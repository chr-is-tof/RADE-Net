import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import csv
import os

from app_utils import (
    create_batch_dict,
    load_dataset,
    build_dataloader,
    save_params,
    save_inference_data_per_frame,
    select_subsets_for_evaluation,
    setup_inference_data_paths_for_evaluation,
    load_inference_data_per_frame,
    set_seeds,
    build_folder_structure_and_paths,
    build_optimizer,
    build_scheduler,
    filter_processed_inference_data,
    select_subsets_for_combined_evaluation
)
from models.model_assembly import build_model
from data.ground_truth import get_ground_truth
from ops.loss_utils import compute_losses
from utils.plotting import visualize, visualize_pkl, paper_plots
from utils.evaluation import evaluation_mAP
from ops.nms import postprocessing
from utils.config_utils import print_config_settings


def is_ddp():
    return "LOCAL_RANK" in os.environ or "RANK" in os.environ


# Sends a tensor to the main process and adapts it using the specified operation
def reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    # If not in DDP mode, simply return the tensor
    if not is_ddp():
        return tensor
    
    world_size = dist.get_world_size()
    dist.reduce(tensor, dst=0, op=op)

    if dist.get_rank() == 0:
        return tensor / world_size
    else:
        return None
    

def train(cfg, model=None, dataloader=None, optimizer=None, scheduler=None, 
          device=None, epochs=35, verbose=False):
    if model is None:
        raise ValueError("Model must be provided for training.")

    if dataloader is None:
        raise ValueError("Dataloader must be provided for training.")
    
    if optimizer is None:
        raise ValueError("Optimizer must be provided for training.")
    
    if device is None:
       raise ValueError("Device must be provided for training.")
    
    if verbose:
        print_config_settings(cfg)
        
    if not is_ddp() or dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=f'{cfg.PATH_TO_LOGS}')

    if cfg.IS_VALIDATE:
        dataset_val = load_dataset(cfg, validate=True)
        if is_ddp():
            sampler_val = DistributedSampler(dataset_val)
            dataloader_val = build_dataloader(cfg, dataset_val, sampler=sampler_val, shuffle=False)
        else:
            dataloader_val = build_dataloader(cfg, dataset_val, sampler=None, shuffle=False)

    train_losses = []
    for epoch in range(epochs): 
        # Set a global current epoch
        cfg.CURRENT_EPOCH = epoch

        # Set train mode
        model.train()

        if is_ddp():
            dataloader.sampler.set_epoch(epoch)
        
        batch_idx = 0
        epoch_train_loss = []

        for rdr_data, ldr_data, cam_paths, gt_paths in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            #print(f"Epoch {epoch+1}/{epochs}, Batch: {batch_idx+1}")
            batch_idx += 1

            optimizer.zero_grad()
            
            # This also loads the data to device
            batch_dict = create_batch_dict(cfg, rdr_data, ldr_data, cam_paths, gt_paths, device)

            # Get ground truth data
            batch_dict = get_ground_truth(cfg, batch_dict, gt_paths, device)

            # Forward pass through the model
            batch_dict = model(batch_dict)

            loss, batch_dict = compute_losses(cfg, batch_dict, focal_loss_type="focal_loss_continuous")

            epoch_train_loss.append(batch_dict['loss_dict']['total_loss'])

            # Backbone / neck / head gradients
            loss.backward()

            # Updates weights according to the optimizer
            optimizer.step()

            # Step the scheduler if provided
            if scheduler is not None and cfg.SCHEDULER in ['LinearLR', 'ExponentialLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                scheduler.step()

            if cfg.VISUALIZE_TRAIN:
                visualize(cfg, batch_dict, cam_paths, batch_idx, batch_dict['loss_dict'])

            # Every process needs to call reduce tensor, but only logging is done by rank 0
            distr_avg_batch_loss = reduce_tensor(batch_dict['loss_dict']['total_loss'], op=dist.ReduceOp.SUM)

            if not is_ddp() or dist.get_rank() == 0:
                writer.add_scalar('LossAvgBatch/Iter', distr_avg_batch_loss, epoch * len(dataloader) + batch_idx)
                writer.add_scalar('LearningRate/Iter', optimizer.param_groups[0]['lr'], epoch * len(dataloader) + batch_idx)
                writer.add_scalar('LossHeatmap/Iter', batch_dict['loss_dict']['focal_loss'], epoch * len(dataloader) + batch_idx)
                writer.add_scalar('LossGWD/Iter', batch_dict['loss_dict']['gwd_loss'], epoch * len(dataloader) + batch_idx)
                writer.add_scalar('LossSmoothL1/Iter', batch_dict['loss_dict']['smooth_l1_loss'], epoch * len(dataloader) + batch_idx)
                if 'classification_loss' in batch_dict['loss_dict']:
                    writer.add_scalar('LossClassification/Iter', batch_dict['loss_dict']['classification_loss'], epoch * len(dataloader) + batch_idx)

        avg_epoch_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        print(f"Completed epoch {epoch + 1}/{epochs}. Epoch loss: {avg_epoch_loss}")
        train_losses.append(avg_epoch_loss)

        if scheduler is not None and cfg.SCHEDULER in ['StepLR', 'MultiStepLR']:
            scheduler.step()

        # Again, every process needs to call reduce_tensor but only rank 0 logs it
        distributed_avg_epoch_loss = reduce_tensor(torch.tensor(avg_epoch_loss), op=dist.ReduceOp.SUM).to(device)
        
        if not is_ddp() or dist.get_rank() == 0:
            writer.add_scalar('LossAvgEpoch', distributed_avg_epoch_loss, epoch)

        # Read as: saves the model weights when true, but only once if DDP is also true
        if (cfg.SAVE_MODELS and epoch % 5 == 0 and epoch != 0) or (cfg.SAVE_MODELS and epoch == epochs - 1) or (cfg.SAVE_ALL_MODELS):
            if not is_ddp() or dist.get_rank() == 0:
                if is_ddp():
                    torch.save(model.module.state_dict(), f"{cfg.PATH_TO_SAVED_MODELS}/epoch{epoch}.pth")
                else:
                    torch.save(model.state_dict(), f"{cfg.PATH_TO_SAVED_MODELS}/epoch{epoch}.pth")
                print(f"Model at epoch {epoch} saved to {cfg.PATH_TO_SAVED_MODELS}/epoch{epoch}.pth")

        if cfg.IS_VALIDATE:
            print("Starting validation ...")

            # Set model to eval mode
            model.eval()

            with torch.no_grad():
                val_loss = []
                for rdr_data, ldr_data, cam_paths, gt_paths in tqdm(dataloader_val, desc="Validation"):
                    batch_dict = create_batch_dict(cfg, rdr_data, ldr_data, cam_paths, gt_paths, device)

                    batch_dict = get_ground_truth(cfg, batch_dict, gt_paths, device)

                    batch_dict = model(batch_dict)

                    loss, batch_dict = compute_losses(cfg, batch_dict, focal_loss_type="focal_loss_continuous")

                    val_loss.append(batch_dict['loss_dict']['total_loss'])
                
                avg_val_loss = sum(val_loss) / len(val_loss)
                distributed_avg_val_loss = reduce_tensor(torch.tensor(avg_val_loss), op=dist.ReduceOp.SUM).to(device)

                if not is_ddp() or dist.get_rank() == 0:
                    print(f"Validation loss after epoch {epoch + 1}: {distributed_avg_val_loss}")
                    writer.add_scalar('LossAvgVal', distributed_avg_val_loss, epoch + 1)
            
            model.train()
            print("Resuming training ...")

    if not is_ddp() or dist.get_rank() == 0:
        writer.close()

    return train_losses


def test(cfg, model=None, dataloader=None, device=None):
    if model is None:
        raise ValueError("Model must be provided for testing.")
    
    if dataloader is None:
        raise ValueError("Dataloader must be provided for testing.")
    
    if device is None:
        raise ValueError("Device must be provided for testing.")

    model.eval()

    losses = []
    with torch.no_grad():
        with open(f"{cfg.PATH_TO_LOGS}/loss_overview_{cfg.INF_MODEL}.csv", 'w', newline='') as csvfile:
            fieldnames = ['subset', 'file_name', 'total_loss', 'focal_loss', 'gwd_loss', 'smooth_l1_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for rdr_data, ldr_data, cam_paths, gt_paths in tqdm(dataloader, desc="Testing"):           
                # This also loads the data to device
                batch_dict = create_batch_dict(cfg, rdr_data, ldr_data, cam_paths, gt_paths, device)

                batch_dict = model(batch_dict)

                batch_dict = get_ground_truth(cfg, batch_dict, gt_paths, device)

                batch_dict = compute_losses(cfg, batch_dict, focal_loss_type="focal_loss_continuous")[1]

                loss_dict = batch_dict['loss_dict']
                
                # Batch size is here always 1 during testing
                parts = gt_paths[0].split(os.sep)
                subset = parts[-2]              # Subset is the second last part
                file_name = parts[-1]           # File name is the last part

                loss_dict['subset'] = subset
                loss_dict['file_name'] = file_name

                writer.writerow(loss_dict)

                losses.append(loss_dict['total_loss'])
                # avg_test_loss = sum(losses) / len(losses)
            
            avg_test_loss = sum(losses) / len(losses)
            print(f"[Model {cfg.INF_MODEL}] Average test loss over the dataset: {avg_test_loss}")
            writer.writerow({'subset': 0, 'file_name': 'overall', 'total_loss': avg_test_loss})


def inference(cfg, config_path, model=None, dataloader=None, device=None, verbose=False):
    if model is None:
        raise ValueError("Model must be provided for inference.")
    
    if dataloader is None:
        raise ValueError("Dataloader must be provided for inference.")
    
    if device is None:
        raise ValueError("Device must be provided for inference.")

    cfg.PATH_TO_SAVED_INFERENCE_DATA = f"{cfg.PATH_TO_SAVED_INFERENCE_DATA}/{cfg.INF_MODEL}"
    cfg.PATH_TO_PLOTS_INFERENCE = f"{cfg.PATH_TO_PLOTS_INFERENCE}/{cfg.INF_MODEL}"

    if not os.path.exists(cfg.PATH_TO_SAVED_INFERENCE_DATA):
        os.makedirs(cfg.PATH_TO_SAVED_INFERENCE_DATA)
    if not os.path.exists(cfg.PATH_TO_PLOTS_INFERENCE):
        os.makedirs(cfg.PATH_TO_PLOTS_INFERENCE)

    if verbose:
        print_config_settings(cfg)

    model.eval()
    batch_idx = 0

    save_params(cfg, cfg.PATH_TO_FOLDERS, config_path, mode='inference')

    with torch.no_grad():
        for rdr_data, ldr_data, cam_paths, gt_paths in tqdm(dataloader, desc="Inference"):           
            # This also loads the data to device
            batch_dict = create_batch_dict(cfg, rdr_data, ldr_data, cam_paths, gt_paths, device)

            batch_dict = model(batch_dict)
            batch_dict = get_ground_truth(cfg, batch_dict, gt_paths, device)
            batch_dict = postprocessing(cfg, batch_dict)

            if cfg.VISUALIZE_INFERENCE:
                visualize(cfg, batch_dict, cam_paths, batch_idx, loss_dict=None)

            if cfg.SAVE_INFERENCE_OUTPUTS:
                save_inference_data_per_frame(cfg, batch_dict, cam_paths)
    

def evaluation(cfg):
    if not cfg.EVALUATE_MAP:
        print("mAP evaluation is disabled in the parameters. Exiting evaluation.")
        return

    if not cfg.INCLUDE_3D:
        print("3D mAP evaluation is currently excluded. Only 2D evaluation will be performed.")

    print_config_settings(cfg)
    
    if cfg.EVALUATION_MODE == 'allweather-allclass':
        print("Evaluation mode set to 'allweather-allclass'. Evaluating on all weather conditions and classes without filtering.")
        all_weather_conditions = ['normal', 'overcast', 'fog', 'rain', 'sleet', 'lightsnow', 'heavysnow']
        all_strings = []
        path_to_saved_inference_data = cfg.PATH_TO_SAVED_INFERENCE_DATA
        for weather in all_weather_conditions:
            cfg.PATH_TO_SAVED_INFERENCE_DATA = f"{path_to_saved_inference_data}/{cfg.INF_MODEL}"
            print(f"Inference data path: {cfg.PATH_TO_SAVED_INFERENCE_DATA}")
            if not os.path.exists(cfg.PATH_TO_SAVED_INFERENCE_DATA):
                raise ValueError(f"Path to saved inference data does not exist: {cfg.PATH_TO_SAVED_INFERENCE_DATA}")

            subsets = select_subsets_for_combined_evaluation(cfg, weather_condition=weather)
            paths = setup_inference_data_paths_for_evaluation(cfg, subsets)
            (frames_pred_boxes, frames_gt_lists,
             frames_subsets, frames_frame_number) = load_inference_data_per_frame(paths)
            map_2d, map_3d = evaluation_mAP(
                cfg, frames_pred_boxes, frames_gt_lists, frames_subsets, frames_frame_number, print_output=True
            )
            aps_2d = map_2d['COCO']
            aps_3d = map_3d['COCO']

            def _format_ap(aps_dict: dict, class_name: str) -> str:
                """Return per-class AP as '%.2f', or '-' if the class was not evaluated."""
                if class_name in aps_dict['aps']:
                    return f"{aps_dict['aps'][class_name] * 100:.2f}"
                return "-"

            classes = ['Sedan', 'Bus or Truck', 'Pedestrian', 'Motorcycle', 'Bicycle']
            per_class_cols = " & ".join(
                f"{_format_ap(aps_3d, cls)} & {_format_ap(aps_2d, cls)}" for cls in classes
            )
            all_strings.append(
                f"{weather} & {aps_3d['mAP']*100:.2f} & {aps_2d['mAP']*100:.2f} & {per_class_cols} \\\\"
            )
        print("\nLaTeX-formatted table rows for each weather condition:")
        for string in all_strings:
            print(string)
        cfg.PATH_TO_SAVED_INFERENCE_DATA = path_to_saved_inference_data
    
    cfg.PATH_TO_SAVED_INFERENCE_DATA = f"{cfg.PATH_TO_SAVED_INFERENCE_DATA}/{cfg.INF_MODEL}"
    print(f"Inference data path: {cfg.PATH_TO_SAVED_INFERENCE_DATA}")
    if not os.path.exists(cfg.PATH_TO_SAVED_INFERENCE_DATA):
        raise ValueError(f"Path to saved inference data does not exist: {cfg.PATH_TO_SAVED_INFERENCE_DATA}")

    subsets = select_subsets_for_evaluation(cfg)
    if cfg.EVALUATE_SINGLE_SUBSET:
        print("Single subset evaluation enabled.")
        subsets = [cfg.EVALUATE_SUBSET]
    print(f"Subset(s) selected for evaluation: {subsets}")

    paths = setup_inference_data_paths_for_evaluation(cfg, subsets)

    (frames_pred_boxes, frames_gt_lists,
     frames_subsets, frames_frame_number) = load_inference_data_per_frame(paths)

    evaluation_mAP(cfg, frames_pred_boxes, frames_gt_lists, frames_subsets, frames_frame_number, print_output=True)

    if cfg.GET_PROBLEMATIC_SAMPLES:
        print("Identifying problematic samples ...")
        frames = []
        for path in paths:
            frames_pred_boxes, frames_gt_lists, _, _ = load_inference_data_per_frame([path])
            # Note: only works for a single method ('VOC', 'PR-AUC', or 'COCO');
            # with multiple methods the return type changes to a dict.
            map_2d, map_3d = evaluation_mAP(frames_pred_boxes, frames_gt_lists, print_output=False, methods=['COCO'])
            name = path.split('/')[-1].replace('boxes_', '').replace('.pkl', '')
            frames.append((name, map_2d, map_3d))
        frames.sort(key=lambda x: (x[1], x[2]))  # Sort by 2D mAP, then by 3D mAP
        print("Top 100 problematic samples (lowest mAP):")
        txt_file = f"{cfg.PATH_TO_LOGS}/problematic_samples_{cfg.INF_MODEL}.txt"
        with open(txt_file, 'w') as f:
            f.write(f"Top 10 problematic samples (lowest mAP) for model {cfg.INF_MODEL}:\n")
            for i, frame in enumerate(frames, 1):
                f.write(f"{i:2d}. 2D mAP: {frame[1]:.4f}, 3D mAP: {frame[2]:.4f}, Name: {frame[0]}\n")

        # Remove samples with mAP lower than threshold
        threshold = 0.8
        filtered_frames = [frame for frame in frames if frame[1] >= threshold and frame[2] >= threshold]

        # Split by weather condition
        weather_conditions = {
            'normal': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15', '16', '17', '18', '19', '20'],
            'overcast': ['13', '22'],
            'fog': ['38', '39', '40', '41', '44', '45'],
            'rain': ['21', '23', '24', '25', '26', '32', '33', '34'],
            'sleet': ['27', '28', '29', '30', '31', '35', '36', '37', '50', '51', '52', '53'],
            'lightsnow': ['42', '43', '48', '49'],
            'heavysnow': ['46', '47', '54', '55', '56', '57', '58']
        }

        # Sort filtered frames into weather categories
        categorized_frames = {key: [] for key in weather_conditions.keys()}
        for frame in filtered_frames:
            for condition, subsets_list in weather_conditions.items():
                if frame[0].split('_')[-3] in subsets_list:
                    categorized_frames[condition].append(frame)
                    break

        # Write each category to separate files
        for condition, frames_list in categorized_frames.items():
            txt_file_condition = f"{cfg.PATH_TO_LOGS}/{cfg.INF_MODEL}_{condition}.txt"
            with open(txt_file_condition, 'w') as f:
                for i, frame in enumerate(frames_list, 1):
                    f.write(f"{i:2d}. 2D mAP: {frame[1]:.4f}, 3D mAP: {frame[2]:.4f}, Name: {frame[0]}\n")


def visualize_inference_from_model(cfg, model=None, dataloader=None, device=None):
    print("Starting visualization of inference results ...")
    print(f"Folder: {cfg.DATE_FOLDER}")
    print(f"Model: {cfg.INF_MODEL}")

    cfg.PATH_TO_SAVED_INFERENCE_DATA = cfg.PATH_TO_SAVED_INFERENCE_DATA + "/" + cfg.INF_MODEL
    if not os.path.exists(cfg.PATH_TO_SAVED_INFERENCE_DATA):
        os.makedirs(cfg.PATH_TO_SAVED_INFERENCE_DATA)

    cfg.PATH_TO_PLOTS_INFERENCE = cfg.PATH_TO_PLOTS_INFERENCE + "/" + cfg.INF_MODEL
    if not os.path.exists(cfg.PATH_TO_PLOTS_INFERENCE):
        os.makedirs(cfg.PATH_TO_PLOTS_INFERENCE)

    if cfg.BATCH_SIZE != 1:
        raise ValueError("For visualization during inference, batch size must be 1.")

    model.eval()
    batch_idx = 0
    with torch.no_grad():
        for rdr_data, ldr_data, cam_paths, gt_paths in tqdm(dataloader, desc="Inference"):           
            batch_dict = create_batch_dict(cfg, rdr_data, ldr_data, cam_paths, gt_paths, device)
            batch_dict = get_ground_truth(cfg, batch_dict, gt_paths, device)

            # Filter based on the choice of classes to save computation
            gt_lists = batch_dict['gt_lists']
            valid = False
            for gt in gt_lists[0]:
                if gt[0] in cfg.CLASSES:
                    valid = True
                    break
            
            if not valid:
                continue

            batch_dict = model(batch_dict)
            batch_dict = postprocessing(batch_dict)

            visualize(cfg, batch_dict, cam_paths, batch_idx, loss_dict=None)

            batch_idx += 1 


def visualize_inference_from_pkl(cfg):
    print("Starting visualization of inference results ...")
    print(f"Folder: {cfg.DATE_FOLDER}")
    print(f"Model: {cfg.INF_MODEL}")

    cfg.PATH_TO_SAVED_INFERENCE_DATA = cfg.PATH_TO_SAVED_INFERENCE_DATA + "/" + cfg.INF_MODEL
    if not os.path.exists(cfg.PATH_TO_SAVED_INFERENCE_DATA):
        raise ValueError(f"Path to saved inference data does not exist: {cfg.PATH_TO_SAVED_INFERENCE_DATA}")

    subsets = select_subsets_for_evaluation(cfg)
    
    if cfg.EVALUATE_SINGLE_SUBSET:
        print("Single subset visualization is enabled.")
        subsets = [cfg.EVALUATE_SUBSET]
    print(f"Subset(s) selected for visualization: {subsets}")

    paths = setup_inference_data_paths_for_evaluation(cfg, subsets)

    # list[list[np.array]], list[list[list]]
    frames_pred_boxes, frames_gt_lists, _, _ = load_inference_data_per_frame(paths)

    visualize_pkl(cfg, frames_pred_boxes, frames_gt_lists, paths, subsets)


def paper(cfg):
    # Weather condition category mapping:
    # normal:      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20
    # overcast:    13, 22
    # fog:         38, 39, 40, 41, 44, 45
    # rain:        21, 23, 24, 25, 26, 32, 33, 34
    # sleet:       27, 28, 29, 30, 31, 35, 36, 37, 50, 51, 52, 53
    # lightsnow:   42, 43, 48, 49
    # heavysnow:   46, 47, 54, 55, 56, 57, 58
    print("Generating paper plots ...")

    weather = 'normal'
    chosen_subset = ['17']
    chosen_frames = ['00480']
    if chosen_subset is None or chosen_frames is None:
        with open(f"{cfg.PATH_TO_LOGS}/{cfg.INF_MODEL}_{weather}.txt", 'r') as f:
            lines = f.readlines()
            for line in lines:
                line_parts = line.strip().split('_')
                subset = line_parts[-3]
                frame_id = line_parts[-1]
                if subset in chosen_subset:
                    chosen_frames.append(frame_id)

    roi = cfg.ROI_TYPE # 'narrow' or 'wide'

    print(f"Weather: {weather}")
    print(f"Subset: {chosen_subset[0]}")
    print(f"Plotting {len(chosen_frames)} frames.")
    paper_plots(cfg, weather=weather, chosen_subset=chosen_subset, chosen_frames=chosen_frames, roi=roi)


def main(cfg, config_path, arg_time=None):
    ### Assertions for the config ###
    assert not (cfg.USE_DATA_PADDING and cfg.RDR_PROCESSING_METHOD == "upsample_azimuth"), \
    "Data padding is not compatible with azimuth upsampling"
    ###                           ###

    set_seeds(cfg.SEED, deterministic=cfg.DETERMINISTIC, neck_type=cfg.MODEL_CFG['rad_neck']['type'])
    torch.autograd.set_detect_anomaly(True)

    print("My PID is", os.getpid())

    if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            print(f"CPU Cores: {os.cpu_count()}")
            print (f"NUM_WORKERS: {cfg.NUM_WORKERS}")
    else:
        print("No GPU available, using CPU.")

    if cfg.MODE == "paper":
        paper(cfg)
        return
    
    if cfg.MODE == "visualize_model":
        dataset = load_dataset(cfg)
        subsets = select_subsets_for_evaluation(cfg)
        print("Subsets that adhere to choice: ", subsets)

        # Filter dataset to only include selected subsets of the KRadar test set
        print("Length before filtering: ", len(dataset.data_file_paths), len(dataset.cam_file_paths), len(dataset.gt_file_paths))
        dataset.data_file_paths = [path for path in dataset.data_file_paths if any(f"/{subset}/" in path for subset in subsets)]
        dataset.cam_file_paths = [path for path in dataset.cam_file_paths if any(f"/{subset}/" in path for subset in subsets)]
        dataset.gt_file_paths = [path for path in dataset.gt_file_paths if any(f"/{subset}/" in path for subset in subsets)]
        print("Length after filtering: ", len(dataset.data_file_paths), len(dataset.cam_file_paths), len(dataset.gt_file_paths))
        
        dataloader = build_dataloader(cfg, dataset, sampler=None, shuffle=False)
        model = build_model(cfg.MODE, cfg.APPROACH, cfg.PATH_TO_SAVED_MODELS, 
                            cfg.INF_PATH_TO_CLUSTER, cfg.INF_MODEL, cfg.MODEL_CFG)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        visualize_inference_from_model(cfg, model, dataloader, device)
        return

    if cfg.MODE == "visualize_pkl":
        visualize_inference_from_pkl(cfg)
        return

    if cfg.MODE == "evaluate":
        evaluation(cfg)
        return
    
    if cfg.MODE == "train":
        build_folder_structure_and_paths(cfg, config_path, arg_time)
        print(f"Folder structure and paths set up. Folder: {cfg.DATE_FOLDER}")
    
    dataset = load_dataset(cfg)
    if cfg.MODE == "test":
        print("Checking for already processed files.")
        filter_processed_inference_data(cfg, dataset)

    if is_ddp():
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
            
        sampler = DistributedSampler(dataset)
        dataloader = build_dataloader(cfg, dataset, sampler=sampler, shuffle=False)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if cfg.MODE == "train":
            dataloader = build_dataloader(cfg, dataset, sampler=None, shuffle=True)
        elif cfg.MODE == "test":
            dataloader = build_dataloader(cfg, dataset, sampler=None, shuffle=False)
        else:
            raise ValueError(f"Unsupported mode: {cfg.MODE}. Use 'train' or 'test'.")

    model = build_model(cfg.MODE, cfg.APPROACH, cfg.PATH_TO_SAVED_MODELS, 
                        cfg.INF_PATH_TO_CLUSTER, cfg.INF_MODEL, cfg.MODEL_CFG)
    model = model.to(device)

    if is_ddp():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if cfg.MODE == "train":
        optimizer = build_optimizer(cfg, model)
        scheduler = build_scheduler(cfg, optimizer, number_samples=len(dataset))

        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")

        train(cfg, model, dataloader, optimizer, scheduler=scheduler, device=device, epochs=cfg.EPOCHS, verbose=cfg.VERBOSE)
    
    elif cfg.MODE == "test":
        inference(cfg, config_path, model, dataloader, device=device, verbose=cfg.VERBOSE)
    else:
        raise ValueError(f"Unsupported mode: {cfg.MODE}. Use 'train' or 'test'.")
    
    # Only destroy process group if using DDP
    if is_ddp():
        dist.destroy_process_group()
    

if __name__ == "__main__":
    print("Please start everything through 'main.py' to ensure proper setup.")