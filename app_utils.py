import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LinearLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
import datetime
import pickle
import json
import shutil


from data.load_data import TesseractDataset, ProcessedDataset, get_paths


def set_seeds(seed=42, deterministic=False, neck_type=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if deterministic:
        # Slows training down but ensures reproducibility
        if neck_type == 'DeformConvResidual':
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("Warning: Using deterministic algorithms with warnings enabled due to DeformConvResidual.")
        else:
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def update_paths(cfg):
    cfg.PATH_TO_FOLDERS = f"{cfg.PATH_TO_EXPERIMENTS}/{cfg.MODEL_CFG['rad_backbone']['type']}/{cfg.DATE_FOLDER}"
    cfg.PATH_TO_TEST_SET = f"{cfg.PATH_TO_FOLDERS}/data_split"
    cfg.PATH_TO_PLOTS_TRAIN = f"{cfg.PATH_TO_FOLDERS}/plots/train"
    cfg.PATH_TO_PLOTS_INFERENCE = f"{cfg.PATH_TO_FOLDERS}/plots/inference"
    cfg.PATH_TO_SAVED_MODELS = f"{cfg.PATH_TO_FOLDERS}/models"
    cfg.PATH_TO_LOGS = f"{cfg.PATH_TO_FOLDERS}/logs"
    cfg.PATH_TO_SAVED_INFERENCE_DATA = f"{cfg.PATH_TO_FOLDERS}/inference"


def save_params(cfg, extended_path, config_path, mode='default', save_mode="python"):
    now = datetime.datetime.now()
    date_time = now.strftime("%y_%m_%d_%H:%M")

    if save_mode == "python":
        src = os.path.join(os.path.dirname(__file__), config_path)
        dst = os.path.join(extended_path, f'params_snapshot_{mode}_{date_time}.py')
        shutil.copy(src, dst)

    else:
        # Create a text file and write all params to it
        param_path = os.path.join(extended_path, f'params_{mode}_{date_time}.txt')
        with open(param_path, 'w') as f:
            for attr in dir(cfg):
                if not attr.startswith("__") and not callable(getattr(cfg, attr)):
                    f.write(f"{attr}: {getattr(cfg, attr)}\n")


def build_folder_structure_and_paths(cfg, config_path, arg_time=None):
    # Create experiment directory
    if not os.path.exists(cfg.PATH_TO_EXPERIMENTS):
        os.makedirs(cfg.PATH_TO_EXPERIMENTS)

    model_type = cfg.MODEL_CFG['rad_backbone']['type']
    model_path = os.path.join(cfg.PATH_TO_EXPERIMENTS, model_type)

    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create timestamped subdirectory
    if arg_time is not None:
        folder_name = arg_time
    else:
        now = datetime.datetime.now()
        folder_name = now.strftime("%y_%m_%d_%H_%M")

    # Set date folder in the params to setup correct paths
    cfg.DATE_FOLDER = folder_name + "_" + cfg.EXPERIMENT_NAME
    update_paths(cfg)

    extended_path = os.path.join(model_path, cfg.DATE_FOLDER)
    if not os.path.exists(extended_path):
        os.makedirs(extended_path)

    save_params(cfg, extended_path, config_path, mode='train')

    # Build subdirectiories for data_split, models, logs, inference, plots (train/inference)
    subdirs = ['data_split', 'models', 'logs', 'inference', 'plots']
    for subdir in subdirs:
        subdir_path = os.path.join(extended_path, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        
        if subdir == 'plots':
            for plot_type in ['train', 'inference']:
                plot_type_path = os.path.join(subdir_path, plot_type)
                if not os.path.exists(plot_type_path):
                    os.makedirs(plot_type_path)
    

def restructure_cluster_path(cfg, cluster_path, is_gt=False):
    # Example: /home/coralidar/projects/Dataset/K-Radar_reduced/1/DERA_tesseract/DERA_tesseract_00142.npy
    if isinstance(cluster_path, list):
        path_list = []
        for cluster_path_single in cluster_path:
            if cluster_path_single.startswith('/home/coralidar/projects/'):
                parts = cluster_path_single.split('/')
                subset = parts[-3]
                folder = parts[-2]
                data = parts[-1]
                if is_gt:
                    # Folder is subset here
                    path_list.append(f"{cfg.PATH_TO_LABELS}/{folder}/{data}")
                else:
                    path_list.append(f"{cfg.PATH_TO_PROCESSED_DATASET}/{subset}/{folder}/{data}")
        return path_list
    else:
        if cluster_path_single.startswith('/home/coralidar/projects/'):
            parts = cluster_path.split('/')
            subset = parts[-3]
            folder = parts[-2]
            data = parts[-1]
            if is_gt:
                # Folder is subset here
                return f"{cfg.PATH_TO_LABELS}/{folder}/{data}"
            else:
                return f"{cfg.PATH_TO_PROCESSED_DATASET}/{subset}/{folder}/{data}"


def load_dataset(cfg, validate=False):
    if validate:
        rdr_paths, cam_paths, ldr_paths, gt_paths = get_paths(cfg, save=True, validate=True)
        return ProcessedDataset(rdr_paths, cam_paths, ldr_paths, gt_paths)

    if cfg.DATA == 'tesseract':
        # Tries different paths based on the existence of the first file
        first_path = f'{cfg.PATH_TO_DATASET}/{cfg.SUBSET}/radar_tesseract/tesseract_{1:05d}.mat'
        if os.path.exists(first_path):
            length = len(os.listdir(f'{cfg.PATH_TO_DATASET}/{cfg.SUBSET}/radar_tesseract'))
            path_to_dataset = [f'{cfg.PATH_TO_DATASET}/{cfg.SUBSET}/radar_tesseract/tesseract_{i:05d}.mat' for i in range(1, length + 1)]
        else:
            path_to_dataset = [f'{cfg.PATH_TO_DATASET}/{cfg.SUBSET}/tesseract_batch/tesseract_{i:05d}.mat' for i in range(1, 4)]

        # Build dataset with all paths
        dataset = TesseractDataset(path_to_dataset)

    elif cfg.DATA == 'processed':
        rdr_paths, cam_paths, ldr_paths, gt_paths = get_paths(cfg, save=True)
        
        if cfg.MODE == 'test' and cfg.CLUSTER == False:
            rdr_paths = restructure_cluster_path(cfg, rdr_paths)
            cam_paths = restructure_cluster_path(cfg, cam_paths)
            ldr_paths = restructure_cluster_path(cfg, ldr_paths)
            gt_paths = restructure_cluster_path(cfg, gt_paths, is_gt=True)
        
        if cfg.APPROACH == 'RadarLidar':
            dataset = ProcessedDataset(cfg, rdr_paths, cam_paths, ldr_paths, gt_paths, return_lidar=True)
        else:
            dataset = ProcessedDataset(cfg, rdr_paths, cam_paths, ldr_paths, gt_paths, return_lidar=False)

    else:
        raise ValueError(f"Unsupported data type: {cfg.DATA}. Must be 'tesseract' or 'processed' or 'experimental'.")

    return dataset


def build_dataloader(cfg, dataset, sampler=None, shuffle=False):
    batch_size = cfg.BATCH_SIZE
    num_workers = cfg.NUM_WORKERS

    if sampler is not None:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        

def build_optimizer(cfg, model):
    if model is None:
        raise ValueError("Combined model must be provided.")

    params = model.parameters()

    return AdamW(params, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)


def build_scheduler(cfg, optimizer, number_samples):
    scheduler = cfg.SCHEDULER
    batch_size = cfg.BATCH_SIZE

    if scheduler == 'None':
        return None
    
    elif scheduler == 'StepLR':
        '''This works on an epoch-level, meaning that the learning rate will be updated every epoch.'''
        step_size = cfg.STEPLR_STEP_SIZE
        gamma = cfg.STEPLR_GAMMA

        # StepLR reduces the learning rate by multiplying it with gamma every step_size epochs. 
        # For example, with step_size=1 and gamma=0.66, the learning rate will be reduced to 66% of its previous value every epoch.
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler == 'MultiStepLR':
        '''This works on an epoch-level, meaning that the learning rate will be updated every epoch.'''
        milestones = cfg.MULTISTEPLR_MILESTONES
        gamma = cfg.MULTISTEPLR_GAMMA

        # MultiStepLR reduces the learning rate by multiplying it with gamma at specific epochs defined in milestones. 
        # For example, with milestones=[3, 6, 9] and gamma=0.5, the learning rate will be reduced to 50% of its previous value at epochs 3, 6, and 9.
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler == 'LinearLR':
        '''This works on an iteration-level, meaning that the learning rate will be updated every iteration.'''
        start_factor = 1.0
        end_factor = cfg.LINLR_MIN_LR / cfg.LEARNING_RATE  # end_factor is the final learning rate as a fraction of the initial learning rate
        total_iters = cfg.EPOCHS * max(1, number_samples // batch_size)

        # LinearLR linearly decreases the learning rate from start_factor * initial_lr to end_factor * initial_lr over total_iters iterations.
        return LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters)
    
    elif scheduler == 'ExponentialLR':
        '''This works on an iteration-level, meaning that the learning rate will be updated every iteration.'''
        gamma = cfg.EXPLR_GAMMA

        # ExponentialLR reduces the learning rate by multiplying it with gamma every iteration. 
        # For example, with gamma=0.66, the learning rate will be reduced to 66% of its previous value every iteration.
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler == 'CosineAnnealingLR':
        '''This works on an iteration-level, meaning that the learning rate will be updated every iteration.'''
        min_lr = cfg.CALR_MIN_LR
        cyclic = cfg.CALR_CYCLIC

        if cyclic:
            # CosineAnnealingLR requires T_max, which is the maximum number of iterations
            # Total_iter should probably be (total number of samples) // (batch size)
            total_iter = max(1, number_samples // batch_size)
        else:
            # If not cyclic, we want to run through the cosine annealing schedule for the entire training duration, 
            # which is epochs * (number of samples // batch size)
            total_iter = cfg.EPOCHS * max(1, number_samples // batch_size)

        return CosineAnnealingLR(optimizer, T_max=total_iter, eta_min=min_lr)
    
    elif scheduler == 'CosineAnnealingWarmRestarts':
        '''This works on an iteration-level, meaning that the learning rate will be updated every iteration.'''
        t0 = cfg.CAWRLR_T0
        eta_min = cfg.CAWRLR_MIN_LR

        # CosineAnnealingWarmRestarts implements a cosine annealing schedule with warm restarts. 
        # The learning rate will follow a cosine curve and then restart to the initial learning rate after T_0 iterations. 
        # After each restart, the period of the cosine curve will be multiplied by T_mult. 
        # For example, with T_0=6 and T_mult=1, the learning rate will restart every 6 iterations and follow the same cosine curve each time.
        return CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=1, eta_min=eta_min)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler}. Only 'None' or 'CosineAnnealingLR' is supported.")


def create_batch_dict(cfg, rdr_data, ldr_data, cam_paths, gt_paths, device):
    batch_size = rdr_data.shape[0]
            
    # Create batch_dict as expected by the the respective model
    if cfg.DATA == 'tesseract':
        batch_dict = {
            'tesseract': rdr_data.to(device),
            'batch_size': batch_size
        }
    elif cfg.DATA == 'processed':
        batch_dict = {
            'rdr_era_dra': rdr_data.to(device),
            'ldr': ldr_data.to(device) if ldr_data is not None else None,
            'cam_paths': cam_paths,
            'gt_paths': gt_paths,
            'batch_size': batch_size
        }

    return batch_dict


def save_inference_data_per_frame(cfg, batch_dict, cam_paths):
    predicted_boxes = batch_dict['predicted_boxes']     # list[Tensor]
    gt_lists = batch_dict['gt_lists']                   # list[list]

    for batch_idx in range(len(predicted_boxes)):
        # Predicted boxes are in format [x, y, z, l, w, h, sin(yaw), cos(yaw), conf.score, class, ra_azi, ra_range]
        pred_boxes = predicted_boxes[batch_idx]         # Predicted boxes for this batch
        gt_list = gt_lists[batch_idx]                   # Complete information of the ground truths

        # Build save path
        inference_path = cfg.PATH_TO_SAVED_INFERENCE_DATA

        cam_path = cam_paths[batch_idx]
        subset = cam_path.split('/')[-3]

        frame = cam_path.split('_')[-1].split('.')[0]  # Extract frame number from path
        save_path = f"{inference_path}/boxes_subset_{subset}_frame_{frame}.pkl"

        with open(save_path, 'wb') as f:
            pickle.dump({
                'predicted_boxes': pred_boxes.cpu().numpy(),
                'gt_list': gt_list,
            }, f)


def select_subsets_for_combined_evaluation(cfg, weather_condition=None):
    with open(f"{cfg.PATH_TO_EXPERIMENTS}/metadata.json", 'r') as f:
        metadata = json.load(f)

    subsets = set(metadata.keys())

    weather_subsets = set()
    for subset, attributes in metadata.items():
        if  attributes['weather_condition'] == weather_condition:
            weather_subsets.add(subset)
    subsets = subsets & weather_subsets

    if not subsets:
        raise ValueError("No subsets found matching the specified criteria.")
    print(f"Weather: {weather_condition}, Selected subsets for evaluation: {subsets}")
    return sorted(subsets, key=lambda x: int(x))


def select_subsets_for_evaluation(cfg):
    # Example structure of metadata:
    # {
    #   "1": {
    #     "classes": ["Bus or Truck", "Sedan"],
    #     "environment": "urban",
    #     "time_of_day": "night",
    #     "weather_condition": "normal"
    #   },
    #   ...

    with open(f"{cfg.PATH_TO_EXPERIMENTS}/metadata.json", 'r') as f:
        metadata = json.load(f)

    subsets = set(metadata.keys())

    # Filter by classes
    if getattr(cfg, "CLASSES", None):
        cls_subsets = set()
        for subset, attributes in metadata.items():
            for cls in cfg.CLASSES:
                if cls in attributes['classes']:
                    cls_subsets.add(subset)
        subsets = subsets & cls_subsets

    # Filter by environment
    if getattr(cfg, "ENVIRONMENT", None):
        env_subsets = set()
        for subset, attributes in metadata.items():
            if attributes['environment'] in cfg.ENVIRONMENT:
                env_subsets.add(subset)
        subsets = subsets & env_subsets

    # Filter by time of day
    if getattr(cfg, "TIME_OF_DAY", None):
        time_subsets = set()
        for subset, attributes in metadata.items():
            if attributes['time_of_day'] in cfg.TIME_OF_DAY:
                time_subsets.add(subset)
        subsets = subsets & time_subsets

    # Filter by weather condition
    if getattr(cfg, "WEATHER_CONDITION", None):
        weather_subsets = set()
        for subset, attributes in metadata.items():
            if attributes['weather_condition'] in cfg.WEATHER_CONDITION:
                weather_subsets.add(subset)
        subsets = subsets & weather_subsets

    if not subsets:
        raise ValueError("No subsets found matching the specified criteria.")

    # Sort numerically but keep as strings
    return sorted(subsets, key=lambda x: int(x))


def setup_inference_data_paths_for_evaluation(cfg, subsets):
    inference_path = cfg.PATH_TO_SAVED_INFERENCE_DATA
    subset_set = set(subsets)
    paths = []

    for file in os.listdir(inference_path):
        if file.endswith('.pkl'):
            # Assumes filename format: boxes_subset_{subset}_frame_{frame}.pkl
            parts = file.split('_')
            if len(parts) >= 4 and parts[0] == 'boxes' and parts[1] == 'subset' and parts[3] == 'frame':
                subset = parts[2]
                if subset in subset_set:
                    paths.append(os.path.join(inference_path, file))

    paths.sort()  # Ensure consistent order

    return paths


def load_inference_data_per_frame(paths):
    frames_pred_boxes = []
    frames_gt_lists = []
    frames_subsets = []
    frames_frame_number = []

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Example path: .../experiments/UNet/2026_02_04_17_10_rad_stem_ndil_hexp_allcls_aniso/inference/epoch10/boxes_subset_10_frame_00847.pkl
        # This extracts the last part after splitting by '/' and then splits by '_' to get subset and frame number
        parts = os.path.basename(path).split('_')
        subset = parts[2]                           # position 3 is subset
        frame_number = parts[4].split('.')[0]       # position 5 is frame number before .pkl

        frames_pred_boxes.append(data['predicted_boxes'])
        frames_gt_lists.append(data['gt_list'])
        frames_subsets.append(int(subset))
        frames_frame_number.append(int(frame_number))

    return frames_pred_boxes, frames_gt_lists, frames_subsets, frames_frame_number


def filter_processed_inference_data(cfg, dataset):
    processed_files = set()

    inference_data_path = f"{cfg.PATH_TO_SAVED_INFERENCE_DATA}/{cfg.INF_MODEL}"
    
    if os.path.exists(inference_data_path):
        for fname in os.listdir(inference_data_path):
            if fname.startswith("boxes_") and fname.endswith(".pkl"):
                # Original: boxes_subset_{subset}_frame_{frame}.pkl
                file_parts = fname.split("_")
                file_subset = file_parts[2]
                file_frame = file_parts[-1].split(".")[0]
                processed_files.add((file_subset, file_frame))
    
    if len(processed_files) == 0:
        print("Nothing to filter, no processed files found.")
        return # Exit early if nothing to filter
    
    print(f"Found {len(processed_files)} already processed files. Skipping them.")
    original_length = len(dataset)
    dataset.filter_processed(processed_files)
    print(f"Dataset reduced from {original_length} to {len(dataset)} samples.")


def verify_paths(rdr_paths, cam_paths, ldr_paths, gt_paths):
    if not (rdr_paths and cam_paths and ldr_paths and gt_paths):
        raise ValueError("All path lists must be provided.")

    assert len(rdr_paths) == len(cam_paths) == len(ldr_paths) == len(gt_paths), "All path lists must have the same length."

    missing_rdr = [p for p in rdr_paths if not os.path.exists(p)]
    missing_cam = [p for p in cam_paths if not os.path.exists(p)]
    missing_ldr = [p for p in ldr_paths if not os.path.exists(p)]
    missing_gt = [p for p in gt_paths if not os.path.exists(p)]

    if missing_rdr:
        print("Missing rdr files:")
        for p in missing_rdr:
            print(p)
    else:
        print("All rdr files exist.")

    if missing_cam:
        print("Missing cam files:")
        for p in missing_cam:
            print(p)
    else:
        print("All cam files exist.")

    if missing_ldr:
        print("Missing ldr files:")
        for p in missing_ldr:
            print(p)
    else:
        print("All ldr files exist.")

    if missing_gt:
        print("Missing gt files:")
        for p in missing_gt:
            print(p)
    else:
        print("All gt files exist.")


if __name__ == "__main__":
    pass