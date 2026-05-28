import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from ops.roi import NARROW, WIDE
from .rdr_processing import process_radar

LDR_PCD_COL_MAP = {
    "x": 0,
    "y": 1,
    "z": 2,
    "intensity": 3,
    "t": 4,
    "reflectivity": 5,
    "ring": 6,
    "ambient": 7,
    "range": 8
}


class TesseractDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        """
        Args:
            file_paths (list): List of paths to .mat files.
            transform (callable, optional): Optional transform to apply to the raw data.
        """
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load raw radar tesseract data from .mat file
        file_path = self.file_paths[idx]
        mat_data = loadmat(file_path)
        
        # Extract radar tesseract (replace 'arrDREA' with your actual key)
        tesseract = mat_data['arrDREA']
        
        # Example preprocessing: log scaling and normalization
        tesseract = 10 * np.log10(tesseract + 1e-10)  # Avoid log(0)
        tesseract = tesseract + 1.2
        tesseract = tesseract / 184.3
        
        # Transpose dimensions to match PyTorch format (D, R, A, E)
        tesseract = np.transpose(tesseract, (0, 1, 3, 2))  # (D, R, A, E)
        
        # Convert to PyTorch tensor
        tesseract = torch.from_numpy(tesseract.astype(np.float32))
        
        # Apply any additional transformations if specified
        if self.transform:
            tesseract = self.transform(tesseract)
        
        return tesseract, idx
    

# In case of different calibration values for different subsets, add this here as additional returns
# and then read it from the train loop of dataloader as a tuple
class ProcessedDataset(Dataset):
    def __init__(self, cfg, rdr_file_paths, cam_file_paths, ldr_file_paths, gt_file_paths, transform=None, return_lidar=False):
        """
        Args:
            file_paths (list): List of paths to processed data files.
            transform (callable, optional): Optional transform to apply to the processed data.
        """
        self.rdr_file_paths = rdr_file_paths
        self.cam_file_paths = cam_file_paths
        self.ldr_file_paths = ldr_file_paths
        self.gt_file_paths = gt_file_paths
        self.transform = transform
        self.rdr_processing_method = cfg.RDR_PROCESSING_METHOD
        self.return_lidar = return_lidar
        self.ldr_col_read = cfg.LDR_COL_READ
        self.ldr_processing_method = cfg.LDR_PROCESSING_METHOD

    def __len__(self):
        return len(self.rdr_file_paths)

    def __getitem__(self, idx):
        # Load processed data from .mat file
        data_file_path = self.rdr_file_paths[idx]
        rdr_data = np.load(data_file_path)

        if self.rdr_processing_method is not None:
            rdr_data = process_radar(rdr_data, self.rdr_processing_method)
                
        # Convert to PyTorch tensor
        rdr_data = torch.from_numpy(rdr_data.astype(np.float32))

        # Load lidar data if requested
        if self.return_lidar:
            ldr_data = read_pcd(self.ldr_file_paths[idx], self.ldr_col_read)
        else:
            # Return empty tensor instead of None for DataLoader compatibility
            ldr_data = torch.empty(0)
        
        # Apply any additional transformations if specified
        if self.transform:
            rdr_data = self.transform(rdr_data)
        
        # Get correspoding cam_path (used for plotting)
        cam_file_path = self.cam_file_paths[idx]

        # Also get the corresponding gt_path
        gt_file_path = self.gt_file_paths[idx]

        return rdr_data, ldr_data, cam_file_path, gt_file_path
    
    def filter_processed(self, processed_files):
        """
        Remove already processed samples from the dataset to avoid redundant inference.
        This also includes redundant data loading.

        Args:
            processed_files is a set of (subset, frame) tuples for fast lookup O(1)

        Makes the method O(n) instead of a naive O(n^2) approach.
        """
        # Filter all lists simultaneously
        filtered_data = []
        filtered_cam = []
        filtered_ldr = []
        filtered_gt = []
        
        for data_path, cam_path, ldr_path, gt_path in zip(self.rdr_file_paths, self.cam_file_paths, self.ldr_file_paths, self.gt_file_paths):
            # Extract subset and frame from cam_path
            # Example: .../Dataset/K-Radar_reduced/1/cam-front/cam-front_00449.png
            path_parts = cam_path.split('/')
            path_subset = path_parts[-3]
            path_frame = path_parts[-1].split('_')[-1].split('.')[0]
            
            # Check if this file was already processed
            if (path_subset, path_frame) not in processed_files:
                filtered_data.append(data_path)
                filtered_cam.append(cam_path)
                filtered_ldr.append(ldr_path)
                filtered_gt.append(gt_path)

        self.rdr_file_paths = filtered_data
        self.cam_file_paths = filtered_cam
        self.ldr_file_paths = filtered_ldr
        self.gt_file_paths = filtered_gt


def read_pcd(file_path, ldr_col_read, filter_near_zero=True, calib=True, apply_rdr_filter=True, 
             apply_roi_filter=False, roi_type="narrow", z_offset=0.0):
    with open(file_path, 'rb') as f:
        # Dynamically skip header until DATA line (in case of variable header length)
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.lower().startswith('data'):
                break

        # np.loadtxt is much faster than list comprehension; cannot use np.frombuffer due to mixed data types
        usecols = [LDR_PCD_COL_MAP[col] for col in ldr_col_read]
        lidar_points = np.loadtxt(f, usecols=usecols, dtype=np.float32)

    # Filter out near-zero points (invalid)
    if filter_near_zero:
        mask = (np.abs(lidar_points[:, 0]) > 0.01) | (np.abs(lidar_points[:, 1]) > 0.01)
        lidar_points = lidar_points[mask]

    if calib:
        # Single vectorized addition across x, y, z columns at once
        # Calib values are hardcoded here, but are the same for all files, so no need to read from file each time
        lidar_points[:, :3] += np.array([-2.54, 0.3, z_offset], dtype=np.float32)

    if apply_rdr_filter:
        x = lidar_points[:, 0]
        y = lidar_points[:, 1]
        azi = -np.arctan2(y, x)
        rng = np.sqrt(x ** 2 + y ** 2)
        azi = np.rad2deg(azi)

        # Default is [-53, 53]
        fov_min, fov_max = [-53, 53]
        range_limit = 118
        
        mask = (azi >= fov_min) & (azi <= fov_max) & (x >= 0) & (rng <= range_limit)
        lidar_points = lidar_points[mask]

    if apply_roi_filter:
        if roi_type == "narrow":
            roi_filter = NARROW
        elif roi_type == "wide":
            roi_filter = WIDE
        else:
            raise ValueError(f"Unknown ROI type '{roi_type}'. Valid options are 'narrow' or 'wide'.")

        # ROI filter is a dictionary with x, y, z min/max values
        x_min, x_max, y_min, y_max, z_min, z_max = roi_filter.values()
        mask = (
            (lidar_points[:, 0] >= x_min) & (lidar_points[:, 0] <= x_max) &
            (lidar_points[:, 1] >= y_min) & (lidar_points[:, 1] <= y_max) &
            (lidar_points[:, 2] >= z_min) & (lidar_points[:, 2] <= z_max)
        )
        lidar_points = lidar_points[mask]

    return lidar_points  # (N, 4) array of x, y, z, intensity


def extract_sensor_numbers(line: str, sensors=('tesseract', 'cam-front', 'os2-64')):
    # Example line: * idx(tesseract_os2-64_cam-front_os1-128_cam-lrr)=00033_00001_00002_00001_00004, timestamp=1643292946.710046076
    # Find the idx(...) part
    match = re.search(r'idx\((.*?)\)=([0-9_]+)', line)
    if not match:
        return None  # Line format not as expected

    sensor_names = match.group(1).split('_')
    sensor_indices = match.group(2).split('_')

    # Map sensor names to their indices
    sensor_mapping = {}
    for name, index in zip(sensor_names, sensor_indices):
        if name in sensors:
            sensor_mapping[name] = index

    return sensor_mapping


def generate_file_paths(cfg) -> tuple[list[list], list[list], list[list], list[list]]:
    if cfg.SUBSET_MODE == 'range':
        subsets = range(cfg.SUBSET_RANGE[0], cfg.SUBSET_RANGE[1])

    elif cfg.SUBSET_MODE == 'choice':
        subsets = cfg.SUBSET_CHOICE

    elif cfg.SUBSET_MODE == 'type':
        raise NotImplementedError("Subset mode 'type' is not implemented yet")
    
    elif cfg.SUBSET_MODE == 'all':
        subsets = range(1, 59)

    else:
        raise NotImplementedError("Subset mode not implemented")
    
    data_name = cfg.DATA_NAME
    extension = cfg.DATA_EXTENSION

    tess_file_paths = []
    cam_file_paths = []
    ldr_file_paths = []
    gt_file_paths = []

    for subset in subsets:
        subset_labels_path = f'{cfg.PATH_TO_LABELS}/{subset}'

        if not os.path.exists(subset_labels_path):
            raise FileNotFoundError(f"Labels path for subset {subset} does not exist: {subset_labels_path}")

        label_files = sorted(
            [f for f in os.listdir(subset_labels_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )

        subset_tess_file_paths = []
        subset_cam_file_paths = []
        subset_ldr_file_paths = []
        subset_gt_file_paths = []


        if cfg.DATA == "processed":
            path = cfg.PATH_TO_PROCESSED_DATASET
        else:
            path = cfg.PATH_TO_PROCESSED_DATASET
        
        for label_file in label_files:
            with open(f'{subset_labels_path}/{label_file}', 'r') as f:
                label_data = f.readline()

                # Mapping order is tesseract, cam-front, and os2-64
                mapping = extract_sensor_numbers(label_data)
                
                if mapping is None:
                    raise ValueError(f"Could not extract sensor numbers from line: {label_data}")
                
                # Map the sensor names to their indices
                for sensor_name, sensor_index in mapping.items():
                    tesseract_path = f'{path}/{subset}/{data_name}/{data_name}_{sensor_index}{extension}'
                    if os.path.exists(tesseract_path) is False:
                        continue
                    if sensor_name == 'tesseract':
                        tesseract_path = f'{path}/{subset}/{data_name}/{data_name}_{sensor_index}{extension}'
                        if os.path.exists(tesseract_path) is False:
                            print(f"Warning: Tesseract path does not exist\n{tesseract_path}")
                        subset_tess_file_paths.append(tesseract_path)
                    elif sensor_name == 'cam-front':
                        subset_cam_file_paths.append(f'{path}/{subset}/{sensor_name}/{sensor_name}_{sensor_index}.png')
                    elif sensor_name == 'os2-64':
                        subset_ldr_file_paths.append(f'{path}/{subset}/{sensor_name}/{sensor_name}_{sensor_index}.pcd')

            subset_gt_file_paths.append(f'{subset_labels_path}/{label_file}')

        tess_file_paths.append(subset_tess_file_paths)
        cam_file_paths.append(subset_cam_file_paths)
        ldr_file_paths.append(subset_ldr_file_paths)
        gt_file_paths.append(subset_gt_file_paths)

    return tess_file_paths, cam_file_paths, ldr_file_paths, gt_file_paths


def train_test_split(cfg, tess_file_paths: list[list], cam_file_paths: list[list],
                     ldr_file_paths: list[list], gt_file_paths: list[list]):
    train_tess_file_paths = []
    train_cam_file_paths = []
    train_ldr_file_paths = []
    train_gt_file_paths = []

    test_tess_file_paths = []
    test_cam_file_paths = []
    test_ldr_file_paths = []
    test_gt_file_paths = []

    test_size = cfg.TEST_SET_SIZE
    random_state = cfg.SEED

    for subset_tess_files, subset_cam_files, subset_ldr_files, subset_gt_files \
        in zip(tess_file_paths, cam_file_paths, ldr_file_paths, gt_file_paths):
        
        # Zip together to keep correspondence
        all_files = list(zip(subset_tess_files, subset_cam_files, subset_ldr_files, subset_gt_files))
        train_files, test_files = sklearn_train_test_split(all_files, test_size=test_size, random_state=random_state)

        # Unzip back to separate lists
        if train_files:
            t, c, l, g = zip(*train_files)
            train_tess_file_paths.extend(t)
            train_cam_file_paths.extend(c)
            train_ldr_file_paths.extend(l)
            train_gt_file_paths.extend(g)
        if test_files:
            t, c, l, g = zip(*test_files)
            test_tess_file_paths.extend(t)
            test_cam_file_paths.extend(c)
            test_ldr_file_paths.extend(l)
            test_gt_file_paths.extend(g)

    return train_tess_file_paths, train_cam_file_paths, train_ldr_file_paths, train_gt_file_paths, \
           test_tess_file_paths, test_cam_file_paths, test_ldr_file_paths, test_gt_file_paths


def save_test_split(cfg, test_tess_file_paths: list, test_cam_file_paths: list,
                    test_ldr_file_paths: list, test_gt_file_paths: list):
    save_path = cfg.PATH_TO_TEST_SET

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # additional_string = "_reduced" if cfg.IS_VALIDATE else ""
    additional_string = ""

    with open(f'{save_path}/test{additional_string}_tess_file_paths.txt', 'w') as f:
        for path in test_tess_file_paths:
            f.write(f"{path}\n")

    with open(f'{save_path}/test{additional_string}_cam_file_paths.txt', 'w') as f:
        for path in test_cam_file_paths:
            f.write(f"{path}\n")

    with open(f'{save_path}/test{additional_string}_ldr_file_paths.txt', 'w') as f:
        for path in test_ldr_file_paths:
            f.write(f"{path}\n")

    with open(f'{save_path}/test{additional_string}_gt_file_paths.txt', 'w') as f:
        for path in test_gt_file_paths:
            f.write(f"{path}\n")


def load_test_split(cfg):
    load_path = cfg.PATH_TO_TEST_SET

    def read_paths(filename):
        with open(os.path.join(load_path, filename), "r") as f:
            return [line.strip() for line in f if line.strip()]

    if cfg.IS_PARTIAL_TEST and not cfg.IS_SAMPLE_TEST and not cfg.IS_VALIDATE:
        part_number = cfg.PART_NUMBER
        test_tess_file_paths = read_paths(f"test_tess_file_paths_subpart_{part_number}.txt")
        test_cam_file_paths = read_paths(f"test_cam_file_paths_subpart_{part_number}.txt")
        test_ldr_file_paths = read_paths(f"test_ldr_file_paths_subpart_{part_number}.txt")
        test_gt_file_paths = read_paths(f"test_gt_file_paths_subpart_{part_number}.txt")
    
    if cfg.IS_SAMPLE_TEST and not cfg.IS_PARTIAL_TEST and not cfg.IS_VALIDATE:
        test_tess_file_paths = read_paths("test_tess_file_paths_sampled.txt")
        test_cam_file_paths = read_paths("test_cam_file_paths_sampled.txt")
        test_ldr_file_paths = read_paths("test_ldr_file_paths_sampled.txt")
        test_gt_file_paths = read_paths("test_gt_file_paths_sampled.txt")
    
    # if cfg.IS_VALIDATE and not cfg.IS_PARTIAL_TEST and not cfg.IS_SAMPLE_TEST:
    #     test_tess_file_paths = read_paths("test_reduced_tess_file_paths.txt")
    #     test_cam_file_paths = read_paths("test_reduced_cam_file_paths.txt")
    #     test_ldr_file_paths = read_paths("test_reduced_ldr_file_paths.txt")
    #     test_gt_file_paths = read_paths("test_reduced_gt_file_paths.txt")
    
    # The logic is okay here because if no flag is chosen, we load the full test set
    # cfg.IS_VALIDATE should only be switched on if we want to validate while training
    if not cfg.IS_PARTIAL_TEST and not cfg.IS_SAMPLE_TEST and not cfg.IS_VALIDATE:
        test_tess_file_paths = read_paths("test_tess_file_paths.txt")
        test_cam_file_paths = read_paths("test_cam_file_paths.txt")
        test_ldr_file_paths = read_paths("test_ldr_file_paths.txt")
        test_gt_file_paths = read_paths("test_gt_file_paths.txt")

    if test_cam_file_paths is None or test_cam_file_paths is None or test_ldr_file_paths is None or test_gt_file_paths is None:
        raise ValueError("One of the test file paths is None. Check configuration.")

    return test_tess_file_paths, test_cam_file_paths, test_ldr_file_paths, test_gt_file_paths


def get_paths(cfg, save=True, validate=False) -> tuple[list, list, list, list]:
    if cfg.MODE == 'train':
        if cfg.SPLIT == 'self':
            if validate:
                raise NotImplementedError("Self split with validation not implemented yet")
            
            tess, cam, ldr, gt = generate_file_paths(cfg)

            train_tess, train_cam, train_ldr, train_gt, \
            test_tess, test_cam, test_ldr, test_gt = train_test_split(cfg, tess, cam, ldr, gt)

            if save:
                save_test_split(cfg, test_tess, test_cam, test_ldr, test_gt)
            
            return train_tess, train_cam, train_ldr, train_gt
        
        elif cfg.SPLIT == 'kradar':
            if validate:
                val_tp, val_cp, val_lp, val_gp = setup_kradar_split(cfg, "val.txt")

                return val_tp, val_cp, val_lp, val_gp
            else:
                if cfg.IS_VALIDATE:
                    train_tp, train_cp, train_lp, train_gp = setup_kradar_split(cfg, "train_reduced.txt")
                else:
                    train_tp, train_cp, train_lp, train_gp = setup_kradar_split(cfg, "train.txt")
            
                if save:
                    # if cfg.IS_VALIDATE:
                    #     test_tp, test_cp, test_lp, test_gp = setup_kradar_split("test_reduced.txt")
                    # else:
                    #     test_tp, test_cp, test_lp, test_gp = setup_kradar_split("test.txt")
                    
                    test_tp, test_cp, test_lp, test_gp = setup_kradar_split(cfg, "test.txt")
                    save_test_split(cfg, test_tp, test_cp, test_lp, test_gp)

                return train_tp, train_cp, train_lp, train_gp
            
        else:
            raise NotImplementedError("Split method not implemented")

    elif cfg.MODE == 'test' or cfg.MODE == 'visualize_model':
        return load_test_split(cfg)
    

def setup_kradar_split(cfg, filename: str) -> tuple[list, list, list, list]:
    with open(f"{cfg.PATH_TO_EXPERIMENTS}/{filename}", "r") as f:
        if cfg.SELECT_SUBSET == 'all':
            train_split = [line.strip() for line in f]
        else:
            train_split = [line.strip() for line in f if line.startswith(cfg.SELECT_SUBSET + ',')]

    if cfg.SELECT_SUBSET == 'all':
        print("loaded all subsets for kradar split")
    else:
        print(f"loaded subset {cfg.SELECT_SUBSET} for kradar split")

    subsets = [line.split(',')[0] for line in train_split]
    gts = [line.split(',')[1] for line in train_split]

    gt_file_paths = []
    path_labels = cfg.PATH_TO_LABELS

    for subset, gt in zip(subsets, gts):
        gt_file_paths.append(f"{path_labels}/{subset}/{gt}")

    tess_file_paths = []
    cam_file_paths = []
    ldr_file_paths = []

    for path in gt_file_paths:
        with open(path, 'r') as f:
            label_data = f.readline()

        # Mapping order is tesseract, cam-front, and os2-64
        mapping = extract_sensor_numbers(label_data)
                
        if mapping is None:
            raise ValueError(f"Could not extract sensor numbers from line: {label_data}")
        
        subset = path.split('/')[-2]
        # Map the sensor names to their indices
        for sensor_name, sensor_index in mapping.items():
            if sensor_name == 'tesseract':
                tesseract_path = f'{cfg.PATH_TO_PROCESSED_DATASET}/{subset}/DERA_tesseract/DERA_tesseract_{sensor_index}.npy'
                if os.path.exists(tesseract_path) is False:
                    print(f"Warning: Tesseract path does not exist\n{tesseract_path}")
                tess_file_paths.append(tesseract_path)
            elif sensor_name == 'cam-front':
                cam_file_paths.append(f'{cfg.PATH_TO_PROCESSED_DATASET}/{subset}/{sensor_name}/{sensor_name}_{sensor_index}.png')
            elif sensor_name == 'os2-64':
                ldr_file_paths.append(f'{cfg.PATH_TO_PROCESSED_DATASET}/{subset}/{sensor_name}/{sensor_name}_{sensor_index}.pcd')

    return tess_file_paths, cam_file_paths, ldr_file_paths, gt_file_paths


if __name__ == "__main__":
    pass