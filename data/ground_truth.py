import os
import numpy as np
import torch

from ops.roi import NARROW, box_center_in_roi
from utils.evaluation import SUBSET_INFO


def is_gt_in_fov(cfg, x, y, range_limit=118):
    azi = -np.arctan2(y, x)
    rng = np.sqrt(x ** 2 + y ** 2)
    azi = np.rad2deg(azi)

    # Default is [-53, 53]
    fov_min, fov_max = cfg.FOV

    if azi < fov_min or azi > fov_max:
        return False
    
    if x < 0 or rng > range_limit:
        return False
    
    return True


def get_calib_values(path_calib, z_offset=0.0):
    f = open(path_calib, 'r')
    lines = f.readlines()
    f.close()
    list_calib = list(map(lambda x: float(x), lines[1].split(',')))
    list_values = [list_calib[1], list_calib[2], z_offset] # X, Y, Z
    return list_values


def extract_subset_from_path(gt_path):
    # Example: .../tools/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility/1/00533_00501.txt
    # Split the path into components
    parts = gt_path.split(os.sep)
    
    # Second last part is the subset number
    return parts[-2]


def read_gt_labels(cfg, gt_path, calib=False, path_calib=None, z_offset=0.0):
    version = cfg.LABEL_VERSION

    f = open(gt_path)
    lines = f.readlines()
    f.close()
    list_tuple_objs = []
    deg2rad = np.pi/180.
    header = (lines[0]).rstrip('\n')
    try:
            temp_idx, tstamp = header.split(', ')
    except: # line breaking error for v2_0
        _, header_prime, line0 = header.split('*')
        header = '*' + header_prime
        temp_idx, tstamp = header.split(', ')
        # print('* b4: ', lines)
        lines.insert(1, '*'+line0)
        lines[0] = header
        # print('* after: ', lines)
    rdr, ldr64, camf, ldr128, camr = temp_idx.split('=')[1].split('_')
    # if rdr != f"{index:05d}":
    #    raise ValueError(f"Wrong index in ground truth file: expected: {index} found: {rdr}")
    tstamp = tstamp.split('=')[1]

    if version == 'v1_1':
        for line in lines[1:]:

            list_vals = line.rstrip('\n').split(',')
            if len(list_vals) != 12:
                print('* split err in ', gt_path)
                continue
            avail = (list_vals[1]).lstrip(' ').rstrip(' ')
            try:
                idx_p = int(list_vals[2])
                idx_b4 = int(list_vals[3])
            except:
                print('* split err in ', gt_path)
                continue
            cls_name = (list_vals[4]).lstrip(' ').rstrip(' ')
            x = float(list_vals[5])
            y = float(list_vals[6])
            z = float(list_vals[7])
            th = float(list_vals[8])*deg2rad
            l = 2*float(list_vals[9])
            w = 2*float(list_vals[10])
            h = 2*float(list_vals[11])

            # Only consider radar data if specified in params
            if cfg.RADAR_ONLY and avail != 'R':
                continue
            
            # Only consider classes which are specified in params
            if cls_name not in cfg.CLASSES_TO_USE:
                continue

            list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p, idx_b4), avail, ('rdr', 'ldr64', 'camf', 'ldr128', 'camr'), (rdr, ldr64, camf, ldr128, camr)))

    elif version == 'v2_0':
            for line in lines[1:]:

                list_vals = line.rstrip('\n').split(', ')
                idx_p = int(list_vals[1])
                cls_name = (list_vals[2])
                x = float(list_vals[3])
                y = float(list_vals[4])
                z = float(list_vals[5])
                th = float(list_vals[6])*deg2rad
                l = 2*float(list_vals[7])
                w = 2*float(list_vals[8])
                h = 2*float(list_vals[9])

                # Only consider classes which are specified in params
                if cls_name not in cfg.CLASSES_TO_USE:
                    continue

                list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), 'R', ('rdr', 'ldr64', 'camf', 'ldr128', 'camr'), (rdr, ldr64, camf, ldr128, camr)))

    elif version == 'v2_1':
        for line in lines[1:]:
            
            list_vals = line.rstrip('\n').split(', ')
            avail = list_vals[1]
            idx_p = int(list_vals[2])
            cls_name = (list_vals[3])
            x = float(list_vals[4])
            y = float(list_vals[5])
            z = float(list_vals[6])
            th = float(list_vals[7])*deg2rad # th in degree
            l = 2*float(list_vals[8])
            w = 2*float(list_vals[9])
            h = 2*float(list_vals[10])

            # Only consider radar data if specified in params
            if cfg.RADAR_ONLY and avail != 'R':
                continue
            
            # Only consider classes which are specified in params
            if cls_name not in cfg.CLASSES_TO_USE:
                continue
            
            list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), avail, ('rdr', 'ldr64', 'camf', 'ldr128', 'camr'), (rdr, ldr64, camf, ldr128, camr)))

    else:
        raise ValueError(f"Unknown label version: {version}")

    if calib:
        list_temp = []
        dx, dy, dz = get_calib_values(path_calib, z_offset)
        for obj in list_tuple_objs:
            cls_name, (x, y, z, th, l, w, h), trk, avail, (rdr_, ldr64_, camf_, ldr128_, camr_), (rdr, ldr64, camf, ldr128, camr) = obj
            x = x + dx
            y = y + dy
            z = z + dz

            # Filter by narrow ROI
            if cfg.TRAINING_ROI and not box_center_in_roi((x, y, z), NARROW):
                continue

            if is_gt_in_fov(cfg, x, y):
                list_temp.append((cls_name, (x, y, z, th, l, w, h), trk, avail, (rdr_, ldr64_, camf_, ldr128_, camr_), (rdr, ldr64, camf, ldr128, camr)))
        list_tuple_objs = list_temp
    else:
        list_temp = []
        for obj in list_tuple_objs:
            cls_name, (x, y, z, th, l, w, h), trk, avail, _, _ = obj
            
            if is_gt_in_fov(cfg, x, y):
                list_temp.append(obj)
        
        list_tuple_objs = list_temp

    return list_tuple_objs


def build_gt_tensor_for_loss(gt_list, device):
    gt_list_tensor = []

    if gt_list is None or len(gt_list) == 0:
        return torch.empty((0, 8), dtype=torch.float32, device=device)

    # GT is (x, y, z, yaw, l, w, h), where yaw is already in radians
    for gt in gt_list:
        # New tensor is (x, y, z, l, w, h, sin(yaw), cos(yaw))
        tensor = torch.zeros(8, dtype=torch.float32, device=device)
        tensor[0] = gt[1][0] # x
        tensor[1] = gt[1][1] # y
        tensor[2] = gt[1][2] # z
        tensor[3] = gt[1][4] # l
        tensor[4] = gt[1][5] # w
        tensor[5] = gt[1][6] # h
        tensor[6] = np.sin(gt[1][3])  # sin(yaw)
        tensor[7] = np.cos(gt[1][3])  # cos(yaw)
        gt_list_tensor.append(tensor)

    return torch.stack(gt_list_tensor)


def build_gt_center_points(cfg, gt_list, device):
    gt_list_tensor = []

    if gt_list is None or len(gt_list) == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    azi_fov = 106           # FOV in degrees
    azi_shift = 53          # Shift to positive index
    
    # Determine bins based on upsampling and padding
    if cfg.RDR_PROCESSING_METHOD == "upsample_azimuth":
        max_azi_idx = 213
    elif cfg.USE_DATA_PADDING:
        max_azi_idx = 111
    else:
        max_azi_idx = 106

    for gt in gt_list: 
        tensor = torch.zeros(2, dtype=torch.float32, device=device)
        x_cart = gt[1][0]  # x
        y_cart = gt[1][1]  # y

        # Convert Cartesian (x, y) to polar (range, azimuth in degrees)
        azi = -np.arctan2(y_cart, x_cart)  # azimuth in radians
        rng = np.sqrt(x_cart**2 + y_cart**2)  # range
        azi = np.rad2deg(azi)  # convert to degrees -> now in [-53°, +53°]
        
        # Shift to positive index -> [0°, 106°]
        azi += azi_shift

        # Scale from degrees to pixel indices: [0°, 106°] → [0, max_azi_idx]
        azi = azi * max_azi_idx / azi_fov

        rng = rng * 255 / 118  # Maps [0m, 118m] → [0, 255] (pixel indices)

        tensor[0] = np.clip(np.round(rng), a_min=0, a_max=255)
        tensor[1] = np.clip(np.round(azi), a_min=0, a_max=max_azi_idx)

        gt_list_tensor.append(tensor)

    return torch.stack(gt_list_tensor)


def build_ground_truth(cfg, gt_path, device):
    if gt_path is None:
        raise ValueError("Ground truth path (gt_path) cannot be None")

    subset = extract_subset_from_path(gt_path)
    calib = cfg.CALIB_GT

    # print(subset, gt_path)

    if cfg.DATA == 'tesseract':
        path_dataset = cfg.PATH_TO_DATASET
    elif cfg.DATA == 'processed':
        path_dataset = cfg.PATH_TO_PROCESSED_DATASET

    path_calib = f'{path_dataset}/{subset}/info_calib/calib_radar_lidar.txt'
    
    gt_list = read_gt_labels(cfg, gt_path, calib=calib, path_calib=path_calib, z_offset=0.0)
    gt_list_tensor = build_gt_tensor_for_loss(gt_list, device=device)
    gt_center_points = build_gt_center_points(cfg, gt_list, device=device)

    return gt_list, gt_list_tensor, gt_center_points


def get_ground_truth(cfg, batch_dict, gt_paths, device):
    gt_lists = []                       # List of ground truth objects
    gt_list_tensors = []                # List of ground truth objects in tensor format
    gt_center_points = []               # List of the center points of ground truth objects
    for gt_path in gt_paths:
        gt_list, gt_list_tensor, gt_centers = build_ground_truth(cfg, gt_path, device)
        gt_lists.append(gt_list)
        gt_list_tensors.append(gt_list_tensor)
        gt_center_points.append(gt_centers)

    batch_dict['gt_lists'] = gt_lists
    batch_dict['gt_list_tensors'] = gt_list_tensors
    batch_dict['gt_center_points'] = gt_center_points

    return batch_dict


##### Below code is for analytical purposes and not part of the main pipeline #####


def check_all_labels(cfg, apply_roi_filter=False):
    classes_dict = {}

    for i in range(1, 59):
        subset_labels_path = f'{cfg.PATH_TO_LABELS}/{i}'

        if not os.path.exists(subset_labels_path):
            raise FileNotFoundError(f"Labels path for subset {i} does not exist: {subset_labels_path}")

        label_files = sorted(
            [f for f in os.listdir(subset_labels_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )

        for label in label_files:
            label_path = os.path.join(subset_labels_path, label)
            if not os.path.isfile(label_path):
                print(f"Skipping non-file: {label_path}")
                continue
            
            gt_list = read_gt_labels(cfg, label_path, calib=False, path_calib=None, z_offset=0.0)
            for gt in gt_list:
                obj_class = gt[0]
                # Filter based on narrow ROI
                gt_values = gt[1]
                x, y, z = gt_values[0], gt_values[1], gt_values[2]
                if apply_roi_filter and not (0 <= x <= 72 and -6.4 <= y <= 6.4 and -2 <= z <= 6):
                    continue
                else:
                    if obj_class not in classes_dict.keys():
                        classes_dict[obj_class] = 1
                    else:
                        classes_dict[obj_class] += 1
    
    return classes_dict


def count_frames_without_gt(cfg):
    count_no_gt = 0
    total_frames = 0

    path_to_test_split = ...
    with open(path_to_test_split, "r") as in_file:
        test_split = [line.strip() for line in in_file]
    from tqdm import tqdm
    for i in tqdm(range(1, 59), desc="Processing Subsets"):
        subset_labels_path = f'{cfg.PATH_TO_LABELS}/{i}'

        if not os.path.exists(subset_labels_path):
            raise FileNotFoundError(f"Labels path for subset {i} does not exist: {subset_labels_path}")

        label_files = sorted(
            [f for f in os.listdir(subset_labels_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )

        for label in label_files:
            string = f"{i},{label}"
            if string not in test_split:
                continue

            label_path = os.path.join(subset_labels_path, label)
            if not os.path.isfile(label_path):
                print(f"Skipping non-file: {label_path}")
                continue

            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) <= 1:  # Only header, no GT objects
                count_no_gt += 1
            
            # gt_list = read_gt_labels(cfg, label_path, calib=False, path_calib=None, z_offset=0.0)
            # total_frames += 1
            # if len(gt_list) == 0:
            #     count_no_gt += 1

        # Delete finished subset in test split to speed up lookup
        test_split = [s for s in test_split if not s.startswith(f"{i},")]

    print(f"Total frames: {total_frames}, Frames without GT: {count_no_gt}, Percentage without GT: {100 * count_no_gt / total_frames:.2f}%")


def count_objects_in_dataset(cfg):
    path_dataset = cfg.PATH_TO_PROCESSED_DATASET
    path_split = cfg.PATH_TO_EXPERIMENTS
    path_labels = cfg.PATH_TO_LABELS
    
    total_objects = 0
    total_objects_train = 0
    total_objects_test = 0

    total_radar_objects = 0
    total_radar_objects_train = 0
    total_radar_objects_test = 0

    total_radar_objects_in_roi = 0
    total_radar_objects_in_roi_train = 0
    total_radar_objects_in_roi_test = 0

    total_counts_per_class = {cls: 0 for cls in cfg.CLASSES_TO_USE}
    total_counts_per_class_train = {cls: 0 for cls in cfg.CLASSES_TO_USE}
    total_counts_per_class_test = {cls: 0 for cls in cfg.CLASSES_TO_USE}

    radar_counts_per_class = {cls: 0 for cls in cfg.CLASSES_TO_USE}
    radar_counts_per_class_train = {cls: 0 for cls in cfg.CLASSES_TO_USE}
    radar_counts_per_class_test = {cls: 0 for cls in cfg.CLASSES_TO_USE}

    radar_in_roi_counts_class_counts = {cls: 0 for cls in cfg.CLASSES_TO_USE}
    radar_in_roi_counts_class_counts_train = {cls: 0 for cls in cfg.CLASSES_TO_USE}
    radar_in_roi_counts_class_counts_test = {cls: 0 for cls in cfg.CLASSES_TO_USE}

    with open(f"{path_split}/train.txt", "r") as in_file:
        train_split = [line.strip() for line in in_file]

    with open(f"{path_split}/test.txt", "r") as in_file:
        test_split = [line.strip() for line in in_file]
    
    for i in range(1, 59):
        subset_labels_path = f'{path_labels}/{i}'

        if not os.path.exists(subset_labels_path):
            raise FileNotFoundError(f"Labels path for subset {i} does not exist: {subset_labels_path}")

        label_files = sorted(
            [f for f in os.listdir(subset_labels_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )

        for label in label_files:
            label_path = os.path.join(subset_labels_path, label)
            if not os.path.isfile(label_path):
                print(f"Skipping non-file: {label_path}")
                continue

            path_calib = f'{path_dataset}/{i}/info_calib/calib_radar_lidar.txt'

            lookup = f"{i},{label}"

            # (1) Get all objects from GT file
            gt_list = read_gt_labels(cfg, label_path, calib=True, path_calib=path_calib, z_offset=0.0)
            for gt in gt_list:
                gt_class = gt[0]
                if gt_class not in cfg.CLASSES_TO_USE:
                    continue
                gt_values = gt[1]
                x, y, z = gt_values[0], gt_values[1], gt_values[2]
                total_objects += 1
                total_counts_per_class[gt_class] += 1

                train, test = False, False
                if lookup in train_split:
                    total_objects_train += 1
                    total_counts_per_class_train[gt_class] += 1
                    train = True
                elif lookup in test_split:
                    total_objects_test += 1
                    total_counts_per_class_test[gt_class] += 1
                    test = True
                else:
                    raise ValueError(f"GT file {lookup} not found in either train or test split")

                # (2) Filter by radar
                if gt[3] == 'R':
                    total_radar_objects += 1
                    radar_counts_per_class[gt_class] += 1
                    if train:
                        total_radar_objects_train += 1
                        radar_counts_per_class_train[gt_class] += 1
                    elif test:
                        total_radar_objects_test += 1
                        radar_counts_per_class_test[gt_class] += 1

                    # (3) Filter by narrow ROI
                    if box_center_in_roi((x, y, z), NARROW):
                        total_radar_objects_in_roi += 1
                        radar_in_roi_counts_class_counts[gt_class] += 1
                        if train:
                            total_radar_objects_in_roi_train += 1
                            radar_in_roi_counts_class_counts_train[gt_class] += 1
                        elif test:
                            total_radar_objects_in_roi_test += 1
                            radar_in_roi_counts_class_counts_test[gt_class] += 1

    print(f"Total objects: {total_objects}, Objects in train: {total_objects_train}, Objects in test: {total_objects_test}")
    print(f"Total radar objects: {total_radar_objects}, Radar objects in train: {total_radar_objects_train}, Radar objects in test: {total_radar_objects_test}")
    print(f"Total radar objects in ROI: {total_radar_objects_in_roi}, Radar objects in ROI in train: {total_radar_objects_in_roi_train}, Radar objects in ROI in test: {total_radar_objects_in_roi_test}")
    
    print("Class counts:")
    for cls, count in total_counts_per_class.items():
        print(f"{cls}: {count}")
    print("Class counts in train:")
    for cls, count in total_counts_per_class_train.items():
        print(f"{cls}: {count}")
    print("Class counts in test:")
    for cls, count in total_counts_per_class_test.items():
        print(f"{cls}: {count}")

    print("Radar class counts:")
    for cls, count in radar_counts_per_class.items():
        print(f"{cls}: {count}")
    print("Radar class counts in train:")
    for cls, count in radar_counts_per_class_train.items():
        print(f"{cls}: {count}")
    print("Radar class counts in test:")
    for cls, count in radar_counts_per_class_test.items():
        print(f"{cls}: {count}")

    print("Radar class counts in ROI:")
    for cls, count in radar_in_roi_counts_class_counts.items():
        print(f"{cls}: {count}")
    print("Radar class counts in ROI in train:")
    for cls, count in radar_in_roi_counts_class_counts_train.items():
        print(f"{cls}: {count}")
    print("Radar class counts in ROI in test:")
    for cls, count in radar_in_roi_counts_class_counts_test.items():
        print(f"{cls}: {count}")

    assert(total_objects == total_objects_train + total_objects_test), \
    f"Total objects ({total_objects}) does not match sum of train and test objects ({total_objects_train + total_objects_test})"
    assert(total_radar_objects == total_radar_objects_train + total_radar_objects_test), \
    f"Total radar objects ({total_radar_objects}) does not match sum of train and test radar objects ({total_radar_objects_train + total_radar_objects_test})"
    assert(total_radar_objects_in_roi == total_radar_objects_in_roi_train + total_radar_objects_in_roi_test), \
    f"Radar objects in ROI ({total_radar_objects_in_roi}) does not match sum of train and test radar objects in ROI ({total_radar_objects_in_roi_train + total_radar_objects_in_roi_test})"

    assert(sum(total_counts_per_class.values()) == total_objects), \
    f"Sum of class counts ({sum(total_counts_per_class.values())}) does not match total objects ({total_objects})"
    assert(sum(total_counts_per_class_train.values()) == total_objects_train), \
    f"Sum of train class counts ({sum(total_counts_per_class_train.values())}) does not match total train objects ({total_objects_train})"
    assert(sum(total_counts_per_class_test.values()) == total_objects_test), \
    f"Sum of test class counts ({sum(total_counts_per_class_test.values())}) does not match total test objects ({total_objects_test})"

    assert(sum(radar_counts_per_class.values()) == total_radar_objects), \
    f"Sum of radar class counts ({sum(radar_counts_per_class.values())}) does not match total radar objects ({total_radar_objects})"
    assert(sum(radar_counts_per_class_train.values()) == total_radar_objects_train), \
    f"Sum of train radar class counts ({sum(radar_counts_per_class_train.values())}) does not match total train radar objects ({total_radar_objects_train})"
    assert(sum(radar_counts_per_class_test.values()) == total_radar_objects_test), \
    f"Sum of test radar class counts ({sum(radar_counts_per_class_test.values())}) does not match total test radar objects ({total_radar_objects_test})"

    assert(sum(radar_in_roi_counts_class_counts.values()) == total_radar_objects_in_roi), \
    f"Sum of radar class counts in ROI ({sum(radar_in_roi_counts_class_counts.values())}) does not match total radar objects in ROI ({total_radar_objects_in_roi})"
    assert(sum(radar_in_roi_counts_class_counts_train.values()) == total_radar_objects_in_roi_train), \
    f"Sum of train radar class counts in ROI ({sum(radar_in_roi_counts_class_counts_train.values())}) does not match total train radar objects in ROI ({total_radar_objects_in_roi_train})"
    assert(sum(radar_in_roi_counts_class_counts_test.values()) == total_radar_objects_in_roi_test), \
    f"Sum of test radar class counts in ROI ({sum(radar_in_roi_counts_class_counts_test.values())}) does not match total test radar objects in ROI ({total_radar_objects_in_roi_test})"


def count_objects_in_dataset_indepth(cfg):
    """
    Count class occurrences for objects, radar, and radar_roi in all environments, times, and weathers.
    """

    ENVIRONMENTS = [
        'urban', 'highway', 'alleyway', 'university',
        'countryside', 'mountain', 'parkinglots', 'shoulder'
    ]
    TIMES = ['day', 'night']
    WEATHERS = [
        'normal', 'rain', 'overcast', 'sleet', 'fog', 'lightsnow', 'heavysnow'
    ]
    LEVELS = ['objects', 'radar', 'radar_roi']
    SPLITS = ['total', 'train', 'test']
    ATTRIBUTES = {'environment': ENVIRONMENTS, 'time': TIMES, 'weather': WEATHERS}
    CLASSES = cfg.CLASSES_TO_USE

    path_split = cfg.PATH_TO_EXPERIMENTS
    path_dataset = cfg.PATH_TO_PROCESSED_DATASET
    path_labels = cfg.PATH_TO_LABELS
    
    # Nested dict: counters[level][split][attribute][attribute_value][class] = count
    counters = {
        level: {
            split: {
                attribute: {
                    value: {cls: 0 for cls in CLASSES}
                    for value in ATTRIBUTES[attribute]
                }
                for attribute in ATTRIBUTES
            }
            for split in SPLITS
        }
        for level in LEVELS
    }

    # Load splits
    with open(f"{path_split}/train.txt") as f:
        train_split = set(line.strip() for line in f)
    with open(f"{path_split}/test.txt") as f:
        test_split = set(line.strip() for line in f)

    # Loop over dataset
    for subset_id in range(1, 59):
        subset_labels_path = f'{path_labels}/{subset_id}'
        if not os.path.exists(subset_labels_path):
            raise FileNotFoundError(f"Labels path for subset {subset_id} does not exist: {subset_labels_path}")
        label_files = sorted(
            [f for f in os.listdir(subset_labels_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )
        info = SUBSET_INFO[subset_id]
        environment, time, weather = info["environment"], info["time"], info["weather"]
        for label in label_files:
            label_path = os.path.join(subset_labels_path, label)
            if not os.path.isfile(label_path):
                continue
            lookup = f"{subset_id},{label}"
            path_calib = f'{path_dataset}/{subset_id}/info_calib/calib_radar_lidar.txt'
            # Determine split
            if lookup in train_split:
                split = 'train'
            elif lookup in test_split:
                split = 'test'
            else:
                raise ValueError(f"GT file {lookup} not in train or test lists")

            gt_list = read_gt_labels(cfg, label_path, calib=True, path_calib=path_calib, z_offset=0.0)
            for gt in gt_list:
                gt_class = gt[0]
                if gt_class not in CLASSES:
                    continue
                # All objects
                for attr, val in zip(['environment', 'time', 'weather'], [environment, time, weather]):
                    counters['objects']['total'][attr][val][gt_class] += 1
                    counters['objects'][split][attr][val][gt_class] += 1
                # Radar objects
                if gt[3] == 'R':
                    for attr, val in zip(['environment', 'time', 'weather'], [environment, time, weather]):
                        counters['radar']['total'][attr][val][gt_class] += 1
                        counters['radar'][split][attr][val][gt_class] += 1
                    # Radar ROI
                    x, y, z = gt[1][:3]
                    if box_center_in_roi((x, y, z), NARROW):
                        for attr, val in zip(['environment', 'time', 'weather'], [environment, time, weather]):
                            counters['radar_roi']['total'][attr][val][gt_class] += 1
                            counters['radar_roi'][split][attr][val][gt_class] += 1

    # ---- Print summary ----
    for level in LEVELS:
        print(f"\n{level}")
        for split in SPLITS:
            print(f"  {split}")
            for cl in CLASSES:
                print(f"    {cl}")
                print("      Environments:")
                for env in ENVIRONMENTS:
                    print(f"        {env:12}: {counters[level][split]['environment'][env][cl]}")
                print("      Times:")
                for t in TIMES:
                    print(f"        {t:12}: {counters[level][split]['time'][t][cl]}")
                print("      Weathers:")
                for w in WEATHERS:
                    print(f"        {w:12}: {counters[level][split]['weather'][w][cl]}")

    return counters


def count_objects_in_dataset_by_distance(cfg):
    """
    Count class occurrences by distance bins for objects, radar, and radar_roi in all environments, times, and weathers.
    """
    LEVELS = ['objects', 'radar', 'radar_roi']
    SPLITS = ['total', 'train', 'test']
    CLASSES = cfg.CLASSES_TO_USE
    DISTANCES = ['0~20m', '20~40m', '40~60m', '60~80m', '80~100m', '100m+']

    path_split = cfg.PATH_TO_EXPERIMENTS
    path_dataset = cfg.PATH_TO_PROCESSED_DATASET
    path_labels = cfg.PATH_TO_LABELS
    
    # Nested dict: counters[level][split][cls][distance] = count
    counters = {
        level: {
            split: {
                cls: {distance: 0 for distance in DISTANCES}
                for cls in CLASSES
            }
            for split in SPLITS
        }
        for level in LEVELS
    }

    # Load splits
    with open(f"{path_split}/train.txt") as f:
        train_split = set(line.strip() for line in f)
    with open(f"{path_split}/test.txt") as f:
        test_split = set(line.strip() for line in f)
    
    # Loop over dataset
    for subset_id in range(1, 59):
        subset_labels_path = f'{path_labels}/{subset_id}'
        if not os.path.exists(subset_labels_path):
            raise FileNotFoundError(f"Labels path for subset {subset_id} does not exist: {subset_labels_path}")
        label_files = sorted(
            [f for f in os.listdir(subset_labels_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )

        for label in label_files:
            label_path = os.path.join(subset_labels_path, label)
            if not os.path.isfile(label_path):
                continue
            lookup = f"{subset_id},{label}"
            path_calib = f'{path_dataset}/{subset_id}/info_calib/calib_radar_lidar.txt'
            # Determine split
            if lookup in train_split:
                split = 'train'
            elif lookup in test_split:
                split = 'test'
            else:
                raise ValueError(f"GT file {lookup} not in train or test lists")

            gt_list = read_gt_labels(cfg, label_path, calib=True, path_calib=path_calib, z_offset=0.0)
            for gt in gt_list:
                gt_class = gt[0]
                if gt_class not in CLASSES:
                    continue
                # All objects
                x, y, z = gt[1][:3]
                dist = np.sqrt(x**2 + y**2)
                if dist <= 20:
                    dist_bin = '0~20m'
                elif dist <= 40:
                    dist_bin = '20~40m'
                elif dist <= 60:
                    dist_bin = '40~60m'
                elif dist <= 80:
                    dist_bin = '60~80m'
                elif dist <= 100:
                    dist_bin = '80~100m'
                else:
                    dist_bin = '100m+'
                counters['objects'][split][gt_class][dist_bin] += 1
                counters['objects']['total'][gt_class][dist_bin] += 1

                # Radar objects
                if gt[3] == 'R':
                    counters['radar'][split][gt_class][dist_bin] += 1
                    counters['radar']['total'][gt_class][dist_bin] += 1

                    # Radar ROI
                    if box_center_in_roi((x, y, z), NARROW):
                        counters['radar_roi'][split][gt_class][dist_bin] += 1
                        counters['radar_roi']['total'][gt_class][dist_bin] += 1
    
    # ---- Print summary ----
    for level in LEVELS:
        print(f"\n{level}")
        for split in SPLITS:
            print(f"  {split}")
            for cl in CLASSES:
                print(f"    {cl}")
                for dist in DISTANCES:
                    print(f"      {dist:8}: {counters[level][split][cl][dist]}")


def get_pedestrian_frames_in_test_set_in_roi(cfg):
    # Read test set
    with open("/home/wsluser/workspace/RADE-Net_private/experiments/test.txt", "r") as in_file:
        test_split = [line.strip() for line in in_file]
    
    for frame in test_split:
        subset, filename = frame.split(',')
        if int(subset) not in [5, 7, 16, 17, 19, 20, 21]:
            continue
        label_path = f"/home/wsluser/workspace/RADE-Net_private/labels/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility/{subset}/{filename}"
        gt_list = read_gt_labels(cfg, label_path, calib=False, path_calib=None, z_offset=0.0)
        for gt in gt_list:
            cls_name = gt[0]
            if cls_name == 'Pedestrian':
                x, y, z = gt[1][:3]
                if box_center_in_roi((x, y, z), NARROW):
                    print(f"Pedestrian in ROI found in frame: {frame} and cam_image: {gt[5][2]}") # gt[5][2] is camf
                    break    


def roughly_estimate_occluded_frames():
    partially = [25, 26, 27, 28, 31, 33, 36, 44, 45, 48, 50, 54, 55]
    strongly = [29, 35, 39, 40, 42, 43, 49, 56]
    fully = [30, 37, 46, 47, 51, 52, 53, 57, 58]

    path_to_labels = "/home/wsluser/workspace/RADE-Net_private/labels/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility"
    
    partially_count = 0
    strongly_count = 0
    fully_count = 0

    for subset in partially:
        label_path = f"{path_to_labels}/{subset}"
        label_files = sorted(
            [f for f in os.listdir(label_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )
        partially_count += len(label_files)

    for subset in strongly:
        label_path = f"{path_to_labels}/{subset}"
        label_files = sorted(
            [f for f in os.listdir(label_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )
        strongly_count += len(label_files)

    for subset in fully:
        label_path = f"{path_to_labels}/{subset}"
        label_files = sorted(
            [f for f in os.listdir(label_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )
        fully_count += len(label_files)

    print(f"Partially occluded frames: {partially_count}")
    print(f"Strongly occluded frames: {strongly_count}")
    print(f"Fully occluded frames: {fully_count}")


def pedestrian_frames_per_train_test_split_in_normal_condition(cfg):
    path_to_labels = "/home/wsluser/workspace/RADE-Net_private/labels/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility"
    path_split = "/home/wsluser/workspace/RADE-Net_private/experiments"

    with open(f"{path_split}/train.txt", "r") as in_file:
        train_split = [line.strip() for line in in_file]

    with open(f"{path_split}/test.txt", "r") as in_file:
        test_split = [line.strip() for line in in_file]

    dic = {}
    for subset in [5, 7, 16, 17, 19, 20]: # Only consider normal condition subsets
        label_path = f"{path_to_labels}/{subset}"
        label_files = sorted(
            [f for f in os.listdir(label_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )
        dic[subset] = {"train": 0, "train_frame_numbers": [], "train_gt": [], "test": 0, "test_frame_numbers": [], "test_gt": []}
        for label in label_files:
            frame_id = f"{subset},{label}"
            gt_list = read_gt_labels(cfg, f"{label_path}/{label}", calib=False, path_calib=None, z_offset=0.0)
            has_pedestrian = any(gt[0] == 'Pedestrian' for gt in gt_list)
            if has_pedestrian:
                if frame_id in train_split:
                    dic[subset]["train"] += 1
                    dic[subset]["train_frame_numbers"].append(gt_list[0][5][2]) # camf
                    dic[subset]["train_gt"].append(frame_id)
                elif frame_id in test_split:
                    dic[subset]["test"] += 1
                    dic[subset]["test_frame_numbers"].append(gt_list[0][5][2]) # camf
                    dic[subset]["test_gt"].append(frame_id)
                else:
                    raise ValueError(f"Frame {frame_id} not found in either train or test split")

    print("Pedestrian frames in normal condition subsets:")
    for subset, counts in dic.items():
        print(f"Subset {subset:>2} - Train: {counts['train']:>3}, Test: {counts['test']:>3}")
        print("Train frames with pedestrians:")
        for frame, file in zip(counts['train_frame_numbers'], counts['train_gt']):
            print(f"Frame: {frame} | Label file: {file}")
        print("Test frames with pedestrians:")
        for frame, file in zip(counts['test_frame_numbers'], counts['test_gt']):
            print(f"Frame: {frame} | Label file: {file}")


def inspect_label_per_sensor_availability(cfg):
    for i in range(1, 59):
        r_avail = 0
        l_avail = 0
        l1_avail = 0

        subset_labels_path = f'{cfg.PATH_TO_LABELS}/{i}'

        if not os.path.exists(subset_labels_path):
            raise FileNotFoundError(f"Labels path for subset {i} does not exist: {subset_labels_path}")

        label_files = sorted(
            [f for f in os.listdir(subset_labels_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )

        for label in label_files:
            label_path = os.path.join(subset_labels_path, label)
            if not os.path.isfile(label_path):
                print(f"Skipping non-file: {label_path}")
                continue
            
            gt_list = read_gt_labels(cfg, label_path, calib=False, path_calib=None, z_offset=0.0)
            for gt in gt_list:
                sensor_avail = gt[3]

                if sensor_avail == 'R':
                    r_avail += 1
                elif sensor_avail == 'L':
                    l_avail += 1
                elif sensor_avail == 'L1':
                    l1_avail += 1
                else:
                    print(f"Unknown sensor availability value: {sensor_avail} in file {label_path}")
        
        print(f"Subset {i:>2} - R: {r_avail:>5}, L: {l_avail:>5}, L1: {l1_avail:>5}")


def inspect_sensor_distribution(cfg):
    distance_dict = {
        'R': {
                '0~20m': 0,
                '20~40m': 0,
                '40~60m': 0,
                '60~80m': 0,
                '80~100m': 0,
                '100~120m': 0,
                '120~140m': 0,
                '140~160m': 0,
                '160~180m': 0,
                '180~200m': 0,
                '200m+': 0
        },
        'L': {
                '0~20m': 0,
                '20~40m': 0,
                '40~60m': 0,
                '60~80m': 0,
                '80~100m': 0,
                '100~120m': 0,
                '120~140m': 0,
                '140~160m': 0,
                '160~180m': 0,
                '180~200m': 0,
                '200m+': 0
        },
        'L1': {
                '0~20m': 0,
                '20~40m': 0,
                '40~60m': 0,
                '60~80m': 0,
                '80~100m': 0,
                '100~120m': 0,
                '120~140m': 0,
                '140~160m': 0,
                '160~180m': 0,
                '180~200m': 0,
                '200m+': 0
        }}
    
    for i in range(1, 59):
        subset_labels_path = f'{cfg.PATH_TO_LABELS}/{i}'

        if not os.path.exists(subset_labels_path):
            raise FileNotFoundError(f"Labels path for subset {i} does not exist: {subset_labels_path}")

        label_files = sorted(
            [f for f in os.listdir(subset_labels_path) if f.endswith('.txt')],
            key=lambda x: int(x.split('_')[0])
        )

        for label in label_files:
            label_path = os.path.join(subset_labels_path, label)
            if not os.path.isfile(label_path):
                print(f"Skipping non-file: {label_path}")
                continue
            
            gt_list = read_gt_labels(cfg, label_path, calib=False, path_calib=None, z_offset=0.0)
            for gt in gt_list:
                sensor_avail = gt[3]
                rng = gt[1][0]
                azi = gt[1][1]    
                dist = np.sqrt(rng**2 + azi**2)
                
                if sensor_avail == 'R':
                    if dist <= 20:
                        dist_bin = '0~20m'
                    elif dist <= 40:
                        dist_bin = '20~40m'
                    elif dist <= 60:
                        dist_bin = '40~60m'
                    elif dist <= 80:
                        dist_bin = '60~80m'
                    elif dist <= 100:
                        dist_bin = '80~100m'
                    elif dist <= 120:
                        dist_bin = '100~120m'
                    elif dist <= 140:
                        dist_bin = '120~140m'
                    elif dist <= 160:
                        dist_bin = '140~160m'
                    elif dist <= 180:
                        dist_bin = '160~180m'
                    elif dist <= 200:
                        dist_bin = '180~200m'
                    else:
                        dist_bin = '200m+'
                    
                    distance_dict['R'][dist_bin] += 1

                elif sensor_avail == 'L':
                    if dist <= 20:
                        dist_bin = '0~20m'
                    elif dist <= 40:
                        dist_bin = '20~40m'
                    elif dist <= 60:
                        dist_bin = '40~60m'
                    elif dist <= 80:
                        dist_bin = '60~80m'
                    elif dist <= 100:
                        dist_bin = '80~100m'
                    elif dist <= 120:
                        dist_bin = '100~120m'
                    elif dist <= 140:
                        dist_bin = '120~140m'
                    elif dist <= 160:
                        dist_bin = '140~160m'
                    elif dist <= 180:
                        dist_bin = '160~180m'
                    elif dist <= 200:
                        dist_bin = '180~200m'
                    else:
                        dist_bin = '200m+'
                    
                    distance_dict['L'][dist_bin] += 1
                elif sensor_avail == 'L1':
                    if dist <= 20:
                        dist_bin = '0~20m'
                    elif dist <= 40:
                        dist_bin = '20~40m'
                    elif dist <= 60:
                        dist_bin = '40~60m'
                    elif dist <= 80:
                        dist_bin = '60~80m'
                    elif dist <= 100:
                        dist_bin = '80~100m'
                    elif dist <= 120:
                        dist_bin = '100~120m'
                    elif dist <= 140:
                        dist_bin = '120~140m'
                    elif dist <= 160:
                        dist_bin = '140~160m'
                    elif dist <= 180:
                        dist_bin = '160~180m'
                    elif dist <= 200:
                        dist_bin = '180~200m'
                    else:
                        dist_bin = '200m+'
                    
                    distance_dict['L1'][dist_bin] += 1
                else:
                    pass
    
    print("Distance distribution for R:")
    for dist_bin, count in distance_dict['R'].items():
        print(f"{dist_bin:8}: {count}")
    print("Distance distribution for L:")
    for dist_bin, count in distance_dict['L'].items():
        print(f"{dist_bin:8}: {count}")
    print("Distance distribution for L1:")
    for dist_bin, count in distance_dict['L1'].items():
        print(f"{dist_bin:8}: {count}")
        

if __name__ == "__main__":
    class DummyCfg:
        PATH_TO_PROCESSED_DATASET = ...
        PATH_TO_EXPERIMENTS = ...
        PATH_TO_LABELS = ...
        RADAR_ONLY = False
        CLASSES_TO_USE = ['Sedan', 'Bus or Truck', 'Pedestrian', 'Motorcycle', 'Bicycle']
        LABEL_VERSION = "v2_1"
        FOV = [-53, 53]
        TRAINING_ROI = False

    cfg = DummyCfg()
    # count_objects_in_dataset(cfg)
    # count_objects_in_dataset_indepth(cfg)
    # cfg.LABEL_VERSION = "v1_1"
    # gt_path_v1_1 = "/home/wsluser/workspace/RADE-Net_private/labels/revise_label/kradar_revised_label_v1_1/1/00136_00104.txt"
    # v1 = read_gt_labels(cfg, gt_path_v1_1, calib=False, path_calib="", z_offset=0.0)
    # cfg.LABEL_VERSION = "v2_1"
    # gt_path_v2_1 = "/home/wsluser/workspace/RADE-Net_private/labels/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility/1/00136_00104.txt"
    # v2 = read_gt_labels(cfg, gt_path_v2_1, calib=False, path_calib="", z_offset=0.0)
    # print(v1)
    # print(v2)

    # get_pedestrian_frames_in_test_set_in_roi(cfg)
    # roughly_estimate_occluded_frames()
    # pedestrian_frames_per_train_test_split_in_normal_condition(cfg)
    cfg.PATH_TO_LABELS = '/home/wsluser/workspace/RADE-Net_private/labels/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility'
    # inspect_label_per_sensor_availability(cfg)
    inspect_sensor_distribution(cfg)
