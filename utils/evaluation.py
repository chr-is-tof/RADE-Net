import numpy as np
import csv
import os
import pandas as pd

from ops.rotated_iou import rotated_iou_2d, rotated_iou_3d
from ops.roi import filter_boxes_in_roi
from .plotting import plot_tp_fp_fn_total_grid, plot_tp_fp_fn_metrics_grid


SUBSET_INFO = {
    1:  {"environment": "urban",       "time": "night", "weather": "normal"},
    2:  {"environment": "highway",     "time": "night", "weather": "normal"},
    3:  {"environment": "highway",     "time": "night", "weather": "normal"},
    4:  {"environment": "highway",     "time": "night", "weather": "normal"},
    5:  {"environment": "urban",       "time": "day",   "weather": "normal"},
    6:  {"environment": "urban",       "time": "night", "weather": "normal"},
    7:  {"environment": "alleyway",    "time": "night", "weather": "normal"},
    8:  {"environment": "university",  "time": "night", "weather": "normal"},
    9:  {"environment": "highway",     "time": "day",   "weather": "normal"},
    10: {"environment": "highway",     "time": "day",   "weather": "normal"},
    11: {"environment": "highway",     "time": "day",   "weather": "normal"},
    12: {"environment": "highway",     "time": "day",   "weather": "normal"},
    13: {"environment": "highway",     "time": "day",   "weather": "overcast"},
    14: {"environment": "urban",       "time": "day",   "weather": "normal"},
    15: {"environment": "urban",       "time": "day",   "weather": "normal"},
    16: {"environment": "university",  "time": "day",   "weather": "normal"},
    17: {"environment": "university",  "time": "day",   "weather": "normal"},
    18: {"environment": "urban",       "time": "day",   "weather": "normal"},
    19: {"environment": "alleyway",    "time": "day",   "weather": "normal"},
    20: {"environment": "urban",       "time": "day",   "weather": "normal"},
    21: {"environment": "alleyway",    "time": "night", "weather": "rain"},
    22: {"environment": "urban",       "time": "night", "weather": "overcast"},
    23: {"environment": "urban",       "time": "night", "weather": "rain"},
    24: {"environment": "urban",       "time": "night", "weather": "rain"},
    25: {"environment": "urban",       "time": "night", "weather": "rain"},
    26: {"environment": "countryside", "time": "day",   "weather": "rain"},
    27: {"environment": "countryside", "time": "day",   "weather": "sleet"},
    28: {"environment": "mountain",    "time": "day",   "weather": "sleet"},
    29: {"environment": "mountain",    "time": "day",   "weather": "sleet"},
    30: {"environment": "parkinglots", "time": "day",   "weather": "sleet"},
    31: {"environment": "countryside", "time": "day",   "weather": "sleet"},
    32: {"environment": "countryside", "time": "day",   "weather": "rain"},
    33: {"environment": "countryside", "time": "day",   "weather": "rain"},
    34: {"environment": "countryside", "time": "night", "weather": "rain"},
    35: {"environment": "parkinglots", "time": "night", "weather": "sleet"},
    36: {"environment": "parkinglots", "time": "night", "weather": "sleet"},
    37: {"environment": "countryside", "time": "night", "weather": "sleet"},
    38: {"environment": "mountain",    "time": "day",   "weather": "fog"},
    39: {"environment": "mountain",    "time": "day",   "weather": "fog"},
    40: {"environment": "mountain",    "time": "day",   "weather": "fog"},
    41: {"environment": "mountain",    "time": "day",   "weather": "fog"},
    42: {"environment": "urban",       "time": "day",   "weather": "lightsnow"},
    43: {"environment": "urban",       "time": "day",   "weather": "lightsnow"},
    44: {"environment": "shoulder",    "time": "day",   "weather": "fog"},
    45: {"environment": "shoulder",    "time": "day",   "weather": "fog"},
    46: {"environment": "highway",     "time": "night", "weather": "heavysnow"},
    47: {"environment": "highway",     "time": "night", "weather": "heavysnow"},
    48: {"environment": "highway",     "time": "night", "weather": "lightsnow"},
    49: {"environment": "highway",     "time": "night", "weather": "lightsnow"},
    50: {"environment": "highway",     "time": "night", "weather": "sleet"},
    51: {"environment": "highway",     "time": "night", "weather": "sleet"},
    52: {"environment": "highway",     "time": "night", "weather": "sleet"},
    53: {"environment": "highway",     "time": "day",   "weather": "sleet"},
    54: {"environment": "urban",       "time": "day",   "weather": "heavysnow"},
    55: {"environment": "urban",       "time": "day",   "weather": "heavysnow"},
    56: {"environment": "urban",       "time": "day",   "weather": "heavysnow"},
    57: {"environment": "urban",       "time": "day",   "weather": "heavysnow"},
    58: {"environment": "urban",       "time": "day",   "weather": "heavysnow"},
}


def evaluation_mAP(cfg, pred_boxes, gt_boxes, subsets, frame_number, print_output=True, methods=['VOC', 'PR-AUC', 'COCO', 'K-Radar']):
    fn = rotated_iou_2d
    iou_threshold = cfg.MAP_2D_IOU_THRESHOLD           # 0.3 based on the K-Radar
    map_2d = compute_dataset_map(cfg, pred_boxes, gt_boxes, subsets, frame_number, 
                                 fn, iou_threshold, '2D mAP', print_output, methods, 
                                 skip_0_gt_frames=cfg.SKIP_0_GT_FRAMES)

    if cfg.INCLUDE_3D:
        fn = rotated_iou_3d
        iou_threshold = cfg.MAP_3D_IOU_THRESHOLD       # 0.3 based on the K-Radar
        map_3d = compute_dataset_map(cfg, pred_boxes, gt_boxes, subsets, frame_number, 
                                     fn, iou_threshold, '3D mAP', print_output, methods,
                                     skip_0_gt_frames=cfg.SKIP_0_GT_FRAMES)

    def print_latex(aps_3d, aps_2d, method_name):
        print(f"\n{method_name} APs:")
        # Name | 3D mAP | 2D mAP | 3D Sedan | 2D Sedan | 3D Bus or Truck | 2D Bus or Truck | 
        # 3D Pedestrian | 2D Pedestrian | 3D Motorcycle | 2D Motorcycle | 3D Bicycle | 2D Bicycle    
        name_latex = cfg.DATE_FOLDER.replace("_", "\\_")
        latex_composite = name_latex + f" & {aps_3d['mAP']*100:.2f} & {aps_2d['mAP']*100:.2f}"
        
        def format_ap(aps_dict, class_name):
            if class_name in aps_dict['aps']:
                return f"{aps_dict['aps'][class_name]*100:.2f}"
            else:
                return "-"
        
        print(latex_composite +\
            f" & {format_ap(aps_3d, 'Sedan')} & {format_ap(aps_2d, 'Sedan')}" +\
            f" & {format_ap(aps_3d, 'Bus or Truck')} & {format_ap(aps_2d, 'Bus or Truck')}" +\
            f" & {format_ap(aps_3d, 'Pedestrian')} & {format_ap(aps_2d, 'Pedestrian')}" +\
            f" & {format_ap(aps_3d, 'Motorcycle')} & {format_ap(aps_2d, 'Motorcycle')}" +\
            f" & {format_ap(aps_3d, 'Bicycle')} & {format_ap(aps_2d, 'Bicycle')} \\\\")

    # Print LaTeX table row
    if 'COCO' in map_2d and 'COCO' in map_3d:
        aps_2d = map_2d['COCO']
        aps_3d = map_3d['COCO']
        print_latex(aps_3d, aps_2d, 'COCO')
        
    if 'K-Radar' in map_2d and 'K-Radar' in map_3d:
        aps_2d = map_2d['K-Radar']
        aps_3d = map_3d['K-Radar']
        print_latex(aps_3d, aps_2d, 'K-Radar')

    return map_2d, map_3d


def aggregate_preds_by_conditions(frame_subset, label, frame_tps, frame_fps, frame_fns, 
                                  info_environment, info_time, info_weather):
    
    def update_info(info_dict, label, key, frame_tps, frame_fps, frame_fns):
        if label not in info_dict:
            info_dict[label] = {}
        if key not in info_dict[label]:
            info_dict[label][key] = {'TP': 0, 'FP': 0, 'FN': 0}
        info_dict[label][key]['TP'] += frame_tps
        info_dict[label][key]['FP'] += frame_fps
        info_dict[label][key]['FN'] += frame_fns


    info = SUBSET_INFO[frame_subset]
    environment = info["environment"]
    time = info["time"]
    weather = info["weather"]

    update_info(info_environment, label, environment, frame_tps, frame_fps, frame_fns)
    update_info(info_time, label, time, frame_tps, frame_fps, frame_fns)
    update_info(info_weather, label, weather, frame_tps, frame_fps, frame_fns)

    return info_environment, info_time, info_weather


def evaluate_correctness(info_environment, info_time, info_weather):

    def sum_stats(info_dict):
        tp, fp, fn = 0, 0, 0
        for label in info_dict:
            for k in info_dict[label]:
                stats = info_dict[label][k]
                tp += stats['TP']
                fp += stats['FP']
                fn += stats['FN']
        return tp, fp, fn


    tp_env, fp_env, fn_env = sum_stats(info_environment)
    tp_time, fp_time, fn_time = sum_stats(info_time)
    tp_weather, fp_weather, fn_weather = sum_stats(info_weather)

    assert(tp_env == tp_time == tp_weather), "TP totals mismatch!"
    assert(fp_env == fp_time == fp_weather), "FP totals mismatch!"
    assert(fn_env == fn_time == fn_weather), "FN totals mismatch!"


def compute_class_aps(cfg, methods, precision, recall, label, results):
    voc_interp_points = 11                             # 11 points for PASCAL VOC-style mAP
    coco_interp_points = 101                           # 101 points for COCO-style mAP
    k_radar_interp_points = 41                         # 41 points for K-Radar-style mAP

    # ===== PASCAL VOC-style mAP =====
    if 'VOC' in methods:
        # Interpolate precision at fixed recall points (PASCAL VOC-style)
        recall_points = np.linspace(0, 1, voc_interp_points)
        precision_interp = np.zeros_like(recall_points)
        for i, r in enumerate(recall_points):
            indices = np.where(recall >= r)[0]
            if indices.size > 0:
                precision_interp[i] = np.max(precision[indices])
            else:
                precision_interp[i] = 0.0

        ap = precision_interp.mean()
        results['VOC']['aps'][label] = ap

    # ===== PR-AUC mAP =====
    if 'PR-AUC' in methods:
        # Compute area under the precision-recall curve using trapezoidal rule
        ap = np.trapezoid(precision, recall)
        results['PR-AUC']['aps'][label] = ap

    # ===== COCO-style mAP =====
    if 'COCO' in methods:
        # Interpolate precision at fixed recall points (COCO-style)
        recall_points = np.linspace(0, 1, coco_interp_points)
        precision_interp = np.zeros_like(recall_points)
        for i, r in enumerate(recall_points):
            indices = np.where(recall >= r)[0]
            if indices.size > 0:
                precision_interp[i] = np.max(precision[indices])
            else:
                precision_interp[i] = 0.0

        ap = precision_interp.mean()
        results['COCO']['aps'][label] = ap

    # ===== K-Radar-style mAP =====
    if 'K-Radar' in methods:
        # Interpolate precision at fixed recall points (K-Radar-style)
        recall_points = np.linspace(0, 1, k_radar_interp_points)
        precision_interp = np.zeros_like(recall_points)
        for i, r in enumerate(recall_points):
            indices = np.where(recall >= r)[0]
            if indices.size > 0:
                precision_interp[i] = np.max(precision[indices])
            else:
                precision_interp[i] = 0.0

        ap = precision_interp.mean()
        results['K-Radar']['aps'][label] = ap

    return results


def export_stats_to_csv(cfg, stats_dict, condition, filename, distance=False):
    path_folders = cfg.PATH_TO_FOLDERS
    path_evaluation = f"{path_folders}/evaluation"

    if not os.path.exists(path_evaluation):
        os.makedirs(path_evaluation)

    if distance:
        with open(f"{path_evaluation}/{filename}", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'distance', 'TP', 'FP', 'FN'])
            labels = [label for eval_type in stats_dict for label in stats_dict[eval_type]]
            for label in set(labels):
                tp_d = stats_dict['tp_distances'][label]
                fp_d = stats_dict['fp_distances'][label]
                fn_d = stats_dict['fn_distances'][label]

                for low, high, bin in [(0, 20, '0~20m'), (20, 40, '20~40m'), (40, 60, '40~60m'), (60, 80, '60~80m')]:
                    tp_count = len([d for d in tp_d if low <= d < high])
                    fp_count = len([d for d in fp_d if low <= d < high])
                    fn_count = len([d for d in fn_d if low <= d < high])
                    row = [label, bin, tp_count, fp_count, fn_count]
                    writer.writerow(row)
        return

    with open(f"{path_evaluation}/{filename}", 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['label', condition, 'TP', 'FP', 'FN'])
        for label in stats_dict:
            for condition_value in stats_dict[label]:
                row = [label, condition_value,
                       int(stats_dict[label][condition_value]['TP']),
                       int(stats_dict[label][condition_value]['FP']),
                       int(stats_dict[label][condition_value]['FN'])]
                writer.writerow(row)
                

def save_mAP_statistics(cfg, info_environment, info_time, info_weather, distances, mAP_type):
    mAP_type = mAP_type[:2]

    export_stats_to_csv(cfg, info_environment, 'environment', f'environment_stats_{mAP_type}.csv')
    export_stats_to_csv(cfg, info_time, 'time', f'time_stats_{mAP_type}.csv')
    export_stats_to_csv(cfg, info_weather, 'weather', f'weather_stats_{mAP_type}.csv')
    export_stats_to_csv(cfg, distances, None, f'distance_stats_{mAP_type}.csv', distance=True)

    dist_df = pd.read_csv(f"{cfg.PATH_TO_FOLDERS}/evaluation/distance_stats_{mAP_type}.csv")
    weather_df = pd.read_csv(f"{cfg.PATH_TO_FOLDERS}/evaluation/weather_stats_{mAP_type}.csv")
    env_df = pd.read_csv(f"{cfg.PATH_TO_FOLDERS}/evaluation/environment_stats_{mAP_type}.csv")
    tod_df = pd.read_csv(f"{cfg.PATH_TO_FOLDERS}/evaluation/time_stats_{mAP_type}.csv")
    
    LABEL_ORDER = ["Sedan", "Bus or Truck", "Pedestrian", "Motorcycle", "Bicycle"]
    PLOTS = [
        # df,        category_col,  category_name,    category_order
        (dist_df,    "distance",    "Distance Range", ["0~20m","20~40m","40~60m","60~80m"]),
        (weather_df, "weather",     "Weather",        ["normal","overcast","fog","rain","sleet","lightsnow","heavysnow"]),
        (env_df,     "environment", "Environment",    ["urban","highway","university","alleyway","mountain","parkinglots","shoulder","countryside"]),
        (tod_df,     "time",        "Time of Day",    ["day","night"]),
    ]
    for df, category_col, category_name, category_order in PLOTS:
        save_path_1 = f"{cfg.PATH_TO_FOLDERS}/evaluation/{category_col}_tp_fp_fn_{mAP_type}.png"
        save_path_2 = f"{cfg.PATH_TO_FOLDERS}/evaluation/{category_col}_metrics_{mAP_type}.png"
        plot_tp_fp_fn_total_grid(
            df,
            category_col=category_col,
            category_name=category_name,
            label_name="Class",
            label_order=LABEL_ORDER,
            category_order=category_order,
            save_path=save_path_1
        )
        plot_tp_fp_fn_metrics_grid(
            df,
            category_col=category_col,
            category_name=category_name,
            label_name="Class",
            label_order=LABEL_ORDER,
            category_order=category_order,
            save_path=save_path_2
        )


def compute_dataset_map(cfg, all_pred_boxes, all_gt_boxes, all_subsets, all_frame_numbers,
                        iou_func, iou_threshold=0.3, mAP_type='mAP', print_output=True, 
                        methods=['VOC', 'PR-AUC', 'COCO'], skip_0_gt_frames=False):
    """
    Compute mean Average Precision (mAP) over the entire dataset, using multiple evaluation methods.

    Methods:
    - PASCAL VOC-style mAP (11-point interpolation)         [2007]
    - PR-AUC mAP (Area under the Precision-Recall curve)    [2010]
    - COCO-style mAP (101-point interpolation)              [2014]

    all_pred_boxes: list of predicted boxes for all frames (each is [N, box_dim])
    all_gt_boxes: list of ground truth boxes for all frames (each is [M, box_dim])
    iou_func: function to compute IoU between two boxes
    """
    label_idx_dict = cfg.LABEL_IDX_DICT
    
    results = {
        'VOC': {'mAP': 0.0, 'aps': {}},
        'PR-AUC': {'mAP': 0.0, 'aps': {}},
        'COCO': {'mAP': 0.0, 'aps': {}},
        'K-Radar': {'mAP': 0.0, 'aps': {}}
    }

    info_environment, info_time, info_weather = {}, {}, {}
    distances = {
        'tp_distances': {label: [] for label in label_idx_dict if label != 'Pedestrian Group' and label != 'Bicycle Group'},
        'fp_distances': {label: [] for label in label_idx_dict if label != 'Pedestrian Group' and label != 'Bicycle Group'},
        'fn_distances': {label: [] for label in label_idx_dict if label != 'Pedestrian Group' and label != 'Bicycle Group'}
    }

    for label, label_idx in label_idx_dict.items():
        conf_scores, tp_list, fp_list = [], [], []
        num_gt = 0

        for frame_pred_boxes, frame_gt_boxes, frame_subset, frame_frame_number in zip(all_pred_boxes, all_gt_boxes, all_subsets, all_frame_numbers):
            # Filter based on ROI
            frame_pred_boxes, frame_gt_boxes = filter_boxes_in_roi(cfg, frame_pred_boxes, frame_gt_boxes)

            # Select predictions and ground truths for this class in this frame
            pred_boxes = frame_pred_boxes[frame_pred_boxes[:, 9] == label_idx]

            # GT is (x, y, z, yaw, l, w, h), where yaw is already in radians
            gt_boxes = np.array([
                [
                    values[0], values[1], values[2], values[4], values[5], values[6],   # x, y, z, l, w, h
                    np.sin(values[3]), np.cos(values[3])                                # sin(yaw), cos(yaw)
                ]
                for gt in frame_gt_boxes if gt[0] == label
                for values in [gt[1]]
            ])

            num_gt += len(gt_boxes)

            # If there are no ground truths and no predictions, skip this frame
            if gt_boxes.shape[0] == 0 and pred_boxes.shape[0] == 0:
                continue
            # If there are no ground truths but predictions, all predictions are false positives
            # (unless skip_0_gt_frames is True, then we skip this frame entirely)
            elif gt_boxes.shape[0] == 0 and pred_boxes.shape[0] > 0:
                if skip_0_gt_frames:
                    # Skip frames with no ground truth to align with other papers
                    continue
                    
                confidence = pred_boxes[:, 8]
                conf_scores.extend(confidence.tolist())
                tp_list.extend([0] * len(confidence))
                fp_list.extend([1] * len(confidence))
                info_environment, info_time, info_weather = aggregate_preds_by_conditions(
                    frame_subset, label, 0, len(confidence), 0,
                    info_environment, info_time, info_weather
                )
                    # FP distances from predicted centers
                for pred_box in pred_boxes:
                    x, y = pred_box[0], pred_box[1]
                    distances['fp_distances'][label].append(np.hypot(x, y))
                continue
            # No preds; only affects recall via num_gt (already counted)
            elif gt_boxes.shape[0] > 0 and pred_boxes.shape[0] == 0:
                if label not in results['VOC']['aps']:
                    results['VOC']['aps'][label] = 0.0
                if label not in results['PR-AUC']['aps']:
                    results['PR-AUC']['aps'][label] = 0.0
                if label not in results['COCO']['aps']:
                    results['COCO']['aps'][label] = 0.0
                if label not in results['K-Radar']['aps']:
                    results['K-Radar']['aps'][label] = 0.0
                info_environment, info_time, info_weather = aggregate_preds_by_conditions(
                    frame_subset, label, 0, 0, len(gt_boxes),
                    info_environment, info_time, info_weather
                )
                # Aggregate by distance
                for gt_box in gt_boxes:
                    x, y = gt_box[0], gt_box[1]
                    distance = np.hypot(x, y)
                    distances['fn_distances'][label].append(distance)
                continue

            # Compute IoU matrix
            iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
            for i, pred_box in enumerate(pred_boxes):
                for j, gt_box in enumerate(gt_boxes):
                    iou_matrix[i, j] = iou_func(pred_box[:8], gt_box[:8])
            
            # Sort predictions by confidence
            confidence = pred_boxes[:, 8]
            sorted_indices = np.argsort(-confidence)

            tp = np.zeros(len(pred_boxes))
            fp = np.zeros(len(pred_boxes))
            matched_gt = set()

            for idx in sorted_indices:
                ious = iou_matrix[idx] if len(gt_boxes) > 0 else np.array([])
                max_iou = np.max(ious) if len(ious) > 0 else 0
                max_gt_idx = np.argmax(ious) if len(ious) > 0 else -1

                if max_iou > iou_threshold and max_gt_idx not in matched_gt:
                    tp[idx] = 1
                    matched_gt.add(max_gt_idx)
                else:
                    fp[idx] = 1
                    # FP distance from predicted center
                    x, y = pred_boxes[idx, 0], pred_boxes[idx, 1]
                    distances['fp_distances'][label].append(np.hypot(x, y))

            # For every frame, track TP, FP, and FN for analysis
            frame_tps = np.sum(tp)
            frame_fps = np.sum(fp)
            frame_fns = len(gt_boxes) - frame_tps
            
            # Aggregate by conditions
            info_environment, info_time, info_weather = aggregate_preds_by_conditions(
                frame_subset, label, frame_tps, frame_fps, frame_fns,
                info_environment, info_time, info_weather
            )

            # Aggregate by distance
            for idx, gt_box in enumerate(gt_boxes):
                x, y = gt_box[0], gt_box[1]
                distance = np.hypot(x, y)
                if idx in matched_gt:
                    distances['tp_distances'][label].append(distance)
                else:
                    distances['fn_distances'][label].append(distance)

            # Store results
            conf_scores.extend(confidence.tolist())
            tp_list.extend(tp.tolist())
            fp_list.extend(fp.tolist())

        # If no ground truths, continue
        if num_gt == 0 or len(tp_list) == 0:
            continue

        # Sort all predictions by confidence (descending)
        conf_scores = np.array(conf_scores)
        tp_list = np.array(tp_list)
        fp_list = np.array(fp_list)
        sort_idx = np.argsort(-conf_scores)
        tp_list = tp_list[sort_idx]
        fp_list = fp_list[sort_idx]

        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum(fp_list)

        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-12)
        recall = tp_cumsum / (num_gt + 1e-12)

        # Add endpoints
        precision = np.concatenate(([1], precision))
        recall = np.concatenate(([0], recall))

        # Compute AP for each method and update results
        results = compute_class_aps(cfg, methods, precision, recall, label, results)

    # Compute mAP for each method
    for method in methods:
        if len(results[method]['aps']) > 0:
            results[method]['mAP'] = np.mean(list(results[method]['aps'].values()))

    # Print results
    if print_output:
        print(f'{mAP_type} Results:')
        for method in methods:
            print(f'  {method} mAP: {results[method]["mAP"]:.4f}')
            for label, ap in results[method]['aps'].items():
                print(f'    Class {label}: AP = {ap:.4f}')

    evaluate_correctness(info_environment, info_time, info_weather)
    save_mAP_statistics(cfg, info_environment, info_time, info_weather, distances, mAP_type)

    return results[methods[0]]['mAP'] if len(methods) == 1 else results


def compute_dataset_map_class_agnostic(cfg, all_pred_boxes, all_gt_boxes, iou_func, iou_threshold=0.3, 
                                       mAP_type='mAP', print_output=True, methods=['VOC', 'PR-AUC', 'COCO']):
    """
    Compute mean Average Precision (mAP) over the entire dataset, using multiple evaluation methods.

    Methods:
    - PASCAL VOC-style mAP (11-point interpolation)         [2007]
    - PR-AUC mAP (Area under the Precision-Recall curve)    [2010]
    - COCO-style mAP (101-point interpolation)              [2014]

    all_pred_boxes: list of predicted boxes for all frames (each is [N, box_dim])
    all_gt_boxes: list of ground truth boxes for all frames (each is [M, box_dim])
    iou_func: function to compute IoU between two boxes
    """
    voc_interp_points = 11                             # 11 points for PASCAL VOC-style mAP
    coco_interp_points = 101                           # 101 points for COCO-style mAP
    
    results = {
        'VOC': {'mAP': 0.0, 'aps': {}},
        'PR-AUC': {'mAP': 0.0, 'aps': {}},
        'COCO': {'mAP': 0.0, 'aps': {}}
    }

    counter_num_pred_boxes = 0
    counter_num_gt_boxes = 0


    conf_scores = []
    tp_list = []
    fp_list = []
    num_gt = 0

    for frame_pred_boxes, frame_gt_boxes in zip(all_pred_boxes, all_gt_boxes):
        # Filter based on ROI
        frame_pred_boxes, frame_gt_boxes = filter_boxes_in_roi(cfg, frame_pred_boxes, frame_gt_boxes)
        pred_boxes = frame_pred_boxes

        # GT is (x, y, z, yaw, l, w, h), where yaw is already in radians
        gt_boxes = np.array([
            [
                values[0], values[1], values[2], values[4], values[5], values[6],   # x, y, z, l, w, h
                np.sin(values[3]), np.cos(values[3])                                # sin(yaw), cos(yaw)
            ]
            for gt in frame_gt_boxes
            for values in [gt[1]]
        ])

        counter_num_pred_boxes += len(pred_boxes)
        counter_num_gt_boxes += len(gt_boxes)

        num_gt += len(gt_boxes)

        # If there are no ground truths and no predictions, skip this frame
        if gt_boxes.shape[0] == 0 and pred_boxes.shape[0] == 0:
            continue
        # If there are no ground truths but predictions, all predictions are false positives
        elif gt_boxes.shape[0] == 0 and pred_boxes.shape[0] > 0:
            confidence = pred_boxes[:, 8]
            conf_scores.extend(confidence.tolist())
            tp_list.extend([0] * len(confidence))
            fp_list.extend([1] * len(confidence))
            continue
        # No preds; only affects recall via num_gt (already counted)
        elif gt_boxes.shape[0] > 0 and pred_boxes.shape[0] == 0:
            continue

        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou_matrix[i, j] = iou_func(pred_box[:8], gt_box[:8])
        
        # Sort predictions by confidence
        confidence = pred_boxes[:, 8]
        sorted_indices = np.argsort(-confidence)

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched_gt = set()

        for idx in sorted_indices:
            ious = iou_matrix[idx] if len(gt_boxes) > 0 else np.array([])
            max_iou = np.max(ious) if len(ious) > 0 else 0
            max_gt_idx = np.argmax(ious) if len(ious) > 0 else -1

            if max_iou > iou_threshold and max_gt_idx not in matched_gt:
                tp[idx] = 1
                matched_gt.add(max_gt_idx)
            else:
                fp[idx] = 1

        # Store results
        conf_scores.extend(confidence.tolist())
        tp_list.extend(tp.tolist())
        fp_list.extend(fp.tolist())

    # Sort all predictions by confidence (descending)
    conf_scores = np.array(conf_scores)
    tp_list = np.array(tp_list)
    fp_list = np.array(fp_list)
    sort_idx = np.argsort(-conf_scores)
    tp_list = tp_list[sort_idx]
    fp_list = fp_list[sort_idx]

    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum(fp_list)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-12)
    recall = tp_cumsum / (num_gt + 1e-12)

    # Add endpoints
    precision = np.concatenate(([1], precision))
    recall = np.concatenate(([0], recall))

    # ===== PASCAL VOC-style mAP =====
    if 'VOC' in methods:
        # Interpolate precision at fixed recall points (PASCAL VOC-style)
        recall_points = np.linspace(0, 1, voc_interp_points)
        precision_interp = np.zeros_like(recall_points)
        for i, r in enumerate(recall_points):
            indices = np.where(recall >= r)[0]
            if indices.size > 0:
                precision_interp[i] = np.max(precision[indices])
            else:
                precision_interp[i] = 0.0

        ap = precision_interp.mean()
        results['VOC']['aps'] = ap

    # ===== PR-AUC mAP =====
    if 'PR-AUC' in methods:
        # Compute area under the precision-recall curve using trapezoidal rule
        ap = np.trapezoid(precision, recall)
        results['PR-AUC']['aps'] = ap

    # ===== COCO-style mAP =====
    if 'COCO' in methods:
        # Interpolate precision at fixed recall points (COCO-style)
        recall_points = np.linspace(0, 1, coco_interp_points)
        precision_interp = np.zeros_like(recall_points)
        for i, r in enumerate(recall_points):
            indices = np.where(recall >= r)[0]
            if indices.size > 0:
                precision_interp[i] = np.max(precision[indices])
            else:
                precision_interp[i] = 0.0

        ap = precision_interp.mean()
        results['COCO']['aps'] = ap

    # Compute mAP for each method
    for method in methods:
        # For class-agnostic, aps is a float, not a dict
        if isinstance(results[method]['aps'], dict):
            if len(results[method]['aps']) > 0:
                results[method]['mAP'] = np.mean(list(results[method]['aps'].values()))
        else:
            results[method]['mAP'] = results[method]['aps']

    # Print results
    if print_output:
        print(f'{mAP_type} Results:')
        for method in methods:
            print(f'  {method} mAP: {results[method]["mAP"]:.4f}')
            # Only print AP directly if it's a float (class-agnostic)
            if not isinstance(results[method]['aps'], dict):
                print(f'    AP = {results[method]["aps"]:.4f}')

    print(f"Number of predicted boxes evaluated: {counter_num_pred_boxes}")

    return results[methods[0]]['mAP'] if len(methods) == 1 else results


if __name__ == "__main__":
    pass