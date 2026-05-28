#!/usr/bin/env python3
"""
Standalone Detection Evaluation Script

This script loads all predictions and ground truth files, identifies True Positives (TP),
False Positives (FP), and False Negatives (FN), and saves visualization images sorted 
into respective folders.

Usage:
    1. Configure the parameters below
    2. Run from project root: python -m utils.evaluate_detections
       Or run directly: python utils/evaluate_detections.py
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the framework
from configs import params_rade as cfg
from ops.rotated_iou import rotated_iou_2d, rotated_iou_3d
from ops.roi import filter_boxes_in_roi, NARROW, WIDE


# ============================================================================
#                           CONFIGURATION PARAMETERS
# ============================================================================

# Experiment folder name (e.g., "26_02_05_15_29_rad_ndil_hexp")
EXPERIMENT = "2026_02_23_10_11_radcam_bl_gatefus_fs75_weightnet_noposenc"

# Model checkpoint name (e.g., "epoch10")
MODEL = "epoch10"

# IoU threshold for matching predictions to ground truth
IOU_THRESHOLD = 0.3

# ROI type for filtering: 'narrow', 'wide', or 'all'
ROI_TYPE = "narrow"

# IoU computation mode: '2d' or '3d'
MODE = "3d"

# Maximum number of samples to process (set to None for all samples)
MAX_SAMPLES = None

# Filter by specific classes (set to None for all classes)
# Example: CLASSES = ['Sedan', 'Bus or Truck']
CLASSES = ['Sedan', 'Bus or Truck']

# ============================================================================


class Args:
    """Simple class to hold configuration parameters."""
    def __init__(self):
        self.experiment = EXPERIMENT
        self.model = MODEL
        self.iou_threshold = IOU_THRESHOLD
        self.roi_type = ROI_TYPE
        self.mode = MODE
        self.max_samples = MAX_SAMPLES
        self.classes = CLASSES


def get_paths(args):
    """Build paths based on the experiment and model."""
    backbone_type = cfg.MODEL_CFG['rad_backbone']['type']
    experiment_path = f"{cfg.PATH_TO_EXPERIMENTS}/{backbone_type}/{args.experiment}"
    inference_path = f"{experiment_path}/inference/{args.model}"
    output_path = f"{experiment_path}/detection_analysis/{args.model}"
    
    return {
        'experiment': experiment_path,
        'inference': inference_path,
        'output': output_path,
        'dataset': cfg.PATH_TO_PROCESSED_DATASET
    }


def setup_output_dirs(output_path):
    """Create output directories for TP, FP, FN images."""
    dirs = {
        'tp': f"{output_path}/true_positives",
        'fp': f"{output_path}/false_positives", 
        'fn': f"{output_path}/false_negatives",
        'summary': f"{output_path}/summary"
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def load_inference_data(inference_path):
    """Load all inference pkl files from the given path."""
    paths = []
    for file in os.listdir(inference_path):
        if file.endswith('.pkl'):
            paths.append(os.path.join(inference_path, file))
    
    paths.sort()
    
    data = []
    for path in paths:
        with open(path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        # Extract subset and frame from filename
        # Format: boxes_subset_{subset}_frame_{frame}.pkl
        parts = os.path.basename(path).split('_')
        subset = parts[2]
        frame = parts[4].split('.')[0]
        
        data.append({
            'path': path,
            'subset': subset,
            'frame': frame,
            'pred_boxes': pkl_data['predicted_boxes'],
            'gt_list': pkl_data['gt_list']
        })
    
    return data


def get_camera_path(dataset_path, subset, gt_list):
    """Build camera image path from ground truth info."""
    if len(gt_list) > 0:
        # GT format: (cls_name, (x, y, z, th, l, w, h), idx_p, avail, (sensor_names), (sensor_indices))
        # sensor_indices = (rdr, ldr64, camf, ldr128, camr)
        camf_idx = gt_list[0][5][2]  # Camera front index
    else:
        # If no GT, try to find camera file by listing directory
        cam_dir = f"{dataset_path}/{subset}/cam-front"
        if os.path.exists(cam_dir):
            cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.png')])
            if cam_files:
                return f"{cam_dir}/{cam_files[0]}"
        return None
    
    cam_path = f"{dataset_path}/{subset}/cam-front/cam-front_{camf_idx}.png"
    return cam_path if os.path.exists(cam_path) else None


def convert_gt_to_boxes(gt_list):
    """Convert ground truth list to box format for IoU computation."""
    gt_boxes = []
    for gt in gt_list:
        cls_name = gt[0]
        x, y, z, yaw, l, w, h = gt[1]
        # Format: [x, y, z, l, w, h, sin(yaw), cos(yaw)]
        gt_boxes.append({
            'label': cls_name,
            'box': np.array([x, y, z, l, w, h, np.sin(yaw), np.cos(yaw)]),
            'raw': gt
        })
    return gt_boxes


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_func, iou_threshold, label_idx_dict):
    """
    Match predictions to ground truth and identify TP, FP, FN.
    
    Returns:
        tp_list: List of (pred_box, gt_box, iou, label) tuples
        fp_list: List of (pred_box, best_iou, reason, label) tuples
        fn_list: List of (gt_box, label) tuples
    """
    tp_list = []
    fp_list = []
    fn_list = []
    
    idx_label_dict = {v: k for k, v in label_idx_dict.items()}
    
    # Group predictions and GTs by class
    for label, label_idx in label_idx_dict.items():
        # Filter predictions for this class
        class_preds = [p for p in range(len(pred_boxes)) 
                       if pred_boxes[p, 9] == label_idx]
        
        # Filter ground truths for this class
        class_gts = [g for g in range(len(gt_boxes)) 
                     if gt_boxes[g]['label'] == label]
        
        if len(class_preds) == 0 and len(class_gts) == 0:
            continue
        
        # No GTs but have predictions -> all FP
        if len(class_gts) == 0:
            for pred_idx in class_preds:
                pred_box = pred_boxes[pred_idx]
                fp_list.append({
                    'pred_box': pred_box,
                    'best_iou': 0.0,
                    'reason': 'no_ground_truth',
                    'label': label
                })
            continue
        
        # Have GTs but no predictions -> all FN
        if len(class_preds) == 0:
            for gt_idx in class_gts:
                fn_list.append({
                    'gt_box': gt_boxes[gt_idx],
                    'label': label
                })
            continue
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(class_preds), len(class_gts)))
        for i, pred_idx in enumerate(class_preds):
            pred_box = pred_boxes[pred_idx][:8]
            for j, gt_idx in enumerate(class_gts):
                gt_box = gt_boxes[gt_idx]['box']
                iou_matrix[i, j] = iou_func(pred_box, gt_box)
        
        # Sort predictions by confidence (descending)
        confidences = [pred_boxes[p, 8] for p in class_preds]
        sorted_indices = np.argsort(-np.array(confidences))
        
        matched_gt = set()
        
        for sorted_idx in sorted_indices:
            pred_idx = class_preds[sorted_idx]
            pred_box = pred_boxes[pred_idx]
            
            ious = iou_matrix[sorted_idx]
            max_iou = np.max(ious)
            max_gt_idx = np.argmax(ious)
            
            if max_iou > iou_threshold and max_gt_idx not in matched_gt:
                # True Positive
                tp_list.append({
                    'pred_box': pred_box,
                    'gt_box': gt_boxes[class_gts[max_gt_idx]],
                    'iou': max_iou,
                    'label': label
                })
                matched_gt.add(max_gt_idx)
            else:
                # False Positive
                reason = 'low_iou' if max_iou <= iou_threshold else 'gt_already_matched'
                fp_list.append({
                    'pred_box': pred_box,
                    'best_iou': max_iou,
                    'reason': reason,
                    'label': label
                })
        
        # Remaining unmatched GTs are False Negatives
        for j, gt_idx in enumerate(class_gts):
            if j not in matched_gt:
                fn_list.append({
                    'gt_box': gt_boxes[gt_idx],
                    'label': label
                })
    
    return tp_list, fp_list, fn_list


def plot_rotated_bbox(ax, x, y, l, w, yaw, color, linewidth=2, label=None):
    """Plot a rotated bounding box."""
    # Create rectangle corners
    corners = np.array([
        [-l/2, -w/2],
        [l/2, -w/2],
        [l/2, w/2],
        [-l/2, w/2],
        [-l/2, -w/2]  # Close the rectangle
    ])
    
    # Rotation matrix
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw, cos_yaw]
    ])
    
    # Rotate and translate
    rotated = corners @ rot.T
    rotated[:, 0] += x
    rotated[:, 1] += y
    
    ax.plot(rotated[:, 0], rotated[:, 1], color=color, linewidth=linewidth, label=label)


def visualize_detection(cam_path, detection, detection_type, output_path, subset, frame, 
                        idx, flip_horizontally=True):
    """
    Visualize a detection on the camera image with BEV overlay.
    
    Args:
        cam_path: Path to camera image
        detection: Detection dict containing box info
        detection_type: 'tp', 'fp', or 'fn'
        output_path: Directory to save the image
        subset: Subset number
        frame: Frame number
        idx: Detection index
        flip_horizontally: Whether to flip y-axis for visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cam_frame = cam_path.split('/')[-1] if cam_path else 'unknown'
    # Left: Camera image
    ax_cam = axes[0]
    if cam_path and os.path.exists(cam_path):
        img = mpimg.imread(cam_path)
        ax_cam.imshow(img)
        ax_cam.set_title('Camera View')
    else:
        ax_cam.text(0.5, 0.5, 'Camera image not available', 
                    ha='center', va='center', fontsize=12)
        ax_cam.set_xlim(0, 1)
        ax_cam.set_ylim(0, 1)
    ax_cam.axis('off')
    
    # Right: BEV visualization
    ax_bev = axes[1]
    ax_bev.set_xlim(-20, 80)
    ax_bev.set_ylim(-30, 30)
    ax_bev.set_xlabel('X (Range) [m]')
    ax_bev.set_ylabel('Y (Azimuth) [m]')
    ax_bev.set_aspect('equal')
    ax_bev.grid(True, alpha=0.3)
    
    # Plot ROI boundaries
    ax_bev.axhline(y=NARROW['y_min'], color='gray', linestyle='--', alpha=0.5, label='Narrow ROI')
    ax_bev.axhline(y=NARROW['y_max'], color='gray', linestyle='--', alpha=0.5)
    ax_bev.axvline(x=NARROW['x_min'], color='gray', linestyle='--', alpha=0.5)
    ax_bev.axvline(x=NARROW['x_max'], color='gray', linestyle='--', alpha=0.5)
    
    # Title and info text
    label = detection.get('label', 'Unknown')
    
    if detection_type == 'tp':
        pred_box = detection['pred_box']
        gt_data = detection['gt_box']
        iou = detection['iou']
        
        # Prediction box
        x, y, z = pred_box[0], pred_box[1], pred_box[2]
        l, w, h = pred_box[3], pred_box[4], pred_box[5]
        yaw = np.arctan2(pred_box[6], pred_box[7])
        conf = pred_box[8]
        
        plot_rotated_bbox(ax_bev, x, y, l, w, yaw, 'green', linewidth=2, label='Prediction')
        
        # GT box (no flip needed - GT is already in correct coordinate frame)
        gt_box = gt_data['box']
        gx, gy, gz = gt_box[0], gt_box[1], gt_box[2]
        gl, gw, gh = gt_box[3], gt_box[4], gt_box[5]
        g_yaw = np.arctan2(gt_box[6], gt_box[7])
        
        plot_rotated_bbox(ax_bev, gx, gy, gl, gw, g_yaw, 'cyan', linewidth=2, label='Ground Truth')
        
        ax_bev.set_title(f'TRUE POSITIVE - {label}\nIoU: {iou:.3f}, Conf: {conf:.3f}')
        filename = f"tp_s{subset}_f{frame}_{idx}_{label}_iou{iou:.2f}_conf{conf:.2f}.png"
        
    elif detection_type == 'fp':
        pred_box = detection['pred_box']
        best_iou = detection['best_iou']
        reason = detection['reason']
        
        x, y, z = pred_box[0], pred_box[1], pred_box[2]
        l, w, h = pred_box[3], pred_box[4], pred_box[5]
        yaw = np.arctan2(pred_box[6], pred_box[7])
        conf = pred_box[8]
        
        plot_rotated_bbox(ax_bev, x, y, l, w, yaw, 'red', linewidth=2, label='False Positive')
        
        ax_bev.set_title(f'FALSE POSITIVE - {label}\nReason: {reason}, Best IoU: {best_iou:.3f}, Conf: {conf:.3f}')
        filename = f"fp_s{subset}_f{frame}_{idx}_{label}_{reason}_conf{conf:.2f}.png"
        
    elif detection_type == 'fn':
        gt_data = detection['gt_box']
        gt_box = gt_data['box']
        
        # GT box (no flip needed - GT is already in correct coordinate frame)
        gx, gy, gz = gt_box[0], gt_box[1], gt_box[2]
        gl, gw, gh = gt_box[3], gt_box[4], gt_box[5]
        g_yaw = np.arctan2(gt_box[6], gt_box[7])
        
        plot_rotated_bbox(ax_bev, gx, gy, gl, gw, g_yaw, 'orange', linewidth=2, label='Missed GT')
        
        ax_bev.set_title(f'FALSE NEGATIVE - {label}\nMissed Ground Truth')
        filename = f"fn_s{subset}_f{frame}_{idx}_{label}.png"
    
    ax_bev.legend(loc='upper right')
    
    # Add frame info
    cam_frame = cam_path.split('/')[-1] if cam_path else 'unknown'
    fig.suptitle(f'Subset: {subset}, Frame: {frame}, cam_frame: {cam_frame}', fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filename


def save_summary_csv(output_dirs, all_detections):
    """Save summary CSV files for TP, FP, FN."""
    import csv
    
    # True Positives
    tp_file = os.path.join(output_dirs['summary'], 'true_positives.csv')
    with open(tp_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subset', 'frame', 'label', 'iou', 'confidence', 
                         'pred_x', 'pred_y', 'pred_z', 'pred_l', 'pred_w', 'pred_h',
                         'gt_x', 'gt_y', 'gt_z', 'gt_l', 'gt_w', 'gt_h'])
        for det in all_detections['tp']:
            pred = det['pred_box']
            gt = det['gt_box']['box']
            writer.writerow([
                det['subset'], det['frame'], det['label'], f"{det['iou']:.4f}", f"{pred[8]:.4f}",
                f"{pred[0]:.2f}", f"{pred[1]:.2f}", f"{pred[2]:.2f}",
                f"{pred[3]:.2f}", f"{pred[4]:.2f}", f"{pred[5]:.2f}",
                f"{gt[0]:.2f}", f"{gt[1]:.2f}", f"{gt[2]:.2f}",
                f"{gt[3]:.2f}", f"{gt[4]:.2f}", f"{gt[5]:.2f}"
            ])
    
    # False Positives
    fp_file = os.path.join(output_dirs['summary'], 'false_positives.csv')
    with open(fp_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subset', 'frame', 'label', 'reason', 'best_iou', 'confidence',
                         'pred_x', 'pred_y', 'pred_z', 'pred_l', 'pred_w', 'pred_h'])
        for det in all_detections['fp']:
            pred = det['pred_box']
            writer.writerow([
                det['subset'], det['frame'], det['label'], det['reason'], 
                f"{det['best_iou']:.4f}", f"{pred[8]:.4f}",
                f"{pred[0]:.2f}", f"{pred[1]:.2f}", f"{pred[2]:.2f}",
                f"{pred[3]:.2f}", f"{pred[4]:.2f}", f"{pred[5]:.2f}"
            ])
    
    # False Negatives
    fn_file = os.path.join(output_dirs['summary'], 'false_negatives.csv')
    with open(fn_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subset', 'frame', 'label',
                         'gt_x', 'gt_y', 'gt_z', 'gt_l', 'gt_w', 'gt_h'])
        for det in all_detections['fn']:
            gt = det['gt_box']['box']
            writer.writerow([
                det['subset'], det['frame'], det['label'],
                f"{gt[0]:.2f}", f"{gt[1]:.2f}", f"{gt[2]:.2f}",
                f"{gt[3]:.2f}", f"{gt[4]:.2f}", f"{gt[5]:.2f}"
            ])
    
    print(f"Summary CSV files saved to {output_dirs['summary']}")


def print_statistics(all_detections):
    """Print detection statistics."""
    tp_count = len(all_detections['tp'])
    fp_count = len(all_detections['fp'])
    fn_count = len(all_detections['fn'])
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print("DETECTION EVALUATION SUMMARY")
    print("="*60)
    print(f"True Positives (TP):   {tp_count}")
    print(f"False Positives (FP):  {fp_count}")
    print(f"False Negatives (FN):  {fn_count}")
    print("-"*60)
    print(f"Precision:             {precision:.4f}")
    print(f"Recall:                {recall:.4f}")
    print(f"F1 Score:              {f1:.4f}")
    print("="*60)
    
    # Per-class statistics
    print("\nPer-Class Statistics:")
    print("-"*60)
    
    labels = set()
    for det in all_detections['tp'] + all_detections['fp'] + all_detections['fn']:
        labels.add(det['label'])
    
    for label in sorted(labels):
        tp_cls = sum(1 for d in all_detections['tp'] if d['label'] == label)
        fp_cls = sum(1 for d in all_detections['fp'] if d['label'] == label)
        fn_cls = sum(1 for d in all_detections['fn'] if d['label'] == label)
        
        prec_cls = tp_cls / (tp_cls + fp_cls) if (tp_cls + fp_cls) > 0 else 0
        rec_cls = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0
        
        print(f"  {label:20s}: TP={tp_cls:4d}, FP={fp_cls:4d}, FN={fn_cls:4d}, "
              f"Prec={prec_cls:.3f}, Rec={rec_cls:.3f}")
    
    # FP reason breakdown
    print("\nFalse Positive Reasons:")
    print("-"*60)
    fp_reasons = {}
    for det in all_detections['fp']:
        reason = det['reason']
        fp_reasons[reason] = fp_reasons.get(reason, 0) + 1
    
    for reason, count in sorted(fp_reasons.items()):
        print(f"  {reason:25s}: {count}")


def main():
    args = Args()
    
    print(f"\n{'='*60}")
    print("Detection Evaluation Script")
    print(f"{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Model: {args.model}")
    print(f"IoU Threshold: {args.iou_threshold}")
    print(f"ROI Type: {args.roi_type}")
    print(f"Mode: {args.mode}")
    
    # Setup paths
    paths = get_paths(args)
    print(f"\nInference path: {paths['inference']}")
    print(f"Output path: {paths['output']}")
    
    if not os.path.exists(paths['inference']):
        print(f"\nError: Inference path does not exist: {paths['inference']}")
        sys.exit(1)
    
    # Create output directories
    output_dirs = setup_output_dirs(paths['output'])
    
    # Load inference data
    print("\nLoading inference data...")
    data = load_inference_data(paths['inference'])
    print(f"Loaded {len(data)} frames")
    
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Processing first {args.max_samples} samples")
    
    # Choose IoU function
    iou_func = rotated_iou_2d if args.mode == '2d' else rotated_iou_3d
    
    # Update cfg for ROI filtering
    cfg.ROI_TYPE = args.roi_type
    
    # Process each frame
    all_detections = {'tp': [], 'fp': [], 'fn': []}
    
    print("\nProcessing frames...")
    for i, frame_data in enumerate(data):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} frames")
        
        subset = frame_data['subset']
        frame = frame_data['frame']
        pred_boxes = frame_data['pred_boxes']
        gt_list = frame_data['gt_list']
        
        # Filter by ROI
        pred_boxes, gt_list = filter_boxes_in_roi(cfg, pred_boxes, gt_list)
        
        # Filter by classes if specified
        if args.classes:
            gt_list = [gt for gt in gt_list if gt[0] in args.classes]
            label_indices = [cfg.LABEL_IDX_DICT[c] for c in args.classes if c in cfg.LABEL_IDX_DICT]
            mask = np.isin(pred_boxes[:, 9], label_indices)
            pred_boxes = pred_boxes[mask]
        
        # Convert GT to box format
        gt_boxes = convert_gt_to_boxes(gt_list)
        
        # Match predictions to GT
        tp_list, fp_list, fn_list = match_predictions_to_gt(
            pred_boxes, gt_boxes, iou_func, args.iou_threshold, cfg.LABEL_IDX_DICT
        )
        
        # Get camera path
        cam_path = get_camera_path(paths['dataset'], subset, gt_list)
        
        # Visualize and save each detection
        for idx, tp in enumerate(tp_list):
            tp['subset'] = subset
            tp['frame'] = frame
            all_detections['tp'].append(tp)
            visualize_detection(cam_path, tp, 'tp', output_dirs['tp'], 
                              subset, frame, idx, cfg.FLIP_GT_HORIZONTALLY)
        
        for idx, fp in enumerate(fp_list):
            fp['subset'] = subset
            fp['frame'] = frame
            all_detections['fp'].append(fp)
            visualize_detection(cam_path, fp, 'fp', output_dirs['fp'],
                              subset, frame, idx, cfg.FLIP_GT_HORIZONTALLY)
        
        for idx, fn in enumerate(fn_list):
            fn['subset'] = subset
            fn['frame'] = frame
            all_detections['fn'].append(fn)
            visualize_detection(cam_path, fn, 'fn', output_dirs['fn'],
                              subset, frame, idx, cfg.FLIP_GT_HORIZONTALLY)
    
    # Save summary CSV files
    save_summary_csv(output_dirs, all_detections)
    
    # Print statistics
    print_statistics(all_detections)
    
    print(f"\nVisualization images saved to:")
    print(f"  True Positives:  {output_dirs['tp']}")
    print(f"  False Positives: {output_dirs['fp']}")
    print(f"  False Negatives: {output_dirs['fn']}")
    print(f"  Summary CSVs:    {output_dirs['summary']}")


if __name__ == "__main__":
    main()
