import numpy as np


# Narrow ROI coordinates as defined by K-Radar
NARROW = {
    "x_min": 0.0,
    "x_max": 72.0,
    "y_min": -6.4,
    "y_max": 6.4,
    "z_min": -2.0,
    "z_max": 6.0
}

# Wide ROI coordinates as defined by K-Radar
WIDE = {
    "x_min": 0.0,
    "x_max": 72.0,
    "y_min": -16.0,
    "y_max": 16.0,
    "z_min": -2.0,
    "z_max": 7.6
}


def box_center_in_roi(center_xyz, roi_dict):
    x, y, z = center_xyz
    return (
        roi_dict['x_min'] <= x <= roi_dict['x_max'] and
        roi_dict['y_min'] <= y <= roi_dict['y_max'] and
        roi_dict['z_min'] <= z <= roi_dict['z_max']
    )


def filter_pred_boxes_in_roi(frame_pred_boxes, roi_dict):
    # Expects columns like [x,y,z,l,w,h,sin_yaw,cos_yaw,score,label_idx]
    mask = []
    
    for b in frame_pred_boxes:
        center = (b[0], b[1], b[2])
        mask.append(box_center_in_roi(center, roi_dict))
    
    mask = np.asarray(mask, dtype=bool)
    return frame_pred_boxes[mask]


def filter_gt_boxes_in_roi(frame_gt_boxes, roi_dict):
    # Expects list of tuples [(label_str, values)], where values=[x,y,z,yaw,l,w,h]
    filtered = []
    
    for gt_box in frame_gt_boxes:
        gt_values = gt_box[1]
        center = (gt_values[0], gt_values[1], gt_values[2])
        if box_center_in_roi(center, roi_dict):
            filtered.append(gt_box)

    return filtered


def filter_boxes_in_roi(cfg, pred_boxes, gt_boxes):
    if cfg.ROI_TYPE == 'narrow':
        roi_dict = NARROW
    elif cfg.ROI_TYPE == 'wide':
        roi_dict = WIDE
    else:
        raise ValueError(f"Unknown ROI_TYPE: {cfg.ROI_TYPE!r}. Expected 'narrow' or 'wide'.")

    filtered_pred_boxes = filter_pred_boxes_in_roi(pred_boxes, roi_dict)
    filtered_gt_boxes = filter_gt_boxes_in_roi(gt_boxes, roi_dict)

    return filtered_pred_boxes, filtered_gt_boxes