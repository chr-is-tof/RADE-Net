import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from .transformation import transform_ra_indices_to_cartesian
from .rotated_iou import rotated_iou_2d
from .roi import WIDE


def pool_and_threshold(cfg, batch_dict):
    heatmap = batch_dict['heatmap']
    # Max Pooling
    max_pooled = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    
    # Create mask of local maxima
    local_max_mask = (heatmap >= max_pooled)
    
    # Create another mask for confidence threshold
    threshold_mask = (heatmap > cfg.CONFIDENCE_SCORE_THRESHOLD)
    
    # Combine masks
    keep_mask = local_max_mask & threshold_mask
   
    # Filter heatmap
    heatmap_thresholded = torch.where(keep_mask, heatmap, torch.tensor(0., device=heatmap.device, dtype=heatmap.dtype))
    
    batch_dict['peaks'] = heatmap_thresholded

    return batch_dict


def build_bounding_boxes_unified(cfg, batch_dict):
    if cfg.USE_SINGULAR_HEATMAP:
        heatmap = batch_dict['peaks']                   # [B, 1, H, W]
        reg = batch_dict['regression']                  # [B, 8, H, W]
        classification = batch_dict['classification']   # [B, C, H, W]
        B, _, H, W = heatmap.shape
        C = classification.shape[1]
    else:
        heatmap = batch_dict['peaks']            # [B, C, H, W]
        reg = batch_dict['regression']           # [B, 8, H, W]
        B, C, H, W = heatmap.shape
    
    # Define ROI boundaries if filtering is enabled
    if cfg.FILTER_BOXES_BY_WIDE_ROI:
        x_min, x_max = WIDE['x_min'], WIDE['x_max']
        y_min, y_max = WIDE['y_min'], WIDE['y_max']
        z_min, z_max = WIDE['z_min'], WIDE['z_max']

    device = heatmap.device

    batch_boxes = []

    for b in range(B):
        if cfg.USE_SINGULAR_HEATMAP:
            # Reduce to single channel
            scores_ = heatmap[b, 0]
            # Get indices where there is a detection
            idx_range, idx_azi = torch.nonzero(scores_, as_tuple=True)
        else:
            # For each pixel, find the class with the highest score and its value
            # [C, H, W] -> [H, W] for class and score
            class_scores, class_indices = torch.max(heatmap[b], dim=0)  # [H, W], [H, W]
            idx_range, idx_azi = torch.nonzero(class_scores, as_tuple=True)

        # If no detections, continue
        if idx_range.numel() == 0:
            batch_boxes.append(torch.empty(0, 12 + C, device=device))
            continue

        x_long, y_lat = transform_ra_indices_to_cartesian(
            idx_azi, 
            idx_range, 
            cfg.USE_DATA_PADDING, 
            True if cfg.RDR_PROCESSING_METHOD == "upsample_azimuth" else False
        )
        
        boxes = []
        for k in range(len(idx_range)):
            # Read out regression predictions
            pred_reg = reg[b][:, idx_range[k].long(), idx_azi[k].long()]
            dx, dy, dz, l, w, h, sin_yaw, cos_yaw = pred_reg
            
            # Add regression offsets to the center positions
            x = x_long[k] + dx
            y = y_lat[k] + dy
            z = dz

            if cfg.FILTER_BOXES_BY_WIDE_ROI:
                # Check if box center is within wide ROI
                if not (x_min <= x.item() <= x_max and
                        y_min <= y.item() <= y_max and
                        z_min <= z.item() <= z_max):
                    continue  # Skip this box

            if cfg.USE_SINGULAR_HEATMAP:
                class_scores = classification[b, :, idx_range[k], idx_azi[k]]  # [C]
                class_score_values = torch.softmax(class_scores, dim=0)
                bbox_class_confidence, bbox_class = torch.max(class_score_values, dim=0)
            else:
                bbox_class = class_indices[idx_range[k], idx_azi[k]]
                bbox_class_confidence = class_scores[idx_range[k], idx_azi[k]]
                class_score_values = heatmap[b, :, idx_range[k], idx_azi[k]]  # [C]

            box = torch.cat([
                torch.tensor([
                    x, y, z, l, w, h, sin_yaw, cos_yaw, 
                    bbox_class_confidence, bbox_class.float(),
                    idx_azi[k].float(), idx_range[k].float()
                ], device=heatmap.device),
                class_score_values
            ])  # [12 + C]

            boxes.append(box)

        batch_boxes.append(torch.stack(boxes) if boxes else torch.empty(0, 12 + C, device=device))

    batch_dict['bboxes'] = batch_boxes

    return batch_dict, device


def get_nms_reduced_bboxes(batch_dict, device):
    bboxes = batch_dict['bboxes']
    batch_size = len(bboxes)

    nms_reduced_bboxes = []
    nms_reduced_indices = []
    for batch_idx in range(batch_size):
        if bboxes[batch_idx].numel() == 0:
            nms_reduced_bboxes.append(torch.empty(0, 12, device=device))
            nms_reduced_indices.append(torch.empty(0, device=device))
            continue

        bbox = bboxes[batch_idx]

        selected_bbox = bbox[:, :8]
        scores = bbox[:, 8]  # confidence scores
        remaining_vals = bbox[:, 9:] # class, x center, y center

        nms_reduced_bbox, nms_reduced_index = nms_rotated(selected_bbox, scores, iou_threshold=0.3)

        full_box = torch.cat((nms_reduced_bbox, remaining_vals[nms_reduced_index]), dim=1)  # [num_kept, 12 + C]
        nms_reduced_bboxes.append(full_box)
        nms_reduced_indices.append(nms_reduced_index)

    batch_dict['predicted_boxes'] = nms_reduced_bboxes
    batch_dict['predicted_boxes_indices'] = nms_reduced_indices
    
    return batch_dict


def nms_rotated(dets: Tensor,
                scores: Tensor,
                iou_threshold: float,
                labels: Optional[Tensor] = None,
                clockwise: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Pure PyTorch implementation of rotated NMS
    Slower than CUDA version but doesn't require compiled extensions
    """
    if dets.shape[0] == 0:
        return dets, torch.empty(0, dtype=torch.long, device=dets.device)
    
    # Sort by scores (descending)
    _, order = scores.sort(0, descending=True)
    keep_inds = []
    
    while len(order) > 0:
        # Keep highest scoring box
        current = order[0]
        keep_inds.append(current)
        
        if len(order) == 1:
            break
        
        # Get remaining boxes
        remaining = order[1:]
        
        # Compute IoU between current box and remaining boxes
        current_box = dets[current]
        ious = []
        for i in remaining:
            iou = rotated_iou_2d(current_box, dets[i])
            #print(f"IoU between box {current} and box {i}: {iou}")
            ious.append(torch.tensor(iou, device=dets.device, dtype=torch.float32))
        ious = torch.stack(ious)
        
        # Keep boxes with IoU below threshold
        keep_mask = ious < iou_threshold
        order = remaining[keep_mask]
    
    keep_inds = torch.tensor(keep_inds, dtype=torch.long, device=dets.device)
    
    # Return in same format as original
    kept_dets = torch.cat((dets[keep_inds], scores[keep_inds].reshape(-1, 1)), dim=1)

    return kept_dets, keep_inds


def postprocessing(cfg, batch_dict):
    batch_dict = pool_and_threshold(cfg, batch_dict)
    batch_dict, device = build_bounding_boxes_unified(cfg, batch_dict)
    batch_dict = get_nms_reduced_bboxes(batch_dict, device)

    return batch_dict
