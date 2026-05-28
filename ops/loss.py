import torch
import torch.nn as nn
import torch.nn.functional as F

from .gwd import gaussian_wasserstein_distance


class FocalLossGaussianBinary(nn.Module):
    # NOTE: Not thoroughly tested yet
    def __init__(self, alpha=2, gamma=4, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, heatmap, gaussian_map):
        B, C, H, W = heatmap.shape
        heatmap = heatmap.view(B, C, -1)
        gaussian_map = gaussian_map.view(B, C, -1)

        eps = 1e-6

        # Compute binary cross entropy loss for each heatmap cell
        bce_loss = F.binary_cross_entropy(heatmap, gaussian_map, reduction='none')  # [B, N]

        # Compute probability for correct class
        p = torch.where(gaussian_map == 1, heatmap, 1 - heatmap)
        p = torch.clamp(p, min=eps, max=(1 - eps))  # Avoid numerical instability

        # Compute alpha factor that balances positive and negative
        alpha_factor = torch.where(gaussian_map == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_factor * torch.pow((1 - p), self.gamma) * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLossGaussianContinuous(nn.Module):
    """
    Focal loss with continuous targets based on a Gaussian distribution around
    the centers of the ground truth. 
    Implementation from CornerNet: https://arxiv.org/abs/1808.01244
    """

    def __init__(self, alpha=2, gamma=4, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, heatmap, gaussian_map):
        eps = 1e-6
        pos_weights = gaussian_map.eq(1)
        neg_weights = torch.pow(1 - gaussian_map, self.gamma)

        # Clamp the heatmap to avoid close to log(0) which results in NaNs
        heatmap = torch.clamp(heatmap, min=eps, max=1-eps)
        heatmap_complement = torch.clamp(1 - heatmap, min=eps, max=1-eps)

        pos_loss = -torch.log(heatmap) * torch.pow(heatmap_complement, self.alpha) * pos_weights
        neg_loss = -torch.log(heatmap_complement) * torch.pow(heatmap, self.alpha) * neg_weights

        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            if torch.sum(pos_weights) > 0:
                return torch.sum(loss) / (torch.sum(pos_weights) + eps)
            else:
                # No positives: average over all pixels (negatives only)
                return torch.mean(neg_loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class SoftFocalLoss(nn.Module):
    """
    Soft focal loss for continuous targets in [0, 1]. The difference to the "hard" focal loss
    for continuous targets is that the loss is applied everywhere and the approach is more 
    "symmetric", where the weighting is applied continuously.
    """
    # NOTE: Not thoroughly tested yet
    def __init__(self, alpha=2, gamma=4, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, heatmap, gaussian_map):
        B, C, H, W = heatmap.shape
        heatmap = heatmap.view(B, C, -1)
        gaussian_map = gaussian_map.view(B, C, -1)

        bce_loss = F.binary_cross_entropy(heatmap, gaussian_map, reduction='none')  # [B, N]

        # Focal weight: (1 - |target - pred|)^gamma
        pt = (heatmap * gaussian_map) + ((1 - heatmap) * (1 - gaussian_map))
        focal_weight = (1 - pt) ** self.gamma

        # Alpha balancing
        alpha_factor = self.alpha * gaussian_map + (1 - self.alpha) * (1 - gaussian_map)

        loss = alpha_factor * focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GWDLoss(nn.Module):
    def __init__(self, decomposition="eigenvalue"):
        super().__init__()
        self.decomposition = decomposition

    def forward(self, cfg, batch_dict):
        regression_maps = batch_dict['regression']          # [B, 8, H, W]
        gt_list_tensors = batch_dict['gt_list_tensors']     # list[Tensor]
        gt_center_points = batch_dict['gt_center_points']   # list[Tensor]

        batch_losses = []
        batch_pred_boxes = []

        B, C, H, W = regression_maps.shape
        for batch_idx in range(B):
            regression_map = regression_maps[batch_idx]     # [8, H, W]
            gt_boxes = gt_list_tensors[batch_idx]           # [num_gt, 8]
            gt_centers = gt_center_points[batch_idx]        # [num_gt, 2] in RA

            # If there are no ground truths, regression loss is 0
            if gt_boxes.numel() == 0:
                batch_losses.append(torch.tensor(0.0, dtype=torch.float32, device=regression_map.device))
                batch_pred_boxes.append(torch.zeros((0, 8), dtype=torch.float32, device=regression_map.device))
                continue

            gwd_losses = []
            pred_boxes = []
            for idx, gt_center in enumerate(gt_centers):
                center_y, center_x = gt_center
                gt_box = gt_boxes[idx]                                        # [8]
                pred_box = regression_map[:, int(center_y), int(center_x)]    # [8]

                transformed_box = self.build_transformed_box(cfg, pred_box, center_y, center_x)

                _, gwd_loss = gaussian_wasserstein_distance(transformed_box, gt_box, decomposition=self.decomposition)
                gwd_losses.append(gwd_loss)
                pred_boxes.append(transformed_box)

            batch_losses.append(torch.mean(torch.stack(gwd_losses)))
            batch_pred_boxes.append(torch.stack(pred_boxes))

        if batch_losses:
            avg_loss_batch = torch.mean(torch.stack(batch_losses))
        else:
            avg_loss_batch = torch.tensor(0.0, device=regression_maps.device)

        batch_dict['train_predicted_boxes'] = batch_pred_boxes

        return avg_loss_batch, batch_dict
    
    def build_transformed_box(self, cfg, pred_box, y_center, x_center):
        """
        y_center: row index, which refers to range
        x_center: column index, which refers to azimuth
        """
        range_meter = y_center * 118/255

        if cfg.RDR_PROCESSING_METHOD == "upsample_azimuth":
            # 214 bins (indices 0-213) covering -53° to 53°
            azimuth_deg = x_center * 106/213 - 53
        elif cfg.USE_DATA_PADDING:
            # 112 bins (indices 0-111) with padding
            azimuth_deg = x_center * 106/111 - 53
        else:
            # 107 bins (indices 0-106)
            azimuth_deg = x_center - 53

        azimuth_rad = torch.deg2rad(azimuth_deg)

        x = range_meter * torch.cos(azimuth_rad)
        
        # Need to flip due to a different coordinate system with the ground truth
        y = range_meter * torch.sin(azimuth_rad) * (-1)

        absolute_center_x = pred_box[0] + x
        absolute_center_y = pred_box[1] + y
        absolute_center_z = pred_box[2]

        pred_box_transformed = torch.stack([
            absolute_center_x, absolute_center_y, absolute_center_z,
            pred_box[3], pred_box[4], pred_box[5],
            pred_box[6], pred_box[7]
        ])

        return pred_box_transformed


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, batch_dict):
        predicted_boxes = batch_dict['train_predicted_boxes']       # list[Tensor]
        gt_list_tensors = batch_dict['gt_list_tensors']             # list[Tensor]
        
        losses = []
        for batch_idx in range(len(predicted_boxes)):
            pred_boxes = predicted_boxes[batch_idx]         # [num_gt, 8]
            gt_boxes = gt_list_tensors[batch_idx]           # [num_gt, 8]

            if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
                losses.append(torch.tensor(0.0, dtype=torch.float32, device=pred_boxes.device))
                continue

            loss = F.smooth_l1_loss(pred_boxes, gt_boxes, reduction='mean')
            losses.append(loss)

        return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=pred_boxes.device)


class ClassificationLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        
        # Order: 'Sedan', 'Bus or Truck', 'Pedestrian', 'Motorcycle', 'Bicycle', 'Pedestrian Group', 'Bicycle Group'
        # Radar-only objects for the train set without ROI-filtering:
        # 'Sedan': 28074, 'Bus or Truck': 5800, 'Bicycle': 586, 
        # 'Pedestrian': 2318, 'Motorcycle': 190

        class_counts = torch.tensor([28074, 5800, 2318, 190, 586], dtype=torch.float)
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(class_counts)  # Normalize weights
        # NOTE: Result is tensor([0.0234, 0.1133, 0.2835, 3.4585, 1.1213]); could clamp weights
        # weights = torch.clamp(weights, max=2.0)

        self.loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    def forward(self, cfg, batch_dict):
        classification_maps = batch_dict['classification']          # [B, num_classes, H, W]
        gt_lists = batch_dict['gt_lists']                           # list[list]
        gt_list_tensors = batch_dict['gt_list_tensors']             # list[Tensor]
        gt_center_points = batch_dict['gt_center_points']           # list[Tensor]

        batch_losses = []
        B, C, H, W = classification_maps.shape
        for batch_idx in range(B):
            classification_map = classification_maps[batch_idx]     # [num_classes, H, W]
            gt_list = gt_lists[batch_idx]                           # list of ground truth boxes
            gt_boxes = gt_list_tensors[batch_idx]                   # [num_gt, 8]
            gt_centers = gt_center_points[batch_idx]                # [num_gt, 2] in RA

            if gt_boxes.numel() == 0:
                # print("WARNING: No GT boxes in this sample.")
                batch_losses.append(torch.tensor(0.0, dtype=torch.float32, device=classification_map.device))
                continue

            cls_losses = []
            for gt, gt_center in zip(gt_list, gt_centers):
                # Convert ground truth class string to index
                label = cfg.LABEL_IDX_DICT.get(gt[0])
                
                if label is None or not (0 <= label < C):
                    print(f"ERROR: Invalid label '{gt[0]}' converted to {label} (must be in [0, {C-1}])")
                    continue

                # Read out classification scores at the center point
                center_y, center_x = gt_center
                int_y, int_x = int(center_y), int(center_x)

                if not (0 <= int_y < H) or not (0 <= int_x < W):
                    print(f"ERROR: Center coord out of bounds! y={center_y} (0,{H-1}), x={center_x} (0,{W-1})")
                    continue
                
                class_scores = classification_map[:, int(center_y), int(center_x)]  # [num_classes]
                
                if torch.isnan(class_scores).any():
                    print(f"ERROR: NaN values found in class scores! ({class_scores.tolist()}) at (y={int_y}, x={int_x})")
                    continue

                if torch.isinf(class_scores).any():
                    print(f"ERROR: Inf values found in class scores! ({class_scores.tolist()}) at (y={int_y}, x={int_x})")
                    continue

                if torch.isneginf(class_scores).all():
                    print(f"ERROR: All class_scores are -inf! ({class_scores.tolist()}) at (y={int_y}, x={int_x})")
                    continue

                if torch.isfinite(class_scores).sum() < C:
                    print(f"ERROR: Not all class_scores are finite ({class_scores.tolist()}) at (y={int_y}, x={int_x})")
                    continue
                
                # Safeguard against extreme values
                class_scores = torch.clamp(class_scores, min=-20, max=20)
                class_scores = class_scores.unsqueeze(0)  # [1, num_classes]
                target = torch.tensor([label], dtype=torch.long, device=classification_map.device)  # [1]
                cls_loss = self.loss_fn(class_scores, target)

                if torch.isfinite(cls_loss):
                    cls_losses.append(cls_loss)
                else:
                    print(f"Skipping NaN/Inf cls_loss: scores={class_scores}, target={target}")

            if cls_losses:
                batch_losses.append(torch.mean(torch.stack(cls_losses)))
            else:
                # All GTs were skipped, assign zero loss for this batch index
                batch_losses.append(torch.tensor(0.0, dtype=torch.float32, device=classification_map.device))

        if batch_losses:
            return torch.mean(torch.stack(batch_losses))
        else:
            return torch.tensor(0.0, device=classification_maps.device)