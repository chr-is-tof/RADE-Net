import torch

from ops.loss import FocalLossGaussianContinuous, SoftFocalLoss, GWDLoss, SmoothL1Loss, ClassificationLoss

import torch.nn.functional as F


def generate_gaussian_heatmap(cfg, gt_lists, gt_center_points, B, C, H, W, device):
    """
    Generates Gaussian heatmaps for object centers.

    When USE_SINGULAR_HEATMAP is True, generates a single heatmap for all classes.
    Otherwise, generates per-class heatmaps.
    """
    sigma = cfg.SIGMA
    
    if cfg.USE_SINGULAR_HEATMAP:
        heatmap = torch.zeros((B, 1, H, W), device=device)
        y = torch.arange(0, H, device=device).view(H, 1).expand(H, W)
        x = torch.arange(0, W, device=device).view(1, W).expand(H, W)
        factor = 2 * sigma ** 2

        for b in range(B):
            heatmap_channel = torch.zeros((H, W), device=device)
            centers = gt_center_points[b]  # [num_gt, 2]
            for idx in range(centers.shape[0]):
                y_center, x_center = centers[idx]
                gaussian = torch.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / factor)
                heatmap_channel += gaussian
            heatmap[b, 0] = torch.clamp(heatmap_channel, 0, 1)
        return heatmap
    
    else:
        heatmap = torch.zeros((B, C, H, W), device=device)

        # Generate a coordinate grid for the heatmap
        y = torch.arange(0, H, device=device).view(H, 1).expand(H, W)  # Shape: [H, W]
        x = torch.arange(0, W, device=device).view(1, W).expand(H, W)  # Shape: [H, W]
        factor = 2 * sigma ** 2

        for b in range(B):
            gt_list = gt_lists[b]
            gt_centers = gt_center_points[b]    # Tensor [num_gt, 2]
            
            for c in range(C):
                # Translate channel idx into label
                label = cfg.IDX_LABEL_DICT.get(c)

                # Get all center points for the current label / channel
                class_indicies = [i for i, gt in enumerate(gt_list) if gt[0] == label]

                if not class_indicies:
                    continue  # No ground truth for this class in the current batch
                
                # Initialize an empty heatmap for the current channel in this batch
                heatmap_channel = torch.zeros((H, W), device=device)
                for idx in class_indicies:
                    y_center, x_center = gt_centers[idx]
                    
                    if cfg.GAUSSIAN_HEATMAP_SIGMA == 'adaptive':
                        length = gt_list[idx][1][4]
                        width = gt_list[idx][1][5]
                        factor = 2 * (max(length, width) * 0.3) ** 2

                    if cfg.GAUSSIAN_HEATMAP_SIGMA == 'anisotropic':
                        length = gt_list[idx][1][4]
                        width = gt_list[idx][1][5]
                        factor_x = 2 * (width / 6) ** 2
                        factor_y = 2 * (length / 6) ** 2

                        # Compute anisotropic Gaussian at all grid points
                        gaussian = torch.exp(-((x - x_center) ** 2 / factor_x + (y - y_center) ** 2 / factor_y))
                        heatmap_channel += gaussian  # Add the current Gaussian to the heatmap
                        continue

                    # Compute Gaussian at all grid points
                    gaussian = torch.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / factor)
                    heatmap_channel += gaussian  # Add the current Gaussian to the heatmap

                # Clamp the resulting heatmap to [0, 1] to avoid overlapping values exceeding 1 which is important for BCE loss
                heatmap[b, c] = torch.clamp(heatmap_channel, 0, 1)

        return heatmap
    

def focal_loss(cfg, batch_dict, loss_type="focal_loss_continuous"):
    heatmap = batch_dict['heatmap']                         # [B, C, H, W]
    gt_lists = batch_dict['gt_lists']                       # list[list]
    gt_center_points = batch_dict['gt_center_points']       # List of [num_gt, 2] tensors

    B, C, H, W = heatmap.shape
    if loss_type == "focal_loss_binary":
        # This implementation misses Gaussian binary map
        raise NotImplementedError("Focal loss for binary targets is not implemented yet.")

    elif loss_type == "focal_loss_continuous":
        gaussian_map = generate_gaussian_heatmap(cfg, gt_lists, gt_center_points, B, C, H, W, heatmap.device)
        criterion = FocalLossGaussianContinuous()
    
    elif loss_type == "soft_focal_loss":
        gaussian_map = generate_gaussian_heatmap(cfg, gt_lists, gt_center_points, B, C, H, W, heatmap.device)
        criterion = SoftFocalLoss()
    
    return criterion(heatmap, gaussian_map)


def bbox_regression_loss(cfg, batch_dict):
    criterion1 = GWDLoss(decomposition="eigenvalue")
    gwd_loss, batch_dict = criterion1(cfg, batch_dict)

    criterion2 = SmoothL1Loss()
    smooth_l1_loss = criterion2(batch_dict)

    return gwd_loss, smooth_l1_loss, batch_dict


def classification_loss(cfg, batch_dict):
    device = batch_dict['heatmap'].device
    criterion = ClassificationLoss(cfg, device)
    cls_loss = criterion(batch_dict)
    return cls_loss


def detection_loss(cfg, batch_dict, focal_loss_type='focal_loss_continuous'):
    # Heatmap loss
    loss_heatmap = focal_loss(cfg, batch_dict, loss_type=focal_loss_type)

    # Bounding box regression loss
    gwd_loss, smooth_l1_loss, batch_dict = bbox_regression_loss(cfg, batch_dict)

    # Compute classification loss if using a singular heatmap
    cls_loss = None
    if cfg.USE_SINGULAR_HEATMAP:
        cls_loss = classification_loss(cfg, batch_dict)

    loss = 0
    loss += 2 * (loss_heatmap / loss_heatmap.detach().mean())
    loss += gwd_loss / gwd_loss.detach().mean()
    loss += smooth_l1_loss / smooth_l1_loss.detach().mean()

    # When loss is 0 (no objects), avoid NaN by skipping cls_loss
    # Theoretically, cls_loss can be 0 due to perfect classification, 
    # but in practice this is rare, so we assume cls_loss > 0 indicates presence of objects
    if cls_loss is not None and cls_loss > 0:
        loss += cls_loss / cls_loss.detach().mean()

    loss_dict = {
        'focal_loss': loss_heatmap.item(),
        'gwd_loss': gwd_loss.item(),
        'smooth_l1_loss': smooth_l1_loss.item(),
        'total_loss': loss_heatmap.item() + gwd_loss.item() + smooth_l1_loss.item()
    }
    if cls_loss is not None:
        loss_dict['classification_loss'] = cls_loss.item()
        loss_dict['total_loss'] += cls_loss.item()

    #print(f"FocalLoss: {loss_heatmap.item():.10f}, GWDLoss: {gwd_loss.item():.10f}, SmoothL1Loss: {smooth_l1_loss.item():.10f}, TotalLoss: {loss_dict['total_loss']:.10f}")
    batch_dict['loss_dict'] = loss_dict

    return loss, batch_dict


def compute_losses(cfg, batch_dict, focal_loss_type='focal_loss_continuous'):
    '''Extensible wrapper function.'''

    loss, batch_dict = detection_loss(cfg, batch_dict, focal_loss_type=focal_loss_type)
    
    return loss, batch_dict


def test_gaussian_heatmap(cfg):
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 1, 3, 256, 112  # 1 batch, 3 classes, 256x112 heatmap

    # Example: two classes, each with one center
    gt_lists = [[
        ('ClassA', (20, 30)),  # (class_name, (y, x))
        ('ClassB', (40, 50)),
        ('ClassC', (230, 80))
    ]]
    gt_center_points = [torch.tensor([[20, 30], [40, 50], [230, 80]], dtype=torch.float32, device=device)]
   
    # Simulate label dictionary for channel mapping
    cfg.IDX_LABEL_DICT = {0: 'ClassA', 1: 'ClassB', 2: 'ClassC'}

    # Generate the heatmaps
    gaussian_heatmaps = generate_gaussian_heatmap(
        cfg, gt_lists, gt_center_points, B, C, H, W, device
    )

    # Plot each class channel
    # for c in range(C):
    #     plt.figure(figsize=(6, 5))
    #     plt.imshow(gaussian_heatmaps[0, c].cpu().numpy(), cmap="hot", origin="lower")
    #     plt.title(f"Gaussian Heatmap for Class {cfg.IDX_LABEL_DICT[c]}")
    #     plt.colorbar()
    #     plt.savefig(f"gaussian_heatmap_class{c}.png")
    #     plt.close()

    for c in range(C):
        plt.figure(figsize=(6, 5))
        plt.imshow(gaussian_heatmaps[0, c].cpu().numpy(), cmap="hot", origin="lower")
        plt.title(f"Gaussian Heatmap for Class {cfg.IDX_LABEL_DICT[c]}")
        plt.colorbar()
        plt.savefig(f"gaussian_heatmap_class_{cfg.IDX_LABEL_DICT[c]}.png")
        plt.close()

    # Transform into cartesian coordinates and plot
    def transform_ra_indices_to_cartesian(azimuth_idx, range_idx, use_padding=False):
        # Convert indices to meters/degrees
        range_meter = range_idx * 118/255

        if use_padding:
            azimuth_deg = azimuth_idx * 106/111 - 53
        else:
            azimuth_deg = azimuth_idx - 53

        # Convert to rad for trig. functions
        azimuth_rad = torch.deg2rad(azimuth_deg)

        # Compute cartesian coordinates
        x_longitudinal = range_meter * torch.cos(azimuth_rad)

        # Need to flip due to a different coordinate system with the ground truth
        y_lateral = range_meter * torch.sin(azimuth_rad)

        return x_longitudinal, y_lateral
    
    ys = torch.arange(0, H, device=device).view(H, 1).expand(H, W)  # Shape: [H, W]
    xs = torch.arange(0, W, device=device).view(1, W).expand(H, W)  # Shape: [H, W]
    for c in range(C):
        heatmap_channel = gaussian_heatmaps[0, c]
        non_zero_indices = torch.nonzero(heatmap_channel > 0.7, as_tuple=False)
        
        x_cart, y_cart = transform_ra_indices_to_cartesian(xs.float(), ys.float(), use_padding=True)
        
        gauss_x_cart = x_cart[non_zero_indices[:, 0], non_zero_indices[:, 1]]
        gauss_y_cart = y_cart[non_zero_indices[:, 0], non_zero_indices[:, 1]]

        plt.figure(figsize=(6, 6))
        plt.scatter(y_cart.cpu().numpy(), x_cart.cpu().numpy(), c='red', s=0.2)
        plt.scatter(gauss_y_cart.cpu().numpy(), gauss_x_cart.cpu().numpy(), c='blue', s=0.2)
        plt.title(f"Cartesian Coordinates for Class {cfg.IDX_LABEL_DICT[c]}")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(f"cartesian_coords_class_{cfg.IDX_LABEL_DICT[c]}.png")
        plt.close()


if __name__ == "__main__":
    from main import load_config
    cfg = load_config("configs/params_thomas.py")
    cfg.GAUSSIAN_HEATMAP_SIGMA = 'fixed'
    test_gaussian_heatmap(cfg)