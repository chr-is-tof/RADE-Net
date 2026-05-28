import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
from matplotlib import pyplot as plt
import torch
import numpy as np
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import os
import pickle
import pandas as pd
import seaborn as sns

from ops.transformation import transform_RA_map_to_XY_map
from ops.roi import filter_boxes_in_roi, NARROW, WIDE


def plot_bbox(cfg, x0, y0, w, h, angle=0, color='cyan', linewidth=1, label=None, conf=None, box_type='gt'):
    ax = plt.gca()
    # Ensure all values are Python floats (not torch tensors)
    if torch.is_tensor(x0): x0 = x0.cpu().item()
    if torch.is_tensor(y0): y0 = y0.cpu().item()
    if torch.is_tensor(w): w = w.cpu().item()
    if torch.is_tensor(h): h = h.cpu().item()
    x0 = x0 - w / 2  # Adjust for center
    y0 = y0 - h / 2  # Adjust for center
    rect = patches.Rectangle((x0, y0), w, h, angle=angle, rotation_point='center', linewidth=linewidth, edgecolor=color, facecolor='none') #, facecolor=(0, 1, 1, 0.3))
    ax.add_patch(rect)

    if label is not None and box_type == 'gt':
        # ax.text(x0 - 0.5, y0 - 3, label, fontsize=8, color='cyan')
        # Annotate with a label (e.g., class name and confidence)
        ax.annotate(
            label,  # label text
            (x0, y0),      # position (bottom-left of box)
            color=color,
            fontsize=5,
            weight='bold',
            xytext=(5, -5),  # offset (pixels)
            textcoords='offset points',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
        )

    if label is not None and box_type == 'pred':
        ax.annotate(
            label,  # label text
            (x0, y0),      # position (bottom-left of box)
            color=color,
            fontsize=5,
            weight='bold',
            xytext=(5, 12.2),  # offset (pixels)
            textcoords='offset points',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
        )

    if (cfg.MODE == 'test' or cfg.MODE == 'visualize_model' or
        cfg.MODE == 'visualize_pkl' or cfg.MODE == 'paper') and conf is not None:
        ax.annotate(
            f'{conf:.3f}',  # label text
            (x0, y0),      # position (bottom-left of box)
            color=color,
            fontsize=5,
            weight='bold',
            xytext=(5, 5),  # offset (pixels)
            textcoords='offset points',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
        )


def plot_all(cfg, batch_dict, cam_paths, batch_idx, loss_dict=None):
    heatmap = batch_dict['heatmap']                     # [B, C, H, W]
    gt_lists = batch_dict['gt_lists']                   # list[list]
    gt_center_points = batch_dict['gt_center_points']   # list[Tensor]
    input_data = batch_dict['rdr_era_dra']              # [B, C, H, W]
    if cfg.MODE == 'train':
        box_preds = batch_dict['train_predicted_boxes']
    elif cfg.MODE == 'test' or cfg.MODE == 'visualize_model':
        box_preds = batch_dict['predicted_boxes']
    flip_horizontally = cfg.FLIP_GT_HORIZONTALLY
    
    B, C, H, W = heatmap.shape
    for b in range(B):
        hmap = heatmap[b]                               # [C, H, W]
        gt_list = gt_lists[b]                           # list of GT boxes for this batch
        gt_center = gt_center_points[b]                 # Tensor of GT center points for this batch
        data = input_data[b]                            # [C, H, W]
        data = torch.max(data[:, :, :107], dim=0)[0]    # Remove padding and max over channels
        cam_path = cam_paths[b]                         # Get the corresponding cam path
        box_pred = box_preds[b] if box_preds else None  # Get the corresponding box predictions if available

        ncols = 4
        nrows = int(np.ceil(C / ncols)) + 2      # + 2 for data and cam pic
        # fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
        fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
        # gs = gridspec.GridSpec(nrows, ncols, height_ratios=[1] * nrows)
        gs = gridspec.GridSpec(nrows, ncols)
        # if nrows == 1:
        #     axes = np.expand_dims(axes, axis=0)  # Ensure axes is always 2D

        hmap_global_min = hmap.min().item()
        hmap_global_max = hmap.max().item()
        for c in range(C):
            row = c // ncols
            col = c % ncols
            ax = fig.add_subplot(gs[row, col])

            class_name = cfg.IDX_LABEL_DICT.get(c, f"Class {c}")

            # --- Left column: Heatmap ---
            im = ax.imshow(hmap[c].detach().cpu().numpy(), cmap='inferno', aspect='auto', 
                           origin='lower', vmin=hmap_global_min, vmax=hmap_global_max)
            ax.set_title(f"Heatmap: {class_name}")
            ax.set_xlabel('Azimuth (bins)')
            ax.set_ylabel('Range (bins)')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # GT centers are in RA coordinates (x=range, y=azimuth)
            if gt_list is not None:
                for idx, gt in enumerate(gt_list):
                    cls = gt[0]
                    if cls == class_name:
                        cx, cy = gt_center[idx]
                        # if flip_horizontally:
                        #     cy = -cy
                        plt.sca(ax)
                        plot_bbox(cfg, cy, cx, w=5, h=10, angle=0, color='cyan', linewidth=1)
            
            # # --- Right: Transformed XY map ---
            # ax_right = axes[c, 1]
            # xy_heatmap = transform_RA_map_to_XY_map(hmap[c].detach().cpu().numpy())
            # sc = ax_right.scatter(xy_heatmap[:, 1], xy_heatmap[:, 0], c=xy_heatmap[:, 2], cmap='inferno', s=1)
            # ax_right.set_title(f"Transformed XY: {class_name}")
            # ax_right.axis('equal')
            # ax_right.axis('off')
            # plt.colorbar(sc, ax=ax_right, fraction=0.046, pad=0.04)
            # if flip_horizontally:
            #     ax_right.invert_xaxis()

        # After plotting heatmaps
        row_for_input = (C // ncols)
        col_for_input = (C % ncols)
        ax_input = fig.add_subplot(gs[row_for_input, col_for_input])
        # ax_input = fig.add_subplot(gs[-2, 0])
        im_input = ax_input.imshow(data.detach().cpu().numpy(), cmap='inferno', aspect='auto', origin='lower')
        ax_input.set_title("Input Data")
        ax_input.set_xlabel('Azimuth (bins)')
        ax_input.set_ylabel('Range (bins)')
        plt.colorbar(im_input, ax=ax_input, fraction=0.046, pad=0.04)

        # Visualizes the transformed data on the second-to-last row to the right of the data
        ax_xy_input = fig.add_subplot(gs[-2, 1:-1])
        xy_input = transform_RA_map_to_XY_map(data.detach().cpu().numpy())
        ax_xy_input.scatter(xy_input[:, 1], xy_input[:, 0], c=xy_input[:, 2], cmap='inferno', s=1)
        ax_xy_input.set_title("Transformed Input Data")
        ax_xy_input.set_xlabel('Azimuth (m)')
        ax_xy_input.set_ylabel('Range (m)')
        plt.colorbar(ax_xy_input.collections[0], ax=ax_xy_input, fraction=0.046, pad=0.04)
        for gt in gt_list:
            label = gt[0]
            cx, cy, cz, yaw, l, w, h = gt[1]
            if flip_horizontally:
                cy = -cy
            plot_bbox(cfg, cy, cx, w, l, angle=np.rad2deg(yaw), color='cyan', label=label)
        
        # Box prediction is in form [dx, dy, dz, l, w, h, sin(yaw), cos(yaw), conf.score, class, ra_azi, ra_range]
        if box_pred is not None:
            for box in box_pred:
                if cfg.MODE == 'train':
                    dx, dy, dz, bl, bw, bh, sinyaw, cosyaw = box.detach().cpu().numpy()
                    conf = None
                elif cfg.MODE == 'test' or cfg.MODE == 'visualize_model':
                    dx, dy, dz, bl, bw, bh, sinyaw, cosyaw, conf = box.detach().cpu().numpy()[:9]
                yaw = np.arctan2(sinyaw, cosyaw)
                if flip_horizontally:
                    dy = -dy
                plot_bbox(cfg, dy, dx, bw, bl, angle=np.rad2deg(yaw), color='springgreen', conf=conf)

        # On the last row place the camera image
        ax_image = fig.add_subplot(gs[-1, :])
        cam_img = mpimg.imread(cam_path)
        ax_image.imshow(cam_img)
        ax_image.set_title("Camera Image")
        ax_image.axis('off')

        if loss_dict is not None:
            fig.suptitle(f'Current Batch Loss for Epoch: {cfg.CURRENT_EPOCH},' \
                         f' FocalLoss: {loss_dict["focal_loss"]:.5f}, GWDLoss: {loss_dict["gwd_loss"]:.5f},' \
                         f' SmoothL1Loss: {loss_dict["smooth_l1_loss"]:.5f}', y=0.99)
        else:
            fig.suptitle(f'Inference', y=0.99)

        subset = cam_path.split('/')[-3]
        # Take any GT (e.g., 0), then 6th entry is tuple of indicices for rdr, ldr64, camf, ...
        frame_idx = gt_list[0][5][2] if gt_list else 'UNK'  # Use frame index from GT if available
        plt.tight_layout()
        if cfg.MODE == 'train':
            save_path = f'{cfg.PATH_TO_PLOTS_TRAIN}/vis_e{cfg.CURRENT_EPOCH}_b{batch_idx}_s{subset}_f{frame_idx}.png'
        elif cfg.MODE == 'test':
            save_path = f'{cfg.PATH_TO_PLOTS_INFERENCE}/inf_vis_s{subset}_f{frame_idx}.png'
        elif cfg.MODE == 'visualize_model':
            save_path = f'{cfg.PATH_TO_PLOTS_INFERENCE}/vis_s{subset}_f{frame_idx}.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=200, format='png')
        plt.close()


def visualize(cfg, batch_dict, cam_paths, batch_idx, loss_dict):
    # if 'train' or 'test' maybe change behaviour
    plot_all(cfg, batch_dict, cam_paths, batch_idx, loss_dict)


def plot_mAP_per_subset(mAP_dict, save_path):
    subsets = list(mAP_dict.keys())
    mAP_values = [mAP_dict[subset] for subset in subsets]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(subsets, mAP_values, color='skyblue')
    plt.ylim(0, 1)
    plt.xlabel('Subset')
    plt.ylabel('mAP')
    plt.title('mAP per Subset')

    # Annotate bars with mAP values
    for bar, mAP in zip(bars, mAP_values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{mAP:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_pkl(cfg, frames_pred_boxes, frames_gt_lists, box_paths, subsets):
    if not (len(frames_pred_boxes) == len(frames_gt_lists) == len(box_paths)):
        raise ValueError("Length of frames_pred_boxes, frames_gt_lists, and box_paths must be the same.")
    
    # Setup folders
    for subset in subsets:
        if not os.path.exists(f'{cfg.PATH_TO_PLOTS_INFERENCE}/{cfg.INF_MODEL}/{subset}'):
            os.makedirs(f'{cfg.PATH_TO_PLOTS_INFERENCE}/{cfg.INF_MODEL}/{subset}')
    
    # Create placeholder radar cone in black
    placeholder_map = np.zeros((256, 112))
    xy_placeholder = transform_RA_map_to_XY_map(placeholder_map)

    save_path = f'{cfg.PATH_TO_PLOTS_INFERENCE}/{cfg.INF_MODEL}'

    for frame_idx in range(len(frames_pred_boxes)):
        frame_pred_boxes = frames_pred_boxes[frame_idx]
        frame_gt_list = frames_gt_lists[frame_idx]
        box_path = box_paths[frame_idx]

        frame_pred_boxes, frame_gt_list = filter_boxes_in_roi(cfg, frame_pred_boxes, frame_gt_list)

        path_parts = box_path.split('_')
        subset = path_parts[-3]
        frame_number = path_parts[-1].split('.')[0]

        plt.figure(figsize=(12, 6))

        plt.scatter(xy_placeholder[:, 1], xy_placeholder[:, 0], c='black', s=1)

        # Plot ground truth boxes
        for gt in frame_gt_list:
            label = gt[0]
            cx, cy, cz, yaw, l, w, h = gt[1]
            if cfg.FLIP_GT_HORIZONTALLY:
                cy = -cy
            plot_bbox(cfg, cy, cx, w, l, angle=np.rad2deg(yaw), color='cyan', label=label)

        # Plot predicted boxes
        for box in frame_pred_boxes:
            dx, dy, dz, bl, bw, bh, sinyaw, cosyaw, conf, label = box[:10]
            label = cfg.IDX_LABEL_DICT.get(int(label), f"Class {int(label)}")
            yaw = np.arctan2(sinyaw, cosyaw)
            if cfg.FLIP_GT_HORIZONTALLY:
                dy = -dy
            plot_bbox(cfg, dy, dx, bw, bl, angle=np.rad2deg(yaw), color='springgreen', conf=conf, label=label, box_type='pred')
        
        plt.title(f'Subset: {subset}, Frame: {frame_number}')
        plt.xlabel('Azimuth (m)')
        plt.ylabel('Range (m)')
        diff = abs(len(frame_gt_list) - len(frame_pred_boxes))
        plt.savefig(f'{save_path}/{subset}/{diff}_pkl_vis_s{subset}_f{frame_number}.png', bbox_inches='tight', dpi=200)
        plt.close()


def draw_box_3d_wireframe(ax, cx, cy, cz, length, width, height, heading, 
                          color='cyan', linestyle='-', linewidth=2, label=None):
    """
    Draw a 3D bounding box wireframe on a 3D matplotlib axis.
    
    Args:
        ax: matplotlib 3D axis
        cx, cy, cz: Center coordinates
        length, width, height: Box dimensions
        heading: Rotation angle around Z-axis (radians)
        color, linestyle, linewidth: Visualization style
        label: Text label to display above the box
    """
    # Define 8 corner points in local frame (before rotation)
    half_l, half_w, half_h = length/2, width/2, height/2
    corners_local = np.array([
        # Bottom face (z = cz - half_h)
        [-half_l, -half_w, -half_h],  # 0: back-left-bottom
        [ half_l, -half_w, -half_h],  # 1: front-left-bottom
        [ half_l,  half_w, -half_h],  # 2: front-right-bottom
        [-half_l,  half_w, -half_h],  # 3: back-right-bottom
        # Top face (z = cz + half_h)
        [-half_l, -half_w,  half_h],  # 4: back-left-top
        [ half_l, -half_w,  half_h],  # 5: front-left-top
        [ half_l,  half_w,  half_h],  # 6: front-right-top
        [-half_l,  half_w,  half_h],  # 7: back-right-top
    ])
    
    # Rotation matrix around Z-axis
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotation_matrix = np.array([
        [cos_h, -sin_h, 0],
        [sin_h,  cos_h, 0],
        [0,      0,     1]
    ])
    
    # Rotate and translate corners to world frame
    corners_world = corners_local @ rotation_matrix.T
    corners_world += np.array([cx, cy, cz])
    
    # Define edges connecting corners
    edge_connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
    ]
    
    # Draw all edges
    for start_idx, end_idx in edge_connections:
        points = corners_world[[start_idx, end_idx]]
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2],
                 color=color, linestyle=linestyle, linewidth=linewidth)
    
    # Add label above the box
    if label is not None:
        ax.text(cx, cy, cz + height/2 + 0.5, label, 
               color=color, fontsize=8, weight='bold',
               bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3'))


def paper_radar(cfg, rdr_path, save_path, pred_boxes, gt_list, roi='narrow'):
    rdr_data = np.load(rdr_path)  # Assuming shape is (C, H, W)
    rdr_data = np.max(rdr_data, axis=0)  # Max over channels
    
    xy_map = transform_RA_map_to_XY_map(rdr_data)

    # Extract x, y, and intensity values
    x = xy_map[:, 1]  # Azimuth (x-axis)
    y = xy_map[:, 0]  # Range (y-axis)
    z = xy_map[:, 2]  # Intensity values

    width_px, height_px = 1608, 1000
    dpi=200

    fig = plt.figure(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)
    #fig = plt.figure()
    # plt.scatter(xy_map[:, 1], xy_map[:, 0], c=xy_map[:, 2], cmap='viridis', s=1)

    # Get the 'jet' colormap
    jet = plt.cm.get_cmap('jet')
    # Sample the first 50% of the colormap
    colors = jet(np.linspace(0, 0.5, 256))  # 0 to 0.5 for first 50%
    # Create a new colormap from these colors
    jet_half = mcolors.LinearSegmentedColormap.from_list('jet_half', colors)

    # Create filled contour plot using triangulation
    plt.tricontourf(x, y, z, levels=50, cmap='jet')

    # Plot ground truth boxes
    for gt in gt_list:
        label = gt[0]
        cx, cy, cz, yaw, l, w, h = gt[1]
        if cfg.FLIP_GT_HORIZONTALLY:
            cy = -cy
        plot_bbox(cfg, cy, cx, w, l, angle=np.rad2deg(yaw), color='white', label=label, box_type='gt')

    # Plot predicted boxes
    for box in pred_boxes:
        dx, dy, dz, bl, bw, bh, sinyaw, cosyaw, conf, label = box[:10]
        yaw = np.arctan2(sinyaw, cosyaw)
        label = cfg.IDX_LABEL_DICT.get(int(label), f"Class {int(label)}")
        if cfg.FLIP_GT_HORIZONTALLY:
            dy = -dy
        plot_bbox(cfg, dy, dx, bw, bl, angle=np.rad2deg(yaw), color='red', label=label, conf=conf, box_type='pred')

    # Include ROI boundaries as dashed lines
    # ROI is a dictionary with 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
    if roi == 'narrow':
        roi_limits = NARROW
    elif roi == 'wide':
        roi_limits = WIDE
    else:
        raise ValueError("roi must be 'narrow' or 'wide'")
    
    plt.plot([roi_limits['y_min'], roi_limits['y_max'], roi_limits['y_max'], roi_limits['y_min'], roi_limits['y_min']],
             [roi_limits['x_min'], roi_limits['x_min'], roi_limits['x_max'], roi_limits['x_max'], roi_limits['x_min']],
             color='black', linestyle='--')
    
    plt.axis('off')
    #plt.ylim([0, 80])
    #plt.xlim([-30, 30])
    #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #plt.subplots_adjust(left=0, right=0, top=0, bottom=0)
    #plt.savefig(save_path, dpi=dpi, pad_inches=0)
    plt.savefig(save_path,dpi=dpi, pad_inches=0, bbox_inches='tight')
    plt.close()


def paper_camera_image(cam_path, save_path, side='left'):
    img = mpimg.imread(cam_path)
    
    # The image is a concatenation of left and right images side by side
    if side == 'left':
        img = img[:, :img.shape[1] // 2, :]
    elif side == 'right':
        img = img[:, img.shape[1] // 2:, :]
    else:
        raise ValueError("side must be 'left' or 'right'")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.margins(0)
    # plt.title('Camera Image')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100, pad_inches=0)
    plt.close()


def paper_plots(cfg, weather=None, chosen_subset=None, chosen_frames=None, roi='narrow'):
    if weather is None:
        raise ValueError("Please provide weather for paper_plots function.")
    if chosen_subset is None:
        raise ValueError("Please provide chosen_subset for paper_plots function.")
    if chosen_frames is None:
        raise ValueError("Please provide chosen_frames for paper_plots function.")

    # Setup save directory
    directory_path = cfg.PATH_TO_FOLDERS
    save_path = f'{directory_path}/paper_plots/{weather}/{chosen_subset[0]}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Find all relevant pkl files based on chosen subsets and frames
    paths = []
    pkl_path = f'{cfg.PATH_TO_SAVED_INFERENCE_DATA}/{cfg.INF_MODEL}'
    print(f"Path to inference: {pkl_path}")
    for file in os.listdir(pkl_path):
        if file.endswith('.pkl'):
            # Assumes filename format: boxes_subset_{subset}_frame_{frame}.pkl
            parts = file.split('_')
            subset = parts[2]
            frame = parts[4].split('.')[0]
            if subset in chosen_subset and frame in chosen_frames:
                paths.append(os.path.join(pkl_path, file))
    paths.sort()

    # Process each pkl file and make plots
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        # Read out subset and frame from path
        parts = path.split('_')
        subset = parts[-3]
        frame_idx = parts[-1].split('.')[0]

        with open(path, 'rb') as f:
            data = pickle.load(f)

        pred_boxes = data['predicted_boxes']
        gt_boxes = data['gt_list']
        
        # Filter based on chosen ROI
        pred_boxes, gt_boxes = filter_boxes_in_roi(cfg, pred_boxes, gt_boxes)
        
        # Assuming at least one gt box exists
        rdr, ldr64, camf, ldr128, camr = gt_boxes[0][-1]
        
        # Build paths for loading data
        data_path = ... # '.../Dataset/K-Radar_reduced'
        rdr_path = f'{data_path}/{subset}/DERA_tesseract/DERA_tesseract_{rdr}.npy'
        cam_path = f'{data_path}/{subset}/cam-front/cam-front_{camf}.png'

        if not os.path.exists(rdr_path):
            raise FileNotFoundError(f"Radar data not found: {rdr_path}")
        
        if not os.path.exists(cam_path):
            raise FileNotFoundError(f"Camera image not found: {cam_path}")
        
        num_preds = len(pred_boxes)
        num_gts = len(gt_boxes)

        # Plot and save radar data
        paper_radar(cfg, rdr_path, f'{save_path}/{num_gts}_{num_preds}_radar_s{subset}_f{frame_idx}.png', pred_boxes, gt_boxes, roi=roi)

        # Plot and save camera image
        paper_camera_image(cam_path, f'{save_path}/camera_image_s{subset}_f{frame_idx}.png', side='left')


def plot_tp_fp_fn_total_grid(
    df: pd.DataFrame,
    category_col: str,                 # e.g. "weather"
    category_name: str | None = None,  # e.g. "Weather" (display)
    label_col: str = "label",
    label_name: str = "Class",         # display name for label axis
    tp_col: str = "TP",
    fp_col: str = "FP",
    fn_col: str = "FN",
    label_order=None,
    category_order=None,
    log_total: bool = False,
    figsize=(16, 10),
    cmap_tp="Blues",
    cmap_fp="Purples",
    cmap_fn="Reds",
    cmap_total="Greys",
    annot_kws=None,
    save_path=None
):
    d = df.copy()

    # display defaults
    if category_name is None:
        category_name = category_col

    # Ensure numeric
    for c in (tp_col, fp_col, fn_col):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    # "Support": ground-truth present = TP + FN
    d["Total"] = d[tp_col] + d[fn_col]

    if label_order is None:
        label_order = sorted(d[label_col].dropna().unique().tolist())
    if category_order is None:
        category_order = sorted(d[category_col].dropna().unique().tolist())

    def make_mat(value_col):
        return (d.pivot_table(index=label_col, columns=category_col, values=value_col, aggfunc="sum")
                 .reindex(index=label_order, columns=category_order)
                 .fillna(0))

    mats = {
        "TP": make_mat(tp_col),
        "FP": make_mat(fp_col),
        "FN": make_mat(fn_col),
        "Total": make_mat("Total"),
    }

    total_for_color = mats["Total"].copy()
    if log_total:
        total_for_color = np.log10(total_for_color.replace(0, np.nan))

    if annot_kws is None:
        annot_kws = dict(fontsize=8)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False, sharey=False)
    axes = axes.ravel()

    sns.heatmap(mats["TP"], ax=axes[0], cmap=cmap_tp, annot=True, fmt=".0f",
                cbar_kws={"label": "Number of TP"}, annot_kws=annot_kws)
    axes[0].set_title("True Positives (TP)")

    sns.heatmap(mats["FP"], ax=axes[1], cmap=cmap_fp, annot=True, fmt=".0f",
                cbar_kws={"label": "Number of FP"}, annot_kws=annot_kws)
    axes[1].set_title("False Positives (FP)")

    sns.heatmap(mats["FN"], ax=axes[2], cmap=cmap_fn, annot=True, fmt=".0f",
                cbar_kws={"label": "Number of FN"}, annot_kws=annot_kws)
    axes[2].set_title("False Negatives (FN)")

    sns.heatmap(total_for_color, ax=axes[3], cmap=cmap_total,
                annot=mats["Total"], fmt=".0f",
                cbar_kws={"label": "log10(TP+FN)" if log_total else "TP+FN"},
                annot_kws=annot_kws)
    axes[3].set_title("Total Support (TP+FN)")

    # axis labels + tick styling
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel(category_name)
        ax.set_ylabel(label_name)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()


def plot_tp_fp_fn_metrics_grid(
    df: pd.DataFrame,
    category_col: str,
    category_name: str | None = None,
    label_col: str = "label",
    label_name: str = "Class",
    tp_col: str = "TP",
    fp_col: str = "FP",
    fn_col: str = "FN",
    label_order=None,
    category_order=None,
    figsize=(18, 12),
    cmap="copper",
    annot_support: bool = True,
    metric_fmt: str = ".2f",
    support_fmt: str = "n={:d}",
    annot_fontsize: int = 8,
    save_path=None
):
    d = df.copy()
    if category_name is None:
        category_name = category_col

    for c in (tp_col, fp_col, fn_col):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    if label_order is None:
        label_order = sorted(d[label_col].dropna().unique().tolist())
    if category_order is None:
        category_order = sorted(d[category_col].dropna().unique().tolist())

    def mat(value_col):
        return (d.pivot_table(index=label_col, columns=category_col, values=value_col, aggfunc="sum")
                 .reindex(index=label_order, columns=category_order)
                 .fillna(0))

    TP = mat(tp_col)
    FP = mat(fp_col)
    FN = mat(fn_col)

    den_prec = (TP + FP).replace(0, np.nan)
    den_rec  = (TP + FN).replace(0, np.nan)
    den_all  = (TP + FP + FN).replace(0, np.nan)

    precision = (TP / den_prec).fillna(0)
    recall    = (TP / den_rec).fillna(0)
    fdr       = (FP / den_prec).fillna(0)
    fnr       = (FN / den_rec).fillna(0)

    f1_den = (precision + recall).replace(0, np.nan)
    f1_score = (2 * precision * recall / f1_den).fillna(0)

    jaccard = (TP / den_all).fillna(0)

    def metric_annot(M: pd.DataFrame) -> pd.DataFrame:
        return M.map(lambda x: format(float(x), metric_fmt))

    def add_support(annot: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
        # support is numeric dataframe aligned with annot
        s = support.fillna(0).astype(int).map(lambda x: support_fmt.format(int(x)))
        return annot + "\n" + s

    annot_precision = metric_annot(precision)
    annot_recall    = metric_annot(recall)
    annot_fdr       = metric_annot(fdr)
    annot_fnr       = metric_annot(fnr)
    annot_f1        = metric_annot(f1_score)
    annot_jaccard   = metric_annot(jaccard)

    if annot_support:
        annot_precision = add_support(annot_precision, TP + FP)
        annot_fdr       = add_support(annot_fdr,       TP + FP)

        annot_recall    = add_support(annot_recall,    TP + FN)
        annot_fnr       = add_support(annot_fnr,       TP + FN)

        annot_f1        = add_support(annot_f1,        TP + FP + FN)
        annot_jaccard   = add_support(annot_jaccard,   TP + FP + FN)

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=False, sharey=False)
    axes = axes.ravel()

    common = dict(cmap=cmap, vmin=0, vmax=1, fmt="", annot_kws={"fontsize": annot_fontsize})

    sns.heatmap(precision, ax=axes[0], annot=annot_precision, cbar_kws={"label": "Precision"}, **common)
    axes[0].set_title("Precision: TP/(TP+FP)")

    sns.heatmap(recall, ax=axes[1], annot=annot_recall, cbar_kws={"label": "Recall"}, **common)
    axes[1].set_title("Recall: TP/(TP+FN)")

    sns.heatmap(fdr, ax=axes[2], annot=annot_fdr, cbar_kws={"label": "FDR"}, **common)
    axes[2].set_title("False Discovery Rate: FP/(TP+FP)")

    sns.heatmap(fnr, ax=axes[3], annot=annot_fnr, cbar_kws={"label": "FNR"}, **common)
    axes[3].set_title("False Negative Rate: FN/(TP+FN)")

    sns.heatmap(f1_score, ax=axes[4], annot=annot_f1, cbar_kws={"label": "F1"}, **common)
    axes[4].set_title("F1 Score: 2*Prec*Recall/(Prec+Recall)")

    sns.heatmap(jaccard, ax=axes[5], annot=annot_jaccard, cbar_kws={"label": "Jaccard"}, **common)
    axes[5].set_title("Jaccard Index: TP/(TP+FP+FN)")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        # ax.tick_params(axis="y", rotation=45)
        ax.set_xlabel(category_name)
        ax.set_ylabel(label_name)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == "__main__":
    pass