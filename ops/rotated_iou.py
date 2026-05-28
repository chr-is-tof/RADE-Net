import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_rotated_corners_np(box):
    # Convert to numpy if it's a tensor
    if torch.is_tensor(box):
        box = box.detach().cpu().numpy()

    x, y, z, l, w, h, sin_yaw, cos_yaw = box

    dx = l / 2  # length along x (range)
    dy = w / 2  # width along y (azimuth)

    corners = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy]
    ])

    rot = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw]
    ])

    rotated = corners @ rot.T
    rotated += np.array([x, y])

    return rotated.astype(np.float32)


# Shoelace formula
def polygon_area(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def rotated_iou_2d(box1, box2):
    poly1 = get_rotated_corners_np(box1)
    poly2 = get_rotated_corners_np(box2)

    # cv2.intersectConvexConvex expects shape (N,1,2)
    poly1_cv2 = poly1.reshape((-1, 1, 2))
    poly2_cv2 = poly2.reshape((-1, 1, 2))
    
    inter, inter_poly = cv2.intersectConvexConvex(poly1_cv2, poly2_cv2)
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    
    inter_area = inter if inter_poly is not None else 0.0
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def rotated_iou_3d(box1, box2):
    # 2D intersection area
    poly1 = get_rotated_corners_np(box1)
    poly2 = get_rotated_corners_np(box2)
    poly1_cv2 = poly1.reshape((-1, 1, 2))
    poly2_cv2 = poly2.reshape((-1, 1, 2))
    inter_area, inter_poly = cv2.intersectConvexConvex(poly1_cv2, poly2_cv2)
    area1 = polygon_area(poly1)
    area2 = polygon_area(poly2)
    inter_area = inter_area if inter_poly is not None else 0.0

    # Since 2D is already known, going to 3D is straightforward by simply adding height to it
    z1, h1 = box1[2].item(), box1[5].item()
    z2, h2 = box2[2].item(), box2[5].item()
    z1_min, z1_max = z1 - h1 / 2, z1 + h1 / 2
    z2_min, z2_max = z2 - h2 / 2, z2 + h2 / 2
    z_inter_min = max(z1_min, z2_min)
    z_inter_max = min(z1_max, z2_max)
    z_inter = max(0.0, z_inter_max - z_inter_min)

    # Computing volumes
    inter_vol = inter_area * z_inter
    vol1 = area1 * h1
    vol2 = area2 * h2
    union_vol = vol1 + vol2 - inter_vol

    if union_vol == 0:
        return 0.0
    
    return inter_vol / union_vol


def plot_rotated_boxes(box1, box2, idx=0):
    # Get corners
    corners1 = get_rotated_corners_np(box1)
    corners2 = get_rotated_corners_np(box2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Plot box1
    poly1 = Polygon(corners1, closed=True, edgecolor='blue', facecolor='none', linewidth=2, label='Box 1')
    ax.add_patch(poly1)
    ax.plot(corners1[:, 0], corners1[:, 1], 'bo')

    # Plot box2
    poly2 = Polygon(corners2, closed=True, edgecolor='red', facecolor='none', linewidth=2, label='Box 2')
    ax.add_patch(poly2)
    ax.plot(corners2[:, 0], corners2[:, 1], 'ro')

    # Plot centers
    ax.plot(box1[0], box1[1], 'bx', markersize=10, label='Center 1')
    ax.plot(box2[0], box2[1], 'rx', markersize=10, label='Center 2')

    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Rotated Bounding Boxes')
    plt.grid(True)
    plt.savefig(f'/home/wsluser/workspace/LRawFusion/K-Radar-main-repo/lraw_decoupeled/rotated_boxes_{idx}.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Swap axes: azimuth (x) is box[..., 1], range (y) is box[..., 0]
    # Plot box1
    poly1 = Polygon(np.stack([corners1[:, 1], corners1[:, 0]], axis=1), closed=True, edgecolor='blue', facecolor='none', linewidth=2, label='Box 1')
    ax.add_patch(poly1)
    ax.plot(corners1[:, 1], corners1[:, 0], 'bo')

    # Plot box2
    poly2 = Polygon(np.stack([corners2[:, 1], corners2[:, 0]], axis=1), closed=True, edgecolor='red', facecolor='none', linewidth=2, label='Box 2')
    ax.add_patch(poly2)
    ax.plot(corners2[:, 1], corners2[:, 0], 'ro')

    # Plot centers
    ax.plot(box1[1], box1[0], 'bx', markersize=10, label='Center 1')
    ax.plot(box2[1], box2[0], 'rx', markersize=10, label='Center 2')

    ax.legend()
    ax.set_xlabel('Azimuth (meters)')
    ax.set_ylabel('Range (meters)')
    ax.set_title('Rotated Bounding Boxes (Azimuth vs Range)')
    ax.invert_xaxis()  # Flips azimuth axis (horizontal)
    plt.grid(True)
    plt.savefig(f'/home/wsluser/workspace/LRawFusion/K-Radar-main-repo/lraw_decoupeled/rotated_boxes_ax_swapped_{idx}.png')
    plt.close()


def get_3d_box_corners(box):
    # box: [x, y, z, l, w, h, sin_yaw, cos_yaw]
    x, y, z, l, w, h, sin_yaw, cos_yaw = box
    dx = l / 2
    dy = w / 2
    dz = h / 2

    # 2D corners in box frame
    corners_2d = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy]
    ])

    # Rotation matrix
    rot = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw]
    ])
    rotated_2d = corners_2d @ rot.T

    # 3D corners (bottom and top faces)
    corners = []
    for z_offset in [-dz, dz]:
        for xy in rotated_2d:
            corners.append([x + xy[0], y + xy[1], z + z_offset])
    corners = np.array(corners).reshape(8, 3)
    return corners


def plot_3d_boxes(boxes, colors=None, labels=None, idx=0):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,0.5])  # Adjust aspect ratio if needed

    if colors is None:
        colors = ['blue', 'red', 'green', 'orange']
    if labels is None:
        labels = [f'Box {i+1}' for i in range(len(boxes))]

    for i, box in enumerate(boxes):
        corners = get_3d_box_corners(box.detach().cpu().numpy())
        # Swap axes: azimuth (x), range (y), height (z)
        faces = [
            [corners[j][[1,0,2]] for j in [0,1,2,3]],  # bottom
            [corners[j][[1,0,2]] for j in [4,5,6,7]],  # top
            [corners[j][[1,0,2]] for j in [0,1,5,4]],  # side
            [corners[j][[1,0,2]] for j in [1,2,6,5]],
            [corners[j][[1,0,2]] for j in [2,3,7,6]],
            [corners[j][[1,0,2]] for j in [3,0,4,7]]
        ]
        poly3d = Poly3DCollection(faces, alpha=0.3, facecolor=colors[i % len(colors)], edgecolor='k', linewidths=1)
        ax.add_collection3d(poly3d)
        # Plot center
        ax.scatter(box[1], box[0], box[2], color=colors[i % len(colors)], s=50, label=labels[i])

    ax.set_xlabel('Azimuth (meters)')
    ax.set_ylabel('Range (meters)')
    ax.set_zlabel('Height (meters)')
    ax.set_title('3D Rotated Bounding Boxes')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'/home/wsluser/workspace/LRawFusion/K-Radar-main-repo/lraw_decoupeled/rotated_boxes_3d_{idx}.png')
    plt.close()


if __name__ == "__main__":
    # Frame 00905, predicted box from model v8_1_epoch299
    box1 = torch.tensor([
        54.447,   # x
        3.2633,   # y
        -0.13086,  # z
        3.3026,   # l
        1.6262,   # w
        1.4421,   # h
        -0.0086683,# sin(yaw)
        0.94541   # cos(yaw)
    ], dtype=torch.float32)

    box2 = torch.tensor([
        53.847,   # x
        3.6433,   # y
        -0.49760,  # z
        3.2819,   # l
        1.6132,   # w
        1.4198,   # h
        0.020237, # sin(yaw)
        0.99980   # cos(yaw)
    ], dtype=torch.float32)

    # Frame 00911, predicted box from model v8_1_epoch299
    box3 = torch.tensor([
        54.413,    # x (range)
        4.1398,    # y (azimuth)
        -0.26169,   # z
        3.2328,    # l
        1.6108,    # w
        1.4546,    # h
        -0.013193,  # sin(yaw)
        0.98508    # cos(yaw)
    ], dtype=torch.float32)

    box4 = torch.tensor([
        54.347,    # x (range)
        4.5420,    # y (azimuth)
        -0.18320,   # z
        3.2819,    # l
        1.6132,    # w
        1.4198,    # h
        0.020237,  # sin(yaw)
        0.99980    # cos(yaw)
    ], dtype=torch.float32)
    
    iou = rotated_iou_2d(box1, box2)
    print(f"Rotated IoU: {iou:.4f}")

    iou = rotated_iou_2d(box3, box4)
    print(f"Rotated IoU: {iou:.4f}")

    iou3d = rotated_iou_3d(box1, box2)
    print(f"3D Rotated IoU: {iou3d:.4f}")

    iou3d = rotated_iou_3d(box3, box4)
    print(f"3D Rotated IoU: {iou3d:.4f}")
    
    plot_3d_boxes([box1, box2], idx=0)
    plot_3d_boxes([box3, box4], idx=1)
    exit()

    plot_rotated_boxes(box1, box2, 0)
    plot_rotated_boxes(box3, box4, 1)