import torch
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .rotated_iou import rotated_iou_2d


# From the paper https://proceedings.mlr.press/v139/yang21l/yang21l.pdf

# NOTE: Boxes already come in the format of (x, y, z, l, w, h, sin(yaw), cos(yaw))
# NOTE: Can avoid computing cos and sin
def box2gaussian(pred_box, gt_box):
    # x1, y1, z1, l1, w1, h1, sin1, cos1 = torch.unbind(boxes1, axis=1)
    # x2, y2, z2, l2, w2, h2, sin2, cos2 = torch.unbind(boxes2, axis=1)
    x1, y1, z1, l1, w1, h1, sin1, cos1 = pred_box
    x2, y2, z2, l2, w2, h2, sin2, cos2 = gt_box

    w1_half = w1 / 2
    h1_half = l1 / 2
    cos1_sq = cos1 ** 2
    sin1_sq = sin1 ** 2
    cos1sin1 = cos1 * sin1

    w2_half = w2 / 2
    h2_half = l2 / 2
    cos2_sq = cos2 ** 2
    sin2_sq = sin2 ** 2
    cos2sin2 = cos2 * sin2

    sigma1_11 = w1_half * cos1_sq + h1_half * sin1_sq
    sigma1_12 = (w1_half - h1_half) * cos1sin1
    sigma1_21 = sigma1_12
    sigma1_22 = w1_half * sin1_sq + h1_half * cos1_sq
    sigma1 = torch.stack([
        torch.stack([sigma1_11, sigma1_12]), 
        torch.stack([sigma1_21, sigma1_22])
    ])

    sigma2_11 = w2_half * cos2_sq + h2_half * sin2_sq
    sigma2_12 = (w2_half - h2_half) * cos2sin2
    sigma2_21 = sigma2_12
    sigma2_22 = w2_half * sin2_sq + h2_half * cos2_sq
    sigma2 = torch.stack([
        torch.stack([sigma2_11, sigma2_12]), 
        torch.stack([sigma2_21, sigma2_22])
    ])

    return x1, y1, x2, y2, sigma1, sigma2


def box2gaussian_batch(boxes):
    x, y, z, l, w, h, sin, cos = torch.unbind(boxes, axis=-1)

    w_half = w / 2
    h_half = l / 2
    cos_sq = cos ** 2
    sin_sq = sin ** 2
    cossin = cos * sin

    # print(x.shape, y.shape, w_half.shape, h_half.shape, cos_sq.shape, sin_sq.shape, cossin.shape)

    sigma_11 = w_half * cos_sq + h_half * sin_sq
    sigma_12 = (w_half - h_half) * cossin
    sigma_21 = sigma_12
    sigma_22 = w_half * sin_sq + h_half * cos_sq
    
    # print(sigma_11.shape, sigma_12.shape, sigma_21.shape, sigma_22.shape)
    # print(sigma_11, sigma_12, sigma_21, sigma_22)
    
    sigma = torch.stack([
        torch.stack([sigma_11, sigma_12], dim=-1),
        torch.stack([sigma_21, sigma_22], dim=-1)
    ], dim=-1)  # Shape [B, 2, 2]
    
    mu = torch.stack([x, y], dim=-1)  # Shape [B, 2]
    
    return mu, sigma


def matrix_sqrt(mat):
    # This computes the matrix square-root via an eigenvalue decomposition
    # NOTE: Clamp eigenvalues for numerical stability: avoids NaNs when tiny negative
    #       eigenvalues appear due to ill-conditioning / floating-point error during training
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eps = 1e-12
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=eps))
    return torch.matmul(eigvecs, torch.matmul(torch.diag(sqrt_eigvals), eigvecs.transpose(0, 1)))


def matrix_sqrt_batch(mat_batch):
    # Eigvals shape: [B, 2], Eigvecs shape: [B, 2, 2]
    eigvals, eigvecs = torch.linalg.eigh(mat_batch)

    eps = 1e-12

    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=eps)) # [B, 2]

    eig_diag = torch.diag_embed(sqrt_eigvals)  # [B, 2, 2]

    return torch.matmul(eigvecs, torch.matmul(eig_diag, eigvecs.transpose(1, 2))) # [B, 2, 2]
 

def gaussian_wasserstein_distance(pred_box, gt_box, decomposition, transformation="none"):
    x1, y1, x2, y2, sigma1, sigma2 = box2gaussian(pred_box, gt_box)

    first_term = (x1 - x2) ** 2 + (y1 - y2) ** 2
    
    # NOTE: Cholesky requires the matrix to be symmetric positive definite (PD)
    #
    # The intermediate matrix is theoretically PSD (of the form X M X^T), but can be
    # singular for degenerate boxes and can lose PD due to numerical error
    #
    # Cholesky may need jitter (eps * I) or a minimum box size to be stable during training, 
    # but eigenvalue decomposition should be stable without modification

    sigma1_sq = torch.matmul(sigma1, sigma1)
    sigma2_sq = torch.matmul(sigma2, sigma2)

    if decomposition == "eigenvalue":
        intermediate = (-2) * matrix_sqrt(torch.matmul(torch.matmul(sigma1, sigma2_sq), sigma1))
    # NOTE: For now, never tested and used
    elif decomposition == "cholesky":
        intermediate = (-2) * torch.linalg.cholesky(torch.matmul(torch.matmul(sigma1, sigma2_sq), sigma1))
    
    second_term = torch.trace(sigma1_sq + sigma2_sq + intermediate)

    gwd = first_term + second_term

    return nonlinear_transformation(gwd, transformation=transformation)


def gaussian_wasserstein_distance_batch(boxes1, boxes2, decomposition, transformation="none"):
    # NOTE: Not thouroughly tested yet
    # Compute the Wasserstein distance for a batch of boxes
    mu1, sigma1 = box2gaussian_batch(boxes1)
    mu2, sigma2 = box2gaussian_batch(boxes2)

    # This represents L2 norm
    first_term = (mu1 - mu2).pow(2).sum(dim=-1) # [B]
    
    sigma1_sq = torch.matmul(sigma1, sigma1)
    sigma2_sq = torch.matmul(sigma2, sigma2)

    if decomposition == "eigenvalue":
        intermediate = (-2) * matrix_sqrt_batch(torch.matmul(torch.matmul(sigma1, sigma2_sq), sigma1))
    elif decomposition == "cholesky":
        intermediate = (-2) * torch.linalg.cholesky(torch.matmul(torch.matmul(sigma1, sigma2_sq), sigma1))

    matrix_add = sigma1_sq + sigma2_sq + intermediate
  
    # torch.trace does not work with batched matrices. However, this accomplishes the same thing
    second_term = torch.diagonal(matrix_add, dim1=-2, dim2=-1).sum(dim=-1) # [B]

    gwd = first_term + second_term

    return nonlinear_transformation(gwd, transformation)


# NOTE: Paper uses tau=2 for "best" results
# NOTE: We opt for tau=1.65 based on empirical results for now, 
#       but this is a hyperparameter that can be tuned
def nonlinear_transformation(gwd, tau=1.65, transformation="sqrt"):
    if transformation == "none":
        iou = 1 / (tau + gwd)
        loss = 1 - iou

    elif transformation == "sqrt":
        iou = 1 / (tau + torch.sqrt(gwd))
        loss = 1 - iou

    elif transformation == "log":
        iou = 1 / (tau + torch.log(gwd + 1))
        loss = 1 - iou

    return iou, loss


def plot_bbox(box, color='cyan'):
    x, y, z, l, w, h, sin, cos = box.detach().cpu().numpy().T
    ax = plt.gca()
    x = x - w / 2  # Adjust for center
    y = y - l / 2  # Adjust for center
    print(f"angle: {np.rad2deg(np.arctan2(sin, cos))}")
    rect = patches.Rectangle((x, y), w, l, angle=np.rad2deg(np.arctan2(sin, cos)), rotation_point='center', linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def test_normal():
    # NOTE: If w=l of a box, then its rotation has no effect on the distance measure

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

    # gt box for frame 00905
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

    # gt box for frame 00911
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

    print("Box1 shape: ", box1.shape)
    print("Box2 shape: ", box2.shape)

    decomposition = "eigenvalue"

    iou1, loss1 = gaussian_wasserstein_distance(box1, box2, decomposition)
    iou2, loss2 = gaussian_wasserstein_distance(box3, box4, decomposition)
    
    print(f"GWD1: {loss1:.4f}, Approx IoU1: {iou1:.4f}")
    print(f"GWD2: {loss2:.4f}, Approx IoU2: {iou2:.4f}")

    iou = rotated_iou_2d(box1, box2)
    print(f"Rotated IoU: {iou:.4f}")

    iou = rotated_iou_2d(box3, box4)
    print(f"Rotated IoU: {iou:.4f}")


def test_batch():
    angle_box1 = 0
    angle_box2 = 45

    sin1, cos1 = np.sin(np.deg2rad(angle_box1)), np.cos(np.deg2rad(angle_box1))
    sin2, cos2 = np.sin(np.deg2rad(angle_box2)), np.cos(np.deg2rad(angle_box2))

    boxes1 = torch.tensor([
        [-9, 0, 0, 3, 1, 0, 0, 1],
        [-1, 0, 0, 4, 1, 0, 0, 1],
        [3, 0, 0, 5, 1, 0, 0.33, 0.1]
    ], dtype=torch.float32)
    boxes2 = torch.tensor([
        [-3.5, 0, 0, 2, 1, 0, 0.8, 0.8],
        [-0.75, 0.25, 0, 2, 1, 0, 0, 1],
        [2, 0.5, 0, 2, 1, 0, 0, 1]
    ], dtype=torch.float32)

    print("Original Box Shapes: ", boxes1.shape, boxes2.shape)

    decomposition = "eigenvalue"

    mu, sigma = box2gaussian_batch(boxes1)
    print(mu.shape, sigma.shape)
    print(mu)
    print(sigma)

    sqrt_mat = matrix_sqrt_batch(sigma)
    print(sqrt_mat.shape)
    print(sqrt_mat)

    gwd_batch = gaussian_wasserstein_distance_batch(boxes1, boxes2, decomposition)
    print(gwd_batch.shape)
    print(gwd_batch)

    color_pairs = [
        ("deepskyblue", "dodgerblue"),      # Both are blue, but one is lighter
        ("mediumseagreen", "forestgreen"),  # Both are green, but one is lighter
        ("orange", "darkorange"),           # Both are orange, but one is deeper
    ]
    
    plt.figure(figsize=(8, 8))
    for i in range(boxes1.shape[0]):
        plot_bbox(boxes1[i], color=color_pairs[i][0])
        plot_bbox(boxes2[i], color=color_pairs[i][1])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("batch_bbox_plot.png")

    if decomposition == "eigenvalue":
        print("Wasserstein Distance (Eig): ", gwd_batch)
        print("Without")
        print("Eig. Loss:         ", (1 - (1 / (1 + gwd_batch))).tolist())
        print("Eig. Approx. IoU:  ", (1 / (1 + gwd_batch)).tolist())
        print("Sqrt")
        print("Eig. Loss:         ", (1 - (1 / (1 + torch.sqrt(gwd_batch)))).tolist())
        print("Eig. Approx. IoU:  ", (1 / (1 + torch.sqrt(gwd_batch))).tolist())
        print("Log")
        print("Eig. Loss:         ", (1 - (1 / (1 + torch.log(gwd_batch + 1)))).tolist())
        print("Eig. Approx. IoU:  ", (1 / (1 + torch.log(gwd_batch + 1))).tolist())

    elif decomposition == "cholesky":
        print("Wasserstein Distance (Chol):", gwd_batch)
        print("Chol. Loss:        ", (1 - (1 / (1 + torch.sqrt(gwd_batch)))).tolist())
        print("Chol. Approx. IoU: ", (1 / (1 + torch.sqrt(gwd_batch))).tolist())
        
    for i in range(boxes1.shape[0]):
        iou = rotated_iou_2d(boxes1[i].numpy(), boxes2[i].numpy())
        print(f"Rotated IoU for box pair {i}: {iou}")


if __name__ == "__main__":
    test_normal()
    # test_batch()
    pass