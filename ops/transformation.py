import numpy as np
import torch


def transform_RA_map_to_XY_map(RA_map):
    RA_cloud = np.zeros([RA_map.shape[0] * RA_map.shape[1], 3])
    idx = 0
    for rng_idx in range(RA_map.shape[0]):
        for azi_idx in range(RA_map.shape[1]):
            rng = rng_idx * 118/255

            # If the width is padded, scale accordingly
            if RA_map.shape[1] == 112:
                azi = azi_idx * 106/111 - 53
            else:
                azi = azi_idx - 53
            
            RA_cloud[idx, :] = [rng, azi, RA_map[rng_idx, azi_idx]]
            idx += 1

    XY_map = np.zeros((RA_cloud.shape[0], 3))
    for i in range(RA_cloud.shape[0]):
        rng, azi, val = RA_cloud[i]
        x = rng * np.cos(np.deg2rad(azi))
        y = rng * np.sin(np.deg2rad(azi))
        XY_map[i] = [x, y, val]

    return XY_map


def transform_ra_indices_to_cartesian(azimuth_idx, range_idx, use_padding=False, upsampled=False):
    # Convert indices to meters/degrees
    range_meter = range_idx * 118/255

    if upsampled:
        azimuth_deg = azimuth_idx * 106/213 - 53
    elif use_padding:
        azimuth_deg = azimuth_idx * 106/111 - 53
    else:
        azimuth_deg = azimuth_idx - 53

    # Convert to rad for trig. functions
    azimuth_rad = torch.deg2rad(azimuth_deg)

    # Compute cartesian coordinates
    x_longitudinal = range_meter * torch.cos(azimuth_rad)

    # Need to flip due to a different coordinate system with the ground truth
    y_lateral = range_meter * torch.sin(azimuth_rad) * (-1)

    return x_longitudinal, y_lateral


if __name__ == "__main__":
    pass