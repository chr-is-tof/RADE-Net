import numpy as np


def upsample_azimuth(
    rdr_data,           # [C, range, azimuth] - [101, 256, 107]
    target_az_bins=214  # Default: double the azimith resolution from 107 to 214 bins
):
    """
    A function that upsamples the azimuth dimension to double the size
    """
    C, r_bins, a_bins = rdr_data.shape
    assert a_bins == 107, f"Expected 107 azimuth bins, got {a_bins}"
    
    # Create an empty array for the upsampled data
    upsampled_rdr_data = np.zeros((C, r_bins, target_az_bins), dtype=np.float32)

    # Duplicate data for each original azimuth bin without interpolation
    for i in range(a_bins):
        upsampled_rdr_data[:, :, 2 * i] = rdr_data[:, :, i]
        upsampled_rdr_data[:, :, 2 * i + 1] = rdr_data[:, :, i]

    return upsampled_rdr_data


def process_radar(rdr_data, method="upsample_azimuth"):
    """
    An extensible wrapper for radar preprocessing steps
    """

    if method == "upsample_azimuth":
        rdr_data = upsample_azimuth(rdr_data)
    else:
        raise ValueError(f"Unknown radar processing method '{method}'. Valid options are 'upsample_azimuth'.")

    return rdr_data