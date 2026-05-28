import numpy as np
import os
from tqdm import tqdm

from data.load_data import read_pcd


def get_all_file_paths():
    """Get all radar and lidar file paths from the training set.
    
    Returns:
        radar_file_paths: List of radar file paths
        lidar_file_paths: List of corresponding lidar file paths
    """
    path_to_dataset = ...
    path_to_train_set = ...

    with open(path_to_train_set, 'r') as f:
        train_files = [line.strip() for line in f.readlines()]

    dic = {}
    for file in train_files:
        subset, frames = file.split(',')
        if subset not in dic:
            dic[subset] = {"radar": [], "lidar": []}
        rdr_frame, ldr_frame = frames.split('_')
        frame = ldr_frame.split('.')[0]
        dic[subset]["radar"].append(rdr_frame)
        dic[subset]["lidar"].append(frame)
    
    radar_file_paths = []
    lidar_file_paths = []
    
    for subset in range(1, 59):
        radar_subset_path = f"{path_to_dataset}/{subset}/DERA_tesseract"
        lidar_subset_path = f"{path_to_dataset}/{subset}/os2-64"
        
        # Process pairs together to maintain correspondence
        for rdr_frame, ldr_frame in zip(dic[str(subset)]["radar"], dic[str(subset)]["lidar"]):
            radar_file_path = os.path.join(radar_subset_path, f"DERA_tesseract_{rdr_frame}.npy")
            lidar_file_path = os.path.join(lidar_subset_path, f"os2-64_{ldr_frame}.pcd")
            radar_file_paths.append(radar_file_path)
            lidar_file_paths.append(lidar_file_path)
        
    return radar_file_paths, lidar_file_paths


def load_radar_frame(file_path):
    rdr_data = np.load(file_path)
    return rdr_data


def load_lidar_frame(file_path):
    ldr_col_read = ["x", "y", "z", "intensity", "reflectivity", "ambient"]
    lidar_data = read_pcd(file_path, ldr_col_read)
    return lidar_data


def analyze_radar_frame(rdr_data):
    # [C, R, A]
    doppler_ra = rdr_data[:64, :, :]
    elevation_ra = rdr_data[64:, :, :]
    
    assert doppler_ra.shape[0] == 64, "Expected 64 Doppler channels"
    assert elevation_ra.shape[0] == 37, "Expected 37 elevation channels"

    # Find min and max values across all Doppler channels
    doppler_min_value = np.min(doppler_ra)
    doppler_max_value = np.max(doppler_ra)

    # Find min and max values across all elevation channels
    elevation_min_value = np.min(elevation_ra)
    elevation_max_value = np.max(elevation_ra)

    return doppler_min_value, doppler_max_value, elevation_min_value, elevation_max_value


def analyze_lidar_frame(ldr_data):
    z = ldr_data[:, 2]
    inten = ldr_data[:, 3]
    refl = ldr_data[:, 4]
    amb = ldr_data[:, 5]

    z_min, z_max = np.min(z), np.max(z)
    inten_min, inten_max = np.min(inten), np.max(inten)
    refl_min, refl_max = np.min(refl), np.max(refl)
    amb_min, amb_max = np.min(amb), np.max(amb)

    return z_min, z_max, inten_min, inten_max, refl_min, refl_max, amb_min, amb_max


def aggregate_frames(radar_file_paths, lidar_file_paths):
    """Aggregate statistics from radar and lidar frames.
    
    Args:
        radar_file_paths: List of radar file paths
        lidar_file_paths: List of corresponding lidar file paths
        
    Returns:
        Tuple of 12 arrays: (doppler_min, doppler_max, elevation_min, elevation_max,
                            z_min, z_max, inten_min, inten_max, 
                            refl_min, refl_max, amb_min, amb_max)
    """
    all_doppler_min_values = []
    all_elevation_min_values = []
    all_doppler_max_values = []
    all_elevation_max_values = []
    all_z_min_values = []
    all_z_max_values = []
    all_inten_min_values = []
    all_inten_max_values = []
    all_refl_min_values = []
    all_refl_max_values = []
    all_amb_min_values = []
    all_amb_max_values = []

    for rdr_path, ldr_path in tqdm(zip(radar_file_paths, lidar_file_paths), 
                                     desc="Processing Frames", 
                                     total=len(radar_file_paths)):
        rdr_data = load_radar_frame(rdr_path)
        ldr_data = load_lidar_frame(ldr_path)
        
        doppler_min, doppler_max, elevation_min, elevation_max = analyze_radar_frame(rdr_data)
        z_min, z_max, inten_min, inten_max, refl_min, refl_max, amb_min, amb_max = analyze_lidar_frame(ldr_data)
        
        all_doppler_min_values.append(doppler_min)
        all_elevation_min_values.append(elevation_min)
        all_doppler_max_values.append(doppler_max)
        all_elevation_max_values.append(elevation_max)
        all_z_min_values.append(z_min)
        all_z_max_values.append(z_max)
        all_inten_min_values.append(inten_min)
        all_inten_max_values.append(inten_max)
        all_refl_min_values.append(refl_min)
        all_refl_max_values.append(refl_max)
        all_amb_min_values.append(amb_min)
        all_amb_max_values.append(amb_max)

    return (all_doppler_min_values, all_doppler_max_values, all_elevation_min_values, all_elevation_max_values,
            all_z_min_values, all_z_max_values, all_inten_min_values, all_inten_max_values,
            all_refl_min_values, all_refl_max_values, all_amb_min_values, all_amb_max_values)


def compute_statistics(min_values, max_values):
    overall_min = np.min(min_values)
    overall_max = np.max(max_values)

    mean_min = np.mean(min_values)
    mean_max = np.mean(max_values)

    median_min = np.median(min_values)
    median_max = np.median(max_values)
    
    quantile_01_min = np.percentile(min_values, 0.1)
    quantile_1_min = np.percentile(min_values, 1)
    quantile_5_min = np.percentile(min_values, 5)
    quantile_95_min = np.percentile(min_values, 95)
    quantile_99_min = np.percentile(min_values, 99)
    quantile_999_min = np.percentile(min_values, 99.9)

    quantile_01_max = np.percentile(max_values, 0.1)
    quantile_1_max = np.percentile(max_values, 1)
    quantile_5_max = np.percentile(max_values, 5)
    quantile_95_max = np.percentile(max_values, 95)
    quantile_99_max = np.percentile(max_values, 99)
    quantile_999_max = np.percentile(max_values, 99.9)

    print(f"Overall Min Value:           {overall_min}")
    print(f"Overall Max Value:           {overall_max}")
    print(f"Mean Min Value:              {mean_min}")
    print(f"Mean Max Value:              {mean_max}")
    print(f"Median Min Value:            {median_min}")
    print(f"Median Max Value:            {median_max}")
    print(f"0.1st Percentile Min Value:  {quantile_01_min}")
    print(f"1st Percentile Min Value:    {quantile_1_min}")
    print(f"5th Percentile Min Value:    {quantile_5_min}")
    print(f"95th Percentile Min Value:   {quantile_95_min}")
    print(f"99th Percentile Min Value:   {quantile_99_min}")
    print(f"99.9th Percentile Min Value: {quantile_999_min}")
    print(f"0.1st Percentile Max Value:  {quantile_01_max}")
    print(f"1st Percentile Max Value:    {quantile_1_max}")
    print(f"5th Percentile Max Value:    {quantile_5_max}")
    print(f"95th Percentile Max Value:   {quantile_95_max}")
    print(f"99th Percentile Max Value:   {quantile_99_max}")
    print(f"99.9th Percentile Max Value: {quantile_999_max}")


def save_statistics(radar_file_paths, lidar_file_paths, 
                   doppler_min, doppler_max, elevation_min, elevation_max,
                   z_min, z_max, inten_min, inten_max, 
                   refl_min, refl_max, amb_min, amb_max,
                   save_path="radar_statistics.npz"):
    """Save min/max values per frame to avoid reloading data.
    
    Args:
        radar_file_paths: List of radar file paths
        lidar_file_paths: List of lidar file paths
        doppler_min/max: Doppler channel statistics
        elevation_min/max: Elevation channel statistics
        z_min/max: Z coordinate statistics
        inten_min/max: Intensity statistics
        refl_min/max: Reflectivity statistics
        amb_min/max: Ambient statistics
        save_path: Path where to save the statistics
    """
    np.savez_compressed(
        save_path,
        radar_file_paths=np.array(radar_file_paths, dtype=object),
        lidar_file_paths=np.array(lidar_file_paths, dtype=object),
        doppler_min=np.array(doppler_min),
        doppler_max=np.array(doppler_max),
        elevation_min=np.array(elevation_min),
        elevation_max=np.array(elevation_max),
        z_min=np.array(z_min),
        z_max=np.array(z_max),
        inten_min=np.array(inten_min),
        inten_max=np.array(inten_max),
        refl_min=np.array(refl_min),
        refl_max=np.array(refl_max),
        amb_min=np.array(amb_min),
        amb_max=np.array(amb_max)
    )
    print(f"Saved statistics for {len(radar_file_paths)} frames to {save_path}")


def load_statistics(save_path="radar_statistics.npz"):
    """Load previously computed min/max statistics.
    
    Args:
        save_path: Path to the saved statistics file
        
    Returns:
        Tuple of (radar_paths, lidar_paths, doppler_min, doppler_max, elevation_min, elevation_max,
                 z_min, z_max, inten_min, inten_max, refl_min, refl_max, amb_min, amb_max)
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Statistics file not found: {save_path}")
    
    data = np.load(save_path, allow_pickle=True)
    radar_file_paths = data['radar_file_paths'].tolist()
    lidar_file_paths = data['lidar_file_paths'].tolist()
    doppler_min = data['doppler_min']
    doppler_max = data['doppler_max']
    elevation_min = data['elevation_min']
    elevation_max = data['elevation_max']
    z_min = data['z_min']
    z_max = data['z_max']
    inten_min = data['inten_min']
    inten_max = data['inten_max']
    refl_min = data['refl_min']
    refl_max = data['refl_max']
    amb_min = data['amb_min']
    amb_max = data['amb_max']
    
    print(f"Loaded statistics for {len(radar_file_paths)} frames from {save_path}")
    return (radar_file_paths, lidar_file_paths, doppler_min, doppler_max, elevation_min, elevation_max,
            z_min, z_max, inten_min, inten_max, refl_min, refl_max, amb_min, amb_max)


if __name__ == "__main__":
    save_path = ...
    
    # Check if statistics already exist
    if os.path.exists(save_path):
        print(f"Loading existing statistics from {save_path}...")
        (radar_file_paths, lidar_file_paths, doppler_min, doppler_max, elevation_min, elevation_max,
         z_min, z_max, inten_min, inten_max, refl_min, refl_max, amb_min, amb_max) = load_statistics(save_path)
    else:
        print("Computing statistics from scratch...")
        radar_file_paths, lidar_file_paths = get_all_file_paths()
        print(f"{len(radar_file_paths)} radar frames found for analysis.")
        print(f"{len(lidar_file_paths)} lidar frames found for analysis.")
        (doppler_min, doppler_max, elevation_min, elevation_max,
         z_min, z_max, inten_min, inten_max, refl_min, refl_max, amb_min, amb_max) = aggregate_frames(radar_file_paths, lidar_file_paths)
        save_statistics(radar_file_paths, lidar_file_paths, doppler_min, doppler_max, elevation_min, elevation_max,
                       z_min, z_max, inten_min, inten_max, refl_min, refl_max, amb_min, amb_max, save_path)
    
    # Compute and display statistics for Doppler channels
    print("\n=== Doppler Channels Statistics ===")
    compute_statistics(doppler_min, doppler_max)

    # Compute and display statistics for Elevation channels
    print("\n=== Elevation Channels Statistics ===")
    compute_statistics(elevation_min, elevation_max)
    
    # Compute and display statistics for Lidar Z coordinate
    print("\n=== Lidar Z Coordinate Statistics ===")
    compute_statistics(z_min, z_max)
    
    # Compute and display statistics for Lidar Intensity
    print("\n=== Lidar Intensity Statistics ===")
    compute_statistics(inten_min, inten_max)
    
    # Compute and display statistics for Lidar Reflectivity
    print("\n=== Lidar Reflectivity Statistics ===")
    compute_statistics(refl_min, refl_max)
    
    # Compute and display statistics for Lidar Ambient
    print("\n=== Lidar Ambient Statistics ===")
    compute_statistics(amb_min, amb_max)

