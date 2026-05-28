from scipy.io import loadmat
import numpy as np
import os
import time


# Calculate statistics and confidence intervals
def calculate_stats(times, name):
    if len(times) == 0:
        return
    
    mean = np.mean(times)
    std = np.std(times, ddof=1)  # Sample standard deviation
    n = len(times)
    
    # 95% confidence interval using t-distribution
    from scipy import stats
    confidence = 0.95
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_of_error = t_value * (std / np.sqrt(n))
    
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    print(f"\n{name} Statistics (n={n}):")
    print(f"  Mean: {mean:.3f}s")
    print(f"  Std:  {std:.3f}s")
    print(f"  95% CI: [{ci_lower:.3f}s, {ci_upper:.3f}s]")
    print(f"  Min:  {np.min(times):.3f}s")
    print(f"  Max:  {np.max(times):.3f}s")


def create_projections_from_tesseract(save_path=None, subset=1, frame_start=1, frame_end=2,
                                      path_to_dataset=None, save=True, calculate_time_stats=False):
    if save_path is None:
        raise ValueError("Please provide a valid save path.")
    
    # Lists to store timing data
    load_times = []
    process_times = []
    total_times = []
    
    for frame_idx in range(frame_start, frame_end):
        frame_start_time = time.time()
        
        # Measure loading time
        load_start = time.time()
        path = f"{path_to_dataset}/{subset}/radar_tesseract/tesseract_{frame_idx:05d}.mat"

        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not on disk")
        
        tesseract = loadmat(path)['arrDREA']
        load_time = time.time() - load_start
        load_times.append(load_time)
        
        # Measure processing time
        process_start = time.time()
        
        # These values are based on the K-Radar dataset normalization
        # as per the original codebase
        tesseract = 10 * np.log10(tesseract + 1e-10)  # Avoid log(0)
        tesseract = tesseract + 1.2
        tesseract = tesseract / 184.3

        DRAE_tess = np.transpose(tesseract, (0, 1, 3, 2))  # (D, R, A, E)
        # Max over channels
        RAE_tess = np.max(DRAE_tess, axis=0)
        DRA_tess = np.max(DRAE_tess, axis=3)
        ERA_tess = np.transpose(RAE_tess, (2, 0, 1))  # Transpose to (E, R, A)

        DconERA_tess = np.concatenate((DRA_tess, ERA_tess), axis=0)  # Concatenate along channel dimension
        
        # Save 3D tensors with shape (C = D+E, R, A)
        DconERA_save_path = f"{save_path}/DERA_tesseract"
        if not os.path.exists(DconERA_save_path):
            os.makedirs(DconERA_save_path)
        
        if save:
            np.save(f"{DconERA_save_path}/DERA_tesseract_{frame_idx:05d}.npy", DconERA_tess)
        
        process_time = time.time() - process_start
        process_times.append(process_time)
        
        total_time = time.time() - frame_start_time
        total_times.append(total_time)
        
        print(f"Frame {frame_idx}/{frame_end-1} | Load: {load_time:.3f}s | Process: {process_time:.3f}s | Total: {total_time:.3f}s", end='\r')

        break  # Remove this break to process all frames in the specified range
    
    print()  # New line after loop
    
    if calculate_time_stats:
        calculate_stats(load_times, "Loading Time")
        calculate_stats(process_times, "Processing Time")
        calculate_stats(total_times, "Total Time")


if __name__ == "__main__":
    """
    This processes the K-Radar dataset tesseract files and creates
    the DERA projections as numpy arrays.

    The code works for specified subsets and frame ranges.
    """
    path_to_dataset = ...       # Example: "path/to/external/hard/drive/K-Radar/"
    subsets = np.arange(1, 59)  # Subsets 1 to 58, adjust as needed
    base_save_path = ...        # Example: ".../Dataset/K-Radar_reduced/"

    if not os.path.exists(base_save_path):
        raise ValueError(f"Save path {base_save_path} does not exist. Please create it first.")  

    for subset in subsets:
        folder = f"{path_to_dataset}/{subset}/radar_tesseract"
        
        num_files = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
        print(f"Number of files in {folder}: {num_files}")
        
        subset_save_path = f'{base_save_path}/{subset}'
        if not os.path.exists(subset_save_path):
            os.makedirs(subset_save_path)

        create_projections_from_tesseract(
            save_path=subset_save_path, 
            subset=subset, 
            frame_start=1, 
            frame_end=(num_files + 1), 
            path_to_dataset=path_to_dataset, 
            save=True,
            calculate_time_stats=True
        )