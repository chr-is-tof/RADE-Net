import importlib.util
import sys
import os
import time

# Command-line arguments mapping to parameter files, 
# which can be extended as needed for multiple configurations
args_dict = {
    "rade": "configs/params_rade.py",
    "radcam": "configs/params_radcam.py",
    # Add more configurations here as needed
}

# Add the parent directory of 'configs' (i.e., project_root) to sys.path
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def save_params_snapshot(params_module, target_path):
    """Dumps all UPPERCASE attributes of params_module to a python file as assignments."""
    with open(target_path, 'w') as f:
        f.write("# Auto-generated params snapshot of the actual parameters used in this run\n\n")
        for key in dir(params_module):
            if key.isupper() and not key.startswith("__"):
                val = getattr(params_module, key)
                f.write(f"{key} = {repr(val)}\n")


def main():
    if len(sys.argv) == 1:
        raise ValueError(f"No argument provided. Provide one of the following arguments: {', '.join(args_dict.keys())}")
    
    if len(sys.argv) > 4:
        raise ValueError("Too many arguments provided.")
    
    arg_params = sys.argv[1]
    arg_time = sys.argv[2] if len(sys.argv) >= 3 else None
    arg_run_dir = sys.argv[3] if len(sys.argv) >= 4 else None

    config_path = args_dict.get(arg_params)
    if config_path is None:
        raise ValueError(f"Unknown argument '{arg_params}'. Valid options are: {', '.join(args_dict.keys())}")

    print(f"Using config file: {config_path}")
    cfg = load_config(config_path)
    
    # Optionally: make a snapshot of params used in this run
    if arg_run_dir is not None:
        snapshot_path = os.path.join(arg_run_dir, 'params_snapshot.py')
        save_params_snapshot(cfg, snapshot_path)
        print(f"Saved params snapshot to {snapshot_path}")

    # Start the pipeline with loaded config
    from app import main as app_main  # Safe import
    app_main(cfg, config_path, arg_time)


if __name__ == "__main__":
    """ 
    Run the application with a selected parameter override file.
    Usage: python main.py [c|t|default]
    """

    current_time = time.time()
    print(f"Script started at {time.ctime(current_time)}")

    main()

    print(f"Total execution time: {time.time() - current_time:.2f} seconds")
    print(f"Script finished at {time.ctime(time.time())}")