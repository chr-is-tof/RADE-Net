# Official RADE-Net Repository

## Table of Contents

- [Research Papers](#research-papers)
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Context](#context)
  - [Setup Environment](#setup-environment)
- [Dataset Preparation](#dataset-preparation)
  - [Downloading K-Radar](#downloading-k-radar)
  - [Creating Projections](#creating-projections)
  - [Create Experiments Folder](#create-experiments-folder)
  - [Advice](#advice)
- [Configuration](#configuration)
  - [Setting Up Paths](#setting-up-paths)
  - [Creating New Parameter Files (Configurations)](#creating-new-parameter-files-configurations)
  - [Key Configuration Options](#key-configuration-options)
- [Usage](#usage)
  - [Training](#training)
  - [Testing / Inference](#testing--inference)
  - [Evaluation](#evaluation)
- [Distributed Training (Experimental)](#distributed-training-experimental)
- [Troubleshooting](#troubleshooting)
- [Acknowledgment](#acknowledgement)
- [License](#license)
- [Citation](#citation)

## Research Papers

- [RADE-Net: Robust Attention Network for Radar-Only Object Detection in Adverse Weather](https://arxiv.org/abs/2602.19994) accepted at [IEEE Intelligent Vehicles Symposium (IV 2026)](https://ieee-iv.org/2026/)

- [DinoRADE: Full Spectral Radar-Camera Fusion with Vision Foundation Model Features for Multi-class Object Detection in Adverse Weather](https://arxiv.org/abs/2604.08074) accepted at [CVPR 2026 DriveX - 4th Workshop on Foundation Models for V2X-based Cooperative Autonomous Driving](https://drivex-workshop.github.io/cvpr2026/)

## Overview

This repository provides radar-based or multimodal object detection models for autonomous driving scenarios. It supports:

- 🎯 Multi-class object detection (e.g., Sedan, Bus/Truck, Pedestrian, Motorcycle, Bicycle)
- 🌧️ Robust performance across weather conditions (e.g., normal, rain, fog, snow, sleet)
- 🔧 Modular architecture with configurable backbone, neck, and head components
- 📊 Comprehensive evaluation with different mAP metrics

## Repository Structure

```python
RADE-Net_private/
├── main.py                     # Entry point for all operations
├── app.py                      # Core training, testing, evaluation logic
├── app_utils.py                # Utility functions for the setup
├── configs/
│   ├── paths.py                # Dataset and output paths configuration
│   ├── params_radcam.py        # Default parameter template for DinoRADE
│   ├── params_rade.py          # Default parameter template for RADE-Net
│   ├── params_<experiment>.py  # Experiment-specific parameter file (example)
│   └── ...                     # Further specified configs
├── data/
│   ├── create_projections.py   # Process K-Radar tesseract to projections
│   ├── load_data.py            # Dataset classes and data loading
|   ├── ground_truth.py         # Ground truth processing
│   └── ...
├── models/
│   ├── model_assembly.py       # Model builder
│   ├── backbone/               # Backbone modules
│   ├── fusion/                 # Fusion modules
│   ├── neck/                   # Neck modules
│   └── head/                   # Detection head modules
├── ops/
│   ├── loss.py                 # Loss functions
│   ├── nms.py                  # Non-maximum suppression
│   ├── rotated_iou.py          # Rotated IoU computation
│   ├── gwd.py                  # Gaussian Wasserstein Distance
│   └── ...
├── utils/
│   ├── evaluation.py           # mAP evaluation
│   ├── plotting.py             # Visualization utilities
│   └── ...
├── experiments/                # Training outputs and logs
├── labels/                     # K-Radar revised labels
```

## Installation

### Prerequisites

- Python 3.11.13 (newer versions may also work)
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- Tested on Ubuntu 24.04 LTS

### Context

This project was developed and tested on Linux, so the commands and examples below assume a Linux environment. Windows and macOS may work but are not officially supported, and we may not be able to provide troubleshooting assistance on those platforms.

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-org/RADE-Net.git
cd RADE-Net

# Create and activate a virtual environment
python -m venv venv_rade
source venv_rade/bin/activate

# Install dependencies
pip install -r requirements.txt   # Optional: --no-cache-dir

# (ONLY required for DinoRADE) Install deformable attention
cd deform_attn
sh ./make.sh
```

*Note*: If the installation of Deformable Attention fails for any reason, please refer to the original implementation and installation instructions in the [official repository](https://github.com/fundamentalvision/Deformable-DETR).

## Dataset Preparation

### Downloading K-Radar

1. Download the K-Radar dataset from the [official repository](https://github.com/kaist-avelab/K-Radar)
2. The dataset should contain:
   - `radar_tesseract/` — Raw 4D radar tensors (`.mat` files)
   - `cam-front/` — Front camera images
   - Ground truth labels

3. The labels are currently included in the repository. If they are missing, download the revised labels (v2.1) and place them in `labels/revise_label/kradar_revised_label_v2_1/`

### Downloading Dinov3 (Optional for RADE-Net)

Please refer to `utils/fusion/hugface_models.py`.

### Creating Projections

The raw radar tesseract data needs to be projected to 3D tensors for efficient training for which a script is provided. **Before running**, edit the script to set your paths:

```python
# In data/create_projections.py
path_to_dataset = ...       # Example"path/to/external/hard/drive/K-Radar/"
subsets = np.arange(1, 59)  # Subsets to 58, adjust as needed
base_save_path = ...        # Example".../Dataset/K-Radar_reduced/"
```

The script will:

1. Load each `.mat` tesseract file
2. Apply log-scaling and normalization
3. Create DERA (Doppler-Elevation-Range-Azimuth) projections
4. Save as `.npy` files for fast loading

Run the script with:

```bash
python -m data.create_projections
```

### Expected Output Structure

After processing the tesseracts, also copy/include the `cam-front` and `info_calib` directories. Your output directory should then look like:

```python
K-Radar_reduced/
├── 1/
|   ├── cam-front/
│   |   ├── cam-front_00001.png
│   |   ├── cam-front_00002.png
│   |   └── ...
│   ├── DERA_tesseract/
│   |   ├── DERA_tesseract_00000.npy
│   |   ├── DERA_tesseract_00001.npy
│   |   └── ...
|   └── info_calib/
|       └── calib_radar_lidar.txt
├── 2/
|   ├── cam-front/
│   |   └── ...
│   ├── DERA_tesseract/
│   |   └── ...
|   └── info_calib/
|       └── ...
└── ...
```

Keep the naming convention used in `create_projections.py` to avoid path issues.

### Create Experiments Folder

To run everything, you also need to create an `experiments` directory (it can be located anywhere). For full functionality, place the following files in that directory:

- `train.txt` — the official K-Radar training split  
- `test.txt` — the official K-Radar test split  
- `metadata.json` — aggregated per-subset settings (available at the repository root)

### Advice

This framework is designed to work with complete subsets, and it does not support selecting individual samples within a subset. If you need to train on specific frames,
create new `train.txt` and `test.txt` files that list only the frames you want to use.

## Configuration

### Setting Up Paths

Edit `configs/paths.py` to match your system:

```python
# Path to a cluster (optional), if you have access
LOC_PATH_TO_CLUSTER = ...

# Folder that contains the processed dataset (3D radar tensors)
# For this, see 'data/create_projections.py'
LOC_PATH_TO_PROCESSED_DATASET = ...

# Folder that contains the revised labels for K-Radar dataset
LOC_PATH_TO_LABELS = ...

# Folder that stores all experiments and their results
LOC_PATH_TO_EXPERIMENTS = ...

# Folder that cotains the Dinov3 pre-trained weights (optional for radar-only approach)
LOC_PATH_TO_DINOV3 = ...
```

### Creating New Parameter Files (Configurations)

Create your own parameter file by copying the template `params.py`. Feel free to change and expand its parameters.

To register your newly created configuration, register it within the dictionary located in `main.py`. Here is an example:

```python
args_dict = {
    "rade"    : "configs/params_rade.py",
    "radcam"  : "configs/params_radcam.py",
    # ...
    # Add further configs
    # "<ID>"  : "configs/params_<config_name>.py" 
}
```

### Key Configuration Options

For more details, see [docs/parameters.md](docs/parameters.md).

## Usage

### Training

While `main.py` takes up to **four** arguments, it is enough to only provide **one** (your config identifier). Start training with your configuration:

```bash
# Use your specified ID from the 'args_dict'
python main.py <ID>       

# The script will:
# 1. Create experiment folder with timestamp
# 2. Save a snapshot of parameters used
# 3. Start training with TensorBoard logging
```

**Training outputs are saved to:**

```python
experiments/<backbone>/<timestamp>_<experiment_name>/     # Chosen <backbone>, e.g. UNet
├── models/                                               # Saved model checkpoints
├── logs/                                                 # TensorBoard logs
├── plots/                                                # Visualization outputs
├── data_split/                                           # Train/test split info
├── evaluation/                                           # Plots and csv files after mAP
└── params_snapshot.py                                    # Exact parameters used
```

**Monitor training:**

```bash
tensorboard --logdir experiments/<your_backbone>/<your_experiment>/logs
```

### Testing / Inference

Set `MODE = 'test'` in your params file and specify the model to evaluate. Also ensure `RAD_BACKBONE_CFG` matches the backbone you are using so that all paths are constructed correctly:

```python
MODE = 'test'
DATE_FOLDER = "25_12_02_15_50_your_experiment"   # Experiment folder name
INF_MODEL = 'epoch10'                            # Model checkpoint to use
SAVE_INFERENCE_OUTPUTS = True                    # Save predictions as .pkl
VISUALIZE_INFERENCE = False                      # Generate visualization plots
```

Run inference:

```bash
python main.py <ID>
```

### Evaluation

Set `MODE = 'evaluate'` to compute mAP metrics on saved inference outputs:

```python
MODE = 'evaluate'
DATE_FOLDER = "25_12_02_15_50_your_experiment"
INF_MODEL = 'epoch10'

# Evaluation settings
INCLUDE_3D = True                    # Include 3D mAP evaluation
MAP_2D_IOU_THRESHOLD = 0.3           # IoU threshold for 2D mAP
MAP_3D_IOU_THRESHOLD = 0.3           # IoU threshold for 3D mAP

# Filter by conditions (leave empty for all)
CLASSES = ['Sedan', 'Bus or Truck', 'Pedestrian', 'Motorcycle', 'Bicycle']
ENVIRONMENT = []        # e.g., ['urban', 'highway']
TIME_OF_DAY = []        # e.g., ['day', 'night']
WEATHER_CONDITION = []  # e.g., ['normal', 'rain', 'fog']
```

Run evaluation:

```bash
python main.py <ID>
```

*Additional context*: `test` mode saves its outputs as `.pkl` files and only needs to be run once (the files are stored in the corresponding experiment folder under `inference`). The subsequent `evaluate` mode reuses these files.

## Distributed Training (Experimental)

The codebase contains an initial implementation for multi-GPU training using PyTorch Distributed Data Parallel (DDP). However, this functionality has **not** been used or validated in the current version and is provided as-is.

*As additional context*: the radar-only models primarily use `GroupNorm`, which does not require cross-GPU synchronization. The radar–camera fusion models use `BatchNorm`; in DDP this may require synchronized batch normalization (e.g., `SyncBatchNorm`) or other adjustments to behave as intended.

To run with DDP:

```bash
# 2 GPUs
torchrun --nproc_per_node=2 main.py <ID>

# 4 GPUs
torchrun --nproc_per_node=4 main.py <ID>
```

When launched with `torchrun`, the training script will automatically:

- share the dataset across GPUs
- synchronize gradients
- save checkpoints only on rank 0

## Troubleshooting

**Out of Memory (OOM):**

- Reduce `BATCH_SIZE`

**Slow data loading:**

- Increase `NUM_WORKERS` (e.g., 4-8)
- Ensure data is on SSD

**Labels not found:**

- Verify `LOC_PATH_TO_LABELS` points to the correct label directory

## Acknowledgement

- We thank the authors of [K-Radar](https://github.com/kaist-avelab/K-Radar) for making the dataset publicly available and for their valuable research contributions.
- The deformable attention implementation (`deform_attn/`) is adapted from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), which itself builds on [Deformable Convolution V2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch). We thank the respective authors for making their work publicly available.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Citation

If you use our project in your research, please cite:

```bibtex
@article{leitgeb_2026_RADENet,
  title={{RADE-Net}: {Robust} {Attention} {Network} for {Radar-only} {Object} {Detection} in {Adverse} {Weather}},
  author={Christof Leitgeb, Thomas Puchleitner, Max Peter Ronecker and Daniel Watzenig},
  journal={arXiv preprint arXiv:2602.19994},
  year={2026},
}
```

and

```bibtex
@article{leitgeb_2026_dinorade,
  title={{DinoRADE}: {Full} {Spectral} {Radar}-{Camera} {Fusion} with {Vision} {Foundation} {Model} {Features} for {Multi}-class {Object} {Detection} in {Adverse} {Weather}},
  author={Christof Leitgeb, Thomas Puchleitner, Max Peter Ronecker and Daniel Watzenig},
  journal={arXiv preprint arXiv:2604.08074},
  year={2026}
}
```
