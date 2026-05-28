from configs.paths import (
    LOC_PATH_TO_CAMCALIB,
    LOC_PATH_TO_DINOV3,
    LOC_PATH_TO_EXPERIMENTS,
    LOC_PATH_TO_DATASET,
    LOC_PATH_TO_PROCESSED_DATASET,
    LOC_PATH_TO_LABELS,
    LOC_PATH_TO_CLUSTER
)

# ── ACTIVE EXPERIMENT ────────────────────────────────────────────────────────
# Change DATE_FOLDER to switch between runs.

DATE_FOLDER = ...

# =============================================================================
# GENERAL SETTINGS)
# =============================================================================

APPROACH        = 'RadarOnly'                 # 'RadarOnly', 'RadarCamera'
EXPERIMENT_NAME = ...                         # Only relevant for folder naming during training
SEED            = 42                          # Default: 42; random seed for reproducibility
DETERMINISTIC   = True                        # Default: True; ensures deterministic behavior for reproducibility (reduces performance)
VERBOSE         = True                        # If True, prints additional information
EPOCHS          = 11                          # Default: 11; in accordance with K-Radar
CURRENT_EPOCH   = 0                           # Updated during training
MODE            = 'train'                     # 'train','test','evaluate','visualize_model','visualize_pkl','paper'
CLUSTER         = True                        # # Default: False; whether to use the cluster for training/inference; if False, it will run locally (make sure to adjust paths accordingly)
SPLIT           = 'kradar'                    # Default: 'kradar'; 'self' or 'kradar'
SELECT_SUBSET   = 'all'                       # '1','2','3','4' or 'all' (only if SPLIT='kradar')
TEST_SET_SIZE   = 0.5                         # Default: 0.5; Only used if SPLIT='self'; proportion of the dataset to be used as test set (may be outdated)

# Radar Data Setting
RDR_PROCESSING_METHOD = None  # None or "upsample_azimuth" (107->214)

# ROI
ROI_TYPE     = 'narrow'             # Default: 'narrow'; Options: 'narrow', 'wide' or 'all'; 'narrow' in accordance with K-Radar
TRAINING_ROI = False                # Default: False; If True, use ROI filtering during training as well

# Validation
IS_VALIDATE = False                 # Default: False; whether to run validation during training; needs a validation set

# Data Loader
BATCH_SIZE  = 10                    # Default: 10; batch size for training and inference; adjust according to GPU memory; <10 for 'upsample_azimuth' method
NUM_WORKERS = 0                     # Default: 0; number of workers for data loading; adjust according to your system
SHUFFLE     = True                  # Default: True; whether to shuffle the dataset during training

# Ground Truth
LABEL_VERSION        = "v1_1"       # 'v1_1', 'v2_0', 'v2_1'
FOV                  = [-53, 53]    # Default: [-53, 53]; Field of View in degrees; filters ground truth
RADAR_ONLY           = True         # Whether to use only radar ground truths for training
CLASSES_TO_USE       = ['Sedan', 'Bus or Truck', 'Pedestrian', 'Motorcycle', 'Bicycle']  # 'Sedan', 'Bus or Truck', 'Pedestrian', 'Motorcycle', 'Bicycle'. 'Pedestrian Group', 'Bicycle Group'
NUM_CLASSES          = len(CLASSES_TO_USE)
FLIP_GT_HORIZONTALLY = True         # Default: True; flip ground truth horizontally
CALIB_GT             = True         # Default: True; calibrate ground truth
LABEL_IDX_DICT = {cls: idx for idx, cls in enumerate(CLASSES_TO_USE)}   # Model output channel i corresponds to CLASSES_TO_USE[i]
IDX_LABEL_DICT = {v: k for k, v in LABEL_IDX_DICT.items()}              # Inverse mapping of LABEL_IDX_DICT


# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Save Settings
SAVE_MODELS     = True   # Default: True; Saves every n-th model and the last model (specified in training loop in app.py)
SAVE_ALL_MODELS = True   # Default: True; Saves all models (overrides SAVE_MODELS)
VISUALIZE_TRAIN = False  # Default: False; visualizes training samples (while training) (may be outdated)

# Dataset Settings
USE_DATA_PADDING = True                 # Whether to use data padding (necessary for UNet)
DATA             = 'processed'          # 'tesseract' or 'processed' ('tesseract' incompatible with UNet)
DATA_NAME        = 'DERA_tesseract'     # Default: 'DERA_tesseract'; name according to folder structure
DATA_EXTENSION   = '.npy'               # Default: '.npy'; file extension of the dataset; likely does not need to be changed unless the dataset format changed
SUBSET_MODE      = 'all'                # 'range','choice','type','all' (only for SPLIT='self')
SUBSET_RANGE     = [1, 2]               # from-to (exclusive); only for SUBSET_MODE='range'
SUBSET_CHOICE    = list(range(1, 59))   # specific subsets;    only for SUBSET_MODE='choice'
SUBSET_TYPE      = ['day', 'night']     #                      only for SUBSET_MODE='type'

# Optimizer
LEARNING_RATE = 0.001                   # Default: 0.001
WEIGHT_DECAY  = 0.01                    # Default: 0.01

# Scheduler ('StepLR'/'MultiStepLR' update per epoch; others update per iteration)
SCHEDULER              = 'CosineAnnealingLR'
# Options: 'None','StepLR','MultiStepLR','LinearLR','ExponentialLR','CosineAnnealingLR','CosineAnnealingWarmRestarts'

STEPLR_STEP_SIZE       = 1
STEPLR_GAMMA           = 0.657933224657568      # gamma s.t. lr * gamma^EPOCHS ≈ 1e-5 (for lr=0.001, EPOCHS=11)

MULTISTEPLR_MILESTONES = [3, 6, 9]
MULTISTEPLR_GAMMA      = 0.5

LINLR_MIN_LR           = 0.00001

EXPLR_GAMMA            = 0.9997602510610437     # gamma s.t. lr * gamma^(EPOCHS*iters_per_epoch) ≈ 1e-5 (for EPOCHS=11, batch_size=10)

CALR_CYCLIC            = True                   # If True, restart cosine schedule every 2nd epoch (multiple cycles)
CALR_MIN_LR            = 0.00001

CAWRLR_T0              = 2                      # Epochs before first restart in CosineAnnealingWarmRestarts
CAWRLR_MIN_LR          = 0.00001

GAUSSIAN_HEATMAP_SIGMA = 'fixed'                # 'fixed', 'adaptive', 'anisotropic'
SIGMA                  = 3                      # Only used if GAUSSIAN_HEATMAP_SIGMA='fixed'


# =============================================================================
# INFERENCE PARAMETERS
# =============================================================================

CONFIDENCE_SCORE_THRESHOLD = 0.3        # Default: 0.3
FILTER_BOXES_BY_WIDE_ROI   = True       # Default: True; applies a ROI filter for faster inference

INF_LOAD_FROM_CLUSTER      = False      
INF_MODEL                  = 'epoch10'  # Default: 'epoch10'; model checkpoint to evaluate
VISUALIZE_INFERENCE        = False
SAVE_INFERENCE_OUTPUTS     = True

IS_PARTIAL_TEST            = False      # Default: False; likely outdated
IS_SAMPLE_TEST             = False      # Default: False; likely outdated
PART_NUMBER                = 1          # 1, 2, 3, or 4; likely outdated; only used if IS_PARTIAL_TEST is True


# =============================================================================
# EVALUATION
# =============================================================================

EVALUATION_MODE          = 'standard'   # 'standard' or 'allweather-allclass'
GET_PROBLEMATIC_SAMPLES  = False        # Default: False
EVALUATE_MAP             = True
INCLUDE_3D               = True
MAP_2D_IOU_THRESHOLD     = 0.3
MAP_3D_IOU_THRESHOLD     = 0.3
SKIP_0_GT_FRAMES         = True        # Default: True; skip frames with no GT (aligns with other papers)

# Subset filters — leave empty to include all
# CLASSES options:           'Sedan','Bus or Truck','Pedestrian','Motorcycle','Bicycle','Pedestrian Group','Bicycle Group'
CLASSES           = ['Sedan', 'Bus or Truck', 'Pedestrian', 'Motorcycle', 'Bicycle']
# ENVIRONMENT options:       'urban','highway','alleyway','university','countryside','mountain','parkinglots','shoulder'
ENVIRONMENT       = []
# TIME_OF_DAY options:       'day','night'
TIME_OF_DAY       = []
# WEATHER_CONDITION options: 'normal','rain','overcast','sleet','fog','lightsnow','heavysnow'
WEATHER_CONDITION = []

EVALUATE_SINGLE_SUBSET = False  # Likely outdated
EVALUATE_SUBSET        = '1'    # '1', '2', '3', '4'; likely outdated


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

USE_SINGULAR_HEATMAP = False                    # Uses a single heatmap for all classes instead of per-class heatmaps
IMAGE_BACKBONE       = 'Dinov3'
RADCAM_FUSION_TYPE   = 'def_att_project_fuse'   # 'concat','deformable_attention','bev_lift_fuse','def_att_project_fuse'
CAMCALIB_PATH        = LOC_PATH_TO_CAMCALIB

DEFORM_ATTN_CFG = {
    'd_model': 128,         # Should match radar and camera feature channels
    'n_heads': 4,           # e.g., 4
    'n_points': 8           # e.g., 8
}
RADAR_PARAMETERS = {
    'range_min': 0,
    'range_max': 118,
    'azimuth_min': -53,
    'azimuth_max': 53,
    'elevation_min': -18,
    'elevation_max': 18
}
RAD2CAM_CALIB = {
    'dx': -1.90 - 0.18,     # Distance between radar and camera in x direction + compensation from ldr2img_t translation part
    'dy': -0.13,            # Compensation from ldr2img_t translation part
    'dz': 0.7 + 0.2         # Height difference between radar and lidar + height difference between lidar and stereo camera
}
CAM_CONFIG = {
    'cam_neck': True,
    'rad_neck': True,
    'cam_neck_output': '256x112', # '256x112' or '224x224' # must match fusion method (concat='256x112', deformable_attention='224x224')
}
BEV_LIFTFUSE_CFG = {
    'depth_levels': 4,
}
DINOV3_CONFIG = {
    'model_name': LOC_PATH_TO_DINOV3,
    'use_adapter': True,
    'adapter_type': 'LoRA', # 'LoRA'
    'lora_config': {
        'r': 16,
        'lora_alpha': 16,
        'target_modules': ["q_proj", "v_proj"],
        'lora_dropout': 0.1,
        'bias': "none",
        'modules_to_save': ["classifier"],
    }
}
UNET_CFG = {
    "mode": MODE,
    "dropout": 0,                   # Default: 0
    "input_stem": False,
    "cbam": True,
    # Describes number of features coming out of the UNet
    "decoder3_out_channels": 128,   # Default: 128; be careful with neck!
}
UNETV2_CFG = {
    "mode": MODE,
    "dropout": 0,                   # Default: 0
    "input_stem": True
}
RESNET_CFG = {
    "input_stem": True,
    "list_layers": [2, 2, 2, 2],                # Number of blocks in each layer; default: [2, 2, 2, 2]
    "resnet_in_channels": [128, 256, 512, 512]  # Number of channels coming out of each layer; default: [128, 256, 512, 512]
}
RAD_BACKBONE_CFG = {
    "type": "UNet",                 # Options: "UNet", "UNetPadCrop", "UNetInterp", "UNetV2", "ResNet"
    "cfg": UNET_CFG
}
RAD_NECK_CFG = {
    "type": "DilatedResidual",                              # Options: 'Default', 'DilatedResidual', 'DeformConvResidual', 'DeformAttn', 'FPN'
    "dilation": (1, 2, 3),                                  # Also in DeformConvResidual
    "modulation": True,                                     # Only used for DeformConvResidual
    "neck_channels": 128,                                   # Default: 128
    "resnet_in_channels": RESNET_CFG['resnet_in_channels']  # Only used if backbone is ResNet and neck is FPN
}
RAD_HEAD_CFG = {
    "single_heatmap": USE_SINGULAR_HEATMAP,
    "multi_channel_type": "combined",   # Options: 'combined', 'heat_split', 'reg_split'
    "heatmap_type": "2DCNN_expanded",   # Options: '2DCNN_default', '2DCNN_expanded', '2DCNN_expanded_residual'
    "reg_type": "2DCNN_expanded",       # Options: '2DCNN_default', '2DCNN_expanded', '2DCNN_expanded_residual', '2DCNN_split_expanded'
    "cls_type": "2DCNN_expanded",       # Options: '2DCNN_expanded', '2DCNN_expanded_residual'
    "in_channels": 128,                 # Default: 128
    "hidden_channels": 128,             # Default: 128
    "num_classes": NUM_CLASSES
}
MODEL_CFG = {
    "data_padding": USE_DATA_PADDING,
    "rad_backbone": RAD_BACKBONE_CFG,
    "rad_neck": RAD_NECK_CFG,
    "rad_head": RAD_HEAD_CFG,
    "image_backbone": IMAGE_BACKBONE,           # Options: 'Dinov3'
    "image_backbone_cfg": DINOV3_CONFIG,        # Use respective config dict for chosen image backbone
    "image_neck": {                             # Current hardcoded values for Dinov3
        "cam_config": CAM_CONFIG,
        "input_dim" : 384,
        "output_dim" : 128 },   
    "radcam_fusion_type": RADCAM_FUSION_TYPE,
    "deform_attn_cfg": DEFORM_ATTN_CFG,
    "radar_parameters": RADAR_PARAMETERS,
    "rad2cam_calib": RAD2CAM_CALIB,
    "cam_config": CAM_CONFIG,
    "bev_liftfuse_cfg": BEV_LIFTFUSE_CFG,
    "camcalib_path": CAMCALIB_PATH,
}


# =============================================================================
# PATHS
# =============================================================================

PATH_TO_DATASET              = LOC_PATH_TO_DATASET
PATH_TO_PROCESSED_DATASET    = LOC_PATH_TO_PROCESSED_DATASET
PATH_TO_LABELS               = LOC_PATH_TO_LABELS
INF_PATH_TO_CLUSTER          = LOC_PATH_TO_CLUSTER
PATH_TO_EXPERIMENTS          = LOC_PATH_TO_EXPERIMENTS

PATH_TO_FOLDERS              = f"{PATH_TO_EXPERIMENTS}/{MODEL_CFG['rad_backbone']['type']}/{DATE_FOLDER}"
PATH_TO_TEST_SET             = f"{PATH_TO_FOLDERS}/data_split"
PATH_TO_PLOTS_TRAIN          = f"{PATH_TO_FOLDERS}/plots/train"
PATH_TO_PLOTS_INFERENCE      = f"{PATH_TO_FOLDERS}/plots/inference"
PATH_TO_SAVED_MODELS         = f"{PATH_TO_FOLDERS}/models"
PATH_TO_LOGS                 = f"{PATH_TO_FOLDERS}/logs"
PATH_TO_SAVED_INFERENCE_DATA = f"{PATH_TO_FOLDERS}/inference"