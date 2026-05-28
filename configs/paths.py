# Path to a cluster (optional), if you have access.
LOC_PATH_TO_CLUSTER = ...

# Folder that contains the raw K-Radar dataset (4D tesseracts) [optional; implementation artifact]
LOC_PATH_TO_DATASET = ...

# Folder that contains the processed dataset (3D radar tensors)
# For this, see 'data/create_projections.py'
LOC_PATH_TO_PROCESSED_DATASET = ...

# Folder that contains the revised labels for K-Radar dataset (specify the version you want to use)
# LOC_PATH_TO_LABELS = '<path_to_your_labels>/revise_label/kradar_revised_label_v2_1/KRadar_revised_visibility'
# LOC_PATH_TO_LABELS = '<path_to_your_labels>/revise_label/kradar_revised_label_v2_0/KRadar_refined_label_by_UWIPL/'
LOC_PATH_TO_LABELS = '<path_to_your_labels>/labels/revise_label/kradar_revised_label_v1_1/'

# Folder that stores all experiments and their results
LOC_PATH_TO_EXPERIMENTS = '<path_to_your_desired_experiment_location>/experiments'

# Folder that contains the calibration files
LOC_PATH_TO_CAMCALIB = '<path_to_your_cam_calib>/calib/cam_calib/common'

# Folder that cotains the Dinov3 pre-trained weights (optional for radar-only approach)
# To download, see 'utils/fusion/hugface_model.py'
LOC_PATH_TO_DINOV3 = '<path_to_dinov3>/vision_models/dinov3_local'