def _section(title):
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def _row(label, value, width=35):
    print(f"  {label:<{width}} {value}")


def _print_model_section(cfg):
    """Print backbone, camera, neck and head config. Shared by train and test modes."""
    backbone = cfg.MODEL_CFG['rad_backbone']['type']
    neck     = cfg.MODEL_CFG['rad_neck']['type']
    head     = cfg.MODEL_CFG['rad_head']

    _section("Model — Backbone")
    _row("Type:", backbone)
    if backbone in ('UNet', 'UNetPadCrop', 'UNetInterp'):
        _row("  Dropout:", cfg.UNET_CFG['dropout'])
        _row("  CBAM:", cfg.UNET_CFG['cbam'])
        _row("  Input Stem:", cfg.UNET_CFG['input_stem'])
        _row("  Decoder3 Out Ch.:", cfg.UNET_CFG['decoder3_out_channels'])
    elif backbone == 'UNetV2':
        _row("  Dropout:", cfg.UNETV2_CFG['dropout'])
        _row("  Input Stem:", cfg.UNETV2_CFG['input_stem'])
    elif backbone == 'ResNet':
        _row("  Layers:", cfg.RESNET_CFG['list_layers'])
        _row("  In Channels:", cfg.RESNET_CFG['resnet_in_channels'])

    if cfg.APPROACH == 'RadarCamera':
        _section("Model — Camera")
        _row("Image Backbone:", cfg.IMAGE_BACKBONE)
        _row("Fusion Type:", cfg.RADCAM_FUSION_TYPE)
        if cfg.IMAGE_BACKBONE == 'Dinov3':
            adapter = cfg.DINOV3_CONFIG['adapter_type'] if cfg.DINOV3_CONFIG['use_adapter'] else 'None'
            _row("  Adapter:", adapter)

    _section("Model — Neck")
    _row("Type:", neck)
    if neck == 'DilatedResidual':
        _row("  Dilation:", cfg.RAD_NECK_CFG['dilation'])
        _row("  Neck Channels:", cfg.RAD_NECK_CFG['neck_channels'])
    elif neck == 'DeformConvResidual':
        _row("  Dilation:", cfg.RAD_NECK_CFG['dilation'])
        _row("  Modulation:", cfg.RAD_NECK_CFG['modulation'])
        _row("  Neck Channels:", cfg.RAD_NECK_CFG['neck_channels'])

    _section("Model — Head")
    _row("Single Heatmap:", head['single_heatmap'])
    _row("Heatmap Type:", head['heatmap_type'])
    _row("Reg Type:", head['reg_type'])
    _row("Cls Type:", head['cls_type'])
    _row("In Ch.:", head['in_channels'])
    _row("Hidden Ch.:", head['hidden_channels'])
    _row("Num Classes:", head['num_classes'])


def print_config_settings(cfg):
    mode = cfg.MODE

    print(f"\n{'═' * 60}")
    print(f"  CONFIG SUMMARY  ─  Mode: {mode.upper()}")
    print(f"{'═' * 60}")

    # ── General (always shown) ────────────────────────────────
    _section("General")
    _row("Date Folder:", cfg.DATE_FOLDER)
    _row("Approach:", cfg.APPROACH)
    _row("Experiment Name:", cfg.EXPERIMENT_NAME)
    _row("Seed:", cfg.SEED)
    _row("Deterministic:", cfg.DETERMINISTIC)
    _row("Verbose:", cfg.VERBOSE)
    _row("Cluster:", cfg.CLUSTER)
    _row("Label Version:", cfg.LABEL_VERSION)
    _row("ROI Type:", cfg.ROI_TYPE)
    _row("Training ROI:", cfg.TRAINING_ROI)
    _row("FOV:", cfg.FOV)
    _row("Classes:", cfg.CLASSES_TO_USE)
    _row("Num Classes:", cfg.NUM_CLASSES)
    _row("Radar Only GT:", cfg.RADAR_ONLY)
    _row("RDR Processing:", cfg.RDR_PROCESSING_METHOD or 'None')

    # ── Mode-specific ─────────────────────────────────────────
    if mode == 'train':
        print(f"\n{'═' * 60}")
        print(f"  TRAINING CONFIGURATION")
        print(f"{'═' * 60}")

        _section("Data")
        _row("Split:", cfg.SPLIT)
        _row("Select Subset:", cfg.SELECT_SUBSET)
        _row("Data:", cfg.DATA)
        _row("Data Name:", cfg.DATA_NAME)
        _row("Data Extension:", cfg.DATA_EXTENSION)
        _row("Subset Mode:", cfg.SUBSET_MODE)
        _row("Calibrate GT:", cfg.CALIB_GT)
        _row("Flip GT Horiz.:", cfg.FLIP_GT_HORIZONTALLY)
        _row("Use Data Padding:", cfg.USE_DATA_PADDING)
        _row("Batch Size:", cfg.BATCH_SIZE)
        _row("Num Workers:", cfg.NUM_WORKERS)
        _row("Shuffle:", cfg.SHUFFLE)
        _row("Validation Set:", cfg.IS_VALIDATE)

        _section("Optimizer & Scheduler")
        _row("Learning Rate:", cfg.LEARNING_RATE)
        _row("Weight Decay:", cfg.WEIGHT_DECAY)
        _row("Epochs:", cfg.EPOCHS)
        _row("Scheduler:", cfg.SCHEDULER)
        if cfg.SCHEDULER == 'StepLR':
            _row("  Step Size:", cfg.STEPLR_STEP_SIZE)
            _row("  Gamma:", cfg.STEPLR_GAMMA)
        elif cfg.SCHEDULER == 'MultiStepLR':
            _row("  Milestones:", cfg.MULTISTEPLR_MILESTONES)
            _row("  Gamma:", cfg.MULTISTEPLR_GAMMA)
        elif cfg.SCHEDULER == 'LinearLR':
            _row("  Min LR:", cfg.LINLR_MIN_LR)
        elif cfg.SCHEDULER == 'ExponentialLR':
            _row("  Gamma:", cfg.EXPLR_GAMMA)
        elif cfg.SCHEDULER == 'CosineAnnealingLR':
            _row("  Min LR:", cfg.CALR_MIN_LR)
            _row("  Cyclic:", cfg.CALR_CYCLIC)
        elif cfg.SCHEDULER == 'CosineAnnealingWarmRestarts':
            _row("  T0:", cfg.CAWRLR_T0)
            _row("  Min LR:", cfg.CAWRLR_MIN_LR)

        _section("Loss")
        _row("Gaussian Heatmap Mode:", cfg.GAUSSIAN_HEATMAP_SIGMA)
        if cfg.GAUSSIAN_HEATMAP_SIGMA == 'fixed':
            _row("  Sigma:", cfg.SIGMA)

        _print_model_section(cfg)

        _section("Save Settings")
        _row("Save Models:", cfg.SAVE_MODELS)
        _row("Save All Models:", cfg.SAVE_ALL_MODELS)
        _row("Visualize Train:", cfg.VISUALIZE_TRAIN)
        _row("Experiment Path:", cfg.PATH_TO_FOLDERS)

    elif mode == 'test':
        print(f"\n{'═' * 60}")
        print(f"  INFERENCE CONFIGURATION")
        print(f"{'═' * 60}")

        _section("Inference Settings")
        _row("Model:", cfg.INF_MODEL)
        _row("Confidence Threshold:", cfg.CONFIDENCE_SCORE_THRESHOLD)
        _row("Filter by Wide ROI:", cfg.FILTER_BOXES_BY_WIDE_ROI)
        _row("Load from Cluster:", cfg.INF_LOAD_FROM_CLUSTER)
        _row("Is Partial Test:", cfg.IS_PARTIAL_TEST)
        if cfg.IS_PARTIAL_TEST:
            _row("  Part Number:", cfg.PART_NUMBER)
        _row("Is Sample Test:", cfg.IS_SAMPLE_TEST)
        _row("Visualize Inference:", cfg.VISUALIZE_INFERENCE)
        _row("Save Outputs:", cfg.SAVE_INFERENCE_OUTPUTS)
        _row("Experiment Path:", cfg.PATH_TO_FOLDERS)

        _print_model_section(cfg)

    elif mode == 'evaluate':
        print(f"\n{'═' * 60}")
        print(f"  EVALUATION CONFIGURATION")
        print(f"{'═' * 60}")

        _section("Evaluation Settings")
        _row("Folder:", cfg.DATE_FOLDER)
        _row("Model:", cfg.INF_MODEL)
        _row("Evaluation Mode:", cfg.EVALUATION_MODE)
        _row("Evaluate mAP:", cfg.EVALUATE_MAP)
        _row("Includes 3D:", cfg.INCLUDE_3D)
        _row("Skip 0-GT Frames:", cfg.SKIP_0_GT_FRAMES)
        _row("mAP 2D IoU Threshold:", cfg.MAP_2D_IOU_THRESHOLD)
        _row("mAP 3D IoU Threshold:", cfg.MAP_3D_IOU_THRESHOLD)
        _row("Classes:", cfg.CLASSES if cfg.CLASSES else 'All')
        _row("Environment:", cfg.ENVIRONMENT if cfg.ENVIRONMENT else 'All')
        _row("Time of Day:", cfg.TIME_OF_DAY if cfg.TIME_OF_DAY else 'All')
        _row("Weather:", cfg.WEATHER_CONDITION if cfg.WEATHER_CONDITION else 'All')
        _row("Get Problematic:", cfg.GET_PROBLEMATIC_SAMPLES)
        _row("Single Subset Eval:", cfg.EVALUATE_SINGLE_SUBSET)
        if cfg.EVALUATE_SINGLE_SUBSET:
            _row("  Subset:", cfg.EVALUATE_SUBSET)

    print(f"\n{'═' * 60}\n")
