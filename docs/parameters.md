# Key Configuration Options

## General Paramters

| Parameter | Options | Description |
| --------- | ------- | ----------- |
| `DATE_FOLDER` | `String` | Experiment folder to load for testing/evaluation |
| `APPROACH` | `'RadarOnly'`, `'RadarCamera'` | Sensor modality configuration |
| `EXPERIMENT_NAME` | `String` | Name appended to the auto-generated folder |
| `MODE` | `'train'`, `'test'`, `'evaluate'` | Operation mode |
| `SEED` | `Integer` | Random seed for reproducibility |
| `DETERMINISTIC` | `True` or `False` | Ensures deterministic operations (slight performance cost) |
| `EPOCHS` | `11` | Number of training epochs; 11 is in accordance with K-Radar |
| `SPLIT` | `'kradar'`, `'self'` | Dataset split method |
| `LABEL_VERSION` | `'v1_1'`, `'v2_0'`, `'v2_1'` | K-Radar label revision to use |
| `ROI_TYPE` | `'narrow'`, `'wide'`, `'all'` | Region of interest filter; `'narrow'` matches K-Radar benchmark |
| `TRAINING_ROI` | `True` or `False` | Apply ROI filter during training as well |
| `IS_VALIDATE` | `True` or `False` | Enable validation set during training |
| `BATCH_SIZE` | `Integer` | Batch size; reduce for GPU memory constraints |
| `RADAR_ONLY` | `True` or `False` | Use only radar-annotated ground truth |
| `CLASSES_TO_USE` | `'Sedan'`, `'Bus or Truck'`, `'Pedestrian'`, `'Motorcycle'`, `'Bicycle'`, `'Pedestrian Group'`, `'Bicycle Group'` | Classes to train and evaluate on |

## Training Paramters

| Parameter | Options | Description |
| --------- | ------- | ----------- |
| `USE_DATA_PADDING` | `True` or `False` | Required for `UNet` backbone |
| `LEARNING_RATE` | `Float` | Initial learning rate; default `0.001` |
| `WEIGHT_DECAY` | `Float` | Optimizer weight decay; default `0.01` |
| `SCHEDULER` | `'None'`, `'StepLR'`, `'MultiStepLR'`, `'LinearLR'`, `'ExponentialLR'`, `'CosineAnnealingLR'`, `'CosineAnnealingWarmRestarts'` | Learning rate scheduler |
| `GAUSSIAN_HEATMAP_SIGMA` | `'fixed'`, `'adaptive'`, `'anisotropic'` | Heatmap target generation strategy |
| `SIGMA` | `Float` | Fixed sigma value; only used when `GAUSSIAN_HEATMAP_SIGMA='fixed'` |
| `SAVE_MODELS` | `True` or `False` | Save checkpoints every n-th epoch and at the end |
| `SAVE_ALL_MODELS` | `True` or `False` | Save a checkpoint after every epoch (overrides `SAVE_MODELS`) |

## Inference Paramters

| Parameter | Options | Description |
| --------- | ------- | ----------- |
| `INF_MODEL` | `String` | Checkpoint to load, e.g. `'epoch10'` |
| `CONFIDENCE_SCORE_THRESHOLD` | `Float` | Minimum heatmap peak score to keep a detection; default `0.3` |
| `FILTER_BOXES_BY_WIDE_ROI` | `True` or `False` | Discard boxes outside the wide ROI before NMS |
| `SAVE_INFERENCE_OUTPUTS` | `True` or `False` | Persist per-frame predictions as `.pkl` files |
| `VISUALIZE_INFERENCE` | `True` or `False` | Generate visualization plots during inference |

## Evaluation Paramters

| Parameter | Options | Description |
| --------- | ------- | ----------- |
| `EVALUATION_MODE` | `'standard'`, `'allweather-allclass'` | Standard evaluates the configured filters; `'allweather-allclass'` loops over all weather conditions |
| `INCLUDE_3D` | `True` or `False` | Compute 3D mAP in addition to 2D |
| `MAP_2D_IOU_THRESHOLD` | `Float` | IoU threshold for 2D mAP matching; default `0.3` |
| `MAP_3D_IOU_THRESHOLD` | `Float` | IoU threshold for 3D mAP matching; default `0.3` |
| `SKIP_0_GT_FRAMES` | `True` or `False` | Exclude frames with no ground truth (aligns with K-Radar benchmark) |
| `CLASSES` | `'Sedan'`, `'Bus or Truck'`, `'Pedestrian'`, `'Motorcycle'`, `'Bicycle'`, `'Pedestrian Group'`, `'Bicycle Group'` | Class filter for evaluation; leave empty for all |
| `ENVIRONMENT` | `'urban'`, `'highway'`, `'alleyway'`, `'university'`, `'countryside'`, `'mountain'`, `'parkinglots'`, `'shoulder'` | Environment filter; leave empty for all |
| `TIME_OF_DAY` | `'day'`, `'night'` | Time-of-day filter; leave empty for all |
| `WEATHER_CONDITION` | `'normal'`, `'rain'`, `'overcast'`, `'sleet'`, `'fog'`, `'lightsnow'`, `'heavysnow'` | Weather filter; leave empty for all |

## Model Paramters

| Parameter | Options | Description |
| --------- | ------- | ----------- |
| `RAD_BACKBONE_CFG['type']` | `'UNet'`, `'UNetPadCrop'`, `'UNetInterp'`, `'UNetV2'`, `'ResNet'` | Radar backbone architecture |
| `RAD_NECK_CFG['type']` | `'Default'`, `'DilatedResidual'`, `'DeformConvResidual'`, `'FPN'` | Neck architecture |
| `USE_SINGULAR_HEATMAP` | `True` or `False` | Single shared heatmap for all classes vs. per-class heatmaps |
| `UNET_CFG['input_stem']` | `True` or `False` | Enable input stem in UNet |
| `UNET_CFG['cbam']` | `True` or `False` | Enable CBAM attention in UNet |
| `UNET_CFG['decoder3_out_channels']` | `Integer` | Number of output channels from the UNet decoder; must match neck |
