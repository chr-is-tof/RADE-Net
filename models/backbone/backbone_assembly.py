from .unet import UNet, UNetPadCrop, UNetInterp, UNetV2
from .resnet import ResNet


def build_backbone(backbone_cfg):
    backbone = backbone_cfg['type']
    
    if backbone == 'UNet':
        return UNet(backbone_cfg)
    elif backbone == 'UNetPadCrop':
        return UNetPadCrop(backbone_cfg)
    elif backbone == 'UNetInterp':
        return UNetInterp(backbone_cfg)
    elif backbone == 'UNetV2':
        return UNetV2(backbone_cfg)
    elif backbone == 'ResNet':
        return ResNet(backbone_cfg)
    else:
        raise ValueError(f"Unknown backbone type: {backbone}")
    