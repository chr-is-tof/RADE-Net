from .neck_nets import (
    DefaultNeck,
    DilatedResidualNeck,
    DeformConvResidualNeck,
    DeformableNeck
)
from .fpn import FPN


def build_neck(neck_cfg):
    neck = neck_cfg["type"]
    dilation = neck_cfg["dilation"]
    modulation = neck_cfg["modulation"]
    channels = neck_cfg["neck_channels"]

    if neck == "Default":
        return DefaultNeck(in_channels=channels)
    elif neck == "DilatedResidual":
        return DilatedResidualNeck(in_channels=channels, dilation=dilation)
    elif neck == "DeformConvResidual":
        return DeformConvResidualNeck(in_channels=channels, dilation=dilation, groups=32, modulation=modulation)
    elif neck == "DeformAttn":
        return DeformableNeck(in_channels=channels, num_heads=4, embed_dim=256, num_levels=1, num_points=8)
    elif neck == "FPN":
        return FPN(neck_cfg, out_c=channels, mode="nearest")
    else:
        raise ValueError(f"Unknown neck type: {neck}")

