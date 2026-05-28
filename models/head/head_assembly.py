from .coupled_heads import (
    MultiChannelHeatmap,
    MultiChannelHeatmapSplit,
    MultiChannelHeatmapRegSplit,
    SingleChannelHeatmap
)


def build_head(head_cfg):
    if head_cfg['single_heatmap']:
        return SingleChannelHeatmap(head_cfg)
    elif head_cfg['multi_channel_type'] == 'combined':
        return MultiChannelHeatmap(head_cfg)
    elif head_cfg['multi_channel_type'] == 'heat_split':
        return MultiChannelHeatmapSplit(head_cfg)
    elif head_cfg['multi_channel_type'] == 'reg_split':
        return MultiChannelHeatmapRegSplit(head_cfg)
    else:
        raise ValueError(f"Unknown head configuration: {head_cfg}")