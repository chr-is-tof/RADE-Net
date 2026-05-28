import torch
import torch.nn as nn

from .head_nets import (
    DefaultCenterHead,
    ExpandedCenterHead,
    ExpandedResidualCenterHead,
    DefaultCenterHead3D,
    DefaultRegHead,
    ExpandedRegHead,
    ExpandedResidualRegHead,
    SplitExpandedRegHead,
    DefaultRegHead3D,
    ExpandedClassificationHead,
    ExpandedResidualClassificationHead
)


class MultiChannelHeatmap(nn.Module):
    def __init__(self, head_cfg):
        super().__init__()

        self.heatmap_type = head_cfg['heatmap_type']
        self.reg_type = head_cfg['reg_type']

        self.in_channels = head_cfg['in_channels']
        self.hidden_channels = head_cfg['hidden_channels']
        self.num_classes = head_cfg['num_classes']

        # ----- Heatmap Head -----

        if self.heatmap_type == '2DCNN_default':
            self.heatmap_head = DefaultCenterHead(self.in_channels, self.num_classes)
        elif self.heatmap_type == '2DCNN_expanded':
            self.heatmap_head = ExpandedCenterHead(self.in_channels, self.hidden_channels, self.num_classes)
        elif self.heatmap_type == '2DCNN_expanded_residual':
            self.heatmap_head = ExpandedResidualCenterHead(self.in_channels, self.hidden_channels, self.num_classes)
        elif self.heatmap_type == '3DCNN':
            self.heatmap_head = DefaultCenterHead3D()
        else:
            raise ValueError(f"Unknown heatmap head type: {self.heatmap_type}")
        
        # ----- Regression Head -----

        if self.reg_type == '2DCNN_default':
            self.reg_head = DefaultRegHead(self.in_channels, out_channels=8)
        elif self.reg_type == '2DCNN_expanded':
            self.reg_head = ExpandedRegHead(self.in_channels, self.hidden_channels, out_channels=8)
        elif self.reg_type == '2DCNN_expanded_residual':
            self.reg_head = ExpandedResidualRegHead(self.in_channels, self.hidden_channels, out_channels=8)
        elif self.reg_type == '3DCNN':
            self.reg_head = DefaultRegHead3D()
        else:
            raise ValueError(f"Unknown regression head type: {self.reg_type}")
        
    def forward(self, batch_dict):
        feat = batch_dict['neck_output']

        if self.heatmap_type.startswith('2DCNN'):
            heatmap = self.heatmap_head(feat)
        elif self.heatmap_type == '3DCNN':
            feat_3d = feat.unsqueeze(1)             # Add depth dimension
            heatmap = self.heatmap_head(feat_3d)
            heatmap = heatmap.squeeze(1)            # Remove depth dimension

        heatmap = torch.sigmoid(heatmap)            # [B, num_classes, H, W]

        if self.reg_type.startswith('2DCNN'):
            reg = self.reg_head(feat)               # [B, 8, H, W]
        elif self.reg_type == '3DCNN':
            feat_3d = feat.unsqueeze(1)             # Add depth dimension
            reg = self.reg_head(feat_3d)
            reg = reg.squeeze(1)                    # Remove depth dimension
        
        batch_dict['heatmap'] = heatmap
        batch_dict['regression'] = reg
        
        return batch_dict
    

class MultiChannelHeatmapRegSplit(nn.Module):
    def __init__(self, head_cfg):
        super().__init__()

        self.heatmap_type = head_cfg['heatmap_type']
        self.reg_type = head_cfg['reg_type']

        self.in_channels = head_cfg['in_channels']
        self.hidden_channels = head_cfg['hidden_channels']
        self.num_classes = head_cfg['num_classes']

        # ----- Heatmap Head -----
        if self.heatmap_type == '2DCNN_default':
            self.heatmap_head = DefaultCenterHead(self.in_channels, self.num_classes)
        elif self.heatmap_type == '2DCNN_expanded':
            self.heatmap_head = ExpandedCenterHead(self.in_channels, self.hidden_channels, num_classes=self.num_classes)
        elif self.heatmap_type == '2DCNN_expanded_residual':
            self.heatmap_head = ExpandedResidualCenterHead(self.in_channels, self.hidden_channels, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown heatmap head type: {self.heatmap_type}")
        
        # ----- Regression Head -----
        if self.reg_type == '2DCNN_default':
            self.reg_head = DefaultRegHead(self.in_channels, out_channels=8)
        elif self.reg_type == '2DCNN_expanded':
            self.reg_head = ExpandedRegHead(self.in_channels, self.hidden_channels, out_channels=8)
        elif self.reg_type == '2DCNN_expanded_residual':
            self.reg_head = ExpandedResidualRegHead(self.in_channels, self.hidden_channels, out_channels=8)
        elif self.reg_type == '2DCNN_split_expanded':
            self.reg_head = SplitExpandedRegHead(self.in_channels, self.hidden_channels)
        else:
            raise ValueError(f"Unknown regression head type: {self.reg_type}")


    def forward(self, batch_dict):
        feat = batch_dict['neck_output']

        heatmap = self.heatmap_head(feat)                       # [B, num_classes, H, W]
        heatmap = torch.sigmoid(heatmap)                       # [B, num_classes, H, W]

        reg = self.reg_head(feat)                                           # [B, 8, H, W]

        batch_dict['heatmap'] = heatmap
        batch_dict['regression'] = reg

        return batch_dict
    

class MultiChannelHeatmapSplit(nn.Module):
    def __init__(self, head_cfg):
        super().__init__()

        self.heatmap_type = head_cfg['heatmap_type']
        self.reg_type = head_cfg['reg_type']

        self.in_channels = head_cfg['in_channels']
        self.hidden_channels = head_cfg['hidden_channels']
        self.num_classes = head_cfg['num_classes']

        self.large_object_classes = 2   # 'sedan' and 'bus or truck'
        self.small_object_classes = self.num_classes - self.large_object_classes

        # ----- Heatmap Head 1 -----

        if self.heatmap_type == '2DCNN_expanded':
            self.heatmap_head_large = ExpandedCenterHead(self.in_channels, self.hidden_channels, num_classes=self.large_object_classes)
        elif self.heatmap_type == '2DCNN_expanded_residual':
            self.heatmap_head_large = ExpandedResidualCenterHead(self.in_channels, self.hidden_channels, num_classes=self.large_object_classes)
        else:
            raise ValueError(f"Unknown heatmap head type: {self.heatmap_type}")
        
        # ----- Heatmap Head 2 -----

        if self.heatmap_type == '2DCNN_expanded':
            self.heatmap_head_small = ExpandedCenterHead(self.in_channels, self.hidden_channels, num_classes=self.small_object_classes)
        elif self.heatmap_type == '2DCNN_expanded_residual':
            self.heatmap_head_small = ExpandedResidualCenterHead(self.in_channels, self.hidden_channels, num_classes=self.small_object_classes)
        else:
            raise ValueError(f"Unknown heatmap head type: {self.heatmap_type}")

        # ----- Regression Head -----

        if self.reg_type == '2DCNN_expanded':
            self.reg_head = ExpandedRegHead(self.in_channels, self.hidden_channels, out_channels=8)
        elif self.reg_type == '2DCNN_expanded_residual':
            self.reg_head = ExpandedResidualRegHead(self.in_channels, self.hidden_channels, out_channels=8)
        else:
            raise ValueError(f"Unknown regression head type: {self.reg_type}")


    def forward(self, batch_dict):
        feat = batch_dict['neck_output']

        heatmap_large = self.heatmap_head_large(feat)                       # [B, large_object_classes, H, W]
        heatmap_small = self.heatmap_head_small(feat)                       # [B, small_object_classes, H, W]
        heatmaps_cat = torch.cat([heatmap_large, heatmap_small], dim=1)     # [B, num_classes, H, W]
        heatmap = torch.sigmoid(heatmaps_cat)                               # [B, num_classes, H, W]

        reg = self.reg_head(feat)                                           # [B, 8, H, W]

        batch_dict['heatmap'] = heatmap
        batch_dict['regression'] = reg

        return batch_dict


class SingleChannelHeatmap(nn.Module):
    def __init__(self, head_cfg):
        super().__init__()

        self.heatmap_head_type = head_cfg['heatmap_type']
        self.reg_head_type = head_cfg['reg_type']
        self.cls_head_type = head_cfg['cls_type']

        self.in_channels = head_cfg['in_channels']
        self.hidden_channels = head_cfg['hidden_channels']
        self.num_classes = head_cfg['num_classes']

        # ----- Heatmap Head -----

        if self.heatmap_head_type == '2DCNN_expanded':
            self.heatmap_head = ExpandedCenterHead(self.in_channels, self.hidden_channels, num_classes=1)
        elif self.heatmap_head_type == '2DCNN_expanded_residual':
            self.heatmap_head = ExpandedResidualCenterHead(self.in_channels, self.hidden_channels, num_classes=1)
        else:
            raise ValueError(f"Unknown heatmap head type: {self.heatmap_head_type}")

        # ----- Regression Head -----

        if self.reg_head_type == '2DCNN_expanded':
            self.reg_head = ExpandedRegHead(self.in_channels, self.hidden_channels, out_channels=8)
        elif self.reg_head_type == '2DCNN_expanded_residual':
            self.reg_head = ExpandedResidualRegHead(self.in_channels, self.hidden_channels, out_channels=8)
        else:
            raise ValueError(f"Unknown regression head type: {self.reg_head_type}")

        # ----- Classification Head -----

        if self.cls_head_type == '2DCNN_expanded':
            self.classification_head = ExpandedClassificationHead(self.in_channels, self.hidden_channels, self.num_classes)
        elif self.cls_head_type == '2DCNN_expanded_residual':
            self.classification_head = ExpandedResidualClassificationHead(self.in_channels, self.hidden_channels, self.num_classes)
        else:
            raise ValueError(f"Unknown classification head type: {self.cls_head_type}")

    def forward(self, batch_dict):
        feat = batch_dict['neck_output']

        heatmap = self.heatmap_head(feat)
        heatmap = torch.sigmoid(heatmap)  # [B, 1, H, W]

        reg = self.reg_head(feat)  # [B, 8, H, W]

        classification = self.classification_head(feat)  # [B, num_classes, H, W]

        batch_dict['heatmap'] = heatmap
        batch_dict['regression'] = reg
        batch_dict['classification'] = classification
        
        return batch_dict