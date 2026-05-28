import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, cfg, out_c=128, mode="nearest"):
        super().__init__()
        
        self.channels = cfg['resnet_in_channels']  # e.g. [128, 256, 512, 512]
        self.mode = mode

        # Lateral projections
        self.lat1 = nn.Conv2d(self.channels[0], out_c, kernel_size=1)
        self.lat2 = nn.Conv2d(self.channels[1], out_c, kernel_size=1)
        self.lat3 = nn.Conv2d(self.channels[2], out_c, kernel_size=1)
        self.lat4 = nn.Conv2d(self.channels[3], out_c, kernel_size=1)

        # Smoothing after each fusion
        self.smooth3 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

    def forward(self, resnet_outputs):
        x1 = resnet_outputs['resnet_output_1']
        x2 = resnet_outputs['resnet_output_2']
        x3 = resnet_outputs['resnet_output_3']
        x4 = resnet_outputs['resnet_output_4']

        p4 = self.lat4(x4)  # [B, C, 32, 14]

        p3 = self.lat3(x3) + F.interpolate(p4, size=x3.shape[-2:], mode=self.mode)  # [B, C, 64, 28]
        p3 = self.smooth3(p3)

        p2 = self.lat2(x2) + F.interpolate(p3, size=x2.shape[-2:], mode=self.mode)  # [B, C, 128, 56]
        p2 = self.smooth2(p2)

        p1 = self.lat1(x1) + F.interpolate(p2, size=x1.shape[-2:], mode=self.mode)  # [B, C, 256, 112]
        p1 = self.smooth1(p1) 

        return p1 # [B, out_c, 256, 112]
    

if __name__ == "__main__":
    # from models.backbone.resnet import ResNet
    # import torch

    # resnet_cfg = {
    #     'list_layers': [2, 2, 2, 2],
    #     'resnet_in_channels': [128, 256, 512, 512],
    #     'input_stem': True
    # }
    # backbone = ResNet(resnet_cfg)
    # neck = FPN(resnet_cfg)

    # dummy_input = torch.randn(2, 101, 256, 112)
    # batch_dict = {'rdr_era_dra': dummy_input}
    # batch_dict = backbone(batch_dict)
    # batch_dict = neck(batch_dict)
    # print(batch_dict['neck_output'].shape)  # Should be [2, out_c, 256, 112]

    from models.model_assembly import build_model
    class CFG:
        MODE = "train"
        APPROACH = "RadarOnly"
        RESNET_CFG = {
            'list_layers': [2, 2, 2, 2],
            'resnet_in_channels': [128, 256, 512, 512],
            'input_stem': True
        }
        RAD_BACKBONE_CFG = {
            "type": "ResNet",
            "cfg": RESNET_CFG
        }
        RAD_NECK_CFG = {
            "type": "FPN",
            "resnet_in_channels": RESNET_CFG['resnet_in_channels'],
            "neck_channels": 128
        }
        RAD_HEAD_CFG = {
            "single_heatmap": False,
            "multi_channel_type": "combined",   # Options: 'combined', 'split'
            "heatmap_type": "2DCNN_expanded",   # Options: '2DCNN_default', '2DCNN_expanded', '2DCNN_expanded_residual', '3DCNN'
            "reg_type": "2DCNN_expanded",       # Options: '2DCNN_default', '2DCNN_expanded', '2DCNN_expanded_residual', '3DCNN'
            "cls_type": "2DCNN_expanded",       # Options: '2DCNN_expanded', '2DCNN_expanded_residual'
            "in_channels": 128,                 # Default: 128
            "hidden_channels": 128,             # Default: 128
            "num_classes": 5
        }
        MODEL_CFG = {
            "data_padding": True,
            "rad_backbone": RAD_BACKBONE_CFG,
            "rad_neck": RAD_NECK_CFG,
            "rad_head": RAD_HEAD_CFG,
        }
    cfg = CFG()
    model = build_model(cfg.MODE, cfg.APPROACH, None, None, None, cfg.MODEL_CFG)
    print(model)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))