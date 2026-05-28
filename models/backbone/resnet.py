import torch
import torch.nn as nn

from models.backbone.unet_blocks import InputStem


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)

        self.act = nn.SiLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out
    

class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.list_layers = cfg['cfg']['list_layers']           # e.g. [2, 2, 2, 2]
        self.in_channels = cfg['cfg']['resnet_in_channels']    # e.g. [128, 128, 256, 512]
        self.input_stem = cfg['cfg']['input_stem']

        if self.input_stem:
            self.transform_input = InputStem(dop_in=64, dop_out=128, ele_in=37, ele_out=64, output_channels=128)
        else:
            self.start_conv = nn.Conv2d(in_channels=101, out_channels=128, kernel_size=3, stride=1, padding=1)

        def build_layers(in_channels, out_channels, num_blocks, stride):
            layers = []
            for i in range(num_blocks):
                layers.append(BasicBlock(in_channels, out_channels, stride if i == 0 else 1))
                in_channels = out_channels
            return nn.Sequential(*layers)

        self.layers1 = build_layers(                128, self.in_channels[0], self.list_layers[0], stride=1)
        self.layers2 = build_layers(self.in_channels[0], self.in_channels[1], self.list_layers[1], stride=2)
        self.layers3 = build_layers(self.in_channels[1], self.in_channels[2], self.list_layers[2], stride=2)
        self.layers4 = build_layers(self.in_channels[2], self.in_channels[3], self.list_layers[3], stride=2)

        
    def forward(self, batch_dict):
        x = batch_dict['rdr_era_dra']

        if self.input_stem:
            x = self.transform_input(x)                         # (B, 128, H, W)
        else:
            x = self.start_conv(x)                              # (B, 128, H, W)

        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)

        output_dict = {
            'resnet_output_1': x1,  # [B, 128, 256, 112]
            'resnet_output_2': x2,  # [B, 256, 128, 56]
            'resnet_output_3': x3,  # [B, 512, 64, 28]
            'resnet_output_4': x4   # [B, 512, 32, 14]
        }
        batch_dict['backbone_output'] = output_dict
    
        return batch_dict
        

if __name__ == "__main__":
    cfg = {
        'list_layers': [2, 2, 2, 2],
        'resnet_in_channels': [128, 256, 512, 512],
        'input_stem': True
    }
    model = ResNet(cfg)
    print(model)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    with torch.no_grad():
        bd = {"rdr_era_dra": torch.randn(2, 101, 256, 112)}
        out = model(bd)
        for k in ["backbone_output_1","backbone_output_2","backbone_output_3","backbone_output_4"]:
            print(k, out[k].shape)