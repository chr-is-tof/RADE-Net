import torch
import torch.nn as nn

from .unet_blocks import (
    InputStem,
    DoubleConvResidual,
    Downsample,
    Bottleneck,
    Upsample,
    ConvolutionalBlockAttention,
    DownsampleBlurConv,
    UpsampleConv
)


class UNet(nn.Module):
    def __init__(self, backbone_cfg):
        super().__init__()

        # Read out the configuration and define stuff here
        # -----
        self.backbone_cfg = backbone_cfg['cfg']
        self.decoder3_out_channels = backbone_cfg['cfg']['decoder3_out_channels']
        self.input_stem = backbone_cfg['cfg']['input_stem']
        self.cbam = backbone_cfg['cfg']['cbam']

        if backbone_cfg['cfg']['mode'] == 'train':
            dropout = backbone_cfg['cfg']['dropout']
        else:
            dropout = 0.0
        # -----
        
        if self.input_stem:
            self.transform_input = InputStem(dop_in=64, dop_out=128, ele_in=37, ele_out=64, output_channels=128)
        else:
            self.start_conv = nn.Conv2d(in_channels=101, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Starting with the first encoder block
        self.encoder_block_1 = DoubleConvResidual(128, 128, dropout=dropout)
        self.downsample_1 = Downsample(kernel_size=2, padding=0)

        # Second encoder block
        self.encoder_block_2 = DoubleConvResidual(128, 256, dropout=dropout)
        self.downsample_2 = Downsample(kernel_size=2, padding=0)

        # Third encoder block
        self.encoder_block_3 = DoubleConvResidual(256, 512, dropout=dropout)
        self.downsample_3 = Downsample(kernel_size=2, padding=0)

        # Bottleneck
        self.bottleneck = Bottleneck(512, 512, dropout=dropout)

        # First decoder block
        self.upsample_1 = Upsample(512, 512, kernel_size=2, padding=0)
        self.cbam_1 = ConvolutionalBlockAttention(512, 512, dropout=dropout)
        self.decoder_block_1 = DoubleConvResidual(1024, 256, dropout=dropout)

        # Second decoder block
        self.upsample_2 = Upsample(256, 256, kernel_size=2, padding=0)
        self.cbam_2 = ConvolutionalBlockAttention(256, 256, dropout=dropout)
        self.decoder_block_2 = DoubleConvResidual(512, 128, dropout=dropout)

        # Third decoder block
        self.upsample_3 = Upsample(128, 128, kernel_size=2, padding=0)
        self.cbam_3 = ConvolutionalBlockAttention(128, 128, dropout=dropout)
        self.decoder_block_3 = DoubleConvResidual(256, self.decoder3_out_channels, dropout=dropout)

    def forward(self, batch_dict):
        x = batch_dict['rdr_era_dra']

        if self.input_stem:
            x = self.transform_input(x)                         # (B, 128, H, W)
        else:
            x = self.start_conv(x)                              # (B, 128, H, W)
        
        # if cfg.USE_DATA_PADDING:
        #     # Pad the input to have width 112 instead of 107
        #     x = torch.nn.functional.pad(x, (0, 5, 0, 0))

        # Encoder
        encoder_1 = self.encoder_block_1(x)                     # (B, 128, H, W)
        down_1 = self.downsample_1(encoder_1)                   # (B, 128, H/2, W/2)
        encoder_2 = self.encoder_block_2(down_1)                # (B, 256, H/2, W/2)
        down_2 = self.downsample_2(encoder_2)                   # (B, 256, H/4, W/4)
        encoder_3 = self.encoder_block_3(down_2)                # (B, 512, H/4, W/4)
        down_3 = self.downsample_3(encoder_3)                   # (B, 512, H/8, W/8)

        # Bottleneck
        bottleneck = self.bottleneck(down_3)                    # (B, 512, H/8, W/8)

        # Decoder
        upsample_1 = self.upsample_1(bottleneck)                # (B, 256, H/4, W/4)

        if self.cbam:
            cbam_1 = self.cbam_1(encoder_3)                     # Apply CBAM to the bottom encoder block
        else:
            cbam_1 = encoder_3

        concat_1 = torch.cat([upsample_1, cbam_1], dim=1)       # (B, 512 + 512 = 1024, H/4, W/4)
        decoder_1 = self.decoder_block_1(concat_1)              # (B, 256, H/4, W/4)

        upsample_2 = self.upsample_2(decoder_1)                 # (B, 128, H/2, W/2)

        if self.cbam:
            cbam_2 = self.cbam_2(encoder_2)                     # Apply CBAM to the middle encoder block
        else:
            cbam_2 = encoder_2

        concat_2 = torch.cat([upsample_2, cbam_2], dim=1)       # (B, 256 + 256 = 512, H/2, W/2)
        decoder_2 = self.decoder_block_2(concat_2)              # (B, 128, H/2, W/2)

        upsample_3 = self.upsample_3(decoder_2)                 # (B, 64, H, W)

        if self.cbam:
            cbam_3 = self.cbam_3(encoder_1)                     # Apply CBAM to the top encoder block
        else:
            cbam_3 = encoder_1

        concat_3 = torch.cat([upsample_3, cbam_3], dim=1)       # (B, 128 + 128 = 256, H, W)
        decoder_3 = self.decoder_block_3(concat_3)              # (B, 128, H, W)

        # Add to batch dict
        batch_dict['backbone_output'] = decoder_3

        return batch_dict
    

class UNetPadCrop(nn.Module):
    def __init__(self, backbone_cfg):
        super().__init__()

        # Read out the configuration and define stuff here
        # -----
        self.backbone_cfg = backbone_cfg['cfg']
        self.decoder3_out_channels = backbone_cfg['cfg']['decoder3_out_channels']
        self.input_stem = backbone_cfg['cfg']['input_stem']
        self.cbam = backbone_cfg['cfg']['cbam']

        if backbone_cfg['cfg']['mode'] == 'train':
            dropout = backbone_cfg['cfg']['dropout']
        else:
            dropout = 0.0
        # -----
        
        if self.input_stem:
            self.transform_input = InputStem(dop_in=64, dop_out=128, ele_in=37, ele_out=64, output_channels=128)
        else:
            self.start_conv = nn.Conv2d(in_channels=101, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Starting with the first encoder block
        self.encoder_block_1 = DoubleConvResidual(128, 128, dropout=dropout)
        self.downsample_1 = Downsample(kernel_size=2, padding=0)

        # Second encoder block
        self.encoder_block_2 = DoubleConvResidual(128, 256, dropout=dropout)
        self.downsample_2 = Downsample(kernel_size=2, padding=0)

        # Third encoder block
        self.encoder_block_3 = DoubleConvResidual(256, 512, dropout=dropout)
        self.downsample_3 = Downsample(kernel_size=2, padding=0)

        # Bottleneck
        self.bottleneck = Bottleneck(512, 512, dropout=dropout)

        # First decoder block
        self.upsample_1 = Upsample(512, 512, kernel_size=2, padding=0)
        self.cbam_1 = ConvolutionalBlockAttention(512, 512, dropout=dropout)
        self.decoder_block_1 = DoubleConvResidual(1024, 256, dropout=dropout)

        # Second decoder block
        self.upsample_2 = Upsample(256, 256, kernel_size=2, padding=0)
        self.cbam_2 = ConvolutionalBlockAttention(256, 256, dropout=dropout)
        self.decoder_block_2 = DoubleConvResidual(512, 128, dropout=dropout)

        # Third decoder block
        self.upsample_3 = Upsample(128, 128, kernel_size=2, padding=0)
        self.cbam_3 = ConvolutionalBlockAttention(128, 128, dropout=dropout)
        self.decoder_block_3 = DoubleConvResidual(256, self.decoder3_out_channels, dropout=dropout)

    def pad_for_downsample(self, x):
        return torch.nn.functional.pad(x, (0, 1, 0, 0))             # Pad width by 1 to the right

    def match_width(self, x, target_tensor):
        _, _, _, W = target_tensor.shape
        return x[:, :, :, :W]

    def forward(self, batch_dict):
        x = batch_dict['rdr_era_dra']

        if self.input_stem:
            x = self.transform_input(x)                             # [B, 128, 256, 107]
        else:
            x = self.start_conv(x)                                  # [B, 128, 256, 107]
        
        # Encoder
        encoder_1_raw = self.encoder_block_1(x)                     # [B, 128, 256, 107]
        encoder_1 = self.pad_for_downsample(encoder_1_raw)          # Pad width to 108
        down_1 = self.downsample_1(encoder_1)                       # [B, 128, 128, 54]

        encoder_2 = self.encoder_block_2(down_1)                    # [B, 256, 128, 54]
        down_2 = self.downsample_2(encoder_2)                       # [B, 256, 64, 27]
        
        encoder_3_raw = self.encoder_block_3(down_2)                # [B, 512, 64, 27]
        encoder_3 = self.pad_for_downsample(encoder_3_raw)          # Pad width to 28
        down_3 = self.downsample_3(encoder_3)                       # [B, 512, 32, 14]

        # Bottleneck
        bottleneck = self.bottleneck(down_3)                        # [B, 512, 32, 14]

        # Decoder
        upsample_1 = self.upsample_1(bottleneck)                    # [B, 512, 64, 28]
        upsample_1 = self.match_width(upsample_1, encoder_3_raw)    # Match width to 27

        if self.cbam:
            cbam_1 = self.cbam_1(encoder_3_raw)                     # Apply CBAM to the bottom encoder block
        else:
            cbam_1 = encoder_3_raw

        concat_1 = torch.cat([upsample_1, cbam_1], dim=1)           # [B, 512 + 512 = 1024, 64, 27]
        decoder_1 = self.decoder_block_1(concat_1)                  # [B, 256, 64, 27]

        upsample_2 = self.upsample_2(decoder_1)                     # [B, 256, 128, 54]

        if self.cbam:
            cbam_2 = self.cbam_2(encoder_2)                         # Apply CBAM to the middle encoder block
        else:
            cbam_2 = encoder_2

        concat_2 = torch.cat([upsample_2, cbam_2], dim=1)           # [B, 256 + 256 = 512, 128, 54]
        decoder_2 = self.decoder_block_2(concat_2)                  # [B, 128, 128, 54]

        upsample_3 = self.upsample_3(decoder_2)                     # [B, 128, 256, 108]
        upsample_3 = self.match_width(upsample_3, encoder_1_raw)    # Match width to 107

        if self.cbam:
            cbam_3 = self.cbam_3(encoder_1_raw)                     # Apply CBAM to the top encoder block
        else:
            cbam_3 = encoder_1_raw

        concat_3 = torch.cat([upsample_3, cbam_3], dim=1)           # [B, 128 + 128 = 256, 256, 107]
        decoder_3 = self.decoder_block_3(concat_3)                  # [B, C, 256, 107]; C is defined by 'decoder3_out_channels'

        # Add to batch dict
        batch_dict['backbone_output'] = decoder_3

        return batch_dict
    

class UNetInterp(nn.Module):
    def __init__(self, backbone_cfg):
        super().__init__()

        # Read out the configuration and define stuff here
        # -----
        self.backbone_cfg = backbone_cfg['cfg']
        self.decoder3_out_channels = backbone_cfg['cfg']['decoder3_out_channels']
        self.input_stem = backbone_cfg['cfg']['input_stem']
        self.cbam = backbone_cfg['cfg']['cbam']

        if backbone_cfg['cfg']['mode'] == 'train':
            dropout = backbone_cfg['cfg']['dropout']
        else:
            dropout = 0.0
        # -----
        
        if self.input_stem:
            self.transform_input = InputStem(dop_in=64, dop_out=128, ele_in=37, ele_out=64, output_channels=128)
        else:
            self.start_conv = nn.Conv2d(in_channels=101, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Starting with the first encoder block
        self.encoder_block_1 = DoubleConvResidual(128, 128, dropout=dropout)
        self.downsample_1 = Downsample(kernel_size=2, padding=0)

        # Second encoder block
        self.encoder_block_2 = DoubleConvResidual(128, 256, dropout=dropout)
        self.downsample_2 = Downsample(kernel_size=2, padding=0)

        # Third encoder block
        self.encoder_block_3 = DoubleConvResidual(256, 512, dropout=dropout)
        self.downsample_3 = Downsample(kernel_size=2, padding=0)

        # Bottleneck
        self.bottleneck = Bottleneck(512, 512, dropout=dropout)

        # First decoder block
        self.upsample_1 = UpsampleConv(512, 512, padding=1)
        self.cbam_1 = ConvolutionalBlockAttention(512, 512, dropout=dropout)
        self.decoder_block_1 = DoubleConvResidual(1024, 256, dropout=dropout)

        # Second decoder block
        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SiLU(inplace=True)
        )
        self.cbam_2 = ConvolutionalBlockAttention(256, 256, dropout=dropout)
        self.decoder_block_2 = DoubleConvResidual(512, 128, dropout=dropout)

        # Third decoder block
        self.upsample_3 = UpsampleConv(128, 128, padding=1)
        self.cbam_3 = ConvolutionalBlockAttention(128, 128, dropout=dropout)
        self.decoder_block_3 = DoubleConvResidual(256, self.decoder3_out_channels, dropout=dropout)

    def interpolate_for_downsample(self, x):
        _, _, H, W = x.shape
        return torch.nn.functional.interpolate(x, size=(H, W + 1), mode='bilinear', align_corners=False)
        
    def forward(self, batch_dict):
        x = batch_dict['rdr_era_dra']

        if self.input_stem:
            x = self.transform_input(x)                             # [B, 128, 256, 107]
        else:
            x = self.start_conv(x)                                  # [B, 128, 256, 107]
        
        # Encoder
        encoder_1_raw = self.encoder_block_1(x)                     # [B, 128, 256, 107]
        encoder_1 = self.interpolate_for_downsample(encoder_1_raw)  # Interpolate width to 108
        down_1 = self.downsample_1(encoder_1)                       # [B, 128, 128, 54]

        encoder_2 = self.encoder_block_2(down_1)                    # [B, 256, 128, 54]
        down_2 = self.downsample_2(encoder_2)                       # [B, 256, 64, 27]
        
        encoder_3_raw = self.encoder_block_3(down_2)                # [B, 512, 64, 27]
        encoder_3 = self.interpolate_for_downsample(encoder_3_raw)  # Interpolate width to 28
        down_3 = self.downsample_3(encoder_3)                       # [B, 512, 32, 14]

        # Bottleneck
        bottleneck = self.bottleneck(down_3)                        # [B, 512, 32, 14]

        # Decoder
        upsample_1 = self.upsample_1(bottleneck, (64, 27))          # [B, 512, 64, 27]

        if self.cbam:
            cbam_1 = self.cbam_1(encoder_3_raw)                     # Apply CBAM to the bottom encoder block
        else:
            cbam_1 = encoder_3_raw

        concat_1 = torch.cat([upsample_1, cbam_1], dim=1)           # [B, 512 + 512 = 1024, 64, 27]
        decoder_1 = self.decoder_block_1(concat_1)                  # [B, 256, 64, 27]

        upsample_2 = self.upsample_2(decoder_1)                     # [B, 256, 128, 54]

        if self.cbam:
            cbam_2 = self.cbam_2(encoder_2)                         # Apply CBAM to the middle encoder block
        else:
            cbam_2 = encoder_2

        concat_2 = torch.cat([upsample_2, cbam_2], dim=1)           # [B, 256 + 256 = 512, 128, 54]
        decoder_2 = self.decoder_block_2(concat_2)                  # [B, 128, 128, 54]

        upsample_3 = self.upsample_3(decoder_2, (256, 107))         # [B, 128, 256, 107]

        if self.cbam:
            cbam_3 = self.cbam_3(encoder_1_raw)                     # Apply CBAM to the top encoder block
        else:
            cbam_3 = encoder_1_raw

        concat_3 = torch.cat([upsample_3, cbam_3], dim=1)           # [B, 128 + 128 = 256, 256, 107]
        decoder_3 = self.decoder_block_3(concat_3)                  # [B, C, 256, 107]; C is defined by 'decoder3_out_channels'

        # Add to batch dict
        batch_dict['backbone_output'] = decoder_3

        return batch_dict


class UNetV2(nn.Module):
    def __init__(self, backbone_cfg):
        super().__init__()

        # Read out the configuration and define stuff here
        # -----
        self.backbone_cfg = backbone_cfg['cfg']
        self.input_stem = backbone_cfg['cfg']['input_stem']
        dropout = backbone_cfg['cfg']['dropout']
        # -----

        if self.input_stem:
            self.transform_input = InputStem(dop_in=64, dop_out=128, ele_in=37, ele_out=64, output_channels=128)
        else:
            self.transform_input = nn.Conv2d(in_channels=101, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Starting with the first encoder block
        self.encoder_block_1 = DoubleConvResidual(128, 128, dropout=dropout)        # [B, 128, H, W]
        self.downsample_1 = DownsampleBlurConv(128, 128)                            # [B, 128, H/2, W/2]; W: 107 -> 54

        # Second encoder block
        self.encoder_block_2 = DoubleConvResidual(128, 256, dropout=dropout)        # [B, 256, H/2, W/2]
        self.downsample_2 = DownsampleBlurConv(256, 256)                            # [B, 256, H/4, W/4]; W: 54 -> 27

        # Third encoder block
        self.encoder_block_3 = DoubleConvResidual(256, 512, dropout=dropout)        # [B, 512, H/4, W/4]
        self.downsample_3 = DownsampleBlurConv(512, 512)                            # [B, 512, H/8, W/8]; W: 27 -> 14

        # Bottleneck
        self.bottleneck = Bottleneck(512, 512, dropout=dropout)                     # [B, 512, H/8, W/8]

        # First decoder block
        self.upsample_1 = UpsampleConv(512, 512, padding=1)                         # [B, 512, H/4, W/4]
        self.cbam_1 = ConvolutionalBlockAttention(512, 512, dropout=dropout)        # [B, 512, H/4, W/4]
        self.decoder_block_1 = DoubleConvResidual(1024, 256, dropout=dropout)       # [B, 256, H/4, W/4]

        # Second decoder block
        self.upsample_2 = UpsampleConv(256, 256, padding=1)                         # [B, 256, H/2, W/2]
        self.cbam_2 = ConvolutionalBlockAttention(256, 256, dropout=dropout)        # [B, 256, H/2, W/2]
        self.decoder_block_2 = DoubleConvResidual(512, 128, dropout=dropout)        # [B, 128, H/2, W/2]

        # Third decoder block
        self.upsample_3 = UpsampleConv(128, 128, padding=1)                         # [B, 128, H, W]
        self.cbam_3 = ConvolutionalBlockAttention(128, 128, dropout=dropout)        # [B, 128, H, W]
        self.decoder_block_3 = DoubleConvResidual(256, 128, dropout=dropout)        # [B, 128, H, W]


    def forward(self, batch_dict):
        x = batch_dict['rdr_era_dra']                           # [B, 101, 256, 107]

        x_transformed = self.transform_input(x)                 # [B, 128, 256, 107]
        
        # Encoder
        encoder_1 = self.encoder_block_1(x_transformed)         # [B, 128, 256, 107]
        down_1 = self.downsample_1(encoder_1)                   # [B, 128, 128, 54]
        encoder_2 = self.encoder_block_2(down_1)                # [B, 256, 128, 54]
        down_2 = self.downsample_2(encoder_2)                   # [B, 256, 64, 27]
        encoder_3 = self.encoder_block_3(down_2)                # [B, 512, 64, 27]
        down_3 = self.downsample_3(encoder_3)                   # [B, 512, 32, 14]

        # Bottleneck
        bottleneck = self.bottleneck(down_3)                    # [B, 512, 32, 14]

        # Decoder
        upsample_1 = self.upsample_1(bottleneck, (64, 27))      # [B, 512, 64, 27]
        cbam_1 = self.cbam_1(encoder_3)                         # Apply CBAM to the bottom encoder block
        concat_1 = torch.cat([upsample_1, cbam_1], dim=1)       # [B, 512 + 512 = 1024, 64, 27]
        decoder_1 = self.decoder_block_1(concat_1)              # [B, 256, 64, 27]

        upsample_2 = self.upsample_2(decoder_1, (128, 54))      # [B, 256, 128, 54]
        cbam_2 = self.cbam_2(encoder_2)                         # Apply CBAM to the middle encoder block
        concat_2 = torch.cat([upsample_2, cbam_2], dim=1)       # [B, 256 + 256 = 512, 128, 54]
        decoder_2 = self.decoder_block_2(concat_2)              # [B, 128, 128, 54]

        upsample_3 = self.upsample_3(decoder_2, (256, 107))     # [B, 128, 256, 107]
        cbam_3 = self.cbam_3(encoder_1)                         # Apply CBAM to the top encoder block
        concat_3 = torch.cat([upsample_3, cbam_3], dim=1)       # [B, 128 + 128 = 256, 256, 107]
        decoder_3 = self.decoder_block_3(concat_3)              # [B, 128, 256, 107]

        batch_dict['backbone_output'] = decoder_3

        return batch_dict


if __name__ == "__main__":
    batch_dict = {'rdr_era_dra' : torch.randn(3, 101, 256, 107)}
    model = UNetPadCrop(backbone_cfg={'cfg': {'input_stem': True, 'dropout': 0.0, 'cbam': True, 'decoder3_out_channels': 128, 'mode': 'train'}})
    batch_dict = model(batch_dict)
    print(batch_dict['backbone_output'].shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total sum of all parameters over all layers: {total_params}")

    model = UNetInterp(backbone_cfg={'cfg': {'input_stem': True, 'dropout': 0.0, 'cbam': True, 'decoder3_out_channels': 128, 'mode': 'train'}})
    batch_dict = model(batch_dict)
    print(batch_dict['backbone_output'].shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total sum of all parameters over all layers: {total_params}")