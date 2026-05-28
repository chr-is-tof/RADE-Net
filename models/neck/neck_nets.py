import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

from nets.ops.modules.ms_deform_attn import MSDeformAttn


###########################
# ----- Neck Blocks ----- #
###########################


class ConvGNAct(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=3, 
                 stride=1, padding=1, dilation=1, groups=32):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.gn = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        return self.act(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=128, kernel_size=3, dilation=1, groups=32):
        super().__init__()

        self.conv1 = ConvGNAct(in_channels, in_channels, kernel_size=kernel_size, 
                               padding=dilation, dilation=dilation, groups=groups)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                               padding=1, dilation=1)
        self.gn = nn.GroupNorm(groups, in_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)         # First conv + GN + Act
        out = self.conv2(out)       # Second conv
        out = self.gn(out)          # GroupNorm
        return self.act(out + x)    # Residual connection + Act
    

class DeformConvGNAct(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=32,
                 modulation=True):
        super().__init__()

        self.k = kernel_size
        self.modulation = modulation

        # offsets: 2*k*k channels
        offset_ch = 2 * kernel_size * kernel_size
        # mask: k*k channels (DCNv2-style)
        mask_ch = kernel_size * kernel_size if modulation else 0

        self.offset_mask = nn.Conv2d(
            in_channels, offset_ch + mask_ch,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation
        )

        self.dcn = DeformConv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            bias=False
        )

        self.gn = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

        # Start as "regular conv" behavior (zero offsets; mask ~ 0.5 if modulation=True)
        nn.init.constant_(self.offset_mask.weight, 0.0)
        nn.init.constant_(self.offset_mask.bias, 0.0)

    def forward(self, x):
        om = self.offset_mask(x)
        k2 = self.k * self.k

        if self.modulation:
            offset, mask = torch.split(om, [2 * k2, k2], dim=1)
            mask = mask.sigmoid()
        else:
            offset, mask = om, None

        # DeformConv2d does not support deterministic mode, so warnings are enabled in set_seeds when using this block
        y = self.dcn(x, offset, mask=mask) if mask is not None else self.dcn(x, offset)

        y = self.gn(y)
        return self.act(y)


class DeformConvResidualBlock(nn.Module):
    def __init__(self, in_channels=128, kernel_size=3, dilation=1, groups=32,
                 modulation=True):
        super().__init__()

        self.conv1 = DeformConvGNAct(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=dilation, dilation=dilation,
            groups=groups,
            modulation=modulation
        )
        
        self.conv2 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, padding=1, dilation=1
        )
        self.gn = nn.GroupNorm(groups, in_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.gn(out)
        return self.act(out + x)


#########################
# ----- Neck Nets ----- #
#########################


class DefaultNeck(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()

        self.neck = nn.Sequential(
            ConvGNAct(in_channels, in_channels),
            ConvGNAct(in_channels, in_channels),
            ConvGNAct(in_channels, in_channels),
        )

    def forward(self, x):
        return self.neck(x)
    

class DynamicDefaultNeck(nn.Module):
    def __init__(self, in_channels=128, num_layers=3):
        super().__init__()

        layers = [ConvGNAct(in_channels, in_channels) for _ in range(num_layers)]
        self.neck = nn.Sequential(*layers)

    def forward(self, x):
        return self.neck(x)


class DilatedResidualNeck(nn.Module):
    def __init__(self, in_channels=128, dilation=(1, 2, 3)):
        super().__init__()

        self.neck = nn.Sequential(
            *[ResidualBlock(in_channels, dilation=d) for d in dilation]
        )

    def forward(self, x):
        return self.neck(x)
    

class DeformConvResidualNeck(nn.Module):
    def __init__(self, in_channels=128, dilation=(1, 2, 3), groups=32, modulation=True):
        super().__init__()

        self.neck = nn.Sequential(
            *[DeformConvResidualBlock(in_channels, dilation=d, groups=groups, modulation=modulation)
              for d in dilation]
        )

    def forward(self, x):
        return self.neck(x)


class DeformableNeck(nn.Module):
    def __init__(self, in_channels=128, num_heads=4, embed_dim=256, num_levels=1, num_points=8):
        super().__init__()
        
        self.neck = nn.Sequential(
           ConvGNAct(in_channels, in_channels),
           DeformableAttention(in_channels, num_heads, embed_dim, num_levels, num_points),
           ConvGNAct(in_channels, in_channels),
           DeformableAttention(in_channels, num_heads, embed_dim, num_levels, num_points),
           ConvGNAct(in_channels, in_channels)
        )

    def forward(self, x):
        return self.neck(x)


class DeformableAttention(nn.Module):
    def __init__(self, in_channels=128, num_heads=8, embed_dim=256, num_levels=1, num_points=4):
        super().__init__()

        self.ms_deform_attn = MSDeformAttn(embed_dim, num_heads, num_levels, num_points)
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        self.norm = nn.GroupNorm(32, in_channels)
        self.act = nn.SiLU(inplace=True)

    @staticmethod
    def get_reference_points(H, W, device):
        """Generates a grid of reference points for a given spatial shape."""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        # Normalize to [0, 1]
        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W
        # Stack to (H*W, 2)
        ref_points = torch.stack((ref_x, ref_y), -1)
        # Expand for num_levels as in deformable attention [B, H*W, num_levels, 2]
        ref_points = ref_points.unsqueeze(1)  # [H*W, 1, 2]
        return ref_points

    def forward(self, x):
        B, C, H, W = x.shape
        x_proj = self.input_proj(x).flatten(2).permute(0, 2, 1)                 # [B, H*W, embed_dim]

        # Create reference points
        reference_points = self.get_reference_points(H, W, x.device)            # [H*W, num_levels, 2]
        reference_points = reference_points.unsqueeze(0).repeat(B, 1, 1, 1)     # [B, H*W, num_levels, 2]

        # Assume single level feature map for simplicity
        spatial_shapes = torch.tensor([[H, W]], dtype=torch.long, device=x.device)
        level_start_index = torch.tensor([0], dtype=torch.long, device=x.device)

        out = self.ms_deform_attn(
            query=x_proj,
            reference_points=reference_points,
            input_flatten=x_proj,  # For single-level, query and value can be the same
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=None  # Optional, None if all positions are valid
        )

        out = out.permute(0, 2, 1).view(B, -1, H, W)  # [B, embed_dim, H, W]
        out = self.output_proj(out)
        out = self.norm(out + x) # Residual connection
        return self.act(out)