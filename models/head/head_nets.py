import torch
import torch.nn as nn


#--------------------------------#
#----- Center Head Networks -----#
#--------------------------------#


class DefaultCenterHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.hidden_channels = 64
        self.groups = 32

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)
    

class ExpandedCenterHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.groups = 32

        self.head = nn.Sequential(
            # Cycle 1
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Cycle 2
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Cycle 3
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Output layer
            nn.Conv2d(self.hidden_channels, num_classes, 1)
        )

    def forward(self, x):
        return self.head(x)
    

class ExpandedResidualCenterHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.groups = 32

        self.conv1 = nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(self.groups, self.hidden_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(self.groups, self.hidden_channels)
        self.act2 = nn.SiLU(inplace=True)

        self.conv3 = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(self.groups, self.hidden_channels)
        self.act3 = nn.SiLU(inplace=True)

        self.output_conv = nn.Conv2d(self.hidden_channels, num_classes, 1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out + residual)  # Residual connection
        
        residual = out
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.act2(out + residual)  # Residual connection

        residual = out
        out = self.conv3(out)
        out = self.gn3(out)
        out = self.act3(out + residual)  # Residual connection

        out = self.output_conv(out)
        return out
    

#-------------------------------#
#----- Regression Networks -----#
#-------------------------------#


class DefaultRegHead(nn.Module):
    def __init__(self, in_channels, out_channels=8):
        super().__init__()

        self.hidden_channels = 64
        self.groups = 32

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1), # [dx, dy, dz, l, w, h, sin(yaw), cos(yaw)]
        )

    def forward(self, x):
        return self.head(x)
    

class ExpandedRegHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=8):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.groups = 32

        self.head = nn.Sequential(
            # Cycle 1
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Cycle 2
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Cycle 3
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Output layer
            nn.Conv2d(self.hidden_channels, out_channels, 1)  # [dx, dy, dz, l, w, h, sin(yaw), cos(yaw)]
        )

    def forward(self, x):
        return self.head(x)
    

class ExpandedResidualRegHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=8):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.groups = 32

        self.conv1 = nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(self.groups, self.hidden_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(self.groups, self.hidden_channels)
        self.act2 = nn.SiLU(inplace=True)

        self.conv3 = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(self.groups, self.hidden_channels)
        self.act3 = nn.SiLU(inplace=True)

        self.output_conv = nn.Conv2d(self.hidden_channels, out_channels, 1)  # [dx, dy, dz, l, w, h, sin(yaw), cos(yaw)]

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out + residual)  # Residual connection
        
        residual = out
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.act2(out + residual)  # Residual connection

        residual = out
        out = self.conv3(out)
        out = self.gn3(out)
        out = self.act3(out + residual)  # Residual connection

        out = self.output_conv(out)
        return out


class SplitExpandedRegHead(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.groups = 32

        self.base = nn.Sequential(
            # Cycle 1
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Cycle 2
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Cycle 3
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True)
        )

        self.mlp_pos = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden_channels, 3, kernel_size=1)  # [dx, dy, dz]
        )

        self.mlp_dim = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden_channels, 3, kernel_size=1)  # [l, w, h]
        )

        self.mlp_angle = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden_channels, 2, kernel_size=1)  # [sin(yaw), cos(yaw)]
        )

    def forward(self, x):
        base = self.base(x)
        
        pos = self.mlp_pos(base)
        dim = self.mlp_dim(base)
        angle = self.mlp_angle(base)

        return torch.cat([pos, dim, angle], dim=1) # [B, 8, H, W] where the 8 channels are [dx, dy, dz, l, w, h, sin(yaw), cos(yaw)]


#-----------------------------------#
#----- Classification Networks -----#
#-----------------------------------#


class DefaultClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.groups = 32

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)     # [B, num_classes, H, W]
    

class ExpandedClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.groups = 32

        self.head = nn.Sequential(
            # Cycle 1
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Cycle 2
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Cycle 3
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(self.groups, self.hidden_channels),
            nn.SiLU(inplace=True),
            # Output layer
            nn.Conv2d(self.hidden_channels, num_classes, 1)
        )

    def forward(self, x):
        return self.head(x)     # [B, num_classes, H, W]
    

class ExpandedResidualClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.groups = 32

        self.conv1 = nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(self.groups, self.hidden_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(self.groups, self.hidden_channels)
        self.act2 = nn.SiLU(inplace=True)

        self.conv3 = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(self.groups, self.hidden_channels)
        self.act3 = nn.SiLU(inplace=True)

        self.output_conv = nn.Conv2d(self.hidden_channels, num_classes, 1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out + residual)  # Residual connection
        
        residual = out
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.act2(out + residual)  # Residual connection

        residual = out
        out = self.conv3(out)
        out = self.gn3(out)
        out = self.act3(out + residual)  # Residual connection

        out = self.output_conv(out)
        return out     # [B, num_classes, H, W]


# ----------------------- #
# ----- 3D Networks ----- #
# ----------------------- #


class DefaultCenterHead3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_channels = 64
        self.groups = 32

        self.head = nn.Sequential(
            # Full 3D convolution
            nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),  #[5, 1, 128, 256, 112] -> [5, 1, 64, 256, 112]
            #nn.GroupNorm(4, 32),
            nn.SiLU(),
            
            # Reduce elevation dimension
            nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),  #[5, 1, 64, 256, 112] -> [5, 1, 32, 256, 112]
            #nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # Final 2D-like convolution
            nn.Conv3d(8, 8, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),  #[5, 1, 32, 256, 112] -> [5, 1, 16, 256, 112]
            #nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            # Output layer
            nn.Conv3d(8, 8, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),  #[5, 1, 16, 256, 112] -> [5, 1, 8, 256, 112]
            #nn.GroupNorm(8, 64),
            nn.SiLU(),

            nn.Conv3d(8, 4, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),  #[5, 1, 8, 256, 112] -> [5, 1, 4, 256, 112]
            #nn.GroupNorm(4, 32),
            nn.SiLU(),

            nn.Conv3d(4, 2, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),  #[5, 1, 4, 256, 112] -> [5, 1, 2, 256, 112]
            #nn.GroupNorm(1, 8),
            nn.SiLU(),

            nn.Conv3d(2, 1, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))  #[5, 1, 2, 256, 112] -> [5, 1, 1, 256, 112]
        )

    def forward(self, x):
        return self.head(x)
    

class DefaultRegHead3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_channels = 64
        self.groups = 32

        self.head = nn.Sequential(
            # Full 3D convolution
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),  #[5, 1, 128, 256, 112] -> [5, 1, 64, 256, 112]
            #nn.GroupNorm(4, groups),
            nn.SiLU(),
            
            # Reduce elevation dimension
            nn.Conv3d(8, 8, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),  #[5, 1, 64, 256, 112] -> [5, 1, 32, 256, 112]
            #nn.GroupNorm(4, groups),
            nn.SiLU(),
            
            # Final 2D-like convolution
            nn.Conv3d(8, 4, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),  #[5, 1, 32, 256, 112] -> [5, 1, 16, 256, 112]
            #nn.GroupNorm(2, 16),
            nn.SiLU(),
            
            # Output layer
            nn.Conv3d(4, 2, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),  #[5, 1, 16, 256, 112] -> [5, 1, 8, 256, 112]
            #nn.GroupNorm(1, 8),
            nn.SiLU(),

            nn.Conv3d(2, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        )

    def forward(self, x):
        return self.head(x)
    