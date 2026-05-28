import torch
import torch.nn as nn
import torch.nn.functional as F


# We use norm after the convolution. Careful with the order of operations. Should probably
# use norm before calling
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0, num_groups=32):
        super().__init__()

        # NOTE: Since we might use a distributed setup, it is important to have a correctly
        # synced BatchNorm. We use GroupNorm instead, which is more stable in distributed setups
        #
        # Also, GroupNorm is more stable for small batch sizes
        if out_channels == 1:
            layer_norm = nn.Identity()
        else:
            layer_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            layer_norm,
            nn.SiLU(inplace=True)
        ]
        
        # Add Dropout if specified
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        
        self.single_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.single_conv(x)
    

class DoubleConvResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            SingleConv(in_channels, out_channels, kernel_size, padding, dropout),
            SingleConv(out_channels, out_channels, kernel_size, padding, dropout)
        )

        if in_channels != out_channels:
            # If the input and output channels are different, we need to adjust the input
            # to match the output channels
            self.adjust_input = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.adjust_input = nn.Identity()

    def forward(self, x):
        return self.double_conv(x) + self.adjust_input(x)
    

class Downsample(nn.Module):
    def __init__(self, kernel_size=2, padding=0):
        super().__init__()

        # After the pooling operation, the remaininzg size is (B, C, H/2, W/2)
        self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=2, padding=padding)

    def forward(self, x):
        return self.pooling(x)
    

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, padding=0):
        super().__init__()

        # After the deconvolution, the new size is (B, C, H*2, W*2)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding)

    def forward(self, x):
        return self.upsample(x)
    

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        # AdapativeMaxPool2d is non-deterministic, therefore we use MaxPool2d with kernel size equal to input size
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        reduced_channels = max(1, in_channels // reduction)

        self.fc1 = nn.Linear(in_channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, in_channels)

    def forward(self, x):
        # max_pool = self.max_pool(x)                                   # Size is collapsed to (B, C, 1, 1)
        max_pool = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)))  # Alternative to AdaptiveMaxPool2d
        avg_pool = self.avg_pool(x)                                     # Size is collapsed to (B, C, 1, 1)

        max_pool = max_pool.view(max_pool.size(0), -1)      # Reshapes to (B, C * H * W)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)      # Reshapes to (B, C * H * W)

        # Input is (B, C)
        max_out = self.fc2(F.relu(self.fc1(max_pool)))      # Pushing the max pooling through the MLP
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))      # Pushing the avg pooling through the MLP

        # Max and avg are added together
        combination = max_out + avg_out

        # Push through the sigmoid to get a scalar between [0, 1], i.e., a weight on a per channel basis
        channel_weights = torch.sigmoid(combination)
        
        # Reshape to fit dimensions of input
        channel_weights = channel_weights.view(x.size(0), -1, 1, 1)

        return x * channel_weights


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # in_channels are 2 due to the concatenation of max and avg pool
        in_channels = 2
        out_channels = 1
        kernel_size = 7         # from the paper

        # Keep dimensions
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # Concatenate over channel dimension
        concatenation = torch.cat([max_pool, avg_pool], dim=1)

        # Collapse to a single channel
        unscaled_spatial_attention = self.convolution(concatenation)
        
        # Push through the sigmoid to get a scalar between [0, 1]
        return torch.sigmoid(unscaled_spatial_attention)
    

class ConvolutionalBlockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0, skip_first_layer=False):
        super().__init__()

        # If skip_first_layer is True, we do not apply the first convolution, which is used for visualization purposes
        if skip_first_layer:
            self.start_conv = nn.Identity()

            self.channel_attention = ChannelAttentionModule(in_channels)
            self.spatial_attention = SpatialAttentionModule()

            self.end_conv = SingleConv(in_channels, out_channels, kernel_size, padding, dropout)
        else:
            self.start_conv = SingleConv(in_channels, out_channels, kernel_size, padding, dropout)
            
            self.channel_attention = ChannelAttentionModule(out_channels)
            self.spatial_attention = SpatialAttentionModule()

            self.end_conv = SingleConv(out_channels, out_channels, kernel_size, padding, dropout)

    def forward(self, x):
        # Start with a simple convolution
        start_conv = self.start_conv(x)

        # Channel attention already multiplies weights with the convolution
        channel_weighted_conv = self.channel_attention(start_conv)

        # Compute spatial attention on the channel weighting conv
        spatial_attention = self.spatial_attention(channel_weighted_conv)

        # Apply the spatial attention to the channel weighted conv
        completely_weighted_conv = channel_weighted_conv * spatial_attention

        # Add a residual connection for performance
        residual = completely_weighted_conv + start_conv

        # Apply output conv
        return self.end_conv(residual)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()

        self.conv1 = DoubleConvResidual(in_channels, out_channels, kernel_size, padding, dropout)
        self.conv2 = DoubleConvResidual(out_channels, out_channels, kernel_size, padding, dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    

class NonLocalBlock(nn.Module):
    def __init__(self, channel_dimension, num_heads, normalization=True, get_attention_weights=False):
        super().__init__()

        self.channel_dim = channel_dimension
        self.num_heads = num_heads
        self.head_dim = channel_dimension // num_heads
        kernel_size = 1
        self.get_attention_weights = get_attention_weights

        self.Q = nn.Conv2d(channel_dimension, channel_dimension, kernel_size=kernel_size)
        self.K = nn.Conv2d(channel_dimension, channel_dimension, kernel_size=kernel_size)
        self.V = nn.Conv2d(channel_dimension, channel_dimension, kernel_size=kernel_size)

        self.out_conv = nn.Conv2d(channel_dimension, channel_dimension, kernel_size=kernel_size)

        if normalization:
            self.norm = nn.GroupNorm(num_groups=32, num_channels=channel_dimension)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        batch_size, channel_dim, height, width = x.size()
        
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # Reshape Q, K, V for multi-head attention
        # Shape is (batch_size, num_heads, head_dim, height, width)
        Q = Q.view(batch_size, self.num_heads, self.head_dim, height, width)
        K = K.view(batch_size, self.num_heads, self.head_dim, height, width)
        V = V.view(batch_size, self.num_heads, self.head_dim, height, width)

        # Permute to (batch_size, num_heads, height, head_dim, width)
        Q = Q.permute(0, 1, 3, 2, 4)
        K = K.permute(0, 1, 3, 2, 4)
        V = V.permute(0, 1, 3, 2, 4)

        # Compute attention scores over width, which results in a shape of (batch_size, num_heads, height, width, width)
        attention_scores = torch.einsum('bnhdi,bnhdj->bnhij', Q, K) / (self.head_dim ** 0.5)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention weights to V, which results in a shape of (batch_size, num_heads, height, head_dim, width)
        attended_values = torch.einsum('bnhij,bnhdj->bnhdi', attention_weights, V)

        # Permute back, which results in a shape of (batch_size, num_heads, head_dim, height, width)
        # Contiguous to ensure memory layout is correct (required by .view())
        attended_values = attended_values.permute(0, 1, 3, 2, 4).contiguous()

        # Reshape back to (batch_size, channel_dim, height, width)
        attended_values = attended_values.view(batch_size, channel_dim, height, width)

        # Apply the output convolution and normalization
        final_conv = self.out_conv(attended_values)

        if self.get_attention_weights:
            # Reduce attention weights to a shape of (batch_size, width, width) by averaging over heads and height
            attention_weights = attention_weights.mean(dim=1).mean(dim=1)

            return self.norm(final_conv), attention_weights
        else:
            return self.norm(final_conv), None
        

class RnnChannelEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=True):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional, batch_first=True)

        self.output_size = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, x):
        # x is of shape (B, H, W, L)
        B, H, W, L = x.shape

        # Reshape to (B*H, W, L) to process each row independently
        x_reshaped = x.view(B * H, W, L)

        # Pass through the LSTM
        lstm_out, _ = self.lstm(x_reshaped)  # lstm_out is of shape (B*H, W, output_size)

        # Reshape back to (B, H, W, output_size)
        output = lstm_out.view(B, H, W, self.output_size)

        return output


class InputStem(nn.Module):
    def __init__(self, dop_in=64, dop_out=128, ele_in=37, ele_out=64, output_channels=192):
        super().__init__()
        
        self.doppler_conv = nn.Sequential(
            nn.Conv2d(dop_in, dop_out, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=dop_out),
            nn.SiLU()
        )

        self.elevation_conv = nn.Sequential(
            nn.Conv2d(ele_in, ele_out, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=ele_out),
            nn.SiLU()
        )

        self.mix_conv = nn.Sequential(
            nn.Conv2d(dop_out + ele_out, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=output_channels),
            nn.SiLU()
        )

    def forward(self, x):
        doppler_data = x[:, :64, :, :]      # First 64 channels are Doppler
        elevation_data = x[:, 64:, :, :]    # Last 37 channels are Elevation

        # The idea of this stem is to process Doppler and Elevation separately
        # and then combine the features. Additionally, we use 1x1 convolutions to
        # adjust the number of channels before mixing and have a more diverse feature set.
        doppler_features = self.doppler_conv(doppler_data)
        elevation_features = self.elevation_conv(elevation_data)

        # Concatenate along the channel dimension
        combined_features = torch.cat([doppler_features, elevation_features], dim=1)

        # Final mixing convolution
        return self.mix_conv(combined_features)


class DepthwiseBlur(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        self.filt_size = 3

        kernel = torch.tensor([
            [1., 2., 1.],
            [2., 4., 2.],
            [1., 2., 1.]
        ]) / 16.0

        k = kernel.view(1, 1, self.filt_size, self.filt_size)

        # Register as buffer to avoid being considered a model parameter
        self.register_buffer("kernel", k) # shape [1,1,3,3]

        # Should be better than zero padding
        self.pad = nn.ReplicationPad2d(self.filt_size // 2) # Reflect / replicate helps at FOV edge

    def forward(self, x):
        # x: [B, C, H, W]; C must equal self.channels
        assert x.size(1) == self.channels, f"Expected {self.channels} channels, got {x.size(1)}"
        
        # expand kernel to [C,1,3,3] and match dtype/device automatically
        weight = self.kernel.expand(self.channels, 1, self.filt_size, self.filt_size)
        weight = weight.to(dtype=x.dtype).contiguous()
        
        x = self.pad(x)

        return nn.functional.conv2d(x, weight, stride=1, padding=0, groups=self.channels)


class DownsampleBlurConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        
        self.blur = DepthwiseBlur(in_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        # Blurs the input before downsampling
        x = self.blur(x)            # [B, C, H, W]
        return self.conv(x)         # [B, C, H/2, W/2]
    

class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mode='bilinear', padding=1):
        super().__init__()
        
        self.mode = mode

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU()
        )

    def forward(self, x, target_size):
        # target_size: (H, W)
        x = F.interpolate(
            x, 
            size=target_size, 
            mode=self.mode, 
            align_corners=False if self.mode in ('bilinear', 'bicubic') else None
        )
        return self.conv(x)


if __name__ == "__main__":
    x = torch.randn(3, 32, 64, 64)                            # Batch size of 3, 32 channels, 64x64 image
    model = DoubleConvResidual(32, 32)
    output = model(x)
    print(f"Shape of DoubleConvResidual should be {x.shape[0], 32, x.shape[2], x.shape[3]}: {output.shape}")

    x2 = torch.randn(3, 32, 5, 5)
    model2 = ChannelAttentionModule(32)
    output = model2(x2)
    print(f"Shape of ChannelAttentionModule should be {x2.shape[0], 32, x2.shape[2], x2.shape[3]}: {output.shape}")

    x3 = torch.rand(3, 32, 5, 5)
    model3 = SpatialAttentionModule()
    output = model3(x3)
    print(f"Shape of ChannelAttentionModule should be {x3.shape[0], 1, x3.shape[2], x3.shape[3]}: {output.shape}")

    x4 = torch.rand(3, 32, 128, 128)
    model4 = ConvolutionalBlockAttention(32, 32)
    output = model4(x4)
    print(f"Shape of ChannelAttentionModule should be {x4.shape[0], 32, x4.shape[2], x4.shape[3]}: {output.shape}")

    x5 = torch.rand(3, 32, 256, 112)
    model5 = NonLocalBlock(32, 4)
    output, _ = model5(x5)
    print(f"Shape of NonLocalBlock should be {x5.shape[0], 32, x5.shape[2], x5.shape[3]}: {output.shape}")

    x5 = torch.rand(3, 32, 256, 112)
    model5 = NonLocalBlock(32, 4, get_attention_weights=True)
    output, attention_weights = model5(x5)
    print(f"Shape of NonLocalBlock should be {x5.shape[0], 32, x5.shape[2], x5.shape[3]}: {output.shape}")
    print(f"Shape of attention weights should be {x5.shape[0], x5.shape[3], x5.shape[3]}: {attention_weights.shape}")

