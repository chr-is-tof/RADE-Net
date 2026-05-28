import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neck.neck_nets import DilatedResidualNeck
from utils.fusion.camera_projection import get_dict_cam_calib_from_yml

class PVCamFPN(torch.nn.Module):
    def __init__(self, input_dim=384, output_channels=256, output_height=37, output_width=112):
        super().__init__()
        
        # Reshape patch tokens to spatial grid (14x14)
        self.patch_size = 14
        self.output_height = output_height
        self.output_width = output_width
        
        # Project from 384 to desired channel dimension
        self.channel_proj = nn.Linear(input_dim, output_channels)
        
        # FPN layers - upsample from 14x14 to 37x112
        self.fpn_layer1 = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_layer2 = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_layer3 = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=(1,4), stride=(1,2), padding=1),  # 56x56 -> 56x112
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.refine = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, batch_dict):
        cam_features = batch_dict['cam_features']  # shape: (B, 1+4+14*14, 384)
        
        B = cam_features.shape[0]
        
        # Extract patch tokens (skip CLS and register tokens)
        # Assuming: 1 CLS token + 4 register tokens + 196 patch tokens
        patch_tokens = cam_features[:, 5:, :]  # (B, 196, 384)
        
        # Project channels
        patch_tokens = self.channel_proj(patch_tokens)  # (B, 196, 128)
        
        # Reshape to spatial grid
        H = W = self.patch_size
        C = patch_tokens.shape[-1]  # 128
        patch_tokens = patch_tokens.permute(0, 2, 1)  # (B, 128, 196)
        patch_tokens = patch_tokens.reshape(B, C, H, W)  # (B, 128, 14, 14)
        
        # Apply FPN upsampling
        x = self.fpn_layer1(patch_tokens)  # (B, 128, 28, 28)
        x = self.fpn_layer2(x)             # (B, 128, 56, 56)
        x = self.fpn_layer3(x)             # (B, 128, 112, 112)

        # Use bilinear interpolation to reach target size (preserves spatial info)
        x = F.interpolate(x, size=(self.output_height, self.output_width), 
                         mode='bilinear', align_corners=False)  # (B, 128, 256, 112)

        x = self.refine(x)
        
        batch_dict['PV_cam_features'] = x
        
        return batch_dict


class CamNeck(torch.nn.Module):
    def __init__(self, cam_cfg, input_dim=384, output_channels=128):
        super().__init__()
        
        self.cam_cfg = cam_cfg
        
        # Project from 384 to desired channel dimension
        self.channel_proj = nn.Linear(input_dim, output_channels)

        if self.cam_cfg['cam_neck_output'] == '180x320':
            # High-res path: 720x1280 input -> 45x80 patch grid (patch_size=16)
            self.patch_h = 45   # 720 // 16
            self.patch_w = 80   # 1280 // 16

            # FPN layers - upsample from 45x80 towards 180x320
            self.fpn_layer1 = nn.Sequential(
                nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),  # 45x80 -> 90x160
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            self.fpn_layer2 = nn.Sequential(
                nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),  # 90x160 -> 180x320
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # Standard path: 224x224 input -> 14x14 patch grid (patch_size=16)
            self.patch_h = 14
            self.patch_w = 14

            # FPN layers - upsample from 14x14 to target size
            self.fpn_layer1 = nn.Sequential(
                nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            self.fpn_layer2 = nn.Sequential(
                nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            self.fpn_layer3 = nn.Sequential(
                nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

            if self.cam_cfg['cam_neck_output'] == '256x112':
                self.fpn_layer4 = nn.Sequential(
                    nn.ConvTranspose2d(output_channels, output_channels, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),  # 112x112 -> 224x112
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True)
                )
            elif self.cam_cfg['cam_neck_output'] == '224x224':
                self.fpn_layer4 = nn.Sequential(
                    nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),  # 112x112 -> 224x224
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                raise ValueError("Invalid cam_neck_output configuration.")

        self.refine = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, batch_dict):
        cam_features = batch_dict['cam_features']
        
        B = cam_features.shape[0]
        
        # Extract patch tokens (skip CLS and register tokens)
        # 1 CLS token + 4 register tokens = 5 prefix tokens
        patch_tokens = cam_features[:, 5:, :]  # (B, patch_h*patch_w, 384)
        
        # Project channels
        patch_tokens = self.channel_proj(patch_tokens)  # (B, patch_h*patch_w, 128)
        
        # Reshape to spatial grid
        H, W = self.patch_h, self.patch_w
        C = patch_tokens.shape[-1]
        patch_tokens = patch_tokens.permute(0, 2, 1)        # (B, C, patch_h*patch_w)
        patch_tokens = patch_tokens.reshape(B, C, H, W)     # (B, C, patch_h, patch_w)
        
        if self.cam_cfg['cam_neck_output'] == '180x320':
            # High-res path: 45x80 -> 90x160 -> 180x320
            x = self.fpn_layer1(patch_tokens)
            x = self.fpn_layer2(x)
        else:
            # Standard path: 14x14 -> 28x28 -> 56x56 -> 112x112 -> target
            x = self.fpn_layer1(patch_tokens)
            x = self.fpn_layer2(x)
            x = self.fpn_layer3(x)
            x = self.fpn_layer4(x)

            if self.cam_cfg['cam_neck_output'] == '256x112':
                # Use bilinear interpolation to reach target size (preserves spatial info)
                x = F.interpolate(x, size=(256, 112), mode='bilinear', align_corners=False)

        x = self.refine(x)
        
        batch_dict['cam_features'] = x
        
        return batch_dict
    

class RadCamConcatFusion(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # No parameters needed for concatenation fusion
        if cfg['cam_config']['cam_neck']:
            self.cam_neck = DilatedResidualNeck(in_channels=128, dilation=(1, 2, 3))
        if cfg['cam_config']['rad_neck']:
            self.rad_neck = DilatedResidualNeck(in_channels=128, dilation=(1, 2, 3))

    def forward(self, batch_dict):
        radar_features = batch_dict['backbone_output']
        cam_features = batch_dict['cam_features']
        # Concatenate along the channel dimension
        radar_features = self.rad_neck(radar_features)
        cam_features = self.cam_neck(cam_features)
        fused_features = torch.cat((radar_features, cam_features), dim=1)
        batch_dict['backbone_output'] = fused_features
        return batch_dict


class RadCamPV2BEVFusion(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.DEFORM_ATTN_CFG['d_model']
        self.n_heads = cfg.DEFORM_ATTN_CFG['n_heads']
        self.n_points = cfg.DEFORM_ATTN_CFG['n_points']
        self.n_levels = 1 
        self.cfg = cfg
        bev_h, bev_w = 256, 112 
        self.bev_depth = cfg.BEV_LIFTFUSE_CFG.get('depth_levels', 4)
        self.pos_embed = nn.Parameter(0.1 *torch.randn(1, self.d_model, self.bev_depth, bev_h, bev_w))

        if cfg.CAM_CONFIG['cam_neck']:
            raise NotImplementedError("cam_neck for PV2BEV fusion not implemented yet.")
        if cfg.CAM_CONFIG['rad_neck']:
            self.rad_neck = DilatedResidualNeck(in_channels=128, dilation=(1, 2, 3))

    @staticmethod
    def get_reference_points(H, W, D, device):
        """Generates a grid of reference points for a given spatial shape."""
        ref_y, ref_x, ref_z = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            torch.linspace(0.5, D - 0.5, D, dtype=torch.float32, device=device),
            indexing='ijk'
        )
        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W
        ref_z = ref_z.reshape(-1) / D
        ref_points = torch.stack((ref_x, ref_y, ref_z), -1)
        return ref_points
    
    def forward(self, batch_dict):
        query_features = batch_dict['backbone_output']    # Radar features (B, C, Hr, Wr)
        value_features = batch_dict['cam_features']     # Camera features (B, C, Hc, Wc)
        if self.cfg.CAM_CONFIG['rad_neck']:
            query_features = self.rad_neck(query_features)
        
        B, C, H, W = query_features.shape
        device = query_features.device
        query_features_3d = query_features.unsqueeze(2).repeat(1, 1, self.bev_depth, 1, 1) # (B, C, Dr=4, Hr, Wr)
        query_features_3d = query_features_3d + self.pos_embed

        # Query is from radar features
        query = query_features_3d.flatten(2).transpose(1, 2) # (B, D*H*W, C)
        query_residual = query.clone()

        # Value is from camera features
        value = value_features.flatten(2).transpose(1, 2) # (B, H*W, C)

        # Reference points are a grid on the query feature map
        reference_points = self.get_reference_points(H, W, self.bev_depth, device)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2) # (B, H*W, 1, 2)

        # The attention module samples from one level (camera features) with shape (H, W)
        input_spatial_shapes = torch.as_tensor([[H, W]], dtype=torch.long, device=device)
        input_level_start_index = torch.as_tensor([0], dtype=torch.long, device=device)

        # The module internally computes sampling offsets and attention weights from the query
        fused_query = self.deform_attn(
            query, 
            reference_points, 
            value, 
            input_spatial_shapes, 
            input_level_start_index
        )
        fused_query = self.dropout(fused_query) + query_residual
        fused_features = fused_query.transpose(1, 2).view(B, C, H, W)
        
        # Project to match the channel dimension of simple concatenation
        fused_features = fused_features.permute(0, 2, 3, 1) # (B, H, W, C)
        fused_features = self.output_proj(fused_features)   # (B, H, W, 2*C)
        fused_features = fused_features.permute(0, 3, 1, 2) # (B, 2*C, H, W)

        batch_dict['backbone_output'] = fused_features
        return batch_dict
    

class RadCamDefAttFusion(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from nets.ops.modules.ms_deform_attn import MSDeformAttn

        self.d_model = cfg['deform_attn_cfg']['d_model']
        self.n_heads = cfg['deform_attn_cfg']['n_heads']
        self.n_points = cfg['deform_attn_cfg']['n_points']
        self.n_levels = 1 
        self.cfg = cfg
        bev_h, bev_w = 256, 112 
        self.pos_embed = nn.Parameter(torch.randn(1, self.d_model, bev_h, bev_w))

        if cfg['cam_config']['cam_neck']:
            self.cam_neck = DilatedResidualNeck(in_channels=128, dilation=(1, 2, 3))
        if cfg['cam_config']['rad_neck']:
            self.rad_neck = DilatedResidualNeck(in_channels=128, dilation=(1, 2, 3))

        self.deform_attn = MSDeformAttn(
            d_model=self.d_model, 
            n_levels=self.n_levels, 
            n_heads=self.n_heads, 
            n_points=self.n_points
        )
        
        self.dropout = nn.Dropout(0.1)
        self.output_proj = nn.Linear(self.d_model, self.d_model * 2)

    @staticmethod
    def get_reference_points(H, W, device):
        """Generates a grid of reference points for a given spatial shape."""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1) / H
        ref_x = ref_x.reshape(-1) / W
        ref_points = torch.stack((ref_x, ref_y), -1)
        return ref_points

    def forward(self, batch_dict):
        query_features = batch_dict['backbone_output']    # Radar features (B, C, H, W)
        value_features = batch_dict['cam_features']     # Camera features (B, C, H, W)
        if self.cfg['cam_config']['rad_neck']:
            query_features = self.rad_neck(query_features)
        if self.cfg['cam_config']['cam_neck']:
            value_features = self.cam_neck(value_features)
        
        B, C, H, W = query_features.shape
        device = query_features.device
        query_features = query_features + self.pos_embed * 0.1
        # Query is from radar features
        query = query_features.flatten(2).transpose(1, 2) # (B, H*W, C)
        query_residual = query.clone()
        # Value is from camera features
        value = value_features.flatten(2).transpose(1, 2) # (B, H*W, C)

        # Reference points are a grid on the query feature map
        reference_points = self.get_reference_points(H, W, device)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2) # (B, H*W, 1, 2)

        # The attention module samples from one level (camera features) with shape (H, W)
        input_spatial_shapes = torch.as_tensor([[H, W]], dtype=torch.long, device=device)
        input_level_start_index = torch.as_tensor([0], dtype=torch.long, device=device)

        fused_query = self.deform_attn(
            query, 
            reference_points, 
            value, 
            input_spatial_shapes, 
            input_level_start_index
        )

        fused_query = self.dropout(fused_query) + query_residual
        fused_features = fused_query.transpose(1, 2).view(B, C, H, W)
        # Project to match the channel dimension of simple concatenation
        fused_features = fused_features.permute(0, 2, 3, 1) # (B, H, W, C)
        fused_features = self.output_proj(fused_features)   # (B, H, W, 2*C)
        fused_features = fused_features.permute(0, 3, 1, 2) # (B, 2*C, H, W)

        batch_dict['backbone_output'] = fused_features
        return batch_dict
    

class FusingCrossAttentionV2(nn.Module):
    """
    utilizes deformable attention to fuse camera- and radar BEV embeddings
    """

    def __init__(self, dim: int = 128, dropout: float = 0.1):
        super(FusingCrossAttentionV2, self).__init__()
        from nets.ops.modules.ms_deform_attn import MSDeformAttn

        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        # Deform.DETR: n_heads=8 n_points=4
        self.fusing_deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=8, n_points=4)

    def forward(self, query: torch.Tensor, input_feats: torch.Tensor, query_pos: torch.Tensor = None) -> torch.Tensor:
        """
        Utilizes deformable attention to fuse camera- and radar BEV embeddings
        Args:
            query: radar BEV embeddings
            input_feats: camera BEV embeddings
            query_pos: (optional) additional position embedding

        Returns:
            torch.Tensor: BEV feature embedding after one fusion block
        """
        query_residual = query.clone()

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        Z, X = 200, 200
        ref_z, ref_x = torch.meshgrid(
            torch.linspace(0.5, Z - 0.5, Z, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X - 0.5, X, dtype=torch.float, device=query.device),
            indexing='ij'
        )
        ref_z = ref_z.reshape(-1)[None] / Z
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_z, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2)  # (B, N, 1, 2)

        input_spatial_shapes = query.new_zeros([1, 2]).long()
        input_spatial_shapes[:] = 200
        input_level_start_index = query.new_zeros([1, ]).long()
        queries = self.fusing_deformable_attention(query, reference_points, input_feats,
                                                   input_spatial_shapes, input_level_start_index)

        return self.dropout(queries) + query_residual
    

class RadCamDefAttProjectionFusion(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from nets.ops.modules.ms_deform_attn import MSDeformAttn

        self.d_model = cfg['deform_attn_cfg']['d_model']
        self.n_heads = cfg['deform_attn_cfg']['n_heads']
        self.n_points = cfg['deform_attn_cfg']['n_points']
        self.expand_to_256 = cfg['deform_attn_cfg']['expand_to_256']
        self.fusion_strategy = cfg['deform_attn_cfg']['fusion_strategy']
        self.elevation_reduction = cfg['deform_attn_cfg']['elevation_reduction']
        self.use_pos_embeddings = cfg['deform_attn_cfg']['positional_embedding']
        self.weighting_network_type = cfg['deform_attn_cfg']['weighting_network']
        self.n_levels = 1 
        self.cfg = cfg
        self.radar_parameters = cfg['radar_parameters']
        self.dict_cam_calib = get_dict_cam_calib_from_yml(cfg['camcalib_path'])

        if cfg['cam_config']['cam_neck_output'] == '180x320':
            self.padding = False
        elif cfg['cam_config']['cam_neck_output'] == '256x112' or cfg['cam_config']['cam_neck_output'] == '224x224':
            self.padding = True
        else:
            raise ValueError("Invalid cam_neck_output configuration.")
        
        # Learnable positional embeddings for radar features
        self.r_bins, self.a_bins, self.e_bins = 256, 112, 10
        if self.use_pos_embeddings == 'learned':
            self.pos_embed = nn.Parameter(torch.randn(1, self.d_model, self.r_bins, self.a_bins, self.e_bins))
        if cfg['cam_config']['cam_neck']:
            self.cam_neck = DilatedResidualNeck(in_channels=128, dilation=(1, 2, 3))
        if cfg['cam_config']['rad_neck']:
            self.rad_neck = DilatedResidualNeck(in_channels=128, dilation=(1, 2, 3))

        self.deform_attn = MSDeformAttn(
            d_model=self.d_model, 
            n_levels=self.n_levels, 
            n_heads=self.n_heads, 
            n_points=self.n_points
        )

        if self.fusion_strategy == 'gated_fusion':
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, 1),
                nn.Sigmoid()
            )
        elif self.fusion_strategy == 'squeeze_excitation':
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(self.d_model, self.d_model // 4, 1),
                nn.ReLU(),
                nn.Conv1d(self.d_model // 4, self.d_model, 1),
                nn.Sigmoid()
            )
        elif self.fusion_strategy == 'dual_path_adaptive_fusion':
            # Memory-efficient: compute weights from sum instead of concat
            self.fusion_weights = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Linear(self.d_model // 2, 2),
                nn.Softmax(dim=-1)  # Normalized weights for radar and camera
            )
        elif self.fusion_strategy == 'cross_modal_self_attention':
            # Memory-efficient channel attention without LayerNorm
            # (LayerNorm creates large temp buffers with 286720 tokens)
            self.channel_attn = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 4),
                nn.ReLU(),
                nn.Linear(self.d_model // 4, self.d_model),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(0.1)
        self.output_proj = nn.Linear(self.d_model, self.d_model * 2)

        if self.elevation_reduction == 'mean':
            self.elevation_reduction_layer = None # for now kept for downwards compatibility, not used in forward pass
        elif self.elevation_reduction == 'single_1d_conv':
            self.elevation_reduction_layer = nn.Sequential(
                nn.Conv3d(self.d_model, self.d_model, kernel_size=(1, 1, self.e_bins)),
                nn.BatchNorm3d(self.d_model),
                nn.ReLU(),
            )
        else:
            raise ValueError("Invalid elevation reduction method.")

        if self.weighting_network_type == 'None':
            self.use_weighting_network = False
        elif self.weighting_network_type == 'CNN':
            self.use_weighting_network = True
            self.weighting_network = nn.Sequential(
                # First conv block: maintain spatial dimensions
                nn.Conv2d(37, 32, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=32),
                nn.ReLU(inplace=True),
                
                # Second conv block: extract spatial features
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=64),
                nn.ReLU(inplace=True),
                
                # Third conv block: refine features
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=32),
                nn.ReLU(inplace=True),
                
                # Output projection: back to E channels
                nn.Conv2d(32, self.e_bins, kernel_size=1),
                # Note: Softmax is applied in forward pass, not here
            )
        
    def get_reference_points(self, device, cam_shape):
        H_cam, W_cam = cam_shape
        range_min, range_max, azimuth_min, azimuth_max, elevation_min, elevation_max = self.radar_parameters['range_min'], self.radar_parameters['range_max'], \
                                                                                           self.radar_parameters['azimuth_min'], self.radar_parameters['azimuth_max'], \
                                                                                           self.radar_parameters['elevation_min'], self.radar_parameters['elevation_max']
        d_range = (range_max - range_min) / self.r_bins
        d_azimuth = (azimuth_max - azimuth_min) / self.a_bins
        d_elevation = (elevation_max - elevation_min) / self.e_bins

        ref_range, ref_azimuth, ref_elevation = torch.meshgrid(
            torch.linspace(range_min + d_range/2, range_max - d_range/2, self.r_bins, dtype=torch.float32, device=device),
            torch.linspace(azimuth_min + d_azimuth/2, azimuth_max - d_azimuth/2, self.a_bins, dtype=torch.float32, device=device),
            torch.linspace(elevation_min + d_elevation/2, elevation_max - d_elevation/2, self.e_bins, dtype=torch.float32, device=device),
            indexing='ij'
        )

        # convert to radiants
        azimuth_rad = -torch.deg2rad(ref_azimuth)
        elevation_rad = torch.deg2rad(ref_elevation)

        # polar to cartesian
        x_cart = ref_range * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
        y_cart = ref_range * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
        z_cart = ref_range * torch.sin(elevation_rad)

        # Stack to create the final Cartesian grid
        ref_points_cartesian = torch.stack((x_cart, y_cart, z_cart), -1)  # (H, W, D, 3)

        # Transform from Radar ro camera coordinates
        ref_points_cartesian[..., 0] -= self.cfg['rad2cam_calib']['dx']
        ref_points_cartesian[..., 1] -= self.cfg['rad2cam_calib']['dy']
        ref_points_cartesian[..., 2] -= self.cfg['rad2cam_calib']['dz']

        # Flatten the grid to a list of points (N, 3) where N = H*W*D
        points_3d = ref_points_cartesian.reshape(-1, 3)

        # Convert to homogeneous coordinates (N, 4)
        ones = torch.ones((points_3d.shape[0], 1), device=device)
        points_3d_h = torch.cat([points_3d, ones], dim=1)

        # Convert numpy matrices to torch tensors
        img_size0, intrinsics0, distortion0, ldr2img0 = self.dict_cam_calib['front0']
        ldr2img_t = torch.from_numpy(ldr2img0).float().to(device)
        intrinsics_t = torch.from_numpy(intrinsics0).float().to(device)

        # Project points to camera coordinate system
        cam_points_h = ldr2img_t @ points_3d_h.T

        # Filter points that are behind the camera (z <= 0)
        depth = cam_points_h[2, :]
        valid_mask_behind = depth > 0

        # Apply intrinsics to project onto the image plane
        img_points_h = intrinsics_t @ cam_points_h

        # Perform perspective division (u,v,w) -> (u/w, v/w)
        img_points_uv = img_points_h[:2, :] / (img_points_h[2, :].clone() + 1e-6)

        # Transpose to get (N, 2)
        img_points_uv = img_points_uv.T

        img_x = img_points_uv[:, 0]
        img_y = img_points_uv[:, 1]
        valid_mask_within_frame = (img_x >= 280) & (img_x < (1280 - 280)) & (img_y >= 0) & (img_y < 720)

        # Normalize to [0, 1]
        img_points_uv[:, 0] = img_points_uv[:, 0] / W_cam
        img_points_uv[:, 1] = img_points_uv[:, 1] / H_cam

        # Create a mask for points outside the image boundaries
        valid_mask_bounds = (img_points_uv[:, 0] >= 0) & (img_points_uv[:, 0] <= 1) & \
                            (img_points_uv[:, 1] >= 0) & (img_points_uv[:, 1] <= 1)
        
        # Combine masks
        valid_mask = valid_mask_behind & valid_mask_bounds & valid_mask_within_frame

        # Clamp the points to be within [0, 1] to avoid sampling errors
        reference_points = torch.clamp(img_points_uv, 0, 1)

        return reference_points, valid_mask

    def forward(self, batch_dict):
        query_features = batch_dict['backbone_output']    # Radar features (B, C, H, W)
        value_features = batch_dict['cam_features']     # Camera features (B, C, H, W)
        if self.cfg['cam_config']['rad_neck']:
            query_features = self.rad_neck(query_features)
        if self.cfg['cam_config']['cam_neck']:
            value_features = self.cam_neck(value_features)
        
        B, C, H, W = query_features.shape
        device = query_features.device

        reference_points, valid_mask = self.get_reference_points(device, (720, 1280))

        if self.use_weighting_network:

            era_dra = batch_dict['rdr_era_dra']
            era = era_dra[:, 64:, :, :] # (B, 1, 37, H, W)
            elevation_logits = self.weighting_network(era)
            elevation_weights = torch.softmax(elevation_logits, dim=1)

        if self.padding:
            #zero pad the value features from 224x224 to 224x398
            padding = (87, 87, 0, 0)  # (left, right, top, bottom)
            padding_mask = torch.zeros(B, value_features.shape[2], value_features.shape[3], device=value_features.device, dtype=torch.bool)
            padding_mask = torch.nn.functional.pad(padding_mask, padding, mode='constant', value=1)
            value_features = torch.nn.functional.pad(value_features, padding, mode='constant', value=0)
            padding_mask = padding_mask.flatten(1)  # (B, H*W, 1)

        else:
            padding_mask = None
            
        B, C_cam, H_cam, W_cam = value_features.shape
        reference_points = reference_points.unsqueeze(1).repeat(B, 1, 1, 1)

        full_residual_query = query_features.clone()
        # duplicate query features self.e_bins times
        query_features = query_features.unsqueeze(-1).repeat(1, 1, 1, 1, self.e_bins) # ([1, 128, 256, 112, 10])

        if self.use_weighting_network:
            elevation_weights = elevation_weights.permute(0, 2, 3, 1).unsqueeze(1) # (B, 1, H, W, E)
            query_features = query_features * elevation_weights

        if self.use_pos_embeddings == 'learned':
            query_features = query_features + self.pos_embed * 0.1

        # Query is from radar features
        query = query_features.flatten(2).transpose(1, 2) # (B, bev_r*bev_a*bev_e, C)
        query_residual = query.clone()

        # Value is from camera features
        value = value_features.flatten(2).transpose(1, 2) # (B, 224*224, C)

        # Mask out invalid queries. This prevents them from contributing to the gradient.
        query = query * valid_mask.view(1, -1, 1).float()

        # The attention module samples from one level (camera features) with shape (H_cam, W_cam)
        input_spatial_shapes = torch.as_tensor([[H_cam, W_cam]], dtype=torch.long, device=device)
        input_level_start_index = torch.as_tensor([0], dtype=torch.long, device=device)

        fused_query = self.deform_attn(
            query, 
            reference_points,
            value, 
            input_spatial_shapes, 
            input_level_start_index,
            input_padding_mask=padding_mask
        )

        fused_query = fused_query.transpose(1, 2).view(B, C, self.r_bins, self.a_bins, self.e_bins)

        if self.elevation_reduction == 'mean': # kept for downwards compatibility with previous experiments
            fused_query = torch.mean(fused_query, dim=4) # (B, C, H, W)
        else:
            fused_query = self.elevation_reduction_layer(fused_query).squeeze(4) # (B, C, H, W)


        if self.fusion_strategy == 'residual':
            fused_query = self.dropout(fused_query) + full_residual_query
        elif self.fusion_strategy == 'force_camera':
            fused_query = self.dropout(fused_query)
        elif self.fusion_strategy == 'force_radar':
            fused_query = full_residual_query
        elif self.fusion_strategy == 'gated_fusion':
            fused_query = fused_query.permute(0,2,3,1)
            full_residual_query = full_residual_query.permute(0,2,3,1)
            gate = self.fusion_gate(torch.cat([full_residual_query, fused_query], dim=-1))  # (B, N, 1)
            fused_query = gate * fused_query + (1 - gate) * full_residual_query
            fused_query = self.dropout(fused_query)
            fused_query = fused_query.permute(0,3,1,2)
        elif self.fusion_strategy == 'squeeze_excitation':
            cam_weights = self.channel_attention(fused_query.transpose(1, 2)).transpose(1, 2)
            fused_query = query_residual + fused_query * cam_weights
            fused_query = self.dropout(fused_query)
        elif self.fusion_strategy == 'dual_path_adaptive_fusion':
            weights = self.fusion_weights(query_residual + fused_query)  # (B, N, 2)
            fused_query = (weights[..., 0:1] * query_residual + 
                        weights[..., 1:2] * fused_query)
            fused_query = self.dropout(fused_query)
        elif self.fusion_strategy == 'cross_modal_self_attention':
            radar_pool = query_residual.mean(dim=1, keepdim=True)  # (B, 1, C)
            camera_pool = fused_query.mean(dim=1, keepdim=True)  # (B, 1, C)
            combined = radar_pool + camera_pool  # (B, 1, C)
            attn_weights = self.channel_attn(combined)  # (B, 1, C)
            fused_query = query_residual + fused_query * attn_weights
            fused_query = self.dropout(fused_query)

        # Reshape and project output
        fused_features = fused_query
    
        if self.expand_to_256:
            # Project to match the channel dimension of simple concatenation
            fused_features = fused_features.permute(0, 2, 3, 1) # (B, H, W, C)
            fused_features = self.output_proj(fused_features)   # (B, H, W, 2*C)
            fused_features = fused_features.permute(0, 3, 1, 2) # (B, 2*C, H, W)
        batch_dict['backbone_output'] = fused_features
        return batch_dict
    
if __name__ == "__main__":
    number = 112

    # Your selected code
    while (number % 8) != 0:
        number += 1

    # Print the result
    print(number)