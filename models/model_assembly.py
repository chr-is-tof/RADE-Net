import torch
import torch.nn as nn
from transformers import AutoImageProcessor
from PIL import Image
from os import path as os_path
import numpy as np

from .backbone.backbone_assembly import build_backbone
from .neck.neck_assembly import build_neck
from .head.head_assembly import build_head
from .fusion.fusion_assembly import (
    build_resnet_backbone,
    build_image_backbone,
    build_image_neck,
    build_fusion
)


class RadarOnly(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.backbone_type = model_cfg['rad_backbone']['type']
        self.data_padding = model_cfg['data_padding']

        self.backbone = build_backbone(model_cfg['rad_backbone'])
        self.neck = build_neck(model_cfg['rad_neck'])
        self.head = build_head(model_cfg['rad_head'])

        backbone_parameter = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        neck_parameter = sum(p.numel() for p in self.neck.parameters() if p.requires_grad)
        head_parameter = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        
        print(f"Backbone paramter: {backbone_parameter}")
        print(f"Neck paramter:     {neck_parameter}")
        print(f"Head paramter:     {head_parameter}")
        print(f"Total parameters:  {backbone_parameter + neck_parameter + head_parameter}")
        
    def forward(self, batch_dict):
        if self.data_padding and not self.backbone_type == 'UNetV2':
            # Pad the input to have width 112 instead of 107
            batch_dict['rdr_era_dra'] = nn.functional.pad(batch_dict['rdr_era_dra'], (0, 5, 0, 0))
        
        batch_dict = self.backbone(batch_dict)
        batch_dict['neck_output'] = self.neck(batch_dict['backbone_output'])
        batch_dict = self.head(batch_dict)
        
        return batch_dict
    

class RadarCamera(torch.nn.Module):
    def __init__(self, model_cfg : dict):
        super().__init__()
        self.double_size = model_cfg['cam_config']['double_size']
        self.backbone_type = model_cfg['rad_backbone']['type']
        self.data_padding = model_cfg['data_padding']

        self.rad_backbone = build_backbone(model_cfg['rad_backbone'])
        self.rad_neck = build_neck(model_cfg['rad_neck'])
        self.head = build_head(model_cfg['rad_head'])

        self.image_backbone_type = model_cfg['image_backbone']

        if self.image_backbone_type == 'Dinov3':
            local_model_path = model_cfg["image_backbone_cfg"]['model_name']
            self.image_processor = AutoImageProcessor.from_pretrained(local_model_path)
            self.image_backbone = build_image_backbone(model_cfg['image_backbone_cfg'], local_model_path)
            self.image_neck = build_image_neck(model_cfg['image_backbone'], model_cfg['image_neck'])
            image_neck_parameter = sum(p.numel() for p in self.image_neck.parameters() if p.requires_grad)
        elif self.image_backbone_type == 'ResNet':
            self.image_backbone = build_resnet_backbone(model_cfg['image_backbone_cfg'])
            image_neck_parameter = 0
        
        self.fusion = build_fusion(model_cfg)
        
        rad_backbone_parameter = sum(p.numel() for p in self.rad_backbone.parameters() if p.requires_grad)
        image_backbone_parameter = sum(p.numel() for p in self.image_backbone.parameters() if p.requires_grad)
        fusion_parameters = sum(p.numel() for p in self.fusion.parameters() if p.requires_grad)
        rad_neck_parameter = sum(p.numel() for p in self.rad_neck.parameters() if p.requires_grad)
        
        head_parameter = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        print(f"Rad backbone paramter:   {rad_backbone_parameter}")
        print(f"Image backbone paramter: {image_backbone_parameter}")
        print(f"Fusion paramter:         {fusion_parameters}")
        print(f"Rad neck paramter:       {rad_neck_parameter}")
        print(f"Image neck paramter:     {image_neck_parameter}")
        print(f"Head paramter:           {head_parameter}")
        total_parameters = (
            rad_backbone_parameter + image_backbone_parameter + fusion_parameters +
            rad_neck_parameter + image_neck_parameter + head_parameter
        )
        print(f"Total parameters:        {total_parameters}")

    def forward(self, batch_dict):
        if self.data_padding and not self.backbone_type == 'UNetV2':
            # Pad the input to have width 112 instead of 107
            batch_dict['rdr_era_dra'] = nn.functional.pad(batch_dict['rdr_era_dra'], (0, 5, 0, 0))

        cam_paths = batch_dict['cam_paths']
        cam_images = []
        for idx in range(len(cam_paths)):
            if self.image_backbone_type == 'Dinov3':
                cam_image  = Image.open(cam_paths[idx])
            elif self.image_backbone_type == 'ResNet':
                cam_image  = Image.open(cam_paths[idx]).convert('RGB')
            width, height = cam_image.size
            cam_image = cam_image.crop((0, 0, width // 2, height))
            if self.double_size:
                width, height = cam_image.size
                cam_image = cam_image.resize((width * 2, height * 2), Image.LANCZOS)
            cam_images.append(cam_image)

        device = next(self.rad_backbone.parameters()).device
        
        if self.image_backbone_type == 'ResNet':
            cam_images_tensor = torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in cam_images]).float().to(device) / 255.0
            image_features = self.image_backbone(cam_images_tensor)
            batch_dict['cam_features'] = image_features
        elif self.image_backbone_type == 'Dinov3':

            # Process the whole batch of images at once
            inputs = self.image_processor(images=cam_images, return_tensors="pt", do_resize=True, size={'shortest_edge': 720}, do_center_crop=True, crop_size={'height': 720, 'width': 1280}).to(device)

            # Get model outputs in a batch
            with torch.no_grad():
                outputs = self.image_backbone(**inputs)
        
            # Extract the CLS token feature [batch_size, 1, feature_dim]
            image_features = outputs.last_hidden_state
            batch_dict['cam_features'] = image_features
            batch_dict = self.image_neck(batch_dict)
        # Forward pass through the backbone
        batch_dict = self.rad_backbone(batch_dict)
        batch_dict = self.fusion(batch_dict)

        # Forward pass through the neck and head
        batch_dict['neck_output'] = self.rad_neck(batch_dict['backbone_output'])
        batch_dict = self.head(batch_dict)

        return batch_dict


class CameraOnly(torch.nn.Module):
    def __init__(self, model_cfg : dict):
        super().__init__()
        self.double_size = model_cfg['cam_config']['double_size']
        self.rad_neck = build_neck(model_cfg['rad_neck'])
        self.data_padding = model_cfg['data_padding']
        self.image_backbone_type = model_cfg['image_backbone']
        
        self.head = build_head(model_cfg['rad_head'])

        if model_cfg["image_backbone"] == 'Dinov3':
            local_model_path = model_cfg["image_backbone_cfg"]['model_name']

        self.image_processor = AutoImageProcessor.from_pretrained(local_model_path)
        self.image_backbone = build_image_backbone(model_cfg['image_backbone_cfg'], local_model_path)
        self.image_neck = build_image_neck(model_cfg['image_backbone'], model_cfg['image_neck'])
        
        image_backbone_parameter = sum(p.numel() for p in self.image_backbone.parameters() if p.requires_grad)
        image_neck_parameter = sum(p.numel() for p in self.image_neck.parameters() if p.requires_grad)
        head_parameter = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(f"Image backbone paramter: {image_backbone_parameter}")
        print(f"Image neck paramter:     {image_neck_parameter}")
        print(f"Head paramter:           {head_parameter}")
        total_parameters = (
            image_backbone_parameter +
            image_neck_parameter + head_parameter
        )
        print(f"Total parameters:        {total_parameters}")

    def forward(self, batch_dict):

        cam_paths = batch_dict['cam_paths']
        cam_images = []
        for idx in range(len(cam_paths)):
            if self.image_backbone_type == 'Dinov3':
                cam_image  = Image.open(cam_paths[idx])
            elif self.image_backbone_type == 'ResNet':
                cam_image  = Image.open(cam_paths[idx]).convert('RGB')
            width, height = cam_image.size
            cam_image = cam_image.crop((0, 0, width // 2, height))
            if self.double_size:
                width, height = cam_image.size
                cam_image = cam_image.resize((width * 2, height * 2), Image.LANCZOS)
            cam_images.append(cam_image)

        device = next(self.image_backbone.parameters()).device
        
        # Process the whole batch of images at once
        inputs = self.image_processor(images=cam_images, return_tensors="pt", do_resize=True, size={'shortest_edge': 720}, do_center_crop=True, crop_size={'height': 720, 'width': 1280}).to(device)
        # Get model outputs in a batch
        with torch.no_grad():
            outputs = self.image_backbone(**inputs)
        
        # Extract the CLS token feature [batch_size, 1, feature_dim]
        image_features = outputs.last_hidden_state

        batch_dict['cam_features'] = image_features
        batch_dict = self.image_neck(batch_dict)

        batch_dict['neck_output'] = self.rad_neck(batch_dict['cam_features'])
        batch_dict = self.head(batch_dict)

        return batch_dict


def build_model(mode : str, approach : str, path_to_models : str, 
                path_to_cluster : str, 
                inf_model : str, model_cfg : dict):
    if mode not in ['train', 'test', 'visualize_model']:
        raise ValueError("Invalid mode. Must be 'train', 'test' or 'visualize_model'.")
    
    if approach == 'RadarOnly':
        model = RadarOnly(model_cfg)
    elif approach == 'RadarCamera':
        model = RadarCamera(model_cfg)
    elif approach == 'CameraOnly':
        model = CameraOnly(model_cfg)
    else:
        raise ValueError(f"Unsupported fusion type: {approach}. Must be 'RadarOnly', 'RadarCamera', 'CameraOnly'.")

    if mode == 'train':
        return model

    elif mode == 'test' or mode == 'visualize_model':
        state_dict_path = f"{path_to_models}/{inf_model}.pth"

        if path_to_cluster:
            state_dict_path = f"{path_to_cluster}/{state_dict_path}"

        if not os_path.exists(state_dict_path):
            raise FileNotFoundError(f"State dict file not found: {state_dict_path}")

        # Load the state dict
        state_dict = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded state dict from {state_dict_path}")
        
        return model
    

