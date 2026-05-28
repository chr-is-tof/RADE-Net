import torch
from transformers import AutoModel
from torchvision.models import resnet50, ResNet50_Weights
from peft import get_peft_model, LoraConfig

from .radcam_nets import (
    RadCamConcatFusion,
    RadCamDefAttFusion,
    RadCamDefAttProjectionFusion,
    CamNeck
)

class FrozenResNetWithProjection(torch.nn.Module):
    """ResNet50 backbone + trainable 1x1 projection.

    stem, layer1, layer2, layer3 are frozen and run under no_grad to save memory.
    layer4 is unfrozen and fine-tuned (activations at 23x40 — memory-safe).
    Only the projection layer is always trainable.
    """
    def __init__(self, frozen_layers, layer4, projection):
        super().__init__()
        self.frozen_layers = frozen_layers
        self.layer4 = layer4
        self.projection = projection
        for param in self.frozen_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.frozen_layers(x)
        x = self.layer4(x)
        return self.projection(x)


def build_resnet_backbone(image_backbone_cfg):
    print(f"Building ResNet Backbone with config: {image_backbone_cfg}")
    local_weights_path = image_backbone_cfg['resnet50_weights_path']
    if local_weights_path:
        print(f"Loading ResNet50 weights from local path: {local_weights_path}")
        resnet_backbone = resnet50(weights=None)
        resnet_backbone.load_state_dict(torch.load(local_weights_path, map_location='cpu'))
    else:
        resnet_backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    # children(): conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    children = list(resnet_backbone.children())
    frozen_layers = torch.nn.Sequential(*children[:-3])   # stem → layer3, frozen
    layer4 = children[-3]                                  # layer4, trainable
    projection = torch.nn.Conv2d(2048, 128, kernel_size=1, bias=False)
    return FrozenResNetWithProjection(frozen_layers, layer4, projection)


def build_image_backbone(image_backbone_cfg, local_model_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    image_backbone = AutoModel.from_pretrained(local_model_path).to(device)
    image_backbone.eval()

    for param in image_backbone.parameters():
        param.requires_grad = False

    print(f"Use '{image_backbone_cfg['adapter_type']}' Adapter for Image Backbone: {image_backbone_cfg['use_adapter']}")

    if image_backbone_cfg['use_adapter']:
        if image_backbone_cfg['adapter_type'] == 'LoRA':
            peft_config = LoraConfig(
                r=image_backbone_cfg['lora_config']['r'],
                lora_alpha=image_backbone_cfg['lora_config']['lora_alpha'],
                target_modules=image_backbone_cfg['lora_config']['target_modules'],
                lora_dropout=image_backbone_cfg['lora_config']['lora_dropout'],
                bias=image_backbone_cfg['lora_config']['bias'],
                modules_to_save=image_backbone_cfg['lora_config']['modules_to_save'],
            )
        else:
            raise ValueError(f"Unknown adapter type: {image_backbone_cfg['adapter_type']}")
        image_backbone = get_peft_model(image_backbone, peft_config)

    return image_backbone


def build_image_neck(name : str, neck_cfg : dict):
    input_dim = neck_cfg['input_dim']
    output_channels = neck_cfg['output_dim']

    if name == 'Dinov3':
        return CamNeck(neck_cfg['cam_config'], input_dim=input_dim, output_channels=output_channels)


def build_fusion(model_cfg : dict):
    fusion_type = model_cfg['radcam_fusion_type']

    if fusion_type == 'concat':
        if not model_cfg['cam_config']['cam_neck_output'] == '256x112':
            raise ValueError("For 'concat' fusion, cam_neck_output must be '256x112'")
        # needs cam_config
        return RadCamConcatFusion(model_cfg)

    elif fusion_type == 'deformable_attention':
        # deform_attn_cfg and cam_config needed
        return RadCamDefAttFusion(model_cfg)

    elif fusion_type == 'def_att_project_fuse':
        # deform_attn_cfg and cam_config needed
        # radar_parameters, camlib_path, rad2cam_calib needed
        return RadCamDefAttProjectionFusion(model_cfg)