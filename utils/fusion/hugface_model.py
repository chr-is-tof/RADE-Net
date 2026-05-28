# =============================================================================
# Download DINOv3 weights from Hugging Face Hub
#
# Run once to fetch and cache the model locally:
#   python utils/fusion/hugface_model.py
# =============================================================================

from transformers import AutoImageProcessor, AutoModel

HF_MODEL_NAME  = "facebook/dinov3-vits16-pretrain-lvd1689m"
SAVE_DIRECTORY = "vision_models/dinov3_local"


if __name__ == "__main__":
    print(f"Downloading '{HF_MODEL_NAME}' from Hugging Face Hub...")

    processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    model = AutoModel.from_pretrained(HF_MODEL_NAME)

    processor.save_pretrained(SAVE_DIRECTORY)
    model.save_pretrained(SAVE_DIRECTORY)

    print(f"Model and processor saved to '{SAVE_DIRECTORY}'")