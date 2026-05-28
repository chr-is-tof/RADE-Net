from pathlib import Path

from PIL import Image
from torchvision import transforms


# Standard preprocessing pipeline for camera images fed into the vision backbone.
# Crops the image to its left half, then # resizes and center-crops to 224x224
_PREPROCESS = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])


def preprocess_camera_image(img_path: str | Path) -> Image.Image:
    """Load a front-camera image and apply the standard preprocessing pipeline.

    Steps:
        1. Crop to left half of the frame.
        2. Resize shortest edge to 224, then center-crop to 224x224.

    Args:
        img_path: Path to the raw camera image (.png / .jpg).

    Returns:
        Preprocessed PIL image ready for the vision backbone.
    """
    image = Image.open(img_path)
    width, height = image.size
    image = image.crop((0, 0, width // 2, height))
    return _PREPROCESS(image)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python camera_preprocess.py <img_path> <save_path>")
        sys.exit(1)

    img_path  = Path(sys.argv[1])
    save_path = Path(sys.argv[2])
    save_path.mkdir(parents=True, exist_ok=True)

    result = preprocess_camera_image(img_path)
    out_file = save_path / (img_path.stem + "_preprocessed.png")
    result.save(out_file)
    print(f"Saved preprocessed image to '{out_file}'")