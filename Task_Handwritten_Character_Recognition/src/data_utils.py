
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import datasets, transforms


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


# --------------------------------------------------
# 2. TRAINING IMAGE TRANSFORM
# --------------------------------------------------
def emnist_transform():
    """
    Convert EMNIST images to tensors.

    The transpose corrects the orientation of EMNIST images
    so they match the normal character display orientation.
    Normalization converts pixel values from [0, 1] to [-1, 1].
    """
    return transforms.Compose([
        transforms.Lambda(lambda image: ImageOps.exif_transpose(image)),
        transforms.ToTensor(),
        transforms.Lambda(lambda tensor: tensor.transpose(1, 2).contiguous()),
        transforms.Normalize((0.5,), (0.5,)),
    ])


# --------------------------------------------------
# 3. LOAD TRAIN, VALIDATION, AND TEST DATASETS
# --------------------------------------------------
def load_datasets():
    """
    Download EMNIST ByClass if needed.

    Training: 90% of the official training set
    Validation: 10% of the official training set
    Testing: official EMNIST test set
    """
    transform = emnist_transform()

    train_full = datasets.EMNIST(
        root=str(DATA_DIR),
        split="byclass",
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.EMNIST(
        root=str(DATA_DIR),
        split="byclass",
        train=False,
        download=True,
        transform=transform,
    )

    targets = np.asarray(train_full.targets)
    indices = np.arange(len(targets))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=42,
        stratify=targets,
    )

    train_set = Subset(train_full, train_idx.tolist())
    val_set = Subset(train_full, val_idx.tolist())

    classes = [str(label) for label in train_full.classes]

    return train_set, val_set, test_set, classes


# --------------------------------------------------
# 4. PREPROCESS USER IMAGE
# --------------------------------------------------
def image_to_tensor(image):
    """
    Prepare a handwritten image for the CNN.

    Input:
        PIL image with a dark character on a light background,
        or a light character on a dark background.

    Output:
        Tensor with shape [1, 1, 28, 28], normalized to [-1, 1].
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))

    # Convert to grayscale.
    image = image.convert("L")

    # Correct the background to dark and character to light,
    # matching the usual EMNIST foreground/background convention.
    grayscale = np.asarray(image, dtype=np.uint8)

    if grayscale.mean() > 127:
        image = ImageOps.invert(image)

    # Preserve grayscale stroke detail instead of hard-thresholding
    # every pixel to black or white.
    grayscale = np.asarray(image, dtype=np.uint8)

    # Find the character area using a threshold only for the crop.
    foreground = grayscale > 30

    if not np.any(foreground):
        raise ValueError(
            "No character detected. Draw a character or upload a clearer image."
        )

    ys, xs = np.where(foreground)

    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1

    image = image.crop((left, top, right, bottom))

    # Fit the character within a 20 × 20 region, preserving aspect ratio.
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center it on a 28 × 28 dark canvas.
    canvas = Image.new("L", (28, 28), color=0)

    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2

    canvas.paste(image, (x_offset, y_offset))

    # Apply the same normalization as training.
    tensor = transforms.ToTensor()(canvas)
    tensor = (tensor - 0.5) / 0.5

    # Add batch dimension: [1, 1, 28, 28].
    return tensor.unsqueeze(0)
