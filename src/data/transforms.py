"""
RSNA Knee Abnormality Detection - Medical Image Augmentations & Transforms.
Applies consistent spatial and intensity augmentations across 2.5D slice stacks.
"""

from typing import Tuple, Union
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_transforms(image_size: Union[int, Tuple[int, int]] = (256, 256)) -> A.Compose:
    """Returns spatial and intensity augmentations for multi-slice knee MRI volumes."""
    if isinstance(image_size, int):
        h, w = image_size, image_size
    else:
        h, w = image_size[0], image_size[1]

    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(-0.06, 0.06), p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(8, 24), hole_width_range=(8, 24), fill=0, p=0.3),
        ToTensorV2(),
    ])


def get_validation_transforms(image_size: Union[int, Tuple[int, int]] = (256, 256)) -> A.Compose:
    """Returns deterministic resizing for validation / test inference."""
    if isinstance(image_size, int):
        h, w = image_size, image_size
    else:
        h, w = image_size[0], image_size[1]

    return A.Compose([
        A.Resize(h, w),
        ToTensorV2(),
    ])


# Convenience aliases
get_train_transforms = get_training_transforms
get_valid_transforms = get_validation_transforms
