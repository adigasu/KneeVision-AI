"""
RSNA Knee Abnormality Detection - Medical Image Augmentations & Transforms.
Applies consistent spatial and intensity augmentations across 2.5D slice stacks.
"""

from typing import Tuple
import numpy as np
import albumentations as A


def get_training_transforms(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """Returns spatial and intensity augmentations for multi-slice knee MRI volumes."""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.CoarseDropout(max_holes=6, max_height=24, max_width=24, min_holes=1, fill_value=0, p=0.3),
    ])


def get_validation_transforms(image_size: Tuple[int, int] = (256, 256)) -> A.Compose:
    """Returns deterministic resizing for validation / test inference."""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
    ])
