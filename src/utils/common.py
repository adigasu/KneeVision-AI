"""
RSNA Knee Abnormality Detection - Utility functions for reproducibility,
configuration management, and logging.
"""

import os
import random
from typing import Any, Dict
import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
