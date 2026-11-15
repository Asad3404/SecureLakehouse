"""
Small helpers shared by every ThreatWeigh script:
  * load_config()  - read config.yaml and apply a compute profile
  * set_seed()     - make Python / NumPy / PyTorch random numbers reproducible
  * get_device()   - use the GPU when there is one, otherwise the CPU
  * project paths  - everything is resolved relative to the project root
"""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import yaml

# The project root is the folder that contains config.yaml (one level above src/).
ROOT = Path(__file__).resolve().parent.parent

# Hugging Face downloads: the "xet" transfer backend is not needed and is
# blocked on some networks, so we use the plain HTTP download path.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Silence a noisy (harmless) warning from the tokenizers library.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def path(p: str | Path) -> Path:
    """Turn a path from config.yaml (relative to the project root) into an absolute Path."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(profile: str | None = None, config_file: str | Path = "config.yaml") -> dict:
    """
    Read config.yaml. If a profile name is given ("full", "budget", "quick"),
    its values are copied into cfg["run"] so that scripts only need to look at
    cfg["run"]["train_n"], cfg["run"]["epochs"], ...
    """
    with open(path(config_file)) as f:
        cfg = yaml.safe_load(f)

    run = {
        "profile": profile or "full",
        "train_n": None,
        "val_n": None,
        "test_n": None,
        "epochs": cfg["train"]["epochs"],
        "patience": cfg["train"]["patience"],
    }
    if profile:
        if profile not in cfg["profiles"]:
            raise ValueError(f"Unknown profile {profile!r}; choose from {list(cfg['profiles'])}")
        run.update(copy.deepcopy(cfg["profiles"][profile]))
    cfg["run"] = run
    return cfg


def set_seed(seed: int) -> None:
    """Seed every random number generator we use, so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the data / TF-IDF stages
        pass


def get_device():
    """Return the best available device: CUDA GPU, Apple MPS, or CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def metric_names(cfg: dict) -> list[str]:
    """The 8 metric names in a fixed order: av, ac, pr, ui, s, c, i, a."""
    return list(cfg["metrics"].keys())
