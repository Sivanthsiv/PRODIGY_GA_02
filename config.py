"""
config.py — Centralised configuration for the image generation project.

All tuneable parameters live here so that the rest of the codebase
stays clean and easy to modify.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env if it exists ───────────────────────────────────
load_dotenv(override=True)

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "outputs")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Settings ───────────────────────────────────────────
# Model ID on HuggingFace Hub.
# Use a non-gated default so first-time runs work without HF auth.
MODEL_ID: str = os.getenv("MODEL_ID", "runwayml/stable-diffusion-v1-5")

# HuggingFace token (needed for gated / private models)
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# ── Generation Defaults ─────────────────────────────────────
# CPU-friendly fast defaults (can still be overridden from CLI/UI/env).
DEFAULT_NUM_INFERENCE_STEPS: int = int(os.getenv("DEFAULT_STEPS", "12"))
DEFAULT_GUIDANCE_SCALE: float = float(os.getenv("DEFAULT_GUIDANCE", "6.5"))
DEFAULT_WIDTH: int = int(os.getenv("DEFAULT_WIDTH", "384"))
DEFAULT_HEIGHT: int = int(os.getenv("DEFAULT_HEIGHT", "384"))
DEFAULT_NEGATIVE_PROMPT: str = (
    "blurry, bad anatomy, bad hands, cropped, worst quality, "
    "low quality, watermark, text, deformed, disfigured"
)

# ── Device ───────────────────────────────────────────────────
import torch

def get_device() -> str:
    """Return the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"           # Apple Silicon GPU
    return "cpu"

DEVICE: str = get_device()

