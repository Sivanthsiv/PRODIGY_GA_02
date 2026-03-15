"""
pipeline.py — Load the Stable Diffusion pipeline and expose a
simple `generate_image()` function.

This module abstracts all HuggingFace Diffusers boiler-plate so that
other parts of the project (CLI, Gradio app) only call one function.
"""

from __future__ import annotations

import traceback
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

from config import (
    MODEL_ID,
    HF_TOKEN,
    DEVICE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
)

# ── Module-level cache ───────────────────────────────────────
_pipeline: StableDiffusionPipeline | None = None


def load_pipeline(model_id: str | None = None) -> StableDiffusionPipeline:
    """
    Load (or return cached) Stable Diffusion pipeline.

    The pipeline is loaded with:
    • float16 precision on CUDA for speed (float32 on CPU / MPS)
    • DPM-Solver++ scheduler for faster, high-quality sampling
    • Safety checker disabled for non-production / research use

    Returns
    -------
    StableDiffusionPipeline
    """
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    # Resolve model at runtime so stale module defaults do not persist across edits.
    resolved_model_id = model_id or os.getenv("MODEL_ID") or MODEL_ID
    has_real_token = bool(HF_TOKEN and HF_TOKEN != "hf_your_token_here")
    if (
        resolved_model_id == "stabilityai/stable-diffusion-2-1"
        and not has_real_token
    ):
        print(
            "⚠️ MODEL_ID points to a gated model without a valid HF token. "
            "Falling back to 'runwayml/stable-diffusion-v1-5'."
        )
        resolved_model_id = "runwayml/stable-diffusion-v1-5"

    print(f"⏳ Loading model '{resolved_model_id}' on [{DEVICE}] …")

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    # Build kwargs — use explicit auth behavior.
    # token=False prevents accidental use of stale cached HuggingFace credentials.
    load_kwargs: dict = dict(
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        token=False,
    )
    if HF_TOKEN and HF_TOKEN != "hf_your_token_here":
        load_kwargs["token"] = HF_TOKEN

    try:
        pipe = StableDiffusionPipeline.from_pretrained(resolved_model_id, **load_kwargs)
    except Exception as exc:
        print(f"❌ Failed to load model: {exc}")
        traceback.print_exc()
        raise RuntimeError(
            f"Could not load model '{resolved_model_id}'. "
            "Make sure you have internet access and enough disk space (~5 GB). "
            "If the model is gated, set a valid HF_TOKEN in your .env file."
        ) from exc

    # Swap to a faster scheduler (DPM-Solver++)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    pipe = pipe.to(DEVICE)

    # Memory optimisations — attention slicing works on all devices
    pipe.enable_attention_slicing()

    _pipeline = pipe
    print("✅ Model loaded successfully!\n")
    return _pipeline


def generate_image(
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int | None = None,
) -> Image.Image:
    """
    Generate a single image from a text prompt.

    Parameters
    ----------
    prompt : str
        The text description of the desired image.
    negative_prompt : str
        Things the model should *avoid* generating.
    num_inference_steps : int
        Denoising steps (higher → better quality, slower).
    guidance_scale : float
        Classifier-free guidance weight (higher → follows prompt more).
    width, height : int
        Output dimensions (must be divisible by 8).
    seed : int | None
        Reproducibility seed. ``None`` → random.

    Returns
    -------
    PIL.Image.Image
        The generated image.
    """
    pipe = load_pipeline()

    # Set up the random generator for reproducibility
    generator = None
    if seed is not None:
        # On CPU, the generator device must be "cpu"
        gen_device = "cpu" if DEVICE == "cpu" else DEVICE
        generator = torch.Generator(device=gen_device).manual_seed(seed)

    print(f"🎨 Generating image for prompt: \"{prompt}\"")
    print(f"   Steps: {num_inference_steps} | Guidance: {guidance_scale} | "
          f"Size: {width}×{height} | Seed: {seed or 'random'}")

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
    except Exception as exc:
        print(f"❌ Generation failed: {exc}")
        traceback.print_exc()
        raise RuntimeError(
            f"Image generation failed: {exc}. "
            "Try reducing image size (e.g. 256×256) or inference steps (e.g. 20) "
            "if you're running out of memory."
        ) from exc

    image: Image.Image = result.images[0]
    print("✅ Image generated!\n")
    return image
