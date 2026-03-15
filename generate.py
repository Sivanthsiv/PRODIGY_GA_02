"""
generate.py — Command-line entry point for image generation.

Usage
-----
    # Basic usage
    python generate.py "a beautiful sunset over mountains, oil painting"

    # With options
    python generate.py "cyberpunk city at night" --steps 75 --guidance 9.0 --seed 42

    # Multiple images with different seeds
    python generate.py "a cat astronaut" --num-images 4

Run ``python generate.py --help`` for full options.
"""

import argparse
import time
from pathlib import Path

from config import (
    OUTPUT_DIR,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
)
from pipeline import generate_image


def slugify(text: str, max_len: int = 50) -> str:
    """Turn a prompt into a filesystem-safe slug."""
    slug = "".join(c if c.isalnum() or c == " " else "" for c in text)
    slug = slug.strip().replace(" ", "_")[:max_len]
    return slug or "image"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts using Stable Diffusion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="Text description of the image to generate.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="What the model should avoid (default: common quality issues).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help=f"Number of denoising steps (default: {DEFAULT_NUM_INFERENCE_STEPS}).",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help=f"Guidance scale (default: {DEFAULT_GUIDANCE_SCALE}).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Image width in pixels (default: {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Image height in pixels (default: {DEFAULT_HEIGHT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Directory to save images (default: {OUTPUT_DIR}).",
    )

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_images):
        seed = args.seed + i if args.seed is not None else None

        start = time.time()
        image = generate_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            width=args.width,
            height=args.height,
            seed=seed,
        )
        elapsed = time.time() - start

        # Build filename: <slug>_<seed>_<timestamp>.png
        slug = slugify(args.prompt)
        ts = int(time.time())
        filename = f"{slug}_{seed or 'rand'}_{ts}.png"
        filepath = out / filename

        image.save(filepath)
        print(f"💾 Saved → {filepath}  ({elapsed:.1f}s)\n")


if __name__ == "__main__":
    main()
