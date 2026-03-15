"""
colab_notebook.py — Self-contained script designed to run in Google Colab.

How to use
----------
1. Open Google Colab (https://colab.research.google.com)
2. Create a new notebook
3. Set Runtime → Change runtime type → GPU (T4 is fine)
4. Copy-paste this entire file into a single code cell and run it.

The script will:
  • Install dependencies
  • Load Stable Diffusion
  • Generate images from sample prompts
  • Display them inline
"""

# ── Step 1: Install dependencies ─────────────────────────────
# fmt: off
# Uncomment the next line when running in Colab
# !pip install -q diffusers transformers accelerate torch safetensors Pillow

# fmt: on

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

# For displaying images in Colab
try:
    from IPython.display import display
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# ── Step 2: Configuration ────────────────────────────────────
MODEL_ID = "stabilityai/stable-diffusion-2-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"🖥️  Device : {DEVICE}")
print(f"📦  Model  : {MODEL_ID}")
print(f"🔢  Dtype  : {DTYPE}\n")


# ── Step 3: Load the pipeline ────────────────────────────────
print("⏳ Loading Stable Diffusion pipeline …")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,
    requires_safety_checker=False,
)

# Use faster scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)

if DEVICE == "cuda":
    pipe.enable_attention_slicing()

print("✅ Pipeline loaded!\n")


# ── Step 4: Define generation function ───────────────────────
def generate(
    prompt: str,
    negative_prompt: str = "blurry, bad anatomy, low quality, watermark",
    steps: int = 50,
    guidance: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: int | None = None,
) -> Image.Image:
    """Generate one image and return it."""
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )
    return result.images[0]


# ── Step 5: Generate images from sample prompts ──────────────
sample_prompts = [
    {
        "prompt": (
            "A majestic mountain landscape at golden hour with a mirror-like "
            "lake, snow-capped peaks, wildflowers, photorealistic, 8K"
        ),
        "seed": 42,
    },
    {
        "prompt": (
            "A futuristic cyberpunk city at night with neon lights reflecting "
            "on wet streets, flying cars, cinematic dramatic lighting"
        ),
        "seed": 123,
    },
    {
        "prompt": (
            "An astronaut playing guitar on the surface of the Moon, Earth "
            "in the background, digital art, trending on ArtStation"
        ),
        "seed": 7,
    },
]

print("🎨 Generating images …\n")

for i, item in enumerate(sample_prompts, 1):
    print(f"── Image {i}/{len(sample_prompts)} ──")
    print(f"   Prompt: {item['prompt'][:80]}…")

    image = generate(prompt=item["prompt"], seed=item["seed"])

    # Save + display
    filename = f"generated_{i}.png"
    image.save(filename)
    print(f"   💾 Saved as {filename}\n")

    if IN_COLAB:
        display(image)
    else:
        image.show()

print("✅ All done!")


# ── Bonus: Interactive prompt ────────────────────────────────
# Uncomment the block below to enter your own prompts interactively:
#
# while True:
#     user_prompt = input("\n✏️  Enter a prompt (or 'quit'): ")
#     if user_prompt.lower() in ("quit", "exit", "q"):
#         break
#     img = generate(user_prompt, seed=None)
#     img.save("custom_output.png")
#     if IN_COLAB:
#         display(img)
#     else:
#         img.show()
