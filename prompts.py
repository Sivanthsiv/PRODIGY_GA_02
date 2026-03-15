"""
prompts.py — Curated sample prompts organised by category.

Each prompt includes the main prompt text, a suggested negative prompt,
and recommended generation parameters.

Usage
-----
    from prompts import SAMPLE_PROMPTS

    for p in SAMPLE_PROMPTS:
        print(p["category"], "—", p["prompt"][:60])
"""

SAMPLE_PROMPTS = [
    # ── Landscapes & Nature ──────────────────────────────────
    {
        "category": "Landscape",
        "prompt": (
            "A breathtaking mountain valley at sunrise with a crystal-clear "
            "lake reflecting snow-capped peaks, wildflowers in the foreground, "
            "golden hour lighting, photorealistic, 8K resolution"
        ),
        "negative_prompt": "blurry, low quality, watermark, text",
        "guidance_scale": 7.5,
        "steps": 50,
    },
    {
        "category": "Landscape",
        "prompt": (
            "An enchanted forest with towering ancient trees, shafts of "
            "sunlight piercing through the canopy, moss-covered stones, "
            "mystical atmosphere, trending on ArtStation, fantasy concept art"
        ),
        "negative_prompt": "blurry, low quality, watermark, text, ugly",
        "guidance_scale": 8.0,
        "steps": 50,
    },

    # ── Fantasy & Sci-Fi ─────────────────────────────────────
    {
        "category": "Fantasy",
        "prompt": (
            "A majestic dragon perched on a cliff overlooking a medieval city, "
            "epic scale, dramatic storm clouds, volumetric lighting, "
            "highly detailed scales, Greg Rutkowski style, digital painting"
        ),
        "negative_prompt": "blurry, bad anatomy, low quality, watermark",
        "guidance_scale": 9.0,
        "steps": 60,
    },
    {
        "category": "Sci-Fi",
        "prompt": (
            "A futuristic space station orbiting a gas giant planet, "
            "solar panels and antenna arrays, Earth visible in the distance, "
            "hard sci-fi, cinematic composition, ultra-detailed, 4K"
        ),
        "negative_prompt": "blurry, low quality, deformed, text",
        "guidance_scale": 7.5,
        "steps": 50,
    },

    # ── Portraits & Characters ───────────────────────────────
    {
        "category": "Portrait",
        "prompt": (
            "Portrait of a cyberpunk hacker with neon-lit hair and augmented "
            "reality visor, rain-soaked alley background, dramatic side lighting, "
            "cinematic color grading, hyper-realistic, 8K"
        ),
        "negative_prompt": "blurry, bad anatomy, extra fingers, deformed face",
        "guidance_scale": 8.5,
        "steps": 55,
    },
    {
        "category": "Portrait",
        "prompt": (
            "An elegant elven queen with flowing silver hair and intricate "
            "crown of vines and crystals, ethereal glow, soft bokeh background, "
            "Artgerm style, fantasy portrait, highly detailed"
        ),
        "negative_prompt": "blurry, bad anatomy, extra limbs, low quality",
        "guidance_scale": 8.0,
        "steps": 50,
    },

    # ── Architecture & Interiors ─────────────────────────────
    {
        "category": "Architecture",
        "prompt": (
            "A cozy reading nook inside a treehouse library, floor-to-ceiling "
            "bookshelves, warm fairy lights, a cup of tea on the windowsill, "
            "rain outside, Studio Ghibli aesthetics, watercolor illustration"
        ),
        "negative_prompt": "blurry, low quality, text, watermark",
        "guidance_scale": 7.0,
        "steps": 50,
    },
    {
        "category": "Architecture",
        "prompt": (
            "An abandoned Art Deco movie theater being reclaimed by nature, "
            "vines growing through cracked marble floors, sunlight streaming "
            "through broken stained-glass ceiling, photorealistic, ethereal"
        ),
        "negative_prompt": "blurry, ugly, low quality, watermark",
        "guidance_scale": 7.5,
        "steps": 50,
    },

    # ── Abstract & Artistic ──────────────────────────────────
    {
        "category": "Abstract",
        "prompt": (
            "An abstract visual of the concept of time as melting clocks in "
            "a surreal desert landscape, Salvador Dalí inspired, vivid colors, "
            "dreamlike atmosphere, oil painting, museum quality"
        ),
        "negative_prompt": "blurry, text, watermark, low resolution",
        "guidance_scale": 10.0,
        "steps": 60,
    },
    {
        "category": "Abstract",
        "prompt": (
            "Fractal geometry blossoming into a field of flowers, mathematical "
            "beauty meets nature, iridescent colors, macro photography style, "
            "octane render, 8K, stunning detail"
        ),
        "negative_prompt": "blurry, low quality, noise, text",
        "guidance_scale": 8.0,
        "steps": 50,
    },
]


# ── Quick-access list of just the prompt strings ─────────────
PROMPT_STRINGS = [p["prompt"] for p in SAMPLE_PROMPTS]


if __name__ == "__main__":
    # Pretty-print all prompts when run directly
    for i, p in enumerate(SAMPLE_PROMPTS, 1):
        print(f"\n{'═' * 60}")
        print(f"  [{p['category'].upper()}]  Prompt #{i}")
        print(f"{'═' * 60}")
        print(f"  {p['prompt']}\n")
        print(f"  Negative : {p['negative_prompt']}")
        print(f"  Guidance : {p['guidance_scale']}  |  Steps : {p['steps']}")
