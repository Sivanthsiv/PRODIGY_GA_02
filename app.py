"""
app.py — Gradio web interface for text-to-image generation.

Launch with:
    python app.py

Then open the URL printed in the terminal (usually http://127.0.0.1:7860).
"""

import time
import gradio as gr
from PIL import Image

from config import (
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    OUTPUT_DIR,
)
from pipeline import generate_image


# ── Sample prompts for the UI ────────────────────────────────
EXAMPLE_PROMPTS = [
    ["A mystical forest with glowing mushrooms and fireflies at twilight, fantasy art, highly detailed"],
    ["A futuristic cyberpunk city at night with neon lights reflecting on wet streets, cinematic lighting"],
    ["An astronaut riding a horse on Mars, digital art, trending on ArtStation"],
    ["A cozy coffee shop interior on a rainy day, warm lighting, Studio Ghibli style, watercolor"],
    ["Portrait of a wise old wizard with a long white beard, intricate robes, dramatic lighting, oil painting"],
    ["A steampunk mechanical owl perched on a gear-shaped tree, brass and copper tones, detailed"],
    ["Underwater ancient city with bioluminescent sea creatures, ethereal blue lighting, concept art"],
    ["A Japanese zen garden in autumn with maple leaves falling, peaceful, soft golden hour light"],
]


def generate_ui(
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: int,
) -> tuple[Image.Image | None, str]:
    """Callback wired to the Gradio interface."""
    if not prompt.strip():
        raise gr.Error("Please enter a prompt!")

    actual_seed = seed if seed >= 0 else None

    try:
        start = time.time()
        image = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            seed=actual_seed,
        )
        elapsed = time.time() - start
    except Exception as exc:
        return None, f"Error: {exc}"

    # Auto-save the image
    ts = int(time.time())
    filepath = OUTPUT_DIR / f"gradio_{actual_seed or 'rand'}_{ts}.png"
    image.save(filepath)

    info = (
        f"⏱️ Generated in {elapsed:.1f}s\n"
        f"📐 Size: {width}×{height}\n"
        f"🔢 Steps: {steps} | Guidance: {guidance}\n"
        f"🎲 Seed: {actual_seed or 'random'}\n"
        f"💾 Saved to: {filepath}"
    )
    return image, info


# ── Build the Gradio interface ───────────────────────────────
with gr.Blocks(
    title="✨ AI Image Generator — Stable Diffusion",
) as demo:

    gr.Markdown(
        """
        # ✨ AI Image Generator
        ### Generate stunning images from text descriptions using **Stable Diffusion**
        ---
        """
    )

    with gr.Row():
        # ── Left column: inputs ──────────────────────────────
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="📝 Prompt",
                placeholder="Describe the image you want to create…",
                lines=3,
            )
            negative_input = gr.Textbox(
                label="🚫 Negative Prompt",
                value=DEFAULT_NEGATIVE_PROMPT,
                lines=2,
            )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                steps_slider = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=DEFAULT_NUM_INFERENCE_STEPS,
                    step=5,
                    label="Inference Steps",
                )
                guidance_slider = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=DEFAULT_GUIDANCE_SCALE,
                    step=0.5,
                    label="Guidance Scale",
                )
                with gr.Row():
                    width_slider = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=DEFAULT_WIDTH,
                        step=64,
                        label="Width",
                    )
                    height_slider = gr.Slider(
                        minimum=256,
                        maximum=1024,
                        value=DEFAULT_HEIGHT,
                        step=64,
                        label="Height",
                    )
                seed_input = gr.Number(
                    label="Seed (-1 = random)",
                    value=-1,
                    precision=0,
                )

            generate_btn = gr.Button(
                "🎨 Generate Image",
                variant="primary",
                elem_id="generate-btn",
            )

        # ── Right column: output ─────────────────────────────
        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")
            output_info = gr.Textbox(label="ℹ️ Generation Info", lines=5, interactive=False)

    # ── Example gallery ──────────────────────────────────────
    gr.Examples(
        examples=EXAMPLE_PROMPTS,
        inputs=[prompt_input],
        label="💡 Try these prompts",
    )

    # ── Wire the button ──────────────────────────────────────
    generate_btn.click(
        fn=generate_ui,
        inputs=[
            prompt_input,
            negative_input,
            steps_slider,
            guidance_slider,
            width_slider,
            height_slider,
            seed_input,
        ],
        outputs=[output_image, output_info],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,              # Set True to get a public link
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
        ),
        css="""
            .gradio-container { max-width: 960px !important; }
            #generate-btn { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                font-size: 1.1em !important;
                padding: 12px 24px !important;
            }
        """,
    )
