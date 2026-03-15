# 🎨 Text-to-Image Generation with Stable Diffusion

> Generate stunning images from text descriptions using pre-trained **Stable Diffusion** models powered by HuggingFace Diffusers.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗_Diffusers-0.27+-yellow)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange?logo=gradio)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [How Diffusion Models Work](#-how-diffusion-models-work)
- [Model Comparison](#-model-comparison)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Prompt Engineering Guide](#-prompt-engineering-guide)
- [Running on Google Colab](#-running-on-google-colab)
- [Web Interface (Gradio)](#-web-interface-gradio)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🔭 Overview

This project demonstrates **text-to-image generation** — the task of creating images from natural-language descriptions. We use **Stable Diffusion**, a state-of-the-art latent diffusion model, through the HuggingFace `diffusers` library.

### What You Get

| Feature | Description |
|---------|-------------|
| **CLI Tool** | Generate images from the terminal with full parameter control |
| **Gradio Web UI** | Beautiful browser-based interface with sliders and examples |
| **Colab Support** | One-click notebook for GPU access on Google Colab |
| **Prompt Library** | 10+ curated prompts across categories |
| **Modular Code** | Clean separation of config, pipeline, and UI logic |

---

## 🧠 How Diffusion Models Work

### The Core Idea

Diffusion models generate images through a two-phase process:

```
TRAINING (Forward Diffusion)             GENERATION (Reverse Diffusion)
┌────────┐    add noise     ┌────────┐    ┌────────┐   remove noise   ┌────────┐
│ Clean  │ ──────────────►  │ Noisy  │    │ Random │ ──────────────►  │ Clean  │
│ Image  │   (many steps)   │ Image  │    │ Noise  │   (many steps)   │ Image  │
└────────┘                  └────────┘    └────────┘                  └────────┘
   The model LEARNS to reverse this process
```

1. **Forward Process (Training):** Gradually add Gaussian noise to real images until they become pure noise. The model learns to predict (and remove) this noise at each step.

2. **Reverse Process (Generation):** Start from random noise, then iteratively denoise it — guided by the text prompt — until a clean image emerges.

### Stable Diffusion Architecture

Stable Diffusion is a **Latent Diffusion Model (LDM)** — it operates in a compressed latent space rather than on raw pixels, which makes it much faster and more memory-efficient.

```
Text Prompt
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    CLIP       │     │    U-Net      │     │    VAE        │
│  Text Encoder │────►│   Denoiser    │────►│   Decoder     │────► Final Image
│              │     │  (iterative)  │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
  Encodes text         Removes noise        Decodes latent
  into embeddings      step by step         to pixel space
```

**Key Components:**
- **CLIP Text Encoder** — Converts your text prompt into a numerical representation (embeddings)
- **U-Net** — The neural network that predicts and removes noise at each step
- **VAE (Variational Autoencoder)** — Compresses images to/from latent space
- **Scheduler** — Controls the noise schedule (we use DPM-Solver++ for speed)

---

## ⚖️ Model Comparison

| Feature | **DALL·E** | **DALL·E Mini / Craiyon** | **Stable Diffusion** |
|---|---|---|---|
| **Developer** | OpenAI | Boris Dayma (community) | Stability AI + Runway |
| **Architecture** | Transformer (VQVAE) | Transformer (smaller) | Latent Diffusion (U-Net) |
| **Parameters** | 12B | ~400M | ~900M |
| **Open Source** | ❌ No | ✅ Yes | ✅ Yes |
| **API Access** | Paid API only | Free (web) | Free (local/API) |
| **Quality** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | Fast (API) | Slow | Fast (with GPU) |
| **GPU Needed** | No (cloud API) | No (cloud) | Yes (≥4 GB VRAM recommended) |
| **Customisation** | Limited | None | Full (fine-tune, LoRA, etc.) |
| **Best For** | Production apps | Quick prototyping | Research, learning, customisation |

> **Why Stable Diffusion?** It's fully open-source, runs locally, has a massive ecosystem of tools and fine-tuned models, and produces high-quality results — perfect for a portfolio project.

---

## 📁 Project Structure

```
Task 2/
├── config.py            # Configuration — model ID, paths, defaults
├── pipeline.py          # Core logic — load model & generate images
├── generate.py          # CLI entry point (argparse)
├── app.py               # Gradio web interface
├── prompts.py           # Curated sample prompts with parameters
├── colab_notebook.py    # Self-contained script for Google Colab
├── requirements.txt     # Python dependencies
├── .env.example         # Template environment variables
├── .gitignore           # Git ignore rules
├── README.md            # This file
└── outputs/             # Generated images (auto-created, git-ignored)
```

---

## 🛠️ Setup & Installation

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| **Python** | 3.9 | 3.10+ |
| **RAM** | 8 GB | 16 GB |
| **GPU VRAM** | 4 GB (float16) | 8+ GB |
| **Disk Space** | ~5 GB (model weights) | ~10 GB |
| **OS** | Windows / Linux / macOS | Any |

> 💡 **No GPU?** The code falls back to CPU (slower, ~2–5 min per image) or use Google Colab for free GPU access.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/text-to-image-generation.git
cd text-to-image-generation
```

### Step 2 — Create a Virtual Environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **PyTorch + CUDA:** If you have an NVIDIA GPU, install the CUDA-enabled version:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### Step 4 — Configure Environment (Optional)

```bash
cp .env.example .env
# Edit .env to set your HuggingFace token and model preferences
```

---

## 🚀 Usage

### Option 1: Command Line (CLI)

```bash
# Basic usage
python generate.py "a beautiful sunset over mountains, oil painting"

# With custom parameters
python generate.py "cyberpunk city at night, neon lights" \
    --steps 75 \
    --guidance 9.0 \
    --seed 42 \
    --width 768 \
    --height 512

# Generate multiple images
python generate.py "a cat wearing a space suit" --num-images 4 --seed 100
```

**CLI Arguments:**

| Argument | Default | Description |
|---|---|---|
| `prompt` | *(required)* | Text description of the image |
| `--negative-prompt` | Common quality issues | What to avoid |
| `--steps` | 50 | Denoising steps (10–100) |
| `--guidance` | 7.5 | Prompt adherence (1–20) |
| `--width` | 512 | Image width (divisible by 8) |
| `--height` | 512 | Image height (divisible by 8) |
| `--seed` | Random | Reproducibility seed |
| `--num-images` | 1 | Batch count |
| `--output-dir` | `outputs/` | Save directory |

### Option 2: Gradio Web Interface

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser. You'll see:
- A text box for your prompt
- Advanced settings (steps, guidance, size, seed)
- An example gallery to try pre-made prompts
- One-click generation with auto-save

---

## ✍️ Prompt Engineering Guide

The quality of your output depends heavily on **how you write the prompt**. Here are best practices:

### Prompt Structure

```
[Subject] + [Details] + [Style] + [Quality Modifiers]
```

### Examples: Good vs. Bad Prompts

| ❌ Vague Prompt | ✅ Engineered Prompt |
|---|---|
| "a cat" | "A fluffy Persian cat sitting on a velvet cushion, soft studio lighting, bokeh background, photorealistic, 8K" |
| "a city" | "A futuristic cyberpunk city at night, neon signs reflecting on wet streets, flying cars, cinematic wide-angle shot, Blade Runner style" |
| "a forest" | "An enchanted forest with glowing mushrooms and fireflies, moonlight filtering through ancient trees, fantasy art, highly detailed, trending on ArtStation" |

### Key Modifiers That Improve Quality

| Category | Examples |
|---|---|
| **Art Style** | oil painting, watercolor, digital art, anime, photorealistic |
| **Lighting** | golden hour, dramatic lighting, soft bokeh, volumetric light |
| **Quality** | highly detailed, 8K, ultra HD, masterpiece, trending on ArtStation |
| **Camera** | wide-angle, macro, aerial view, close-up portrait |
| **Artists** | Greg Rutkowski style, Studio Ghibli, Artgerm, Alphonse Mucha |

### Negative Prompts Matter Too

Always include a negative prompt to filter out common artefacts:

```
blurry, bad anatomy, bad hands, cropped, worst quality, low quality,
watermark, text, deformed, disfigured, extra fingers, extra limbs
```

### The `guidance_scale` Parameter

| Value | Effect |
|---|---|
| 1–3 | Very creative, ignores prompt |
| 5–8 | Balanced (recommended) |
| 10–15 | Strictly follows prompt |
| 15+ | Over-saturated, artefacts |

---

## ☁️ Running on Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. **Runtime → Change runtime type → GPU** (T4 is free)
4. Paste this into the first cell:

```python
!pip install -q diffusers transformers accelerate torch safetensors Pillow
```

5. Copy the contents of `colab_notebook.py` into the second cell
6. Run both cells — images will display inline!

> 💡 **Tip:** Colab gives you a free NVIDIA T4 GPU (16 GB VRAM), which is more than enough for Stable Diffusion.

---

## 🌐 Web Interface (Gradio)

The Gradio UI (`app.py`) provides:

- 📝 **Prompt input** with multi-line text box
- 🚫 **Negative prompt** pre-filled with quality defaults
- ⚙️ **Advanced settings** — steps, guidance, size, seed
- 💡 **Example gallery** — click to auto-fill prompts
- 🎨 **One-click generation** with progress indicator
- 💾 **Auto-save** to `outputs/` directory
- ℹ️ **Generation info** — timing, parameters, file path

```bash
python app.py
# Opens at http://127.0.0.1:7860
```

To create a **public shareable link** (e.g., for demo):
```python
demo.launch(share=True)
```

---

## 🚀 Future Improvements

### 1. Image-to-Image Generation
Transform an existing image based on a prompt:
```python
from diffusers import StableDiffusionImg2ImgPipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_ID)
```

### 2. ControlNet Integration
Guide generation with edges, poses, or depth maps for more control.

### 3. LoRA Fine-Tuning
Train lightweight adapters on your own images (e.g., specific art styles, characters):
```bash
pip install peft
# Use HuggingFace PEFT + LoRA for fine-tuning
```

### 4. Flask REST API
Build a production-ready API:
```python
from flask import Flask, request, send_file
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def api_generate():
    prompt = request.json["prompt"]
    image = generate_image(prompt)
    image.save("temp.png")
    return send_file("temp.png", mimetype="image/png")
```

### 5. Batch Processing
Generate images from a CSV of prompts for large-scale asset creation.

### 6. Deployment Options

| Platform | Best For | Cost |
|----------|----------|------|
| **HuggingFace Spaces** | Free demo hosting | Free tier |
| **Google Cloud Run** | Scalable API | Pay-per-use |
| **AWS SageMaker** | Enterprise | $$$ |
| **Replicate** | Easy API deployment | Pay-per-use |
| **Local Docker** | Self-hosted | Hardware costs |

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Stability AI](https://stability.ai/) — Stable Diffusion model
- [HuggingFace](https://huggingface.co/) — Diffusers library & model hub
- [Gradio](https://gradio.app/) — Web UI framework
- [CompVis](https://github.com/CompVis/latent-diffusion) — Original Latent Diffusion research

---

<p align="center">
  Built with ❤️ as an internship project at Prodigy InfoTech
</p>
