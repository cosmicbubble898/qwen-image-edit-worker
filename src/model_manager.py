"""
Model Manager — handles loading, caching, and LoRA management for Qwen-Image-Edit.
"""

import os
import re
import time
import torch
from typing import Optional

# Globals
_pipeline = None
_loaded_loras: list[str] = []

# Config from env
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "/models")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2511")
MODEL_PRECISION = os.environ.get("MODEL_PRECISION", "bf16")
# Phr00t v23 NSFW converted model path (baked into Docker image)
NSFW_MODEL_DIR = os.environ.get("NSFW_MODEL_DIR", "/models/qwen-nsfw")
NSFW_LORAS_ENV = os.environ.get("NSFW_LORAS", "")  # comma-separated LoRA names
# Directory on the network volume holding arbitrary user LoRA .safetensors files.
# A request's loras:[{name}] resolves `name` to a file here (or MODEL_CACHE_DIR/loras),
# then falls back to the LORA_REGISTRY (HuggingFace) for the built-in named ones.
LORA_DIR = os.environ.get("LORA_DIR", "/runpod-volume/loras")

# Performance optimizations
ENABLE_ATTENTION_SLICING = os.environ.get("ENABLE_ATTENTION_SLICING", "true").lower() == "true"
ENABLE_VAE_SLICING = os.environ.get("ENABLE_VAE_SLICING", "true").lower() == "true"
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "false").lower() == "true"  # Experimental

# Known NSFW LoRAs (HuggingFace paths)
LORA_REGISTRY = {
    "gnass": {
        "repo": "gnass-org/GNASS-Qwen-Edit-NSFW",
        "filename": "gnass_qwen_edit_nsfw.safetensors",
        "default_weight": 0.7,
    },
    "bfs_face_v5": {
        "repo": "Alissonerdx/BFS-Best-Face-Swap",
        "filename": "bfs_head_v5_2511_merged_version_rank_16_fp16.safetensors",
        "default_weight": 1.0,
    },
    "lightning_4step": {
        "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "default_weight": 1.0,
    },
}


def get_torch_dtype():
    """Get torch dtype from config."""
    if MODEL_PRECISION == "fp8":
        return torch.float8_e4m3fn
    elif MODEL_PRECISION == "fp16":
        return torch.float16
    return torch.bfloat16


def get_pipeline():
    """Get or load the pipeline (singleton)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = _load_pipeline()
    return _pipeline


def _load_pipeline():
    """Load the Qwen-Image-Edit pipeline with Phr00t v23 NSFW weights."""
    from diffusers import QwenImageEditPlusPipeline

    dtype = get_torch_dtype()
    start = time.time()

    # Priority 1: Phr00t v23 NSFW (converted, baked into Docker)
    if os.path.exists(os.path.join(NSFW_MODEL_DIR, "transformer")):
        print(f"[MODEL] Loading Phr00t v23 NSFW from {NSFW_MODEL_DIR} (precision={MODEL_PRECISION})...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            NSFW_MODEL_DIR,
            torch_dtype=dtype,
        )
    # Priority 2: Cached base model
    else:
        local_path = os.path.join(MODEL_CACHE_DIR, "qwen-image-edit-2511")
        if os.path.exists(local_path) and os.listdir(local_path):
            print(f"[MODEL] Loading from cache: {local_path}")
            source = local_path
        else:
            print(f"[MODEL] Downloading from HuggingFace: {MODEL_ID}")
            source = MODEL_ID

        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            source,
            torch_dtype=dtype,
            cache_dir=MODEL_CACHE_DIR if source == MODEL_ID else None,
        )

    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)
    
    # Performance optimizations
    if ENABLE_ATTENTION_SLICING:
        pipeline.enable_attention_slicing(slice_size="auto")
        print("[MODEL] Attention slicing enabled (reduces VRAM)")
    if ENABLE_VAE_SLICING:
        pipeline.enable_vae_slicing()
        print("[MODEL] VAE slicing enabled (reduces VRAM)")
    
    # Experimental: torch.compile for faster inference (requires PyTorch 2.0+)
    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        print("[MODEL] Compiling transformer with torch.compile (may take 1-2 min first run)...")
        pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")

    # Disable any safety filters
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    if hasattr(pipeline, "feature_extractor"):
        pipeline.feature_extractor = None
    if hasattr(pipeline, "watermarker"):
        pipeline.watermarker = None

    elapsed = time.time() - start
    print(f"[MODEL] Pipeline loaded in {elapsed:.1f}s")

    # Load default NSFW LoRAs from env
    if NSFW_LORAS_ENV:
        lora_names = [l.strip() for l in NSFW_LORAS_ENV.split(",") if l.strip()]
        for name in lora_names:
            load_lora(pipeline, name)

    return pipeline


def _adapter_name(name: str) -> str:
    """diffusers adapter names must be simple identifiers — sanitize the filename."""
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    return "lora_" + (re.sub(r"[^A-Za-z0-9_]", "_", stem)[:48] or "x")


def _resolve_lora_file(name: str) -> Optional[str]:
    """Find a LoRA .safetensors on disk by name. Network volume first, then the
    baked model-cache loras dir. Tries the name as-is and with .safetensors."""
    candidates = [name]
    if not name.endswith(".safetensors"):
        candidates.append(name + ".safetensors")
    for base in (LORA_DIR, os.path.join(MODEL_CACHE_DIR, "loras")):
        for n in candidates:
            p = os.path.join(base, n)
            if os.path.isfile(p):
                return p
    return None


def load_lora(pipeline, lora_name: str, weight: Optional[float] = None) -> Optional[str]:
    """Load a single LoRA and return its adapter name (or None on failure).

    Resolution order:
      1. A file on the network volume / cache (arbitrary user LoRAs, by filename).
      2. A built-in entry in LORA_REGISTRY (downloaded from HuggingFace).
    Unknown names are logged loudly so a silent no-op is impossible to miss.
    """
    global _loaded_loras
    adapter = _adapter_name(lora_name)
    if adapter in _loaded_loras:
        return adapter

    start = time.time()
    try:
        path = _resolve_lora_file(lora_name)
        if path:
            print(f"[LORA] Loading '{lora_name}' from volume file {path}...")
            pipeline.load_lora_weights(path, adapter_name=adapter)
        elif lora_name in LORA_REGISTRY:
            info = LORA_REGISTRY[lora_name]
            local = os.path.join(MODEL_CACHE_DIR, "loras", info["filename"])
            if os.path.isfile(local):
                print(f"[LORA] Loading '{lora_name}' from cache {local}...")
                pipeline.load_lora_weights(local, adapter_name=adapter)
            else:
                print(f"[LORA] Loading '{lora_name}' from HuggingFace {info['repo']}...")
                pipeline.load_lora_weights(info["repo"], weight_name=info["filename"], adapter_name=adapter)
        else:
            print(f"[LORA] NOT FOUND: '{lora_name}' — looked in {LORA_DIR}, "
                  f"{os.path.join(MODEL_CACHE_DIR, 'loras')}, and registry {list(LORA_REGISTRY)}")
            return None

        _loaded_loras.append(adapter)
        print(f"[LORA] '{lora_name}' -> adapter '{adapter}' in {time.time() - start:.1f}s")
        return adapter

    except Exception as e:
        print(f"[LORA] FAILED to load '{lora_name}': {e}")
        return None


def set_runtime_loras(pipeline, loras: list[dict]):
    """Load + activate the LoRAs for one generation. Stacks all that load.
    loras: [{"name": "AnimeNSFW3-diffusers.safetensors", "weight": 1.0}, ...]
    """
    if not loras:
        return

    active, weights = [], []
    for lora in loras:
        name = lora.get("name", "")
        if not name:
            continue
        weight = lora.get("weight")
        weight = 1.0 if weight is None else float(weight)
        adapter = load_lora(pipeline, name, weight)
        if adapter:
            active.append(adapter)
            weights.append(weight)

    if active:
        pipeline.set_adapters(active, adapter_weights=weights)
        print(f"[LORA] Active adapters: {dict(zip(active, weights))}")
    else:
        # Requested LoRAs but none applied — surface it instead of silently no-op'ing.
        print(f"[LORA] WARNING: requested {[l.get('name') for l in loras]} but none loaded")
