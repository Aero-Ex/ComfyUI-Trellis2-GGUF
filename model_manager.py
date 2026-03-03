"""
model_manager.py
================
Single source of truth for all Trellis2 model file management.

- File resolution: given a model basename and format, returns the absolute local path.
- Downloading: downloads missing files from HuggingFace into models/Trellis2.
- ONLY called from Trellis2LoadModel.process in nodes.py — no other code should download.

All other internal code (trellis2/models/__init__.py etc.) calls `resolve_local_path` only,
which does a pure local lookup and raises FileNotFoundError if the file is missing.
"""

import os
from typing import Optional

import folder_paths

# ─────────────────────────────────────────────────────────────────────────────
# Repository identifiers
# ─────────────────────────────────────────────────────────────────────────────
GGUF_REPO   = "Aero-Ex/Trellis2-GGUF"
DINOV3_REPO = "Aero-Ex/Dinov3"
SDNQ_REPO   = "Aero-Ex/Trellis2-SDNQ"

# ─────────────────────────────────────────────────────────────────────────────
# Aero-Ex/Trellis2-GGUF subfolder map  (basename prefix → HF subfolder)
# ─────────────────────────────────────────────────────────────────────────────
REPO_PATH_MAP = {
    "ss_dec_":                 "decoders/Stage1/",
    "shape_dec_":              "decoders/Stage2/",
    "tex_dec_":                "decoders/Stage2/",
    "ss_flow_":                "refiner/",
    "slat_flow_img2shape_":    "shape/",
    "slat_flow_imgshape2tex_": "texture/",
    "shape_enc_":              "encoders/",
    "tex_enc_":                "encoders/",
}

# Models that are hardcoded in the pipeline but missing from pipeline.json
EXTRA_MODELS = [
    "ckpts/ss_dec_conv3d_16l8_fp16",
    "ckpts/shape_enc_next_dc_f16c32_fp16",
]

# Encoder/decoder prefixes — these always use plain safetensors (fp16 baked in name)
ENC_DEC_PREFIXES = ("ss_dec_", "shape_dec_", "tex_dec_", "shape_enc_", "tex_enc_")


def get_models_dir() -> str:
    """Absolute path to models/Trellis2."""
    return os.path.join(folder_paths.models_dir, "Trellis2")


def remote_path(basename: str, suffix: str) -> str:
    """Map a model basename to its path inside the Aero-Ex HF repo."""
    if suffix.endswith(".gguf"):
        suffix = suffix.replace("_Q5_K.gguf", "_Q5_K_M.gguf")
        suffix = suffix.replace("_Q4_K.gguf", "_Q4_K_M.gguf")
    for prefix, folder in REPO_PATH_MAP.items():
        if basename.startswith(prefix):
            return f"{folder}{basename}{suffix}"
    return f"{basename}{suffix}"


def is_enc_dec(basename: str) -> bool:
    return any(basename.startswith(p) for p in ENC_DEC_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# Local file resolution (NO downloads, raises FileNotFoundError if missing)
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_paths(basename: str, suffix: str) -> list[str]:
    """
    Return all local candidate paths for a given model file, in priority order.
    Searches:
      1. Flat root:  models/Trellis2/<basename><suffix>
      2. Nested:     models/Trellis2/<folder>/<basename><suffix>   (Aero-Ex layout)
    """
    root = get_models_dir()
    nested = remote_path(basename, suffix)
    return [
        os.path.join(root, f"{basename}{suffix}"),   # flat root
        os.path.join(root, nested),                  # nested  (e.g. refiner/file_Q6_K.gguf)
    ]


def resolve_local_path(basename: str, enable_gguf: bool = False, gguf_quant: str = "Q8_0",
                       precision: Optional[str] = None) -> tuple[str, str, bool]:
    """
    Find the local files for a model.

    Returns:
        (config_path, model_path, is_gguf)

    Raises:
        FileNotFoundError if the model is not found locally.
    """
    # enc/dec: never GGUF, never precision suffix (fp16 is in the basename)
    if is_enc_dec(basename):
        enable_gguf = False
        precision = None

    # JSON config candidates
    json_candidates = _candidate_paths(basename, ".json")
    config_path = next((p for p in json_candidates if os.path.exists(p)), None)

    if enable_gguf:
        # GGUF candidates: quantized first, then plain
        gguf_sfx = f"_{gguf_quant}.gguf"
        gguf_candidates = _candidate_paths(basename, gguf_sfx) + _candidate_paths(basename, ".gguf")
        model_path = next((p for p in gguf_candidates if os.path.exists(p)), None)
        if model_path:
            if config_path is None:
                config_path = model_path  # json not strictly needed for GGUF
            return config_path, model_path, True
    else:
        # Safetensors candidates: precision-specific first, then plain
        sfx_list = []
        if precision:
            sfx_list.append(f"_{precision}.safetensors")
        sfx_list.append(".safetensors")
        for sfx in sfx_list:
            st_candidates = _candidate_paths(basename, sfx)
            model_path = next((p for p in st_candidates if os.path.exists(p)), None)
            if model_path:
                if config_path is None:
                    raise FileNotFoundError(f"Model file found at {model_path} but no JSON config for {basename}")
                return config_path, model_path, False

    raise FileNotFoundError(
        f"Model not found locally: {basename} "
        f"(gguf={enable_gguf}, quant={gguf_quant}, precision={precision}). "
        f"Run Trellis2LoadModel first to download all required files."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Downloader  (called ONLY from Trellis2LoadModel.process in nodes.py)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Downloader  (called ONLY from Trellis2LoadModel.process in nodes.py)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_model_files(
    model_format: str,
    pipeline_config: dict,
) -> dict:
    """
    Download all required model files if not present locally.
    Called once during Trellis2LoadModel.process.

    Args:
        model_format: e.g. "GGUF Q6_K", "Safetensors (BF16)", "sdnq_int8_svd64"
        pipeline_config: parsed pipeline.json dict

    Returns:
        dict mapping model_key -> resolved (config_path, model_path, is_gguf)
    """
    from huggingface_hub import hf_hub_download

    root = get_models_dir()
    os.makedirs(root, exist_ok=True)

    enable_gguf = model_format.startswith("GGUF")
    enable_sdnq = model_format.startswith("sdnq")
    
    gguf_quant = model_format.split(" ")[1] if enable_gguf else "Q8_0"
    precision = None
    if "(BF16)" in model_format:
        precision = "bf16"
    elif "(FP8)" in model_format:
        precision = "fp8"

    # ── Ensure pipeline.json ──────────────────────────────────────────────
    pipeline_json_path = os.path.join(root, "pipeline.json")
    if not os.path.exists(pipeline_json_path):
        print(f"[ModelManager] Downloading pipeline.json from {GGUF_REPO}...")
        hf_hub_download(repo_id=GGUF_REPO, filename="pipeline.json", local_dir=root)

    # ── Ensure DINOv3 ─────────────────────────────────────────────────────
    dinov3_new  = os.path.join(root, "dinov3", "facebook", "dinov3-vitl16-pretrain-lvd1689m", "model.safetensors")
    dinov3_old  = os.path.join(folder_paths.models_dir, "Aero-Ex", "Dinov3", "facebook",
                               "dinov3-vitl16-pretrain-lvd1689m", "model.safetensors")
    if not os.path.exists(dinov3_new) and not os.path.exists(dinov3_old):
        print(f"[ModelManager] Downloading DINOv3 from {DINOV3_REPO}...")
        dinov3_dir = os.path.join(root, "dinov3")
        for fn in ["config.json", "model.safetensors", "preprocessor_config.json"]:
            hf_hub_download(
                repo_id=DINOV3_REPO,
                filename=f"facebook/dinov3-vitl16-pretrain-lvd1689m/{fn}",
                local_dir=dinov3_dir,
            )

    # ── Ensure all model components ───────────────────────────────────────
    models_cfg = pipeline_config.get("args", {}).get("models", {})
    
    # Merge EXTRA_MODELS into the collection to ensure they are downloaded
    all_models = dict(models_cfg)
    for extra_path in EXTRA_MODELS:
        extra_key = extra_path.split("/")[-1]
        if extra_key not in all_models:
            all_models[extra_key] = extra_path

    print(f"[ModelManager] Verifying {len(all_models)} model components ({model_format})...")

    resolved = {}
    for key, rel_path in all_models.items():
        basename = rel_path.split("/")[-1]
        enc_dec = is_enc_dec(basename)

        # ── SDNQ Download Logic ──────────────────────────────────────────
        if enable_sdnq and not enc_dec:
            # SDNQ files go into models/Trellis2/sdnq/
            sdnq_dir = os.path.join(root, "sdnq")
            os.makedirs(sdnq_dir, exist_ok=True)
            
            # Map name: 'ss_flow_img_dit_1_3B_64_bf16' -> 'ss_flow_img_dit_1_3B_64_int8_svd64'
            # Note: The model_format might be 'sdnq_int8_svd64'
            suffix_to_add = model_format.replace("sdnq_", "")  # 'int8_svd64'
            
            # Usually base name ends with _bf16, _fp16, or similar
            base = basename
            for p in ["_bf16", "_fp16", "_fp8"]:
                if base.endswith(p):
                    base = base[:len(base)-len(p)]
                    break
            
            sdnq_basename = f"{base}_{suffix_to_add}"
            
            # Each SDNQ component needs 3 files
            sdnq_suffixes = [".json", ".safetensors", "_quantization_config.json"]
            for sfx in sdnq_suffixes:
                fn = sdnq_basename + sfx
                local_f = os.path.join(sdnq_dir, fn)
                if not os.path.exists(local_f):
                    print(f"[ModelManager] Downloading SDNQ file {fn} from {SDNQ_REPO}...")
                    try:
                        hf_hub_download(repo_id=SDNQ_REPO, filename=fn, local_dir=sdnq_dir)
                    except Exception as e:
                        print(f"[ModelManager] ⚠ Failed to download SDNQ {fn}: {e}")
            
            # For SDNQ, we don't 'resolve_local_path' the usual way as it expects 
            # safetensors in root or nested folders, whereas SDNQ is in sdnq/ folder.
            # But the pipeline's from_pretrained will handle this by taking 'path' = root/sdnq/name
            resolved[key] = os.path.join(sdnq_dir, sdnq_basename)
            continue

        # ── Regular GGUF / Safetensors Download Logic ─────────────────────
        # Determine suffixes to fetch
        if enc_dec:
            # enc/dec: always plain safetensors (fp16 in name already)
            suffixes = [".json", ".safetensors"]
            use_gguf = False
            prec = None
        elif enable_gguf:
            suffixes = [".json", f"_{gguf_quant}.gguf"]
            use_gguf = True
            prec = None
        else:
            sfx = f"_{precision}.safetensors" if precision else ".safetensors"
            suffixes = [".json", sfx]
            use_gguf = False
            prec = precision

        for sfx in suffixes:
            candidates = _candidate_paths(basename, sfx)
            if any(os.path.exists(c) for c in candidates):
                continue  # already present

            # Need to download
            hf_filename = remote_path(basename, sfx)
            print(f"[ModelManager] Downloading {hf_filename} from {GGUF_REPO}...")
            try:
                hf_hub_download(repo_id=GGUF_REPO, filename=hf_filename, local_dir=root)
            except Exception as e:
                print(f"[ModelManager] ⚠ Failed to download {hf_filename}: {e}")

        # Record resolved paths for caller
        try:
            resolved[key] = resolve_local_path(basename, enable_gguf=use_gguf,
                                               gguf_quant=gguf_quant, precision=prec)
        except FileNotFoundError as e:
            print(f"[ModelManager] ✘ Could not resolve {key} after download attempt: {e}")
            resolved[key] = None

    return resolved
