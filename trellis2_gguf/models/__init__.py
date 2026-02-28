import importlib
import sys

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

# Mapping: local ckpts/ prefix -> Aero-Ex/Trellis2-GGUF subfolder
# Used to build the exact remote filename without listing the whole repo
_GGUF_REPO = "Aero-Ex/Trellis2-GGUF"
_REPO_PATH_MAP = {
    "ss_dec_":                    "decoders/Stage1/",
    "shape_dec_":                 "decoders/Stage2/",
    "tex_dec_":                   "decoders/Stage2/",
    "ss_flow_":                   "refiner/",
    "slat_flow_img2shape_":       "shape/",
    "slat_flow_imgshape2tex_":    "texture/",
    "shape_enc_":                 "encoders/",
    "tex_enc_":                   "encoders/",
}

def _remote_path(basename: str, suffix: str) -> str:
    """Map a ckpts basename to its Aero-Ex repo path (e.g. 'shape/..._Q8_0.gguf')."""
    # Normalize suffix for GGUF quants that might have different naming in repo
    if suffix.endswith(".gguf"):
        # Handle Q5_K -> Q5_K_M mapping if common
        if "_Q5_K.gguf" in suffix: suffix = suffix.replace("_Q5_K.gguf", "_Q5_K_M.gguf")
        if "_Q4_K.gguf" in suffix: suffix = suffix.replace("_Q4_K.gguf", "_Q4_K_M.gguf")

    for prefix, folder in _REPO_PATH_MAP.items():
        if basename.startswith(prefix):
            return f"{folder}{basename}{suffix}"
    return f"{basename}{suffix}"   # fallback: root of repo


def _tensor_shape(t):
    """Get the true shape of a tensor, handling GGMLTensor (quantized) and regular tensors."""
    # GGMLTensor stores original unquantized shape in .tensor_shape
    ts = getattr(t, "tensor_shape", None)
    if ts is not None:
        return tuple(ts)
    return tuple(t.shape)


def _infer_arch_from_sd(sd: dict, config: dict) -> dict:
    """
    Patch config['args'] from actual weight shapes in the state dict.
    This makes loading robust against GGUF files whose architecture differs
    from what the .json says (e.g. Aero-Ex builds with model_channels=1632).
    """
    args = config.get("args", {})

    # model_channels — t_embedder MLP hidden dim or input projection output
    for key in ("t_embedder.mlp.0.weight", "t_embedder.mlp.2.weight", 
                "input_layer.weight", "input_proj.weight", "x_embedder.weight",
                "blocks.0.self_attn.to_qkv.weight"):
        if key in sd:
            shape = _tensor_shape(sd[key])
            # For linear weights (out, in), model_channels is usually the 'out' for input_layer 
            # and 'in' for blocks.0.self_attn.to_qkv.
            if "blocks.0.self_attn" in key:
                args["model_channels"] = shape[1]
            elif "input_layer" in key or "input_proj" in key or "x_embedder" in key:
                args["model_channels"] = shape[0]
            else:
                # Timestep embedder case
                args["model_channels"] = shape[0]
            break

    # in_channels — input projection input
    for key in ("input_layer.weight", "input_proj.weight", "x_embedder.weight"):
        if key in sd:
            args["in_channels"] = _tensor_shape(sd[key])[1]
            break

    # out_channels — final linear output
    for key in ("output_layer.1.weight", "final_layer.linear.weight", "proj_out.weight"):
        if key in sd:
            args["out_channels"] = _tensor_shape(sd[key])[0]
            break

    # num_blocks — count unique block indices
    import re
    block_ids = {int(m.group(1)) for k in sd for m in [re.match(r"^blocks\.(\d+)\.", k)] if m}
    if block_ids:
        args["num_blocks"] = max(block_ids) + 1

    # mlp_ratio — infer from first block's MLP hidden size
    mc = args.get("model_channels")
    if mc:
        for key in ("blocks.0.mlp.mlp.0.weight", "blocks.0.ff.net.0.proj.weight"):
            if key in sd:
                mlp_hidden = _tensor_shape(sd[key])[0]
                args["mlp_ratio"] = mlp_hidden / mc
                break

    # num_heads — infer from attention q projection if available
    if mc:
        for key in ("blocks.0.attn.to_q.weight", "blocks.0.attn1.to_q.weight"):
            if key in sd:
                q_dim = _tensor_shape(sd[key])[0]
                num_head_channels = args.get("num_head_channels", 64)
                args["num_heads"] = q_dim // num_head_channels
                break

    config["args"] = args
    return config


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, enable_gguf: bool = False, gguf_quant: str = "Q8_0", **kwargs):
    """
    Load a model from a pretrained checkpoint.

    This function ONLY performs local file resolution — it NEVER downloads anything.
    All downloading must be done first via Trellis2GGUFLoadModel which calls
    model_manager.ensure_model_files().

    Args:
        path: Absolute path prefix for the model (without .json / .safetensors / .gguf).
              e.g.  /path/to/models/Trellis2GGUF/ckpts/ss_flow_img_dit_1_3B_64_bf16
        enable_gguf: Load as GGUF instead of Safetensors.
        gguf_quant:  GGUF quantization type (e.g. "Q6_K").
        precision:   Safetensors precision suffix override (e.g. "fp8", "bf16").
    """
    import os
    import sys
    import json
    import torch
    import torch.nn.init as init
    import inspect
    from safetensors.torch import load_file
    from ..utils import gguf_utils

    # Import model_manager for centralized file resolution
    import importlib.util, sys as _sys
    _mm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model_manager.py")
    if "trellis2_gguf_model_manager" not in _sys.modules:
        spec = importlib.util.spec_from_file_location("trellis2_gguf_model_manager", _mm_path)
        _mm = importlib.util.module_from_spec(spec)
        _sys.modules["trellis2_gguf_model_manager"] = _mm
        spec.loader.exec_module(_mm)
    model_manager = _sys.modules["trellis2_gguf_model_manager"]

    if gguf_quant and gguf_quant.startswith("GGUF_"):
        gguf_quant = gguf_quant[5:]

    precision = kwargs.pop("precision", None)
    basename = os.path.basename(path)

    print(f"[Trellis2GGUF-models] from_pretrained: {basename}  enable_gguf={enable_gguf}  quant={gguf_quant}  precision={precision}", file=sys.stderr)

    # ── Resolve local paths via model_manager (NO downloads here) ────────
    config_file, model_file, is_gguf = model_manager.resolve_local_path(
        basename,
        enable_gguf=enable_gguf,
        gguf_quant=gguf_quant,
        precision=precision,
    )

    print(f"[Trellis2GGUF-models] Resolved: {os.path.basename(model_file)}", file=sys.stderr)

    # ── Load checkpoint ───────────────────────────────────────────────────
    with open(config_file, 'r') as f:
        config = json.load(f)

    if is_gguf:
        print(f"[TRELLIS2] Loading GGUF: {os.path.basename(model_file)}", file=sys.stderr)
        sd, metadata = gguf_utils.load_gguf_checkpoint(model_file)
        if metadata:
            meta_map = {
                'trellis.attention.head_count': 'num_heads',
                'trellis.model.model_channels':  'model_channels',
                'trellis.model.num_blocks':       'num_blocks',
                'trellis.model.in_channels':      'in_channels',
                'trellis.model.out_channels':     'out_channels',
            }
            for k, v in meta_map.items():
                if k in metadata:
                    config['args'][v] = metadata[k]
        config = _infer_arch_from_sd(sd, config)
        print(f"[Trellis2GGUF-models]   Arch inferred from sd: model_channels={config['args'].get('model_channels')} "
              f"num_blocks={config['args'].get('num_blocks')} "
              f"mlp_ratio={config['args'].get('mlp_ratio')}", file=sys.stderr)
    else:
        sd = load_file(model_file)

    # ── Build model (skip random init for speed) ──────────────────────────
    model_class = __getattr__(config['name'])
    _orig_init_weights = getattr(model_class, 'initialize_weights', None)
    if _orig_init_weights:
        model_class.initialize_weights = lambda self: None

    _init_funcs = ['normal_', 'kaiming_uniform_', 'uniform_', 'zeros_', 'ones_',
                   'kaiming_normal_', 'xavier_uniform_', 'xavier_normal_', 'constant_']
    _orig_inits = {name: getattr(init, name) for name in _init_funcs if hasattr(init, name)}
    _noop = lambda tensor, *args, **kwargs: tensor
    for name in _orig_inits:
        setattr(init, name, _noop)

    try:
        merged_args = {**config['args'], **kwargs}
        sig = inspect.signature(model_class.__init__)
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        final_args = {k: v for k, v in merged_args.items() if has_var_kwargs or k in sig.parameters}
        model = model_class(**final_args)
    finally:
        for name, fn in _orig_inits.items():
            setattr(init, name, fn)
        if _orig_init_weights:
            model_class.initialize_weights = _orig_init_weights

    if is_gguf:
        model = gguf_utils.convert_to_ggml(model)

    model.load_state_dict(sd, strict=False)

    if not is_gguf:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            sd_loaded = load_file(model_file, device=device)
            model.load_state_dict(sd_loaded, strict=False)
        except Exception as e:
            print(f"[Trellis2GGUF-models] Fast load failed ({e}), using CPU fallback", file=sys.stderr)
            model.load_state_dict(load_file(model_file), strict=False)

    return model

# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
