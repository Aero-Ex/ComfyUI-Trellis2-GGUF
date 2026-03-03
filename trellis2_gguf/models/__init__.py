import importlib
import sys
import os
import json
import logging
import inspect
import torch
from typing import Any, Optional
from ..utils import sdnq_utils
from ..utils import gguf_utils

logger = logging.getLogger("Trellis2")

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


def from_pretrained(path: str, enable_gguf: bool = False, gguf_quant: str = "Q8_0",
                    enable_sdnq: bool = False, sdnq_use_quantized_matmul: bool = True,
                    sdnq_torch_compile: bool = False,
                    **kwargs):
    """
    Load a model from a pretrained checkpoint.

    This function ONLY performs local file resolution — it NEVER downloads anything.
    All downloading must be done first via Trellis2LoadModel which calls
    model_manager.ensure_model_files().

    Args:
        path: Absolute path prefix for the model (without .json / .safetensors / .gguf).
              For SDNQ: pass the SDNQ directory path (already resolved by the pipeline).
              e.g.  /path/to/models/Trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16
        enable_gguf: Load as GGUF instead of Safetensors.
        gguf_quant:  GGUF quantization type (e.g. "Q6_K").
        precision:   Safetensors precision suffix override (e.g. "fp8", "bf16").
        enable_sdnq: Load as SDNQ quantized model (uint4 + SVD). Path must be a directory.
        sdnq_use_quantized_matmul: Use int8 quantized matmul (requires Triton).
    """
    from safetensors.torch import load_file
    from ..utils import gguf_utils

    # Import model_manager for centralized file resolution
    import importlib.util
    _mm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model_manager.py")
    if "trellis2_model_manager" not in sys.modules:
        spec = importlib.util.spec_from_file_location("trellis2_model_manager", _mm_path)
        _mm = importlib.util.module_from_spec(spec)
        sys.modules["trellis2_model_manager"] = _mm
        spec.loader.exec_module(_mm)
    model_manager = sys.modules["trellis2_model_manager"]

    if gguf_quant and gguf_quant.startswith("GGUF_"):
        gguf_quant = gguf_quant[5:]

    precision = kwargs.pop("precision", None)
    basename = os.path.basename(path)

    logger.debug("Loading %s  sdnq=%s  gguf=%s", basename, enable_sdnq, enable_gguf)

    # ------------------------------------------------------------------ #
    # SDNQ loading: flat layout  sdnq/{name}_quantization_config.json
    #               or legacy    sdnq/{name}/quantization_config.json
    # ------------------------------------------------------------------ #
    _sdnq_is_flat   = os.path.isfile(path + "_quantization_config.json")
    _sdnq_is_subdir = os.path.isdir(path) and os.path.exists(os.path.join(path, "quantization_config.json"))
    _is_sdnq = _sdnq_is_flat or _sdnq_is_subdir or (enable_sdnq and (os.path.isdir(path) or _sdnq_is_flat))
    if _is_sdnq and (_sdnq_is_flat or _sdnq_is_subdir or os.path.isdir(path)):
        # Resolve files for both layouts
        if _sdnq_is_flat:
            _sdnq_dir = os.path.dirname(path)  # parent dir for display
            _sdnq_name = os.path.basename(path)
            _qconfig_path = path + "_quantization_config.json"
            _parent = os.path.dirname(path)
            # model config json: {name}.json (same name, not the quantization one)
            _json_files = [f for f in os.listdir(_parent)
                           if f == _sdnq_name + ".json"]
            if not _json_files:
                # fallback: any json starting with the name that isn't a quant config
                _json_files = [f for f in os.listdir(_parent)
                               if f.startswith(_sdnq_name) and f.endswith(".json")
                               and not f.endswith("_quantization_config.json")]
            _st_files = [f for f in os.listdir(_parent)
                         if f == _sdnq_name + ".safetensors"]
            if not _json_files:
                raise FileNotFoundError(f"[Trellis2-SDNQ] No model config .json for {_sdnq_name} in {_parent}")
            if not _st_files:
                raise FileNotFoundError(f"[Trellis2-SDNQ] No .safetensors for {_sdnq_name} in {_parent}")
            _config_file  = os.path.join(_parent, _json_files[0])
            _weights_file = os.path.join(_parent, _st_files[0])
            _cache_path   = path + ".pt"  # sdnq/{name}.pt
        else:
            _sdnq_dir = path
            _qconfig_path = os.path.join(_sdnq_dir, "quantization_config.json")
            _json_files = [f for f in os.listdir(_sdnq_dir)
                           if f.endswith(".json") and f != "quantization_config.json"]
            if not _json_files:
                raise FileNotFoundError(f"[Trellis2-SDNQ] No model config .json found in {_sdnq_dir}")
            _config_file = os.path.join(_sdnq_dir, _json_files[0])
            _st_files = [f for f in os.listdir(_sdnq_dir) if f.endswith(".safetensors")]
            if not _st_files:
                raise FileNotFoundError(f"[Trellis2-SDNQ] No .safetensors weights found in {_sdnq_dir}")
            _weights_file = os.path.join(_sdnq_dir, _st_files[0])
            _cache_path   = _sdnq_dir.rstrip("/\\") + ".pt"

        logger.debug("SDNQ weights: %s", os.path.basename(_weights_file))

        _dev = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")

        # Set SDNQ engine options in environment before sdnq is imported
        os.environ["SDNQ_USE_TORCH_COMPILE"] = "1" if sdnq_torch_compile else "0"

        try:
            import sdnq  # noqa: just verify sdnq is importable before proceeding
        except ImportError as _e:
            raise ImportError(
                f"[Trellis2-SDNQ] Cannot import sdnq. Make sure it is installed (pip install sdnq).\n"
                f"  Original error: {_e}"
            ) from _e

        # SDNQ loading tools
        _pkg_root = __name__.rsplit('.', 1)[0]

        # Cache: flat → {name}.pt; legacy → {subdir}.pt (already set above in each branch)
        if os.path.isfile(_cache_path) and os.path.getmtime(_cache_path) >= os.path.getmtime(_weights_file):
            _csz = os.path.getsize(_cache_path) / 1024**2
            logger.info("SDNQ: loading from cache  %s  (%.0f MB)", os.path.basename(_cache_path), _csz)
            
            sdnq_utils.apply_sparse_linear_patch(_pkg_root)
            sdnq_utils.alias_trellis_package(_pkg_root)

            _model = torch.load(_cache_path, map_location="cpu", weights_only=False)
            return sdnq_utils.finalize_sdnq_model(_model, _pkg_root, _dev, sdnq_use_quantized_matmul, sdnq_torch_compile)

        # Fast build — shape-derived SDNQDequantizer, no SVD on random weights, no .pt cache
        from ..utils.sdnq_fast_loader import fast_sdnq_load
        return fast_sdnq_load(
            weights_file=_weights_file,
            config_file=_config_file,
            qconfig_file=_qconfig_path,
            pkg_root=_pkg_root,
            use_quantized_matmul=sdnq_use_quantized_matmul,
            torch_compile=sdnq_torch_compile,
            device=_dev,
        )
    # ------------------------------------------------------------------ #

    # ── Resolve local paths via model_manager (NO downloads here) ────────
    config_file, model_file, is_gguf = model_manager.resolve_local_path(
        basename,
        enable_gguf=enable_gguf,
        gguf_quant=gguf_quant,
        precision=precision,
    )

    logger.debug("Resolved: %s", os.path.basename(model_file))

    # ── Load checkpoint ───────────────────────────────────────────────────
    with open(config_file, 'r') as f:
        config = json.load(f)

    if is_gguf:
        logger.info("GGUF: %s", os.path.basename(model_file))
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
        # Always infer from actual weight shapes — most reliable source of truth
        # (handles GGUF files built with different dims than the json specifies)
        config = _infer_arch_from_sd(sd, config)
        logger.debug("Arch: model_channels=%s  num_blocks=%s  mlp_ratio=%s",
                     config['args'].get('model_channels'),
                     config['args'].get('num_blocks'),
                     config['args'].get('mlp_ratio'))
    else:
        sd = load_file(model_file)

    # Build model (skip random init for speed)
    model_class = __getattr__(config['name'])
    _orig_init_weights = getattr(model_class, 'initialize_weights', None)
    if _orig_init_weights:
        model_class.initialize_weights = lambda self: None

    init = torch.nn.init
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
        # Reload directly to CUDA for FP16 models (avoids CPU overhead)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            sd_loaded = load_file(model_file, device=device)
            model.load_state_dict(sd_loaded, strict=False)
        except Exception as e:
            logger.warning("Direct CUDA load failed (%s), using CPU fallback", e)
            model.load_state_dict(load_file(model_file), strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
