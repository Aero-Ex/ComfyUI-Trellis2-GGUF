"""
sdnq_fast_loader.py

Fast SDNQ model loader that bypasses .pt cache and the slow sdnq_post_load_quant
(which does SVD + quantization on random init weights, taking ~40s).

HOW IT WORKS
------------
The standard cold-build path is:
  1. Instantiate model with random bf16 weights          (~3s)
  2. sdnq_post_load_quant:
       for each nn.Linear → SVD + quantize random weights (~40s ← THE BOTTLENECK)
  3. load_state_dict(assign=True) from .safetensors      (~5s)
       → overwrites step-2 output with real pre-computed weights
  4. torch.save(model, .pt)                              (~5s)

Step 2 wastes ~40s computing SVD on random weights that are immediately discarded.

The fast loader does:
  1. Instantiate model                                    (~3s)
  2. For each SDNQ layer:
       - Compute SDNQDequantizer from weight shapes + config  (NO SVD)
       - Replace nn.Linear with SDNQLinear using that dequantizer
  3. load_state_dict(assign=True) from .safetensors      (~5s)

Total: ~8s vs ~53s — no .pt cache file needed.

SHAPE-DERIVATION STRATEGY
--------------------------
All SDNQDequantizer fields can be derived WITHOUT running SVD/quantization:

  result_dtype          = nn.Linear.weight.dtype        (bf16)
  original_shape        = nn.Linear.weight.shape        ((out, in))
  original_stride       = nn.Linear.weight.stride()
  quantized_weight_shape= sd[path+'.weight'].shape      (from safetensors directly)
  result_shape          = derived from scale.shape
                          (None if no grouping, original_shape if grouped)
  group_size            = derived from scale.shape
                          (safetensors scale tells us num_of_groups)
  re_quantize_for_matmul= True for int8 (if bit-width or signedness mismatch)
  weights_dtype         = from quantization_config.json
  quantized_matmul_dtype= 'int8' (default for integer types)
  svd_rank/steps        = from quantization_config.json
  use_quantized_matmul  = False initially (set later by apply_sdnq_options)
  layer_class_name      = nn.Linear.__class__.__name__
"""

import os
import sys
import json
import logging
import inspect
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger("Trellis2")

# ─────────────────────────────────────────────────────────────────────────────

def _resolve_module(model: nn.Module, path: str):
    """
    Navigate model by dotted path.
    Returns (parent_module, child_name, child_module).
    e.g. 'blocks.0.self_attn.to_qkv' → (SparseMultiHeadAttention, 'to_qkv', Linear)
    """
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    child_name = parts[-1]
    child = getattr(parent, child_name)
    return parent, child_name, child


def _build_dequantizer(
    original_shape: torch.Size,
    original_stride,
    result_dtype: torch.dtype,
    layer_class_name: str,
    scale_shape: torch.Size,
    weights_dtype: str,
    svd_rank: int,
    svd_steps: int,
    has_zero_point: bool,
    use_quantized_matmul: bool = False,
    use_stochastic_rounding: bool = False,
):
    """
    Build an SDNQDequantizer using tensor shapes instead of actual tensor math.

    quantized_weight_shape is the UNPACKED output shape passed to the dequantizer as
    the .view() target — it must equal original_shape (no grouping) or
    (out, num_groups, group_size) (grouped). It is NOT the packed storage shape
    from safetensors (which has last-dim halved for 4-bit types).

    group_size / result_shape are derived from the stored scale.shape:
      - scale.ndim == 2 → (out, 1): no grouping → group_size=-1, result_shape=None
      - scale.ndim == 3 → (out, num_groups, 1): grouped → derive num_groups, group_size
    """
    from sdnq.common import dtype_dict
    from sdnq.dequantizer import SDNQDequantizer

    # Derive quantized_matmul_dtype
    if dtype_dict[weights_dtype]["is_integer"]:
        quantized_matmul_dtype = "int8"
    elif dtype_dict[weights_dtype]["num_bits"] == 8:
        quantized_matmul_dtype = "float8_e4m3fn"
    else:
        quantized_matmul_dtype = "float16"

    # re_quantize_for_matmul (mirrors sdnq_quantize_layer_weight logic)
    # For int8: is_unsigned=False; bit widths must match
    w_info = dtype_dict[weights_dtype]
    q_info = dtype_dict[quantized_matmul_dtype]
    re_quantize_for_matmul = bool(
        w_info.get("is_unsigned", False)
        or (w_info.get("is_integer", False) != q_info.get("is_integer", False))
        or (w_info.get("num_bits", 8) > q_info.get("num_bits", 8))
        or (
            w_info.get("is_packed", False)
            and not w_info.get("is_integer", False)
            and not q_info.get("is_integer", False)
        )
    )

    # Derive group_size, result_shape, and quantized_weight_shape from scale.shape.
    # For linear: after quantization+grouping, scale.shape = (out, num_groups, 1) or (out, 1)
    # (Note: if transpose happened, scale.shape would be (1, out) or similar — but for int8
    #  re_quantize_for_matmul=True means transpose never happens, so (out, ...) is guaranteed)
    channel_size = original_shape[-1]

    if len(scale_shape) == 3:
        # Grouped: scale.shape = (out, num_groups, 1)
        num_of_groups = scale_shape[1]
        group_size = channel_size // num_of_groups
        result_shape = original_shape              # (out, in) — reshape target after unpack
        # unpacked weight shape = (out, num_groups, group_size)
        quantized_weight_shape = (original_shape[0], num_of_groups, group_size)
        # grouped weights always require re-quantize for matmul (can't directly matmul grouped)
        re_quantize_for_matmul = True
    elif len(scale_shape) == 2 and scale_shape[1] == 1:
        # No grouping: scale.shape = (out, 1)
        group_size = -1
        result_shape = None
        # unpacked weight shape = original_shape = (out, in)
        quantized_weight_shape = original_shape
    elif len(scale_shape) == 2 and scale_shape[0] == 1:
        # Transposed: scale.shape = (1, out)
        group_size = -1
        result_shape = None
        quantized_weight_shape = original_shape
    else:
        # Fallback
        group_size = -1
        result_shape = None
        quantized_weight_shape = original_shape

    return SDNQDequantizer(
        result_dtype=result_dtype,
        result_shape=result_shape,
        original_shape=original_shape,
        original_stride=list(original_stride),
        quantized_weight_shape=quantized_weight_shape,
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=quantized_matmul_dtype,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_quantized_matmul=use_quantized_matmul,
        re_quantize_for_matmul=re_quantize_for_matmul,
        use_stochastic_rounding=use_stochastic_rounding,
        layer_class_name=layer_class_name,
        # is_packed/is_unsigned/is_integer/is_integer_matmul are set inside __init__
        # from dtype_dict[weights_dtype] — do NOT pass them explicitly
    )


def fast_sdnq_load(
    weights_file: str,
    config_file: str,
    qconfig_file: str,
    pkg_root: str,
    use_quantized_matmul: bool = True,
    torch_compile: bool = False,
    device: str = "cuda:0",
) -> nn.Module:
    """
    Load an SDNQ model from its three files (weights+config+qconfig) in ~8s
    without building/reading a .pt cache.

    Args:
        weights_file:       path to {name}.safetensors
        config_file:        path to {name}.json  (architecture)
        qconfig_file:       path to {name}_quantization_config.json
        pkg_root:           e.g. 'trellis2_gguf'
        use_quantized_matmul: enable int8 Triton matmul
        torch_compile:      torch.compile the model (SLatFlowModel only)
        device:             target device
    """
    import importlib
    from safetensors import safe_open
    from sdnq.file_loader import load_files as sdnq_load_files
    from sdnq.layers import SDNQLinear, get_sdnq_wrapper_class
    from sdnq.forward import get_forward_func

    # ── Read configs ──────────────────────────────────────────────────────────
    with open(config_file) as f:
        model_config = json.load(f)
    with open(qconfig_file) as f:
        qcfg = json.load(f)

    weights_dtype = qcfg.get("weights_dtype", "int8")
    svd_rank      = int(qcfg.get("svd_rank", 32))
    svd_steps     = int(qcfg.get("svd_steps", 8))
    use_stoch_rnd = bool(qcfg.get("use_stochastic_rounding", False))

    # Build per-layer dtype map from modules_dtype_dict.
    # Config keys end in '.weight', our layer paths do not.
    # e.g. {"blocks.0.self_attn.to_qkv": "int6", ...}
    _per_layer_dtype: dict = {}
    for dtype_name, paths in qcfg.get("modules_dtype_dict", {}).items():
        for p in paths:
            # strip trailing '.weight' if present
            key = p[:-len(".weight")] if p.endswith(".weight") else p
            _per_layer_dtype[key] = dtype_name

    # ── Scan safetensors keys ─────────────────────────────────────────────────
    with safe_open(weights_file, framework="pt") as fh:
        all_keys = set(fh.keys())

    # Layers with .svd_down key = SDNQ layers
    sdnq_paths = sorted(
        k[: -len(".svd_down")]
        for k in all_keys
        if k.endswith(".svd_down")
    )

    # ── Instantiate model (disabled random init) ──────────────────────────────
    _models_mod = importlib.import_module(f"{pkg_root}.models")
    model_class = _models_mod.__getattr__(model_config["name"])

    sig = inspect.signature(model_class.__init__)
    has_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    final_args = {
        k: v
        for k, v in model_config["args"].items()
        if has_var_kw or k in sig.parameters
    }

    # Disable random init (avoid writing random values to weights we'll immediately overwrite)
    init = torch.nn.init
    _init_funcs = [
        "normal_", "kaiming_uniform_", "uniform_", "zeros_", "ones_",
        "kaiming_normal_", "xavier_uniform_", "xavier_normal_", "constant_",
    ]
    _orig = {n: getattr(init, n) for n in _init_funcs if hasattr(init, n)}
    _noop = lambda t, *a, **k: t
    for n in _orig:
        setattr(init, n, _noop)
    _orig_init_weights = getattr(model_class, "initialize_weights", None)
    if _orig_init_weights:
        model_class.initialize_weights = lambda self: None

    try:
        model = model_class(**final_args)
    finally:
        for n, fn in _orig.items():
            setattr(init, n, fn)
        if _orig_init_weights:
            model_class.initialize_weights = _orig_init_weights

    model.eval()

    # ── Apply sparse linear patch ─────────────────────────────────────────────
    from . import sdnq_utils
    sdnq_utils.apply_sparse_linear_patch(pkg_root)
    sdnq_utils.alias_trellis_package(pkg_root)

    # ── Load safetensors into CPU dict ────────────────────────────────────────
    logger.info("SDNQ: loading %s", os.path.basename(weights_file))
    sd = sdnq_load_files([weights_file], device="cpu")

    # ── Replace nn.Linear → SDNQLinear for each SDNQ path ─────────────────────
    logger.debug("SDNQ: wiring %d quantized layers", len(sdnq_paths))
    for path in sdnq_paths:
        parent, child_name, layer = _resolve_module(model, path)

        # Tensors from safetensors for this layer
        weight    = sd[path + ".weight"]
        scale     = sd[path + ".scale"]
        svd_up_t  = sd.get(path + ".svd_up")
        svd_down_t = sd.get(path + ".svd_down")
        zero_pt   = sd.get(path + ".zero_point")
        bias_t    = sd.get(path + ".bias")

        # Build dequantizer from shapes (no SVD computation!)
        layer_dtype = _per_layer_dtype.get(path, weights_dtype)
        dequantizer = _build_dequantizer(
            original_shape=layer.weight.shape,
            original_stride=layer.weight.stride(),
            result_dtype=layer.weight.dtype,
            layer_class_name=layer.__class__.__name__,
            scale_shape=scale.shape,
            weights_dtype=layer_dtype,
            svd_rank=svd_rank,
            svd_steps=svd_steps,
            has_zero_point=(zero_pt is not None),
            use_quantized_matmul=False,     # set later by apply_sdnq_options
            use_stochastic_rounding=use_stoch_rnd,
        )

        # Set dequantizer on the original layer before wrapping
        # (SDNQLayer.__init__ copies all __dict__ entries, including this)
        layer.sdnq_dequantizer = dequantizer

        # Replace nn.Linear with SDNQLinear
        forward_func = get_forward_func(
            layer.__class__.__name__,
            dequantizer.quantized_matmul_dtype,
            dequantizer.use_quantized_matmul,
        )
        sdnq_layer = get_sdnq_wrapper_class(layer, forward_func)

        # Assign tensors directly as Parameters
        sdnq_layer.weight    = nn.Parameter(weight, requires_grad=False)
        sdnq_layer.scale     = nn.Parameter(scale, requires_grad=False)
        sdnq_layer.zero_point = nn.Parameter(zero_pt, requires_grad=False) if zero_pt is not None else None
        sdnq_layer.svd_up    = nn.Parameter(svd_up_t, requires_grad=False) if svd_up_t is not None else None
        sdnq_layer.svd_down  = nn.Parameter(svd_down_t, requires_grad=False) if svd_down_t is not None else None
        if bias_t is not None:
            sdnq_layer.bias  = nn.Parameter(bias_t, requires_grad=False)

        setattr(parent, child_name, sdnq_layer)

    # ── Load remaining weights (norms, embeddings, biases, etc.) ─────────────
    # Keys already assigned to SDNQ layers
    sdnq_keys = set()
    for path in sdnq_paths:
        for suffix in ("weight", "scale", "svd_up", "svd_down", "zero_point", "bias"):
            sdnq_keys.add(f"{path}.{suffix}")

    remaining_sd = {k: v for k, v in sd.items() if k not in sdnq_keys}
    missing, unexpected = model.load_state_dict(remaining_sd, strict=False)

    _ignored_expected = {"weight", "scale", "svd_up", "svd_down", "zero_point", "bias"}  # already assigned
    truly_missing = [k for k in missing if k.rsplit(".", 1)[-1] not in _ignored_expected]
    if truly_missing:
        logger.warning("SDNQ: unexpected missing keys after load: %s", truly_missing[:5])
    if unexpected:
        logger.warning("SDNQ: unexpected extra keys after load: %s", unexpected[:5])

    del sd

    # ── Finalize (device, matmul, compile) ────────────────────────────────────
    return sdnq_utils.finalize_sdnq_model(
        model, pkg_root, device, use_quantized_matmul, torch_compile
    )
