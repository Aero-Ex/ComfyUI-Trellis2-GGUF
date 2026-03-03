import os
import sys
import logging
import torch
import json
import inspect
import importlib
import pkgutil
from typing import Any, Optional

logger = logging.getLogger("Trellis2")

def apply_sparse_linear_patch(pkg_name: str):
    """
    Patch SDNQ's internal layer mapping to recognize Trellis2's SparseLinear.
    """
    try:
        import sdnq.common as _sc
        import sdnq.layers as _sl
        import sdnq.quantizer as _sq
        from sdnq.layers import SDNQLinear as _SDNQL
        
        # Construct relative import for SparseLinear based on pkg_name
        # e.g. ComfyUI-Trellis2-GGUF.trellis2_gguf.modules.sparse.linear
        _sl_mod = importlib.import_module(f"{pkg_name}.modules.sparse.linear")
        _SL = getattr(_sl_mod, "SparseLinear")
        
        for _attr in ("linear_types", "allowed_types"):
            for _mod in (_sc, _sq):
                _col = getattr(_mod, _attr, None)
                if _col is not None and "SparseLinear" not in _col:
                    _col.add("SparseLinear")
        
        _orig_gw = _sl.get_sdnq_wrapper_class
        def _patched_gw(_layer, _ff, _orig=_orig_gw):
            if isinstance(_layer, _SL):
                return _SDNQL(_layer, _ff)
            return _orig(_layer, _ff)
        
        _sl.get_sdnq_wrapper_class = _patched_gw
        if hasattr(_sq, 'get_sdnq_wrapper_class'):
            _sq.get_sdnq_wrapper_class = _patched_gw
            
    except Exception as _pe:
        logger.warning("SparseLinear patch failed: %s", _pe)

def wrap_sparse_linear_sdnq_layers(model: torch.nn.Module, pkg_name: str):
    """
    Wrap forward_func on ALL SDNQLayer instances so that SparseTensor inputs
    have .feats extracted before the int8 matmul and re-wrapped afterwards.

    SLatFlowModel passes SparseTensors through regular nn.Linear layers too
    (e.g. to_qkv, to_out, FFN), so we must wrap every SDNQLayer regardless
    of original_class, not just those that replaced SparseLinear.
    The wrapper is a no-op for plain tensor inputs (just adds a hasattr check).
    """
    try:
        from sdnq.layers import SDNQLinear as _SDNQL
    except Exception:
        return

    for _mod in model.modules():
        if isinstance(_mod, _SDNQL) and hasattr(_mod, 'forward_func'):
            _orig_ff = _mod.forward_func
            def _sparse_aware(_layer, _input, _ff=_orig_ff):
                if hasattr(_input, 'feats'):
                    _result = _ff(_layer, _input.feats)
                    return _input.replace(_result)
                return _ff(_layer, _input)
            _mod.forward_func = _sparse_aware

def alias_trellis_package(pkg_name: str):
    """
    Pre-populate sys.modules with trellis2.* -> current_package.* 
    and trellis2_gguf.* -> current_package.* aliases.
    This allows unpickling models saved under different package structures.
    """
    try:
        _il = importlib
        _pu = pkgutil
        _t2pkg = _il.import_module(pkg_name)
        
        def _alias_tree(pkg, old_prefix, new_prefix):
            for _mi in _pu.walk_packages(pkg.__path__, prefix=new_prefix + "."):
                _old = old_prefix + _mi.name[len(new_prefix):]
                if _old not in sys.modules:
                    try:
                        sys.modules[_old] = _il.import_module(_mi.name)
                    except Exception:
                        pass
            sys.modules.setdefault(old_prefix, pkg)
            
        _alias_tree(_t2pkg, "trellis2", pkg_name)
        _alias_tree(_t2pkg, "trellis2_gguf", pkg_name)
    except Exception as e:
        logger.warning("Package aliasing failed: %s", e)

def finalize_sdnq_model(model: torch.nn.Module, pkg_name: str, device: str = "cuda:0", 
                        use_quantized_matmul: bool = True, torch_compile: bool = False):
    """
    Perform final model preparation: move to device, enable quantized matmul, 
    wrap sparse layers, and optionally compile.
    """
    # Manually sync sdnq backend options to ensure they match the requested state
    # even if the backend was already imported.
    try:
        import sdnq.common as _sc
        import sdnq.sdnext as _sn
        _sc.use_torch_compile = torch_compile
        _sn.shared.opts.sdnq_dequantize_compile = torch_compile
    except Exception as _sync_err:
        logger.debug("Failed to sync sdnq backend defaults: %s", _sync_err)

    from sdnq.loader import apply_sdnq_options_to_model
    
    model = model.to(device).eval()
    if use_quantized_matmul:
        logger.info("Enabling int8 quantized matmul")
        try:
            model = apply_sdnq_options_to_model(model, use_quantized_matmul=True)
        except RuntimeError as _re:
            logger.warning("Quantized matmul unavailable, falling back: %s", _re)
            
    wrap_sparse_linear_sdnq_layers(model, pkg_name)
    
    if torch_compile:
        _cls_name = type(model).__name__
        if "SLat" in _cls_name:
            logger.info("Applying torch.compile to %s", _cls_name)
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.suppress_errors = True
            # Enable on-disk FX graph cache so compiled kernels survive restarts.
            # TORCHINDUCTOR_CACHE_DIR is set at plugin load time (__init__.py).
            torch._inductor.config.fx_graph_cache = True
            model = torch.compile(model, dynamic=True, fullgraph=False)
        else:
            logger.debug("Skipping torch.compile for %s (SDNQ kernels already fused)", _cls_name)
            
    return model
