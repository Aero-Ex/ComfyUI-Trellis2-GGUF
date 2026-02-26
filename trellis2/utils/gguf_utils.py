# Adapted from ComfyUI-GGUF (c) City96 || Apache-2.0
# Ported from ComfyUI-TRELLIS2 to ComfyUI-Trellis2 with fixes for:
#   - MultiHeadRMSNorm / SparseMultiHeadRMSNorm use gamma not weight
#   - convert_to_ggml covers SparseGroupNorm32 and SparseLayerNorm32
import sys
import os
import gguf
import torch
import logging
import warnings
import re
import importlib.util
from typing import Dict, Any, Tuple, Optional, List

# Try to find ComfyUI-GGUF for native support
_GGUF_STRATEGY_LOGGED = False

def chained_hasattr(obj, chained_attr):
    probe = obj
    for attr in chained_attr.split('.'):
        if hasattr(probe, attr):
            probe = getattr(probe, attr)
        else:
            return False
    return True

# A backward and forward compatible way to get `torch.compiler.disable`.
def get_torch_compiler_disable_decorator():
    def dummy_decorator(*args, **kwargs):
        def noop(x):
            return x
        return noop

    from packaging import version

    if not chained_hasattr(torch, "compiler.disable"):
        logging.info("ComfyUI-GGUF: Torch too old for torch.compile - bypassing")
        return dummy_decorator # torch too old
    elif version.parse(torch.__version__) >= version.parse("2.8"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile")
        return dummy_decorator # torch compile works
    if chained_hasattr(torch, "_dynamo.config.nontraceable_tensor_subclasses"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile (nightly)")
        return dummy_decorator # torch compile works, nightly before 2.8 release
    else:
        logging.info("ComfyUI-GGUF: Partial torch compile only, consider updating pytorch")
        return torch.compiler.disable

torch_compiler_disable = get_torch_compiler_disable_decorator()
def _setup_native_gguf():
    global GGMLTensor, GGMLLayer, _native_dequantize_tensor, is_quantized, HAS_GGUF_OPS, get_orig_shape

    # Try local import first (in case it's in sys.path)
    try:
        from . import dequant as local_dequant
    except:
        local_dequant = None

    # Check for ComfyUI-GGUF in custom_nodes.
    # This file lives at:  custom_nodes/ComfyUI-Trellis2/trellis2/utils/gguf_utils.py
    # Three levels up reaches custom_nodes/.
    custom_nodes_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    gguf_path = os.path.join(custom_nodes_path, "ComfyUI-GGUF")

    if not os.path.isdir(gguf_path):
        return False

    try:
        def load_module(name, path):
            spec = importlib.util.spec_from_file_location(f"gguf_native.{name}", path)
            module = importlib.util.module_from_spec(spec)
            module.__package__ = "gguf_native"
            sys.modules[f"gguf_native.{name}"] = module
            spec.loader.exec_module(module)
            return module

        import types
        gguf_native = types.ModuleType("gguf_native")
        gguf_native.__path__ = [gguf_path]
        sys.modules["gguf_native"] = gguf_native

        gguf_dequant = load_module("dequant", os.path.join(gguf_path, "dequant.py"))
        gguf_ops = load_module("ops", os.path.join(gguf_path, "ops.py"))
        gguf_loader = load_module("loader", os.path.join(gguf_path, "loader.py"))

        GGMLTensor = gguf_ops.GGMLTensor
        GGMLLayer = gguf_ops.GGMLLayer
        _native_dequantize_tensor = gguf_dequant.dequantize_tensor
        is_quantized = gguf_dequant.is_quantized
        get_orig_shape = gguf_loader.get_orig_shape
        HAS_GGUF_OPS = True
        print("[Trellis2] Using native ComfyUI-GGUF support (ops/dequant/loader)", file=sys.stderr)
        return True
    except Exception as e:
        if "comfy" in str(e):
            pass  # Expected in isolated subprocess envs
        else:
            print(
                f"[Trellis2] ComfyUI-GGUF import failed: {e}. Using internal GGUF implementation.",
                file=sys.stderr,
            )
        for k in list(sys.modules.keys()):
            if k.startswith("gguf_native"):
                del sys.modules[k]
        HAS_GGUF_OPS = False
        _native_dequantize_tensor = None
        return False


_native_dequantize_tensor = None

# ── Fallback dequantization implementations ──────────────────────────────────

TORCH_COMPATIBLE_QTYPES = (
    None,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.BF16,
)

QTYPES_TO_DTYPE = {
    gguf.GGMLQuantizationType.F32: torch.float32,
    gguf.GGMLQuantizationType.F16: torch.float16,
    gguf.GGMLQuantizationType.BF16: torch.bfloat16,
}



def is_torch_compatible(tensor):
    return tensor is None or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES


def is_quantized(tensor):
    return not is_torch_compatible(tensor)


def to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = (ql | (qh << 4))
    return (d * qs) + m


def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d  = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs)


def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m


def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d  = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return (d * qs)


# K Quants
QK_K = 256
K_SCALE_SIZE = 12


def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8))


def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    ql, qh, scales, d, = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))

    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))

    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))


# IQ quants
KVALUES = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.int8,
)


def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size//2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int64)

    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))
    del kvalues # should still be view, but just to be safe

    return (d * qs)

def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = to_uint16(scales_h)

    shift_a = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2))
    shift_b = torch.tensor([2 * i for i in range(QK_K // 32)], device=d.device, dtype=torch.uint8).reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03

    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & 0x0F

    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64)).reshape((n_blocks, -1, 32))
    del kvalues # see IQ4_NL
    del shift_a
    del shift_b

    return (dl * qs).reshape((n_blocks, -1))


dequantize_functions = {
    gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
    gguf.GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS: dequantize_blocks_IQ4_XS,
}


def dequantize(data, qtype, oshape, dtype=None):
    """Dequantize tensor back to usable shape/dtype."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


def dequantize_tensor(tensor, dtype=None, dequant_dtype=None, device=None, scale=None):
    """Dequantize a GGML or FP8 tensor to a plain torch.Tensor.

    Always returns a plain ``torch.Tensor`` (not a GGMLTensor subclass) so that
    standard PyTorch ops (attention, group-norm, etc.) work without surprises.
    """
    global _GGUF_STRATEGY_LOGGED

    if _native_dequantize_tensor is not None:
        if not _GGUF_STRATEGY_LOGGED:
            print("[Trellis2] GGUF: native CUDA dequant (ComfyUI-GGUF ops)", file=sys.stderr)
            _GGUF_STRATEGY_LOGGED = True
        result = _native_dequantize_tensor(tensor, dtype=dtype, dequant_dtype=dequant_dtype)
        if device is not None:
            result = result.to(device)
        if not type(result) is torch.Tensor:
            result = result.as_subclass(torch.Tensor)
        return result

    if isinstance(tensor, torch.nn.Parameter):
        tensor = tensor.data

    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", None)

    if oshape is None:
        if isinstance(tensor, torch.nn.Parameter):
            oshape = getattr(tensor.data, "tensor_shape", tensor.shape)
        else:
            oshape = tensor.shape

    if qtype in TORCH_COMPATIBLE_QTYPES:
        result = tensor.as_subclass(torch.Tensor)
        if qtype in QTYPES_TO_DTYPE and result.dtype != QTYPES_TO_DTYPE[qtype]:
            result = result.view(QTYPES_TO_DTYPE[qtype])
        if oshape is not None:
            result = result.reshape(oshape)
        result = result.to(dtype)
    elif qtype in dequantize_functions:
        if not _GGUF_STRATEGY_LOGGED:
            print("[Trellis2] GGUF: internal fallback dequant", file=sys.stderr)
            _GGUF_STRATEGY_LOGGED = True
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        result = dequantize(tensor.as_subclass(torch.Tensor), qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        new = gguf.quants.dequantize(tensor.cpu().numpy(), qtype)
        result = torch.from_numpy(new).to(tensor.device, dtype=dtype)
        target_numel = torch.Size(oshape).numel()
        if result.numel() != target_numel:
            result = result.reshape(-1)[:target_numel].reshape(oshape)

    if device is not None:
        result = result.to(device)

    if not type(result) is torch.Tensor:
        result = result.as_subclass(torch.Tensor)

    return result


# ── GGMLTensor ────────────────────────────────────────────────────────────────

class GGMLTensor(torch.Tensor):
    """torch.Tensor subclass that carries GGML quantization metadata."""

    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def __new__(cls, data, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, data, **kwargs)

    def to(self, *args, **kwargs):
        tensor_type = getattr(self, "tensor_type", None)
        tensor_shape = getattr(self, "tensor_shape", None)
        patches = getattr(self, "patches", []).copy()
        new = super().to(*args, **kwargs)
        new.tensor_type = tensor_type
        new.tensor_shape = tensor_shape
        new.patches = patches
        return new

    # Avoid copying quantized bytes unnecessarily
    def clone(self, *args, **kwargs): return self
    def detach(self, *args, **kwargs): return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
                new_tensor,
                tensor_type = getattr(self, "tensor_type", None),
                tensor_shape = size,
                patches = getattr(self, "patches", []).copy()
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape") or self.tensor_shape is None:
            self.tensor_shape = torch.Size(super().size())
        return self.tensor_shape

    def size(self, dim=None):
        if dim is not None:
            return self.shape[dim]
        return self.shape

    def dim(self):
        return len(self.shape)

    @property
    def ndim(self):
        return self.dim()


# ── GGMLLayer ─────────────────────────────────────────────────────────────────

class GGMLLayer(torch.nn.Module):
    """Mixin for any nn.Module that may hold GGML-quantized parameters.

    Two important design notes for this codebase:
    - ``MultiHeadRMSNorm`` / ``SparseMultiHeadRMSNorm`` store their learnable
      scale in ``self.gamma`` (not ``self.weight``).  ``cast_bias_weight`` and
      ``is_ggml_quantized`` therefore also inspect ``gamma``.
    - ``nn.Parameter`` strips custom tensor-subclass attributes on construction,
      so the logical shape and quant type are stored separately as
      ``_ggml_weight_shape`` / ``_ggml_weight_type`` on the module.
    - The **native** ComfyUI-GGUF ``GGMLLayer.is_ggml_quantized()`` reads
      ``self.weight`` and ``self.bias`` unconditionally (ignoring its kwargs).
      We set ``bias = None`` at class level so subclasses without a bias param
      (e.g. ``GGMLMultiHeadRMSNorm``) do not raise ``AttributeError``.
    """

    comfy_cast_weights = True
    dequant_dtype = None

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """
        Custom loader to handle GGMLTensor assignment. nn.Parameter strips custom
        attributes, so we detect quantized tensors here and assign them directly
        to bypass the PyTorch copy_() which fails on shape/storage mismatches.
        """
        # MultiHeadRMSNorm / SparseMultiHeadRMSNorm use gamma
        # Linear/LayerNorm/GroupNorm use weight and potential bias
        for name in ("weight", "bias", "gamma"):
            key = f"{prefix}{name}"
            if key in state_dict:
                val = state_dict[key]
                # Detect a GGMLTensor (quantised)
                if hasattr(val, "tensor_type") and val.tensor_type not in {None, 0, 1}:
                    # Store metadata before nn.Parameter strips it
                    setattr(self, f"_ggml_{name}_shape", getattr(val, "tensor_shape", val.shape))
                    setattr(self, f"_ggml_{name}_type",  getattr(val, "tensor_type", None))
                    # Assign directly — do NOT call copy_() on mismatched shapes
                    setattr(self, name, torch.nn.Parameter(val, requires_grad=False))
                    # Consume from state_dict so super()._load_from_state_dict doesn't try to load it
                    state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = getattr(self, "weight", None)
        if bias is None:
            bias = getattr(self, "bias", None)
        # Also check gamma (MultiHeadRMSNorm / SparseMultiHeadRMSNorm)
        gamma = getattr(self, "gamma", None)
        if is_quantized(weight) or is_quantized(bias) or is_quantized(gamma):
            return True
        if hasattr(self, "_ggml_weight_type") and self._ggml_weight_type not in {None, 0, 1}:
            return True
        return False

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return None
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
        if not type(weight) is torch.Tensor:
            weight = weight.as_subclass(torch.Tensor)
        return weight

    @torch_compiler_disable()
    def cast_bias_weight(self, input=None, dtype=None, device=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if device is None:
                device = input.device

        bias = None
        if hasattr(self, "bias") and self.bias is not None:
            bias_data = self.bias.data if isinstance(self.bias, torch.nn.Parameter) else self.bias
            # Ensure device match
            if device is not None and bias_data.device != device:
                bias_data = bias_data.to(device)

            # Restore metadata stripped by nn.Parameter
            stored_shape = getattr(self, "_ggml_bias_shape", None)
            stored_type = getattr(self, "_ggml_bias_type", None)
            if stored_shape is not None and getattr(bias_data, "tensor_shape", None) is None:
                bias_data.tensor_shape = stored_shape
            if stored_type is not None and getattr(bias_data, "tensor_type", None) is None:
                bias_data.tensor_type = stored_type

            bias = self.get_weight(bias_data, dtype)

        weight = None
        # Support both weight (Linear/Norm) and gamma (MultiHeadRMSNorm)
        # Prioritize gamma if both exist (e.g. via alias property)
        has_gamma = hasattr(self, "gamma") and self.gamma is not None
        has_weight = hasattr(self, "weight") and self.weight is not None
        
        meta_prefix = "weight"
        raw_weight = None
        
        if has_gamma:
            raw_weight = self.gamma
            meta_prefix = "gamma"
        elif has_weight:
            raw_weight = self.weight
            meta_prefix = "weight"

        if raw_weight is not None:
            w_data = raw_weight.data if isinstance(raw_weight, torch.nn.Parameter) else raw_weight
            if device is not None and w_data.device != device:
                w_data = w_data.to(device)

            # Restore metadata stripped by nn.Parameter
            stored_shape = getattr(self, f"_ggml_{meta_prefix}_shape", None)
            stored_type = getattr(self, f"_ggml_{meta_prefix}_type", None)
            if stored_shape is not None and getattr(w_data, "tensor_shape", None) is None:
                w_data.tensor_shape = stored_shape
            if stored_type is not None and getattr(w_data, "tensor_type", None) is None:
                w_data.tensor_type = stored_type

            weight = self.get_weight(w_data, dtype)

        return weight, bias


# ── Helper: recover original shape from GGUF metadata ────────────────────────

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


# Initialise native support — FALLBACK IS ALWAYS TRUE for conversion
_setup_native_gguf()
HAS_GGUF_OPS = True


# ── Key-remapping for GGUF files exported with Flux wrapper ──────────────────

def remap_gguf_state_dict(sd: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Remap GGUF state-dict keys from Flux-wrapper naming to Trellis2 naming."""
    arch = metadata.get("general.architecture", "flux")
    if arch != "flux":
        return sd

    new_sd = {}
    remap_count = 0

    mappings = {
        "time_in.": "t_embedder.",
        "guidance_in.": "adaLN_modulation.1.",
        "img_in.": "input_layer.",
        "final_layer.": "out_layer.",
        "blk.": "blocks.",
    }

    block_mappings = {
        ".attn.q.": ".self_attn.to_q.",
        ".attn.k.": ".self_attn.to_k.",
        ".attn.v.": ".self_attn.to_v.",
        ".attn.qkv.": ".self_attn.to_qkv.",
        ".attn.out.": ".self_attn.to_out.",
        ".attn.proj.": ".self_attn.to_proj.",
        ".cross_attn.q.": ".cross_attn.to_q.",
        ".cross_attn.k.": ".cross_attn.to_k.",
        ".cross_attn.v.": ".cross_attn.to_v.",
        ".cross_attn.kv.": ".cross_attn.to_kv.",
        ".cross_attn.out.": ".cross_attn.to_out.",
        ".cross_attn.proj.": ".cross_attn.to_proj.",
        ".ln1.": ".norm1.",
        ".ln2.": ".norm2.",
        ".ln3.": ".norm3.",
    }

    for k, v in sd.items():
        original_k = k
        for old, new in mappings.items():
            if k.startswith(old):
                k = k.replace(old, new, 1)
                break
        if "blocks." in k:
            for old, new in block_mappings.items():
                if old in k:
                    k = k.replace(old, new)
        if k != original_k:
            remap_count += 1
        new_sd[k] = v

    return new_sd


# ── GGUF checkpoint loader ────────────────────────────────────────────────────

def load_gguf_checkpoint(path):
    """Read a GGUF file and return (state_dict, metadata).

    - ``state_dict``: maps tensor names → GGMLTensor (quantized bytes + metadata).
    - ``metadata``: GGUF header fields (strings, ints, floats, bools).
    """
    print(f"[Trellis2] Loading GGUF checkpoint: {os.path.basename(path)}", file=sys.stderr)
    reader = gguf.GGUFReader(path)
    state_dict = {}
    metadata = {}

    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:
                if field.types[0] == gguf.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] == gguf.GGUFValueType.INT32:
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.F32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except Exception:
            continue

    if metadata:
        print(f"[Trellis2]   {len(metadata)} metadata fields", file=sys.stderr)
        for k, v in metadata.items():
            if any(kw in k for kw in ("version", "architecture", "quant")):
                print(f"[Trellis2]     {k}: {v}", file=sys.stderr)

    tensor_counts: Dict[str, int] = {}
    for tensor in reader.tensors:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)

        shape = get_orig_shape(reader, tensor.name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        qtype = tensor.tensor_type.name
        tensor_counts[qtype] = tensor_counts.get(qtype, 0) + 1

        if tensor.tensor_type in QTYPES_TO_DTYPE:
            target_dtype = QTYPES_TO_DTYPE[tensor.tensor_type]
            if torch_tensor.dtype != target_dtype:
                torch_tensor = torch_tensor.view(target_dtype)
            torch_tensor = torch_tensor.view(*shape)

        state_dict[tensor.name] = GGMLTensor(
            torch_tensor,
            tensor_type=tensor.tensor_type,
            tensor_shape=shape,
        )

    state_dict = remap_gguf_state_dict(state_dict, metadata)

    # Convenience: surface latent_channels to metadata callers
    if "flux.latent_channels" in metadata:
        metadata["latent_channels"] = metadata["flux.latent_channels"]
    elif "input_layer.weight" in state_dict:
        metadata["latent_channels"] = state_dict["input_layer.weight"].shape[1]

    return state_dict, metadata


# ── Custom GGML replacement layers ───────────────────────────────────────────

class GGMLSparseLinear(GGMLLayer):
    """Drop-in for nn.Linear that dequantizes GGML weights on each forward."""

    def forward(self, input):
        weight, bias = self.cast_bias_weight(input)
        return torch.nn.functional.linear(input, weight, bias)


class GGMLMultiHeadRMSNorm(GGMLLayer):
    """Drop-in for MultiHeadRMSNorm / SparseMultiHeadRMSNorm.

    These layers store their learnable scale in ``self.gamma`` (shape [H, D]),
    not ``self.weight``.  The RMS-norm formula is:
        out = (x / rms(x)) * gamma * scale   where scale = sqrt(dim)

    A ``weight`` property aliases ``gamma`` so that the native ComfyUI-GGUF
    ``GGMLLayer.is_ggml_quantized()`` (which unconditionally reads
    ``self.weight``) does not raise an AttributeError.
    """

    # bias sentinel (RMSNorm has no bias; native is_ggml_quantized reads self.bias)
    bias = None

    @property
    def weight(self):
        """Alias for gamma so native ComfyUI-GGUF code that reads self.weight works."""
        return getattr(self, "gamma", None)

    @weight.setter
    def weight(self, value):
        self.gamma = value

    def forward(self, x):
        # Dequantize gamma — cast_bias_weight reads self.gamma via the weight alias
        # We must pass a plain tensor to cast_bias_weight; unwrap VarLenTensor first.
        # Use duck-typing so we don't need to import VarLenTensor here.
        is_sparse = hasattr(x, "feats") and hasattr(x, "replace")
        feats = x.feats if is_sparse else x
        gamma, _ = self.cast_bias_weight(feats)
        scale = getattr(self, "scale", feats.shape[-1] ** 0.5)
        # Exact formula matching MultiHeadRMSNorm / SparseMultiHeadRMSNorm:
        #   out = F.normalize(x.float(), dim=-1) * gamma * scale
        if is_sparse:
            x_type = x.dtype
            normed = torch.nn.functional.normalize(feats.float(), dim=-1) * gamma * scale
            return x.replace(normed).to(x_type)
        else:
            x_type = x.dtype
            normed = torch.nn.functional.normalize(x.float(), dim=-1) * gamma * scale
            return normed.to(x_type)


class GGMLGroupNorm32(GGMLLayer):
    """Drop-in for GroupNorm32 / SparseGroupNorm32."""

    def forward(self, x):
        weight, bias = self.cast_bias_weight(x)
        return torch.nn.functional.group_norm(x, self.num_groups, weight, bias, self.eps)


class GGMLLayerNorm32(GGMLLayer):
    """Drop-in for LayerNorm32 / SparseLayerNorm32."""

    def forward(self, x):
        weight, bias = self.cast_bias_weight(x)
        return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


# ── convert_to_ggml ───────────────────────────────────────────────────────────

def convert_to_ggml(module):
    """Recursively replace standard layers with GGML-capable equivalents.

    Skips:
    - ``SparseLinear`` — already inherits ``GGMLLayer`` directly.
    - VAE / Encoder / Decoder / Dino models — never GGUF-quantised.

    Replaces:
    - ``nn.Linear``                              → GGMLSparseLinear
    - ``MultiHeadRMSNorm`` / ``SparseMultiHeadRMSNorm`` → GGMLMultiHeadRMSNorm
    - ``GroupNorm32`` / ``SparseGroupNorm32`` / ``nn.GroupNorm``  → GGMLGroupNorm32
    - ``LayerNorm32`` / ``SparseLayerNorm32`` / ``nn.LayerNorm``  → GGMLLayerNorm32
    - ``nn.Conv2d``                              → GGMLOps.Conv2d (if native available)
    """
    if not HAS_GGUF_OPS:
        return module

    model_class_name = module.__class__.__name__
    if any(x in model_class_name for x in ["VAE", "Encoder", "Decoder", "Dino"]):
        logging.info(f"[Trellis2] Skipping GGUF conversion for: {model_class_name}")
        return module

    for name, child in module.named_children():
        child_class_name = child.__class__.__name__

        new_layer = None

        if child_class_name == "SparseLinear":
            # Already a GGMLLayer — no replacement needed
            continue

        elif isinstance(child, torch.nn.Linear):
            new_layer = GGMLSparseLinear()
            for attr in ("in_features", "out_features"):
                if hasattr(child, attr):
                    setattr(new_layer, attr, getattr(child, attr))

        elif child_class_name in ("MultiHeadRMSNorm", "SparseMultiHeadRMSNorm"):
            # Guard on gamma (not weight — these layers have no weight param)
            if getattr(child, "gamma", None) is not None:
                new_layer = GGMLMultiHeadRMSNorm()
                for attr in ("eps", "num_heads", "head_dim", "scale"):
                    if hasattr(child, attr):
                        setattr(new_layer, attr, getattr(child, attr))

        elif child_class_name in ("GroupNorm32", "SparseGroupNorm32") or isinstance(
            child, torch.nn.GroupNorm
        ):
            if getattr(child, "weight", None) is not None:
                new_layer = GGMLGroupNorm32()
                for attr in ("num_groups", "num_channels", "eps"):
                    if hasattr(child, attr):
                        setattr(new_layer, attr, getattr(child, attr))

        elif child_class_name in ("LayerNorm32", "SparseLayerNorm32") or isinstance(
            child, torch.nn.LayerNorm
        ):
            if getattr(child, "weight", None) is not None:
                new_layer = GGMLLayerNorm32()
                for attr in ("normalized_shape", "eps"):
                    if hasattr(child, attr):
                        setattr(new_layer, attr, getattr(child, attr))

        elif isinstance(child, torch.nn.Conv2d):
            try:
                from gguf_native.ops import GGMLOps
                new_layer = GGMLOps.Conv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    child.padding,
                    child.dilation,
                    child.groups,
                    child.bias is not None,
                )
            except Exception:
                pass

        if new_layer is not None:
            # Transfer gamma (RMSNorm) or weight/bias
            if hasattr(child, "gamma"):
                new_layer.gamma = child.gamma
            if hasattr(child, "weight"):
                new_layer.weight = child.weight
            if hasattr(child, "bias"):
                new_layer.bias = child.bias
            setattr(module, name, new_layer)
        else:
            convert_to_ggml(child)

    return module
