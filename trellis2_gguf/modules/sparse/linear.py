import torch
import torch.nn as nn
import torch.nn.functional as F
from . import VarLenTensor
from ...utils.gguf_utils import GGMLLayer

__all__ = [
    'SparseLinear'
]


def chunked_apply(module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size <= 0 or x.shape[0] <= chunk_size:
        return module(x)
    
    # Process first chunk to determine output shape and dtype
    out_0 = module(x[0:chunk_size])
    out_shape = (x.shape[0],) + out_0.shape[1:]
    out = torch.empty(out_shape, device=x.device, dtype=out_0.dtype)
    out[0:chunk_size] = out_0
    
    # Process remaining chunks
    for i in range(chunk_size, x.shape[0], chunk_size):
        out[i:i+chunk_size] = module(x[i:i+chunk_size])
    return out


class SparseLinear(GGMLLayer, nn.Linear):
    """Sparse linear layer that is GGUF-aware via the GGMLLayer mixin.

    When a GGUF checkpoint is loaded, ``_load_from_state_dict`` detects the
    quantised GGMLTensor, stores the logical shape / quant-type on the module
    (because nn.Parameter strips custom tensor attributes), and assigns the
    parameter directly — bypassing PyTorch's ``copy_()`` which would fail due
    to the shape mismatch between quantised bytes and the logical shape.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features, out_features, bias)
        self.low_vram = False
        self.chunk_size = 65536

    def _forward_feats(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_ggml_quantized():
            weight, bias = self.cast_bias_weight(x)
            return F.linear(x, weight, bias)
        return super(nn.Linear, self).forward(x)  # nn.Linear.forward

    def forward(self, input: VarLenTensor) -> VarLenTensor:
        if self.is_ggml_quantized():
            # GGUF path: dequantize weights then run linear on raw feats
            weight, bias = self.cast_bias_weight(input.feats)
            return input.replace(F.linear(input.feats, weight, bias))
        # Non-GGUF path: cast feats to weight dtype to avoid Float/Half mismatch
        feats = input.feats.to(self.weight.dtype)
        if self.low_vram:
            return input.replace(chunked_apply(super(GGMLLayer, self).forward, feats, self.chunk_size))
        return input.replace(super(GGMLLayer, self).forward(feats))
