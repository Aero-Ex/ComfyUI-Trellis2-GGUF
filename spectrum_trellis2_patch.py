import torch
import torch.nn as nn
import numpy as np
import inspect
from typing import Optional, Tuple, Union, Dict, List, Any
from easydict import EasyDict as edict

# --- Core Spectrum Math (High Precision Variant) ---

# We enforce float32 for all mathematical operations to reach 100% parity potential
MATH_DTYPE = torch.float32

def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    shape = x.shape
    return x.reshape(1, -1) if x.ndim == 1 else x.reshape(1, -1), shape

def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return x_flat.reshape(shape)

class HighPrecisionForecaster(nn.Module):
    def __init__(self, M: int = 3, K: int = 10, lam: float = 1e-3, feature_shape = None):
        super().__init__()
        self.M = M
        self.K = K
        self.lam = lam
        self.register_buffer("t_buf", torch.empty(0, dtype=MATH_DTYPE))
        self._H_buf = None # (K, F) in float32
        self._shape = None
        self._coef = None  # (P, F) in float32
        self.feature_shape = feature_shape

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        """Fixed global range mapping [0, 1000] to [-1, 1] for Chebyshev stability."""
        t_min = torch.tensor(0.0, device=t.device, dtype=MATH_DTYPE)
        t_max = torch.tensor(1000.0, device=t.device, dtype=MATH_DTYPE)
        mid = 0.5 * (t_min + t_max)
        rng = (t_max - t_min)
        return (t.to(MATH_DTYPE) - mid) * 2.0 / rng

    def update(self, t: float | torch.Tensor, h: torch.Tensor) -> None:
        device = h.device
        t = torch.as_tensor(t, dtype=MATH_DTYPE, device=device)
        h_flat, shape = _flatten(h)
        h_flat = h_flat.to(MATH_DTYPE) # Higher precision for buffer
        
        if self._shape is None:
            self._shape = shape
        
        if self.t_buf.numel() == 0:
            self.t_buf = t[None]
            self._H_buf = h_flat
        else:
            self.t_buf = torch.cat([self.t_buf, t[None]], dim=0)
            self._H_buf = torch.cat([self._H_buf, h_flat], dim=0)
            if self.t_buf.numel() > self.K:
                self.t_buf = self.t_buf[-self.K:]
                self._H_buf = self._H_buf[-self.K:]
        self._coef = None

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        taus = taus.reshape(-1, 1).to(MATH_DTYPE)
        K = taus.shape[0]
        T0 = torch.ones((K, 1), device=taus.device, dtype=MATH_DTYPE)
        if self.M == 0: return T0
        T1 = taus
        cols = [T0, T1]
        for m in range(2, self.M + 1):
            Tm = 2.0 * taus * cols[-1] - cols[-2]
            cols.append(Tm)
        return torch.cat(cols[: self.M + 1], dim=1)

    def _fit_if_needed(self) -> None:
        if self._coef is not None:
            return
        taus = self._taus(self.t_buf)
        X = self._build_design(taus) # (K, P)
        H = self._H_buf            # (K, F)
        P = X.shape[1]
        
        # Ridge Regression solve in float32
        lamI = self.lam * torch.eye(P, device=X.device, dtype=MATH_DTYPE)
        Xt = X.transpose(0, 1)
        XtX = Xt @ X + lamI
        XtH = Xt @ H
        
        # Using linalg.solve for better precision than manual cholesky inverse
        try:
            self._coef = torch.linalg.solve(XtX, XtH)
        except:
            # Fallback for singular matrix
            jitter = 1e-6 * XtX.diag().mean()
            self._coef = torch.linalg.solve(XtX + jitter * torch.eye(P, device=X.device), XtH)

    def predict(self, t_star: float | torch.Tensor) -> torch.Tensor:
        device = self.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=MATH_DTYPE, device=device)
        self._fit_if_needed()
        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None])
        h_flat = x_star @ self._coef
        return _unflatten(h_flat, self._shape)

class HighPrecisionSpectrum(nn.Module):
    def __init__(self, forecaster, w: float = 0.5):
        super().__init__()
        self.forecaster = forecaster
        self.w = w

    def _local_taylor_2nd_order(self, t_star: torch.Tensor) -> torch.Tensor:
        H = self.forecaster._H_buf
        t = self.forecaster.t_buf
        if t.numel() < 2:
            return H[-1].clone()
        
        # Consistent with Spectrum repo's fractional k logic
        h_i = H[-1]; t_i = t[-1]
        h_im1 = H[-2]; t_im1 = t[-2]
        
        dt_last = (t_i - t_im1) # Float32 maintain sign
        if dt_last.abs() < 1e-8: dt_last = torch.sign(dt_last) * 1e-8 if dt_last != 0 else torch.tensor(1e-8, device=t.device)
        
        k = (t_star - t_i) / dt_last
        
        # 1st order change
        dh1 = (h_i - h_im1)
        out = h_i + k * dh1
        
        # 2nd order correction (Newton forward form)
        if t.numel() >= 3:
            h_im2 = H[-3]
            d2 = (h_i - 2.0 * h_im1 + h_im2)
            out = out + 0.5 * k * (k - 1.0) * d2
            
        return out

    def predict(self, t_star: float | torch.Tensor):
        device = self.forecaster.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=MATH_DTYPE, device=device)
        
        h_cheb = self.forecaster.predict(t_star)
        h_taylor = _unflatten(self._local_taylor_2nd_order(t_star), self.forecaster._shape)
        
        h_mix = (1.0 - self.w) * h_taylor + self.w * h_cheb
        return h_mix

    def update(self, t, h):
        return self.forecaster.update(t, h)

# --- Dual-Stream Implementation ---

def patch_spectrum(model: nn.Module, w: float = 0.5, M: int = 4, interval: int = 3, warmup: int = 3):
    """
    Monkey-patches a Trellis2 model with High-Precision Dual-Stream Spectrum.
    """
    if hasattr(model, '_spectrum_patched'):
        return model
        
    orig_forward = model.forward
    
    # We maintain TWO completely separate mathematical states to prevent CFG bleed
    model._spectrum_streams = {
        'pos': {'forecaster': None, 'step_count': 0},
        'neg': {'forecaster': None, 'step_count': 0},
    }
    # Parameters tuned for maximum balance between speed and 100% parity goal
    model._spectrum_config = {
        'interval': interval,
        'warmup': warmup,
        'w': w,
        'M': M,
        'K': 10,
        'lam': 1e-3,
    }
    model._spectrum_patched = True

    def spectrum_forward(self, x, t, cond=None, *args, **kwargs):
        # 1. State Identification (Dual-Stream logic)
        t_scalar = t[0].item() if torch.is_tensor(t) and t.ndim > 0 else (t.item() if torch.is_tensor(t) else t)
        
        if not hasattr(self, '_spectrum_last_t') or abs(self._spectrum_last_t - t_scalar) > 1e-4:
            self._spectrum_last_t = t_scalar
            self._spectrum_call_idx = 0
        else:
            self._spectrum_call_idx += 1
            
        stream_id = 'pos' if self._spectrum_call_idx == 0 else 'neg'
        stream = self._spectrum_streams[stream_id]
        cfg = self._spectrum_config
        step = stream['step_count']
        
        is_warmup = step < cfg['warmup']
        is_check = (step - cfg['warmup']) % (cfg['interval'] + 1) == 0 if not is_warmup else True
        
        if is_warmup or is_check:
            # ACTUAL FORWARD (MATERIALIZATION)
            # Filter kwargs for strict signatures
            sig = inspect.signature(orig_forward)
            has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            f_kwargs = kwargs if has_var_kwargs else {k: v for k, v in kwargs.items() if k in sig.parameters}
            
            out = orig_forward(x, t, cond=cond, *args, **f_kwargs)
            feat = out.feats if hasattr(out, 'feats') else out
            
            # Update high-precision forecaster
            if stream['forecaster'] is None:
                forecaster = HighPrecisionForecaster(M=cfg['M'], K=cfg['K'], lam=cfg['lam'], feature_shape=feat.shape[1:])
                stream['forecaster'] = HighPrecisionSpectrum(forecaster, w=cfg['w'])
            
            stream['forecaster'].update(t_scalar, feat)
            stream['step_count'] += 1
            return out
        else:
            # SPECTRUM FORECAST (ACCELERATION)
            print(f"[Spectrum] Forecasting {stream_id} step {step} (t={t_scalar:.1f})")
            pred_feat = stream['forecaster'].predict(t_scalar).to(x.dtype if hasattr(x, 'dtype') else feat.dtype)
            
            if hasattr(x, 'replace'):
                out = x.replace(feats=pred_feat)
            else:
                out = pred_feat
            
            stream['step_count'] += 1
            return out

    model.forward = spectrum_forward.__get__(model, model.__class__)
    return model

def apply_spectrum_to_pipeline(pipeline, steps: Dict[str, int], w: float = 0.6, interval: int = 1):
    """
    Applies high-precision Spectrum.
    Interval=1 is recommended for 100% parity goal in low-step models.
    """
    if hasattr(pipeline, 'models'):
        m = pipeline.models
        if m.get('sparse_structure_flow_model') is not None:
            patch_spectrum(m['sparse_structure_flow_model'], w=w, interval=interval)
        if m.get('shape_slat_flow_model_512') is not None:
            patch_spectrum(m['shape_slat_flow_model_512'], w=w, interval=interval)
        if m.get('tex_slat_flow_model_512') is not None:
            patch_spectrum(m['tex_slat_flow_model_512'], w=w, interval=interval)
            
    return pipeline
