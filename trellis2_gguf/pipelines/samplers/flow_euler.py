from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            guidance_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)


class FlowEulerMultiViewSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with multi-view blending.
    """
    def __init__(self, sigma_min: float, resolution: int):
        super().__init__(sigma_min)
        self.resolution = resolution
    
    def _compute_view_weights_sparse(self, coords, views, front_axis='z', blend_temperature=2.0) -> torch.Tensor:
        """
        Compute blending weights for sparse voxels.
        """
        # Normalize coords to [-1, 1] range (roughly)
        z = (coords[:, 1].float() / self.resolution) * 2 - 1.0
        x = (coords[:, 3].float() / self.resolution) * 2 - 1.0
        
        if front_axis == 'z':
            # Front (+Z), Back (-Z), Right (+X), Left (-X)
            view_vectors = {
                'front': torch.stack([torch.zeros_like(z), z], dim=1), # (0, z)
                'back':  torch.stack([torch.zeros_like(z), -z], dim=1),
                'right': torch.stack([x, torch.zeros_like(x)], dim=1),
                'left':  torch.stack([-x, torch.zeros_like(x)], dim=1),
            }
        else: # front_axis == 'x' (swapped)
            # Front (+X), Back (-X), Right (+Z), Left (-Z)
             view_vectors = {
                'front': torch.stack([x, torch.zeros_like(x)], dim=1),
                'back':  torch.stack([-x, torch.zeros_like(x)], dim=1),
                'right': torch.stack([torch.zeros_like(z), z], dim=1),
                'left':  torch.stack([torch.zeros_like(z), -z], dim=1),
            }

        scores = []
        for view in views:
            if view in view_vectors:
                v_vec = view_vectors[view]
                score = v_vec.sum(dim=1)
                scores.append(score)
            else:
                scores.append(torch.full_like(z, -10.0))
        
        scores = torch.stack(scores, dim=1) # (N, num_views)
        weights = torch.softmax(scores * blend_temperature, dim=1)
        return weights

    def _compute_view_weights_dense(self, shape, device, views, front_axis='z', blend_temperature=2.0) -> torch.Tensor:
        """
        Compute blending weights for dense grid (B, C, D, H, W).
        Returns weights of shape (1, 1, D, H, W, NumViews) for easy broadcasting (actually we want (1, 1, D, H, W) per view)
        """
        # shape is (B, C, D, H, W)
        D, H, W = shape[2], shape[3], shape[4]
        
        # Create meshgrid in [-1, 1]
        # We assume D is Z axis, W is X axis (usually D, H, W = Z, Y, X in 3D tensors?)
        # Let's verify standard: (Batch, Channel, Depth, Height, Width) -> (B, C, Z, Y, X)
        
        dz = torch.linspace(-1, 1, D, device=device)
        dy = torch.linspace(-1, 1, H, device=device)
        dx = torch.linspace(-1, 1, W, device=device)
        
        # meshgrid 'ij' indexing: (D, H, W) order
        grid_z, grid_y, grid_x = torch.meshgrid(dz, dy, dx, indexing='ij') 
        
        # Flatten for vector calc? Or keep structural. Keep structural.
        
        if front_axis == 'z':
             # Front (+Z), Back (-Z), Right (+X), Left (-X)
             # Vectors are scalar fields here
             view_scores = {
                 'front': grid_z,
                 'back': -grid_z,
                 'right': grid_x,
                 'left': -grid_x,
             }
        else:
             view_scores = {
                 'front': grid_x,
                 'back': -grid_x,
                 'right': grid_z,
                 'left': -grid_z,
             }
             
        scores = []
        for view in views:
            if view in view_scores:
                scores.append(view_scores[view])
            else:
                scores.append(torch.full_like(grid_z, -10.0))
                
        # Stack: (NumViews, D, H, W)
        scores = torch.stack(scores, dim=0) 
        
        # Softmax over views dimension (0)
        weights = torch.softmax(scores * blend_temperature, dim=0)
        
        # Reshape for broadcasting: (NumViews, 1, 1, D, H, W) -> No wait, loop is over views.
        # We want to return something we can index like weights[i] -> (1, 1, D, H, W)
        
        # Current shape: (NumViews, D, H, W)
        return weights

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        conds: Dict[str, Any], # Changed: expects dict of {view: cond}
        views: List[str],      # Changed: list of view keys corresponding to conds
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        **kwargs
    ):
        """
        Sample with multi-view blending.
        """
        is_sparse = hasattr(x_t, 'coords')
        
        if is_sparse:
            # 1. Compute per-voxel weights based on current sparse coords
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
            # weights: (N, NumViews)
        else:
            # Dense tensor (B, C, D, H, W)
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            # weights: (NumViews, D, H, W)
        
        # 2. Run model for each view and blend predictions
        pred_v_accum = 0
        
        for i, view in enumerate(views):
            cond = conds[view]
            # Use _inference_model to support mixins (CFG, etc)
            # If cond is a dict containing 'cond' and 'neg_cond' (from pipeline.get_cond), unpack it
            if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                pred_v_view = self._inference_model(model, x_t, t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
            else:
                pred_v_view = self._inference_model(model, x_t, t, cond=cond, **kwargs)
            
            # Weighted accumulation
            if is_sparse:
                # weights[:, i] is (N,), pred_v_view might be SparseTensor or Tensor (N, C)
                w = weights[:, i].unsqueeze(1)
                
                v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                pred_v_accum += v_feats * w
            else:
                # Dense
                # weights[i] is (D, H, W). pred_v_view is (B, C, D, H, W)
                w = weights[i].unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
                pred_v_accum += pred_v_view * w
                
        if is_sparse:
            # Re-wrap accumulated features into a SparseTensor matching x_t
            # pred_v_accum is (N, C) tensor now
            pred_v = x_t.replace(feats=pred_v_accum)
        else:
            pred_v = pred_v_accum
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)

        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        conds: Dict[str, Any], # {view: cond}
        views: List[str],      # ['front', 'back', ...]
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling MultiView",
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        **kwargs
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.sample_once(
                model, sample, t, t_prev, 
                conds=conds, 
                views=views,
                front_axis=front_axis, 
                blend_temperature=blend_temperature, 
                **kwargs
            )
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerMultiViewGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerMultiViewSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with multi-view blending, CFG, and guidance interval.
    """
    pass

# RK4 and RK5 Samplers

class FlowRK4Sampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using the 4th-order Runge-Kutta method.
    """
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        dt = t_prev - t
        
        # Helper to extract just the velocity prediction
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v

        # RK4 intermediate slopes
        k1 = get_v(x_t, t)
        k2 = get_v(x_t + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = get_v(x_t + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = get_v(x_t + dt * k3, t + dt)
        
        # RK4 integration
        pred_x_prev = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        # We need to return pred_x_0 as well to satisfy the pipeline's logging/tracking
        # We compute x_start_eps based on the k1 velocity (equivalent to the Euler estimation of x_0)
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK5Sampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Butcher's 5th-order Runge-Kutta method.
    """
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        dt = t_prev - t
        
        # Helper to extract just the velocity prediction
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v

        # Intermediate time step fractions for Butcher's RK5
        c2, c3, c4, c5, c6 = 1/4, 1/4, 1/2, 3/4, 1.0
        
        k1 = get_v(x_t, t)
        k2 = get_v(x_t + dt * (1/4 * k1), t + dt * c2)
        k3 = get_v(x_t + dt * (1/8 * k1 + 1/8 * k2), t + dt * c3)
        k4 = get_v(x_t + dt * (-1/2 * k2 + 1.0 * k3), t + dt * c4)
        k5 = get_v(x_t + dt * (3/16 * k1 + 9/16 * k4), t + dt * c5)
        k6 = get_v(x_t + dt * (-3/7 * k1 + 2/7 * k2 + 12/7 * k3 - 12/7 * k4 + 8/7 * k5), t + dt * c6)
        
        # Final RK5 Integration
        pred_x_prev = x_t + dt * (7/90 * k1 + 32/90 * k3 + 12/90 * k4 + 32/90 * k5 + 7/90 * k6)
        
        # Estimate x_0 based on k1 for tracking
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


# --- Classifier Free Guidance (CFG) Wrappers ---
class FlowRK4CfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowRK4Sampler):
    """RK4 sampling with classifier-free guidance."""
    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0, guidance_strength: float = 3.0, verbose: bool = True, **kwargs):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)

class FlowRK5CfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowRK5Sampler):
    """RK5 sampling with classifier-free guidance."""
    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps: int = 50, rescale_t: float = 1.0, guidance_strength: float = 3.0, verbose: bool = True, **kwargs):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)
        
class FlowRK4GuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK4Sampler):
    """RK4 with CFG and Guidance Intervals."""
    pass

class FlowRK5GuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK5Sampler):
    """RK5 with CFG and Guidance Intervals."""
    pass        
    
# RK4 and RK5 for MultiView

class FlowRK4MultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view flow matching using 4th-order Runge-Kutta."""
    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, 
        conds: Dict[str, Any], views: List[str], 
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        
        # Calculate spatial blending weights ONCE for the current step
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            
        # Helper function to compute the blended velocity for a given intermediate x and t
        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                
                if is_sparse:
                    w = weights[:, i].unsqueeze(1)
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum += v_feats * w
                else:
                    w = weights[i].unsqueeze(0).unsqueeze(0)
                    pred_v_accum += pred_v_view * w
                    
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            else:
                return pred_v_accum

        # RK4 Evaluations
        k1 = get_blended_v(x_t, t)
        k2 = get_blended_v(x_t + k1 * (0.5 * dt), t + 0.5 * dt)
        k3 = get_blended_v(x_t + k2 * (0.5 * dt), t + 0.5 * dt)
        k4 = get_blended_v(x_t + k3 * dt, t + dt)
        
        pred_x_prev = x_t + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6.0)
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK5MultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view flow matching using Butcher's 5th-order Runge-Kutta."""
    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, 
        conds: Dict[str, Any], views: List[str], 
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            
        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                
                if is_sparse:
                    w = weights[:, i].unsqueeze(1)
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum += v_feats * w
                else:
                    w = weights[i].unsqueeze(0).unsqueeze(0)
                    pred_v_accum += pred_v_view * w
                    
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            else:
                return pred_v_accum

        # Butcher Tableau Intermediate steps
        c2, c3, c4, c5, c6 = 1/4, 1/4, 1/2, 3/4, 1.0
        
        k1 = get_blended_v(x_t, t)
        k2 = get_blended_v(x_t + k1 * (dt * 1/4), t + dt * c2)
        k3 = get_blended_v(x_t + (k1 * 1/8 + k2 * 1/8) * dt, t + dt * c3)
        k4 = get_blended_v(x_t + (k2 * -1/2 + k3 * 1.0) * dt, t + dt * c4)
        k5 = get_blended_v(x_t + (k1 * 3/16 + k4 * 9/16) * dt, t + dt * c5)
        k6 = get_blended_v(x_t + (k1 * -3/7 + k2 * 2/7 + k3 * 12/7 + k4 * -12/7 + k5 * 8/7) * dt, t + dt * c6)
        
        pred_x_prev = x_t + (k1 * 7/90 + k3 * 32/90 + k4 * 12/90 + k5 * 32/90 + k6 * 7/90) * dt
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK4MultiViewGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK4MultiViewSampler):
    pass

class FlowRK5MultiViewGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK5MultiViewSampler):
    pass

# Heun (RK2)

class FlowHeunSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Heun's Method (2nd-order Runge-Kutta).
    Requires 2 NFEs per step.
    """
    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        dt = t_prev - t
        
        # Helper to extract just the velocity prediction
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v

        # Step 1: Predictor (Euler step)
        k1 = get_v(x_t, t)
        x_temp = x_t + k1 * dt
        
        # Step 2: Corrector
        k2 = get_v(x_temp, t + dt)
        
        # Average the two velocities for the final step
        pred_x_prev = x_t + 0.5 * dt * (k1 + k2)
        
        # Estimate x_0 based on k1 for tracking/logging
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

# --- CFG Wrapper for Heun ---
class FlowHeunGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowHeunSampler):
    """Heun sampling with CFG and Guidance Intervals."""
    pass
    
class FlowHeunMultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view flow matching using Heun's method (2nd-order Runge-Kutta)."""
    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float, 
        conds: Dict[str, Any], views: List[str], 
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        
        # Calculate spatial blending weights ONCE for the current step
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)
            
        # Helper function to compute the blended velocity for a given intermediate x and t
        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                
                if is_sparse:
                    w = weights[:, i].unsqueeze(1)
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum += v_feats * w
                else:
                    w = weights[i].unsqueeze(0).unsqueeze(0)
                    pred_v_accum += pred_v_view * w
                    
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            else:
                return pred_v_accum

        # Heun's Method (RK2) Evaluations
        # Step 1: Predictor (Euler step)
        k1 = get_blended_v(x_t, t)
        x_temp = x_t + k1 * dt
        
        # Step 2: Corrector
        k2 = get_blended_v(x_temp, t + dt)
        
        # Combine
        pred_x_prev = x_t + 0.5 * dt * (k1 + k2)
        
        # Estimate x_0 based on k1 for tracking
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

# --- CFG Wrapper for Heun Multi-View ---
class FlowHeunMultiViewGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowHeunMultiViewSampler):
    pass

