from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline, _find_sdnq_model_dir
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel

from .. import models

import gc
import os
import folder_paths
import trimesh
import o_voxel
import cumesh
import nvdiffrast.torch as dr
import cv2
import flex_gemm
from flex_gemm.ops.grid_sample import grid_sample_3d

import random

from comfy.utils import ProgressBar

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import contextlib
@contextlib.contextmanager
def hide_src_module():
    import sys
    import os
    import torch
    
    # Patch valeoai_NAF_main to be a regular package so it takes precedence over other nodes
    hub_dir = torch.hub.get_dir()
    naf_dir = os.path.join(hub_dir, "valeoai_NAF_main")
    if os.path.exists(naf_dir):
        for d in ["src", "src/model"]:
            init_path = os.path.join(naf_dir, d, "__init__.py")
            if os.path.exists(os.path.join(naf_dir, d)) and not os.path.exists(init_path):
                open(init_path, 'w').close()

    old_src_modules = {}
    for k in list(sys.modules.keys()):
        if k == 'src' or k.startswith('src.'):
            old_src_modules[k] = sys.modules.pop(k)
    try:
        yield
    finally:
        for k, v in old_src_modules.items():
            sys.modules[k] = v


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    # model_names_to_load = [
        # 'sparse_structure_flow_model',
        # 'sparse_structure_decoder',
        # 'shape_slat_flow_model_512',
        # 'shape_slat_flow_model_1024',
        # 'shape_slat_decoder',
        # 'tex_slat_flow_model_512',
        # 'tex_slat_flow_model_1024',
        # 'tex_slat_decoder',
    # ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self._low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.precision = None
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'
        
        self.PIXAL3D_IMAGE_COND_CONFIGS = {
            "ss": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 512,
                "grid_resolution": 16,
            },
            "shape_512": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 512,
                "grid_resolution": 32,
                "use_naf_upsample": True,
                "naf_target_size": 512,
            },
            "shape_1024": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 1024,
                "grid_resolution": 64,
                "use_naf_upsample": True,
                "naf_target_size": 512,
            },
            "tex_1024": {
                "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "image_size": 1024,
                "grid_resolution": 64,
                "use_naf_upsample": True,
                "naf_target_size": 1024,
            },
        }

    @property
    def low_vram(self) -> bool:
        return self._low_vram

    @low_vram.setter
    def low_vram(self, value: bool):
        self._low_vram = value
        for m in self.models.values():
            if hasattr(m, 'low_vram'):
                m.low_vram = value
        if hasattr(self, 'image_cond_model') and hasattr(self.image_cond_model, 'low_vram'):
            self.image_cond_model.low_vram = value

    def _cond_to(self, cond: dict, device: torch.device) -> dict:
        # Move only tensors; keep other items unchanged
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond.items()}

    def _cond_cpu(self, cond: dict) -> dict:
        return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in cond.items()}

    def _cleanup_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _get_sampler(self, sampler_name: Optional[str], default_sampler: samplers.Sampler, **kwargs) -> samplers.Sampler:
        """
        Get a sampler by name, or return the default sampler.
        """
        if sampler_name is None or sampler_name == "" or sampler_name == "default":
            return default_sampler
        
        # Determine the base class name based on the default sampler
        base_class = default_sampler.__class__.__name__
        
        # Map simple names to class prefixes
        name_map = {
            "euler": "FlowEuler",
            "heun": "FlowHeun",
            "rk4": "FlowRK4",
            "rk5": "FlowRK5",
        }
        prefix = name_map.get(sampler_name.lower(), "FlowEuler")
        
        # Construct the target class name by replacing the prefix
        # e.g., FlowEulerGuidanceIntervalSampler -> FlowRK4GuidanceIntervalSampler
        if base_class.startswith("FlowEuler"):
            target_class_name = base_class.replace("FlowEuler", prefix)
        elif base_class.startswith("FlowHeun"):
            target_class_name = base_class.replace("FlowHeun", prefix)
        elif base_class.startswith("FlowRK4"):
            target_class_name = base_class.replace("FlowRK4", prefix)
        elif base_class.startswith("FlowRK5"):
            target_class_name = base_class.replace("FlowRK5", prefix)
        else:
            # Fallback if the base class doesn't match expected patterns
            target_class_name = prefix + "Sampler"
            
        target_class = getattr(samplers, target_class_name, None)
        if target_class is None:
            print(f"[Trellis2] Warning: Sampler {target_class_name} not found, using default {base_class}")
            return default_sampler
            
        # Initialize the new sampler with parameters from the default one if possible
        # Most samplers in this repo take sigma_min in __init__
        try:
            init_args = {'sigma_min': getattr(default_sampler, 'sigma_min', 1e-3)}
            # If the default sampler has other attributes like resolution, copy them
            for attr in ['resolution']:
                if hasattr(default_sampler, attr):
                    init_args[attr] = getattr(default_sampler, attr)
            # Override with explicitly passed kwargs
            init_args.update(kwargs)
            return target_class(**init_args)
        except Exception as e:
            print(f"[Trellis2] Warning: Could not initialize {target_class_name}: {e}")
            return default_sampler

    def GetSamplerName(self, sampler_name: Optional[str]) -> str:
        """
        Helper to normalize sampler names.
        """
        if sampler_name is None or sampler_name == "" or sampler_name == "default":
            return "Euler"
        name_map = {
            "euler": "Euler",
            "heun": "Heun",
            "rk4": "RK4",
            "rk5": "RK5",
        }
        return name_map.get(sampler_name.lower(), "Euler")

    def move_all_to_cpu(self):
        """Move all models in the pipeline to CPU."""
        print("[Trellis2] Offloading all models to CPU...")
        for name, model in self.models.items():
            if model is not None:
                model.cpu()
        if self.image_cond_model is not None:
            self.image_cond_model.cpu()
        if self.rembg_model is not None:
            self.rembg_model.cpu()
        self._cleanup_cuda()

    def _sdnq_remap(self, path: str) -> str:
        """Remap a bf16 model path to its SDNQ directory if enable_sdnq is set."""
        if not getattr(self, 'enable_sdnq', False):
            return path
        sdnq_dir = _find_sdnq_model_dir(path, svd_rank=getattr(self, 'sdnq_svd_rank', 32))
        if sdnq_dir:
            print(f'[Trellis2-SDNQ] {os.path.basename(path)} → SDNQ: {os.path.basename(sdnq_dir)}')
            return sdnq_dir
        print(f'[Trellis2-SDNQ] WARNING: No SDNQ dir for {os.path.basename(path)}, using original')
        return path

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json", keep_models_loaded = True,
                        enable_gguf: bool = False, gguf_quant: str = "Q8_0", precision: str = None,
                        enable_sdnq: bool = False, sdnq_use_quantized_matmul: bool = True,
                        sdnq_torch_compile: bool = False,
                        sdnq_svd_rank: int = 32,
                        isPixal3D: bool = False) -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        #pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        #pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.image_cond_model = None
        pipeline.rembg_model = None
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'
        pipeline.path = path
        pipeline.keep_models_loaded = keep_models_loaded
        pipeline.last_processing = ''
        pipeline.enable_gguf = enable_gguf
        pipeline.gguf_quant = gguf_quant
        pipeline.precision = precision
        pipeline.enable_sdnq = enable_sdnq
        pipeline.sdnq_use_quantized_matmul = sdnq_use_quantized_matmul
        pipeline.sdnq_torch_compile = sdnq_torch_compile
        pipeline.sdnq_svd_rank = sdnq_svd_rank
        pipeline.isPixal3D = isPixal3D

        pipeline._pretrained_args['models']['sparse_structure_decoder'] = os.path.join(folder_paths.models_dir,"Trellis2","decoders","Stage1","ss_dec_conv3d_16l8_fp16")
        # Check both the new consolidated location and the old legacy location for DINOv3
        dinov3_new = os.path.join(folder_paths.models_dir,"Trellis2","dinov3","facebook","dinov3-vitl16-pretrain-lvd1689m")
        dinov3_old = os.path.join(folder_paths.models_dir,"Aero-Ex","Dinov3","facebook","dinov3-vitl16-pretrain-lvd1689m")
        dinov3_user = os.path.join(folder_paths.models_dir,"facebook","dinov3-vitl16-pretrain-lvd1689m")
        if os.path.exists(dinov3_new):
            facebook_model_path = dinov3_new
        elif os.path.exists(dinov3_user):
            facebook_model_path = dinov3_user
        else:
            facebook_model_path = dinov3_old
        pipeline._pretrained_args['image_cond_model']['args']['model_name'] = facebook_model_path
        
        for k in pipeline.PIXAL3D_IMAGE_COND_CONFIGS:
            pipeline.PIXAL3D_IMAGE_COND_CONFIGS[k]["model_name"] = facebook_model_path

        return pipeline
        
    def load_moge_model(self):
        if hasattr(self,'moge_model') and self.moge_model is not None:
            return self.moge_model
        
        model_name = "Ruicheng/moge-2-vitl"
        moge_model_path = os.path.join(folder_paths.models_dir, "Ruicheng","moge-2-vitl")
        
        if not os.path.exists(moge_model_path):
            print(f"Downloading MoGe model to: {moge_model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_name,
                local_dir=moge_model_path,
                local_dir_use_symlinks=False,
            )
        
        moge_model_path = os.path.join(moge_model_path,'model.pt')
        
        print('Loading MoGe model ...')
        try:
            from moge.model.v2 import MoGeModel
        except ImportError:
            # Fallback for relative import if running inside the GGUF module
            import sys
            trellis2_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "ComfyUI-Trellis2"))
            if trellis2_path not in sys.path:
                sys.path.append(trellis2_path)
            from moge.model.v2 import MoGeModel
            
        self.moge_model = MoGeModel.from_pretrained(moge_model_path).to(self.device)
        self.moge_model.eval()

    def unload_moge_model(self):
        if hasattr(self,'moge_model') and self.moge_model is not None:
            del self.moge_model
            self.moge_model = None
            self._cleanup_cuda() 

    def get_moge_camera_config(self, image):
        try:
            from trellis2.utils.camera import get_camera_params_wild_moge
        except ImportError:
            import importlib.util
            import os
            camera_py_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "ComfyUI-Trellis2", "trellis2", "utils", "camera.py"))
            spec = importlib.util.spec_from_file_location("trellis2_camera", camera_py_path)
            trellis2_camera = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(trellis2_camera)
            get_camera_params_wild_moge = trellis2_camera.get_camera_params_wild_moge
            
        camera_config = get_camera_params_wild_moge(image, self.moge_model)
        print(camera_config)
        return camera_config

    def _get_build_pixal3d_image_cond_model(self):
        import sys
        for name, module in list(sys.modules.items()):
            if name.startswith('torch.') or name.startswith('torchvision.'): continue
            try:
                if hasattr(module, 'build_pixal3d_image_cond_model'):
                    func = getattr(module, 'build_pixal3d_image_cond_model')
                    if callable(func):
                        return func
            except Exception:
                pass
        raise ImportError("Could not find build_pixal3d_image_cond_model in loaded modules. Make sure ComfyUI-Trellis2 is updated and installed.")

    def get_proj_cond_ss(
        self,
        image: list,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
        image_cond_model = None
    ) -> dict:
        """
        Get proj conditioning for sparse structure stage.

        Args:
            image: List of PIL images.
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': ...}
        """
        print('Getting Proj Image Cond ...')
        device = self.device
        # Use MoGe to estimate camera params from the image when available
        try:
            self.load_moge_model()
            cfg = self.get_moge_camera_config(image[0] if isinstance(image, list) else image)
            camera_angle_x = cfg['camera_angle_x']
            distance = cfg['distance']
            mesh_scale = cfg['mesh_scale']
            if self.low_vram:
                self.unload_moge_model()
        except Exception as e:
            print(f"[MoGe] Falling back to defaults: {e}")
        if self.low_vram:
            image_cond_model.to(device)
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        with hide_src_module():
            z_global, z_proj = image_cond_model(
                image, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor,
            )
        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': torch.zeros_like(z_proj)},
        }

    @torch.no_grad()
    def get_proj_cond_shape(
        self,
        image_cond_model: nn.Module,
        image: list,
        coords: torch.Tensor,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
        grid_resolution_override: int = None,
    ) -> dict:
        """
        Get proj conditioning for shape/texture stages (sparse-token aligned).

        Args:
            image_cond_model: The proj image cond model for this stage.
            image: List of PIL images.
            coords: Sparse structure coordinates [N, 4] (batch_idx, x, y, z).
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.
            grid_resolution_override: Override the grid resolution if not None.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': SparseTensor}
        """
        print('Getting Projected Image Cond ...')
        device = self.device
        # Use MoGe to estimate camera params from the image when available
        try:
            self.load_moge_model()
            cfg = self.get_moge_camera_config(image[0] if isinstance(image, list) else image)
            camera_angle_x = cfg['camera_angle_x']
            distance = cfg['distance']
            mesh_scale = cfg['mesh_scale']
            if self.low_vram:
                self.unload_moge_model()
        except Exception as e:
            print(f"[MoGe] Falling back to defaults: {e}")
        target_size = getattr(image_cond_model, 'naf_target_size', 512)
        is_texture_stage = False
        if isinstance(target_size, (list, tuple)):
            is_texture_stage = any(x == 1024 for x in target_size)
        else:
            is_texture_stage = target_size == 1024

        if self.low_vram and not is_texture_stage:
            image_cond_model.to(device)
            image_cond_model.naf_tile_factor = 4
        else:
            if self.low_vram:
                image_cond_model.to(device)
            image_cond_model.naf_tile_factor = 1

        orig_grid_res = image_cond_model.grid_resolution
        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            image_cond_model.grid_resolution = grid_resolution_override
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=grid_resolution_override,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        B = 1
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        # monkeypatch the forward of this specific image_cond_model instance
        # or its class if it hasn't been monkeypatched yet!
        if not hasattr(image_cond_model, '_monkeypatched_for_cpu'):
            original_forward = image_cond_model.forward.__func__ if hasattr(image_cond_model.forward, '__func__') else image_cond_model.forward
            
            def optimized_forward_instance(self_instance, image, camera_angle_x=None, distance=None, mesh_scale=None, transform_matrix=None):
                if self_instance.grid_resolution < 32:
                    return original_forward(self_instance, image, camera_angle_x, distance, mesh_scale, transform_matrix)
                
                import torch
                import numpy as np
                from PIL import Image

                if isinstance(image, torch.Tensor):
                    assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
                elif isinstance(image, list):
                    assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
                    image = [i.resize((self_instance.image_size, self_instance.image_size), Image.LANCZOS) for i in image]
                    image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
                    image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
                    image = torch.stack(image).cuda()
                else:
                    raise ValueError(f"Unsupported type of image: {type(image)}")
                
                B = image.shape[0]
                
                if self_instance.use_naf_upsample:
                    image_for_naf = image.clone()
                
                image = self_instance.transform(image)
                
                with torch.no_grad():
                    z = self_instance.extract_features(image)
                    
                    z_clstoken = z[:, 0:1]
                    num_reg = getattr(self_instance.model.config, 'num_register_tokens', 4)
                    z_regtokens = z[:, 1:1+num_reg]
                    z_patchtokens = z[:, 1+num_reg:]
                    
                    z_patchtokens_spatial = z_patchtokens.reshape(
                        B, self_instance.patch_number, self_instance.patch_number, -1
                    )
                    
                    if camera_angle_x is None or distance is None or mesh_scale is None:
                        raise ValueError("camera_angle_x, distance, and mesh_scale must be provided")
                    
                    z_proj_lr = self_instance.proj_grid(
                        z_patchtokens_spatial, 
                        camera_angle_x, 
                        distance, 
                        mesh_scale,
                        transform_matrix
                    )
                    
                    if self_instance.use_naf_upsample:
                        self_instance._load_naf()
                        lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)

                        K = getattr(self_instance, 'naf_tile_factor', 1) or 1
                        if K <= 1:
                            hr_features = self_instance.naf_model(
                                image_for_naf, lr_features_bchw, self_instance.naf_target_size
                            )
                            z_proj_hr = self_instance.proj_grid(
                                hr_features,
                                camera_angle_x,
                                distance,
                                mesh_scale,
                                transform_matrix,
                                BHWC=False
                            )
                            del hr_features
                        else:
                            z_proj_hr = self_instance._proj_naf_tiled(
                                image_for_naf, lr_features_bchw,
                                camera_angle_x, distance, mesh_scale, transform_matrix,
                                tile_factor=int(K),
                            )

                        z_proj_lr_cpu = z_proj_lr.cpu()
                        z_proj_hr_cpu = z_proj_hr.cpu()
                        del z_proj_lr, z_proj_hr
                        torch.cuda.empty_cache()
                        z_proj = torch.cat([z_proj_lr_cpu, z_proj_hr_cpu], dim=-1)
                    else:
                        z_proj = z_proj_lr.cpu()
                        del z_proj_lr
                        torch.cuda.empty_cache()
                    
                    global_features = torch.cat([z_clstoken, z_regtokens], dim=1)
                    return global_features, z_proj

            import types
            image_cond_model.forward = types.MethodType(optimized_forward_instance, image_cond_model)
            image_cond_model._monkeypatched_for_cpu = True
            print("[Trellis2-GGUF] Successfully applied instance monkeypatch to image_cond_model!")

        with hide_src_module():
            z_global, z_proj = image_cond_model(
                image, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor,
            )
        grid_res = image_cond_model.grid_resolution
        z_proj_grid = z_proj.reshape(B, grid_res, grid_res, grid_res, -1)
        batch_indices = coords[:, 0].long().to(z_proj.device)
        x_coords = coords[:, 1].long().to(z_proj.device)
        y_coords = coords[:, 2].long().to(z_proj.device)
        z_coords = coords[:, 3].long().to(z_proj.device)
        z_proj_sparse = z_proj_grid[batch_indices, x_coords, y_coords, z_coords].to(device)
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)
        del z_proj_grid, z_proj
        torch.cuda.empty_cache()

        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            image_cond_model.grid_resolution = orig_grid_res
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=orig_grid_res,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        if self.low_vram:
            image_cond_model.cpu()
        return {
            'cond': {'global': z_global, 'proj': z_proj_st},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)},
        }

    def load_pixal3d_image_cond_ss(self):    
        if hasattr(self,'pixal3d_image_cond_ss') and self.pixal3d_image_cond_ss is not None:
            return self.pixal3d_image_cond_ss
        print('Loading Pixal3D Image Cond SS Model ...')
        build_func = self._get_build_pixal3d_image_cond_model()
        model = build_func(self.PIXAL3D_IMAGE_COND_CONFIGS["ss"])
        self.pixal3d_image_cond_ss = model
        return model
        
    def unload_pixal3d_image_cond_ss(self):
        if hasattr(self,'pixal3d_image_cond_ss') and self.pixal3d_image_cond_ss is not None:
            del self.pixal3d_image_cond_ss
            self.pixal3d_image_cond_ss = None
            self._cleanup_cuda() 
            
    def load_pixal3d_image_cond_shape_512(self):    
        if hasattr(self,'pixal3d_image_cond_shape_512') and self.pixal3d_image_cond_shape_512 is not None:
            return self.pixal3d_image_cond_shape_512
        print('Loading Pixal3D Image Cond Shape 512 Model ...')
        build_func = self._get_build_pixal3d_image_cond_model()
        model = build_func(self.PIXAL3D_IMAGE_COND_CONFIGS["shape_512"])
        self.pixal3d_image_cond_shape_512 = model
        return model
        
    def unload_pixal3d_image_cond_shape_512(self):
        if hasattr(self,'pixal3d_image_cond_shape_512') and self.pixal3d_image_cond_shape_512 is not None:
            del self.pixal3d_image_cond_shape_512
            self.pixal3d_image_cond_shape_512 = None
            self._cleanup_cuda() 
            
    def load_pixal3d_image_cond_shape_1024(self):    
        if hasattr(self,'pixal3d_image_cond_shape_1024') and self.pixal3d_image_cond_shape_1024 is not None:
            return self.pixal3d_image_cond_shape_1024
        print('Loading Pixal3D Image Cond Shape 1024 Model ...')
        build_func = self._get_build_pixal3d_image_cond_model()
        model = build_func(self.PIXAL3D_IMAGE_COND_CONFIGS["shape_1024"])
        self.pixal3d_image_cond_shape_1024 = model
        return model
        
    def unload_pixal3d_image_cond_shape_1024(self):
        if hasattr(self,'pixal3d_image_cond_shape_1024') and self.pixal3d_image_cond_shape_1024 is not None:
            del self.pixal3d_image_cond_shape_1024
            self.pixal3d_image_cond_shape_1024 = None
            self._cleanup_cuda() 

    def load_pixal3d_image_cond_tex_1024(self):    
        if hasattr(self,'pixal3d_image_cond_tex_1024') and self.pixal3d_image_cond_tex_1024 is not None:
            return self.pixal3d_image_cond_tex_1024
        print('Loading Pixal3D Image Cond Tex 1024 Model ...')
        build_func = self._get_build_pixal3d_image_cond_model()
        model = build_func(self.PIXAL3D_IMAGE_COND_CONFIGS["tex_1024"])
        self.pixal3d_image_cond_tex_1024 = model
        return model
        
    def unload_pixal3d_image_cond_tex_1024(self):
        if hasattr(self,'pixal3d_image_cond_tex_1024') and self.pixal3d_image_cond_tex_1024 is not None:
            del self.pixal3d_image_cond_tex_1024
            self.pixal3d_image_cond_tex_1024 = None
            self._cleanup_cuda()

    def load_sparse_structure_model(self):        
        if self.models['sparse_structure_flow_model'] is None:
            print('Loading Sparse Structure model ...')
            _path = self._sdnq_remap(os.path.join(self.path, self._pretrained_args['models']['sparse_structure_flow_model']))
            self.models['sparse_structure_flow_model'] = models.from_pretrained(
                _path,
                enable_gguf=getattr(self, 'enable_gguf', False),
                gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                precision=getattr(self, 'precision', None),
                enable_sdnq=getattr(self, 'enable_sdnq', False),
                sdnq_use_quantized_matmul=getattr(self, 'sdnq_use_quantized_matmul', True),
                sdnq_torch_compile=getattr(self, 'sdnq_torch_compile', False),
                isPixal3D=getattr(self, 'isPixal3D', False),
            )
            self.models['sparse_structure_flow_model'].eval()
            self.models['sparse_structure_flow_model'].to(self._device)
        
        if self.models['sparse_structure_decoder'] is None:            
            self.models['sparse_structure_decoder'] = models.from_pretrained(self._pretrained_args['models']['sparse_structure_decoder'], isPixal3D=getattr(self, 'isPixal3D', False))
            self.models['sparse_structure_decoder'].eval()        
            self.models['sparse_structure_decoder'].to(self._device)
            if hasattr(self.models['sparse_structure_decoder'], 'low_vram'):
                self.models['sparse_structure_decoder'].low_vram = self.low_vram
    
    def unload_sparse_structure_model(self):
        if self.models['sparse_structure_flow_model'] is not None:
            del self.models['sparse_structure_flow_model']
            self.models['sparse_structure_flow_model'] = None            
            
        if self.models['sparse_structure_decoder'] is not None:
            del self.models['sparse_structure_decoder']
            self.models['sparse_structure_decoder'] = None
        
        self._cleanup_cuda()
            
    def load_image_cond_model(self):
        if self.image_cond_model is None:
            print('Loading Image Cond model ...')
            self.image_cond_model = getattr(image_feature_extractor, self._pretrained_args['image_cond_model']['name'])(**self._pretrained_args['image_cond_model']['args'])
            self.image_cond_model.to(self._device)
            
    def unload_image_cond_model(self):
        if self.image_cond_model is not None:
            del self.image_cond_model
            self.image_cond_model = None            
            self._cleanup_cuda()
            
    def load_shape_slat_flow_model_512(self):        
        if self.models['shape_slat_flow_model_512'] is None:
            print('Loading Shape Slat Flow 512 model ...')
            _path = self._sdnq_remap(os.path.join(self.path, self._pretrained_args['models']['shape_slat_flow_model_512']))
            self.models['shape_slat_flow_model_512'] = models.from_pretrained(
                _path,
                enable_gguf=getattr(self, 'enable_gguf', False),
                gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                precision=getattr(self, 'precision', None),
                enable_sdnq=getattr(self, 'enable_sdnq', False),
                sdnq_use_quantized_matmul=getattr(self, 'sdnq_use_quantized_matmul', True),
                sdnq_torch_compile=getattr(self, 'sdnq_torch_compile', False),
                isPixal3D=getattr(self, 'isPixal3D', False),
            )
            self.models['shape_slat_flow_model_512'].eval()
            self.models['shape_slat_flow_model_512'].to(self._device)
            
    def unload_shape_slat_flow_model_512(self):
        if self.models['shape_slat_flow_model_512'] is not None:
            del self.models['shape_slat_flow_model_512']
            self.models['shape_slat_flow_model_512'] = None
            self._cleanup_cuda()
            
    def load_tex_slat_flow_model_512(self):
        if 'tex_slat_flow_model_512' not in self._pretrained_args.get('models', {}):
            return
        if self.models.get('tex_slat_flow_model_512') is None:
            print('Loading Texture Slat Flow 512 model ...')
            _path = self._sdnq_remap(os.path.join(self.path, self._pretrained_args['models']['tex_slat_flow_model_512']))
            self.models['tex_slat_flow_model_512'] = models.from_pretrained(
                _path,
                enable_gguf=getattr(self, 'enable_gguf', False),
                gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                precision=getattr(self, 'precision', None),
                enable_sdnq=getattr(self, 'enable_sdnq', False),
                sdnq_use_quantized_matmul=getattr(self, 'sdnq_use_quantized_matmul', True),
                sdnq_torch_compile=getattr(self, 'sdnq_torch_compile', False),
                isPixal3D=getattr(self, 'isPixal3D', False),
            )
            self.models['tex_slat_flow_model_512'].eval()
            self.models['tex_slat_flow_model_512'].to(self._device)          

    def unload_tex_slat_flow_model_512(self):
        if self.models.get('tex_slat_flow_model_512') is not None:
            del self.models['tex_slat_flow_model_512']
            self.models['tex_slat_flow_model_512'] = None
            self._cleanup_cuda()

    def load_tex_slat_decoder(self):        
        if self.models['tex_slat_decoder'] is None:
            print('Loading Texture Slat decoder model ...')
            self.models['tex_slat_decoder'] = models.from_pretrained(
                os.path.join(self.path, self._pretrained_args['models']['tex_slat_decoder']),
                precision=getattr(self, 'precision', None),
                isPixal3D=getattr(self, 'isPixal3D', False)
            )
            self.models['tex_slat_decoder'].eval()
            self.models['tex_slat_decoder'].to(self._device)
            if hasattr(self.models['tex_slat_decoder'], 'low_vram'):
                self.models['tex_slat_decoder'].low_vram = self.low_vram

    def unload_tex_slat_decoder(self):
        if self.models['tex_slat_decoder'] is not None:
            del self.models['tex_slat_decoder']
            self.models['tex_slat_decoder'] = None
            self._cleanup_cuda()
            
    def load_shape_slat_decoder(self):        
        if self.models['shape_slat_decoder'] is None:
            print('Loading Shape Slat decoder model ...')
            self.models['shape_slat_decoder'] = models.from_pretrained(
                os.path.join(self.path, self._pretrained_args['models']['shape_slat_decoder']),
                precision=getattr(self, 'precision', None),
                isPixal3D=getattr(self, 'isPixal3D', False)
            )
            self.models['shape_slat_decoder'].eval()
            self.models['shape_slat_decoder'].to(self._device)
            if hasattr(self.models['shape_slat_decoder'], 'low_vram'):
                self.models['shape_slat_decoder'].low_vram = self.low_vram

    def unload_shape_slat_decoder(self):
        if self.models['shape_slat_decoder'] is not None:
            del self.models['shape_slat_decoder']
            self.models['shape_slat_decoder'] = None
            self._cleanup_cuda()

    def load_shape_slat_flow_model_1024(self):        
        if self.models['shape_slat_flow_model_1024'] is None:
            print('Loading Shape Slat Flow 1024 model ...')
            _path = self._sdnq_remap(os.path.join(self.path, self._pretrained_args['models']['shape_slat_flow_model_1024']))
            self.models['shape_slat_flow_model_1024'] = models.from_pretrained(
                _path,
                enable_gguf=getattr(self, 'enable_gguf', False),
                gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                precision=getattr(self, 'precision', None),
                enable_sdnq=getattr(self, 'enable_sdnq', False),
                sdnq_use_quantized_matmul=getattr(self, 'sdnq_use_quantized_matmul', True),
                sdnq_torch_compile=getattr(self, 'sdnq_torch_compile', False),
                isPixal3D=getattr(self, 'isPixal3D', False),
            )
            self.models['shape_slat_flow_model_1024'].eval()
            self.models['shape_slat_flow_model_1024'].to(self._device)           

    def unload_shape_slat_flow_model_1024(self):
        if self.models['shape_slat_flow_model_1024'] is not None:
            del self.models['shape_slat_flow_model_1024']
            self.models['shape_slat_flow_model_1024'] = None
            self._cleanup_cuda()

    def load_tex_slat_flow_model_1024(self):        
        if self.models['tex_slat_flow_model_1024'] is None:
            print('Loading Texture Slat Flow 1024 model ...')
            _path = self._sdnq_remap(os.path.join(self.path, self._pretrained_args['models']['tex_slat_flow_model_1024']))
            self.models['tex_slat_flow_model_1024'] = models.from_pretrained(
                _path,
                enable_gguf=getattr(self, 'enable_gguf', False),
                gguf_quant=getattr(self, 'gguf_quant', 'Q8_0'),
                precision=getattr(self, 'precision', None),
                enable_sdnq=getattr(self, 'enable_sdnq', False),
                sdnq_use_quantized_matmul=getattr(self, 'sdnq_use_quantized_matmul', True),
                sdnq_torch_compile=getattr(self, 'sdnq_torch_compile', False),
                isPixal3D=getattr(self, 'isPixal3D', False),
            )
            self.models['tex_slat_flow_model_1024'].eval()
            self.models['tex_slat_flow_model_1024'].to(self._device)                   

    def unload_tex_slat_flow_model_1024(self):
        if self.models['tex_slat_flow_model_1024'] is not None:
            del self.models['tex_slat_flow_model_1024']
            self.models['tex_slat_flow_model_1024'] = None
            self._cleanup_cuda()

    def load_shape_slat_encoder(self):        
        if self.models['shape_slat_encoder'] is None:
            print('Loading Shape Slat Encoder model ...')
            self.models['shape_slat_encoder'] = models.from_pretrained(f"{self.path}/ckpts/shape_enc_next_dc_f16c32_fp16", isPixal3D=getattr(self, 'isPixal3D', False))
            self.models['shape_slat_encoder'].eval()
            self.models['shape_slat_encoder'].to(self._device)
            if hasattr(self.models['shape_slat_encoder'], 'low_vram'):
                self.models['shape_slat_encoder'].low_vram = self.low_vram

    def unload_shape_slat_encoder(self):
        if self.models['shape_slat_encoder'] is not None:
            del self.models['shape_slat_encoder']
            self.models['shape_slat_encoder'] = None
            self._cleanup_cuda()      

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            if self.image_cond_model is not None:
                self.image_cond_model.to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self.rembg_model.cpu()
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        resolution: int,
        include_neg_cond: bool = True,
        *,
        fusion_mode: str = "concat",   # "concat" or "mean"
        max_views: int = 4,            # safety cap for 3090
    ) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image: One of:
                - PIL Image
                - list[PIL Image] (multi-view)
                - torch.Tensor batch (B,H,W,C) from ComfyUI
            resolution: Conditioning resolution (e.g. 512 or 1024)
            include_neg_cond: Whether to include negative conditioning
            fusion_mode: "concat" (recommended) or "mean"
            max_views: Max number of views to fuse when list/batch is provided

        Returns:
            dict with keys: cond (+ neg_cond if include_neg_cond)
        """
        self.image_cond_model.image_size = resolution

        # ---- Normalize input into what image_cond_model expects ----
        # Most implementations expect PIL image or list[PIL images].
        if isinstance(image, torch.Tensor):
            # Expect ComfyUI IMAGE tensor: (B,H,W,C) float in [0,1]
            if image.ndim == 4:
                # Lazy import to avoid circulars if tensor2pil is in nodes/utils
                from .nodes import tensor2pil 
                images = [tensor2pil(image[i]) for i in range(min(int(image.shape[0]), max_views))]
            else:
                raise ValueError(f"Expected image tensor with shape (B,H,W,C), got {tuple(image.shape)}")
        elif isinstance(image, Image.Image):
            images = [image]
        elif isinstance(image, (list, tuple)):
            # list of PIL images
            images = list(image)[:max_views]
            if not images:
                raise ValueError("Empty image list provided to get_cond().")
            if not all(isinstance(im, Image.Image) for im in images):
                raise TypeError("get_cond() received a list/tuple but not all elements are PIL Images.")
        else:
            raise TypeError(f"Unsupported image type for get_cond(): {type(image)}")

        if self.low_vram:
            self.image_cond_model.to(self.device)

        # ---- Extract per-view conditioning ----
        with hide_src_module():
            cond = self.image_cond_model(images)

        # Normalize shapes:
        # Common outputs:
        #   - (V, N, D) for multi-view
        #   - (1, N, D) for single-view list length=1
        #   - (N, D) for some single-image extractors
        if cond.ndim == 2:
            # (N, D) -> (1, N, D)
            cond = cond.unsqueeze(0)
        elif cond.ndim != 3:
            raise RuntimeError(f"Unexpected cond ndim={cond.ndim}, shape={tuple(cond.shape)}")

        # If we passed multiple views, fuse them into one conditioning sequence
        if cond.shape[0] > 1:
            if fusion_mode == "concat":
                # (V, N, D) -> (1, V*N, D)
                cond = cond.reshape(1, -1, cond.shape[-1])
            elif fusion_mode == "mean":
                # (V, N, D) -> (1, N, D)
                cond = cond.mean(dim=0, keepdim=True)
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        if self.low_vram:
            self.image_cond_model.cpu()

        if not include_neg_cond:
            return {"cond": cond}

        neg_cond = torch.zeros_like(cond)
        return {"cond": cond, "neg_cond": neg_cond}

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        sampler: str = None,
        fill_holes: bool = True,
        hole_structure: int = 1,
        hole_iterations: int = 1,
        hole_fill_algorithm: str = "remove_small_holes",
        keep_only_shell: bool = True,
        verbose: bool = True,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        if self.low_vram:
            cond = self._cond_to(cond, self.device)                         
        # Sample sparse structure latent
        
        if isinstance(cond, dict) and 'cond' in cond:
            c = cond['cond']
            if isinstance(c, dict):
                print("COND global:", c['global'].abs().mean().item(), "proj:", c['proj'].abs().mean().item())
            else:
                print("COND:", c.abs().mean().item())

        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        
        active_sampler = self._get_sampler(sampler, self.sparse_structure_sampler)
        # Debug: log conditioning shapes and stats
        _c = cond.get('cond', cond)
        if isinstance(_c, dict):
            for k, v in _c.items():
                if isinstance(v, torch.Tensor):
                    print(f"  [SS cond '{k}'] shape={v.shape}, dtype={v.dtype}, mean={v.float().mean():.4f}, std={v.float().std():.4f}")
        _nc = cond.get('neg_cond', None)
        if _nc is not None and isinstance(_nc, dict):
            for k, v in _nc.items():
                if isinstance(v, torch.Tensor):
                    print(f"  [SS neg_cond '{k}'] shape={v.shape}, dtype={v.dtype}, mean={v.float().mean():.4f}")
        z_s = active_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        print("z_s stats:", z_s.shape, z_s.max().item(), z_s.min().item(), z_s.mean().item(), z_s.dtype)
        tmp_decoded = decoder(z_s)
        print("decoded stats:", tmp_decoded.shape, tmp_decoded.max().item(), tmp_decoded.min().item(), tmp_decoded.mean().item(), tmp_decoded.dtype)
        decoded = tmp_decoded > 0
        if self.low_vram:
            decoder.cpu()
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        coords = coords.cpu()
        del decoded
        del z_s
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()
        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        sampler: str = None,
        verbose: bool = True,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        proj_image_cond_model=None,
        proj_images=None,
        **kwargs,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        if self.low_vram:
            cond = self._cond_to(cond, self.device)

        coords_dev = coords.to(self.device)
        # Rebuild proj cond with coords if proj model is provided
        if proj_image_cond_model is not None and proj_images is not None:
            grid_res = int(coords[:, 1:].max().item()) + 1
            cond = self.get_proj_cond_shape(proj_image_cond_model, proj_images, coords,
                                            grid_resolution_override=grid_res)
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        active_sampler = self._get_sampler(sampler, self.shape_slat_sampler)
        slat = active_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()

        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        sampler: str = None,
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        proj_image_cond_model=None,
        proj_images=None,
        **kwargs,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR

        if self.low_vram:
            lr_cond = self._cond_to(lr_cond, self.device)
            cond = self._cond_to(cond, self.device)

        coords_dev = coords.to(self.device)                         
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)
            
        active_sampler = self._get_sampler(sampler, self.shape_slat_sampler)
        slat = active_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (LR)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
            self._cleanup_cuda()                                
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            lr_cond = self._cond_cpu(lr_cond)
            self._cleanup_cuda()

        # Upsample       
        self.load_shape_slat_decoder()
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        
        if not self.keep_models_loaded:
            self.unload_shape_slat_decoder()
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                hr_resolution = 512
                break
        
        coords_dev = coords.to(self.device)

        # Free LR proj cond memory before HR proj cond
        if self.low_vram:
            self._cleanup_cuda()
        # Rebuild proj cond with HR coords if proj model is provided
        if proj_image_cond_model is not None and proj_images is not None:
            hr_grid_res = int(coords[:, 1:].max().item()) + 1
            cond = self.get_proj_cond_shape(proj_image_cond_model, proj_images, coords,
                                            grid_resolution_override=hr_grid_res)
            if self.low_vram:
                self._cleanup_cuda()

        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
            
        active_sampler = self._get_sampler(sampler, self.shape_slat_sampler)
        slat = active_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()

        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
        use_tiled: bool = True,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        
        self.load_shape_slat_decoder()
        
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True, useTiled=use_tiled)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
            self._cleanup_cuda()                
        
        if not self.keep_models_loaded:        
            self.unload_shape_slat_decoder()
            
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
        sampler: str = None,
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        proj_image_cond_model=None,
        proj_images=None,
        **kwargs,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        if self.low_vram:
            cond = self._cond_to(cond, self.device)
        # Rebuild proj cond with shape_slat coords if proj model provided
        if proj_image_cond_model is not None and proj_images is not None:
            tex_coords = shape_slat.coords.cpu()
            tex_grid_res = int(tex_coords[:, 1:].max().item()) + 1
            cond = self.get_proj_cond_shape(proj_image_cond_model, proj_images, tex_coords,
                                            grid_resolution_override=tex_grid_res)
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
            
        active_sampler = self._get_sampler(sampler, self.tex_slat_sampler)
        slat = active_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                    

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()                         
        return slat

    def decode_tex_slat(self, slat: SparseTensor, subs: torch.Tensor = None) -> SparseTensor:
        """
        Decode the structured latent to texture voxels.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        # Comprehensive offload to ensure maximum VRAM for decoder
        self.move_all_to_cpu()
        
        self.load_tex_slat_decoder()
        
        if self.low_vram:
            self.models['tex_slat_decoder'].to(self.device)
            self.models['tex_slat_decoder'].low_vram = True                                               
            
        if subs is None:
            if getattr(self, 'use_tiled_decoder_for_texture', False):
                tile_overlap = getattr(self, 'tiled_decoder_overlap', 48)
                tile_size = getattr(self, 'tiled_decoder_size', 120)
                ret = self.models['tex_slat_decoder']._tiled_forward(slat, tile_size=tile_size, overlap=tile_overlap) * 0.5 + 0.5
            else:
                ret = self.models['tex_slat_decoder'](slat) * 0.5 + 0.5
        else:
            slat.clear_spatial_cache()
            if getattr(self, 'use_tiled_decoder_for_texture', False):
                tile_overlap = getattr(self, 'tiled_decoder_overlap', 48)
                tile_size = getattr(self, 'tiled_decoder_size', 120)
                ret = self.models['tex_slat_decoder']._tiled_forward(slat, guide_subs=subs, tile_size=tile_size, overlap=tile_overlap) * 0.5 + 0.5
            else:
                ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
            
        if self.low_vram:
            self.models['tex_slat_decoder'].cpu()
            self.models['tex_slat_decoder'].low_vram = False
            self._cleanup_cuda()
        
        if not self.keep_models_loaded:
            self.unload_tex_slat_decoder()
        
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
        use_tiled: bool = True,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        def strip_ggml_sparse(st):
            if st is None: return None
            feats = torch.empty(st.feats.shape, dtype=st.feats.dtype, device=st.feats.device).copy_(st.feats)
            coords = torch.empty(st.coords.shape, dtype=st.coords.dtype, device=st.coords.device).copy_(st.coords)
            return st.replace(feats=feats, coords=coords)

        shape_slat = strip_ggml_sparse(shape_slat)
        tex_slat = strip_ggml_sparse(tex_slat)

        meshes, subs = self.decode_shape_slat(shape_slat, resolution, use_tiled=use_tiled)
        
        if subs is not None:
            subs = [strip_ggml_sparse(sub) for sub in subs]

        if self.low_vram:
            self._cleanup_cuda()                                                         
            
        if tex_slat is None:
            if self.low_vram:
                self._cleanup_cuda()                                                         
            out_mesh = []
            for m in meshes:
                out_mesh.append(
                    MeshWithVoxel(
                        m.vertices, m.faces,
                        origin = [-0.5, -0.5, -0.5],
                        voxel_size = 1 / resolution,
                        coords = None,
                        attrs = None,
                        voxel_shape = None,
                        layout=self.pbr_attr_layout
                    )
                )
            return out_mesh
        
        else:    
            tex_voxels = self.decode_tex_slat(tex_slat, subs)
            if self.low_vram:
                self._cleanup_cuda()                                                         
            out_mesh = []
            for m, v in zip(meshes, tex_voxels):
                out_mesh.append(
                    MeshWithVoxel(
                        m.vertices.float(), m.faces.int(),
                        origin = [-0.5, -0.5, -0.5],
                        voxel_size = 1 / resolution,
                        coords = v.coords[:, 1:],
                        attrs = v.feats.float(),
                        voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                        layout=self.pbr_attr_layout
                    )
                )
            return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = False,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        max_views: int = 4,
        generate_texture_slat = True,
        use_tiled: bool = True,
        pbar = None,
        sampler: str = None,
        sparse_structure_sampler: str = None,
        shape_sampler: str = None,
        tex_sampler: str = None,
        verbose: bool = False,
        fill_holes: bool = True,
        hole_iterations: int = 1,
        dino_lock: float = 0.00,
        dino_substeps: int = 4,
        hole_fill_algorithm: str = "remove_small_holes",
        dino_foundation_cap: float = 0.92,
        keep_only_shell: bool = True,
        **kwargs,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        # if pipeline_type == '512':
            # assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_512' in self.models, "No 512 resolution texture SLat flow model found."
        # elif pipeline_type == '1024':
            # assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        # elif pipeline_type == '1024_cascade':
            # assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            # assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        # elif pipeline_type == '1536_cascade':
            # assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            # assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            # assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        # else:
            # raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        # Accept either a single PIL image or a list of PIL images (multi-view)
        if isinstance(image, (list, tuple)):
            images = list(image)
        else:
            images = [image]

        if preprocess_image:
            images = [self.preprocess_image(im) for im in images]
            
        seed_all(seed)
        
        # Load sparse structure model early so we can check image_attn_mode
        self.load_sparse_structure_model()
        ss_model = self.models['sparse_structure_flow_model']
        ss_attn_mode = getattr(ss_model, 'image_attn_mode', None)

        # Get Image Cond
        self.load_image_cond_model()
        # Use proj conditioning if sparse structure model requires it
        if ss_attn_mode == 'proj':
            proj_cond_model = self.load_pixal3d_image_cond_ss()
            cond_512 = self.get_proj_cond_ss(images, image_cond_model=proj_cond_model)
        else:
            # Multi-view conditioning happens inside get_cond()
            cond_512 = self.get_cond(images, 512, max_views = max_views)
        cond_1024 = self.get_cond(images, 1024, max_views = max_views) if pipeline_type != '512' else None
        
        if pbar is not None:
            pbar.update(1)
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()
        
        #ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        
        # Sampling Sparse Structure
        self.load_sparse_structure_model()        
        coords = self.sample_sparse_structure(
            cond_512, sparse_structure_resolution,
            num_samples, sparse_structure_sampler_params,
            sampler=sparse_structure_sampler or sampler
        )
        
        if pbar is not None:
            pbar.update(1)
        
        if not self.keep_models_loaded:
            self.unload_sparse_structure_model()
        
        # Sampling Shape
        # Build proj conds for shape/tex stages if needed (Pixal3D models require proj conditioning)
        shape_512_model = None
        shape_1024_model = None
        tex_1024_model = None
        if ss_attn_mode == 'proj':
            # coords are in sparse_structure_resolution space; proj grid must match
            coords_grid_res = int(coords[:, 1:].max().item()) + 1
            shape_512_model = self.load_pixal3d_image_cond_shape_512()
            cond_512 = self.get_proj_cond_shape(shape_512_model, images, coords,
                                                grid_resolution_override=coords_grid_res)
            if pipeline_type != '512':
                shape_1024_model = self.load_pixal3d_image_cond_shape_1024()
                cond_1024 = self.get_proj_cond_shape(shape_1024_model, images, coords,
                                                     grid_resolution_override=coords_grid_res)
                tex_1024_model = self.load_pixal3d_image_cond_tex_1024()

        if pipeline_type == '512':            
            self.unload_shape_slat_flow_model_1024()
            self.load_shape_slat_flow_model_512()            
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_512_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_1024()
                self.load_tex_slat_flow_model_512()
                tex_slat = self.sample_tex_slat(
                    cond_512, self.models['tex_slat_flow_model_512'],
                    shape_slat, tex_slat_sampler_params,
                    sampler=tex_sampler or sampler,
                    proj_image_cond_model=shape_512_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                
                if pbar is not None:
                    pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_512()
            
            res = 512
        elif pipeline_type == '1024':
            self.unload_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_1024_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_1024()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=tex_sampler or sampler,
                    proj_image_cond_model=tex_1024_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                
                if pbar is not None:
                    pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
                
            res = 1024
        elif pipeline_type == '1024_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()            
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_1024_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
            # Free VRAM before texture stage
            if ss_attn_mode == 'proj' and self.low_vram:
                self.unload_pixal3d_image_cond_ss()
                self.unload_pixal3d_image_cond_shape_512()
                self.unload_pixal3d_image_cond_shape_1024()
                self._cleanup_cuda()

            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=tex_sampler or sampler,
                    proj_image_cond_model=tex_1024_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif pipeline_type == '2048_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 2048,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_1024_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
                
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=tex_sampler or sampler,
                    proj_image_cond_model=tex_1024_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                
                if pbar is not None:
                    pbar.update(1)
                
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif pipeline_type == '4096_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 4096,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_1024_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
                
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=tex_sampler or sampler,
                    proj_image_cond_model=tex_1024_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                        
                if pbar is not None:
                    pbar.update(1)  
                
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif pipeline_type == '1536_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()                
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                sampler=shape_sampler or sampler,
                proj_image_cond_model=shape_1024_model if ss_attn_mode == 'proj' else None,
                proj_images=images if ss_attn_mode == 'proj' else None,
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
            
            if generate_texture_slat:            
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=tex_sampler or sampler,
                    proj_image_cond_model=tex_1024_model if ss_attn_mode == 'proj' else None,
                    proj_images=images if ss_attn_mode == 'proj' else None,
                )
                
                if pbar is not None:
                    pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()               
            
        # Clean up proj cond models when not keeping loaded
        if not self.keep_models_loaded and ss_attn_mode == 'proj':
            self.unload_pixal3d_image_cond_ss()
            self.unload_pixal3d_image_cond_shape_512()
            self.unload_pixal3d_image_cond_shape_1024()
            self.unload_pixal3d_image_cond_tex_1024()
        torch.cuda.empty_cache()
        if generate_texture_slat:
            out_mesh = self.decode_latent(shape_slat, tex_slat, res, use_tiled=use_tiled)
        else:
            out_mesh = self.decode_latent(shape_slat, None, res, use_tiled=use_tiled)
        torch.cuda.empty_cache()
        if pbar is not None:
            pbar.update(1)
        if return_latent:
            if generate_texture_slat:
                return out_mesh, (shape_slat, tex_slat, res)
            else:
                return out_mesh, (shape_slat, None, res)
        else:
            return out_mesh

    @torch.no_grad()
    def run_multiview(
        self,
        front: Image.Image,
        back: Image.Image = None,
        left: Image.Image = None,
        right: Image.Image = None,
        seed: int = 42,
        pipeline_type: str = None,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        max_num_tokens: int = 49152,
        sparse_structure_resolution: int = 32,
        generate_texture_slat: bool = True,
        use_tiled: bool = True,
        return_latent: bool = False,
        pbar: ProgressBar = None,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = None,
        sparse_structure_sampler: str = None,
        shape_sampler: str = None,
        tex_sampler: str = None,
        verbose: bool = False,
        fill_holes: bool = True,
        hole_iterations: int = 1,
        dino_lock: float = 0.00,
        dino_substeps: int = 4,
        hole_fill_algorithm: str = "remove_small_holes",
        dino_foundation_cap: float = 0.92,
        keep_only_shell: bool = True,
        **kwargs,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline with named multi-view images and spatial blending.
        """
        if pipeline_type is None:
            pipeline_type = self.default_pipeline_type
        
        seed_all(seed)
        
        # Collect views
        views_dict = {'front': front}
        if back is not None: views_dict['back'] = back
        if left is not None: views_dict['left'] = left
        if right is not None: views_dict['right'] = right
        
        views_list = list(views_dict.keys())

        # 1. Conditioning
        # Calculate conditioning per view
        conds = {}     # 1024 or None (if 512)
        lr_conds = {}  # 512 (for cascade)
        conds_512 = {} # Explicit 512 storage for structure sampling
        conds_1024 = {}
        
        self.load_image_cond_model()
        
        if pipeline_type == '512':
             for v, img in views_dict.items():
                c = self.get_cond([img], 512)
                conds[v] = c
                conds_512[v] = c
                
        elif pipeline_type == '1024':
             for v, img in views_dict.items():
                c1024 = self.get_cond([img], 1024)
                conds[v] = c1024
                conds_1024[v] = c1024
                # Does 1024 pipeline use 512 for structure? 
                # run() says: cond_512 = get_cond(..., 512). So yes.
                conds_512[v] = self.get_cond([img], 512)
                
        elif 'cascade' in pipeline_type:
            # 1024_cascade or 1536_cascade
             for v, img in views_dict.items():
                c512 = self.get_cond([img], 512)
                c1024 = self.get_cond([img], 1024)
                lr_conds[v] = c512
                conds[v] = c1024
                conds_512[v] = c512
                conds_1024[v] = c1024
                
        if not self.keep_models_loaded:
            self.unload_image_cond_model()

        if pbar is not None:
            pbar.update(1)
        
        self.load_sparse_structure_model()          
        coords = self.sample_sparse_structure_multiview(
            conds_512, 
            views_list,
            sparse_structure_resolution,
            sampler_params=sparse_structure_sampler_params,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            sampler=sparse_structure_sampler or sampler,
        )
        
        if not self.keep_models_loaded:
            self.unload_sparse_structure_model()        
            
        if pbar is not None:
            pbar.update(1)

        # 3. Shape Slat MultiView
        shape_slat = None
        res = 0
        
        if pipeline_type == '1024_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat = self.sample_shape_slat_cascade_multiview(
                lr_conds, conds, views_list,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sampler=shape_sampler or sampler
            )
            res = 1024
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
                
        elif pipeline_type == '1536_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()
            shape_slat = self.sample_shape_slat_cascade_multiview(
                lr_conds, conds, views_list,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sampler=shape_sampler or sampler
            )
            res = 1536
             
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
        
        elif pipeline_type == '512': # Single stage
             self.load_shape_slat_flow_model_512()
             shape_slat = self.sample_shape_slat_multiview(
                 conds, views_list,
                 self.models['shape_slat_flow_model_512'],
                 coords, shape_slat_sampler_params,
                 front_axis=front_axis,
                 blend_temperature=blend_temperature,
                 sampler=shape_sampler or sampler
             )
             res = 512
             if not self.keep_models_loaded:
                 self.unload_shape_slat_flow_model_512()
        
        elif pipeline_type == '1024': # Single stage
             self.load_shape_slat_flow_model_1024()
             shape_slat = self.sample_shape_slat_multiview(
                 conds, views_list,
                 self.models['shape_slat_flow_model_1024'],
                 coords, shape_slat_sampler_params,
                 front_axis=front_axis,
                 blend_temperature=blend_temperature,
                 sampler=shape_sampler or sampler
             )
             res = 1024
             if not self.keep_models_loaded:
                 self.unload_shape_slat_flow_model_1024()

        if pbar is not None:
            pbar.update(1)

        # Texture Slat MultiView
        tex_slat = None
        if generate_texture_slat:
            tex_model_key = 'tex_slat_flow_model_1024'
            if pipeline_type == '512':
                tex_model_key = 'tex_slat_flow_model_512'
                self.load_tex_slat_flow_model_512()
                flow_model = self.models['tex_slat_flow_model_512']
                tex_conds = conds_512
            else:
                self.load_tex_slat_flow_model_1024()
                flow_model = self.models['tex_slat_flow_model_1024']
                tex_conds = conds_1024
            
            tex_slat = self.sample_tex_slat_multiview(
                tex_conds, views_list,
                shape_slat=shape_slat, 
                flow_model=flow_model,
                sampler_params=tex_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sampler=tex_sampler or sampler
            )  
             
            if not self.keep_models_loaded:
                if pipeline_type == '512':
                    self.unload_tex_slat_flow_model_512()
                else:
                    self.unload_tex_slat_flow_model_1024()

            if pbar is not None:
                pbar.update(1)
                 
        torch.cuda.empty_cache()
        if generate_texture_slat:
            out_mesh = self.decode_latent(shape_slat, tex_slat, res, use_tiled=use_tiled)
        else:
            out_mesh = self.decode_latent(shape_slat, None, res, use_tiled=use_tiled)            
        torch.cuda.empty_cache()
        
        if pbar is not None:
             pbar.update(1)
             
        if return_latent:
             return out_mesh, (shape_slat, tex_slat, res)
        else:
             return out_mesh

    def sample_sparse_structure_multiview(
        self,
        conds: dict,
        views: list,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = None,
        fill_holes: bool = True,
        hole_structure: int = 1,
        hole_iterations: int = 1,
        hole_fill_algorithm: str = "remove_small_holes",
        keep_only_shell: bool = True,
        verbose: bool = True,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ) -> torch.Tensor:
        """
        Sample sparse structures with multi-view blending.
        """
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)
                
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        
        default_sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution
        )
        active_sampler = self._get_sampler(sampler, default_sampler)
        
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        
        if self.low_vram:
            flow_model.to(self.device)
            
        z_s = active_sampler.sample(
            flow_model,
            noise,
            conds=conds,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure (MultiView)",
        ).samples
        
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()
            
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
            
        # Standard decoding logic from sample_sparse_structure
        decoded = decoder(z_s) > 0
        
        if self.low_vram:
            decoder.cpu()
            self._cleanup_cuda()
            
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        
        # Extract coordinates (N, 4) -> (b, d, h, w)
        # argwhere returns (b, c, d, h, w), so we want [0, 2, 3, 4]
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        
        coords = coords.cpu()
        del decoded
        del z_s
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()
            
        return coords

    def sample_shape_slat_multiview(
        self,
        conds: dict,
        views: list,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = None,
    ) -> SparseTensor:
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)

        coords_dev = coords.to(self.device)                         
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        
        default_sampler = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution,
        )
        active_sampler = self._get_sampler(sampler, default_sampler)
        
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        
        if self.low_vram:
            flow_model.to(self.device)
            
        slat = active_sampler.sample(
            flow_model,
            noise,
            conds=conds,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (MultiView)",
        ).samples
        
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()

        return slat

    def sample_shape_slat_cascade_multiview(
        self,
        lr_conds: dict,
        conds: dict,
        views: list,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = None,
    ) -> SparseTensor:
        # LR
        if self.low_vram:
            for v in lr_conds:
                lr_conds[v] = self._cond_to(lr_conds[v], self.device)
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)

        coords_dev = coords.to(self.device)                         
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels, device=self.device),
            coords=coords_dev,
        )
        
        default_sampler_lr = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model_lr.resolution,
        )
        active_sampler_lr = self._get_sampler(sampler, default_sampler_lr)
        sampler_params_combined = {**self.shape_slat_sampler_params, **sampler_params}
        slat = active_sampler_lr.sample(
            flow_model_lr,
            noise,
            conds=lr_conds,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            **sampler_params_combined,
            verbose=True,
            tqdm_desc="Sampling shape SLat (MultiView LR)",
        ).samples
        
        if self.low_vram:
            flow_model_lr.cpu()
            self._cleanup_cuda()                                
        
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        del coords_dev
        
        # Upsample logic
        self.load_shape_slat_decoder()
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
            
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                hr_resolution = 512
                break

        # HR
        default_sampler_hr = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution,
        )
        active_sampler_hr = self._get_sampler(sampler, default_sampler_hr)
        
        d_slat = active_sampler_hr.sample(
            flow_model,
            noise,
            conds=conds,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            **sampler_params_combined,
            verbose=True,
            tqdm_desc="Sampling shape SLat (MultiView HR)",
        ).samples
        
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()
            
        slat = d_slat * std + mean
        
        if self.low_vram:
            for v in lr_conds:
                lr_conds[v] = self._cond_cpu(lr_conds[v])
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()
            
        return slat


    def sample_tex_slat_multiview(
        self,
        conds: dict,
        views: list,
        shape_slat: SparseTensor,
        flow_model,
        sampler_params: dict = {},
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = None,
    ) -> SparseTensor:
        """
        Sample structured latent for texture with multi-view blending.
        """
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)

        # Normalize shape slat for conditioning
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat_normalized = (shape_slat - mean) / std

        #coords = shape_slat.coords
        #coords_dev = coords.to(self.device)
        
        # Calculate noise channels: total input - concat cond channels
        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise_channels = in_channels - shape_slat.feats.shape[1]
        
        # noise = SparseTensor(
            # feats=torch.randn(coords.shape[0], noise_channels, device=self.device),
            # coords=coords_dev,
        # )
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        
        default_sampler_tex = samplers.FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=1e-5,
            resolution=flow_model.resolution,
        )
        active_sampler_tex = self._get_sampler(sampler, default_sampler_tex)
        
        if self.low_vram:
            flow_model.to(self.device)
            
        slat = active_sampler_tex.sample(
            flow_model,
            noise,
            conds=conds,
            views=views,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            concat_cond=shape_slat_normalized,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat (MultiView)",
        ).samples
        
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        #del coords_dev
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()
            
        return slat

    def sample_shape_slat_cascade_advanced(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        low_res_sampler_params: dict = {},
        high_res_sampler_params: dict = {},
        max_num_tokens: int = 999999,
        sparse_structure_resolution: int = 32,
        low_res_sampler_name: str = 'euler',
        high_res_sampler_name: str = 'euler',
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # LR

        if self.low_vram:
            lr_cond = self._cond_to(lr_cond, self.device)
            cond = self._cond_to(cond, self.device)

        coords_dev = coords.to(self.device)                         
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **low_res_sampler_params}
        if self.low_vram:
            flow_model_lr.to(self.device)            
        
        active_sampler = self._get_sampler(
            low_res_sampler_name, 
            self.shape_slat_sampler, 
            steps=sampler_params.get("steps", 12)
        )
            
        slat = active_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (LR)",
        ).samples
        if self.low_vram:
            flow_model_lr.cpu()
            self._cleanup_cuda()                                
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            lr_cond = self._cond_cpu(lr_cond)
            self._cleanup_cuda()

        # Upsample       
        self.load_shape_slat_decoder()
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        
        if not self.keep_models_loaded:
            self.unload_shape_slat_decoder()
            
        ratio = (sparse_structure_resolution / 32)
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                print(f"Num Tokens: {num_tokens}")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 512
                break
        
        coords_dev = coords.to(self.device)                                           
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )        
        
        sampler_params = {**self.shape_slat_sampler_params, **high_res_sampler_params}
        
        active_sampler_hr = self._get_sampler(
            high_res_sampler_name, 
            self.shape_slat_sampler, 
            steps=sampler_params.get("steps", 12)
        )
        
        if self.low_vram:
            flow_model.to(self.device)
        slat = active_sampler_hr.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (HR)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()

        return slat, hr_resolution   

    def sample_tex_slat_advanced(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
        sampler_name: str = 'euler',
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        proj_image_cond_model=None,
        proj_images=None,
        **kwargs,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
        """
        if self.low_vram:
            cond = self._cond_to(cond, self.device)
        # Rebuild proj cond with shape_slat coords if proj model provided
        if proj_image_cond_model is not None and proj_images is not None:
            tex_coords = shape_slat.coords.cpu()
            tex_grid_res = int(tex_coords[:, 1:].max().item()) + 1
            cond = self.get_proj_cond_shape(proj_image_cond_model, proj_images, tex_coords,
                                            grid_resolution_override=tex_grid_res)
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        
        active_sampler = self._get_sampler(
            sampler_name, 
            self.tex_slat_sampler, 
            steps=sampler_params.get("steps", 12)
        )
        
        if self.low_vram:
            flow_model.to(self.device)
        slat = active_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                    

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()                         
        return slat        
            
    @torch.no_grad()
    def run_cascade(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        low_res_shape_slat_sampler_params: dict = {},
        high_res_shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        pipeline_type: str = '1024_cascade',
        max_num_tokens: int = 999999,
        sparse_structure_resolution: int = 32,
        generate_texture_slat = True,
        use_tiled: bool = True,
        pbar = None,
        sparse_structure_sampler = 'euler',
        low_res_shape_sampler = 'euler',
        high_res_shape_sampler = 'euler',
        tex_sampler = 'euler',
        max_views: int = 4
    ) -> List[MeshWithVoxel]:
        
        if isinstance(image, (list, tuple)):
            images = list(image)
        else:
            images = [image]
            
        seed_all(seed)
        
        # Load sparse structure model early so we can check image_attn_mode
        self.load_sparse_structure_model()
        ss_model = self.models['sparse_structure_flow_model']
        ss_attn_mode = getattr(ss_model, 'image_attn_mode', None)

        # Get Image Cond
        self.load_image_cond_model()
        # Use proj conditioning if sparse structure model requires it
        if ss_attn_mode == 'proj':
            proj_cond_model = self.load_pixal3d_image_cond_ss()
            cond_512 = self.get_proj_cond_ss(images, image_cond_model=proj_cond_model)
        else:
            # Multi-view conditioning happens inside get_cond()
            cond_512 = self.get_cond(images, 512, max_views = max_views)
        cond_1024 = self.get_cond(images, 1024, max_views = max_views) if pipeline_type != '512' else None
        
        if pbar is not None:
            pbar.update(1)
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()       
                
        # Sampling Sparse Structure
        self.load_sparse_structure_model()        
        coords = self.sample_sparse_structure(
            cond_512, sparse_structure_resolution,
            num_samples, sparse_structure_sampler_params,
            sampler=sparse_structure_sampler
        )
        
        if pbar is not None:
            pbar.update(1)
        
        if not self.keep_models_loaded:
            self.unload_sparse_structure_model()
        
        # Sampling Shape
        if pipeline_type == '1024_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()            
            shape_slat, res = self.sample_shape_slat_cascade_advanced(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, low_res_shape_slat_sampler_params, high_res_shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution,
                low_res_shape_sampler, high_res_shape_sampler
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat_advanced(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, tex_sampler
                )
                
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif pipeline_type == '1536_cascade':
            self.load_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()            
            shape_slat, res = self.sample_shape_slat_cascade_advanced(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, low_res_shape_slat_sampler_params, high_res_shape_slat_sampler_params,
                max_num_tokens,
                sparse_structure_resolution,
                low_res_shape_sampler, high_res_shape_sampler
            )
            
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
                self.unload_shape_slat_flow_model_1024()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat_advanced(
                    cond_1024, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, tex_sampler
                )
                
            if pbar is not None:
                pbar.update(1)
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()         
            
        torch.cuda.empty_cache()
        if generate_texture_slat:
            out_mesh = self.decode_latent(shape_slat, tex_slat, res, use_tiled=use_tiled)
        else:
            out_mesh = self.decode_latent(shape_slat, None, res, use_tiled=use_tiled)
        torch.cuda.empty_cache()
        pbar.update(1)              

        return out_mesh

    def preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Preprocess the input mesh.
        """
        mesh = mesh.copy()
        vertices = mesh.vertices.copy()
        
        vertices_min = vertices.min(axis=0)
        vertices_max = vertices.max(axis=0)
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        tmp = vertices[:, 1].copy()
        vertices[:, 1] = -vertices[:, 2]
        vertices[:, 2] = tmp
        assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), 'vertices out of range'
        
        mesh.vertices = vertices
        return mesh

    def encode_shape_slat(
        self,
        mesh: trimesh.Trimesh,
        resolution: int = 1024,
        use_tiled_encoder: bool = False,
        encoder_tile_size: int = 512,
        encoder_overlap: int = 24,
    ) -> SparseTensor:
        """
        Encode the meshes to structured latent.

        Args:
            mesh (trimesh.Trimesh): The mesh to encode.
            resolution (int): The resolution of mesh
            use_tiled_encoder (bool): Whether to use spatial chunking during encoding to save VRAM.
            encoder_tile_size (int): The size of the spatial chunks.
            encoder_overlap (int): The overlap between spatial chunks.
        
        Returns:
            SparseTensor: The encoded structured latent.
        """
        print('Converting mesh to flexible dual grid ...')
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
            
        vertices = SparseTensor(
            feats=dual_vertices * resolution - voxel_indices,
            coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
        ).to(self.device)
        intersected = vertices.replace(intersected).to(self.device)
            
        self.load_shape_slat_encoder()
            
        if self.low_vram:
            self.models['shape_slat_encoder'].to(self.device)
            
        if use_tiled_encoder:
            print(f"Encoding shape slat with tiles (size: {encoder_tile_size}, overlap: {encoder_overlap})...")
            
        import inspect
        encoder_forward = getattr(self.models['shape_slat_encoder'], 'forward', None)
        has_use_tiled = False
        if encoder_forward is not None:
            try:
                sig = inspect.signature(encoder_forward)
                has_use_tiled = 'use_tiled' in sig.parameters
            except Exception:
                pass

        if has_use_tiled:
            shape_slat = self.models['shape_slat_encoder'](
                vertices, intersected, 
                use_tiled=use_tiled_encoder, 
                tile_size=encoder_tile_size, 
                overlap=encoder_overlap
            )
        else:
            shape_slat = self.models['shape_slat_encoder'](vertices, intersected)
        
        if self.low_vram:
            self.models['shape_slat_encoder'].cpu()
            
        if not self.keep_models_loaded:
            self.unload_shape_slat_encoder()
            
        return shape_slat

    def postprocess_mesh(
        self,
        mesh: trimesh.Trimesh,
        pbr_voxel: SparseTensor,
        resolution: int = 1024,
        texture_size: int = 1024,
        texture_alpha_mode = 'OPAQUE',
        double_side_material = True,
        bake_on_vertices = False,
        use_custom_normals = False,
        uv_unwrap_method = 'Xatlas',
        mesh_cluster_threshold_cone_half_angle_rad = 60.0,
        inpainting = 'telea'
    ):        
        vertices = mesh.vertices
        faces = mesh.faces
        normals = np.asarray(mesh.vertex_normals).copy()
        
        vertices_torch = torch.from_numpy(vertices).float().cuda()
        faces_torch = torch.from_numpy(faces).int().cuda()
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = mesh.visual.uv.copy()
            uvs[:, 1] = 1 - uvs[:, 1]
            uvs_torch = torch.from_numpy(uvs).float().cuda()
        else:
            if self.low_vram:
                print("[Trellis2] Low VRAM: Offloading voxel grid to CPU for mesh unwrapping...")
                pbr_voxel = pbr_voxel.cpu()
                self._cleanup_cuda()

            _cumesh = cumesh.CuMesh()
            _cumesh.init(vertices_torch, faces_torch)
            print('Unwrapping mesh ...')
            
            if uv_unwrap_method == 'Blender':
                from ..utils.unwrap_utils import blender_unwrap_glb, check_bpy_available
                if not check_bpy_available():
                    print("[Trellis2] WARNING: bpy not installed, falling back to Xatlas.")
                    uv_unwrap_method = 'Xatlas'
                    
            if uv_unwrap_method == 'Blender':
                _out_verts, _out_faces = _cumesh.read()
                new_verts, new_faces, new_uvs, vmap_blender = blender_unwrap_glb(_out_verts.cpu().numpy(), _out_faces.cpu().numpy())
                if new_verts is None:
                    print("[Trellis2] ERROR: Blender unwrap failed, falling back to Xatlas.")
                    uv_unwrap_method = 'Xatlas'
                else:
                    vertices_torch = torch.from_numpy(new_verts).cuda().float()
                    faces_torch = torch.from_numpy(new_faces).cuda().int()
                    uvs_torch = torch.from_numpy(new_uvs).cuda().float()
                    vmap = torch.from_numpy(vmap_blender).cuda().long()
                    
            if uv_unwrap_method == 'Smart':
                from ..utils.unwrap_utils import python_smart_unwrap_glb
                _out_verts, _out_faces = _cumesh.read()
                new_verts, new_faces, new_uvs, vmap_blender = python_smart_unwrap_glb(
                    _out_verts.cpu().numpy(), 
                    _out_faces.cpu().numpy(),
                    angle_limit=np.radians(mesh_cluster_threshold_cone_half_angle_rad)
                )
                if new_verts is None:
                    print("[Trellis2] ERROR: Smart unwrap failed, falling back to Xatlas.")
                    uv_unwrap_method = 'Xatlas'
                else:
                    vertices_torch = torch.from_numpy(new_verts).cuda().float()
                    faces_torch = torch.from_numpy(new_faces).cuda().int()
                    uvs_torch = torch.from_numpy(new_uvs).cuda().float()
                    vmap = torch.from_numpy(vmap_blender).cuda().long()
                    
            if uv_unwrap_method == 'Xatlas':
                vertices_torch, faces_torch, uvs_torch, vmap = _cumesh.uv_unwrap(
                    compute_charts_kwargs={
                        "threshold_cone_half_angle_rad": np.radians(mesh_cluster_threshold_cone_half_angle_rad),
                        "refine_iterations": 0,
                        "global_iterations": 1,
                        "smooth_strength": 1,
                    },
                    return_vmaps=True,
                    verbose=True,
                )
            
            del _cumesh
            gc.collect()      

            if self.low_vram:
                print("[Trellis2] Low VRAM: Moving voxel grid back to GPU...")
                pbr_voxel = pbr_voxel.to(self.device)
                self._cleanup_cuda()
            
            vertices_torch = vertices_torch.cuda()
            faces_torch = faces_torch.cuda()
            uvs_torch = uvs_torch.cuda()
            vertices = vertices_torch.cpu().numpy()
            faces = faces_torch.cpu().numpy()
            uvs = uvs_torch.cpu().numpy()
            normals = normals[vmap.cpu().numpy()]

        # --- Branch: Bake On Vertices (skip UV unwrapping and texture creation) ---
        if bake_on_vertices:
            aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
            
            # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
            if isinstance(aabb, (list, tuple)):
                aabb = np.array(aabb)
            if isinstance(aabb, np.ndarray):
                aabb = torch.tensor(aabb, dtype=torch.float32, device=pbr_voxel.coords.device)

            voxel_size = 1 / resolution

            # Calculate grid dimensions based on AABB and voxel size                
            if voxel_size is not None:
                if isinstance(voxel_size, float):
                    voxel_size = [voxel_size, voxel_size, voxel_size]
                if isinstance(voxel_size, (list, tuple)):
                    voxel_size = np.array(voxel_size)
                if isinstance(voxel_size, np.ndarray):
                    voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=pbr_voxel.coords.device)
                grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
            else:
                if isinstance(grid_size, int):
                    grid_size = [grid_size, grid_size, grid_size]
                if isinstance(grid_size, (list, tuple)):
                    grid_size = np.array(grid_size)
                if isinstance(grid_size, np.ndarray):
                    grid_size = torch.tensor(grid_size, dtype=torch.int32, device=pbr_voxel.coords.device)
                voxel_size = (aabb[1] - aabb[0]) / grid_size
            
            print('Baking colors on vertices...')
            out_vertices = vertices_torch
            out_faces = faces_torch
            out_normals = normals           
            
            # Sample attributes directly at vertex positions from the voxel grid
            # No BVH mapping needed - the voxel grid contains all the color information
            vertex_attrs = grid_sample_3d(
                pbr_voxel.feats,
                pbr_voxel.coords,
                shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
                grid=((out_vertices - aabb[0]) / voxel_size).reshape(1, -1, 3),
                mode='trilinear',
            )
            
            # Extract base color and alpha per vertex (vertex_attrs shape: N_vertices x C)
            base_color_idx = self.pbr_attr_layout['base_color']
            alpha_idx = self.pbr_attr_layout['alpha']
            
            # Get RGB values and squeeze any extra dimensions to get (N, 3)
            vertex_colors_rgb = vertex_attrs[..., base_color_idx].cpu().numpy()
            vertex_colors_rgb = np.squeeze(vertex_colors_rgb)  # Remove batch dims if any
            if vertex_colors_rgb.ndim == 1:
                vertex_colors_rgb = vertex_colors_rgb[None, :]  # Ensure at least 2D
            vertex_colors_rgb = np.clip(vertex_colors_rgb * 255, 0, 255).astype(np.uint8)
            
            # Handle alpha based on texture_alpha_mode
            if texture_alpha_mode == "OPAQUE":
                # For OPAQUE mode, use full alpha (255)
                vertex_alpha = np.full((vertex_colors_rgb.shape[0], 1), 255, dtype=np.uint8)
            else:
                vertex_alpha = vertex_attrs[..., alpha_idx].cpu().numpy()
                vertex_alpha = np.squeeze(vertex_alpha)  # Remove batch dims if any
                vertex_alpha = np.clip(vertex_alpha * 255, 0, 255).astype(np.uint8)
                # Ensure alpha is 2D with shape (N, 1)
                if vertex_alpha.ndim == 1:
                    vertex_alpha = vertex_alpha[:, None]
            
            # Combine into RGBA
            vertex_colors_rgba = np.concatenate([vertex_colors_rgb, vertex_alpha], axis=-1)
            
            print("Finalizing mesh with vertex colors...")
            
            vertices_np = out_vertices.cpu().numpy()
            faces_np = out_faces.cpu().numpy()
            normals_np = out_normals
            
            # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
            vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
            normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
            
            # Create mesh with vertex colors using ColorVisuals
            if use_custom_normals:
                textured_mesh = trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_normals=normals_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )
            else:
                textured_mesh = trimesh.Trimesh(
                    vertices=vertices_np,
                    faces=faces_np,
                    vertex_colors=vertex_colors_rgba,
                    process=False,
                )                
            
            # Return empty placeholder textures for vertex color mode
            placeholder_texture = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
            return (textured_mesh, placeholder_texture, placeholder_texture,)
                
        # rasterize
        print('Finalizing mesh ...')
        ctx = dr.RasterizeCudaContext()
        uvs_torch = torch.cat([uvs_torch * 2 - 1, torch.zeros_like(uvs_torch[:, :1]), torch.ones_like(uvs_torch[:, :1])], dim=-1).unsqueeze(0)
        rast, _ = dr.rasterize(
            ctx, uvs_torch, faces_torch,
            resolution=[texture_size, texture_size],
        )
        
        torch.cuda.synchronize()
        
        mask = rast[0, ..., 3] > 0
        pos = dr.interpolate(vertices_torch.unsqueeze(0), rast, faces_torch)[0][0]
        
        attrs = torch.zeros(texture_size, texture_size, pbr_voxel.shape[1], device=self.device)
        attrs[mask] = flex_gemm.ops.grid_sample.grid_sample_3d(
            pbr_voxel.feats,
            pbr_voxel.coords,
            shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
            grid=((pos[mask] + 0.5) * resolution).reshape(1, -1, 3),
            mode='trilinear',
        ).float()
        
        torch.cuda.synchronize()
        
        # construct mesh
        mask = mask.cpu().numpy()
        base_color = np.clip(attrs[..., self.pbr_attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., self.pbr_attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., self.pbr_attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., self.pbr_attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        # extend
        if inpainting == 'telea':
            inpainting_algo = cv2.INPAINT_TELEA
        else:
            inpainting_algo = cv2.INPAINT_NS
            
        mask = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask, 3, inpainting_algo)
        metallic = cv2.inpaint(metallic, mask, 1, inpainting_algo)[..., None]
        roughness = cv2.inpaint(roughness, mask, 1, inpainting_algo)[..., None]
        alpha = cv2.inpaint(alpha, mask, 1, inpainting_algo)[..., None]
        
        baseColorTexture = Image.fromarray(np.concatenate([base_color, alpha], axis=-1))
        metallicRoughnessTexture = Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1))
        
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=baseColorTexture,
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=metallicRoughnessTexture,
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode=texture_alpha_mode,
            doubleSided=True,
        )

        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
        normals[:, 1], normals[:, 2] = normals[:, 2], -normals[:, 1]
        uvs[:, 1] = 1 - uvs[:, 1] # Flip UV V-coordinate
        
        if use_custom_normals:
            textured_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=normals,
                process=False,
                visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
            )
        else:
            textured_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False,
                visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
            )
            
        return textured_mesh, baseColorTexture, metallicRoughnessTexture

    @torch.no_grad()
    def texture_mesh(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        resolution: int = 1024,
        texture_size: int = 2048,
        texture_alpha_mode = 'OPAQUE',
        double_side_material = True,
        max_views = 4,
        bake_on_vertices = False,
        use_custom_normals = False,
        uv_unwrap_method: str = 'Xatlas',
        mesh_cluster_threshold_cone_half_angle_rad=60.0,
        use_tiled_encoder: bool = False,
        encoder_tile_size: int = 512,
        encoder_overlap: int = 24,
        use_tiled_decoder: bool = False,
        decoder_tile_size: int = 120,
        decoder_overlap: int = 48,
        sampler: str = None,
        inpainting: str = 'telea',
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ):
        
        self.use_tiled_decoder_for_texture = use_tiled_decoder
        self.tiled_decoder_size = decoder_tile_size
        self.tiled_decoder_overlap = decoder_overlap
        
        mesh = self.preprocess_mesh(mesh)
        seed_all(seed)
        
        # Accept either a single PIL image or a list of PIL images (multi-view)
        if isinstance(image, (list, tuple)):
            images = list(image)
        else:
            images = [image]
        
        self.load_image_cond_model()        
        cond_resolution = resolution
        if cond_resolution>1024:
            cond_resolution = 1024
            
        cond = self.get_cond(images, cond_resolution, max_views = max_views)
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()
        
        shape_slat = self.encode_shape_slat(
            mesh, 
            resolution,
            use_tiled_encoder=use_tiled_encoder,
            encoder_tile_size=encoder_tile_size,
            encoder_overlap=encoder_overlap
        )
        
        if resolution==512:
            self.unload_tex_slat_flow_model_1024()
            self.load_tex_slat_flow_model_512()
            tex_model = self.models['tex_slat_flow_model_512']
            
            tex_slat = self.sample_tex_slat(
                cond, tex_model,
                shape_slat, tex_slat_sampler_params,
                sampler=sampler
            )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_512()
        else:
            self.unload_tex_slat_flow_model_512()
            self.load_tex_slat_flow_model_1024()
            tex_model = self.models['tex_slat_flow_model_1024']
            
            tex_slat = self.sample_tex_slat(
                cond, tex_model,
                shape_slat, tex_slat_sampler_params, sampler=sampler
            )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()

        torch.cuda.empty_cache()
        pbr_voxel = self.decode_tex_slat(tex_slat)
        torch.cuda.empty_cache()
        
        out_mesh, baseColorTexture, metallicRoughnessTexture = self.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size, texture_alpha_mode, double_side_material, bake_on_vertices, use_custom_normals, uv_unwrap_method, mesh_cluster_threshold_cone_half_angle_rad, inpainting)
        return out_mesh, baseColorTexture, metallicRoughnessTexture
        
    @torch.no_grad()
    def texture_mesh_multiview(
        self,
        mesh: trimesh.Trimesh,
        front: Image.Image,
        back: Image.Image,
        left: Image.Image,
        right: Image.Image,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        resolution: int = 1024,
        texture_size: int = 2048,
        texture_alpha_mode = 'OPAQUE',
        double_side_material = True,
        bake_on_vertices = False,
        use_custom_normals = False,
        uv_unwrap_method: str = 'Xatlas',
        mesh_cluster_threshold_cone_half_angle_rad=60.0,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        use_tiled_encoder: bool = False,
        encoder_tile_size: int = 512,
        encoder_overlap: int = 24,
        use_tiled_decoder: bool = False,
        decoder_tile_size: int = 120,
        decoder_overlap: int = 48,
        sampler: str = None,
        inpainting: str = 'telea',
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ):
        
        self.use_tiled_decoder_for_texture = use_tiled_decoder
        self.tiled_decoder_size = decoder_tile_size
        self.tiled_decoder_overlap = decoder_overlap
        
        mesh = self.preprocess_mesh(mesh)
        seed_all(seed)
        
        self.load_image_cond_model()        
        # Collect views
        views_dict = {'front': front}
        if back is not None: views_dict['back'] = back
        if left is not None: views_dict['left'] = left
        if right is not None: views_dict['right'] = right
        
        views_list = list(views_dict.keys())

        # 1. Conditioning
        # Calculate conditioning per view
        conds = {}
        
        self.load_image_cond_model()
        
        if resolution == 512:
             for v, img in views_dict.items():
                c = self.get_cond([img], 512)
                conds[v] = c
        else:
             for v, img in views_dict.items():
                c = self.get_cond([img], 1024)
                conds[v] = c
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()
        
        shape_slat = self.encode_shape_slat(
            mesh, 
            resolution,
            use_tiled_encoder=use_tiled_encoder,
            encoder_tile_size=encoder_tile_size,
            encoder_overlap=encoder_overlap
        )
        
        if resolution==512:
            self.unload_tex_slat_flow_model_1024()
            self.load_tex_slat_flow_model_512()
            tex_model = self.models['tex_slat_flow_model_512']
            
            tex_slat = self.sample_tex_slat_multiview(
                conds, views_list,
                shape_slat=shape_slat, 
                flow_model=tex_model,
                sampler_params=tex_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sampler=sampler,
            )            
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_512()
        else:
            self.unload_tex_slat_flow_model_512()
            self.load_tex_slat_flow_model_1024()
            tex_model = self.models['tex_slat_flow_model_1024']
            
            tex_slat = self.sample_tex_slat_multiview(
                conds, views_list,
                shape_slat=shape_slat, 
                flow_model=tex_model,
                sampler_params=tex_slat_sampler_params,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                sampler=sampler,
            )                          
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
                
        torch.cuda.empty_cache()
        pbr_voxel = self.decode_tex_slat(tex_slat)
        torch.cuda.empty_cache()
        
        out_mesh, baseColorTexture, metallicRoughnessTexture = self.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size, texture_alpha_mode, double_side_material, bake_on_vertices, use_custom_normals, uv_unwrap_method, mesh_cluster_threshold_cone_half_angle_rad, inpainting)
        return out_mesh, baseColorTexture, metallicRoughnessTexture        
    
    def get_coords_from_trimesh(self, mesh, resolution):
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()
        
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
        
        coords = torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)                
        coords = coords.cpu()
        
        #print(coords)
        
        del voxel_indices
        del dual_vertices
        del intersected
        
        if self.low_vram:
            self._cleanup_cuda() 
        
        return coords;
        
    def sample_mesh_slat(
        self,
        mesh_slat,
        cond: dict,
        flow_model,
        resolution: int,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        downsampling = 16,
        use_tiled_upsample: bool = False,
        upsample_tile_size: int = 16,
        upsample_overlap: int = 2,
        sampler: str = None,
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ) -> SparseTensor:
        # Upsample       
        self.load_shape_slat_decoder()
        print('Decoding mesh slat ...')
        if self.low_vram:
            self.models['shape_slat_decoder'].to(self.device)
            self.models['shape_slat_decoder'].low_vram = True
        decoder = self.models['shape_slat_decoder']
        if use_tiled_upsample:
            print(f'[Trellis2] Tiled upsample (tile={upsample_tile_size}, overlap={upsample_overlap}) ...')
            hr_coords = decoder._tiled_upsample(mesh_slat, upsample_times=4,
                                                tile_size=upsample_tile_size,
                                                overlap=upsample_overlap)
        else:
            hr_coords = decoder.upsample(mesh_slat, upsample_times=4)
        if self.low_vram:
            self.models['shape_slat_decoder'].cpu()
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        
        if not self.keep_models_loaded:
            self.unload_shape_slat_decoder()
        
        #downsampling = 16
        lr_resolution = resolution
        # if hr_resolution == 512:
            # downsampling = 16
        # elif hr_resolution == 1024:
            # downsampling = 32
        # elif hr_resolution == 1536:
            # downsampling = 32
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // downsampling)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and resolution >= 1024:
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                hr_resolution = 512
                break
        
        coords_dev = coords.to(self.device)                                           
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            flow_model.to(self.device)
        active_sampler = self._get_sampler(sampler, self.shape_slat_sampler)
        slat = active_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()                                

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        del coords_dev
        if self.low_vram:
            cond = self._cond_cpu(cond)
            self._cleanup_cuda()

        return slat, hr_resolution        
    
    @torch.no_grad()
    def refine_mesh(
        self,
        mesh: trimesh.Trimesh,
        image,
        seed: int = 42,
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        resolution: int = 1024,
        max_num_tokens = 50000,
        generate_texture_slat = True,
        return_latent = False,
        downsampling = 16,
        use_tiled: bool = True,
        max_views: int = 4,
        use_tiled_encoder: bool = False,
        encoder_tile_size: int = 512,
        encoder_overlap: int = 24,
        use_tiled_decoder: bool = False,
        decoder_tile_size: int = 120,
        decoder_overlap: int = 48,
        use_tiled_upsample: bool = False,
        upsample_tile_size: int = 16,
        upsample_overlap: int = 2,
        sampler: str = None,
        verbose: bool = False,
        dino_lock: float = 0.0,
        dino_substeps: int = 4,
        dino_foundation_cap: float = 0.92,
        **kwargs,
    ):
        self.use_tiled_decoder_for_texture = use_tiled_decoder
        self.tiled_decoder_size = decoder_tile_size
        self.tiled_decoder_overlap = decoder_overlap
        mesh = self.preprocess_mesh(mesh)
        seed_all(seed)
        
        self.load_image_cond_model()
        
        if isinstance(image, (list, tuple)):
            images = list(image)
        else:
            images = [image]        
        
        if resolution == 512:
            cond = self.get_cond(images, 512, max_views = max_views)
        else:
            cond = self.get_cond(images, 1024, max_views = max_views)
        
        if not self.keep_models_loaded:
            self.unload_image_cond_model()        
        
        mesh_slat = self.encode_shape_slat(
            mesh, 
            resolution,
            use_tiled_encoder=use_tiled_encoder,
            encoder_tile_size=encoder_tile_size,
            encoder_overlap=encoder_overlap
        )
        
        if resolution==512:
            self.unload_shape_slat_flow_model_1024()
            self.load_shape_slat_flow_model_512()            
            shape_slat, res = self.sample_mesh_slat(
                mesh_slat,
                cond,
                self.models['shape_slat_flow_model_512'],
                512,
                shape_slat_sampler_params,
                max_num_tokens,
                downsampling,
                use_tiled_upsample=use_tiled_upsample,
                upsample_tile_size=upsample_tile_size,
                upsample_overlap=upsample_overlap,
                sampler=sampler,
            )
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_512()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_1024()
                self.load_tex_slat_flow_model_512()
                tex_slat = self.sample_tex_slat(
                    cond, self.models['tex_slat_flow_model_512'],
                    shape_slat, tex_slat_sampler_params,
                    sampler=sampler
                )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_512()                
        elif resolution == 1024:
            self.unload_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()            
            shape_slat, res = self.sample_mesh_slat(
                mesh_slat,
                cond,
                self.models['shape_slat_flow_model_1024'],
                1024,
                shape_slat_sampler_params,
                max_num_tokens,
                downsampling,
                use_tiled_upsample=use_tiled_upsample,
                upsample_tile_size=upsample_tile_size,
                upsample_overlap=upsample_overlap,
                sampler=sampler,
            )
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_1024()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=sampler
                )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()
        elif resolution == 1536:
            self.unload_shape_slat_flow_model_512()
            self.load_shape_slat_flow_model_1024()            
            shape_slat, res = self.sample_mesh_slat(
                mesh_slat,
                cond,
                self.models['shape_slat_flow_model_1024'],
                1536,
                shape_slat_sampler_params,
                max_num_tokens,
                downsampling,
                use_tiled_upsample=use_tiled_upsample,
                upsample_tile_size=upsample_tile_size,
                upsample_overlap=upsample_overlap,
                sampler=sampler,
            )
            
            if not self.keep_models_loaded:
                self.unload_shape_slat_flow_model_1024()
            
            if generate_texture_slat:
                self.unload_tex_slat_flow_model_512()
                self.load_tex_slat_flow_model_1024()
                tex_slat = self.sample_tex_slat(
                    cond, self.models['tex_slat_flow_model_1024'],
                    shape_slat, tex_slat_sampler_params, sampler=sampler
                )
            
            if not self.keep_models_loaded:
                self.unload_tex_slat_flow_model_1024()                 
                
        torch.cuda.empty_cache()
        if generate_texture_slat:
            out_mesh = self.decode_latent(shape_slat, tex_slat, res, use_tiled=use_tiled)
        else:
            out_mesh = self.decode_latent(shape_slat, None, res, use_tiled=use_tiled)
        torch.cuda.empty_cache()
        
        if return_latent:
            if generate_texture_slat:
                return out_mesh, (shape_slat, tex_slat, res)
            else:
                return out_mesh, (shape_slat, None, res)
        else:
            return out_mesh      
