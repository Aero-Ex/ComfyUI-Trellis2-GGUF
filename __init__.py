import sys
import os
import torch

# Locate and add ComfyUI-Trellis2 to sys.path to prepare monkeypatches
custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trellis2_path = os.path.join(custom_nodes_dir, "ComfyUI-Trellis2")
if os.path.exists(trellis2_path) and trellis2_path not in sys.path:
    sys.path.append(trellis2_path)

# ── Monkeypatch DinoV3ProjFeatureExtractor.forward ─────────────────────────
try:
    from trellis2.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    original_forward = DinoV3ProjFeatureExtractor.forward

    def optimized_forward(
        self,
        image,
        camera_angle_x=None,
        distance=None,
        mesh_scale=None,
        transform_matrix=None,
    ):
        if self.grid_resolution < 32:
            return original_forward(self, image, camera_angle_x, distance, mesh_scale, transform_matrix)

        import torch
        import numpy as np
        from PIL import Image

        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((self.image_size, self.image_size), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        B = image.shape[0]
        
        if self.use_naf_upsample:
            image_for_naf = image.clone()
        
        image = self.transform(image)
        
        with torch.no_grad():
            z = self.extract_features(image)
            
            z_clstoken = z[:, 0:1]
            num_reg = getattr(self.model.config, 'num_register_tokens', 4)
            z_regtokens = z[:, 1:1+num_reg]
            z_patchtokens = z[:, 1+num_reg:]
            
            z_patchtokens_spatial = z_patchtokens.reshape(
                B, self.patch_number, self.patch_number, -1
            )
            
            if camera_angle_x is None or distance is None or mesh_scale is None:
                raise ValueError("camera_angle_x, distance, and mesh_scale must be provided")
            
            z_proj_lr = self.proj_grid(
                z_patchtokens_spatial, 
                camera_angle_x, 
                distance, 
                mesh_scale,
                transform_matrix
            )
            
            if self.use_naf_upsample:
                self._load_naf()
                lr_features_bchw = z_patchtokens_spatial.permute(0, 3, 1, 2)

                K = getattr(self, 'naf_tile_factor', 1) or 1
                if K <= 1:
                    hr_features = self.naf_model(
                        image_for_naf, lr_features_bchw, self.naf_target_size
                    )
                    z_proj_hr = self.proj_grid(
                        hr_features,
                        camera_angle_x,
                        distance,
                        mesh_scale,
                        transform_matrix,
                        BHWC=False
                    )
                    del hr_features
                else:
                    z_proj_hr = self._proj_naf_tiled(
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

    DinoV3ProjFeatureExtractor.forward = optimized_forward
    print("[Trellis2-GGUF] Successfully monkeypatched DinoV3ProjFeatureExtractor.forward")
except Exception as e:
    print(f"[Trellis2-GGUF] Warning: Failed to monkeypatch DinoV3ProjFeatureExtractor.forward: {e}")

# ── Monkeypatch Trellis2ImageTo3DPipeline.get_proj_cond_shape ─────────────
try:
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor

    def optimized_get_proj_cond_shape(
        self,
        image_cond_model,
        image,
        coords,
        camera_angle_x=0.8575560450553894,
        distance=2.0,
        mesh_scale=1.0,
        grid_resolution_override=None,
    ):
        device = self.device
        if camera_angle_x is None:
            cam_angle = self.get_moge_camera_config(image)
        else:
            cam_angle = camera_angle_x
            
        if not torch.is_tensor(cam_angle):
            cam_angle = torch.tensor([cam_angle], device=device)
            
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
        
        B = image.shape[0] if isinstance(image, torch.Tensor) else len(image)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
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

    Trellis2ImageTo3DPipeline.get_proj_cond_shape = optimized_get_proj_cond_shape
    print("[Trellis2-GGUF] Successfully monkeypatched Trellis2ImageTo3DPipeline.get_proj_cond_shape")
except Exception as e:
    print(f"[Trellis2-GGUF] Warning: Failed to monkeypatch Trellis2ImageTo3DPipeline.get_proj_cond_shape: {e}")


from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]