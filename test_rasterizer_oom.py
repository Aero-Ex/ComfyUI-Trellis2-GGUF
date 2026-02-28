import torch
import numpy as np
import trimesh
import sys
import os
import gc

# Mock ComfyUI dependencies before importing nodes
import sys
from unittest.mock import MagicMock

# Define mocks
mock_folder_paths = MagicMock()
mock_folder_paths.get_input_directory.return_value = "/tmp"
mock_folder_paths.get_output_directory.return_value = "/tmp"
sys.modules["folder_paths"] = mock_folder_paths

mock_node_helpers = MagicMock()
sys.modules["node_helpers"] = mock_node_helpers

mock_mm = MagicMock()
sys.modules["comfy.model_management"] = mock_mm

mock_comfy = MagicMock()
sys.modules["comfy"] = mock_comfy
mock_comfy_utils = MagicMock()
sys.modules["comfy.utils"] = mock_comfy_utils

# Add path to the comfy texturing custom nodes
custom_node_path = "/home/aero/comfy/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF"
if custom_node_path not in sys.path:
    sys.path.insert(0, custom_node_path)

# Mock specific trellis2_gguf components if needed, or allow them to be imported normally
# since we added custom_node_path to sys.path
try:
    from nodes import Trellis2GGUFPostProcessAndUnWrapAndRasterizer
    from trellis2_gguf.representations.mesh.base import MeshWithVoxel
except ImportError as e:
    print(f"Import failed: {e}")
    # Fallback/Emergency mocks for nodes import
    sys.modules["nodes.trellis2_gguf"] = MagicMock()
    sys.modules["nodes.trellis2_gguf.pipelines"] = MagicMock()
    from nodes import Trellis2GGUFPostProcessAndUnWrapAndRasterizer
    from trellis2_gguf.representations.mesh.base import MeshWithVoxel
import cumesh

def test_rasterizer():
    print("Initializing test...")
    # 1. Create a synthetic mesh (~10.4M faces)
    # subdivisions 9 is 5,242,880 faces. We combine two to get ~10.4M
    print("Generating a ~10.4M face synthetic mesh...")
    m1 = trimesh.creation.icosphere(subdivisions=9)
    m2 = trimesh.creation.icosphere(subdivisions=9)
    m2.vertices += 0.5 # Move it slightly
    sphere = trimesh.util.concatenate([m1, m2])
    
    vertices = torch.from_numpy(sphere.vertices).float().cuda()
    faces = torch.from_numpy(sphere.faces).int().cuda()
    print(f"Mesh: {vertices.shape[0]:,} verts, {faces.shape[0]:,} faces")

    # Assign dummy UVs to bypass uv_unwrap which fails at this scale
    dummy_uvs = np.random.rand(vertices.shape[0], 2).astype(np.float32)
    
    # 2. Create a dummy MeshWithVoxel
    # 2M active voxels with 6 attributes (RGBA, Metallic, Roughness)
    num_voxels = 2000000
    attrs = torch.randn(num_voxels, 6).float().cuda()
    coords = torch.randint(0, 512, (num_voxels, 3)).int().cuda()
    
    pbr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }

    # Mock MeshWithVoxel to have .visual
    class MockMesh(MeshWithVoxel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            class Visual:
                def __init__(self, uv): self.uv = uv
            self.visual = Visual(dummy_uvs)

    mesh_voxel = MockMesh(
        vertices=vertices,
        faces=faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1/512,
        coords=coords,
        attrs=attrs,
        voxel_shape=torch.Size([512, 512, 512]),
        layout=pbr_layout
    )

    # 3. Create BVH
    print("Building BVH...")
    bvh = cumesh.bvh.cuBVH(vertices, faces)
    # Mocking bvh.vertices/faces as the node expects
    bvh.vertices = vertices
    bvh.faces = faces

    # 4. Instantiate Node
    node = Trellis2GGUFPostProcessAndUnWrapAndRasterizer()
    
    # constructed args
    
    # 5. Run process with high texture size
    # texture_size = 4096 (16M texels)
    print("Running rasterizer with 4096 texture...")
    
    args = {
        'mesh': mesh_voxel,
        'mesh_cluster_threshold_cone_half_angle_rad': 60,
        'mesh_cluster_refine_iterations': 0,
        'mesh_cluster_global_iterations': 1,
        'mesh_cluster_smooth_strength': 1,
        'texture_size': 4096,
        'remesh': False,
        'remesh_band': 1.0,
        'remesh_project': 0.0,
        'target_face_num': 7000000,
        'simplify_method': "Cumesh",
        'fill_holes': False,
        'texture_alpha_mode': "OPAQUE",
        'dual_contouring_resolution': "Auto",
        'double_side_material': True,
        'remove_floaters': False,
        'bake_on_vertices': False,
        'use_custom_normals': False,
        'bvh': bvh,
        'remove_inner_faces': False
    }

    # Clear local names IMMEDIATELY
    del mesh_voxel, bvh, sphere, m1, m2, vertices, faces
    
    torch.cuda.empty_cache()
    gc.collect()
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"Free VRAM right before process call: {free_mem:.2f} GB")
    
    try:
        # We need to mock ProgressBar
        import nodes
        class MockPBar:
            def update(self, *args): pass
        nodes.ProgressBar = lambda x: MockPBar()

        # Call process
        node.process(**args)
        print("SUCCESS: Finished without OOM")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\nVRAM Summary:")
    print(torch.cuda.memory_summary())

if __name__ == "__main__":
    # Print GPU info
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total GPU Memory: {total_mem:.2f} GB")
    torch.cuda.empty_cache()
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"Initial Free GPU Memory: {free_mem:.2f} GB")

    test_rasterizer()
