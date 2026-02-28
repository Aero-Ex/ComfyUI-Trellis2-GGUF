import torch
import trimesh
import gc
import sys
import time

# Add path to the comfy texturing custom nodes to test the remeshing functions directly
sys.path.append("/home/aero/comfy/ComfyUI/custom_nodes/ComfyUI-Trellis2-GGUF")
import cumesh.remeshing

def test_mesh(mesh_path=None):
    if mesh_path and os.path.exists(mesh_path):
        print(f"Loading real mesh from {mesh_path}...")
        sphere = trimesh.load(mesh_path, force='mesh')
    else:
        print("Generating a 30M face synthetic mesh (icosphere)...")
        # Subdivisions 10 yields 20,971,520 faces. Subdivisions 11 is 83,886,080.
        sphere = trimesh.creation.icosphere(subdivisions=10)
    
    vertices = torch.from_numpy(sphere.vertices).float() * 0.4
    faces = torch.from_numpy(sphere.faces).int()
    
    print(f"Generated mesh with {vertices.shape[0]:,} vertices and {faces.shape[0]:,} faces.")
    
    # We will simulate the same inputs passed into dual contouring
    center = torch.zeros(3, device='cuda')
    scale = 1.0
    resolution = 2048 # Simulate high res dual contouring texturing map
    
    print("\nStarting memory test on Dual Contouring Remeshing...")
    try:
        new_v, new_f = cumesh.remeshing.remesh_narrow_band_dc_quad(
            vertices=vertices,
            faces=faces,
            center=center,
            scale=scale,
            resolution=resolution,
            band=1.0,
            project_back=0.0,
            verbose=True,
            remove_inner_faces=True
        )
        print(f"\nSUCCESS: Remeshed into {new_v.shape[0]:,} vertices and {new_f.shape[0]:,} faces.")
        
        del vertices, faces, sphere
        torch.cuda.empty_cache()
        gc.collect()
        import cumesh as cumesh_module
        
        # --- 1. Performance Test on Mesh Simplification ---
        print("\nStarting performance test on Mesh Simplification...")
        print(f"Simplifying {new_f.shape[0]:,} faces down to 1,000,000...")
        
        mesh = cumesh_module.CuMesh()
        mesh.init(new_v, new_f)
        
        start_time = time.time()
        mesh.simplify(target_num_faces=5_000_000, verbose=True)
        end_time = time.time()
        
        final_v, final_f = mesh.read()
        print(f"\nSUCCESS: Simplified in {end_time - start_time:.2f} seconds.")
        print(f"Final counts: {final_v.shape[0]:,} vertices and {final_f.shape[0]:,} faces.")
        
        if (end_time - start_time) > 300: # 5 minutes threshold
            print("WARNING: Simplification is still quite slow!")
        else:
            print("PERFORMANCE PASS: Simplification completed within reasonable time.")
            
        # Cleanup
        del mesh
        gc.collect()
        torch.cuda.empty_cache()
        
        # --- 2. Memory Test on UV Unwrapping (on the simplified mesh) ---
        print("\nStarting memory test on UV Unwrapping (simplified mesh)...")
        mesh = cumesh_module.CuMesh()
        mesh.init(final_v, final_f)
        
        out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": 1.5,
                "refine_iterations": 2,
                "global_iterations": 2,
                "smooth_strength": 0.5,                
            },
            xatlas_compute_charts_kwargs={
                "max_iterations": 1,
                "max_cost": 100000.0,
                "normal_deviation_weight": 0.0,
                "roundness_weight": 0.0,
                "straightness_weight": 0.0,
                "normal_seam_weight": 0.0,
                "texture_seam_weight": 0.0,
                "max_chart_area": 100.0,
                "max_boundary_length": 50.0,
            },
            return_vmaps=True,
            verbose=True,
        )
        print(f"\nSUCCESS: Unwrapped. UVs shape: {out_uvs.shape[0]:,}")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nCUDA Memory Summary:")
    print(torch.cuda.memory_summary())

if __name__ == "__main__":
    import os
    user_mesh = "/home/aero/comfy/ComfyUI/output/OnlyMesh_00002_.glb"
    test_mesh(user_mesh)
