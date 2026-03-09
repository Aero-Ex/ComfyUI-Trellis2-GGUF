import torch
import numpy as np
import trimesh
import os
import time
import tempfile

def check_bpy_available():
    try:
        import bpy
        import bmesh
        return True
    except Exception:
        return False

def python_smart_unwrap_glb(vertices: np.ndarray, faces: np.ndarray, margin=0.01, angle_limit=1.15192):
    try:
        from smart_uv import smart_uv_unwrap
    except ImportError:
        # Try finding it in the sibling directory Smart-UV-Projection
        import sys
        import os
        lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Smart-UV-Projection")
        if os.path.exists(lib_path) and lib_path not in sys.path:
            sys.path.append(lib_path)
        
        try:
            from smart_uv import smart_uv_unwrap
        except ImportError:
            print(f"[Trellis2-GGUF] Error: smart_uv package not found at {lib_path}. Please install it.")
            raise
        
    print(f"[Trellis2-GGUF] Starting pure Python Smart UV unwrapping...")
    t0 = time.time()
    
    new_vertices, new_faces, new_uvs, vmap = smart_uv_unwrap(
        vertices, faces, margin=margin, angle_limit=angle_limit, area_weight=0.0
    )
    
    print(f"[Trellis2-GGUF] Pure Python Smart UV project completed in {time.time() - t0:.2f}s")
    print(f"[Trellis2-GGUF] Original vertices: {len(vertices)}, New Vertices: {len(new_vertices)}")
    
    return new_vertices, new_faces, new_uvs, vmap

def blender_unwrap_glb(vertices: np.ndarray, faces: np.ndarray, margin=0.01):
    import bpy
    print(f"[Trellis2-GGUF] Starting Blender native unwrapping via temp GLB...")
    t0 = time.time()
    # 1. Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # 2. Inject mesh directly via PyData (prevents coordinate rotation from GLTF import)
    mesh_data = bpy.data.meshes.new("mesh")
    # from_pydata is extremely slow for large meshes, use foreach_set instead
    mesh_data.vertices.add(len(vertices))
    # Blender expects flattened 1D arrays for foreach_set
    mesh_data.vertices.foreach_set("co", vertices.flatten())
    
    mesh_data.loops.add(len(faces) * 3)
    mesh_data.polygons.add(len(faces))
    
    mesh_data.loops.foreach_set("vertex_index", faces.flatten())
    
    loop_starts = np.arange(0, len(faces) * 3, 3, dtype=np.int32)
    loop_totals = np.full(len(faces), 3, dtype=np.int32)
    
    mesh_data.polygons.foreach_set("loop_start", loop_starts)
    mesh_data.polygons.foreach_set("loop_total", loop_totals)
    
    mesh_data.update()
    mesh_data.validate()
    
    mesh_obj = bpy.data.objects.new("obj", mesh_data)
    bpy.context.collection.objects.link(mesh_obj)
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    
    print(f"[Trellis2-GGUF] Mesh injected directly into Blender in {time.time() - t0:.2f}s")
    
    # 3. Unwrap
    t1 = time.time()
    
    # Ensure there is a UV layer
    if not mesh_obj.data.uv_layers:
        mesh_obj.data.uv_layers.new()
        
    mesh_obj.data.uv_layers.active = mesh_obj.data.uv_layers[0]

    # Enter EDIT mode and select all
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    print("[Trellis2-GGUF] Running smart project...")
    # Smart UV Project using settings from user screenshot
    # Angle Limit: 66 degrees = 1.1519173 radians
    # Island Margin: 0.0
    # Scale to Bounds: False
    bpy.ops.uv.smart_project(
        angle_limit=1.15192, 
        margin_method='SCALED', 
        island_margin=0.0,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=False
    )
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print(f"[Trellis2-GGUF] Smart UV project completed in {time.time() - t1:.2f}s")
    
    # 4. Extract data
    t2 = time.time()
    
    mesh = mesh_obj.data
    mesh.calc_loop_triangles() # Ensure triangles for extraction
    
    V = len(mesh.vertices)
    F = len(mesh.loop_triangles)
    
    blender_vertices = np.empty(V * 3, dtype=np.float32)
    mesh.vertices.foreach_get('co', blender_vertices)
    blender_vertices = blender_vertices.reshape(-1, 3)
    
    # Extract loop indices (which reference vertices) from triangles
    faces_loops = np.empty(F * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get('loops', faces_loops)

    # Convert loop indices to vertex indices
    faces_verts = np.empty(F * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get('vertices', faces_verts)
    blender_faces = faces_verts.reshape(-1, 3)
    
    uv_layer = mesh.uv_layers.active.data
    uvs = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    uv_layer.foreach_get('uv', uvs)
    uvs = uvs.reshape(-1, 2)
    
    # UVs are per loop, 
    flat_uvs = uvs[faces_loops]
    
    # Clamp UVs strictly to [0, 1] to prevent nvdiffrast clipping outside viewport
    flat_uvs = np.clip(flat_uvs, 0.0, 1.0)
    
    # Deduplicate based on vertex coordinates + UVs to recreate Trimesh split format
    flat_faces = blender_faces.reshape(-1)
    quantized_uvs = (flat_uvs * 1e5).astype(np.int32)
    
    pack = np.column_stack((flat_faces, quantized_uvs))
    
    unique_pairs, unique_indices, inverse_indices = np.unique(pack, axis=0, return_index=True, return_inverse=True)
    
    vmap = flat_faces[unique_indices].astype(np.int32)
    new_vertices = blender_vertices[vmap]
    new_uvs = flat_uvs[unique_indices]
    new_faces = inverse_indices.reshape(F, 3).astype(np.int32)
    
    print(f"[Trellis2-GGUF] Original vertices: {len(vertices)}, New Vertices: {len(new_vertices)}")
        
    return new_vertices, new_faces, new_uvs, vmap
