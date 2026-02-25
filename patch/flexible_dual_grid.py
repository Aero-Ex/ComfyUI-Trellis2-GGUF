from typing import *
import numpy as np
import torch
from .. import _C

from tqdm import tqdm

__all__ = [
    "mesh_to_flexible_dual_grid",
    "flexible_dual_grid_to_mesh",
    "tiled_flexible_dual_grid_to_mesh",
]

@torch.no_grad()
def tiled_flexible_dual_grid_to_mesh(
    coords, dual_vertices, intersected_flag, split_weight,
    aabb, grid_size, tile_size=128, overlap=1, train=False
    ):
    device = coords.device

    # Keep everything on GPU
    min_coord = coords.min(dim=0).values
    max_coord = coords.max(dim=0).values

    # Calculate ranges on CPU just for the loop logic
    x_range = range(min_coord[0].item(), max_coord[0].item() + 1, tile_size)
    y_range = range(min_coord[1].item(), max_coord[1].item() + 1, tile_size)
    z_range = range(min_coord[2].item(), max_coord[2].item() + 1, tile_size)

    all_vertices, all_faces = [], []
    vertex_count = 0

    pbar = tqdm(total=len(x_range)*len(y_range)*len(z_range), desc="Fast Tiled Decoding")

    for x in x_range:
        for y in y_range:
            for z in z_range:
                # 1. Masking directly on GPU
                lower = torch.tensor([x, y, z], device=device)
                upper = lower + tile_size
                mask = torch.all((coords >= lower - overlap) & (coords < upper + overlap), dim=1)

                if mask.any():
                    # 2. CRITICAL: .contiguous() ensures the memory layout is
                    # exactly what the custom CUDA kernel expects.
                    t_coords = coords[mask].contiguous()
                    t_dual = dual_vertices[mask].contiguous()
                    t_inter = intersected_flag[mask].contiguous()
                    t_weight = split_weight[mask].contiguous()

                    if not t_inter.any():
                        pbar.update(1)
                        continue

                    try:
                        t_verts, t_faces = flexible_dual_grid_to_mesh(
                            t_coords, t_dual, t_inter, t_weight,
                            aabb=aabb, grid_size=grid_size, train=train
                        )

                        if t_verts is not None and t_verts.shape[0] > 0:
                            all_vertices.append(t_verts)
                            all_faces.append(t_faces + vertex_count)
                            vertex_count += t_verts.shape[0]
                    except RuntimeError:
                        continue

                pbar.update(1)

    pbar.close()
    if not all_vertices: return None, None

    # Final Merge and Deduplication
    unique_v, inverse = torch.unique(torch.cat(all_vertices, dim=0), dim=0, return_inverse=True)
    return unique_v, inverse[torch.cat(all_faces, dim=0)]


def _init_hashmap(grid_size, capacity, device):
    VOL = (grid_size[0] * grid_size[1] * grid_size[2]).item()

    # If the number of elements in the tensor is less than 2^32, use uint32 as the hashmap type, otherwise use uint64.
    if VOL < 2**32:
        hashmap_keys = torch.full((capacity,), torch.iinfo(torch.uint32).max, dtype=torch.uint32, device=device)
    elif VOL < 2**64:
        hashmap_keys = torch.full((capacity,), torch.iinfo(torch.uint64).max, dtype=torch.uint64, device=device)
    else:
        raise ValueError(f"The spatial size is too large to fit in a hashmap. Get volumn {VOL} > 2^64.")

    hashmap_vals = torch.zeros((capacity,), dtype=torch.uint32, device=device)

    return hashmap_keys, hashmap_vals


@torch.no_grad()
def mesh_to_flexible_dual_grid(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    aabb: Union[list, tuple, np.ndarray, torch.Tensor] = None,
    face_weight: float = 1.0,
    boundary_weight: float = 1.0,
    regularization_weight: float = 0.1,
    timing: bool = False,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Voxelize a mesh into a sparse voxel grid.

    Args:
        vertices (torch.Tensor): The vertices of the mesh.
        faces (torch.Tensor): The faces of the mesh.
        voxel_size (float, list, tuple, np.ndarray, torch.Tensor): The size of each voxel.
        grid_size (int, list, tuple, np.ndarray, torch.Tensor): The size of the grid.
            NOTE: One of voxel_size and grid_size must be provided.
        aabb (list, tuple, np.ndarray, torch.Tensor): The axis-aligned bounding box of the mesh.
            If not provided, it will be computed automatically.
        face_weight (float): The weight of the face term in the QEF when solving the dual vertices.
        boundary_weight (float): The weight of the boundary term in the QEF when solving the dual vertices.
        regularization_weight (float): The weight of the regularization term in the QEF when solving the dual vertices.
        timing (bool): Whether to time the voxelization process.

    Returns:
        torch.Tensor: The indices of the voxels that are occupied by the mesh.
            The shape of the tensor is (N, 3), where N is the number of occupied voxels.
        torch.Tensor: The dual vertices of the mesh.
        torch.Tensor: The intersected flag of each voxel.
    """

    # Load mesh
    vertices = vertices.float()
    faces = faces.int()

    # Voxelize settings
    assert voxel_size is not None or grid_size is not None, "Either voxel_size or grid_size must be provided"

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        assert isinstance(voxel_size, torch.Tensor), f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, but got {type(voxel_size)}"
        assert voxel_size.dim() == 1, f"voxel_size must be a 1D tensor, but got {voxel_size.shape}"
        assert voxel_size.size(0) == 3, f"voxel_size must have 3 elements, but got {voxel_size.size(0)}"

    if grid_size is not None:
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32)
        assert isinstance(grid_size, torch.Tensor), f"grid_size must be an int, list, tuple, np.ndarray, or torch.Tensor, but got {type(grid_size)}"
        assert grid_size.dim() == 1, f"grid_size must be a 1D tensor, but got {grid_size.shape}"
        assert grid_size.size(0) == 3, f"grid_size must have 3 elements, but got {grid_size.size(0)}"

    if aabb is not None:
        if isinstance(aabb, (list, tuple)):
            aabb = np.array(aabb)
        if isinstance(aabb, np.ndarray):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
        assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
        assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
        assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Auto adjust aabb
    if aabb is None:
        min_xyz = vertices.min(dim=0).values
        max_xyz = vertices.max(dim=0).values

        if voxel_size is not None:
            padding = torch.ceil((max_xyz - min_xyz) / voxel_size) * voxel_size - (max_xyz - min_xyz)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5
        if grid_size is not None:
            padding = (max_xyz - min_xyz) / (grid_size - 1)
            min_xyz -= padding * 0.5
            max_xyz += padding * 0.5

        aabb = torch.stack([min_xyz, max_xyz], dim=0).float().cuda()

    # Fill voxel size or grid size
    if voxel_size is None:
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    if grid_size is None:
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()

    # subdivide mesh
    vertices = vertices - aabb[0].reshape(1, 3)
    grid_range = torch.stack([torch.zeros_like(grid_size), grid_size], dim=0).int()

    ret = _C.mesh_to_flexible_dual_grid_cpu(
        vertices,
        faces,
        voxel_size,
        grid_range,
        face_weight,
        boundary_weight,
        regularization_weight,
        timing,
    )

    return ret


def flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    dual_vertices: torch.Tensor,
    intersected_flag: torch.Tensor,
    split_weight: Union[torch.Tensor, None],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    train: bool = False,
):
    """
    Extract mesh from sparse voxel structures using flexible dual grid.

    Args:
        coords (torch.Tensor): The coordinates of the voxels.
        dual_vertices (torch.Tensor): The dual vertices.
        intersected_flag (torch.Tensor): The intersected flag.
        split_weight (torch.Tensor): The split weight of each dual quad. If None, the algorithm
            will split based on minimum angle.
        aabb (list, tuple, np.ndarray, torch.Tensor): The axis-aligned bounding box of the mesh.
        voxel_size (float, list, tuple, np.ndarray, torch.Tensor): The size of each voxel.
        grid_size (int, list, tuple, np.ndarray, torch.Tensor): The size of the grid.
            NOTE: One of voxel_size and grid_size must be provided.
        train (bool): Whether to use training mode.

    Returns:
        vertices (torch.Tensor): The vertices of the mesh.
        faces (torch.Tensor): The faces of the mesh.
    """
    # Static variables
    if not hasattr(flexible_dual_grid_to_mesh, "edge_neighbor_voxel_offset"):
        flexible_dual_grid_to_mesh.edge_neighbor_voxel_offset = torch.tensor([
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],     # x-axis
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],     # y-axis
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],     # z-axis
        ], dtype=torch.int, device=coords.device).unsqueeze(0)
    if not hasattr(flexible_dual_grid_to_mesh, "quad_split_1"):
        flexible_dual_grid_to_mesh.quad_split_1 = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=coords.device, requires_grad=False)
    if not hasattr(flexible_dual_grid_to_mesh, "quad_split_2"):
        flexible_dual_grid_to_mesh.quad_split_2 = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=coords.device, requires_grad=False)
    if not hasattr(flexible_dual_grid_to_mesh, "quad_split_train"):
        flexible_dual_grid_to_mesh.quad_split_train = torch.tensor([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4], dtype=torch.long, device=coords.device, requires_grad=False)

    # AABB
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    assert isinstance(voxel_size, torch.Tensor), f"voxel_size must be a float, list, tuple, np.ndarray, or torch.Tensor, but got {type(voxel_size)}"
    assert voxel_size.dim() == 1, f"voxel_size must be a 1D tensor, but got {voxel_size.shape}"
    assert voxel_size.size(0) == 3, f"voxel_size must have 3 elements, but got {voxel_size.size(0)}"
    assert isinstance(grid_size, torch.Tensor), f"grid_size must be an int, list, tuple, np.ndarray, or torch.Tensor, but got {type(grid_size)}"
    assert grid_size.dim() == 1, f"grid_size must be a 1D tensor, but got {grid_size.shape}"
    assert grid_size.size(0) == 3, f"grid_size must have 3 elements, but got {grid_size.size(0)}"

    # Extract mesh
    N = dual_vertices.shape[0]
    mesh_vertices = (coords.float() + dual_vertices * 0.5) / grid_size - 0.5

    # Store active voxels into hashmap
    hashmap = _init_hashmap(grid_size, 2 * N, device=coords.device)
    _C.hashmap_insert_3d_idx_as_val_cuda(*hashmap, torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1), *grid_size.tolist())

    # Find connected voxels
    edge_neighbor_voxel = coords.reshape(N, 1, 1, 3) + flexible_dual_grid_to_mesh.edge_neighbor_voxel_offset      # (N, 3, 4, 3)
    connected_voxel = edge_neighbor_voxel[intersected_flag]                           # (M, 4, 3)
    M = connected_voxel.shape[0]
    connected_voxel_hash_key = torch.cat([
        torch.zeros((M * 4, 1), dtype=torch.int, device=coords.device),
        connected_voxel.reshape(-1, 3)
    ], dim=1)
    connected_voxel_indices = _C.hashmap_lookup_3d_cuda(*hashmap, connected_voxel_hash_key, *grid_size.tolist()).reshape(M, 4).int()
    connected_voxel_valid = (connected_voxel_indices != 0xffffffff).all(dim=1)
    quad_indices = connected_voxel_indices[connected_voxel_valid].int()                             # (L, 4)
    L = quad_indices.shape[0]

    # Construct triangles
    if not train:
        mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)
        chunk_size = 1048576 # 2^20
        mesh_triangles = torch.empty((L * 2, 3), dtype=torch.int32, device=coords.device)

        for i in range(0, L, chunk_size):
            end = min(i + chunk_size, L)
            quad_indices_chunk = quad_indices[i:end]

            if split_weight is None:
                # if split 1
                atempt_triangles_0 = quad_indices_chunk[:, flexible_dual_grid_to_mesh.quad_split_1]
                v0 = mesh_vertices[atempt_triangles_0[:, 0]]
                v1 = mesh_vertices[atempt_triangles_0[:, 1]]
                v2 = mesh_vertices[atempt_triangles_0[:, 2]]
                v3 = mesh_vertices[atempt_triangles_0[:, 3]] # Wait, split 1 is 0,1,2, 0,2,3
                # Re-indexing to get correct vertices for normals
                # split_1: [0, 1, 2, 0, 2, 3] -> triangles (0,1,2) and (0,2,3)
                # Actually atempt_triangles_0 is (chunk_size, 6)
                # Let's be more precise
                t0 = atempt_triangles_0[:, :3]
                t1 = atempt_triangles_0[:, 3:]
                normals0 = torch.cross(mesh_vertices[t0[:, 1]] - mesh_vertices[t0[:, 0]], mesh_vertices[t0[:, 2]] - mesh_vertices[t0[:, 0]])
                normals1 = torch.cross(mesh_vertices[t1[:, 1]] - mesh_vertices[t1[:, 0]], mesh_vertices[t1[:, 2]] - mesh_vertices[t1[:, 0]])
                align0 = (normals0 * normals1).sum(dim=1, keepdim=True).abs()

                # if split 2
                atempt_triangles_1 = quad_indices_chunk[:, flexible_dual_grid_to_mesh.quad_split_2]
                t0 = atempt_triangles_1[:, :3]
                t1 = atempt_triangles_1[:, 3:]
                normals0 = torch.cross(mesh_vertices[t0[:, 1]] - mesh_vertices[t0[:, 0]], mesh_vertices[t0[:, 2]] - mesh_vertices[t0[:, 0]])
                normals1 = torch.cross(mesh_vertices[t1[:, 1]] - mesh_vertices[t1[:, 0]], mesh_vertices[t1[:, 2]] - mesh_vertices[t1[:, 0]])
                align1 = (normals0 * normals1).sum(dim=1, keepdim=True).abs()

                # select split
                mesh_triangles[i*2:end*2] = torch.where(align0 > align1, atempt_triangles_0, atempt_triangles_1).reshape(-1, 3)
            else:
                split_weight_ws = split_weight[quad_indices_chunk]
                split_weight_ws_02 = split_weight_ws[:, 0] * split_weight_ws[:, 2]
                split_weight_ws_13 = split_weight_ws[:, 1] * split_weight_ws[:, 3]
                mesh_triangles[i*2:end*2] = torch.where(
                    split_weight_ws_02 > split_weight_ws_13,
                    quad_indices_chunk[:, flexible_dual_grid_to_mesh.quad_split_1],
                    quad_indices_chunk[:, flexible_dual_grid_to_mesh.quad_split_2]
                ).reshape(-1, 3)
    else:
        assert split_weight is not None, "split_weight must be provided in training mode"
        mesh_vertices = (coords.float() + dual_vertices) * voxel_size + aabb[0].reshape(1, 3)
        quad_vs = mesh_vertices[quad_indices]
        mean_v02 = (quad_vs[:, 0] + quad_vs[:, 2]) / 2
        mean_v13 = (quad_vs[:, 1] + quad_vs[:, 3]) / 2
        split_weight_ws = split_weight[quad_indices]
        split_weight_ws_02 = split_weight_ws[:, 0] * split_weight_ws[:, 2]
        split_weight_ws_13 = split_weight_ws[:, 1] * split_weight_ws[:, 3]
        mid_vertices = (
            split_weight_ws_02 * mean_v02 +
            split_weight_ws_13 * mean_v13
        ) / (split_weight_ws_02 + split_weight_ws_13)
        mesh_vertices = torch.cat([mesh_vertices, mid_vertices], dim=0)
        quad_indices = torch.cat([quad_indices, torch.arange(N, N + L, device='cuda').unsqueeze(1)], dim=1)
        mesh_triangles = quad_indices[:, flexible_dual_grid_to_mesh.quad_split_train].reshape(-1, 3)

    return mesh_vertices, mesh_triangles
