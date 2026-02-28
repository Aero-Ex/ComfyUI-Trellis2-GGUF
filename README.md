# 🌀 ComfyUI Wrapper for [https://github.com/microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)

---

<img width="883" height="566" alt="{09272892-57D6-4EB8-B27B-6B875916982A}" src="https://github.com/user-attachments/assets/a7788f13-141c-4072-9143-b8b1ee1ead2a" />

---

<img width="980" height="579" alt="{F6FE6B7B-94B7-44C6-8C89-02E7C81EBF7E}" src="https://github.com/user-attachments/assets/ad27111c-beb8-48ef-8613-c533a3a5cacd" />

---

## 📋 Changelog

| Date | Description |
| --- | --- |
| **2026-02-21** | Fixed "Vertical lines" bug |
| **2026-02-17** | Disabled Triton Cache (trying to fix vertical lines bug)<br>Fixed "Weld Vertices"<br>Added "Reconstruct Mesh with Quad" node |
| **2026-02-13** | Added the node "Weld Vertices"<br>Added the resolution 1536 for "Mesh Texturing" |
| **2026-02-12** | Added the node "Mesh With Voxel Multi-View Generator" |
|| Added the node "Mesh Texturing Multi-View" |
|| Added new example workflows |
| **2026-02-10** | Improved progress bar when filling holes with meshlib |
| **2026-02-09** | Fixed "Mesh Texturing" node<br>"mesh_cluster_threshold_cone_half_angle_rad" was not used |
| **2026-02-08** | Fixed "Fill Holes" node progress bar<br>Updated Cumesh package<br>Added "Remesh with Quad" node<br>Added "Batch Simplify Mesh and Export" node|
| **2026-02-07** | Updated Cumesh package<br>Improved "Remesh" node when removing inner layer|
| **2026-02-02** | Added node "Smooth Normals"<br>Useful for "Low Poly" mesh to remove the "blocky" aspect|
|| Added "remove_background" parameter for "PreProcess Image" node<br>Using rembg package|
| **2026-01-30** | Updated Cumesh, updated nodes, updated workflows|
||PostProcess UnWrap and Rasterize : removed fill_holes_max_perimeter <br> using fill holes from Meshlib|
||Remesh : added "remove_inner_faces" -> same algorithm as "Reconstruct Mesh"|
||Mesh Texturing: added "mesh_cluster_threshold_cone_half_angle_rad"|
| **2026-01-29** |Updated cumesh -> Remesh and Reconstruct made by chunk|
| **2026-01-28** |Added the node "Fill Holes With Meshlib"|
||Trying to fix caching issue|
| **2026-01-27** |Added the node "Trellis2ReconstructMesh"|
||"Multiple Images" support for "Mesh Refiner" node|
| **2026-01-21** |Added a "Continue" node|
||Added the option "bake_on_vertices" for "Mesh Texturing" node|
||Added "padding" option for "Preprocess Image" node|
| **2026-01-20** |Added node "Simplify Trimesh"|
||Fixed crash with "remove_infinite_vertices" in "PostProcess Mesh" node|
||Fixed texture generation|
| **2026-01-19** |Updated Cumesh|
| **2026-01-12** |Can pass multiple images to "Mesh Texturing" node (experimental)|
||Applied latest fixes from Microsoft|
| **2026-01-05** |Implemented "Tiled" Decoder|
||Updated Cumesh and O_voxel|

---

## REQUIREMENTS ##

You need to have access to facebook dinov3 models in order to use Trellis.2

[https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)

Clone the repository in ComfyUI models folder under "facebook/dinov3-vitl16-pretrain-lvd1689m"

So in ComfyUI/models/facebook/dinov3-vitl16-pretrain-lvd1689m

---

## ⚙️ Installation Guide

> Tested on **Windows 11** with **Python 3.11** and **Torch = 2.7.0 + cu128**.

### 1. Install Wheels

#### For a standard python environment:

**If you use Torch v2.7.0:**
```bash
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch270/cumesh-0.0.1-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch270/nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch270/nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch270/flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch270/o_voxel-0.0.1-cp311-cp311-win_amd64.whl
```

**If you use Torch v2.8.0:**
```bash
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch280/cumesh-0.0.1-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch280/nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch280/nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch280/flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python -m pip install ComfyUI/custom_nodes/ComfyUI-Trellis2/wheels/Windows/Torch280/o_voxel-0.0.1-cp311-cp311-win_amd64.whl
```

---

#### For ComfyUI Portable:

**If you use Torch v2.7.0:**
```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch270\cumesh-0.0.1-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch270\nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch270\nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch270\flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch270\o_voxel-0.0.1-cp311-cp311-win_amd64.whl
```

**If you use Torch v2.8.0:**
```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch280\cumesh-0.0.1-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch280\nvdiffrast-0.4.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch280\nvdiffrec_render-0.0.0-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch280\flex_gemm-0.0.1-cp311-cp311-win_amd64.whl
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Trellis2\wheels\Windows\Torch280\o_voxel-0.0.1-cp311-cp311-win_amd64.whl
```

---

**Check the folder wheels for the other versions**

---

### 2. Custom Build

#### o_voxel

Use my own version of Trellis.2 here: https://github.com/visualbruno/TRELLIS.2

#### Cumesh 

Use my own version of Cumesh here: https://github.com/visualbruno/CuMesh

---

### 3. Requirements.txt

#### For a standard python environment:

```bash
python -m pip install -r ComfyUI/custom_nodes/ComfyUI-Trellis2/requirements.txt
```

---

#### For ComfyUI Portable:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Trellis2\requirements.txt
```
