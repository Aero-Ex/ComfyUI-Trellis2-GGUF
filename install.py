import os
import sys
import subprocess
import platform
import argparse
import shutil
import site

def run_command(command):
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    return True

def is_uv_available():
    try:
        subprocess.check_call([sys.executable, "-m", "uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except:
        return False

def get_env_info():
    info = {}
    
    # OS
    if platform.system() == "Windows":
        info['platform'] = "win_amd64"
    else:
        info['platform'] = "manylinux_2_35_x86_64" # Default specialized tag
        
    # Python version
    py_ver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    info['python'] = py_ver
    
    # Torch and CUDA
    try:
        import torch
        torch_ver = torch.__version__.split('+')[0].split('.')
        info['torch'] = f"torch{torch_ver[0]}{torch_ver[1]}"
        
        if torch.version.cuda:
            cuda_ver = torch.version.cuda.split('.')
            info['cuda'] = f"cu{cuda_ver[0]}{cuda_ver[1]}"
        else:
            info['cuda'] = None
    except ImportError:
        print("PyTorch not found. Please install PyTorch with CUDA support first.")
        sys.exit(1)
        
    return info

def apply_patches(dry_run=False):
    print("\n--- Applying Patches ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    patch_dir = os.path.join(current_dir, "patch")
    
    if not os.path.exists(patch_dir):
        print(f"Patch folder not found at {patch_dir}. Skipping patching.")
        return

    # Mapping: filename in patch/ -> target library name
    patch_mapping = {
        "remeshing.py": "cumesh",
        "flexible_dual_grid.py": "o_voxel"
    }
    
    import importlib.util

    for patch_file, lib_name in patch_mapping.items():
        src = os.path.join(patch_dir, patch_file)
        if not os.path.exists(src):
            continue

        print(f"Searching for target of {patch_file} in {lib_name}...")
        
        lib_path = None
        try:
            spec = importlib.util.find_spec(lib_name)
            if spec and spec.origin:
                lib_path = os.path.dirname(spec.origin)
        except Exception as e:
            print(f"  Error searching for {lib_name}: {e}")

        if not lib_path:
            print(f"  Warning: Library '{lib_name}' not found. Skipping patch {patch_file}.")
            continue

        # Environment-independent search within the library directory
        target_path = None
        for root, dirs, files in os.walk(lib_path):
            if patch_file in files:
                target_path = os.path.join(root, patch_file)
                break
        
        if target_path:
            if dry_run:
                print(f"  [Dry Run] Would patch: {src} -> {target_path}")
            else:
                try:
                    shutil.copy(src, target_path)
                    print(f"  Successfully patched: {target_path}")
                except Exception as e:
                    print(f"  Error patching {target_path}: {e}")
        else:
            print(f"  Warning: Could not find '{patch_file}' within {lib_path}. Library might be a different version.")

def show_recommendations():
    print("\n" + "="*50)
    print("RECOMMENDED ENVIRONMENTS")
    print("="*50)
    print("If you encounter 404 errors, it is likely because your environment")
    print("combination is not yet supported on the wheel repository.")
    print("\nPreferred configurations:")
    print("1. CUDA 12.4 + PyTorch 2.5 (Most compatible)")
    print("2. CUDA 12.6 + PyTorch 2.6 (Latest compatible)")
    print("3. CUDA 12.1 + PyTorch 2.4 (Legacy compatible)")
    print("\nYou can install the preferred Torch version using:")
    print("pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124")
    print("="*50 + "\n")

def install_flash_attn(pip_base, env, dry_run=False):
    import urllib.request
    import re
    
    print("\n--- Installing flash-attn ---")
    index_url = "https://pozzettiandrea.github.io/cuda-wheels/flash-attn/"
    
    try:
        req = urllib.request.Request(index_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
    except Exception as e:
        print(f"  Warning: Could not access flash-attn wheel index: {e}")
        return False

    cuda = env['cuda']
    torch_tag = env['torch']
    python_tag = env['python']
    platform_tag = "linux_x86_64" if platform.system() != "Windows" else "win_amd64"
    
    # Match torch version with optional dot (e.g., torch25 or torch2.5)
    # torch_tag is like 'torch25'
    torch_base = torch_tag[:5] # 'torch'
    torch_ver = torch_tag[5:] # '25'
    torch_pattern = rf"{torch_base}{torch_ver[0]}\.?{torch_ver[1]}" # torch2\.?5
    
    # Regex to find the GitHub release URL for the matching wheel
    pattern = rf'https://github\.com/[^"\s]+flash_attn-[^"\s]+\+{cuda}{torch_pattern}-{python_tag}-{python_tag}-{platform_tag}\.whl'
    
    matches = re.findall(pattern, html)
    if matches:
        wheel_url = matches[0]
       
        print(f"  Found matching wheel: {wheel_url}")
        if dry_run:
            print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', wheel_url])}")
            return True
        else:
            return run_command(pip_base + ["install", wheel_url])
    else:
        print(f"  Warning: No matching flash-attn wheel found for {cuda}, {torch_tag}, {python_tag} on {platform_tag}.")
        print("  Please consider switching to a recommended environment for flash-attn support.")
        return False

def install():
    parser = argparse.ArgumentParser(description="ComfyUI-Trellis2 Installation")
    parser.add_argument("--dry-run", action="store_true", help="Print the steps without installing")
    args = parser.parse_args()

    print("--- ComfyUI-Trellis2 Installation ---")
    if args.dry_run:
        print("(DRY RUN MODE)")
    
    env = get_env_info()
    print(f"Detected Environment:")
    print(f"  OS: {platform.system()}")
    print(f"  Python: {env['python']}")
    print(f"  PyTorch: {env['torch']}")
    print(f"  CUDA: {env['cuda']}")
    
    if not env['cuda']:
        print("Error: CUDA not detected in PyTorch. These wheels require CUDA.")
        sys.exit(1)
    
    # 1. Install standard requirements
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, "requirements.txt")
    
    use_uv = is_uv_available()
    pip_base = [sys.executable, "-m", "uv", "pip"] if use_uv else [sys.executable, "-m", "pip"]
    
    if os.path.exists(requirements_path):
        print(f"\nInstalling requirements from {requirements_path}...")
        if args.dry_run:
            print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', '-r', requirements_path])}")
        else:
            run_command(pip_base + ["install", "-r", requirements_path])

    # 2. Install flash-attn from Pozzetti's wheels
    install_flash_attn(pip_base, env, dry_run=args.dry_run)
    
    # 3. Define other CUDA wheels
    packages = [
        {"name": "cumesh", "tag": "cumesh-latest", "file": "cumesh", "version": "0.0.1", "linux_tag": "manylinux_2_35_x86_64"},
        {"name": "flex-gemm", "tag": "flex_gemm-latest", "file": "flex_gemm", "version": "1.0.0", "linux_tag": "manylinux_2_34_x86_64.manylinux_2_35_x86_64"},
        {"name": "nvdiffrast", "tag": "nvdiffrast-latest", "file": "nvdiffrast", "version": "0.4.0", "linux_tag": "manylinux_2_34_x86_64.manylinux_2_35_x86_64"},
        {"name": "nvdiffrec-render", "tag": "nvdiffrec_render-latest", "file": "nvdiffrec_render", "version": "0.0.1", "linux_tag": "manylinux_2_34_x86_64.manylinux_2_35_x86_64"},
        {"name": "o-voxel", "tag": "o_voxel-latest", "file": "o_voxel", "version": "0.0.1", "linux_tag": "manylinux_2_34_x86_64.manylinux_2_35_x86_64"},
    ]
    
    print("\nInstalling CUDA wheels...")
    
    errors_occurred = False
    for pkg in packages:
        platform_tag = env['platform']
        # Special case for packages that use different manylinux tags
        if platform_tag.startswith("manylinux"):
            platform_tag = pkg['linux_tag']
            
        wheel_name = f"{pkg['file']}-{pkg['version']}+{env['cuda']}{env['torch']}-{env['python']}-{env['python']}-{platform_tag}.whl"
        url = f"https://github.com/PozzettiAndrea/cuda-wheels/releases/download/{pkg['tag']}/{wheel_name}"
        
        print(f"Installing {pkg['name']}...")
        if args.dry_run:
            print(f"  [Dry Run] Would run: {' '.join(pip_base + ['install', url])}")
        else:
            if not run_command(pip_base + ["install", url]):
                print(f"Warning: Failed to install {pkg['name']}.")
                errors_occurred = True

    if errors_occurred:
        show_recommendations()

    # Apply Patches
    apply_patches(dry_run=args.dry_run)

    print("\nInstallation complete!")

if __name__ == "__main__":
    install()
