import os
import ctypes
import importlib.util

def force_load_bpy_libs():
    """
    Forcefully load bpy's bundled libembree library globally to prevent 
    symbol conflicts with older versions (e.g. from pymeshlab).
    """
    try:
        spec = importlib.util.find_spec('bpy')
        if spec:
            bpy_dir = os.path.dirname(spec.origin)
            embree_path = os.path.join(bpy_dir, "lib", "libembree4.so.4")
            if os.path.exists(embree_path):
                # Use os module for RTLD flags as they are more complete than ctypes in some environments
                mode = os.RTLD_GLOBAL | os.RTLD_NOW
                if hasattr(os, 'RTLD_DEEPBIND'):
                    mode |= os.RTLD_DEEPBIND
                
                ctypes.CDLL(embree_path, mode=mode)
                print(f"[Trellis2-GGUF] Pre-startup: Successfully force-loaded {embree_path} globally.")
    except Exception as e:
        print(f"[Trellis2-GGUF] Pre-startup warning: Failed to force-load bpy libs: {e}")

# Run the fix
force_load_bpy_libs()
