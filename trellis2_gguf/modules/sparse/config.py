from typing import *

CONV = 'flex_gemm'
DEBUG = False
ATTN = 'sdpa'

_ATTN_PKG = {
    'flash_attn': 'flash_attn',
    'flash_attn_3': 'flash_attn_interface',
    'xformers': 'xformers',
    'sdpa': None,
}

def _attn_available(name):
    pkg = _ATTN_PKG.get(name)
    if pkg is None:
        return True
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

def _autodetect_attn():
    global ATTN
    for candidate in ['flash_attn', 'xformers', 'sdpa']:
        if _attn_available(candidate):
            ATTN = candidate
            return
    ATTN = 'sdpa'

def __from_env():
    import os

    global CONV
    global DEBUG
    global ATTN

    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in ['none', 'spconv', 'torchsparse', 'flex_gemm']:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'

    if env_sparse_attn_backend is not None and env_sparse_attn_backend in _ATTN_PKG:
        if _attn_available(env_sparse_attn_backend):
            ATTN = env_sparse_attn_backend
        else:
            print(f"[SPARSE] Requested attn backend '{env_sparse_attn_backend}' not available, auto-detecting...")
            _autodetect_attn()
    else:
        _autodetect_attn()

    print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}")


__from_env()
    

def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']):
    global ATTN
    ATTN = backend
