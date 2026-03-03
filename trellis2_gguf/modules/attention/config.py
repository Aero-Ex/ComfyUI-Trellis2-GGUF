from typing import *

BACKEND = 'sdpa'
DEBUG = False

_BACKEND_PKG = {
    'flash_attn': 'flash_attn',
    'flash_attn_3': 'flash_attn_interface',
    'xformers': 'xformers',
    'sdpa': None,
    'naive': None,
}

def _backend_available(name):
    pkg = _BACKEND_PKG.get(name)
    if pkg is None:
        return True
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

def _autodetect_backend():
    global BACKEND
    for candidate in ['flash_attn', 'xformers', 'sdpa']:
        if _backend_available(candidate):
            BACKEND = candidate
            return
    BACKEND = 'sdpa'

def __from_env():
    import os

    global BACKEND
    global DEBUG

    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')

    if env_attn_backend is not None and env_attn_backend in _BACKEND_PKG:
        if _backend_available(env_attn_backend):
            BACKEND = env_attn_backend
        else:
            print(f"[ATTENTION] Requested backend '{env_attn_backend}' not available, auto-detecting...")
            _autodetect_backend()
    else:
        _autodetect_backend()

    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    print(f"[ATTENTION] Using backend: {BACKEND}")


__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
