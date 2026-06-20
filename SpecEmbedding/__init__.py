import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
