"""SpecEmbedding package.

Importing the package is intentionally free of filesystem side effects.
Executable entry points create writable runtime caches when their ``main``
functions run.
"""

import os

# Third-party imports (Numba/Matplotlib) may initialize caches before an
# executable reaches main(). Point them at writable locations without creating
# directories or disabling JIT during package import.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
