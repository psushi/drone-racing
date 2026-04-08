import platform

from crazyflow.utils import enable_cache

import ece484_fly.envs  # noqa: F401, register environments with gymnasium

# Avoid loading stale AOT cache artifacts on macOS (can trigger CPU feature mismatch warnings).
if platform.system() != "Darwin":
    enable_cache()  # Enable persistent caching of jax functions
