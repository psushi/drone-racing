import jax
import numpy as np

def normalize_actions(actions: np.ndarray, actions_low: np.ndarray, actions_high: np.ndarray) -> np.ndarray:
    return np.clip(actions, actions_low, actions_high)


def select_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        if jax.devices("gpu"):
            return "gpu"
    except RuntimeError:
        pass
    return "cpu"

