from jinja2.nodes import For
import numpy as np

def normalize_actions(actions: np.ndarray, actions_low: np.ndarray, actions_high: np.ndarray) -> np.ndarray:
    norm_actions = np.tanh(actions)
    norm_actions = actions_low + ( norm_actions + 1) *  0.5 * (actions_high - actions_low)
    return norm_actions



