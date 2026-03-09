import numpy as np


def pattern_overlap(state: np.ndarray, pattern: np.ndarray) -> float:
    """
    Compute overlap between a retrieved state and a stored binary pattern.

    Defined as:
        (# of active pattern neurons that are active in the state)
        / (# of active neurons in the pattern)

    Parameters
    ----------
    state : np.ndarray
        Binary network state.
    pattern : np.ndarray
        Binary stored pattern.

    Returns
    -------
    float
        Overlap in [0, 1].
    """
    if state.shape != pattern.shape:
        raise ValueError("state and pattern must have the same shape")

    n_pattern_active = pattern.sum()
    if n_pattern_active == 0:
        raise ValueError("pattern has no active neurons")

    return np.sum((state == 1) & (pattern == 1)) / n_pattern_active


def retrieval_success(
    state: np.ndarray,
    pattern: np.ndarray,
    threshold: float = 0.9,
) -> int:
    """
    Return 1 if overlap exceeds threshold, else 0.
    """
    ov = pattern_overlap(state, pattern)
    return int(ov >= threshold)