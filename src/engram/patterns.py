import numpy as np


def generate_sparse_pattern(
    n_neurons: int,
    sparsity: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a sparse binary pattern with values in {0, 1}.

    Parameters
    ----------
    n_neurons : int
        Total number of neurons in the network.
    sparsity : float
        Fraction of neurons active in the pattern (e.g. 0.1 for 10%).
    rng : np.random.Generator | None
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Binary vector of shape (n_neurons,) with approximately
        sparsity * n_neurons active neurons.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0 < sparsity < 1):
        raise ValueError("sparsity must be between 0 and 1")

    pattern = np.zeros(n_neurons, dtype=int)
    n_active = int(round(n_neurons * sparsity))
    active_indices = rng.choice(n_neurons, size=n_active, replace=False)
    pattern[active_indices] = 1
    return pattern


def corrupt_pattern(
    pattern: np.ndarray,
    corruption_level: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Corrupt a binary pattern by flipping a fraction of bits.

    Parameters
    ----------
    pattern : np.ndarray
        Binary input pattern.
    corruption_level : float
        Fraction of bits to flip.
    rng : np.random.Generator | None
        Random number generator.

    Returns
    -------
    np.ndarray
        Corrupted binary pattern.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0 <= corruption_level <= 1):
        raise ValueError("corruption_level must be between 0 and 1")

    corrupted = pattern.copy()
    n_flip = int(round(len(pattern) * corruption_level))

    if n_flip == 0:
        return corrupted

    flip_indices = rng.choice(len(pattern), size=n_flip, replace=False)
    corrupted[flip_indices] = 1 - corrupted[flip_indices]
    return corrupted


def generate_cs_input(
    pattern: np.ndarray,
    cue_fraction: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a CS-like external input vector that partially overlaps
    the stored engram.

    This keeps a subset of active engram neurons 'cued' and leaves the
    rest at zero.

    Parameters
    ----------
    pattern : np.ndarray
        Stored binary engram pattern.
    cue_fraction : float
        Fraction of active engram neurons included in the CS cue.
    rng : np.random.Generator | None
        Random number generator.

    Returns
    -------
    np.ndarray
        Binary cue vector with partial overlap with the pattern.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0 <= cue_fraction <= 1):
        raise ValueError("cue_fraction must be between 0 and 1")

    active_indices = np.where(pattern == 1)[0]
    n_cued = int(round(len(active_indices) * cue_fraction))

    cs_input = np.zeros_like(pattern)

    if n_cued == 0:
        return cs_input

    cued_indices = rng.choice(active_indices, size=n_cued, replace=False)
    cs_input[cued_indices] = 1
    return cs_input

def generate_noisy_cs_input(
    pattern: np.ndarray,
    cue_fraction: float,
    background_fraction: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a CS-like input with:
    - partial overlap onto true engram neurons
    - background activation of non-engram neurons

    Parameters
    ----------
    pattern : np.ndarray
        Stored binary engram pattern.
    cue_fraction : float
        Fraction of active engram neurons included in the cue.
    background_fraction : float
        Fraction of inactive/non-engram neurons also activated.
    rng : np.random.Generator | None
        Random number generator.

    Returns
    -------
    np.ndarray
        Binary cue vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0 <= cue_fraction <= 1):
        raise ValueError("cue_fraction must be between 0 and 1")
    if not (0 <= background_fraction <= 1):
        raise ValueError("background_fraction must be between 0 and 1")

    cs_input = np.zeros_like(pattern)

    engram_idx = np.where(pattern == 1)[0]
    non_engram_idx = np.where(pattern == 0)[0]

    n_cued = int(round(len(engram_idx) * cue_fraction))
    n_background = int(round(len(non_engram_idx) * background_fraction))

    if n_cued > 0:
        cued_idx = rng.choice(engram_idx, size=n_cued, replace=False)
        cs_input[cued_idx] = 1

    if n_background > 0:
        bg_idx = rng.choice(non_engram_idx, size=n_background, replace=False)
        cs_input[bg_idx] = 1

    return cs_input