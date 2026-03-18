import numpy as np


class HopfieldNetwork:
    """
    Sparse Hopfield network for storing and retrieving sparse binary patterns.
    """

    def __init__(self, n_neurons: int, sparsity: float):
        self.n = n_neurons
        self.a = sparsity
        self.W = np.zeros((n_neurons, n_neurons))

    def reset_weights(self) -> None:
        """
        Reset all synaptic weights to zero.
        """
        self.W = np.zeros((self.n, self.n))

    def _hebbian_increment(self, pattern: np.ndarray) -> np.ndarray:
        """
        Compute the centered Hebbian outer product contribution for a pattern.

        This returns the unnormalized weight increment matrix.
        """
        if pattern.shape != (self.n,):
            raise ValueError("pattern must have shape (n_neurons,)")

        centered = pattern - self.a
        return np.outer(centered, centered)

    def store_patterns(self, patterns: list[np.ndarray]) -> None:
        """
        Store sparse binary patterns using a centered Hebbian rule.

        This is the 'parallel' / all-at-once storage: the weight matrix
        is recomputed from scratch given the full list of patterns.
        """
        W = np.zeros((self.n, self.n))

        for p in patterns:
            W += self._hebbian_increment(p)

        W /= (self.n * self.a * (1 - self.a))
        np.fill_diagonal(W, 0.0)
        self.W = W

    def store_pattern_sequential(self, pattern: np.ndarray) -> None:
        """
        Store one new pattern by incrementally updating the weights.

        This uses the same centered Hebbian rule and normalization as
        'store_patterns', but updates the existing weight matrix rather
        than recomputing it from scratch.
        """
        delta_W = self._hebbian_increment(pattern) / (self.n * self.a * (1 - self.a))
        self.W += delta_W
        np.fill_diagonal(self.W, 0.0)


    def local_field(
        self,
        state: np.ndarray,
        external_input: np.ndarray | None = None,
        beta: float = 1.0,
    ) -> np.ndarray:
        """
        Compute the local field h for all neurons.
        """
        if external_input is None:
            external_input = np.zeros(self.n)

        return self.W @ state + beta * external_input

    def energy(
        self,
        state: np.ndarray,
        external_input: np.ndarray | None = None,
        beta: float = 1.0,
    ) -> float:
        """
        Hopfield-like energy for binary state with external field.
        """
        if external_input is None:
            external_input = np.zeros(self.n)

        recurrent_term = -0.5 * state @ self.W @ state
        external_term = -beta * external_input @ state
        return float(recurrent_term + external_term)

    def update_synchronous(
        self,
        state: np.ndarray,
        external_input: np.ndarray | None = None,
        beta: float = 1.0,
        theta: float = 0.0,
    ) -> np.ndarray:
        """
        One synchronous update of all neurons.
        """
        h = self.local_field(state, external_input=external_input, beta=beta)
        return (h > theta).astype(int)

    def run_synchronous(
        self,
        initial_state: np.ndarray,
        external_input: np.ndarray | None = None,
        beta: float = 1.0,
        theta: float = 0.0,
        max_steps: int = 50,
    ) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
        """
        Run synchronous dynamics until convergence.
        """
        state = initial_state.copy()
        trajectory = [state.copy()]
        energies = [self.energy(state, external_input=external_input, beta=beta)]

        for _ in range(max_steps):
            new_state = self.update_synchronous(
                state,
                external_input=external_input,
                beta=beta,
                theta=theta,
            )

            trajectory.append(new_state.copy())
            energies.append(self.energy(new_state, external_input=external_input, beta=beta))

            if np.array_equal(new_state, state):
                break

            state = new_state

        return state, trajectory, energies

    def run(
        self,
        initial_state: np.ndarray,
        external_input: np.ndarray | None = None,
        beta: float = 1.0,
        theta: float = 0.0,
        max_steps: int = 50,
        asynchronous: bool = False,
        n_sweeps: int = 20,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
        """
        Convenience wrapper to run either synchronous or asynchronous dynamics.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial binary state.
        external_input : np.ndarray | None
            Optional external field.
        beta : float
            Strength of the external field.
        theta : float
            Firing threshold.
        max_steps : int
            Maximum number of synchronous updates (if asynchronous=False).
        asynchronous : bool
            If True, run asynchronous updates; otherwise synchronous.
        n_sweeps : int
            Number of sweeps for asynchronous updates.
        rng : np.random.Generator | None
            Random generator for asynchronous update order.
        """
        if asynchronous:
            return self.run_asynchronous(
                initial_state=initial_state,
                external_input=external_input,
                beta=beta,
                theta=theta,
                n_sweeps=n_sweeps,
                rng=rng,
            )
        return self.run_synchronous(
            initial_state=initial_state,
            external_input=external_input,
            beta=beta,
            theta=theta,
            max_steps=max_steps,
        )

    def run_asynchronous(
        self,
        initial_state: np.ndarray,
        external_input: np.ndarray | None = None,
        beta: float = 1.0,
        theta: float = 0.0,
        n_sweeps: int = 20,
        rng: np.random.Generator | None = None,
        record_every_sweep: bool = True,
    ) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
        """
        Run asynchronous updates.
        
        One sweep = n single-neuron updates in random order.
        """
        if rng is None:
            rng = np.random.default_rng()

        state = initial_state.copy()
        trajectory = [state.copy()]
        energies = [self.energy(state, external_input=external_input, beta=beta)]

        for _ in range(n_sweeps):
            prev_state = state.copy()
            update_order = rng.permutation(self.n)

            for i in update_order:
                h_i = self.W[i] @ state + (
                    0.0 if external_input is None else beta * external_input[i]
                )
                state[i] = int(h_i > theta)

            if record_every_sweep:
                trajectory.append(state.copy())
                energies.append(self.energy(state, external_input=external_input, beta=beta))

            if np.array_equal(state, prev_state):
                break

        return state, trajectory, energies