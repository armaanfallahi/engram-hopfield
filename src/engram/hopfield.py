import numpy as np


class HopfieldNetwork:
    """
    Sparse Hopfield network for storing and retrieving engrams.
    """

    def __init__(self, n_neurons: int, sparsity: float):
        self.n = n_neurons
        self.a = sparsity
        self.W = np.zeros((n_neurons, n_neurons))

    def store_patterns(self, patterns: list[np.ndarray]):
        """
        Store patterns using sparse Hebbian rule.
        """
        n_patterns = len(patterns)

        W = np.zeros((self.n, self.n))

        for p in patterns:
            centered = p - self.a
            W += np.outer(centered, centered)

        W /= (self.n * self.a * (1 - self.a))

        np.fill_diagonal(W, 0)

        self.W = W

    def update(self, state: np.ndarray, external_input=None, beta=1.0, theta=0.0):
        """
        Single synchronous update step.
        """
        if external_input is None:
            external_input = np.zeros(self.n)

        h = self.W @ state + beta * external_input

        new_state = (h > theta).astype(int)

        return new_state

    def run(
        self,
        initial_state: np.ndarray,
        external_input=None,
        beta=1.0,
        theta=0.0,
        max_steps=50,
    ):
        """
        Run network dynamics until convergence.
        """
        state = initial_state.copy()

        trajectory = [state.copy()]

        for _ in range(max_steps):

            new_state = self.update(state, external_input, beta, theta)

            trajectory.append(new_state.copy())

            if np.array_equal(new_state, state):
                break

            state = new_state

        return state, trajectory