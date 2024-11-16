# https://chatgpt.com/share/6738c29b-12fc-8011-b082-ea45aa8f7fa2
import numpy as np
from typing import Any, Dict, List

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with spin flavors on a triangular lattice.

    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary containing model parameters.
            - 'a': Lattice constant.
            - 't_s_n': Dictionary of hopping parameters t_s(n) for each flavor s.
            - 'U_n': List of interaction parameters U(n) for each hopping vector n.
            - 'n_vectors': List of hopping vectors n (as numpy arrays).
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int = 10, parameters: Dict[str, Any] = None, filling_factor: float = 0.5):
        if parameters is None:
            parameters = {}
        self.lattice = 'triangular'
        self.D = (2,)  # Number of flavors identified: spin up and spin down
        self.basis_order = {
            '0': 'spin',
            'Order': ['up', 'down']
        }
        # Basis Order:
        # 0: spin_up
        # 1: spin_down

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature
        self.a = parameters.get('a', 1.0)  # Lattice constant

        # Generate k-space
        self.k_space = self.generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        # Hopping vectors n (default to nearest neighbors)
        default_n_vectors = [
            np.array([1, 0]),
            np.array([0.5, np.sqrt(3)/2]),
            np.array([-0.5, np.sqrt(3)/2]),
            np.array([-1, 0]),
            np.array([-0.5, -np.sqrt(3)/2]),
            np.array([0.5, -np.sqrt(3)/2])
        ]
        self.n_vectors = parameters.get('n_vectors', default_n_vectors)

        # Hopping parameters t_s_n for each flavor s
        default_t_s_n = {
            0: [1.0 for _ in self.n_vectors],  # spin_up
            1: [1.0 for _ in self.n_vectors]   # spin_down
        }
        self.t_s_n = parameters.get('t_s_n', default_t_s_n)

        # Interaction parameters U_n for each hopping vector n
        default_U_n = [1.0 for _ in self.n_vectors]  # For simplicity, U(n) = 1.0
        self.U_n = parameters.get('U_n', default_U_n)

        # Precompute U(0) and U(k - q)
        self.U0 = self.compute_U0()      # U(0)
        self.U_kq = self.compute_U_kq()  # U(k - q)

    def generate_k_space(self, lattice: str, N_shell: int, a: float) -> np.ndarray:
        # Function to generate k-space points for the triangular lattice
        # For simplicity, we generate a grid in the first Brillouin zone
        num_points = 2 * N_shell + 1
        kx = np.linspace(-np.pi / a, np.pi / a, num_points)
        ky = np.linspace(-np.pi / a, np.pi / a, num_points)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_space = np.vstack([kx_grid.ravel(), ky_grid.ravel()]).T
        return k_space

    def compute_U0(self) -> float:
        # Compute U(0) from U_n
        U0 = np.sum(self.U_n)
        return U0

    def compute_U_kq(self) -> np.ndarray:
        # Compute U(k - q) for all k and q
        N_k = self.N_k
        U_kq = np.zeros((N_k, N_k), dtype=np.float32)
        for idx_k in range(N_k):
            k = self.k_space[idx_k]
            for idx_q in range(N_k):
                q = self.k_space[idx_q]
                k_minus_q = k - q
                U_kq_value = 0.0
                for U_n_value, n in zip(self.U_n, self.n_vectors):
                    phase = np.dot(k_minus_q, n)
                    U_kq_value += U_n_value * np.cos(phase)  # Assuming U_n are real
                U_kq[idx_k, idx_q] = U_kq_value
        return U_kq

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        N_k = self.N_k
        H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
        # Compute E_s(k) = sum over n of t_s(n) * cos(k . n)
        E_s_k = np.zeros(self.D + (N_k,), dtype=np.float32)
        for s in range(self.D[0]):
            for idx_k, k in enumerate(self.k_space):
                E_s_k_value = 0.0
                for t_s_n_value, n in zip(self.t_s_n[s], self.n_vectors):
                    phase = np.dot(k, n)
                    E_s_k_value += t_s_n_value * np.cos(phase)  # Assuming t_s_n are real
                E_s_k[s, idx_k] = E_s_k_value
            # H_nonint[s, s, :] = -E_s(k)
            H_nonint[s, s, :] = -E_s_k[s, :]  # Negative sign from the Hamiltonian
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = self.expand(exp_val)  # Shape: (D, D, N_k)
        N_k = self.N_k
        H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

        # Compute total density n_total
        n_total = np.sum(exp_val.diagonal(offset=0, axis1=0, axis2=1))  # Sum over s and k

        # Hartree term: H_int[s', s', k'] += U0 * n_total / N_k
        for s_prime in range(self.D[0]):
            H_int[s_prime, s_prime, :] += self.U0 * n_total / N_k

        # Fock term: H_int[s', s, q] += - (1 / N_k) * sum over k of U(k - q) * exp_val[s, s', k]
        for s_prime in range(self.D[0]):
            for s in range(self.D[0]):
                # Compute convolution over k
                convolution = np.dot(self.U_kq.T, exp_val[s, s_prime, :])
                H_int[s_prime, s, :] += - (1 / N_k) * convolution

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened * D_flattened, N_k).
            return_flat (bool): Whether to flatten the output Hamiltonian.

        Returns:
            np.ndarray: The total Hamiltonian with shape (D_flattened, D_flattened, N_k) if return_flat is True,
                        else with shape (D, D, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total  # Shape: D, D, N_k

    def flatten(self, ham: np.ndarray) -> np.ndarray:
        """
        Flattens the Hamiltonian from shape (D, D, N_k) to (D_flattened, D_flattened, N_k).

        Args:
            ham (np.ndarray): Hamiltonian with shape (D, D, N_k).

        Returns:
            np.ndarray: Flattened Hamiltonian.
        """
        return ham.reshape((np.prod(self.D), np.prod(self.D), self.N_k))

    def expand(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Expands the expectation value from shape (D_flattened, D_flattened, N_k) to (D, D, N_k).

        Args:
            exp_val (np.ndarray): Expectation value with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: Expanded expectation value.
        """
        return exp_val.reshape(self.D + self.D + (self.N_k,))
