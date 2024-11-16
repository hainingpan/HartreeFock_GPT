# https://chatgpt.com/share/6738bfef-25e8-8011-a372-1da8563660a0
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with spin degrees of freedom on a triangular lattice.

    Args:
        N_shell (int): Number of shells in k-space lattice (determines the size of k-space grid).
        parameters (dict): Dictionary containing model parameters 't_s_n' and 'U_n'.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=5, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'triangular'   # Lattice symmetry
        self.D = (2,)  # Number of flavors (spin up and spin down)
        self.basis_order = {'0': 'spin. Order: up, down'}
        # 0: spin up, 1: spin down

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0.0  # Temperature
        self.k_space = generate_k_space(self.lattice, N_shell)
        self.N_k = self.k_space.shape[0]

        # Model parameters with default values
        self.t_s_n = parameters.get('t_s_n', self.default_t_s_n())
        self.U_n = parameters.get('U_n', self.default_U_n())

        # Precompute E_s(k) and U(k)
        self.E_s_k = self.compute_E_s_k()  # Energy dispersion for each spin
        self.U_k = self.compute_U_k()      # Interaction potential in k-space

        # U(0) for Hartree term
        self.U_0 = self.U_n.get((0, 0), 0)

    def default_t_s_n(self):
        # Default hopping amplitudes t_s(n)
        t = 1.0  # Default hopping amplitude
        # For spin-independent hopping
        t_s_n = {'up': {}, 'down': {}}
        # Nearest neighbor vectors on triangular lattice
        n_vectors = self.get_nearest_neighbor_vectors()
        for s in ['up', 'down']:
            for n in n_vectors:
                t_s_n[s][n] = t
        return t_s_n

    def default_U_n(self):
        # Default interaction potential U(n)
        U0 = 1.0  # Default on-site interaction
        U_n = {}
        U_n[(0, 0)] = U0
        return U_n

    def get_nearest_neighbor_vectors(self):
        # Returns nearest neighbor vectors for triangular lattice
        n_vectors = [
            (1, 0),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (0, -1),
            (1, -1)
        ]
        return n_vectors

    def compute_E_s_k(self):
        # Compute E_s(k) = sum_n t_s(n) * exp(-i k . n)
        E_s_k = {}
        for s in ['up', 'down']:
            E_k = np.zeros(self.N_k, dtype=np.float64)
            for n, t in self.t_s_n[s].items():
                n_vector = np.array(n)
                phase = np.exp(-1j * np.dot(self.k_space, n_vector))
                E_k += t * phase.real  # Assuming t is real
            E_s_k[s] = E_k
        return E_s_k

    def compute_U_k(self):
        # Compute U(k) = sum_n U(n) * exp(-i k . n)
        U_k = np.zeros(self.N_k, dtype=np.float64)
        for n, U in self.U_n.items():
            n_vector = np.array(n)
            phase = np.exp(-1j * np.dot(self.k_space, n_vector))
            U_k += U * phase.real  # Assuming U is real
        return U_k

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((self.D[0], self.D[0], self.N_k), dtype=np.float64)
        # Kinetic energy for spin up
        H_nonint[0, 0, :] = self.E_s_k['up']  # H[up, up, k]
        # Kinetic energy for spin down
        H_nonint[1, 1, :] = self.E_s_k['down']  # H[down, down, k]
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = self.expand(exp_val)  # Shape (D, D, N_k)
        H_int = np.zeros((self.D[0], self.D[0], self.N_k), dtype=np.float64)

        # Hartree term: Adds to diagonal elements
        for s in range(self.D[0]):
            n_s = np.mean(exp_val[s, s, :])  # Average density for spin s
            # Hartree contribution
            H_int[0, 0, :] += self.U_0 * n_s  # Adds to H[up, up, k]
            H_int[1, 1, :] += self.U_0 * n_s  # Adds to H[down, down, k]

        # Fock term: Adds to off-diagonal elements
        # Compute convolution using FFT for efficiency
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                # Compute convolution over k' of U(k - k') and exp_val[s, s', k']
                exp_k = exp_val[s, s_prime, :]
                convolution = np.fft.ifft(
                    np.fft.fft(exp_k) * np.conj(np.fft.fft(self.U_k))
                ).real / self.N_k
                # Fock contribution
                H_int[s_prime, s, :] -= convolution  # Adds to H[s', s, k]

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to flatten the Hamiltonian. Defaults to True.

        Returns:
            np.ndarray: The total Hamiltonian with shape (D_flattened, D_flattened, N_k) if return_flat is True,
                        otherwise with shape (D, D, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total

    def flatten(self, ham):
        # Flatten the Hamiltonian from shape (D, D, N_k) to (D_flattened, D_flattened, N_k)
        D_flat = np.prod(self.D)
        return ham.reshape((D_flat, D_flat, self.N_k))

    def expand(self, exp_val):
        # Expand exp_val from shape (D_flattened, D_flattened, N_k) to (D, D, N_k)
        return exp_val.reshape(self.D + self.D + (self.N_k,))
