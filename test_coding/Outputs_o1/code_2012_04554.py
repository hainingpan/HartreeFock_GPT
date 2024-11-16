# https://chatgpt.com/share/6738c165-a45c-8011-86c8-e1ff26d98af5
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary containing model parameters 't', 'U', and 'a'.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int = 5, parameters: dict = None, filling_factor: float = 0.5):
        if parameters is None:
            parameters = {'t': 1.0, 'U': 1.0, 'a': 1.0}
        self.lattice = 'triangular'
        self.D = (2,)  # Number of flavors (spin)
        self.basis_order = {
            '0': 'spin',
            'Order': ['up (0)', 'down (1)']
        }
        # This is the basis order that the Hamiltonian will follow.

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0.0  # Temperature
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        self.t = parameters.get('t', 1.0)  # Hopping amplitude
        self.U = parameters.get('U', 1.0)  # Interaction strength

        # Hopping vectors for triangular lattice (nearest neighbors)
        self.n_vectors = np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [-0.5, np.sqrt(3)/2],
            [-1, 0],
            [-0.5, -np.sqrt(3)/2],
            [0.5, -np.sqrt(3)/2]
        ]) * self.a  # Scaled by lattice constant

        return

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        N_k = self.N_k
        H_nonint = np.zeros((self.D[0], self.D[0], N_k), dtype=np.float32)

        # Compute E(k) for each k
        E_k = np.zeros(N_k, dtype=np.float32)
        for idx, k in enumerate(self.k_space):
            # Compute the dispersion relation E_s(k)
            E_k_complex = self.t * np.sum(np.exp(-1j * np.dot(self.n_vectors, k)))
            E_k[idx] = -np.real(E_k_complex)  # Negative sign from the Hamiltonian definition

        # Assign E(k) to diagonal elements for both spins
        # H[s, s, k] corresponds to -E_s(k) c_{k,s}^\dagger c_{k,s}
        H_nonint[0, 0, :] = E_k  # Spin up
        H_nonint[1, 1, :] = E_k  # Spin down

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = self.expand(exp_val)  # Reshape to (D, D, N_k)
        N_k = self.N_k
        H_int = np.zeros((self.D[0], self.D[0], N_k), dtype=np.float32)

        # Calculate the mean densities for each spin
        n_s = np.mean(np.real(np.diagonal(exp_val, axis1=0, axis2=1)), axis=1)  # n_s = ⟨c_{k,s}^† c_{k,s}⟩
        n_total = np.sum(n_s)  # Total density

        # Hartree term: Adds U * n_total to diagonal elements H[s', s', k]
        # Accounts for interactions of particles with the average total density
        for s_prime in range(self.D[0]):
            H_int[s_prime, s_prime, :] += self.U * n_total  # Hartree interaction

        # Fock term: Adds -U * mean_exp_val to elements H[s', s, k]
        # Accounts for exchange interactions between particles
        for s_prime in range(self.D[0]):
            for s in range(self.D[0]):
                mean_exp_val = np.mean(np.real(exp_val[s, s_prime, :]))  # ⟨c_{k,s}^† c_{k,s'}⟩
                H_int[s_prime, s, :] += -self.U * mean_exp_val  # Fock interaction

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool, optional): Whether to flatten the Hamiltonian. Defaults to True.

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total  # Shape (D, D, N_k)

    def flatten(self, ham):
        """
        Flattens the Hamiltonian from shape (D, D, N_k) to (D_flattened, D_flattened, N_k).
        """
        D_flattened = np.prod(self.D)
        return ham.reshape((D_flattened, D_flattened, self.N_k))

    def expand(self, exp_val):
        """
        Expands the expectation value from shape (D_flattened, D_flattened, N_k) to (D, D, N_k).
        """
        return exp_val.reshape(self.D + self.D + (self.N_k,))
