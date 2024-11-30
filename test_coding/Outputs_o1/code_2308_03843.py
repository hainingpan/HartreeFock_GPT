# https://chatgpt.com/share/674856d0-cf34-800e-8188-cea355e99ab5
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system on a triangular lattice.

    Args:
      N_shell (int): Number of shells in k-space for generating k-points.
      parameters (dict): Dictionary containing model parameters 't', 'mu', 'N_b', and 'Vq'.
      filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=10, parameters: dict={'t':1.0, 'mu':0.0, 'N_b':2}, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Lattice symmetry ('square' or 'triangular')
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.t = parameters.get('t', 1.0)  # Hopping parameter
        self.mu = parameters.get('mu', 0.0)  # Chemical potential
        self.N_b = parameters.get('N_b', 2)  # Number of basis functions (orbitals)
        self.D = (2, self.N_b)  # Flavors: spin and orbital
        self.basis_order = {'0': 'spin: up, down', '1': f'orbital: 0 to {self.N_b - 1}'}

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Assume temperature is zero
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Interaction potential V_{\alpha\beta}(q)
        self.Vq = parameters.get('Vq', None)  # Interaction potential function

        # If Vq is not provided, define a default Vq
        if self.Vq is None:
            def Vq_default(alpha, beta, q_vec):
                # For simplicity, assume a constant interaction
                return 1.0
            self.Vq = Vq_default

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D + D + (N_k,)).
        """
        D_spin, D_orbital = self.D
        N_k = self.N_k
        H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)  # Shape (2, N_b, 2, N_b, N_k)

        # Compute the dispersion ε(k) for the triangular lattice
        a1 = np.array([1, 0])
        a2 = np.array([-0.5, np.sqrt(3)/2])
        a3 = np.array([-0.5, -np.sqrt(3)/2])

        for k_idx in range(N_k):
            k_vec = self.k_space[k_idx]  # k = (kx, ky)

            # Calculate ε(k) = -2t [cos(k·a1) + cos(k·a2) + cos(k·a3)]
            epsilon_k = -2 * self.t * (
                np.cos(np.dot(k_vec, a1)) +
                np.cos(np.dot(k_vec, a2)) +
                np.cos(np.dot(k_vec, a3))
            )

            for sigma in range(D_spin):
                for alpha in range(D_orbital):
                    for beta in range(D_orbital):
                        if alpha == beta:
                            # Diagonal elements: ε(k) - μ
                            H_nonint[sigma, alpha, sigma, beta, k_idx] = epsilon_k - self.mu
                        else:
                            # Off-diagonal elements: zero (no hopping between different orbitals)
                            H_nonint[sigma, alpha, sigma, beta, k_idx] = 0.0  # No inter-orbital hopping

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened * D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D + D + (N_k,)).
        """
        D_spin, D_orbital = self.D
        N_k = self.N_k

        # Expand exp_val to shape (D_spin, D_orbital, D_orbital, N_k)
        exp_val = self.expand(exp_val)  # Shape (2, N_b, N_b, N_k)

        H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)  # Shape (2, N_b, 2, N_b, N_k)

        # Compute Hartree and Fock contributions
        for sigma in range(D_spin):
            for alpha in range(D_orbital):
                for beta in range(D_orbital):
                    # Hartree term (diagonal in orbital indices)
                    if alpha == beta:
                        h_H = 0.0
                        for gamma in range(D_orbital):
                            for sigma_prime in range(D_spin):
                                rho_gamma_gamma = exp_val[sigma_prime, gamma, gamma, :]  # Shape (N_k,)
                                sum_rho = np.sum(rho_gamma_gamma) / N_k  # Average over k'

                                V_beta_gamma_0 = self.Vq(beta, gamma, np.array([0, 0]))  # q=0

                                h_H += V_beta_gamma_0 * sum_rho

                        # Hartree term is the same for all k
                        H_int[sigma, alpha, sigma, beta, :] += h_H  # Hartree contribution

                    # Fock term
                    h_F = np.zeros(N_k, dtype=np.float32)  # Shape (N_k,)

                    for k_idx in range(N_k):
                        k_vec = self.k_space[k_idx]

                        h_F_k = 0.0

                        for k_prime_idx in range(N_k):
                            k_prime_vec = self.k_space[k_prime_idx]
                            q_vec = k_vec - k_prime_vec

                            V_alpha_beta_q = self.Vq(alpha, beta, q_vec)

                            rho_alpha_beta = exp_val[sigma, alpha, beta, k_prime_idx]

                            h_F_k += V_alpha_beta_q * rho_alpha_beta

                        h_F[k_idx] = - h_F_k / N_k  # Negative sign and division by N_k

                    H_int[sigma, alpha, sigma, beta, :] += h_F  # Fock contribution

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened * D_flattened, N_k).
            return_flat (bool, optional): If True, returns the Hamiltonian flattened. Defaults to True.

        Returns:
            np.ndarray: The total Hamiltonian with shape (D_flattened, D_flattened, N_k) if return_flat is True.
                        Otherwise, returns shape (D + D + (N_k,)).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total  # Shape: D + D + (N_k,)

    def flatten(self, ham):
        """
        Flattens the Hamiltonian from shape (D + D + (N_k,)) to (D_flattened, D_flattened, N_k).
        """
        D_flattened = np.prod(self.D)
        return ham.reshape((D_flattened, D_flattened, self.N_k))

    def expand(self, exp_val):
        """
        Expands the exp_val from shape (D_flattened, D_flattened, N_k) to (D + D + (N_k,)).
        """
        return exp_val.reshape(self.D + self.D + (self.N_k,))
