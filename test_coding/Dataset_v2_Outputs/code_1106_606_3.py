"""Generate using GPT4-o1, https://chatgpt.com/share/67201579-da74-8011-9b41-8c72c6bb0440"""
from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for the given system.

    Args:
        N_shell (int): Number of shells for the k-space lattice.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int = 5, parameters: dict[str, Any] = None, filling_factor: float = 0.5):
        self.lattice = 'square'  # Lattice symmetry
        self.D = (2, 3)  # Tuple of flavors: (spin, orbital)
        self.basis_order = {
            '0': 'spin (up, down)',     # Flavor type 0
            '1': 'orbital (p_x, p_y, d)'  # Flavor type 1
        }

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0.0  # Temperature
        self.k_space = generate_k_space(self.lattice, N_shell)
        self.N_k = self.k_space.shape[0]

        # Default model parameters
        if parameters is None:
            parameters = {}
        self.t_pd = parameters.get('t_pd', 1.0)      # Hopping between p and d orbitals
        self.t_pp = parameters.get('t_pp', 0.5)      # Hopping between p_x and p_y orbitals
        self.Delta = parameters.get('Delta', 0.0)    # On-site energy difference
        self.U_p = parameters.get('U_p', 0.0)        # On-site interaction on p orbitals
        self.U_d = parameters.get('U_d', 0.0)        # On-site interaction on d orbital
        self.V_pp = parameters.get('V_pp', 0.0)      # Interaction between p orbitals
        self.V_pd = parameters.get('V_pd', 0.0)      # Interaction between p and d orbitals
        self.mu = parameters.get('mu', 0.0)          # Chemical potential

        # Effective interaction parameters
        self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        self.V_pp_tilde = 8 * self.V_pp - self.U_p
        self.U_d_tilde = self.U_d - 4 * self.V_pd

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (2, 3, 2, 3, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=np.float32)
        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        cos_kx_over_2 = np.cos(kx / 2)
        cos_ky_over_2 = np.cos(ky / 2)

        gamma1_kx = -2 * self.t_pd * cos_kx_over_2  # γ₁(kₓ)
        gamma1_ky = -2 * self.t_pd * cos_ky_over_2  # γ₁(k_y)
        gamma2_k = -4 * self.t_pp * cos_kx_over_2 * cos_ky_over_2  # γ₂(k)

        # Non-interacting off-diagonal terms
        # H[s1, o1, s2, o2, k]
        for s in range(2):  # Spin index: 0 (up), 1 (down)
            # γ₂(k) between p_x and p_y
            H_nonint[s, 0, s, 1, :] = gamma2_k
            H_nonint[s, 1, s, 0, :] = gamma2_k  # Symmetric term

            # γ₁(kₓ) between p_x and d
            H_nonint[s, 0, s, 2, :] = gamma1_kx
            H_nonint[s, 2, s, 0, :] = gamma1_kx  # Symmetric term

            # γ₁(k_y) between p_y and d
            H_nonint[s, 1, s, 2, :] = gamma1_ky
            H_nonint[s, 2, s, 1, :] = gamma1_ky  # Symmetric term

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (2, 3, 2, 3, N_k).
        """
        exp_val = self.expand(exp_val)  # Reshape exp_val to (2, 3, 2, 3, N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=np.float32)

        # Compute n^p and η
        # Occupations for p_x and p_y orbitals
        n_px = np.mean(
            exp_val[0, 0, 0, 0, :] + exp_val[1, 0, 1, 0, :]
        )
        n_py = np.mean(
            exp_val[0, 1, 0, 1, :] + exp_val[1, 1, 1, 1, :]
        )
        n_p = n_px + n_py  # Total p orbital occupation

        # Occupation for d orbital
        n_d = np.mean(
            exp_val[0, 2, 0, 2, :] + exp_val[1, 2, 1, 2, :]
        )

        n = n_p + n_d  # Total hole density

        # Nematic order parameter η
        eta = n_px - n_py

        # Compute ξ_x, ξ_y, ξ_d
        xi_x = self.Delta + (self.U_p_tilde * n_p) / 4 - (self.V_pp_tilde * eta) / 4 - self.mu
        xi_y = self.Delta + (self.U_p_tilde * n_p) / 4 + (self.V_pp_tilde * eta) / 4 - self.mu
        xi_d = (self.U_d_tilde * (n - n_p)) / 2 - self.mu

        # Interacting diagonal terms
        for s in range(2):  # Spin index: 0 (up), 1 (down)
            # ξ_x on p_x orbital
            H_int[s, 0, s, 0, :] = xi_x
            # ξ_y on p_y orbital
            H_int[s, 1, s, 1, :] = xi_y
            # ξ_d on d orbital
            H_int[s, 2, s, 2, :] = xi_d

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool, optional): If True, returns a flattened Hamiltonian. Defaults to True.

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total  # Shape: (2, 3, 2, 3, N_k)

    def expand(self, exp_val: np.ndarray) -> np.ndarray:
        """Reshapes exp_val to (2, 3, 2, 3, N_k)."""
        return exp_val.reshape(self.D + self.D + (self.N_k,))

    def flatten(self, H_total: np.ndarray) -> np.ndarray:
        """Flattens H_total to shape (D_flattened, D_flattened, N_k)."""
        D_flat = np.prod(self.D)
        return H_total.reshape((D_flat, D_flat, self.N_k))
