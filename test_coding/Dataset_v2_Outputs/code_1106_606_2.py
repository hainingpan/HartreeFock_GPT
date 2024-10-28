"""Generate using GPT4-4o, https://chatgpt.com/share/67201533-d024-8011-80dc-029d5f23a9f2"""
from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a multi-orbital system on a square lattice.
    
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=10, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'  # Lattice type
        self.D = (2, 3)  # Spin (up, down) and Orbital (p_x, p_y, d)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0.0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)
        self.t_pp = parameters.get('t_pp', 0.5)
        self.Delta = parameters.get('Delta', 1.0)
        self.mu = parameters.get('mu', 0.0)
        
        # Effective interaction parameters
        self.Up_tilde = parameters.get('Up_tilde', 1.0)
        self.Vpp_tilde = parameters.get('Vpp_tilde', 0.5)
        self.Ud_tilde = parameters.get('Ud_tilde', 1.0)
    
    def generate_non_interacting(self) -> np.ndarray:
        """Generates the non-interacting part of the Hamiltonian."""
        H_nonint = np.zeros((2, 3, 2, 3, self.N_k), dtype=np.float32)
        kx, ky = self.k_space[:, 0], self.k_space[:, 1]

        # Diagonal terms: on-site energies
        H_nonint[:, 0, :, 0, :] = self.Delta - self.mu  # p_x
        H_nonint[:, 1, :, 1, :] = self.Delta - self.mu  # p_y
        H_nonint[:, 2, :, 2, :] = -self.mu             # d orbital
        
        # Off-diagonal hopping terms
        H_nonint[:, 0, :, 1, :] = -4 * self.t_pp * np.cos(kx / 2) * np.cos(ky / 2)  # p_x <-> p_y
        H_nonint[:, 0, :, 2, :] = -2 * self.t_pd * np.cos(kx / 2)  # p_x <-> d
        H_nonint[:, 1, :, 2, :] = -2 * self.t_pd * np.cos(ky / 2)  # p_y <-> d

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generates the interacting part of the Hamiltonian based on exp_val."""
        exp_val = self.expand(exp_val)
        H_int = np.zeros((2, 3, 2, 3, self.N_k), dtype=np.float32)

        # Calculate mean-field parameters
        n_p = np.sum(exp_val[:, :2, :, :2, :]) / 4
        eta = (np.sum(exp_val[:, 0, :, 0, :]) - np.sum(exp_val[:, 1, :, 1, :])) / 4
        
        # Apply interactions
        H_int[:, 0, :, 0, :] += self.Up_tilde * n_p / 4 - self.Vpp_tilde * eta / 4
        H_int[:, 1, :, 1, :] += self.Up_tilde * n_p / 4 + self.Vpp_tilde * eta / 4
        H_int[:, 2, :, 2, :] += self.Ud_tilde * (1 - n_p) / 2

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        return self.flatten(H_total) if flatten else H_total

    def flatten(self, ham):
        return ham.reshape((np.prod(self.D), np.prod(self.D), self.N_k))

    def expand(self, exp_val):
        return exp_val.reshape((2, 3, 2, 3, self.N_k))
