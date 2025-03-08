import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a square-centered lattice with N (Nitrogen) atoms at vertices
    and B (Boron) atoms at centers, considering different spin states.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'  # Square-centered lattice
        self.D = (2, 2)  # (atom_type, spin)
        self.basis_order = {'0': 'atom_type', '1': 'spin'}
        # 0: atom_type: Nitrogen (N), Boron (B)
        # 1: spin: up, down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t_N = parameters.get('t_N', 1.0)  # Hopping for N atoms
        self.t_B = parameters.get('t_B', 1.0)  # Hopping for B atoms
        self.t_BN = parameters.get('t_BN', 0.5)  # Hopping between N and B atoms
        self.Delta = parameters.get('Delta', 0.0)  # On-site energy difference for N atoms
        
        # Interaction parameters
        self.U_N = parameters.get('U_N', 3.0)  # Coulomb repulsion for N atoms
        self.U_B = parameters.get('U_B', 3.0)  # Coulomb repulsion for B atoms
        self.V_B = parameters.get('V_B', 0.65)  # Additional interaction for B atoms
        self.V_BN = parameters.get('V_BN', 1.0)  # Interaction between B and N atoms
        
        return

    def generate_non_interacting(self) -> np.ndarray:
        """Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (*self.D, *self.D, self.N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate dispersion terms for the square lattice
        # For square lattice, nearest neighbors are at (±1,0) and (0,±1)
        dispersion = 2 * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))
        
        # Calculate the phase factor for the N-B hopping
        # For square-centered lattice, B is at (0.5, 0.5) relative to N
        phase_NB = np.exp(-1j * (self.k_space[:, 0] + self.k_space[:, 1]) / 2)
        phase_BN = np.exp(1j * (self.k_space[:, 0] + self.k_space[:, 1]) / 2)
        
        # N atom hopping (for both spins)
        for spin in range(2):
            # Dispersion term
            H_nonint[0, spin, 0, spin, :] = -self.t_N * dispersion + self.Delta
            
            # B atom hopping (for both spins)
            H_nonint[1, spin, 1, spin, :] = -self.t_B * dispersion
            
            # N-B hopping terms
            H_nonint[0, spin, 1, spin, :] = -self.t_BN * phase_NB
            H_nonint[1, spin, 0, spin, :] = -self.t_BN * phase_BN
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value with shape (np.prod(self.D), np.prod(self.D), self.N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (*self.D, *self.D, self.N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate the averages of density operators for each atom type and spin
        n_N_up = np.mean(exp_val[0, 0, 0, 0, :])  # <a^†_{k,up} a_{k,up}>
        n_N_down = np.mean(exp_val[0, 1, 0, 1, :])  # <a^†_{k,down} a_{k,down}>
        n_B_up = np.mean(exp_val[1, 0, 1, 0, :])  # <b^†_{k,up} b_{k,up}>
        n_B_down = np.mean(exp_val[1, 1, 1, 1, :])  # <b^†_{k,down} b_{k,down}>
        
        # U_N term: Coulomb repulsion for N atoms (opposite spins)
        H_int[0, 0, 0, 0, :] = self.U_N / self.N_k * n_N_down  # up-up affected by down
        H_int[0, 1, 0, 1, :] = self.U_N / self.N_k * n_N_up  # down-down affected by up
        
        # U_B term: Coulomb repulsion for B atoms (opposite spins)
        H_int[1, 0, 1, 0, :] = self.U_B / self.N_k * n_B_down  # up-up affected by down
        H_int[1, 1, 1, 1, :] = self.U_B / self.N_k * n_B_up  # down-down affected by up
        
        # V_B term: Additional interaction for B atoms (all spins)
        H_int[1, 0, 1, 0, :] += 2 * self.V_B / self.N_k * (n_B_up + n_B_down)  # B up-up
        H_int[1, 1, 1, 1, :] += 2 * self.V_B / self.N_k * (n_B_up + n_B_down)  # B down-down
        
        # V_BN term: Interaction between B and N atoms
        # B affecting N
        H_int[0, 0, 0, 0, :] += self.V_BN / self.N_k * (n_B_up + n_B_down)  # N up-up
        H_int[0, 1, 0, 1, :] += self.V_BN / self.N_k * (n_B_up + n_B_down)  # N down-down
        
        # N affecting B
        H_int[1, 0, 1, 0, :] += self.V_BN / self.N_k * (n_N_up + n_N_down)  # B up-up
        H_int[1, 1, 1, 1, :] += self.V_BN / self.N_k * (n_N_up + n_N_down)  # B down-down
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            return_flat (bool): Whether to return a flattened array, default is True.
            
        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
