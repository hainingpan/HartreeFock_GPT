import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a two-orbital model (N and B atoms) with spin.
    
    The Hamiltonian includes:
    - Hopping terms for N atoms, B atoms, and between N and B atoms
    - On-site energy for N atoms
    - Various interaction terms (U_N, U_B, V_B, V_BN)
    
    Args:
        N_shell (int): Number of shells in the k-space grid.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor for the system.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'  # Square-centered lattice
        self.D = (2, 2)  # (atom_type, spin)
        self.basis_order = {'0': 'atom_type', '1': 'spin'}
        # atom_type: 0 = N (atoms at vertices), 1 = B (atoms at center)
        # spin: 0 = up, 1 = down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t_N = parameters.get('t_N', 1.0)  # Hopping parameter for N atoms
        self.t_B = parameters.get('t_B', 1.0)  # Hopping parameter for B atoms
        self.t_BN = parameters.get('t_BN', 0.5)  # Hopping parameter between N and B atoms
        self.Delta = parameters.get('Delta', 0.0)  # On-site energy for N atoms
        
        # Interaction parameters
        self.U_N = parameters.get('U_N', 3.0)  # On-site interaction for N atoms
        self.U_B = parameters.get('U_B', 3.0)  # On-site interaction for B atoms
        self.V_B = parameters.get('V_B', 0.65)  # Interaction for B atoms regardless of spin
        self.V_BN = parameters.get('V_BN', 1.0)  # Interaction between B and N atoms
        
        return
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (*D, *D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        kx, ky = self.k_space[:, 0], self.k_space[:, 1]
        
        # Dispersion for N atoms: -2 * t_N * (cos(kx) + cos(ky))
        epsilon_N = -2 * self.t_N * (np.cos(kx) + np.cos(ky))
        
        # Dispersion for B atoms: -2 * t_B * (cos(kx) + cos(ky))
        epsilon_B = -2 * self.t_B * (np.cos(kx) + np.cos(ky))
        
        # Hopping between N and B atoms: -4 * t_BN * cos(kx/2) * cos(ky/2)
        # For a square-centered lattice, B atoms are at (±0.5, ±0.5) relative to N atoms
        phi = -4 * self.t_BN * np.cos(kx/2) * np.cos(ky/2)
        
        # Kinetic terms for N atoms (both spins) + on-site energy Delta
        H_nonint[0, 0, 0, 0, :] = epsilon_N + self.Delta  # N, spin up
        H_nonint[0, 1, 0, 1, :] = epsilon_N + self.Delta  # N, spin down
        
        # Kinetic terms for B atoms (both spins)
        H_nonint[1, 0, 1, 0, :] = epsilon_B  # B, spin up
        H_nonint[1, 1, 1, 1, :] = epsilon_B  # B, spin down
        
        # Hopping terms between N and B atoms (both spins)
        H_nonint[0, 0, 1, 0, :] = phi  # N to B, spin up
        H_nonint[0, 1, 1, 1, :] = phi  # N to B, spin down
        H_nonint[1, 0, 0, 0, :] = np.conj(phi)  # B to N, spin up
        H_nonint[1, 1, 0, 1, :] = np.conj(phi)  # B to N, spin down
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian using expectation values.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (*D, *D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate mean field expectation values
        # For N atoms
        n_N_up = np.mean(exp_val[0, 0, 0, 0, :])  # <a^dagger_{k, up} a_{k, up}>
        n_N_down = np.mean(exp_val[0, 1, 0, 1, :])  # <a^dagger_{k, down} a_{k, down}>
        
        # For B atoms
        n_B_up = np.mean(exp_val[1, 0, 1, 0, :])  # <b^dagger_{k, up} b_{k, up}>
        n_B_down = np.mean(exp_val[1, 1, 1, 1, :])  # <b^dagger_{k, down} b_{k, down}>
        
        # Interaction term U_B for B atoms (different spins)
        H_int[1, 0, 1, 0, :] += self.U_B / self.N_k * n_B_down  # B, spin up interacting with B, spin down
        H_int[1, 1, 1, 1, :] += self.U_B / self.N_k * n_B_up  # B, spin down interacting with B, spin up
        
        # Interaction term U_N for N atoms (different spins)
        H_int[0, 0, 0, 0, :] += self.U_N / self.N_k * n_N_down  # N, spin up interacting with N, spin down
        H_int[0, 1, 0, 1, :] += self.U_N / self.N_k * n_N_up  # N, spin down interacting with N, spin up
        
        # Interaction term V_B for B atoms (all spins)
        H_int[1, 0, 1, 0, :] += 2 * self.V_B / self.N_k * (n_B_up + n_B_down)  # B, spin up
        H_int[1, 1, 1, 1, :] += 2 * self.V_B / self.N_k * (n_B_up + n_B_down)  # B, spin down
        
        # Interaction term V_BN for B-N interaction
        # N atoms interacting with B atom density
        H_int[0, 0, 0, 0, :] += self.V_BN / self.N_k * (n_B_up + n_B_down)  # N, spin up
        H_int[0, 1, 0, 1, :] += self.V_BN / self.N_k * (n_B_up + n_B_down)  # N, spin down
        
        # B atoms interacting with N atom density
        H_int[1, 0, 1, 0, :] += self.V_BN / self.N_k * (n_N_up + n_N_down)  # B, spin up
        H_int[1, 1, 1, 1, :] += self.V_BN / self.N_k * (n_N_up + n_N_down)  # B, spin down
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian by combining non-interacting and interacting parts.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            return_flat (bool): Whether to return the flattened Hamiltonian.
            
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
