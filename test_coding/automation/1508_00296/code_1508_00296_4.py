import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a square-centered lattice with N atoms at vertices
    and B atoms at centers. Includes hopping and interaction terms.
    
    Args:
        N_shell (int): Number of shells in the Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'   # Square-centered lattice with N at vertices and B at centers
        self.D = (2, 2)  # (atom_type, spin)
        self.basis_order = {'0': 'atom_type', '1': 'spin'}
        # Order:
        # atom_type: N (0), B (1)
        # spin: up (0), down (1)
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # Temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t_N = parameters.get('t_N', 1.0)  # Hopping parameter for N atoms
        self.t_B = parameters.get('t_B', 1.0)  # Hopping parameter for B atoms
        self.t_BN = parameters.get('t_BN', 0.5)  # Hopping parameter between N and B atoms
        self.Delta = parameters.get('Delta', 0.0)  # Energy shift for N atoms
        
        # Interaction parameters
        self.U_N = parameters.get('U_N', 3.0)  # Interaction strength for N atoms
        self.U_B = parameters.get('U_B', 0.65)  # Interaction strength for B atoms
        self.V_B = parameters.get('V_B', 0.65)  # Additional interaction for B atoms
        self.V_BN = parameters.get('V_BN', 1.0)  # Interaction between N and B atoms
        
        return

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate the dispersion relation for the square-centered lattice
        # For the sum over n and n', we use 4*cos(kx/2)*cos(ky/2)
        dispersion = 4 * np.cos(self.k_space[:, 0] * 0.5) * np.cos(self.k_space[:, 1] * 0.5)
        
        # N atom terms (for both spin up and down)
        H_nonint[0, 0, 0, 0, :] = self.t_N * dispersion + self.Delta  # N atom, spin up
        H_nonint[0, 1, 0, 1, :] = self.t_N * dispersion + self.Delta  # N atom, spin down
        
        # B atom terms (for both spin up and down)
        H_nonint[1, 0, 1, 0, :] = self.t_B * dispersion  # B atom, spin up
        H_nonint[1, 1, 1, 1, :] = self.t_B * dispersion  # B atom, spin down
        
        # Hopping terms between N and B atoms
        # N to B hopping (for both spin up and down)
        H_nonint[0, 0, 1, 0, :] = self.t_BN * dispersion  # N to B, spin up
        H_nonint[0, 1, 1, 1, :] = self.t_BN * dispersion  # N to B, spin down
        
        # B to N hopping (for both spin up and down)
        H_nonint[1, 0, 0, 0, :] = self.t_BN * dispersion  # B to N, spin up
        H_nonint[1, 1, 0, 1, :] = self.t_BN * dispersion  # B to N, spin down
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
        
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate the mean densities for each atom type and spin
        n_N_up = np.mean(exp_val[0, 0, 0, 0, :])  # <a†_{k,↑} a_{k,↑}>
        n_N_down = np.mean(exp_val[0, 1, 0, 1, :])  # <a†_{k,↓} a_{k,↓}>
        n_B_up = np.mean(exp_val[1, 0, 1, 0, :])  # <b†_{k,↑} b_{k,↑}>
        n_B_down = np.mean(exp_val[1, 1, 1, 1, :])  # <b†_{k,↓} b_{k,↓}>
        
        # U_B term (interaction between different spins for B atoms)
        H_int[1, 0, 1, 0, :] = self.U_B * n_B_down / self.N_k  # B atom, spin up affected by spin down
        H_int[1, 1, 1, 1, :] = self.U_B * n_B_up / self.N_k    # B atom, spin down affected by spin up
        
        # U_N term (interaction between different spins for N atoms)
        H_int[0, 0, 0, 0, :] = self.U_N * n_N_down / self.N_k  # N atom, spin up affected by spin down
        H_int[0, 1, 0, 1, :] = self.U_N * n_N_up / self.N_k    # N atom, spin down affected by spin up
        
        # 2V_B term (interaction for all spin combinations for B atoms)
        B_total = 2 * self.V_B * (n_B_up + n_B_down) / self.N_k
        H_int[1, 0, 1, 0, :] += B_total  # B atom, spin up
        H_int[1, 1, 1, 1, :] += B_total  # B atom, spin down
        
        # V_BN term (B affecting N)
        BN_from_B = self.V_BN * (n_B_up + n_B_down) / self.N_k
        H_int[0, 0, 0, 0, :] += BN_from_B  # N atom, spin up affected by B
        H_int[0, 1, 0, 1, :] += BN_from_B  # N atom, spin down affected by B
        
        # V_BN term (N affecting B)
        BN_from_N = self.V_BN * (n_N_up + n_N_down) / self.N_k
        H_int[1, 0, 1, 0, :] += BN_from_N  # B atom, spin up affected by N
        H_int[1, 1, 1, 1, :] += BN_from_N  # B atom, spin down affected by N
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian. Default is True.
        
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
