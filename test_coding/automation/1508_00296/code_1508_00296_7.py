import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a square-centered lattice with N atoms at vertices and B atoms at centers.
    
    The Hamiltonian models a system with two types of atoms (N and B) and spin degrees of freedom (up and down).
    It includes hopping between atoms, on-site potentials, and various interaction terms.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any] = {}, filling_factor: float = 0.5):
        self.lattice = 'square'   # Lattice symmetry ('square' for square-centered lattice)
        self.D = (2, 2)  # Flavors: (atom_type, spin)
        self.basis_order = {'0': 'atom_type', '1': 'spin'}
        # Order for each flavor:
        # atom_type: N, B
        # spin: up, down

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
        self.Delta = parameters.get('Delta', 0.0)  # On-site potential for N atoms

        # Interaction parameters
        self.U_N = parameters.get('U_N', 3.0)  # On-site interaction for N atoms
        self.U_B = parameters.get('U_B', 1.0)  # On-site interaction for B atoms
        self.V_B = parameters.get('V_B', 0.65)  # Additional interaction for B atoms
        self.V_BN = parameters.get('V_BN', 1.0)  # Interaction between N and B atoms

        return

    def generate_non_interacting(self) -> np.ndarray:
        """Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate lattice dispersion for square lattice
        # For nearest neighbor hopping on square lattice
        disp_N = -2 * (np.cos(self.k_space[:, 0] * self.a) + np.cos(self.k_space[:, 1] * self.a))
        disp_B = -2 * (np.cos(self.k_space[:, 0] * self.a) + np.cos(self.k_space[:, 1] * self.a))
        
        # For hopping between N and B atoms (assuming they are at distance a/2 in both x and y)
        disp_NB = 4 * np.cos(self.k_space[:, 0] * self.a / 2) * np.cos(self.k_space[:, 1] * self.a / 2)
        
        # N atoms with up spin (0,0)
        H_nonint[0, 0, 0, 0, :] = self.t_N * disp_N + self.Delta
        
        # N atoms with down spin (0,1)
        H_nonint[0, 1, 0, 1, :] = self.t_N * disp_N + self.Delta
        
        # B atoms with up spin (1,0)
        H_nonint[1, 0, 1, 0, :] = self.t_B * disp_B
        
        # B atoms with down spin (1,1)
        H_nonint[1, 1, 1, 1, :] = self.t_B * disp_B
        
        # Hopping between N and B atoms (up spin)
        H_nonint[0, 0, 1, 0, :] = self.t_BN * disp_NB
        H_nonint[1, 0, 0, 0, :] = self.t_BN * np.conjugate(disp_NB)
        
        # Hopping between N and B atoms (down spin)
        H_nonint[0, 1, 1, 1, :] = self.t_BN * disp_NB
        H_nonint[1, 1, 0, 1, :] = self.t_BN * np.conjugate(disp_NB)
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)

        # Calculate mean expectation values
        n_N_up = np.mean(exp_val[0, 0, 0, 0, :])   # <a^†_{k, ↑} a_{k, ↑}>
        n_N_down = np.mean(exp_val[0, 1, 0, 1, :]) # <a^†_{k, ↓} a_{k, ↓}>
        n_B_up = np.mean(exp_val[1, 0, 1, 0, :])   # <b^†_{k, ↑} b_{k, ↑}>
        n_B_down = np.mean(exp_val[1, 1, 1, 1, :]) # <b^†_{k, ↓} b_{k, ↓}>

        # N atoms with up spin (0,0)
        # U_N term: interaction with opposite spin
        # V_BN term: interaction with B atoms of both spins
        H_int[0, 0, 0, 0, :] = (self.U_N / self.N_k) * n_N_down + \
                             (self.V_BN / self.N_k) * (n_B_up + n_B_down)
        
        # N atoms with down spin (0,1)
        H_int[0, 1, 0, 1, :] = (self.U_N / self.N_k) * n_N_up + \
                             (self.V_BN / self.N_k) * (n_B_up + n_B_down)
        
        # B atoms with up spin (1,0)
        # U_B term: interaction with opposite spin
        # 2V_B term: interaction with all B atoms
        # V_BN term: interaction with N atoms of both spins
        H_int[1, 0, 1, 0, :] = (self.U_B / self.N_k) * n_B_down + \
                             (2 * self.V_B / self.N_k) * (n_B_up + n_B_down) + \
                             (self.V_BN / self.N_k) * (n_N_up + n_N_down)
        
        # B atoms with down spin (1,1)
        H_int[1, 1, 1, 1, :] = (self.U_B / self.N_k) * n_B_up + \
                             (2 * self.V_B / self.N_k) * (n_B_up + n_B_down) + \
                             (self.V_BN / self.N_k) * (n_N_up + n_N_down)
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, return the flattened Hamiltonian. Default is True.

        Returns:
            np.ndarray: The total Hamiltonian with appropriate shape.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
