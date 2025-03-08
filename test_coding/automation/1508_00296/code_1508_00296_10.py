import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a square-centered lattice with N atoms at vertices and B atoms at centers.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'   # Lattice symmetry
        self.D = (2, 2)  # (atom_type, spin)
        self.basis_order = {'0': 'atom_type', '1': 'spin'}
        # Order for each flavor:
        # 0: atom_type. Order: N (at vertices), B (at center)
        # 1: spin. Order: up, down

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
        self.U_N = parameters.get('U_N', 3.0)  # Interaction strength for N atoms
        self.U_B = parameters.get('U_B', 3.0)  # Interaction strength for B atoms
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
        
        # Compute the dispersion relations for hopping terms
        # For a square lattice, nearest-neighbor hopping contribution
        kx, ky = self.k_space[:, 0], self.k_space[:, 1]
        f_N = 2 * (np.cos(kx * self.a) + np.cos(ky * self.a))  # N-N hopping
        f_B = 2 * (np.cos(kx * self.a) + np.cos(ky * self.a))  # B-B hopping
        
        # For N-B hopping (from vertex to center)
        g_NB = np.exp(-1j * (kx * self.a/2 + ky * self.a/2))  # N-B hopping
        
        # N-N hopping terms (spin up and down)
        H_nonint[0, 0, 0, 0, :] = self.t_N * f_N + self.Delta  # N atom, spin up
        H_nonint[0, 1, 0, 1, :] = self.t_N * f_N + self.Delta  # N atom, spin down
        
        # B-B hopping terms (spin up and down)
        H_nonint[1, 0, 1, 0, :] = self.t_B * f_B  # B atom, spin up
        H_nonint[1, 1, 1, 1, :] = self.t_B * f_B  # B atom, spin down
        
        # N-B hopping terms (both spins)
        H_nonint[0, 0, 1, 0, :] = self.t_BN * g_NB  # N to B, spin up
        H_nonint[0, 1, 1, 1, :] = self.t_BN * g_NB  # N to B, spin down
        
        # B-N hopping terms (complex conjugate)
        H_nonint[1, 0, 0, 0, :] = self.t_BN * np.conjugate(g_NB)  # B to N, spin up
        H_nonint[1, 1, 0, 1, :] = self.t_BN * np.conjugate(g_NB)  # B to N, spin down
        
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
        
        # Calculate mean occupation numbers for each atom type and spin
        mean_N_up = np.mean(exp_val[0, 0, 0, 0, :])  # Mean occupation of N atoms, spin up
        mean_N_down = np.mean(exp_val[0, 1, 0, 1, :])  # Mean occupation of N atoms, spin down
        mean_B_up = np.mean(exp_val[1, 0, 1, 0, :])  # Mean occupation of B atoms, spin up
        mean_B_down = np.mean(exp_val[1, 1, 1, 1, :])  # Mean occupation of B atoms, spin down
        
        # U_B terms: Interaction between different spins on B atoms
        H_int[1, 0, 1, 0, :] += self.U_B * mean_B_down  # B atom, spin up affected by spin down
        H_int[1, 1, 1, 1, :] += self.U_B * mean_B_up    # B atom, spin down affected by spin up
        
        # U_N terms: Interaction between different spins on N atoms
        H_int[0, 0, 0, 0, :] += self.U_N * mean_N_down  # N atom, spin up affected by spin down
        H_int[0, 1, 0, 1, :] += self.U_N * mean_N_up    # N atom, spin down affected by spin up
        
        # 2V_B terms: Interaction between all spins on B atoms
        total_B = mean_B_up + mean_B_down
        H_int[1, 0, 1, 0, :] += 2 * self.V_B * total_B  # B atom, spin up affected by all B
        H_int[1, 1, 1, 1, :] += 2 * self.V_B * total_B  # B atom, spin down affected by all B
        
        # V_BN terms: Interaction between B and N atoms
        total_N = mean_N_up + mean_N_down
        
        # B affecting N
        H_int[0, 0, 0, 0, :] += self.V_BN * total_B  # N atom, spin up affected by all B
        H_int[0, 1, 0, 1, :] += self.V_BN * total_B  # N atom, spin down affected by all B
        
        # N affecting B
        H_int[1, 0, 1, 0, :] += self.V_BN * total_N  # B atom, spin up affected by all N
        H_int[1, 1, 1, 1, :] += self.V_BN * total_N  # B atom, spin down affected by all N
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns a flattened Hamiltonian. Default is True.

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
