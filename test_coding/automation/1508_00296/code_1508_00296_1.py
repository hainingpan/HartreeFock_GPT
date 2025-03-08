import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a two-band model with N atoms at vertices and B atoms at the center
    of a square lattice. The model includes hopping terms and various interaction terms.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'
        self.D = (4,)  # 4 flavors: N atom spin up/down, B atom spin up/down
        self.basis_order = {'0': 'atom-spin'}
        # Order for each flavor:
        # 0: N atom with spin up
        # 1: N atom with spin down
        # 2: B atom with spin up
        # 3: B atom with spin down

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
        
        # On-site energy parameter
        self.Delta = parameters.get('Delta', 0.0)  # On-site energy for N atoms
        
        # Interaction parameters
        self.U_N = parameters.get('U_N', 3.0)  # Interaction strength for N atoms with opposite spin
        self.U_B = parameters.get('U_B', 3.0)  # Interaction strength for B atoms with opposite spin
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
        
        # Compute the dispersion relation for a square lattice
        dispersion = -2 * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))
        
        # Hopping term for N atoms and on-site energy
        H_nonint[0, 0, :] = self.t_N * dispersion + self.Delta  # N up to N up
        H_nonint[1, 1, :] = self.t_N * dispersion + self.Delta  # N down to N down
        
        # Hopping term for B atoms
        H_nonint[2, 2, :] = self.t_B * dispersion  # B up to B up
        H_nonint[3, 3, :] = self.t_B * dispersion  # B down to B down
        
        # Hopping between N and B atoms
        # Phase factor for hopping from N to B and vice versa
        phase_NB = np.exp(-1j * (self.k_space[:, 0] / 2 + self.k_space[:, 1] / 2))
        phase_BN = np.exp(1j * (self.k_space[:, 0] / 2 + self.k_space[:, 1] / 2))
        
        H_nonint[0, 2, :] = self.t_BN * phase_BN  # N up to B up
        H_nonint[1, 3, :] = self.t_BN * phase_BN  # N down to B down
        H_nonint[2, 0, :] = self.t_BN * phase_NB  # B up to N up
        H_nonint[3, 1, :] = self.t_BN * phase_NB  # B down to N down
        
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

        # Calculate the mean densities for each flavor
        n_N_up = np.mean(exp_val[0, 0, :])  # <a^†_{k,↑} a_{k,↑}>
        n_N_down = np.mean(exp_val[1, 1, :])  # <a^†_{k,↓} a_{k,↓}>
        n_B_up = np.mean(exp_val[2, 2, :])  # <b^†_{k,↑} b_{k,↑}>
        n_B_down = np.mean(exp_val[3, 3, :])  # <b^†_{k,↓} b_{k,↓}>

        # U_N term: Interaction between N atoms with opposite spins
        H_int[0, 0, :] = self.U_N * n_N_down / self.N_k  # N up interacts with N down
        H_int[1, 1, :] = self.U_N * n_N_up / self.N_k    # N down interacts with N up
        
        # U_B term: Interaction between B atoms with opposite spins
        H_int[2, 2, :] = self.U_B * n_B_down / self.N_k  # B up interacts with B down
        H_int[3, 3, :] = self.U_B * n_B_up / self.N_k    # B down interacts with B up
        
        # V_B term: Additional interaction for B atoms with any spin
        H_int[2, 2, :] += 2 * self.V_B * (n_B_up + n_B_down) / self.N_k  # B up interacts with all B
        H_int[3, 3, :] += 2 * self.V_B * (n_B_up + n_B_down) / self.N_k  # B down interacts with all B
        
        # V_BN term: Interaction between N and B atoms
        H_int[0, 0, :] += self.V_BN * (n_B_up + n_B_down) / self.N_k  # N up interacts with all B
        H_int[1, 1, :] += self.V_BN * (n_B_up + n_B_down) / self.N_k  # N down interacts with all B
        H_int[2, 2, :] += self.V_BN * (n_N_up + n_N_down) / self.N_k  # B up interacts with all N
        H_int[3, 3, :] += self.V_BN * (n_N_up + n_N_down) / self.N_k  # B down interacts with all N
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian. Defaults to True.

        Returns:
            np.ndarray: The total Hamiltonian. If return_flat is True, shape is 
                        (np.prod(D), np.prod(D), N_k), otherwise shape is (D, D, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
