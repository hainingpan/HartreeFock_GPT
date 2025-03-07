import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a three-orbital (p_x, p_y, d) model on a square lattice.
    
    This model describes a system with three orbitals per site and includes both
    non-interacting hopping terms and interacting terms that depend on mean-field
    expectation values. The interacting terms capture Coulomb interactions and
    give rise to potential nematic ordering between p_x and p_y orbitals.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={
        't_pd': 1.0,
        't_pp': 0.5,
        'U_p': 3.0,
        'U_d': 7.0,
        'V_pp': 1.0,
        'V_pd': 1.0,
        'Delta': 3.0,
        'T': 0,
        'a': 1.0
    }, filling_factor: float=0.5):
        self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular').
        self.D = (2, 3) # Number of flavors identified: (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: orbital p_x, p_y, d

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0) # temperature, default to 0
        self.a = parameters.get('a', 1.0) # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0) # p-d hopping parameter
        self.t_pp = parameters.get('t_pp', 0.5) # p-p hopping parameter
        self.U_p = parameters.get('U_p', 3.0) # Coulomb repulsion on p orbitals
        self.U_d = parameters.get('U_d', 7.0) # Coulomb repulsion on d orbitals
        self.V_pp = parameters.get('V_pp', 1.0) # Inter-site Coulomb repulsion between p orbitals
        self.V_pd = parameters.get('V_pd', 1.0) # Inter-site Coulomb repulsion between p and d orbitals
        self.Delta = parameters.get('Delta', 3.0) # Energy difference between p and d orbitals

        # Derived effective interaction parameters
        self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        self.V_pp_tilde = 8 * self.V_pp - self.U_p
        self.U_d_tilde = self.U_d - 4 * self.V_pd

        return

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        This includes all hopping terms between orbitals.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate hopping terms for each k point
        for k_idx, k in enumerate(self.k_space):
            # Calculate gamma terms
            gamma_1_x = -2 * self.t_pd * np.cos(k[0] / 2)  # p_x to d hopping
            gamma_1_y = -2 * self.t_pd * np.cos(k[1] / 2)  # p_y to d hopping
            gamma_2 = -4 * self.t_pp * np.cos(k[0] / 2) * np.cos(k[1] / 2)  # p_x to p_y hopping
            
            # Fill the Hamiltonian for both spin up and spin down
            for s in range(2):
                # p_x to p_y hopping
                H_nonint[s, 0, s, 1, k_idx] = gamma_2
                H_nonint[s, 1, s, 0, k_idx] = gamma_2
                
                # p_x to d hopping
                H_nonint[s, 0, s, 2, k_idx] = gamma_1_x
                H_nonint[s, 2, s, 0, k_idx] = gamma_1_x
                
                # p_y to d hopping
                H_nonint[s, 1, s, 2, k_idx] = gamma_1_y
                H_nonint[s, 2, s, 1, k_idx] = gamma_1_y
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.
        This includes all terms that depend on occupation numbers.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        # Unflatten the expectation value tensor
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate mean occupation for each orbital and spin
        n_px_up = np.mean(exp_val[0, 0, 0, 0, :])  # <p_x,up|p_x,up>
        n_px_down = np.mean(exp_val[1, 0, 1, 0, :])  # <p_x,down|p_x,down>
        n_py_up = np.mean(exp_val[0, 1, 0, 1, :])  # <p_y,up|p_y,up>
        n_py_down = np.mean(exp_val[1, 1, 1, 1, :])  # <p_y,down|p_y,down>
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :])  # <d,up|d,up>
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :])  # <d,down|d,down>
        
        # Calculate total densities
        n_p = n_px_up + n_px_down + n_py_up + n_py_down  # Total p orbital density
        n = n_p + n_d_up + n_d_down  # Total density
        
        # Calculate nematic order parameter
        eta = (n_px_up + n_px_down) - (n_py_up + n_py_down)
        
        # Calculate chemical potential (derived from the formula provided)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Calculate the diagonal terms in the Hamiltonian
        xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu
        xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu
        xi_d = self.U_d_tilde * (n - n_p) / 2 - mu
        
        # Fill the diagonal terms for each spin and orbital
        for s in range(2):  # Loop over spin
            # All of these are interacting terms depending on occupation numbers
            H_int[s, 0, s, 0, :] = xi_x  # p_x diagonal term
            H_int[s, 1, s, 1, :] = xi_y  # p_y diagonal term
            H_int[s, 2, s, 2, :] = xi_d  # d diagonal term
        
        # Note: f(n^p, eta) is a constant energy shift for all k-points
        # It doesn't contribute to the matrix elements directly but affects the total energy
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian.

        Returns:
            np.ndarray: The total Hamiltonian with shape (D, D, N_k) or flattened.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
