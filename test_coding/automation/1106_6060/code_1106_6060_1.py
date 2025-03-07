import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a three-orbital system with p_x, p_y, and d orbitals.
    The class implements the mean-field Hamiltonian given by the equation:
    H_MF = sum_k,s C^dag_ks H_ks C_ks + f(n^p, eta),
    where C^dag_ks = (p^dag_xks, p^dag_yks, d^dag_ks) and H_ks is a 3x3 matrix.
    
    Args:
        N_shell (int): Number shell in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, representing the ratio of occupied states to total states.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_pd':1.0, 't_pp':0.5, 'Delta':1.0, 'U_p':1.0, 'V_pp':0.5, 'V_pd':0.5, 'U_d':1.0, 'T':0, 'a':1.0}, filling_factor: float=0.5):
        self.lattice = 'square'  # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
        self.D = (2, 3)  # Number of flavors identified: (spin, orbital).
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: p_x, p_y, d

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
        self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping
        self.Delta = parameters.get('Delta', 1.0)  # Energy offset
        self.U_p = parameters.get('U_p', 1.0)  # On-site interaction for p orbitals
        self.V_pp = parameters.get('V_pp', 0.5)  # Nearest-neighbor interaction between p orbitals
        self.V_pd = parameters.get('V_pd', 0.5)  # Nearest-neighbor interaction between p and d orbitals
        self.U_d = parameters.get('U_d', 1.0)  # On-site interaction for d orbitals

        # Effective interaction parameters
        self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        self.V_pp_tilde = 8 * self.V_pp - self.U_p
        self.U_d_tilde = self.U_d - 4 * self.V_pd
        
        return

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Non-interacting part: gamma1 and gamma2 terms
        for s in range(2):  # Loop over spin
            for k in range(self.N_k):  # Loop over k points
                k_x, k_y = self.k_space[k]
                
                # Calculate the gamma terms
                gamma1_x = -2 * self.t_pd * np.cos(k_x / 2)
                gamma1_y = -2 * self.t_pd * np.cos(k_y / 2)
                gamma2 = -4 * self.t_pp * np.cos(k_x / 2) * np.cos(k_y / 2)
                
                # p_x-p_y coupling (gamma2 term)
                H_nonint[s, 0, s, 1, k] = gamma2
                H_nonint[s, 1, s, 0, k] = gamma2
                
                # p_x-d coupling (gamma1_x term)
                H_nonint[s, 0, s, 2, k] = gamma1_x
                H_nonint[s, 2, s, 0, k] = gamma1_x
                
                # p_y-d coupling (gamma1_y term)
                H_nonint[s, 1, s, 2, k] = gamma1_y
                H_nonint[s, 2, s, 1, k] = gamma1_y
                
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
        
        # Calculate the mean values for each orbital and spin
        n_px_up = np.mean(exp_val[0, 0, 0, 0, :].real)  # <p^dag_x,up p_x,up>
        n_py_up = np.mean(exp_val[0, 1, 0, 1, :].real)  # <p^dag_y,up p_y,up>
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :].real)   # <d^dag_up d_up>
        n_px_down = np.mean(exp_val[1, 0, 1, 0, :].real)  # <p^dag_x,down p_x,down>
        n_py_down = np.mean(exp_val[1, 1, 1, 1, :].real)  # <p^dag_y,down p_y,down>
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :].real)  # <d^dag_down d_down>
        
        # Calculate n^p, eta, and n
        n_p = n_px_up + n_py_up + n_px_down + n_py_down
        eta = (n_px_up + n_px_down) - (n_py_up + n_py_down)
        n = n_p + n_d_up + n_d_down
        
        # Calculate chemical potential
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Calculate xi_x, xi_y, xi_d (the diagonal elements)
        xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu
        xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu
        xi_d = self.U_d_tilde * (n - n_p) / 2 - mu
        
        # Fill the diagonal terms for both spin up and spin down
        for s in range(2):  # Loop over spin
            for k in range(self.N_k):  # Loop over k points
                # p_x diagonal term
                H_int[s, 0, s, 0, k] = xi_x
                # p_y diagonal term
                H_int[s, 1, s, 1, k] = xi_y
                # d diagonal term
                H_int[s, 2, s, 2, k] = xi_d
        
        # Note: The function f(n^p, eta) is a constant energy shift that contributes to the total energy,
        # but doesn't affect the eigenvectors. It's not added to the Hamiltonian matrix here.
        # f_term = -self.U_p_tilde * (n_p**2) / 8 + self.V_pp_tilde * (eta**2) / 8 - self.U_d_tilde * ((n - n_p)**2) / 4
                
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns flattened Hamiltonian.

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
