import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a three-orbital system with px, py, and d orbitals.
    
    The Hamiltonian includes hopping terms between orbitals, interaction effects, 
    and a nematic order parameter. The model is based on a square lattice.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters like t_pd, t_pp, Delta, etc.
        filling_factor (float): Filling factor for the system.
    """
    def __init__(self, N_shell: int = 3, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'  # Lattice symmetry
        self.D = (2, 3)  # (|spin|, |orbital|)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: px, py, d orbitals
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # Temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
        self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping
        self.Delta = parameters.get('Delta', 1.0)  # Energy offset
        self.U_p = parameters.get('U_p', 1.0)  # Coulomb interaction on p orbitals
        self.U_d = parameters.get('U_d', 1.0)  # Coulomb interaction on d orbitals
        self.V_pp = parameters.get('V_pp', 0.5)  # p-p interaction
        self.V_pd = parameters.get('V_pd', 0.5)  # p-d interaction
        
        # Derived interaction parameters
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
        
        for k_idx in range(self.N_k):
            kx, ky = self.k_space[k_idx]
            
            # Calculate hopping terms
            gamma_1_kx = -2 * self.t_pd * np.cos(kx/2)  # px-d hopping
            gamma_1_ky = -2 * self.t_pd * np.cos(ky/2)  # py-d hopping
            gamma_2_k = -4 * self.t_pp * np.cos(kx/2) * np.cos(ky/2)  # px-py hopping
            
            # For each spin
            for s in range(2):
                # px-py hopping
                H_nonint[s, 0, s, 1, k_idx] = gamma_2_k
                H_nonint[s, 1, s, 0, k_idx] = gamma_2_k
                
                # px-d hopping
                H_nonint[s, 0, s, 2, k_idx] = gamma_1_kx
                H_nonint[s, 2, s, 0, k_idx] = gamma_1_kx
                
                # py-d hopping
                H_nonint[s, 1, s, 2, k_idx] = gamma_1_ky
                H_nonint[s, 2, s, 1, k_idx] = gamma_1_ky
        
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
        
        # Calculate expectation values
        # n_p: total density of holes on oxygen sites (px, py)
        n_x_up = np.mean(exp_val[0, 0, 0, 0, :])    # <p†_x↑ p_x↑>
        n_x_down = np.mean(exp_val[1, 0, 1, 0, :])  # <p†_x↓ p_x↓>
        n_y_up = np.mean(exp_val[0, 1, 0, 1, :])    # <p†_y↑ p_y↑>
        n_y_down = np.mean(exp_val[1, 1, 1, 1, :])  # <p†_y↓ p_y↓>
        
        n_p = (n_x_up + n_x_down) + (n_y_up + n_y_down)
        
        # n: total density of holes (px, py, d)
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :])    # <d†↑ d↑>
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :])  # <d†↓ d↓>
        n = n_p + (n_d_up + n_d_down)
        
        # η: nematic order parameter (difference between px and py occupations)
        eta = (n_x_up + n_x_down) - (n_y_up + n_y_down)
        
        # Calculate chemical potential (to factor out constant terms)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Calculate the diagonal elements with interaction effects
        xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu
        xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu
        xi_d = self.U_d_tilde * (n - n_p) / 2 - mu
        
        # Update Hamiltonian with interaction terms
        for s in range(2):
            for k_idx in range(self.N_k):
                # Diagonal terms with interaction effects
                H_int[s, 0, s, 0, k_idx] = xi_x  # px orbital
                H_int[s, 1, s, 1, k_idx] = xi_y  # py orbital
                H_int[s, 2, s, 2, k_idx] = xi_d  # d orbital
        
        # Store the constant energy shift for later use in energy calculations
        self.f_np_eta = -self.U_p_tilde * (n_p**2) / 8 + self.V_pp_tilde * (eta**2) / 8 - self.U_d_tilde * ((n - n_p)**2) / 4
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian. Default is True.
            
        Returns:
            np.ndarray: The total Hamiltonian with shape (D, D, N_k) or (D_flattened, D_flattened, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
