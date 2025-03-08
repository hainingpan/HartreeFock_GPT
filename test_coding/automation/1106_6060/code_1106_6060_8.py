import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for the three-band Emery model with p-d hybridization.
    
    The model includes p_x, p_y orbitals (oxygen) and d orbital (copper) with spin.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'
        self.D = (2, 3)  # (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Order for each flavor:
        # spin: up, down
        # orbital: p_x, p_y, d

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        self.Delta = parameters.get('Delta', 1.0)  # Energy offset for p orbitals
        self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
        self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping
        self.U_p = parameters.get('U_p', 3.0)  # On-site interaction for p orbitals
        self.U_d = parameters.get('U_d', 8.0)  # On-site interaction for d orbitals
        self.V_pp = parameters.get('V_pp', 1.0)  # Nearest-neighbor interaction between p orbitals
        self.V_pd = parameters.get('V_pd', 1.0)  # Nearest-neighbor interaction between p and d orbitals
        
        # Effective interaction parameters
        self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        self.V_pp_tilde = 8 * self.V_pp - self.U_p
        self.U_d_tilde = self.U_d - 4 * self.V_pd
        
        return

    def compute_order_parameters(self, exp_val):
        """
        Compute the order parameters from the expectation values.
        
        Args:
            exp_val: Expectation values, shape (2, 3, 2, 3, N_k).
            
        Returns:
            tuple: (n_p, n, eta) order parameters.
        """
        # Calculate densities for each orbital and spin
        n_px_up = np.mean(exp_val[0, 0, 0, 0, :])
        n_px_down = np.mean(exp_val[1, 0, 1, 0, :])
        n_py_up = np.mean(exp_val[0, 1, 0, 1, :])
        n_py_down = np.mean(exp_val[1, 1, 1, 1, :])
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :])
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :])
        
        # Total p and d densities
        n_px = n_px_up + n_px_down
        n_py = n_py_up + n_py_down
        n_p = n_px + n_py
        n_d = n_d_up + n_d_down
        n = n_p + n_d
        
        # Nematic order parameter
        eta = n_px - n_py
        
        return n_p, n, eta

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (2, 3, 2, 3, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate hopping terms gamma_1 and gamma_2
        k_x = self.k_space[:, 0]
        k_y = self.k_space[:, 1]
        gamma_1_x = -2 * self.t_pd * np.cos(k_x/2)
        gamma_1_y = -2 * self.t_pd * np.cos(k_y/2)
        gamma_2 = -4 * self.t_pp * np.cos(k_x/2) * np.cos(k_y/2)
        
        # Assign hopping terms to Hamiltonian for both spin channels
        for s in range(2):
            # p_x - p_y hopping
            H_nonint[s, 0, s, 1, :] = gamma_2  # p_x to p_y
            H_nonint[s, 1, s, 0, :] = gamma_2  # p_y to p_x
            
            # p_x - d hopping
            H_nonint[s, 0, s, 2, :] = gamma_1_x  # p_x to d
            H_nonint[s, 2, s, 0, :] = gamma_1_x  # d to p_x
            
            # p_y - d hopping
            H_nonint[s, 1, s, 2, :] = gamma_1_y  # p_y to d
            H_nonint[s, 2, s, 1, :] = gamma_1_y  # d to p_y
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (2, 3, 2, 3, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute order parameters
        n_p, n, eta = self.compute_order_parameters(exp_val)
        
        # Calculate chemical potential (not an independent variable)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Calculate orbital energies
        xi_x = self.Delta + self.U_p_tilde * n_p/4 - self.V_pp_tilde * eta/4 - mu
        xi_y = self.Delta + self.U_p_tilde * n_p/4 + self.V_pp_tilde * eta/4 - mu
        xi_d = self.U_d_tilde * (n - n_p)/2 - mu
        
        # Assign diagonal terms to Hamiltonian for both spin channels
        for s in range(2):
            H_int[s, 0, s, 0, :] = xi_x  # p_x orbital energy
            H_int[s, 1, s, 1, :] = xi_y  # p_y orbital energy
            H_int[s, 2, s, 2, :] = xi_d  # d orbital energy
            
        # Add the f(n_p, eta) term
        # This is a constant energy contribution, but we distribute it evenly across k-points
        f_term = -self.U_p_tilde * (n_p**2)/8 + self.V_pp_tilde * (eta**2)/8 - self.U_d_tilde * ((n - n_p)**2)/4
        
        # Since f_term is a global energy shift, we can add it to any diagonal element
        # Here we distribute it evenly to all diagonal elements
        constant_shift = f_term / (2 * 3 * self.N_k)  # Divide by number of diagonal elements
        for s in range(2):
            for o in range(3):
                H_int[s, o, s, o, :] += constant_shift
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return a flattened array. Default is True.
            
        Returns:
            np.ndarray: The total Hamiltonian, either flattened or not.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
