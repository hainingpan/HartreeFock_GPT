import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a three-band Emery model.
    
    Args:
        N_shell (int): Number of shells in the Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'
        self.D = (2, 3)  # (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: px, py, d

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # Hopping between p and d orbitals
        self.t_pp = parameters.get('t_pp', 0.5)  # Hopping between p orbitals
        self.Delta = parameters.get('Delta', 2.0)  # Energy difference between p and d orbitals
        self.U_p = parameters.get('U_p', 4.0)  # On-site repulsion for p orbitals
        self.U_d = parameters.get('U_d', 8.0)  # On-site repulsion for d orbitals
        self.V_pp = parameters.get('V_pp', 1.0)  # Inter-site repulsion between p orbitals
        self.V_pd = parameters.get('V_pd', 1.5)  # Inter-site repulsion between p and d orbitals
        
        # Effective interaction parameters
        self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        self.V_pp_tilde = 8 * self.V_pp - self.U_p
        self.U_d_tilde = self.U_d - 4 * self.V_pd
        
        return

    def compute_order_parameters(self, exp_val: np.ndarray):
        """
        Compute order parameters from the expectation values.
        
        Args:
            exp_val: Expectation value array with shape (D1, D2, ..., DN, N_k).
            
        Returns:
            Tuple containing (n_p, eta, n, mu).
        """
        # Compute the densities based on expectation values
        n_px_up = np.mean(exp_val[0, 0, :])  # <p^†_{x↑}p_{x↑}>
        n_py_up = np.mean(exp_val[0, 1, :])  # <p^†_{y↑}p_{y↑}>
        n_d_up = np.mean(exp_val[0, 2, :])   # <d^†_{↑}d_{↑}>
        n_px_down = np.mean(exp_val[1, 0, :])  # <p^†_{x↓}p_{x↓}>
        n_py_down = np.mean(exp_val[1, 1, :])  # <p^†_{y↓}p_{y↓}>
        n_d_down = np.mean(exp_val[1, 2, :])   # <d^†_{↓}d_{↓}>
        
        # Total density on oxygen sites (p orbitals)
        n_p = n_px_up + n_py_up + n_px_down + n_py_down
        
        # Nematic order parameter
        eta = n_px_up + n_px_down - n_py_up - n_py_down
        
        # Total density
        n = n_p + n_d_up + n_d_down
        
        # Chemical potential (as defined in the problem)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        return n_p, eta, n, mu

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D1, D2, ..., DN, D1, D2, ..., DN, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate the hopping terms for each k point
        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        
        # gamma_1(k_i) = -2t_{pd}cos(k_i/2)
        gamma_1_kx = -2 * self.t_pd * np.cos(kx / 2)
        gamma_1_ky = -2 * self.t_pd * np.cos(ky / 2)
        
        # gamma_2(k) = -4t_{pp}cos(k_x/2)cos(k_y/2)
        gamma_2_k = -4 * self.t_pp * np.cos(kx / 2) * np.cos(ky / 2)
        
        # Fill the Hamiltonian for both spin up and spin down
        for s in range(2):
            # Hopping between p_x and p_y orbitals
            H_nonint[s, 0, s, 1, :] = gamma_2_k
            H_nonint[s, 1, s, 0, :] = gamma_2_k
            
            # Hopping between p_x and d orbitals
            H_nonint[s, 0, s, 2, :] = gamma_1_kx
            H_nonint[s, 2, s, 0, :] = gamma_1_kx
            
            # Hopping between p_y and d orbitals
            H_nonint[s, 1, s, 2, :] = gamma_1_ky
            H_nonint[s, 2, s, 1, :] = gamma_1_ky
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D1, D2, ..., DN, D1, D2, ..., DN, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute order parameters
        n_p, eta, n, mu = self.compute_order_parameters(exp_val)
        
        # Calculate the diagonal interaction terms
        xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu
        xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu
        xi_d = self.U_d_tilde * (n - n_p) / 2 - mu
        
        # Fill the diagonal interaction terms for both spin up and spin down
        for s in range(2):
            H_int[s, 0, s, 0, :] = xi_x  # p_x orbital
            H_int[s, 1, s, 1, :] = xi_y  # p_y orbital
            H_int[s, 2, s, 2, :] = xi_d  # d orbital
        
        # Add the constant term f(n^p, eta)
        f_constant = -self.U_p_tilde * (n_p**2) / 8 + self.V_pp_tilde * (eta**2) / 8 - self.U_d_tilde * ((n - n_p)**2) / 4
        
        # Distribute the constant term evenly across all diagonal elements
        for s in range(2):
            for o in range(3):
                H_int[s, o, s, o, :] += f_constant / (2 * 3)  # Divide by total number of diagonal elements
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns a flattened Hamiltonian.
            
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
