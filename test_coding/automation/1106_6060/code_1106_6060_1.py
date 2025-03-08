import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a three-orbital (p_x, p_y, d) model on a square lattice.
    
    Args:
        N_shell (int): Number of shells in the reciprocal space.
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
        self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping parameter
        self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping parameter
        self.Delta = parameters.get('Delta', 0.0)  # Energy level difference
        self.U_p = parameters.get('U_p', 1.0)  # On-site Coulomb repulsion for p orbitals
        self.V_pp = parameters.get('V_pp', 0.5)  # Inter-site Coulomb interaction between p orbitals
        self.V_pd = parameters.get('V_pd', 0.5)  # Inter-site Coulomb interaction between p and d orbitals
        self.U_d = parameters.get('U_d', 1.0)  # On-site Coulomb repulsion for d orbitals
        
        # Derived parameters (effective interaction strengths)
        self.Utilde_p = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        self.Vtilde_pp = 8 * self.V_pp - self.U_p
        self.Utilde_d = self.U_d - 4 * self.V_pd
        
        return
    
    def compute_order_parameters(self, exp_val):
        """
        Compute the order parameters from the expectation values.
        
        Args:
            exp_val (np.ndarray): Expectation values with shape (*self.D, *self.D, self.N_k).
        
        Returns:
            tuple: Order parameters: n_p (total p-site occupation), eta (nematic order parameter), n (total occupation).
        """
        # Extract occupation numbers for each orbital and spin
        n_x_up = np.mean(exp_val[0, 0, 0, 0, :])  # <p^\dagger_x,up p_x,up>
        n_x_down = np.mean(exp_val[1, 0, 1, 0, :])  # <p^\dagger_x,down p_x,down>
        n_y_up = np.mean(exp_val[0, 1, 0, 1, :])  # <p^\dagger_y,up p_y,up>
        n_y_down = np.mean(exp_val[1, 1, 1, 1, :])  # <p^\dagger_y,down p_y,down>
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :])  # <d^\dagger_up d_up>
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :])  # <d^\dagger_down d_down>
        
        # Compute order parameters
        n_p = n_x_up + n_x_down + n_y_up + n_y_down  # total p-orbital occupation
        eta = (n_x_up + n_x_down) - (n_y_up + n_y_down)  # nematic order parameter
        n = n_p + n_d_up + n_d_down  # total occupation
        
        return n_p, eta, n
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generate the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: Non-interacting Hamiltonian with shape (*self.D, *self.D, self.N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Generate the non-interacting terms for each k-point and spin
        for s in range(2):  # Loop over spins
            # Hopping terms (independent of spin)
            for k in range(self.N_k):
                kx, ky = self.k_space[k]
                
                # Compute hopping parameters
                gamma_1_x = -2 * self.t_pd * np.cos(kx / 2)  # p_x-d hopping
                gamma_1_y = -2 * self.t_pd * np.cos(ky / 2)  # p_y-d hopping
                gamma_2 = -4 * self.t_pp * np.cos(kx / 2) * np.cos(ky / 2)  # p_x-p_y hopping
                
                # Fill the matrix for this k-point and spin
                # p_x - p_y hopping
                H_nonint[s, 0, s, 1, k] = gamma_2
                H_nonint[s, 1, s, 0, k] = gamma_2
                
                # p_x - d hopping
                H_nonint[s, 0, s, 2, k] = gamma_1_x
                H_nonint[s, 2, s, 0, k] = gamma_1_x
                
                # p_y - d hopping
                H_nonint[s, 1, s, 2, k] = gamma_1_y
                H_nonint[s, 2, s, 1, k] = gamma_1_y
                
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation values with shape (D_flattened, D_flattened, self.N_k).
        
        Returns:
            np.ndarray: Interacting Hamiltonian with shape (*self.D, *self.D, self.N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute the order parameters
        n_p, eta, n = self.compute_order_parameters(exp_val)
        
        # Compute the chemical potential (depends on order parameters)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Interacting terms (diagonal elements that depend on order parameters)
        for s in range(2):  # Loop over spins
            # On-site energies for each orbital
            # xi_x term for p_x orbital
            xi_x = self.Delta + self.Utilde_p * n_p / 4 - self.Vtilde_pp * eta / 4 - mu
            H_int[s, 0, s, 0, :] = xi_x
            
            # xi_y term for p_y orbital
            xi_y = self.Delta + self.Utilde_p * n_p / 4 + self.Vtilde_pp * eta / 4 - mu
            H_int[s, 1, s, 1, :] = xi_y
            
            # xi_d term for d orbital
            xi_d = self.Utilde_d * (n - n_p) / 2 - mu
            H_int[s, 2, s, 2, :] = xi_d
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generate the total Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation values with shape (D_flattened, D_flattened, self.N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian.
        
        Returns:
            np.ndarray: Total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        # Note: The constant term f(n_p, eta) is not included in the Hamiltonian matrix
        # as it doesn't affect the eigenvectors. It would be added to total energy calculations.
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
