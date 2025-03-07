import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a three-orbital model with px, py, and d orbitals.
    
    The Hamiltonian describes a square lattice with three orbitals per unit cell and
    includes both kinetic energy terms and interaction terms.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system.
    """
    def __init__(self, N_shell: int = 5, 
                 parameters: dict[str, Any] = {
                     't_pd': 1.0, 't_pp': 0.5, 
                     'U_p': 3.0, 'U_d': 8.0, 
                     'V_pp': 1.0, 'V_pd': 1.0,
                     'Delta': 3.5, 'T': 0, 'a': 1.0
                 }, 
                 filling_factor: float = 0.5):
        """
        Initialize the Hartree-Fock Hamiltonian for the three-orbital model.
        """
        self.lattice = 'square'  # Lattice symmetry
        self.D = (2, 3)  # (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: px, py, d orbitals
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
        self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping
        self.U_p = parameters.get('U_p', 3.0)    # On-site repulsion for p orbitals
        self.U_d = parameters.get('U_d', 8.0)    # On-site repulsion for d orbitals
        self.V_pp = parameters.get('V_pp', 1.0)  # Inter-site p-p repulsion
        self.V_pd = parameters.get('V_pd', 1.0)  # Inter-site p-d repulsion
        self.Delta = parameters.get('Delta', 3.5)  # Energy difference between p and d orbitals
        
        # Derived interaction parameters
        self.tilde_U_p = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        self.tilde_V_pp = 8 * self.V_pp - self.U_p
        self.tilde_U_d = self.U_d - 4 * self.V_pd
        
        return

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # For both spin up and down
        for s in range(2):
            for k_idx in range(self.N_k):
                kx, ky = self.k_space[k_idx]
                
                # Calculate gamma terms
                gamma_1_kx = -2 * self.t_pd * np.cos(kx/2)  # px-d hopping
                gamma_1_ky = -2 * self.t_pd * np.cos(ky/2)  # py-d hopping
                gamma_2_k = -4 * self.t_pp * np.cos(kx/2) * np.cos(ky/2)  # px-py hopping
                
                # Fill the 3x3 Hamiltonian matrix for each k-point and spin
                # Off-diagonal elements - non-interacting part
                H_nonint[s, 0, s, 1, k_idx] = gamma_2_k  # px-py hopping
                H_nonint[s, 1, s, 0, k_idx] = gamma_2_k  # py-px hopping
                
                H_nonint[s, 0, s, 2, k_idx] = gamma_1_kx  # px-d hopping
                H_nonint[s, 2, s, 0, k_idx] = gamma_1_kx  # d-px hopping
                
                H_nonint[s, 1, s, 2, k_idx] = gamma_1_ky  # py-d hopping
                H_nonint[s, 2, s, 1, k_idx] = gamma_1_ky  # d-py hopping
        
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
        
        # Calculate densities from expectation values
        # n_p: total density of holes on oxygen sites (px and py)
        n_px_up = np.mean(exp_val[0, 0, 0, 0, :])  # <p^†_{x↑} p_{x↑}>
        n_px_down = np.mean(exp_val[1, 0, 1, 0, :])  # <p^†_{x↓} p_{x↓}>
        n_py_up = np.mean(exp_val[0, 1, 0, 1, :])  # <p^†_{y↑} p_{y↑}>
        n_py_down = np.mean(exp_val[1, 1, 1, 1, :])  # <p^†_{y↓} p_{y↓}>
        
        n_p = n_px_up + n_px_down + n_py_up + n_py_down
        
        # Calculate d-orbital densities
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :])  # <d^†_{↑} d_{↑}>
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :])  # <d^†_{↓} d_{↓}>
        n_d = n_d_up + n_d_down
        
        # Total density
        n = n_p + n_d
        
        # Nematic order parameter η: difference between px and py occupations
        eta = (n_px_up + n_px_down) - (n_py_up + n_py_down)
        
        # Calculate chemical potential (assuming it's set to ensure correct filling)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Calculate ξ terms
        xi_x = self.Delta + self.tilde_U_p * n_p/4 - self.tilde_V_pp * eta/4 - mu
        xi_y = self.Delta + self.tilde_U_p * n_p/4 + self.tilde_V_pp * eta/4 - mu
        xi_d = self.tilde_U_d * (n - n_p)/2 - mu
        
        # Add diagonal interacting terms for both spin up and down
        for s in range(2):
            for k_idx in range(self.N_k):
                # Diagonal elements - interacting part
                H_int[s, 0, s, 0, k_idx] = xi_x  # px orbital energy + interaction
                H_int[s, 1, s, 1, k_idx] = xi_y  # py orbital energy + interaction
                H_int[s, 2, s, 2, k_idx] = xi_d  # d orbital energy + interaction
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian.
            
        Returns:
            np.ndarray: The total Hamiltonian with shape (D, D, N_k) or flattened shape.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
