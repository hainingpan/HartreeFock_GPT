import numpy as np
from typing import Any, Dict, Tuple
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a three-orbital (p_x, p_y, d) model on a square lattice.
    
    This model represents a simplified model for cuprate superconductors with oxygen p orbitals
    and copper d orbitals, including interactions.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor (default 0.5).
    """
    def __init__(self, N_shell: int, parameters: Dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'
        self.D = (2, 3)  # (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Spin: up, down
        # Orbital: p_x, p_y, d
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
        self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping
        self.Delta = parameters.get('Delta', 2.0)  # p-d energy level difference
        
        # Interaction parameters
        self.U_p = parameters.get('U_p', 3.0)  # On-site interaction for p orbitals
        self.U_d = parameters.get('U_d', 8.0)  # On-site interaction for d orbitals
        self.V_pp = parameters.get('V_pp', 1.0)  # Inter-site interaction between p orbitals
        self.V_pd = parameters.get('V_pd', 1.5)  # Inter-site interaction between p and d orbitals
        
        return
    
    def generate_non_interacting(self) -> np.ndarray:
        """Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: Non-interacting Hamiltonian with shape (2, 3, 2, 3, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Extract k-space coordinates
        k_x = self.k_space[:, 0]
        k_y = self.k_space[:, 1]
        
        # Compute hopping terms
        gamma_1_kx = -2 * self.t_pd * np.cos(k_x/2)  # p_x-d hopping
        gamma_1_ky = -2 * self.t_pd * np.cos(k_y/2)  # p_y-d hopping
        gamma_2 = -4 * self.t_pp * np.cos(k_x/2) * np.cos(k_y/2)  # p_x-p_y hopping
        
        # Fill the Hamiltonian matrix for both spin channels
        for s in range(2):
            # p_x-d and d-p_x hopping
            H_nonint[s, 0, s, 2, :] = gamma_1_kx
            H_nonint[s, 2, s, 0, :] = gamma_1_kx
            
            # p_y-d and d-p_y hopping
            H_nonint[s, 1, s, 2, :] = gamma_1_ky
            H_nonint[s, 2, s, 1, :] = gamma_1_ky
            
            # p_x-p_y and p_y-p_x hopping
            H_nonint[s, 0, s, 1, :] = gamma_2
            H_nonint[s, 1, s, 0, :] = gamma_2
            
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val: Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: Interacting Hamiltonian with shape (2, 3, 2, 3, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute hole densities from expectation values
        n_p = 0  # Total density of holes on oxygen sites
        eta = 0  # Nematic order parameter
        n = 0   # Total density of holes
        
        for s in range(2):  # For both spin up and down
            # Density on oxygen sites (p_x and p_y)
            n_px_s = np.mean(exp_val[s, 0, s, 0, :])  # p_x spin-s occupation
            n_py_s = np.mean(exp_val[s, 1, s, 1, :])  # p_y spin-s occupation
            n_d_s = np.mean(exp_val[s, 2, s, 2, :])   # d spin-s occupation
            
            n_p += n_px_s + n_py_s            # Total p orbital occupation
            eta += n_px_s - n_py_s            # Nematic order parameter
            n += n_px_s + n_py_s + n_d_s      # Total occupation
        
        # Compute effective interaction parameters
        U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        V_pp_tilde = 8 * self.V_pp - self.U_p
        U_d_tilde = self.U_d - 4 * self.V_pd
        
        # Compute chemical potential (not an independent variable)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Compute single-particle terms
        xi_x = self.Delta + U_p_tilde * n_p/4 - V_pp_tilde * eta/4 - mu
        xi_y = self.Delta + U_p_tilde * n_p/4 + V_pp_tilde * eta/4 - mu
        xi_d = U_d_tilde * (n - n_p)/2 - mu
        
        # Compute the energy shift from the f(n^p, η) term
        f_np_eta = (-U_p_tilde * n_p**2/8 + 
                   V_pp_tilde * eta**2/8 - 
                   U_d_tilde * (n - n_p)**2/4)
        
        # Set the diagonal elements for both spin channels
        for s in range(2):
            # p_x orbital
            H_int[s, 0, s, 0, :] = xi_x
            
            # p_y orbital
            H_int[s, 1, s, 1, :] = xi_y
            
            # d orbital
            H_int[s, 2, s, 2, :] = xi_d
            
        # Add the global energy shift as a constant to all k-points
        # Note: In practice, this constant shift doesn't affect the eigenstates
        # but is important for total energy calculations
        constant_shift = f_np_eta / self.N_k
        for s1 in range(2):
            for o1 in range(3):
                H_int[s1, o1, s1, o1, :] += constant_shift
                
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val: Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat: Whether to return the flattened Hamiltonian.
            
        Returns:
            np.ndarray: Total Hamiltonian, either flattened or with shape (2, 3, 2, 3, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
