import numpy as np
from typing import Any, Dict
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for the Emery model with p_x, p_y, and d orbitals.
    
    This implements a three-orbital model on a square lattice with p-d and p-p hoppings,
    and includes interactions that can lead to nematic ordering.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system (default is 0.5).
    """
    def __init__(self, N_shell: int, parameters: Dict[str, Any] = {}, filling_factor: float = 0.5):
        self.lattice = 'square'
        self.D = (2, 3)  # (spin, orbital)
        self.basis_order = {
            '0': 'spin',     # 0: up, 1: down
            '1': 'orbital'   # 0: p_x, 1: p_y, 2: d
        }
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # Hopping between p and d orbitals
        self.t_pp = parameters.get('t_pp', 0.5)  # Hopping between p orbitals
        self.Delta = parameters.get('Delta', 1.0)  # Energy difference between p and d orbitals
        
        # Interaction parameters
        self.U_p = parameters.get('U_p', 1.0)  # On-site interaction for p orbitals
        self.U_d = parameters.get('U_d', 1.0)  # On-site interaction for d orbitals
        self.V_pp = parameters.get('V_pp', 0.5)  # Inter-p-orbital interaction
        self.V_pd = parameters.get('V_pd', 0.5)  # p-d interaction
        
        # Effective interaction parameters as defined in the model
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
        
        # Extract k_x and k_y for convenience
        k_x = self.k_space[:, 0]
        k_y = self.k_space[:, 1]
        
        # Calculate gamma terms for hopping
        gamma_1_x = -2 * self.t_pd * np.cos(k_x / 2)  # p_x-d hopping
        gamma_1_y = -2 * self.t_pd * np.cos(k_y / 2)  # p_y-d hopping
        gamma_2 = -4 * self.t_pp * np.cos(k_x / 2) * np.cos(k_y / 2)  # p_x-p_y hopping
        
        # Fill the Hamiltonian matrix for each spin (spin-independent Hamiltonian)
        for s in range(2):  # For both spin up and down
            # p_x - p_x term: only constant Delta (no k-dependence)
            H_nonint[s, 0, s, 0, :] = self.Delta
            
            # p_y - p_y term: only constant Delta (no k-dependence)
            H_nonint[s, 1, s, 1, :] = self.Delta
            
            # d - d term is 0 by default (reference energy)
            
            # Off-diagonal terms (hopping)
            # p_x - p_y hopping
            H_nonint[s, 0, s, 1, :] = gamma_2
            H_nonint[s, 1, s, 0, :] = gamma_2  # Hermitian conjugate
            
            # p_x - d hopping
            H_nonint[s, 0, s, 2, :] = gamma_1_x
            H_nonint[s, 2, s, 0, :] = gamma_1_x  # Hermitian conjugate
            
            # p_y - d hopping
            H_nonint[s, 1, s, 2, :] = gamma_1_y
            H_nonint[s, 2, s, 1, :] = gamma_1_y  # Hermitian conjugate
        
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
        # For p_x orbital (sum over both spins)
        n_px_up = np.mean(exp_val[0, 0, 0, 0, :])  # <p_x,up | p_x,up>
        n_px_down = np.mean(exp_val[1, 0, 1, 0, :])  # <p_x,down | p_x,down>
        n_px = n_px_up + n_px_down
        
        # For p_y orbital (sum over both spins)
        n_py_up = np.mean(exp_val[0, 1, 0, 1, :])  # <p_y,up | p_y,up>
        n_py_down = np.mean(exp_val[1, 1, 1, 1, :])  # <p_y,down | p_y,down>
        n_py = n_py_up + n_py_down
        
        # For d orbital (sum over both spins)
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :])  # <d,up | d,up>
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :])  # <d,down | d,down>
        n_d = n_d_up + n_d_down
        
        # Calculate total p and total densities
        n_p = n_px + n_py  # Total p-orbital density
        n = n_p + n_d      # Total density
        
        # Calculate nematic order parameter
        eta = n_px - n_py  # Difference between p_x and p_y occupation
        
        # Calculate chemical potential (as defined in the model)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Calculate xi terms (diagonal elements with interaction corrections)
        xi_x = self.Delta + self.U_p_tilde * n_p/4 - self.V_pp_tilde * eta/4 - mu
        xi_y = self.Delta + self.U_p_tilde * n_p/4 + self.V_pp_tilde * eta/4 - mu
        xi_d = self.U_d_tilde * (n - n_p)/2 - mu
        
        # Fill the diagonal elements of the interacting Hamiltonian
        for s in range(2):  # For both spin up and down
            H_int[s, 0, s, 0, :] = xi_x  # p_x diagonal term
            H_int[s, 1, s, 1, :] = xi_y  # p_y diagonal term
            H_int[s, 2, s, 2, :] = xi_d  # d diagonal term
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
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
