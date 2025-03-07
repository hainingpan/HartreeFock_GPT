import numpy as np
from typing import Any, Dict
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a 3-orbital model (p_x, p_y, d) with spin.
    
    This class implements a three-orbital model with spin degrees of freedom on a square lattice.
    The Hamiltonian includes both on-site and inter-site interactions and can exhibit
    nematic order depending on the parameters.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor for the system.
    """
    
    def __init__(self, N_shell: int = 3, parameters: Dict[str, Any] = None, filling_factor: float = 0.5):
        self.lattice = 'square'  # Lattice symmetry
        self.D = (2, 3)  # (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: orbital p_x, p_y, d
        
        # Default parameters if none provided
        if parameters is None:
            parameters = {}
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # temperature, fixed to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
        self.t_pp = parameters.get('t_pp', 0.3)  # p-p hopping
        self.Delta = parameters.get('Delta', 3.0)  # Energy level difference
        self.U_p = parameters.get('U_p', 3.0)  # On-site Coulomb repulsion on p orbitals
        self.U_d = parameters.get('U_d', 6.0)  # On-site Coulomb repulsion on d orbitals
        self.V_pp = parameters.get('V_pp', 1.0)  # Inter-site Coulomb repulsion between p orbitals
        self.V_pd = parameters.get('V_pd', 1.5)  # Inter-site Coulomb repulsion between p and d orbitals
        
        # Compute effective interaction parameters
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
        
        # Extract k_x and k_y
        k_x = self.k_space[:, 0]
        k_y = self.k_space[:, 1]
        
        # Compute gamma1 and gamma2 for all k-points
        gamma1_x = -2 * self.t_pd * np.cos(k_x/2)  # p_x-d hopping
        gamma1_y = -2 * self.t_pd * np.cos(k_y/2)  # p_y-d hopping
        gamma2 = -4 * self.t_pp * np.cos(k_x/2) * np.cos(k_y/2)  # p_x-p_y hopping
        
        # Fill the Hamiltonian matrix for each k-point and spin
        for s in range(2):  # Spin
            # p_x - p_y hopping (gamma2)
            H_nonint[s, 0, s, 1, :] = gamma2
            H_nonint[s, 1, s, 0, :] = gamma2
            
            # p_x - d hopping (gamma1_x)
            H_nonint[s, 0, s, 2, :] = gamma1_x
            H_nonint[s, 2, s, 0, :] = gamma1_x
            
            # p_y - d hopping (gamma1_y)
            H_nonint[s, 1, s, 2, :] = gamma1_y
            H_nonint[s, 2, s, 1, :] = gamma1_y
        
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
        
        # Compute expectation values for holes on different orbitals
        n_p_up_x = np.mean(exp_val[0, 0, 0, 0, :])  # <p_x^dag p_x> for spin up
        n_p_up_y = np.mean(exp_val[0, 1, 0, 1, :])  # <p_y^dag p_y> for spin up
        n_p_down_x = np.mean(exp_val[1, 0, 1, 0, :])  # <p_x^dag p_x> for spin down
        n_p_down_y = np.mean(exp_val[1, 1, 1, 1, :])  # <p_y^dag p_y> for spin down
        
        # Calculate n_p (total p orbital occupancy) and eta (nematic order parameter)
        n_p = n_p_up_x + n_p_up_y + n_p_down_x + n_p_down_y
        eta = n_p_up_x + n_p_down_x - n_p_up_y - n_p_down_y
        
        # Calculate d orbital occupancy
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :])  # <d^dag d> for spin up
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :])  # <d^dag d> for spin down
        n_d = n_d_up + n_d_down
        
        # Total hole density
        n = n_p + n_d
        
        # Compute the chemical potential
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Compute xi_x, xi_y, and xi_d (diagonal elements of H_ks)
        xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu
        xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu
        xi_d = self.U_d_tilde * (n - n_p) / 2 - mu
        
        # Fill the diagonal elements of the Hamiltonian
        for s in range(2):  # Spin
            H_int[s, 0, s, 0, :] = xi_x  # p_x orbital interaction term
            H_int[s, 1, s, 1, :] = xi_y  # p_y orbital interaction term
            H_int[s, 2, s, 2, :] = xi_d  # d orbital interaction term
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
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
