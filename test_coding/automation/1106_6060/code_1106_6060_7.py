import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a three-orbital (p_x, p_y, d) model with spin.
    
    This class implements the mean-field Hamiltonian for a model with three orbitals
    (p_x, p_y, d) and two spin states (up, down). The Hamiltonian includes both
    non-interacting terms (hopping) and interacting terms (on-site and inter-site
    interactions).
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor for the system.
    """
    def __init__(self, N_shell: int = 1, parameters: dict[str, Any] = None, filling_factor: float = 0.5):
        if parameters is None:
            parameters = {}
        
        self.lattice = 'square'  # Lattice symmetry
        self.D = (2, 3)  # Number of flavors: (spin, orbital)
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
        self.Delta = parameters.get('Delta', 1.0)  # Energy offset for p orbitals
        
        # Interaction parameters
        self.U_p = parameters.get('U_p', 2.0)  # On-site interaction for p orbitals
        self.U_d = parameters.get('U_d', 4.0)  # On-site interaction for d orbitals
        self.V_pp = parameters.get('V_pp', 0.5)  # Inter-site interaction between p orbitals
        self.V_pd = parameters.get('V_pd', 0.5)  # Inter-site interaction between p and d orbitals
        
        # Effective interaction parameters
        self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        self.V_pp_tilde = 8 * self.V_pp - self.U_p
        self.U_d_tilde = self.U_d - 4 * self.V_pd
        
        return
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        This includes the hopping terms between different orbitals.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Extract k_x and k_y components
        k_x = self.k_space[:, 0]
        k_y = self.k_space[:, 1]
        
        # Calculate gamma values for each k-point
        gamma_1_kx = -2 * self.t_pd * np.cos(k_x / 2)  # p_x-d hopping
        gamma_1_ky = -2 * self.t_pd * np.cos(k_y / 2)  # p_y-d hopping
        gamma_2_k = -4 * self.t_pp * np.cos(k_x / 2) * np.cos(k_y / 2)  # p_x-p_y hopping
        
        # Fill the non-interacting Hamiltonian matrix for both spin channels
        for s in range(2):  # Loop over spin
            # Off-diagonal elements (non-interacting)
            # p_x - p_y coupling
            H_nonint[s, 0, s, 1, :] = gamma_2_k  # <p^†_x p_y>
            H_nonint[s, 1, s, 0, :] = gamma_2_k  # <p^†_y p_x>
            
            # p_x - d coupling
            H_nonint[s, 0, s, 2, :] = gamma_1_kx  # <p^†_x d>
            H_nonint[s, 2, s, 0, :] = gamma_1_kx  # <d^† p_x>
            
            # p_y - d coupling
            H_nonint[s, 1, s, 2, :] = gamma_1_ky  # <p^†_y d>
            H_nonint[s, 2, s, 1, :] = gamma_1_ky  # <d^† p_y>
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.
        
        This includes the diagonal terms that depend on the expectation values
        of density operators.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate densities from expectation values
        n_p_x_up = np.mean(exp_val[0, 0, 0, 0, :])  # <p^†_x↑ p_x↑>
        n_p_x_down = np.mean(exp_val[1, 0, 1, 0, :])  # <p^†_x↓ p_x↓>
        n_p_y_up = np.mean(exp_val[0, 1, 0, 1, :])  # <p^†_y↑ p_y↑>
        n_p_y_down = np.mean(exp_val[1, 1, 1, 1, :])  # <p^†_y↓ p_y↓>
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :])  # <d^†↑ d↑>
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :])  # <d^†↓ d↓>
        
        # Calculate derived quantities
        n_p = n_p_x_up + n_p_x_down + n_p_y_up + n_p_y_down  # Total p-orbital density
        n = n_p + n_d_up + n_d_down  # Total density
        eta = (n_p_x_up + n_p_x_down) - (n_p_y_up + n_p_y_down)  # Nematic order parameter
        
        # Calculate chemical potential (adjusted to reflect definition in the problem)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Calculate diagonal elements that depend on densities
        xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu  # p_x energy
        xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu  # p_y energy
        xi_d = self.U_d_tilde * (n - n_p) / 2 - mu  # d energy
        
        # Fill the interacting part of the Hamiltonian
        for s in range(2):  # Loop over spin
            # Diagonal elements (interacting)
            H_int[s, 0, s, 0, :] = xi_x  # p_x orbital
            H_int[s, 1, s, 1, :] = xi_y  # p_y orbital
            H_int[s, 2, s, 2, :] = xi_d  # d orbital
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        This combines the non-interacting and interacting parts of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return a flattened Hamiltonian.
            
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
