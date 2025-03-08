import numpy as np
from typing import Any, Dict
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a three-orbital (p_x, p_y, d) model with nematic order.
    
    This implementation models a square lattice with three orbitals (p_x, p_y, d)
    and includes interactions that can lead to nematic ordering between p_x and p_y orbitals.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: Dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'
        self.D = (2, 3)  # (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Orbital basis: p_x (0), p_y (1), d (2)
        # Spin basis: up (0), down (1)

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        self.Delta = parameters.get('Delta', 1.0)  # Energy difference between p and d orbitals
        self.t_pd = parameters.get('t_pd', 1.0)  # Hopping between p and d orbitals
        self.t_pp = parameters.get('t_pp', 0.5)  # Hopping between p orbitals
        
        # Interaction parameters
        self.U_p = parameters.get('U_p', 2.0)  # On-site interaction for p orbitals
        self.U_d = parameters.get('U_d', 4.0)  # On-site interaction for d orbitals
        self.V_pp = parameters.get('V_pp', 1.0)  # Interaction between p orbitals
        self.V_pd = parameters.get('V_pd', 1.5)  # Interaction between p and d orbitals
        
        return

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (2, 3, 2, 3, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Extract k-space points
        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        
        # Calculate hopping terms
        gamma_1_x = -2 * self.t_pd * np.cos(kx/2)  # p_x to d hopping
        gamma_1_y = -2 * self.t_pd * np.cos(ky/2)  # p_y to d hopping
        gamma_2 = -4 * self.t_pp * np.cos(kx/2) * np.cos(ky/2)  # p_x to p_y hopping
        
        # For both spin up and spin down (identical for both spins)
        for s in range(2):
            # Off-diagonal hopping terms
            H_nonint[s, 0, s, 2, :] = gamma_1_x  # p_x to d
            H_nonint[s, 2, s, 0, :] = gamma_1_x  # d to p_x (Hermitian conjugate)
            
            H_nonint[s, 1, s, 2, :] = gamma_1_y  # p_y to d
            H_nonint[s, 2, s, 1, :] = gamma_1_y  # d to p_y (Hermitian conjugate)
            
            H_nonint[s, 0, s, 1, :] = gamma_2  # p_x to p_y
            H_nonint[s, 1, s, 0, :] = gamma_2  # p_y to p_x (Hermitian conjugate)
            
            # The diagonal terms that don't depend on expectation values
            # (just the non-interacting part, such as site energy Δ)
            # These will be set to the Delta parameter in this case
            H_nonint[s, 0, s, 0, :] = self.Delta  # p_x on-site energy
            H_nonint[s, 1, s, 1, :] = self.Delta  # p_y on-site energy
            # d on-site energy is 0 by convention (energy reference)
            
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (2, 3, 2, 3, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate expectation values
        n_px_up = exp_val[0, 0, :].mean()
        n_px_down = exp_val[1, 0, :].mean()
        n_py_up = exp_val[0, 1, :].mean()
        n_py_down = exp_val[1, 1, :].mean()
        n_d_up = exp_val[0, 2, :].mean()
        n_d_down = exp_val[1, 2, :].mean()
        
        # Total p and d occupancies
        n_p = n_px_up + n_px_down + n_py_up + n_py_down
        n_d = n_d_up + n_d_down
        n_total = n_p + n_d
        
        # Nematic order parameter
        eta = n_px_up + n_px_down - n_py_up - n_py_down
        
        # Calculate effective interaction parameters
        U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        V_pp_tilde = 8 * self.V_pp - self.U_p
        U_d_tilde = self.U_d - 4 * self.V_pd
        
        # Calculate chemical potential
        mu = 2 * self.V_pd * n_total - self.V_pd * n_total**2
        
        # Calculate the interacting part for diagonal elements
        xi_x = U_p_tilde * n_p/4 - V_pp_tilde * eta/4 - mu
        xi_y = U_p_tilde * n_p/4 + V_pp_tilde * eta/4 - mu
        xi_d = U_d_tilde * (n_total - n_p)/2 - mu
        
        # Update diagonal elements for both spin channels
        for s in range(2):
            H_int[s, 0, s, 0, :] = xi_x  # p_x orbital
            H_int[s, 1, s, 1, :] = xi_y  # p_y orbital
            H_int[s, 2, s, 2, :] = xi_d  # d orbital
            
        # Note: The f(n^p, η) term contributes a constant energy shift
        # which doesn't affect the eigenvectors of the Hamiltonian
        # f_correction = -U_p_tilde * n_p**2/8 + V_pp_tilde * eta**2/8 - U_d_tilde * (n_total - n_p)**2/4
        # This term would add a scalar to the total energy but not affect the HF iterations
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            return_flat (bool): Whether to return the flattened array.
            
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
