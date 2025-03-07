import numpy as np
from typing import Any, Dict, Optional
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for the Emery model on a square lattice.
    
    This implements the three-band Emery model with p_x, p_y and d orbitals.
    The model includes hopping between orbitals and interaction terms
    that create a self-consistent mean field problem.
    
    Args:
        N_shell (int): Number of shells in the k-space lattice.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor determining electron density.
    """
    def __init__(self, N_shell: int=10, parameters: Optional[Dict[str, Any]]=None, filling_factor: float=0.5):
        self.lattice = 'square'
        self.D = (2, 3)  # (spin, orbital)
        self.basis_order = {
            '0': 'spin',    # up (s=0), down (s=1)
            '1': 'orbital'  # p_x (o=0), p_y (o=1), d (o=2)
        }
        
        # Occupancy parameters
        self.nu = filling_factor
        self.T = 0.0  # Temperature is set to zero
        
        # Default parameters if none provided
        if parameters is None:
            parameters = {}
            
        # Lattice constant
        self.a = parameters.get('a', 1.0)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters with defaults
        self.t_pd = parameters.get('t_pd', 1.0)    # p-d hopping parameter
        self.t_pp = parameters.get('t_pp', 0.3)    # p-p hopping parameter
        self.Delta = parameters.get('Delta', 3.0)  # Energy difference between p and d orbitals
        
        # Interaction parameters
        self.U_p = parameters.get('U_p', 4.0)      # On-site repulsion for p orbitals
        self.U_d = parameters.get('U_d', 8.0)      # On-site repulsion for d orbitals
        self.V_pp = parameters.get('V_pp', 1.0)    # Inter-site p-p repulsion
        self.V_pd = parameters.get('V_pd', 1.0)    # Inter-site p-d repulsion
        
        # Derived interaction parameters
        self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        self.V_pp_tilde = 8 * self.V_pp - self.U_p
        self.U_d_tilde = self.U_d - 4 * self.V_pd

    def generate_non_interacting(self) -> np.ndarray:
        """Generates the non-interacting part of the Hamiltonian.
        
        This includes the kinetic terms (hopping) and the orbital energy difference.
        
        Returns:
            np.ndarray: Non-interacting Hamiltonian with shape (D + D + (N_k,))
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=np.complex128)
        
        # Compute gamma1 and gamma2 for each k-point
        gamma1_x = -2 * self.t_pd * np.cos(self.k_space[:, 0] / 2)
        gamma1_y = -2 * self.t_pd * np.cos(self.k_space[:, 1] / 2)
        gamma2 = -4 * self.t_pp * np.cos(self.k_space[:, 0] / 2) * np.cos(self.k_space[:, 1] / 2)
        
        # Define H_nonint for each spin (no mixing between spins)
        for s in range(2):
            # Diagonal elements - only include Delta for p orbitals
            H_nonint[s, 0, s, 0, :] = self.Delta  # p_x-p_x
            H_nonint[s, 1, s, 1, :] = self.Delta  # p_y-p_y
            # d orbital is reference (zero energy)
            
            # Off-diagonal hopping terms
            H_nonint[s, 0, s, 1, :] = gamma2      # p_x-p_y
            H_nonint[s, 1, s, 0, :] = gamma2      # p_y-p_x
            
            H_nonint[s, 0, s, 2, :] = gamma1_x    # p_x-d
            H_nonint[s, 2, s, 0, :] = gamma1_x    # d-p_x
            
            H_nonint[s, 1, s, 2, :] = gamma1_y    # p_y-d
            H_nonint[s, 2, s, 1, :] = gamma1_y    # d-p_y
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generates the interacting part of the Hamiltonian.
        
        Computes the mean-field interaction terms based on the expectation values.
        
        Args:
            exp_val: Expectation values with shape (D_flattened*D_flattened, N_k)
            
        Returns:
            np.ndarray: Interacting Hamiltonian with shape (D + D + (N_k,))
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=np.complex128)
        
        # Calculate occupation numbers for each orbital and spin
        n_px_up = np.mean(exp_val[0, 0, 0, 0, :].real)    # <p_x,up^† p_x,up>
        n_px_down = np.mean(exp_val[1, 0, 1, 0, :].real)  # <p_x,down^† p_x,down>
        n_py_up = np.mean(exp_val[0, 1, 0, 1, :].real)    # <p_y,up^† p_y,up>
        n_py_down = np.mean(exp_val[1, 1, 1, 1, :].real)  # <p_y,down^† p_y,down>
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :].real)     # <d_up^† d_up>
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :].real)   # <d_down^† d_down>
        
        # Calculate order parameters
        n_p = n_px_up + n_px_down + n_py_up + n_py_down  # Total p-orbital density
        eta = n_px_up + n_px_down - n_py_up - n_py_down  # Nematic order parameter
        n = n_p + n_d_up + n_d_down                      # Total density
        
        # Chemical potential (fixed by total density)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Double-counting correction term
        f_n_eta = -self.U_p_tilde * (n_p**2) / 8 + self.V_pp_tilde * (eta**2) / 8 - self.U_d_tilde * ((n - n_p)**2) / 4
        
        # On-site energies with interactions
        xi_x = self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu
        xi_y = self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu
        xi_d = self.U_d_tilde * (n - n_p) / 2 - mu
        
        # Add interaction terms to diagonal elements for each spin
        for s in range(2):
            H_int[s, 0, s, 0, :] = xi_x  # p_x-p_x
            H_int[s, 1, s, 1, :] = xi_y  # p_y-p_y
            H_int[s, 2, s, 2, :] = xi_d  # d-d
        
        # Note: The double-counting term f(n^p, η) is not added to the Hamiltonian
        # as it's a constant energy shift that doesn't affect eigenstates
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """Generates the total Hartree-Fock Hamiltonian.
        
        Combines the non-interacting and interacting parts of the Hamiltonian.
        
        Args:
            exp_val: Expectation values with shape (D_flattened*D_flattened, N_k)
            return_flat: Whether to return the Hamiltonian in flattened form
            
        Returns:
            np.ndarray: Total Hamiltonian
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
