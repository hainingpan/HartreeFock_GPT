import numpy as np
from typing import Any, Dict, Optional, Tuple
from HF import *

class HartreeFockHamiltonian:
    """A Hartree-Fock Hamiltonian implementation for the three-band Hubbard model (Emery model).
    
    This class implements a three-band Hubbard model with p_x, p_y (oxygen), and d (copper) 
    orbitals on a square lattice. The model includes hopping between d and p orbitals (t_pd), 
    between p orbitals (t_pp), as well as on-site and inter-site Coulomb interactions.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
    """
    def __init__(self, N_shell: int = 3, parameters: Dict[str, Any] = None):
        # Set default parameters if none provided
        if parameters is None:
            parameters = {}
        
        # Lattice parameters
        self.lattice = 'square'
        self.D = (2, 3)  # (spin, orbital)
        self.basis_order = {
            '0': 'spin: up, down',
            '1': 'orbital: p_x, p_y, d'
        }
        
        # Temperature and lattice constant
        self.T = parameters.get('T', 0)  # Temperature, default is 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        
        # Generate k-space
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # d-p hopping parameter
        self.t_pp = parameters.get('t_pp', 0.3)  # p-p hopping parameter
        self.Delta = parameters.get('Delta', 3.0)  # Energy level difference between p and d orbitals
        
        # Interaction parameters
        self.U_p = parameters.get('U_p', 4.0)  # On-site Coulomb for p orbitals
        self.U_d = parameters.get('U_d', 8.0)  # On-site Coulomb for d orbitals
        self.V_pp = parameters.get('V_pp', 1.0)  # Inter-site Coulomb between p orbitals
        self.V_pd = parameters.get('V_pd', 1.0)  # Inter-site Coulomb between p and d orbitals
        
        # Total number of holes
        self.n_total = parameters.get('n', 1.0)  # Total hole density

    def generate_non_interacting(self) -> np.ndarray:
        """Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Extract k-points
        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        
        # Calculate hopping terms
        gamma_1_x = -2 * self.t_pd * np.cos(kx/2)  # d-p_x hopping
        gamma_1_y = -2 * self.t_pd * np.cos(ky/2)  # d-p_y hopping
        gamma_2 = -4 * self.t_pp * np.cos(kx/2) * np.cos(ky/2)  # p_x-p_y hopping
        
        # Set up the non-interacting Hamiltonian for each spin
        for s in range(2):  # Loop over spin up (s=0) and spin down (s=1)
            # Only add the constant part (Delta) to the diagonal elements
            # p_x diagonal
            H_nonint[s, 0, s, 0, :] = self.Delta  # p_x orbital energy
            
            # p_y diagonal
            H_nonint[s, 1, s, 1, :] = self.Delta  # p_y orbital energy
            
            # d diagonal is set to 0 (reference energy level)
            
            # Off-diagonal hopping terms
            # p_x-p_y hopping
            H_nonint[s, 0, s, 1, :] = gamma_2
            H_nonint[s, 1, s, 0, :] = gamma_2
            
            # p_x-d hopping
            H_nonint[s, 0, s, 2, :] = gamma_1_x
            H_nonint[s, 2, s, 0, :] = gamma_1_x
            
            # p_y-d hopping
            H_nonint[s, 1, s, 2, :] = gamma_1_y
            H_nonint[s, 2, s, 1, :] = gamma_1_y
            
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        # Unflatten expectation values to full shape
        exp_val = unflatten(exp_val, self.D, self.N_k)
        
        # Initialize interacting Hamiltonian
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate expectation values
        # Density on p_x orbital (spin up + spin down)
        n_px = np.mean(exp_val[0, 0, 0, 0, :] + exp_val[1, 0, 1, 0, :])
        
        # Density on p_y orbital (spin up + spin down)
        n_py = np.mean(exp_val[0, 1, 0, 1, :] + exp_val[1, 1, 1, 1, :])
        
        # Density on d orbital (spin up + spin down)
        n_d = np.mean(exp_val[0, 2, 0, 2, :] + exp_val[1, 2, 1, 2, :])
        
        # Total density on p orbitals
        n_p = n_px + n_py
        
        # Total density
        n = n_p + n_d
        
        # Nematic order parameter
        eta = n_px - n_py
        
        # Calculate effective interactions
        U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        V_pp_tilde = 8 * self.V_pp - self.U_p
        U_d_tilde = self.U_d - 4 * self.V_pd
        
        # Chemical potential (adjusted to maintain total density)
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Calculate diagonal energy terms
        xi_x = self.Delta + U_p_tilde * n_p/4 - V_pp_tilde * eta/4 - mu
        xi_y = self.Delta + U_p_tilde * n_p/4 + V_pp_tilde * eta/4 - mu
        xi_d = U_d_tilde * (n - n_p)/2 - mu
        
        # Set up the interacting Hamiltonian
        for s in range(2):  # Loop over spin
            # Diagonal terms
            H_int[s, 0, s, 0, :] = xi_x - self.Delta  # p_x
            H_int[s, 1, s, 1, :] = xi_y - self.Delta  # p_y
            H_int[s, 2, s, 2, :] = xi_d  # d
        
        # Additional constant term f(n^p, η)
        # This term is uniform for all states, so we don't add it to H_int
        # It only shifts the total energy of the system without affecting the eigenstates
        # f_constant = -U_p_tilde * (n_p)**2/8 + V_pp_tilde * eta**2/8 - U_d_tilde * (n - n_p)**2/4
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian.
            
        Returns:
            np.ndarray: The total Hamiltonian. If return_flat is True, shape is 
                        (D_flattened, D_flattened, N_k), otherwise (D, D, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
