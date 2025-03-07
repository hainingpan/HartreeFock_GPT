import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a three-orbital (p_x, p_y, d) model on a square lattice.
    
    This implements the mean-field Hamiltonian:
    H_MF = sum_{k,s} C^dag_{k,s} H_{k,s} C_{k,s} + f(n^p, eta)
    
    where C^dag_{k,s} = (p^dag_{x,k,s}, p^dag_{y,k,s}, d^dag_{k,s})
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular')
        self.D = (2, 3)  # Number of flavors: (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: orbital p_x, p_y, d

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
        self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping
        self.Delta = parameters.get('Delta', 3.0)  # On-site energy difference
        
        # Interaction parameters
        self.U_p = parameters.get('U_p', 4.0)  # On-site Coulomb repulsion at oxygen sites
        self.U_d = parameters.get('U_d', 8.0)  # On-site Coulomb repulsion at copper sites
        self.V_pp = parameters.get('V_pp', 1.0)  # Nearest-neighbor interaction between oxygen sites
        self.V_pd = parameters.get('V_pd', 1.0)  # Nearest-neighbor interaction between p and d orbitals
        
        # Derived interaction parameters (from Eq. Uptilde, Vptilde, Udtilde)
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
        
        # Calculate the k-dependent hopping terms
        gamma_1_kx = -2 * self.t_pd * np.cos(self.k_space[:, 0] / 2)  # p_x-d hopping
        gamma_1_ky = -2 * self.t_pd * np.cos(self.k_space[:, 1] / 2)  # p_y-d hopping
        gamma_2_k = -4 * self.t_pp * np.cos(self.k_space[:, 0] / 2) * np.cos(self.k_space[:, 1] / 2)  # p_x-p_y hopping
        
        # Fill the Hamiltonian matrix for each spin
        for s in range(2):  # Loop over spin (up, down)
            # Non-interacting on-site energy terms (Delta term)
            H_nonint[s, 0, s, 0, :] = self.Delta  # p_x orbital on-site energy
            H_nonint[s, 1, s, 1, :] = self.Delta  # p_y orbital on-site energy
            # d orbital on-site energy is 0
            
            # Off-diagonal hopping terms
            # p_x-p_y hopping
            H_nonint[s, 0, s, 1, :] = gamma_2_k
            H_nonint[s, 1, s, 0, :] = gamma_2_k
            
            # p_x-d hopping
            H_nonint[s, 0, s, 2, :] = gamma_1_kx
            H_nonint[s, 2, s, 0, :] = gamma_1_kx
            
            # p_y-d hopping
            H_nonint[s, 1, s, 2, :] = gamma_1_ky
            H_nonint[s, 2, s, 1, :] = gamma_1_ky
        
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
        
        # Calculate expectation values for hole densities
        n_p_x_up = np.mean(exp_val[0, 0, 0, 0, :])    # <p^dag_{x,up} p_{x,up}>
        n_p_x_down = np.mean(exp_val[1, 0, 1, 0, :])  # <p^dag_{x,down} p_{x,down}>
        n_p_y_up = np.mean(exp_val[0, 1, 0, 1, :])    # <p^dag_{y,up} p_{y,up}>
        n_p_y_down = np.mean(exp_val[1, 1, 1, 1, :])  # <p^dag_{y,down} p_{y,down}>
        n_d_up = np.mean(exp_val[0, 2, 0, 2, :])      # <d^dag_{up} d_{up}>
        n_d_down = np.mean(exp_val[1, 2, 1, 2, :])    # <d^dag_{down} d_{down}>
        
        # Calculate derived quantities
        n_p = n_p_x_up + n_p_x_down + n_p_y_up + n_p_y_down  # Total hole density on oxygen sites
        n = n_p + n_d_up + n_d_down  # Total hole density
        eta = (n_p_x_up + n_p_x_down) - (n_p_y_up + n_p_y_down)  # Nematic order parameter
        
        # Calculate the chemical potential
        mu = 2 * self.V_pd * n - self.V_pd * n**2
        
        # Calculate the interacting on-site energies
        xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu
        xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu
        xi_d = self.U_d_tilde * (n - n_p) / 2 - mu
        
        # Calculate f(n^p, eta) constant term
        f_const = -self.U_p_tilde * (n_p**2) / 8 + self.V_pp_tilde * (eta**2) / 8 - self.U_d_tilde * ((n - n_p)**2) / 4
        
        # Update the diagonal elements of the Hamiltonian
        for s in range(2):  # Loop over spin
            # p_x orbital interaction term
            H_int[s, 0, s, 0, :] = xi_x
            
            # p_y orbital interaction term
            H_int[s, 1, s, 1, :] = xi_y
            
            # d orbital interaction term
            H_int[s, 2, s, 2, :] = xi_d
        
        # Add constant term f(n^p, eta) distributed evenly across all states
        # This is a constant energy shift that doesn't affect the eigenstates
        # We distribute it evenly across diagonal elements to maintain total energy
        f_per_state = f_const / (2 * 3 * self.N_k)  # Divide by total number of states
        for s in range(2):
            for o in range(3):
                H_int[s, o, s, o, :] += f_per_state
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns the flattened Hamiltonian.

        Returns:
            np.ndarray: The total Hamiltonian with shape (D_flattened, D_flattened, N_k) if return_flat is True,
                      otherwise with shape (D, D, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
