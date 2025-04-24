import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Implementation of the Hartree-Fock Hamiltonian for a three-orbital model (px, py, d)
    with spin degrees of freedom on a square lattice.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'  # Lattice type
        self.D = (2, 3)  # (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: px, py, d

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # Temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # Hopping between p and d orbitals
        self.t_pp = parameters.get('t_pp', 0.5)  # Hopping between p orbitals
        self.Delta = parameters.get('Delta', 1.0)  # On-site energy difference
        self.U_p_tilde = parameters.get('U_p_tilde', 2.0)  # Effective interaction on p orbitals
        self.V_pp_tilde = parameters.get('V_pp_tilde', 1.0)  # Effective interaction between p orbitals
        self.U_d_tilde = parameters.get('U_d_tilde', 2.0)  # Effective interaction on d orbital
        self.n = parameters.get('n', 1.0)  # Total hole density
        self.mu = parameters.get('mu', 0.0)  # Chemical potential
        
        return

    def compute_order_parameters(self, exp_val: np.ndarray):
        """
        Computes the order parameters n^p and eta from the expectation values.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            
        Returns:
            tuple: (n_p, eta) where n_p is the total density of holes on oxygen sites and
                  eta is the nematic order parameter.
        """
        # Compute the occupancies for each orbital and spin
        n_x_up = np.mean(exp_val[0, 0, 0, 0, :])
        n_x_down = np.mean(exp_val[1, 0, 1, 0, :])
        n_y_up = np.mean(exp_val[0, 1, 0, 1, :])
        n_y_down = np.mean(exp_val[1, 1, 1, 1, :])
        
        # Compute n^p and eta
        n_p = n_x_up + n_x_down + n_y_up + n_y_down
        eta = (n_x_up + n_x_down) - (n_y_up + n_y_down)
        
        return n_p, eta

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian (hopping terms).
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D[0], D[1], D[0], D[1], N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Extract k-coordinates
        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        
        # Compute the hopping elements
        gamma_1_kx = -2 * self.t_pd * np.cos(kx / 2)  # Hopping between px and d
        gamma_1_ky = -2 * self.t_pd * np.cos(ky / 2)  # Hopping between py and d
        gamma_2_k = -4 * self.t_pp * np.cos(kx / 2) * np.cos(ky / 2)  # Hopping between px and py
        
        # Spin up
        # Off-diagonal hopping terms
        H_nonint[0, 0, 0, 1, :] = gamma_2_k  # px to py hopping
        H_nonint[0, 1, 0, 0, :] = gamma_2_k  # py to px hopping
        H_nonint[0, 0, 0, 2, :] = gamma_1_kx  # px to d hopping
        H_nonint[0, 2, 0, 0, :] = gamma_1_kx  # d to px hopping
        H_nonint[0, 1, 0, 2, :] = gamma_1_ky  # py to d hopping
        H_nonint[0, 2, 0, 1, :] = gamma_1_ky  # d to py hopping
        
        # Spin down (same hopping terms as spin up)
        H_nonint[1, 0, 1, 1, :] = gamma_2_k  # px to py hopping
        H_nonint[1, 1, 1, 0, :] = gamma_2_k  # py to px hopping
        H_nonint[1, 0, 1, 2, :] = gamma_1_kx  # px to d hopping
        H_nonint[1, 2, 1, 0, :] = gamma_1_kx  # d to px hopping
        H_nonint[1, 1, 1, 2, :] = gamma_1_ky  # py to d hopping
        H_nonint[1, 2, 1, 1, :] = gamma_1_ky  # d to py hopping
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian (on-site energies).
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D[0], D[1], D[0], D[1], N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute order parameters
        n_p, eta = self.compute_order_parameters(exp_val)
        
        # Calculate the on-site energies
        xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - self.mu
        xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - self.mu
        xi_d = self.U_d_tilde * (self.n - n_p) / 2 - self.mu
        
        # Fill in the on-site energies for both spin up and down
        # Spin up
        H_int[0, 0, 0, 0, :] = xi_x  # px orbital
        H_int[0, 1, 0, 1, :] = xi_y  # py orbital
        H_int[0, 2, 0, 2, :] = xi_d  # d orbital
        
        # Spin down (same on-site energies as spin up)
        H_int[1, 0, 1, 0, :] = xi_x  # px orbital
        H_int[1, 1, 1, 1, :] = xi_y  # py orbital
        H_int[1, 2, 1, 2, :] = xi_d  # d orbital
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the Hamiltonian in flattened form.
            
        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total  # (s1, o1, s2, o2, k)
