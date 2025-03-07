import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a three-orbital model (px, py, d) with spin.
    
    The Hamiltonian describes a system with px, py, and d orbitals on a square lattice
    with both on-site and inter-site interactions.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor for the system.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'Delta': 0.0, 't_pd': 1.0, 't_pp': 0.5, 'U_p': 1.0, 'V_pp': 0.5, 'V_pd': 0.5, 'U_d': 1.0, 'T': 0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'square'   # Lattice symmetry
        self.D = (2, 3)  # Number of flavors: (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital_flavor'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: orbital_flavor: px, py, d
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # Temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.Delta = parameters.get('Delta', 0.0)  # Energy difference between p and d orbitals
        self.t_pd = parameters.get('t_pd', 1.0)  # Hopping between p and d orbitals
        self.t_pp = parameters.get('t_pp', 0.5)  # Hopping between p orbitals
        self.U_p = parameters.get('U_p', 1.0)  # On-site interaction for p orbitals
        self.V_pp = parameters.get('V_pp', 0.5)  # Inter-site interaction between p orbitals
        self.V_pd = parameters.get('V_pd', 0.5)  # Inter-site interaction between p and d orbitals
        self.U_d = parameters.get('U_d', 1.0)  # On-site interaction for d orbitals
        
        # Derived interaction parameters
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
        
        # Loop over spin (interactions are diagonal in spin)
        for s in range(self.D[0]):
            # Diagonal elements (constant parts)
            H_nonint[s, 0, s, 0, :] = self.Delta  # ξ_x (constant part)
            H_nonint[s, 1, s, 1, :] = self.Delta  # ξ_y (constant part)
            # ξ_d (constant part) is 0.0, already initialized to zero
            
            # Off-diagonal elements
            for k_idx in range(self.N_k):
                kx, ky = self.k_space[k_idx]
                
                # γ_1(k_x) for p_x-d interaction
                gamma_1_kx = -2 * self.t_pd * np.cos(kx / 2)
                H_nonint[s, 0, s, 2, k_idx] = gamma_1_kx
                H_nonint[s, 2, s, 0, k_idx] = gamma_1_kx  # Hermitian conjugate
                
                # γ_1(k_y) for p_y-d interaction
                gamma_1_ky = -2 * self.t_pd * np.cos(ky / 2)
                H_nonint[s, 1, s, 2, k_idx] = gamma_1_ky
                H_nonint[s, 2, s, 1, k_idx] = gamma_1_ky  # Hermitian conjugate
                
                # γ_2(k) for p_x-p_y interaction
                gamma_2_k = -4 * self.t_pp * np.cos(kx / 2) * np.cos(ky / 2)
                H_nonint[s, 0, s, 1, k_idx] = gamma_2_k
                H_nonint[s, 1, s, 0, k_idx] = gamma_2_k  # Hermitian conjugate
        
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
        
        # Calculate expectation values
        n_p = 0.0  # Total density of holes on oxygen sites (px, py)
        eta = 0.0  # Nematic order parameter (px-py asymmetry)
        n = 0.0    # Total density of holes
        
        for s in range(self.D[0]):
            # Sum over all k points for each expectation value
            n_x = np.mean(exp_val[s, 0, s, 0, :]).real
            n_y = np.mean(exp_val[s, 1, s, 1, :]).real
            n_d = np.mean(exp_val[s, 2, s, 2, :]).real
            
            n_p += n_x + n_y  # Sum px and py densities for both spins
            eta += n_x - n_y  # Difference between px and py densities
            n += n_x + n_y + n_d  # Total density including d orbitals
        
        # Calculate chemical potential based on total density
        mu = 2 * self.V_pd * n - self.V_pd * n * n
        
        # Loop over spin (interactions are diagonal in spin)
        for s in range(self.D[0]):
            # Diagonal elements (interaction parts)
            xi_x = self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu
            xi_y = self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu
            xi_d = self.U_d_tilde * (n - n_p) / 2 - mu
            
            H_int[s, 0, s, 0, :] = xi_x  # px-px interaction
            H_int[s, 1, s, 1, :] = xi_y  # py-py interaction
            H_int[s, 2, s, 2, :] = xi_d  # d-d interaction
        
        # Add the constant term f(n^p, η)/N to the total energy
        # This is a constant energy shift that doesn't affect the eigenstates
        f_term = -self.U_p_tilde * n_p**2 / 8 + self.V_pp_tilde * eta**2 / 8 - self.U_d_tilde * (n - n_p)**2 / 4
        
        # Distribute the constant energy shift equally among all diagonal elements
        for s in range(self.D[0]):
            for o in range(self.D[1]):
                H_int[s, o, s, o, :] += f_term / (self.D[0] * self.D[1])
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns the flattened Hamiltonian.

        Returns:
            np.ndarray: The total Hamiltonian with shape (D_flattened, D_flattened, N_k) if return_flat=True,
                        else with shape (D, D, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
