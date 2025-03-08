import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for the Emery model describing copper-oxide planes in high-Tc superconductors.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor for the system. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_pd': 1.0, 't_pp': 0.5, 'Delta': 3.0, 'U_p': 4.0, 'V_pp': 1.0, 'V_pd': 1.0, 'U_d': 8.0, 'n': 1.0, 'T': 0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'square'  # Lattice symmetry
        self.D = (2, 3)  # Number of flavors (spin, orbital)
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
        self.t_pd = parameters.get('t_pd', 1.0)  # Hopping between p and d orbitals
        self.t_pp = parameters.get('t_pp', 0.5)  # Hopping between p orbitals
        self.Delta = parameters.get('Delta', 3.0)  # Energy difference between p and d orbitals
        self.U_p = parameters.get('U_p', 4.0)  # On-site Coulomb repulsion for p orbitals
        self.V_pp = parameters.get('V_pp', 1.0)  # Coulomb repulsion between neighboring p orbitals
        self.V_pd = parameters.get('V_pd', 1.0)  # Coulomb repulsion between p and d orbitals
        self.U_d = parameters.get('U_d', 8.0)  # On-site Coulomb repulsion for d orbitals
        self.n = parameters.get('n', 1.0)  # Total density of holes
        
        # Effective interaction parameters
        self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd  # Effective U_p
        self.V_pp_tilde = 8 * self.V_pp - self.U_p  # Effective V_pp
        self.U_d_tilde = self.U_d - 4 * self.V_pd  # Effective U_d
        
        # Chemical potential
        self.mu = 2 * self.V_pd * self.n - self.V_pd * self.n**2  # Chemical potential
        
        return
        
    def compute_order_parameters(self, exp_val: np.ndarray):
        """Compute order parameters from expectation values.
        
        Args:
            exp_val (np.ndarray): Expectation value tensor.
            
        Returns:
            tuple: Tuple containing n_p (total density of holes on oxygen sites) and eta (nematic order parameter).
        """
        # Unpack spin and orbital indices
        spin_up, spin_down = 0, 1
        p_x, p_y, d = 0, 1, 2
        
        # Compute n_p: total density of holes on oxygen sites (p_x and p_y)
        n_p_x_up = np.mean(exp_val[spin_up, p_x, spin_up, p_x, :])
        n_p_x_down = np.mean(exp_val[spin_down, p_x, spin_down, p_x, :])
        n_p_y_up = np.mean(exp_val[spin_up, p_y, spin_up, p_y, :])
        n_p_y_down = np.mean(exp_val[spin_down, p_y, spin_down, p_y, :])
        
        n_p = (n_p_x_up + n_p_x_down) + (n_p_y_up + n_p_y_down)
        
        # Compute eta: nematic order parameter (difference between p_x and p_y hole densities)
        eta = (n_p_x_up + n_p_x_down) - (n_p_y_up + n_p_y_down)
        
        return n_p, eta
        
    def compute_energy_constant(self, n_p, eta):
        """Compute the constant energy term f(n^p, eta) / N.
        
        This term is part of the total energy calculation but is not included
        in the Hamiltonian matrix elements.
        
        Args:
            n_p (float): Total density of holes on oxygen sites.
            eta (float): Nematic order parameter.
            
        Returns:
            float: Constant energy term per k-point.
        """
        f_const = -self.U_p_tilde * (n_p**2) / 8 + self.V_pp_tilde * (eta**2) / 8 - self.U_d_tilde * (self.n - n_p)**2 / 4
        
        return f_const / self.N_k  # Distribute the constant term over all k points
        
    def generate_non_interacting(self) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: Non-interacting Hamiltonian tensor.
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Unpack spin and orbital indices
        spin_up, spin_down = 0, 1
        p_x, p_y, d = 0, 1, 2
        
        for s in [spin_up, spin_down]:
            # Gamma_1 terms (hopping between p and d orbitals)
            gamma_1_x = -2 * self.t_pd * np.cos(self.k_space[:, 0] / 2)
            gamma_1_y = -2 * self.t_pd * np.cos(self.k_space[:, 1] / 2)
            
            # Gamma_2 term (hopping between p_x and p_y orbitals)
            gamma_2 = -4 * self.t_pp * np.cos(self.k_space[:, 0] / 2) * np.cos(self.k_space[:, 1] / 2)
            
            # Off-diagonal elements of the Hamiltonian (within the same spin block)
            H_nonint[s, p_x, s, p_y, :] = gamma_2  # Hopping between p_x and p_y orbitals
            H_nonint[s, p_y, s, p_x, :] = gamma_2  # Hermitian conjugate
            H_nonint[s, p_x, s, d, :] = gamma_1_x  # Hopping between p_x and d orbitals
            H_nonint[s, d, s, p_x, :] = gamma_1_x  # Hermitian conjugate
            H_nonint[s, p_y, s, d, :] = gamma_1_y  # Hopping between p_y and d orbitals
            H_nonint[s, d, s, p_y, :] = gamma_1_y  # Hermitian conjugate
        
        return H_nonint
        
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value tensor.
            
        Returns:
            np.ndarray: Interacting Hamiltonian tensor.
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute order parameters
        n_p, eta = self.compute_order_parameters(exp_val)
        
        # Unpack spin and orbital indices
        spin_up, spin_down = 0, 1
        p_x, p_y, d = 0, 1, 2
        
        for s in [spin_up, spin_down]:
            # Diagonal elements of the Hamiltonian (within the same spin block)
            # p_x orbital energy with interaction effects
            H_int[s, p_x, s, p_x, :] = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - self.mu
            
            # p_y orbital energy with interaction effects
            H_int[s, p_y, s, p_y, :] = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - self.mu
            
            # d orbital energy with interaction effects
            H_int[s, d, s, d, :] = self.U_d_tilde * (self.n - n_p) / 2 - self.mu
        
        return H_int
        
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """Generate the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value tensor.
            return_flat (bool): Whether to return the flattened Hamiltonian.
            
        Returns:
            np.ndarray: Total Hamiltonian tensor.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
