import numpy as np
from typing import Dict, Any, Tuple
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a 2D triangular lattice with spin degrees of freedom.
    
    Implements a Hamiltonian with:
    - Kinetic energy terms E_s(k) = sum_n t_s(n) * e^(-i k·n)
    - Hartree interaction term U(0) * <n_s> * n_s'
    - Fock (exchange) interaction term -U(k2-k1) * <c_k1,s^† c_k1,s'> * c_k2,s'^† c_k2,s
    """
    def __init__(self, 
                 parameters: dict = {},
                 N_shell: int = 10,
                 Nq_shell: int = 1,
                 filling_factor: float = 0.5,
                 temperature: float = 0):
        """Initialize the Hartree-Fock Hamiltonian.
        
        Args:
            parameters: Dictionary containing model parameters
            N_shell: Number of shells in k-space
            Nq_shell: Number of shells in q-space
            filling_factor: Electron filling factor
            temperature: Temperature for Fermi-Dirac distribution
        """
        self.lattice = 'triangular'
        
        self.N_shell = N_shell
        self.nu = filling_factor
        self.T = temperature
        
        # define parameters
        # Hopping parameters for different lattice vectors
        self.ts = parameters.get('ts', {(1,0): 1.0, (0,1): 1.0, (1,1): 0.5})  
        # On-site Coulomb interaction
        self.U0 = parameters.get('U0', 1.0)  
        # Characteristic momentum scale for U(k)
        self.k0 = parameters.get('k0', 1.0)  
        
        # define lattice constant
        self.a_G = parameters.get('a_G', 0.246)  # graphene lattice constant, nm
        self.epsilon = parameters.get('epsilon', 0.0)  # strain
        self.theta = parameters.get('theta', 1.0)  # twist angle, degrees
        self.a_M = self.a_G/np.sqrt(self.epsilon**2 + np.deg2rad(self.theta)**2)  # moire lattice constant, nm

        # define k-space within first Brillouin zone (momentum space meshgrid)
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        # define q-space for extended Brillouin zone (reciprocal lattice connnecting different Gamma points)
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)

        # define helper functions
        self.Nk = len(self.k_space)
        self.Nq = len(self.q)

        # degree of freedom including the reciprocal lattice vectors
        self.D = (2,)  # (spin)

        # define high symmetry points
        self.high_symm = generate_high_symmetry_points(self.lattice, self.a_M)
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}
    
    def U_k_func(self, k_diff):
        """Calculate the momentum-dependent Coulomb interaction U(k).
        
        Args:
            k_diff: Momentum difference k2-k1
            
        Returns:
            float: Value of U(k_diff)
        """
        # Simple model: U(k) = U0 * exp(-k^2/k0^2)
        return self.U0 * np.exp(-np.sum(k_diff**2) / self.k0**2)
    
    def generate_kinetic(self, k=None) -> np.ndarray:
        """Generate kinetic energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Kinetic energy Hamiltonian
        """
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
        
        H_K = np.zeros(self.D + self.D + (self.Nk,), dtype=complex)
        
        # Calculate E_s(k) for each spin and k-point
        for s in range(2):  # spin index
            for k_idx in range(self.Nk):
                k_point = np.array([kx[k_idx], ky[k_idx]])
                E_k = 0
                for n, t_value in self.ts.items():
                    E_k += t_value * np.exp(-1j * (k_point[0] * n[0] + k_point[1] * n[1]))
                H_K[s, s, k_idx] = -E_k  # Negative sign as per the Hamiltonian
        
        return H_K
    
    def H_V(self, k=None) -> np.ndarray:
        """Generate potential energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Potential energy Hamiltonian
        """
        # In this model, there are no explicit potential terms in the non-interacting part
        if k is None:
            k = self.k_space
        return np.zeros(self.D + self.D + (len(k),), dtype=complex)
    
    def generate_non_interacting(self, k=None) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Non-interacting Hamiltonian (kinetic + potential)
        """
        return self.generate_kinetic(k) + self.H_V(k)
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        exp_val = unflatten(exp_val, self.D, self.Nk)
        H_int = np.zeros(self.D + self.D + (self.Nk,), dtype=complex)
        
        # Hartree term: U(0) * <n_s> * n_s'
        for s in range(2):  # spin s
            # Calculate average density for spin s across all k
            n_s = np.mean(exp_val[s, s, :])
            for s_prime in range(2):  # spin s'
                # Add contribution to the diagonal elements for spin s'
                H_int[s_prime, s_prime, :] += (1.0 / self.Nk) * self.U0 * n_s
        
        # Fock term: -U(k2-k1) * <c_k1,s^† c_k1,s'> * c_k2,s'^† c_k2,s
        for s in range(2):  # spin s
            for s_prime in range(2):  # spin s'
                for k1_idx in range(self.Nk):
                    k1 = self.k_space[k1_idx]
                    for k2_idx in range(self.Nk):
                        k2 = self.k_space[k2_idx]
                        # Calculate momentum-dependent interaction U(k2-k1)
                        U_k = self.U_k_func(k2 - k1)
                        # Add the Fock term (exchange interaction)
                        H_int[s_prime, s, k2_idx] -= (1.0 / self.Nk) * U_k * exp_val[s, s_prime, k1_idx]
        
        return H_int
        
    def generate_Htotal(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the total Hamiltonian including both non-interacting and interacting parts.
        
        This function is required by the solve() function in HF.py.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Total Hamiltonian (non-interacting + interacting)
        """
        return self.generate_non_interacting() + self.generate_interacting(exp_val)
