import numpy as np
from typing import Dict, Any, Optional
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a 4-band model on a triangular lattice.
    
    The Hamiltonian consists of non-interacting 4x4 matrix elements
    and interacting terms from Hartree-Fock theory.
    """
    def __init__(self, 
                 parameters: Dict[str, Any] = {},
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
        
        # Hopping parameters for the non-interacting Hamiltonian
        self.gamma0 = parameters.get('gamma0', 1.0)  # Interlayer hopping
        self.gamma1 = parameters.get('gamma1', 0.4)  # Vertical hopping
        self.gamma3 = parameters.get('gamma3', 0.3)  # Trigonal warping
        self.gamma4 = parameters.get('gamma4', 0.1)  # Electron-hole asymmetry
        
        # Interaction parameters
        self.V0 = parameters.get('V0', 1.0)  # Direct interaction potential at q=0
        self.a = parameters.get('a', 1.0)  # Lattice constant in nm
        
        # Define lattice constant
        self.a_M = self.a  # Moiré lattice constant
        
        # Generate k-space within first Brillouin zone
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        # Generate q-space for extended Brillouin zone
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        
        # Helper values
        self.Nk = len(self.k_space)
        self.Nq = len(self.q)
        
        # Degree of freedom: 4 bands
        self.D = (4,)
        
        # Define high symmetry points
        self.high_symm = generate_high_symmetry_points(self.lattice, self.a_M)
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}
        
        # Calculate real-space unit cell area for normalization
        self.A = get_area(self.a_M, self.lattice)
    
    def f(self, k: np.ndarray) -> complex:
        """Calculate the complex function f(k) used in the Hamiltonian.
        
        Args:
            k: k-point coordinates [kx, ky]
            
        Returns:
            Complex value of f(k)
        """
        kx, ky = k
        term1 = np.exp(1j * ky * self.a / np.sqrt(3))
        term2 = 1 + 2 * np.exp(-1j * 3 * ky * self.a / (2 * np.sqrt(3))) * np.cos(kx * self.a / 2)
        return term1 * term2
    
    def generate_kinetic(self, k: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate kinetic energy terms of the Hamiltonian.
        This is not used in the current model but included for completeness.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Kinetic energy Hamiltonian (zeros in this model)
        """
        if k is None:
            k = self.k_space
        
        # This model doesn't separate kinetic terms explicitly
        return np.zeros((*self.D, *self.D, len(k)), dtype=complex)
    
    def H_V(self, k: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate potential energy terms of the Hamiltonian.
        This is not used in the current model but included for completeness.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Potential energy Hamiltonian (zeros in this model)
        """
        if k is None:
            k = self.k_space
            
        # This model doesn't separate potential terms explicitly
        return np.zeros((*self.D, *self.D, len(k)), dtype=complex)
    
    def generate_non_interacting(self, k: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian from the 4x4 matrix H_0.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Non-interacting Hamiltonian
        """
        if k is None:
            k = self.k_space
        
        Nk = len(k)
        H_0 = np.zeros((*self.D, *self.D, Nk), dtype=complex)
        
        for idx in range(Nk):
            f_val = self.f(k[idx])
            f_conj = np.conjugate(f_val)
            
            # Fill the 4x4 matrix based on the given Hamiltonian
            # First row
            H_0[0, 1, idx] = self.gamma0 * f_val
            H_0[0, 2, idx] = self.gamma4 * f_val
            H_0[0, 3, idx] = self.gamma3 * f_conj
            
            # Second row
            H_0[1, 0, idx] = self.gamma0 * f_conj
            H_0[1, 2, idx] = self.gamma1
            H_0[1, 3, idx] = self.gamma4 * f_val
            
            # Third row
            H_0[2, 0, idx] = self.gamma4 * f_conj
            H_0[2, 1, idx] = self.gamma1
            H_0[2, 3, idx] = self.gamma0 * f_val
            
            # Fourth row
            H_0[3, 0, idx] = self.gamma3 * f_val
            H_0[3, 1, idx] = self.gamma4 * f_conj
            H_0[3, 2, idx] = self.gamma0 * f_conj
        
        return H_0
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian.
        
        Implements the Hartree-Fock interaction terms:
        1. Direct terms: V(0)/A * <n_λ> contributes to H[λ',λ',:]
        2. Exchange terms: -V(k1-k2)/A * <c_λ^†c_λ'> contributes to H[λ',λ,:]
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        exp_val = unflatten(exp_val, self.D, self.Nk)
        H_int = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
        
        # Direct term (Hartree term)
        for lambda_idx in range(self.D[0]):
            n_lambda = np.mean(exp_val[lambda_idx, lambda_idx, :])  # Average occupation of band lambda
            for lambda_prime_idx in range(self.D[0]):
                # V(0)/A * <c_λ^†c_λ> contributes to H[λ',λ',:]
                H_int[lambda_prime_idx, lambda_prime_idx, :] += self.V0 / self.A * n_lambda
        
        # Exchange term (Fock term)
        # For simplicity, we approximate V(k1-k2) with V0 and handle the momentum dependence
        # through the expectation values directly
        for lambda_idx in range(self.D[0]):
            for lambda_prime_idx in range(self.D[0]):
                if lambda_idx != lambda_prime_idx:
                    for k_idx in range(self.Nk):
                        # -V(k1-k2)/A * <c_λ^†c_λ'> contributes to H[λ',λ,:]
                        # We use the mean expectation value as a simplification
                        H_int[lambda_prime_idx, lambda_idx, k_idx] -= self.V0 / self.A * np.mean(exp_val[lambda_idx, lambda_prime_idx, :])
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the total Hamiltonian including both non-interacting and interacting parts.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Total Hamiltonian (non-interacting + interacting)
        """
        H_non_int = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        
        # Total Hamiltonian is the sum of non-interacting and interacting parts
        H_total = H_non_int + H_int
        
        # Return flattened Hamiltonian for compatibility with solver
        return flattened(H_total, self.D, self.Nk)
