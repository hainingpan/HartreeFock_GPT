import numpy as np
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a 6-flavor triangular lattice model."""
    
    def __init__(self, 
                 parameters: dict = {
                     'gamma0': 1.0, 'gamma1': 0.1, 'gamma2': 0.05, 
                     'gamma3': 0.08, 'gammaN': 0.02, 
                     'U_H': 0.5, 'U_X': 0.3
                 },
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
        
        # Model parameters
        self.gamma0 = parameters.get('gamma0', 1.0)  # Hopping parameter
        self.gamma1 = parameters.get('gamma1', 0.1)  # Interlayer coupling
        self.gamma2 = parameters.get('gamma2', 0.05) # Secondary coupling
        self.gamma3 = parameters.get('gamma3', 0.08) # Trigonal warping
        self.gammaN = parameters.get('gammaN', 0.02) # Asymmetry parameter
        
        # Interaction parameters
        self.U_H = parameters.get('U_H', 0.5)  # Hartree interaction strength
        self.U_X = parameters.get('U_X', 0.3)  # Exchange interaction strength
        
        # Lattice constant (in Angstroms)
        self.a = 2.46
        
        # Generate k-space within first Brillouin zone
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        # Generate q-space for extended Brillouin zone
        self.q_index, self.q = get_q(Nq_shell, self.a)
        
        # Define dimensions
        self.Nk = len(self.k_space)
        self.Nq = len(self.q)
        
        # Degree of freedom - 6 flavors
        self.D = (6,)
        
        # Define high symmetry points
        self.high_symm = generate_high_symmetry_points(self.lattice, self.a)

    def calculate_f(self, k=None):
        """Calculate the complex function f(k) for the given k-points.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Complex values of f(k)
        """
        if k is None:
            kx, ky = self.k_space[:, 0], self.k_space[:, 1]
        else:
            kx, ky = k[:, 0], k[:, 1]
        
        a = self.a
        # f(k) = e^(i k_y a / √3) * (1 + 2 * e^(-i 3 k_y a / 2√3) * cos(k_x a / 2))
        term1 = np.exp(1j * ky * a / np.sqrt(3))
        term2 = 1 + 2 * np.exp(-1j * 3 * ky * a / (2 * np.sqrt(3))) * np.cos(kx * a / 2)
        return term1 * term2

    def generate_non_interacting(self, k=None) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Non-interacting Hamiltonian
        """
        if k is None:
            k = self.k_space
        
        # Calculate f(k) for the given k-points
        f_k = self.calculate_f(k)
        f_k_conj = np.conjugate(f_k)
        
        # Initialize Hamiltonian
        H_0 = np.zeros((*self.D, *self.D, len(k)), dtype=complex)
        
        # Fill in the matrix based on the provided Hamiltonian structure
        # First row
        H_0[0, 1, :] = self.gamma0 * f_k
        H_0[0, 3, :] = self.gamma3 * f_k_conj + self.gammaN
        H_0[0, 5, :] = self.gamma2
        
        # Second row
        H_0[1, 0, :] = self.gamma0 * f_k_conj
        H_0[1, 2, :] = self.gamma1
        
        # Third row
        H_0[2, 1, :] = self.gamma1
        H_0[2, 3, :] = self.gamma0 * f_k
        H_0[2, 5, :] = self.gamma3 * f_k_conj
        
        # Fourth row
        H_0[3, 0, :] = self.gamma3 * f_k + np.conjugate(self.gammaN)
        H_0[3, 2, :] = self.gamma0 * f_k_conj
        H_0[3, 4, :] = self.gamma1
        
        # Fifth row
        H_0[4, 3, :] = self.gamma1
        H_0[4, 5, :] = self.gamma0 * f_k
        
        # Sixth row
        H_0[5, 0, :] = self.gamma2
        H_0[5, 2, :] = self.gamma3 * f_k
        H_0[5, 4, :] = self.gamma0 * f_k_conj
        
        return H_0

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        exp_val = unflatten(exp_val, self.D, self.Nk)
        H_int = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
        
        # Calculate Hartree terms
        # U_H^{λλ'} [Σ_{k'} <c^†_{k',λ'} c_{k',λ'}>] c^†_{k,λ} c_{k,λ}
        for lambda_prime in range(6):
            # Calculate average density for each flavor λ'
            n_lambda_prime = np.mean(exp_val[lambda_prime, lambda_prime, :])
            
            # Add Hartree contribution to each flavor λ
            for lambda_val in range(6):
                H_int[lambda_val, lambda_val, :] += self.U_H * n_lambda_prime
        
        # Calculate Exchange terms
        # -U_X^{λλ'} <c^†_{k',λ'} c_{k',λ}> c^†_{k,λ} c_{k,λ'}
        for lambda_prime in range(6):
            for lambda_val in range(6):
                # Use the expectation value directly without momentum dependence for simplicity
                # In a more detailed model, this would depend on (k'-k)
                H_int[lambda_val, lambda_prime, :] -= self.U_X * exp_val[lambda_prime, lambda_val, :]
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the total Hamiltonian including both non-interacting and interacting parts.
        
        This function is required by the solve() function in HF.py.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Total Hamiltonian (non-interacting + interacting)
        """
        H_0 = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_0 + H_int
        
        # Return flattened Hamiltonian
        return flattened(H_total, self.D, self.Nk)
