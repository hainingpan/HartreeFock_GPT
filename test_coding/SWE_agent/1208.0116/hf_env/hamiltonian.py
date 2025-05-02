from typing import Any
import numpy as np

from HF import *
import run_tests

#LLM Edits Start: Create the Hamiltonian Class for this system
class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a triangular lattice with 6x6 matrix.
    
    The Hamiltonian includes a non-interacting part H_0 and an interacting
    part V_HF implementing Hartree-Fock approximation with Hartree and 
    Exchange Coulomb interactions.
    
    Args:
        parameters: Dictionary containing model parameters
        N_shell: Number of shells in k-space
        Nq_shell: Number of shells in q-space
        filling_factor: Electron filling factor
        temperature: Temperature for Fermi-Dirac distribution
    """
    def __init__(self, 
                 parameters: dict = None,
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
        if parameters is None:
            parameters = {}
            
        self.lattice = 'triangular'
        
        self.N_shell = N_shell
        self.nu = filling_factor
        self.T = temperature
        
        # Lattice parameters
        self.a = parameters.get('a', 2.46)  # Lattice constant in Angstrom
        
        # Hopping parameters
        self.gamma0 = parameters.get('gamma0', 3.0)  # Nearest-neighbor hopping
        self.gamma1 = parameters.get('gamma1', 0.4)  # Interlayer hopping
        self.gamma2 = parameters.get('gamma2', 0.01)  # Next-nearest interlayer hopping
        self.gamma3 = parameters.get('gamma3', 0.3)  # Trigonal warping
        self.gammaN = parameters.get('gammaN', 0.0)  # Bias term
        
        # Interaction parameters
        self.U_H = parameters.get('U_H', 1.0)  # Hartree interaction strength
        self.U_X = parameters.get('U_X', 1.0)  # Exchange interaction strength
        
        # define lattice constant
        self.a_M = self.a  # Using the provided lattice constant
        
        # define k-space within first Brillouin zone (momentum space meshgrid)
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        # define q-space for extended Brillouin zone (reciprocal lattice connecting different Gamma points)
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        
        # define helper functions
        self.Nk = len(self.k_space)
        self.N_k = self.Nk  # Alias for compatibility with function calls
        self.Nq = len(self.q)
        
        # degree of freedom including the reciprocal lattice vectors
        # This is the dimension of the Hamiltonian
        self.D = (6,)
        
        # define high symmetry points
        self.high_symm = generate_high_symmtry_points(self.lattice, self.a_M)
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}
        
        # Area of unit cell (for Coulomb interactions)
        self.A = get_area(self.a_M, self.lattice)
    
    def _calculate_f(self, k=None):
        """Calculate the f(k) function in the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Complex array with f(k) values
        """
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
        
        # Calculate f(k) = e^(i k_y a/√3) * (1 + 2e^(-i 3k_y a/2√3) * cos(k_x a/2))
        term1 = np.exp(1j * ky * self.a_M / np.sqrt(3))
        term2 = 1 + 2 * np.exp(-1j * 3 * ky * self.a_M / (2 * np.sqrt(3))) * np.cos(kx * self.a_M / 2)
        
        return term1 * term2
    
    def generate_non_interacting(self, k=None) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Non-interacting Hamiltonian
        """
        if k is None:
            N_k = self.Nk
        else:
            N_k = len(k)
            
        # Initialize the Hamiltonian matrix
        H_0 = np.zeros((self.D[0], self.D[0], N_k), dtype=complex)
        
        # Calculate f(k)
        f = self._calculate_f(k)
        
        # Fill the Hamiltonian matrix according to the given structure
        # Note: The matrix indices are 0-5 instead of 1-6 as in the equation
        
        # Row 0
        H_0[0, 1, :] = -self.gamma0 * f
        H_0[0, 3, :] = -self.gamma3 * f.conj() - self.gammaN
        H_0[0, 5, :] = -self.gamma2
        
        # Row 1
        H_0[1, 0, :] = -self.gamma0 * f.conj()
        H_0[1, 2, :] = -self.gamma1
        
        # Row 2
        H_0[2, 1, :] = -self.gamma1
        H_0[2, 3, :] = -self.gamma0 * f
        H_0[2, 5, :] = -self.gamma3 * f.conj()
        
        # Row 3
        H_0[3, 0, :] = -self.gamma3 * f - self.gammaN.conjugate() if isinstance(self.gammaN, complex) else -self.gamma3 * f - self.gammaN
        H_0[3, 2, :] = -self.gamma0 * f.conj()
        H_0[3, 4, :] = -self.gamma1
        
        # Row 4
        H_0[4, 3, :] = -self.gamma1
        H_0[4, 5, :] = -self.gamma0 * f
        
        # Row 5
        H_0[5, 0, :] = -self.gamma2
        H_0[5, 2, :] = -self.gamma3 * f
        H_0[5, 4, :] = -self.gamma0 * f.conj()
        
        return H_0
    
    def generate_kinetic(self, k=None) -> np.ndarray:
        """Generate kinetic energy terms of the Hamiltonian.
        
        This is identical to generate_non_interacting for this model.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Kinetic energy Hamiltonian
        """
        return self.generate_non_interacting(k)
    
    def H_V(self, k=None) -> np.ndarray:
        """Generate potential energy terms of the Hamiltonian.
        
        For this model, there are no separate potential terms in the non-interacting part.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Zero array with correct shape
        """
        if k is None:
            N_k = self.Nk
        else:
            N_k = len(k)
            
        # Return a zero matrix since all terms are included in generate_non_interacting
        return np.zeros((self.D[0], self.D[0], N_k), dtype=complex)
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        # Initialize the interaction Hamiltonian
        H_int = np.zeros((self.D[0], self.D[0], self.Nk), dtype=complex)
        
        # Calculate mean occupations for all orbitals
        # The exp_val tensor has shape (D_flattened, D_flattened, N_k)
        n_lambda = np.zeros(self.D[0])
        for lambda_idx in range(self.D[0]):
            # Average occupation for each orbital across all k-points
            n_lambda[lambda_idx] = np.mean(np.real(exp_val[lambda_idx, lambda_idx, :]))
        
        # Calculate Hartree term contributions
        for lambda_idx in range(self.D[0]):
            # Apply Hartree terms to diagonal elements
            for lambda_prime in range(self.D[0]):
                # U_H^{lambda lambda'} * <c^dag_{k' lambda'} c_{k' lambda'}>
                H_int[lambda_idx, lambda_idx, :] += self.U_H * n_lambda[lambda_prime]
        
        # Calculate Exchange term contributions
        # This is a simplification for the momentum dependence of U_X
        for k_idx in range(self.Nk):
            for lambda_idx in range(self.D[0]):
                for lambda_prime in range(self.D[0]):
                    # -U_X^{lambda lambda'}(k'-k) * <c^dag_{k' lambda'} c_{k' lambda}>
                    H_int[lambda_idx, lambda_prime, k_idx] -= self.U_X * exp_val[lambda_prime, lambda_idx, k_idx]
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the total Hamiltonian including both non-interacting and interacting parts.
        
        This function is required by the solve() function in HF.py.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Total Hamiltonian (non-interacting + interacting)
        """
        H_non_int = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        
        return H_non_int + H_int
# LLM Edits end

if __name__=='__main__':
    # LLM Edits Start: Instantiate Hamiltonian in different limits    
    # Default parameters
    ham = HartreeFockHamiltonian(N_shell=10)
    
    # Infinitesimal coupling limit - very small interaction parameters
    ham_infinitesimal_u = HartreeFockHamiltonian(
        parameters={
            'a': 2.46,
            'gamma0': 3.0,
            'gamma1': 0.4,
            'gamma2': 0.01,
            'gamma3': 0.3,
            'gammaN': 0.0,
            'U_H': 0.01,  # Very small Hartree interaction
            'U_X': 0.01   # Very small Exchange interaction
        },
        N_shell=10
    )
    
    # Large coupling limit - strong interaction parameters
    ham_large_u = HartreeFockHamiltonian(
        parameters={
            'a': 2.46,
            'gamma0': 3.0,
            'gamma1': 0.4,
            'gamma2': 0.01,
            'gamma3': 0.3,
            'gammaN': 0.0,
            'U_H': 10.0,  # Large Hartree interaction
            'U_X': 10.0   # Large Exchange interaction
        },
        N_shell=10
    )
    # LLM Edits End
        
    run_tests.generate_artifacts(ham, ham_infinitesimal_u, ham_large_u)