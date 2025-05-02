from typing import Any
import numpy as np

from HF import *
import run_tests

#LLM Edits Start: Create the Hamiltonian Class for this system
class HartreeFockHamiltonian:
    """Implementation of the Hartree-Fock Hamiltonian for a triangular lattice system.
    
    This class implements the Hamiltonian:
    H_0 = [
        [0, γ0·f, γ4·f, γ3·f*],
        [γ0·f*, 0, γ1, γ4·f],
        [γ4·f*, γ1, 0, γ0·f],
        [γ3·f, γ4·f*, γ0·f*, 0]
    ]
    
    with interaction terms:
    V^HF_quadratic = 1/A·∑_λλ'k1k2 ⟨c†_k1λ c_k1λ⟩·c†_k2λ' c_k2λ'·V(0) - 
                     1/A·∑_λλ'k1k2 ⟨c†_k1λ c_k1λ'⟩·c†_k2λ' c_k2λ·V(k1-k2)
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
        # Set default parameters if none provided
        if parameters is None:
            parameters = {}
            
        self.lattice = 'triangular'
        
        self.N_shell = N_shell
        self.nu = filling_factor
        self.T = temperature
        
        # Model parameters (γ values) with defaults
        self.gamma0 = parameters.get('gamma0', 3.0)  # Nearest-neighbor hopping
        self.gamma1 = parameters.get('gamma1', 0.4)  # Interlayer hopping
        self.gamma3 = parameters.get('gamma3', 0.3)  # Trigonal warping
        self.gamma4 = parameters.get('gamma4', 0.15)  # Electron-hole asymmetry
        
        # Interaction parameters
        self.V0 = parameters.get('V0', 1.0)  # On-site interaction strength
        self.a = parameters.get('a', 1.0)  # Lattice constant (in appropriate units)
        
        # Define lattice constant for moiré pattern
        # For this simple model, we'll use the direct lattice constant
        self.a_M = self.a
        
        # define k-space within first Brillouin zone (momentum space meshgrid)
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        # define q-space for extended Brillouin zone (reciprocal lattice connecting different Gamma points)
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        
        # define helper functions
        self.Nk = len(self.k_space)
        self.N_k = self.Nk  # Alias for compatibility
        self.Nq = len(self.q)
        
        # Degree of freedom: 4 orbitals in this model
        self.D = (4,)
        
        # Define high symmetry points
        self.high_symm = generate_high_symmtry_points(self.lattice, self.a_M)
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}
        
        # Calculate area of unit cell (for normalization in interaction terms)
        self.area = get_area(self.a_M, self.lattice)
        
    def compute_f(self, k=None):
        """Compute the complex function f(k) used in the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Complex values of f(k) for each k-point
        """
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
            
        # f(k) = e^(i k_y a/√3) * (1 + 2e^(-i 3k_y a/2√3) * cos(k_x a/2))
        term1 = np.exp(1j * ky * self.a / np.sqrt(3))
        term2 = 1 + 2 * np.exp(-1j * 3 * ky * self.a / (2 * np.sqrt(3))) * np.cos(kx * self.a / 2)
        
        return term1 * term2
        
    def generate_non_interacting(self, k=None) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Non-interacting Hamiltonian (kinetic + potential)
        """
        if k is None:
            N_k = self.Nk
        else:
            N_k = len(k)
            
        # Initialize Hamiltonian matrix
        H0 = np.zeros((*self.D, *self.D, N_k), dtype=complex)
        
        # Compute f(k) for all k-points
        f_k = self.compute_f(k)
        f_k_conj = np.conjugate(f_k)
        
        # Populate Hamiltonian matrix based on the equation in the prompt
        # [0, γ0·f, γ4·f, γ3·f*]
        # [γ0·f*, 0, γ1, γ4·f]
        # [γ4·f*, γ1, 0, γ0·f]
        # [γ3·f, γ4·f*, γ0·f*, 0]
        
        # First row
        H0[0, 1, :] = self.gamma0 * f_k
        H0[0, 2, :] = self.gamma4 * f_k
        H0[0, 3, :] = self.gamma3 * f_k_conj
        
        # Second row
        H0[1, 0, :] = self.gamma0 * f_k_conj
        H0[1, 2, :] = self.gamma1
        H0[1, 3, :] = self.gamma4 * f_k
        
        # Third row
        H0[2, 0, :] = self.gamma4 * f_k_conj
        H0[2, 1, :] = self.gamma1
        H0[2, 3, :] = self.gamma0 * f_k
        
        # Fourth row
        H0[3, 0, :] = self.gamma3 * f_k
        H0[3, 1, :] = self.gamma4 * f_k_conj
        H0[3, 2, :] = self.gamma0 * f_k_conj
        
        return H0
        
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        exp_val = unflatten(exp_val, self.D, self.Nk)
        H_int = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
        
        # Implementation of the interacting Hamiltonian:
        # V^HF_quadratic = 1/A·∑_λλ'k1k2 ⟨c†_k1λ c_k1λ⟩·c†_k2λ' c_k2λ'·V(0) - 
        #                  1/A·∑_λλ'k1k2 ⟨c†_k1λ c_k1λ'⟩·c†_k2λ' c_k2λ·V(k1-k2)
        
        # First term (direct/Hartree term): ⟨c†_k1λ c_k1λ⟩·c†_k2λ' c_k2λ'·V(0)/A
        # Calculate the average occupation for each orbital
        n_orbitals = np.zeros(4)
        for orb in range(4):
            n_orbitals[orb] = np.mean(np.real(exp_val[orb, orb, :]))
        
        # Apply direct interaction for each orbital pair
        for orb1 in range(4):
            for orb2 in range(4):
                if orb1 != orb2:  # Off-diagonal terms
                    H_int[orb1, orb1, :] += n_orbitals[orb2] * self.V0 / self.area
        
        # Second term (exchange/Fock term): -⟨c†_k1λ c_k1λ'⟩·c†_k2λ' c_k2λ·V(k1-k2)/A
        # For simplicity, we'll use a momentum-independent V(k1-k2) = V0
        for orb1 in range(4):
            for orb2 in range(4):
                if orb1 != orb2:  # Off-diagonal terms
                    # Mean expectation value across k-points for this orbital pair
                    exchange_avg = np.mean(exp_val[orb1, orb2, :])
                    # Apply exchange interaction
                    H_int[orb2, orb1, :] -= exchange_avg * self.V0 / self.area
        
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
# LLM Edits end

if __name__=='__main__':
    # LLM Edits Start: Instantiate Hamiltonian in different limits    
    # Default parameters
    ham = HartreeFockHamiltonian(N_shell=10)
    
    # Infinitesimal coupling limit (very small interaction strength)
    ham_infinitesimal_u = HartreeFockHamiltonian(
        N_shell=10,
        parameters={'gamma0': 3.0, 'gamma1': 0.4, 'gamma3': 0.3, 'gamma4': 0.15, 'V0': 0.01}
    )
    
    # Large coupling limit (strong interaction strength)
    ham_large_u = HartreeFockHamiltonian(
        N_shell=10,
        parameters={'gamma0': 3.0, 'gamma1': 0.4, 'gamma3': 0.3, 'gamma4': 0.15, 'V0': 10.0}
    )
    # LLM Edits End
        
    run_tests.generate_artifacts(ham, ham_infinitesimal_u, ham_large_u)