from typing import Any
import numpy as np

from HF import *
import run_tests

#LLM Edits Start: Create the Hamiltonian Class for this system
class HartreeFockHamiltonian:
    def __init__(self, 
                 parameters: dict = None,
                 N_shell: int = 10,
                 Nq_shell: int = 1,
                 filling_factor: float = 0.5,
                 temperature: float = 0):
        """Initialize the Hartree-Fock Hamiltonian for twisted bilayer graphene.
        
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
            
        # Basic settings
        self.lattice = 'triangular'
        self.N_shell = N_shell
        self.nu = filling_factor
        self.T = temperature
        
        # Physical parameters
        self.hbar = parameters.get('hbar', 1.0)
        self.v_D = parameters.get('v_D', 1.0)  # Dirac velocity
        self.theta = parameters.get('theta', 1.0)  # Twist angle in degrees
        self.a_G = parameters.get('a_G', 0.246)  # Lattice constant of graphene in nm
        self.epsilon = parameters.get('epsilon', 0.8)  # Strain parameter
        
        # Tunneling parameters
        self.omega0 = parameters.get('omega0', 0.8)  # Interlayer tunneling strength
        self.omega1 = parameters.get('omega1', 0.9)  # Interlayer tunneling strength
        self.phi = parameters.get('phi', 0.0)  # Phase for tunneling
        
        # Interaction strengths
        self.V0 = parameters.get('V0', 1.0)  # Hartree interaction strength
        self.V1 = parameters.get('V1', 1.0)  # Fock interaction strength
        self.psi = parameters.get('psi', 0.0)  # Phase for interaction
        self.omega = np.exp(1j * 2 * np.pi / 3)  # Complex phase factor
        
        # Compute moire lattice constant
        self.a_M = self.a_G / np.sqrt(self.epsilon**2 + np.deg2rad(self.theta)**2)  # moire lattice constant, nm
        
        # Define k-space within first Brillouin zone (momentum space meshgrid)
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        # Define q-space for extended Brillouin zone (reciprocal lattice connecting different Gamma points)
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        
        # Define helper variables
        self.Nk = len(self.k_space)
        self.N_k = self.Nk  # Adding N_k for compatibility
        self.Nq = len(self.q)
        
        # Degree of freedom including the reciprocal lattice vectors
        # Each layer has 2 sublattices (A,B), and we have 2 layers
        self.D = (2, 2, self.Nq)  # (layer, sublattice, reciprocal lattice vector)
        
        # Define high symmetry points
        self.high_symm = generate_high_symmtry_points(self.lattice, self.a_M)
        # Define reciprocal lattice vectors for the moire pattern
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}
        
    def generate_kinetic(self, k=None) -> np.ndarray:
        """Generate kinetic energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Kinetic energy Hamiltonian
        """
        # Generate kinetic terms
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
            
        H_K = np.zeros((self.D + self.D + (self.Nk,)), dtype=complex)
        
        # Dirac Hamiltonian for each layer (±θ/2)
        for idx, q in enumerate(self.q):
            # Momentum measured from Dirac point for top layer (+θ/2)
            kx_top = kx + q[0] - self.high_symm["K"][0]
            ky_top = ky + q[1] - self.high_symm["K"][1]
            k_mag_top = np.sqrt(kx_top**2 + ky_top**2)
            theta_k_top = np.arctan2(ky_top, kx_top)
            
            # Momentum measured from Dirac point for bottom layer (-θ/2)
            kx_bot = kx + q[0] - self.high_symm["K"][0]
            ky_bot = ky + q[1] - self.high_symm["K"][1]
            k_mag_bot = np.sqrt(kx_bot**2 + ky_bot**2)
            theta_k_bot = np.arctan2(ky_bot, kx_bot)
            
            # Dirac Hamiltonian for top layer (+θ/2)
            # Off-diagonal elements for top layer
            H_K[0, 0, idx, 0, 1, idx, :] = -self.hbar * self.v_D * k_mag_top * np.exp(1j * (theta_k_top - self.theta/2))
            H_K[0, 1, idx, 0, 0, idx, :] = -self.hbar * self.v_D * k_mag_top * np.exp(-1j * (theta_k_top - self.theta/2))
            
            # Dirac Hamiltonian for bottom layer (-θ/2)
            # Off-diagonal elements for bottom layer
            H_K[1, 0, idx, 1, 1, idx, :] = -self.hbar * self.v_D * k_mag_bot * np.exp(1j * (theta_k_bot + self.theta/2))
            H_K[1, 1, idx, 1, 0, idx, :] = -self.hbar * self.v_D * k_mag_bot * np.exp(-1j * (theta_k_bot + self.theta/2))
            
        return H_K
    
    def H_V(self, k=None) -> np.ndarray:
        """Generate potential energy terms of the Hamiltonian (tunneling between layers).
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Potential energy Hamiltonian
        """
        # Generate potential terms (tunneling)
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
            
        H_M = np.zeros((self.D + self.D + (self.Nk,)), dtype=complex)
        
        # Assign V0 to diagonal elements
        for idx, q in enumerate(self.q):
            H_M[0, 0, idx, 0, 0, idx, :] = self.V0
            H_M[0, 1, idx, 0, 1, idx, :] = self.V0
            H_M[1, 0, idx, 1, 0, idx, :] = self.V0
            H_M[1, 1, idx, 1, 1, idx, :] = self.V0
        
        # Assign tunneling terms between layers
        for idx1, q1 in enumerate(self.q):
            for idx2, q2 in enumerate(self.q):
                diff_q = q1 - q2
                # Tunneling terms for j=0,1,2 (momentum boosts)
                for j in range(3):
                    # Compute T_j matrix elements
                    T_j_00 = self.omega0
                    T_j_01 = self.omega1 * np.cos(j * self.phi) + 1j * self.omega1 * np.sin(j * self.phi)
                    T_j_10 = self.omega1 * np.cos(j * self.phi) - 1j * self.omega1 * np.sin(j * self.phi)
                    T_j_11 = self.omega0
                    
                    # Check if diff_q matches a reciprocal lattice vector
                    if j == 0 and np.allclose(diff_q, np.zeros(2)):
                        # Tunneling from bottom to top layer
                        H_M[0, 0, idx1, 1, 0, idx2, :] = T_j_00
                        H_M[0, 0, idx1, 1, 1, idx2, :] = T_j_01
                        H_M[0, 1, idx1, 1, 0, idx2, :] = T_j_10
                        H_M[0, 1, idx1, 1, 1, idx2, :] = T_j_11
                        # Hermitian conjugate for tunneling from top to bottom
                        H_M[1, 0, idx2, 0, 0, idx1, :] = np.conj(T_j_00)
                        H_M[1, 1, idx2, 0, 0, idx1, :] = np.conj(T_j_01)
                        H_M[1, 0, idx2, 0, 1, idx1, :] = np.conj(T_j_10)
                        H_M[1, 1, idx2, 0, 1, idx1, :] = np.conj(T_j_11)
                    elif j == 1 and np.allclose(diff_q, self.g[1]):
                        # Tunneling with momentum boost b_1
                        H_M[0, 0, idx1, 1, 0, idx2, :] = T_j_00
                        H_M[0, 0, idx1, 1, 1, idx2, :] = T_j_01
                        H_M[0, 1, idx1, 1, 0, idx2, :] = T_j_10
                        H_M[0, 1, idx1, 1, 1, idx2, :] = T_j_11
                        # Hermitian conjugate
                        H_M[1, 0, idx2, 0, 0, idx1, :] = np.conj(T_j_00)
                        H_M[1, 1, idx2, 0, 0, idx1, :] = np.conj(T_j_01)
                        H_M[1, 0, idx2, 0, 1, idx1, :] = np.conj(T_j_10)
                        H_M[1, 1, idx2, 0, 1, idx1, :] = np.conj(T_j_11)
                    elif j == 2 and np.allclose(diff_q, self.g[2]):
                        # Tunneling with momentum boost b_2
                        H_M[0, 0, idx1, 1, 0, idx2, :] = T_j_00
                        H_M[0, 0, idx1, 1, 1, idx2, :] = T_j_01
                        H_M[0, 1, idx1, 1, 0, idx2, :] = T_j_10
                        H_M[0, 1, idx1, 1, 1, idx2, :] = T_j_11
                        # Hermitian conjugate
                        H_M[1, 0, idx2, 0, 0, idx1, :] = np.conj(T_j_00)
                        H_M[1, 1, idx2, 0, 0, idx1, :] = np.conj(T_j_01)
                        H_M[1, 0, idx2, 0, 1, idx1, :] = np.conj(T_j_10)
                        H_M[1, 1, idx2, 0, 1, idx1, :] = np.conj(T_j_11)
                        
        return H_M
        
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

        # Calculate density matrix fluctuation (relative to isolated layers at charge neutrality)
        delta_rho = exp_val.copy()
        
        # Compute Hartree term (direct interaction)
        # Σ^H_αG;βG'(k) = (1/A)∑_α' V_α'α(G'-G) δρ_α'α'(G-G') δ_αβ
        for layer1 in range(2):
            for sublattice1 in range(2):
                for idx1, q1 in enumerate(self.q):
                    for layer2 in range(2):
                        for sublattice2 in range(2):
                            for idx2, q2 in enumerate(self.q):
                                if layer1 == layer2 and sublattice1 == sublattice2:
                                    # Diagonal elements - Hartree term
                                    for other_layer in range(2):
                                        for other_sublattice in range(2):
                                            for other_idx, other_q in enumerate(self.q):
                                                # Calculate interaction potential
                                                V_hartree = self.V0 / self.Nk
                                                
                                                # Calculate density fluctuation for Hartree term
                                                delta_rho_hartree = np.mean(delta_rho[other_layer, other_sublattice, other_idx, 
                                                                                       other_layer, other_sublattice, other_idx, :])
                                                
                                                # Add Hartree term to diagonal element
                                                H_int[layer1, sublattice1, idx1, layer2, sublattice2, idx2, :] += V_hartree * delta_rho_hartree
        
        # Compute Fock term (exchange interaction)
        # Σ^F_αG;βG'(k) = -(1/A)∑_G'',k' V_αβ(G''+k'-k) δρ_α,G+G'';β,G'+G''(k')
        for layer1 in range(2):
            for sublattice1 in range(2):
                for idx1, q1 in enumerate(self.q):
                    for layer2 in range(2):
                        for sublattice2 in range(2):
                            for idx2, q2 in enumerate(self.q):
                                # Calculate Fock term (exchange interaction)
                                for idx_q, q_exchange in enumerate(self.q):
                                    # Calculate interaction potential for Fock term
                                    V_fock = self.V1 / self.Nk
                                    
                                    # Calculate density matrix for Fock term
                                    delta_rho_fock = np.mean(delta_rho[layer1, sublattice1, (idx1 + idx_q) % self.Nq, 
                                                                       layer2, sublattice2, (idx2 + idx_q) % self.Nq, :])
                                    
                                    # Add Fock term to Hamiltonian with negative sign
                                    H_int[layer1, sublattice1, idx1, layer2, sublattice2, idx2, :] -= V_fock * delta_rho_fock
                                    
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
    
    # Infinitesimal coupling - very small interaction strengths
    ham_infinitesimal_u = HartreeFockHamiltonian(
        N_shell=10,
        parameters={
            'V0': 0.01,  # Very small Hartree interaction
            'V1': 0.01,  # Very small Fock interaction
            'omega0': 1.0,
            'omega1': 1.0,
            'theta': 1.0,
            'a_G': 0.246,
            'v_D': 1.0,
            'hbar': 1.0,
            'epsilon': 0.8
        }
    )
    
    # Large coupling - strong interaction strengths
    ham_large_u = HartreeFockHamiltonian(
        N_shell=10,
        parameters={
            'V0': 10.0,  # Large Hartree interaction
            'V1': 10.0,  # Large Fock interaction
            'omega0': 1.0,
            'omega1': 1.0,
            'theta': 1.0,
            'a_G': 0.246,
            'v_D': 1.0,
            'hbar': 1.0,
            'epsilon': 0.8
        }
    )
    # LLM Edits End
        
    run_tests.generate_artifacts(ham, ham_infinitesimal_u, ham_large_u)
