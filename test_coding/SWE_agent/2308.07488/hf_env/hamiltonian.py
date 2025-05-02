from typing import Any
import numpy as np

from HF import *
import run_tests

#LLM Edits Start: Create the Hamiltonian Class for this system
class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a twisted bilayer system with Coulomb interaction.
    
    This implementation is based on the continuum model with moiré potential and
    Coulomb interaction as described in the references.
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
        # Default parameters if none provided
        if parameters is None:
            parameters = {}
            
        # Set lattice type
        self.lattice = 'triangular'
        
        # Basic parameters
        self.N_shell = N_shell
        self.nu = filling_factor
        self.T = temperature
        
        # System parameters
        self.hbar = parameters.get('hbar', 1.0)  # Reduced Planck constant
        self.m_star = parameters.get('m_star', 1.0)  # Effective mass
        self.V = parameters.get('V', 15.0)  # Intralayer moiré potential strength
        self.w = parameters.get('w', 110.0)  # Interlayer moiré potential strength
        self.phi = parameters.get('phi', np.pi/3)  # Phase in intralayer potential
        self.Delta_D = parameters.get('Delta_D', 10.0)  # Dirac mass
        
        # Coulomb interaction parameters
        self.e2 = parameters.get('e2', 100.0)  # e^2 value (interaction strength)
        self.epsilon = parameters.get('epsilon', 4.0)  # Dielectric constant
        self.epsilon0 = parameters.get('epsilon0', 1.0)  # Vacuum permittivity
        self.d_gate = parameters.get('d_gate', 20.0)  # Distance to gate
        self.d = parameters.get('d', 0.6)  # Interlayer distance
        self.A = parameters.get('A', 1.0)  # Area normalization factor
        
        # Twist angle and derived parameters
        self.theta = parameters.get('theta', 3.0)  # Twist angle in degrees
        self.epsilon_strain = parameters.get('epsilon_strain', 0.0)  # Strain parameter
        self.a_G = parameters.get('a_G', 0.246)  # Graphene lattice constant, nm
        
        # Define moire lattice constant
        self.a_M = self.a_G/np.sqrt(self.epsilon_strain**2 + np.deg2rad(self.theta)**2)  # moire lattice constant, nm

        # Define k-space within first Brillouin zone (momentum space meshgrid)
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        # Define q-space for extended Brillouin zone (reciprocal lattice connecting different Gamma points)
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)

        # Define helper functions
        self.Nk = len(self.k_space)
        self.N_k = len(self.k_space)  # Alias for compatibility
        self.Nq = len(self.q)
        
        # Define high symmetry points
        self.high_symm = generate_high_symmtry_points(self.lattice, self.a_M)
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}
        
        # Define kappa (k-space shift for top and bottom layers)
        self.kappa_plus = np.array([0.0, 0.0])  # Shift for top layer
        self.kappa_minus = np.array([0.0, 0.0])  # Shift for bottom layer
        
        # Define G_vectors for moiré potentials
        # G1, G3, G5 for intralayer potential
        # G2, G3 for interlayer potential
        self.G = []
        for i in range(1, 6):
            angle = (i-1) * 60  # in degrees
            mag = 4 * np.pi / (np.sqrt(3) * self.a_M)
            self.G.append(mag * np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]))
        
        # Degree of freedom including the reciprocal lattice vectors
        # In this model: (layer, q-vector)
        # 2 layers (bottom, top)
        self.D = (2, self.Nq)
        
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
            
        # Initialize Hamiltonian
        H_K = np.zeros((self.D + self.D + (self.Nk,)), dtype=complex)
        
        for idx, q in enumerate(self.q):
            # Compute (k-kappa)^2 terms for top and bottom layers
            k_minus_kappa_plus_x = kx + q[0] - self.kappa_plus[0]
            k_minus_kappa_plus_y = ky + q[1] - self.kappa_plus[1]
            k_minus_kappa_minus_x = kx + q[0] - self.kappa_minus[0]
            k_minus_kappa_minus_y = ky + q[1] - self.kappa_minus[1]
            
            # Kinetic energy for bottom layer (-ħ²(k-κ₊)²/2m*)
            H_K[0, idx, 0, idx, :] = -self.hbar**2 * (k_minus_kappa_plus_x**2 + k_minus_kappa_plus_y**2) / (2 * self.m_star)
            
            # Kinetic energy for top layer (-ħ²(k-κ₋)²/2m*)
            H_K[1, idx, 1, idx, :] = -self.hbar**2 * (k_minus_kappa_minus_x**2 + k_minus_kappa_minus_y**2) / (2 * self.m_star)
        
        return H_K
    
    def H_V(self, k=None) -> np.ndarray:
        """Generate potential energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Potential energy Hamiltonian
        """
        # Generate potential terms
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
            
        # Initialize Hamiltonian for potential terms
        H_M = np.zeros((self.D + self.D + (self.Nk,)), dtype=complex)
        
        # 1. Add intralayer moiré potentials
        # Delta_b and Delta_t are the intralayer potentials for bottom and top
        # defined as 2V*sum_{i=1,3,5}cos(G_i·r±phi)
        # In momentum space, this becomes V term on diagonal for each G vector
        
        # For each q-vector, add the moiré potential
        for idx, q in enumerate(self.q):
            # Diagonal term for bottom layer: Delta_D/2
            H_M[0, idx, 0, idx, :] += self.Delta_D / 2
            
            # Diagonal term for top layer: -Delta_D/2
            H_M[1, idx, 1, idx, :] -= self.Delta_D / 2
            
            # Add Δ_b/Δ_t potentials for each G connection
            for idx2, q2 in enumerate(self.q):
                diff_q = q - q2
                
                # Check if diff_q matches any of the G vectors or their negatives
                for i in [0, 2, 4]:  # G1, G3, G5
                    # Bottom layer with +phi phase
                    if np.allclose(diff_q, self.G[i]):
                        H_M[0, idx, 0, idx2, :] += self.V * np.exp(1j * self.phi)
                    if np.allclose(diff_q, -self.G[i]):
                        H_M[0, idx, 0, idx2, :] += self.V * np.exp(-1j * self.phi)
                    
                    # Top layer with -phi phase
                    if np.allclose(diff_q, self.G[i]):
                        H_M[1, idx, 1, idx2, :] += self.V * np.exp(-1j * self.phi)
                    if np.allclose(diff_q, -self.G[i]):
                        H_M[1, idx, 1, idx2, :] += self.V * np.exp(1j * self.phi)
                
                # Interlayer potential Delta_T
                # Delta_T = w(1 + e^(-iG2·r) + e^(-iG3·r))
                
                # Direct tunneling (q = q2)
                if np.allclose(diff_q, np.zeros(2)):
                    H_M[0, idx, 1, idx2, :] += self.w
                    H_M[1, idx2, 0, idx, :] += self.w  # Hermitian conjugate
                
                # G2 tunneling
                if np.allclose(diff_q, -self.G[1]):
                    H_M[0, idx, 1, idx2, :] += self.w
                    H_M[1, idx2, 0, idx, :] += self.w  # Hermitian conjugate
                
                # G3 tunneling
                if np.allclose(diff_q, -self.G[2]):
                    H_M[0, idx, 1, idx2, :] += self.w
                    H_M[1, idx2, 0, idx, :] += self.w  # Hermitian conjugate
        
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
        
        # Calculate Coulomb interaction for each q-vector
        # V_{ll'}(q) = (e^2)/(2εε0|q|) * [tanh(d_gate|q|) + (1-δ_{ll'})(e^(-d|q|)-1)]
        
        # For each k point
        for k_idx in range(self.Nk):
            # For each layer l and l'
            for l in range(2):  # l = 0 (bottom), 1 (top)
                for l_prime in range(2):  # l' = 0 (bottom), 1 (top)
                    # For each q-vector
                    for q_idx, q in enumerate(self.q):
                        # Calculate |q|
                        q_mag = np.sqrt(q[0]**2 + q[1]**2) + 1e-10  # Add small constant to avoid division by zero
                        
                        # Calculate the Coulomb potential V_{ll'}(q)
                        # V_{ll'}(q) = (e^2)/(2εε0|q|) * [tanh(d_gate|q|) + (1-δ_{ll'})(e^(-d|q|)-1)]
                        V_q = self.e2 / (2 * self.epsilon * self.epsilon0 * q_mag)
                        
                        # Factor 1: tanh(d_gate|q|)
                        factor1 = np.tanh(self.d_gate * q_mag)
                        
                        # Factor 2: (1-δ_{ll'})(e^(-d|q|)-1)
                        delta_ll = 1 if l == l_prime else 0
                        factor2 = (1 - delta_ll) * (np.exp(-self.d * q_mag) - 1)
                        
                        # Combine factors
                        V_q *= (factor1 + factor2)
                        
                        # Direct Hartree term: <c†_{l,k+q} c_{l,k}> c†_{l',k'-q} c_{l',k'}
                        for idx1 in range(self.Nq):
                            for idx2 in range(self.Nq):
                                # Find the corresponding indices in k-space
                                k_plus_q = (k_idx + q_idx) % self.Nk  # Approximation for k+q
                                
                                # Direct term
                                direct_term = V_q * exp_val[l, idx1, l, idx1, k_plus_q]
                                H_int[l_prime, idx2, l_prime, idx2, k_idx] += direct_term / self.A
                                
                                # Exchange term (subtracted): -<c†_{l,k+q} c_{l',k'}> c†_{l',k'-q} c_{l,k}
                                exchange_term = V_q * exp_val[l, idx1, l_prime, idx2, k_plus_q]
                                H_int[l_prime, idx2, l, idx1, k_idx] -= exchange_term / self.A
        
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
    ham = HartreeFockHamiltonian(
        N_shell=10,
        parameters={
            'hbar': 1.0,
            'm_star': 1.0,
            'V': 15.0,
            'w': 110.0,
            'phi': np.pi/3,
            'Delta_D': 10.0,
            'e2': 100.0,
            'epsilon': 4.0,
            'epsilon0': 1.0,
            'd_gate': 20.0,
            'd': 0.6,
            'A': 1.0,
            'theta': 3.0,
            'a_G': 0.246
        }
    )
    
    # Infinitesimal coupling (very weak interaction)
    ham_infinitesimal_u = HartreeFockHamiltonian(
        N_shell=10,
        parameters={
            'hbar': 1.0,
            'm_star': 1.0,
            'V': 15.0,
            'w': 110.0,
            'phi': np.pi/3,
            'Delta_D': 10.0,
            'e2': 1.0,  # Very small interaction strength
            'epsilon': 40.0,  # Higher dielectric constant to reduce interaction
            'epsilon0': 1.0,
            'd_gate': 20.0,
            'd': 0.6,
            'A': 10.0,  # Larger area to reduce interaction density
            'theta': 3.0,
            'a_G': 0.246
        }
    )
    
    # Large coupling (strong interaction)
    ham_large_u = HartreeFockHamiltonian(
        N_shell=10,
        parameters={
            'hbar': 1.0,
            'm_star': 1.0,
            'V': 15.0,
            'w': 110.0,
            'phi': np.pi/3,
            'Delta_D': 10.0,
            'e2': 1000.0,  # Very large interaction strength
            'epsilon': 1.0,  # Lower dielectric constant to enhance interaction
            'epsilon0': 1.0,
            'd_gate': 5.0,  # Shorter gate distance to enhance interaction
            'd': 0.3,  # Smaller interlayer distance
            'A': 0.1,  # Smaller area to increase interaction density
            'theta': 3.0,
            'a_G': 0.246
        }
    )
    # LLM Edits End
        
    run_tests.generate_artifacts(ham, ham_infinitesimal_u, ham_large_u)
