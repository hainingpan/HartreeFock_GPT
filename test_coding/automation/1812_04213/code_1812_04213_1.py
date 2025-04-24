import numpy as np
from typing import Optional, Dict, Any
from HF import flattened, unflatten

class HartreeFockHamiltonian:
    def __init__(self, 
                 parameters: Dict[str, Any] = None,
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
        if parameters is None:
            parameters = {}
            
        self.lattice = 'triangular'  # Moiré lattice symmetry
        
        self.N_shell = N_shell
        self.Nq_shell = Nq_shell
        self.nu = filling_factor
        self.T = temperature
        
        # Physical constants and model parameters
        self.hbar_vD = parameters.get('hbar_vD', 1.0)  # Dirac velocity constant
        self.theta = parameters.get('theta', 1.1)  # Twist angle in degrees
        self.omega0 = parameters.get('omega0', 0.08)  # Interlayer tunneling parameter (sigma0 term)
        self.omega1 = parameters.get('omega1', 0.1)  # Interlayer tunneling parameter (sigma_x/y term)
        self.phi = parameters.get('phi', 0.0)  # Phase in tunneling Hamiltonian
        self.V = parameters.get('V', 1.0)  # Interaction potential strength
        self.a = parameters.get('a', 0.246)  # Graphene lattice constant in nm
        self.psi = parameters.get('psi', 0.0)  # Additional phase factor
        
        # Derived parameters
        self.epsilon = 2 * np.sin(np.deg2rad(self.theta/2))
        self.a_M = self.a / self.epsilon  # moiré lattice constant, nm
        
        # Define k-space within first Brillouin zone
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        # Define q-space for extended Brillouin zone
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        
        # Helper dimensions
        self.Nk = len(self.k_space)
        self.Nq = len(self.q)
        
        # Degree of freedom: (layer, sublattice, q-vector)
        self.D = (2, 2, self.Nq)
        
        # Define high symmetry points
        self.high_symm = generate_high_symmetry_points(self.lattice, self.a_M)
        
        # Define g vectors (reciprocal lattice vectors)
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}
        
        # Complex phase for 120-degree rotation
        self.omega = np.exp(1j * 2 * np.pi / 3)

    def generate_kinetic(self, k=None) -> np.ndarray:
        """Generate kinetic energy terms of the Hamiltonian (Dirac Hamiltonian).
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Kinetic energy Hamiltonian
        """
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
        
        # Initialize Hamiltonian tensor
        H_K = np.zeros((2, 2, self.Nq, 2, 2, self.Nq, self.Nk), dtype=complex)
        
        # Dirac points for each layer
        K_plus = 4 * np.pi / (3 * self.a) * np.array([np.cos(np.deg2rad(self.theta/2)), np.sin(np.deg2rad(self.theta/2))])
        K_minus = 4 * np.pi / (3 * self.a) * np.array([np.cos(np.deg2rad(-self.theta/2)), np.sin(np.deg2rad(-self.theta/2))])
        
        # Loop over all q vectors
        for idx_q, q in enumerate(self.q):
            # For each k-point, calculate momentum relative to Dirac point
            
            # Top layer (layer 0): h_{theta/2}(k)
            k_bar_top = np.array([kx + q[0] - K_plus[0], ky + q[1] - K_plus[1]]).T
            k_bar_mag_top = np.sqrt(k_bar_top[:,0]**2 + k_bar_top[:,1]**2)
            theta_k_bar_top = np.arctan2(k_bar_top[:,1], k_bar_top[:,0])
            
            # Off-diagonal elements for top layer (sublattice A to B)
            H_K[0, 0, idx_q, 0, 1, idx_q, :] = -self.hbar_vD * k_bar_mag_top * np.exp(1j * (theta_k_bar_top - np.deg2rad(self.theta/2)))
            # Off-diagonal elements for top layer (sublattice B to A)
            H_K[0, 1, idx_q, 0, 0, idx_q, :] = -self.hbar_vD * k_bar_mag_top * np.exp(-1j * (theta_k_bar_top - np.deg2rad(self.theta/2)))
            
            # Bottom layer (layer 1): h_{-theta/2}(k)
            k_bar_bottom = np.array([kx + q[0] - K_minus[0], ky + q[1] - K_minus[1]]).T
            k_bar_mag_bottom = np.sqrt(k_bar_bottom[:,0]**2 + k_bar_bottom[:,1]**2)
            theta_k_bar_bottom = np.arctan2(k_bar_bottom[:,1], k_bar_bottom[:,0])
            
            # Off-diagonal elements for bottom layer (sublattice A to B)
            H_K[1, 0, idx_q, 1, 1, idx_q, :] = -self.hbar_vD * k_bar_mag_bottom * np.exp(1j * (theta_k_bar_bottom - np.deg2rad(-self.theta/2)))
            # Off-diagonal elements for bottom layer (sublattice B to A)
            H_K[1, 1, idx_q, 1, 0, idx_q, :] = -self.hbar_vD * k_bar_mag_bottom * np.exp(-1j * (theta_k_bar_bottom - np.deg2rad(-self.theta/2)))
        
        return H_K
    
    def H_V(self, k=None) -> np.ndarray:
        """Generate interlayer tunneling terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Interlayer tunneling Hamiltonian
        """
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
        
        # Initialize Hamiltonian tensor
        H_T = np.zeros((2, 2, self.Nq, 2, 2, self.Nq, self.Nk), dtype=complex)
        
        # Loop over all reciprocal lattice vectors
        for idx_q1, q1 in enumerate(self.q):
            for idx_q2, q2 in enumerate(self.q):
                diff_q = q1 - q2
                
                # Check if diff_q matches any of the g vectors
                for j in range(1, 4):  # j = 1, 2, 3
                    if np.allclose(diff_q, self.g[j]):
                        # T_j matrix elements (top to bottom tunneling)
                        # T_j = omega0*sigma0 + omega1*cos(j*phi)*sigma_x + omega1*sin(j*phi)*sigma_y
                        
                        # sigma0 term (diagonal)
                        H_T[0, 0, idx_q1, 1, 0, idx_q2, :] = self.omega0
                        H_T[0, 1, idx_q1, 1, 1, idx_q2, :] = self.omega0
                        
                        # sigma_x term (off-diagonal real part)
                        cos_term = self.omega1 * np.cos((j-1) * self.phi)
                        H_T[0, 0, idx_q1, 1, 1, idx_q2, :] += cos_term
                        H_T[0, 1, idx_q1, 1, 0, idx_q2, :] += cos_term
                        
                        # sigma_y term (off-diagonal imaginary part)
                        sin_term = self.omega1 * np.sin((j-1) * self.phi)
                        H_T[0, 0, idx_q1, 1, 1, idx_q2, :] += 1j * sin_term
                        H_T[0, 1, idx_q1, 1, 0, idx_q2, :] -= 1j * sin_term
                        
                    if np.allclose(diff_q, -self.g[j]):
                        # Hermitian conjugate (bottom to top tunneling)
                        # Conjugate of sigma0 term
                        H_T[1, 0, idx_q1, 0, 0, idx_q2, :] = self.omega0
                        H_T[1, 1, idx_q1, 0, 1, idx_q2, :] = self.omega0
                        
                        # Conjugate of sigma_x term
                        cos_term = self.omega1 * np.cos((j-1) * self.phi)
                        H_T[1, 0, idx_q1, 0, 1, idx_q2, :] += cos_term
                        H_T[1, 1, idx_q1, 0, 0, idx_q2, :] += cos_term
                        
                        # Conjugate of sigma_y term
                        sin_term = self.omega1 * np.sin((j-1) * self.phi)
                        H_T[1, 0, idx_q1, 0, 1, idx_q2, :] -= 1j * sin_term
                        H_T[1, 1, idx_q1, 0, 0, idx_q2, :] += 1j * sin_term
        
        return H_T
    
    def generate_non_interacting(self, k=None) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Non-interacting Hamiltonian (kinetic + tunneling)
        """
        return self.generate_kinetic(k) + self.H_V(k)
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian (Hartree and Fock terms).
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        exp_val = unflatten(exp_val, self.D, self.Nk)
        H_int = np.zeros((2, 2, self.Nq, 2, 2, self.Nq, self.Nk), dtype=complex)
        
        # Calculate delta_rho (difference from isolated layers)
        # For simplicity, assume exp_val already represents delta_rho
        delta_rho = exp_val
        
        # Hartree term calculation
        # Σ^H_α,G;β,G'(k) = (1/A)∑_α' V_α'α(G'-G) δρ_α'α'(G-G') δ_αβ
        
        # Calculate summed density for each G-G'
        for l1 in range(2):  # layer index
            for s1 in range(2):  # sublattice index
                for q1_idx, q1 in enumerate(self.q):
                    for l2 in range(2):
                        for s2 in range(2):
                            for q2_idx, q2 in enumerate(self.q):
                                # Hartree term (only contributes when l1=l2 and s1=s2)
                                if l1 == l2 and s1 == s2:
                                    q_diff = q2 - q1
                                    
                                    # Sum over all possible α' (layer and sublattice indices)
                                    for l_prime in range(2):
                                        for s_prime in range(2):
                                            # Calculate mean density
                                            density_sum = 0
                                            for q_idx in range(self.Nq):
                                                q_idx2 = np.argmin(np.sum((self.q - (q1 - q2 + self.q[q_idx]))**2, axis=1))
                                                density_sum += np.mean(delta_rho[l_prime, s_prime, q_idx, l_prime, s_prime, q_idx2, :])
                                            
                                            # Calculate Coulomb potential
                                            # Use regularized potential V/|q| with small cutoff to avoid divergence
                                            potential = self.V / (np.sqrt(np.sum(q_diff**2) + 0.01))
                                            
                                            # Add Hartree contribution
                                            H_int[l1, s1, q1_idx, l2, s2, q2_idx, :] += potential * density_sum / self.Nk
                
                                # Fock term calculation
                                # Σ^F_α,G;β,G'(k) = -(1/A)∑_{G'',k'} V_αβ(G''+k'-k) δρ_α,G+G'';β,G'+G''(k')
                                
                                # For each G'' (using q vectors as G'')
                                for q_pp_idx, q_pp in enumerate(self.q):
                                    # Find indices for G+G'' and G'+G''
                                    q1_plus_qpp = q1 + q_pp
                                    q2_plus_qpp = q2 + q_pp
                                    
                                    # Find closest vectors in our q-grid
                                    idx_q1_plus_qpp = np.argmin(np.sum((self.q - q1_plus_qpp.reshape(1, 2))**2, axis=1))
                                    idx_q2_plus_qpp = np.argmin(np.sum((self.q - q2_plus_qpp.reshape(1, 2))**2, axis=1))
                                    
                                    # For each k-point, calculate the exchange contribution
                                    for k_idx in range(self.Nk):
                                        k = self.k_space[k_idx]
                                        
                                        # For simplicity, use k' = k (equivalent to considering only one k' point)
                                        k_prime = k
                                        k_prime_idx = k_idx
                                        
                                        # Calculate momentum transfer
                                        q_transfer = q_pp + k_prime - k
                                        
                                        # Calculate Coulomb potential
                                        potential = self.V / (np.sqrt(np.sum(q_transfer**2) + 0.01))
                                        
                                        # Add Fock contribution
                                        H_int[l1, s1, q1_idx, l2, s2, q2_idx, k_idx] -= potential * delta_rho[l1, s1, idx_q1_plus_qpp, l2, s2, idx_q2_plus_qpp, k_prime_idx] / self.Nk
        
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
        H_total = H_non_int + H_int
        
        # Return flattened Hamiltonian as required by solve()
        return flattened(H_total, self.D, self.Nk)
