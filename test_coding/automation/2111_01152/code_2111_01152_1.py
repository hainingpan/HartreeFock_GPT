import numpy as np
from typing import Dict, Any
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a moiré superlattice of MoTe2/WSe2.
    
    This class implements the Hartree-Fock Hamiltonian for a heterobilayer 
    of MoTe2 and WSe2 with triangular lattice symmetry.
    """
    def __init__(self, 
                 parameters: Dict[str, Any] = None,
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
        
        # Default parameters if none provided
        if parameters is None:
            parameters = {}
            
        # Physical constants
        self.hbar = parameters.get('hbar', 1.0)  # Reduced Planck constant
        self.me = parameters.get('me', 1.0)  # Electron rest mass
        
        # Lattice constants
        self.a_b = parameters.get('a_b', 3.575)  # MoTe2 lattice constant in Angstrom
        self.a_t = parameters.get('a_t', 3.32)   # WSe2 lattice constant in Angstrom
        self.theta = parameters.get('theta', 0.0)  # Twist angle in degrees
        self.epsilon = parameters.get('epsilon', 1.0)  # Dielectric constant
        
        # Effective masses
        self.m_b = parameters.get('m_b', 0.65) * self.me  # Bottom layer effective mass
        self.m_t = parameters.get('m_t', 0.35) * self.me  # Top layer effective mass
        
        # Potential parameters
        self.V_b = parameters.get('V_b', 10.0)  # Bottom layer potential amplitude (meV)
        self.psi_b = parameters.get('psi_b', -14.0)  # Bottom layer potential phase (degrees)
        self.V_zt = parameters.get('V_zt', 0.0)  # Band offset (meV)
        self.w = parameters.get('w', 10.0)  # Tunneling strength (meV)
        self.d = parameters.get('d', 0.5)  # Screening length for Coulomb interaction (nm)
        
        # Calculate moiré lattice constant
        self.a_G = self.a_b * self.a_t / abs(self.a_b - self.a_t)  # Moiré lattice constant (initial)
        self.a_M = self.a_G/np.sqrt(self.epsilon**2 + np.deg2rad(self.theta)**2)  # Moiré lattice constant, nm
        
        # Define k-space and q-space
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        
        # Helper dimensions
        self.Nk = len(self.k_space)
        self.Nq = len(self.q)
        
        # Degree of freedom including the reciprocal lattice vectors
        self.D = (2, 2, self.Nq)  # (layer, valley, q-vectors)
        
        # Define high symmetry points
        self.high_symm = generate_high_symmetry_points(self.lattice, self.a_M)
        self.omega = np.exp(1j * 2 * np.pi / 3)  # Phase factor for C3z symmetry
        
        # Define kappa (corner of moiré Brillouin zone)
        self.kappa = 4 * np.pi / (3 * self.a_M) * np.array([1.0, 0.0])
        
        # Define g vectors for potential
        self.g = {}
        for j in range(1, 4):
            self.g[j] = rotation_mat(120*(j-1))@self.high_symm["Gamma'"]
    
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
        
        # Initialize kinetic Hamiltonian
        H_K = np.zeros((2, 2, self.Nq, 2, 2, self.Nq, len(kx)), dtype=np.complex128)
        
        # Calculate k magnitude squared
        k_squared = kx**2 + ky**2
        
        # For each q vector
        for idx_q in range(self.Nq):
            # Bottom layer, +K valley: -ℏ²k²/2m_b
            H_K[0, 0, idx_q, 0, 0, idx_q, :] = -self.hbar**2 * k_squared / (2 * self.m_b)
            
            # Top layer, +K valley: -ℏ²(k-κ)²/2m_t
            kx_shifted = kx - self.kappa[0]
            ky_shifted = ky - self.kappa[1]
            k_minus_kappa_squared = kx_shifted**2 + ky_shifted**2
            H_K[1, 0, idx_q, 1, 0, idx_q, :] = -self.hbar**2 * k_minus_kappa_squared / (2 * self.m_t)
            
            # Bottom layer, -K valley: -ℏ²k²/2m_b
            H_K[0, 1, idx_q, 0, 1, idx_q, :] = -self.hbar**2 * k_squared / (2 * self.m_b)
            
            # Top layer, -K valley: -ℏ²(k+κ)²/2m_t
            kx_shifted = kx + self.kappa[0]
            ky_shifted = ky + self.kappa[1]
            k_plus_kappa_squared = kx_shifted**2 + ky_shifted**2
            H_K[1, 1, idx_q, 1, 1, idx_q, :] = -self.hbar**2 * k_plus_kappa_squared / (2 * self.m_t)
        
        return H_K
    
    def H_V(self, k=None) -> np.ndarray:
        """Generate potential energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Potential energy Hamiltonian
        """
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
        
        # Initialize potential Hamiltonian
        H_M = np.zeros((2, 2, self.Nq, 2, 2, self.Nq, len(kx)), dtype=np.complex128)
        
        # Add band offset to top layer
        for idx_q in range(self.Nq):
            H_M[1, 0, idx_q, 1, 0, idx_q, :] = self.V_zt
            H_M[1, 1, idx_q, 1, 1, idx_q, :] = self.V_zt
        
        # Bottom layer potential (Δ_b)
        psi_rad = np.deg2rad(self.psi_b)
        for idx_q1 in range(self.Nq):
            for idx_q2 in range(self.Nq):
                for j in [1, 3, 5]:  # j values from the potential definition
                    g_j_idx = (j - 1) // 2 + 1  # Map to our g indices (1, 2, 3)
                    if g_j_idx in self.g:
                        q_diff = self.q[idx_q1] - self.q[idx_q2]
                        if np.allclose(q_diff, self.g[g_j_idx]) or np.allclose(q_diff, -self.g[g_j_idx]):
                            # Apply to both valleys
                            potential_term = self.V_b * np.exp(1j * psi_rad) if np.allclose(q_diff, self.g[g_j_idx]) else self.V_b * np.exp(-1j * psi_rad)
                            # Bottom layer potential for +K valley
                            H_M[0, 0, idx_q1, 0, 0, idx_q2, :] += potential_term
                            # Bottom layer potential for -K valley
                            H_M[0, 1, idx_q1, 0, 1, idx_q2, :] += potential_term
        
        # Interlayer tunneling (Δ_T,τ)
        for idx_q1 in range(self.Nq):
            for idx_q2 in range(self.Nq):
                # Direct tunneling term
                if np.allclose(self.q[idx_q1], self.q[idx_q2]):
                    # For +K valley
                    H_M[0, 0, idx_q1, 1, 0, idx_q2, :] = self.w
                    H_M[1, 0, idx_q2, 0, 0, idx_q1, :] = np.conj(self.w)
                    
                    # For -K valley
                    H_M[0, 1, idx_q1, 1, 1, idx_q2, :] = self.w
                    H_M[1, 1, idx_q2, 0, 1, idx_q1, :] = np.conj(self.w)
                
                # g2 tunneling term
                q_diff = self.q[idx_q1] - self.q[idx_q2]
                if np.allclose(q_diff, self.g[2]):
                    # For +K valley
                    H_M[0, 0, idx_q1, 1, 0, idx_q2, :] = self.w * self.omega
                    H_M[1, 0, idx_q2, 0, 0, idx_q1, :] = np.conj(self.w * self.omega)
                    
                    # For -K valley (note valley dependence)
                    H_M[0, 1, idx_q1, 1, 1, idx_q2, :] = self.w * np.conj(self.omega)
                    H_M[1, 1, idx_q2, 0, 1, idx_q1, :] = np.conj(self.w * np.conj(self.omega))
                
                # g3 tunneling term
                if np.allclose(q_diff, self.g[3]):
                    # For +K valley
                    H_M[0, 0, idx_q1, 1, 0, idx_q2, :] = self.w * self.omega**2
                    H_M[1, 0, idx_q2, 0, 0, idx_q1, :] = np.conj(self.w * self.omega**2)
                    
                    # For -K valley (note valley dependence)
                    H_M[0, 1, idx_q1, 1, 1, idx_q2, :] = self.w * np.conj(self.omega**2)
                    H_M[1, 1, idx_q2, 0, 1, idx_q1, :] = np.conj(self.w * np.conj(self.omega**2))
        
        return H_M
    
    def generate_non_interacting(self, k=None) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Non-interacting Hamiltonian (kinetic + potential)
        """
        return self.generate_kinetic(k) + self.H_V(k)
    
    def calculate_coulomb_interaction(self, q):
        """Calculate the Coulomb interaction for a given q vector.
        
        Args:
            q: The momentum transfer vector
            
        Returns:
            float: The Coulomb interaction strength
        """
        q_mag = np.linalg.norm(q)
        if q_mag < 1e-10:  # Avoid division by zero
            return 0.0
        return 2 * np.pi * np.tanh(q_mag * self.d) / (self.epsilon * q_mag)
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        # Unflatten the expectation value
        exp_val = unflatten(exp_val, self.D, self.Nk)
        
        # Initialize interaction Hamiltonian
        H_int = np.zeros((2, 2, self.Nq, 2, 2, self.Nq, self.Nk), dtype=np.complex128)
        
        # Hartree term calculation
        for l2 in range(2):  # Layer index
            for tau2 in range(2):  # Valley index
                for q2_idx in range(self.Nq):
                    for q3_idx in range(self.Nq):
                        q_diff = self.q[q2_idx] - self.q[q3_idx]
                        v_q = self.calculate_coulomb_interaction(q_diff)
                        
                        # Sum over all states for the density
                        hartree_sum = 0
                        for l1 in range(2):
                            for tau1 in range(2):
                                for q1_idx in range(self.Nq):
                                    q4_idx = q1_idx  # For diagonal elements in density
                                    if np.allclose(self.q[q1_idx] + self.q[q2_idx], self.q[q4_idx] + self.q[q3_idx]):
                                        # Take the mean over k points for the density
                                        rho = np.mean(exp_val[l1, tau1, q1_idx, l1, tau1, q4_idx, :])
                                        hartree_sum += rho * v_q
                        
                        # Add the Hartree contribution
                        H_int[l2, tau2, q2_idx, l2, tau2, q3_idx, :] += hartree_sum / self.Nk
        
        # Fock term calculation
        for l2 in range(2):  # Layer index for creation
            for tau2 in range(2):  # Valley index for creation
                for q2_idx in range(self.Nq):  # q-vector index for creation
                    for l1 in range(2):  # Layer index for annihilation
                        for tau1 in range(2):  # Valley index for annihilation
                            for q4_idx in range(self.Nq):  # q-vector index for annihilation
                                for k2_idx in range(self.Nk):  # k-point index
                                    k2 = self.k_space[k2_idx]
                                    
                                    # For each expectation value
                                    for q1_idx in range(self.Nq):
                                        for q3_idx in range(self.Nq):
                                            if np.allclose(self.q[q1_idx] + self.q[q2_idx], self.q[q3_idx] + self.q[q4_idx]):
                                                # Calculate the momentum transfer
                                                k1 = k2  # Same k-point in exp_val
                                                q_transfer = k1 + self.q[q1_idx] - k2 - self.q[q4_idx]
                                                v_q = self.calculate_coulomb_interaction(q_transfer)
                                                
                                                # Get the expectation value
                                                rho = exp_val[l1, tau1, q1_idx, l2, tau2, q3_idx, k2_idx]
                                                
                                                # Add the Fock contribution (note the negative sign)
                                                H_int[l2, tau2, q2_idx, l1, tau1, q4_idx, k2_idx] -= rho * v_q / self.Nk
        
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
        
        # Return flattened Hamiltonian for solve() function
        return flattened(H_total, self.D, self.Nk)
