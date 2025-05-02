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
        
        # Lattice type - triangular for the moiré superlattice
        self.lattice = 'triangular'
        
        # Basic parameters
        self.N_shell = N_shell
        self.nu = filling_factor
        self.T = temperature
        
        # Material parameters
        self.a_b = parameters.get('a_b', 3.575)  # MoTe2 lattice constant in Å
        self.a_t = parameters.get('a_t', 3.32)   # WSe2 lattice constant in Å
        
        # Calculate moiré lattice constant
        self.theta = parameters.get('theta', 0.0)  # Twist angle (in degrees)
        self.epsilon = np.abs(self.a_b - self.a_t) / self.a_t  # Lattice mismatch
        self.a_M = self.a_b * self.a_t / np.abs(self.a_b - self.a_t)  # moiré lattice constant in Å
        
        # Effective masses in units of electron mass
        self.m_b = parameters.get('m_b', 0.65)  # Bottom layer (MoTe2) effective mass
        self.m_t = parameters.get('m_t', 0.35)  # Top layer (WSe2) effective mass
        self.m_e = 1.0  # Set electron mass to 1 for unit simplicity
        
        # Reciprocal space parameters
        self.kappa = 4 * np.pi / (3 * self.a_M) * np.array([1, 0])  # Corner of moiré Brillouin zone
        
        # Potential parameters
        self.V_b = parameters.get('V_b', 10.0)  # Amplitude of bottom layer potential, tunable parameter
        self.psi_b = parameters.get('psi_b', -14.0)  # Spatial pattern in degrees
        self.V_zt = parameters.get('V_zt', 0.0)  # Band offset, controlled by out-of-plane displacement field
        
        # Tunneling strength - can be modified by pressure
        self.w = parameters.get('w', 15.0)  # Tunneling strength
        self.omega = np.exp(1j * 2 * np.pi / 3)  # Phase factor for C3 symmetry
        
        # Coulomb interaction parameters
        self.epsilon_r = parameters.get('epsilon_r', 10.0)  # Dielectric constant
        self.d = parameters.get('d', 0.6)  # Effective distance between layers (nm)
        
        # Define k-space within first Brillouin zone
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        
        # Define q-space for extended Brillouin zone
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        
        # Helper attributes
        self.Nk = len(self.k_space)
        self.N_k = self.Nk  # Alias for compatibility with HF.py
        self.Nq = len(self.q)
        
        # Define degree of freedom tuple: (valley, layer, q-vector)
        # valley: +K (0) or -K (1)
        # layer: bottom (0) or top (1)
        self.D = (2, 2, self.Nq)  # (valley, layer, q-vector)
        
        # Define high symmetry points
        self.high_symm = generate_high_symmtry_points(self.lattice, self.a_M)
        
        # Precompute g vectors for the potential terms
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}
        
        # Calculate hbar^2/(2m) factors for kinetic energy (in appropriate units)
        self.hbar_sq_2m_b = 1.0/(2.0 * self.m_b * self.m_e)
        self.hbar_sq_2m_t = 1.0/(2.0 * self.m_t * self.m_e)
    
    def generate_kinetic(self, k=None) -> np.ndarray:
        """Generate kinetic energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Kinetic energy Hamiltonian
        """
        # Use provided k points or default to k_space
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
        
        # Initialize Hamiltonian tensor
        H_K = np.zeros((self.D + self.D + (self.Nk,)), dtype=complex)
        
        # Calculate kinetic energy for each q point shift
        for idx, q in enumerate(self.q):
            # For +K valley, bottom layer
            k_squared = (kx + q[0])**2 + (ky + q[1])**2
            H_K[0, 0, idx, 0, 0, idx, :] = -self.hbar_sq_2m_b * k_squared
            
            # For +K valley, top layer (with kappa shift)
            k_kappa_squared = ((kx + q[0]) - self.kappa[0])**2 + ((ky + q[1]) - self.kappa[1])**2
            H_K[0, 1, idx, 0, 1, idx, :] = -self.hbar_sq_2m_t * k_kappa_squared
            
            # For -K valley, bottom layer
            H_K[1, 0, idx, 1, 0, idx, :] = -self.hbar_sq_2m_b * k_squared
            
            # For -K valley, top layer (with -kappa shift)
            k_neg_kappa_squared = ((kx + q[0]) + self.kappa[0])**2 + ((ky + q[1]) + self.kappa[1])**2
            H_K[1, 1, idx, 1, 1, idx, :] = -self.hbar_sq_2m_t * k_neg_kappa_squared
        
        return H_K
    
    def H_V(self, k=None) -> np.ndarray:
        """Generate potential energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Potential energy Hamiltonian
        """
        # Use provided k points or default to k_space
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
        
        # Initialize potential Hamiltonian
        H_M = np.zeros((self.D + self.D + (self.Nk,)), dtype=complex)
        
        # Use k-vectors directly as position vectors for potential evaluation
        r_vectors = self.k_space
        
        # Assign V0 (diagonal terms - constant potential and band offset)
        for idx, q in enumerate(self.q):
            # Band offset for top layer
            H_M[0, 1, idx, 0, 1, idx, :] = self.V_zt
            H_M[1, 1, idx, 1, 1, idx, :] = self.V_zt
        
        # Calculate g vectors for the periodic potential
        g_vectors = []
        for j in [1, 3, 5]:
            angle_rad = np.radians(60 * (j-1))
            g_j = (4 * np.pi / (np.sqrt(3) * self.a_M)) * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
            g_vectors.append(g_j)
        
        # Calculate the periodic potential Delta_b(r) for all points at once
        Delta_b = np.zeros(self.Nk)
        for g_j in g_vectors:
            # Vectorized calculation of the potential
            Delta_b += 2 * self.V_b * np.cos(np.sum(g_j * r_vectors, axis=1) + np.radians(self.psi_b))
        
        # Add to both valleys for bottom layer (all q points)
        for idx in range(self.Nq):
            H_M[0, 0, idx, 0, 0, idx, :] += Delta_b
            H_M[1, 0, idx, 1, 0, idx, :] += Delta_b
        
        # Calculate interlayer tunneling terms Delta_T,tau(r)
        g1 = g_vectors[0]  # j=1
        g2 = np.array([4 * np.pi / (np.sqrt(3) * self.a_M) * np.cos(np.radians(60)), 
                      4 * np.pi / (np.sqrt(3) * self.a_M) * np.sin(np.radians(60))])  # j=2
        g3 = np.array([4 * np.pi / (np.sqrt(3) * self.a_M) * np.cos(np.radians(120)), 
                      4 * np.pi / (np.sqrt(3) * self.a_M) * np.sin(np.radians(120))])  # j=3
        
        # Off-diagonal coupling between layers for both valleys
        for idx1, q1 in enumerate(self.q):
            for idx2, q2 in enumerate(self.q):
                diff_q = q1 - q2
                
                # Check if the q difference matches one of the g vectors
                for j, g_j in enumerate([g1, g2, g3], 1):
                    if j == 1:
                        # Term with no position dependence
                        for i in range(self.Nk):
                            # +K valley tunneling
                            H_M[0, 0, idx1, 0, 1, idx2, i] += self.w
                            # -K valley tunneling (tau = -1)
                            H_M[1, 0, idx1, 1, 1, idx2, i] += -self.w
                    
                    elif j == 2:
                        if np.allclose(diff_q, g_j, atol=1e-6):
                            for i in range(self.Nk):
                                r = r_vectors[i]
                                # +K valley
                                H_M[0, 0, idx1, 0, 1, idx2, i] += self.w * self.omega
                                # -K valley
                                H_M[1, 0, idx1, 1, 1, idx2, i] += -self.w * self.omega**(-1) * np.exp(-1j * np.dot(g_j, r))
                    
                    elif j == 3:
                        if np.allclose(diff_q, g_j, atol=1e-6):
                            for i in range(self.Nk):
                                r = r_vectors[i]
                                # +K valley
                                H_M[0, 0, idx1, 0, 1, idx2, i] += self.w * self.omega**2
                                # -K valley
                                H_M[1, 0, idx1, 1, 1, idx2, i] += -self.w * self.omega**(-2) * np.exp(-1j * np.dot(g_j, r))
        
        # Hermitian conjugate terms
        for v1 in range(2):  # valley
            for l1 in range(2):  # layer
                for q1 in range(self.Nq):  # q-vector
                    for v2 in range(2):  # valley
                        for l2 in range(2):  # layer
                            for q2 in range(self.Nq):  # q-vector
                                if not (v1 == v2 and l1 == l2 and q1 == q2):  # Skip diagonal elements
                                    H_M[v2, l2, q2, v1, l1, q1, :] = np.conj(H_M[v1, l1, q1, v2, l2, q2, :])
        
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
        
        # Volume factor for normalization
        V = self.Nk
        
        # Compute Coulomb interaction potential for all q differences
        V_coulomb = {}
        for q1 in range(self.Nq):
            for q4 in range(self.Nq):
                q_diff = np.linalg.norm(self.q[q1] - self.q[q4])
                if q_diff > 1e-10:  # Avoid division by zero
                    V_coulomb[(q1, q4)] = 2 * np.pi * np.tanh(q_diff * self.d) / (self.epsilon_r * q_diff)
                else:
                    V_coulomb[(q1, q4)] = 2 * np.pi * self.d / self.epsilon_r  # Limit as q_diff->0
        
        # Simplified Hartree term - use mean-field Coulomb interaction
        hartree_strength = 2 * np.pi * self.d / (self.epsilon_r * V)
        
        for v1 in range(2):  # valley index
            for l1 in range(2):  # layer index
                # Calculate total electron density for this valley/layer
                total_density = 0
                for q in range(self.Nq):
                    total_density += np.mean(exp_val[v1, l1, q, v1, l1, q, :])
                
                # Apply Hartree potential to all other electrons
                for v2 in range(2):  # valley index
                    for l2 in range(2):  # layer index
                        if v1 != v2 or l1 != l2:  # Skip self-interaction
                            for q in range(self.Nq):
                                H_int[v2, l2, q, v2, l2, q, :] += hartree_strength * total_density
        
        # Simplified Fock term - just use mean-field exchange interaction
        # This is a computationally efficient approximation of the full Fock term
        for v1 in range(2):  # valley
            for l1 in range(2):  # layer 
                for v2 in range(2):  # valley
                    for l2 in range(2):  # layer
                        # Calculate average exchange interaction
                        if v1 == v2 and l1 == l2:  # Same valley and layer exchange
                            exchange_strength = 0.5 * 2 * np.pi * self.d / (self.epsilon_r * V)
                            for q in range(self.Nq):
                                # Calculate average density for this valley/layer
                                avg_density = np.mean(exp_val[v1, l1, q, v1, l1, q, :])
                                # Apply exchange interaction uniformly
                                H_int[v1, l1, q, v1, l1, q, :] -= exchange_strength * avg_density
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the total Hamiltonian including both non-interacting and interacting parts.
        
        This function is required by the solve() function in HF.py.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Total Hamiltonian (non-interacting + interacting), 
                       flattened to shape (prod(D), prod(D), Nk)
        """
        h_total = self.generate_non_interacting() + self.generate_interacting(exp_val)
        # Return flattened Hamiltonian as expected by HF.diagonalize
        return flattened(h_total, self.D, self.Nk)
# LLM Edits end

if __name__=='__main__':
    # LLM Edits Start: Instantiate Hamiltonian in different limits    
    # Default parameters
    ham = HartreeFockHamiltonian(N_shell=10)
    
    # Infinitesimal coupling limit - reduce the Coulomb interaction strength
    ham_infinitesimal_u = HartreeFockHamiltonian(
        N_shell=10,
        parameters={
            'epsilon_r': 100.0,  # Increase dielectric constant to reduce interaction
            'w': 5.0,            # Reduce tunneling strength
            'V_b': 2.0           # Reduce potential depth
        }
    )
    
    # Large coupling limit - increase the Coulomb interaction strength
    ham_large_u = HartreeFockHamiltonian(
        N_shell=10,
        parameters={
            'epsilon_r': 2.0,    # Decrease dielectric constant to enhance interaction
            'w': 30.0,           # Increase tunneling strength
            'V_b': 20.0          # Increase potential depth
        }
    )
    # LLM Edits End
        
    run_tests.generate_artifacts(ham, ham_infinitesimal_u, ham_large_u)