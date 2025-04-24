import numpy as np
from typing import Dict, Tuple, Optional
from scipy import constants
from HF import *

class HartreeFockHamiltonian:
    def __init__(self, 
                 parameters: Dict = None,
                 N_shell: int = 10,
                 Nq_shell: int = 1,
                 filling_factor: float = 0.5,
                 temperature: float = 0):
        """Initialize the Hartree-Fock Hamiltonian for twisted bilayer system.
        
        Args:
            parameters: Dictionary containing model parameters
            N_shell: Number of shells in k-space
            Nq_shell: Number of shells in q-space
            filling_factor: Electron filling factor
            temperature: Temperature for Fermi-Dirac distribution
        """
        self.lattice = 'triangular'
        
        # Default parameters if not provided
        if parameters is None:
            parameters = {}
            
        # Define basic parameters
        self.N_shell = N_shell
        self.nu = filling_factor
        self.T = temperature
        
        # Physical constants
        self.hbar = constants.hbar  # Reduced Planck constant
        self.e = constants.e  # Elementary charge
        self.epsilon0 = constants.epsilon_0  # Vacuum permittivity
        
        # Model parameters
        self.m_star = parameters.get('m_star', 0.5 * constants.m_e)  # Effective mass
        self.kappa_plus = parameters.get('kappa_plus', np.array([0.0, 0.0]))  # Momentum shift for layer 1
        self.kappa_minus = parameters.get('kappa_minus', np.array([0.0, 0.0]))  # Momentum shift for layer 2
        self.Delta_D = parameters.get('Delta_D', 0.01)  # Dirac mass term (eV)
        self.V = parameters.get('V', 0.01)  # Moire potential strength (eV)
        self.phi = parameters.get('phi', 0.0)  # Phase in moire potential
        self.w = parameters.get('w', 0.1)  # Interlayer coupling strength (eV)
        self.epsilon = parameters.get('epsilon', 4.0)  # Dielectric constant
        self.d_gate = parameters.get('d_gate', 10.0)  # Distance to gate (nm)
        self.d = parameters.get('d', 0.6)  # Interlayer distance (nm)
        self.theta = parameters.get('theta', 1.05)  # Twist angle in degrees
        
        # Lattice parameters
        self.a_G = parameters.get('a_G', 0.246)  # Graphene lattice constant (nm)
        self.a_M = self.a_G/np.sqrt(self.epsilon**2 + np.deg2rad(self.theta)**2)  # Moiré lattice constant (nm)
        
        # Generate k-space grid
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        self.Nk = len(self.k_space)
        
        # Generate q-space vectors for interactions
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        self.Nq = len(self.q)
        
        # Define degrees of freedom
        self.D = (2, 2, self.Nq)  # (layer, sublattice, q-vector)
        
        # Define high symmetry points and G vectors
        self.high_symm = generate_high_symmetry_points(self.lattice, self.a_M)
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1, 4)}
        
        # Define complex phase factors
        self.omega = np.exp(1j * 2 * np.pi / 3)
        self.psi = 0.0  # Phase parameter
        
        # System volume/area
        self.A = get_area(self.a_M, self.lattice) * self.Nk

    def generate_kinetic(self, k=None) -> np.ndarray:
        """Generate kinetic energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Kinetic energy Hamiltonian
        """
        if k is None:
            kx, ky = self.k_space[:, 0], self.k_space[:, 1]
        else:
            kx, ky = k[:, 0], k[:, 1]
        
        H_K = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
        
        # Prefactor for kinetic energy
        prefactor = -self.hbar**2 / (2 * self.m_star)
        
        for idx_q1, q1 in enumerate(self.q):
            for idx_q2, q2 in enumerate(self.q):
                # Only consider terms where q1 = q2 for kinetic energy
                if np.allclose(q1, q2):
                    # For each k-point, calculate (k+q-kappa)^2
                    for i in range(self.Nk):
                        # Bottom layer (layer 0)
                        k_plus_q1 = np.array([kx[i], ky[i]]) + q1
                        k_diff_plus = k_plus_q1 - self.kappa_plus
                        k_squared_plus = k_diff_plus[0]**2 + k_diff_plus[1]**2
                        H_K[0, 0, idx_q1, 0, 0, idx_q2, i] += prefactor * k_squared_plus
                        H_K[0, 1, idx_q1, 0, 1, idx_q2, i] += prefactor * k_squared_plus
                        
                        # Top layer (layer 1)
                        k_diff_minus = k_plus_q1 - self.kappa_minus
                        k_squared_minus = k_diff_minus[0]**2 + k_diff_minus[1]**2
                        H_K[1, 0, idx_q1, 1, 0, idx_q2, i] += prefactor * k_squared_minus
                        H_K[1, 1, idx_q1, 1, 1, idx_q2, i] += prefactor * k_squared_minus
        
        return H_K
    
    def H_V(self, k=None) -> np.ndarray:
        """Generate potential energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Potential energy Hamiltonian
        """
        if k is None:
            kx, ky = self.k_space[:, 0], self.k_space[:, 1]
        else:
            kx, ky = k[:, 0], k[:, 1]
            
        H_M = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
        
        # Add intralayer moire potentials (Delta_b/t)
        for idx_q1, q1 in enumerate(self.q):
            for idx_q2, q2 in enumerate(self.q):
                diff_q = q1 - q2
                
                # For each G_i (i=1,3,5), check if diff_q matches G_i or -G_i
                for i in [1, 3, 5]:
                    # We need to map i to corresponding g_j index (i=1->j=1, i=3->j=2, i=5->j=3)
                    j = (i + 1) // 2
                    
                    # Check if diff_q matches G_i
                    if i <= 3 and np.allclose(diff_q, self.g[j]):
                        # Bottom layer potential with +phi
                        H_M[0, 0, idx_q1, 0, 0, idx_q2, :] += self.V * np.exp(1j * self.phi)
                        H_M[0, 1, idx_q1, 0, 1, idx_q2, :] += self.V * np.exp(1j * self.phi)
                        
                        # Top layer potential with -phi
                        H_M[1, 0, idx_q1, 1, 0, idx_q2, :] += self.V * np.exp(-1j * self.phi)
                        H_M[1, 1, idx_q1, 1, 1, idx_q2, :] += self.V * np.exp(-1j * self.phi)
                    
                    # Check if diff_q matches -G_i
                    if i <= 3 and np.allclose(diff_q, -self.g[j]):
                        # Bottom layer potential with -phi
                        H_M[0, 0, idx_q1, 0, 0, idx_q2, :] += self.V * np.exp(-1j * self.phi)
                        H_M[0, 1, idx_q1, 0, 1, idx_q2, :] += self.V * np.exp(-1j * self.phi)
                        
                        # Top layer potential with +phi
                        H_M[1, 0, idx_q1, 1, 0, idx_q2, :] += self.V * np.exp(1j * self.phi)
                        H_M[1, 1, idx_q1, 1, 1, idx_q2, :] += self.V * np.exp(1j * self.phi)
        
        # Add interlayer tunneling (Delta_T)
        for idx_q1, q1 in enumerate(self.q):
            for idx_q2, q2 in enumerate(self.q):
                diff_q = q1 - q2
                
                # Gamma point (no momentum transfer)
                if np.allclose(diff_q, np.zeros(2)):
                    H_M[0, 0, idx_q1, 1, 0, idx_q2, :] += self.w
                    H_M[1, 0, idx_q1, 0, 0, idx_q2, :] += self.w
                
                # G_2 momentum transfer
                if np.allclose(diff_q, self.g[2]):
                    H_M[0, 0, idx_q1, 1, 0, idx_q2, :] += self.w
                    H_M[1, 0, idx_q1, 0, 0, idx_q2, :] += self.w * np.exp(1j * 2 * np.pi / 3)
                
                # G_3 momentum transfer
                if np.allclose(diff_q, self.g[3]):
                    H_M[0, 0, idx_q1, 1, 0, idx_q2, :] += self.w
                    H_M[1, 0, idx_q1, 0, 0, idx_q2, :] += self.w * np.exp(1j * 4 * np.pi / 3)
        
        # Add Dirac mass term (Delta_D)
        for idx_q, q in enumerate(self.q):
            # Bottom layer gets +Delta_D/2
            H_M[0, 0, idx_q, 0, 0, idx_q, :] += 0.5 * self.Delta_D
            H_M[0, 1, idx_q, 0, 1, idx_q, :] += 0.5 * self.Delta_D
            
            # Top layer gets -Delta_D/2
            H_M[1, 0, idx_q, 1, 0, idx_q, :] -= 0.5 * self.Delta_D
            H_M[1, 1, idx_q, 1, 1, idx_q, :] -= 0.5 * self.Delta_D
            
        return H_M
    
    def generate_non_interacting(self, k=None) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Non-interacting Hamiltonian (kinetic + potential)
        """
        return self.generate_kinetic(k) + self.H_V(k)
    
    def coulomb_interaction(self, q) -> np.ndarray:
        """Calculate the Coulomb interaction matrix V_{ll'} for a given q.
        
        Args:
            q: Momentum transfer vector
            
        Returns:
            np.ndarray: Coulomb interaction matrix between layers
        """
        # Calculate |q|
        q_mag = np.linalg.norm(q)
        
        # Avoid division by zero
        if q_mag < 1e-10:
            q_mag = 1e-10
            
        # Prefactor
        prefactor = self.e**2 / (2 * self.epsilon * self.epsilon0 * q_mag)
        
        # Initialize the interaction matrix (layer x layer)
        V_ll = np.zeros((2, 2), dtype=complex)
        
        # Diagonal elements (intralayer)
        for l in range(2):
            V_ll[l, l] = prefactor * np.tanh(self.d_gate * q_mag)
            
        # Off-diagonal elements (interlayer)
        V_ll[0, 1] = prefactor * (np.exp(-self.d * q_mag) - 1)
        V_ll[1, 0] = V_ll[0, 1]
        
        return V_ll
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        exp_val = unflatten(exp_val, self.D, self.Nk)
        H_int = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
        
        # Calculate Hartree and Fock terms
        for idx_k in range(self.Nk):
            for idx_q, q_val in enumerate(self.q):
                # Calculate Coulomb matrix for this q
                V_ll = self.coulomb_interaction(q_val)
                
                # Loop over layers and sublattices
                for l in range(2):  # layer index
                    for l_prime in range(2):  # layer' index
                        for tau in range(2):  # sublattice index
                            for tau_prime in range(2):  # sublattice' index
                                # Hartree term (direct)
                                hartree_term = 0
                                for idx_q_prime, q_prime in enumerate(self.q):
                                    # Calculate density terms
                                    rho_l_tau = np.mean(exp_val[l, tau, idx_q_prime, l, tau, :, :])
                                    
                                    # Contribution to Hartree term
                                    hartree_term += V_ll[l, l_prime] * rho_l_tau / self.A
                                
                                # Add Hartree term to Hamiltonian
                                H_int[l_prime, tau_prime, idx_q, l_prime, tau_prime, idx_q, idx_k] += hartree_term
                                
                                # Fock term (exchange)
                                fock_term = 0
                                for idx_q_prime, q_prime in enumerate(self.q):
                                    # Only consider terms where q_prime - q matches a valid reciprocal vector
                                    for idx_q_diff, q_diff in enumerate(self.q):
                                        if np.allclose(q_prime - q_val, q_diff):
                                            # Calculate exchange density matrix elements
                                            exchange_term = np.mean(exp_val[l, tau, idx_q_prime, l_prime, tau_prime, idx_q_diff, :])
                                            
                                            # Contribution to Fock term
                                            fock_term -= V_ll[l, l_prime] * exchange_term / self.A
                                
                                # Add Fock term to Hamiltonian
                                H_int[l_prime, tau_prime, idx_q, l, tau, idx_q, idx_k] += fock_term
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the total Hamiltonian including both non-interacting and interacting parts.
        
        This function is required by the solve() function in HF.py.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Total Hamiltonian (non-interacting + interacting)
        """
        return flattened(self.generate_non_interacting() + self.generate_interacting(exp_val), self.D, self.Nk)
