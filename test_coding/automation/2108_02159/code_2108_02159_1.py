import numpy as np
from typing import Dict, Any, Tuple
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a moiré continuum model with Coulomb interactions.

    This class implements a moiré continuum model on a triangular lattice with
    Hartree-Fock mean-field treatment of Coulomb interactions.
    """
    def __init__(self, 
                 parameters: Dict[str, Any] = {'hbar': 1.0, 'm_star': 1.0, 'V_M': 1.0, 'phi': 0.0, 'A': 1.0},
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
        self.lattice = 'triangular'  # Lattice symmetry
        
        self.N_shell = N_shell
        self.nu = filling_factor
        self.T = temperature
        
        # Model parameters
        self.hbar = parameters.get('hbar', 1.0)  # Planck constant
        self.m_star = parameters.get('m_star', 1.0)  # Effective mass
        self.V_M = parameters.get('V_M', 1.0)  # Moiré modulation strength
        self.phi = parameters.get('phi', 0.0)  # Moiré modulation phase
        self.A = parameters.get('A', 1.0)  # Area of the unit cell
        self.epsilon = parameters.get('epsilon', 1.0)  # Dielectric constant for Coulomb
        self.theta = parameters.get('theta', 1.0)  # Twist angle in degrees
        
        # Define lattice constant
        self.a_G = parameters.get('a_G', 0.246)  # Graphene lattice constant in nm
        self.a_M = self.a_G/np.sqrt(self.epsilon**2 + np.deg2rad(self.theta)**2)  # moiré lattice constant, nm
        
        # Define k-space within first Brillouin zone (momentum space meshgrid)
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        
        # Define q-space for extended Brillouin zone
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        
        # Helper variables
        self.Nk = len(self.k_space)
        self.Nq = len(self.q)
        
        # Degree of freedom including reciprocal lattice vectors
        self.D = (2, self.Nq)  # 2 for layer/orbital index
        
        # Define high symmetry points
        self.high_symm = generate_high_symmetry_points(self.lattice, self.a_M)
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}
        
        # Calculate moiré potential coefficients
        self.V = [self.V_M * np.exp((-1)**(j-1) * 1j * self.phi) for j in range(1, 7)]
        
        # Coulomb potential parameters (simplified model)
        self.V0 = parameters.get('V0', 1.0)  # On-site interaction
        self.r_screen = parameters.get('r_screen', 10.0)  # Screening length

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
        
        H_K = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
        
        # Kinetic energy is diagonal in layer/orbital and b indices
        for alpha in range(2):  # Loop over layer/orbital index
            for idx, q in enumerate(self.q):
                # Calculate (k+b)^2 for each k-point
                kq_x = kx + q[0]
                kq_y = ky + q[1]
                k_squared = kq_x**2 + kq_y**2
                
                # Kinetic energy: -ħ²/(2m*) * (k+b)²
                H_K[alpha, idx, alpha, idx, :] = -self.hbar**2/(2*self.m_star) * k_squared
        
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
        
        H_M = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
        
        # Moiré potential couples different b vectors
        # For each pair of b vectors that differ by one of the six basic moiré vectors
        for alpha in range(2):  # Loop over layer/orbital index (potential is diagonal in this index)
            for idx1, q1 in enumerate(self.q):
                for idx2, q2 in enumerate(self.q):
                    diff_q = q1 - q2
                    
                    # Check if the difference is one of the six basic moiré vectors
                    for j in range(1, 4):  # Check the three primary g vectors
                        if np.allclose(diff_q, self.g[j]):
                            H_M[alpha, idx1, alpha, idx2, :] = self.V[j-1]
                        if np.allclose(diff_q, -self.g[j]):
                            H_M[alpha, idx1, alpha, idx2, :] = np.conj(self.V[j-1])
                    
                    # The other three vectors (rotated by 60°)
                    for j in range(1, 4):
                        rotated_g = rotation_mat(60) @ self.g[j]
                        if np.allclose(diff_q, rotated_g):
                            H_M[alpha, idx1, alpha, idx2, :] = self.V[j+2]
                        if np.allclose(diff_q, -rotated_g):
                            H_M[alpha, idx1, alpha, idx2, :] = np.conj(self.V[j+2])
        
        return H_M

    def generate_non_interacting(self, k=None) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Non-interacting Hamiltonian (kinetic + potential)
        """
        return self.generate_kinetic(k) + self.H_V(k)

    def coulomb_potential(self, q_vec) -> np.ndarray:
        """Calculate the Coulomb potential in momentum space.
        
        Args:
            q_vec: Momentum vector
            
        Returns:
            np.ndarray: Coulomb potential V(q)
        """
        # Simple model of screened Coulomb potential: V(q) = 2πe²/(ε|q|) * exp(-|q|r_screen)
        # For small |q|, we regularize to avoid divergence
        q_mag = np.linalg.norm(q_vec, axis=-1)
        q_min = 1e-10  # Regularization
        
        # For simplicity, we use a 2D screened Coulomb potential
        v_q = np.zeros_like(q_mag, dtype=complex)
        mask = q_mag > q_min
        v_q[mask] = 2*np.pi/(self.epsilon * q_mag[mask]) * np.exp(-q_mag[mask] * self.r_screen)
        v_q[~mask] = 2*np.pi/(self.epsilon * q_min) * np.exp(-q_min * self.r_screen)
        
        return v_q

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        exp_val = unflatten(exp_val, self.D, self.Nk)
        H_int = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)

        # Direct term (Hartree)
        for alpha in range(2):
            for beta in range(2):
                if alpha == beta:  # Direct term is diagonal in layer/orbital index
                    for idx1, q1 in enumerate(self.q):
                        for idx2, q2 in enumerate(self.q):
                            # Calculate b'-b
                            delta_q = q2 - q1
                            
                            # Calculate V(b'-b)
                            v_q = self.coulomb_potential(delta_q)
                            
                            # Calculate density matrix sum
                            rho_sum = 0
                            for alpha_prime in range(2):
                                for idx_bpp, q_bpp in enumerate(self.q):
                                    # Sum over all k' points
                                    rho_sum += np.mean(exp_val[alpha_prime, idx1+idx_bpp, alpha_prime, idx2+idx_bpp, :])
                            
                            # Add direct term contribution
                            H_int[alpha, idx1, beta, idx2, :] += v_q * rho_sum / self.A

        # Exchange term (Fock)
        for alpha in range(2):
            for beta in range(2):
                for idx1, q1 in enumerate(self.q):
                    for idx2, q2 in enumerate(self.q):
                        for idx_bpp, q_bpp in enumerate(self.q):
                            # Calculate q = b''+k'-k
                            for k_idx in range(self.Nk):
                                k = self.k_space[k_idx]
                                for k_prime_idx in range(self.Nk):
                                    k_prime = self.k_space[k_prime_idx]
                                    
                                    q_exch = q_bpp + (k_prime - k)
                                    v_q = self.coulomb_potential(q_exch)
                                    
                                    # Exchange contribution
                                    H_int[alpha, idx1, beta, idx2, k_idx] -= v_q * exp_val[alpha, idx1+idx_bpp, beta, idx2+idx_bpp, k_prime_idx] / self.A

        return H_int
        
    def generate_Htotal(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the total Hamiltonian including both non-interacting and interacting parts.
        
        This function is required by the solve() function in HF.py.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Total Hamiltonian (non-interacting + interacting)
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        # Return flattened Hamiltonian for the solver
        return flattened(H_total, self.D, self.Nk)
