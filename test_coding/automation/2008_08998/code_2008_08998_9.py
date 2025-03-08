import numpy as np
from typing import Any, Dict, Tuple, List
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a triangular lattice with a sqrt(3) × sqrt(3) superlattice.
    
    This class implements the Hamiltonian for a system with spin and reciprocal lattice vector
    degrees of freedom, with both tight-binding terms and Coulomb interactions treated
    at the Hartree-Fock level.
    
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Model parameters including hopping and interaction strengths.
        filling_factor (float, optional): Filling factor of the system. Defaults to 0.5.
    """
    def __init__(self, N_shell, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2, 3)  # (spin, reciprocal_lattice_vector)
        self.basis_order = {'0': 'spin', '1': 'reciprocal_lattice_vector'}
        # 0: spin up, spin down
        # 1: Gamma, K, K'
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Tight-binding parameters
        self.t1 = parameters.get('t1', 6.0)  # meV, nearest-neighbor hopping
        self.t2 = parameters.get('t2', 1.0)  # meV, next-nearest-neighbor hopping
        
        # Coulomb interaction parameters
        self.d = parameters.get('d', 10.0)  # nm, screening length
        self.coulomb_constant = parameters.get('coulomb_constant', 1440.0)  # meV·nm
        self.epsilon_r = parameters.get('epsilon_r', 1.0)  # relative dielectric constant
        self.U0 = parameters.get('U0', 1000.0/self.epsilon_r)  # meV, on-site repulsion
        
        # High symmetry points in reciprocal space
        self.high_sym_points = generate_high_symmtry_points(self.lattice, self.a)
        self.q_vectors = np.array([
            self.high_sym_points["Gamma"],  # Gamma point
            self.high_sym_points["K"],      # K point
            self.high_sym_points["K'"]      # K' point
        ])
        
        return

    def get_nearest_neighbor_vectors(self):
        """
        Returns the integer coordinate offsets (n1, n2) corresponding to the 
        nearest neighbors in a 2D triangular Bravais lattice. These offsets are ONLY
        valid for a lattice whose two primitive vectors are separated by 120°.

        To obtain the real-space displacements for each neighbor, multiply these 
        integer pairs by the primitive vectors a1 and a2, i.e.:
            R_neighbor = n1 * a1 + n2 * a2

        For a 2D triangular lattice, there are six nearest neighbors, given by:
        """
        n_vectors = [
            (1, 0),
            (0, 1),
            (1, 1),
            (-1, 0),
            (0, -1),
            (-1, -1),
        ]
        return n_vectors
    
    def get_next_nearest_neighbor_vectors(self):
        """Returns the integer coordinate offsets for next-nearest neighbors in a 2D triangular lattice."""
        n_vectors = [
            (2, 0),
            (0, 2),
            (2, 2),
            (-2, 0),
            (0, -2),
            (-2, -2),
            (1, -1),
            (-1, 1),
            (2, 1),
            (1, 2),
            (-2, -1),
            (-1, -2),
        ]
        return n_vectors
    
    def compute_coulomb_potential(self, q_diff):
        """
        Computes the Coulomb potential in momentum space for a given q difference.
        
        Args:
            q_diff: Difference between two q vectors
            
        Returns:
            float: The Coulomb potential value U(q_diff)
        """
        # For simplicity, we use a model Coulomb potential
        # In a full implementation, this would be a proper Fourier transform of the real-space potential
        q_mag = np.linalg.norm(q_diff)
        if q_mag < 1e-10:
            return self.U0  # On-site repulsion
        else:
            # Screened Coulomb potential in momentum space
            return self.coulomb_constant / self.epsilon_r * (1.0 / q_mag - 1.0 / np.sqrt(q_mag**2 + 1.0/self.d**2))
    
    def check_momentum_conservation(self, q_alpha_idx, q_beta_idx, q_gamma_idx, q_delta_idx):
        """
        Checks if momentum conservation is satisfied: q_alpha + q_beta = q_gamma + q_delta + G
        where G is a reciprocal lattice vector.
        
        Args:
            q_alpha_idx, q_beta_idx, q_gamma_idx, q_delta_idx: Indices for q-vectors
            
        Returns:
            bool: True if momentum conservation is satisfied
        """
        q_alpha = self.q_vectors[q_alpha_idx]
        q_beta = self.q_vectors[q_beta_idx]
        q_gamma = self.q_vectors[q_gamma_idx]
        q_delta = self.q_vectors[q_delta_idx]
        
        # Check if q_alpha + q_beta - q_gamma - q_delta is a reciprocal lattice vector
        # For simplicity, we'll check if it's close to zero or any of the high symmetry points
        q_diff = q_alpha + q_beta - q_gamma - q_delta
        
        # Check against possible reciprocal lattice vectors (including zero)
        for G in [np.array([0, 0])]:  # Could include other reciprocal lattice vectors
            if np.allclose(q_diff, G, atol=1e-10):
                return True
        
        return False

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get nearest and next-nearest neighbor vectors in real space
        nn_vectors = self.get_nearest_neighbor_vectors()
        nnn_vectors = self.get_next_nearest_neighbor_vectors()
        
        # Convert integer offsets to real space vectors
        nn_real = []
        for n in nn_vectors:
            R_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            nn_real.append(R_n)
            
        nnn_real = []
        for n in nnn_vectors:
            R_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            nnn_real.append(R_n)
        
        # Compute hopping terms for each spin, q-vector, and momentum
        for s in range(2):  # Spin
            for q1 in range(3):  # Creation q-vector
                for q2 in range(3):  # Annihilation q-vector
                    for k_idx in range(self.N_k):  # Crystal momentum p
                        p = self.k_space[k_idx]
                        
                        # Nearest-neighbor hopping
                        for R_n in nn_real:
                            phase = np.exp(-1j * np.dot(p + self.q_vectors[q1], R_n))
                            if q1 == q2:  # Only diagonal terms in q-space
                                H_nonint[s, q1, s, q2, k_idx] += -self.t1 * phase
                        
                        # Next-nearest-neighbor hopping
                        for R_n in nnn_real:
                            phase = np.exp(-1j * np.dot(p + self.q_vectors[q1], R_n))
                            if q1 == q2:  # Only diagonal terms in q-space
                                H_nonint[s, q1, s, q2, k_idx] += -self.t2 * phase
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian using Hartree-Fock mean field theory.
        
        Args:
            exp_val (np.ndarray): Expectation value tensor.
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Average values of exp_val for the Hartree and Fock terms
        exp_val_avg = np.zeros((2, 3, 2, 3), dtype=complex)  # (s, q_alpha, s', q_gamma)
        for s in range(2):
            for q_alpha in range(3):
                for s_prime in range(2):
                    for q_gamma in range(3):
                        exp_val_avg[s, q_alpha, s_prime, q_gamma] = np.mean(exp_val[s, q_alpha, s_prime, q_gamma, :])
        
        # Hartree term
        for s in range(2):  # Target spin
            for s_prime in range(2):  # Source spin
                for q_alpha in range(3):  # Source creation q
                    for q_delta in range(3):  # Source annihilation q
                        for q_beta in range(3):  # Target creation q
                            for q_gamma in range(3):  # Target annihilation q
                                # Check momentum conservation
                                if self.check_momentum_conservation(q_alpha, q_beta, q_gamma, q_delta):
                                    # Compute Coulomb interaction for this q difference
                                    U_q = self.compute_coulomb_potential(self.q_vectors[q_alpha] - self.q_vectors[q_delta])
                                    
                                    # Add Hartree term to Hamiltonian (density-density interaction)
                                    for k_idx in range(self.N_k):
                                        H_int[s, q_beta, s, q_gamma, k_idx] += (
                                            (1.0/self.N_k) * U_q * exp_val_avg[s_prime, q_alpha, s_prime, q_delta]
                                        )
        
        # Fock term
        for s in range(2):  # Target creation spin
            for s_prime in range(2):  # Target annihilation spin
                for s_source in range(2):  # Source creation spin
                    for s_source_prime in range(2):  # Source annihilation spin
                        if s == s_source and s_prime == s_source_prime:  # Only consider terms with matching spins
                            for q_alpha in range(3):  # Source creation q
                                for q_gamma in range(3):  # Source annihilation q
                                    for q_beta in range(3):  # Target creation q
                                        for q_delta in range(3):  # Target annihilation q
                                            # Check momentum conservation
                                            if self.check_momentum_conservation(q_alpha, q_beta, q_gamma, q_delta):
                                                for k_idx in range(self.N_k):
                                                    # For each p, we need U(p_alpha + q_alpha - p_beta - q_delta)
                                                    # For simplicity, we use just the q difference
                                                    U_p_q = self.compute_coulomb_potential(
                                                        self.q_vectors[q_alpha] - self.q_vectors[q_delta]
                                                    )
                                                    
                                                    # Add Fock term (exchange interaction)
                                                    H_int[s, q_beta, s_prime, q_delta, k_idx] -= (
                                                        (1.0/self.N_k) * U_p_q * exp_val_avg[s_source, q_alpha, s_source_prime, q_gamma]
                                                    )
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian by combining non-interacting and interacting parts.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            return_flat (bool, optional): Whether to return a flattened Hamiltonian. Defaults to True.
            
        Returns:
            np.ndarray: Total Hamiltonian, either flattened or in full tensor form.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
