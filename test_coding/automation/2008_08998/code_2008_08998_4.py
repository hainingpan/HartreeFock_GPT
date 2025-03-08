import numpy as np
from typing import Any, Dict, Tuple
from scipy.spatial.distance import cdist
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a system with spin and valley degrees of freedom
    
    Implements the Hamiltonian for a triangular lattice with electron interactions,
    including tight-binding, Hartree, and Fock terms.
    
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2, 3)  # (spin, q-vector)
        self.basis_order = {
            '0': 'spin',
            '1': 'reciprocal_lattice_vector'
        }
        # Basis order details:
        # spin: 0 = up, 1 = down
        # reciprocal_lattice_vector: 0 = Γ, 1 = K, 2 = K'

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Hopping parameters
        self.t1 = parameters.get('t1', 6.0)  # Nearest-neighbor hopping (meV)
        self.t2 = parameters.get('t2', 1.0)  # Next-nearest-neighbor hopping (meV)

        # Interaction parameters
        self.epsilon_r = parameters.get('epsilon_r', 1.0)  # Relative dielectric constant
        self.d = parameters.get('d', 10.0)  # Screening length (nm)
        self.coulomb_const = parameters.get('coulomb_const', 1440.0)  # e²/ε₀ in meV·nm
        self.u_onsite = parameters.get('u_onsite', 1000.0 / self.epsilon_r)  # On-site interaction (meV)
        
        # Define high symmetry points (Γ, K, K')
        self.high_sym_points = self._get_high_symmetry_points()
        
        return

    def _get_high_symmetry_points(self):
        """Get the three high symmetry points Γ, K, K'"""
        points = generate_high_symmtry_points(self.lattice, self.a)
        return np.array([points["Gamma"], points["K"], points["K'"]])
        
    def get_nearest_neighbor_vectors(self):
        """
        Returns the integer coordinate offsets (n1, n2) corresponding to the 
        nearest neighbors in a 2D triangular Bravais lattice. These offsets are ONLY
        valid for a lattice whose two primitive vectors are separated by 120°.

        To obtain the real-space displacements for each neighbor, multiply these 
        integer pairs by the primitive vectors a1 and a2, i.e.:
            R_neighbor = n1 * a1 + n2 * a2

        For a 2D triangular lattice, there are six nearest neighbors.
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
        """Returns the integer coordinate offsets for next-nearest neighbors"""
        nn_vectors = [
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
            (-1, -2)
        ]
        return nn_vectors
    
    def compute_interaction_potential(self, q_diff):
        """Compute interaction potential U(q) for a given momentum difference"""
        # For simplicity, we're using a screened Coulomb potential in momentum space
        # In a more detailed implementation, this would involve Fourier transforms of the real-space potential
        r_space_positions = get_shell_index_triangle(5)  # Sample over a grid of real-space positions
        
        # Convert to real coordinates
        r_vectors = []
        for i, j in zip(*r_space_positions):
            r = i * self.primitive_vectors[0] + j * self.primitive_vectors[1]
            r_vectors.append(r)
        r_vectors = np.array(r_vectors)
        
        # Calculate real-space potential
        u_r = np.zeros(len(r_vectors))
        for i, r in enumerate(r_vectors):
            r_norm = np.linalg.norm(r)
            if r_norm > 0:
                u_r[i] = self.coulomb_const / self.epsilon_r * (1/r_norm - 1/np.sqrt(r_norm**2 + self.d**2))
            else:
                u_r[i] = self.u_onsite
        
        # Compute Fourier transform to get U(q)
        u_q = 0
        for i, r in enumerate(r_vectors):
            u_q += u_r[i] * np.exp(1j * np.dot(q_diff, r))
        
        return u_q / len(r_vectors)  # Normalize
    
    def momentum_conserving_delta(self, q_sum, allowed_g=None):
        """Check if momentum is conserved up to a reciprocal lattice vector"""
        if allowed_g is None:
            # Define a few reciprocal lattice vectors to check against
            g_vectors = get_reciprocal_vectors_triangle(self.a)
            allowed_g = [np.array([0, 0])]  # Start with G = 0
            
            # Add some small integer multiples of reciprocal vectors
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i != 0 or j != 0:  # Skip the zero vector
                        allowed_g.append(i * g_vectors[0] + j * g_vectors[1])
        
        # Check if q_sum equals any of the allowed G vectors (within numerical precision)
        for g in allowed_g:
            if np.linalg.norm(q_sum - g) < 1e-6:
                return True
        
        return False

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get real-space positions for nearest and next-nearest neighbors
        nn_vectors = self.get_nearest_neighbor_vectors()
        nnn_vectors = self.get_next_nearest_neighbor_vectors()
        
        # Convert to real coordinates
        nn_positions = []
        for n1, n2 in nn_vectors:
            r = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
            nn_positions.append(r)
            
        nnn_positions = []
        for n1, n2 in nnn_vectors:
            r = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
            nnn_positions.append(r)
        
        # Loop over all flavors and k-points
        for s in range(2):  # Spin
            for q_idx in range(3):  # q-vector
                # Only non-zero terms are between q and Gamma (q=0)
                if q_idx == 0:  # q = Gamma
                    # Diagonal terms (no hopping)
                    H_nonint[s, q_idx, s, q_idx, :] = 0  # On-site energy (set to 0)
                else:
                    q_vec = self.high_sym_points[q_idx]
                    
                    # Nearest-neighbor hopping
                    for k_idx, k in enumerate(self.k_space):
                        # Sum over nearest neighbors
                        hopping_sum = 0
                        for r in nn_positions:
                            hopping_sum += self.t1 * np.exp(-1j * np.dot(k + q_vec, r))
                        
                        # Sum over next-nearest neighbors
                        for r in nnn_positions:
                            hopping_sum += self.t2 * np.exp(-1j * np.dot(k + q_vec, r))
                        
                        # Assign to Hamiltonian
                        H_nonint[s, q_idx, s, 0, k_idx] = -hopping_sum
                        H_nonint[s, 0, s, q_idx, k_idx] = -np.conjugate(hopping_sum)  # Hermitian conjugate
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Loop over all flavor combinations
        for s1 in range(2):  # Spin 1
            for s2 in range(2):  # Spin 2
                for q_alpha in range(3):  # q_alpha
                    for q_beta in range(3):  # q_beta
                        for q_gamma in range(3):  # q_gamma
                            for q_delta in range(3):  # q_delta
                                
                                # Check momentum conservation
                                q_sum = self.high_sym_points[q_alpha] + self.high_sym_points[q_beta] - self.high_sym_points[q_gamma] - self.high_sym_points[q_delta]
                                if not self.momentum_conserving_delta(q_sum):
                                    continue  # Skip if momentum not conserved
                                
                                # For Hartree term (same spin)
                                if s1 == s2:
                                    # Compute U(q_alpha - q_delta)
                                    q_diff = self.high_sym_points[q_alpha] - self.high_sym_points[q_delta]
                                    U_q = self.compute_interaction_potential(q_diff)
                                    
                                    # Compute expectation value
                                    for k_idx in range(self.N_k):
                                        exp_val_hartree = np.mean(exp_val[s1, q_alpha, s1, q_delta, :])
                                        H_int[s2, q_beta, s2, q_gamma, k_idx] += (1/self.N_k) * U_q * exp_val_hartree
                                
                                # For Fock term (can be different spins)
                                # Compute U(p_alpha + q_alpha - p_beta - q_delta)
                                # For simplicity, we're using an average p_alpha and p_beta
                                for k_idx in range(self.N_k):
                                    p_avg = self.k_space[k_idx]
                                    q_diff = p_avg + self.high_sym_points[q_alpha] - p_avg - self.high_sym_points[q_delta]
                                    U_q = self.compute_interaction_potential(q_diff)
                                    
                                    # Compute expectation value
                                    exp_val_fock = np.mean(exp_val[s1, q_alpha, s2, q_gamma, :])
                                    H_int[s2, q_beta, s1, q_delta, k_idx] -= (1/self.N_k) * U_q * exp_val_fock
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the Hamiltonian in flattened form.

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
