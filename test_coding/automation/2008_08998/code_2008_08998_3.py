import numpy as np
from typing import Any, Tuple, Dict, List
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with spin and reciprocal lattice vectors in a triangular lattice.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor for the system. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2, 3)  # (spin, q-vector)
        self.basis_order = {
            '0': 'spin',
            '1': 'q-vector'
        }
        # spin: up, down
        # q-vector: Gamma, K, K'
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # High symmetry points in the original BZ
        self.high_sym_points = generate_high_symmtry_points(self.lattice, self.a)
        self.q_vectors = [
            self.high_sym_points["Gamma"],  # Gamma point
            self.high_sym_points["K"],      # K point
            self.high_sym_points["K'"]      # K' point
        ]
        
        # Hopping parameters
        self.t1 = parameters.get('t1', 6.0)  # nearest-neighbor hopping (meV)
        self.t2 = parameters.get('t2', 1.0)  # next-nearest-neighbor hopping (meV)
        
        # Interaction parameters
        self.epsilon_r = parameters.get('epsilon_r', 10.0)  # relative dielectric constant
        self.coulomb_const = 1440.0  # e^2/epsilon_0 in meV·nm
        self.d = parameters.get('d', 10.0)  # screening length in nm
        self.U_onsite = 1000.0 / self.epsilon_r  # onsite repulsion in meV
        
        # Precompute interaction potential in real space and momentum space
        self.U_momentum = self._compute_interaction_potential()
        
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
    
    def _compute_interaction_potential(self):
        """Compute the interaction potential in momentum space."""
        # Get real space lattice positions up to a certain range
        max_range = 20  # Adjust this for accuracy vs performance
        real_space_positions = []
        for i in range(-max_range, max_range+1):
            for j in range(-max_range, max_range+1):
                if i == 0 and j == 0:
                    continue  # Skip origin for now
                pos = i * self.primitive_vectors[0] + j * self.primitive_vectors[1]
                r = np.linalg.norm(pos)
                real_space_positions.append((pos, r))
        
        # Compute U(r) for each position
        U_real = {}
        for pos, r in real_space_positions:
            U_r = (self.coulomb_const / self.epsilon_r) * (1/r - 1/np.sqrt(r**2 + self.d**2))
            U_real[tuple(pos)] = U_r
        
        # Add onsite term
        U_real[(0.0, 0.0)] = self.U_onsite
        
        # Compute U(q) for each q vector pair
        U_q = np.zeros((3, 3), dtype=complex)
        for qa_idx in range(3):
            for qd_idx in range(3):
                q_diff = self.q_vectors[qa_idx] - self.q_vectors[qd_idx]
                U_q_val = 0.0
                for pos, r in real_space_positions:
                    phase = np.exp(1j * np.dot(q_diff, pos))
                    U_q_val += U_real[tuple(pos)] * phase
                # Add onsite term
                U_q_val += self.U_onsite
                U_q[qa_idx, qd_idx] = U_q_val
        
        return U_q
    
    def _compute_hopping_amplitude(self, k_vector, q_i, q_j):
        """
        Compute the hopping amplitude between q_i and q_j at crystal momentum k.
        
        Args:
            k_vector (np.ndarray): Crystal momentum in the folded BZ
            q_i (np.ndarray): Initial reciprocal lattice vector
            q_j (np.ndarray): Final reciprocal lattice vector
            
        Returns:
            complex: Hopping amplitude
        """
        # Get nearest and next-nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        nnn_vectors = []
        for n1 in nn_vectors:
            for n2 in nn_vectors:
                vec = (n1[0] + n2[0], n1[1] + n2[1])
                if vec not in nn_vectors and vec != (0, 0) and vec not in nnn_vectors:
                    nnn_vectors.append(vec)
        
        # Calculate hopping amplitude
        amplitude = 0.0
        
        # Nearest-neighbor hopping
        for n_vec in nn_vectors:
            R_n = n_vec[0] * self.primitive_vectors[0] + n_vec[1] * self.primitive_vectors[1]
            phase = np.exp(-1j * np.dot(k_vector + q_j - q_i, R_n))
            amplitude += self.t1 * phase
        
        # Next-nearest-neighbor hopping
        for n_vec in nnn_vectors:
            R_n = n_vec[0] * self.primitive_vectors[0] + n_vec[1] * self.primitive_vectors[1]
            phase = np.exp(-1j * np.dot(k_vector + q_j - q_i, R_n))
            amplitude += self.t2 * phase
        
        return amplitude
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Loop over all k-points, spins, and q-vectors
        for k_idx in range(self.N_k):
            k_vector = self.k_space[k_idx]
            
            for s in range(2):  # spin up, down
                for q_i in range(3):  # Gamma, K, K'
                    for q_j in range(3):  # Gamma, K, K'
                        # Calculate hopping term
                        hopping_amplitude = self._compute_hopping_amplitude(
                            k_vector, self.q_vectors[q_i], self.q_vectors[q_j])
                        
                        # Set the non-interacting Hamiltonian element
                        H_nonint[s, q_i, s, q_j, k_idx] = -hopping_amplitude
        
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
        
        # Calculate average expectation values over all k-points
        density_matrix = np.zeros((2, 3, 2, 3), dtype=complex)  # [s, q_alpha, s', q_delta]
        
        for k_idx in range(self.N_k):
            for s in range(2):  # spin index
                for q_alpha in range(3):  # q_alpha index
                    for s_prime in range(2):  # s' index
                        for q_delta in range(3):  # q_delta index
                            density_matrix[s, q_alpha, s_prime, q_delta] += exp_val[s, q_alpha, s_prime, q_delta, k_idx]
        
        density_matrix /= self.N_k  # Average over all k-points
        
        # Loop over all k-points to calculate the interacting Hamiltonian
        for k_idx in range(self.N_k):
            # Momenta conservation constraint (delta function)
            # For each (q_alpha, q_beta, q_gamma, q_delta) where q_alpha + q_beta = q_gamma + q_delta (mod G)
            for q_alpha in range(3):
                for q_beta in range(3):
                    for q_gamma in range(3):
                        q_delta = (q_alpha + q_beta - q_gamma) % 3  # Conservation of momentum mod 3
                        
                        # Hartree term
                        for s in range(2):  # s index
                            for s_prime in range(2):  # s' index
                                # U(q_alpha - q_delta) * <c^†_q_alpha,s c_q_delta,s> * c^†_q_beta,s' c_q_gamma,s'
                                hartree_factor = self.U_momentum[q_alpha, q_delta] * density_matrix[s, q_alpha, s, q_delta]
                                H_int[s_prime, q_beta, s_prime, q_gamma, k_idx] += hartree_factor / self.N_k
                        
                        # Fock term
                        for s in range(2):  # s index
                            for s_prime in range(2):  # s' index
                                # -U(p_alpha + q_alpha - p_beta - q_delta) * <c^†_q_alpha,s c_q_gamma,s'> * c^†_q_beta,s' c_q_delta,s
                                # Simplify by using an average interaction potential for momentum dependence
                                fock_factor = -self.U_momentum[q_alpha, q_gamma] * density_matrix[s, q_alpha, s_prime, q_gamma]
                                H_int[s_prime, q_beta, s, q_delta, k_idx] += fock_factor / self.N_k
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian.
            
        Returns:
            np.ndarray: The total Hamiltonian with appropriate shape.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
