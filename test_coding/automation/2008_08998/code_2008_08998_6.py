import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with folded Brillouin zone.
    
    This class implements a Hamiltonian with both non-interacting hopping terms and
    interacting Hartree-Fock terms with Coulomb interaction.
    
    Args:
        N_shell (int): Number of shells in the Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2, 3)  # (spin, q-vector)
        self.basis_order = {'0': 'spin', '1': 'q_vector'}
        # Order: 0: spin up, spin down; 1: Gamma, K, K'
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t1 = parameters.get('t1', 6.0)  # nearest-neighbor hopping (meV)
        self.t2 = parameters.get('t2', 1.0)  # next-nearest-neighbor hopping (meV)
        
        # Interaction parameters
        self.epsilon_r = parameters.get('epsilon_r', 1.0)  # relative dielectric constant
        self.d = parameters.get('d', 10.0)  # screening length (nm)
        self.coulomb_const = parameters.get('coulomb_const', 1440.0)  # e^2/epsilon_0 (meV·nm)
        self.U_onsite = parameters.get('U_onsite', 1000.0 / self.epsilon_r)  # on-site interaction (meV)
        
        # Get high symmetry points for q vectors
        self.high_sym_points = generate_high_symmtry_points(self.lattice, self.a)
        self.q_vectors = [
            self.high_sym_points["Gamma"],
            self.high_sym_points["K"],
            self.high_sym_points["K'"]
        ]
        
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
        """Returns the integer coordinate offsets for next-nearest neighbors."""
        nn_vectors = self.get_nearest_neighbor_vectors()
        nnn_vectors = []
        
        # Next-nearest neighbors can be obtained by adding two nearest-neighbor vectors
        for i in range(len(nn_vectors)):
            for j in range(i+1, len(nn_vectors)):
                v1 = nn_vectors[i]
                v2 = nn_vectors[j]
                # Check if the vectors aren't in opposite directions
                if v1 != (-v2[0], -v2[1]):
                    nnn_vector = (v1[0] + v2[0], v1[1] + v2[1])
                    if nnn_vector not in nnn_vectors:
                        nnn_vectors.append(nnn_vector)
        
        return nnn_vectors
    
    def compute_coulomb_interaction(self, q_vector):
        """
        Computes the Coulomb interaction U(q) for a given q vector.
        
        Args:
            q_vector (np.ndarray): The q vector for which to compute the interaction.
            
        Returns:
            float: The Coulomb interaction value.
        """
        q_magnitude = np.linalg.norm(q_vector)
        
        if q_magnitude < 1e-10:  # q ≈ 0 (on-site interaction)
            return self.U_onsite
        else:
            # Screened Coulomb potential: U(q) = (e^2/epsilon_0*epsilon_r) * (1/q - 1/sqrt(q^2 + d^2))
            # in momentum space
            coulomb_factor = self.coulomb_const / self.epsilon_r
            return coulomb_factor * (1/q_magnitude - 1/np.sqrt(q_magnitude**2 + self.d**2))

    def check_momentum_conservation(self, q_alpha, q_beta, q_gamma, q_delta):
        """
        Checks if momentum conservation is satisfied: q_alpha + q_beta = q_gamma + q_delta (mod G).
        
        Args:
            q_alpha, q_beta, q_gamma, q_delta: Indices of q vectors.
            
        Returns:
            bool: True if momentum conservation is satisfied, False otherwise.
        """
        q_sum_in = self.q_vectors[q_alpha] + self.q_vectors[q_beta]
        q_sum_out = self.q_vectors[q_gamma] + self.q_vectors[q_delta]
        
        # Check if difference is a reciprocal lattice vector G
        diff = q_sum_in - q_sum_out
        
        # Get reciprocal lattice vectors
        recip_vectors = get_reciprocal_vectors_triangle(self.a)
        
        # Check if diff is close to any G = n1*G1 + n2*G2
        for n1 in range(-1, 2):
            for n2 in range(-1, 2):
                G = n1 * recip_vectors[0] + n2 * recip_vectors[1]
                if np.linalg.norm(diff - G) < 1e-10:
                    return True
        
        return False

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get nearest and next-nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        nnn_vectors = self.get_next_nearest_neighbor_vectors()
        
        # Calculate hopping terms for each spin and q-vector combination
        for s in range(2):  # spin index
            for q_alpha in range(3):  # source q index
                for q_beta in range(3):  # destination q index
                    q_alpha_vec = self.q_vectors[q_alpha]
                    q_beta_vec = self.q_vectors[q_beta]
                    
                    # Sum over all k points in the Brillouin zone
                    for k_idx in range(self.N_k):
                        p = self.k_space[k_idx]
                        
                        # Nearest-neighbor hopping contribution
                        t_nn = 0
                        for n in nn_vectors:
                            R_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                            phase = np.exp(-1j * np.dot(p + q_beta_vec - q_alpha_vec, R_n))
                            t_nn += self.t1 * phase
                        
                        # Next-nearest-neighbor hopping contribution
                        t_nnn = 0
                        for n in nnn_vectors:
                            R_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                            phase = np.exp(-1j * np.dot(p + q_beta_vec - q_alpha_vec, R_n))
                            t_nnn += self.t2 * phase
                        
                        # Total hopping term (negative sign from the Hamiltonian)
                        H_nonint[s, q_alpha, s, q_beta, k_idx] = -(t_nn + t_nnn)
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian (Hartree-Fock terms).
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Precompute Coulomb interaction for all possible q-vector differences
        U_q = {}
        for q_alpha in range(3):
            for q_delta in range(3):
                q_diff = self.q_vectors[q_alpha] - self.q_vectors[q_delta]
                U_q[(q_alpha, q_delta)] = self.compute_coulomb_interaction(q_diff)
        
        # Calculate Hartree term
        for s in range(2):  # spin index
            for s_prime in range(2):  # second spin index
                for q_alpha in range(3):  # q_alpha index
                    for q_beta in range(3):  # q_beta index
                        for q_gamma in range(3):  # q_gamma index
                            for q_delta in range(3):  # q_delta index
                                # Check momentum conservation
                                if self.check_momentum_conservation(q_alpha, q_beta, q_gamma, q_delta):
                                    # Coulomb interaction U(q_alpha - q_delta)
                                    U_hartree = U_q[(q_alpha, q_delta)]
                                    
                                    # Calculate expectation value <c^\dagger_q_alpha,s c_q_delta,s>
                                    n_hartree = np.mean(exp_val[s, q_alpha, s, q_delta, :])
                                    
                                    # Add Hartree contribution to H_int for all k points
                                    for k_idx in range(self.N_k):
                                        # Hartree term: U(q_α-q_δ) <c^\dagger_q_α,s c_q_δ,s> c^\dagger_q_β,s' c_q_γ,s'
                                        H_int[s_prime, q_beta, s_prime, q_gamma, k_idx] += (
                                            (1.0 / self.N_k) * U_hartree * n_hartree
                                        )
        
        # Calculate Fock term
        for s in range(2):  # spin index
            for s_prime in range(2):  # second spin index
                for q_alpha in range(3):  # q_alpha index
                    for q_beta in range(3):  # q_beta index
                        for q_gamma in range(3):  # q_gamma index
                            for q_delta in range(3):  # q_delta index
                                # Check momentum conservation
                                if self.check_momentum_conservation(q_alpha, q_beta, q_gamma, q_delta):
                                    # For the Fock term, the interaction depends on p_α+q_α-p_β-q_δ
                                    # As a simplification, we'll use U(q_alpha-q_gamma) for all p
                                    U_fock = U_q[(q_alpha, q_gamma)]
                                    
                                    # Calculate expectation value <c^\dagger_q_alpha,s c_q_gamma,s'>
                                    n_fock = np.mean(exp_val[s, q_alpha, s_prime, q_gamma, :])
                                    
                                    # Add Fock contribution to H_int for all k points
                                    for k_idx in range(self.N_k):
                                        # Fock term: -U(p_α+q_α-p_β-q_δ) <c^\dagger_q_α,s c_q_γ,s'> c^\dagger_q_β,s' c_q_δ,s
                                        # Note the negative sign
                                        H_int[s_prime, q_beta, s, q_delta, k_idx] -= (
                                            (1.0 / self.N_k) * U_fock * n_fock
                                        )
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian.
            
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
