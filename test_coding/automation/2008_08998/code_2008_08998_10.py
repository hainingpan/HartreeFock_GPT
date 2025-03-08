import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with spin and
    high-symmetry points in the reciprocal space.
    
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2, 3)  # (|spin|, |reciprocal_lattice_vector|)
        self.basis_order = {
            '0': 'spin',
            '1': 'reciprocal_lattice_vector'
        }
        # Order for each flavor:
        # 0: spin: spin_up (↑), spin_down (↓)
        # 1: reciprocal_lattice_vector: Γ, K, K'
        
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
        
        # Coulomb interaction parameters
        self.epsilon_r = parameters.get('epsilon_r', 1.0)  # Relative dielectric constant
        self.coulomb_const = 1440.0  # e^2/ε_0 in meV·nm
        self.screening_length = parameters.get('d', 10.0)  # Screening length in nm
        self.onsite_U = parameters.get('onsite_U', 1000.0 / self.epsilon_r)  # Onsite interaction (meV)
        
        # Generate high-symmetry points for q vectors
        self.high_sym_points = generate_high_symmtry_points(self.lattice, self.a)
        self.q_vectors = np.array([
            self.high_sym_points["Gamma"],  # Γ point
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
    
    def calculate_interaction_potential(self, p):
        """
        Calculate the interaction potential U(p) in momentum space.
        
        Args:
            p (np.ndarray): Momentum vector.
            
        Returns:
            float: Interaction potential value.
        """
        # For p=0, return the onsite interaction
        if np.all(np.abs(p) < 1e-10):
            return self.onsite_U
        
        # For p≠0, calculate the potential from the Fourier transform
        # of the real-space potential U(r) = e^2/(ε_0 ε_r) (1/r - 1/sqrt(r^2+d^2))
        
        # We need to sum over lattice sites, but since the potential decays with distance,
        # we can limit the sum to a reasonable number of neighbors
        n_max = 10  # Consider neighbors up to n_max*a distance
        potential = 0.0
        
        for n1 in range(-n_max, n_max+1):
            for n2 in range(-n_max, n_max+1):
                R_n = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
                r = np.linalg.norm(R_n)
                
                if r > 0:  # Exclude r=0 case (handled separately)
                    # Real-space potential: U(r) = e^2/(ε_0 ε_r) (1/r - 1/sqrt(r^2+d^2))
                    U_r = self.coulomb_const / self.epsilon_r * (1/r - 1/np.sqrt(r**2 + self.screening_length**2))
                    # Fourier transform: U(p) = Σ_R U(R) exp(i p·R)
                    potential += U_r * np.exp(1j * np.dot(p, R_n))
        
        return potential.real  # Return real part (imaginary should be zero for real U(r))
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (*self.D, *self.D, self.N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get nearest and next-nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Calculate next-nearest neighbor vectors
        nnn_vectors = []
        for i, n1 in enumerate(nn_vectors):
            for n2 in nn_vectors[i+1:]:
                n_sum = (n1[0] + n2[0], n1[1] + n2[1])
                if n_sum not in nnn_vectors and (-n_sum[0], -n_sum[1]) not in nnn_vectors:
                    nnn_vectors.append(n_sum)
        
        # Calculate hopping terms for each k-point
        for k_idx, k in enumerate(self.k_space):
            for s in range(self.D[0]):  # Spin index
                for q1 in range(self.D[1]):  # q index for creation operator
                    for q2 in range(self.D[1]):  # q index for annihilation operator
                        # Momentum conservation: same q vectors for creation and annihilation
                        if q1 == q2:
                            # Calculate sum over lattice sites
                            hopping_term = 0.0
                            
                            # Nearest-neighbor hopping
                            for n in nn_vectors:
                                R_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                                phase = np.exp(-1j * np.dot(k + self.q_vectors[q1], R_n))
                                hopping_term += self.t1 * phase
                            
                            # Next-nearest-neighbor hopping
                            for n in nnn_vectors:
                                R_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                                phase = np.exp(-1j * np.dot(k + self.q_vectors[q1], R_n))
                                hopping_term += self.t2 * phase
                            
                            # Assign to Hamiltonian
                            H_nonint[s, q1, s, q2, k_idx] = -hopping_term
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian (Hartree and Fock terms).
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (*self.D, *self.D, self.N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Process each k-point
        for k_idx, k in enumerate(self.k_space):
            # For each flavor combination
            for s1 in range(self.D[0]):  # Spin of creation operator
                for q_beta in range(self.D[1]):  # q vector of creation operator
                    for s2 in range(self.D[0]):  # Spin of annihilation operator
                        for q_gamma in range(self.D[1]):  # q vector of annihilation operator
                            
                            # Hartree term: Contributes when spins are the same
                            if s1 == s2:
                                for s in range(self.D[0]):  # Spin in expectation value
                                    for q_alpha in range(self.D[1]):  # q vector in expectation value (creation)
                                        for q_delta in range(self.D[1]):  # q vector in expectation value (annihilation)
                                            
                                            # Check momentum conservation: q_alpha + q_beta = q_gamma + q_delta
                                            # This is simplified here - in practice, would need to check with modulo reciprocal lattice vectors
                                            vector_sum1 = self.q_vectors[q_alpha] + self.q_vectors[q_beta]
                                            vector_sum2 = self.q_vectors[q_gamma] + self.q_vectors[q_delta]
                                            
                                            if np.allclose(vector_sum1, vector_sum2, atol=1e-10):
                                                # Calculate interaction potential
                                                U_q = self.calculate_interaction_potential(
                                                    self.q_vectors[q_alpha] - self.q_vectors[q_delta]
                                                )
                                                
                                                # Hartree term
                                                H_int[s1, q_beta, s2, q_gamma, k_idx] += (
                                                    U_q / self.N_k * 
                                                    exp_val[s, q_alpha, s, q_delta, k_idx]
                                                )
                            
                            # Fock term
                            for s in range(self.D[0]):  # First spin in expectation value
                                for sprime in range(self.D[0]):  # Second spin in expectation value
                                    for q_alpha in range(self.D[1]):  # q vector in expectation value (creation)
                                        for q_gamma_prime in range(self.D[1]):  # q vector in expectation value (annihilation)
                                            
                                            # Check momentum conservation and spin conservation
                                            vector_sum1 = self.q_vectors[q_alpha] + self.q_vectors[q_beta]
                                            vector_sum2 = self.q_vectors[q_gamma_prime] + self.q_vectors[q_gamma]
                                            
                                            if np.allclose(vector_sum1, vector_sum2, atol=1e-10) and s1 == sprime and s2 == s:
                                                # Calculate interaction potential with momentum transfer
                                                U_p = self.calculate_interaction_potential(
                                                    k + self.q_vectors[q_alpha] - k - self.q_vectors[q_gamma_prime]
                                                )
                                                
                                                # Fock term (negative sign)
                                                H_int[s1, q_beta, s2, q_gamma, k_idx] -= (
                                                    U_p / self.N_k * 
                                                    exp_val[s, q_alpha, sprime, q_gamma_prime, k_idx]
                                                )
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            return_flat (bool): Whether to return the flattened Hamiltonian. Default is True.
            
        Returns:
            np.ndarray: The total Hamiltonian, either flattened or in high-rank form.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
