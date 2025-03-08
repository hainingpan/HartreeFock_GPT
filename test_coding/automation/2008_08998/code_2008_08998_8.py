import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with folded Brillouin zone.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2, 3)  # (spin, q-vector)
        self.basis_order = {
            '0': 'spin',           # up, down
            '1': 'q-vector'        # Gamma, K, K'
        }
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t1 = parameters.get('t1', 6.0)  # nearest-neighbor hopping in meV
        self.t2 = parameters.get('t2', 1.0)  # next-nearest-neighbor hopping in meV
        
        # Interaction parameters
        self.e2_over_epsilon0 = parameters.get('e2_over_epsilon0', 1440.0)  # Coulomb constant in meV·nm
        self.epsilon_r = parameters.get('epsilon_r', 1.0)  # relative dielectric constant
        self.d = parameters.get('d', 10.0)  # screening length in nm
        self.U_onsite = parameters.get('U_onsite', 1000.0 / self.epsilon_r)  # onsite interaction in meV
        
        # High symmetry points
        self.high_symmetry = generate_high_symmtry_points(self.lattice, self.a)
        self.q_vectors = [
            self.high_symmetry['Gamma'],  # Gamma point
            self.high_symmetry['K'],      # K point
            self.high_symmetry['K\'']     # K' point
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
        """
        Returns the integer coordinate offsets for next-nearest neighbors
        in a 2D triangular Bravais lattice.
        """
        nn_vectors = [
            (2, 0),
            (1, -1),
            (-1, -2),
            (-2, -1),
            (-1, 1),
            (0, 2)
        ]
        return nn_vectors
    
    def _compute_interaction_potential(self, r):
        """
        Computes the interaction potential U(r) in real space.
        
        Args:
            r (float): Distance in nm.
            
        Returns:
            float: Interaction potential at distance r in meV.
        """
        if r < 1e-10:  # Essentially r=0
            return self.U_onsite
        
        return self.e2_over_epsilon0 / self.epsilon_r * (1/r - 1/np.sqrt(r**2 + self.d**2))
    
    def _compute_interaction_potential_k(self, k_vector):
        """
        Computes the interaction potential U(k) in momentum space.
        
        This is a simplified implementation that approximates the Fourier transform
        of the real-space potential.
        
        Args:
            k_vector (numpy.ndarray): Momentum vector.
            
        Returns:
            float: Interaction potential in momentum space in meV.
        """
        k = np.linalg.norm(k_vector)
        if k < 1e-10:  # Avoid division by zero
            return self.U_onsite
        
        # Simplified 2D Fourier transform of the screened Coulomb potential
        return self.e2_over_epsilon0 / self.epsilon_r * 2*np.pi/k * (1 - np.exp(-k*self.d))
    
    def _check_momentum_conservation(self, q_alpha_idx, q_beta_idx, q_gamma_idx, q_delta_idx):
        """
        Checks if momentum conservation is satisfied up to a reciprocal lattice vector.
        
        Args:
            q_alpha_idx, q_beta_idx, q_gamma_idx, q_delta_idx: Indices of q-vectors.
            
        Returns:
            bool: True if momentum is conserved, False otherwise.
        """
        q_alpha = self.q_vectors[q_alpha_idx]
        q_beta = self.q_vectors[q_beta_idx]
        q_gamma = self.q_vectors[q_gamma_idx]
        q_delta = self.q_vectors[q_delta_idx]
        
        # Check if q_alpha + q_beta = q_gamma + q_delta (up to a reciprocal lattice vector)
        q_sum_in = q_alpha + q_beta
        q_sum_out = q_gamma + q_delta
        
        # Get reciprocal vectors
        recip_vectors = get_reciprocal_vectors_triangle(self.a)
        
        # Check if the difference is a reciprocal lattice vector
        diff = q_sum_in - q_sum_out
        
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                G = i * recip_vectors[0] + j * recip_vectors[1]
                if np.allclose(diff, G, atol=1e-8):
                    return True
        
        return False
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian (tight-binding).
        
        Returns:
            numpy.ndarray: Non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get nearest and next-nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        nnn_vectors = self.get_next_nearest_neighbor_vectors()
        
        # For each k-point, compute the hopping terms
        for k_idx in range(self.N_k):
            k = self.k_space[k_idx]
            
            # For each spin
            for s in range(2):
                # For each q-vector
                for q_idx in range(3):
                    q = self.q_vectors[q_idx]
                    
                    # Tight-binding term (diagonal in spin and q-vector)
                    hopping_term = 0.0
                    
                    # Nearest-neighbor hopping
                    for nn in nn_vectors:
                        R_n = nn[0] * self.primitive_vectors[0] + nn[1] * self.primitive_vectors[1]
                        phase = np.exp(-1j * np.dot(k + q, R_n))
                        hopping_term += -self.t1 * phase
                    
                    # Next-nearest-neighbor hopping
                    for nnn in nnn_vectors:
                        R_n = nnn[0] * self.primitive_vectors[0] + nnn[1] * self.primitive_vectors[1]
                        phase = np.exp(-1j * np.dot(k + q, R_n))
                        hopping_term += -self.t2 * phase
                    
                    # Add to Hamiltonian (diagonal in spin and q-vector)
                    H_nonint[s, q_idx, s, q_idx, k_idx] = hopping_term
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian (Hartree-Fock terms).
        
        Args:
            exp_val (numpy.ndarray): Expectation value array.
            
        Returns:
            numpy.ndarray: Interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # For each k-point
        for k_idx in range(self.N_k):
            k = self.k_space[k_idx]
            
            # Hartree term
            for s in range(2):        # s
                for s_prime in range(2):  # s'
                    for q_alpha in range(3):
                        for q_delta in range(3):
                            for q_beta in range(3):
                                for q_gamma in range(3):
                                    # Check momentum conservation
                                    if self._check_momentum_conservation(q_alpha, q_beta, q_gamma, q_delta):
                                        # Compute interaction potential
                                        U_q = self._compute_interaction_potential_k(
                                            self.q_vectors[q_alpha] - self.q_vectors[q_delta]
                                        )
                                        
                                        # Hartree term
                                        H_int[s_prime, q_beta, s_prime, q_gamma, k_idx] += (
                                            U_q / self.N_k * exp_val[s, q_alpha, s, q_delta, k_idx]
                                        )
            
            # Fock term
            for s in range(2):        # s
                for s_prime in range(2):  # s'
                    for q_alpha in range(3):
                        for q_gamma in range(3):
                            for q_beta in range(3):
                                for q_delta in range(3):
                                    # Check momentum conservation
                                    if self._check_momentum_conservation(q_alpha, q_beta, q_gamma, q_delta):
                                        # Compute interaction potential for Fock term
                                        U_p = self._compute_interaction_potential_k(
                                            k + self.q_vectors[q_alpha] - k - self.q_vectors[q_delta]
                                        )
                                        
                                        # Fock term (negative sign)
                                        H_int[s_prime, q_beta, s, q_delta, k_idx] -= (
                                            U_p / self.N_k * exp_val[s, q_alpha, s_prime, q_gamma, k_idx]
                                        )
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian by combining non-interacting
        and interacting parts.
        
        Args:
            exp_val (numpy.ndarray): Expectation value array.
            return_flat (bool, optional): If True, returns a flattened Hamiltonian. Defaults to True.
            
        Returns:
            numpy.ndarray: Total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
