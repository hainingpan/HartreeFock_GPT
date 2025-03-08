import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice system with spin and
    reciprocal lattice vector (q) degrees of freedom.
    
    Args:
        N_shell (int): Number of momentum shells in the Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor for the system. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Lattice symmetry
        self.D = (2, 3)  # Number of flavors: (spin, q-vector)
        self.basis_order = {'0': 'spin', '1': 'q_vector'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: Gamma point, K point, K' point
        
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
        self.d = parameters.get('d', 10.0)  # Screening length (nm)
        self.e2_epsilon0 = 1440.0  # Coulomb constant (meV·nm)
        self.U_onsite = 1000.0 / self.epsilon_r  # On-site interaction (meV)
        
        # Define the high-symmetry points in the BZ
        high_sym_points = generate_high_symmtry_points(self.lattice, self.a)
        self.q_vectors = np.zeros((3, 2))  # [Gamma, K, K']
        self.q_vectors[0] = high_sym_points["Gamma"]  # Gamma point
        self.q_vectors[1] = high_sym_points["K"]      # K point
        self.q_vectors[2] = high_sym_points["K'"]     # K' point
        
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
    
    def calculate_U(self, p):
        """
        Calculate the Coulomb interaction potential in momentum space.
        
        Args:
            p (np.ndarray): Momentum vector.
            
        Returns:
            float: Coulomb potential value at momentum p.
        """
        if np.all(p == 0):
            return self.U_onsite
        
        # Get real-space vectors within a certain range for Fourier transform
        n_shell = 10  # Sufficient for convergence
        ij, r_vectors = get_q(n_shell, self.a)
        
        # Calculate real-space potential U(r)
        U_r = np.zeros(len(r_vectors))
        for i, r in enumerate(r_vectors):
            r_norm = np.linalg.norm(r)
            if r_norm > 0:
                U_r[i] = (self.e2_epsilon0 / self.epsilon_r) * (1/r_norm - 1/np.sqrt(r_norm**2 + self.d**2))
        
        # Calculate momentum-space potential U(p)
        U_p = 0
        for i, r in enumerate(r_vectors):
            U_p += U_r[i] * np.exp(1j * np.dot(p, r))
        
        return U_p.real
    
    def momentum_conservation_delta(self, q_alpha, q_beta, q_gamma, q_delta):
        """
        Check if momentum conservation is satisfied: q_alpha + q_beta = q_gamma + q_delta (up to a reciprocal lattice vector).
        
        Args:
            q_alpha, q_beta, q_gamma, q_delta: Reciprocal lattice vectors.
            
        Returns:
            bool: True if momentum conservation is satisfied, False otherwise.
        """
        # Get the reciprocal lattice vectors
        recip_vectors = get_reciprocal_vectors_triangle(self.a)
        
        # Calculate q_alpha + q_beta - q_gamma - q_delta
        q_diff = q_alpha + q_beta - q_gamma - q_delta
        
        # Check if q_diff is approximately a reciprocal lattice vector
        for i in range(-1, 2):
            for j in range(-1, 2):
                G = i * recip_vectors[0] + j * recip_vectors[1]
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
        
        # Get nearest and next-nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        nnn_vectors = get_shell_index_triangle(2)  # Next-nearest neighbors
        
        # Calculate the hopping terms for each spin and q-vector combination
        for s in range(2):  # spin index
            for q_idx1 in range(3):  # q-vector index for creation operator
                for q_idx2 in range(3):  # q-vector index for annihilation operator
                    q1 = self.q_vectors[q_idx1]
                    q2 = self.q_vectors[q_idx2]
                    
                    # Loop over all k-points
                    for k_idx in range(self.N_k):
                        k = self.k_space[k_idx]
                        
                        # Nearest-neighbor hopping
                        for n in nn_vectors:
                            R_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                            H_nonint[s, q_idx1, s, q_idx2, k_idx] -= self.t1 * np.exp(-1j * np.dot(k + q1, R_n))
                        
                        # Next-nearest-neighbor hopping
                        for n in nnn_vectors:
                            R_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                            H_nonint[s, q_idx1, s, q_idx2, k_idx] -= self.t2 * np.exp(-1j * np.dot(k + q1, R_n))
        
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
        
        # Calculate Hartree and Fock terms
        for s1 in range(2):  # spin index s
            for s2 in range(2):  # spin index s'
                for q_alpha in range(3):
                    for q_beta in range(3):
                        for q_gamma in range(3):
                            for q_delta in range(3):
                                # Check momentum conservation
                                if not self.momentum_conservation_delta(
                                    self.q_vectors[q_alpha],
                                    self.q_vectors[q_beta],
                                    self.q_vectors[q_gamma],
                                    self.q_vectors[q_delta]):
                                    continue
                                
                                # Hartree term
                                # U(q_alpha - q_delta) * <c_{q_alpha,s}^dagger(p) c_{q_delta,s}(p)> * c_{q_beta,s'}^dagger(p) c_{q_gamma,s'}(p)
                                U_hartree = self.calculate_U(self.q_vectors[q_alpha] - self.q_vectors[q_delta])
                                hartree_mean = np.mean(exp_val[s1, q_alpha, s1, q_delta, :])
                                
                                # Hartree term contributes to H[s2, q_beta, s2, q_gamma, :]
                                H_int[s2, q_beta, s2, q_gamma, :] += (1.0 / self.N_k) * U_hartree * hartree_mean
                                
                                # Fock term
                                # Note: For precise calculation, U should depend on p_alpha + q_alpha - p_beta - q_delta
                                # but for simplicity, we'll use the q-dependence only.
                                U_fock = self.calculate_U(self.q_vectors[q_alpha] - self.q_vectors[q_gamma])
                                fock_mean = np.mean(exp_val[s1, q_alpha, s2, q_gamma, :])
                                
                                # Fock term contributes to H[s2, q_beta, s1, q_delta, :]
                                H_int[s2, q_beta, s1, q_delta, :] -= (1.0 / self.N_k) * U_fock * fock_mean
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            return_flat (bool): If True, returns the flattened Hamiltonian.
            
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
