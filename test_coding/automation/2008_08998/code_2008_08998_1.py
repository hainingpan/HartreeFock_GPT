import numpy as np
from typing import Any, Dict, Tuple, List
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with a √3 × √3 superlattice.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor of the system. Defaults to 0.5.
    """
    def __init__(self, N_shell: int, parameters: Dict[str, Any] = {}, filling_factor: float = 0.5):
        self.lattice = 'triangular'
        self.D = (2, 3)  # (spin, high-symmetry point)
        self.basis_order = {'0': 'spin', '1': 'q_vector'}
        # Order for each flavor:
        # 0: spin up (0), spin down (1)
        # 1: Γ (0), K (1), K' (2)
        
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
        self.e2_epsilon0 = parameters.get('e2_epsilon0', 1440.0)  # Coulomb constant (meV·nm)
        self.d = parameters.get('d', 10.0)  # Screening length (nm)
        self.epsilon_r = parameters.get('epsilon_r', 1.0)  # Relative dielectric constant
        self.U_onsite = parameters.get('U_onsite', 1000.0/self.epsilon_r)  # On-site repulsion (meV)
        
        # Define high-symmetry points
        self.high_symmetry_points = self._get_high_symmetry_points()
        
        return
    
    def _get_high_symmetry_points(self):
        """Get the high-symmetry points in the original Brillouin zone."""
        high_sym_pts = generate_high_symmtry_points(self.lattice, self.a)
        return {
            "Gamma": high_sym_pts["Gamma"],
            "K": high_sym_pts["K"],
            "K'": high_sym_pts["K'"]
        }
    
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
        """Returns the integer coordinate offsets for next-nearest neighbors in a triangular lattice."""
        n_vectors = [
            (2, 0), (0, 2), (2, 2),
            (-2, 0), (0, -2), (-2, -2),
            (1, -1), (-1, 1), (2, 1),
            (1, 2), (-1, -2), (-2, -1)
        ]
        return n_vectors
    
    def _calculate_coulomb_potential_real(self, r):
        """Calculate the real-space Coulomb potential.
        
        Args:
            r (float): Distance in nm.
            
        Returns:
            float: Potential value in meV.
        """
        if r < 1e-6:  # Very close to zero, use on-site value
            return self.U_onsite
        else:
            # Screened Coulomb potential
            return (self.e2_epsilon0 / self.epsilon_r) * (1.0/r - 1.0/np.sqrt(r**2 + self.d**2))
    
    def _calculate_coulomb_potential_momentum(self, q_diff):
        """Calculate the Coulomb potential in momentum space.
        
        Args:
            q_diff (numpy.ndarray): Momentum difference vector.
            
        Returns:
            float: Potential value in meV.
        """
        # This is a simplified implementation
        # In a real implementation, this would involve a Fourier transform of the real-space potential
        # over all lattice sites
        
        # For demonstration, we'll approximate using the norm of q_diff
        q_norm = np.linalg.norm(q_diff)
        if q_norm < 1e-6:
            return self.U_onsite  # On-site potential
        else:
            # Approximate Fourier transform for the screened Coulomb potential
            # This is a crude approximation and would need to be replaced with a proper calculation
            return (self.e2_epsilon0 / self.epsilon_r) * (1.0/q_norm)
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (2, 3, 2, 3, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get nearest and next-nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        nnn_vectors = self.get_next_nearest_neighbor_vectors()
        
        # Get high-symmetry points
        gamma_pt = self.high_symmetry_points["Gamma"]
        k_pt = self.high_symmetry_points["K"]
        kp_pt = self.high_symmetry_points["K'"]
        q_vectors = [gamma_pt, k_pt, kp_pt]
        
        # Calculate hopping terms
        for s in range(2):  # spin
            for q1 in range(3):  # q vector for creation
                for q2 in range(3):  # q vector for annihilation
                    for k_idx in range(self.N_k):  # k point
                        k = self.k_space[k_idx]
                        
                        # Term for nearest-neighbor hopping
                        for n_vec in nn_vectors:
                            R_n = n_vec[0] * self.primitive_vectors[0] + n_vec[1] * self.primitive_vectors[1]
                            phase = np.exp(-1j * np.dot(k + q_vectors[q1] - q_vectors[q2], R_n))
                            H_nonint[s, q1, s, q2, k_idx] -= self.t1 * phase
                        
                        # Term for next-nearest-neighbor hopping
                        for n_vec in nnn_vectors:
                            R_n = n_vec[0] * self.primitive_vectors[0] + n_vec[1] * self.primitive_vectors[1]
                            phase = np.exp(-1j * np.dot(k + q_vectors[q1] - q_vectors[q2], R_n))
                            H_nonint[s, q1, s, q2, k_idx] -= self.t2 * phase
        
        return H_nonint
    
    def _momentum_conserving_indices(self):
        """
        Generate all combinations of q_alpha, q_beta, q_gamma, q_delta that satisfy
        q_alpha + q_beta = q_gamma + q_delta (up to a reciprocal lattice vector).
        
        Returns:
            List[Tuple[int, int, int, int]]: List of valid (q_alpha, q_beta, q_gamma, q_delta) index tuples.
        """
        # This is a simplified implementation
        # In a real implementation, this would check all combinations against the delta function
        
        valid_combinations = []
        
        # For simplicity, we'll consider these combinations:
        # 1. All same: (i, i, i, i)
        for i in range(3):
            valid_combinations.append((i, i, i, i))
        
        # 2. Swap pairs: (i, j, j, i)
        for i in range(3):
            for j in range(3):
                if i != j:
                    valid_combinations.append((i, j, j, i))
        
        # Additional combinations would require detailed knowledge of the reciprocal lattice
        # and would need to be verified against the momentum conservation condition
        
        return valid_combinations
    
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
        
        # Get high-symmetry points
        gamma_pt = self.high_symmetry_points["Gamma"]
        k_pt = self.high_symmetry_points["K"]
        kp_pt = self.high_symmetry_points["K'"]
        q_vectors = [gamma_pt, k_pt, kp_pt]
        
        # Get momentum-conserving combinations
        valid_combinations = self._momentum_conserving_indices()
        
        # Calculate Hartree and Fock terms
        for p_beta_idx in range(self.N_k):
            p_beta = self.k_space[p_beta_idx]
            
            # Loop through all valid combinations of q indices
            for q_alpha, q_beta, q_gamma, q_delta in valid_combinations:
                # For simplicity, assume p_alpha = p_beta for the expectation values
                p_alpha_idx = p_beta_idx
                
                # Hartree term: <c_q_α,s^†(p_α) c_q_δ,s(p_α)> c_q_β,s'^†(p_β) c_q_γ,s'(p_β)
                for s in range(2):  # Spin for expectation value
                    hartree_exp_val = exp_val[s, q_alpha, s, q_delta, p_alpha_idx]
                    
                    # Calculate Coulomb potential for Hartree term
                    U_hartree = self._calculate_coulomb_potential_momentum(q_vectors[q_alpha] - q_vectors[q_delta])
                    
                    for sp in range(2):  # Spin for operators
                        # Add Hartree term to H_int
                        H_int[sp, q_beta, sp, q_gamma, p_beta_idx] += (1.0 / self.N_k) * U_hartree * hartree_exp_val
                
                # Fock term: <c_q_α,s^†(p_α) c_q_γ,s'(p_α)> c_q_β,s'^†(p_β) c_q_δ,s(p_β)
                for s in range(2):  # Spin for first operator in expectation
                    for sp in range(2):  # Spin for second operator in expectation
                        fock_exp_val = exp_val[s, q_alpha, sp, q_gamma, p_alpha_idx]
                        
                        # Calculate Coulomb potential for Fock term
                        # This is a simplification; the real calculation would be more complex
                        U_fock = self._calculate_coulomb_potential_momentum(
                            p_alpha_idx + q_vectors[q_alpha] - p_beta - q_vectors[q_delta]
                        )
                        
                        # Add Fock term to H_int
                        H_int[sp, q_beta, s, q_delta, p_beta_idx] -= (1.0 / self.N_k) * U_fock * fock_exp_val
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool, optional): Whether to return a flattened Hamiltonian. Defaults to True.
            
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
