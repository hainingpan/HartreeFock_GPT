import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with spin and reciprocal lattice vectors
    corresponding to high-symmetry points in the folded Brillouin zone.
    
    Args:
        N_shell (int): Number of shells in k-space for discretization.
        parameters (dict): Dictionary of model parameters.
        filling_factor (float): Filling factor (default: 0.5).
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2, 3)  # (|spin|, |q|)
        self.basis_order = {'0': 'spin', '1': 'reciprocal_lattice_vector'}
        # Spin: 0 = up, 1 = down
        # Reciprocal lattice vector: 0 = Γ, 1 = K, 2 = K'
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t1 = parameters.get('t1', 6.0)  # Nearest-neighbor hopping (meV)
        self.t2 = parameters.get('t2', 1.0)  # Next-nearest-neighbor hopping (meV)
        self.epsilon_r = parameters.get('epsilon_r', 1.0)  # Relative dielectric constant
        self.d = parameters.get('d', 10.0)  # Screening length (nm)
        self.coulomb_const = parameters.get('coulomb_const', 1440.0)  # e^2/epsilon_0 (meV·nm)
        self.onsite_u = parameters.get('onsite_u', 1000.0/self.epsilon_r)  # Onsite interaction (meV)
        
        # High-symmetry points in original BZ
        self.high_symmetry_points = generate_high_symmtry_points(self.lattice, self.a)
        self.gamma = self.high_symmetry_points["Gamma"]
        self.K = self.high_symmetry_points["K"]
        self.Kprime = self.high_symmetry_points["K'"]
        self.q_vectors = [self.gamma, self.K, self.Kprime]
        
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
    
    def interaction_potential_momentum(self, q_diff):
        """
        Computes the interaction potential in momentum space.
        
        Args:
            q_diff (np.ndarray): Momentum difference vector.
            
        Returns:
            float: Interaction potential in momentum space.
        """
        # For a screened Coulomb potential in 2D
        q_norm = np.linalg.norm(q_diff)
        if q_norm < 1e-10:  # Avoid division by zero
            return self.onsite_u
        else:
            # Approximate Fourier transform of the screened potential
            return 2*np.pi*self.coulomb_const/self.epsilon_r / (q_norm + 1/self.d)
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: Non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get neighbor vectors for hopping
        nn_vectors = self.get_nearest_neighbor_vectors()  # nearest neighbors
        
        # Precompute neighbor vectors in real space
        nn_real = []
        for nn in nn_vectors:
            R_n = nn[0] * self.primitive_vectors[0] + nn[1] * self.primitive_vectors[1]
            nn_real.append(R_n)
        
        # Next nearest neighbors (using shell index 2)
        nnn_real = []
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                if abs(n1) + abs(n2) == 2 and n1 != n2:  # Next-nearest neighbors
                    R_n = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
                    nnn_real.append(R_n)
        
        # Loop over all k-points (p in folded BZ)
        for k_idx in range(self.N_k):
            p = self.k_space[k_idx]
            
            # Loop over spins
            for s in range(2):
                for q_alpha_idx in range(3):
                    q_alpha = self.q_vectors[q_alpha_idx]
                    
                    for q_delta_idx in range(3):
                        q_delta = self.q_vectors[q_delta_idx]
                        
                        # Kinetic term (only diagonal in q for tight-binding)
                        if q_alpha_idx == q_delta_idx:
                            hopping_term = 0j
                            
                            # Nearest-neighbor hopping
                            for R_n in nn_real:
                                phase = np.exp(-1j * np.dot(p + q_alpha, R_n))
                                hopping_term += -self.t1 * phase
                            
                            # Next-nearest-neighbor hopping
                            for R_n in nnn_real:
                                phase = np.exp(-1j * np.dot(p + q_alpha, R_n))
                                hopping_term += -self.t2 * phase
                            
                            # Add to Hamiltonian - kinetic energy term
                            H_nonint[s, q_alpha_idx, s, q_delta_idx, k_idx] = hopping_term
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian (Hartree-Fock terms).
        
        Args:
            exp_val (np.ndarray): Expectation value with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: Interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Loop over all k-points (p_beta in the equation)
        for k_idx in range(self.N_k):
            p_beta = self.k_space[k_idx]
            
            # Hartree term
            for s_prime in range(2):  # s' in equation
                for s in range(2):     # s in equation
                    for q_alpha_idx in range(3):
                        q_alpha = self.q_vectors[q_alpha_idx]
                        
                        for q_beta_idx in range(3):
                            q_beta = self.q_vectors[q_beta_idx]
                            
                            for q_gamma_idx in range(3):
                                q_gamma = self.q_vectors[q_gamma_idx]
                                
                                for q_delta_idx in range(3):
                                    q_delta = self.q_vectors[q_delta_idx]
                                    
                                    # Momentum conservation check
                                    if np.allclose(q_alpha + q_beta, q_gamma + q_delta, atol=1e-10):
                                        # Interaction potential
                                        U_hartree = self.interaction_potential_momentum(q_alpha - q_delta)
                                        
                                        # Sum over all p_alpha (approximate integration)
                                        for p_alpha_idx in range(self.N_k):
                                            # Get density expectation value: <c^†_{q_alpha,s}(p_alpha) c_{q_delta,s}(p_alpha)>
                                            exp_hartree = exp_val[s, q_alpha_idx, s, q_delta_idx, p_alpha_idx]
                                            
                                            # Add Hartree term: c^†_{q_beta,s'}(p_beta) c_{q_gamma,s'}(p_beta)
                                            H_int[s_prime, q_beta_idx, s_prime, q_gamma_idx, k_idx] += \
                                                U_hartree * exp_hartree / self.N_k
            
            # Fock term
            for s_prime in range(2):  # s' in equation
                for s in range(2):     # s in equation
                    for q_alpha_idx in range(3):
                        q_alpha = self.q_vectors[q_alpha_idx]
                        
                        for q_beta_idx in range(3):
                            q_beta = self.q_vectors[q_beta_idx]
                            
                            for q_gamma_idx in range(3):
                                q_gamma = self.q_vectors[q_gamma_idx]
                                
                                for q_delta_idx in range(3):
                                    q_delta = self.q_vectors[q_delta_idx]
                                    
                                    # Momentum conservation check
                                    if np.allclose(q_alpha + q_beta, q_gamma + q_delta, atol=1e-10):
                                        # Sum over all p_alpha (approximate integration)
                                        for p_alpha_idx in range(self.N_k):
                                            p_alpha = self.k_space[p_alpha_idx]
                                            
                                            # Interaction potential with momentum transfer for exchange
                                            U_fock = self.interaction_potential_momentum(
                                                p_alpha + q_alpha - p_beta - q_delta
                                            )
                                            
                                            # Get exchange expectation value: <c^†_{q_alpha,s}(p_alpha) c_{q_gamma,s'}(p_alpha)>
                                            exp_fock = exp_val[s, q_alpha_idx, s_prime, q_gamma_idx, p_alpha_idx]
                                            
                                            # Add Fock term (with negative sign): c^†_{q_beta,s'}(p_beta) c_{q_delta,s}(p_beta)
                                            H_int[s_prime, q_beta_idx, s, q_delta_idx, k_idx] += \
                                                -U_fock * exp_fock / self.N_k
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian.
            
        Returns:
            np.ndarray: Total Hamiltonian, either flattened or with full shape.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
