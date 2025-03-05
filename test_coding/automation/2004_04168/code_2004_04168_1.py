import numpy as np
from typing import Any, Dict
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with kinetic, Hartree, and Fock terms.
    
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system.
    """
    def __init__(self, N_shell: int=3, parameters: Dict[str, Any]={'t_s': 1.0, 'U_0': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # Two spin flavors (up and down)
        self.basis_order = {'0': 'spin'}
        # Order for spin: 0 = spin_up, 1 = spin_down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = self.a * np.array([[1, 0], [0.5, np.sqrt(3)/2]])  # Primitive vectors for triangular lattice
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameter
        self.t_s = parameters.get('t_s', 1.0)  # Nearest-neighbor hopping
        
        # Interaction parameter
        self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction
        
        return
    
    def get_nearest_neighbor_vectors(self):
        """
        Returns the integer coordinate offsets corresponding to the 
        nearest neighbors in a 2D triangular Bravais lattice.
        
        For a 2D triangular lattice, there are six nearest neighbors.
        """
        n_vectors = [
            (1, 0),    # Right
            (0, 1),    # Upper-right
            (-1, 1),   # Upper-left
            (-1, 0),   # Left
            (0, -1),   # Lower-left
            (1, -1)    # Lower-right
        ]
        return n_vectors
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting (kinetic) part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D[0], D[0], N_k).
        """
        H_nonint = np.zeros((self.D[0], self.D[0], self.N_k), dtype=complex)
        
        # Get nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Calculate E_s(k) for each k-point and spin
        for s in range(self.D[0]):
            for k_idx, k in enumerate(self.k_space):
                E_k = 0.0
                # Sum over nearest neighbors
                for n_vec in nn_vectors:
                    # Convert n_vec to real space using primitive vectors
                    n_real = n_vec[0] * self.primitive_vectors[0] + n_vec[1] * self.primitive_vectors[1]
                    # Calculate dot product k⋅n
                    k_dot_n = np.dot(k, n_real)
                    # Add to E_k: t_s(n) * e^(-i k⋅n)
                    E_k += self.t_s * np.exp(-1j * k_dot_n)
                
                # Set diagonal element - represents kinetic energy term
                H_nonint[s, s, k_idx] = E_k
        
        return H_nonint
    
    def calculate_U_k(self, k_diff):
        """
        Calculate U(k) for a given k difference.
        
        Args:
            k_diff (numpy.ndarray): Difference between two k-points.
            
        Returns:
            complex: U(k) value.
        """
        # In general, U(k) = sum_n U(n) * e^(-i k⋅n)
        # For simplicity, we assume only on-site interaction U(0)
        # For a more complete model, we would include interactions at different neighbor distances
        
        if np.allclose(k_diff, 0):
            return self.U_0  # U(0) for on-site interaction
        else:
            return 0.0  # Simplified: no interaction for k_diff != 0
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting (Hartree + Fock) part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D[0], D[0], N_k).
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)
        H_int = np.zeros((self.D[0], self.D[0], self.N_k), dtype=complex)
        
        # Hartree term: (1/N) * sum_{s,s',k1,k2} U(0) * <c_s^†(k1) c_s(k1)> * c_s'^†(k2) c_s'(k2)
        # Compute average density for each spin
        n_s = np.zeros(self.D[0], dtype=complex)
        for s in range(self.D[0]):
            n_s[s] = np.mean(exp_val[s, s, :])
        
        # Apply Hartree term to diagonal elements
        for s_prime in range(self.D[0]):
            for k2_idx in range(self.N_k):
                hartree_term = self.U_0 * sum(n_s)  # U(0) * sum_s <n_s>
                H_int[s_prime, s_prime, k2_idx] += hartree_term
        
        # Fock term: -(1/N) * sum_{s,s',k1,k2} U(k1-k2) * <c_s^†(k1) c_s'(k1)> * c_s'^†(k2) c_s(k2)
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                for k2_idx, k2 in enumerate(self.k_space):
                    for k1_idx, k1 in enumerate(self.k_space):
                        k_diff = k1 - k2
                        U_k_diff = self.calculate_U_k(k_diff)
                        # Note the negative sign in the Fock term
                        H_int[s_prime, s, k2_idx] -= (U_k_diff * exp_val[s, s_prime, k1_idx]) / self.N_k
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            return_flat (bool): Whether to return a flattened Hamiltonian.
            
        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened_hamiltonian(H_total, self.D, self.N_k)
        else:
            return H_total
