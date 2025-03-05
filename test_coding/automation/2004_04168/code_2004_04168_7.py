import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with spin degrees of freedom on a triangular lattice.
    
    The Hamiltonian includes:
    - Kinetic energy: E_s(k) c^\dagger_s(k) c_s(k)
    - Hartree term: \frac{1}{N} \sum_{s, s'} \sum_{k_1, k_2} U(0) \langle c_s^\dagger(k_1) c_s(k_1) \rangle c_{s'}^\dagger(k_2) c_{s'}(k_2)
    - Fock term: -\frac{1}{N} \sum_{s, s'} \sum_{k_1, k_2} U(k_1 - k_2) \langle c_s^\dagger(k_1) c_{s'}(k_1) \rangle c_{s'}^\dagger(k_2) c_s(k_2)
    
    Args:
        N_shell: Number of k-space shells to include.
        parameters: Dictionary containing model parameters.
        filling_factor: Filling factor of the system. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=10, parameters: dict[str, Any]={'t_s': 1.0, 'U_0': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Lattice type: triangular
        self.D = (2,)  # Dimension tuple: 2 spin states (up, down)
        self.basis_order = {'0': 'spin'}  # Basis ordering
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature set to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        # Define the primitive (Bravais) lattice vectors for a 2D triangular lattice
        self.primitive_vectors = self.a * np.array([[1, 0], [0.5, np.sqrt(3)/2]])  
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_s = parameters.get('t_s', 1.0)  # Hopping parameter
        self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction strength
        self.U_1 = parameters.get('U_1', 0.5)  # Nearest-neighbor interaction strength
        self.U_2 = parameters.get('U_2', 0.2)  # Next-nearest-neighbor interaction strength
        
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
            (1, 0),    # Right
            (0, 1),    # Up-right
            (-1, 1),   # Up-left
            (-1, 0),   # Left
            (0, -1),   # Down-left
            (1, -1),   # Down-right
        ]
        return n_vectors
    
    def compute_dispersion(self, k_points):
        """
        Computes the dispersion relation E_s(k) = \sum_{n} t_s(n) e^{-i k \cdot n} for each k point.
        
        Args:
            k_points: Array of k points.
            
        Returns:
            Energy values for each k point.
        """
        neighbors = self.get_nearest_neighbor_vectors()
        E_s = np.zeros(k_points.shape[0], dtype=complex)
        
        # Compute E_s(k) = sum_n t_s(n) * exp(-i * k * n)
        for n in neighbors:
            # Convert integer coordinates to real space positions
            r_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            
            # For simplicity, assume t_s(n) is the same for all neighbors
            t_n = self.t_s
            
            # Compute k dot r_n for all k points
            k_dot_r = np.sum(k_points * r_n, axis=1)
            
            # Add contribution to E_s
            E_s += t_n * np.exp(-1j * k_dot_r)
            
        return E_s
    
    def compute_interaction(self, k_diff):
        """
        Computes the interaction term U(k) = \sum_{n} U(n) e^{-i k \cdot n} for a given k difference.
        
        Args:
            k_diff: Difference between k points.
            
        Returns:
            Interaction strength.
        """
        # Compute U(k) = sum_n U(n) * exp(-i * k * n)
        neighbors = self.get_nearest_neighbor_vectors()
        U_k = self.U_0  # On-site interaction
        
        for n in neighbors:
            # Convert integer coordinates to real space positions
            r_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            
            # Compute k dot r_n
            k_dot_r = np.sum(k_diff * r_n)
            
            # Add contribution from nearest neighbors
            U_k += self.U_1 * np.exp(-1j * k_dot_r)
            
        # Could add higher-order interactions if needed
        
        return U_k
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian: H_Kinetic = \sum_{s, k} E_s(k) c^\dagger_s(k) c_s(k)
        
        Returns:
            Non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Compute dispersion relation E_s(k)
        E_k = self.compute_dispersion(self.k_space)
        
        # Set diagonal elements for both spin up and spin down
        # They share the same dispersion relation in this case
        H_nonint[0, 0, :] = E_k  # Spin up kinetic energy term
        H_nonint[1, 1, :] = E_k  # Spin down kinetic energy term
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian, including both Hartree and Fock terms.
        
        Args:
            exp_val: Expectation value array.
            
        Returns:
            Interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Hartree term: U(0) * <c_s^†(k1) c_s(k1)> * c_s'^†(k2) c_s'(k2)
        n_up = np.mean(exp_val[0, 0, :]).real  # Average density for spin up
        n_down = np.mean(exp_val[1, 1, :]).real  # Average density for spin down
        
        # Hartree term contributes to diagonal elements
        H_int[0, 0, :] = self.U_0 / self.N_k * n_down  # Spin up interacting with average spin down
        H_int[1, 1, :] = self.U_0 / self.N_k * n_up    # Spin down interacting with average spin up
        
        # Fock term: -U(k1-k2) * <c_s^†(k1) c_s'(k1)> * c_s'^†(k2) c_s(k2)
        for k2_idx in range(self.N_k):
            for k1_idx in range(self.N_k):
                k_diff = self.k_space[k1_idx] - self.k_space[k2_idx]
                U_k_diff = self.compute_interaction(k_diff)
                
                # Diagonal Fock terms (s = s')
                H_int[0, 0, k2_idx] -= U_k_diff / self.N_k * exp_val[0, 0, k1_idx]
                H_int[1, 1, k2_idx] -= U_k_diff / self.N_k * exp_val[1, 1, k1_idx]
                
                # Off-diagonal Fock terms (s ≠ s')
                H_int[0, 1, k2_idx] -= U_k_diff / self.N_k * exp_val[0, 1, k1_idx]
                H_int[1, 0, k2_idx] -= U_k_diff / self.N_k * exp_val[1, 0, k1_idx]
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hamiltonian by combining non-interacting and interacting parts.
        
        Args:
            exp_val: Expectation value.
            return_flat: Whether to return a flattened Hamiltonian.
            
        Returns:
            Total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened_hamiltonian(H_total, self.D, self.N_k)
        else:
            return H_total
