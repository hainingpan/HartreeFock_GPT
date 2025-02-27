import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with spin degrees of freedom.
    
    The Hamiltonian consists of:
    1. Kinetic term: E_s(k) = sum_n t_s(n) exp(-i k·n)
    2. Hartree term: U(0)/N sum_{s,s',k1,k2} <c_s^dagger(k1) c_s(k1)> c_s'^dagger(k2) c_s'(k2)
    3. Fock term: -1/N sum_{s,s',k1,k2} U(k1-k2) <c_s^dagger(k1) c_s'(k1)> c_s'^dagger(k2) c_s(k2)
    
    Args:
        N_shell: Number of shells for the k-space grid
        parameters: Dictionary containing model parameters
        filling_factor: Filling factor of the system (default: 0.5)
    """
    def __init__(self, N_shell: int=5, parameters: dict={'t_up': 1.0, 't_down': 1.0, 'U0': 1.0, 'Un': 0.5}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # Tuple representing the flavor dimensions (spin up, spin down)
        self.basis_order = {'0': 'spin'}
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0.0  # Temperature (set to 0)
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = self.a * np.array([[0,1],[np.sqrt(3)/2,-1/2]])  # Define primitive vectors for triangular lattice
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_up = parameters.get('t_up', 1.0)  # Hopping parameter for spin up
        self.t_down = parameters.get('t_down', 1.0)  # Hopping parameter for spin down
        self.U0 = parameters.get('U0', 1.0)  # On-site interaction strength U(0)
        self.Un = parameters.get('Un', 0.5)  # Nearest-neighbor interaction strength U(n)
        
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
            (1, 0),   # First nearest neighbor
            (0, 1),   # Second nearest neighbor
            (-1, 1),  # Third nearest neighbor
            (-1, 0),  # Fourth nearest neighbor
            (0, -1),  # Fifth nearest neighbor
            (1, -1)   # Sixth nearest neighbor
        ]
        return n_vectors
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian:
        H_Kinetic = sum_{s,k} E_s(k) c_s^dagger(k) c_s(k)
        
        Returns:
            np.ndarray: Non-interacting Hamiltonian with shape (D, D, N_k)
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Get nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Calculate dispersion E_s(k) = sum_n t_s(n) exp(-i k·n)
        for k_idx, k in enumerate(self.k_space):
            E_up = 0.0
            E_down = 0.0
            
            for n in nn_vectors:
                # Convert integer coordinates to real-space coordinates
                n_real = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                
                # Calculate k·n
                k_dot_n = np.dot(k, n_real)
                
                # Calculate exp(-i k·n)
                exp_factor = np.exp(-1j * k_dot_n)
                
                # Add contribution to E_s(k)
                E_up += self.t_up * exp_factor
                E_down += self.t_down * exp_factor
            
            # Add to diagonal elements of Hamiltonian
            H_nonint[0, 0, k_idx] = E_up  # Spin up kinetic term
            H_nonint[1, 1, k_idx] = E_down  # Spin down kinetic term
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian (Hartree and Fock terms)
        
        Args:
            exp_val: Expectation value array
            
        Returns:
            np.ndarray: Interacting Hamiltonian with shape (D, D, N_k)
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Extract expectation values
        n_up = exp_val[0, 0, :]  # <c_up^dagger(k) c_up(k)>
        n_down = exp_val[1, 1, :]  # <c_down^dagger(k) c_down(k)>
        p_up_down = exp_val[0, 1, :]  # <c_up^dagger(k) c_down(k)>
        p_down_up = exp_val[1, 0, :]  # <c_down^dagger(k) c_up(k)>
        
        # Calculate average densities
        avg_n_up = np.mean(n_up)
        avg_n_down = np.mean(n_down)
        
        # Calculate U(k1-k2) for all k-point pairs
        nn_vectors = self.get_nearest_neighbor_vectors()
        U_k_diff = np.zeros((self.N_k, self.N_k), dtype=complex)
        
        for k1_idx in range(self.N_k):
            for k2_idx in range(self.N_k):
                k_diff = self.k_space[k1_idx] - self.k_space[k2_idx]
                U_k_diff[k1_idx, k2_idx] = self.U0  # Add U(0) term
                
                for n in nn_vectors:
                    # Convert to real-space coordinates
                    n_real = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                    
                    # Calculate (k1-k2)·n
                    k_dot_n = np.dot(k_diff, n_real)
                    
                    # Add U(n) * exp(-i (k1-k2)·n) to U(k1-k2)
                    U_k_diff[k1_idx, k2_idx] += self.Un * np.exp(-1j * k_dot_n)
        
        # Hartree terms (diagonal elements)
        for k_idx in range(self.N_k):
            # Spin up-up interaction from both up and down density
            H_int[0, 0, k_idx] += self.U0 * (avg_n_up + avg_n_down) / self.N_k
            
            # Spin down-down interaction from both up and down density
            H_int[1, 1, k_idx] += self.U0 * (avg_n_up + avg_n_down) / self.N_k
        
        # Fock terms
        for k2_idx in range(self.N_k):
            for k1_idx in range(self.N_k):
                # Diagonal elements (same spin)
                H_int[0, 0, k2_idx] -= U_k_diff[k1_idx, k2_idx] * n_up[k1_idx] / self.N_k
                H_int[1, 1, k2_idx] -= U_k_diff[k1_idx, k2_idx] * n_down[k1_idx] / self.N_k
                
                # Off-diagonal elements (spin flip)
                H_int[0, 1, k2_idx] -= U_k_diff[k1_idx, k2_idx] * p_up_down[k1_idx] / self.N_k
                H_int[1, 0, k2_idx] -= U_k_diff[k1_idx, k2_idx] * p_down_up[k1_idx] / self.N_k
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hamiltonian by combining non-interacting and interacting parts
        
        Args:
            exp_val: Expectation value array
            return_flat: Whether to return flattened Hamiltonian
            
        Returns:
            np.ndarray: Total Hamiltonian
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened_hamiltonian(H_total, self.D, self.N_k)
        else:
            return H_total
