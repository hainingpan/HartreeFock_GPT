import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with spin-1/2 fermions on a triangular lattice.
    
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int = 10, parameters: dict = None, filling_factor: float = 0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # LM Task: has to define this tuple - we have 2 spin flavors
        self.basis_order = {'0': 'spin'}
        # Order for spin:
        # 0: spin up
        # 1: spin down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature, assuming T=0 as specified
        
        # Set default parameters if none are provided
        if parameters is None:
            parameters = {
                'a': 1.0,
                't_0': 1.0,  # On-site hopping
                't_1': 0.1,  # First-shell hopping
                'U_0': 1.0,  # On-site interaction
                'U_1': 0.5,  # First-shell interaction
            }
        
        self.a = parameters.get('a', 1.0)  # Lattice constant
        # Define the primitive vectors for a 2D triangular lattice
        self.primitive_vectors = self.a * np.array([[1, 0], [0.5, np.sqrt(3)/2]])
        
        # Generate k-space
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t_0 = parameters.get('t_0', 1.0)  # On-site hopping
        self.t_1 = parameters.get('t_1', 0.1)  # First-shell hopping
        
        # Interaction parameters
        self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction
        self.U_1 = parameters.get('U_1', 0.5)  # First-shell interaction
        
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
            (1, 0),   # +a1
            (0, 1),   # +a2
            (-1, 1),  # -a1 + a2
            (-1, 0),  # -a1
            (0, -1),  # -a2
            (1, -1)   # a1 - a2
        ]
        return n_vectors
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generate the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Get the nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Compute E_s(k) for each spin and k-point
        for s in range(self.D[0]):
            for i, k in enumerate(self.k_space):
                # On-site term
                E_s_k = self.t_0
                
                # Nearest-neighbor hopping - calculating E_s(k) = sum_n t_s(n) * exp(-i*k*n)
                for n in nn_vectors:
                    # Convert lattice coordinates to cartesian
                    n_cart = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                    phase = np.exp(-1j * np.dot(k, n_cart))
                    E_s_k += self.t_1 * phase
                
                # Assign to diagonal elements (kinetic terms)
                H_nonint[s, s, i] = E_s_k
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val: The expectation value with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)  # Unflatten exp_val to match Hamiltonian shape
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Hartree term: (1/N) * sum_s,s',k1,k2 U(0) * <c_s^†(k1) c_s(k1)> * c_s'^†(k2) c_s'(k2)
        for s_prime in range(self.D[0]):
            for s in range(self.D[0]):
                # Calculate average density for spin s across all k-points
                avg_density_s = np.mean(exp_val[s, s, :])
                # Add Hartree contribution to H_int[s_prime, s_prime, :]
                H_int[s_prime, s_prime, :] += (self.U_0 / self.N_k) * avg_density_s
        
        # Fock term: -(1/N) * sum_s,s',k1,k2 U(k1-k2) * <c_s^†(k1) c_s'(k1)> * c_s'^†(k2) c_s(k2)
        # For simplicity, we approximate U(k1-k2) with U(0) for all k1,k2
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                # Calculate average correlation between spin s and s_prime
                avg_correlation = np.mean(exp_val[s, s_prime, :])
                # Add Fock contribution to H_int[s_prime, s, :]
                H_int[s_prime, s, :] -= (self.U_0 / self.N_k) * avg_correlation
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generate the total Hamiltonian.
        
        Args:
            exp_val: The expectation value.
            return_flat: Whether to return the flattened Hamiltonian.
            
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
