import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Defines the Hartree-Fock Hamiltonian for a system with a triangular lattice,
    with nearest and next-nearest neighbor hopping, and on-site and nearest-neighbor
    interactions.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor (default is 0.5).
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_1': 6.0, 't_2': 1.0, 'U_0': 1.0, 'U_1': 0.5, 'T': 0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # Spin flavor
        self.basis_order = {'0': 'spin'}
        # Order for spin flavor: 0 = spin_up, 1 = spin_down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # Temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters (in meV)
        self.t_1 = parameters.get('t_1', 6.0)  # Nearest-neighbor hopping in meV
        self.t_2 = parameters.get('t_2', 1.0)  # Next-nearest-neighbor hopping in meV
        
        # Interaction strengths
        self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction
        self.U_1 = parameters.get('U_1', 0.5)  # Nearest-neighbor interaction
        
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
    
    def compute_Es(self, k: np.ndarray) -> np.ndarray:
        """
        Compute the energy dispersion E_s(k) for a triangular lattice.
        
        Args:
            k (np.ndarray): k points in the first Brillouin zone of shape (N_k, 2).
            
        Returns:
            np.ndarray: Energy dispersion E_s(k) of shape (N_k,).
        """
        # Get primitive lattice vectors
        a1 = self.primitive_vectors[0]
        a2 = self.primitive_vectors[1]
        
        # Initialize energy
        E_s = np.zeros(k.shape[0], dtype=complex)
        
        # Nearest neighbor hopping
        for n in self.get_nearest_neighbor_vectors():
            n_vector = n[0] * a1 + n[1] * a2
            k_dot_n = np.sum(k * n_vector, axis=1)
            E_s += self.t_1 * np.exp(-1j * k_dot_n)
        
        # Next nearest neighbor hopping
        # Define the next-nearest neighbors for a triangular lattice
        next_nearest_neighbors = [
            (2, 0), (0, 2), (-1, 2), (-2, 0), (0, -2), (1, -2)
        ]
        for n in next_nearest_neighbors:
            n_vector = n[0] * a1 + n[1] * a2
            k_dot_n = np.sum(k * n_vector, axis=1)
            E_s += self.t_2 * np.exp(-1j * k_dot_n)
        
        return E_s
    
    def compute_U(self, k: np.ndarray) -> np.ndarray:
        """
        Compute the interaction U(k) for a triangular lattice.
        
        Args:
            k (np.ndarray): k points in the first Brillouin zone, can be of shape (N_k, 2) or (2,).
            
        Returns:
            np.ndarray: Interaction U(k) of shape (N_k,) or scalar.
        """
        # Get primitive lattice vectors
        a1 = self.primitive_vectors[0]
        a2 = self.primitive_vectors[1]
        
        # Reshape k if it's a single wave vector
        single_k = k.ndim == 1
        if single_k:
            k = k.reshape(1, 2)
        
        # Initialize U(k)
        U_k = np.zeros(k.shape[0], dtype=complex)
        
        # On-site interaction
        U_k += self.U_0
        
        # Nearest neighbor interaction
        for n in self.get_nearest_neighbor_vectors():
            n_vector = n[0] * a1 + n[1] * a2
            k_dot_n = np.sum(k * n_vector, axis=1)
            U_k += self.U_1 * np.exp(-1j * k_dot_n)
        
        # Return a scalar if input was a single wave vector
        if single_k:
            return U_k[0]
        return U_k
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute E_s(k) for all k-points
        E_s = self.compute_Es(self.k_space)
        
        # Populate the Hamiltonian with the kinetic term
        for s in range(self.D[0]):
            H_nonint[s, s, :] = E_s
        
        return H_nonint
    
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
        
        # Hartree term: U(0) <c_s†(k1) c_s(k1)> c_s'†(k2) c_s'(k2)
        for s in range(self.D[0]):
            # Mean density for species s across all k-points
            n_s = np.mean(exp_val[s, s, :])
            for s_prime in range(self.D[0]):
                # Populate the Hamiltonian with the Hartree term
                H_int[s_prime, s_prime, :] += self.U_0 * n_s / self.N_k
        
        # Fock term: -U(k1-k2) <c_s†(k1) c_s'(k1)> c_s'†(k2) c_s(k2)
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                # For all k2 points
                for k2_idx in range(self.N_k):
                    k2 = self.k_space[k2_idx]
                    # For all k1 points
                    for k1_idx in range(self.N_k):
                        k1 = self.k_space[k1_idx]
                        # Compute U(k1 - k2)
                        k_diff = k1 - k2
                        u_k1_minus_k2 = self.compute_U(k_diff)
                        # Fock term
                        H_int[s_prime, s, k2_idx] -= u_k1_minus_k2 * exp_val[s, s_prime, k1_idx] / self.N_k
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, return the flattened Hamiltonian.
            
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
