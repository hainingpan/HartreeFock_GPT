import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a spin system on a triangular lattice.
    
    Args:
        N_shell (int): Number of shells in the Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=5, parameters: dict={'t0': 1.0, 't1': 0.1, 'U0': 1.0, 'U1': 0.5, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # Only spin flavor (up, down)
        self.basis_order = {'0': 'spin'}  # Order for spin: up (0), down (1)
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature is 0 as specified in the problem
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = self.a * np.array([[1, 0], [0.5, np.sqrt(3)/2]])  # Define primitive vectors for triangular lattice
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters (for E_s(k))
        self.t0 = parameters.get('t0', 1.0)  # On-site hopping
        self.t1 = parameters.get('t1', 0.1)  # Nearest-neighbor hopping
        
        # Interaction parameters (for U(k))
        self.U0 = parameters.get('U0', 1.0)  # On-site interaction
        self.U1 = parameters.get('U1', 0.5)  # Nearest-neighbor interaction
        
        return
        
    def get_nearest_neighbor_vectors(self):
        """
        Returns the integer coordinate offsets (n1, n2) corresponding to the 
        nearest neighbors in a 2D triangular Bravais lattice. These offsets are ONLY
        valid for a lattice whose two primitive vectors are separated by 120°.

        To obtain the real-space displacements for each neighbor, multiply these 
        integer pairs by the primitive vectors a1 and a2, i.e.:
            R_neighbor = n1 * a1 + n2 * a2

        For a 2D triangular lattice, there are six nearest neighbors.
        """
        n_vectors = [
            (1, 0),    # Nearest neighbor along a1
            (0, 1),    # Nearest neighbor along a2
            (-1, 1),   # Nearest neighbor at a2 - a1
            (-1, 0),   # Nearest neighbor at -a1
            (0, -1),   # Nearest neighbor at -a2
            (1, -1)    # Nearest neighbor at a1 - a2
        ]
        return n_vectors
        
    def compute_dispersion(self, k_points):
        """
        Computes the dispersion relation E_s(k) for the given k points.
        
        Args:
            k_points (np.ndarray): Array of k points with shape (N_k, 2).
            
        Returns:
            np.ndarray: Dispersion values with shape (N_k,).
        """
        # For simplicity, we assume that the dispersion is the same for both spin directions
        dispersion = np.zeros(k_points.shape[0], dtype=complex)
        
        # On-site contribution
        dispersion += self.t0
        
        # Nearest-neighbor contribution
        nn_vectors = self.get_nearest_neighbor_vectors()
        for n in nn_vectors:
            n_vec = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            for i in range(k_points.shape[0]):
                dispersion[i] += self.t1 * np.exp(-1j * np.dot(k_points[i], n_vec))
            
        return dispersion
        
    def compute_interaction(self, k_diff):
        """
        Computes the interaction potential U(k) for the given k difference.
        
        Args:
            k_diff (np.ndarray): k difference vector with shape (2,).
            
        Returns:
            float: Interaction value.
        """
        interaction = self.U0  # On-site contribution
        
        # Nearest-neighbor contribution
        nn_vectors = self.get_nearest_neighbor_vectors()
        for n in nn_vectors:
            n_vec = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            interaction += self.U1 * np.exp(-1j * np.dot(k_diff, n_vec))
            
        return interaction
        
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D + D + (N_k,)).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Calculate the dispersion relation
        dispersion = self.compute_dispersion(self.k_space)
        
        # Add the kinetic energy term for both spin up and spin down
        for s in range(self.D[0]):
            H_nonint[s, s, :] = dispersion  # Corresponds to E_s(k) c_s^†(k) c_s(k)
            
        return H_nonint
        
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D + D + (N_k,)).
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Hartree term: (1/N) ∑_{s,s'} ∑_{k1,k2} U(0) <c_s^†(k1) c_s(k1)> c_s'^†(k2) c_s'(k2)
        for s in range(self.D[0]):
            # Calculate average density for spin s
            avg_density_s = np.mean(exp_val[s, s, :])
            
            for sp in range(self.D[0]):
                # Contribution to H_int[sp, sp, :] for all k2
                H_int[sp, sp, :] += self.U0 / self.N_k * avg_density_s
        
        # Fock term: -(1/N) ∑_{s,s'} ∑_{k1,k2} U(k1-k2) <c_s^†(k1) c_s'(k1)> c_s'^†(k2) c_s(k2)
        for k2 in range(self.N_k):
            for k1 in range(self.N_k):
                k_diff = self.k_space[k1] - self.k_space[k2]
                U_k_diff = self.compute_interaction(k_diff)
                
                for s in range(self.D[0]):
                    for sp in range(self.D[0]):
                        # Fock term contribution to H_int[sp, s, k2]
                        H_int[sp, s, k2] -= U_k_diff / self.N_k * exp_val[s, sp, k1]
        
        return H_int
        
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            return_flat (bool, optional): Whether to return a flattened Hamiltonian. Defaults to True.
            
        Returns:
            np.ndarray: The total Hamiltonian, either flattened or not.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return flattened_hamiltonian(H_total, self.D, self.N_k)
        else:
            return H_total
