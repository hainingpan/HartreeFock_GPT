import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system on a triangular lattice with spin.
    
    The Hamiltonian consists of:
    1. Kinetic term with nearest-neighbor and next-nearest-neighbor hopping
    2. Hartree term with on-site interaction
    3. Fock term with on-site and nearest-neighbor interactions
    
    Args:
        N_shell (int): Number of shells in k-space for Brillouin zone sampling
        parameters (dict): Dictionary of model parameters
        filling_factor (float): Filling factor, default is 0.5
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t1': 6.0, 't2': 1.0, 'U0': 1.0, 'U1': 0.5, 'T': 0.0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # Two spin states (up and down)
        self.basis_order = {'0': 'spin'}
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t1 = parameters.get('t1', 6.0)  # nearest-neighbor hopping (in meV)
        self.t2 = parameters.get('t2', 1.0)  # next-nearest-neighbor hopping (in meV)
        
        # Interaction parameters
        self.U0 = parameters.get('U0', 1.0)  # on-site interaction
        self.U1 = parameters.get('U1', 0.5)  # nearest-neighbor interaction
        
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
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Nearest-neighbor vectors (using the provided method)
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Next-nearest-neighbor vectors (using `get_shell_index_triangle(2)` and filtering)
        shell2_indices = get_shell_index_triangle(2)
        all_shell2_vectors = [(i, j) for i, j in zip(shell2_indices[0], shell2_indices[1])]
        # Filter out the nearest neighbors (shell 1) to get only the next-nearest neighbors
        nnn_vectors = [vec for vec in all_shell2_vectors if vec not in nn_vectors and vec != (0, 0)]
        
        # Calculate kinetic energy terms E_s(k) for each spin
        for s in range(self.D[0]):  # Loop over spin states
            for k_idx in range(self.N_k):  # Loop over k-points
                k = self.k_space[k_idx]
                E_k = 0.0
                
                # Nearest-neighbor hopping
                for n_i, n_j in nn_vectors:
                    # Convert lattice coordinates to real space using primitive vectors
                    n = n_i * self.primitive_vectors[0] + n_j * self.primitive_vectors[1]
                    E_k += self.t1 * np.exp(-1j * np.dot(k, n))
                
                # Next-nearest-neighbor hopping
                for n_i, n_j in nnn_vectors:
                    # Convert lattice coordinates to real space using primitive vectors
                    n = n_i * self.primitive_vectors[0] + n_j * self.primitive_vectors[1]
                    E_k += self.t2 * np.exp(-1j * np.dot(k, n))
                
                H_nonint[s, s, k_idx] = E_k
        
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
        
        # Compute the mean density for each spin
        mean_density = np.zeros(self.D[0], dtype=complex)
        for s in range(self.D[0]):
            mean_density[s] = np.mean(exp_val[s, s, :])
        
        # Hartree term: U(0) * <n_s> * n_s'
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                # The Hartree term adds to the on-site energy for spin s_prime
                H_int[s_prime, s_prime, :] += (self.U0 / self.N_k) * mean_density[s]
        
        # Fock term: -U(k1-k2) * <c_s^dag(k1) c_s'(k1)> * c_s'^dag(k2) c_s(k2)
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                for k2_idx in range(self.N_k):
                    k2 = self.k_space[k2_idx]
                    
                    fock_sum = 0.0
                    for k1_idx in range(self.N_k):
                        k1 = self.k_space[k1_idx]
                        dk = k1 - k2
                        
                        # Calculate U(k1-k2) including on-site and nearest-neighbor interactions
                        U_dk = self.U0  # On-site contribution
                        
                        # Add nearest-neighbor contributions
                        for n_i, n_j in self.get_nearest_neighbor_vectors():
                            # Convert lattice coordinates to real space using primitive vectors
                            n = n_i * self.primitive_vectors[0] + n_j * self.primitive_vectors[1]
                            U_dk += self.U1 * np.exp(-1j * np.dot(dk, n))
                        
                        fock_sum += U_dk * exp_val[s, s_prime, k1_idx]
                    
                    # Add the Fock term to the Hamiltonian
                    H_int[s_prime, s, k2_idx] -= fock_sum / self.N_k
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, return a flattened Hamiltonian.
        
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
