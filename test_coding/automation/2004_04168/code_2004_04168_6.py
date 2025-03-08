import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """HartreeFock Hamiltonian for a two-band system on a triangular lattice with
    nearest-neighbor and next-nearest-neighbor hopping, and on-site and 
    nearest-neighbor interactions.
    
    Args:
      N_shell (int): Number of shells in the first Brillouin zone.
      parameters (dict): Dictionary containing model parameters such as 't_1', 't_2', 'U_0', 'U_1'.
      filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_1': 6.0, 't_2': 1.0, 'U_0': 1.0, 'U_1': 0.5}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,) # 2 spin flavors
        self.basis_order = {'0': 'spin'}
        # Order for each flavor:
        # 0: spin up
        # 1: spin down

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        self.t_1 = parameters.get('t_1', 6.0) # Nearest-neighbor hopping in meV
        self.t_2 = parameters.get('t_2', 1.0) # Next-nearest-neighbor hopping in meV
        self.U_0 = parameters.get('U_0', 1.0) # On-site interaction
        self.U_1 = parameters.get('U_1', 0.5) # Nearest-neighbor interaction

        return

    def get_nearest_neighbor_vectors(self):
        """
        # Returns the integer coordinate offsets (n1, n2) corresponding to the 
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
        """
        Returns the integer coordinate offsets (n1, n2) corresponding to the 
        next-nearest neighbors in a 2D triangular Bravais lattice.
        """
        nn_vectors = [
            (2, 0),
            (0, 2),
            (2, 2),
            (-2, 0),
            (0, -2),
            (-2, -2),
            (1, -1),
            (-1, 1),
            (2, 1),
            (1, 2),
            (-2, -1),
            (-1, -2)
        ]
        return nn_vectors

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
          np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D,*self.D,self.N_k), dtype=complex)
        
        # Get nearest and next-nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        nnn_vectors = self.get_next_nearest_neighbor_vectors()
        
        # Compute dispersion E_s(k) using the hopping parameters
        # E_s(k) = sum_n t_s(n) * exp(-i * k . n)
        E_k = np.zeros(self.N_k, dtype=complex)
        
        # Contribution from nearest neighbors
        for nn in nn_vectors:
            # Convert neighbor vector to real space
            r = nn[0] * self.primitive_vectors[0] + nn[1] * self.primitive_vectors[1]
            # Compute k . r for all k points
            k_dot_r = np.sum(self.k_space * r, axis=1)
            # Add contribution to dispersion
            E_k += self.t_1 * np.exp(-1j * k_dot_r)
        
        # Contribution from next-nearest neighbors
        for nnn in nnn_vectors:
            # Convert neighbor vector to real space
            r = nnn[0] * self.primitive_vectors[0] + nnn[1] * self.primitive_vectors[1]
            # Compute k . r for all k points
            k_dot_r = np.sum(self.k_space * r, axis=1)
            # Add contribution to dispersion
            E_k += self.t_2 * np.exp(-1j * k_dot_r)
        
        # Set diagonal elements for both spin up and spin down (assuming they have the same dispersion)
        H_nonint[0, 0, :] = E_k  # Spin up
        H_nonint[1, 1, :] = E_k  # Spin down
        
        return H_nonint

    def compute_U_k(self, k):
        """
        Compute the momentum-dependent interaction potential.
        
        Args:
            k (np.ndarray): Momentum vector.
            
        Returns:
            complex: Interaction potential U(k).
        """
        U_k = self.U_0  # On-site interaction
        
        # Add contribution from nearest-neighbor interaction
        nn_vectors = self.get_nearest_neighbor_vectors()
        for nn in nn_vectors:
            r = nn[0] * self.primitive_vectors[0] + nn[1] * self.primitive_vectors[1]
            k_dot_r = np.sum(k * r)
            U_k += self.U_1 * np.exp(-1j * k_dot_r)
            
        return U_k

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
          exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
          np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D,*self.D,self.N_k), dtype=complex)

        # Compute average occupation for each flavor
        n_up = np.mean(exp_val[0, 0, :])  # Average occupation of spin up
        n_down = np.mean(exp_val[1, 1, :])  # Average occupation of spin down
        
        # Hartree term: H_Hartree = (1/N) * sum_{s,s',k1,k2} U(0) * <c_s^dag(k1) c_s(k1)> * c_s'^dag(k2) c_s'(k2)
        # For each spin s', this contributes to H[s',s',k2]
        H_int[0, 0, :] += self.U_0 * (n_up + n_down)  # Spin up interaction with average density
        H_int[1, 1, :] += self.U_0 * (n_up + n_down)  # Spin down interaction with average density
        
        # Fock term: H_Fock = -(1/N) * sum_{s,s',k1,k2} U(k1-k2) * <c_s^dag(k1) c_s'(k1)> * c_s'^dag(k2) c_s(k2)
        # This contributes to H[s',s,k2]
        
        # For each k2 and each pair of flavors (s,s')
        for k2_idx in range(self.N_k):
            for s in range(2):
                for s_prime in range(2):
                    # Compute \sum_{k1} U(k1-k2) * <c_s^dag(k1) c_s'(k1)>
                    fock_sum = 0.0
                    for k1_idx in range(self.N_k):
                        k_diff = self.k_space[k1_idx] - self.k_space[k2_idx]
                        U_k1_minus_k2 = self.compute_U_k(k_diff)
                        fock_sum += U_k1_minus_k2 * exp_val[s, s_prime, k1_idx]
                    
                    # Contribute to H[s',s,k2]
                    H_int[s_prime, s, k2_idx] -= fock_sum / self.N_k
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns a flattened version of the Hamiltonian.

        Returns:
            np.ndarray: The total Hamiltonian, either flattened or with shape (D, D, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
