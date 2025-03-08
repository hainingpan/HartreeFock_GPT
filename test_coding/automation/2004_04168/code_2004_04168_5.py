import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with nearest and next-nearest neighbor hopping,
    on-site and nearest-neighbor interactions.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor for the system, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t1': 6.0, 't2': 1.0, 'U0': 1.0, 'U1': 0.5, 'T': 0.0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # 2 spin states (up, down)
        self.basis_order = {'0': 'spin'}
        # Order for spin: 0=up, 1=down
        
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
        Generates the non-interacting part of the Hamiltonian (kinetic term).
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Calculate E_s(k) for each k point
        for s in range(self.D[0]):  # Loop over spin
            for k_idx in range(self.N_k):  # Loop over k points
                k = self.k_space[k_idx]
                
                # Nearest-neighbor hopping contribution
                E_k = 0
                for n1, n2 in nn_vectors:
                    # Convert to real space vector
                    n_vec = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
                    # Add contribution to E_s(k)
                    E_k += self.t1 * np.exp(-1j * np.dot(k, n_vec))
                
                # Next-nearest-neighbor hopping contribution
                for n1 in [-1, 0, 1]:
                    for n2 in [-1, 0, 1]:
                        if abs(n1) + abs(n2) == 2:  # Next-nearest neighbors
                            n_vec = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
                            E_k += self.t2 * np.exp(-1j * np.dot(k, n_vec))
                
                # Set the diagonal element for this spin and k-point
                H_nonint[s, s, k_idx] = E_k
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian (Hartree and Fock terms).
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate densities for each spin
        n_up = np.mean(exp_val[0, 0, :])  # Mean density of up spins
        n_down = np.mean(exp_val[1, 1, :])  # Mean density of down spins
        
        # Hartree term: on-site interaction between different spins
        H_int[0, 0, :] += (1.0 / self.N_k) * self.U0 * n_down  # Up-up interaction with down density
        H_int[1, 1, :] += (1.0 / self.N_k) * self.U0 * n_up    # Down-down interaction with up density
        
        # Fock term: exchange interaction between spins
        # We need to calculate U(k1-k2) for all pairs of k-points
        for k1_idx in range(self.N_k):
            for k2_idx in range(self.N_k):
                k_diff = self.k_space[k1_idx] - self.k_space[k2_idx]
                
                # Calculate U(k_diff)
                U_k_diff = self.U0  # On-site contribution
                
                # Add nearest-neighbor contribution
                for n1, n2 in self.get_nearest_neighbor_vectors():
                    n_vec = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
                    U_k_diff += self.U1 * np.exp(-1j * np.dot(k_diff, n_vec))
                
                # Add Fock contribution for each spin pair
                H_int[1, 0, k2_idx] -= (1.0 / self.N_k) * U_k_diff * exp_val[0, 1, k1_idx]
                H_int[0, 1, k2_idx] -= (1.0 / self.N_k) * U_k_diff * exp_val[1, 0, k1_idx]
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns a flattened Hamiltonian.
            
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
