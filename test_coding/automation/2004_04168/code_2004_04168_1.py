import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """A class implementing the Hartree-Fock Hamiltonian for a triangular lattice model with 
    on-site and nearest-neighbor interactions.

    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_1': 6.0, 't_2': 1.0, 'U_0': 1.0, 'U_1': 0.5, 'T': 0.0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # Dimension for spin (up, down)
        self.basis_order = {'0': 'spin'}
        # Order for spin: 0 = up, 1 = down

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Hopping parameters
        self.t_1 = parameters.get('t_1', 6.0)  # nearest-neighbor hopping (6 meV)
        self.t_2 = parameters.get('t_2', 1.0)  # next-nearest-neighbor hopping (1 meV)
        
        # Interaction parameters
        self.U_0 = parameters.get('U_0', 1.0)  # on-site interaction
        self.U_1 = parameters.get('U_1', 0.5)  # nearest-neighbor interaction

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
        n_vectors = [
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
        return n_vectors

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get the nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        nnn_vectors = self.get_next_nearest_neighbor_vectors()
        
        # Compute E_s(k) for each k-point using the hopping parameters
        for k_idx in range(self.N_k):
            k = self.k_space[k_idx]
            
            # Calculate E_s(k) = sum_n t_s(n) * exp(-i * k · n)
            # For both spin up and down (same dispersion)
            E_k = 0.0
            
            # Nearest-neighbor contribution (t_1)
            for nn_vec in nn_vectors:
                k_dot_n = k[0] * nn_vec[0] + k[1] * nn_vec[1]
                E_k += self.t_1 * np.exp(-1j * k_dot_n)
            
            # Next-nearest-neighbor contribution (t_2)
            for nnn_vec in nnn_vectors:
                k_dot_n = k[0] * nnn_vec[0] + k[1] * nnn_vec[1]
                E_k += self.t_2 * np.exp(-1j * k_dot_n)
            
            # Assign the same dispersion to both spin up and spin down
            H_nonint[0, 0, k_idx] = E_k
            H_nonint[1, 1, k_idx] = E_k
        
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
        
        # Calculate mean densities for Hartree term
        n_up = np.mean(exp_val[0, 0, :])  # <c_up^dagger(k_1) c_up(k_1)>
        n_down = np.mean(exp_val[1, 1, :])  # <c_down^dagger(k_1) c_down(k_1)>
        
        # Calculate mean spin-flip terms for Fock term
        n_updown = np.mean(exp_val[0, 1, :])  # <c_up^dagger(k_1) c_down(k_1)>
        n_downup = np.mean(exp_val[1, 0, :])  # <c_down^dagger(k_1) c_up(k_1)>
        
        # Get nearest neighbor vectors for U(k) calculation
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Hartree term: U(0) * n_s * c_s'^dagger c_s'
        # For spin up electrons interacting with average spin down density
        H_int[0, 0, :] = self.U_0 * n_down
        
        # For spin down electrons interacting with average spin up density
        H_int[1, 1, :] = self.U_0 * n_up
        
        # Fock term: -U(k_1 - k_2) * <c_s^dagger c_s'> * c_s'^dagger c_s
        for k_idx in range(self.N_k):
            k = self.k_space[k_idx]
            
            # Calculate U(k) for this k-point
            U_k = self.U_0  # On-site contribution
            for nn_vec in nn_vectors:
                k_dot_n = k[0] * nn_vec[0] + k[1] * nn_vec[1]
                U_k += self.U_1 * np.exp(-1j * k_dot_n)
            
            # Fock term for spin down->up transitions
            H_int[0, 1, k_idx] -= U_k * n_downup
            
            # Fock term for spin up->down transitions  
            H_int[1, 0, k_idx] -= U_k * n_updown
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian by combining non-interacting and interacting parts.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns a flattened Hamiltonian.
        
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
