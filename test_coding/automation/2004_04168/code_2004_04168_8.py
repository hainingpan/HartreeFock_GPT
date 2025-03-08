import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with spin-dependent interactions.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor for the bands. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'U_0': 1.0, 'U_1': 0.5, 't_1': 6.0, 't_2': 1.0, 'T': 0.0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # 2 spin flavors (up and down)
        self.basis_order = {'0': 'spin'}
        # 0: spin up, spin down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t_1 = parameters.get('t_1', 6.0)  # Nearest-neighbor hopping (6 meV)
        self.t_2 = parameters.get('t_2', 1.0)  # Next-nearest-neighbor hopping (1 meV)
        
        # Interaction parameters
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
    
    def get_next_nearest_neighbor_vectors(self):
        """Returns the integer coordinate offsets for next-nearest neighbors in a triangular lattice"""
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
            (-1, -2),
        ]
        return n_vectors

    def compute_dispersion(self, k_points):
        """
        Compute the energy dispersion E_s(k) for each k-point.
        
        Args:
            k_points (np.ndarray): Array of k-points with shape (N_k, 2).
            
        Returns:
            np.ndarray: Energy dispersion for each k-point with shape (N_k,).
        """
        # Get the primitive vectors in real space
        a1, a2 = self.primitive_vectors
        
        # Get the nearest and next-nearest neighbor offsets
        nn_offsets = self.get_nearest_neighbor_vectors()
        nnn_offsets = self.get_next_nearest_neighbor_vectors()
        
        # Compute the positions of the nearest and next-nearest neighbors
        nn_positions = [n1 * a1 + n2 * a2 for n1, n2 in nn_offsets]
        nnn_positions = [n1 * a1 + n2 * a2 for n1, n2 in nnn_offsets]
        
        # Compute the dispersion E_s(k)
        E_k = np.zeros(len(k_points), dtype=complex)
        
        for k in range(len(k_points)):
            # Contribution from nearest neighbors (t_1 term)
            for pos in nn_positions:
                E_k[k] += self.t_1 * np.exp(-1j * np.dot(k_points[k], pos))
            
            # Contribution from next-nearest neighbors (t_2 term)
            for pos in nnn_positions:
                E_k[k] += self.t_2 * np.exp(-1j * np.dot(k_points[k], pos))
        
        return E_k

    def compute_interaction_potential(self, k_diff):
        """
        Compute the interaction potential U(k) for a given k-difference.
        
        Args:
            k_diff (np.ndarray): k-point difference vector.
            
        Returns:
            complex: Interaction potential U(k_diff).
        """
        # Get the primitive vectors in real space
        a1, a2 = self.primitive_vectors
        
        # Get the nearest neighbor offsets
        nn_offsets = self.get_nearest_neighbor_vectors()
        
        # Compute the positions of the nearest neighbors
        nn_positions = [n1 * a1 + n2 * a2 for n1, n2 in nn_offsets]
        
        # Compute U(k) = U_0 + U_1 * sum_{n in nearest neighbors} e^(-i k·n)
        U_k = self.U_0
        for pos in nn_positions:
            U_k += self.U_1 * np.exp(-1j * np.dot(k_diff, pos))
        
        return U_k

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute the dispersion E_s(k) for each k-point
        E_k = self.compute_dispersion(self.k_space)
        
        # Set the diagonal elements for both spin channels
        # Since the dispersion is the same for both spin up and spin down,
        # we set both diagonal elements to the same value
        H_nonint[0, 0, :] = E_k  # Up-up channel
        H_nonint[1, 1, :] = E_k  # Down-down channel
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian including Hartree and Fock terms.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Hartree term: U(0) * <c_s†(k1) c_s(k1)> * c_s'†(k2) c_s'(k2)
        # Calculate the mean density for each spin
        n_up = np.mean(exp_val[0, 0, :])  # Mean of <c_up†(k) c_up(k)>
        n_down = np.mean(exp_val[1, 1, :])  # Mean of <c_down†(k) c_down(k)>
        
        # Add the Hartree term to the diagonal elements
        H_int[0, 0, :] += self.U_0 * n_down  # Up-up channel gets contribution from down density
        H_int[1, 1, :] += self.U_0 * n_up    # Down-down channel gets contribution from up density
        
        # Fock term: -U(k1-k2) * <c_s†(k1) c_s'(k1)> * c_s'†(k2) c_s(k2)
        # For each k2 point, sum over all k1 points with the appropriate interaction
        for k2 in range(self.N_k):
            for k1 in range(self.N_k):
                # Compute U(k1-k2)
                k_diff = self.k_space[k1] - self.k_space[k2]
                U_k = self.compute_interaction_potential(k_diff)
                
                # Add Fock contributions to all matrix elements
                # Up-up channel gets contribution from up-up correlation
                H_int[0, 0, k2] -= (1 / self.N_k) * U_k * exp_val[0, 0, k1]
                
                # Up-down channel gets contribution from down-up correlation
                H_int[0, 1, k2] -= (1 / self.N_k) * U_k * exp_val[1, 0, k1]
                
                # Down-up channel gets contribution from up-down correlation
                H_int[1, 0, k2] -= (1 / self.N_k) * U_k * exp_val[0, 1, k1]
                
                # Down-down channel gets contribution from down-down correlation
                H_int[1, 1, k2] -= (1 / self.N_k) * U_k * exp_val[1, 1, k1]
        
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
