import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with spin-dependent hopping and interactions.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters:
            - t1: Nearest-neighbor hopping parameter (default: 6.0 meV)
            - t2: Next-nearest-neighbor hopping parameter (default: 1.0 meV)
            - U0: On-site interaction strength (default: 1.0)
            - U1: Nearest-neighbor interaction strength (default: 0.5)
            - T: Temperature (default: 0)
            - a: Lattice constant (default: 1.0)
        filling_factor (float): Filling factor for the system, between 0 and 1.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t1': 6.0, 't2': 1.0, 'U0': 1.0, 'U1': 0.5, 'T': 0.0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'   # Lattice symmetry
        self.D = (2,)  # Number of flavors (spin)
        self.basis_order = {'0': 'spin'}
        # Order for each flavor:
        # 0: spin up, spin down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t1 = parameters.get('t1', 6.0)  # Nearest-neighbor hopping (in meV)
        self.t2 = parameters.get('t2', 1.0)  # Next-nearest-neighbor hopping (in meV)
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
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get nearest and next-nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Next-nearest neighbor vectors for a triangular lattice
        nnn_vectors = [
            (2, 0), (0, 2), (2, 2), (-2, 0), (0, -2), (-2, -2),
            (1, -1), (-1, 1), (2, 1), (1, 2), (-2, -1), (-1, -2)
        ]
        
        # Calculate dispersion relation E_s(k) for each k-point and spin
        for s in range(self.D[0]):  # Loop over spins
            for k_idx in range(self.N_k):
                k = self.k_space[k_idx]
                
                # Contribution from nearest-neighbor hopping
                for nn in nn_vectors:
                    delta_r = nn[0] * self.primitive_vectors[0] + nn[1] * self.primitive_vectors[1]
                    H_nonint[s, s, k_idx] += self.t1 * np.exp(-1j * np.dot(k, delta_r))
                
                # Contribution from next-nearest-neighbor hopping
                for nnn in nnn_vectors:
                    delta_r = nnn[0] * self.primitive_vectors[0] + nnn[1] * self.primitive_vectors[1]
                    H_nonint[s, s, k_idx] += self.t2 * np.exp(-1j * np.dot(k, delta_r))
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian, including Hartree and Fock terms.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate average spin densities for Hartree term
        n_up = np.mean(exp_val[0, 0, :])    # Average density of spin-up
        n_down = np.mean(exp_val[1, 1, :])  # Average density of spin-down
        
        # Calculate Hartree term (on-site interaction)
        # U(0) * <c_s^dagger(k1) c_s(k1)> * c_s'^dagger(k2) c_s'(k2)
        H_int[0, 0, :] += self.U0 * n_down  # Contribution from spin-down to spin-up
        H_int[1, 1, :] += self.U0 * n_up    # Contribution from spin-up to spin-down
        
        # Calculate Fock term (on-site interaction)
        # -U0 * <c_s^dagger(k1) c_s'(k1)> * c_s'^dagger(k2) c_s(k2)
        
        # For s = s', the Fock term contributes to diagonal elements
        H_int[0, 0, :] -= self.U0 * n_up    # Contribution from spin-up to spin-up
        H_int[1, 1, :] -= self.U0 * n_down  # Contribution from spin-down to spin-down
        
        # For s != s', the Fock term contributes to off-diagonal elements
        # <c_up^dagger(k1) c_down(k1)>
        n_up_down = np.mean(exp_val[0, 1, :])
        # <c_down^dagger(k1) c_up(k1)>
        n_down_up = np.mean(exp_val[1, 0, :])
        
        H_int[0, 1, :] -= self.U0 * n_up_down  # Contribution from mixed terms
        H_int[1, 0, :] -= self.U0 * n_down_up  # Contribution from mixed terms
        
        # Calculate Fock term (nearest-neighbor interaction)
        # -U1(k1-k2) * <c_s^dagger(k1) c_s'(k1)> * c_s'^dagger(k2) c_s(k2)
        
        # Get nearest-neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Calculate U1(k1-k2) for all pairs of k1, k2
        for s in range(self.D[0]):  # Loop over spins
            for s_prime in range(self.D[0]):  # Loop over spins
                for k2_idx in range(self.N_k):
                    k2 = self.k_space[k2_idx]
                    for k1_idx in range(self.N_k):
                        k1 = self.k_space[k1_idx]
                        k_diff = k1 - k2
                        
                        # Calculate U(k1-k2) for nearest-neighbor interaction
                        U_nn = 0
                        for nn in nn_vectors:
                            delta_r = nn[0] * self.primitive_vectors[0] + nn[1] * self.primitive_vectors[1]
                            U_nn += self.U1 * np.exp(-1j * np.dot(k_diff, delta_r))
                        
                        # Add nearest-neighbor Fock term
                        H_int[s, s_prime, k2_idx] -= U_nn / self.N_k * exp_val[s, s_prime, k1_idx]
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns the flattened Hamiltonian.

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
