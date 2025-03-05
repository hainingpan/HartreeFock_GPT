import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with spin-dependent hopping and
    distance-dependent interactions.
    
    Args:
        N_shell (int): Number of shells for the k-space grid.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int=10, parameters: dict[str, Any]=None, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # LM Task: Tuple defining the number of spin flavors
        self.basis_order = {'0': 'spin'}  # Order: spin_up, spin_down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature set to 0
        
        if parameters is None:
            parameters = {
                'a': 1.0,        # Lattice constant
                't_up': 1.0,     # Hopping parameter for spin up
                't_down': 1.0,   # Hopping parameter for spin down
                'U_0': 1.0,      # On-site interaction
                'U_1': 0.5,      # Nearest neighbor interaction
                'U_2': 0.25,     # Next nearest neighbor interaction
                'U_3': 0.1       # Third nearest neighbor interaction
            }
        
        self.a = parameters['a']  # Lattice constant
        # Define the primitive (Bravais) lattice vectors for a 2D triangular lattice
        self.primitive_vectors = self.a * np.array([[1, 0], [0.5, np.sqrt(3)/2]])
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t_up = parameters['t_up']      # Hopping amplitude for spin up
        self.t_down = parameters['t_down']  # Hopping amplitude for spin down
        
        # Interaction parameters
        self.U_0 = parameters['U_0']  # On-site Coulomb repulsion
        self.U_1 = parameters['U_1']  # Nearest neighbor interaction
        self.U_2 = parameters['U_2']  # Next nearest neighbor interaction 
        self.U_3 = parameters['U_3']  # Third nearest neighbor interaction
        
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
            (1, 0),   # Right
            (0, 1),   # Upper-right
            (-1, 1),  # Upper-left
            (-1, 0),  # Left
            (0, -1),  # Lower-left
            (1, -1)   # Lower-right
        ]
        return n_vectors
    
    def get_second_neighbor_vectors(self):
        """Returns the coordinate offsets for second nearest neighbors."""
        n_vectors = [
            (1, 1),    # Upper-right-right
            (-1, 2),   # Upper-upper-left
            (-2, 1),   # Upper-left-left
            (-1, -1),  # Lower-left-left
            (1, -2),   # Lower-lower-right
            (2, -1)    # Lower-right-right
        ]
        return n_vectors
    
    def get_third_neighbor_vectors(self):
        """Returns the coordinate offsets for third nearest neighbors."""
        n_vectors = [
            (2, 0),    # Right-right
            (0, 2),    # Upper-upper-right
            (-2, 2),   # Upper-upper-left-left
            (-2, 0),   # Left-left
            (0, -2),   # Lower-lower-left
            (2, -2)    # Lower-lower-right-right
        ]
        return n_vectors

    def energy_dispersion(self, k, spin):
        """
        Computes the energy dispersion E_s(k) for a given spin.
        
        Args:
            k: k-points array
            spin: 0 for spin up, 1 for spin down
            
        Returns:
            Energy dispersion for the given spin and k-points
        """
        t = self.t_up if spin == 0 else self.t_down
        E_k = np.zeros(len(k), dtype=complex)
        
        # Sum over nearest neighbors: E_s(k) = sum_n t_s(n) * exp(-i * k . n)
        for n in self.get_nearest_neighbor_vectors():
            n_vector = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            for i, k_point in enumerate(k):
                E_k[i] += t * np.exp(-1j * np.dot(k_point, n_vector))
        
        return np.real(E_k)
    
    def interaction_potential(self, k_diff):
        """
        Computes the interaction potential U(k) for given k-difference.
        
        Args:
            k_diff: k-point difference (k1 - k2)
            
        Returns:
            Interaction potential value
        """
        # On-site interaction
        U_k = np.ones(len(k_diff)) * self.U_0
        
        # Add contributions from different neighbor shells
        for n, U_val in zip([self.get_nearest_neighbor_vectors(), 
                            self.get_second_neighbor_vectors(), 
                            self.get_third_neighbor_vectors()],
                           [self.U_1, self.U_2, self.U_3]):
            for vec in n:
                n_vector = vec[0] * self.primitive_vectors[0] + vec[1] * self.primitive_vectors[1]
                for i, k in enumerate(k_diff):
                    U_k[i] += U_val * np.exp(-1j * np.dot(k, n_vector))
        
        return np.real(U_k)

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian (kinetic energy).
        
        Returns:
            np.ndarray: Non-interacting Hamiltonian with shape (D+D+(N_k,)).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Compute energy dispersion for each spin
        H_nonint[0, 0, :] = self.energy_dispersion(self.k_space, 0)  # Spin up kinetic energy
        H_nonint[1, 1, :] = self.energy_dispersion(self.k_space, 1)  # Spin down kinetic energy
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian (Hartree and Fock terms).
        
        Args:
            exp_val: Expectation values array
            
        Returns:
            np.ndarray: Interacting Hamiltonian with shape (D+D+(N_k,)).
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Hartree term calculations
        n_up = np.mean(exp_val[0, 0, :])    # Average density of spin up
        n_down = np.mean(exp_val[1, 1, :])  # Average density of spin down
        
        # Add Hartree terms to diagonal elements
        H_int[0, 0, :] = self.U_0 * n_down / self.N_k  # Spin up interacting with average spin down
        H_int[1, 1, :] = self.U_0 * n_up / self.N_k    # Spin down interacting with average spin up
        
        # Fock term calculations
        for k2_idx in range(self.N_k):
            k2 = self.k_space[k2_idx]
            for k1_idx in range(self.N_k):
                k1 = self.k_space[k1_idx]
                k_diff = k1 - k2
                U_k = self.interaction_potential(np.array([k_diff]))[0]
                
                # Fock terms for all spin combinations
                # s=0,s'=0: Spin up-up interaction
                H_int[0, 0, k2_idx] -= U_k * exp_val[0, 0, k1_idx] / self.N_k
                
                # s=1,s'=1: Spin down-down interaction
                H_int[1, 1, k2_idx] -= U_k * exp_val[1, 1, k1_idx] / self.N_k
                
                # s=0,s'=1: Spin up-down interaction
                H_int[1, 0, k2_idx] -= U_k * exp_val[0, 1, k1_idx] / self.N_k
                
                # s=1,s'=0: Spin down-up interaction
                H_int[0, 1, k2_idx] -= U_k * exp_val[1, 0, k1_idx] / self.N_k
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hamiltonian by combining non-interacting and interacting parts.
        
        Args:
            exp_val: Expectation values array
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
