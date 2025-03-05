import numpy as np
from typing import Any, Dict
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with spin-dependent hopping and interactions.
    
    Args:
        N_shell (int): Number of shells to include in k-space.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, defaults to 0.5.
    """
    def __init__(self, N_shell: int = 5, parameters: Dict[str, Any] = None, filling_factor: float = 0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # Spin degree of freedom (up and down)
        self.basis_order = {'0': 'spin'}  # Order for each flavor: 0: spin up, spin down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature set to 0
        
        # Default parameters if none provided
        if parameters is None:
            parameters = {
                'a': 1.0,  # Lattice constant
                't_0_up': 0.0,  # On-site energy for spin up
                't_1_up': 1.0,  # Nearest-neighbor hopping for spin up
                't_0_down': 0.0,  # On-site energy for spin down
                't_1_down': 1.0,  # Nearest-neighbor hopping for spin down
                'U_0': 1.0,  # On-site interaction
                'U_1': 0.5   # Nearest-neighbor interaction
            }
        
        # Lattice parameters
        self.a = parameters.get('a', 1.0)
        # Primitive vectors for triangular lattice
        self.primitive_vectors = self.a * np.array([
            [1, 0],
            [0.5, np.sqrt(3)/2]
        ])
        
        # Hopping parameters for different shells and spins
        self.t_up = {0: parameters.get('t_0_up', 0.0),
                    1: parameters.get('t_1_up', 1.0)}
        self.t_down = {0: parameters.get('t_0_down', 0.0),
                      1: parameters.get('t_1_down', 1.0)}
        
        # Interaction parameters for different shells            
        self.U = {0: parameters.get('U_0', 1.0),
                 1: parameters.get('U_1', 0.5)}
        
        # Generate k-space
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
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
            (0, 1),   # Up-right
            (-1, 1),  # Up-left
            (-1, 0),  # Left
            (0, -1),  # Down-left
            (1, -1)   # Down-right
        ]
        return n_vectors
    
    def calculate_dispersion(self, k, spin):
        """
        Calculate the dispersion relation E_s(k) for a given k-point and spin.
        
        Args:
            k: k-point coordinates
            spin: 0 for up, 1 for down
            
        Returns:
            Energy dispersion
        """
        # Choose hopping parameters based on spin
        t_s = self.t_up if spin == 0 else self.t_down
        
        # On-site energy
        E = t_s[0]
        
        # Add contribution from nearest neighbors
        neighbors = self.get_nearest_neighbor_vectors()
        for n in neighbors:
            # Convert n to real space vector using primitive vectors
            real_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            # Add hopping contribution
            E += t_s[1] * np.exp(-1j * np.dot(k, real_n))
        
        return E
    
    def calculate_interaction(self, k_diff):
        """
        Calculate the interaction U(k) for a given k-point difference.
        
        Args:
            k_diff: k-point difference coordinates
            
        Returns:
            Interaction strength
        """
        # On-site interaction
        U_k = self.U[0]
        
        # Add contribution from nearest neighbors
        neighbors = self.get_nearest_neighbor_vectors()
        for n in neighbors:
            # Convert n to real space vector using primitive vectors
            real_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            # Add interaction contribution
            U_k += self.U[1] * np.exp(-1j * np.dot(k_diff, real_n))
        
        return U_k
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Calculate dispersion for each k-point and spin
        for i, k in enumerate(self.k_space):
            # Diagonal elements corresponding to kinetic energy terms
            H_nonint[0, 0, i] = self.calculate_dispersion(k, 0)  # Spin up
            H_nonint[1, 1, i] = self.calculate_dispersion(k, 1)  # Spin down
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        # Unflatten exp_val to shape (D, D, N_k)
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)
        
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Hartree term: U(0) * <n_s(k1)> * n_s'(k2)
        for s in range(2):  # Spin up and down
            # Calculate average density for spin s
            n_s = np.mean(exp_val[s, s, :])
            
            for s_prime in range(2):  # Spin up and down
                # Add Hartree term to diagonal elements for spin s'
                H_int[s_prime, s_prime, :] += self.U[0] * n_s / self.N_k
        
        # Fock term: -U(k1-k2) * <c_s†(k1) c_s'(k1)> * c_s'†(k2) c_s(k2)
        for k2_idx in range(self.N_k):
            k2 = self.k_space[k2_idx]
            for k1_idx in range(self.N_k):
                k1 = self.k_space[k1_idx]
                k_diff = k1 - k2
                U_k_diff = self.calculate_interaction(k_diff)
                
                for s in range(2):  # Spin s
                    for s_prime in range(2):  # Spin s'
                        # Fock term contributes to off-diagonal elements
                        H_int[s_prime, s, k2_idx] -= U_k_diff * exp_val[s, s_prime, k1_idx] / self.N_k
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            return_flat (bool): Whether to return the flattened Hamiltonian.
            
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
