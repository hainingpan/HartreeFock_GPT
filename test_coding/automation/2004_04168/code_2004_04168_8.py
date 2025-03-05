import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with spin-dependent hopping and interactions.
    
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system.
    """
    def __init__(self, N_shell: int = 5, parameters: dict[str, Any] = None, filling_factor: float = 0.5):
        if parameters is None:
            parameters = {'t_up': 1.0, 't_down': 1.0, 'U_0': 1.0, 'U_1': 0.5, 'a': 1.0}
        
        self.lattice = 'triangular'
        self.D = (2,)  # Spin-up and spin-down
        self.basis_order = {'0': 'spin'}
        # Order for each flavor: 0 = up, 1 = down

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Zero temperature
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = self.a * np.array([[1, 0], [0.5, np.sqrt(3)/2]])  # Triangular lattice primitive vectors
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t_up = parameters.get('t_up', 1.0)  # Hopping parameter for spin-up
        self.t_down = parameters.get('t_down', 1.0)  # Hopping parameter for spin-down
        
        # Interaction parameters
        self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction strength
        self.U_1 = parameters.get('U_1', 0.5)  # Nearest-neighbor interaction strength
    
    def get_nearest_neighbor_vectors(self):
        """
        Returns the integer coordinate offsets (n1, n2) corresponding to the 
        nearest neighbors in a 2D triangular Bravais lattice.

        For a 2D triangular lattice, there are six nearest neighbors.
        """
        n_vectors = [
            (1, 0),   # Right
            (0, 1),   # Top-right
            (-1, 1),  # Top-left
            (-1, 0),  # Left
            (0, -1),  # Bottom-left
            (1, -1)   # Bottom-right
        ]
        return n_vectors
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generate the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: Non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Get nearest neighbor vectors for the triangular lattice
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Calculate dispersion relation E_s(k) for each spin and k-point
        for k_idx, k in enumerate(self.k_space):
            # Spin up dispersion
            dispersion_up = 0
            # Spin down dispersion
            dispersion_down = 0
            
            for n in nn_vectors:
                # Convert n from integer coordinates to real space displacement
                n_real = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                # Calculate k·n
                k_dot_n = np.dot(k, n_real)
                # Add contribution to dispersion
                dispersion_up += self.t_up * np.exp(-1j * k_dot_n)
                dispersion_down += self.t_down * np.exp(-1j * k_dot_n)
            
            # Assign to Hamiltonian
            H_nonint[0, 0, k_idx] = dispersion_up  # Spin up kinetic term
            H_nonint[1, 1, k_idx] = dispersion_down  # Spin down kinetic term
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation values.
            
        Returns:
            np.ndarray: Interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Get nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Calculate average densities for Hartree term
        density_up = np.mean(exp_val[0, 0, :])
        density_down = np.mean(exp_val[1, 1, :])
        
        # Hartree term: (1/N) ∑_{s,s'} ∑_{k1,k2} U(0) ⟨c_s^†(k1) c_s(k1)⟩ c_{s'}^†(k2) c_{s'}(k2)
        # For spin up (s'=0)
        H_int[0, 0, :] += self.U_0 * density_down / self.N_k  # Interaction with average spin down density
        # For spin down (s'=1)
        H_int[1, 1, :] += self.U_0 * density_up / self.N_k  # Interaction with average spin up density
        
        # Fock term: -(1/N) ∑_{s,s'} ∑_{k1,k2} U(k1-k2) ⟨c_s^†(k1) c_{s'}(k1)⟩ c_{s'}^†(k2) c_s(k2)
        for k1_idx in range(self.N_k):
            for k2_idx in range(self.N_k):
                k_diff = self.k_space[k1_idx] - self.k_space[k2_idx]
                
                # Calculate U(k_diff) = U_0 + U_1 * sum_n e^(-i k_diff·n)
                U_k_diff = self.U_0
                for n in nn_vectors:
                    n_real = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                    k_dot_n = np.dot(k_diff, n_real)
                    U_k_diff += self.U_1 * np.exp(-1j * k_dot_n)
                
                # Apply Fock terms for all spin combinations
                # For s=0, s'=0 (up-up)
                H_int[0, 0, k2_idx] -= (U_k_diff / self.N_k) * exp_val[0, 0, k1_idx]
                # For s=0, s'=1 (up-down)
                H_int[1, 0, k2_idx] -= (U_k_diff / self.N_k) * exp_val[0, 1, k1_idx]
                # For s=1, s'=0 (down-up)
                H_int[0, 1, k2_idx] -= (U_k_diff / self.N_k) * exp_val[1, 0, k1_idx]
                # For s=1, s'=1 (down-down)
                H_int[1, 1, k2_idx] -= (U_k_diff / self.N_k) * exp_val[1, 1, k1_idx]
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generate the total Hamiltonian (non-interacting + interacting).
        
        Args:
            exp_val (np.ndarray): Expectation values.
            return_flat (bool): If True, returns the flattened Hamiltonian.
            
        Returns:
            np.ndarray: Total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened_hamiltonian(H_total, self.D, self.N_k)
        else:
            return H_total
