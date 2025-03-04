import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a spin system on a triangular lattice.
    
    Args:
        N_shell (int): Number of "shells" in the k-space.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=10, parameters: dict={'t_up': 1.0, 't_down': 1.0, 'U_0': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # Spin up and spin down
        self.basis_order = {'0': 'spin'}
        # Order for each flavor:
        # 0: spin up, spin down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = self.a * np.array([[1, 0], [0.5, np.sqrt(3)/2]])  # Define the primitive (Bravais) lattice vectors for a 2D triangular lattice
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_up = parameters.get('t_up', 1.0)  # Hopping parameter for spin up
        self.t_down = parameters.get('t_down', 1.0)  # Hopping parameter for spin down
        self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction strength
        
    def get_nearest_neighbor_vectors(self):
        """
        Returns the integer coordinate offsets (n1, n2) corresponding to the 
        nearest neighbors in a 2D triangular Bravais lattice.
        """
        n_vectors = [
            (1, 0),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (0, -1),
            (1, -1)
        ]
        return n_vectors
    
    def compute_energy_dispersion(self, k_points, s):
        """
        Compute the energy dispersion for the given k points and spin.
        
        Args:
            k_points (np.ndarray): Array of k points.
            s (int): Spin index (0 for up, 1 for down).
            
        Returns:
            np.ndarray: Energy dispersion.
        """
        nn_vectors = self.get_nearest_neighbor_vectors()
        t = self.t_up if s == 0 else self.t_down
        
        # Compute the energy dispersion: E_s(k) = sum_n t_s(n) * exp(-i * k . n)
        energy = np.zeros(len(k_points), dtype=complex)
        for n in nn_vectors:
            # Convert n to real space using primitive vectors
            r = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            # Compute k . r
            k_dot_r = k_points[:, 0] * r[0] + k_points[:, 1] * r[1]
            # Add contribution to energy
            energy += t * np.exp(-1j * k_dot_r)
            
        return energy.real  # Return the real part as the energy should be real
    
    def compute_interaction_potential(self, k_diff):
        """
        Compute the interaction potential U(k) for the given k difference.
        
        Args:
            k_diff (np.ndarray): Array of k differences.
            
        Returns:
            np.ndarray: Interaction potential.
        """
        # Implementation of U(k) = sum_n U(n) * exp(-i * k . n)
        # For simplicity, we'll assume U(0) = U_0 (on-site interaction only)
        # For more complex interactions, this would need to be expanded
        return np.ones(len(k_diff)) * self.U_0
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Compute the kinetic energy for spin up and spin down
        # H_Kinetic = sum_{s, k} E_s(k) c^dagger_s(k) c_s(k)
        H_nonint[0, 0, :] = self.compute_energy_dispersion(self.k_space, 0)  # Spin up
        H_nonint[1, 1, :] = self.compute_energy_dispersion(self.k_space, 1)  # Spin down
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Hartree term: H_Hartree = (1/N) * sum_{s,s',k1,k2} U(0) * <c_s^dag(k1) c_s(k1)> * c_s'^dag(k2) c_s'(k2)
        for s in range(self.D[0]):
            for sp in range(self.D[0]):
                # Average density of spin s across all k points
                avg_density_s = np.mean(exp_val[s, s, :])
                # Hartree term contributes to diagonal elements of H for spin s'
                H_int[sp, sp, :] += (self.U_0 / self.N_k) * avg_density_s
        
        # Fock term: H_Fock = -(1/N) * sum_{s,s',k1,k2} U(k1-k2) * <c_s^dag(k1) c_s'(k1)> * c_s'^dag(k2) c_s(k2)
        for s in range(self.D[0]):
            for sp in range(self.D[0]):
                for k1 in range(self.N_k):
                    for k2 in range(self.N_k):
                        # Compute k1 - k2
                        k_diff = self.k_space[k1] - self.k_space[k2]
                        # Compute U(k1 - k2)
                        U_k = self.compute_interaction_potential(np.array([k_diff]))
                        # Fock term contributes to off-diagonal elements of H
                        H_int[sp, s, k2] -= (U_k[0] / self.N_k) * exp_val[s, sp, k1]
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array.
            return_flat (bool, optional): Whether to return the flattened Hamiltonian. Defaults to True.
            
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
