import numpy as np
from typing import Any, Dict
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with spin-dependent hopping and interactions.
    
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary of model parameters.
        filling_factor (float): Filling factor of the system.
    """
    def __init__(self, N_shell: int = 5, parameters: Dict[str, Any] = None, filling_factor: float = 0.5):
        if parameters is None:
            parameters = {
                'a': 1.0,  # Lattice constant
                't_0': 1.0,  # On-site hopping
                't_1': 0.2,  # Nearest neighbor hopping
                'U_0': 1.0,  # On-site interaction
                'U_1': 0.5   # Nearest neighbor interaction
            }
        
        self.lattice = 'triangular'
        self.D = (2,)  # Spin up and spin down
        self.basis_order = {'0': 'spin'}
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature set to 0
        self.a = parameters.get('a', 1.0)
        
        # Define the primitive (Bravais) lattice vectors for a 2D triangular lattice
        self.primitive_vectors = self.a * np.array([[1, 0], [0.5, np.sqrt(3)/2]])
        
        # Generate k-space
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t_0 = parameters.get('t_0', 1.0)  # On-site hopping
        self.t_1 = parameters.get('t_1', 0.2)  # Nearest neighbor hopping
        
        # Interaction parameters
        self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction
        self.U_1 = parameters.get('U_1', 0.5)  # Nearest neighbor interaction
        
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
            (1, 0),   # Neighbor to the right
            (0, 1),   # Neighbor up-right
            (-1, 1),  # Neighbor up-left
            (-1, 0),  # Neighbor to the left
            (0, -1),  # Neighbor down-left
            (1, -1)   # Neighbor down-right
        ]
        return n_vectors
    
    def compute_dispersion(self, k, spin):
        """
        Compute the dispersion E_s(k) for a given wavevector k and spin s.
        
        Args:
            k (numpy.ndarray): Wavevector (kx, ky)
            spin (int): Spin index (0 for up, 1 for down)
            
        Returns:
            float: Dispersion energy for the given k and spin
        """
        # On-site term
        E = self.t_0
        
        # Nearest neighbor hopping
        neighbors = self.get_nearest_neighbor_vectors()
        for n in neighbors:
            n_vector = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            phase = np.exp(-1j * np.dot(k, n_vector))
            E += self.t_1 * phase.real
        
        return E
    
    def compute_interaction(self, k_diff):
        """
        Compute the interaction potential U(k_diff) for a given wavevector difference.
        
        Args:
            k_diff (numpy.ndarray): Difference between two wavevectors
            
        Returns:
            float: Interaction potential at the given k difference
        """
        # On-site interaction
        U_k = self.U_0
        
        # Nearest neighbor interaction
        neighbors = self.get_nearest_neighbor_vectors()
        for n in neighbors:
            n_vector = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            phase = np.exp(-1j * np.dot(k_diff, n_vector))
            U_k += self.U_1 * phase.real
        
        return U_k
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            numpy.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Compute dispersion for each k-point and spin
        for s in range(2):  # Loop over spins
            for i in range(self.N_k):
                k = self.k_space[i]
                H_nonint[s, s, i] = self.compute_dispersion(k, s)
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.
        
        Args:
            exp_val (numpy.ndarray): Expectation value with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            numpy.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Hartree term: U(0) * <c_s^dag(k1) c_s(k1)> * c_s'^dag(k2) c_s'(k2)
        # This contributes to the diagonal elements H[s', s', k]
        for s in range(2):  # Loop over first spin index
            n_s = np.mean(exp_val[s, s, :])  # Average density of spin s
            for s_prime in range(2):  # Loop over second spin index
                H_int[s_prime, s_prime, :] += self.U_0 / self.N_k * n_s
        
        # Fock term: -U(k1-k2) * <c_s^dag(k1) c_s'(k1)> * c_s'^dag(k2) c_s(k2)
        # This contributes to elements H[s', s, k]
        for s in range(2):  # Loop over first spin index
            for s_prime in range(2):  # Loop over second spin index
                for i in range(self.N_k):  # Loop over k1
                    for j in range(self.N_k):  # Loop over k2
                        k_diff = self.k_space[i] - self.k_space[j]
                        U_k_diff = self.compute_interaction(k_diff)
                        H_int[s_prime, s, j] -= U_k_diff / self.N_k * exp_val[s, s_prime, i]
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hamiltonian by combining the non-interacting and interacting parts.
        
        Args:
            exp_val (numpy.ndarray): Expectation value with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns a flattened version of the Hamiltonian.
            
        Returns:
            numpy.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened_hamiltonian(H_total, self.D, self.N_k)
        else:
            return H_total
