import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with spin degrees of freedom on a triangular lattice.
    
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system. Default is 0.5.
    """
    def __init__(self, N_shell: int=10, parameters: dict=None, filling_factor: float=0.5):
        if parameters is None:
            parameters = {'t0': 1.0, 't1': 0.2, 'U0': 1.0, 'U1': 0.5, 'a': 1.0}
            
        self.lattice = 'triangular'
        self.D = (2,)  # LM Task: has to define this tuple - 2 spin states (up and down)
        self.basis_order = {'0': 'spin'}
        # Order for each flavor:
        # 0: spin up, spin down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0.0  # Temperature set to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        
        # Define the primitive (Bravais) lattice vectors for a 2D triangular lattice
        self.primitive_vectors = self.a * np.array([[1, 0], [0.5, np.sqrt(3)/2]])  # They are separated by 60°
        
        # Generate k-space
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters (t) for different neighbor shells
        self.t0 = parameters.get('t0', 1.0)  # On-site energy
        self.t1 = parameters.get('t1', 0.2)  # Nearest-neighbor hopping amplitude
        
        # Interaction parameters (U) for different neighbor shells
        self.U0 = parameters.get('U0', 1.0)  # On-site (Hubbard) interaction
        self.U1 = parameters.get('U1', 0.5)  # Nearest-neighbor interaction
    
    def get_nearest_neighbor_vectors(self):
        """
        Returns the integer coordinate offsets (n1, n2) corresponding to the 
        nearest neighbors in a 2D triangular Bravais lattice. These offsets are ONLY
        valid for a lattice whose two primitive vectors are separated by 60°.

        To obtain the real-space displacements for each neighbor, multiply these 
        integer pairs by the primitive vectors a1 and a2, i.e.:
            R_neighbor = n1 * a1 + n2 * a2

        For a 2D triangular lattice, there are six nearest neighbors, given by:
        """
        n_vectors = [
            (1, 0),    # Right
            (0, 1),    # Upper-right
            (-1, 1),   # Upper-left
            (-1, 0),   # Left
            (0, -1),   # Lower-left
            (1, -1)    # Lower-right
        ]
        return n_vectors
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian using the dispersion relation.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Get nearest neighbor vectors for the triangular lattice
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Compute dispersion relation for all k-points
        for k in range(self.N_k):
            k_point = self.k_space[k]
            energy = self.t0  # On-site energy
            
            # Add nearest-neighbor hopping contribution
            for n_vec in nn_vectors:
                # Convert to real space vector: R = n1*a1 + n2*a2
                R = n_vec[0] * self.primitive_vectors[0] + n_vec[1] * self.primitive_vectors[1]
                # Add contribution: t1 * exp(-i k·R)
                energy += self.t1 * np.exp(-1j * np.dot(k_point, R))
            
            # Set diagonal elements for both spin up and spin down
            # Note: Since E_s(k) could be spin-dependent in general, we keep this flexibility
            H_nonint[0, 0, k] = np.real(energy)  # Spin up
            H_nonint[1, 1, k] = np.real(energy)  # Spin down
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian including Hartree and Fock terms.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Calculate average densities for each flavor (spin)
        n_s = np.zeros(self.D[0])
        for s in range(self.D[0]):
            n_s[s] = np.mean(exp_val[s, s, :])
        
        # Hartree term: U(0) * n_s * c_s'^dagger * c_s'
        for s_prime in range(self.D[0]):
            for s in range(self.D[0]):
                # On-site interaction U(0)
                H_int[s_prime, s_prime, :] += (1.0 / self.N_k) * self.U0 * n_s[s]
        
        # Fock term: -U(k1-k2) * <c_s^dagger(k1) c_s'(k1)> * c_s'^dagger(k2) c_s(k2)
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                # Calculate the interaction for all possible k1-k2
                for k2 in range(self.N_k):
                    fock_term = 0.0
                    for k1 in range(self.N_k):
                        # Calculate k1 - k2
                        k_diff = self.k_space[k1] - self.k_space[k2]
                        
                        # Calculate U(k1 - k2) = U0 + U1 * sum_n exp(-i (k1-k2)·R_n)
                        U_k = self.U0  # On-site interaction
                        
                        # Add nearest-neighbor contribution
                        for n_vec in self.get_nearest_neighbor_vectors():
                            R = n_vec[0] * self.primitive_vectors[0] + n_vec[1] * self.primitive_vectors[1]
                            U_k += self.U1 * np.exp(-1j * np.dot(k_diff, R))
                        
                        # Add contribution to Fock term
                        fock_term += np.real(U_k) * exp_val[s, s_prime, k1]
                    
                    # Add Fock term contribution (note the negative sign)
                    H_int[s_prime, s, k2] -= (1.0 / self.N_k) * fock_term
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generate the total Hamiltonian by combining non-interacting and interacting parts.
        
        Args:
            exp_val (np.ndarray): Expectation values.
            return_flat (bool): Whether to return a flattened Hamiltonian. Default is True.
            
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
