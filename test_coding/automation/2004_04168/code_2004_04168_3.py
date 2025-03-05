import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with hopping and interaction terms.
    
    Args:
        N_shell (int): Number of shells for the k-space grid.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=10, parameters: dict={'t_0': 1.0, 't_1': 0.1, 'U_0': 1.0, 'U_1': 0.5}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # LM Task: defines tuple - spin up and spin down
        self.basis_order = {'0': 'spin'}
        # Basis order: 0: spin up, 1: spin down

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0.0  # Temperature set to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = self.a * np.array([[0,1],[np.sqrt(3)/2,-1/2]])  # Define the primitive lattice vectors for a 2D triangular lattice
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Hopping parameters (t_s(n))
        self.t_0 = parameters.get('t_0', 1.0)  # On-site hopping
        self.t_1 = parameters.get('t_1', 0.1)  # Nearest-neighbor hopping
        
        # Interaction parameters (U(n))
        self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction strength
        self.U_1 = parameters.get('U_1', 0.5)  # Nearest-neighbor interaction strength
        
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
            (-1, 1),
            (-1, 0),
            (0, -1),
            (1, -1)
        ]
        return n_vectors

    def compute_E_s(self, k, s):
        """
        Computes the energy dispersion E_s(k) for a given k and spin s.

        Args:
            k (np.ndarray): Momentum vector.
            s (int): Spin index (0 for up, 1 for down).

        Returns:
            float: Energy dispersion value.
        """
        # For simplicity, assume t_s(n) is independent of spin
        n_vectors = self.get_nearest_neighbor_vectors()
        E = self.t_0  # On-site hopping
        for n in n_vectors:
            # Compute dot product k.n
            k_dot_n = k[0] * n[0] + k[1] * n[1]
            E += self.t_1 * np.exp(-1j * k_dot_n)
        return E.real  # Assuming real values for simplicity

    def compute_U_k(self, k):
        """
        Computes the interaction potential U(k) for a given momentum k.

        Args:
            k (np.ndarray): Momentum vector.

        Returns:
            float: Interaction potential value.
        """
        n_vectors = self.get_nearest_neighbor_vectors()
        U = self.U_0  # On-site interaction
        for n in n_vectors:
            # Compute dot product k.n
            k_dot_n = k[0] * n[0] + k[1] * n[1]
            U += self.U_1 * np.exp(-1j * k_dot_n)
        return U.real  # Assuming real values for simplicity

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Compute the dispersion relation for each k and spin
        for s in range(self.D[0]):
            for i, k in enumerate(self.k_space):
                # Kinetic term: E_s(k) c^dagger_s(k) c_s(k)
                H_nonint[s, s, i] = self.compute_E_s(k, s)
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten_exp_val(exp_val, self.D, self.N_k)  # Reshape to (D, D, N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)
        
        # Hartree term: U(0) <c_s^dagger(k1) c_s(k1)> c_s'^dagger(k2) c_s'(k2)
        for s_prime in range(self.D[0]):
            for k2 in range(self.N_k):
                # For each s_prime, sum over all spins s the average density
                for s in range(self.D[0]):
                    average_density_s = np.mean(exp_val[s, s, :])  # Average density for spin s
                    H_int[s_prime, s_prime, k2] += self.U_0 * average_density_s
        
        # Fock term: -U(k1-k2) <c_s^dagger(k1) c_s'(k1)> c_s'^dagger(k2) c_s(k2)
        for s_prime in range(self.D[0]):
            for s in range(self.D[0]):
                for k2 in range(self.N_k):
                    # For each s', s and k2, sum over all k1
                    fock_term = 0
                    for k1 in range(self.N_k):
                        k_diff = self.k_space[k1] - self.k_space[k2]
                        U_k_diff = self.compute_U_k(k_diff)
                        fock_term += U_k_diff * exp_val[s, s_prime, k1]
                    H_int[s_prime, s, k2] -= fock_term / self.N_k
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).
            return_flat (bool, optional): Whether to return a flattened Hamiltonian. Defaults to True.

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
