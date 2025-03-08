import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Implementation of the Hartree-Fock Hamiltonian for a triangular lattice with spin-dependent interactions.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters like hopping (t) and interaction strengths (U0, U1).
        filling_factor (float): The filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Triangular lattice as specified
        self.D = (2,)  # Spin up and down
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
        self.t = parameters.get('t', 1.0)  # Hopping parameter
        self.U0 = parameters.get('U0', 1.0)  # On-site interaction
        self.U1 = parameters.get('U1', 0.5)  # Nearest neighbor interaction
        # Add more interactions for different shells as needed
        
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
    
    def calculate_U_k(self, k):
        """
        Calculate the Fourier transform of the interaction potential U(n) at momentum k.
        
        Args:
            k (np.ndarray): Momentum vector.
            
        Returns:
            complex: The Fourier transform of U(n) at momentum k.
        """
        # On-site interaction (independent of k)
        U_k = self.U0
        
        # Add nearest neighbor interaction
        n_vectors = self.get_nearest_neighbor_vectors()
        for n in n_vectors:
            real_space_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            k_dot_n = np.dot(k, real_space_n)
            U_k += self.U1 * np.exp(-1j * k_dot_n)
        
        # Add more shells as needed
        
        return U_k
        
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Energy dispersion for each spin and k point
        n_vectors = self.get_nearest_neighbor_vectors()
        
        for s in range(self.D[0]):  # For each spin
            for k_idx in range(self.N_k):
                k = self.k_space[k_idx]
                # Compute E_s(k) = sum_n t_s(n) * exp(-i k.n)
                E_k = 0
                for n in n_vectors:
                    real_space_n = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
                    k_dot_n = np.dot(k, real_space_n)
                    E_k += self.t * np.exp(-1j * k_dot_n)
                
                # Assign to the corresponding matrix element - negative sign from Hamiltonian definition
                H_nonint[s, s, k_idx] = -E_k
        
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
        
        # Calculate mean densities for use in Hartree term
        n_up = np.mean(exp_val[0, 0, :])  # Mean density of spin up
        n_down = np.mean(exp_val[1, 1, :])  # Mean density of spin down
        
        # Hartree term: U(0) * mean density interacting with each spin
        H_int[0, 0, :] += (1.0 / self.N_k) * self.U0 * n_down  # Spin up interacting with mean spin down
        H_int[1, 1, :] += (1.0 / self.N_k) * self.U0 * n_up  # Spin down interacting with mean spin up
        
        # Fock term: -(1/N) sum_{s,s',k,q} U(k-q) <c_{k,s}^\dagger c_{k,s'}> c_{q,s'}^\dagger c_{q,s}
        for s in range(self.D[0]):  # s = 0,1
            for sp in range(self.D[0]):  # s' = 0,1
                for k_idx in range(self.N_k):
                    for q_idx in range(self.N_k):
                        # Compute U(k-q)
                        k = self.k_space[k_idx]
                        q = self.k_space[q_idx]
                        k_minus_q = k - q
                        U_k_minus_q = self.calculate_U_k(k_minus_q)
                        
                        # Update the Fock term for the sp,s element at momentum q
                        H_int[sp, s, q_idx] -= (1.0 / self.N_k) * U_k_minus_q * exp_val[s, sp, k_idx]
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): If True, returns the flattened Hamiltonian, else returns the original shape.

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
