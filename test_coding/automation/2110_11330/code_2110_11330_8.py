import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice system with momentum-dependent interactions.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t':1.0, 'U0':1.0, 'T':0, 'a':1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Lattice symmetry
        self.D = (2,)  # Number of quantum states
        self.basis_order = {'0': 'quantum_index'}  # The 0th index represents quantum index (s)
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # Temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t = parameters.get('t', 1.0)  # Hopping parameter
        self.U0 = parameters.get('U0', 1.0)  # On-site interaction strength
        
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
        Generates the non-interacting part of the Hamiltonian:
        H = -∑_s∑_k E_s(k) c_k,s^† c_k,s
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute energy dispersion for triangular lattice
        # E_s(k) = ∑_n t_s(n) e^(-i k·n)
        for s in range(self.D[0]):
            E_k = np.zeros(self.N_k, dtype=complex)
            for n_vec in self.get_nearest_neighbor_vectors():
                # Compute k dot n for each nearest neighbor
                k_dot_n = self.k_space[:, 0] * n_vec[0] + self.k_space[:, 1] * n_vec[1]
                E_k += self.t * np.exp(-1j * k_dot_n)
            
            # Non-interacting term in Hamiltonian
            H_nonint[s, s, :] = -E_k
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian, which includes:
        1. Hartree term: (1/N)∑_s,s'∑_k,k' U(0) ⟨c_k,s^† c_k,s⟩ c_k',s'^† c_k',s'
        2. Fock term: -(1/N)∑_s,s'∑_k,q U(k-q) ⟨c_k,s^† c_k,s'⟩ c_q,s'^† c_q,s
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate average occupancies for Hartree term
        n_avg = np.zeros(self.D[0])
        for s in range(self.D[0]):
            n_avg[s] = np.mean(exp_val[s, s, :])
        
        # 1. Hartree term
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                # Add contribution from Hartree term: U(0)⟨c_k,s^† c_k,s⟩c_k',s'^† c_k',s'
                H_int[s_prime, s_prime, :] += self.U0 * n_avg[s] / self.N_k
        
        # 2. Fock term
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                for k in range(self.N_k):
                    for q in range(self.N_k):
                        # Calculate momentum difference k-q
                        k_vec = self.k_space[k]
                        q_vec = self.k_space[q]
                        diff = k_vec - q_vec
                        
                        # Calculate U(k-q) - in a real implementation this would be more sophisticated
                        # For simplicity, using a distance-dependent decay
                        U_k_q = self.U0 * np.exp(-np.linalg.norm(diff))
                        
                        # Add contribution from Fock term: -U(k-q)⟨c_k,s^† c_k,s'⟩c_q,s'^† c_q,s
                        H_int[s_prime, s, q] -= U_k_q * exp_val[s, s_prime, k] / self.N_k
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian by combining non-interacting and interacting parts.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian.
            
        Returns:
            np.ndarray: The total Hamiltonian, either in its original shape or flattened.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
