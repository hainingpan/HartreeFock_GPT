import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Implements a Hartree-Fock Hamiltonian for a system on a triangular lattice with
    hopping and interaction terms.
    
    The Hamiltonian includes:
    - Non-interacting term: -sum_s sum_k E_s(k) c_k,s^dagger c_k,s
    - Hartree term: (1/N) * sum_s,s' sum_k,k' U(0) <c_k,s^dagger c_k,s> c_k',s'^dagger c_k',s'
    - Fock term: -(1/N) * sum_s,s' sum_k,q U(k-q) <c_k,s^dagger c_k,s'> c_q,s'^dagger c_q,s
    
    Args:
      N_shell: Number of shells in the first Brillouin zone.
      parameters: Dictionary containing model parameters.
      filling_factor: Filling factor for the system. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t': 1.0, 'U': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # 2 spin states
        self.basis_order = {'0': 'spin'}  # up, down

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters
        self.t = parameters.get('t', 1.0)  # Hopping parameter
        self.U = parameters.get('U', 1.0)  # Interaction strength

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
        Generates the non-interacting part of the Hamiltonian based on the energy dispersion.
        
        E_s(k) = sum_{n} t_s(n) * exp(-i k·n) where n runs over all hopping pairs.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate energy dispersion using nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        for s in range(self.D[0]):  # Loop through spins
            for k_idx in range(self.N_k):
                k = self.k_space[k_idx]
                E_k = 0.0
                
                # Sum over all nearest neighbors
                for n1, n2 in nn_vectors:
                    # Convert to real space vector using primitive vectors
                    n_vector = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
                    # Add hopping contribution to energy
                    E_k += self.t * np.exp(-1j * np.dot(k, n_vector))
                
                # Set the diagonal element (kinetic term)
                H_nonint[s, s, k_idx] = -E_k
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian (Hartree and Fock terms).
        
        Args:
            exp_val: Expectation value array with shape (D_flattened, D_flattened, N_k).
        
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Hartree term: (1/N) * sum_{s,s'} sum_{k,k'} U(0) * <c_k,s^dagger c_k,s> * c_k',s'^dagger c_k',s'
        for s in range(self.D[0]):
            # Calculate average density for spin s
            n_s = np.mean(exp_val[s, s, :])
            
            # Apply Hartree term to all spin s' diagonal elements
            for s_prime in range(self.D[0]):
                H_int[s_prime, s_prime, :] += (1.0 / self.N_k) * self.U * n_s
        
        # Fock term: -(1/N) * sum_{s,s'} sum_{k,q} U(k-q) * <c_k,s^dagger c_k,s'> * c_q,s'^dagger c_q,s
        # For simplicity, using a constant U(k-q) = U(0) = self.U
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                # Average spin-flip expectation value
                exchange_ss_prime = np.mean(exp_val[s, s_prime, :])
                
                # Apply Fock term (negative sign already included)
                H_int[s_prime, s, :] -= (1.0 / self.N_k) * self.U * exchange_ss_prime
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian by combining non-interacting and 
        interacting parts.
        
        Args:
            exp_val: Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat: Whether to return the Hamiltonian in flattened form.
        
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
