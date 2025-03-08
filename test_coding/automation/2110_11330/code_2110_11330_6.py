import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hamiltonian for a triangular lattice with spin-1/2 particles.
    
    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary of model parameters.
        filling_factor (float): Filling factor of the system (default 0.5).
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t': 1.0, 'U': 1.0, 'U_1': 0.5, 'T': 0.0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Triangular lattice
        self.D = (2,)  # Spin up and spin down
        self.basis_order = {'0': 'spin'}
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t = parameters.get('t', 1.0)  # Hopping parameter
        self.U = parameters.get('U', 1.0)  # On-site interaction
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

    def compute_energy_dispersion(self, k):
        """
        Compute the energy dispersion E_s(k) for the triangular lattice.
        
        Args:
            k (np.ndarray): 2D wavevector.
            
        Returns:
            float: The energy dispersion.
        """
        E_k = 0.0
        n_vectors = self.get_nearest_neighbor_vectors()
        for n1, n2 in n_vectors:
            # Calculate real-space displacement vector r = n1 * a1 + n2 * a2
            r = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
            
            # Dot product of k with r
            k_dot_r = np.dot(k, r)
            
            # Add contribution to energy dispersion
            E_k += self.t * np.exp(-1j * k_dot_r)
        
        return E_k

    def compute_U_k(self, k):
        """
        Compute the Fourier transform of the real-space interaction potential.
        
        Args:
            k (np.ndarray): 2D wavevector.
            
        Returns:
            float: The Fourier transform of the interaction potential.
        """
        # On-site interaction
        U_k = self.U
        
        # Add contribution from nearest neighbors
        nn_vectors = self.get_nearest_neighbor_vectors()
        for n1, n2 in nn_vectors:
            # Calculate real-space displacement vector
            r = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
            
            # Dot product of k with r
            k_dot_r = np.dot(k, r)
            
            # Add contribution to interaction potential
            U_k += self.U_1 * np.exp(-1j * k_dot_r)
        
        return U_k

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate energy dispersion for each k-point
        for k_idx in range(self.N_k):
            k = self.k_space[k_idx]
            E_k = self.compute_energy_dispersion(k)
            
            # Assign energy to non-interacting Hamiltonian for each spin
            # -E_s(k) c_k,s^† c_k,s term
            for s in range(self.D[0]):
                H_nonint[s, s, k_idx] = -E_k
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian (Hartree and Fock terms).
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Hartree term: (1/N) * sum_{s,s'} * sum_{k,k'} * U(0) * <c_{k,s}^† c_{k,s}> * c_{k',s'}^† c_{k',s'}
        for s in range(self.D[0]):
            # Calculate average density for spin s
            n_s = np.mean(exp_val[s, s, :])
            
            for sp in range(self.D[0]):
                for k_idx in range(self.N_k):
                    # Add Hartree term to H_int[s', s', k']
                    H_int[sp, sp, k_idx] += self.U * n_s / self.N_k
        
        # Fock term: -(1/N) * sum_{s,s'} * sum_{k,q} * U(k-q) * <c_{k,s}^† c_{k,s'}> * c_{q,s'}^† c_{q,s}
        for s in range(self.D[0]):
            for sp in range(self.D[0]):
                for k_idx in range(self.N_k):
                    for q_idx in range(self.N_k):
                        # Get k and q vectors
                        k = self.k_space[k_idx]
                        q = self.k_space[q_idx]
                        
                        # Calculate k-q
                        k_minus_q = k - q
                        
                        # Compute U(k-q)
                        U_k_minus_q = self.compute_U_k(k_minus_q)
                        
                        # Add Fock term to H_int[s', s, q]
                        H_int[sp, s, q_idx] -= U_k_minus_q * exp_val[s, sp, k_idx] / self.N_k
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian (default True).
            
        Returns:
            np.ndarray: The total Hamiltonian, flattened if return_flat is True.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
