import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """Implements a Hartree-Fock Hamiltonian for a system with spin-dependent 
    hopping and interaction terms on a triangular lattice.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_0': 1.0, 't_1': 0.5, 'U_0': 1.0, 'T': 0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (2,)  # Two spin flavors
        self.basis_order = {'0': 'spin'}
        # Order for spin flavor: 0=up, 1=down

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters for hopping
        self.t_0 = parameters.get('t_0', 1.0)  # Onsite hopping parameter
        self.t_1 = parameters.get('t_1', 0.5)  # Nearest-neighbor hopping parameter
        
        # Model parameters for interaction
        self.U_0 = parameters.get('U_0', 1.0)  # Interaction strength parameter
        
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

    def calculate_energy_dispersion(self, k_vec, s):
        """Calculates the energy dispersion E_s(k) for a given momentum and spin.
        
        E_{s}(k) = sum_{n} t_{s}(n) * exp(-i * k · n)
        
        Args:
            k_vec (np.ndarray): Momentum vector.
            s (int): Spin index.
            
        Returns:
            complex: Energy dispersion at the given momentum and spin.
        """
        E_k = self.t_0  # Onsite term
        
        # Nearest-neighbor hopping terms
        for n1, n2 in self.get_nearest_neighbor_vectors():
            n_vec = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
            k_dot_n = np.dot(k_vec, n_vec)
            E_k += self.t_1 * np.exp(-1j * k_dot_n)
        
        return E_k
    
    def calculate_U_k_minus_q(self, k_vec, q_vec):
        """Calculates the interaction potential U(k-q) between momenta k and q.
        
        U(k-q) = sum_{n} U(n) * exp(-i * (k-q) · n)
        
        Args:
            k_vec (np.ndarray): Momentum k.
            q_vec (np.ndarray): Momentum q.
            
        Returns:
            complex: The interaction potential U(k-q).
        """
        k_minus_q = k_vec - q_vec
        U_k_minus_q = self.U_0  # Constant term
        
        # Sum over all the relevant neighbor vectors
        for n1, n2 in self.get_nearest_neighbor_vectors():
            n_vec = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
            k_minus_q_dot_n = np.dot(k_minus_q, n_vec)
            U_k_minus_q += self.U_0 * np.exp(-1j * k_minus_q_dot_n)
        
        return U_k_minus_q

    def generate_non_interacting(self) -> np.ndarray:
        """Generates the non-interacting part of the Hamiltonian.
        
        H_0 = -sum_{s}sum_{k} E_{s}(k) c_{k,s}^† c_{k,s}
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate energy dispersion for each spin and k-point
        for s in range(self.D[0]):
            for k in range(self.N_k):
                k_vec = self.k_space[k]
                # Negative sign from the Hamiltonian definition
                H_nonint[s, s, k] = -self.calculate_energy_dispersion(k_vec, s)
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generates the interacting part of the Hamiltonian.
        
        H_Hartree = (1/N)sum_{s,s'}sum_{k,k'} U(0) <c_{k,s}^† c_{k,s}> c_{k',s'}^† c_{k',s'}
        H_Fock = -(1/N)sum_{s,s'}sum_{k,q} U(k-q) <c_{k,s}^† c_{k,s'}> c_{q,s'}^† c_{q,s}
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Hartree term: U(0) * <c_{k,s}^† c_{k,s}> * c_{k',s'}^† c_{k',s'}
        for s in range(self.D[0]):
            n_s = np.mean(exp_val[s, s, :])  # Average occupation of spin s
            for s_prime in range(self.D[0]):
                # Add Hartree contribution to diagonal elements
                H_int[s_prime, s_prime, :] += self.U_0 * n_s / self.N_k
        
        # Fock term: -U(k-q) * <c_{k,s}^† c_{k,s'}> * c_{q,s'}^† c_{q,s}
        for q in range(self.N_k):
            q_vec = self.k_space[q]
            for s in range(self.D[0]):
                for s_prime in range(self.D[0]):
                    fock_sum = 0.0
                    for k in range(self.N_k):
                        k_vec = self.k_space[k]
                        U_k_minus_q = self.calculate_U_k_minus_q(k_vec, q_vec)
                        fock_sum += U_k_minus_q * exp_val[s, s_prime, k]
                    
                    # Add Fock contribution (note the negative sign)
                    H_int[s_prime, s, q] -= fock_sum / self.N_k
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """Generates the total Hartree-Fock Hamiltonian.
        
        H_total = H_non_interacting + H_interacting
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian. Default is True.
            
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
