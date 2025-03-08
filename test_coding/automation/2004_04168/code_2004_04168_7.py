import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with two spin states on a triangular lattice.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor of the system. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t1': 6.0, 't2': 1.0, 'U0': 1.0, 'U1': 0.5}, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Lattice symmetry
        self.D = (2,)  # Number of flavors (spin up, spin down)
        self.basis_order = {'0': 'spin'}
        # Order for each flavor:
        # 0: spin up
        # 1: spin down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0.0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t1 = parameters.get('t1', 6.0)  # Nearest-neighbor hopping (6 meV)
        self.t2 = parameters.get('t2', 1.0)  # Next-nearest-neighbor hopping (1 meV)
        self.U0 = parameters.get('U0', 1.0)  # On-site interaction
        self.U1 = parameters.get('U1', 0.5)  # Nearest-neighbor interaction
        
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
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Get the nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Compute the dispersion relation for each k point
        for i in range(self.N_k):
            k = self.k_space[i]
            
            # Nearest-neighbor contribution
            E_nn = 0
            for n1, n2 in nn_vectors:
                k_dot_n = k[0] * n1 + k[1] * n2
                E_nn += np.exp(-1j * k_dot_n)
            
            # Next-nearest-neighbor contribution
            E_nnn = 0
            for n1, n2 in [(2, 0), (0, 2), (2, 1), (1, 2), (-1, 2), (-2, 1), 
                          (-2, 0), (0, -2), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                k_dot_n = k[0] * n1 + k[1] * n2
                E_nnn += np.exp(-1j * k_dot_n)
            
            # Total dispersion: E_s(k) = sum_n t_s(n) * e^(-i k·n)
            E_k = self.t1 * E_nn + self.t2 * E_nnn
            
            # The dispersion is the same for both spin states
            H_nonint[0, 0, i] = np.real(E_k)  # Use real part for numerical stability
            H_nonint[1, 1, i] = np.real(E_k)
        
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
        
        # Compute the average densities for each spin
        n_up = np.mean(exp_val[0, 0, :])
        n_down = np.mean(exp_val[1, 1, :])
        
        # Hartree term: U0 * n_s * c_s'^dag(k2) c_s'(k2)
        # This term contributes to H_int[s', s', k2]
        H_int[0, 0, :] += self.U0 * n_down  # Interaction of spin up with average spin down density
        H_int[1, 1, :] += self.U0 * n_up    # Interaction of spin down with average spin up density
        
        # Fock term: -U(k1-k2) * <c_s^dag(k1) c_s'(k1)> * c_s'^dag(k2) c_s(k2)
        # This term contributes to H_int[s', s, k2]
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        for k2 in range(self.N_k):
            for k1 in range(self.N_k):
                # Compute the momentum difference
                dk = self.k_space[k1] - self.k_space[k2]
                
                # Compute interaction strength U(k1-k2) = U0 + U1 * sum_n e^(-i(k1-k2)·n)
                U_k1_k2 = self.U0  # On-site interaction
                for n1, n2 in nn_vectors:
                    k_dot_n = dk[0] * n1 + dk[1] * n2
                    U_k1_k2 += self.U1 * np.exp(-1j * k_dot_n)
                
                # Use the real part for numerical stability
                U_k1_k2 = np.real(U_k1_k2)
                
                # Add the Fock term for all spin combinations
                for s in range(self.D[0]):
                    for s_prime in range(self.D[0]):
                        H_int[s_prime, s, k2] -= (1.0 / self.N_k) * U_k1_k2 * exp_val[s, s_prime, k1]
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the Hamiltonian in a flattened form.
            
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
