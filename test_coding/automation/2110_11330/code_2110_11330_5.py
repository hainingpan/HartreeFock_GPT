import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with spin-1/2 particles on a triangular lattice.
    
    Args:
        N_shell (int): Number of shells in the Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_up': 1.0, 't_down': 1.0, 'U_0': 1.0, 'T': 0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Lattice symmetry
        self.D = (2,)  # Number of flavors (spin up and down)
        self.basis_order = {'0': 'spin'}
        # Order for each flavor:
        # 0: spin_up, 1: spin_down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_up = parameters.get('t_up', 1.0)  # Hopping parameter for spin up
        self.t_down = parameters.get('t_down', 1.0)  # Hopping parameter for spin down
        self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction strength
        
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
        
        # Get nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Calculate energy dispersion for each k-point
        for k_idx, k in enumerate(self.k_space):
            # Energy dispersion for spin up
            E_up = 0
            # Energy dispersion for spin down
            E_down = 0
            
            # Sum over nearest neighbors
            for n1, n2 in nn_vectors:
                # Calculate real space vector
                R = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
                # Calculate phase factor
                phase = np.exp(-1j * np.dot(k, R))
                # Add contribution to energy dispersion
                E_up += self.t_up * phase
                E_down += self.t_down * phase
            
            # Set kinetic energy terms
            H_nonint[0, 0, k_idx] = -E_up  # Kinetic energy for spin up
            H_nonint[1, 1, k_idx] = -E_down  # Kinetic energy for spin down
        
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
        
        # Hartree term: U(0) * <c_{k,s}^dagger c_{k,s}> * c_{k',s'}^dagger c_{k',s'}
        # Calculate average density for each spin
        n_up = np.mean(exp_val[0, 0, :])  # Average density for spin up
        n_down = np.mean(exp_val[1, 1, :])  # Average density for spin down
        total_density = n_up + n_down
        
        # Add Hartree contributions (diagonal elements)
        H_int[0, 0, :] += self.U_0 * total_density / self.N_k  # Hartree term for spin up
        H_int[1, 1, :] += self.U_0 * total_density / self.N_k  # Hartree term for spin down
        
        # Fock term: -U(k-q) * <c_{k,s}^dagger c_{k,s'}> * c_{q,s'}^dagger c_{q,s}
        # For simplicity, assuming U(k-q) = U_0 for all k, q pairs
        
        # Calculate average exchange correlations
        exchange_up_down = np.mean(exp_val[0, 1, :])  # <c_{k,up}^dagger c_{k,down}>
        exchange_down_up = np.mean(exp_val[1, 0, :])  # <c_{k,down}^dagger c_{k,up}>
        
        # Add Fock contributions (off-diagonal elements)
        H_int[1, 0, :] -= self.U_0 * exchange_up_down / self.N_k  # Fock term for spin down->up
        H_int[0, 1, :] -= self.U_0 * exchange_down_up / self.N_k  # Fock term for spin up->down
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
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
