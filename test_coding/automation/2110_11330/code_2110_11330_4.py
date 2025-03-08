import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a triangular lattice with spin.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float): Filling factor, typically between 0 and 1. Default is 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_0': 0.0, 't_1': 1.0, 'U_0': 1.0, 'U_1': 0.5, 'T': 0, 'a': 1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Lattice symmetry ('triangular')
        self.D = (2,)  # Number of spin flavors (up and down)
        self.basis_order = {'0': 'spin'}
        # Order for each flavor:
        # 0: spin up
        # 1: spin down

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.primitive_vectors = get_primitive_vectors_triangle(self.a)
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Hopping parameters
        self.t_0 = parameters.get('t_0', 0.0)  # On-site hopping (typically 0 for most models)
        self.t_1 = parameters.get('t_1', 1.0)  # Nearest-neighbor hopping

        # Interaction parameters
        self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction (U(0))
        self.U_1 = parameters.get('U_1', 0.5)  # Nearest-neighbor interaction (U(1))

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

    def compute_energy_dispersion(self) -> np.ndarray:
        """
        Computes the energy dispersion E_s(k) based on the hopping parameters.
        
        Returns:
            np.ndarray: Energy dispersion for each k-point with shape (N_k,).
        """
        # Get nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Compute energy dispersion for each k-point
        energy = np.zeros(self.N_k)
        
        # On-site term (typically 0)
        energy += self.t_0
        
        # Nearest-neighbor hopping
        for n in nn_vectors:
            # Convert integer offsets to real-space vectors
            n_vec = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            # Add contribution to energy
            energy += self.t_1 * np.cos(np.dot(self.k_space, n_vec))
        
        return energy

    def compute_interaction_U(self, k_diff: np.ndarray) -> float:
        """
        Computes the interaction term U(k-q) based on the interaction parameters.
        
        Args:
            k_diff (np.ndarray): Array of shape (2,) containing the difference between k-points.
            
        Returns:
            float: Interaction strength for the given k-difference.
        """
        # Get nearest neighbor vectors
        nn_vectors = self.get_nearest_neighbor_vectors()
        
        # Compute interaction term for the k-difference
        U = self.U_0  # On-site interaction
        
        # Nearest-neighbor interaction
        for n in nn_vectors:
            # Convert integer offsets to real-space vectors
            n_vec = n[0] * self.primitive_vectors[0] + n[1] * self.primitive_vectors[1]
            # Add contribution to interaction
            U += self.U_1 * np.cos(np.dot(k_diff, n_vec))
        
        return U

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute energy dispersion for each k-point
        energy = self.compute_energy_dispersion()
        
        # Fill diagonal elements for each spin
        for s in range(self.D[0]):
            H_nonint[s, s, :] = -energy  # Note the negative sign from the original Hamiltonian
        
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
        
        # Compute averaged expectation values for the Hartree term
        avg_density = np.zeros(self.D[0])
        for s in range(self.D[0]):
            avg_density[s] = np.mean(exp_val[s, s, :])
        
        # Hartree term: (1/N) * sum_{s,s',k,k'} U(0) * <c_{k,s}^dag c_{k,s}> * c_{k',s'}^dag c_{k',s'}
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                H_int[s_prime, s_prime, :] += self.U_0 * avg_density[s] / self.N_k
        
        # Fock term: -(1/N) * sum_{s,s',k,q} U(k-q) * <c_{k,s}^dag c_{k,s'}> * c_{q,s'}^dag c_{q,s}
        for q in range(self.N_k):  # This is the momentum index for the Hamiltonian element
            for k in range(self.N_k):  # This is the momentum in the expectation value
                # Compute interaction strength based on momentum difference
                k_diff = self.k_space[k] - self.k_space[q]
                U_kq = self.compute_interaction_U(k_diff)
                
                for s in range(self.D[0]):  # These are the spin indices
                    for s_prime in range(self.D[0]):
                        H_int[s, s_prime, q] -= U_kq * exp_val[s, s_prime, k] / self.N_k
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the Hamiltonian in flattened form. Default is True.
            
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
