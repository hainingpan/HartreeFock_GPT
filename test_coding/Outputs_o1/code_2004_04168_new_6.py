import numpy as np
from typing import Any
from HF import *


# Assume that generate_k_space is defined elsewhere; it should return an array of k-points.
# For example:
# def generate_k_space(lattice: str, N_shell: int, a: float) -> np.ndarray:
#     # ... implementation ...
#     return k_space

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a model defined on a 2D triangular lattice.
    
    The Hamiltonian consists of:
      - A kinetic term:
          E_s(k) = sum_n t_s(n) exp(-i k·n)
          H_kinetic = sum_{s,k} E_s(k) c†_s(k)c_s(k)
      - An interaction term split into Hartree and Fock parts:
          H_Hartree = (1/N) U(0) sum_{s,s'} [sum_{k1}⟨c†_s(k1)c_s(k1)⟩] c†_{s'}(k)c_{s'}(k)
          H_Fock   = -(1/N) sum_{s,s'} [sum_{k1} U(k1-k)⟨c†_s(k1)c_{s'}(k1)⟩] c†_{s'}(k)c_s(k)
    
    Model parameters that do not depend on exp_val (i.e. not updated in each iteration)
    are set via the 'parameters' dictionary.
    """
    def __init__(self, N_shell: int = 10, 
                 parameters: dict[str, Any] = None, 
                 filling_factor: float = 0.5):
        if parameters is None:
            parameters = {
                'a': 1.0,
                't_up': 1.0,    # Hopping amplitude for spin up electrons
                't_down': 1.0,  # Hopping amplitude for spin down electrons
                'U_on': 1.0,    # On-site interaction U(n=0)
                'U_nn': 0.5     # Nearest-neighbor interaction (for n ≠ 0)
            }
        self.lattice = 'triangular'
        self.D = (2,)  # Only one flavor (spin) with 2 components.
        self.basis_order = {'0': 'spin_up', '1': 'spin_down'}
        
        # Occupancy and temperature parameters.
        self.nu = filling_factor
        self.T = 0  # Temperature is set to zero.
        
        # Lattice constant and primitive vectors for the triangular lattice.
        self.a = parameters.get('a', 1.0)
        # Define the two primitive (Bravais) lattice vectors.
        # (These are chosen so that the two vectors are separated by 120°.)
        self.primitive_vectors = self.a * np.array([[0, 1],
                                                     [np.sqrt(3)/2, -1/2]])
        
        # Generate the k-space using a predefined function.
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters (non–exp_val dependent)
        self.t_up = parameters.get('t_up', 1.0)
        self.t_down = parameters.get('t_down', 1.0)
        self.U_on = parameters.get('U_on', 1.0)    # On-site interaction contribution.
        self.U_nn = parameters.get('U_nn', 0.5)      # Nearest-neighbor interaction.
        
        return

    def get_nearest_neighbor_vectors(self):
        """
        Returns the integer coordinate offsets (n1, n2) corresponding to the
        nearest neighbors in a 2D triangular Bravais lattice. To obtain the real-space
        displacement for each neighbor, multiply the integer pair by the primitive vectors.
        """
        n_vectors = [
            (1, 0),
            (0, 1),
            (1, 1),
            (-1, 0),
            (0, -1),
            (-1, -1)
        ]
        return n_vectors

    def compute_band_energy(self, k: np.ndarray, t: float) -> complex:
        """
        Computes the band energy for a given momentum k and hopping amplitude t.
        The energy is given by: E(k) = t * sum_{n in NN} exp(-i k·R_n)
        where R_n is the displacement corresponding to each nearest neighbor.
        """
        energy = 0.0 + 0.0j
        n_vectors = self.get_nearest_neighbor_vectors()
        for n in n_vectors:
            # Convert integer offset to a real-space vector: R = n1*a1 + n2*a2.
            R_n = n[0]*self.primitive_vectors[0] + n[1]*self.primitive_vectors[1]
            energy += np.exp(-1j * np.dot(k, R_n))
        return t * energy

    def compute_Uq(self, delta_k: np.ndarray) -> complex:
        """
        Computes the momentum–dependent interaction:
          U(q) = sum_n U(n) exp(-i q·R_n)
        Here, we assume:
          - U(n=0) = U_on  (on-site term)
          - U(n) = U_nn for each nearest neighbor vector.
        """
        # On-site contribution (n = 0).
        Uq = self.U_on  # exp(-i q·0) = 1.
        # Add contributions from all nearest neighbors.
        n_vectors = self.get_nearest_neighbor_vectors()
        for n in n_vectors:
            R_n = n[0]*self.primitive_vectors[0] + n[1]*self.primitive_vectors[1]
            Uq += self.U_nn * np.exp(-1j * np.dot(delta_k, R_n))
        return Uq

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non–interacting (kinetic) part of the Hamiltonian.
        For each spin s, the kinetic energy is given by:
          E_s(k) = sum_{n} t_s(n) exp(-i k·R_n)
        and enters only the diagonal elements.
        Returns:
            np.ndarray: Hamiltonian with shape (D, D, N_k)
        """
        H_nonint = np.zeros((np.prod(self.D), np.prod(self.D), self.N_k), dtype=complex)
        # Loop over k-points.
        for idx, k in enumerate(self.k_space):
            # Compute the band energies for spin up (s = 0) and spin down (s = 1).
            E_up = self.compute_band_energy(k, self.t_up)
            E_down = self.compute_band_energy(k, self.t_down)
            H_nonint[0, 0, idx] = E_up
            H_nonint[1, 1, idx] = E_down
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian, including both Hartree
        and Fock contributions.
        
        The Hartree term adds a momentum–independent diagonal shift:
            (1/N) U(0) * (⟨n_up⟩ + ⟨n_down⟩)
        where U(0) = U_on + U_nn*(number of NN).
        
        The Fock term contributes off–diagonal corrections:
            - (1/N) sum_{k1} U(k1 - k2) ⟨c†_s(k1)c_{s'}(k1)⟩
        
        Args:
            exp_val (np.ndarray): Expectation value array (flattened shape: (D_flat, D_flat, N_k)).
        
        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        # Expand exp_val to shape (2, 2, N_k)
        exp_val = self.expand(exp_val)
        H_int = np.zeros((np.prod(self.D), np.prod(self.D), self.N_k), dtype=complex)
        
        # --- Hartree Term ---
        # Compute the average densities (over k) for spin up and spin down.
        n_up = np.mean(exp_val[0, 0, :])
        n_down = np.mean(exp_val[1, 1, :])
        # U(0) is given by the full sum over all U(n) with n = 0 and over the nearest neighbors.
        neighbor_vectors = self.get_nearest_neighbor_vectors()
        U0_effective = self.U_on + self.U_nn * len(neighbor_vectors)
        # The Hartree contribution (same for both spins) adds to the diagonal.
        hartree_contrib = U0_effective * (n_up + n_down)
        H_int[0, 0, :] += hartree_contrib
        H_int[1, 1, :] += hartree_contrib
        
        # --- Fock Term ---
        # For each momentum point k2 (indexed by j), sum over k1 (indexed by i).
        # Precompute a U_matrix with U(k1 - k2) for all k1, k2.
        U_matrix = np.zeros((self.N_k, self.N_k), dtype=complex)
        for j in range(self.N_k):
            for i in range(self.N_k):
                delta_k = self.k_space[i] - self.k_space[j]
                U_matrix[j, i] = self.compute_Uq(delta_k)
        
        # Now add the Fock term contribution to all matrix elements.
        # Note: The Fock term couples the (s, s') elements via the expectation value.
        for s in range(2):
            for s_prime in range(2):
                for j in range(self.N_k):
                    # Sum over all k1 indices.
                    fock_sum = np.sum(U_matrix[j, :] * exp_val[s, s_prime, :])
                    H_int[s, s_prime, j] += - (1.0 / self.N_k) * fock_sum
                    
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.
        
        Args:
            exp_val (np.ndarray): Expectation value array (flattened: (D_flat, D_flat, N_k)).
            return_flat (bool, optional): If True, returns the Hamiltonian flattened 
                to shape (np.prod(D), np.prod(D), N_k). Defaults to True.
        
        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total

    def flatten(self, ham: np.ndarray) -> np.ndarray:
        """
        Flattens the Hamiltonian to shape (np.prod(D), np.prod(D), N_k).
        """
        return ham.reshape((np.prod(self.D), np.prod(self.D), self.N_k))

    def expand(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Expands the flattened expectation value to shape (D, D, N_k).
        """
        return exp_val.reshape(self.D + self.D + (self.N_k,))
