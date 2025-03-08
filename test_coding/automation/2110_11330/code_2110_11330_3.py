import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice with spin-dependent dispersion and interactions.
  
  Args:
    N_shell (int): Number of shells in the first Brillouin zone.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float): Filling factor, defaults to 0.5.
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={'t': 1.0, 'U0': 1.0, 'U1': 0.5, 'T': 0, 'a': 1.0}, filling_factor: float=0.5):
    self.lattice = 'triangular'  # Lattice symmetry
    self.D = (2,)  # Spin flavors (up, down)
    self.basis_order = {'0': 'spin'}
    # 0: spin up, 1: spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0.0)  # temperature, default to 0
    self.a = parameters.get('a', 1.0)  # Lattice constant
    self.primitive_vectors = get_primitive_vectors_triangle(self.a)
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

    # Model parameters
    self.t = parameters.get('t', 1.0)  # Hopping parameter (energy scale)
    self.U0 = parameters.get('U0', 1.0)  # On-site interaction
    self.U1 = parameters.get('U1', 0.5)  # Nearest neighbor interaction
    
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
    Computes the energy dispersion E_s(k) for a given momentum k.
    
    Args:
      k (np.ndarray): Momentum vector.
    
    Returns:
      complex: The energy dispersion at momentum k.
    """
    E_k = 0
    nn_vectors = self.get_nearest_neighbor_vectors()
    
    for n1, n2 in nn_vectors:
        # Real-space displacement
        nn_real = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
        # Calculate dot product k·n
        k_dot_n = np.dot(k, nn_real)
        # Add to energy (assuming t_s is spin-independent)
        E_k += self.t * np.exp(-1j * k_dot_n)
    
    return E_k

  def compute_U_k_minus_q(self, k, q):
    """
    Computes the interaction U(k-q) in momentum space.
    
    Args:
      k (np.ndarray): Momentum k.
      q (np.ndarray): Momentum q.
    
    Returns:
      complex: The interaction U(k-q).
    """
    # On-site interaction (U0) is independent of k-q
    U_k_minus_q = self.U0
    
    # Get nearest neighbors
    nn_vectors = self.get_nearest_neighbor_vectors()
    
    # Add contribution from nearest neighbors
    for n1, n2 in nn_vectors:
        # Real-space displacement
        nn_real = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
        # Calculate dot product (k-q)·n
        k_minus_q_dot_n = np.dot(k - q, nn_real)
        # Add to interaction strength with phase factor
        U_k_minus_q += self.U1 * np.exp(-1j * k_minus_q_dot_n)
    
    return U_k_minus_q

  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.
    
    Returns:
      np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    H_nonint = np.zeros((*self.D,*self.D,self.N_k), dtype=complex)
    
    # Calculate the energy dispersion E_s(k) for each k
    for k_idx, k in enumerate(self.k_space):
        E_k = self.compute_energy_dispersion(k)
        
        # Assign to both spin up and spin down (assuming t_up = t_down)
        H_nonint[0, 0, k_idx] = -E_k  # Spin up diagonal element
        H_nonint[1, 1, k_idx] = -E_k  # Spin down diagonal element
    
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
    H_int = np.zeros((*self.D,*self.D,self.N_k), dtype=complex)
    
    # Calculate average densities for the Hartree term
    n_up = np.mean(exp_val[0, 0, :])    # Average spin up density
    n_down = np.mean(exp_val[1, 1, :])  # Average spin down density
    
    # Hartree term: U(0) * n_down for spin up, U(0) * n_up for spin down
    # H_Hartree = (1/N) * sum_{s,s'} sum_{k,k'} U(0) <c_k,s^† c_k,s> c_k',s'^† c_k',s'
    H_int[0, 0, :] += self.U0 * n_down  # Spin up interacting with average spin down
    H_int[1, 1, :] += self.U0 * n_up    # Spin down interacting with average spin up
    
    # Calculate the Fock term for each q-point
    # H_Fock = -(1/N) * sum_{s,s'} sum_{k,q} U(k-q) <c_k,s^† c_k,s'> c_q,s'^† c_q,s
    for q_idx, q in enumerate(self.k_space):
        # For each pair of spins
        for s in range(self.D[0]):
            for s_prime in range(self.D[0]):
                # Sum over all k
                fock_sum = 0
                for k_idx, k in enumerate(self.k_space):
                    U_k_minus_q = self.compute_U_k_minus_q(k, q)
                    fock_sum += U_k_minus_q * exp_val[s, s_prime, k_idx]
                
                # Add Fock term contribution to the Hamiltonian
                H_int[s_prime, s, q_idx] -= (1.0 / self.N_k) * fock_sum
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """
    Generates the total Hartree-Fock Hamiltonian.
    
    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
      return_flat (bool): Whether to return the flattened version of the Hamiltonian.
    
    Returns:
      np.ndarray: The total Hamiltonian, either flattened or not.
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if return_flat:
      return flattened(H_total, self.D, self.N_k)
    else:
      return H_total
