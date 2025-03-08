import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a spin-1/2 system on a triangular lattice with
  hopping and interaction terms.
  
  Args:
    N_shell (int): Number of shells in the first Brillouin zone.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float): Filling factor of the system. Defaults to 0.5.
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={'t1': 6.0, 't2': 1.0, 'U0': 1.0, 'U1': 0.5, 'T': 0.0, 'a': 1.0}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (2,)  # Two spin flavors
    self.basis_order = {'0': 'spin'}
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
    self.t1 = parameters.get('t1', 6.0)  # Nearest-neighbor hopping (in meV)
    self.t2 = parameters.get('t2', 1.0)  # Next-nearest-neighbor hopping (in meV)
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
    
  def get_next_nearest_neighbor_vectors(self):
    """
    Returns the integer coordinate offsets (n1, n2) corresponding to the 
    next-nearest neighbors in a 2D triangular Bravais lattice.
    """
    n_vectors = [
        (2, 0),
        (0, 2),
        (2, 2),
        (-2, 0),
        (0, -2),
        (-2, -2),
        (1, -1),
        (-1, 1),
        (-1, -2),
        (-2, -1),
        (1, 2),
        (2, 1)
    ]
    return n_vectors
  
  def compute_Es(self, k):
    """
    Compute the dispersion relation E_s(k) = sum_n t_s(n) exp(-i k . n)
    
    Args:
        k (numpy.ndarray): Array of k-points with shape (N_k, 2).
        
    Returns:
        numpy.ndarray: E_s(k) for all k-points with shape (N_k,).
    """
    E_s = np.zeros(k.shape[0], dtype=complex)
    
    # Get the nearest neighbor vectors
    nn_vectors = self.get_nearest_neighbor_vectors()
    nnn_vectors = self.get_next_nearest_neighbor_vectors()
    
    # Compute the contribution from nearest neighbors (t1 term)
    for n1, n2 in nn_vectors:
        r = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
        E_s += self.t1 * np.exp(-1j * (k[:, 0] * r[0] + k[:, 1] * r[1]))
    
    # Compute the contribution from next-nearest neighbors (t2 term)
    for n1, n2 in nnn_vectors:
        r = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
        E_s += self.t2 * np.exp(-1j * (k[:, 0] * r[0] + k[:, 1] * r[1]))
    
    return E_s.real  # The result should be real
  
  def compute_U(self, k):
    """
    Compute the interaction potential U(k) = U_0 + U_1 sum_delta exp(-i k . delta),
    where delta runs over all nearest-neighbor vectors.
    
    Args:
        k (numpy.ndarray): Array of k-vectors with shape (N, 2).
        
    Returns:
        numpy.ndarray: U(k) for all k-vectors with shape (N,).
    """
    # On-site interaction is constant
    U_k = np.ones(k.shape[0], dtype=complex) * self.U0
    
    # Get the nearest neighbor vectors
    nn_vectors = self.get_nearest_neighbor_vectors()
    
    # Compute the contribution from nearest neighbors
    for n1, n2 in nn_vectors:
        r = n1 * self.primitive_vectors[0] + n2 * self.primitive_vectors[1]
        U_k += self.U1 * np.exp(-1j * (k[:, 0] * r[0] + k[:, 1] * r[1]))
    
    return U_k.real  # The result should be real

  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.
    
    Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Compute the dispersion relation for all k-points
    E_s = self.compute_Es(self.k_space)
    
    # The kinetic energy is the same for both spin up and spin down
    H_nonint[0, 0, :] = E_s  # Spin up
    H_nonint[1, 1, :] = E_s  # Spin down
    
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
    
    # Hartree term calculations
    n_up = np.mean(exp_val[0, 0, :])    # Mean occupation of spin up
    n_down = np.mean(exp_val[1, 1, :])  # Mean occupation of spin down
    
    # Hartree contribution to diagonal elements
    H_int[0, 0, :] += self.U0 * n_down  # Spin up interacting with mean spin down
    H_int[1, 1, :] += self.U0 * n_up    # Spin down interacting with mean spin up
    
    # For the Fock term, we need to compute U(k1-k2) for all pairs of k1 and k2
    for k1_idx in range(self.N_k):
        k1 = self.k_space[k1_idx]
        for k2_idx in range(self.N_k):
            k2 = self.k_space[k2_idx]
            k_diff = k1 - k2
            U_k_diff = self.compute_U(k_diff.reshape(1, 2))[0]
            
            # Fock contribution to all matrix elements
            H_int[0, 0, k2_idx] -= (1.0 / self.N_k) * U_k_diff * exp_val[0, 0, k1_idx]
            H_int[1, 1, k2_idx] -= (1.0 / self.N_k) * U_k_diff * exp_val[1, 1, k1_idx]
            H_int[0, 1, k2_idx] -= (1.0 / self.N_k) * U_k_diff * exp_val[0, 1, k1_idx]
            H_int[1, 0, k2_idx] -= (1.0 / self.N_k) * U_k_diff * exp_val[1, 0, k1_idx]
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """
    Generates the total Hartree-Fock Hamiltonian.
    
    Args:
        exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
        return_flat (bool, optional): If True, returns the flattened Hamiltonian. Defaults to True.
        
    Returns:
        np.ndarray: The total Hamiltonian. If return_flat is True, the shape is (D_flattened, D_flattened, N_k),
                    otherwise it's (D, D, N_k).
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    
    if return_flat:
        return flattened(H_total, self.D, self.N_k)
    else:
        return H_total
