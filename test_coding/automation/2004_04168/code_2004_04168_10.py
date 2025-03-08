import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice model with spin degrees of freedom.
  
  Implements a model with:
  - Nearest-neighbor (t_1) and next-nearest-neighbor (t_2) hopping
  - On-site (U_0) and nearest-neighbor (U_1) interactions
  
  Args:
    N_shell (int): Number of shells in k-space
    parameters (dict): Dictionary containing model parameters
    filling_factor (float): Filling factor of the system (default: 0.5)
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_1': 6, 't_2': 1, 'U_0': 1.0, 'U_1': 0.1}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (2,)  # Spin up and spin down
    self.basis_order = {'0': 'spin'}
    # this is the basis order that the Hamiltonian will follow

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0.0)  # temperature, default to 0
    self.a = parameters.get('a', 1.0)  # Lattice constant
    self.primitive_vectors = get_primitive_vectors_triangle(self.a)
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

    # Hopping parameters
    self.t_1 = parameters.get('t_1', 6)  # Nearest-neighbor hopping (meV)
    self.t_2 = parameters.get('t_2', 1)  # Next-nearest-neighbor hopping (meV)
    
    # Interaction parameters
    self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction strength
    self.U_1 = parameters.get('U_1', 0.1)  # Nearest-neighbor interaction strength

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
    
    This includes the kinetic energy term with nearest-neighbor and 
    next-nearest-neighbor hopping on a triangular lattice.
    
    Returns:
        np.ndarray: Non-interacting Hamiltonian with shape (D, D, N_k)
    """
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Get nearest-neighbor vectors
    nn_vectors = self.get_nearest_neighbor_vectors()
    
    # Compute kinetic energy for each k-point
    for k_idx in range(self.N_k):
        k = self.k_space[k_idx]
        
        # Initialize energy
        E_k = 0.0
        
        # Nearest-neighbor hopping contribution
        for n_vec in nn_vectors:
            k_dot_n = k[0] * n_vec[0] + k[1] * n_vec[1]
            E_k += self.t_1 * np.exp(-1j * k_dot_n)
        
        # Next-nearest-neighbor hopping (combinations of nearest neighbors)
        nnn_vectors = []
        for i, n1 in enumerate(nn_vectors):
            for n2 in nn_vectors[i+1:]:  # Avoid double-counting
                nnn_vec = (n1[0] + n2[0], n1[1] + n2[1])
                if nnn_vec not in nnn_vectors and nnn_vec != (0, 0):
                    nnn_vectors.append(nnn_vec)
        
        for n_vec in nnn_vectors:
            k_dot_n = k[0] * n_vec[0] + k[1] * n_vec[1]
            E_k += self.t_2 * np.exp(-1j * k_dot_n)
        
        # Assign the same energy to both spin up and spin down
        H_nonint[0, 0, k_idx] = E_k
        H_nonint[1, 1, k_idx] = E_k
    
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian using Hartree-Fock approximation.
    
    Includes:
    - Hartree term: interaction between average densities
    - Fock term: exchange interaction
    
    Args:
        exp_val (np.ndarray): Expectation values with shape (D_flattened, D_flattened, N_k)
        
    Returns:
        np.ndarray: Interacting Hamiltonian with shape (D, D, N_k)
    """
    exp_val = unflatten(exp_val, self.D, self.N_k)
    H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)

    # Calculate average densities for Hartree term
    n_up = np.mean(exp_val[0, 0, :])    # <c_up^dagger(k) c_up(k)>
    n_down = np.mean(exp_val[1, 1, :])  # <c_down^dagger(k) c_down(k)>
    
    # Hartree term - on-site interaction only
    H_int[0, 0, :] += self.U_0 * n_down / self.N_k  # Interaction of spin up with average spin down density
    H_int[1, 1, :] += self.U_0 * n_up / self.N_k    # Interaction of spin down with average spin up density
    
    # Fock term - requires interaction potential U(k1-k2)
    nn_vectors = self.get_nearest_neighbor_vectors()
    
    for k2_idx in range(self.N_k):
        for k1_idx in range(self.N_k):
            k_diff = self.k_space[k1_idx] - self.k_space[k2_idx]
            
            # Calculate interaction potential U(k1-k2)
            U_k = self.U_0  # On-site contribution
            
            # Nearest-neighbor contribution
            for n_vec in nn_vectors:
                k_dot_n = k_diff[0] * n_vec[0] + k_diff[1] * n_vec[1]
                U_k += self.U_1 * np.exp(-1j * k_dot_n)
            
            # Fock exchange terms
            # For s=up, s'=down: c_down^dagger(k2) c_up(k2)
            H_int[1, 0, k2_idx] -= U_k * exp_val[0, 1, k1_idx] / self.N_k
            
            # For s=down, s'=up: c_up^dagger(k2) c_down(k2)
            H_int[0, 1, k2_idx] -= U_k * exp_val[1, 0, k1_idx] / self.N_k
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """
    Generates the total Hartree-Fock Hamiltonian by combining non-interacting and
    interacting parts.
    
    Args:
        exp_val (np.ndarray): Expectation values
        return_flat (bool): Whether to return a flattened array
        
    Returns:
        np.ndarray: Total Hamiltonian, flattened if return_flat=True
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    
    if return_flat:
      return flattened(H_total, self.D, self.N_k)
    else:
      return H_total
