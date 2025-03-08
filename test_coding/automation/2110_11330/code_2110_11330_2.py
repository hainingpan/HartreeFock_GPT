import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a spin-1/2 system on a triangular lattice.
  
  The Hamiltonian includes:
  - Non-interacting dispersion term: -∑_s ∑_k E_s(k) c^†_ks c_ks
  - Hartree term: (1/N) ∑_s,s' ∑_k,k' U(0) <c^†_ks c_ks> c^†_k's' c_k's'
  - Fock term: -(1/N) ∑_s,s' ∑_k,q U(k-q) <c^†_ks c_ks'> c^†_qs' c_qs
  
  Args:
    N_shell: Number of shells in k-space
    parameters: Dictionary of model parameters
    filling_factor: Filling factor of the system
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (2,)  # Dimension for spin (up, down)
    self.basis_order = {'0': 'spin'}  # 0: up, 1: down
    
    # Occupancy parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0.0)  # Temperature
    self.a = parameters.get('a', 1.0)  # Lattice constant
    self.primitive_vectors = get_primitive_vectors_triangle(self.a)
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]
    
    # Hopping parameters - determine energy dispersion
    self.t_up = parameters.get('t_up', 1.0)  # Hopping parameter for spin up
    self.t_down = parameters.get('t_down', 1.0)  # Hopping parameter for spin down
    
    # Interaction parameters
    self.U_0 = parameters.get('U_0', 1.0)  # On-site interaction U(0)
    self.U_1 = parameters.get('U_1', 0.5)  # Nearest neighbor interaction
    self.U_2 = parameters.get('U_2', 0.25)  # Next-nearest neighbor interaction
    
    return

  def get_nearest_neighbor_vectors(self):
    """
    # Returns the integer coordinate offsets (n1, n2) corresponding to the 
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
    Generates the non-interacting part of the Hamiltonian: -∑_s ∑_k E_s(k) c^†_ks c_ks
    where E_s(k) = ∑_n t_s(n) e^(-ik·n)
    
    Returns:
      Non-interacting Hamiltonian with shape (D, D, N_k)
    """
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Get nearest neighbor vectors for hopping
    n_vectors = self.get_nearest_neighbor_vectors()
    
    # Calculate energy dispersion for each k-point
    for k_idx, k in enumerate(self.k_space):
        # Calculate E_up(k) - dispersion for spin up
        E_up = 0
        for n in n_vectors:
            k_dot_n = k[0] * n[0] + k[1] * n[1]
            E_up += self.t_up * np.exp(-1j * k_dot_n)
        
        # Calculate E_down(k) - dispersion for spin down
        E_down = 0
        for n in n_vectors:
            k_dot_n = k[0] * n[0] + k[1] * n[1]
            E_down += self.t_down * np.exp(-1j * k_dot_n)
        
        # Set diagonal elements (spin up and spin down)
        # Negative sign comes from Hamiltonian definition
        H_nonint[0, 0, k_idx] = -E_up  # Dispersion for spin up
        H_nonint[1, 1, k_idx] = -E_down  # Dispersion for spin down
    
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian (Hartree + Fock terms)
    
    Args:
      exp_val: Expectation value array with shape (D_flattened, D_flattened, N_k)
    
    Returns:
      Interacting Hamiltonian with shape (D, D, N_k)
    """
    exp_val = unflatten(exp_val, self.D, self.N_k)
    H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Calculate average densities for Hartree term
    n_up = np.mean(exp_val[0, 0, :])    # <c_k,up^† c_k,up>
    n_down = np.mean(exp_val[1, 1, :])  # <c_k,down^† c_k,down>
    
    # Hartree term: (1/N) ∑_s,s' ∑_k,k' U(0) <c^†_ks c_ks> c^†_k's' c_k's'
    # Add U(0) * n_s' contribution to each state
    H_int[0, 0, :] = self.U_0 * n_down  # U(0) * n_down for spin up
    H_int[1, 1, :] = self.U_0 * n_up    # U(0) * n_up for spin down
    
    # Fock term: -(1/N) ∑_s,s' ∑_k,q U(k-q) <c^†_ks c_ks'> c^†_qs' c_qs
    for q_idx, q in enumerate(self.k_space):
        for k_idx, k in enumerate(self.k_space):
            # Calculate U(k-q) - interaction potential in momentum space
            k_minus_q = k - q
            k_minus_q_mag = np.linalg.norm(k_minus_q)
            
            # Simple model for U(k-q) based on momentum difference
            if k_minus_q_mag < 1e-10:
                U_k_minus_q = self.U_0  # On-site interaction
            elif k_minus_q_mag < 2.0 * np.pi / (np.sqrt(3) * self.a):
                U_k_minus_q = self.U_1  # Nearest-neighbor interaction
            else:
                U_k_minus_q = self.U_2  # Next-nearest-neighbor interaction
            
            # Apply Fock term for all spin combinations
            for s in range(2):
                for s_prime in range(2):
                    # -U(k-q) * <c^†_ks c_ks'> / N_k
                    fock_term = -U_k_minus_q * exp_val[s, s_prime, k_idx] / self.N_k
                    H_int[s_prime, s, q_idx] += fock_term
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """
    Generates the total Hartree-Fock Hamiltonian by combining non-interacting and interacting parts.
    
    Args:
      exp_val: Expectation value array
      return_flat: If True, returns flattened Hamiltonian
    
    Returns:
      Total Hamiltonian
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if return_flat:
      return flattened(H_total, self.D, self.N_k)
    else:
      return H_total
