import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """
  Three-orbital (px, py, d) model on a square lattice with spin.
  
  This class implements a mean-field Hamiltonian for a three-orbital model
  with interactions between p and d orbitals. The model includes nematic 
  ordering between px and py orbitals.
  
  Args:
    N_shell (int): Number of shells in the first Brillouin zone.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float): Filling factor of the system.
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
    self.lattice = 'square'   # Lattice symmetry
    self.D = (2, 3) # Number of flavors (spin, orbital)
    self.basis_order = {'0': 'spin', '1': 'orbital'}
    # Order for each flavor:
    # 0: spin up, spin down
    # 1: px, py, d

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0) # temperature, default to 0
    self.a = parameters.get('a', 1.0) # Lattice constant
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

    # Model parameters
    self.t_pd = parameters.get('t_pd', 1.0) # p-d hopping parameter
    self.t_pp = parameters.get('t_pp', 0.5) # p-p hopping parameter
    self.Delta = parameters.get('Delta', 1.0) # Energy difference between p and d orbitals
    self.U_p = parameters.get('U_p', 1.0) # On-site Coulomb repulsion for p-orbitals
    self.U_d = parameters.get('U_d', 2.0) # On-site Coulomb repulsion for d-orbitals
    self.V_pp = parameters.get('V_pp', 0.5) # Inter-site Coulomb repulsion between p-orbitals
    self.V_pd = parameters.get('V_pd', 0.75) # Inter-site Coulomb repulsion between p and d orbitals
    
    # Compute effective interaction parameters
    self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
    self.V_pp_tilde = 8 * self.V_pp - self.U_p
    self.U_d_tilde = self.U_d - 4 * self.V_pd

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.

    Returns:
      np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # For each k-point and spin
    for s in range(2):  # spin loop
      for k_idx in range(self.N_k):
        kx, ky = self.k_space[k_idx]
        
        # Calculate hopping parameters
        gamma_1_x = -2 * self.t_pd * np.cos(kx / 2)  # px-d hopping
        gamma_1_y = -2 * self.t_pd * np.cos(ky / 2)  # py-d hopping
        gamma_2 = -4 * self.t_pp * np.cos(kx / 2) * np.cos(ky / 2)  # px-py hopping
        
        # Hopping between px and py
        H_nonint[s, 0, s, 1, k_idx] = gamma_2
        H_nonint[s, 1, s, 0, k_idx] = gamma_2
        
        # Hopping between px and d
        H_nonint[s, 0, s, 2, k_idx] = gamma_1_x
        H_nonint[s, 2, s, 0, k_idx] = gamma_1_x
        
        # Hopping between py and d
        H_nonint[s, 1, s, 2, k_idx] = gamma_1_y
        H_nonint[s, 2, s, 1, k_idx] = gamma_1_y
    
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
    
    # Compute expectation values
    # Total density of holes (n)
    n = 0
    for s in range(2):  # spin loop
      for o in range(3):  # orbital loop
        n += np.mean(exp_val[s, o, s, o, :])
    
    # Density of holes on oxygen sites (n_p)
    n_p = 0
    for s in range(2):  # spin loop
      for o in range(2):  # p_x and p_y orbitals
        n_p += np.mean(exp_val[s, o, s, o, :])
    
    # Nematic order parameter (eta)
    eta = 0
    for s in range(2):  # spin loop
      eta += np.mean(exp_val[s, 0, s, 0, :]) - np.mean(exp_val[s, 1, s, 1, :])
    
    # Calculate chemical potential (mu)
    mu = 2 * self.V_pd * n - self.V_pd * n**2
    
    # Calculate on-site energies
    xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu
    xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu
    xi_d = self.U_d_tilde * (n - n_p) / 2 - mu
    
    # Apply on-site energies
    for s in range(2):  # spin loop
      for k_idx in range(self.N_k):
        H_int[s, 0, s, 0, k_idx] = xi_x  # px orbital
        H_int[s, 1, s, 1, k_idx] = xi_y  # py orbital
        H_int[s, 2, s, 2, k_idx] = xi_d  # d orbital
    
    # Calculate f(n_p, eta)/N - the energy offset term
    f_term = -self.U_p_tilde * (n_p**2) / 8 + self.V_pp_tilde * (eta**2) / 8 - self.U_d_tilde * ((n - n_p)**2) / 4
    
    # Add constant energy term to diagonal elements
    for s in range(2):  # spin loop
      for o in range(3):  # orbital loop
        for k_idx in range(self.N_k):
          H_int[s, o, s, o, k_idx] += f_term / self.N_k  # Normalized by number of k-points
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """
    Generates the total Hartree-Fock Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
      return_flat (bool): Whether to return the Hamiltonian in flattened form.

    Returns:
      np.ndarray: The total Hamiltonian with appropriate shape based on return_flat.
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    
    if return_flat:
      return flattened(H_total, self.D, self.N_k)
    else:
      return H_total
