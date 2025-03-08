import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """
  Emery model Hamiltonian for CuO2 planes with p_x, p_y, and d orbitals.
  
  Args:
    N_shell (int): Number of shells in the first Brillouin zone.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float): Filling factor for the system.
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
    self.lattice = 'square'
    self.D = (2, 3)  # (spin, orbital)
    self.basis_order = {'0': 'spin', '1': 'orbital'}
    # 0: spin - up, down
    # 1: orbital - p_x, p_y, d

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0)  # temperature, default to 0
    self.a = parameters.get('a', 1.0)  # Lattice constant
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

    # Model parameters
    self.Delta = parameters.get('Delta', 1.0)  # Energy difference between p and d orbitals
    self.t_pd = parameters.get('t_pd', 1.0)    # Hopping between p and d orbitals
    self.t_pp = parameters.get('t_pp', 0.5)    # Hopping between p_x and p_y orbitals
    
    # Interaction parameters
    self.U_p = parameters.get('U_p', 3.0)      # On-site interaction for p orbitals
    self.U_d = parameters.get('U_d', 8.0)      # On-site interaction for d orbitals
    self.V_pp = parameters.get('V_pp', 1.0)    # Nearest-neighbor interaction between p orbitals
    self.V_pd = parameters.get('V_pd', 1.0)    # Nearest-neighbor interaction between p and d orbitals
    
    # Effective interaction parameters
    self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
    self.V_pp_tilde = 8 * self.V_pp - self.U_p
    self.U_d_tilde = self.U_d - 4 * self.V_pd
    
    # Total density of holes (a conserved quantity)
    self.n = parameters.get('n', 1.0)
    
    # Chemical potential (not an independent parameter in this model)
    self.mu = 2 * self.V_pd * self.n - self.V_pd * self.n**2
    
    return

  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.
    
    Returns:
      np.ndarray: The non-interacting Hamiltonian with shape (*self.D, *self.D, self.N_k).
    """
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Extract k-space coordinates
    kx = self.k_space[:, 0]
    ky = self.k_space[:, 1]
    
    # Calculate the hopping terms
    gamma_1_kx = -2 * self.t_pd * np.cos(kx / 2)  # p_x-d hopping
    gamma_1_ky = -2 * self.t_pd * np.cos(ky / 2)  # p_y-d hopping
    gamma_2_k = -4 * self.t_pp * np.cos(kx / 2) * np.cos(ky / 2)  # p_x-p_y hopping
    
    # Fill in the Hamiltonian matrix for both spin up and down
    for s in range(2):  # spin loop
        # Off-diagonal terms - same for both spins
        # p_x-p_y hopping
        H_nonint[s, 0, s, 1, :] = gamma_2_k
        H_nonint[s, 1, s, 0, :] = gamma_2_k
        
        # p_x-d hopping
        H_nonint[s, 0, s, 2, :] = gamma_1_kx
        H_nonint[s, 2, s, 0, :] = gamma_1_kx
        
        # p_y-d hopping
        H_nonint[s, 1, s, 2, :] = gamma_1_ky
        H_nonint[s, 2, s, 1, :] = gamma_1_ky
        
        # Diagonal terms - constant part only (the interacting parts will be added later)
        # p_x and p_y orbital energy
        H_nonint[s, 0, s, 0, :] = self.Delta - self.mu
        H_nonint[s, 1, s, 1, :] = self.Delta - self.mu
        
        # d orbital energy - constant part only
        H_nonint[s, 2, s, 2, :] = -self.mu
    
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.
    
    Args:
      exp_val (np.ndarray): Expectation value array with shape (prod(*self.D), prod(*self.D), self.N_k).
    
    Returns:
      np.ndarray: The interacting Hamiltonian with shape (*self.D, *self.D, self.N_k).
    """
    exp_val = unflatten(exp_val, self.D, self.N_k)
    H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Calculate n^p - total density of holes on oxygen sites
    n_p_up_x = np.mean(exp_val[0, 0, 0, 0, :])  # <p_x^dag_up p_x_up>
    n_p_down_x = np.mean(exp_val[1, 0, 1, 0, :])  # <p_x^dag_down p_x_down>
    n_p_up_y = np.mean(exp_val[0, 1, 0, 1, :])  # <p_y^dag_up p_y_up>
    n_p_down_y = np.mean(exp_val[1, 1, 1, 1, :])  # <p_y^dag_down p_y_down>
    
    n_p = n_p_up_x + n_p_down_x + n_p_up_y + n_p_down_y
    
    # Calculate the nematic order parameter eta
    eta = n_p_up_x + n_p_down_x - n_p_up_y - n_p_down_y
    
    # Additional quantities needed for the Hamiltonian
    n_d = self.n - n_p  # Total density on d orbitals
    
    # Fill in the interacting part of the Hamiltonian for both spins
    for s in range(2):  # spin loop
        # p_x orbital - interaction part
        H_int[s, 0, s, 0, :] = self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4
        
        # p_y orbital - interaction part
        H_int[s, 1, s, 1, :] = self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4
        
        # d orbital - interaction part
        H_int[s, 2, s, 2, :] = self.U_d_tilde * n_d / 2
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """
    Generates the total Hamiltonian including non-interacting and interacting parts.
    
    Args:
      exp_val (np.ndarray): Expectation value array.
      return_flat (bool): Whether to return the flattened Hamiltonian.
    
    Returns:
      np.ndarray: The total Hamiltonian.
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    
    # Note: The f(n^p, eta) term is a constant energy shift and doesn't affect 
    # the eigenstates, so we don't need to add it to the Hamiltonian matrix.
    # If total energy is needed, it could be calculated separately.
    
    if return_flat:
      return flattened(H_total, self.D, self.N_k)
    else:
      return H_total
