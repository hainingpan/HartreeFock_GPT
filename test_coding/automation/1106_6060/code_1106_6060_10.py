import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a three-orbital (p_x, p_y, d) model on a square lattice.
  
  Args:
    N_shell: Number of shells in the Brillouin zone
    parameters: Dictionary containing model parameters
    filling_factor: Filling factor of the system (default: 0.5)
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={
      't_pd': 1.0,      # p-d hopping parameter
      't_pp': 0.5,      # p-p hopping parameter
      'Delta': 1.0,     # Energy level of p orbitals relative to d
      'U_p': 3.0,       # On-site Coulomb repulsion for p orbitals
      'U_d': 5.0,       # On-site Coulomb repulsion for d orbitals
      'V_pp': 1.0,      # Inter-site interaction between p orbitals
      'V_pd': 1.0,      # Inter-site interaction between p and d orbitals
      'T': 0,           # Temperature
      'a': 1.0          # Lattice constant
    }, filling_factor: float=0.5):
    
    self.lattice = 'square'
    self.D = (2, 3)  # (spin, orbital)
    self.basis_order = {'0': 'spin', '1': 'orbital'}
    # Spin: 0=up, 1=down
    # Orbital: 0=p_x, 1=p_y, 2=d

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0)  # temperature, default to 0
    self.a = parameters.get('a', 1.0)  # Lattice constant
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

    # Model parameters
    self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
    self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping
    self.Delta = parameters.get('Delta', 1.0)  # p-orbital energy level

    # Interaction parameters
    self.U_p = parameters.get('U_p', 3.0)  # p-orbital Coulomb
    self.U_d = parameters.get('U_d', 5.0)  # d-orbital Coulomb
    self.V_pp = parameters.get('V_pp', 1.0)  # p-p interaction
    self.V_pd = parameters.get('V_pd', 1.0)  # p-d interaction

    # Effective interaction parameters
    self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
    self.V_pp_tilde = 8 * self.V_pp - self.U_p
    self.U_d_tilde = self.U_d - 4 * self.V_pd

    return

  def compute_order_parameters(self, exp_val):
    """Computes order parameters from expectation values."""
    # Ensure exp_val is unflattened
    exp_val = unflatten(exp_val, self.D, self.N_k)
    
    # Calculate hole densities for each orbital and spin
    n_px_up = np.mean(exp_val[0, 0, 0, 0, :].real)
    n_px_down = np.mean(exp_val[1, 0, 1, 0, :].real)
    n_py_up = np.mean(exp_val[0, 1, 0, 1, :].real)
    n_py_down = np.mean(exp_val[1, 1, 1, 1, :].real)
    n_d_up = np.mean(exp_val[0, 2, 0, 2, :].real)
    n_d_down = np.mean(exp_val[1, 2, 1, 2, :].real)
    
    # Total p orbital occupation
    n_p = n_px_up + n_px_down + n_py_up + n_py_down
    
    # Nematic order parameter (difference between p_x and p_y occupations)
    eta = (n_px_up + n_px_down) - (n_py_up + n_py_down)
    
    # Total hole density
    n = n_p + n_d_up + n_d_down
    
    return n_p, eta, n

  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Loop over both spin states (up and down)
    for s in range(2):
        # Calculate hopping terms
        gamma1_kx = -2 * self.t_pd * np.cos(self.k_space[:, 0] / 2)  # p_x-d hopping
        gamma1_ky = -2 * self.t_pd * np.cos(self.k_space[:, 1] / 2)  # p_y-d hopping
        gamma2_k = -4 * self.t_pp * np.cos(self.k_space[:, 0] / 2) * np.cos(self.k_space[:, 1] / 2)  # p_x-p_y hopping
        
        # p_x - p_y hopping
        H_nonint[s, 0, s, 1, :] = gamma2_k  # p_x to p_y
        H_nonint[s, 1, s, 0, :] = gamma2_k  # p_y to p_x (Hermitian conjugate)
        
        # p_x - d hopping
        H_nonint[s, 0, s, 2, :] = gamma1_kx  # p_x to d
        H_nonint[s, 2, s, 0, :] = gamma1_kx  # d to p_x (Hermitian conjugate)
        
        # p_y - d hopping
        H_nonint[s, 1, s, 2, :] = gamma1_ky  # p_y to d
        H_nonint[s, 2, s, 1, :] = gamma1_ky  # d to p_y (Hermitian conjugate)
        
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""
    exp_val = unflatten(exp_val, self.D, self.N_k)
    H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Compute order parameters
    n_p, eta, n = self.compute_order_parameters(exp_val)
    
    # Calculate chemical potential (not an independent parameter)
    mu = 2 * self.V_pd * n - self.V_pd * n**2
    
    # Calculate site energies with mean-field interactions
    xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - mu  # p_x energy
    xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - mu  # p_y energy
    xi_d = self.U_d_tilde * (n - n_p) / 2 - mu  # d energy
    
    # Set diagonal elements of the Hamiltonian for each spin
    for s in range(2):
        H_int[s, 0, s, 0, :] = xi_x  # p_x orbital energy
        H_int[s, 1, s, 1, :] = xi_y  # p_y orbital energy
        H_int[s, 2, s, 2, :] = xi_d  # d orbital energy
    
    # Note: f(n_p,eta) is a constant energy shift that doesn't affect eigenstates
    # If needed, it can be calculated as:
    # f_term = (-self.U_p_tilde * n_p**2 / 8 + self.V_pp_tilde * eta**2 / 8 
    #          - self.U_d_tilde * (n - n_p)**2 / 4) * self.N_k
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """Generates the total Hamiltonian by combining non-interacting and interacting parts."""
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if return_flat:
      return flattened(H_total, self.D, self.N_k)
    else:
      return H_total
