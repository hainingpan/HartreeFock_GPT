from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a three-orbital model on a square lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary of model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int = 10, parameters: dict[str, Any] = {'t_pd': 1.0, 't_pp': 0.5, 'U_p': 2.0, 'U_d': 1.0, 'V_pp': 0.8, 'V_pd': 0.3, 'Delta': 0.1, 'mu': 0.0}, filling_factor: float = 0.5):
    self.lattice = 'square'
    self.D = (3, 2)  # (orbital, spin)
    self.basis_order = {'0': 'orbital', '1': 'spin'}
    # Order for each flavor:
    # 0: px, py, d
    # 1: up, down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Default temperature
    self.k_space = generate_k_space(self.lattice, N_shell)

    # Model parameters with default values
    self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
    self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping
    self.U_p = parameters.get('U_p', 2.0)  # On-site interaction on p orbitals
    self.U_d = parameters.get('U_d', 1.0)  # On-site interaction on d orbital
    self.V_pp = parameters.get('V_pp', 0.8)  # p-p interorbital interaction
    self.V_pd = parameters.get('V_pd', 0.3)  # p-d interorbital interaction
    self.Delta = parameters.get('Delta', 0.1)  # Crystal field splitting
    self.mu = parameters.get('mu', 0.0)  # Chemical Potential

    # Effective interaction parameters (calculated once during initialization)
    self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
    self.V_pp_tilde = 8 * self.V_pp - self.U_p
    self.U_d_tilde = self.U_d - 4 * self.V_pd
    self.aM = 1.0 #Setting lattice constant to 1 for simplicity

    return

  def generate_non_interacting(self) -> np.ndarray:
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + (N_k,), dtype=np.complex64)

    kx = self.k_space[:, 0]
    ky = self.k_space[:, 1]

    gamma_1x = -2 * self.t_pd * np.cos(kx / 2)
    gamma_1y = -2 * self.t_pd * np.cos(ky / 2)
    gamma_2 = -4 * self.t_pp * np.cos(kx / 2) * np.cos(ky / 2)


    for s in range(self.D[1]): # spin
        H_nonint[0, 0, s, 0, 0, s, :] = self.Delta - self.mu #Added constant terms
        H_nonint[0, 1, s, 0, 1, s, :] = self.Delta - self.mu#Added constant terms
        H_nonint[1, 0, s, 1, 0, s, :] = -self.mu # Added constant term
        
        H_nonint[0, 0, s, 0, 1, s, :] = gamma_2
        H_nonint[0, 1, s, 0, 0, s, :] = gamma_2
        H_nonint[0, 0, s, 1, 0, s, :] = gamma_1x
        H_nonint[1, 0, s, 0, 0, s, :] = gamma_1x
        H_nonint[0, 1, s, 1, 0, s, :] = gamma_1y
        H_nonint[1, 0, s, 0, 1, s, :] = gamma_1y

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    exp_val = exp_val.reshape(self.D + (self.k_space.shape[0],))  # Reshape to (3, 2, Nk)
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.complex64)

    n_p = np.sum(exp_val[0:2, :, :]) / (N_k * self.aM**2) # Sum over px and py for both spins. Dividing by Nk and Area
    n = np.sum(exp_val) / (N_k * self.aM**2) # Sum over all orbitals and spins. Dividing by Nk and Area
    # exp_val[orbital, spin, k]
    n_px = np.sum(exp_val[0, :, :]) / (N_k * self.aM**2) # Sum over spin for px orbitals, then mean
    n_py = np.sum(exp_val[1, :, :]) / (N_k * self.aM**2) # Sum over spin for py orbitals, then mean
    eta = n_px - n_py

    xi_x_int = self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4
    xi_y_int = self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4
    xi_d_int = self.U_d_tilde * (n - n_p) / 2
    f_int =  -(self.U_p_tilde * (n_p)**2) / 8 + (self.V_pp_tilde * (eta)**2) / 8 - (self.U_d_tilde * (n - n_p)**2) / 4

    for s in range(self.D[1]):
      H_int[0, 0, s, 0, 0, s, :] = xi_x_int + f_int # px up, px up
      H_int[0, 1, s, 0, 1, s, :] = xi_y_int + f_int# py up, py up
      H_int[1, 0, s, 1, 0, s, :] = xi_d_int + f_int# d up, d up

    return H_int



  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool = True) -> np.ndarray:
    N_k = exp_val.shape[-1]
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if flatten:
      return self.flatten(H_total)
    else:
      return H_total

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D), np.prod(self.D), self.k_space.shape[0]))

  def expand(self, exp_val):
    return exp_val.reshape(self.D + (self.k_space.shape[0],))




