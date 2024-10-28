from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a three-orbital model on a square lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell:int=1, parameters: dict={'t_pd': 1.0, 't_pp': 0.5, 'Delta': 2.0, 'U_p': 3.0, 'U_d': 4.0, 'V_pd': 1.0, 'V_pp': 0.5}, filling_factor: float=0.5):
    self.lattice = 'square'
    self.D = (3, 2) # orbital, spin
    self.basis_order = {'0': 'orbital', '1': 'spin'}
    # Order for each flavor:
    # 0: px, py, d
    # 1: up, down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Default temperature
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.N_k = self.k_space.shape[0]

    # Model parameters with default values
    self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
    self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping
    self.Delta = parameters.get('Delta', 2.0) # Charge transfer energy
    self.U_p = parameters.get('U_p', 3.0)  # On-site interaction on p-orbital
    self.U_d = parameters.get('U_d', 4.0)  # On-site interaction on d-orbital
    self.V_pd = parameters.get('V_pd', 1.0)  # Intersite interaction between p and d orbitals
    self.V_pp = parameters.get('V_pp', 0.5)  # Intersite interaction between p orbitals


    # Derived interaction parameters (precompute for efficiency)
    self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
    self.V_pp_tilde = 8 * self.V_pp - self.U_p
    self.U_d_tilde = self.U_d - 4 * self.V_pd
    self.aM = 1.0  # Lattice constant (Assuming a = 1)

    return

  def generate_non_interacting(self) -> np.ndarray:
    H_nonint = np.zeros(self.D + (self.N_k,), dtype=np.float32)

    gamma_1x = -2 * self.t_pd * np.cos(self.k_space[:, 0] / 2)
    gamma_1y = -2 * self.t_pd * np.cos(self.k_space[:, 1] / 2)
    gamma_2 = -4 * self.t_pp * np.cos(self.k_space[:, 0] / 2) * np.cos(self.k_space[:, 1] / 2)

    for s in range(self.D[1]):  # Iterate over spins
        H_nonint[0, 1, s, :] = gamma_2  # gamma_2(k) p_x - p_y
        H_nonint[1, 0, s, :] = gamma_2  # gamma_2(k) p_y - p_x

        H_nonint[0, 2, s, :] = gamma_1x  # gamma_1(kx) p_x - d
        H_nonint[2, 0, s, :] = gamma_1x  # gamma_1(kx) d - p_x

        H_nonint[1, 2, s, :] = gamma_1y # gamma_1(ky) p_y - d
        H_nonint[2, 1, s, :] = gamma_1y  # gamma_1(ky) d - p_y

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    exp_val = self.expand(exp_val)
    H_int = np.zeros(self.D + (self.N_k,), dtype=np.float32)

    # Calculate n^p, n, and eta
    n_p = 2 * np.mean(exp_val[0, 0, :] + exp_val[0, 1, :] + exp_val[1, 0, :] + exp_val[1, 1, :]) #px_up + px_down + py_up + py_down
    n = np.mean(np.sum(exp_val, axis=(0, 1)))
    eta = np.mean(exp_val[0, 0, :] + exp_val[0, 1, :] - (exp_val[1, 0, :] + exp_val[1, 1, :]))

    xi_x = self.Delta + (self.U_p_tilde * n_p / 4) - (self.V_pp_tilde * eta / 4) # Removed mu since it will be incorporated in the chemical potential class
    xi_y = self.Delta + (self.U_p_tilde * n_p / 4) + (self.V_pp_tilde * eta / 4) # Removed mu since it will be incorporated in the chemical potential class
    xi_d = (self.U_d_tilde * (n - n_p) / 2) # Removed mu since it will be incorporated in the chemical potential class
    f = -(self.U_p_tilde * (n_p**2) / 8) + (self.V_pp_tilde * (eta**2) / 8) - (self.U_d_tilde * ((n - n_p)**2) / 4)
   
    for s in range(self.D[1]):  # Iterate over spins
        H_int[0, 0, s, :] = xi_x + f # px-px
        H_int[1, 1, s, :] = xi_y + f # py-py
        H_int[2, 2, s, :] = xi_d + f # d-d

    return H_int



# ... (rest of the class remains the same)


