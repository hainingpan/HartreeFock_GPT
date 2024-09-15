import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  The Hartree-Fock Hamiltonian for a 2D system with three orbitals (px, py, d)
  and spin degeneracy. The Hamiltonian is given by:
  H = H_0 + H_int,
  where H_0 is the non-interacting part and H_int is the interacting part.

  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters
        't_pd', 't_pp', 'Delta', 'U_p', 'V_pp', 'U_d', 'V_pd'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'t_pd':1.0, 't_pp': 0.5, 'Delta':1.0, 'U_p': 1.0, 'V_pp':0.4, 'U_d': 1.0, 'V_pd': 0.2}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'square'
    self.D = (2, 3) # (spin, orbital)
    self.basis_order = {'0': 'spin', '1':'orbital'}
    # Order for each flavor:
    # 0: spin up, spin down
    # 1: p_x, p_y, d

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)

    # Model parameters
    self.t_pd = parameters['t_pd'] # Hopping between p and d orbitals.
    self.t_pp = parameters['t_pp'] # Hopping between p orbitals.
    self.Delta = parameters['Delta'] # Crystal field splitting.
    self.U_p = parameters['U_p'] # On-site interaction on p orbital.
    self.V_pp = parameters['V_pp'] # Nearest-neighbour interaction between p orbitals.
    self.U_d = parameters['U_d'] # On-site interaction on d orbital.
    self.V_pd = parameters['V_pd'] # Nearest-neighbour interaction between p and d orbitals.

    # Effective interaction parameters.
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
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    
    # Define hopping terms.
    gamma_1_x = -2 * self.t_pd * np.cos(self.k_space[:, 0] / 2)
    gamma_1_y = -2 * self.t_pd * np.cos(self.k_space[:, 1] / 2)
    gamma_2 = -4 * self.t_pp * np.cos(self.k_space[:, 0] / 2) * np.cos(self.k_space[:, 1] / 2)

    # Assign matrix elements for both spins.
    for s in range(2):
        # Assign diagonal elements of kinetic energy
        H_nonint[s, 0, s, 0, :] = self.Delta # p_x, p_x
        H_nonint[s, 1, s, 1, :] = self.Delta # p_y, p_y
        
        # Assign off-diagonal elements of kinetic energy
        H_nonint[s, 0, s, 1, :] = gamma_2 # p_x, p_y
        H_nonint[s, 1, s, 0, :] = gamma_2 # p_y, p_x

        H_nonint[s, 0, s, 2, :] = gamma_1_x # p_x, d
        H_nonint[s, 2, s, 0, :] = gamma_1_x # d, p_x

        H_nonint[s, 1, s, 2, :] = gamma_1_y # p_y, d
        H_nonint[s, 2, s, 1, :] = gamma_1_y # d, p_y
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    
    # Calculate relevant densities.
    n = np.mean(np.sum(exp_val, axis=(0, 1), keepdims=True), axis=-1)
    n_x_p = np.mean(exp_val[0, 0, :] + exp_val[1, 0, :]) # p_x orbital
    n_y_p = np.mean(exp_val[0, 1, :] + exp_val[1, 1, :]) # p_y orbital
    n_p = n_x_p + n_y_p
    eta = n_y_p - n_x_p

    # Calculate diagonal elements of interaction energy
    xi_x = self.Delta + (self.U_p_tilde * n_p / 4) - (self.V_pp_tilde * eta / 4)
    xi_y = self.Delta + (self.U_p_tilde * n_p / 4) + (self.V_pp_tilde * eta / 4)
    xi_d = (self.U_d_tilde * (n - n_p) / 2)

    # Calculate the constant term
    f = (-self.U_p_tilde * n_p**2 / 8) + (self.V_pp_tilde * eta**2 / 8) - (self.U_d_tilde * (n - n_p)**2 / 4)
    
    # Assign matrix elements for both spins.
    for s in range(2):
        H_int[s, 0, s, 0, :] = xi_x # p_x, p_x
        H_int[s, 1, s, 1, :] = xi_y # p_y, p_y
        H_int[s, 2, s, 2, :] = xi_d # d, d

        # Adding constant energy contribution
        H_int[s, 0, s, 0, :] +=  f # p_x, p_x
        H_int[s, 1, s, 1, :] +=  f # p_y, p_y
        H_int[s, 2, s, 2, :] +=  f # d, d
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) ->np.ndarray:
    """
      Generates the total Hartree-Fock Hamiltonian.

      Args:
          exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

      Returns:
          np.ndarray: The total Hamiltonian with shape (D, D, N_k).
    """
    N_k = exp_val.shape[-1]
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if flatten:
      return flatten(H_total)
    else:
      return H_total

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D,self.D, self.Nk))
