import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  2D three-orbital model with Hartree-Fock mean-field approximation.

  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters 't_pd', 't_pp',
                        'U_p', 'U_d', 'V_pp', 'V_pd', 'Delta', 'mu'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'t_pd':1.0, 't_pp': 1.0, 'U_p':1.0, 'U_d':1.0,
                                                          'V_pp':1.0, 'V_pd':1.0, 'Delta':1.0, 'mu':1.0},
                                                          filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2, 3) # Number of flavors identified: (spin, orbital)
    self.basis_order = {'0': 'spin', '1':'orbital'}
    # Order for each flavor:
    # 0: spin up, spin down
    # 1: p_x, p_y, d

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.t_pd = parameters['t_pd'] # Hopping parameter between p and d orbitals
    self.t_pp = parameters['t_pp'] # Hopping parameter between p orbitals
    self.U_p = parameters['U_p'] # On-site interaction strength for p orbitals
    self.U_d = parameters['U_d'] # On-site interaction strength for d orbital
    self.V_pp = parameters['V_pp'] # Nearest-neighbor interaction strength between p orbitals
    self.V_pd = parameters['V_pd'] # Nearest-neighbor interaction strength between p and d orbitals
    self.Delta = parameters['Delta'] # Crystal field splitting
    self.mu = parameters['mu'] # Chemical potential

    # Defining effective interaction parameters
    self.U_p_tilde = self.U_p + 8*self.V_pp - 8*self.V_pd
    self.V_pp_tilde = 8*self.V_pp - self.U_p
    self.U_d_tilde = self.U_d - 4*self.V_pd

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    # Kinetic energy terms
    H_nonint[0, 0, 0, 1, :] = -4 * self.t_pp * np.cos(self.k_space[:, 0]/2) * np.cos(self.k_space[:, 1]/2) # gamma_2(k) for spin up
    H_nonint[1, 0, 1, 1, :] = -4 * self.t_pp * np.cos(self.k_space[:, 0]/2) * np.cos(self.k_space[:, 1]/2) # gamma_2(k) for spin down
    H_nonint[0, 1, 0, 0, :] = -4 * self.t_pp * np.cos(self.k_space[:, 0]/2) * np.cos(self.k_space[:, 1]/2) # gamma_2(k) for spin up
    H_nonint[1, 1, 1, 0, :] = -4 * self.t_pp * np.cos(self.k_space[:, 0]/2) * np.cos(self.k_space[:, 1]/2) # gamma_2(k) for spin down

    H_nonint[0, 0, 0, 2, :] = -2 * self.t_pd * np.cos(self.k_space[:, 0]/2) # gamma_1(k_x) for spin up
    H_nonint[1, 0, 1, 2, :] = -2 * self.t_pd * np.cos(self.k_space[:, 0]/2) # gamma_1(k_x) for spin down
    H_nonint[0, 2, 0, 0, :] = -2 * self.t_pd * np.cos(self.k_space[:, 0]/2) # gamma_1(k_x) for spin up
    H_nonint[1, 2, 1, 0, :] = -2 * self.t_pd * np.cos(self.k_space[:, 0]/2) # gamma_1(k_x) for spin down

    H_nonint[0, 1, 0, 2, :] = -2 * self.t_pd * np.cos(self.k_space[:, 1]/2) # gamma_1(k_y) for spin up
    H_nonint[1, 1, 1, 2, :] = -2 * self.t_pd * np.cos(self.k_space[:, 1]/2) # gamma_1(k_y) for spin down
    H_nonint[0, 2, 0, 1, :] = -2 * self.t_pd * np.cos(self.k_space[:, 1]/2) # gamma_1(k_y) for spin up
    H_nonint[1, 2, 1, 1, :] = -2 * self.t_pd * np.cos(self.k_space[:, 1]/2) # gamma_1(k_y) for spin down

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) # 2, 3, N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    # Calculate the mean densities for spin up and spin down
    n = np.mean(np.sum(exp_val, axis=(0, 1))) # <c^\dagger_{k',o,s} c_{k',o,s}>
    n_p_x = np.mean(exp_val[0, 0, :] + exp_val[1, 0, :]) # <c^\dagger_{k',p_x,s} c_{k',p_x,s}>
    n_p_y = np.mean(exp_val[0, 1, :] + exp_val[1, 1, :]) # <c^\dagger_{k',p_y,s} c_{k',p_y,s}>
    n_p =  np.mean(exp_val[0, 0, :] + exp_val[1, 0, :] + exp_val[0, 1, :] + exp_val[1, 1, :]) # <c^\dagger_{k',p_i,s} c_{k',p_i,s}>
    eta = np.mean(exp_val[0, 0, :] + exp_val[1, 0, :] - exp_val[0, 1, :] - exp_val[1, 1, :])

    # Hartree-Fock terms
    H_int[0, 0, 0, 0, :] = self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4  + self.U_d_tilde * (n - n_p) / 2 # xi_x for spin up
    H_int[1, 0, 1, 0, :] = self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4  + self.U_d_tilde * (n - n_p) / 2 # xi_x for spin down
    H_int[0, 1, 0, 1, :] = self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4  + self.U_d_tilde * (n - n_p) / 2 # xi_y for spin up
    H_int[1, 1, 1, 1, :] = self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4  + self.U_d_tilde * (n - n_p) / 2 # xi_y for spin down
    H_int[0, 2, 0, 2, :] = self.U_d_tilde * (n - n_p) / 2 # xi_d for spin up
    H_int[1, 2, 1, 2, :] = self.U_d_tilde * (n - n_p) / 2 # xi_d for spin down

    H_int += -self.U_p_tilde * (n_p)**2 / 8 + self.V_pp_tilde * (eta)**2 / 8 - self.U_d_tilde * (n - n_p)**2 / 4 # f(n_p, eta)

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
      return self.flatten(H_total)
    else:
      return H_total #l1, s1, q1, ....k

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D,self.D, self.Nk))
