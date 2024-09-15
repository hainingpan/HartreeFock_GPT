from HF import *
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters 't' and 'U'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
    temperature (float, optional): Temperature. Defaults to 0.0.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'hbar': 1.0, 'm_b': 1.0, 'm_t':1.0, 'kappa': 1.0, 'Delta_b': 1.0, 'Delta_t': 1.0, 'Delta_T': 1.0, 'e': 1.0, 'epsilon': 1.0, 'd': 1.0}, filling_factor: float=0.5):
    self.lattice = 'triangular'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2, 2) # Number of flavors identified.
    self.basis_order = {'0': 'band', '1': 'valley'}
    # Order for each flavor:
    # 0: b, t
    # 1: +K, -K

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.k_space = generate_k_space(symmetry=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.hbar = parameters['hbar'] # Reduced Planck constant
    self.m_b = parameters['m_b'] # Mass of the b band
    self.m_t = parameters['m_t'] # Mass of the t band
    self.kappa = parameters['kappa'] # Momentum space shift
    self.Delta_b = parameters['Delta_b'] # Potential of the b band
    self.Delta_t = parameters['Delta_t'] # Potential of the t band
    self.Delta_T = parameters['Delta_T'] # Potential of the T band
    self.e = parameters['e'] # Charge of the electron
    self.epsilon = parameters['epsilon'] # Dielectric constant
    self.d = parameters['d'] # Screening length

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    # Kinetic energy for b+K, t+K, b-K, t-K.
    H_nonint[0, 0, 0, 0, 0, 0, :] = -self.hbar**2 * self.k_space[:, 0]**2 / (2 * self.m_b)
    H_nonint[0, 1, 0, 0, 1, 0, :] =  -self.hbar**2 * (self.k_space[:, 0] - self.kappa)**2 / (2 * self.m_t)
    H_nonint[1, 0, 1, 1, 0, 1, :] = -self.hbar**2 * self.k_space[:, 0]**2 / (2 * self.m_b)
    H_nonint[1, 1, 1, 1, 1, 1, :] =  -self.hbar**2 * (self.k_space[:, 0] + self.kappa)**2 / (2 * self.m_t)

    # Potential energy for b+K, t+K, b-K, t-K.
    H_nonint[0, 0, 0, 0, 0, 0, :] += self.Delta_b # Delta_b(r)
    H_nonint[0, 1, 0, 0, 0, 0, :] += self.Delta_T # Delta_{T,+K}(r)
    H_nonint[0, 0, 0, 0, 1, 0, :] += self.Delta_T # Delta_{T,+K}^*(r)
    H_nonint[0, 1, 0, 0, 1, 0, :] += self.Delta_t # Delta_t(r)
    H_nonint[1, 0, 1, 1, 0, 1, :] += self.Delta_b # Delta_b(r)
    H_nonint[1, 1, 1, 1, 0, 1, :] += self.Delta_T # Delta_{T,-K}(r)
    H_nonint[1, 0, 1, 1, 1, 1, :] += self.Delta_T # Delta_{T,-K}^*(r)
    H_nonint[1, 1, 1, 1, 1, 1, :] += self.Delta_t # Delta_t(r)
  
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

    # Calculate the mean densities for spin up and spin down
    for l1 in range(2):
      for tau1 in range(2):
        for l2 in range(2):
          for tau2 in range(2):
            for q1 in range(N_k):
              for q2 in range(N_k):
                for q3 in range(N_k):
                  for q4 in range(N_k):
                    if q1 + q2 == q3 + q4:
                      # Hartree terms
                      H_int[l2, tau2, q2, l2, tau2, q3, :] += np.mean(exp_val[l1, tau1, q1, l1, tau1, q4, :]) * 2 * np.pi * self.e**2 * np.tanh(np.abs(q1 - q4) * self.d) / (self.epsilon * np.abs(q1-q4))
                      # Fock terms
                      H_int[l2, tau2, q2, l1, tau1, q4, :] -= np.mean(exp_val[l1, tau1, q1, l2, tau2, q3, :]) * 2 * np.pi * self.e**2 * np.tanh(np.abs(self.k_space + q1 - self.k_space - q4) * self.d) / (self.epsilon * np.abs(self.k_space + q1 - self.k_space - q4)) # V(k+q1 - k - q4)
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
      return H_total

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D,self.D, self.Nk))
