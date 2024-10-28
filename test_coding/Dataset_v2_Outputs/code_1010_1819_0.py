from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'gamma_0':1.0, 'gamma_1':1.0, 'gamma_3':1.0, 'gamma_4':1.0, 'V':1.0, 'a':1.0}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (4,)
    self.basis_order = {'0': 'orbital'}
    # 0: orbital 0, orbital 1, orbital 2, orbital 3

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Assuming T=0
    self.k_space = generate_k_space(self.lattice, N_shell) # LM Task: added N_shell input parameter
    self.a = parameters['a'] # Lattice Constant
    self.aM = self.a**2 * np.sqrt(3)/2 # Area of the unit cell.
    # Model parameters
    self.gamma_0 = parameters['gamma_0']
    self.gamma_1 = parameters['gamma_1']
    self.gamma_3 = parameters['gamma_3']
    self.gamma_4 = parameters['gamma_4']
    self.V = parameters['V'] # Interaction strength V(k) # Defaulting to a constant V for now


    return

  def f(self, k):
      return np.exp(1j * k[:, 1] * self.a / np.sqrt(3)) * (1 + 2 * np.exp(-1j * 3 * k[:, 1] * self.a / (2*np.sqrt(3))) * np.cos(k[:, 0] * self.a / 2))


  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.

    Returns:
      np.ndarray: The non-interacting Hamiltonian.
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros((4, 4, N_k), dtype=np.complex64)
    f_k = self.f(self.k_space)
    
    H_nonint[0, 1, :] = self.gamma_0 * f_k # gamma_0 f(k)
    H_nonint[1, 0, :] = self.gamma_0 * np.conj(f_k) # gamma_0 f*(k)
    H_nonint[0, 2, :] = self.gamma_4 * f_k # gamma_4 f(k)
    H_nonint[2, 0, :] = self.gamma_4 * np.conj(f_k) # gamma_4 f*(k)
    H_nonint[0, 3, :] = self.gamma_3 * np.conj(f_k)  # gamma_3 f*(k)
    H_nonint[3, 0, :] = self.gamma_3 * f_k  # gamma_3 f(k)
    H_nonint[1, 2, :] = self.gamma_1 # gamma_1
    H_nonint[2, 1, :] = self.gamma_1 # gamma_1
    H_nonint[1, 3, :] = self.gamma_4 * f_k # gamma_4 f(k)
    H_nonint[3, 1, :] = self.gamma_4 * np.conj(f_k) # gamma_4 f*(k)
    H_nonint[2, 3, :] = self.gamma_0 * f_k # gamma_0 f(k)
    H_nonint[3, 2, :] = self.gamma_0 * np.conj(f_k) # gamma_0 f*(k)
    return H_nonint


  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array.

    Returns:
      np.ndarray: The interacting Hamiltonian.
    """
    exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros((4, 4, N_k), dtype=np.complex64)
    Nk = self.k_space.shape[0]


    for l in range(4): # lambda'
        for lp in range(4): # lambda
            H_int[l, lp, :] += np.mean(exp_val[lp, lp,:]) * self.V * (1/self.aM) #Hartree term

    # Fock term: #Note: V(k) is a constant for this example. Make sure the full expression is used for non-constant V(k)
    for k1 in range(N_k):
        for k2 in range(N_k):
            for l in range(4): # lambda
                for lp in range(4): # lambda'
                    H_int[lp, l, k2] -= exp_val[l, lp, k1] * self.V * (1/self.aM)
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
      """Generates the total Hamiltonian."""
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
      return exp_val.reshape((self.D[0], self.D[0], self.k_space.shape[0]))



