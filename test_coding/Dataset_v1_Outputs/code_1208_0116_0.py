from HF import *
import numpy as np
from typing import Any
from scipy.fft import fft, ifft, fftfreq

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'gamma_0':1.0, 'gamma_1':1.0, 'gamma_2':1.0, 'gamma_3':1.0, 'gamma_N':1.0, 'U_H':1.0, 'U_X':1.0}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'triangular'
    self.D = (2, 2) # LM Task: has to define this tuple. Number of flavors identified: sublattice, spin
    self.basis_order = {'0': 'sublattice', '1': 'spin'}
    # Order for each flavor:
    # 0: sublattice A, sublattice B
    # 1: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0 # Assume T = 0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)

    # All other parameters such as interaction strengths
    self.gamma_0 = parameters['gamma_0'] # Tight-binding parameter
    self.gamma_1 = parameters['gamma_1'] # Tight-binding parameter
    self.gamma_2 = parameters['gamma_2'] # Tight-binding parameter
    self.gamma_3 = parameters['gamma_3'] # Tight-binding parameter
    self.gamma_N = parameters['gamma_N'] # Tight-binding parameter
    self.U_H = parameters['U_H'] # Strength of Hartree interaction
    self.U_X = parameters['U_X'] # Strength of Exchange interaction
    self.aM = 2.46 # Lattice constant in Angstrom

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128) # Initialize Hamiltonian
    # Kinetic energy
    #H_nonint[0, 0, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  
    f_k = np.exp(1j*self.k_space[:, 1] * self.aM / np.sqrt(3)) * (1 + 2 * np.exp(-1j * 3 * self.k_space[:, 1] * self.aM / (2 * np.sqrt(3))) * np.cos(self.k_space[:, 0] * self.aM / 2))
    H_nonint[0, 0, 1, 1, :] = self.gamma_0 * f_k # c^{\dag}_{{\bf k} A \uparrow} c_{{\bf k} B \uparrow}
    H_nonint[1, 1, 0, 0, :] = self.gamma_0 * np.conjugate(f_k)  # c^{\dag}_{{\bf k} B \uparrow} c_{{\bf k} A \uparrow}
    H_nonint[1, 0, 1, 1, :] = self.gamma_1 # c^{\dag}_{{\bf k} B \uparrow} c_{{\bf k} B \downarrow}
    H_nonint[0, 1, 0, 0, :] = self.gamma_1 # c^{\dag}_{{\bf k} A \downarrow} c_{{\bf k} A \uparrow}
    H_nonint[0, 1, 1, 1, :] = self.gamma_2 # c^{\dag}_{{\bf k} A \downarrow} c_{{\bf k} B \downarrow}
    H_nonint[1, 1, 0, 1, :] = self.gamma_2 # c^{\dag}_{{\bf k} B \downarrow} c_{{\bf k} A \downarrow}
    H_nonint[0, 0, 1, 0, :] = (self.gamma_3 * np.conjugate(f_k) + self.gamma_N) # c^{\dag}_{{\bf k} A \uparrow} c_{{\bf k} B \downarrow}
    H_nonint[1, 0, 0, 0, :] = (self.gamma_3 * f_k  + np.conjugate(self.gamma_N)) # c^{\dag}_{{\bf k} B \downarrow} c_{{\bf k} A \uparrow}
    H_nonint[1, 1, 0, 1, :] = self.gamma_3 * f_k # c^{\dag}_{{\bf k} B \downarrow} c_{{\bf k} A \downarrow}
    H_nonint[0, 1, 1, 1, :] = self.gamma_3 * np.conjugate(f_k) # c^{\dag}_{{\bf k} A \downarrow} c_{{\bf k} B \downarrow}

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) # 2, 2, N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128)
    for l in range(2):
      for s in range(2):
        H_int[l, s, l, s, :] += self.U_H * np.mean(exp_val[l, s, l, s, :])  # Hartree term: U_H^{\lambda \lambda^{\prime}}
\left[ \sum_{{\bf k}^{\prime}}
\left<  c^{\dag}_{{\bf k}^{\prime} \lambda^{\prime}} c_{{\bf k}^{\prime} \lambda^{\prime}} \right>  \right]
c^{\dag}_{{\bf k} \lambda} c_{{\bf k} \lambda}
        for lp in range(2):
          for sp in range(2):
            # Add other interactions here
            H_int[l, s, lp, sp, :] -= self.U_X * exp_val[lp, sp, l, s, :] # Exchange term: U_{X}^{\lambda \lambda'}
\left({\bf k}^{\prime} - {\bf k} \right)
\left<  c^{\dag}_{{\bf k}^{\prime} \lambda^{\prime}} c_{{\bf k}^{\prime} \lambda} \right>
c^{\dag}_{{\bf k} \lambda} c_{{\bf k} \lambda^{\prime}}  
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
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.k_space.shape[0]))

  def expand(self, exp_val):
    return exp_val.reshape((self.D + self.D + (self.k_space.shape[0],)))
