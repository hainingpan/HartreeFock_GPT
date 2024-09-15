from HF import *
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_kx (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters:
        'tN': Nearest-neighbor hopping for the a orbital.
        'tB': Nearest-neighbor hopping for the b orbital.
        'tBN': Nearest-neighbor hopping between a and b orbitals.
        'Delta': On-site energy difference between a and b orbitals.
        'UN': Hubbard U interaction strength for the a orbital.
        'UB': Hubbard U interaction strength for the b orbital.
        'VBN': Density-density interaction strength between a and b orbitals.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
    temperature (float, optional): Temperature. Defaults to 0.0.
    n (str | None, optional): Parameter used in chemical potential calculation. Defaults to None.
  """
  def __init__(self, N_kx: int=10, parameters: dict={'tN': 1.0, 'tB': 1.0, 'tBN':0.5, 'Delta': 0.0, 'UN': 1.0, 'UB': 1.0, 'VBN': 0.5}, filling_factor: float=0.5):
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2, 2) # Number of flavors identified: orbital, spin.
    self.basis_order = {'0': 'orbital', '1': 'spin'}
    # Order for each flavor:
    # 0: a, b
    # 1: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.n = n # Number of particles in the system.
    self.k_space = generate_k_space(symmetry=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.tN = parameters['tN'] # Nearest-neighbor hopping for the a orbital.
    self.tB = parameters['tB'] # Nearest-neighbor hopping for the b orbital.
    self.tBN = parameters['tBN'] # Nearest-neighbor hopping between a and b orbitals.
    self.Delta = parameters['Delta'] # On-site energy difference between a and b orbitals.
    self.UN = parameters['UN'] # Hubbard U interaction strength for the a orbital.
    self.UB = parameters['UB'] # Hubbard U interaction strength for the b orbital.
    self.VBN = parameters['VBN'] # Density-density interaction strength between a and b orbitals.

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    # Kinetic energy for a and b orbitals, spin up and down.
    H_nonint[0, 0, 0, 0, :] = -2 * self.tN * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1])) + self.Delta  
    H_nonint[0, 1, 0, 1, :] = -2 * self.tN * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1])) + self.Delta  
    H_nonint[1, 0, 1, 0, :] = -2 * self.tB * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))
    H_nonint[1, 1, 1, 1, :] = -2 * self.tB * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  

    # Hybridization between a and b orbitals, spin up and down
    H_nonint[0, 0, 1, 0, :] = -2 * self.tBN * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))
    H_nonint[0, 1, 1, 1, :] = -2 * self.tBN * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))
    H_nonint[1, 0, 0, 0, :] = -2 * self.tBN * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))
    H_nonint[1, 1, 0, 1, :] = -2 * self.tBN * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) # (2, 2, 2, 2, N_k)
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    # Calculate the mean densities for spin up and spin down for both orbitals
    n_bu = np.mean(exp_val[1, 0, 1, 0, :]) # <b^+_{k',up} b_{k',up}>
    n_bd = np.mean(exp_val[1, 1, 1, 1, :]) # <b^+_{k',down} b_{k',down}>
    n_au = np.mean(exp_val[0, 0, 0, 0, :]) # <a^+_{k',up} a_{k',up}>
    n_ad = np.mean(exp_val[0, 1, 0, 1, :]) # <a^+_{k',down} a_{k',down}>

    # Hartree-Fock terms
    H_int[1, 1, 1, 1, :] = (self.UB + 2*self.VB) * n_bd + (self.UB + self.VBN)*n_bu + self.VBN * (n_au + n_ad) # Interaction of b, down with: average b, down density, average b, up density, average a, up density, and average a, down density
    H_int[1, 0, 1, 0, :] = (self.UB + 2*self.VB) * n_bu + (self.UB + self.VBN)*n_bd + self.VBN * (n_au + n_ad) # Interaction of b, up with: average b, up density, average b, down density, average a, up density, and average a, down density
    H_int[0, 1, 0, 1, :] = self.UN * n_ad + self.VBN*(n_bu + n_bd) # Interaction of a, down with: average a, down density, average b, up density, and average b, down density
    H_int[0, 0, 0, 0, :] = self.UN * n_au + self.VBN*(n_bu + n_bd) # Interaction of a, up with: average a, up density, average b, up density, and average b, down density
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
