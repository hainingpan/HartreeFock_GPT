import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_kx (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters 'V_M', 'phi' and 'm_star'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
    temperature (float, optional): Temperature. Defaults to 0.0.
    n (str | None, optional): Parameter used in chemical potential calculation. Defaults to None.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'V_M':1.0, 'phi': 0.0, 'm_star': 1.0}, filling_factor: float=0.5):
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (6, 2) # (|moire reciprocal lattice vectors|, |spin|)
    self.basis_order = {'0': 'moire reciprocal lattice vector', '1': 'spin'}
    # Order for each flavor:
    # 0: b1, b2, b3, b4, b5, b6
    # 1: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.n = n # Number of particles in the system.
    self.k_space = generate_k_space(symmetry=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.V_M = parameters['V_M'] # Moire modulation strength
    self.phi = parameters['phi'] # Moire modulation shape
    self.m_star = parameters['m_star'] # Effective mass

    self.aM = 1.0 # TODO: Define the lattice constant

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    # Kinetic energy
    for i in range(self.D[0]): # moire reciprocal lattice vectors
      for j in range(self.D[1]): #spin
        H_nonint[i, j, i, j, :] = - (hbar**2)/(2*self.m_star) * (self.k_space + b[i])**2
    
    # Moire potential
    for i in range(6):
        V_j = self.V_M * np.exp((-1)**(i-1) * 1j * self.phi)
        H_nonint[i, :,  i+ b_j, :, :] += V_j  #TODO: Define b_j
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
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    # TODO: Define V_{alpha, beta}
    # Calculate the mean densities for spin up and spin down
    n_up = np.mean(exp_val[0, 0, :]) # <c_{k',up}^\dagger c_{k',up}>
    n_down = np.mean(exp_val[1, 1, :]) # <c_{k',down}^\dagger c_{k',down}>

    # Hartree-Fock terms
    H_int[0, 0, :] = self.U * n_down # Interaction of spin up with average spin down density
    H_int[1, 1, :] = self.U * n_up # Interaction of spin down with average spin up density
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
