import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters 't_up', 't_down', 'U'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'t_up':1.0, 't_down':1.0, 'U':1.0}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2,) # Number of flavors identified.
    self.basis_order = {'0': 'spin'}
    # Order for each flavor:
    # 0: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.t_up = parameters['t_up'] # Hopping parameter for up spin
    self.t_down = parameters['t_down'] # Hopping parameter for down spin
    self.U = parameters['U'] # Interaction strength

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros((self.D[0], self.D[0], N_k), dtype=np.float32)
    # Kinetic energy for spin up and spin down
    for i in range(len(self.k_space)):
        H_nonint[0, 0, i] =  self.t_up * np.sum(np.exp(-1j*np.dot(self.k_space[i], self.n))) # E_s(k) for spin up
        H_nonint[1, 1, i] =  self.t_down * np.sum(np.exp(-1j*np.dot(self.k_space[i], self.n))) # E_s(k) for spin down
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

    # Hartree Terms
    for s in range(self.D[0]):
        for sp in range(self.D[0]):
            for k1 in range(N_k):
                for k2 in range(N_k):
                    H_int[sp, sp, k2] += 1/N_k * self.U * exp_val[s, s, k1] # U(0) <c^\dagger_s(k1) c_s(k1)> c^\dagger_{s'}(k2) c_{s'}(k2)

    # Fock Terms
    for s in range(self.D[0]):
        for sp in range(self.D[0]):
            for k1 in range(N_k):
                for k2 in range(N_k):
                    H_int[sp, s, k2] += -1/N_k * self.U * exp_val[s, sp, k1] # -U(k1-k2) <c^\dagger_s(k1) c_{s'}(k1)> c^\dagger_{s'}(k2) c_s(k2)
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
