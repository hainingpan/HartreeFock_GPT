import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_kx (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
    temperature (float, optional): Temperature. Defaults to 0.0.
    n (str | None, optional): Parameter used in chemical potential calculation. Defaults to None.
  """
  def __init__(self, N_kx: int=10, parameters: dict={'tN': 1.0, 'tB': 1.0, 'tBN': 1.0, 'Delta': 1.0, 'UB': 1.0, 'UN': 1.0, 'VB': 1.0, 'VBN': 1.0}, filling_factor: float=0.5): #LM Task: Modify parameter_kwargs
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'. #LM Task: Define the lattice constant, used for the area.
    self.D = (2, 2) # Number of flavors identified. #LM Task: Define this tuple.
    self.basis_order = {'0': 'orbital', '1': 'spin'} #LM Task: Define this dictionary.
    # Order for each flavor:
    # 0: a, b
    # 1: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.n = n # Number of particles in the system.
    self.k_space = generate_k_space(symmetry=self.lattice, N_shell) #LM Task: Modify generate_k_space to take symmetry as an argument.
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters #LM Task: Define all parameters here.
    self.tN = parameters['tN'] # Hopping parameter for the N orbital
    self.tB = parameters['tB'] # Hopping parameter for the B orbital
    self.tBN = parameters['tBN'] # Hopping parameter between N and B orbitals
    self.Delta = parameters['Delta'] # On-site energy difference between N and B orbitals
    self.UN = parameters['UN'] # Hubbard U interaction strength on the N orbital
    self.UB = parameters['UB'] # Hubbard U interaction strength on the B orbital
    self.VB = parameters['VB'] # Density-density interaction strength on the B orbital
    self.VBN = parameters['VBN'] # Density-density interaction strength between the N and B orbitals
    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32) #2, 2, 2, 2, N_k
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    H_nonint[0, 0, 0, 0, 0, 0, :] = self.tN  #  a^{\dagger}_{k \uparrow} a_{k \uparrow}
    H_nonint[0, 1, 0, 0, 1, 0, :] = self.tN  #  a^{\dagger}_{k \downarrow} a_{k \downarrow}
    H_nonint[1, 0, 0, 1, 0, 0, :] = self.tB #  b^{\dagger}_{k \uparrow} b_{k \uparrow}
    H_nonint[1, 1, 0, 1, 1, 0, :] = self.tB # b^{\dagger}_{k \downarrow} b_{k \downarrow}
    H_nonint[0, 0, 0, 1, 0, 0, :] = self.tBN # a^{\dagger}_{k \uparrow} b_{k \uparrow}
    H_nonint[0, 1, 0, 1, 1, 0, :] = self.tBN # a^{\dagger}_{k \downarrow} b_{k \downarrow}
    H_nonint[1, 0, 0, 0, 0, 0, :] = self.tBN # b^{\dagger}_{k \uparrow} a_{k \uparrow}
    H_nonint[1, 1, 0, 0, 1, 0, :] = self.tBN # b^{\dagger}_{k \downarrow} a_{k \downarrow}
    H_nonint[0, 0, 0, 0, 0, 0, :] += self.Delta # a^{\dagger}_{k \uparrow} a_{k \uparrow}
    H_nonint[0, 1, 0, 0, 1, 0, :] += self.Delta # a^{\dagger}_{k \downarrow} a_{k \downarrow}
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
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32) #2, 2, 2, 2, N_k

    # Calculate the mean densities for spin up and spin down
    n_bu = np.mean(exp_val[1, 0, :]) # <b^{\dagger}_{k',up} b_{k',up}>
    n_bd = np.mean(exp_val[1, 1, :]) # <b^{\dagger}_{k',down} b_{k',down}>
    n_au = np.mean(exp_val[0, 0, :]) # <a^{\dagger}_{k',up} a_{k',up}>
    n_ad = np.mean(exp_val[0, 1, :]) # <a^{\dagger}_{k',down} a_{k',down}>

    # Hartree-Fock terms
    H_int[1, 1, 0, 1, 1, 0, :] += self.UB * n_bu / N_k # U_B <b^{\dagger}_{k, \uparrow} b_{k, \uparrow} > b^{\dagger}_{k, \downarrow} b_{k, \downarrow}
    H_int[1, 0, 0, 1, 0, 0, :] += self.UB * n_bd / N_k # U_B <b^{\dagger}_{k, \downarrow} b_{k, \downarrow} > b^{\dagger}_{k, \uparrow} b_{k, \uparrow}
    H_int[0, 1, 0, 0, 1, 0, :] += self.UN * n_au / N_k # U_N <a^{\dagger}_{k, \uparrow} a_{k, \uparrow} > a^{\dagger}_{k, \downarrow} a_{k, \downarrow}
    H_int[0, 0, 0, 0, 0, 0, :] += self.UN * n_ad / N_k # U_N <a^{\dagger}_{k, \downarrow} a_{k, \downarrow} > a^{\dagger}_{k, \uparrow} a_{k, \uparrow}
    H_int[1, 0, 0, 1, 0, 0, :] += 2 * self.VB * n_bu / N_k # 2V_B<b^{\dagger}_{k, \uparrow} b_{k, \uparrow} > b^{\dagger}_{k, \uparrow} b_{k, \uparrow}
    H_int[1, 1, 0, 1, 1, 0, :] += 2 * self.VB * n_bd / N_k # 2V_B<b^{\dagger}_{k, \downarrow} b_{k, \downarrow} > b^{\dagger}_{k, \downarrow} b_{k, \downarrow}
    H_int[0, 0, 0, 0, 0, 0, :] += self.VBN * n_bu / N_k # V_BN<b^{\dagger}_{k, \uparrow} b_{k, \uparrow} > a^{\dagger}_{k, \uparrow} a_{k, \uparrow}
    H_int[0, 1, 0, 0, 1, 0, :] += self.VBN * n_bd / N_k # V_BN<b^{\dagger}_{k, \downarrow} b_{k, \downarrow} > a^{\dagger}_{k, \downarrow} a_{k, \downarrow}
    H_int[1, 0, 0, 1, 0, 0, :] += self.VBN * n_au / N_k # V_BN<a^{\dagger}_{k, \uparrow} a_{k, \uparrow} > b^{\dagger}_{k, \uparrow} b_{k, \uparrow}
    H_int[1, 1, 0, 1, 1, 0, :] += self.VBN * n_ad / N_k # V_BN<a^{\dagger}_{k, \downarrow} a_{k, \downarrow} > b^{\dagger}_{k, \downarrow} b_{k, \downarrow}
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
