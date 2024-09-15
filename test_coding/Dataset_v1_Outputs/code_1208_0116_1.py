from HF import *
import numpy as np
from typing import Any
from numpy import pi

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'gamma_0':1.0, 'gamma_1':1.0, 'gamma_2':1.0, 'gamma_3':1.0, 'gamma_N':1.0, 'U_H':1.0, 'U_X':1.0}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    # LM Task: Replace `parameter_kwargs` with all constants and parameters that do NOT appear in EXP-VAL DEPENDENT TERMS.
    # These should be accessible from `generate_Htotal` via self.<parameter_name>. Make sure the `init` and `generate_Htotal` functions are consistent.
    self.lattice = 'triangular'
    self.D = (2, 3) # LM Task: has to define this tuple.
    self.basis_order = {'0': 'spin', '1': 'sublattice'}
    # Order for each flavor:
    # 0: spin up, spin down
    # 1: sublattice A, sublattice B, sublattice C

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # All other parameters such as interaction strengths
    self.gamma_0 = parameters['gamma_0'] # Tight-binding parameter
    self.gamma_1 = parameters['gamma_1'] # Tight-binding parameter
    self.gamma_2 = parameters['gamma_2'] # Tight-binding parameter
    self.gamma_3 = parameters['gamma_3'] # Tight-binding parameter
    self.gamma_N = parameters['gamma_N'] # Tight-binding parameter
    #self.param_1 = parameters['param_1'] # Brief phrase explaining physical significance of `param_1`
    #...
    #self.param_p = parameters['param_p'] # Brief phrase explaining physical significance of `param_p`
    self.U_H = parameters['U_H'] # Hartree interaction
    self.U_X = parameters['U_X'] # Exchange interaction
    self.aM = 2.46 # # LM Task: Define the lattice constant, used for the area.
    # Any other problem specific parameters.

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128)
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    #H_nonint[0, 0, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  
    #H_nonint[1, 1, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))
    a = self.aM
    kx = self.k_space[:, 0]
    ky = self.k_space[:, 1]
    f = np.exp(1j*ky*a/np.sqrt(3))*(1 + 2*np.exp(-1j*3*ky*a/(2*np.sqrt(3)))*np.cos(kx*a/2))
    H_nonint[0, 1, :] = self.gamma_0*f # c_{up, A}^\dagger c_{down, B}
    H_nonint[0, 4, :] = self.gamma_3*np.conjugate(f) + self.gamma_N # c_{up, A}^\dagger c_{down, C}
    H_nonint[0, 5, :] = self.gamma_2 # c_{up, A}^\dagger c_{down, C}
    H_nonint[1, 0, :] = self.gamma_0*np.conjugate(f) # c_{down, A}^\dagger c_{up, B}
    H_nonint[1, 2, :] = self.gamma_1 # c_{down, A}^\dagger c_{up, C}
    H_nonint[2, 1, :] = self.gamma_1 # c_{up, B}^\dagger c_{down, A}
    H_nonint[2, 3, :] = self.gamma_0*f # c_{up, B}^\dagger c_{down, C}
    H_nonint[2, 5, :] = self.gamma_3*np.conjugate(f) # c_{up, B}^\dagger c_{down, C}
    H_nonint[3, 0, :] = self.gamma_3*f + np.conjugate(self.gamma_N) # c_{down, C}^\dagger c_{up, A}
    H_nonint[3, 2, :] = self.gamma_0*np.conjugate(f) # c_{down, C}^\dagger c_{up, B}
    H_nonint[3, 4, :] = self.gamma_1 # c_{down, C}^\dagger c_{up, A}
    H_nonint[4, 1, :] = self.gamma_1 # c_{up, A}^\dagger c_{down, B}
    H_nonint[4, 5, :] = self.gamma_0*f # c_{up, A}^\dagger c_{down, C}
    H_nonint[5, 0, :] = self.gamma_2 # c_{down, B}^\dagger c_{up, A}
    H_nonint[5, 2, :] = self.gamma_3*f # c_{down, B}^\dagger c_{up, C}
    H_nonint[5, 4, :] = self.gamma_0*np.conjugate(f) # c_{down, B}^\dagger c_{up, A}
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    #assert exp_val.shape[0] == self.D, "Dimension of exp_val equal the number of flavors identified."
    exp_val = self.expand(exp_val) # 2, 2, N_k
    Nk = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (Nk,), dtype=np.complex128)

    # If more complicated functions of `exp_val` occur in multiple places,
    # one may add additional functions to the class of the form `func(self, exp_val)`.
    # Eg: the compute_order_parameter(exp_val) function for Emery in Emery_model_upd.
    # Otherwise define dependent expressions below
    #exp0 = function of exp_val
    #exp1 = function of exp_val
    #...
    #exp_e = function of exp_val
    for l in range(2):
      for s in range(3):
        for lp in range(2):
          for sp in range(3):
            # Hartree terms
            H_int[l, s, l, s, :] += self.U_H*np.sum(exp_val[lp, sp, lp, sp, :], axis=0) # \sum_{k'} <c_{k',lp,sp}^\dagger c_{k',lp,sp}> c_{k,l,s}^\dagger c_{k,l,s}

            # Exchange terms
            H_int[l, s, lp, sp, :] -= self.U_X*exp_val[lp, sp, l, s, :] #  <c_{k',lp,sp}^\dagger c_{k',l,s}> c_{k,l,s}^\dagger c_{k,lp,sp}

    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
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
      return H_total #l1, s1, q1, ....k

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D,self.D, self.Nk))
