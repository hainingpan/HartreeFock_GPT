from HF import *
import numpy as np
from typing import Any
from scipy.constants import hbar

class HartreeFockHamiltonian:
  """
  2D Twisted Bilayer Graphene Hamiltonian with Hartree-Fock interaction.

  Args:
    N_shell (int): Number of k-point shells to generate.
    parameters (dict[str, Any]): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor (between 0 and 1). Defaults to 0.5.
  """
  def __init__(self, N_shell: int, parameters:dict[str, Any], filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'triangular'
    self.D = (2, 2) # (layer, sublattice)
    self.basis_order = {'0': 'layer', '1': 'sublattice'}
    # Order for each flavor:
    # 0: layer: top, bottom
    # 1: sublattice: A, B

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)

    # All other parameters such as interaction strengths
    self.theta = parameters['theta'] # Twist angle
    self.a = parameters['a'] # Lattice constant of monolayer graphene.
    self.aM = self.a / (2 * np.sin(self.theta / 2)) # Moire lattice constant
    self.vF = parameters['vF'] # Fermi velocity

    self.omega0 = parameters['omega0'] # Interlayer tunneling parameter
    self.omega1 = parameters['omega1'] # Interlayer tunneling parameter
    
    #TODO: Are these parameters of the model? Or specific to the interaction?
    self.V = V # Interaction potential
    self.rho_iso = rho_iso # Density matrix of isolated layers at charge neutrality

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
    for i in range(N_k):
        H_nonint[0, 0, i] =  #TODO: Insert expression for h_theta/2(k)
        H_nonint[1, 1, i] =  #TODO: Insert expression for h_-theta/2(k')
        H_nonint[0, 1, i] =  #TODO: Insert expression for h_T(r)
        H_nonint[1, 0, i] =  #TODO: Insert expression for h^\dagger_T(r)
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

    #TODO: Define these based on the expressions given in the problem statement.
    delta_rho_GG = function of exp_val # \delta \rho_{\alpha'\alpha'}(\bm{G}-\bm{G}')
    delta_rho_GGpp = function of exp_val # \delta \rho_{\alpha, \bm{G}+\bm{G}'';\beta, \bm{G}'+\bm{G}''} (\bm{k}')

    for i in range(N_k):
        H_int[0, 0, i] =  #TODO: Insert expression for Sigma^H_{0,0} + Sigma^F_{0,0}
        H_int[1, 1, i] =  #TODO: Insert expression for Sigma^H_{1,1} + Sigma^F_{1,1}
        H_int[0, 1, i] =  #TODO: Insert expression for Sigma^F_{0,1}
        H_int[1, 0, i] =  #TODO: Insert expression for Sigma^F_{1,0}
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
