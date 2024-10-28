from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a two-orbital model on a triangular lattice.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters 't_s', 'U_n'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'t_s':([1.0, -0.5, -0.5], [1.0, -0.5, -0.5]), 'U_n':([1.0],)}, filling_factor: float=0.5):
    self.lattice = 'triangular'  # Lattice symmetry ('square' or 'triangular').
    self.D = (2,)             # Number of orbitals.
    self.basis_order = {'0': 'orbital'}
    # Order for each flavor:
    # 0: s0, s1

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0             # Temperature, defaults to 0.
    self.k_space = generate_k_space(self.lattice, N_shell) # k points are generated using lattice type and N_shell
    self.Nk = self.k_space.shape[0]
    # Model parameters

    self.t_s = parameters.get('t_s', ([1.0, -0.5, -0.5], [1.0, -0.5, -0.5]))  # Hopping parameters for each orbital
    self.U_n = parameters.get('U_n', ([1.0],))       # Interaction strength parameters

    self.aM = 1/np.sqrt(2) # Lattice constant for triangular lattice is 1/sqrt(2), assuming first neighbor hopping


    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """

    H_nonint = np.zeros((self.D[0], self.D[0], self.Nk), dtype=np.complex128) # Assuming aM=1

    # Kinetic energy for each orbital
    for s in range(self.D[0]): #for orbitals s0 and s1
      E_s = np.zeros(self.Nk,dtype=np.complex128)
      for i in range(len(self.t_s[s])): #for nn hopping in the Hamiltonian
          E_s += self.t_s[s][i] * np.exp(-1j*np.dot(self.k_space,generate_hopping_vectors(self.lattice)[i]))
      H_nonint[s, s, :] = E_s
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D, D, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    #exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros((self.D[0], self.D[0], N_k), dtype=np.complex128)

    # Hartree term
    U_0 = self.U_n[0][0]
    for s in range(self.D[0]):
        for sp in range(self.D[0]):
            H_int[sp, sp, :] += (U_0 / N_k) * np.sum(exp_val[s, s, :])

    # Fock term
    for s in range(self.D[0]):
        for sp in range(self.D[0]):
           H_int[s,s,:] -=  (U_0 / N_k) * np.sum(exp_val[s, sp, :])


    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
    """
      Generates the total Hartree-Fock Hamiltonian.

      Args:
          exp_val (np.ndarray): Expectation value array with shape (D, D, N_k).

      Returns:
          np.ndarray: The total Hamiltonian with shape (D, D, N_k).
    """
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
    return exp_val.reshape((self.D[0],self.D[0], self.Nk))

def generate_k_space(lattice, N_shell):
    if lattice == 'square':
        N_kx = 2 * (N_shell + 1)
        k_space = np.zeros((N_kx * N_kx, 2))
        ind = 0
        for nx in range(-N_shell, N_shell+1):
          for ny in range(-N_shell, N_shell+1):
            k_space[ind,0] = 2*np.pi * nx/N_kx
            k_space[ind,1] = 2*np.pi * ny/N_kx
            ind+=1
        return k_space

    if lattice=='triangular':
      Nk=int(3*(N_shell**2)+3*(N_shell)+1) #Triangular lattice size
      k_space = np.zeros((Nk, 2))
      ind=0;
      for nx in range(-N_shell, N_shell+1):
          for ny in range(-N_shell, N_shell+1):
              if -nx - ny >= -N_shell:
                  k_space[ind,0] = nx * 2 *np.pi/np.sqrt(3.)
                  k_space[ind,1] = (-nx - 2 *ny) * 2 *np.pi /3.
                  ind+=1
      return k_space

def generate_hopping_vectors(lattice):
    if lattice == 'square':
        return [(1., 0.), (0., 1.), (-1., 0.), (0., -1.)]
    if lattice == 'triangular':
        return [(1., 0.), (-0.5, np.sqrt(3)/2), (-0.5, -np.sqrt(3)/2)]

