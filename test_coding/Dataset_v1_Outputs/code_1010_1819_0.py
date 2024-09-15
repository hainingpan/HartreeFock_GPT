from HF import *
import numpy as np
from typing import Any
from scipy.signal import convolve

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict={'gamma_0':1.0, 'gamma_1':1.0, 'gamma_3':1.0, 'gamma_4':1.0, 'V':1.0}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'triangular'
    self.D = (4,) # LM Task: has to define this tuple.
    self.basis_order = {'0': 'orbital'} # LM Task: Define which flavors to account for
    # this is the basis order that the Hamiltonian will follow
    # 0: orbital_0, orbital_1, orbital_2, orbital_3

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0 # LM Task: assume T = 0.
    self.k_space = generate_k_space(lattice=self.lattice, N_shell) # LM Task: Define the lattice.

    # All other parameters such as interaction strengths
    self.gamma_0 = parameters['gamma_0'] # Tight-binding model parameter.
    self.gamma_1 = parameters['gamma_1'] # Tight-binding model parameter.
    self.gamma_3 = parameters['gamma_3'] # Tight-binding model parameter.
    self.gamma_4 = parameters['gamma_4'] # Tight-binding model parameter.
    self.V = parameters['V'] # Interaction strength.
    self.aM = 1 # Lattice constant.
    # Any other problem specific parameters.

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros((self.D[0], self.D[0], N_k), dtype=np.complex128)
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    H_nonint[0, 1, :] = self.gamma_0*self.f(self.k_space) #gamma_0 f  c_0^\dagger c_1
    H_nonint[1, 0, :] = np.conjugate(H_nonint[0, 1, :]) #gamma_0 f^*  c_1^\dagger c_0

    H_nonint[0, 2, :] = self.gamma_4*self.f(self.k_space) #gamma_4 f  c_0^\dagger c_2
    H_nonint[2, 0, :] = np.conjugate(H_nonint[0, 2, :]) #gamma_4 f^*  c_2^\dagger c_0

    H_nonint[0, 3, :] = self.gamma_3*np.conjugate(self.f(self.k_space)) #gamma_3 f^*  c_0^\dagger c_3
    H_nonint[3, 0, :] = np.conjugate(H_nonint[0, 3, :]) #gamma_3 f  c_3^\dagger c_0

    H_nonint[1, 2, :] = self.gamma_1 #gamma_1  c_1^\dagger c_2
    H_nonint[2, 1, :] = np.conjugate(H_nonint[1, 2, :]) #gamma_1  c_2^\dagger c_1

    H_nonint[1, 3, :] = self.gamma_4*self.f(self.k_space) #gamma_4 f  c_1^\dagger c_3
    H_nonint[3, 1, :] = np.conjugate(H_nonint[1, 3, :]) #gamma_4 f^*  c_3^\dagger c_1

    H_nonint[2, 3, :] = self.gamma_0*self.f(self.k_space) #gamma_0 f  c_2^\dagger c_3
    H_nonint[3, 2, :] = np.conjugate(H_nonint[2, 3, :]) #gamma_0 f^*  c_3^\dagger c_2
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

    # Calculate the mean densities for spin up and spin down
    # Hartree Terms
    for lambda_1 in range(4):
      for lambda_2 in range(4):
        H_int[lambda_1, lambda_2, :] += self.V*np.mean(exp_val[lambda_1, lambda_2, :], axis=0)  
    # V*<c_{k_1,lambda_1}^\dagger c_{k_1,lambda_2}> c_{k_2,lambda_2}^\dagger c_{k_2,lambda_1}

    #Fock Terms
    for lambda_1 in range(4):
      for lambda_2 in range(4):
        temp =  -self.V*convolve(exp_val[lambda_1, lambda_2, :], self.V, mode='same')/self.aM
        H_int[lambda_1, lambda_2, :] += temp #-V(k_1-k_2)<c_{k_1,lambda_1}^\dagger c_{k_1,lambda_2}> c_{k_2,lambda_2}^\dagger c_{k_2,lambda_1}

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
      return H_total #l1, s1, q1, ....k

  def f(self, k):
    return np.exp(1j*k[:, 1]*self.aM/np.sqrt(3))*(1 + 2*np.exp(-1j*3*k[:, 1]*self.aM/(2*np.sqrt(3)))*np.cos(k[:, 0]*self.aM/2))

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.k_space.shape[0]))

  def expand(self, exp_val):
    return exp_val.reshape((self.D[0],self.D[0], self.k_space.shape[0]))
