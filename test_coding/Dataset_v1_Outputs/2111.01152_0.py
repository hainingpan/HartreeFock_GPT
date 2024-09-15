import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'hbar': 1.0545718e-34, 'm_b': 9.10938356e-31, 'm_t': 9.10938356e-31, 'kappa': 1e9, 'e': 1.602176634e-19, 'epsilon': 8.8541878128e-12, 'd':1e-9, 'Delta_b': 0, 'Delta_t': 0, 'Delta_T_plusK': 0, 'Delta_T_minusK': 0}, filling_factor: float=0.5):
    # Lattice and basis setup
    self.lattice = 'square'
    self.D = (2, 2) # Number of flavors identified: (layer, valley)
    self.basis_order = {'0': 'layer', '1': 'valley'}
    # Order for each flavor:
    # 0: layer: bottom layer, top layer
    # 1: valley: K, K'

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0 # Assume temperature is 0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)
    self.Nk = self.k_space.shape[0]

    # Model parameters
    self.hbar = parameters['hbar'] # Reduced Planck constant
    self.m_b = parameters['m_b'] # Effective mass in bottom layer
    self.m_t = parameters['m_t'] # Effective mass in top layer
    self.kappa = parameters['kappa'] # Valley offset
    self.e = parameters['e'] # Elementary charge
    self.epsilon = parameters['epsilon'] # Vacuum permittivity
    self.d = parameters['d'] # Screening length
    #On-site energies
    self.Delta_b = parameters['Delta_b'] # Potential in bottom layer
    self.Delta_t = parameters['Delta_t'] # Potential in top layer
    #Interlayer Hoppings
    self.Delta_T_plusK  = parameters['Delta_T_plusK'] # Potential for K valley
    self.Delta_T_minusK = parameters['Delta_T_minusK'] # Potential for K' valley
    self.aM = 1 # LM Task: Define the lattice constant, used for the area.

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
    H_nonint[0, 0, :] = -((self.hbar**2)*(self.k_space[:, 0]**2) + (self.hbar**2)*(self.k_space[:, 1]**2))/(2*self.m_b) # Bottom layer, K valley
    H_nonint[1, 1, :] = -((self.hbar**2)*((self.k_space[:, 0]-self.kappa)**2) + (self.hbar**2)*((self.k_space[:, 1]-self.kappa)**2))/(2*self.m_t) # Top layer, K valley
    H_nonint[2, 2, :] = -((self.hbar**2)*(self.k_space[:, 0]**2) + (self.hbar**2)*(self.k_space[:, 1]**2))/(2*self.m_b) # Bottom layer, K' valley
    H_nonint[3, 3, :] = -((self.hbar**2)*((self.k_space[:, 0]+self.kappa)**2) + (self.hbar**2)*((self.k_space[:, 1]+self.kappa)**2))/(2*self.m_t) # Top layer, K' valley
    # Potential energy
    H_nonint[0, 0, :] += self.Delta_b # Bottom layer, K valley
    H_nonint[1, 1, :] += self.Delta_t # Top layer, K valley
    H_nonint[2, 2, :] += self.Delta_b # Bottom layer, K' valley
    H_nonint[3, 3, :] += self.Delta_t # Top layer, K' valley

    #Interlayer Hopping
    H_nonint[0, 1, :] += self.Delta_T_plusK  # Bottom layer to top layer, K valley
    H_nonint[1, 0, :] += np.conjugate(self.Delta_T_plusK) # Top layer to bottom layer, K valley
    H_nonint[2, 3, :] += self.Delta_T_minusK # Bottom layer to top layer, K' valley
    H_nonint[3, 2, :] += np.conjugate(self.Delta_T_minusK) # Top layer to bottom layer, K' valley

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) # (2, 2, 2, 2, Nk)
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    
    #Hartree Terms
    for l1 in range(self.D[0]):
      for tau1 in range(self.D[1]):
        for q1 in range(self.Nk):
          for l2 in range(self.D[0]):
            for tau2 in range(self.D[1]):
              for q2 in range(self.Nk):
                for q3 in range(self.Nk):
                  q4 = q1+q2-q3
                  if q4>=0 and q4<self.Nk:
                    H_int[l2, tau2, q2, l2, tau2, q3, :] += (1/self.aM) * np.mean(exp_val[l1, tau1, q1, l1, tau1, q4, :]) * 2*np.pi*(self.e**2)*np.tanh(np.linalg.norm(self.k_space[q1]-self.k_space[q4])*self.d)/(self.epsilon*np.linalg.norm(self.k_space[q1]-self.k_space[q4]))

    #Fock Terms
    for l1 in range(self.D[0]):
      for tau1 in range(self.D[1]):
        for q1 in range(self.Nk):
          for l2 in range(self.D[0]):
            for tau2 in range(self.D[1]):
              for q2 in range(self.Nk):
                for q3 in range(self.Nk):
                  q4 = q1+q2-q3
                  if q4>=0 and q4<self.Nk:
                    H_int[l2, tau2, q2, l1, tau1, q4, :] += -(1/self.aM) * np.mean(exp_val[l1, tau1, q1, l2, tau2, q3, :]) * 2*np.pi*(self.e**2)*np.tanh(np.linalg.norm(self.k_space[q1]-self.k_space[q4])*self.d)/(self.epsilon*np.linalg.norm(self.k_space[q1]-self.k_space[q4]))

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

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D + self.D + (self.Nk,)))
