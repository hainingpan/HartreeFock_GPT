import numpy as np
from typing import Any
from scipy.constants import hbar, e, epsilon_0

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict[str, Any]={'V':1.0, 'phi': 0.0, 'w': 0.1, 'm_eff': 1.0, 'Delta_D': 0.1, 'd_gate': 1.0, 'd': 1.0}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (2, 2) # (layer, spin)
    self.basis_order = {'0': 'layer', '1': 'spin'}
    # Order for each flavor:
    # 0: layer: top, bottom
    # 1: spin: up, down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)
    self.Nk = self.k_space.shape[0]
    # N_kx = 2*(N_shell+1) for a square lattice

    # All other parameters such as interaction strengths
    self.V = parameters['V'] # Moire potential strength
    self.phi = parameters['phi'] # Moire potential phase
    self.w = parameters['w'] # Interlayer coupling strength
    self.m_eff = parameters['m_eff'] # Effective mass
    self.Delta_D = parameters['Delta_D'] # Layer splitting
    self.d_gate = parameters['d_gate'] # Gate distance
    self.d = parameters['d'] # Interlayer distance
    self.aM = get_A() # LM Task: Define the lattice constant, used for the area.
    # Any other problem specific parameters.
    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128) # layer1, spin1, layer2, spin2, k
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    # Assuming kappa_+ = - kappa_- for simplicity, to be modified based on actual definition in the problem
    H_nonint[0, 0, 0, 0, :] = -hbar**2 * ((self.k_space[:, 0] + self.kappa)**2 + self.k_space[:, 1]**2)/(2*self.m_eff) + self.Delta_b # top layer, spin up
    H_nonint[1, 0, 1, 0, :] = -hbar**2 * ((self.k_space[:, 0] - self.kappa)**2 + self.k_space[:, 1]**2)/(2*self.m_eff) + self.Delta_t # bottom layer, spin up
    # Assuming spin up and spin down have the same non-interacting Hamiltonian, to be modified if that's not the case
    H_nonint[0, 1, 0, 1, :] = H_nonint[0, 0, 0, 0, :] # top layer, spin down
    H_nonint[1, 1, 1, 1, :] = H_nonint[1, 0, 1, 0, :] # bottom layer, spin down

    H_nonint[0, 0, 0, 0, :] += 1/2 * self.Delta_D # top layer, spin up
    H_nonint[1, 0, 1, 0, :] -= 1/2 * self.Delta_D # bottom layer, spin up
    H_nonint[0, 1, 0, 1, :] += 1/2 * self.Delta_D # top layer, spin up
    H_nonint[1, 1, 1, 1, :] -= 1/2 * self.Delta_D # bottom layer, spin up

    H_nonint[0, 0, 1, 0, :] += self.Delta_T # top layer, spin up
    H_nonint[1, 0, 0, 0, :] += np.conj(self.Delta_T)  # bottom layer, spin up
    H_nonint[0, 1, 1, 1, :] += self.Delta_T # top layer, spin up
    H_nonint[1, 1, 0, 1, :] += np.conj(self.Delta_T)  # bottom layer, spin up
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) # 2, 2, 2, 2, N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128) # layer1, spin1, layer2, spin2, k

    # Calculate the mean densities for spin up and spin down
    # Hartree Terms
    for l in range(2):
      for s in range(2):
        for lp in range(2):
          for sp in range(2):
            H_int[lp, sp, lp, sp, :] += 1/self.aM * self.V_ll(q=0) * np.mean(exp_val[l, s, l, s, :]) # <c_{k',up}^\dagger c_{k',up}>
    #Fock terms
    for l in range(2):
      for s in range(2):
        for lp in range(2):
          for sp in range(2):
            H_int[lp, sp, l, s, :] -= 1/self.aM * self.V_ll(q=0) * (exp_val[l, s, lp, sp, :]) # <c_{k',up}^\dagger c_{k',up}>

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

  def V_ll(self, q):
    return e**2 / (2 * epsilon_0 * epsilon_r(q)) * (np.tanh(d_gate * q) + (1 - delta(l, lp)) * (np.exp(- d * q) - 1))
