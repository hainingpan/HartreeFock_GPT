import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_kx (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_kx: int=10, parameters: dict={'A':1.0, 'Eg':1.0, 'me':1.0, 'mh':1.0, 'hbar':1.0}, filling_factor: float=0.5): # LM Task: Modify parameters
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2, 2) # Number of flavors identified.
    self.basis_order = {'0': 'spin', '1': 'band'} # LM Task: Define the basis order.
    # Order for each flavor:
    # 0: spin up, spin down
    # 1: conduction, valence

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.k_space = generate_k_space(symmetry=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.A = parameters['A'] # Rashba spin-orbit coupling strength
    self.Eg = parameters['Eg'] # Band gap
    self.me = parameters['me'] # Electron effective mass
    self.mh = parameters['mh'] # Hole effective mass
    self.hbar = parameters['hbar'] # Reduced Planck constant

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    #k = self.k_space
    #kx, ky = k[:, 0], k[:, 1]
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32) # (2, 2, 2, 2, N_k)
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    H_nonint[0, 0, 0, 0, 0, 0, :] = self.hbar**2 / (2 * self.me) * (self.k_space[:, 0]- self.Q/2)**2 + self.Eg / 2 #spin up, conduction, spin up, conduction
    H_nonint[0, 0, 1, 0, 0, 0, :] = self.A * (self.k_space[:, 0] + 1j * self.k_space[:, 1]) #spin up, valence, spin up, conduction
    H_nonint[0, 0, 0, 0, 0, 1, :] = self.A * (self.k_space[:, 0] - 1j * self.k_space[:, 1]) #spin up, conduction, spin up, valence
    H_nonint[0, 0, 1, 0, 0, 1, :] = - self.hbar**2 / (2 * self.mh) * (self.k_space[:, 0] + self.Q/2)**2 - self.Eg / 2 #spin up, valence, spin up, valence
    H_nonint[0, 1, 0, 0, 1, 0, :] = self.hbar**2 / (2 * self.me) * (self.k_space[:, 0] - self.Q/2)**2 + self.Eg / 2 #spin down, conduction, spin down, conduction
    H_nonint[0, 1, 1, 0, 1, 0, :] = - self.A * (self.k_space[:, 0] - 1j * self.k_space[:, 1]) #spin down, valence, spin down, conduction
    H_nonint[0, 1, 0, 0, 1, 1, :] = - self.A * (self.k_space[:, 0] + 1j * self.k_space[:, 1]) #spin down, conduction, spin down, valence
    H_nonint[0, 1, 1, 0, 1, 1, :] = - self.hbar**2 / (2 * self.mh) * (self.k_space[:, 0] + self.Q/2)**2 - self.Eg / 2 #spin down, valence, spin down, valence
    H_nonint[1, 0, 0, 1, 0, 0, :] = self.hbar**2 / (2 * self.me) * (self.k_space[:, 0] - self.Q/2)**2 + self.Eg / 2 #spin up, conduction, spin up, conduction
    H_nonint[1, 0, 1, 1, 0, 0, :] = self.A * (self.k_space[:, 0] + 1j * self.k_space[:, 1]) #spin up, valence, spin up, conduction
    H_nonint[1, 0, 0, 1, 0, 1, :] = self.A * (self.k_space[:, 0] - 1j * self.k_space[:, 1]) #spin up, conduction, spin up, valence
    H_nonint[1, 0, 1, 1, 0, 1, :] = - self.hbar**2 / (2 * self.mh) * (self.k_space[:, 0] + self.Q/2)**2 - self.Eg / 2 #spin up, valence, spin up, valence
    H_nonint[1, 1, 0, 1, 1, 0, :] = self.hbar**2 / (2 * self.me) * (self.k_space[:, 0] - self.Q/2)**2 + self.Eg / 2 #spin down, conduction, spin down, conduction
    H_nonint[1, 1, 1, 1, 1, 0, :] = - self.A * (self.k_space[:, 0] - 1j * self.k_space[:, 1]) #spin down, valence, spin down, conduction
    H_nonint[1, 1, 0, 1, 1, 1, :] = - self.A * (self.k_space[:, 0] + 1j * self.k_space[:, 1]) #spin down, conduction, spin down, valence
    H_nonint[1, 1, 1, 1, 1, 1, :] = - self.hbar**2 / (2 * self.mh) * (self.k_space[:, 0] + self.Q/2)**2 - self.Eg / 2 #spin down, valence, spin down, valence
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
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    # Calculate the mean densities for spin up and spin down
    #n_up = np.mean(exp_val[0, 0, :]) # <c_{k',up}^\dagger c_{k',up}>
    #n_down = np.mean(exp_val[1, 1, :]) # <c_{k',down}^\dagger c_{k',down}>
    for b1 in range(2):
      for s1 in range(2):
        for n1 in range(2):
          for b2 in range(2):
            for s2 in range(2):
              for n2 in range(2):
                for k in range(N_k):
                  H_int[b1, s1, n1, b2, s2, n2, k] = self.V * exp_val[b1, s1, n1, b2, s2, n2, k] # Assuming V is a parameter, adjust accordingly
    
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
