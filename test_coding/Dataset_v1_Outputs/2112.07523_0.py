import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  The Hartree-Fock Hamiltonian for a BHZ model.

  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'me':1.0, 'mh':1.0, 'Eg':1.0, 'A':1.0, 'Q':(0.0, 0.0), 'V':1.0}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'square'  
    self.D = (2, 2) # (spin, band)
    self.basis_order = {'0': 'spin', '1':'band'}
    # Order for each flavor:
    # 0: spin up, spin down
    # 1: conduction band, valence band

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.hbar = 1.054e-34 # Reduced Planck constant
    self.me = parameters['me'] # Electron effective mass
    self.mh = parameters['mh'] # Hole effective mass
    self.Eg = parameters['Eg'] # Band gap energy
    self.A = parameters['A'] # Spin-orbit coupling strength
    self.Q = parameters['Q'] # Momentum transfer
    self.V = parameters['V'] # Interaction strength

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros((self.D[0], self.D[1], self.D[0], self.D[1], N_k), dtype=np.complex128) #spin1, band1, spin2, band2, k
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    H_nonint[0, 0, 0, 0, :] =  (self.hbar**2/(2*self.me)) * ((self.k_space[:, 0]-self.Q[0]/2)**2 + (self.k_space[:, 1]-self.Q[1]/2)**2) + self.Eg/2
    H_nonint[0, 1, 0, 1, :] = -(self.hbar**2/(2*self.mh)) * ((self.k_space[:, 0]+self.Q[0]/2)**2 + (self.k_space[:, 1]+self.Q[1]/2)**2) - self.Eg/2
    H_nonint[0, 0, 0, 1, :] = self.A * (self.k_space[:, 0] + 1j*self.k_space[:, 1])
    H_nonint[0, 1, 0, 0, :] = self.A * (self.k_space[:, 0] - 1j*self.k_space[:, 1])

    H_nonint[1, 0, 1, 0, :] =  (self.hbar**2/(2*self.me)) * ((self.k_space[:, 0]-self.Q[0]/2)**2 + (self.k_space[:, 1]-self.Q[1]/2)**2) + self.Eg/2
    H_nonint[1, 1, 1, 1, :] = -(self.hbar**2/(2*self.mh)) * ((self.k_space[:, 0]+self.Q[0]/2)**2 + (self.k_space[:, 1]+self.Q[1]/2)**2) - self.Eg/2
    H_nonint[1, 0, 1, 1, :] = -self.A * (self.k_space[:, 0] - 1j*self.k_space[:, 1])
    H_nonint[1, 1, 1, 0, :] = -self.A * (self.k_space[:, 0] + 1j*self.k_space[:, 1])
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
    H_int = np.zeros((self.D[0], self.D[1],  self.D[0], self.D[1], N_k), dtype=np.complex128) #spin1, band1, spin2, band2, k

    # Calculate the mean densities for spin up and spin down
    # Hartree term
    for b1 in range(self.D[0]):
        for s1 in range(self.D[1]):
            for n1 in range(self.D[2]):
                for b2 in range(self.D[0]):
                    for s2 in range(self.D[1]):
                        for n2 in range(self.D[2]):
                            for k in range(N_k):
                                for np in range(self.D[2]):
                                    H_int[b1, s1, n1, b1, s1, n2, k] += 1/(N_k) * self.V * exp_val[b2, s2, np, b2, s2, np, k]
                            
    #Fock term
    #                for b1 in range(self.D[0]):
    #    for s1 in range(self.D[1]):
    #        for n1 in range(self.D[2]):
    #            for b2 in range(self.D[0]):
    #                for s2 in range(self.D[1]):
    #                    for n2 in range(self.D[2]):
    #                        for k in range(N_k):
    #                            for np in range(self.D[2]):
    #                                H_int[b2, s2, n1, b1, s1, n2, k] += 1/(N_k) * self.V * exp_val[b1, s1, np, b2, s2, np, k]

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
