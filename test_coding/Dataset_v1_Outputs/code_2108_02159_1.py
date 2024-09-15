from HF import *
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'hbar': 1.0, 'm_star': 1.0, 'V_M': 1.0, 'phi': 0.0}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'triangular'
    self.D = (6, 2) # Number of flavors identified: reciprocal lattice vectors, spin
    self.basis_order = {'0': 'reciprocal_lattice_vector', '1': 'spin'}
    # Order for each flavor:
    # 0: b1, b2, b3, b4, b5, b6
    # 1: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.hbar = parameters['hbar'] # Reduced Planck constant
    self.m_star = parameters['m_star'] # Effective mass
    self.V_M = parameters['V_M'] # Moire modulation strength
    self.phi = parameters['phi'] # Moire modulation shape parameter
    self.aM = 1.0 # Lattice constant # LM Task: Define the lattice constant, used for the area.
    

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32) # 6, 2, 6, 2, N_k

    # Kinetic energy
    for i in range(6): # reciprocal lattice vectors
      for j in range(2): # spin
        H_nonint[i, j, i, j, :] = -((self.hbar**2)/(2*self.m_star)) * np.sum((self.k_space + self.reciprocal_lattice_vectors[i])**2, axis=1)
    
    # Moire potential
    for i in range(6): #reciprocal lattice vectors
      for j in range(2): # spin
        H_nonint[i, j, (i+1)%6, j, :] += self.V_M*np.exp((-1)**(i)*1j*self.phi)
        H_nonint[(i+1)%6, j, i, j, :] += np.conjugate(self.V_M*np.exp((-1)**(i)*1j*self.phi)) # Ensuring hermiticity
    
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) #6, 2, 6, 2, N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    
    # Calculate the mean densities for spin up and spin down
    # Hartree Term
    density_hartree = np.sum(exp_val, axis = 3) # Summing over the second reciprocal lattice vector index
    density_hartree = np.sum(density_hartree, axis=2) # Summing over k

    for i in range(6):
      for j in range(2):
        for ip in range(6):
          for jp in range(2):
            H_int[i, j, ip, jp, :] += (1/self.aM) * self.V_interacting[i, j, ip, jp] * density_hartree[ip, jp]
    
    # Fock Term
    for i in range(6):
      for j in range(2):
        for ip in range(6):
          for jp in range(2):
            for k in range(N_k):
              for kp in range(N_k):
                for bpp in range(6):
                  H_int[i, j, ip, jp, k] -= (1/self.aM) * self.V_interacting[i, j, ip, jp] * exp_val[i, j, ip, jp, kp] # Assuming V_interacting is defined elsewhere # LM Task: Define the interaction.
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
