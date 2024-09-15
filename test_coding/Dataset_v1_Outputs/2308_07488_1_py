import numpy as np
from typing import Any
from scipy.constants import  hbar, e, epsilon_0

class HartreeFockHamiltonian:
  """
  The Hartree-Fock Hamiltonian for the TMD Heterobilayer problem.
  """
  def __init__(self, N_shell:int=1, parameters:dict[str, Any]={'V':1.0, 'w':0.2, 'm_star': 0.5, 'd_gate': 10, 'd':1.0, 'phi':0.0}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'triangular'
    self.D = (2, 2) # layer, spin
    # Order of tuples follows order of creation/annihilation operators.
    self.basis_order = {'0': 'layer', '1':'spin'}
    # Order for each flavor:
    # 0: bottom, top
    # 1: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0 # Assume zero temperature
    self.k_space = generate_k_space(lattice=self.lattice, N_shell)
    N_k = self.k_space.shape[0]

    # All other parameters
    self.V = parameters['V'] # Moire potential strength
    self.w = parameters['w'] # Interlayer moire potential strength
    self.m_star = parameters['m_star'] # Effective mass
    self.d_gate = parameters['d_gate'] # Gate distance
    self.d = parameters['d'] # Interlayer distance
    self.phi = parameters['phi'] # Moire potential phase

    # Calculate reciprocal lattice vectors
    a = 4*np.pi/(np.sqrt(3)*a_moire)
    self.G1 = 2*np.pi/a*np.array([np.sqrt(3), 1.0])
    self.G2 = 2*np.pi/a*np.array([np.sqrt(3), -1.0])
    self.G3 = 2*np.pi/a*np.array([0.0, -2.0])

    self.kappa_plus = (self.G1 + self.G2)/3
    self.kappa_minus = (2*self.G2 + self.G1)/3

    self.aM = np.sqrt(3)/2*a**2 # Area of the Moire unit cell

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.
      H[l1, s1, l2, s2, k] = < l1, s1, k|H0|l2, s2, k>

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    # Kinetic energy for bottom layer, spin up and down.
    H_nonint[0, 0, 0, 0, :] = -hbar**2/(2*self.m_star) * ((self.k_space[:, 0] - self.kappa_plus[0])**2 + (self.k_space[:, 1] - self.kappa_plus[1])**2)
    H_nonint[0, 1, 0, 1, :] = -hbar**2/(2*self.m_star) * ((self.k_space[:, 0] - self.kappa_plus[0])**2 + (self.k_space[:, 1] - self.kappa_plus[1])**2)

    # Kinetic energy for top layer, spin up and down.
    H_nonint[1, 0, 1, 0, :] =  -hbar**2/(2*self.m_star) * ((self.k_space[:, 0] - self.kappa_minus[0])**2 + (self.k_space[:, 1] - self.kappa_minus[1])**2)
    H_nonint[1, 1, 1, 1, :] =  -hbar**2/(2*self.m_star) * ((self.k_space[:, 0] - self.kappa_minus[0])**2 + (self.k_space[:, 1] - self.kappa_minus[1])**2)

    # Moire potential for bottom and top layers.
    H_nonint[0, 0, 0, 0, :] += 2*self.V*(np.cos(self.G1[0] * self.k_space[:, 0] + self.G1[1] * self.k_space[:, 1] + self.phi) \
                         + np.cos(self.G2[0] * self.k_space[:, 0] + self.G2[1] * self.k_space[:, 1] + self.phi)  \
                         + np.cos(self.G3[0] * self.k_space[:, 0] + self.G3[1] * self.k_space[:, 1] + self.phi))
    
    H_nonint[1, 1, 1, 1, :] += 2*self.V*(np.cos(self.G1[0] * self.k_space[:, 0] + self.G1[1] * self.k_space[:, 1] - self.phi) \
                         + np.cos(self.G2[0] * self.k_space[:, 0] + self.G2[1] * self.k_space[:, 1] - self.phi)  \
                         + np.cos(self.G3[0] * self.k_space[:, 0] + self.G3[1] * self.k_space[:, 1] - self.phi))

    # Interlayer Moire potential
    H_nonint[0, 0, 1, 0, :] = self.w*(1 + np.exp(-1j * (self.G2[0] * self.k_space[:, 0] + self.G2[1] * self.k_space[:, 1])) + np.exp(-1j * (self.G3[0] * self.k_space[:, 0] + self.G3[1] * self.k_space[:, 1])))
    H_nonint[0, 1, 1, 1, :] = self.w*(1 + np.exp(-1j * (self.G2[0] * self.k_space[:, 0] + self.G2[1] * self.k_space[:, 1])) + np.exp(-1j * (self.G3[0] * self.k_space[:, 0] + self.G3[1] * self.k_space[:, 1])))
    H_nonint[1, 0, 0, 0, :] = np.conj(H_nonint[0, 0, 1, 0, :])
    H_nonint[1, 1, 0, 1, :] = np.conj(H_nonint[0, 1, 1, 1, :])

    # Exchange field.
    H_nonint[0, 0, 0, 0, :] += 1/2 * Delta_D
    H_nonint[0, 1, 0, 1, :] += 1/2 * Delta_D
    H_nonint[1, 0, 1, 0, :] += - 1/2 * Delta_D
    H_nonint[1, 1, 1, 1, :] += - 1/2 * Delta_D

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
    
    for l1 in range(2):
        for l2 in range(2):
            for s1 in range(2):
                for s2 in range(2):
                    for k in range(N_k):
                        for q in range(N_k):
                            #Hartree Terms
                            H_int[l1, s1, l1, s1, k] += 1/self.aM * self.V_int(q=q) * exp_val[l2, s2, l2, s2, k]
                            #Fock Terms
                            H_int[l1, s1, l1, s1, k] -= 1/self.aM * self.V_int(q=k-q) * exp_val[l1, s1, l2, s2, q]
    return H_int

  def V_int(self, q:int):
      return (e**2 /(2 * epsilon_0 * epsilon_r * np.sqrt((self.k_space[q, 0] - self.k_space[0, 0])**2 + (self.k_space[q, 1] - self.k_space[0, 1])**2))) \
                *(np.tanh(self.d_gate * np.sqrt((self.k_space[q, 0] - self.k_space[0, 0])**2 + (self.k_space[q, 1] - self.k_space[0, 1])**2) ) \
                + (np.exp(- self.d * np.sqrt((self.k_space[q, 0] - self.k_space[0, 0])**2 + (self.k_space[q, 1] - self.k_space[0, 1])**2) ) - 1))

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
