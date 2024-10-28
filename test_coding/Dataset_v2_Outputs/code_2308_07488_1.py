from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """Continuum Model Hamiltonian with Coulomb Interaction.

  Args:
      N_shell (int): Number of k-point shells for triangular lattice.
      parameters (dict[str, Any]): Model parameters. Includes:
          'V': Moiré potential strength.
          'phi': Moiré potential phase.
          'w': Interlayer coupling strength.
          'm_star': Effective mass.
          'hbar': Reduced Planck constant.
          'e': Elementary charge.
          'epsilon': Dielectric constant.
          'epsilon_0': Vacuum permittivity.
          'd_gate': Gate distance.
          'd': Layer separation.
      filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell:int=10, parameters: dict[str, Any]={'V':1.0, 'phi':0.0, 'w':1.0, 'm_star':1.0, 'hbar':1.0, 'e':1.0, 'epsilon':1.0, 'epsilon_0':1.0, 'd_gate':1.0, 'd':1.0}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'triangular'
    self.D = (2, 2) # level, spin
    self.basis_order = {'0': 'level', '1': 'spin'}
    # 0: level. Order: bottom, top
    # 1: spin. Order: up, down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0 # Assuming zero temperature.
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.Nk = self.k_space.shape[0]
    self.N_shell = N_shell

    # Model parameters
    self.V = parameters['V'] # Moiré potential strength
    self.phi = parameters['phi'] # Moiré potential phase
    self.w = parameters['w'] # Interlayer coupling strength
    self.m_star = parameters['m_star'] # Effective mass
    self.hbar = parameters['hbar'] # Reduced Planck constant
    self.e = parameters['e'] # Elementary charge
    self.epsilon = parameters['epsilon'] # Dielectric constant
    self.epsilon_0 = parameters['epsilon_0'] # Vacuum permittivity
    self.d_gate = parameters['d_gate'] # Gate distance
    self.d = parameters['d'] # Layer separation
    self.aM = 4 * np.pi / (np.sqrt(3)*(2*(self.N_shell) + 1)) # Area of the moiré unit cell.

    return


  def generate_non_interacting(self) -> np.ndarray:
    H_nonint = np.zeros(self.D + (self.Nk,), dtype=np.float32)

    kappa_plus = np.array([4*np.pi/(3*self.aM), 0])  # Assuming appropriate definition for kappa_plus
    kappa_minus = np.array([-4*np.pi/(3*self.aM), 0]) # Assuming appropriate definition for kappa_minus

    # Kinetic terms
    H_nonint[0, 0, :] = -((self.hbar**2) / (2 * self.m_star)) * np.sum((self.k_space - kappa_plus)**2, axis=1) # bottom level, up spin
    H_nonint[1, 1, :] = -((self.hbar**2) / (2 * self.m_star)) * np.sum((self.k_space - kappa_minus)**2, axis=1) # top level, up spin
    # Moiré potentials and interlayer coupling (assuming r is a function of k)
    Delta_b = 2*self.V*(np.cos(self.k_space[:,0]+ self.phi) + np.cos((1/2) * self.k_space[:,0] - (np.sqrt(3)/2) *self.k_space[:,1]+self.phi)+np.cos((1/2) * self.k_space[:,0] + (np.sqrt(3)/2) *self.k_space[:,1]+self.phi))
    Delta_t = 2*self.V*(np.cos(self.k_space[:,0]-self.phi) + np.cos((1/2) * self.k_space[:,0] - (np.sqrt(3)/2) *self.k_space[:,1]-self.phi)+np.cos((1/2) * self.k_space[:,0] + (np.sqrt(3)/2) *self.k_space[:,1]-self.phi))
    Delta_T = self.w*(1 + np.exp(-1j * self.k_space[:,0]) + np.exp(-1j * ((1/2)*self.k_space[:,0] + (np.sqrt(3)/2)*self.k_space[:,1])))
    H_nonint[0, 0, :] += Delta_b  # bottom level, up spin
    H_nonint[1, 1, :] += Delta_t   # top level, up spin

    H_nonint[0, 1, :] = Delta_T # bottom to top, up spin
    H_nonint[1, 0, :] = np.conj(Delta_T) # top to bottom, up spin
    
    Delta_D = 0.5*(Delta_b - Delta_t) # Assuming Delta_D can be defined like this.
    H_nonint[0, 0, :] += Delta_D # bottom level, up spin
    H_nonint[1, 1, :] -= Delta_D # top level, up spin
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + (self.Nk,), dtype=np.float32)


    for l in range(2):
        for tau in range(2): # Assuming tau represents spin: 0 for up, 1 for down
            for lp in range(2):
                for taup in range(2):
                    for k in range(N_k):
                        for kp in range(N_k):
                            for q_idx in range(N_k): # Iterate over all q values in k_space
                                q = self.k_space[q_idx]
                                #k_plus_q = self.k_space + q
                                k_plus_q_idx = (k + q_idx) % N_k
                                #kp_minus_q = self.k_space - q
                                kp_minus_q_idx = (kp - q_idx + N_k)% N_k

                                V_llp = (self.e**2)/(2*self.epsilon*self.epsilon_0 * np.linalg.norm(q)) * (np.tanh(self.d_gate*np.linalg.norm(q)) + (1-int(l==lp))* (np.exp(-self.d * np.linalg.norm(q))-1))
                                H_int[lp, taup, kp] += (1/self.aM)* V_llp * (exp_val[l, tau, k_plus_q_idx, l, tau, k] * int(kp == kp) - exp_val[l, tau, k_plus_q_idx, lp, taup, kp] * int(k == kp_minus_q_idx))
    return H_int



  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
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
    return exp_val.reshape((self.D + (self.Nk,)))

