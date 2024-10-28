from HF import *

import numpy as np
from typing import Any
from scipy.linalg import block_diag

class HartreeFockHamiltonian:
  """Hartree-Fock Hamiltonian for twisted bilayer graphene.

  Args:
    N_shell (int): Number of k-point shells to include.
    parameters (dict[str, Any]): Dictionary of model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=1, parameters: dict[str, Any]={'v_D': 1.0, 'omega_0': 0.1, 'omega_1': 0.05, 'theta': np.pi/6, 'V':1.0}, filling_factor: float=0.5): # LM Task: Updated parameters
    self.lattice = 'triangular'
    self.D = (2, 3)  # layer, reciprocal_lattice_vector
    self.basis_order = {'0': 'layer', '1': 'reciprocal_lattice_vector'} # LM Task: Updated Basis Order
    # 0: top layer, bottom layer
    # 1: q_0, q_1, q_2

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Default temperature
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.Nk = len(self.k_space)

    # Model parameters
    self.v_D = parameters.get('v_D', 1.0)  # Dirac velocity. Default to 1.0
    self.omega_0 = parameters.get('omega_0', 0.1)  # Interlayer tunneling parameter. Default to 0.1
    self.omega_1 = parameters.get('omega_1', 0.05)  # Interlayer tunneling parameter. Default to 0.05
    self.theta = parameters.get('theta', np.pi/6)  # Twist angle. Default to pi/6
    self.V = parameters.get('V', 1.0) # Interaction strength. Default to 1.0

    # Moiré lattice parameters
    self.a = 1.0  # Default graphene lattice constant
    self.aM = self.a / (2 * np.sin(self.theta / 2))  # Moiré lattice constant  # LM Task: Define aM

    # Reciprocal lattice vectors
    self.b1 = np.array([1/2, np.sqrt(3)/2]) * 4*np.pi / (np.sqrt(3) * self.aM)
    self.b2 = np.array([-1/2, np.sqrt(3)/2]) * 4*np.pi / (np.sqrt(3) * self.aM)
    self.q = [np.array([0.0, 0.0]), self.b1, self.b2]  # Momentum boosts


  def generate_non_interacting(self) -> np.ndarray:
      H_nonint = np.zeros((self.D[0], self.D[0], self.D[1], self.D[1], self.Nk), dtype=np.complex128)
      for k_idx, k in enumerate(self.k_space):
          # Dirac Hamiltonians for isolated rotated graphene layers
          k_bar_top = k - np.array([0,4*np.pi/(3*self.a)])
          k_bar_bottom = k - np.array([0,-4*np.pi/(3*self.a)])
          theta_k_bar_top = np.arctan2(k_bar_top[1], k_bar_top[0])
          theta_k_bar_bottom = np.arctan2(k_bar_bottom[1], k_bar_bottom[0])

          h_top = -self.v_D * np.linalg.norm(k_bar_top) * np.array([[0, np.exp(1j * (theta_k_bar_top - self.theta/2))],
                                                                [np.exp(-1j * (theta_k_bar_top - self.theta/2)), 0]])
          h_bottom = -self.v_D * np.linalg.norm(k_bar_bottom) * np.array([[0, np.exp(1j * (theta_k_bar_bottom + self.theta/2))],
                                                                      [np.exp(-1j * (theta_k_bar_bottom + self.theta/2)), 0]])


          # Tunneling Hamiltonian
          h_T = np.zeros((2,2),dtype=np.complex128)
          for j in range(3):
             Tj = self.omega_0 * np.eye(2) + self.omega_1 * np.cos(j*self.theta) * np.array([[0,1],[1,0]]) + self.omega_1 * np.sin(j*self.theta) * np.array([[0,-1j],[1j,0]])
             h_T += Tj # * np.exp(-1j * np.dot(self.q[j], r)) where r needs to be defined on a triangular lattice

          # Construct non-interacting Hamiltonian
          H_nonint[:, :, :, :, k_idx] = block_diag(h_top, h_bottom)
          H_nonint[0,1,:,:,k_idx] = h_T
          H_nonint[1,0,:,:,k_idx] = h_T.conj().T
          
      return H_nonint.reshape(np.prod(self.D),np.prod(self.D), self.Nk)



  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    exp_val = self.expand(exp_val) # 2, 3, 2, 3, N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128)

    # Calculate delta_rho (Placeholder: replace with actual calculation)
    delta_rho = exp_val - 0 # Placeholder for rho_iso

    # Placeholder for Sigma^H and Sigma^F (replace with actual calculation based on delta_rho)
    Sigma_H = self.V * delta_rho # Placeholder. Needs to be function of delta_rho.
    Sigma_F = self.V * delta_rho # Placeholder. Needs to be a function of delta_rho.

    H_int[0, 0, :, :, :] = Sigma_H + Sigma_F
    H_int[0, 1, :, :, :] = Sigma_F
    H_int[1, 0, :, :, :] = Sigma_F
    H_int[1, 1, :, :, :] = Sigma_H + Sigma_F

    return H_int.reshape(np.prod(self.D),np.prod(self.D), self.Nk)


  # ... (rest of the class remains the same, with updated D and basis_order)

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D[0], self.D[1], self.D[0], self.D[1], self.Nk)) # updated



