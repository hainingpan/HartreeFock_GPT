from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Continuum model Hamiltonian for twisted bilayer graphene with Hartree-Fock mean-field interaction.

  Args:
      N_shell (int): Number of k-point shells in the triangular lattice.
      parameters (dict): Dictionary of model parameters. Default values provided.
      filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict[str, Any]={'m_star': 0.057, 'V': 0.0265, 'w': 0.112, 'phi': np.pi/3, 'Delta_D': 0.0, 'e': 1.0, 'epsilon': 3.0, 'd_gate': 1.0, 'd': 1.0, 'h_bar': 1.0}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (2, 2)  # (layer, spin)
    self.basis_order = {'0': 'layer', '1': 'spin'}
    # Order for each flavor:
    # 0: layer: bottom, top
    # 1: spin: up, down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Default temperature
    self.k_space = generate_k_space(self.lattice, N_shell)
    self.Nk = len(self.k_space)

    # Model parameters with default values
    self.m_star = parameters.get('m_star', 0.057)       # Effective mass
    self.V = parameters.get('V', 0.0265)              # Moiré potential strength
    self.w = parameters.get('w', 0.112)              # Interlayer tunneling strength
    self.phi = parameters.get('phi', np.pi / 3)        # Moiré potential phase
    self.Delta_D = parameters.get('Delta_D', 0.0)      # Displacement field
    self.e = parameters.get('e', 1.0)                 # Elementary charge
    self.epsilon = parameters.get('epsilon', 3.0)      # Dielectric constant
    self.d_gate = parameters.get('d_gate', 1.0)        # Gate distance
    self.d = parameters.get('d', 1.0)                 # Layer separation
    self.h_bar = parameters.get('h_bar', 1.0)       # Reduced Planck constant
    self.aM = get_A() # LM Task: Define the lattice constant, used for the area.
    return

  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""

    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + (N_k,), dtype=np.float32) # 2, 2, N_k

    # Kinetic terms + Moiré potentials + Displacement field
    kappa_plus = np.array([0, 0]) # Assuming kappa_plus = [0, 0] since it is not explicitly given
    kappa_minus = np.array([0, 0]) # Assuming kappa_minus = [0, 0] since it is not explicitly given
    G1 = np.array([0, 0]) # Assuming G1 = [0, 0] since it is not explicitly given
    G2 = np.array([0, 0]) # Assuming G2 = [0, 0] since it is not explicitly given
    G3 = np.array([0, 0]) # Assuming G3 = [0, 0] since it is not explicitly given
    r = np.array([0, 0]) # Assuming r = [0, 0] since it is not explicitly given, although ideally this should be a sum over r in principle

    for k_idx, k in enumerate(self.k_space):
        H_nonint[0, 0, k_idx] = - (self.h_bar**2) * np.dot(k - kappa_plus, k - kappa_plus) / (2 * self.m_star) + 2 * self.V * np.sum([np.cos(np.dot(Gi, r) + self.phi) for Gi in [G1, G2, G3]]) + 0.5 * self.Delta_D
        H_nonint[1, 1, k_idx] = - (self.h_bar**2) * np.dot(k - kappa_minus, k - kappa_minus) / (2 * self.m_star) + 2 * self.V * np.sum([np.cos(np.dot(Gi, r) - self.phi) for Gi in [G1, G2, G3]]) - 0.5 * self.Delta_D
        Delta_T = self.w * (1 + np.exp(-1j * np.dot(G2, r)) + np.exp(-1j * np.dot(G3, r)))
        H_nonint[0, 1, k_idx] = Delta_T
        H_nonint[1, 0, k_idx] = np.conj(Delta_T)

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""
    exp_val = self.expand(exp_val)
    N_k = self.k_space.shape[0]
    H_int = np.zeros(self.D + (N_k,), dtype=np.float32)

    for l in range(2): # layer
        for tau in range(2): # spin
            for k_idx, k in enumerate(self.k_space):
                for lp in range(2):
                    for taup in range(2):
                        for kp_idx, kp in enumerate(self.k_space):
                            q = k - kp
                            V_llp = (self.e**2) / (2 * self.epsilon *  abs(q) ) * (np.tanh(self.d_gate * abs(q)) + (1 - int(l==lp)) * (np.exp(-self.d * abs(q)) - 1))
                            # Hartree term
                            H_int[lp, taup, kp_idx] += (1/self.aM) * V_llp * exp_val[l, tau, k_idx] # <c_{l tau k+q}^\dagger c_{l tau k}> c_{lp taup kp-q}^\dagger c_{lp taup kp}
                            # Fock term
                            H_int[l, tau, k_idx] -= (1/self.aM) * V_llp * exp_val[lp, taup, kp_idx]  # - <c_{l tau k+q}^\dagger c_{lp taup kp}> c_{lp taup kp-q}^\dagger c_{l tau k}

    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool = True) -> np.ndarray:
    """Generates the total Hamiltonian."""
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if flatten:
        return self.flatten(H_total)
    else:
        return H_total

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D), np.prod(self.D), self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape(self.D + (self.Nk,))

