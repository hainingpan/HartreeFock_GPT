from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Hamiltonian for a two-layer (N and B) system with spin.

  Args:
    N_shell (int): Number of k-point shells.
    parameters (dict): Dictionary of model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int = 10, parameters: dict = {'tN': 1.0, 'tB': 1.0, 'tBN': 0.1, 'Delta': 0.0, 'UN': 1.0, 'UB': 1.0, 'VB': 0.1, 'VBN': 0.1}, filling_factor: float = 0.5):
    self.lattice = 'square'
    self.D = (2, 2)
    self.basis_order = {'0': 'layer', '1': 'spin'}
    # Order for each flavor:
    # 0: layer N, layer B
    # 1: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0  # Temperature
    self.k_space = generate_k_space(self.lattice, N_shell)

    # Model parameters with default values
    self.tN = parameters.get('tN', 1.0)  # Hopping in layer N
    self.tB = parameters.get('tB', 1.0)  # Hopping in layer B
    self.tBN = parameters.get('tBN', 0.1) # Hopping between layers N and B
    self.Delta = parameters.get('Delta', 0.0) # On-site potential difference between layers
    self.UN = parameters.get('UN', 1.0)  # Interaction strength in layer N
    self.UB = parameters.get('UB', 1.0)  # Interaction strength in layer B
    self.VB = parameters.get('VB', 0.1)  # Intra-layer interaction in layer B
    self.VBN = parameters.get('VBN', 0.1) # Inter-layer interaction between N and B


    return

  def generate_non_interacting(self) -> np.ndarray:
    """Generates the non-interacting part of the Hamiltonian."""
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + (N_k,), dtype=np.float32)

    # Kinetic terms for layer N and B, spin up and down
    H_nonint[0, 0, :] = self.tN * np.sum(np.exp(-1j*np.dot(self.k_space, n)) for n in get_neighbors(self.lattice)) + self.Delta # Layer N, spin up
    H_nonint[0, 1, :] = self.tN * np.sum(np.exp(-1j*np.dot(self.k_space, n)) for n in get_neighbors(self.lattice)) + self.Delta # Layer N, spin down
    H_nonint[1, 0, :] = self.tB * np.sum(np.exp(-1j*np.dot(self.k_space, n)) for n in get_neighbors(self.lattice)) # Layer B, spin up
    H_nonint[1, 1, :] = self.tB * np.sum(np.exp(-1j*np.dot(self.k_space, n)) for n in get_neighbors(self.lattice)) # Layer B, spin down


    # Interlayer hopping term
    H_nonint[0, 1, :] += self.tBN * np.sum(np.exp(-1j*np.dot(self.k_space, n)) for n in get_neighbors(self.lattice)) # N to B hopping
    H_nonint[1, 0, :] += self.tBN * np.sum(np.exp(1j*np.dot(self.k_space, n)) for n in get_neighbors(self.lattice)) # B to N hopping

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generates the interacting part of the Hamiltonian."""
    exp_val = self.expand(exp_val)
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + (N_k,), dtype=np.float32)

    # Calculate mean densities
    n_bu = np.mean(exp_val[1, 0, :])  # <b^+_{k, up} b_{k, up}>
    n_bd = np.mean(exp_val[1, 1, :])  # <b^+_{k, down} b_{k, down}>
    n_nu = np.mean(exp_val[0, 0, :])  # <a^+_{k, up} a_{k, up}>
    n_nd = np.mean(exp_val[0, 1, :])  # <a^+_{k, down} a_{k, down}>

    # Interaction terms
    H_int[0, 0, :] = self.UN * n_nd + self.VBN * (n_bu + n_bd) # Layer N, spin up
    H_int[0, 1, :] = self.UN * n_nu + self.VBN * (n_bu + n_bd) # Layer N, spin down
    H_int[1, 0, :] = self.UB * n_bd + 2 * self.VB * n_bd + self.VBN * (n_nu + n_nd) # Layer B, spin up
    H_int[1, 1, :] = self.UB * n_bu + 2 * self.VB * n_bu + self.VBN * (n_nu + n_nd) # Layer B, spin down

    return H_int

  # ... (rest of the class remains the same as in the example)

