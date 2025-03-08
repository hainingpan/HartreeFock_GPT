import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a system with p_x, p_y, and d orbitals.
  
  Args:
    N_shell (int): Number of shells in the first Brillouin zone.
    parameters (dict): Dictionary containing model parameters.
      - t_pd (float): Hopping parameter between p and d orbitals.
      - t_pp (float): Hopping parameter between p orbitals.
      - Delta (float): Energy difference between p and d orbitals.
      - U_p (float): Coulomb repulsion on p orbitals.
      - V_pp (float): Inter-site Coulomb repulsion between p orbitals.
      - U_d (float): Coulomb repulsion on d orbital.
      - V_pd (float): Inter-site Coulomb repulsion between p and d orbitals.
      - T (float): Temperature (default: 0).
      - a (float): Lattice constant (default: 1.0).
    filling_factor (float): Filling factor (default: 0.5).
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_pd':1.0, 't_pp':1.0, 'Delta':0.0, 'U_p':0.0, 'V_pp':0.0, 'U_d':0.0, 'V_pd':0.0, 'T':0, 'a':1.0}, filling_factor: float=0.5):
    self.lattice = 'square'   # Lattice type
    self.D = (2, 3)           # (spin, orbital)
    self.basis_order = {'0': 'spin', '1': 'orbital'}
    # Order for each flavor:
    # spin: up (0), down (1)
    # orbital: p_x (0), p_y (1), d (2)

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0)      # Temperature
    self.a = parameters.get('a', 1.0)    # Lattice constant
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

    # Hopping parameters
    self.t_pd = parameters.get('t_pd', 1.0)  # Hopping between p and d orbitals
    self.t_pp = parameters.get('t_pp', 1.0)  # Hopping between p orbitals
    
    # Energy parameters
    self.Delta = parameters.get('Delta', 0.0)  # Energy difference between p and d orbitals
    
    # Interaction parameters
    self.U_p = parameters.get('U_p', 0.0)    # Coulomb repulsion on p orbitals
    self.V_pp = parameters.get('V_pp', 0.0)  # Inter-site Coulomb repulsion between p orbitals
    self.U_d = parameters.get('U_d', 0.0)    # Coulomb repulsion on d orbital
    self.V_pd = parameters.get('V_pd', 0.0)  # Inter-site Coulomb repulsion between p and d orbitals

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.

    Returns:
      np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Hopping terms (off-diagonal in orbital space)
    for s in range(2):  # Iterate over spin
      # Hopping between p_x and d orbitals: γ1(kx) term
      H_nonint[s, 0, s, 2, :] = -2 * self.t_pd * np.cos(self.k_space[:, 0] / 2)
      H_nonint[s, 2, s, 0, :] = -2 * self.t_pd * np.cos(self.k_space[:, 0] / 2)
      
      # Hopping between p_y and d orbitals: γ1(ky) term
      H_nonint[s, 1, s, 2, :] = -2 * self.t_pd * np.cos(self.k_space[:, 1] / 2)
      H_nonint[s, 2, s, 1, :] = -2 * self.t_pd * np.cos(self.k_space[:, 1] / 2)
      
      # Hopping between p_x and p_y orbitals: γ2(k) term
      H_nonint[s, 0, s, 1, :] = -4 * self.t_pp * np.cos(self.k_space[:, 0] / 2) * np.cos(self.k_space[:, 1] / 2)
      H_nonint[s, 1, s, 0, :] = -4 * self.t_pp * np.cos(self.k_space[:, 0] / 2) * np.cos(self.k_space[:, 1] / 2)
    
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = unflatten(exp_val, self.D, self.N_k)
    H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)

    # Calculate expectation values for each orbital and spin
    n_px_up = np.real(exp_val[0, 0, 0, 0, :].mean())
    n_px_down = np.real(exp_val[1, 0, 1, 0, :].mean())
    n_py_up = np.real(exp_val[0, 1, 0, 1, :].mean())
    n_py_down = np.real(exp_val[1, 1, 1, 1, :].mean())
    n_d_up = np.real(exp_val[0, 2, 0, 2, :].mean())
    n_d_down = np.real(exp_val[1, 2, 1, 2, :].mean())
    
    # Calculate derived quantities
    n_p = n_px_up + n_px_down + n_py_up + n_py_down  # Total density on p orbitals
    eta = (n_px_up + n_px_down) - (n_py_up + n_py_down)  # Nematic order parameter
    n = self.nu  # Total density of holes, equal to the filling factor
    
    # Calculate effective interaction parameters using equations (12)-(14)
    U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
    V_pp_tilde = 8 * self.V_pp - self.U_p
    U_d_tilde = self.U_d - 4 * self.V_pd
    
    # Calculate chemical potential
    mu = 2 * self.V_pd * n - self.V_pd * n**2
    
    # Diagonal elements of the matrix (interacting terms)
    for s in range(2):  # Iterate over spin
      # p_x orbital: ξx term
      H_int[s, 0, s, 0, :] = self.Delta + U_p_tilde * n_p / 4 - V_pp_tilde * eta / 4 - mu
      
      # p_y orbital: ξy term
      H_int[s, 1, s, 1, :] = self.Delta + U_p_tilde * n_p / 4 + V_pp_tilde * eta / 4 - mu
      
      # d orbital: ξd term
      H_int[s, 2, s, 2, :] = U_d_tilde * (n - n_p) / 2 - mu
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """
    Generates the total Hartree-Fock Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
      return_flat (bool): If True, returns a flattened Hamiltonian (default: True).

    Returns:
      np.ndarray: The total Hamiltonian, flattened if return_flat is True.
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if return_flat:
      return flattened(H_total, self.D, self.N_k)
    else:
      return H_total
