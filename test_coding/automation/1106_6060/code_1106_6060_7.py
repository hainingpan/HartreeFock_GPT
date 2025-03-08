import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """Hartree-Fock Hamiltonian for a three-orbital (px, py, d) model on a square lattice.
  
  This class implements the mean-field Hamiltonian for a model of copper-oxygen planes
  with px, py oxygen orbitals and d copper orbitals, including interactions that can lead
  to nematic ordering.
  
  Args:
    N_shell: Number of shells in reciprocal space.
    parameters: Dictionary containing model parameters.
    filling_factor: Filling factor for the system, default 0.5.
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
    self.lattice = 'square'
    self.D = (2, 3)  # (spin, orbital)
    self.basis_order = {'0': 'spin', '1': 'orbital'}
    # Order for each flavor:
    # spin: up, down
    # orbital: px, py, d

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0)  # temperature, default to 0
    self.a = parameters.get('a', 1.0)  # Lattice constant
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

    # Model parameters
    self.Delta = parameters.get('Delta', 1.0)  # Charge transfer energy
    self.t_pd = parameters.get('t_pd', 1.0)  # Oxygen-copper hopping
    self.t_pp = parameters.get('t_pp', 0.5)  # Oxygen-oxygen hopping
    
    # Interaction parameters
    self.U_p = parameters.get('U_p', 3.0)  # On-site repulsion for oxygen
    self.U_d = parameters.get('U_d', 5.0)  # On-site repulsion for copper
    self.V_pp = parameters.get('V_pp', 1.0)  # Oxygen-oxygen repulsion
    self.V_pd = parameters.get('V_pd', 1.5)  # Oxygen-copper repulsion
    
    # Derived interaction parameters
    self.U_p_tilde = self.U_p + 8*self.V_pp - 8*self.V_pd  # Effective oxygen repulsion
    self.V_pp_tilde = 8*self.V_pp - self.U_p  # Effective oxygen-oxygen repulsion
    self.U_d_tilde = self.U_d - 4*self.V_pd  # Effective copper repulsion

    return

  def generate_non_interacting(self) -> np.ndarray:
    """Generate the non-interacting part of the Hamiltonian matrix.
    
    Returns:
      np.ndarray: Non-interacting Hamiltonian with shape (D, D, N_k).
    """
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Extract k-space coordinates
    kx = self.k_space[:, 0]
    ky = self.k_space[:, 1]
    
    # Hopping terms
    gamma_1_kx = -2 * self.t_pd * np.cos(kx/2)  # px-d hopping
    gamma_1_ky = -2 * self.t_pd * np.cos(ky/2)  # py-d hopping
    gamma_2_k = -4 * self.t_pp * np.cos(kx/2) * np.cos(ky/2)  # px-py hopping
    
    # For spin up (s=0)
    # Diagonal terms (charge transfer energy)
    H_nonint[0, 0, 0, 0, :] = self.Delta  # px-px
    H_nonint[0, 1, 0, 1, :] = self.Delta  # py-py
    
    # Off-diagonal hopping terms
    H_nonint[0, 0, 0, 1, :] = gamma_2_k  # px-py
    H_nonint[0, 1, 0, 0, :] = gamma_2_k  # py-px
    H_nonint[0, 0, 0, 2, :] = gamma_1_kx  # px-d
    H_nonint[0, 2, 0, 0, :] = gamma_1_kx  # d-px
    H_nonint[0, 1, 0, 2, :] = gamma_1_ky  # py-d
    H_nonint[0, 2, 0, 1, :] = gamma_1_ky  # d-py
    
    # For spin down (s=1) - same structure as spin up
    H_nonint[1, 0, 1, 0, :] = self.Delta  # px-px
    H_nonint[1, 1, 1, 1, :] = self.Delta  # py-py
    
    H_nonint[1, 0, 1, 1, :] = gamma_2_k  # px-py
    H_nonint[1, 1, 1, 0, :] = gamma_2_k  # py-px
    H_nonint[1, 0, 1, 2, :] = gamma_1_kx  # px-d
    H_nonint[1, 2, 1, 0, :] = gamma_1_kx  # d-px
    H_nonint[1, 1, 1, 2, :] = gamma_1_ky  # py-d
    H_nonint[1, 2, 1, 1, :] = gamma_1_ky  # d-py
    
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """Generate the interacting part of the Hamiltonian based on expectation values.
    
    Args:
      exp_val: Expectation value array with shape (D_flattened, D_flattened, N_k).
      
    Returns:
      np.ndarray: Interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = unflatten(exp_val, self.D, self.N_k)
    H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Calculate expectation values for different orbitals
    # Sum over all k points for each orbital and spin
    n_px_up = np.mean(exp_val[0, 0, 0, 0, :])  # <p†_x↑ p_x↑>
    n_py_up = np.mean(exp_val[0, 1, 0, 1, :])  # <p†_y↑ p_y↑>
    n_d_up = np.mean(exp_val[0, 2, 0, 2, :])   # <d†↑ d↑>
    
    n_px_down = np.mean(exp_val[1, 0, 1, 0, :])  # <p†_x↓ p_x↓>
    n_py_down = np.mean(exp_val[1, 1, 1, 1, :])  # <p†_y↓ p_y↓>
    n_d_down = np.mean(exp_val[1, 2, 1, 2, :])   # <d†↓ d↓>
    
    # Calculate derived quantities
    n_p = n_px_up + n_py_up + n_px_down + n_py_down  # Total oxygen hole density
    n_d = n_d_up + n_d_down  # Total copper hole density
    n = n_p + n_d  # Total hole density
    
    # Nematic order parameter (difference between px and py occupations)
    eta = (n_px_up + n_px_down) - (n_py_up + n_py_down)
    
    # Calculate chemical potential (dependent on n)
    mu = 2 * self.V_pd * n - self.V_pd * n**2
    
    # For spin up (s=0)
    # Diagonal terms with interaction
    H_int[0, 0, 0, 0, :] = self.U_p_tilde * n_p/4 - self.V_pp_tilde * eta/4 - mu  # px-px
    H_int[0, 1, 0, 1, :] = self.U_p_tilde * n_p/4 + self.V_pp_tilde * eta/4 - mu  # py-py
    H_int[0, 2, 0, 2, :] = self.U_d_tilde * (n - n_p)/2 - mu  # d-d
    
    # For spin down (s=1) - same interaction terms
    H_int[1, 0, 1, 0, :] = self.U_p_tilde * n_p/4 - self.V_pp_tilde * eta/4 - mu  # px-px
    H_int[1, 1, 1, 1, :] = self.U_p_tilde * n_p/4 + self.V_pp_tilde * eta/4 - mu  # py-py
    H_int[1, 2, 1, 2, :] = self.U_d_tilde * (n - n_p)/2 - mu  # d-d
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """Generate the total Hamiltonian combining non-interacting and interacting parts.
    
    Args:
      exp_val: Expectation value array with shape (D_flattened, D_flattened, N_k).
      return_flat: Whether to return the Hamiltonian in flattened form.
      
    Returns:
      np.ndarray: Total Hamiltonian, either flattened or in full tensor form.
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    
    if return_flat:
      return flattened(H_total, self.D, self.N_k)
    else:
      return H_total
