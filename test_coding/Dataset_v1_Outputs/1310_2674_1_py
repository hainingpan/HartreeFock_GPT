import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters: dict={'t':1.0, 'U':1.0}, filling_factor: float=0.5):
    # LM Task: Update lattice
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    # LM Task: What is D?
    self.D = (2, 2) # Number of orbitals and spins
    # LM Task: Update the basis order
    self.basis_order = {'0': 'orbital', '1':'spin'}
    # Order for each flavor:
    # 0: orbital. Order: 0, 1, ...
    # 1: spin. Order: up, down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0. # Assuming T = 0
    self.k_space = generate_k_space(lattice=self.lattice, N_shell=N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # LM Task: Define the lattice constant
    self.aM = 1. # Assuming lattice constant is 1

    # Model parameters
    # LM Task: Add all parameters from the model that do NOT appear in EXP-VAL DEPENDENT TERMS.
    # These should be accessible from `generate_Htotal` via self.<parameter_name>.
    self.epsilon = parameters.get('epsilon', np.zeros(self.D[0])) # On-site energy for each orbital
    self.t_ij = parameters.get('t_ij', np.ones((self.D[0], self.D[0])))  # Hopping parameters between orbitals
    self.U = parameters.get('U', np.zeros((self.D[1], self.D[1], self.D[0], self.D[0], self.D[0], self.D[0]))) # Interaction strengths
    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    # Kinetic energy
    for alpha in range(self.D[0]):
        for spin_index in range(self.D[1]):
            H_nonint[alpha, spin_index, alpha, spin_index, :] += self.epsilon[alpha] # On-site energy

    for alpha in range(self.D[0]):
        for beta in range(self.D[0]):
            for spin_index in range(self.D[1]):
                H_nonint[alpha, spin_index, beta, spin_index, :] -= self.t_ij[alpha, beta] * np.exp(-1j * np.dot(self.k_space, self.aM))  # Hopping term

    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = self.expand(exp_val) # D[0], D[1], D[0], D[1], N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    # Calculate the mean densities for spin up and spin down
    for alpha in range(self.D[0]):
        for alpha_prime in range(self.D[0]):
            for beta in range(self.D[0]):
                for beta_prime in range(self.D[0]):
                    for spin_index in range(self.D[1]):
                        for spin_prime_index in range(self.D[1]):
                            # Hartree Terms
                            H_int[alpha_prime, spin_prime_index, beta_prime, spin_prime_index, :] += self.U[spin_index, spin_prime_index, alpha, alpha_prime, beta, beta_prime] * exp_val[alpha, spin_index, :, beta, spin_index, :]

                            # Fock Terms
                            H_int[alpha_prime, spin_prime_index, beta, spin_index, :] -= self.U[spin_index, spin_prime_index, alpha, alpha_prime, beta, beta_prime] * exp_val[alpha, spin_index, :, beta_prime, spin_prime_index, :]
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
      return self.flatten(H_total)
    else:
      return H_total

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D,self.D, self.Nk))
