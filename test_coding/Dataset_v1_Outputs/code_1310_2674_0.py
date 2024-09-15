import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  """
  2-band tight-binding model on a square lattice with onsite interactions.
  The Hamiltonian is given by:
    H = \sum_{k,alpha,sigma} epsilon_{alpha} d^{dagger}_{k,alpha,sigma} d_{k,alpha,sigma}
       - \sum_{k,alpha,beta,sigma} t^{alpha,beta}_{k} d^{dagger}_{k,alpha,sigma} d_{k,beta,sigma}
       + \sum_{alpha,alpha',beta,beta',sigma,sigma'} \sum_{k}
         U^{sigma,sigma'}_{alpha,alpha',beta,beta'}(0)  
         [ <d^{dagger}_{k,alpha,sigma} d_{k,beta,sigma}> d^{dagger}_{k,alpha',sigma'} d_{k,beta',sigma'}
         - <d^{dagger}_{k,alpha,sigma} d_{k,beta',sigma'}> d^{dagger}_{k,alpha',sigma'} d_{k,beta,sigma} ]

  Args:
    N_shell (int): Number of k-points in the x-direction.
    parameters (dict[str, Any]): Dictionary containing model parameters:
        - 'epsilon': Onsite energy for each orbital.
        - 't': Hopping parameters.
        - 'U': Interaction strength.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
  """
  def __init__(self, N_shell: int=10, parameters:dict[str, Any]={'epsilon': [0.0, 0.0], 't': [[1.0, 0.5],[0.5, 1.0]], 'U': [[[[1.0, 0.5],[0.5, 1.0]],[[0.5, 1.0],[1.0, 0.5]]],[[[0.5, 1.0],[1.0, 0.5]],[[1.0, 0.5],[0.5, 1.0]]]]}, filling_factor: float=0.5): #TODO: To add space_dim or not?
    self.lattice = 'square' # LM Task: Define the lattice type.
    self.D = (2, 2) # LM Task: has to define this tuple.
    self.basis_order = {'0': 'orbital', '1': 'spin'}
    # this is the basis order that the Hamiltonian will follow
    # 0: orbital. Order: 0, 1, ....
    # 1: spin. Order: up, down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0.0 # LM Task: Define the temperature
    self.k_space = generate_k_space(lattice=self.lattice, N_shell = N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # All other parameters such as interaction strengths
    self.epsilon = parameters['epsilon'] #  Onsite energy for each orbital.
    self.t = parameters['t'] # Hopping parameters.
    self.U = parameters['U'] # Interaction strength.
    #self.param_1 = parameters['param_1'] # Brief phrase explaining physical significance of `param_1`
    #...
    #self.param_p = parameters['param_p'] # Brief phrase explaining physical significance of `param_p`

    self.aM = 1 # LM Task: Define the lattice constant, used for the area.
    # Any other problem specific parameters.

    return

  def generate_non_interacting(self) -> np.ndarray:
    """
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    for alpha in range(self.D[0]):
      for spin in range(self.D[1]):
        H_nonint[alpha, spin, alpha, spin, :] = self.epsilon[alpha] #  \epsilon_{\alpha} d^{\dagger}_{\bf k,\alpha,\sigma} d^{\phantom\dagger}_{\bf k,\alpha,\sigma}
    for alpha in range(self.D[0]):
      for beta in range(self.D[0]):
        for spin in range(self.D[1]):
          H_nonint[alpha, spin, beta, spin, :] = -self.t[alpha][beta]*(np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1])) #t^{\alpha\beta}_{\bf k} d^{\dagger}_{\bf k,\alpha,\sigma} d^{\phantom\dagger}_{\bf k,\beta,\sigma}  
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

    for alpha in range(self.D[0]):
      for alphap in range(self.D[0]):
        for beta in range(self.D[0]):
          for betap in range(self.D[0]):
            for spin in range(self.D[1]):
              for spinp in range(self.D[1]):
                H_int[alphap, spinp, betap, spinp, :] += self.U[spin][spinp][alpha][alphap] * exp_val[alpha, spin, :]*exp_val[beta, spin, :] # U^{\sigma,\sigma^\prime}_{\alpha,\alpha^\prime,\beta,\beta^\prime}(0)  
                                                                                                                                            #  \langle d^{\dagger}_{\bm k,\alpha,\sigma} d^{\phantom\dagger}_{\bm k,\beta,\sigma} \rangle d^{\dagger}_{\bm k,\alpha^\prime,\sigma^\prime} d^{\phantom\dagger}_{\bm k,\beta^\prime,\sigma^\prime}
                H_int[alphap, spinp, beta, spin, :] -= self.U[spin][spinp][alpha][alphap] * exp_val[alpha, spin, :]*exp_val[betap, spinp, :] # U^{\sigma,\sigma^\prime}_{\alpha,\alpha^\prime,\beta,\beta^\prime}(0)  
                                                                                                                                             # \langle d^{\dagger}_{\bm k,\alpha,\sigma} d^{\phantom\dagger}_{\bm k,\beta^\prime,\sigma^\prime} \rangle d^{\dagger}_{\bm k,\alpha^\prime,\sigma^\prime} d^{\phantom\dagger}_{\bm k,\beta,\sigma}
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
      return H_total #l1, s1, q1, ....k

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.k_space.shape[0]))

  def expand(self, exp_val):
    return exp_val.reshape((self.D + (self.k_space.shape[0],)))
