import numpy as np
from typing import Any
from scipy.constants import hbar

class HartreeFockHamiltonian:
    """
    Args:
        N_shell (int): Number of k-points in the x-direction.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=10, parameters: dict={'theta': np.pi/4, 'vF': 1.0, 'w0': 0.1, 'w1':0.05 }, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Lattice symmetry ('square' or 'triangular'). Defaults to 'triangular'.
        self.D = (2, 2) # Number of flavors identified: valley, sublattice
        self.basis_order = {'0': 'valley', '1':'sublattice'}
        # Order for each flavor:
        # 0: valley: K, K'
        # 1: sublattice: A, B

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0 # temperature
        self.k_space = generate_k_space(symmetry=self.lattice, N_shell)
        self.Nk = self.k_space.shape[0]
        # N_kx = 2*(N_shell+1) for a square lattice

        # Model parameters
        self.theta = parameters.get('theta', np.pi/4) # Angle between layers
        self.vF = parameters.get('vF', 1.0) # Fermi velocity
        self.w0 = parameters.get('w0', 0.1) # Sublattice-independent interlayer hopping
        self.w1 = parameters.get('w1', 0.05) # Sublattice-dependent interlayer hopping
        self.phi = parameters.get('phi', 2*np.pi/3) # Twist angle

        self.a = 1 # Lattice constant of monolayer graphene
        self.aM = self.a / (2*np.sin(self.theta/2)) # Moire lattice constant

        # Define reciprocal lattice vectors
        self.b1 = np.array([1/2, np.sqrt(3)/2]) * 4*np.pi / (np.sqrt(3) * self.aM)
        self.b2 = np.array([-1/2, np.sqrt(3)/2]) * 4*np.pi / (np.sqrt(3) * self.aM)
        self.G = np.array([self.b1, self.b2]) # Reciprocal lattice vectors

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, Nk).
        """
        N_k = self.k_space.shape[0]
        H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128)
        # Kinetic energy for spin up and spin down.
        # They are identical in this case, but we keep them separate for clarity

        # Define the Dirac Hamiltonian
        for i, k in enumerate(self.k_space):
          H_nonint[0, 0, :, :, i] = self.h_theta(k, self.theta/2)  
          H_nonint[1, 1, :, :, i] = self.h_theta(k, -self.theta/2)

        # Interlayer tunneling
        H_nonint[0, 1, :, :, :] = self.hT() #h_T(r)
        H_nonint[1, 0, :, :, :] = np.conj(np.transpose(H_nonint[0, 1, :, :, :], axes=(1, 0, 2))) #h_T(r)^\dagger

        return H_nonint
    
    def h_theta(self, k, theta):
      """
      Dirac Hamiltonian for rotated graphene.
      """
      k_ = k - self.Dirac_momentum(theta) #k - K_{theta}
      angle = np.arctan(k_[1]/k_[0])

      return -hbar * self.vF * np.linalg.norm(k_) * np.array([[0, np.exp(1j*(angle-theta))],
                                                                 [np.exp(-1j*(angle-theta)), 0]])
    
    def Dirac_momentum(self, theta):
      return (4*np.pi/(3*self.a)) * np.array([np.cos(theta), np.sin(theta)])


    def hT(self):
      return self.w0 * np.eye(2) + self.w1 * (np.cos(self.phi) * np.array([[0, 1], [1, 0]])  + np.sin(self.phi) * np.array([[0, -1j], [1j, 0]]))
    
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
        H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128)

        # Calculate the mean densities for spin up and spin down
        # Assuming the interaction is density-density interaction.

        # Hartree-Fock terms
        #H_int[0, 0, :] = self.U * n_down # Interaction of spin up with average spin down density
        #H_int[1, 1, :] = self.U * n_up # Interaction of spin down with average spin up density
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
      return exp_val.reshape((self.D + (self.Nk,)))
