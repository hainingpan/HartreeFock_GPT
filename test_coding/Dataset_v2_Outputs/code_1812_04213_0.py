from HF import *

import numpy as np
from typing import Any
from scipy.linalg import block_diag

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for twisted bilayer graphene.

    Args:
        N_shell (int): Number of k-point shells to include.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=1, parameters: dict[str, Any]={'v_D': 1.0, 'omega_0': 1.0, 'omega_1': 1.0, 'theta': np.pi/6, 'V':1.0}, filling_factor: float=0.5):
        self.lattice = 'triangular' # Moire lattice type
        self.D = (2, 3)  # (layer, reciprocal_lattice_vector)
        self.basis_order = {'0': 'layer', '1': 'reciprocal_lattice_vector'}
        # Order for each flavor:
        # 0: top, bottom
        # 1: q_0, q_1, q_2

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0.0  # Assuming zero temperature
        self.k_space = generate_k_space(self.lattice, N_shell)
        self.Nk = self.k_space.shape[0]
        self.N_shell = N_shell
        # Model parameters
        self.v_D = parameters.get('v_D', 1.0)  # Dirac velocity
        self.omega_0 = parameters.get('omega_0', 1.0)  # Interlayer tunneling strength
        self.omega_1 = parameters.get('omega_1', 1.0)  # Interlayer tunneling modulation strength
        self.theta = parameters.get('theta', np.pi/6)  # Twist angle
        self.V = parameters.get('V', 1.0) # Interaction strength

        self.a = 1.42 # Angstrom. Monolayer graphene lattice constant.
        self.aM = self.a/(2*np.sin(self.theta/2))  # Moire lattice constant
        self.phi = 2*np.pi/3 # Phase in T_j
        self.A =  np.sqrt(3)/2* self.aM**2
        self.b1 = np.array([1/2, np.sqrt(3)/2])*4*np.pi/(np.sqrt(3)*self.aM)
        self.b2 = np.array([-1/2, np.sqrt(3)/2])*4*np.pi/(np.sqrt(3)*self.aM)
        self.q = [np.array([0.0, 0.0]), self.b1, self.b2]
        return

    def generate_non_interacting(self) -> np.ndarray:
        """Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: Non-interacting Hamiltonian.
        """
        N_k = self.k_space.shape[0]
        H_nonint = np.zeros(self.D + (N_k,), dtype=np.complex128)

        for k_idx, k in enumerate(self.k_space):
            kbar_plus = k - np.array([0, 4*np.pi/(3*self.a)]) # K_theta Dirac point. Assuming theta = pi/3 to get simplified value
            kbar_minus = k - np.array([0, -4*np.pi/(3*self.a)])

            theta_k_plus = np.arctan2(kbar_plus[1],kbar_plus[0])
            theta_k_minus = np.arctan2(kbar_minus[1], kbar_minus[0])


            h_plus = -self.v_D * np.linalg.norm(kbar_plus) * np.array([[0, np.exp(1j*(theta_k_plus - self.theta/2))],
                                                                   [np.exp(-1j*(theta_k_plus - self.theta/2)), 0]])
            h_minus = -self.v_D * np.linalg.norm(kbar_minus) * np.array([[0, np.exp(1j*(theta_k_minus + self.theta/2))],
                                                                       [np.exp(-1j*(theta_k_minus + self.theta/2)), 0]])

            # Tunneling Hamiltonian
            h_T = sum([self.omega_0 * np.eye(2) + self.omega_1*np.cos(j*self.phi)*np.array([[0, 1], [1, 0]]) + \
                       self.omega_1*np.sin(j*self.phi)*np.array([[0, -1j], [1j, 0]]) for j in range(3)])


            H_nonint[:, :, k_idx] = np.block([[h_plus, h_T], [h_T.conj().T, h_minus]])

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array.

        Returns:
            np.ndarray: Interacting Hamiltonian.
        """

        exp_val = self.expand(exp_val)
        N_k = exp_val.shape[-1]
        H_int = np.zeros(self.D + (N_k,), dtype=np.complex128)

        rho = np.mean(exp_val, axis=2)

        # Assuming rho_iso is predefined and accessible
        delta_rho = rho - self.rho_iso

        # Hartree term
        Sigma_H = self.V * delta_rho

        # Fock term - Simplified Placeholder. Needs full implementation based on Eq.\ref{fockse}
        Sigma_F = -self.V * exp_val

        for k_idx in range(self.Nk):
            H_int[:, :, k_idx] = Sigma_H + Sigma_F[:, :, k_idx]

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
        """Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array.
            flatten (bool, optional): Whether to flatten the Hamiltonian. Defaults to True.

        Returns:
            np.ndarray: Total Hamiltonian.
        """
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

