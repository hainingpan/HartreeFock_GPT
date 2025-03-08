import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for a system with N and B atoms on a square-centered lattice.
    
    This implements a model with two types of atoms (N at vertices and B at center) and two spin states.
    The Hamiltonian includes hopping terms and various interaction terms.
    
    Args:
        N_shell (int): Number of shells in the first Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor for the system. Defaults to 0.5.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
        self.lattice = 'square'
        self.D = (2, 2)  # (site, spin)
        self.basis_order = {'0': 'site', '1': 'spin'}
        # Order for each flavor:
        # site: 0=N (vertex), 1=B (center)
        # spin: 0=up, 1=down

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters for hopping
        self.t_N = parameters.get('t_N', 1.0)  # Hopping parameter for N atoms
        self.t_B = parameters.get('t_B', 1.0)  # Hopping parameter for B atoms
        self.t_BN = parameters.get('t_BN', 0.5)  # Hopping parameter between N and B atoms
        self.Delta = parameters.get('Delta', 0)  # On-site energy for N atoms

        # Model parameters for interactions
        self.U_N = parameters.get('U_N', 3.0)  # Interaction strength between N atoms of opposite spins
        self.U_B = parameters.get('U_B', 0.0)  # Interaction strength between B atoms of opposite spins
        self.V_B = parameters.get('V_B', 0.65)  # Additional interaction for B atoms
        self.V_BN = parameters.get('V_BN', 1.0)  # Interaction between N and B atoms

        return

    def generate_non_interacting(self) -> np.ndarray:
        """Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (*self.D, *self.D, self.N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute dispersion for nearest-neighbor hopping on square lattice
        # The sum over n e^(-ik·n) for nearest neighbors gives 2(cos(k_x) + cos(k_y))
        disp_N = 2 * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))
        disp_B = 2 * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))
        
        # Compute N-B hopping term
        # For a square-centered lattice, the sum over n' e^(-ik·n') gives 4*cos(k_x/2)*cos(k_y/2)
        # where n' are the vectors connecting N and B sites
        disp_NB = 4 * np.cos(self.k_space[:, 0] / 2) * np.cos(self.k_space[:, 1] / 2)
        
        # On-site energies and hopping for N atoms (both spins)
        H_nonint[0, 0, 0, 0, :] = self.t_N * disp_N + self.Delta  # N up-up
        H_nonint[0, 1, 0, 1, :] = self.t_N * disp_N + self.Delta  # N down-down
        
        # On-site energies and hopping for B atoms (both spins)
        H_nonint[1, 0, 1, 0, :] = self.t_B * disp_B  # B up-up
        H_nonint[1, 1, 1, 1, :] = self.t_B * disp_B  # B down-down
        
        # Hopping between N and B atoms (spin-conserving)
        H_nonint[0, 0, 1, 0, :] = self.t_BN * disp_NB  # N up to B up
        H_nonint[1, 0, 0, 0, :] = self.t_BN * np.conj(disp_NB)  # B up to N up
        H_nonint[0, 1, 1, 1, :] = self.t_BN * disp_NB  # N down to B down
        H_nonint[1, 1, 0, 1, :] = self.t_BN * np.conj(disp_NB)  # B down to N down
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (*self.D, *self.D, self.N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Compute the mean densities for N and B atoms with both spins
        n_N_up = np.mean(exp_val[0, 0, 0, 0, :])    # <a†_{k,up} a_{k,up}>
        n_N_down = np.mean(exp_val[0, 1, 0, 1, :])  # <a†_{k,down} a_{k,down}>
        n_B_up = np.mean(exp_val[1, 0, 1, 0, :])    # <b†_{k,up} b_{k,up}>
        n_B_down = np.mean(exp_val[1, 1, 1, 1, :])  # <b†_{k,down} b_{k,down}>
        
        # Interaction terms for N atoms
        # U_N term for opposite spins: U_N/N sum_{k,σ≠σ'} <a†_{k,σ} a_{k,σ}> a†_{k,σ'} a_{k,σ'}
        H_int[0, 0, 0, 0, :] = self.U_N * n_N_down  # N up interacting with N down
        H_int[0, 1, 0, 1, :] = self.U_N * n_N_up    # N down interacting with N up
        
        # V_BN term for N atoms: V_BN/N sum_{k,σ,σ'} <b†_{k,σ} b_{k,σ}> a†_{k,σ'} a_{k,σ'}
        H_int[0, 0, 0, 0, :] += self.V_BN * (n_B_up + n_B_down)  # N up interacting with all B
        H_int[0, 1, 0, 1, :] += self.V_BN * (n_B_up + n_B_down)  # N down interacting with all B
        
        # Interaction terms for B atoms
        # U_B term for opposite spins: U_B/N sum_{k,σ≠σ'} <b†_{k,σ} b_{k,σ}> b†_{k,σ'} b_{k,σ'}
        H_int[1, 0, 1, 0, :] = self.U_B * n_B_down  # B up interacting with B down
        H_int[1, 1, 1, 1, :] = self.U_B * n_B_up    # B down interacting with B up
        
        # 2V_B term for all B-B interactions: 2V_B/N sum_{k,σ,σ'} <b†_{k,σ} b_{k,σ}> b†_{k,σ'} b_{k,σ'}
        H_int[1, 0, 1, 0, :] += 2 * self.V_B * (n_B_up + n_B_down)  # B up interacting with all B
        H_int[1, 1, 1, 1, :] += 2 * self.V_B * (n_B_up + n_B_down)  # B down interacting with all B
        
        # V_BN term for B atoms: V_BN/N sum_{k,σ,σ'} <a†_{k,σ} a_{k,σ}> b†_{k,σ'} b_{k,σ'}
        H_int[1, 0, 1, 0, :] += self.V_BN * (n_N_up + n_N_down)  # B up interacting with all N
        H_int[1, 1, 1, 1, :] += self.V_BN * (n_N_up + n_N_down)  # B down interacting with all N
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool, optional): If True, returns the flattened Hamiltonian. Defaults to True.

        Returns:
            np.ndarray: The total Hamiltonian. If return_flat is True, shape is (np.prod(self.D), np.prod(self.D), self.N_k).
                        Otherwise, shape is (*self.D, *self.D, self.N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
