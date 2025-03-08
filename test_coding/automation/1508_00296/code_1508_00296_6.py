import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a square-centered lattice with N atoms at vertices and B atoms at centers.
    
    Args:
        N_shell: Number of shells in k-space.
        parameters: Dictionary of model parameters.
        filling_factor: Occupancy factor for states.
    """
    def __init__(self, N_shell: int, parameters: dict[str, Any]={'t_N': 1.0, 't_B': 1.0, 't_BN': 0.5, 'Delta': 0.0, 'U_N': 3.0, 'U_B': 0.0, 'V_B': 0.65, 'V_BN': 1.0}, filling_factor: float=0.5):
        self.lattice = 'square'   # Square-centered lattice
        self.D = (4,)  # N_up, N_down, B_up, B_down
        self.basis_order = {'0': 'atom_spin'}
        # Order: 0: N_up, 1: N_down, 2: B_up, 3: B_down
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # Temperature, default to 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Hopping parameters
        self.t_N = parameters.get('t_N', 1.0)  # Hopping for N atoms
        self.t_B = parameters.get('t_B', 1.0)  # Hopping for B atoms
        self.t_BN = parameters.get('t_BN', 0.5)  # Hopping between N and B atoms
        self.Delta = parameters.get('Delta', 0.0)  # On-site energy difference for N atoms
        
        # Interaction parameters
        self.U_N = parameters.get('U_N', 3.0)  # Interaction between N atoms with different spins
        self.U_B = parameters.get('U_B', 0.0)  # Interaction between B atoms with different spins
        self.V_B = parameters.get('V_B', 0.65)  # Interaction between B atoms with all spins
        self.V_BN = parameters.get('V_BN', 1.0)  # Interaction between N and B atoms
        
        return

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.
        
        Returns:
            The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate dispersion for square lattice
        k_x = self.k_space[:, 0]
        k_y = self.k_space[:, 1]
        # Sum over nearest neighbors phase factors
        nearest_neighbors_N = 2 * (np.cos(k_x * self.a) + np.cos(k_y * self.a))
        nearest_neighbors_B = 2 * (np.cos(k_x * self.a) + np.cos(k_y * self.a))
        # Phase factor for N-B hopping (connects center to vertices)
        NB_hopping = 4 * np.cos(k_x * self.a / 2) * np.cos(k_y * self.a / 2)
        
        # N atom hopping + on-site energy
        H_nonint[0, 0, :] = self.t_N * nearest_neighbors_N + self.Delta  # N atoms, spin up
        H_nonint[1, 1, :] = self.t_N * nearest_neighbors_N + self.Delta  # N atoms, spin down
        
        # B atom hopping
        H_nonint[2, 2, :] = self.t_B * nearest_neighbors_B  # B atoms, spin up
        H_nonint[3, 3, :] = self.t_B * nearest_neighbors_B  # B atoms, spin down
        
        # N-B hopping (off-diagonal terms)
        H_nonint[0, 2, :] = self.t_BN * NB_hopping  # N to B, spin up
        H_nonint[2, 0, :] = self.t_BN * np.conj(NB_hopping)  # B to N, spin up (hermitian conjugate)
        H_nonint[1, 3, :] = self.t_BN * NB_hopping  # N to B, spin down
        H_nonint[3, 1, :] = self.t_BN * np.conj(NB_hopping)  # B to N, spin down (hermitian conjugate)
        
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian using mean-field theory.
        
        Args:
            exp_val: Expectation value array with shape (D_flattened, D_flattened, N_k).
            
        Returns:
            The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Calculate mean densities
        n_N_up = np.mean(exp_val[0, 0, :])   # Mean density of N atoms with spin up
        n_N_down = np.mean(exp_val[1, 1, :]) # Mean density of N atoms with spin down
        n_B_up = np.mean(exp_val[2, 2, :])   # Mean density of B atoms with spin up
        n_B_down = np.mean(exp_val[3, 3, :]) # Mean density of B atoms with spin down
        
        # U_N term: Interaction between N atoms with different spins
        H_int[0, 0, :] += self.U_N / self.N_k * n_N_down  # N atoms, spin up affected by spin down
        H_int[1, 1, :] += self.U_N / self.N_k * n_N_up    # N atoms, spin down affected by spin up
        
        # U_B term: Interaction between B atoms with different spins
        H_int[2, 2, :] += self.U_B / self.N_k * n_B_down  # B atoms, spin up affected by spin down
        H_int[3, 3, :] += self.U_B / self.N_k * n_B_up    # B atoms, spin down affected by spin up
        
        # V_B term: Interaction between B atoms with all spins
        H_int[2, 2, :] += 2 * self.V_B / self.N_k * (n_B_up + n_B_down)  # B atoms, spin up
        H_int[3, 3, :] += 2 * self.V_B / self.N_k * (n_B_up + n_B_down)  # B atoms, spin down
        
        # V_BN term: Interaction between N and B atoms
        # N atoms affected by B atoms
        H_int[0, 0, :] += self.V_BN / self.N_k * (n_B_up + n_B_down)  # N atoms, spin up
        H_int[1, 1, :] += self.V_BN / self.N_k * (n_B_up + n_B_down)  # N atoms, spin down
        
        # B atoms affected by N atoms
        H_int[2, 2, :] += self.V_BN / self.N_k * (n_N_up + n_N_down)  # B atoms, spin up
        H_int[3, 3, :] += self.V_BN / self.N_k * (n_N_up + n_N_down)  # B atoms, spin down
        
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian by combining non-interacting and interacting parts.
        
        Args:
            exp_val: Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat: If True, returns the Hamiltonian in flattened form.
            
        Returns:
            The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
