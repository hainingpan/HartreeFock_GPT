import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
    """A class representing the Hartree-Fock Hamiltonian for a three-orbital (p_x, p_y, d) model on a square lattice.
    
    This class implements the mean-field Hamiltonian for a model with p_x, p_y, and d orbitals,
    with both non-interacting hopping terms and interacting terms dependent on orbital occupations.
    
    Args:
        N_shell (int, optional): Number of shells in the Brillouin zone. Default is 3.
        parameters (dict, optional): Dictionary of model parameters. Defaults include:
            't_pd': Copper-oxygen hopping parameter
            't_pp': Oxygen-oxygen hopping parameter
            'Delta': Energy difference between p and d orbitals
            'U_p': On-site interaction for p orbitals
            'V_pp': Nearest-neighbor p-p interaction
            'V_pd': Nearest-neighbor p-d interaction
            'U_d': On-site interaction for d orbitals
            'n': Total hole density
            'T': Temperature (default 0)
        filling_factor (float, optional): Filling factor for occupation computation. Default is 0.5.
    """
    def __init__(self, N_shell: int = 3, parameters: dict[str, Any]={'t_pd': 1.0, 't_pp': 0.5, 'Delta': 2.0, 
                                                               'U_p': 4.0, 'V_pp': 1.0, 'V_pd': 1.0, 'U_d': 8.0, 'n': 1.0, 'T': 0},
                 filling_factor: float = 0.5):
        self.lattice = 'square'   # Lattice symmetry ('square')
        self.D = (2, 3)  # Number of flavors: (spin, orbital)
        self.basis_order = {'0': 'spin', '1': 'orbital'}
        # Order for each flavor:
        # 0: spin up, spin down
        # 1: orbital p_x, p_y, d

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = parameters.get('T', 0)  # temperature, default to 0
        self.a = 1.0  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        self.t_pd = parameters.get('t_pd', 1.0)  # p-d hopping
        self.t_pp = parameters.get('t_pp', 0.5)  # p-p hopping
        self.Delta = parameters.get('Delta', 2.0)  # Energy difference between p and d orbitals
        self.U_p = parameters.get('U_p', 4.0)  # On-site p orbital interaction
        self.V_pp = parameters.get('V_pp', 1.0)  # p-p interaction
        self.V_pd = parameters.get('V_pd', 1.0)  # p-d interaction
        self.U_d = parameters.get('U_d', 8.0)  # On-site d orbital interaction
        self.n = parameters.get('n', 1.0)  # Total hole density
        
        # Compute effective interaction parameters
        self.U_p_tilde = self.U_p + 8 * self.V_pp - 8 * self.V_pd
        self.V_pp_tilde = 8 * self.V_pp - self.U_p
        self.U_d_tilde = self.U_d - 4 * self.V_pd
        
        # Compute the chemical potential
        self.mu = 2 * self.V_pd * self.n - self.V_pd * self.n**2
        
        return

    def generate_non_interacting(self) -> np.ndarray:
        """Generates the non-interacting part of the Hamiltonian.
        
        This includes the off-diagonal hopping terms gamma_1 and gamma_2.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Iterate over all spins
        for s in range(self.D[0]):
            # Off-diagonal hopping terms gamma_2(k) between p_x and p_y
            gamma_2_k = -4 * self.t_pp * np.cos(self.k_space[:, 0]/2) * np.cos(self.k_space[:, 1]/2)
            H_nonint[s, 0, s, 1, :] = gamma_2_k  # p_x to p_y hopping
            H_nonint[s, 1, s, 0, :] = gamma_2_k  # p_y to p_x hopping (hermitian conjugate)
            
            # Off-diagonal hopping terms gamma_1(k_x) between p_x and d
            gamma_1_kx = -2 * self.t_pd * np.cos(self.k_space[:, 0]/2)
            H_nonint[s, 0, s, 2, :] = gamma_1_kx  # p_x to d hopping
            H_nonint[s, 2, s, 0, :] = gamma_1_kx  # d to p_x hopping (hermitian conjugate)
            
            # Off-diagonal hopping terms gamma_1(k_y) between p_y and d
            gamma_1_ky = -2 * self.t_pd * np.cos(self.k_space[:, 1]/2)
            H_nonint[s, 1, s, 2, :] = gamma_1_ky  # p_y to d hopping
            H_nonint[s, 2, s, 1, :] = gamma_1_ky  # d to p_y hopping (hermitian conjugate)
        
        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generates the interacting part of the Hamiltonian.
        
        This includes the diagonal terms xi_x, xi_y, and xi_d which depend on expectation values.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = unflatten(exp_val, self.D, self.N_k)
        H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
        
        # Extract the diagonal elements for each spin and orbital
        # exp_val has shape (2, 3, 2, 3, N_k) for (spin1, orbital1, spin2, orbital2, k)
        
        # Calculate p_x and p_y occupations for each spin
        n_px_up = np.mean(exp_val[0, 0, 0, 0, :])    # <c^dag_{up,px,k} c_{up,px,k}>
        n_px_down = np.mean(exp_val[1, 0, 1, 0, :])  # <c^dag_{down,px,k} c_{down,px,k}>
        n_py_up = np.mean(exp_val[0, 1, 0, 1, :])    # <c^dag_{up,py,k} c_{up,py,k}>
        n_py_down = np.mean(exp_val[1, 1, 1, 1, :])  # <c^dag_{down,py,k} c_{down,py,k}>
        
        # Total p orbital density: sum of all p_x and p_y occupations
        n_p = n_px_up + n_px_down + n_py_up + n_py_down
        
        # Nematic order parameter: difference between p_x and p_y occupations
        eta = (n_px_up + n_px_down) - (n_py_up + n_py_down)
        
        # Compute diagonal elements according to the Hamiltonian definition
        xi_x = self.Delta + self.U_p_tilde * n_p / 4 - self.V_pp_tilde * eta / 4 - self.mu
        xi_y = self.Delta + self.U_p_tilde * n_p / 4 + self.V_pp_tilde * eta / 4 - self.mu
        xi_d = self.U_d_tilde * (self.n - n_p) / 2 - self.mu
        
        # Fill in diagonal elements for each spin
        for s in range(self.D[0]):
            H_int[s, 0, s, 0, :] = xi_x  # p_x orbital energy
            H_int[s, 1, s, 1, :] = xi_y  # p_y orbital energy
            H_int[s, 2, s, 2, :] = xi_d  # d orbital energy
        
        # Note: The constant term f(n^p, eta) affects the total energy but not the eigenvalues
        # or eigenvectors, so it's not included in the Hamiltonian matrix.
        
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """Generates the total Hartree-Fock Hamiltonian.
        
        This combines the non-interacting and interacting parts of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool, optional): Whether to return a flattened Hamiltonian. Default is True.

        Returns:
            np.ndarray: The total Hamiltonian. If return_flat is True, the shape is 
                        (D_flattened, D_flattened, N_k), otherwise (D, D, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        
        if return_flat:
            return flattened(H_total, self.D, self.N_k)
        else:
            return H_total
