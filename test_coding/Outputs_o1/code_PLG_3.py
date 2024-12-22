# https://chatgpt.com/share/67437c20-927c-8011-8c7d-ad269376f826
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for rhombohedral-stacked pentalayer graphene.

    Args:
        N_k (int): Number of k-points in the Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_k: int=100, parameters: dict=None, filling_factor: float=0.5):
        self.lattice = 'triangular'  # Lattice symmetry ('square' or 'triangular')
        self.D = (10,)  # Number of flavors identified
        self.basis_order = {
            '0': 'A1',
            '1': 'B1',
            '2': 'A2',
            '3': 'B2',
            '4': 'A3',
            '5': 'B3',
            '6': 'A4',
            '7': 'B4',
            '8': 'A5',
            '9': 'B5'
        }
        # Order for each flavor

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature
        self.N_k = N_k  # Number of k-points
        self.k_space = self.generate_k_space(self.lattice, self.N_k)
        # self.area = ... # Area of the system, may be needed

        # Model parameters (with default values)
        if parameters is None:
            parameters = {
                'gamma_0': 2600,     # meV
                'gamma_1': 356.1,    # meV
                'gamma_2': -15,      # meV
                'gamma_3': -293,     # meV
                'gamma_4': -144,     # meV
                'delta': 12.2,       # meV
                'u_a': 16.4,         # meV
                'u_d': 0.0,          # Default value
                'e_charge': 1.602e-19,  # Coulombs
                'epsilon_0': 8.854e-12,  # F/m
                'epsilon_r': 5,         # Relative dielectric constant
                'd_s': 30e-9,           # Gate distance in meters
            }

        # Extract parameters
        self.gamma_0 = parameters['gamma_0']
        self.gamma_1 = parameters['gamma_1']
        self.gamma_2 = parameters['gamma_2']
        self.gamma_3 = parameters['gamma_3']
        self.gamma_4 = parameters['gamma_4']
        self.delta = parameters['delta']
        self.u_a = parameters['u_a']
        self.u_d = parameters['u_d']
        self.e_charge = parameters['e_charge']
        self.epsilon_0 = parameters['epsilon_0']
        self.epsilon_r = parameters['epsilon_r']
        self.d_s = parameters['d_s']

        # Precompute V_C(0)
        self.V_C_0 = self.compute_V_C_0()

        return

    def generate_k_space(self, lattice_type: str, N_k: int) -> np.ndarray:
        """
        Generates the k-space grid.

        Args:
            lattice_type (str): 'square' or 'triangular'
            N_k (int): Number of k-points

        Returns:
            np.ndarray: k-space points of shape (N_k, 2)
        """
        # Implement the k-space generation according to the lattice
        if lattice_type == 'triangular':
            # For simplicity, generate a regular grid
            N_side = int(np.sqrt(N_k))
            kx = np.linspace(-np.pi, np.pi, N_side)
            ky = np.linspace(-np.pi / np.sqrt(3), np.pi / np.sqrt(3), N_side)
            kx, ky = np.meshgrid(kx, ky)
            kx = kx.flatten()
            ky = ky.flatten()
            k_space = np.stack((kx, ky), axis=-1)
        else:
            # Square lattice
            N_side = int(np.sqrt(N_k))
            kx = np.linspace(-np.pi, np.pi, N_side)
            ky = np.linspace(-np.pi, np.pi, N_side)
            kx, ky = np.meshgrid(kx, ky)
            kx = kx.flatten()
            ky = ky.flatten()
            k_space = np.stack((kx, ky), axis=-1)
        self.N_k = k_space.shape[0]
        return k_space

    def compute_V_C_0(self) -> float:
        """
        Computes the Coulomb interaction at q=0.

        Returns:
            float: V_C(0)
        """
        # At q = 0, V_C(0) is undefined due to division by zero.
        # Physically, the long-wavelength limit can be handled by considering screening, or we can set V_C(0) to zero.
        # For simplicity, we set V_C(0) = 0
        return 0.0

    def compute_V_C_q(self, q_magnitude: np.ndarray) -> np.ndarray:
        """
        Computes the Coulomb interaction for given q magnitudes.

        Args:
            q_magnitude (np.ndarray): Array of q magnitudes

        Returns:
            np.ndarray: Coulomb interaction values for each q
        """
        # Avoid division by zero
        q_magnitude = np.where(q_magnitude == 0, 1e-10, q_magnitude)
        V_C_q = (self.e_charge ** 2) / (2 * self.epsilon_0 * self.epsilon_r * q_magnitude)
        V_C_q *= np.tanh(q_magnitude * self.d_s)
        return V_C_q

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        N_k = self.k_space.shape[0]
        D = self.D[0]
        H_nonint = np.zeros((D, D, N_k), dtype=np.float32)

        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        k_plus = kx + 1j * ky  # For K valley
        # If necessary, adjust for K' valley

        # Compute v_i terms
        sqrt3_over_2 = np.sqrt(3) / 2
        v0 = sqrt3_over_2 * self.gamma_0 * k_plus  # v0 = (√3/2) * γ0 * k_plus
        v0_dag = np.conj(v0)
        v3 = sqrt3_over_2 * self.gamma_3 * k_plus
        v3_dag = np.conj(v3)
        v4 = sqrt3_over_2 * self.gamma_4 * k_plus
        v4_dag = np.conj(v4)

        # Precompute constants
        gamma1 = self.gamma_1
        gamma2_over_2 = self.gamma_2 / 2
        delta = self.delta
        ua = self.u_a
        ud = self.u_d

        # Build H_nonint for each k-point
        for idx in range(N_k):
            H_k = np.zeros((D, D), dtype=complex)
            # Fill H_k according to H_0 matrix elements
            H_k[0, 0] = 2 * ud
            H_k[0, 1] = v0_dag[idx]
            H_k[0, 2] = v4_dag[idx]
            H_k[0, 3] = v3[idx]
            H_k[0, 5] = gamma2_over_2

            H_k[1, 0] = v0[idx]
            H_k[1, 1] = 2 * ud + delta
            H_k[1, 2] = gamma1
            H_k[1, 3] = v4_dag[idx]

            H_k[2, 0] = v4[idx]
            H_k[2, 1] = gamma1
            H_k[2, 2] = ud + ua
            H_k[2, 3] = v0_dag[idx]
            H_k[2, 4] = v4_dag[idx]
            H_k[2, 5] = v3[idx]
            H_k[2, 7] = gamma2_over_2

            H_k[3, 0] = v3_dag[idx]
            H_k[3, 1] = v4[idx]
            H_k[3, 2] = v0[idx]
            H_k[3, 3] = ud + ua
            H_k[3, 4] = gamma1
            H_k[3, 5] = v4_dag[idx]

            H_k[4, 2] = v4[idx]
            H_k[4, 3] = gamma1
            H_k[4, 4] = ua
            H_k[4, 5] = v0_dag[idx]
            H_k[4, 6] = v4_dag[idx]
            H_k[4, 7] = v3[idx]
            H_k[4, 9] = gamma2_over_2

            H_k[5, 0] = gamma2_over_2
            H_k[5, 2] = v3_dag[idx]
            H_k[5, 3] = v4[idx]
            H_k[5, 4] = v0[idx]
            H_k[5, 5] = ua
            H_k[5, 6] = gamma1
            H_k[5, 7] = v4_dag[idx]

            H_k[6, 4] = v4[idx]
            H_k[6, 5] = gamma1
            H_k[6, 6] = -ud + ua
            H_k[6, 7] = v0_dag[idx]
            H_k[6, 8] = v4_dag[idx]
            H_k[6, 9] = v3[idx]

            H_k[7, 2] = gamma2_over_2
            H_k[7, 5] = v3_dag[idx]
            H_k[7, 6] = v0[idx]
            H_k[7, 7] = -ud + ua
            H_k[7, 8] = gamma1
            H_k[7, 9] = v4_dag[idx]

            H_k[8, 6] = v4[idx]
            H_k[8, 7] = gamma1
            H_k[8, 8] = -2 * ud + delta
            H_k[8, 9] = v0_dag[idx]

            H_k[9, 4] = gamma2_over_2
            H_k[9, 6] = v3_dag[idx]
            H_k[9, 7] = v4[idx]
            H_k[9, 8] = v0[idx]
            H_k[9, 9] = -2 * ud

            # Assign H_k to H_nonint[:, :, idx]
            H_nonint[:, :, idx] = H_k.real  # Convert to real if necessary

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = self.expand(exp_val)  # Shape: (D, D, N_k)
        N_k = exp_val.shape[-1]
        D = self.D[0]
        H_int = np.zeros((D, D, N_k), dtype=np.float32)

        # Compute mean densities n_mu for each flavor mu
        n_mu = np.mean(exp_val.diagonal(axis1=0, axis2=1), axis=-1)  # Shape: (D,)

        # Hartree term (diagonal)
        for mu in range(D):
            H_int[mu, mu, :] += self.V_C_0 * n_mu[mu]  # Add Hartree term

        # Fock term (off-diagonal)
        # Compute V_C(k' - k) and perform sum over k'
        # This can be computationally intensive; we may approximate or vectorize

        # Precompute q = k' - k for all k, k'
        k_space = self.k_space  # Shape: (N_k, 2)
        q_x = k_space[:, np.newaxis, 0] - k_space[np.newaxis, :, 0]  # Shape: (N_k, N_k)
        q_y = k_space[:, np.newaxis, 1] - k_space[np.newaxis, :, 1]  # Shape: (N_k, N_k)
        q_magnitude = np.sqrt(q_x**2 + q_y**2)  # Shape: (N_k, N_k)

        V_C_q = self.compute_V_C_q(q_magnitude)  # Shape: (N_k, N_k)

        # Normalize by area (assuming area per k-point is 1 for simplicity)
        A = 1.0  # Replace with actual area if available

        # For each k, compute sum over k' of V_C_q * exp_val[mu, nu, :]
        for k_idx in range(N_k):
            for mu in range(D):
                for nu in range(D):
                    Fock_term = np.sum(V_C_q[k_idx, :] * exp_val[mu, nu, :]) / A
                    H_int[nu, mu, k_idx] -= Fock_term / N_k  # Add Fock term

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            flatten (bool): Whether to flatten the Hamiltonian.

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if flatten:
            return self.flatten(H_total)
        else:
            return H_total

    def flatten(self, ham: np.ndarray) -> np.ndarray:
        """
        Flattens the Hamiltonian.

        Args:
            ham (np.ndarray): Hamiltonian with shape (D, D, N_k).

        Returns:
            np.ndarray: Flattened Hamiltonian with shape (D_flattened, D_flattened, N_k).
        """
        return ham.reshape((np.prod(self.D), np.prod(self.D), self.N_k))

    def expand(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Expands the expectation values.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: Expanded expectation values with shape (D, D, N_k).
        """
        return exp_val.reshape(self.D + self.D + (self.N_k,))
