# https://chatgpt.com/share/674379bb-9930-8011-821b-2211f29798c2
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for rhombohedral-stacked pentalayer graphene.

    Args:
        N_shell (int): Number of k-point shells for k-space sampling.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """

    def __init__(self, N_shell: int = 5, parameters: dict = None, filling_factor: float = 0.5):
        self.lattice = 'triangular'
        self.D = (10,)  # Number of flavors identified.
        self.basis_order = {
            '0': 'sublattice-layer index. Order: A1, B1, A2, B2, A3, B3, A4, B4, A5, B5'
        }

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature
        self.a = 1.0  # Lattice constant (can be adjusted as needed)

        # Generate k-space
        self.k_space = self.generate_k_space(N_shell)
        self.N_k = self.k_space.shape[0]

        # Physical constants
        self.e = 1.602e-19  # Elementary charge in Coulombs
        self.epsilon_0 = 8.854e-12  # Vacuum permittivity in F/m

        # Model parameters
        if parameters is None:
            parameters = {}
        self.gamma_0 = parameters.get('gamma_0', 2600.0)  # meV
        self.gamma_1 = parameters.get('gamma_1', 356.1)
        self.gamma_2 = parameters.get('gamma_2', -15.0)
        self.gamma_3 = parameters.get('gamma_3', -293.0)
        self.gamma_4 = parameters.get('gamma_4', -144.0)
        self.delta = parameters.get('delta', 12.2)
        self.u_a = parameters.get('u_a', 16.4)
        self.u_d = parameters.get('u_d', 0.0)  # Default value
        self.epsilon = parameters.get('epsilon', 5.0)  # Effective dielectric constant
        self.d_s = parameters.get('d_s', 30e-9)  # Gate distance in meters

        # System area (approximate for triangular lattice)
        self.A = self.N_k * (self.a ** 2 * np.sqrt(3) / 2)

    def generate_k_space(self, N_shell):
        """
        Generates k-space points for a triangular lattice.

        Args:
            N_shell (int): Number of k-point shells.

        Returns:
            np.ndarray: Array of k-points.
        """
        # Placeholder function; implement actual k-space generation
        # For simplicity, create a grid in kx and ky
        k_max = np.pi / self.a
        kx = np.linspace(-k_max, k_max, 2 * N_shell + 1)
        ky = np.linspace(-k_max, k_max, 2 * N_shell + 1)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_space = np.vstack([kx_grid.ravel(), ky_grid.ravel()]).T
        return k_space

    def V_C0(self):
        """
        Calculates V_C at q=0 (regularized to avoid divergence).

        Returns:
            float: V_C(0)
        """
        return (self.e ** 2) / (2 * self.epsilon_0 * self.epsilon) * self.d_s

    def V_C(self, q):
        """
        Calculates the Coulomb interaction V_C(q).

        Args:
            q (np.ndarray): Magnitude of momentum transfer.

        Returns:
            np.ndarray: Coulomb potential V_C(q).
        """
        V_C_q = (self.e ** 2) / (2 * self.epsilon_0 * self.epsilon * q) * np.tanh(q * self.d_s)
        # Regularize at q=0
        V_C_q[q == 0] = self.V_C0()
        return V_C_q

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D_flattened, D_flattened, N_k).
        """
        N_k = self.N_k
        H_nonint = np.zeros((10, 10, N_k), dtype=np.complex128)

        # Compute v_i terms
        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        k_plus = kx + 1j * ky  # Using valley index +
        factor = (np.sqrt(3) / 2)
        v0 = factor * self.gamma_0 * k_plus
        v0_dag = np.conj(v0)
        v3 = factor * self.gamma_3 * k_plus
        v3_dag = np.conj(v3)
        v4 = factor * self.gamma_4 * k_plus
        v4_dag = np.conj(v4)

        # Model parameters
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2
        delta = self.delta
        u_a = self.u_a
        u_d = self.u_d

        # Build H0 for each k-point
        for idx_k in range(N_k):
            H = np.zeros((10, 10), dtype=np.complex128)
            # Shortcuts for k-dependent terms
            v0_k = v0[idx_k]
            v0_dag_k = v0_dag[idx_k]
            v3_k = v3[idx_k]
            v3_dag_k = v3_dag[idx_k]
            v4_k = v4[idx_k]
            v4_dag_k = v4_dag[idx_k]

            # Fill the Hamiltonian matrix H at k
            # Entries correspond to the given H0 matrix
            H[0, 0] = 2 * u_d
            H[0, 1] = v0_dag_k
            H[0, 2] = v4_dag_k
            H[0, 3] = v3_k
            H[0, 5] = gamma_2 / 2

            H[1, 0] = v0_k
            H[1, 1] = 2 * u_d + delta
            H[1, 2] = gamma_1
            H[1, 3] = v4_dag_k

            H[2, 0] = v4_k
            H[2, 1] = gamma_1
            H[2, 2] = u_d + u_a
            H[2, 3] = v0_dag_k
            H[2, 4] = v4_dag_k
            H[2, 5] = v3_k
            H[2, 7] = gamma_2 / 2

            H[3, 0] = v3_dag_k
            H[3, 1] = v4_k
            H[3, 2] = v0_k
            H[3, 3] = u_d + u_a
            H[3, 4] = gamma_1
            H[3, 5] = v4_dag_k

            H[4, 2] = v4_k
            H[4, 3] = gamma_1
            H[4, 4] = u_a
            H[4, 5] = v0_dag_k
            H[4, 6] = v4_dag_k
            H[4, 7] = v3_k
            H[4, 9] = gamma_2 / 2

            H[5, 0] = gamma_2 / 2
            H[5, 2] = v3_dag_k
            H[5, 3] = v4_k
            H[5, 4] = v0_k
            H[5, 5] = u_a
            H[5, 6] = gamma_1
            H[5, 7] = v4_dag_k

            H[6, 4] = v4_k
            H[6, 5] = gamma_1
            H[6, 6] = -u_d + u_a
            H[6, 7] = v0_dag_k
            H[6, 8] = v4_dag_k
            H[6, 9] = v3_k

            H[7, 2] = gamma_2 / 2
            H[7, 5] = v3_dag_k
            H[7, 6] = v4_k
            H[7, 7] = -u_d + u_a
            H[7, 8] = gamma_1
            H[7, 9] = v4_dag_k

            H[8, 6] = v4_k
            H[8, 7] = gamma_1
            H[8, 8] = -2 * u_d + delta
            H[8, 9] = v0_dag_k

            H[9, 4] = gamma_2 / 2
            H[9, 6] = v3_dag_k
            H[9, 7] = v4_k
            H[9, 8] = v0_k
            H[9, 9] = -2 * u_d

            # Ensure Hermiticity
            H = H + H.conj().T - np.diag(H.diagonal())

            # Assign to H_nonint
            H_nonint[:, :, idx_k] = H

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D_flattened, D_flattened, N_k).
        """
        N_k = self.N_k
        H_int = np.zeros((10, 10, N_k), dtype=np.complex128)

        # Compute n_total
        n_total = np.sum(exp_val.diagonal(axis1=0, axis2=1))  # Sum over all k' and diagonal elements
        V_C0 = self.V_C0()

        # Add to diagonal elements (Hartree term)
        for i in range(10):
            H_int[i, i, :] += (1 / self.A) * V_C0 * n_total

        # Compute exchange term (Fock term)
        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        kx_diff = kx[None, :] - kx[:, None]
        ky_diff = ky[None, :] - ky[:, None]
        q = np.sqrt(kx_diff ** 2 + ky_diff ** 2) + 1e-12  # Add small number to avoid division by zero
        V_C_q = self.V_C(q)  # Shape (N_k, N_k)

        # Compute the exchange interaction
        for idx_k in range(N_k):
            for i in range(10):
                for j in range(10):
                    # Sum over k'
                    exchange_sum = np.sum(V_C_q[idx_k, :] * exp_val[i, j, :])
                    H_int[j, i, idx_k] += - (1 / self.A) * exchange_sum  # Note the index swap for H[j, i, k]

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool, optional): Whether to flatten the output. Defaults to True.

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return H_total.reshape((np.prod(self.D), np.prod(self.D), self.N_k))
        else:
            return H_total  # Shape: (D_flattened, D_flattened, N_k)
