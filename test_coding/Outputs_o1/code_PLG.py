"""Generate Pentalayer graphene with o1, https://chatgpt.com/share/67322b4e-42f0-8011-98eb-67e7c5866569"""
import numpy as np
from typing import Any
from scipy.constants import e, epsilon_0

def generate_k_space(lattice, N_kx, a):
    # Placeholder function; replace with actual implementation
    # Generates k-space grid for the given lattice
    # For simplicity, we consider a square grid here
    kx = np.linspace(-np.pi / a, np.pi / a, N_kx)
    ky = np.linspace(-np.pi / a, np.pi / a, N_kx)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_space = np.vstack([kx_grid.ravel(), ky_grid.ravel()]).T
    return k_space

class HartreeFockHamiltonian:
    """
    Args:
        N_kx (int): Number of k-points in the x-direction.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_kx: int=10, parameters: dict[str, Any]=None, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (5, 2)  # Number of layers and sublattices
        self.basis_order = {
            '0': 'layer. Order: 1,2,3,4,5',
            '1': 'sublattice. Order: A,B'
        }

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature
        self.a = parameters['a'] if parameters and 'a' in parameters else 1.0  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_kx, self.a)
        self.N_k = self.k_space.shape[0]

        # Model parameters with default values
        params = parameters if parameters else {}
        self.gamma_0 = params.get('gamma_0', 2600.0)
        self.gamma_1 = params.get('gamma_1', 356.1)
        self.gamma_2 = params.get('gamma_2', -15.0)
        self.gamma_3 = params.get('gamma_3', -293.0)
        self.gamma_4 = params.get('gamma_4', -144.0)
        self.delta = params.get('delta', 12.2)
        self.u_a = params.get('u_a', 16.4)
        self.u_d = params.get('u_d', 0.0)  # Displacement field

        # Interaction parameters
        self.epsilon = params.get('epsilon', 10.0)  # Dielectric constant
        self.d_s = params.get('d_s', 1e-9)  # Gate distance in meters

        # Valley parameter (+1 or -1)
        self.valley = params.get('valley', '+')
        self.sign = 1 if self.valley == '+' else -1

        return

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        N_k = self.N_k
        D = self.D
        H_nonint = np.zeros(D + D + (N_k,), dtype=np.complex128)

        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]

        # Compute v_i and v_i^\dagger functions
        def v_i(kx, ky, gamma_i):
            return (np.sqrt(3)/2) * gamma_i * (self.sign * kx + 1j * ky)

        def v_i_dagger(kx, ky, gamma_i):
            return np.conj(v_i(kx, ky, gamma_i))

        # Map (layer, sublattice) to flat index
        def idx(layer, sublattice):
            return layer * 2 + sublattice

        # Now fill in the Hamiltonian matrix elements
        for idx_k in range(N_k):
            # Initialize the 10x10 matrix for each k-point
            H_k = np.zeros((10, 10), dtype=np.complex128)

            # Get kx and ky for this k-point
            kx_k = kx[idx_k]
            ky_k = ky[idx_k]

            # Compute v0, v3, v4 and their conjugates
            v0 = v_i(kx_k, ky_k, self.gamma_0)
            v0_dag = v_i_dagger(kx_k, ky_k, self.gamma_0)

            v3 = v_i(kx_k, ky_k, self.gamma_3)
            v3_dag = v_i_dagger(kx_k, ky_k, self.gamma_3)

            v4 = v_i(kx_k, ky_k, self.gamma_4)
            v4_dag = v_i_dagger(kx_k, ky_k, self.gamma_4)

            # Fill in the matrix elements according to the Hamiltonian
            H_k[0, 0] = 2 * self.u_d
            H_k[0, 1] = v0_dag
            H_k[0, 2] = v4_dag
            H_k[0, 3] = v3
            H_k[0, 5] = self.gamma_2 / 2

            H_k[1, 0] = v0
            H_k[1, 1] = 2 * self.u_d + self.delta
            H_k[1, 2] = self.gamma_1
            H_k[1, 3] = v4_dag

            H_k[2, 0] = v4
            H_k[2, 1] = self.gamma_1
            H_k[2, 2] = self.u_d + self.u_a
            H_k[2, 3] = v0_dag
            H_k[2, 4] = v4_dag
            H_k[2, 5] = v3
            H_k[2, 7] = self.gamma_2 / 2

            H_k[3, 0] = v3_dag
            H_k[3, 1] = v4
            H_k[3, 2] = v0
            H_k[3, 3] = self.u_d + self.u_a
            H_k[3, 4] = self.gamma_1
            H_k[3, 5] = v4_dag

            H_k[4, 2] = v4
            H_k[4, 3] = self.gamma_1
            H_k[4, 4] = self.u_a
            H_k[4, 5] = v0_dag
            H_k[4, 6] = v4_dag
            H_k[4, 7] = v3
            H_k[4, 9] = self.gamma_2 / 2

            H_k[5, 0] = self.gamma_2 / 2
            H_k[5, 2] = v3_dag
            H_k[5, 3] = v4
            H_k[5, 4] = v0
            H_k[5, 5] = self.u_a
            H_k[5, 6] = self.gamma_1
            H_k[5, 7] = v4_dag

            H_k[6, 4] = v4
            H_k[6, 5] = self.gamma_1
            H_k[6, 6] = -self.u_d + self.u_a
            H_k[6, 7] = v0_dag
            H_k[6, 8] = v4_dag
            H_k[6, 9] = v3

            H_k[7, 2] = self.gamma_2 / 2
            H_k[7, 5] = v3_dag
            H_k[7, 6] = v0
            H_k[7, 7] = -self.u_d + self.u_a
            H_k[7, 8] = self.gamma_1
            H_k[7, 9] = v4_dag

            H_k[8, 6] = v4
            H_k[8, 7] = self.gamma_1
            H_k[8, 8] = -2 * self.u_d + self.delta
            H_k[8, 9] = v0_dag

            H_k[9, 4] = self.gamma_2 / 2
            H_k[9, 6] = v3_dag
            H_k[9, 7] = v4
            H_k[9, 8] = v0
            H_k[9, 9] = -2 * self.u_d

            # Ensure Hermiticity
            H_k = H_k + H_k.conj().T - np.diag(H_k.diagonal())

            # Reshape H_k to (D, D) shape
            H_nonint[:, :, idx_k] = H_k.reshape(self.D + self.D)

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = self.expand(exp_val)  # Shape (D1, D2, N_k)
        N_k = self.N_k
        D = self.D
        H_int = np.zeros(D + D + (N_k,), dtype=np.complex128)

        # Compute the interaction potential V_C(q)
        # V_C(q) = (e^2) / (2 * epsilon_0 * epsilon * q) * tanh(q * d_s)
        # Compute q vectors (difference between k and k')
        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]

        kx_grid_k = kx[:, np.newaxis]  # Shape (N_k, 1)
        ky_grid_k = ky[:, np.newaxis]
        kx_grid_kp = kx[np.newaxis, :]
        ky_grid_kp = ky[np.newaxis, :]

        qx = kx_grid_k - kx_grid_kp  # Shape (N_k, N_k)
        qy = ky_grid_k - ky_grid_kp
        q = np.sqrt(qx**2 + qy**2) + 1e-10  # Avoid division by zero

        V_C_q = (e**2) / (2 * epsilon_0 * self.epsilon * q) * np.tanh(q * self.d_s)

        # Normalize interaction potential
        V_C_q /= self.N_k

        # Compute H_int[μ, ν, k] = sum_{k'} V_C_q[k, k'] * exp_val[μ, ν, k']
        for mu in range(10):
            for nu in range(10):
                # exp_val[mu, nu, :] is shape (N_k,)
                # H_int[mu, nu, :] = V_C_q.dot(exp_val[mu, nu, :])
                H_int_mu_nu = np.dot(V_C_q, exp_val[mu, nu, :])  # Shape (N_k,)
                H_int_mu_nu = H_int_mu_nu  # Interaction term for (mu, nu)
                H_int[mu // 2, mu % 2, nu // 2, nu % 2, :] = H_int_mu_nu

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total  # Shape: D1, D2, N_k

    def flatten(self, ham):
        return ham.reshape((np.prod(self.D), np.prod(self.D), self.N_k))

    def expand(self, exp_val):
        return exp_val.reshape(self.D + self.D + (self.N_k,))
