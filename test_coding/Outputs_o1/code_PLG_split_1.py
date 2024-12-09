# https://chatgpt.com/share/674b39ac-bb10-8011-91de-d6ca0ab38aa5
from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for rhombohedral-stacked pentalayer graphene with Coulomb interactions.

    Args:
        N_shell (int): Number of k-point shells. Defaults to 10.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int = 10, parameters: dict[str, Any] = None, filling_factor: float = 0.5):
        self.lattice = 'triangular'  # Lattice symmetry ('square' or 'triangular')
        self.D = (10,)  # Number of flavors identified
        self.basis_order = {
            '0': ['A1', 'B1', 'A2', 'B2', 'A3', 'B3', 'A4', 'B4', 'A5', 'B5']
        }

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Assume temperature T = 0
        self.a = 1.0  # Lattice constant, default value
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        self.area = (np.sqrt(3)/2) * (self.a * N_shell * 2 * np.pi / self.a)**2  # Approximate area in reciprocal space

        # Physical constants
        self.e_charge = 1.602176634e-19  # Elementary charge in Coulombs
        self.epsilon_0 = 8.8541878128e-12  # Vacuum permittivity in F/m

        # Interaction parameters
        default_parameters = {
            'gamma0': 2600.0,  # in meV
            'gamma1': 356.1,
            'gamma2': -15.0,
            'gamma3': -293.0,
            'gamma4': -144.0,
            'delta': 12.2,
            'ua': 16.4,
            'ud': 0.0,  # Default value for ud
            'valley': 1,  # +1 or -1
            'epsilon_r': 5.0,  # Relative dielectric constant
            'd_s': 30e-9,  # Gate distance in meters
        }
        if parameters is None:
            parameters = default_parameters
        else:
            # Use provided parameters, fill in defaults where necessary
            for key, value in default_parameters.items():
                parameters.setdefault(key, value)

        self.gamma0 = parameters['gamma0']
        self.gamma1 = parameters['gamma1']
        self.gamma2 = parameters['gamma2']
        self.gamma3 = parameters['gamma3']
        self.gamma4 = parameters['gamma4']
        self.delta = parameters['delta']
        self.ua = parameters['ua']
        self.ud = parameters['ud']
        self.valley = parameters['valley']
        self.epsilon_r = parameters['epsilon_r']
        self.d_s = parameters['d_s']

        # Avoid division by zero in V_C(0)
        q_min = 1e-6  # Small minimum q
        self.V_C0 = (self.e_charge**2) / (2 * self.epsilon_0 * self.epsilon_r * q_min) * np.tanh(q_min * self.d_s)

        return

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        N_k = self.N_k
        D = self.D[0]  # D = 10
        H_nonint = np.zeros((D, D, N_k), dtype=np.complex128)

        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        valley = self.valley  # +1 or -1

        # Define v_i(k)
        sqrt3_over_2 = np.sqrt(3) / 2
        phi_k = valley * kx + 1j * ky  # For valley = +1 or -1

        # Velocity terms
        v0 = sqrt3_over_2 * self.gamma0 * phi_k        # v0(k)
        v0_dag = np.conj(v0)                           # v0†(k)

        v3 = sqrt3_over_2 * self.gamma3 * phi_k        # v3(k)
        v3_dag = np.conj(v3)                           # v3†(k)

        v4 = sqrt3_over_2 * self.gamma4 * phi_k        # v4(k)
        v4_dag = np.conj(v4)                           # v4†(k)

        gamma1 = self.gamma1
        gamma2_half = self.gamma2 / 2.0  # γ2 / 2

        ua = self.ua
        ud = self.ud
        delta = self.delta

        # Now fill in the Hamiltonian matrix elements
        # Note that H is Hermitian, so H[i, j, :] = H[j, i, :].conj()

        # Diagonal elements
        H_nonint[0, 0, :] = 2 * ud
        H_nonint[1, 1, :] = 2 * ud + delta
        H_nonint[2, 2, :] = ud + ua
        H_nonint[3, 3, :] = ud + ua
        H_nonint[4, 4, :] = ua
        H_nonint[5, 5, :] = ua
        H_nonint[6, 6, :] = -ud + ua
        H_nonint[7, 7, :] = -ud + ua
        H_nonint[8, 8, :] = -2 * ud + delta
        H_nonint[9, 9, :] = -2 * ud

        # Off-diagonal elements
        # H[0,1] = v0†
        H_nonint[0, 1, :] = v0_dag
        H_nonint[1, 0, :] = v0      # H Hermitian

        # H[0,2] = v4†
        H_nonint[0, 2, :] = v4_dag
        H_nonint[2, 0, :] = v4

        # H[0,3] = v3
        H_nonint[0, 3, :] = v3
        H_nonint[3, 0, :] = v3_dag

        # H[0,5] = γ2 / 2
        H_nonint[0, 5, :] = gamma2_half
        H_nonint[5, 0, :] = gamma2_half  # γ2 is real

        # H[1,2] = γ1
        H_nonint[1, 2, :] = gamma1
        H_nonint[2, 1, :] = gamma1

        # H[1,3] = v4†
        H_nonint[1, 3, :] = v4_dag
        H_nonint[3, 1, :] = v4

        # H[2,3] = v0†
        H_nonint[2, 3, :] = v0_dag
        H_nonint[3, 2, :] = v0

        # H[2,5] = v3
        H_nonint[2, 5, :] = v3
        H_nonint[5, 2, :] = v3_dag

        # H[2,7] = γ2 / 2
        H_nonint[2, 7, :] = gamma2_half
        H_nonint[7, 2, :] = gamma2_half

        # H[3,4] = γ1
        H_nonint[3, 4, :] = gamma1
        H_nonint[4, 3, :] = gamma1

        # H[3,5] = v4†
        H_nonint[3, 5, :] = v4_dag
        H_nonint[5, 3, :] = v4

        # H[4,5] = v0†
        H_nonint[4, 5, :] = v0_dag
        H_nonint[5, 4, :] = v0

        # H[4,7] = v3
        H_nonint[4, 7, :] = v3
        H_nonint[7, 4, :] = v3_dag

        # H[4,9] = gamma2_half
        H_nonint[4, 9, :] = gamma2_half
        H_nonint[9, 4, :] = gamma2_half

        # H[5,6] = γ1
        H_nonint[5, 6, :] = gamma1
        H_nonint[6, 5, :] = gamma1

        # H[5,7] = v4†
        H_nonint[5, 7, :] = v4_dag
        H_nonint[7, 5, :] = v4

        # H[6,7] = v0†
        H_nonint[6, 7, :] = v0_dag
        H_nonint[7, 6, :] = v0

        # H[6,9] = v3
        H_nonint[6, 9, :] = v3
        H_nonint[9, 6, :] = v3_dag

        # H[7,8] = γ1
        H_nonint[7, 8, :] = gamma1
        H_nonint[8, 7, :] = gamma1

        # H[7,9] = v4†
        H_nonint[7, 9, :] = v4_dag
        H_nonint[9, 7, :] = v4

        # H[8,9] = v0†
        H_nonint[8, 9, :] = v0_dag
        H_nonint[9, 8, :] = v0

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
        N_k = self.N_k
        D = self.D[0]
        H_int = np.zeros((D, D, N_k), dtype=np.complex128)

        # Compute mean occupation numbers n_nu
        n_nu = np.mean(np.real(np.diagonal(exp_val, axis1=0, axis2=1)), axis=1)  # Shape: (D,)

        # Hartree term: V_C(0) * sum_nu n_nu * delta_mu_nu
        V_C0 = self.V_C0  # Precomputed in __init__

        # Hartree potential
        V_H = V_C0 * np.sum(n_nu)  # Scalar
        for mu in range(D):
            H_int[mu, mu, :] += V_H  # Adds to diagonal elements

        # Fock term: - (1/A) * sum_k' V_C(k - k') * exp_val[mu, nu, k']
        # Prepare interaction potential V_C(q)
        V_C_q = np.zeros((N_k, N_k), dtype=np.float64)  # Interaction matrix
        for i in range(N_k):
            for j in range(N_k):
                qx = self.k_space[i, 0] - self.k_space[j, 0]
                qy = self.k_space[i, 1] - self.k_space[j, 1]
                q = np.sqrt(qx**2 + qy**2) + 1e-6  # Avoid division by zero
                V_C = (self.e_charge**2) / (2 * self.epsilon_0 * self.epsilon_r * q) * np.tanh(q * self.d_s)
                V_C_q[i, j] = V_C

        # Compute Fock term
        for mu in range(D):
            for nu in range(D):
                # exp_val_mu_nu: Shape (N_k,)
                exp_val_mu_nu = exp_val[mu, nu, :]
                # Compute convolution over k'
                Sigma_k = np.zeros(N_k, dtype=np.complex128)
                for k_idx in range(N_k):
                    # Sum over k'
                    Sigma_k[k_idx] = np.sum(V_C_q[k_idx, :] * exp_val_mu_nu) / self.area
                # Update H_int
                H_int[nu, mu, :] -= Sigma_k  # Note indices [nu, mu, :]

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hamiltonian H_total = H_nonint + H_int.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return a flattened Hamiltonian.

        Returns:
            np.ndarray: The total Hamiltonian with shape (D_flat, D_flat, N_k) if return_flat is True,
                        or (D, D, N_k) if return_flat is False.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total

    def flatten(self, ham):
        """
        Flattens the Hamiltonian from shape (D, D, N_k) to (D_flat, D_flat, N_k).
        """
        D_flat = np.prod(self.D)
        return ham.reshape((D_flat, D_flat, self.N_k))

    def expand(self, exp_val):
        """
        Expands the expectation values from shape (D_flat, D_flat, N_k) to (D, D, N_k).
        """
        return exp_val.reshape(self.D + self.D + (self.N_k,))
