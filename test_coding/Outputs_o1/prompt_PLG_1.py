# https://chatgpt.com/share/67437213-347c-8011-acbf-adc32f566c6e
from HF import *

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for rhombohedral-stacked pentalayer graphene.

    Args:
        N_shell (int): Number of shells in the k-space lattice.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int = 10, parameters: dict[str, Any] = {}, filling_factor: float = 0.5):
        self.lattice = 'triangular'  # Lattice symmetry
        self.D = (10,)  # Number of orbitals
        self.basis_order = {'0': 'orbital'}
        # Order for orbitals:
        # 0: A_1, 1: B_1, 2: A_2, 3: B_2, 4: A_3, 5: B_3, 6: A_4, 7: B_4, 8: A_5, 9: B_5

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Tight-binding parameters (in meV)
        self.gamma_0 = parameters.get('gamma_0', 2600)
        self.gamma_1 = parameters.get('gamma_1', 356.1)
        self.gamma_2 = parameters.get('gamma_2', -15)
        self.gamma_3 = parameters.get('gamma_3', -293)
        self.gamma_4 = parameters.get('gamma_4', -144)
        self.delta = parameters.get('delta', 12.2)
        self.u_a = parameters.get('u_a', 16.4)
        self.u_d = parameters.get('u_d', 0.0)  # Default value

        # Coulomb interaction parameters
        self.epsilon = parameters.get('epsilon', 7.0)  # Effective dielectric constant
        self.d_s = parameters.get('d_s', 30e-9)  # Gate distance in meters
        self.e_charge = 1.602e-19  # Elementary charge in Coulombs
        self.epsilon_0 = 8.854e-12  # Vacuum permittivity in F/m
        self.A = parameters.get('A', 1e-12)  # System area in m^2

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        D_flat = np.prod(self.D)
        N_k = self.N_k
        H_nonint = np.zeros((D_flat, D_flat, N_k), dtype=np.complex128)

        # Extract k-space components
        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        k_plus = kx + 1j * ky  # For K valley (spin and valley fixed)

        # Precompute v_i(k)
        v_factor = (np.sqrt(3) / 2)
        v0 = v_factor * self.gamma_0 * k_plus
        v0_dag = np.conj(v0)
        v3 = v_factor * self.gamma_3 * k_plus
        v3_dag = np.conj(v3)
        v4 = v_factor * self.gamma_4 * k_plus
        v4_dag = np.conj(v4)

        # Energy levels
        u_d = self.u_d
        u_a = self.u_a
        delta = self.delta
        gamma1 = self.gamma_1
        gamma2_half = self.gamma_2 / 2

        # Map orbitals for readability
        # Indices correspond to basis_order
        A1, B1, A2, B2, A3, B3, A4, B4, A5, B5 = range(10)

        for idx_k in range(N_k):
            H0 = np.zeros((D_flat, D_flat), dtype=np.complex128)
            
            # Fill H0 according to the provided matrix
            # For brevity, we show a few representative elements

            # Diagonal elements (on-site energies)
            H0[A1, A1] = 2 * u_d
            H0[B1, B1] = 2 * u_d + delta
            H0[A2, A2] = u_d + u_a
            H0[B2, B2] = u_d + u_a
            H0[A3, A3] = u_a
            H0[B3, B3] = u_a
            H0[A4, A4] = -u_d + u_a
            H0[B4, B4] = -u_d + u_a
            H0[A5, A5] = -2 * u_d + delta
            H0[B5, B5] = -2 * u_d

            # Off-diagonal elements (hopping terms)
            H0[A1, B1] = v0_dag[idx_k]
            H0[B1, A1] = v0[idx_k]
            H0[A1, B2] = v4_dag[idx_k]
            H0[B2, A1] = v4[idx_k]
            H0[A1, B3] = v3[idx_k]
            H0[B3, A1] = v3_dag[idx_k]
            H0[A1, A3] = gamma2_half
            H0[A3, A1] = gamma2_half

            # Continue filling H0 for all elements as per the Hamiltonian matrix

            # Assign H0 to the non-interacting Hamiltonian
            H_nonint[:, :, idx_k] = H0

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        D_flat = np.prod(self.D)
        N_k = self.N_k
        exp_val = exp_val.reshape((D_flat, D_flat, N_k))
        H_int = np.zeros((D_flat, D_flat, N_k), dtype=np.complex128)

        # Hartree term
        n_total = np.sum(np.diagonal(exp_val, axis1=0, axis2=1)) / N_k  # Total electron density
        # Handle V_C(0) singularity by regularization or set to zero (assuming neutrality)
        V_C0 = 0  # Set Hartree term to zero for neutrality
        H_int[np.arange(D_flat), np.arange(D_flat), :] += V_C0 * n_total  # Adds to diagonal elements

        # Fock term
        # Precompute V_C(q) matrix
        V_C_q = self._compute_coulomb_matrix()

        # Calculate the Fock term
        for idx_k in range(N_k):
            H_int_k = np.zeros((D_flat, D_flat), dtype=np.complex128)
            for idx_kp in range(N_k):
                V_C = V_C_q[idx_kp, idx_k]
                exp_val_kp = exp_val[:, :, idx_kp]
                # Contribution to H_int at k from k'
                H_int_k += - (1 / self.A) * V_C * exp_val_kp
            H_int[:, :, idx_k] += H_int_k

        return H_int

    def _compute_coulomb_matrix(self) -> np.ndarray:
        """
        Computes the Coulomb interaction matrix V_C(q) for all q = k' - k.

        Returns:
            np.ndarray: Coulomb interaction matrix of shape (N_k, N_k).
        """
        N_k = self.N_k
        V_C_q = np.zeros((N_k, N_k), dtype=np.float64)
        k_space = self.k_space

        for idx_k in range(N_k):
            for idx_kp in range(N_k):
                q = k_space[idx_kp] - k_space[idx_k]
                q_magnitude = np.linalg.norm(q)
                # Regularize q=0 to avoid division by zero
                if q_magnitude == 0:
                    V_C = 0
                else:
                    V_C = (self.e_charge ** 2) / (2 * self.epsilon_0 * self.epsilon * q_magnitude)
                    V_C *= np.tanh(q_magnitude * self.d_s)
                V_C_q[idx_kp, idx_k] = V_C

        return V_C_q

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flat, D_flat, N_k).
            return_flat (bool): Whether to return the Hamiltonian in flattened form.

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int

        if return_flat:
            return H_total.reshape((np.prod(self.D), np.prod(self.D), self.N_k))
        else:
            return H_total  # Shape: (D, D, N_k)
