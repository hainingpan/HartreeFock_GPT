# https://chatgpt.com/share/674853d8-a7c4-800e-aaf8-048fb6600043

import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a bilayer system with valley degrees of freedom.

    Args:
        N_shell (int): Number of shells in k-space.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=10, parameters: dict[str, Any]=None, filling_factor: float=0.5):
        if parameters is None:
            parameters = {}
        self.lattice = 'triangular'  # Given lattice type
        self.D = (2, 2)  # Number of flavors: (layer, valley)
        self.basis_order = {
            '0': 'layer. Order: bottom (b), top (t)',
            '1': 'valley. Order: +K, -K'
        }

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature assumed to be zero
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = self.generate_k_space(N_shell)
        self.N_k = self.k_space.shape[0]

        # Physical constants and model parameters
        self.hbar = parameters.get('hbar', 1.0)  # Planck's constant (set to 1 for simplicity)
        self.m_b = parameters.get('m_b', 1.0)  # Effective mass for bottom layer
        self.m_t = parameters.get('m_t', 1.0)  # Effective mass for top layer
        self.kappa = parameters.get('kappa', 0.1)  # Wavevector shift

        # Potential parameters
        self.Delta_b = parameters.get('Delta_b', 0.0)  # Potential in bottom layer
        self.Delta_t = parameters.get('Delta_t', 0.0)  # Potential in top layer
        self.Delta_T_plusK = parameters.get('Delta_T_plusK', 0.0)
        self.Delta_T_minusK = parameters.get('Delta_T_minusK', 0.0)

        # Interaction parameters
        self.e = parameters.get('e', 1.0)  # Electron charge (set to 1 for simplicity)
        self.d = parameters.get('d', 1.0)  # Screening length
        self.epsilon = parameters.get('epsilon', 1.0)  # Dielectric constant
        self.V0 = parameters.get('V0', 1.0)  # Overall interaction strength

        return

    def generate_k_space(self, N_shell):
        """
        Generates k-space points for the triangular lattice.

        Args:
            N_shell (int): Number of shells in k-space.

        Returns:
            np.ndarray: Array of k-points.
        """
        # Placeholder implementation for k-space generation
        # Replace with actual implementation as needed
        kx = np.linspace(-np.pi, np.pi, N_shell)
        ky = np.linspace(-np.pi, np.pi, N_shell)
        kx, ky = np.meshgrid(kx, ky)
        k_points = np.column_stack((kx.flatten(), ky.flatten()))
        return k_points

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=np.float32)
        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        k = np.sqrt(kx**2 + ky**2)  # Magnitude of k

        # Kinetic energy terms
        # H[bottom, +K, bottom, +K, :] = - (ħ² k²) / (2 m_b)
        H_nonint[0, 0, 0, 0, :] = - (self.hbar ** 2) * (k ** 2) / (2 * self.m_b) + self.Delta_b
        # H[top, +K, top, +K, :] = - (ħ² (k - κ)²) / (2 m_t)
        H_nonint[1, 0, 1, 0, :] = - (self.hbar ** 2) * ((k - self.kappa) ** 2) / (2 * self.m_t) + self.Delta_t
        # H[bottom, -K, bottom, -K, :] = - (ħ² k²) / (2 m_b)
        H_nonint[0, 1, 0, 1, :] = - (self.hbar ** 2) * (k ** 2) / (2 * self.m_b) + self.Delta_b
        # H[top, -K, top, -K, :] = - (ħ² (k + κ)²) / (2 m_t)
        H_nonint[1, 1, 1, 1, :] = - (self.hbar ** 2) * ((k + self.kappa) ** 2) / (2 * self.m_t) + self.Delta_t

        # Potential terms (off-diagonal elements)
        # H[bottom, +K, top, +K, :] = Δ_{T,+K}
        H_nonint[0, 0, 1, 0, :] = self.Delta_T_plusK
        # H[top, +K, bottom, +K, :] = Δ_{T,+K}^*
        H_nonint[1, 0, 0, 0, :] = np.conj(self.Delta_T_plusK)
        # H[bottom, -K, top, -K, :] = Δ_{T,-K}
        H_nonint[0, 1, 1, 1, :] = self.Delta_T_minusK
        # H[top, -K, bottom, -K, :] = Δ_{T,-K}^*
        H_nonint[1, 1, 0, 1, :] = np.conj(self.Delta_T_minusK)

        return H_nonint

    def V_q(self, q):
        """
        Interaction potential V(q).

        Args:
            q (np.ndarray): Momentum transfer vector.

        Returns:
            np.ndarray: Interaction potential.
        """
        q_magnitude = np.linalg.norm(q, axis=-1) + 1e-8  # Avoid division by zero
        V_q = (2 * np.pi * self.e ** 2 * np.tanh(q_magnitude * self.d)) / (self.epsilon * q_magnitude)
        return V_q

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = self.expand(exp_val)  # Shape: (2, 2, 2, 2, N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=np.float32)

        # Hartree term: diagonal in indices
        # Summing over l1, τ1, q1, q4
        n_total = np.sum(exp_val, axis=(0, 1, 2, 3))  # Total density
        V_Hartree = self.V0 * n_total  # Simplified Hartree potential

        # Adding Hartree potential to diagonal elements
        for l2 in range(2):
            for tau2 in range(2):
                # H_int[l2, tau2, l2, tau2, :] += V_Hartree
                H_int[l2, tau2, l2, tau2, :] += V_Hartree  # Hartree term

        # Fock term: off-diagonal in indices
        # For simplicity, we consider only diagonal elements in k-space
        for l1 in range(2):
            for tau1 in range(2):
                for l2 in range(2):
                    for tau2 in range(2):
                        # Fock exchange term
                        Fock_exchange = -self.V0 * exp_val[l1, tau1, l2, tau2, :]  # Fock term
                        H_int[l2, tau2, l1, tau1, :] += Fock_exchange

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool, optional): Whether to flatten the Hamiltonian. Defaults to True.

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total  # Shape: (2, 2, 2, 2, N_k)

    def flatten(self, H):
        """
        Flattens the Hamiltonian from (D1, D2, D1, D2, N_k) to (D_flat, D_flat, N_k).

        Args:
            H (np.ndarray): Hamiltonian array.

        Returns:
            np.ndarray: Flattened Hamiltonian.
        """
        D_flat = np.prod(self.D)
        return H.reshape(D_flat, D_flat, self.N_k)

    def expand(self, exp_val):
        """
        Expands the expectation values to the shape (D1, D2, D1, D2, N_k).

        Args:
            exp_val (np.ndarray): Flattened expectation values.

        Returns:
            np.ndarray: Expanded expectation values.
        """
        return exp_val.reshape(self.D + self.D + (self.N_k,))
