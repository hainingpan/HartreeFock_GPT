# https://chatgpt.com/share/673a866f-4d74-8011-b174-53f59c629b53
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a bilayer system with layer and valley degrees of freedom.

    Args:
        N_shell (int): Number of k-point shells.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int=5, parameters: dict={'hbar':1.0, 'm_b':1.0, 'm_t':1.0, 'kappa':1.0, 'e':1.0, 'd':1.0, 'epsilon':1.0, 'V0':1.0}, filling_factor: float=0.5):
        # LM Task: Define lattice type and tuple of flavors D
        self.lattice = 'triangular'
        self.D = (2, 2)  # (layer, valley)
        self.basis_order = {
            '0': 'layer. Order: bottom (0), top (1)',
            '1': 'valley. Order: +K (0), -K (1)'
        }

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = self.generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

        # Physical parameters
        self.hbar = parameters['hbar']  # Reduced Planck constant
        self.m_b = parameters['m_b']    # Mass in bottom layer
        self.m_t = parameters['m_t']    # Mass in top layer
        self.kappa = parameters['kappa']  # Offset in momentum
        self.e = parameters['e']        # Electron charge
        self.d = parameters['d']        # Screening length
        self.epsilon = parameters['epsilon']  # Dielectric constant
        self.V0 = parameters['V0']      # Interaction strength scaling

        # Potential parameters
        self.Delta_b = parameters.get('Delta_b', 0.0)  # Potential in bottom layer
        self.Delta_t = parameters.get('Delta_t', 0.0)  # Potential in top layer
        self.Delta_T_plusK = parameters.get('Delta_T_plusK', 0.0)  # Tunneling potential at +K
        self.Delta_T_minusK = parameters.get('Delta_T_minusK', 0.0)  # Tunneling potential at -K

        return

    def generate_k_space(self, lattice: str, N_shell: int, a: float):
        """
        Generates k-space points for the given lattice.

        Args:
            lattice (str): Type of lattice ('square' or 'triangular').
            N_shell (int): Number of shells.
            a (float): Lattice constant.

        Returns:
            np.ndarray: Array of k-points.
        """
        # This function is assumed to be predefined.
        # For simplicity, we'll use a placeholder here.
        k_points = np.linspace(-np.pi / a, np.pi / a, 2 * N_shell + 1)
        kx, ky = np.meshgrid(k_points, k_points)
        k_space = np.vstack([kx.ravel(), ky.ravel()]).T
        return k_space

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=np.float32)

        kx = self.k_space[:, 0]
        ky = self.k_space[:, 1]
        k = np.sqrt(kx**2 + ky**2)

        # Diagonal kinetic terms
        # Layer 0 (bottom), Valley +K (0)
        H_nonint[0, 0, 0, 0, :] = - (self.hbar**2 * k**2) / (2 * self.m_b) + self.Delta_b
        # Layer 1 (top), Valley +K (0)
        H_nonint[1, 0, 1, 0, :] = - (self.hbar**2 * (k - self.kappa)**2) / (2 * self.m_t) + self.Delta_t
        # Layer 0 (bottom), Valley -K (1)
        H_nonint[0, 1, 0, 1, :] = - (self.hbar**2 * k**2) / (2 * self.m_b) + self.Delta_b
        # Layer 1 (top), Valley -K (1)
        H_nonint[1, 1, 1, 1, :] = - (self.hbar**2 * (k + self.kappa)**2) / (2 * self.m_t) + self.Delta_t

        # Off-diagonal potential terms
        # Tunneling between bottom and top layers at +K
        H_nonint[0, 0, 1, 0, :] = self.Delta_T_plusK
        H_nonint[1, 0, 0, 0, :] = np.conj(self.Delta_T_plusK)
        # Tunneling between bottom and top layers at -K
        H_nonint[0, 1, 1, 1, :] = self.Delta_T_minusK
        H_nonint[1, 1, 0, 1, :] = np.conj(self.Delta_T_minusK)

        return H_nonint

    def compute_interaction_potential(self, q):
        """
        Computes the interaction potential V(q).

        Args:
            q (np.ndarray): Momentum transfer vector.

        Returns:
            float: Interaction potential value.
        """
        q_magnitude = np.linalg.norm(q, axis=-1)
        V_q = (2 * np.pi * self.e**2 * np.tanh(q_magnitude * self.d)) / (self.epsilon * q_magnitude + 1e-12)
        return V_q * self.V0  # Scale by interaction strength

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = self.expand(exp_val)  # Shape: (D, D, N_k)
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=np.float32)

        # Hartree Term (Diagonal in indices)
        # Sum over l1, tau1 (layer and valley)
        for l1 in range(self.D[0]):
            for tau1 in range(self.D[1]):
                # Compute density n_{l1, tau1}(k)
                n_lt = exp_val[l1, tau1, l1, tau1, :]

                # Hartree potential contributed to all other states
                for l2 in range(self.D[0]):
                    for tau2 in range(self.D[1]):
                        # Skip if same indices to avoid double counting
                        # Compute difference in k-space (assuming periodic boundary conditions)
                        q = self.k_space[:, np.newaxis, :] - self.k_space[np.newaxis, :, :]
                        V_q = self.compute_interaction_potential(q)

                        # Mean-field approximation: integrate over k'
                        n_mean = np.mean(n_lt)
                        H_int[l2, tau2, l2, tau2, :] += (n_mean / self.N_k) * V_q.diagonal()

        # Fock Term (Off-diagonal in indices)
        for l1 in range(self.D[0]):
            for tau1 in range(self.D[1]):
                for l2 in range(self.D[0]):
                    for tau2 in range(self.D[1]):
                        # Skip if same indices (already considered in Hartree term)
                        if (l1 == l2) and (tau1 == tau2):
                            continue
                        # Compute exchange term
                        # Compute difference in k-space
                        q = self.k_space[:, np.newaxis, :] - self.k_space[np.newaxis, :, :]
                        V_q = self.compute_interaction_potential(q)

                        # Mean-field approximation: integrate over k'
                        exp_exchange = exp_val[l1, tau1, l2, tau2, :]
                        exchange_mean = np.mean(exp_exchange)
                        H_int[l2, tau2, l1, tau1, :] -= (exchange_mean / self.N_k) * V_q.diagonal()

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool, optional): If True, returns a flattened Hamiltonian. Defaults to True.

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total  # Shape: (l1, tau1, l2, tau2, N_k)

    def flatten(self, ham):
        """
        Flattens the Hamiltonian from shape (D, D, N_k) to (D_flattened, D_flattened, N_k).

        Args:
            ham (np.ndarray): Hamiltonian array with shape (D, D, N_k).

        Returns:
            np.ndarray: Flattened Hamiltonian.
        """
        D_flat = np.prod(self.D)
        return ham.reshape((D_flat, D_flat, self.N_k))

    def expand(self, exp_val):
        """
        Expands the expectation values from shape (D_flattened, D_flattened, N_k) to (D, D, N_k).

        Args:
            exp_val (np.ndarray): Expectation value array.

        Returns:
            np.ndarray: Expanded expectation values.
        """
        return exp_val.reshape(self.D + self.D + (self.N_k,))
