# https://chatgpt.com/share/6748516c-8e94-800e-ad8b-bcdb565a66e6
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for twisted bilayer graphene projected onto valley K.

    Args:
        N_G (int): Number of reciprocal lattice vectors included in the plane-wave expansion.
        N_k (int): Number of k-points in the Brillouin zone.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.

    Attributes:
        lattice (str): Lattice symmetry ('square' or 'triangular'). Defaults to 'triangular'.
        D (tuple): Tuple of flavors.
        basis_order (dict): Basis order mapping.
    """

    def __init__(self, N_G: int=10, N_k: int=10, parameters: dict=None, filling_factor: float=0.5):
        self.lattice = 'triangular'
        self.D = (4, N_G)  # 4 orbitals (layer and sublattice combinations), N_G reciprocal lattice vectors
        self.basis_order = {
            '0': 'Orbital (Layer and Sublattice)',
            'Orbital Order': {
                '0': '(Layer: Top, Sublattice: A)',
                '1': '(Layer: Top, Sublattice: B)',
                '2': '(Layer: Bottom, Sublattice: A)',
                '3': '(Layer: Bottom, Sublattice: B)'
            },
            '1': 'Reciprocal Lattice Vector G'
        }

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature
        self.a = parameters.get('a', 2.46e-10)  # Lattice constant of monolayer graphene (in meters)
        self.theta = parameters.get('theta', 1.05)  # Twist angle (in degrees)
        self.a_M = self.a / (2 * np.sin(np.deg2rad(self.theta / 2)))  # Moiré lattice constant
        self.k_space = self.generate_k_space(self.lattice, N_k)
        self.N_k = self.k_space.shape[0]
        self.N_G = N_G

        # Model parameters with default values
        if parameters is None:
            parameters = {}
        self.hbar = parameters.get('hbar', 1.0545718e-34)  # Reduced Planck constant
        self.v_D = parameters.get('v_D', 1e6)  # Dirac velocity in m/s
        self.phi = parameters.get('phi', 0)  # Phase parameter
        self.omega_0 = parameters.get('omega_0', 110e-3 * 1.60218e-19)  # Interlayer coupling parameter (in Joules)
        self.omega_1 = parameters.get('omega_1', 110e-3 * 1.60218e-19)  # Interlayer coupling parameter (in Joules)
        self.A = parameters.get('A', self.a_M**2 * np.sqrt(3) / 2)  # Area of the Moiré unit cell

        # Interaction potential V (placeholder)
        self.V = lambda q: 1 / (np.abs(q) + 1e-10)  # Coulomb potential (simplified)

    def generate_k_space(self, lattice: str, N_k: int):
        # Placeholder function to generate k-space points
        kx = np.linspace(-np.pi / self.a_M, np.pi / self.a_M, N_k)
        ky = np.linspace(-np.pi / self.a_M, np.pi / self.a_M, N_k)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_space = np.column_stack((kx_grid.flatten(), ky_grid.flatten()))
        return k_space

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D_total, D_total, N_k).
        """
        D_total = np.prod(self.D)
        H_nonint = np.zeros((D_total, D_total, self.N_k), dtype=np.complex128)

        # Construct h_theta(k) matrices for each k-point
        h_theta_over_2 = self.h_theta(self.k_space, self.theta / 2)
        h_minus_theta_over_2 = self.h_theta(self.k_space, -self.theta / 2)

        # Construct h_T matrix
        h_T = self.h_T()

        # Map indices for layer and sublattice
        for k_index in range(self.N_k):
            # Diagonal blocks for Top and Bottom layers
            h_block = np.zeros((4, 4), dtype=np.complex128)
            h_block[0:2, 0:2] = h_theta_over_2[:, :, k_index]  # Top layer
            h_block[2:4, 2:4] = h_minus_theta_over_2[:, :, k_index]  # Bottom layer

            # Off-diagonal blocks for interlayer tunneling
            h_block[0:2, 2:4] = h_T  # Tunneling from Bottom to Top
            h_block[2:4, 0:2] = h_T.conj().T  # Tunneling from Top to Bottom

            # Expand h_block over reciprocal lattice vectors
            H_nonint[:, :, k_index] = np.kron(np.eye(self.N_G), h_block)

        return H_nonint

    def h_theta(self, k_space, theta):
        """
        Constructs the h_theta(k) matrix for each k-point.

        Args:
            k_space (np.ndarray): Array of k-points.
            theta (float): Rotation angle in degrees.

        Returns:
            np.ndarray: h_theta matrices for each k-point.
        """
        N_k = k_space.shape[0]
        h_theta_k = np.zeros((2, 2, N_k), dtype=np.complex128)
        for k_index, k in enumerate(k_space):
            # Rotate k by theta
            theta_rad = np.deg2rad(theta)
            k_rot = self.rotate_vector(k, theta_rad)
            k_mag = np.linalg.norm(k_rot)
            theta_k = np.arctan2(k_rot[1], k_rot[0])
            h = -self.hbar * self.v_D * k_mag * np.array([
                [0, np.exp(1j * (theta_k - theta_rad))],
                [np.exp(-1j * (theta_k - theta_rad)), 0]
            ])
            h_theta_k[:, :, k_index] = h
        return h_theta_k

    def h_T(self):
        """
        Constructs the interlayer tunneling matrix h_T(r).

        Returns:
            np.ndarray: h_T matrix.
        """
        # Define T_j matrices
        T_j = []
        for j in range(3):
            angle = j * self.phi
            T = self.omega_0 * np.eye(2) + \
                self.omega_1 * np.cos(angle) * np.array([[0, 1], [1, 0]]) + \
                self.omega_1 * np.sin(angle) * np.array([[0, -1j], [1j, 0]])
            T_j.append(T)

        # Sum over j
        h_T = sum(T_j)
        return h_T

    def rotate_vector(self, vec, angle):
        """
        Rotates a 2D vector by a given angle.

        Args:
            vec (np.ndarray): 2D vector.
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: Rotated vector.
        """
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        return rotation_matrix @ vec

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D_total, D_total, N_k).
        """
        D_total = np.prod(self.D)
        H_int = np.zeros((D_total, D_total, self.N_k), dtype=np.complex128)

        # Reshape exp_val to match Hamiltonian dimensions
        exp_val = exp_val.reshape((D_total, D_total, self.N_k))

        # Compute delta rho
        rho = exp_val  # Assuming exp_val contains rho
        rho_iso = self.compute_rho_iso()
        delta_rho = rho - rho_iso

        # Compute Hartree and Fock self-energies
        Sigma_H = self.compute_Sigma_H(delta_rho)
        Sigma_F = self.compute_Sigma_F(delta_rho)

        # Total self-energy
        H_int = Sigma_H + Sigma_F

        return H_int

    def compute_rho_iso(self):
        """
        Computes the density matrix of isolated layers filled up to charge neutrality.

        Returns:
            np.ndarray: rho_iso matrix.
        """
        D_total = np.prod(self.D)
        rho_iso = np.zeros((D_total, D_total, self.N_k), dtype=np.complex128)
        # Placeholder: In practice, compute rho_iso based on isolated layers
        return rho_iso

    def compute_Sigma_H(self, delta_rho):
        """
        Computes the Hartree self-energy.

        Args:
            delta_rho (np.ndarray): Difference in density matrix.

        Returns:
            np.ndarray: Hartree self-energy matrix.
        """
        D_total = np.prod(self.D)
        Sigma_H = np.zeros((D_total, D_total, self.N_k), dtype=np.complex128)

        # Hartree term contributes to diagonal elements (alpha == beta)
        for k_index in range(self.N_k):
            for alpha in range(D_total):
                # Sum over all G and k'
                delta_rho_sum = np.sum(delta_rho[:, :, k_index])
                V_q = self.V(0)  # Simplified interaction potential
                Sigma_H[alpha, alpha, k_index] += V_q * delta_rho_sum

        return Sigma_H

    def compute_Sigma_F(self, delta_rho):
        """
        Computes the Fock self-energy.

        Args:
            delta_rho (np.ndarray): Difference in density matrix.

        Returns:
            np.ndarray: Fock self-energy matrix.
        """
        D_total = np.prod(self.D)
        Sigma_F = np.zeros((D_total, D_total, self.N_k), dtype=np.complex128)

        # Fock term involves exchange interactions
        for k_index in range(self.N_k):
            V_q = self.V(0)  # Simplified interaction potential
            Sigma_F[:, :, k_index] -= V_q * delta_rho[:, :, k_index]

        return Sigma_F

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to return the flattened Hamiltonian.

        Returns:
            np.ndarray: Total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int
        if return_flat:
            D_total = np.prod(self.D)
            return H_total.reshape((D_total, D_total, self.N_k))
        else:
            return H_total  # Shape: D1, D2, N_k

