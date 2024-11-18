# https://chatgpt.com/share/6738d9db-9144-800e-a797-7c917c4b574e
import numpy as np
from typing import Any, Dict

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system defined on a triangular lattice with spin and orbital degrees of freedom.

    Args:
        N_shell (int): Number of k-point shells to consider.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int = 5, parameters: Dict[str, Any] = None, filling_factor: float = 0.5):
        self.lattice = 'triangular'  # Lattice type

        # Tuple of flavors
        self.D = (2, 3)  # (Spin, Orbital)

        # Basis order
        self.basis_order = {
            '0': 'Spin. Order: 0: up, 1: down',
            '1': 'Orbital index. Order: 0, 1, 2'
        }

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature is assumed to be zero
        self.parameters = parameters if parameters is not None else {}
        self.a = self.parameters.get('a', 2.46)  # Lattice constant (in Ångströms)
        self.k_space = self.generate_k_space(N_shell)
        self.N_k = self.k_space.shape[0]
        self.A = self.compute_area()  # Total area of the system

        # Model parameters
        self.gamma_0 = self.parameters.get('gamma_0', 1.0)
        self.gamma_1 = self.parameters.get('gamma_1', 0.1)
        self.gamma_2 = self.parameters.get('gamma_2', 0.05)
        self.gamma_3 = self.parameters.get('gamma_3', 0.05)
        self.gamma_N = self.parameters.get('gamma_N', 0.01)
        # Interaction strengths
        self.U_H0 = self.parameters.get('U_H0', 1.0)  # Hartree interaction for same λ
        self.U_H1 = self.parameters.get('U_H1', 0.5)  # Hartree interaction for different λ
        self.U_X0 = self.parameters.get('U_X0', 1.0)  # Exchange interaction strength
        self.q0 = self.parameters.get('q0', 1.0)      # Exchange interaction decay parameter

    def generate_k_space(self, N_shell: int) -> np.ndarray:
        """
        Generates k-space points for the triangular lattice.

        Args:
            N_shell (int): Number of k-point shells.

        Returns:
            np.ndarray: Array of k-space points.
        """
        # Simplified k-space generation for a triangular lattice
        kx = np.linspace(-np.pi / self.a, np.pi / self.a, 2 * N_shell + 1)
        ky = np.linspace(-np.pi / self.a, np.pi / self.a, 2 * N_shell + 1)
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_space = np.vstack([kx_grid.ravel(), ky_grid.ravel()]).T
        return k_space

    def compute_area(self) -> float:
        """
        Computes the total area of the system.

        Returns:
            float: Total area.
        """
        # For a triangular lattice, area per unit cell is (sqrt(3)/2) * a^2
        area_per_cell = (np.sqrt(3) / 2) * self.a ** 2
        return area_per_cell * self.N_k

    def f_function(self, k: np.ndarray) -> np.ndarray:
        """
        Computes the function f(k) used in the Hamiltonian.

        Args:
            k (np.ndarray): k-space points.

        Returns:
            np.ndarray: Values of f(k) for each k-point.
        """
        a = self.a
        kx = k[:, 0]
        ky = k[:, 1]
        exp_factor = np.exp(1j * ky * a / np.sqrt(3))
        cos_term = np.cos(kx * a / 2)
        f_k = exp_factor * (1 + 2 * np.exp(-1j * 3 * ky * a / (2 * np.sqrt(3))) * cos_term)
        return f_k

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D_spin, D_orbital, D_spin, D_orbital, N_k).
        """
        N_k = self.N_k
        D_spin, D_orbital = self.D
        H_nonint = np.zeros((D_spin, D_orbital, D_spin, D_orbital, N_k), dtype=complex)

        # Compute f(k) for all k-points
        f_k = self.f_function(self.k_space)
        f_k_conj = np.conj(f_k)

        # Filling in the Hamiltonian matrix elements based on H_0
        for k_idx in range(N_k):
            # Non-zero elements from H_0
            # Spin up (σ = 0)
            H_nonint[0, 0, 0, 1, k_idx] = -self.gamma_0 * f_k[k_idx]         # γ₀ f(k)
            H_nonint[0, 1, 0, 0, k_idx] = -self.gamma_0 * f_k_conj[k_idx]    # γ₀ f*(k)
            H_nonint[0, 1, 0, 2, k_idx] = -self.gamma_1                      # γ₁
            H_nonint[0, 2, 0, 1, k_idx] = -self.gamma_1                      # γ₁
            H_nonint[0, 0, 1, 0, k_idx] = -self.gamma_3 * f_k_conj[k_idx] - self.gamma_N  # γ₃ f*(k) + γ_N
            H_nonint[0, 0, 1, 2, k_idx] = -self.gamma_2                      # γ₂
            H_nonint[0, 2, 1, 2, k_idx] = -self.gamma_3 * f_k_conj[k_idx]    # γ₃ f*(k)

            # Spin down (σ = 1)
            H_nonint[1, 0, 1, 1, k_idx] = -self.gamma_0 * f_k[k_idx]         # γ₀ f(k)
            H_nonint[1, 1, 1, 0, k_idx] = -self.gamma_0 * f_k_conj[k_idx]    # γ₀ f*(k)
            H_nonint[1, 1, 1, 2, k_idx] = -self.gamma_1                      # γ₁
            H_nonint[1, 2, 1, 1, k_idx] = -self.gamma_1                      # γ₁
            H_nonint[1, 0, 0, 0, k_idx] = -self.gamma_3 * f_k[k_idx] - np.conj(self.gamma_N)  # γ₃ f(k) + γ_N*
            H_nonint[1, 2, 0, 0, k_idx] = -self.gamma_2                      # γ₂
            H_nonint[1, 2, 0, 2, k_idx] = -self.gamma_3 * f_k[k_idx]         # γ₃ f(k)

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D_spin, D_orbital, D_spin, D_orbital, N_k).
        """
        D_spin, D_orbital = self.D
        N_k = self.N_k
        exp_val = self.expand(exp_val)  # Shape: (D_spin, D_orbital, D_spin, D_orbital, N_k)
        H_int = np.zeros((D_spin, D_orbital, D_spin, D_orbital, N_k), dtype=complex)

        # Compute n_{λ'} = sum over k' of exp_val[σ', l', σ', l', k'] / N_k
        n_lambda = np.sum(np.diagonal(exp_val, axis1=0, axis2=2), axis=(2, 3)) / N_k  # Shape: (D_spin, D_orbital)

        # Hartree term
        for σ in range(D_spin):
            for l in range(D_orbital):
                for σ_prime in range(D_spin):
                    for l_prime in range(D_orbital):
                        U_H = self.compute_U_H((σ, l), (σ_prime, l_prime))
                        n_lp = n_lambda[σ_prime, l_prime]
                        # Add Hartree term to diagonal elements
                        H_int[σ, l, σ, l, :] += U_H * n_lp

        # Exchange term
        for k_idx in range(N_k):
            k = self.k_space[k_idx]
            for σ in range(D_spin):
                for l in range(D_orbital):
                    for σ_prime in range(D_spin):
                        for l_prime in range(D_orbital):
                            # Compute delta_k = k' - k for all k'
                            delta_k = self.k_space - k  # Shape: (N_k, 2)
                            U_X = self.compute_U_X((σ, l), (σ_prime, l_prime), delta_k)  # Shape: (N_k,)
                            p_lambda = exp_val[σ_prime, l_prime, σ, l, :]  # Shape: (N_k,)
                            # Sum over k'
                            exchange_sum = np.sum(U_X * p_lambda) / N_k
                            # Subtract exchange term
                            H_int[σ, l, σ_prime, l_prime, k_idx] -= exchange_sum

        return H_int

    def compute_U_H(self, λ: tuple, λ_prime: tuple) -> float:
        """
        Computes the Hartree Coulomb integral U_H^{λ λ'}.

        Args:
            λ (tuple): (σ, l) indices for the first state.
            λ_prime (tuple): (σ', l') indices for the second state.

        Returns:
            float: The Hartree integral U_H^{λ λ'}.
        """
        if λ == λ_prime:
            return self.U_H0
        else:
            return self.U_H1

    def compute_U_X(self, λ: tuple, λ_prime: tuple, delta_k: np.ndarray) -> np.ndarray:
        """
        Computes the Exchange Coulomb integral U_X^{λ λ'}(q).

        Args:
            λ (tuple): (σ, l) indices for the first state.
            λ_prime (tuple): (σ', l') indices for the second state.
            delta_k (np.ndarray): Momentum transfer vectors q = k' - k.

        Returns:
            np.ndarray: Exchange integrals for each q.
        """
        q_magnitude = np.linalg.norm(delta_k, axis=1)
        U_X = self.U_X0 * np.exp(-q_magnitude ** 2 / self.q0 ** 2)
        return U_X

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool, optional): If True, returns the flattened Hamiltonian.

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int

        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total  # Shape: (D_spin, D_orbital, D_spin, D_orbital, N_k)

    def flatten(self, ham: np.ndarray) -> np.ndarray:
        """
        Flattens the Hamiltonian to shape (D_flattened, D_flattened, N_k).

        Args:
            ham (np.ndarray): Hamiltonian array.

        Returns:
            np.ndarray: Flattened Hamiltonian.
        """
        D_flattened = np.prod(self.D)
        return ham.reshape(D_flattened, D_flattened, self.N_k)

    def expand(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Expands the expectation values to shape (D_spin, D_orbital, D_spin, D_orbital, N_k).

        Args:
            exp_val (np.ndarray): Flattened expectation value array.

        Returns:
            np.ndarray: Expanded expectation values.
        """
        D_spin, D_orbital = self.D
        return exp_val.reshape(D_spin, D_orbital, D_spin, D_orbital, self.N_k)

# Example usage:
# parameters = {
#     'a': 2.46,
#     'gamma_0': 1.0,
#     'gamma_1': 0.1,
#     'gamma_2': 0.05,
#     'gamma_3': 0.05,
#     'gamma_N': 0.01,
#     'U_H0': 1.0,
#     'U_H1': 0.5,
#     'U_X0': 1.0,
#     'q0': 1.0
# }
# hf_hamiltonian = HartreeFockHamiltonian(N_shell=5, parameters=parameters)
# exp_val = np.random.rand(np.prod(hf_hamiltonian.D) ** 2, hf_hamiltonian.N_k)  # Placeholder for actual exp_val
# H_total = hf_hamiltonian.generate_Htotal(exp_val)
