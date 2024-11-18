# https://chatgpt.com/share/6738d9db-9144-800e-a797-7c917c4b574e
import numpy as np
from typing import Any, Dict

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system defined on a triangular lattice with 4 orbitals.

    Args:
        N_shell (int): Number of k-point shells to consider.
        parameters (dict): Dictionary containing model parameters.
        filling_factor (float, optional): Filling factor. Defaults to 0.5.
    """
    def __init__(self, N_shell: int = 5, parameters: Dict[str, Any] = None, filling_factor: float = 0.5):
        # Lattice type
        self.lattice = 'triangular'  # Lattice symmetry ('square' or 'triangular')
        
        # Tuple of flavors
        self.D = (4,)  # Number of orbitals
        
        # Basis order
        self.basis_order = {'0': 'Orbital index'}
        # Order for each orbital:
        # 0: Orbital 0
        # 1: Orbital 1
        # 2: Orbital 2
        # 3: Orbital 3

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature is assumed to be zero
        self.parameters = parameters if parameters is not None else {}
        self.a = self.parameters.get('a', 1.0)  # Lattice constant
        self.k_space = self.generate_k_space(N_shell)
        self.N_k = self.k_space.shape[0]
        self.A = self.compute_area()  # Total area of the system

        # Model parameters
        self.gamma_0 = self.parameters.get('gamma_0', 1.0)  # Hopping parameter γ₀
        self.gamma_1 = self.parameters.get('gamma_1', 0.1)  # Hopping parameter γ₁
        self.gamma_3 = self.parameters.get('gamma_3', 0.05)  # Hopping parameter γ₃
        self.gamma_4 = self.parameters.get('gamma_4', 0.02)  # Hopping parameter γ₄
        self.V0 = self.parameters.get('V0', 1.0)  # Interaction strength at q=0
        self.Vq = self.parameters.get('Vq', self.default_Vq)  # Interaction as a function of q

    def generate_k_space(self, N_shell: int) -> np.ndarray:
        """
        Generates k-space points for the triangular lattice.

        Args:
            N_shell (int): Number of k-point shells.

        Returns:
            np.ndarray: Array of k-space points.
        """
        # This is a placeholder for the actual k-space generation
        # For simplicity, we create a grid in the first Brillouin zone
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
        # For a triangular lattice, the area of the primitive cell is (sqrt(3)/2) * a^2
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

    def default_Vq(self, q: np.ndarray) -> float:
        """
        Default interaction function V(q).

        Args:
            q (np.ndarray): Momentum transfer vector.

        Returns:
            float: Interaction strength for the given q.
        """
        # Simple Coulomb interaction as an example
        epsilon = 1e-2  # Small constant to avoid division by zero
        return self.V0 / (np.linalg.norm(q, axis=1) + epsilon)

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        N_k = self.N_k
        D = self.D[0]
        H_nonint = np.zeros((D, D, N_k), dtype=complex)

        # Compute f(k) for all k-points
        f_k = self.f_function(self.k_space)
        f_k_conj = np.conj(f_k)

        # Filling in the Hamiltonian matrix elements based on H_0
        H_nonint[0, 1, :] = self.gamma_0 * f_k        # γ₀ f(k)
        H_nonint[0, 2, :] = self.gamma_4 * f_k        # γ₄ f(k)
        H_nonint[0, 3, :] = self.gamma_3 * f_k_conj   # γ₃ f*(k)

        H_nonint[1, 0, :] = self.gamma_0 * f_k_conj   # γ₀ f*(k)
        H_nonint[1, 2, :] = self.gamma_1              # γ₁
        H_nonint[1, 3, :] = self.gamma_4 * f_k        # γ₄ f(k)

        H_nonint[2, 0, :] = self.gamma_4 * f_k_conj   # γ₄ f*(k)
        H_nonint[2, 1, :] = self.gamma_1              # γ₁
        H_nonint[2, 3, :] = self.gamma_0 * f_k        # γ₀ f(k)

        H_nonint[3, 0, :] = self.gamma_3 * f_k        # γ₃ f(k)
        H_nonint[3, 1, :] = self.gamma_4 * f_k_conj   # γ₄ f*(k)
        H_nonint[3, 2, :] = self.gamma_0 * f_k_conj   # γ₀ f*(k)

        return H_nonint

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        D = self.D[0]
        N_k = self.N_k
        exp_val = self.expand(exp_val)  # Shape: (D, D, N_k)
        H_int = np.zeros((D, D, N_k), dtype=complex)

        # Compute n_λ = (1/A) * sum over k of exp_val[λ, λ, k]
        n_lambda = np.sum(exp_val.diagonal(axis1=0, axis2=1), axis=1) / self.A  # Shape: (D,)

        # First term: H_int[λ', λ', k] += (V(0)/A) * sum_{λ} n_λ
        V0_term = (self.V0 / self.A) * np.sum(n_lambda)
        for λ_prime in range(D):
            H_int[λ_prime, λ_prime, :] += V0_term  # Adding to diagonal elements

        # Second term: H_int[λ', λ, k] -= (1/A) * sum over k' of exp_val[λ, λ', k'] * V(k' - k)
        for λ in range(D):
            for λ_prime in range(D):
                # Compute exp_val[λ, λ', k']
                exp_val_kp = exp_val[λ, λ_prime, :]  # Shape: (N_k,)

                # Compute V(k' - k) for all k and k'
                # This results in a matrix of shape (N_k, N_k)
                delta_k = self.k_space[:, np.newaxis, :] - self.k_space[np.newaxis, :, :]  # Shape: (N_k, N_k, 2)
                V_q = self.Vq(delta_k.reshape(-1, 2)).reshape(N_k, N_k)  # Interaction matrix

                # Compute the sum over k'
                interaction_sum = np.dot(V_q, exp_val_kp) / self.A  # Shape: (N_k,)

                # Subtract from H_int[λ', λ, k]
                H_int[λ_prime, λ, :] -= interaction_sum

        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool, optional): If True, returns the flattened Hamiltonian.

        Returns:
            np.ndarray: The total Hamiltonian with shape (D*D, N_k) if flattened, else (D, D, N_k).
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int

        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total  # Shape: (D, D, N_k)

    def flatten(self, ham: np.ndarray) -> np.ndarray:
        """
        Flattens the Hamiltonian from shape (D, D, N_k) to (D*D, N_k).

        Args:
            ham (np.ndarray): Hamiltonian array to flatten.

        Returns:
            np.ndarray: Flattened Hamiltonian.
        """
        D = self.D[0]
        return ham.reshape(D * D, self.N_k)

    def expand(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Expands the expectation values from shape (D_flattened, D_flattened, N_k) to (D, D, N_k).

        Args:
            exp_val (np.ndarray): Expectation value array to expand.

        Returns:
            np.ndarray: Expanded expectation values.
        """
        D = self.D[0]
        return exp_val.reshape(D, D, self.N_k)

    # Additional methods or helper functions can be added here as needed

# Example usage:
# parameters = {'a': 1.0, 'gamma_0': 1.0, 'gamma_1': 0.1, 'gamma_3': 0.05, 'gamma_4': 0.02, 'V0': 1.0}
# hf_hamiltonian = HartreeFockHamiltonian(N_shell=5, parameters=parameters)
# exp_val = np.random.rand(4*4, hf_hamiltonian.N_k)  # Placeholder for actual expectation values
# H_total = hf_hamiltonian.generate_Htotal(exp_val)
