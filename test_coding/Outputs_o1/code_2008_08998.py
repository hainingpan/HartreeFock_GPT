# https://chatgpt.com/share/6738bfc5-ce4c-8011-a9c0-fbc5f148c0d6
import numpy as np
from typing import Any

def generate_k_space(lattice: str, N_shell: int, a: float):
    """
    Generates k-space points for a given lattice.
    """
    k_points = []
    for nx in range(-N_shell, N_shell + 1):
        for ny in range(-N_shell, N_shell + 1):
            if lattice == 'triangular':
                # Reciprocal lattice vectors for triangular lattice
                b1 = (2 * np.pi / a) * np.array([1.0, -1 / np.sqrt(3)])
                b2 = (2 * np.pi / a) * np.array([0.0, 2 / np.sqrt(3)])
                k = nx * b1 + ny * b2
                k_points.append(k)
            else:
                # Implement other lattices if needed
                pass
    k_space = np.array(k_points)
    return k_space

def generate_n_vectors(lattice: str, N_shell: int, a: float):
    """
    Generates lattice vectors n up to N_shell in a given lattice.
    """
    n_vectors = []
    for nx in range(-N_shell, N_shell + 1):
        for ny in range(-N_shell, N_shell + 1):
            if lattice == 'triangular':
                # Real-space lattice vectors for triangular lattice
                a1 = a * np.array([1.0, 0.0])
                a2 = a * np.array([0.5, np.sqrt(3) / 2])
                n = nx * a1 + ny * a2
                n_vectors.append(n)
            else:
                # Implement other lattices if needed
                pass
    n_vectors = np.array(n_vectors)
    return n_vectors

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a system with spin on a triangular lattice.

    Attributes:
        lattice (str): Type of lattice ('triangular').
        D (tuple): Tuple of flavors (2,), representing spin.
        basis_order (dict): Mapping of basis indices to spin states.
        nu (float): Filling factor.
        T (float): Temperature of the system.
        a (float): Lattice constant.
        N_shell (int): Number of shells in k-space.
        k_space (np.ndarray): Array of k-points.
        N_k (int): Number of k-points.
        t_s_n (callable): Function returning hopping parameter t_s(n).
        U_n (callable): Function returning interaction potential U(n).
        n_vectors (np.ndarray): Array of real-space lattice vectors n.
        epsilon_k_s (np.ndarray): Non-interacting energy dispersion ε_{k,s}.
        U_k (np.ndarray): Fourier transform of interaction potential U(k).
        U_k_diff (np.ndarray): Matrix of U(k - k') for all k and k'.
    """
    def __init__(self, N_shell: int = 1, parameters: dict[str, Any] = {}, filling_factor: float = 0.5):
        self.lattice = 'triangular'  # Lattice symmetry
        self.D = (2,)  # Number of flavors identified (spin up and down)
        self.basis_order = {'0': 'spin', 'Order': ['up', 'down']}
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0  # Temperature
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.N_shell = N_shell
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]
        
        # Model parameters
        # t_s_n should be a function t_s_n(s, n) returning t_s(n)
        self.t_s_n = parameters.get('t_s_n', lambda s, n: 1.0)  # Default hopping parameter
        # U_n should be a function U_n(n) returning U(n)
        self.U_n = parameters.get('U_n', lambda n: 1.0)  # Default interaction strength
        
        # Generate real-space lattice vectors n
        self.n_vectors = generate_n_vectors(self.lattice, N_shell, self.a)
        
        # Precompute ε_{k,s}
        self.epsilon_k_s = self.compute_epsilon_k_s()
        
        # Precompute U(k)
        self.U_k = self.compute_U_k()
        
        # Precompute U(k - k') matrix
        self.U_k_diff = self.compute_U_k_diff()
        
        return
    
    def compute_epsilon_k_s(self):
        """
        Computes ε_{k,s} = -∑_{n} t_s(n) e^{-i k ⋅ n}
        """
        epsilon_k_s = np.zeros((2, self.N_k))
        for s in range(2):  # s = 0 (up), s = 1 (down)
            for i_k, k in enumerate(self.k_space):
                sum_over_n = 0.0
                for n in self.n_vectors:
                    t_s_n = self.t_s_n(s, n)  # Hopping parameter t_s(n)
                    phase = np.exp(-1j * np.dot(k, n))
                    sum_over_n += t_s_n * phase
                epsilon_k_s[s, i_k] = -np.real(sum_over_n)
        return epsilon_k_s
    
    def compute_U_k(self):
        """
        Computes U(k) = ∑_{n} U(n) e^{-i k ⋅ n}
        """
        U_k = np.zeros(self.N_k)
        for i_k, k in enumerate(self.k_space):
            sum_over_n = 0.0
            for n in self.n_vectors:
                U_n = self.U_n(n)  # Interaction potential U(n)
                phase = np.exp(-1j * np.dot(k, n))
                sum_over_n += U_n * phase
            U_k[i_k] = np.real(sum_over_n)
        return U_k
    
    def compute_U_of_k(self, k_vector):
        """
        Computes U(k) for a given k_vector.
        """
        sum_over_n = 0.0
        for n in self.n_vectors:
            U_n = self.U_n(n)
            phase = np.exp(-1j * np.dot(k_vector, n))
            sum_over_n += U_n * phase
        U_k = np.real(sum_over_n)
        return U_k
    
    def compute_U_k_diff(self):
        """
        Computes U(k - k') for all pairs of k and k'.
        """
        N_k = self.N_k
        U_k_diff = np.zeros((N_k, N_k))
        for i in range(N_k):
            for j in range(N_k):
                k_diff = self.k_space[i] - self.k_space[j]  # Difference in k vectors
                U_k_diff[i, j] = self.compute_U_of_k(k_diff)
        return U_k_diff
    
    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
        """
        H_nonint = np.zeros((2, 2, self.N_k), dtype=np.float32)
        # Non-interacting energy dispersion for spin up and down
        H_nonint[0, 0, :] = self.epsilon_k_s[0, :]  # Spin up
        H_nonint[1, 1, :] = self.epsilon_k_s[1, :]  # Spin down
        return H_nonint
    
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = self.expand(exp_val)  # Expand to shape (2, 2, N_k)
        H_int = np.zeros((2, 2, self.N_k), dtype=np.float32)
        
        # Extract n_s(k) for each spin s
        n_s_k = np.zeros((2, self.N_k))
        for s in range(2):
            n_s_k[s, :] = exp_val[s, s, :]  # Diagonal elements (density)
        
        # Hartree term: Adds interactions proportional to the densities
        for s_prime in range(2):
            for k_index in range(self.N_k):
                sum_over_s = 0.0
                for s in range(2):
                    # Sum over all k' and spins s
                    sum_over_k_prime = np.sum(self.U_k_diff[k_index, :] * n_s_k[s, :])
                    sum_over_s += sum_over_k_prime
                H_int[s_prime, s_prime, k_index] += sum_over_s  # Adds to diagonal elements
        
        # Fock term: Adds exchange interactions
        for s_prime in range(2):
            for s in range(2):
                m_s_sprime_k = exp_val[s, s_prime, :]  # Off-diagonal elements
                for k_index in range(self.N_k):
                    # Sum over all k'
                    sum_over_k_prime = np.sum(self.U_k_diff[k_index, :] * m_s_sprime_k)
                    H_int[s_prime, s, k_index] -= sum_over_k_prime  # Adds to H[s', s, k]
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Generates the total Hartree-Fock Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).
            return_flat (bool): Whether to flatten the output Hamiltonian.

        Returns:
            np.ndarray: The total Hamiltonian.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_nonint + H_int  # Total Hamiltonian
        if return_flat:
            return self.flatten(H_total)
        else:
            return H_total  # Shape: (D1, D2, N_k)
    
    def flatten(self, H):
        """
        Flattens the Hamiltonian from shape (D1, D2, N_k) to (D_flattened, D_flattened, N_k).

        Args:
            H (np.ndarray): Hamiltonian to flatten.

        Returns:
            np.ndarray: Flattened Hamiltonian.
        """
        D_flattened = np.prod(self.D)
        return H.reshape((D_flattened, D_flattened, self.N_k))
    
    def expand(self, exp_val):
        """
        Expands exp_val from shape (D_flattened, D_flattened, N_k) to (D1, D2, N_k).

        Args:
            exp_val (np.ndarray): Expectation value array to expand.

        Returns:
            np.ndarray: Expanded expectation value array.
        """
        return exp_val.reshape(self.D + self.D + (self.N_k,))
