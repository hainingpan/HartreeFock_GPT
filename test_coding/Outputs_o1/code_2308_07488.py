# https://chatgpt.com/share/6748585b-3158-800e-a05f-639177789e8e
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for a continuum model in the K valley with layer and spin degrees of freedom.
    
    Attributes:
        lattice (str): Lattice type, 'triangular' in this case.
        D (tuple): Tuple of flavors, representing layer and spin.
        basis_order (dict): Mapping of flavor indices to physical labels.
        nu (float): Filling factor.
        T (float): Temperature, set to 0.
        a (float): Lattice constant.
        k_space (np.ndarray): Array of k-points in the Brillouin zone.
        N_k (int): Number of k-points.
        m_star (float): Effective mass.
        hbar (float): Reduced Planck constant.
        V (float): Amplitude of intralayer moiré potential.
        w (float): Interlayer coupling strength.
        phi (float): Phase in moiré potential.
        Delta_D (float): Detuning energy.
        epsilon (float): Dielectric constant.
        e_charge (float): Elementary charge.
        d_gate (float): Gate distance.
        d (float): Layer separation.
        G_vectors (np.ndarray): Reciprocal lattice vectors.
        kappa_plus (np.ndarray): Shift vector for bottom layer.
        kappa_minus (np.ndarray): Shift vector for top layer.
    """

    def __init__(self, N_shell: int = 5, parameters: dict = None, filling_factor: float = 0.5):
        # Lattice and flavors
        self.lattice = 'triangular'
        self.D = (2, 2)  # (layer, spin)
        self.basis_order = {
            '0': 'layer. Order: bottom (b), top (t)',
            '1': 'spin. Order: up (↑), down (↓)'
        }

        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = 0.0  # Temperature is 0
        self.a = parameters.get('a', 1.0)  # Lattice constant
        self.k_space = self.generate_k_space(N_shell)
        self.N_k = self.k_space.shape[0]

        # Physical constants and parameters
        self.hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
        self.m_star = parameters.get('m_star', 0.5 * 9.10938356e-31)  # Effective mass
        self.V = parameters.get('V', 1.0)  # Intralayer moiré potential amplitude
        self.w = parameters.get('w', 0.1)  # Interlayer coupling strength
        self.phi = parameters.get('phi', 0.0)  # Phase shift
        self.Delta_D = parameters.get('Delta_D', 0.0)  # Detuning energy
        self.epsilon = parameters.get('epsilon', 4.0)  # Dielectric constant
        self.e_charge = 1.60217662e-19  # Elementary charge (C)
        self.d_gate = parameters.get('d_gate', 10e-9)  # Gate distance
        self.d = parameters.get('d', 0.5e-9)  # Layer separation

        # Reciprocal lattice vectors and shift vectors
        self.G_vectors = self.generate_reciprocal_vectors()
        self.kappa_plus = parameters.get('kappa_plus', np.array([0.0, 0.0]))
        self.kappa_minus = parameters.get('kappa_minus', np.array([0.0, 0.0]))

        return

    def generate_k_space(self, N_shell: int) -> np.ndarray:
        """
        Generates k-space points for a triangular lattice.

        Args:
            N_shell (int): Number of shells in reciprocal lattice.

        Returns:
            np.ndarray: Array of k-points.
        """
        # Implementation details would go here
        # For simplicity, we return a placeholder array
        k_points = np.random.rand(100, 2) * 2 * np.pi / self.a
        return k_points

    def generate_reciprocal_vectors(self) -> np.ndarray:
        """
        Generates reciprocal lattice vectors for the triangular lattice.

        Returns:
            np.ndarray: Array of reciprocal lattice vectors.
        """
        # Placeholder implementation
        G1 = (4 * np.pi / (self.a * np.sqrt(3))) * np.array([np.sqrt(3) / 2, 0.5])
        G2 = (4 * np.pi / (self.a * np.sqrt(3))) * np.array([0, 1])
        G3 = G1 - G2
        return np.array([G1, G2, G3])

    def generate_non_interacting(self) -> np.ndarray:
        """
        Generates the non-interacting part of the Hamiltonian.

        Returns:
            np.ndarray: Non-interacting Hamiltonian with shape (D, D, N_k).
        """
        D_layer, D_spin = self.D
        N_k = self.N_k
        H_nonint = np.zeros(self.D + self.D + (N_k,), dtype=np.float64)

        # Kinetic terms and moiré potentials
        for s in range(D_spin):
            # Spin-independent terms
            for k_idx, k_vec in enumerate(self.k_space):
                # Bottom layer (l = 0)
                delta_k_b = k_vec - self.kappa_plus
                kinetic_b = (np.linalg.norm(delta_k_b) ** 2) * (self.hbar ** 2) / (2 * self.m_star)
                Delta_b = self.calculate_moire_potential(self.G_vectors, k_vec, layer='bottom')
                H_nonint[0, s, 0, s, k_idx] = -kinetic_b + Delta_b + 0.5 * self.Delta_D

                # Top layer (l = 1)
                delta_k_t = k_vec - self.kappa_minus
                kinetic_t = (np.linalg.norm(delta_k_t) ** 2) * (self.hbar ** 2) / (2 * self.m_star)
                Delta_t = self.calculate_moire_potential(self.G_vectors, k_vec, layer='top')
                H_nonint[1, s, 1, s, k_idx] = -kinetic_t + Delta_t - 0.5 * self.Delta_D

                # Off-diagonal terms (interlayer coupling)
                Delta_T = self.calculate_interlayer_potential(k_vec)
                H_nonint[0, s, 1, s, k_idx] = Delta_T
                H_nonint[1, s, 0, s, k_idx] = np.conj(Delta_T)

        return H_nonint

    def calculate_moire_potential(self, G_vectors, k_vec, layer='bottom') -> float:
        """
        Calculates the moiré potential for a given layer and k-vector.

        Args:
            G_vectors (np.ndarray): Reciprocal lattice vectors.
            k_vec (np.ndarray): k-vector.
            layer (str): 'bottom' or 'top'.

        Returns:
            float: Moiré potential value.
        """
        phase = self.phi if layer == 'bottom' else -self.phi
        Delta = 2 * self.V * np.sum([np.cos(np.dot(G, k_vec) + phase) for G in G_vectors])
        return Delta

    def calculate_interlayer_potential(self, k_vec) -> complex:
        """
        Calculates the interlayer coupling potential.

        Args:
            k_vec (np.ndarray): k-vector.

        Returns:
            complex: Interlayer coupling potential.
        """
        Delta_T = self.w * (1 + np.exp(-1j * np.dot(self.G_vectors[1], k_vec)) +
                            np.exp(-1j * np.dot(self.G_vectors[2], k_vec)))
        return Delta_T

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interacting part of the Hamiltonian.

        Args:
            exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

        Returns:
            np.ndarray: Interacting Hamiltonian with shape (D, D, N_k).
        """
        exp_val = self.expand(exp_val)  # Shape: (2, 2, 2, 2, N_k)
        D_layer, D_spin = self.D
        N_k = self.N_k
        H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float64)

        # Compute mean occupation numbers n_{l τ}
        n_lt = np.mean(np.diagonal(exp_val, axis1=0, axis2=2), axis=-1)  # Shape: (2, 2)

        # Interaction strengths V_{ll'}
        V_ll = self.calculate_coulomb_interaction()

        # Add interaction terms to the Hamiltonian
        for l in range(D_layer):
            for s in range(D_spin):
                interaction_energy = 0.0
                for l_prime in range(D_layer):
                    for s_prime in range(D_spin):
                        interaction_energy += V_ll[l, l_prime] * n_lt[l_prime, s_prime]
                for k_idx in range(N_k):
                    H_int[l, s, l, s, k_idx] += interaction_energy

        return H_int

    def calculate_coulomb_interaction(self) -> np.ndarray:
        """
        Calculates the Coulomb interaction matrix V_{ll'} at q=0.

        Returns:
            np.ndarray: Coulomb interaction matrix V_{ll'} of shape (2, 2).
        """
        V_ll = np.zeros((2, 2))
        q = 0.0  # At q=0 for simplicity
        prefactor = (self.e_charge ** 2) / (2 * self.epsilon * 8.854187817e-12 * q if q != 0 else 1e-9)
        tanh_term = np.tanh(self.d_gate * q)
        delta_ll = np.eye(2)
        V_ll = prefactor * (tanh_term + (1 - delta_ll) * (np.exp(-self.d * q) - 1))
        return V_ll

    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
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
            return self.flatten(H_total)
        else:
            return H_total  # Shape: (l1, s1, l2, s2, k)

    def flatten(self, ham: np.ndarray) -> np.ndarray:
        """
        Flattens the Hamiltonian from (D, D, N_k) to (D_flattened, D_flattened, N_k).

        Args:
            ham (np.ndarray): Hamiltonian array.

        Returns:
            np.ndarray: Flattened Hamiltonian.
        """
        D_flat = np.prod(self.D)
        return ham.reshape((D_flat, D_flat, self.N_k))

    def expand(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Expands the expectation values from (D_flattened, D_flattened, N_k) to (D, D, N_k).

        Args:
            exp_val (np.ndarray): Expectation value array.

        Returns:
            np.ndarray: Expanded expectation value array.
        """
        return exp_val.reshape(self.D + self.D + (self.N_k,))
