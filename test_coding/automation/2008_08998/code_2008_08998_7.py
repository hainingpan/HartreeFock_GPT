import numpy as np
from typing import Any
from HF import *

class HartreeFockHamiltonian:
  """
  Hartree-Fock Hamiltonian for a triangular lattice with spin and folded Brillouin zone.
  
  Args:
    N_shell (int): Number of shells in the first Brillouin zone.
    parameters (dict): Dictionary containing model parameters.
    filling_factor (float): Filling factor for the system, default is 0.5.
  """
  def __init__(self, N_shell: int, parameters: dict[str, Any]={}, filling_factor: float=0.5):
    self.lattice = 'triangular'
    self.D = (2, 3)  # (|spin|, |reciprocal_lattice_vector|)
    self.basis_order = {'0': 'spin', '1': 'reciprocal_lattice_vector'}
    # Order for each flavor:
    # 0: spin up, spin down
    # 1: Gamma point, K point, K' point
    
    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters.get('T', 0.0)  # temperature, default to 0
    self.a = parameters.get('a', 1.0)  # Lattice constant
    self.primitive_vectors = get_primitive_vectors_triangle(self.a)
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]
    
    # Hopping parameters
    self.t1 = parameters.get('t1', 6.0)  # Nearest-neighbor hopping in meV
    self.t2 = parameters.get('t2', 1.0)  # Next-nearest-neighbor hopping in meV
    
    # Interaction parameters
    self.epsilon_r = parameters.get('epsilon_r', 1.0)  # Relative dielectric constant
    self.d = parameters.get('d', 10.0)  # Screening length in nm
    self.coulomb_const = 1440.0  # e²/ε₀ in meV·nm
    self.u_onsite = 1000.0 / self.epsilon_r  # Onsite interaction in meV
    
    return

  def get_nearest_neighbor_vectors(self):
    """
    Returns the integer coordinate offsets (n1, n2) corresponding to the 
    nearest neighbors in a 2D triangular Bravais lattice. These offsets are ONLY
    valid for a lattice whose two primitive vectors are separated by 120°.

    To obtain the real-space displacements for each neighbor, multiply these 
    integer pairs by the primitive vectors a1 and a2, i.e.:
        R_neighbor = n1 * a1 + n2 * a2

    For a 2D triangular lattice, there are six nearest neighbors, given by:
    """
    n_vectors = [
        (1, 0),
        (0, 1),
        (1, 1),
        (-1, 0),
        (0, -1),
        (-1, -1),
    ]
    return n_vectors
    
  def get_next_nearest_neighbor_vectors(self):
    """
    Returns the integer coordinate offsets for next-nearest neighbors in a triangular lattice.
    """
    nn_vectors = [
        (2, 0),
        (0, 2),
        (2, 2),
        (-2, 0),
        (0, -2),
        (-2, -2),
        (1, -1),
        (-1, 1),
        (2, 1),
        (1, 2),
        (-1, -2),
        (-2, -1),
    ]
    return nn_vectors
    
  def get_q_point(self, q_idx):
    """
    Returns the q-point (reciprocal lattice vector) based on the index.
    
    Args:
        q_idx (int): Index of the q-point (0: Gamma, 1: K, 2: K')
        
    Returns:
        np.ndarray: The q-point vector
    """
    high_sym_points = generate_high_symmtry_points(self.lattice, self.a)
    if q_idx == 0:
        return high_sym_points["Gamma"]
    elif q_idx == 1:
        return high_sym_points["K"]
    else:  # q_idx == 2
        return high_sym_points["K'"]
    
  def calculate_interaction_potential(self, q_vec):
    """
    Calculate the interaction potential U(q) in momentum space.
    
    Args:
        q_vec (np.ndarray): Momentum vector
        
    Returns:
        float: Interaction potential value
    """
    q_mag = np.linalg.norm(q_vec)
    if q_mag < 1e-10:  # q≈0 case
        return self.u_onsite
    else:
        # Screened Coulomb potential in momentum space
        return self.coulomb_const / (self.epsilon_r * (q_mag + 1/self.d))
    
  def is_momentum_conserved(self, q_alpha, q_beta, q_gamma, q_delta):
    """
    Checks if momentum is conserved up to a reciprocal lattice vector.
    
    Args:
        q_alpha, q_beta, q_gamma, q_delta: Indices of q-points
        
    Returns:
        bool: True if momentum is conserved, False otherwise
    """
    q_alpha_vec = self.get_q_point(q_alpha)
    q_beta_vec = self.get_q_point(q_beta)
    q_gamma_vec = self.get_q_point(q_gamma)
    q_delta_vec = self.get_q_point(q_delta)
    
    # Check if q_alpha + q_beta - q_gamma - q_delta is close to a reciprocal lattice vector
    diff_vec = q_alpha_vec + q_beta_vec - q_gamma_vec - q_delta_vec
    
    # Get reciprocal lattice vectors
    recip_vecs = get_reciprocal_vectors_triangle(self.a)
    
    # Check if diff_vec is close to any linear combination of reciprocal lattice vectors
    for i in range(-1, 2):
        for j in range(-1, 2):
            G = i * recip_vecs[0] + j * recip_vecs[1]
            if np.allclose(diff_vec, G, atol=1e-10):
                return True
    
    return False
    
  def generate_non_interacting(self) -> np.ndarray:
    """
    Generates the non-interacting part of the Hamiltonian.

    Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    """
    H_nonint = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # Get neighbor vectors
    nn_vectors = self.get_nearest_neighbor_vectors()
    nnn_vectors = self.get_next_nearest_neighbor_vectors()
    
    # For each k-point (p in our notation)
    for k_idx in range(self.N_k):
        p = self.k_space[k_idx]
        
        # For each spin and q
        for s in range(2):  # spin
            for q_alpha in range(3):  # q_alpha (Gamma, K, K')
                for q_beta in range(3):  # q_beta (Gamma, K, K')
                    # Initialize hopping terms
                    hopping_term = 0.0
                    
                    # Get q vectors
                    q_alpha_vec = self.get_q_point(q_alpha)
                    q_beta_vec = self.get_q_point(q_beta)
                    
                    # For each real-space lattice position
                    for n_vec in nn_vectors:
                        R_n = n_vec[0] * self.primitive_vectors[0] + n_vec[1] * self.primitive_vectors[1]
                        # Nearest-neighbor hopping
                        hopping_term += self.t1 * np.exp(-1j * np.dot(p + q_beta_vec, R_n))
                    
                    for n_vec in nnn_vectors:
                        R_n = n_vec[0] * self.primitive_vectors[0] + n_vec[1] * self.primitive_vectors[1]
                        # Next-nearest-neighbor hopping
                        hopping_term += self.t2 * np.exp(-1j * np.dot(p + q_beta_vec, R_n))
                    
                    # Check if q_alpha + p = q_beta + p (i.e., q_alpha = q_beta)
                    if q_alpha == q_beta:
                        H_nonint[s, q_alpha, s, q_beta, k_idx] = -hopping_term
    
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    """
    Generates the interacting part of the Hamiltonian (Hartree-Fock).

    Args:
        exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

    Returns:
        np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    """
    exp_val = unflatten(exp_val, self.D, self.N_k)
    H_int = np.zeros((*self.D, *self.D, self.N_k), dtype=complex)
    
    # For each k-point (p_beta in our notation)
    for k_idx in range(self.N_k):
        p_beta = self.k_space[k_idx]
        
        # For each spin pair
        for s in range(2):  # spin s
            for s_prime in range(2):  # spin s'
                # For each q combination with momentum conservation
                for q_alpha in range(3):
                    for q_beta in range(3):
                        for q_gamma in range(3):
                            for q_delta in range(3):
                                # Check momentum conservation
                                if self.is_momentum_conserved(q_alpha, q_beta, q_gamma, q_delta):
                                    # For Hartree term: only affects diagonal in spin
                                    U_hartree = self.calculate_interaction_potential(
                                        self.get_q_point(q_alpha) - self.get_q_point(q_delta)
                                    )
                                    hartree_exp_val = np.mean(exp_val[s, q_alpha, s, q_delta, :])
                                    H_int[s_prime, q_beta, s_prime, q_gamma, k_idx] += \
                                        U_hartree * hartree_exp_val / self.N_k
                                    
                                    # For Fock term: exchanges spin s and s'
                                    U_fock = self.calculate_interaction_potential(
                                        p_beta + self.get_q_point(q_alpha) - p_beta - self.get_q_point(q_delta)
                                    )
                                    fock_exp_val = np.mean(exp_val[s, q_alpha, s_prime, q_gamma, :])
                                    H_int[s_prime, q_beta, s, q_delta, k_idx] -= \
                                        U_fock * fock_exp_val / self.N_k
    
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    """
    Generates the total Hartree-Fock Hamiltonian.

    Args:
        exp_val (np.ndarray): Expectation value array.
        return_flat (bool): Whether to return flattened Hamiltonian.

    Returns:
        np.ndarray: The total Hamiltonian.
    """
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if return_flat:
        return flattened(H_total, self.D, self.N_k)
    else:
        return H_total
