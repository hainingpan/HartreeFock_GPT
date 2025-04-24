import numpy as np
from scipy import optimize


def flattened(ham: np.ndarray, N_flavor, N_k: int):
    """Flattens a Hamiltonian or expectation value tensor from high-rank to rank-3.

    Args:
      ham: Hamiltonian or expectation value tensor with shape (*N_flavor, *N_flavor, N_k).
      N_flavor: Tuple or int. The flavor dimensions to be flattened.
      N_k: Number of k-points (size of the last dimension).

    Returns:
      ndarray: Flattened tensor with shape (np.prod(N_flavor), np.prod(N_flavor), N_k).
    """
    return ham.reshape((np.product(N_flavor), np.product(N_flavor), N_k))


def unflatten(ham: np.ndarray, N_flavor, N_k: int):
    """Unflattens a Hamiltonian or expectation value tensor from rank-3 to high-rank.

    Args:
      ham: Flattened tensor with shape (np.prod(N_flavor), np.prod(N_flavor), N_k).
      N_flavor: Tuple or int. The flavor dimensions.
      N_k: Number of k-points (size of the last dimension).

    Returns:
      ndarray: High-rank tensor with shape (*N_flavor, *N_flavor, N_k).
    """
    return ham.reshape(N_flavor + N_flavor + (N_k,))


def compute_mu(en: np.ndarray, nu: float, T: float =0):
    """Compute the chemical potential based on energy levels and filling factor.
    This function calculates the chemical potential (mu) for a given energy array
    and filling factor. For zero temperature (T=0), it uses a sorting approach
    to find the Fermi level. For finite temperature (T>0), it numerically solves
    for the chemical potential that gives the desired filling factor using the
    Fermi-Dirac distribution.
    
    Args:
      en (np.ndarray): All energy levels with shape (N_level, N_k), where N_level is the number of energy levels
      nu (float): Filling factor, typically between 0 and 1.
      T (float, optional): Temperature. Default is 0.
    
    Returns:
      float: Chemical potential (mu) corresponding to the given filling factor.
    """

    if T == 0:
        flattened_en = en.flatten()

        # Sort the flattened energy array to find the Fermi level based on filling factor
        sorted_en = np.sort(flattened_en)

        # Determine the index for the Fermi level based on the filling factor
        fermi_index = min(int(np.floor(nu * len(sorted_en))), len(sorted_en) - 1)

        # Fermi energy is the energy at the Fermi index
        mu = sorted_en[fermi_index]

    else:
        # For finite temperature, we need to solve for mu numerically
        flattened_en = en.flatten()
        # Define a function to solve for mu
        def fermi_distribution_diff(mu_guess):
            occupancies = 1.0 / (1.0 + np.exp((flattened_en - mu_guess) / T))
            return occupancies.mean() - nu
        
        # Find the mu value that gives the target filling
        mu = optimize.brentq(fermi_distribution_diff, 
                np.min(flattened_en) - 10*T, 
                np.max(flattened_en) + 10*T)

    return mu

def get_occupancy(en: np.ndarray, mu: float, T: float=0):
    """Compute the occupancy of each state at each k point.

    Args:
      en: Energies with shape (N_level, N_k), level index first, then total k pts.
      mu: Chemical potential, at T=0, fermi energy.
      T: Temperature. For T=0, uses step function. For T>0, uses Fermi-Dirac distribution.

    Returns:
      occupancy: Occupancy with the same shape as `en`.
    """
    occupancy = np.zeros_like(en)  # next steps might complain if en is complex
    if T == 0:
        # Compute occupancy: 1 if energy <= Fermi level, 0 otherwise
        occupancy = (en <= mu).astype(int)
    else:
        # Fermi-Dirac distribution for finite temperature
        occupancy = 1.0 / (1.0 + np.exp((en - mu) / T))
    return occupancy


def contract_indices(wf: np.ndarray, occupancy: np.ndarray):
    """Computes the expected value using the wavefunction and occupancy.

    Args:
      wf: Wavefunction with shape (N_flavor, N_level, N_k), where the first
        index is the flavor index, the second index is the level index, and the
        third index is for different k points.
      occupancy: Occupancy of each state at each k point, with shape (N_level,
        N_k), where the first index is the level index, and the second index is
        for different k points.

    Returns:
      exp_val: The expected value with shape (N_flavor, N_k), where the first
      index is for the flavor,
              and the second index is for different k points.
    """

    # Contraction rule in words:
    # Sum over the level index (l) for each spin (s) and k-point (k),
    # multiplying occupancy (l, k), conjugate wavefunction (s, l, k), and wavefunction (S, l, k).

    # np.einsum contraction:
    exp_val = np.einsum("lk,slk,Slk->sSk", occupancy, wf.conj(), wf)
    return exp_val


def diagonalize(h_total: np.ndarray):
    """Diagonalizes the total Hamiltonian for each k point, sorts the eigenvalues and eigenvectors.

    Args:
      h_total: The total Hamiltonian with shape (N_flavor, N_flavor, N_k). If

    Returns:
      wf: Eigenvectors (wavefunctions) with shape (N_flavor, N_level, N_k), where normally, N_level=N_flavor
      en: Eigenvalues (energies) with shape (N_flavor, N_k).
    """
    if h_total.ndim != 3:
        N_flavor = h_total.shape[: (h_total.ndim - 1) // 2]
        h_total = flattened(h_total, N_flavor, h_total.shape[-1])

    N_k = h_total.shape[-1]
    D = h_total.shape[0]
    wf = np.zeros_like(h_total)  # Wavefunctions (eigenvectors).
    en = np.zeros((D, N_k))  # Eigenvalues (energies)

    # Ensure H_total is Hermitian by symmetrizing it. # Doubt: Is this a step for all problems?
    h_total_sym = (h_total + h_total.transpose((1, 0, 2)).conj()) / 2

    # Loop over each k point
    for i in range(N_k):
        # Diagonalize the DxD Hamiltonian for this k point
        vals, vecs = np.linalg.eigh(h_total_sym[:, :, i])

        # Sort eigenvalues and eigenvectors
        sort_index = np.argsort(vals)
        vals_sorted = vals[sort_index]
        vecs_sorted = vecs[:, sort_index]

        # Store the sorted eigenvalues and eigenvectors
        en[:, i] = vals_sorted
        wf[:, :, i] = vecs_sorted
    return wf, en


def get_exp_val(wf, en, nu, T):
    """Computes the expected values from the wavefunction, eigenenergies, and filling factor.

    Args:
      wf: Wavefunctions with shape (N_flavor, N_level, N_k).
      en: Eigenenergies with shape (N_level, N_k).
      nu: Filling factor. float
      T: Temperature

    Returns:
    - exp_val: numpy.ndarray: Expected values with shape (N_flavor, N_flavor, N_k).
    """
    # 1. Compute the chemical potential
    mu = compute_mu(en, nu, T)
    # 2. Compute the occupancy based on energies and chemical potential
    occ = get_occupancy(en, mu,T)
    # 3. Compute the expected value from wavefunction and occupancy
    exp_val = contract_indices(wf, occ)
    return exp_val


def solve(hamiltonian, exp_val_0, N_iterations):
    """
    Self-consistently solve for the wavefunction, eigenvalues, and expected value.

    Args:
      hamiltonian (HartreeFockHamiltonian): The initialized Hamiltonian class.
      exp_val_0 (numpy array): Initial ansatz for the expected value.
      N_iterations: Maximum number of iterations to run the self-consistent
      solver.

    Returns:
      wf (numpy array): Wavefunction with shape (N_flavor, N_level, N_k).
      en (numpy array): Energies with shape (N_level, N_k).
      exp_val (numpy array): Expected value with shape (N_flavor, N_flavor, N_k).
    """
    exp_val = exp_val_0.copy()
    # assert len(exp_val.shape)==3
    wf, en = None, None
    if hasattr(hamiltonian, "Nk"):
        N_k = hamiltonian.Nk
    elif hasattr(hamiltonian, "N_k"):
        N_k = hamiltonian.N_k
    else:
        raise ValueError("Nk not found in hamiltonian")

    nu = hamiltonian.nu
    T = hamiltonian.T
    conv = 100
    for iteration in range(N_iterations):
        # 1. `get_energy`
        htotal = hamiltonian.generate_Htotal(exp_val)
        wf, en = diagonalize(htotal)

        # 2. Update exp val from diagonalized Htotal
        new_exp_val = get_exp_val(wf, en, nu, T)
        new_exp_val = unflatten(new_exp_val, hamiltonian.D, N_k)

        # 3. Check for convergence (optional improvement could involve
        # setting a tolerance)
        prev_conv = conv
        conv = np.max(np.abs(new_exp_val - exp_val))
        if conv < 1e-7:
            print(f"Convergence reached at iteration {iteration}")
            break

        # if iteration > 2 and np.abs(conv-prev_conv) < 1e-20:
        #   print(f"Did not converge")
        #   break

        # Update the expected value for the next iteration
        exp_val = new_exp_val

    # Return the final wavefunction, energies, and expected values
    return wf, en, exp_val


def get_shell_index_triangle(n_shell):
    """Generates indices for a triangular grid in reciprocal space. Assume the two basis vectors are separated by 120 degrees. In order to get the actual coordinate, we need to multiply by the basis vectors, i.e., i * basis_vector_1 + j * basis_vector_2.
    Args:
      n_shell (int): number of the shell in reciprocal space.
    Returns:
      tuple: A pair of lists (i, j) with integer coordinates for each point
        in the triangular grid. Both lists have the same length.
    """
    i, j = [], []
    for jindex in range(-n_shell, n_shell + 1):
        for iindex in range(
            max(-n_shell, -n_shell + jindex), min(n_shell + jindex, n_shell) + 1
        ):
            i.append(iindex)
            j.append(jindex)
    return i, j


def get_reciprocal_vectors_triangle(a):
    """Computes the reciprocal lattice vectors for a triangular lattice. The two reciprocal are separated by 120 degrees, which are 4pi/(3a) * [cos(60deg), sin(60deg)] and 4pi/(3a) * [cos(-60deg), sin(-60deg)].
    The reciprocal lattice vectors are in units of 4pi/(3a), which is the distance from Gamma to K point.
    Args:
      a (float): Real space lattice constant.
    Returns:
      np.ndarray: Array of shape (2, 2) containing the two reciprocal lattice vectors.
            Each row represents a vector with components [x, y].
            Units are 4π/(3a), which is the distance from Gamma to K point.
    """
    return (
        np.array(
            [
                [np.cos(np.radians(60)), np.sin(np.radians(60))],
                [np.cos(np.radians(-60)), np.sin(np.radians(-60))],
            ]
        )
        * 4* np.pi/ (3 * a)
    )

def get_primitive_vectors_triangle(a):
  """
  Calculate the primitive (Bravais) lattice vectors for a 2D triangular lattice: They are separated by 120°
  Parameters:
  a (float): Lattice constant.
  Returns:
  numpy.ndarray: 2x2 array of primitive vectors.
  """
  return a * np.array([[0,1],[np.sqrt(3)/2,-1/2]])


def rotation_mat(theta_deg):
    """Creates a 2D rotation matrix for the given angle.
    Args:
      theta_deg (float): The rotation angle in degrees.
    Returns:
      numpy.ndarray: A 2x2 rotation matrix representing a counterclockwise
        rotation of theta_deg degrees.
    """

    return np.array(
        [
            [np.cos(np.radians(theta_deg)), -np.sin(np.radians(theta_deg))],
            [np.sin(np.radians(theta_deg)), np.cos(np.radians(theta_deg))],
        ]
    )


def generate_k_space(lattice: str, n_shell: int, a: float = 1.0):
    """Returns the k-space grid.

    Args:
      lattice: square | triangular
      n_shell: Number of "shells" or layers of the square/ triangular lattice. For
        the square lattice, the number of k points on each edge is (2 * n_shell) +
        1.
      a: Lattice constant. Default is 1.0.

    Returns:
      An N_k * 2 array
    """
    if lattice == "square":
        N_kx = N_ky = (2 * n_shell) + 1
        vec = np.array([[2 * np.pi / a, 0], [0, 2 * np.pi / a]])
        kx_range = np.linspace(-0.5, 0.5, N_kx, endpoint=False) + .5/N_kx
        ky_range = np.linspace(-0.5, 0.5, N_ky, endpoint=False) + .5/N_ky
        kx, ky = np.meshgrid(kx_range, ky_range)
        k_space = np.vstack([kx.ravel(), ky.ravel()]).T @ vec
        return k_space

    elif lattice == "triangular":
        reciprocal_vects = get_reciprocal_vectors_triangle(a)
        i, j = get_shell_index_triangle(n_shell)
        i = np.array(i) / (n_shell)
        j = np.array(j) / (n_shell)

        k_space = np.column_stack((i, j)) @ reciprocal_vects

        return k_space @ rotation_mat(90)

    else:
        raise ValueError(f"Unsupported lattice: {lattice}")


def get_q(n_shell, a):
    """Computes distant Gamma point in the momentum space.
    Args:
      n_shell (int): number of the shells in the triangular lattice.
      a (float): Lattice constant or scaling parameter.
    Returns:
      tuple: A tuple containing two arrays:
        - ij (numpy.ndarray): Array of shape (N, 2) representing lattice coordinates.
        - q (numpy.ndarray): Array of shape (N, 2) representing reciprocal space coordinates.
          Calculated as the matrix product of ij and the reciprocal lattice vectors.
    """
    i, j = get_shell_index_triangle(n_shell)
    ij = np.column_stack((i, j))
    return ij, ij @ get_reciprocal_vectors_triangle(a)


def get_area(a, lattice):
    """Computes the area of the real space unit cell for a given lattice type = 'square' or 'triangular'.

    Args:
      a (float): The lattice constant value.
      lattice (str): Type of lattice. Special handling for 'triangular',
        defaults to square/rectangular for other values.

    Returns:
      float: Area of the real space unit cell.
    """
    if lattice == "triangular":
        return a**2 * np.sin(np.deg2rad(120))
    else:
        return a**2


def generate_high_symmetry_points(lattice, a_M):
    """Returns high symmetry points in the 2D Brillouin zone.
    Calculates the coordinates of special k-points in the Brillouin zone for
    triangular or square lattices using the provided lattice constant.
    For triangular lattices, the high symmetry points are Gamma, Gamma', M, K, M', K'; 
    {
            "Gamma": np.array([0, 0]),
            "Gamma'": 4 * np.pi / (np.sqrt(3) * a_M) * np.array([1, 0]),
            "M": 2 * np.pi / (np.sqrt(3) * a_M) * np.array([1, 0]),
            "M'": 2 * np.pi / (np.sqrt(3) * a_M) * np.array([1 / 2, np.sqrt(3) / 2]),
            "K": 4 * np.pi / (3 * a_M) * np.array([np.sqrt(3) / 2, 1 / 2]),
            "K'": 4 * np.pi / (3 * a_M) * np.array([np.sqrt(3) / 2, -1 / 2]),
        }
    For square lattices, the points are Gamma, Gamma', M, K, M', K';
    {
            "Gamma": np.array([0, 0]),
            "Gamma'": 2 * np.pi / (a_M) * np.array([1, 0]),
            "M": 2 * np.pi / (a_M) * np.array([1 / 2, 0]),
            "M'": 2 * np.pi / (a_M) * np.array([0, 1 / 2]),
            "K": 2 * np.pi / (a_M) * np.array([1 / 2, 1 / 2]),
            "K'": 2 * np.pi / (a_M) * np.array([1 / 2, -1 / 2]),
        }
    Args:
      lattice: str, the lattice type ('triangular' or 'square')
      a_M: float, the lattice constant
    Returns:
      dict: A dictionary of high symmetry points where keys are point labels (str)
      and values are numpy arrays of shape (2,) containing the k-space coordinates.
      Points include 'Gamma', 'M', 'K', and their variations.
    """
    if lattice == "triangular":
        high_symm = {
            "Gamma": np.array([0, 0]),
            "Gamma'": 4 * np.pi / (np.sqrt(3) * a_M) * np.array([1, 0]),
            "M": 2 * np.pi / (np.sqrt(3) * a_M) * np.array([1, 0]),
            "M'": 2 * np.pi / (np.sqrt(3) * a_M) * np.array([1 / 2, np.sqrt(3) / 2]),
            "K": 4 * np.pi / (3 * a_M) * np.array([np.sqrt(3) / 2, 1 / 2]),
            "K'": 4 * np.pi / (3 * a_M) * np.array([np.sqrt(3) / 2, -1 / 2]),
        }
    elif lattice == "square":
        high_symm = {
            "Gamma": np.array([0, 0]),
            "Gamma'": 2 * np.pi / (a_M) * np.array([1, 0]),
            "M": 2 * np.pi / (a_M) * np.array([1 / 2, 0]),
            "M'": 2 * np.pi / (a_M) * np.array([0, 1 / 2]),
            "K": 2 * np.pi / (a_M) * np.array([1 / 2, 1 / 2]),
            "K'": 2 * np.pi / (a_M) * np.array([1 / 2, -1 / 2]),
        }

    return high_symm
