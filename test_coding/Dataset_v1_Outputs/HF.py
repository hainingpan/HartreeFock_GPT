"""HF library."""

import numpy as np
from scipy import optimize


def fermi_dbn(energy, T):
  return 1/(np.exp((energy)/T)+1)

def flattened_hamiltonian(ham):
  if ham.shape==3:
    return ham
  else:
    un_flattened = list(ham.shape) #l1, s1...l2, s2..., k

#TODO: Haining tthinks the T!=0 way rn is a bit unstable.
def compute_mu(en: np.ndarray, nu: float, T: float, n):
  """Compute the chemical potential."""
  flat_en = en.flatten()

  if T == 0:
    flattened_en = en.flatten()

    # Sort the flattened energy array to find the Fermi level based on filling factor
    sorted_en = np.sort(flattened_en)

    # Determine the index for the Fermi level based on the filling factor
    fermi_index = min(int(np.floor(nu * len(sorted_en))), len(sorted_en)-1)

    # Fermi energy is the energy at the Fermi index
    mu = sorted_en[fermi_index]

  else:
    raise NotImplementedError("T!=0 needs to be checked.")
    def func(x):
      return n - 2*np.sum(fermi_dbn(flat_en[:]-x, T))/en.size

    mu = optimize.fsolve(func, nu)
    # Replaced this since it had 0.5 hard-coded with what I assume should be nu.
    if np.isclose(func(mu), 0.0):
      print("mu was found")
    else:
      print("mu wasn't found")
  return mu


def get_occupancy(en: np.ndarray, T: float, mu: float):
  """Compute the occupancy of each state at each k point.

  Args:
    en: Energies with shape (N_level, N_k), level index first, then total k pts.
    T: Temperature.
    mu: Chemical potential, at T=0, fermi energy.

  Returns:
    occupancy: Occupancy with the same shape as `en`.
  """
  occupancy = np.zeros_like(en) # next steps might complain if en is complex
  if T == 0:
    # Compute occupancy: 1 if energy <= Fermi level, 0 otherwise
    occupancy = (en <= mu).astype(int)
  else:
    for idx in range(len(en[:, 1])):
      occupancy[idx,:] = fermi_dbn(en[idx,:]-mu, T)
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
    h_total: The total Hamiltonian with shape (N_flavor, N_flavor, N_k).

  Returns:
    wf: Eigenvectors (wavefunctions) with shape (N_flavor, N_flavor, N_k).
    en: Eigenvalues (energies) with shape (N_flavor, N_k).
  """
  N_k = h_total.shape[-1]
  D = h_total.shape[0]
  wf = np.zeros_like(h_total)  # Wavefunctions (eigenvectors).
  en = np.zeros((D, N_k))  # Eigenvalues (energies)

  # Ensure H_total is Hermitian by symmetrizing it. # Doubt: Is this a step for all problems?
  h_total_sym = (h_total + h_total.transpose((1, 0, 2))) / 2

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


def get_exp_val(wf, en, nu, T, n):
  """Computes the expected values from the wavefunction, eigenenergies, and filling factor.
  TODO: This assumes the exp val is diagonal..
  Args:
    wf: Wavefunctions with shape (N_flavor, N_level, N^2).
    en: Eigenenergies with shape (N_flavor, N^2).
    nu: Filling factor.
    T: Temperature
    n:

  Returns:
  - numpy.ndarray: Expected values with shape (N_flavor, N^2).
  """
  # 1. Compute the chemical potential
  mu = compute_mu(en, nu, T, n)
  # 2. Compute the occupancy based on energies and chemical potential
  occ = get_occupancy(en, T, mu)
  # 3. Compute the expected value from wavefunction and occupancy
  exp_val = contract_indices(wf, occ)
  return exp_val


def solve(hamiltonian, exp_val_0, N_iterations):
    """
    Self-consistently solve for the wavefunction, eigenvalues, and expected value.

    Args:
      hamiltonian (Hamiltonian): The initialized Hamiltonian.
      exp_val_0 (numpy array): Initial ansatz for the expected value.
      N_iterations: Maximum number of iterations to run the self-consistent
      solver.

    Returns:
      wf (numpy array): Wavefunction with shape (N_flavor, N_level, N_k).
      en (numpy array): Energies with shape (N_level, N_k).
      exp_val (numpy array): Expected value with shape (N_flavor, N_flavor, N_k).
    """
    exp_val = exp_val_0.copy()
    assert len(exp_val.shape)==3
    wf, en = None, None

    nu = hamiltonian.nu
    T = hamiltonian.T
    n = hamiltonian.n
    conv = 100
    for iteration in range(N_iterations):
      # 1. `get_energy`
      htotal = hamiltonian.generate_Htotal(exp_val)
      wf, en = diagonalize(htotal)

      # 2. Update exp val from diagonalized Htotal
      new_exp_val = get_exp_val(wf, en, nu, T, n)

      # 3. Check for convergence (optional improvement could involve
      # setting a tolerance)
      prev_conv = conv
      conv = np.max(np.abs(new_exp_val - exp_val))

      if conv < 1e-7:
        print(f'Convergence reached at iteration {iteration}')
        break

      if iteration > 2 and np.abs(conv-prev_conv) < 1e-20:
        print(f"Did not converge")
        break

      # Update the expected value for the next iteration
      exp_val = new_exp_val

    # Return the final wavefunction, energies, and expected values
    return wf, en, exp_val


def query_parameters(hamiltonian):
  """Queries the parameters from the problem description."""
  return hamiltonian.__dict__


def reset_parameters(
    hamiltonian, parameter_kwargs: dict[str, float]
):
  """Resets the parameters from the problem description."""
  for k in parameter_kwargs:
    hamiltonian.__dict__[k] = parameter_kwargs[k]


def get_shell_index(n_shell):
    """Returns indices of the reciprocal lattice grid."""
    x, y = [], []
    for yindex in range(-n_shell, n_shell + 1):
      for xindex in range(max(-n_shell, -n_shell + yindex), min(n_shell + yindex, n_shell) + 1):
        x.append(xindex)
        y.append(yindex)
    return x,y

def get_reciprocal_vectors():
  return np.array([[np.cos(np.radians(60)), np.sin(np.radians(60))],[np.cos(np.radians(-60)), np.sin(np.radians(-60))]])

# def rotation_mat(theta_deg):
#   return np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
#                     [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
def rotation_mat(theta_deg):
  return np.array([[np.cos(np.radians(theta_deg)), -np.sin(np.radians(theta_deg))],
                    [np.sin(np.radians(theta_deg)), np.cos(np.radians(theta_deg))]])


def generate_k_space(lattice: str, n_shell: int):
  """
    Args:
      lattice: square | triangular
      n_shell: Number of "shells" or layers of the square/ tirangular lattice.
      # Edit: N_k will be a "resolution" for the BZ zone.
      Symmetry:
    Returns:
      An N_k * 2 array
  """
  if lattice == 'square':
    N_kx = (2*n_shell)+1
    k_range = np.linspace(-np.pi, np.pi, N_kx, endpoint=False)
    kx, ky = np.meshgrid(k_range, k_range)
    k_space = np.vstack([kx.ravel(), ky.ravel()]).T
    return k_space

  if lattice == 'triangular':
    reciprocal_vects = get_reciprocal_vectors()
    x,y = get_shell_index(n_shell)
    x=[x_el//(2*n_shell) for x_el in x]
    y=[y_el//(2*n_shell) for y_el in y]

    k_space = np.column_stack((x, y)) @ reciprocal_vects

    return k_space @ rotation_mat(90)

    # this returns integer coordinates n, m such that \vec{r} = n \vec{A_1} + m \vec{A_2}.
    # Lattice vectors in real space A1 = (1,0), A2 = (-sqrt(3)/2, 1/2)
    # Reciprocal lattice vectors are G1 and G2 for k-space
    # k = n \vec{G_1} + m \vec{G_2}
    # BZ_coord_space = np.column_stack((x, y))

    # np.array([G1_x, G2_x], [G1_y, G2_y])
    # k_space = np.matmul(BZ_coord_space, reciprocal_vects)
    # return k_space
  # TODO: add other symmetries
  else:
    raise ValueError(f'Unsupported lattice: {lattice}')

def get_q(n_shell):
    x,y = get_shell_index(n_shell)
    xy=np.column_stack((x, y))
    return xy, xy @ get_reciprocal_vectors()

def get_A(aM, lattice):
  if lattice == 'triangular':
    return aM**2 * np.sin(np.deg2rad(60))
  else:
    return aM**2