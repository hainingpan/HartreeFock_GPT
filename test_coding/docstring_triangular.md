`
class HartreeFockHamiltonian:
  def __init__(self, N_shell, parameters:dict[str, Any], filling_factor: float=0.5):
    self.lattice = 'square' | 'triangular'
    self.D = # LM Task: has to define this tuple.
    self.basis_order = {'0': 'flavor_type_0', '1': 'flavor_type_1', ... 'D-1': 'flavor_type_D-1'}
    # this is the basis order that the Hamiltonian will follow

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = temperature
    self.a = parameters['a'] # Lattice constant
    self.primitive_vectors = self.a * np.array([[0,1],[np.sqrt(3)/2,-1/2]]) # # Define the primitive (Bravais) lattice vectors for a 2D triangular lattice:They are separated by 120°
    # For a 2D square lattice, one would use two orthogonal primitive vectors.
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

    # All other parameters such as interaction strengths
    #self.param_0 = parameters['param_0'] # Brief phrase explaining physical significance of `param_0`
    #self.param_1 = parameters['param_1'] # Brief phrase explaining physical significance of `param_1`
    #...
    #self.param_p = parameters['param_p'] # Brief phrase explaining physical significance of `param_p`
    # Any other problem specific parameters.

    return

   def get_nearest_neighbor_vectors(self):
        """
        # Returns the integer coordinate offsets (n1, n2) corresponding to the 
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
            ...
        ]
        return n_vectors
    

  def generate_non_interacting(self) -> np.ndarray:
    H_nonint = np.zeros((self.D+ self.D+ (self.N_k,)), dtype=complex)
    #H_nonint[0, 0, :] = `code expression corresponding to all terms that contribute to H_nonint[0, 0]`
    #...
    #H_nonint[d, d, :] = `code expression corresponding to all terms that contribute to H_nonint[d, d]`
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    exp_val = expand(exp_val, self.D)
    H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=complex)

    # If more complicated functions of `exp_val` occur in multiple places,
    # one may add additional functions to the class of the form `func(self, exp_val)`.
    # Eg: the compute_order_parameter(exp_val) function for Emery in Emery_model_upd.
    # Otherwise define dependent expressions below
    #exp0 = function of exp_val
    #exp1 = function of exp_val
    #...
    #exp_e = function of exp_val

    H_int[0, 0, :] = #`code expression corresponding to all terms that contribute to H_int[0, 0]`
    #...
    H_int[d, d, :] = #`code expression corresponding to all terms that contribute to H_int[d, d]`
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) -> np.ndarray:
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if return_flat:
      return flattened(H_total,self.D,N_k)
    else:
      return H_total #l1, s1, q1, ....k
`


`
def flattened_hamiltonian(ham,N_flavor,N_k):
  """Flattens the Hamiltonian from N_flavor+ N_flavor+(N_k,) to (np.prod(N_flavor), np.prod(N_flavor), N_k)"""
  
def unflatten_exp_val(exp_val, N_flavor,N_k):
  """Unflattens the expected value from (np.prod(N_flavor), np.prod(N_flavor), N_k) to N_flavor+ N_flavor+(N_k,)"""


def compute_mu(en: np.ndarray, nu: float, T: float):
  """Compute the chemical potential."""

def get_occupancy(en: np.ndarray, T: float, mu: float):
  """Compute the occupancy of each state at each k point.

  Args:
    en: Energies with shape (N_level, N_k), level index first, then total k pts.
    T: Temperature.
    mu: Chemical potential, at T=0, fermi energy.

  Returns:
    occupancy: Occupancy with the same shape as `en`.
  """

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

def diagonalize(h_total: np.ndarray):
  """Diagonalizes the total Hamiltonian for each k point, sorts the eigenvalues and eigenvectors.

  Args:
    h_total: The total Hamiltonian with shape (N_flavor, N_flavor, N_k).

  Returns:
    wf: Eigenvectors (wavefunctions) with shape (N_flavor, N_flavor, N_k).
    en: Eigenvalues (energies) with shape (N_flavor, N_k).
  """

def get_exp_val(wf, en, nu, T):
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

def generate_high_symmtry_points(lattice, a_M):
      """Returns the high symmetry points in the Brillouin zone, given the lattice constant"""
  `