import numpy as np
from typing import Any

############################################################
# Predefined helper functions (stubs):
############################################################

def generate_k_space(lattice: str, N_shell: int, a: float=1.0) -> np.ndarray:
    """
    Generate the discrete k-points in the Brillouin zone
    for the chosen lattice (e.g. triangular).
    Returns an array of shape (N_k, 2). 
    """
    # LM Task: Implementation not shown; assume it is provided.
    k_space = []
    # ...
    return np.array(k_space)

def get_q(Nq_shell: int, a: float=1.0):
    """
    Generate the set of moire reciprocal lattice vectors.
    Returns indices and vectors, e.g. (q_index, q_vectors).
    """
    # LM Task: Implementation not shown; assume it is provided.
    q_index = []
    q_vecs = []
    return q_index, np.array(q_vecs)

def rotation_mat(angle_deg: float) -> np.ndarray:
    """ Returns the 2D rotation matrix for angle in degrees. """
    theta = np.deg2rad(angle_deg)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def generate_high_symmtry_points(lattice: str, a_M: float):
    """ Returns a dictionary of high-symmetry momenta for plotting/band-structure. """
    # LM Task: Implementation not shown; assume it is provided.
    high_symm = {}
    # ...
    return high_symm

def expand(exp_val: np.ndarray, D: tuple[int, ...]) -> np.ndarray:
    """ 
    Reshape/unsqueeze exp_val from flattened shape (D_flat, D_flat, N_k) 
    to shape (D, D, N_k).
    """
    D_flat = np.prod(D)
    # e.g. exp_val.shape = (D_flat, D_flat, N_k)
    return exp_val.reshape(D + D + (exp_val.shape[-1],))

############################################################
# The HartreeFockHamiltonian Class
############################################################

class HartreeFockHamiltonian:
    """
    Example Hartree-Fock Hamiltonian class for a TMD heterobilayer
    with bottom/top layers, ±K valleys, and moire reciprocal vectors.
    """

    def __init__(self, 
                 parameters: dict[str, Any] = {}, 
                 N_shell: int = 10,
                 Nq_shell: int = 1,
                 filling_factor: float = 0.5):
        """
        Args:
            parameters (dict[str, Any]): Model parameters not depending on exp_val.
            N_shell (int): # of discrete k-points in each direction (mesh size).
            Nq_shell (int): # of shells of moire reciprocal-lattice vectors.
            filling_factor (float): e.g. doping fraction.
        """
        # Lattice type
        self.lattice = 'triangular'

        # Model / physical parameters
        # Assign defaults if not provided in `parameters`.
        self.hbar = parameters.get('hbar', 1.0545718e-34)  # Planck's const / 2π
        self.m_b = parameters.get('m_b', 0.45)             # (example) bottom mass
        self.m_t = parameters.get('m_t', 0.35)             # (example) top mass
        self.kappa = parameters.get('kappa', 0.5)          # momentum offset
        self.a_G = parameters.get('a_G', 1.0)              # bare lattice const
        self.theta = parameters.get('theta', 0.0)          # twist angle
        self.epsilon = parameters.get('epsilon', 10.0)     # screening param
        self.Vb = parameters.get('Vb', 10.0e-3)            # amplitude of ∆_b
        self.psi_b = parameters.get('psi_b', -14.0)        # phase (in degrees!)
        self.w    = parameters.get('w', 5.0e-3)            # tunneling strength
        self.omega = np.exp(1j*2*np.pi/3)                  # C3z symmetry factor
        self.d = parameters.get('d', 10.0)                 # layer separation
        self.e_charge = parameters.get('e_charge', 1.602e-19)  # elementary charge
        self.eps_0 = parameters.get('eps_0', 8.854e-12)     # vacuum permittivity
        self.eps_r = parameters.get('eps_r', 12.0)          # relative perm
        self.T = 0.0  # temperature set to zero

        # Additional placeholders:
        # self.Vt = ...  # ∆_t(r) amplitude if needed
        # etc.

        # Define moire lattice constant
        # a_M = a_G / sqrt(epsilon^2 + theta^2 in radians^2)
        # (the form below is an example; adjust as needed)
        self.a_M = self.a_G / np.sqrt(self.epsilon**2 + np.deg2rad(self.theta)**2)

        # Generate k-space
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        self.Nk = len(self.k_space)

        # Generate q-space
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        self.Nq = len(self.q)

        # The dimension of the Hamiltonian in flavor space
        # layer in {b, t} => 2
        # valley in {+K, -K} => 2
        # moire reciprocal vectors => self.Nq
        self.D = (2, 2, self.Nq)

        # For convenience, generate high symmetry points
        self.high_symm = generate_high_symmtry_points(self.lattice, self.a_M)
        # For example, we might define a set of 3 equivalent Γ' points by 120° rotation:
        self.g = {
            j: rotation_mat(120*(j-1)) @ self.high_symm.get("Gamma'", np.zeros(2)) 
            for j in range(1,4)
        }

        # Save filling factor
        self.filling_factor = filling_factor


    def generate_kinetic(self, k: np.ndarray=None) -> np.ndarray:
        """
        Generate the single-particle kinetic term, 
        shape = (D, D, N_k).
        
        If k is None, use self.k_space. Otherwise k should be an
        array of shape (N_k_alt, 2) for a user-defined mesh.
        """
        if k is None:
            k = self.k_space  # shape (N_k, 2)
        N_k_in = k.shape[0]

        # Allocate the Hamiltonian
        # shape in flavor space => (2,2,Nq, 2,2,Nq, N_k_in)
        H_K = np.zeros(self.D + self.D + (N_k_in,), dtype=np.complex128)

        # The 4x4 block structure in real continuum models 
        # translates to (l=2, τ=2) for each q.
        # We'll write them as diagonal blocks in the code:

        # Indices in code: H_K[l, tau, q, l', tau', q', kidx]
        #
        # match to:
        #   [b,+K], [t,+K], [b,-K], [t,-K]
        # with doping each block by the moiré q as well.

        for ik in range(N_k_in):
            # magnitude: k^2, or (k - κ)^2, etc
            # For simplicity, let kx = k[ik,0], ky = k[ik,1].
            # define "k±κ" as a shift in (say) kx-direction:
            kx, ky = k[ik]
            k_plus  = np.sqrt((kx - self.kappa)**2 + (ky)**2)
            k_minus = np.sqrt((kx + self.kappa)**2 + (ky)**2)
            k_norm  = np.sqrt(kx**2 + ky**2)

            # fill diagonal blocks:
            # layer=b, valley=+K => -hbar^2 k^2/(2 m_b)
            # layer=t, valley=+K => -hbar^2 (k-κ)^2/(2 m_t)
            # layer=b, valley=-K => -hbar^2 k^2/(2 m_b)
            # layer=t, valley=-K => -hbar^2 (k+κ)^2/(2 m_t)

            E_b_plusK = - (self.hbar**2)/(2*self.m_b) * (k_norm**2)
            E_t_plusK = - (self.hbar**2)/(2*self.m_t) * (k_plus**2)
            E_b_minusK= - (self.hbar**2)/(2*self.m_b) * (k_norm**2)
            E_t_minusK= - (self.hbar**2)/(2*self.m_t) * (k_minus**2)

            # place them on the diagonal for each q
            for iq in range(self.Nq):
                # bottom, +K
                H_K[0, 0, iq, 0, 0, iq, ik] = E_b_plusK
                # top, +K
                H_K[1, 0, iq, 1, 0, iq, ik] = E_t_plusK
                # bottom, -K
                H_K[0, 1, iq, 0, 1, iq, ik] = E_b_minusK
                # top, -K
                H_K[1, 1, iq, 1, 1, iq, ik] = E_t_minusK

        return H_K

    def H_V(self, k: np.ndarray=None) -> np.ndarray:
        """
        Generate the periodic potential part, shape = (D, D, N_k).
        For illustration, we place ∆_b(r), ∆_t(r)=0, and ∆_{T,±K}(r).
        Typically, one would do a Fourier transform from r->q, 
        or directly store ∆(k) for each k.  We show the structure only.
        """
        if k is None:
            k = self.k_space
        N_k_in = k.shape[0]

        H_M = np.zeros(self.D + self.D + (N_k_in,), dtype=np.complex128)

        # The 4×4 block in real space becomes 
        #    [∆_b(r),         ∆_{T,+K}(r),   0,             0]
        #    [∆_{T,+K}^*(r),  ∆_t(r),       0,             0]
        #    [0,              0,           ∆_b(r),        ∆_{T,-K}(r)]
        #    [0,              0,           ∆_{T,-K}^*(r), ∆_t(r)]
        #
        # with layer={b,t} → index 0 or 1,
        #    valley=+K → index 0; valley=-K → index 1.
        #
        # We have ∆_t(r)=0 in this model, so we skip that piece,
        # and set ∆_b(r) = 2 * Vb * ∑ cos(g_j·r + psi_b),
        # ∆_{T,±K} as given by eqn for interlayer tunneling.
        #
        # Here, we illustrate *constant placeholders* for each k.

        for ik in range(N_k_in):
            for iq in range(self.Nq):
                # "Compute" the real-space potential => a placeholder number:
                Delta_b_val = 0.001  # e.g. some small amplitude
                Delta_t_val = 0.0    # set to zero in problem statement

                # For interlayer tunneling:
                # Delta_{T,+K} ~ w (1 + ω^{+K} e^{i...} + ω^{2K} e^{i...})
                # In code, we just place a placeholder or test expression:
                Delta_T_plusK  = 0.001 * np.exp(1j*0.0)
                Delta_T_minusK = 0.001 * np.exp(-1j*0.2)

                # Fill in block structure:
                # layer=b(0), valley=+K(0) => row = [0,0], col = [0,0]
                # layer=t(1), valley=+K(0) => row = [1,0], col = [1,0]
                # ...
                H_M[0, 0, iq, 0, 0, iq, ik] = Delta_b_val
                H_M[1, 0, iq, 1, 0, iq, ik] = Delta_t_val
                H_M[0, 0, iq, 1, 0, iq, ik] = Delta_T_plusK
                H_M[1, 0, iq, 0, 0, iq, ik] = Delta_T_plusK.conjugate()

                H_M[0, 1, iq, 0, 1, iq, ik] = Delta_b_val
                H_M[1, 1, iq, 1, 1, iq, ik] = Delta_t_val
                H_M[0, 1, iq, 1, 1, iq, ik] = Delta_T_minusK
                H_M[1, 1, iq, 0, 1, iq, ik] = Delta_T_minusK.conjugate()

        return H_M

    def generate_non_interacting(self, k: np.ndarray=None) -> np.ndarray:
        """
        Sum up the kinetic and the single-particle (periodic) potential parts.
        """
        H_kin = self.generate_kinetic(k)
        H_pot = self.H_V(k)
        return H_kin + H_pot

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Build the Hartree-Fock interaction part, which depends on exp_val.
        Shape of the returned matrix = (D, D, N_k).
        """
        # 1) Unflatten exp_val from shape (D_flat, D_flat, N_k) -> (D, D, N_k)
        exp_val = expand(exp_val, self.D)

        H_int = np.zeros(self.D + self.D + (self.Nk,), dtype=np.complex128)

        # ---------------------------------------------------------------------
        # 2) "Hartree" term: 
        #    + (1/V)* sum_{l1,q1,q4} < b_{l1,τ1,q1}^dagger b_{l1,τ1,q4} > b_{l2,τ2,q2}^dagger b_{l2,τ2,q3} 
        #    * V(|q1-q4|) δ_{q1+q2,q3+q4}
        #
        #    Pseudocode: for each (l2, τ2, q2), (l2, τ2, q3):
        #      sum over l1, τ1, q1, q4 => exp_val[l1,τ1,q1, l1,τ1,q4, k], etc
        #
        # 3) "Fock" term: 
        #    - (1/V)* sum < b_{l1,τ1,q1}^dagger b_{l2,τ2,q3} > b_{l2,τ2,q2}^dagger b_{l1,τ1,q4} ...
        #
        #  We won't fully implement all the loops. Instead, we illustrate how 
        #  you would incorporate these terms, referencing `exp_val[...]`.
        # ---------------------------------------------------------------------

        # Example placeholder: Just showing that we can write something like
        #    H_int[l2,τ2,q2, l2,τ2,q3, k] += sum_{...} [ coupling * exp_val[l1,τ1,q1, l1,τ1,q4, k] ]
        #
        # Real code would check delta_{q1+q2, q3+q4}, handle V(|q1-q4|), etc.

        # HARTEE-like contribution (schematic)
        # for kidx in range(self.Nk):
        #     for l2 in range(2):
        #         for tau2 in range(2):
        #             for q2 in range(self.Nq):
        #                 for q3 in range(self.Nq):
        #                     # sum over (l1, tau1, q1, q4)
        #                     val = 0.0j
        #                     for l1 in range(2):
        #                         for tau1 in range(2):
        #                             for q1 in range(self.Nq):
        #                                 for q4 in range(self.Nq):
        #                                     # example: the "mean field" from exp_val:
        #                                     mf = exp_val[l1, tau1, q1, l1, tau1, q4, kidx]
        #                                     # some factor: V(|q1-q4|)
        #                                     # check delta_{q1+q2, q3+q4}
        #                                     # accumulate into val
        #                                     ...
        #                     H_int[l2, tau2, q2, l2, tau2, q3, kidx] += val

        # FOCK-like contribution (schematic)
        # similarly, we pick out 
        #    exp_val[l1, tau1, q1, l2, tau2, q3, kidx]
        # etc.
        # and incorporate the minus sign, the potential, the delta, etc.

        # For demonstration, we return H_int=0. 
        # In a real code, you'd fill in these matrix elements as above.
        return H_int

    def generate_Htotal(self, exp_val: np.ndarray, flatten: bool = True) -> np.ndarray:
        """
        The total Hartree-Fock Hamiltonian = H_non_interacting + H_interacting.
        """
        H_nonint = self.generate_non_interacting()
        H_int    = self.generate_interacting(exp_val)
        H_total  = H_nonint + H_int

        if flatten:
            return self.flatten(H_total)
        else:
            return H_total

    def flatten(self, ham: np.ndarray) -> np.ndarray:
        """
        Flatten from shape (D, D, N_k) to 
        (D_flat, D_flat, N_k) = (2*2*Nq, 2*2*Nq, N_k).
        """
        D_flat = np.prod(self.D)
        return ham.reshape((D_flat, D_flat, self.Nk))

    def expand(self, ham_flat: np.ndarray) -> np.ndarray:
        """
        The inverse of flatten, if needed.
        """
        return ham_flat.reshape(self.D + self.D + (self.Nk,))

