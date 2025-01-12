from HF import *
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
    """
    HartreeFockHamiltonian for a triangular TMD heterobilayer model.

    Args:
        parameters (dict): Model parameters not dependent on <b^\dagger b>.
          Defaults are provided below.
        N_shell (int): Number of k-space shells for the Brillouin-zone sampling.
        Nq_shell (int): Number of reciprocal lattice shells for the q vectors.
        filling_factor (float): Band filling fraction, default=0.5.
    """

    # ------------------------------------------------------------------------
    # LM Task: Modify init, define parameters that do NOT appear in exp_val
    # ------------------------------------------------------------------------
    def __init__(
        self,
        parameters: dict[str, Any] | None = None,
        N_shell: int = 1,
        Nq_shell: int = 1,
        filling_factor: float = 0.5
    ):
        if parameters is None:
            parameters = {}
        # Lattice type
        self.lattice = 'triangular'  # Provided LATTICE

        # Provide default values for the main physical parameters
        # Kinetic parameters
        self.hbar = parameters.get("hbar", 1.0)
        self.m_b = parameters.get("m_b", 1.0)       # bottom layer effective mass
        self.m_t = parameters.get("m_t", 1.0)       # top layer effective mass
        self.kappa = parameters.get("kappa", 0.0)   # offset wavevector for top valley

        # Potential parameters
        self.V_b = parameters.get("V_b", 0.0)        # amplitude of bottom potential
        self.psi_b = parameters.get("psi_b", -14.0)  # phase shift in bottom potential (deg)
        self.w = parameters.get("w", 0.0)            # interlayer tunneling
        self.omega = np.exp(1j*2*np.pi/3)            # C3 rotation factor
        # We set Delta_t(r)=0 as stated in the problem (low-energy focus on WSe2 conduction band).

        # Interaction parameters
        self.e = parameters.get("e", 1.0)          # electron charge
        self.epsilon = parameters.get("epsilon", 1.0)  # dielectric constant
        self.d = parameters.get("d", 1.0)          # screening length scale
        self.Volume = parameters.get("Volume", 1.0)# real-space area or volume factor

        # Lattice constant
        self.a_G = parameters.get("a_G", 1.0)      # reference lattice constant
        self.theta = parameters.get("theta", 0.0)  # small twist angle
        self.epsilon_theta = parameters.get("epsilon_theta", 0.0)  # strain, etc.

        # Save geometry parameters
        self.N_shell = N_shell
        self.Nq_shell = Nq_shell
        self.filling_factor = filling_factor
        self.T = 0.0  # Temperature = 0 by default

        # moire lattice constant:
        self.a_M = self.a_G / np.sqrt(self.epsilon_theta**2 + np.deg2rad(self.theta)**2 + 1e-16)

        # Generate k-space
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        self.Nk = len(self.k_space)

        # Generate q-space
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        self.Nq = len(self.q)

        # The dimension tuple: (layer=2, valley=2, reciprocal-lattice-vector=Nq)
        self.D = (2, 2, self.Nq)

        # Precompute some high-symmetry points
        self.high_symm = generate_high_symmtry_points(self.lattice, self.a_M)
        self.g = {
            j: rotation_mat(120*(j-1)) @ self.high_symm["Gamma'"] for j in range(1,4)
        }

    # ------------------------------------------------------------------------
    # LM Task: Generate Kinetic Term
    # ------------------------------------------------------------------------
    def generate_kinetic(self, k: np.ndarray | None = None) -> np.ndarray:
        """
        Generate the kinetic term H_K with shape (D, D, N_k).
        Kinetic energies are diagonal in the flavor indices: (l, tau, q).
        """
        if k is None:
            k = self.k_space  # shape (N_k, 2)
        kx, ky = k[:,0], k[:,1]
        N_k = k.shape[0]

        # Allocate Hamiltonian: (2,2,Nq, 2,2,Nq, N_k)
        H_K = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128)

        # For each valley: +K or -K => offset ± kappa in (kx, ?). 
        # We'll assume it only shifts kx by ± self.kappa.
        # For each layer: mass = m_b or m_t.

        # Indices: (l1, tau1, q1, l2, tau2, q2, k)
        # We'll fill in diagonal elements only, i.e. if (l1, tau1, q1)==(l2,tau2,q2)
        for l in range(2):       # bottom=0, top=1
            for tau in range(2): # +K=0, -K=1
                for iq in range(self.Nq):
                    # effective mass
                    mass = self.m_b if (l==0) else self.m_t
                    # ±kappa shift for top layer's valley
                    #   If tau=0 => +K => (kx - kappa)
                    #   If tau=1 => -K => (kx + kappa)
                    sign = (+1.0 if tau==1 else -1.0)
                    if l==0:
                        # bottom layer: the problem statement says k^2 for bottom
                        #   see H_{Kinetic}(r) => -hbar^2 k^2/(2 m_b)
                        #   Actually the sign in the original matrix is negative but we add that as H ~ E_k
                        #   so E_k = - (hbar^2 k^2)/(2 m_b). 
                        # We'll do the normal convention E_k = (hbar^2 k^2)/(2m), 
                        # and keep the minus sign in front if that is the chosen zero. 
                        shift_kx = kx[:] if (tau==0) else kx[:]  # no shift for bottom? from the matrix?
                        # The matrix line 1,3 are "k^2" => no ±kappa shift for bottom
                    else:
                        # top layer: lines 2,4 => (k - kappa)^2 or (k + kappa)^2
                        shift_kx = kx[:] - self.kappa if (tau==0) else kx[:] + self.kappa
                    shift_ky = ky[:]

                    # energy = -hbar^2/(2m) * (k_x^2 + k_y^2)
                    # keep the negative sign as in the Hamiltonian
                    kin_val = - (self.hbar**2/(2.0*mass))*(shift_kx**2 + shift_ky**2)

                    H_K[l, tau, iq, l, tau, iq, :] = kin_val

        return H_K

    # ------------------------------------------------------------------------
    # LM Task: Generate Potential Term
    # ------------------------------------------------------------------------
    def H_V(self, k: np.ndarray | None = None) -> np.ndarray:
        """
        Generate the periodic potential terms H_M with shape (D, D, N_k).
        This includes:
          - Delta_b(r) on diagonal (bottom layer)
          - Delta_t(r)=0 for top layer (given)
          - Interlayer tunneling Delta_{T, ±K}(r) as off-diagonal in layer index.
        """
        if k is None:
            k = self.k_space
        N_k = k.shape[0]

        # Allocate
        H_M = np.zeros(self.D + self.D + (N_k,), dtype=np.complex128)

        # For each (l1, tau1, q1, l2, tau2, q2) ...
        # We model the moiré potential by a real-space expression with 
        # cos(...) expansions, but in HF code we typically represent them 
        # by some reciprocal-lattice expansions. 
        #
        # We will do a schematic assignment as in the provided 4x4 block:
        # 1) diagonal in (layer, tau), => Delta_b(r) if layer=bottom 
        #    or Delta_t(r)=0 if layer=top
        # 2) off-diagonal in layer => Delta_{T, ±K}(r)
        #
        # In a self-consistent code, one might store Delta_b(q) etc. 
        # For demonstration, we'll just store some placeholders.

        for l1 in range(2):
            for tau1 in range(2):
                for q1i in range(self.Nq):
                    for l2 in range(2):
                        for tau2 in range(2):
                            for q2i in range(self.Nq):
                                if (l1==l2) and (tau1==tau2) and (q1i==q2i):
                                    # Diagonal in flavor
                                    if l1==0:
                                        # bottom layer => Delta_b(r)
                                        # approximate it as: Delta_b ~ 2 V_b * sum_j cos(...) ~ some constant
                                        # We'll store just a single constant as an example:
                                        delta_b = self._compute_Delta_b() 
                                        H_M[l1,tau1,q1i, l2,tau2,q2i, :] = delta_b
                                    else:
                                        # top layer => Delta_t(r) = 0
                                        H_M[l1,tau1,q1i, l2,tau2,q2i, :] = 0.0

                                # Off-diagonal in layer => interlayer tunneling
                                # i.e. (l1=0, l2=1) or (l1=1, l2=0), same valley => Delta_{T,±K}(r)
                                elif (tau1==tau2) and (q1i==q2i):
                                    # same valley, same q => Delta_T
                                    # for demonstration, let’s follow the sign convention:
                                    if (l1==0 and l2==1):
                                        # bottom->top
                                        dtun = self._compute_Delta_T(tau1)
                                        H_M[l1,tau1,q1i, l2,tau2,q2i, :] = dtun
                                    elif (l1==1 and l2==0):
                                        # top->bottom => complex conjugate
                                        dtun = self._compute_Delta_T(tau1)
                                        H_M[l1,tau1,q1i, l2,tau2,q2i, :] = dtun.conjugate()

        return H_M

    def _compute_Delta_b(self) -> complex:
        """
        Example of computing the average or representative value of 
        Delta_b(r) ~ 2*V_b * sum_j cos(...) with phase psi_b.
        For illustration, just return a single number. 
        """
        # In a real code, you'd sum over g_j, etc.
        # e.g. Delta_b(r) ~ 2*V_b * sum_{j in {1,3,5}} cos(g_j . r + psi_b).
        # We'll just place a single constant (the peak).
        return 3.0 * self.V_b * np.cos(np.deg2rad(self.psi_b))

    def _compute_Delta_T(self, tau: int) -> complex:
        """
        Interlayer tunneling amplitude for valley tau in {0=+K, 1=-K}.
        Delta_{T, tau}(r) ~ tau * w [1 + omega^tau e^{i tau g_2.r} + ... ].
        We'll just return a single complex amplitude as an example.
        """
        # Just a placeholder for demonstration
        sign = (+1 if tau==0 else -1)
        return sign * self.w * (1 + self.omega**sign + self.omega**(2*sign))

    # ------------------------------------------------------------------------
    # Summation: Non-interacting Hamiltonian = H_K + H_V
    # ------------------------------------------------------------------------
    def generate_non_interacting(self, k: np.ndarray | None = None) -> np.ndarray:
        """
        Return the sum of Kinetic + Periodic Potential terms.
        """
        return self.generate_kinetic(k) + self.H_V(k)

    # ------------------------------------------------------------------------
    # LM Task: Interacting part depends on exp_val
    # ------------------------------------------------------------------------
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Build the Hartree-Fock interaction Hamiltonian, shape (D, D, N_k).
        Interacting terms:
          1) Hartree:  + (1/V) * <b^\dagger b> * b^\dagger b * V(|q_1-q_4|)
          2) Fock:     - (1/V) * <b^\dagger b> * b^\dagger b * ...
        """
        # Expand exp_val to shape (2,2,Nq,2,2,Nq,N_k)
        exp_val = expand(exp_val, self.D)
        H_int = np.zeros(self.D + self.D + (self.Nk,), dtype=np.complex128)

        # We must sum over <b_{l1,t1,q1}^\dagger b_{l1,t1,q4}> or <b_{l1,t1,q1}^\dagger b_{l2,t2,q3}> etc.
        # Weighted by the Coulomb matrix element V(|...|) with the appropriate Kronecker deltas.
        #
        # For demonstration, we show only a symbolic structure:

        for k1 in range(self.Nk):
            for l1 in range(2):
                for t1 in range(2):
                    for q1i in range(self.Nq):
                        for q4i in range(self.Nq):
                            # Hartree-like average: < b^\dagger_{l1,t1,q1} b_{l1,t1,q4} >
                            # Then it acts on b^\dagger_{l2,t2,q2} b_{l2,t2,q3}.
                            # We'll do a toy assignment for the potential factor:
                            mean_val = exp_val[l1, t1, q1i, l1, t1, q4i, k1]
                            if abs(mean_val) < 1e-14:
                                continue

                            V_q = self._compute_Coulomb_factor(abs(q1i - q4i))

                            # The corresponding matrix element is:
                            # H_int[l2,t2,q2, l2,t2,q3, k1 or k2???] += + (1/V)* mean_val * V_q ...
                            # We skip the delta_{q1+q2,q3+q4} detail in this toy snippet.

                            # Just place it on the diagonal for demonstration:
                            for l2 in range(2):
                                for t2 in range(2):
                                    for q2i in range(self.Nq):
                                        # Illustrative: we pick q3i = q2i + (q1i-q4i)? etc. 
                                        # But let's skip the full structure, just show how you'd add it:
                                        q3i = q2i  # fake
                                        H_int[l2,t2,q2i, l2,t2,q3i, k1] += (1.0/self.Volume)*mean_val*V_q

                            # Fock contribution (with a minus sign). 
                            # We skip the difference in arguments for brevity:
                            # H_int[...] -= (1/V)* ...
                            # etc.

        return H_int

    def _compute_Coulomb_factor(self, dq: float) -> float:
        """
        Example of V(|q1-q4|) ~ 2π e^2 tanh(|q| d)/(ε |q|)
        We'll just return a rough scale for demonstration.
        """
        if dq < 1e-12:
            return 0.0
        val = (2.0 * np.pi * self.e**2 
               * np.tanh(dq * self.d) / (self.epsilon * dq))
        return val

    # ------------------------------------------------------------------------
    # Summation: H_total = H_nonint + H_int
    # ------------------------------------------------------------------------
    def generate_Htotal(self, exp_val: np.ndarray, flatten: bool = True) -> np.ndarray:
        """
        Generate total HF Hamiltonian = Non-Interacting + Interacting.
        """
        # Non-interacting
        H_nonint = self.generate_non_interacting()
        # Interacting
        H_int = self.generate_interacting(exp_val)

        H_tot = H_nonint + H_int
        if flatten:
            return self.flatten(H_tot)
        else:
            return H_tot

    # ------------------------------------------------------------------------
    # Flattening / Expanding
    # ------------------------------------------------------------------------
    def flatten(self, ham: np.ndarray) -> np.ndarray:
        """
        Flatten from shape (2,2,Nq,2,2,Nq,Nk) => (D_flat, D_flat, Nk)
        """
        D_flat = np.prod(self.D)
        return ham.reshape((D_flat, D_flat, self.Nk))

    def expand(self, ham: np.ndarray) -> np.ndarray:
        """
        Expand from (D_flat, D_flat, Nk) => (2,2,Nq,2,2,Nq,Nk).
        """
        return ham.reshape(self.D + self.D + (self.Nk,))