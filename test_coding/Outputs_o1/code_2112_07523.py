import numpy as np
from typing import Any
from HF import *
########################################
# We assume you already have or will provide
# generate_k_space(lattice, N_shell, a)
# flatten(H_total, D)
# expand(exp_val, D)
# in your codebase.  We do not redefine them here.
########################################

class HartreeFockHamiltonian:
    """
    Hartree-Fock Hamiltonian for the BHZ + Excitonic Interactions model.
    
    Args:
      N_shell (int): Controls size of the discrete k-mesh in Brillouin Zone.
      parameters (dict[str, Any]): Model parameters, e.g. 
          {
            'm_e': ... ,    # electron effective mass
            'm_h': ... ,    # hole effective mass
            'E_g': ... ,    # band gap
            'A':   ... ,    # Dirac linear coupling
            'Q':   ... ,    # Momentum shift between conduction & valence
            'V0':  ... ,    # Typical scale for Coulomb interactions
            'epsilon': ...,
            'd': ...,
            'S': ...        # 2D area (if needed)
            ...
          }
      filling_factor (float, optional): Occupancy in conduction band, etc.
      temperature (float, optional): System temperature. Defaults to 0.
    """
    def __init__(self,
                 N_shell: int=4,
                 parameters: dict[str, Any]=None,
                 filling_factor: float=0.5,
                 temperature: float=0.0):
        
        if parameters is None:
            parameters = {}
        
        # LATTICE
        self.lattice = 'square'   # Given in the problem statement
        
        # -------------- LM Task: define the dimension + basis --------------
        # Here we have spin x band => 2 x 2 = 4 internal states
        # We keep it as a 2D tuple for clarity:
        self.D = (2, 2)  
        
        # Provide a dictionary describing how we flatten them to 0..3
        self.basis_order = {
            "0": "spin_up, conduction",
            "1": "spin_up, valence",
            "2": "spin_down, conduction",
            "3": "spin_down, valence"
        }
        
        # Occupancy relevant parameters
        self.nu = filling_factor
        self.T = temperature
        
        # Set default values if not provided
        self.m_e  = parameters.get('m_e', 0.5)    # electron effective mass
        self.m_h  = parameters.get('m_h', 0.5)    # hole (hh) effective mass
        self.E_g  = parameters.get('E_g', 1.0)    # band gap
        self.A    = parameters.get('A',   1.0)    # Dirac coupling
        self.Q    = parameters.get('Q',   np.array([0.0, 0.0])) # shift
        self.V0   = parameters.get('V0',  1.0)    # typical Coulomb scale
        self.eps  = parameters.get('epsilon', 10) # dielectric const
        self.d    = parameters.get('d',   10.0)   # interlayer distance
        self.S    = parameters.get('S',   1e4)    # sample area
        
        # Lattice constant 'a' (needed in generate_k_space)
        self.a = parameters.get('a', 1.0)

        # Build the k-mesh
        self.k_space = generate_k_space(self.lattice, N_shell, self.a)
        self.N_k = self.k_space.shape[0]

    # -----------------------------------------------------------------------
    def generate_non_interacting(self) -> np.ndarray:
        """
        Construct the non-interacting BHZ Hamiltonian blocks: 
          h_up(k) and h_down(k), each 2x2 in conduction/valence space.
        
        Returns:
            np.ndarray of shape (2,2,2,2,N_k) if unflattened, or (4,4,N_k) if flattened,
            but we will keep it unflattened initially: (D + D + (N_k,))
        """
        # Prepare the container: shape = (2,2, 2,2, N_k)
        H_nonint = np.zeros(self.D + self.D + (self.N_k,), dtype=np.complex128)
        
        # For convenience:
        kx = self.k_space[:,0]
        ky = self.k_space[:,1]
        # Q/2 shift in momentum
        Qx, Qy = self.Q / 2.0
        
        # Let us define for spin up block: (s=0)
        # conduction (b=0) diagonal: +E_g/2 + [hbar^2/(2m_e)](k - Q/2)^2
        # valence   (b=1) diagonal: -E_g/2 - [hbar^2/(2m_h)](k + Q/2)^2
        # off diag: A(k_x +/- i k_y)
        
        # For spin down block: (s=1)
        # same diagonal terms, but off-diagonal has a sign flip: A(k_x - i k_y) vs. A(k_x + i k_y), etc.
        
        # We'll define a small helper:
        def k_sq(kx, ky):
            return kx**2 + ky**2
        
        # Non-interacting energies for conduction/valence:
        #   For spin-up: 
        #       conduction diag = + E_g/2 + (hbar^2/2m_e)*(k - Q/2)^2
        #       valence diag    = - E_g/2 - (hbar^2/2m_h)*(k + Q/2)^2
        #
        #   For spin-down the same diagonal terms appear, but the off-diagonal differs in sign.
        
        # We'll treat hbar^2/(2m_e) as one parameter.  For simplicity, 
        # let's define them as:
        alpha_e = 1.0/(2.0*self.m_e)  # "hbar^2" taken as 1 in convenient units
        alpha_h = 1.0/(2.0*self.m_h)
        
        for ik in range(self.N_k):
            # Shifted wavevectors:
            kxm = kx[ik] - Qx  # (k - Q/2)
            kym = ky[ik] - Qy
            kxp = kx[ik] + Qx  # (k + Q/2)
            kyp = ky[ik] + Qy
            
            # For spin-up block => s=0
            # conduction(b=0) diagonal
            H_nonint[0,0,0,0,ik] = +self.E_g/2.0 + alpha_e*(kxm**2 + kym**2)
            # valence(b=1) diagonal
            H_nonint[0,1,0,1,ik] = -self.E_g/2.0 - alpha_h*(kxp**2 + kyp**2)
            
            # Off-diagonals in the up-block
            # (conduction <-> valence) => A(kx + i ky), A(kx - i ky)
            # top-right: conduction->valence
            H_nonint[0,0,0,1,ik] = self.A*(kx[ik] + 1j*ky[ik])
            # bottom-left: valence->conduction
            H_nonint[0,1,0,0,ik] = self.A*(kx[ik] - 1j*ky[ik])
            
            # For spin-down block => s=1
            # conduction(b=0) diagonal
            H_nonint[1,0,1,0,ik] = +self.E_g/2.0 + alpha_e*(kxm**2 + kym**2)
            # valence(b=1) diagonal
            H_nonint[1,1,1,1,ik] = -self.E_g/2.0 - alpha_h*(kxp**2 + kyp**2)
            
            # Off-diagonals in the down-block
            # sign changes: conduction<->valence => -A(kx - i ky), etc
            H_nonint[1,0,1,1,ik] = -self.A*(kx[ik] - 1j*ky[ik])
            H_nonint[1,1,1,0,ik] = -self.A*(kx[ik] + 1j*ky[ik])
        
        return H_nonint

    # -----------------------------------------------------------------------
    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """
        Generates the interaction (Hartree + Fock) contributions, which depend
        on the density-matrix expectation values stored in 'exp_val'.
        
        Args:
          exp_val (np.ndarray): shape (D_flattened, D_flattened, N_k) or flattened.
        
        Returns:
          H_int (np.ndarray): shape = (2,2,2,2,N_k) unflattened. 
        """
        # Expand to shape (2,2, 2,2, N_k) so that indexing matches [s1,b1, s2,b2, k]
        rho = expand(exp_val, self.D)   # i.e. shape (2,2, 2,2, N_k)
        
        H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=np.complex128)
        
        # -------------------------------------------------------------------
        # Example: We show how we might incorporate a simple Hartree shift
        # from exciton density n_x = sum_{s,n,k} rho_{c s n}^{c s n}(k). 
        # In the full model, you would implement the full sums over n,n',n'' 
        # from eq. (2) and eq. (5).  Here we only illustrate the structure.
        # -------------------------------------------------------------------
        
        # 1) Compute exciton density n_x from conduction occupancy:
        #    n_x = (1/S) * sum_{k, spin, conduction-band} [rho_{c, spin}^c, spin (k)]
        #    in code: conduction index b=0, valence = b=1 (assuming we set that order).
        
        conduction_index = 0
        # sum over spin up/down => s=0,1
        # shape of rho is (2,2,2,2,N_k), so conduction -> valence = b1=0 or b2=0
        # Diagonal conduction occupancy => rho[s,b=0, s,b=0, k]
        
        n_x_val = 0.0
        for s in range(2):
            for ik in range(self.N_k):
                n_x_val += rho[s,conduction_index, s,conduction_index, ik].real
        
        n_x_val /= self.S  # factor of 1/S as in eq.(4)
        
        # Potential shift: 4 pi e^2 n_x d / epsilon
        # We'll store it in a variable:
        V_Hx = 4.0 * np.pi * (1.0) * n_x_val * self.d / self.eps
        # (Here we treat e^2 = 1 for illustration, or you can set e=1, etc.)
        
        # We add that shift to conduction vs valence band energies accordingly.
        # For example, eq.(4) states that conduction layer is raised by +(...),
        # hole (valence) layer is lowered by the same amount, etc.
        
        # Let's say conduction states get +V_Hx:
        for s in range(2):
            for ik in range(self.N_k):
                H_int[s,0, s,0, ik] += V_Hx   # conduction diagonal
                
        # valence states might get -V_Hx:
        for s in range(2):
            for ik in range(self.N_k):
                H_int[s,1, s,1, ik] -= V_Hx
        
        # 2) Fock and Coulomb-exchange terms:
        #    The full expression from eq.(2),(5) is quite involved.  Typically one
        #    would do nested sums over b,b', s,s', n,n', k,k' with the potentials
        #    V_{b b'}(...).  Each leads to a term in H_int[...] that depends on
        #    the expectation values rho_{b's'n'}^{b's'n''+...}(k').  
        #    We will not expand it fully here but you would implement it similarly,
        #    indexing the correct conduction/valence, spin, and k-points.
        
        # For example, a schematic Fock contribution might look like:
        # H_int[b,s, b,s, k] += - (1/S)*sum_{b',s',n',k'} V_{b b'}(...) * rho[b,s,n']^{b',s',n'}(k') 
        # etc.
        
        return H_int

    # -----------------------------------------------------------------------
    def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool = True) -> np.ndarray:
        """
        Builds total Hartree-Fock Hamiltonian = Non-interacting + Interacting.
        """
        H_nonint = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        
        H_total = H_nonint + H_int
        
        if return_flat:
            return flatten(H_total, self.D)  # shape => (4,4,N_k) if D=(2,2)
        else:
            return H_total  # shape => (2,2,2,2,N_k)
