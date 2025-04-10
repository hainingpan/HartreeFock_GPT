class HartreeFockHamiltonian:
    def __init__(self, 
                 parameters: dict,
                 N_shell: int = 10,
                 Nq_shell: int = 1,
                 filling_factor: float = 0.5,
                 temperature: float = 0):
        """Initialize the Hartree-Fock Hamiltonian.
        
        Args:
            parameters: Dictionary containing model parameters
            N_shell: Number of shells in k-space
            Nq_shell: Number of shells in q-space
            filling_factor: Electron filling factor
            temperature: Temperature for Fermi-Dirac distribution
        """
        self.lattice = 'square' | 'triangular'
        
        self.N_shell = N_shell
        self.nu = filling_factor
        self.T = temperature
        # define parameters
        #self.param_0 = parameters['param_0'] # Brief phrase explaining physical significance of `param_0`
        #self.param_1 = parameters['param_1'] # Brief phrase explaining physical significance of `param_1`
        #...
        #self.param_p = parameters['param_p'] # Brief phrase explaining physical significance of `param_p`
        # Any other problem specific parameters.

        # define lattice constant
        self.a_M = self.a_G/np.sqrt(epsilon**2 + np.deg2rad(self.theta)**2) # moire lattice constant, nm

        # define k-space within first Brillouin zone (momentum space meshgrid)
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        # define q-space for extended Brillouin zone (reciprocal lattice connnecting different Gamma points)
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)

        # define helper functions
        self.Nk = len(self.k_space)
        self.Nq = len(self.q)

        # degree of freedome including the reciprocal lattice vectors
        self.D = (...,) + (self.Nq)

        # define high symmetry points
        self.high_symm = generate_high_symmetry_points(self.lattice, self.a_M)
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}
    

    def generate_kinetic(self, k=None) -> np.ndarray:
        """Generate kinetic energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Kinetic energy Hamiltonian
        """
        # Generate kinetic terms
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx,ky = k[:,0],k[:,1]
        H_K = np.zeros((self.D+ self.D+ (self.N_k,)), dtype=complex)
        for idx,q in enumerate(self.q):
            kx_total = kx + q
            ky_total = ky + q
            H_K[...] = `code expression computing the kinetic term from the momentum (kx_total,ky_total)`
        return H_K
    
    
    def H_V(self, k=None) -> np.ndarray:
        """Generate potential energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Potential energy Hamiltonian
        """
        # Generate potential terms
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
        H_M = np.zeros((self.D + self.D + (self.Nk,)), dtype=complex)
        # assign V0 to diagonal elements
        for idx, q in enumerate(self.q):
            H_M[0,idx,0,idx,:] = self.V0
            H_M[1,idx,1,idx,:] = self.V0
        
        # assign V1 to off-diagonal elements
        for idx1, q1 in enumerate(self.q):
            for idx2, q2 in enumerate(self.q):
                diff_q = q1 - q2
                for j in self.g.keys():
                    if np.allclose(diff_q, self.g[j]):
                        H_M[0,idx1,0,idx2,:] = self.V1*np.exp(1j*self.psi)
                        H_M[0,idx1,1,idx2,:] = self.V1*np.exp(1j*self.psi)*self.omega**(-j)
                        H_M[1,idx1,0,idx2,:] = self.V1*np.exp(1j*self.psi)*self.omega**(j+1)
                        H_M[1,idx1,1,idx2,:] = self.V1*np.exp(1j*self.psi)*self.omega
                    if np.allclose(diff_q, -self.g[j]):
                        H_M[0,idx1,0,idx2,:] = (self.V1*np.exp(1j*self.psi)).conj()
                        H_M[0,idx1,1,idx2,:] = (self.V1*np.exp(1j*self.psi)*self.omega**(-j)).conj()
                        H_M[1,idx1,0,idx2,:] = (self.V1*np.exp(1j*self.psi)*self.omega**(j+1)).conj()
                        H_M[1,idx1,1,idx2,:] = (self.V1*np.exp(1j*self.psi)*self.omega).conj()
        return H_M

        
    def generate_non_interacting(self, k=None) -> np.ndarray:
        """Generate the non-interacting part of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Non-interacting Hamiltonian (kinetic + potential)
        """
        return self.generate_kinetic(k) + self.H_V(k)

    def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the interacting part of the Hamiltonian.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        exp_val = unflatten(exp_val, self.D, self.Nk)
        H_int = np.zeros(self.D + self.D + (self.Nk,), dtype=complex)

        # If more complicated functions of `exp_val` occur in multiple places,
        # one may add additional functions to the class of the form `func(self, exp_val)`.
        # Eg: the compute_order_parameter(exp_val) function for Emery in Emery_model_upd.
        # Otherwise define dependent expressions below
        #exp0 = function of exp_val
        #exp1 = function of exp_val
        #...
        #exp_e = function of exp_val

        H_int[..., ..., :] = #`code expression corresponding to all terms that contribute to each matrix element`
        #...
        return H_int
        
    def generate_Htotal(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the total Hamiltonian including both non-interacting and interacting parts.
        
        This function is required by the solve() function in HF.py.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Total Hamiltonian (non-interacting + interacting)
        """
        return self.generate_non_interacting() + self.generate_interacting(exp_val)