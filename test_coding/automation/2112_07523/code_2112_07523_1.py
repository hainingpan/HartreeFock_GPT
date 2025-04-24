import numpy as np
from typing import Dict, Any, Tuple
from HF import *

class HartreeFockHamiltonian:
    """Hartree-Fock Hamiltonian for the BHZ model with interactions.
    
    Implements the BHZ (Bernevig-Hughes-Zhang) model with Hartree-Fock interactions
    for electrons and holes with spin.
    """
    def __init__(self, 
                 parameters: Dict[str, Any] = {},
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
        self.lattice = 'square'
        
        self.N_shell = N_shell
        self.nu = filling_factor
        self.T = temperature
        
        # Define parameters with default values
        self.hbar = parameters.get('hbar', 1.0)  # Reduced Planck constant
        self.m_e = parameters.get('m_e', 1.0)    # Effective mass for electrons
        self.m_h = parameters.get('m_h', 1.0)    # Effective mass for holes
        self.E_g = parameters.get('E_g', 1.0)    # Energy gap
        self.A = parameters.get('A', 1.0)        # Coupling parameter
        self.Q = parameters.get('Q', np.array([0.1, 0.0]))  # Momentum vector
        self.epsilon = parameters.get('epsilon', 1.0)  # Dielectric constant
        self.d = parameters.get('d', 1.0)        # Layer separation
        self.V_cc = parameters.get('V_cc', 1.0)  # Coulomb interaction between conduction bands
        self.V_vv = parameters.get('V_vv', 1.0)  # Coulomb interaction between valence bands
        self.V_cv = parameters.get('V_cv', 1.0)  # Coulomb interaction between conduction and valence bands
        self.a_G = parameters.get('a_G', 1.0)    # Lattice constant 
        
        # Define lattice constant
        self.a_M = self.a_G

        # Define k-space within first Brillouin zone
        self.k_space = generate_k_space(self.lattice, N_shell, a=self.a_M)
        # Define q-space for extended Brillouin zone
        self.q_index, self.q = get_q(Nq_shell, a=self.a_M)
        
        # Helper properties
        self.Nk = len(self.k_space)
        self.Nq = len(self.q)
        
        # Define degree of freedom tuple
        self.D = (2, 2, self.Nq)  # (spin, band, reciprocal lattice vector)
        
        # Define high symmetry points
        self.high_symm = generate_high_symmetry_points(self.lattice, self.a_M)
        self.g = {j: rotation_mat(120*(j-1))@self.high_symm["Gamma'"] for j in range(1,4)}

    def generate_kinetic(self, k=None) -> np.ndarray:
        """Generate kinetic energy terms of the Hamiltonian.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Kinetic energy Hamiltonian
        """
        if k is None:
            kx, ky = self.k_space[:,0], self.k_space[:,1]
        else:
            kx, ky = k[:,0], k[:,1]
        
        H_K = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
        
        for n in range(self.Nq):
            # Spin up, conduction band
            H_K[0, 0, n, 0, 0, n, :] = self.hbar**2/(2*self.m_e)*((kx-self.Q[0]/2)**2 + 
                                                                (ky-self.Q[1]/2)**2) + self.E_g/2
            
            # Spin up, conduction to valence coupling
            H_K[0, 0, n, 0, 1, n, :] = self.A * (kx + 1j*ky)
            
            # Spin up, valence to conduction coupling
            H_K[0, 1, n, 0, 0, n, :] = self.A * (kx - 1j*ky)
            
            # Spin up, valence band
            H_K[0, 1, n, 0, 1, n, :] = -self.hbar**2/(2*self.m_h)*((kx+self.Q[0]/2)**2 + 
                                                                  (ky+self.Q[1]/2)**2) - self.E_g/2
            
            # Spin down, conduction band
            H_K[1, 0, n, 1, 0, n, :] = self.hbar**2/(2*self.m_e)*((kx-self.Q[0]/2)**2 + 
                                                                (ky-self.Q[1]/2)**2) + self.E_g/2
            
            # Spin down, conduction to valence coupling
            H_K[1, 0, n, 1, 1, n, :] = -self.A * (kx - 1j*ky)
            
            # Spin down, valence to conduction coupling
            H_K[1, 1, n, 1, 0, n, :] = -self.A * (kx + 1j*ky)
            
            # Spin down, valence band
            H_K[1, 1, n, 1, 1, n, :] = -self.hbar**2/(2*self.m_h)*((kx+self.Q[0]/2)**2 + 
                                                                  (ky+self.Q[1]/2)**2) - self.E_g/2
            
        return H_K
    
    def H_V(self, k=None) -> np.ndarray:
        """Generate potential energy terms of the Hamiltonian.
        
        In this model, potential energy terms are already included in the kinetic energy
        terms because the BHZ model combines them.
        
        Args:
            k: Optional k-points array. If None, use self.k_space
            
        Returns:
            np.ndarray: Potential energy Hamiltonian (zeros in this case)
        """
        return np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
    
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
        
        Implements both Hartree and Fock terms from the mean-field approximation.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Interacting Hamiltonian
        """
        exp_val = unflatten(exp_val, self.D, self.Nk)
        H_int = np.zeros((*self.D, *self.D, self.Nk), dtype=complex)
        
        # Calculate the density matrix 
        rho = np.zeros_like(exp_val)
        for b in range(2):  # band index
            for s in range(2):  # spin index
                for n in range(self.Nq):  # reciprocal lattice index
                    for bp in range(2):  # band index (prime)
                        for sp in range(2):  # spin index (prime)
                            for np_ in range(self.Nq):  # reciprocal lattice index (prime)
                                # Calculate rho based on equation: 
                                # rho(b',s',n',b,s,n,k) = <a^†(b',s',n',k) a(b,s,n,k)> - δ(b,b')δ(b,v)δ(s,s')δ(n,n')
                                rho[bp, sp, np_, b, s, n, :] = exp_val[bp, sp, np_, b, s, n, :]
                                # Subtract the reference configuration (valence bands filled)
                                if bp == b == 1 and sp == s and np_ == n:  # b=b'=valence
                                    rho[bp, sp, np_, b, s, n, :] -= 1.0
        
        # Calculate exciton density
        # n_x = (1/S) * Σ_{s,n,k} ρ_{c,s,n}^{c,s,n}(k) = -(1/S) * Σ_{s,n,k} ρ_{v,s,n}^{v,s,n}(k)
        n_x = 0.0
        for s in range(2):
            for n in range(self.Nq):
                n_x += np.mean(rho[0, s, n, 0, s, n, :])  # Sum over conduction bands
        
        # Hartree term implementation
        for b in range(2):  # band index
            for s in range(2):  # spin index
                for n in range(self.Nq):  # reciprocal lattice index
                    for np in range(self.Nq):  # reciprocal lattice index (prime)
                        # Direct Coulomb interaction 
                        for bp in range(2):  # band index (prime)
                            # Set the interaction potential based on band types
                            if b == 0 and bp == 0:  # conduction-conduction
                                V = self.V_cc
                            elif b == 1 and bp == 1:  # valence-valence
                                V = self.V_vv
                            else:  # conduction-valence or valence-conduction
                                V = self.V_cv
                                
                            # Add the Hartree term
                            for sp in range(2):  # spin index (prime)
                                for npp in range(self.Nq):  # reciprocal lattice index (double prime)
                                    # Hartree contribution from all occupied states
                                    q_factor = np.linalg.norm((np-n)*self.Q) 
                                    # V_{b,b'}((n'-n)*Q) * ρ_{b',s',n''}^{b',s',n''+n'-n}(k')
                                    hartree_contrib = V * np.exp(-q_factor) * np.mean(
                                        rho[bp, sp, npp, bp, sp, npp+(np-n), :]
                                    )
                                    H_int[b, s, np, b, s, n, :] += hartree_contrib / self.Nk
        
        # Special case: for n'=n, add the electrostatic potential energy difference
        # 4πe²n_x d/ε between electron and hole layers
        for s in range(2):  # spin index
            for n in range(self.Nq):  # reciprocal lattice index
                H_int[0, s, n, 0, s, n, :] += 4 * np.pi * n_x * self.d / self.epsilon  # Conduction band
                H_int[1, s, n, 1, s, n, :] -= 4 * np.pi * n_x * self.d / self.epsilon  # Valence band
                
        # Fock term implementation
        for b in range(2):  # band index
            for s in range(2):  # spin index
                for n in range(self.Nq):  # reciprocal lattice index
                    for bp in range(2):  # band index (prime)
                        for sp in range(2):  # spin index (prime)
                            for np in range(self.Nq):  # reciprocal lattice index (prime)
                                # Set the interaction potential based on band types
                                if b == 0 and bp == 0:  # conduction-conduction
                                    V = self.V_cc
                                elif b == 1 and bp == 1:  # valence-valence
                                    V = self.V_vv
                                else:  # conduction-valence or valence-conduction
                                    V = self.V_cv
                                    
                                # Add the Fock term
                                for npp in range(self.Nq):  # reciprocal lattice index (double prime)
                                    # For simplicity, we're using a q-dependent interaction that decreases with distance
                                    q_factor = np.linalg.norm((npp-n)*self.Q + 0.1*(np.random.rand(2)-0.5))  # Add small k'-k term
                                    # -V_{b,b'}((n''-n)*Q + k'-k) * ρ_{b,s,n''}^{b',s',n''+n'-n}(k')
                                    fock_contrib = -V * np.exp(-q_factor) * np.mean(
                                        rho[b, s, npp, bp, sp, npp+(np-n), :]
                                    )
                                    H_int[bp, sp, np, b, s, n, :] += fock_contrib / self.Nk
                                
        return H_int
    
    def generate_Htotal(self, exp_val: np.ndarray) -> np.ndarray:
        """Generate the total Hamiltonian including both non-interacting and interacting parts.
        
        This function is required by the solve() function in HF.py.
        
        Args:
            exp_val: Expected values tensor used to construct the interacting Hamiltonian
            
        Returns:
            np.ndarray: Total Hamiltonian (non-interacting + interacting)
        """
        H_non_int = self.generate_non_interacting()
        H_int = self.generate_interacting(exp_val)
        H_total = H_non_int + H_int
        
        # Return flattened Hamiltonian for solver compatibility
        return flattened(H_total, self.D, self.Nk)
