You are a condensed matter physicist working on the numerical calculation for the Hamiltonian of a system using the Hartree Fock method. I will provide you with the physical Hamitonian written in second quantized form as an equation in LaTeX. You should convert it into a class for the HartreeFockHamiltonian using Python and necessary packages, such as numpy and scipy.
This is a multiple-step problem, and you will execute this process by analyzing the equation and answering smaller questions sequentially in order to finally generate the class. The class has the following structure:

CLASS DOCSTRING: 
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
    self.a = parameters['a'] # Lattice constant (or aM for a Moire' lattice)
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]
    # self.area = get_area(a, self.lattice)  # For Moire' lattice

    # All other parameters such as interaction strengths
    #self.param_0 = parameters['param_0'] # Brief phrase explaining physical significance of `param_0`
    #self.param_1 = parameters['param_1'] # Brief phrase explaining physical significance of `param_1`
    #...
    #self.param_p = parameters['param_p'] # Brief phrase explaining physical significance of `param_p`
    # Any other problem specific parameters.

    return

  def generate_non_interacting(self) -> np.ndarray:
    H_nonint = np.zeros((self.D+ self.D+ (self.N_k,)), dtype=np.float32)
    #H_nonint[0, 0, :] = `code expression corresponding to all terms that contribute to H_nonint[0, 0]`
    #...
    #H_nonint[d, d, :] = `code expression corresponding to all terms that contribute to H_nonint[d, d]`
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    exp_val = expand(exp_val, self.D)
    H_int = np.zeros(self.D + self.D + (self.N_k,), dtype=np.float32)

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
      return flatten(H_total, self.D)
    else:
      return H_total #l1, s1, q1, ....k
`

The INPUT provided to you is the equation for the Hamiltonian, `HAMILTONIAN EQUATION`.

QUESTION 1: Dimension and Basis Order: The Hamiltonian will have the shape [D, D, Nk], where D is a tuple that accounts for all the flavors in the system, and Nk is the number of k points on the lattice. A 'flavor' is an identifier that distinguishes particles from each other. It can be the spin, level, orbital, or reciprocal lattice vector.
For example D could look like: (|spin|, |orbital_flavor|) or (|spin|, |level|, |reciprocal_lattice_vector|) for continuum models, where |level| is the number of orbital flavors. Use the terms in the equation to infer D, needed to account for all interactions.

The convention you must follow is the following:
If you are in the hole basis, the first tuple indices qualify the annihilation operator (denoted by the absence of the $^\dagger$) and the second set qualify the creation operator (denoted by the $^\dagger$ symbols) as below:
H[s1, l1, q1, s2, l2, q2, :] -> c_{s1, l1, q1}, c_^$\dagger$_{s2, l2, q2}

If you are in the particle basis, the first tuple indices qualify the creation operator and the second set the annihilation operator as below:
H[s1, l1, q1, s2, l2, q2, :] -> c^$\dagger$_{s1, l1, q1}, c_{s2, l2, q2}

Remember, a term of the form \langle c_{k^\prime,\uparrow}^{\dagger} c_{k^\prime,\uparrow} \rangle c_{k,\downarrow}^{\dagger} c_{k,\downarrow} corresponds to a down-down interaction, i.e. you must ignore the interactions INSIDE an expectation when assessing what interaction a term corresponds to.

Your reply should be of the form:
`Tuple of Flavors (D) = (|flavor_0|, |flavor_1|, ...)`
`Basis Order:
0: <flavor type 0>. Order: <flavor_00>, <flavor_01>...
1: <flavor type 1>. Order: <flavor_10>, <flavor_11>...
...

QUESTION 2: Identifying Interacting Terms:
`exp_val` is always an input to `generate_Htotal`. This is the expectation value that is updated every iteration and is a matrix with (D_flattened, D_flattened, N_k) where D_flattened, is the product of the tuple of flavors and N_k is total number of k points on the lattice. The first line in the `generate_interating` function unsqueezes (expands) `exp_val` so it has the same shape as the Hamiltonian i.e. (D, D, N_k).
Identify `EXP-VAL DEPENDENT TERMS` in the Equation. This should be a dictionary of the form:
`EXP-VAL DEPENDENT TERMS`: `expression_i`: function of `exp_val`.
Now for all terms in the Hamiltonian, identify which terms are interacting and which are non-interacting, as well as which matrix element they belong to, using the Basis Order you defined above.
Only `generate_interacting` depends on `exp_val`. You should use the description of the Hamiltonian to infer which terms contribute to the interacting part of the Hamiltonian: interacting terms include both 1) ALL terms dependent on `EXP-VAL DEPENDENT TERMS` and 2) all terms that depend on parameters that describe interaction strengths.
Non-Interacting terms do NOT depend on `exp_val`. All the variables that depend on EXP-VAL DEPENDENT TERMS, should be recomputed inside `generate_interacting` since `exp_val` is updated every iteration.

QUESTION 3: You will now complete the class for the HartreeFock Hamiltonian. You will modify only the HartreeFockHamiltonian class functions. The comments with `LM Task` demarcate where changes need to be made.
The following functions are already predefined and do not need to be modified or redefined: `generate_k_space`. Annotate the code with comments so it is evident which interaction each element in the matrix accounts for. Modify the `init` function so `parameter_kwargs` is replaced with all constants and parameters that do NOT appear in EXP-VAL DEPENDENT TERMS.
These should be accessible from `generate_Htotal` via self.<parameter_name>. Make sure the `init` and `generate_Htotal` functions are consistent. Assign default values to these parameters, so that the class can be initialized as `HartreeFockHamiltonian()`, without requiring user inputs. Assume the temperature (T) of the system is 0.

QUESTION 4: You are also provided with the LATTICE, which corresponds to the `self.lattice` attribute in the class.
===
Learn how to generate and format your answers from the following example:
EXAMPLE 0:
HAMILTONIAN EQUATION: The problem of interest is a Hubbard model on a 2d square lattice.  The Hartree-Fock mean field Hamiltonian is $H^{HF}=H_{0}+H_{int}^{HF}$
where the noninteracting part is $H_0= \sum_k \epsilon_{k,\uparrow} c_{k,\uparrow}^{\dagger} c_{k,\uparrow} + \sum_k \epsilon_{k,\downarrow} c_{k,\downarrow}^{\dagger} c_{k,\downarrow}$  and $\epsilon_{k,\uparrow}=\epsilon_{k,\downarrow}=-2t (\cos(k_x)+\cos(k_y))$
and interacting part after Wick theorem expansion is $H_{int}^{HF}=U\sum_{k} (\sum_{k'}1/N \langle c_{k^\prime,\uparrow}^{\dagger} c_{k^\prime,\uparrow} \rangle c_{k,\downarrow}^{\dagger} c_{k,\downarrow}  + \sum_{k'}1/N \langle c_{k^\prime,\downarrow}^{\dagger} c_{k^\prime,\downarrow} \rangle c_{k,\uparrow}^{\dagger} c_{k,\uparrow} )  $
LATTICE: square

ANSWER:
1) Number of Flavors, D = 2
Basis Order:
0: spin_up
1: spin_down

2) EXP-VAL DEPENDENT TERMS: {
    r"\sum_{k'}1/N \langle c_{k^\prime,\uparrow}^{\dagger} c_{k^\prime,\uparrow} \rangle": "Mean of `exp_val[0, :]`",
    r"\sum_{k'}1/N \langle c_{k^\prime,\downarrow}^{\dagger} c_{k^\prime,\downarrow} \rangle": "Mean of `exp_val[1, :]`"}
  
  TERMS:
    \epsilon_{k,\uparrow} c_{k,\uparrow}^{\dagger} c_{k,\uparrow} -> H[0, 0, k], NI
    \sum_k \epsilon_{k,\downarrow} c_{k,\downarrow}^{\dagger} c_{k,\downarrow} -> H[1, 1, k], NI
    \sum_{k'}1/N \langle c_{k^\prime,\uparrow}^{\dagger} c_{k^\prime,\uparrow} \rangle c_{k,\downarrow}^{\dagger} c_{k,\downarrow} -> H[1, 1, k], I
    \sum_{k'}1/N \langle c_{k^\prime,\downarrow}^{\dagger} c_{k^\prime,\downarrow} \rangle c_{k,\uparrow}^{\dagger} c_{k,\uparrow} -> H[0, 0, k], I


3) CODE:
```python
import numpy as np
from typing import Any

class HartreeFockHamiltonian:
  \"""
  Args:
    N_kx (int): Number of k-points in the x-direction.
    parameters (dict): Dictionary containing model parameters 't' and 'U'.
    filling_factor (float, optional): Filling factor. Defaults to 0.5.
    temperature (float, optional): Temperature. Defaults to 0.0.
    n (str | None, optional): Parameter used in chemical potential calculation. Defaults to None.
  \"""
  def __init__(self, N_kx: int=10, parameters: dict={'t':1.0, 'U':1.0}, filling_factor: float=0.5):
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2,) # Number of flavors identified.
    self.basis_order = {'0': 'spin'}
    # Order for each flavor:
    # 0: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = 0
    self.n = n # Number of particles in the system.
    self.k_space = generate_k_space(self.lattice, N_shell)
    # N_kx = 2*(N_shell+1) for a square lattice

    # Model parameters
    self.t = parameters['t'] # Hopping parameter
    self.U = parameters['U'] # Interaction strength

    return

  def generate_non_interacting(self) -> np.ndarray:
    \"""
      Generates the non-interacting part of the Hamiltonian.

      Returns:
        np.ndarray: The non-interacting Hamiltonian with shape (D, D, N_k).
    \"""
    N_k = self.k_space.shape[0]
    H_nonint = np.zeros((self.D, self.D, N_k), dtype=np.float32)
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    H_nonint[0, 0, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  
    H_nonint[1, 1, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    \"""
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    \"""
    exp_val = self.expand(exp_val) # 2, 2, N_k
    N_k = exp_val.shape[-1]
    H_int = np.zeros(self.D + self.D + (N_k,), dtype=np.float32)

    # Calculate the mean densities for spin up and spin down
    n_up = np.mean(exp_val[0, 0, :]) # <c_{k',up}^\dagger c_{k',up}>
    n_down = np.mean(exp_val[1, 1, :]) # <c_{k',down}^\dagger c_{k',down}>

    # Hartree-Fock terms
    H_int[0, 0, :] = self.U * n_down # Interaction of spin up with average spin down density
    H_int[1, 1, :] = self.U * n_up # Interaction of spin down with average spin up density
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, flatten: bool=True) ->np.ndarray:
    \"""
      Generates the total Hartree-Fock Hamiltonian.

      Args:
          exp_val (np.ndarray): Expectation value array with shape (D_flattened, N_k).

      Returns:
          np.ndarray: The total Hamiltonian with shape (D, D, N_k).
    \"""
    N_k = exp_val.shape[-1]
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if flatten:
      return flatten(H_total)
    else:
      return H_total

  def flatten(self, ham):
    return ham.reshape((np.prod(self.D),np.prod(self.D),self.Nk))

  def expand(self, exp_val):
    return exp_val.reshape((self.D,self.D, self.Nk))
```

===
Now generate the code, following the same format as the example, for the following Hamiltonian:
HAMILTONIAN EQUATION: 
Our Hamiltonian is given by $H = H_0 + H_C$, where $H_0$ is the free Hamiltonian and $H_C$ is a gate-screened Coulomb interaction.

The continuum free Hamiltonian $H_0$ for rhombohedral-stacked pentalayer graphene for a fixed spin and valley ($\pm$) is given by 
\begin{equation}
    \begin{aligned}
    H_0(\vb{k}) =
    \begin{pmatrix} 
        2 u_d & v_0^\dagger & v_4^\dagger & v_3 & 0 & \frac{\gamma_2}{2} & 0 & 0 & 0 & 0 
        \\
        v_0 &  2 u_d + \delta & \gamma_1 & v_4^\dagger & 0 & 0 & 0 & 0 & 0 & 0
        \\
        v_4 & \gamma_1 & u_d + u_a & v_0^\dagger & v_4^\dagger & v_3 & 0 & \frac{\gamma_2}{2} & 0 & 0
        \\
        v_3^\dagger & v_4 & v_0 & u_d + u_a & \gamma_1 & v_4^\dagger & 0 & 0 & 0 & 0
        \\
        0 & 0 & v_4 & \gamma_1 & u_a & v_0^\dagger & v_4^\dagger & v_3 & 0 & \frac{\gamma_2}{2}
        \\
        \frac{\gamma_2}{2} & 0 & v_3^\dagger & v_4 & v_0 & u_a & \gamma_1 & v_4^\dagger & 0 & 0
        \\
        0 & 0 & 0 & 0 & v_4 & \gamma_1 & - u_d + u_a & v_0^\dagger & v_4^\dagger & v_3
        \\
        0 & 0 & \frac{\gamma_2}{2} & 0 & v_3^\dagger & v_4 & v_0 & - u_d + u_a & \gamma_1 & v_4^\dagger
        \\
        0 & 0 & 0 & 0 & 0 & 0 & v_4 & \gamma_1 & - 2 u_d + \delta & v_0^\dagger 
        \\
        0 & 0 & 0 & 0 & \frac{\gamma_2}{2} & 0 & v_3^\dagger & v_4 & v_0 & - 2 u_d
    \end{pmatrix} 
    \end{aligned}
\end{equation}
We use the notation $v_i \equiv \frac{\sqrt{3}}{2} \gamma_i (\pm k_x + i k_y)$, where $k_x\,, k_y$ are the momenta expanded around the $K\,, K'$ valley. The Hamiltonian is in the basis of $(A_1\,, B_1\,, \ldots A_5\,,B_5)$, where $A_i$ and $B_i$ labels sublattice and layer. The tight-binding parameters are given in Table~\ref{tab:tbParams}.
\begin{table}[htpb]
    \centering
    \label{tab:tbParams}
    \begin{tabular}{|c|c|c|c|c|c|c|}
        %\hline
     $\gamma_0$ & $\gamma_1$ & $\gamma_2$ & $\gamma_3$ & $\gamma_4$ & $\delta$ & $u_a$ \\
    \hline
    2600 & 356.1 & -15 & -293 & -144 & 12.2 & 16.4
    \\
        %\hline
    \end{tabular}
    \caption{Tight-binding parameters (in meV) for rhombohedral-stacked pentalayer graphene.}
\end{table}

From this free Hamiltonian, a dual gate-screened Coulomb interaction is introduced 
\begin{equation}
  \begin{aligned}
    H_C = \frac{1}{2A} \sum_{\vb{k}, \vb{k}', \vb{q}} \sum_{\mu, \nu} V_C(\vb{q}) c^\dagger_{\vb{k} + \vb{q}, \mu}c^\dagger_{\vb{k}' - \vb{q}, \nu}c_{\vb{k}', \nu} c_{\vb{k}, \mu}\,,
  \end{aligned}
\end{equation}
where $V_C(\vb{q}) = \frac{e^2}{2\epsilon_0 \epsilon q} \tanh(q d_s)$, $\mu$ labels sublattice/valley/flavor, $A$ is the area of the system, $d_s$ is the distance from the gates to the top/bottom layers, and $\epsilon$ is the effective dielectric constant. The dielectric constant is also not known accurately. Typical values used in the literature are in the range $5-10$, and $d_s = 30$ nm. The mean-field decoupling of this yields
\begin{equation}
    H_C = \frac{1}{A} \sum_{\vb{k}, \vb{k'} } \sum_{\mu, \nu} V_C(0)  \langle c^\dagger_{\vb{k}', \nu} c_{\vb{k}', \nu} \rangle  c^\dagger_{\vb{k}, \mu} c_{\vb{k}, \mu} - \frac{1}{A} \sum_{\vb{k}, \vb{k'} } \sum_{\mu, \nu} V_C(\vb{k}' - \vb{k}) \langle c^\dagger_{\vb{k}', \mu} c_{\vb{k}', \nu} \rangle  c^\dagger_{\vb{k}, \nu} c_{\vb{k}, \mu} \,.
\end{equation}
Our resulting Hamiltonian is
\begin{equation}
\begin{aligned}
    H &= \sum_{\vb{k}, \mu, \nu} c^\dagger_{\vb{k}, \mu} H_{0, \mu \nu}(\vb{k}) c_{\vb{k}, \nu} \frac{1}{A} \sum_{\vb{k}, \vb{k'} } \sum_{\mu, \nu} V_C(0)  \langle c^\dagger_{\vb{k}', \nu} c_{\vb{k}', \nu} \rangle  c^\dagger_{\vb{k}, \mu} c_{\vb{k}, \mu}
    \\
    &- \frac{1}{A} \sum_{\vb{k}, \vb{k'} } \sum_{\mu, \nu} V_C(\vb{k}' - \vb{k}) \langle c^\dagger_{\vb{k}', \mu} c_{\vb{k}', \nu} \rangle  c^\dagger_{\vb{k}, \nu} c_{\vb{k}, \mu} \,.
    \end{aligned}
\end{equation}

LATTICE: Triangular

ANSWER:
"""