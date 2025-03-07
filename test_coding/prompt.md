You are a condensed matter physicist working on the numerical calculation for the Hamiltonian of a system using the Hartree Fock method. I will provide you with the physical Hamitonian written in second quantized form as an equation in LaTeX. You should convert it into a class for the HartreeFockHamiltonian using Python and necessary packages, such as numpy and scipy.
This is a multiple-step problem, and you will execute this process by analyzing the equation and answering smaller questions sequentially in order to finally generate the class. The class has the following structure:

CLASS DOCSTRING: {{DOCSTRING}}

The following function is already defined in the library HF.py. So you don't need to defined them if you need them.
{{HF_DOCSTRING}}

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
from HF import *

class HartreeFockHamiltonian:
  \"""
  Args:
    N_shell (int): Number shell in the first Broullouin zone.
    parameters (dict): Dictionary containing model parameters 't' and 'U'.
  \"""
  def __init__(self, N_shell: int, parameters: dict[str, Any]={'t':1.0, 'U':1.0, 'T':0, 'a':1.0},filling_factor: float=0.5):
    self.lattice = 'square'   # Lattice symmetry ('square' or 'triangular'). Defaults to 'square'.
    self.D = (2,) # Number of flavors identified.
    self.basis_order = {'0': 'spin'}
    # Order for each flavor:
    # 0: spin up, spin down

    # Occupancy relevant parameters
    self.nu = filling_factor
    self.T = parameters['T'] # temperature, default to 0
    self.a = parameters['a'] # Lattice constant
    self.k_space = generate_k_space(self.lattice, N_shell, self.a)
    self.N_k = self.k_space.shape[0]

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
    H_nonint = np.zeros((*self.D,*self.D,self.N_k), dtype=complex)
    # Kinetic energy for spin up and spin down.
    # They are identical in this case, but we keep them separate for clarity
    H_nonint[0, 0, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  
    H_nonint[1, 1, :] = -2 * self.t * (np.cos(self.k_space[:, 0]) + np.cos(self.k_space[:, 1]))  
    return H_nonint

  def generate_interacting(self, exp_val: np.ndarray) -> np.ndarray:
    \"""
    Generates the interacting part of the Hamiltonian.

    Args:
      exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

    Returns:
      np.ndarray: The interacting Hamiltonian with shape (D, D, N_k).
    \"""
    exp_val = unflatten(exp_val, self.D, self.N_k)
    H_int = np.zeros((*self.D,*self.D,self.N_k), dtype=complex)

    # Calculate the mean densities for spin up and spin down
    n_up = np.mean(exp_val[0, 0, :]) # <c_{k',up}^\dagger c_{k',up}>
    n_down = np.mean(exp_val[1, 1, :]) # <c_{k',down}^\dagger c_{k',down}>

    # Hartree-Fock terms
    H_int[0, 0, :] = self.U * n_down # Interaction of spin up with average spin down density
    H_int[1, 1, :] = self.U * n_up # Interaction of spin down with average spin up density
    return H_int

  def generate_Htotal(self, exp_val: np.ndarray, return_flat: bool=True) ->np.ndarray:
    \"""
      Generates the total Hartree-Fock Hamiltonian.

      Args:
          exp_val (np.ndarray): Expectation value array with shape (D_flattened, D_flattened, N_k).

      Returns:
          np.ndarray: The total Hamiltonian with shape (D, D, N_k).
    \"""
    H_nonint = self.generate_non_interacting()
    H_int = self.generate_interacting(exp_val)
    H_total = H_nonint + H_int
    if return_flat:
      return flattened(H_total,self.D,self.N_k)
    else:
      return H_total
```

===
Now generate the code, following the same format as the example, for the following Hamiltonian:
HAMILTONIAN EQUATION: {{HAMILTONIAN}}
LATTICE: {{SYMMETRY}}
ANSWER:
"""