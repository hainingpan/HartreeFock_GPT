# Preamble 
You are a physicist working on numerical calculation for a Hubbard model using Hartree Fock calculation. 
I will provide you the physical Hamiltonian written in second quantized form.
You should convert them into a function using Python and necessary packages, such as Numpy and scipy.
This is a multiple-step problem, and I will divide them into bite-size subproblems. So you need to code each subproblem and encapsulate each of them into a function. I will tell you the input and output of the function.
The general framework of this code is a `class` object in Python, where `self.__init__` sets the parameters.
The problem of interest is a Hubbard model on a 2d square lattice.  The Hartree-Fock mean field Hamiltonian is $H^{HF}=H_{0}+H_{int}^{HF}$
where the noninteracting part is $H_0= \sum_k \epsilon_{k,\uparrow} c_{k,\uparrow}^{\dagger} c_{k,\uparrow} + \sum_k \epsilon_{k,\downarrow} c_{k,\downarrow}^{\dagger} c_{k,\downarrow}$  and $\epsilon_{k,\uparrow}=\epsilon_{k,\downarrow}=-2t (\cos(k_x)+\cos(k_y))$ 
and interacting part after Wick theorem expansion is $H_{int}^{HF}=U\sum_{k} (\sum_{k'}1/N \langle c_{k^\prime,\uparrow}^{\dagger} c_{k^\prime,\uparrow} \rangle c_{k,\downarrow}^{\dagger} c_{k,\downarrow}  + \sum_{k'}1/N \langle c_{k^\prime,\downarrow}^{\dagger} c_{k^\prime,\downarrow} \rangle c_{k,\uparrow}^{\dagger} c_{k,\uparrow} )  $ 

You can start to build up a generic framework that only includes the initialized function`__init__` now. You can leave the content empty. Then I will tell you each step and you should dynamically update the content of the function.

# generate_k_space
You should create a function `generate_k_space` to generate the mesh grid of k points in the first Brillouin zone in the momentum space. 
No input is needed.
kx, and ky should be equally divided into `self.N` points in both axes. To avoid double counting on the boundary first Brillouin zone, we conventionally include only the negative side of the boundary of the Brillouin zone, i.e.,  including -pi, but excluding +pi.
The output should be a 2D array with the shape of (N^2,2), where the first axis is the number of total k points in the momentum space, the second axis is the kx, and ky component of each k point.

Finally, you should update this function into `__init__` and store the return value.

# generate_noninteracting
You should create a function to construct the noninteracting term `generate_noninteracting`

No input is needed.
You should compute $\epsilon_{k,\uparrow}$ and $\epsilon_{k,\downarrow}$, by substituting each k points.
The output should be a 3d array with the shape of (2,2,N^2), the first axis and second axis are both for spin up \uparrow and spin down \downarrow. The third axis is for different k points.

# generate_H_int
You should create a function to construct the interacting term after Hartree Fock decomposition `generate_H_int`

The input is an expected value `exp_val` with the shape of (2, N^2), where the first index is for spin up  \uparrow and down \downarrow, and the second index is for different k points. `exp_val` will be defined later, 
The output is also a 3d array with the shape of (2,2,N^2), the first index and second index are both for spin up \uparrow and spin down \downarrow. The third axis is for different k points.

# get_occupancy
You should create a function for the occupancy of each state at each k point, `get_occupancy``.

The first input is the energies, `en` with the shape of (2,N^2). The first index is the level index. The second index is for different k points. 
The second input is the filling factor `nu`. 
You should first determine the  Fermi level based on the filling factor. Namely, choose the correct energy (Fermi level) such that the proportion of the state with energy below the Fermi level is just the filling factor. The Fermi level is a universal value for all spins.
Then, compute the occupancy of each k point by ensuring that all states below the Fermi level are occupied and all states above the Fermi level are empty.  Thus, the occupancy will have the same shape of the array as the energy `en`.

The output is the occupancy with the shape of (2, N^2). The first index is the level index. The second index is for different k points. 

# compute_exp_val
You should create a function to compute the expected value (`exp_val`) from the wavefunction, `compute_exp_val`.

The first input is the wavefunction `wf` with the shape of (2, 2, N^2). The first index is for spin up \uparrow and down \downarrow. The second index is the level index. The third index is for different k points. Namely, the wave function is arranged along each column. 
The second input is the occupancy of each state at each k point, with the shape of (2, N^2). The first index is the level index. The second index is for different k points. 
The expected value is defined as $\langle c_{k,s}^{\dagger} c_{k,s} \rangle$, where s is the spin index.
Because the expected value $\langle .. \rangle$ is averaged over the "level index", it effectively carries a "weight" which is the occupancy of each k point
THus, the expected value is a product of three terms---  (1) the occupancy, (2) the ket version of `wf`, (2) bra version of `wf` (remember the hermitian conjugate). 
The output is the expected value `exp_val`  with the shape of (2, N^2), where the first index is for spin up  \uparrow and down \downarrow, and the second index is for different k points.
To efficiently compute the summation, you should avoid the loop but use `np.einsum` to perform the tensor contraction. 
The order and number of indices in `np.einsum` are very important.
So before you write anything, you should first reiterate the contraction rule.
You need to first remind yourself what are the indices of four tensors (`occupancy`, `wf`, the conjugate of `wf`, and the output `exp_val`).
Keep an eye on the number and order of each index. 
Note that only indices with the same physical meaning can be contracted. 

# generate_total
You should create a function to compute the total Hamiltonian `generate_total`.

The input is the interacting term with the shape of (2,2,N^2), the first index and second index are both for spin up \uparrow and spin down \downarrow. The third axis is for different k points.
The total Hamiltonian is the sum of the noninteracting term `H0``(which was already computed before) and the interacting term.
The output is the total Hamiltonian with the shape of (2,2,N^2), the first index and second index are both for spin up \uparrow and spin down \downarrow. The third axis is for different k points.

# diagonalize
You should create a function to diagonalize the total Hamiltonian for each k point, `diagonalize`.

The input is the total Hamiltonian with the shape of (2,2,N^2), the first index and second index are both for spin up \uparrow and spin down \downarrow. The third axis is for different k points.
You should first make H_total hermitian by symmetrizing the first and second index.
You should then loop over each k point, compute the full eigenvector and eigenvalue of each (2,2) matrix, and sort them by the value of eigenvalue.
The first output is the eigenvector, i.e., wavefunction `wf`, with the shape of (2, 2, N^2). The first index is for spin up \uparrow and down \downarrow. The second index is the level index. The third index is for different k points. Namely, the wave function is arranged along each column. 
The second output is the eigenvalue, i.e., energies `en` with the shape of (2,N^2). The first index is the level index. The second index is for different k points. 

# get_energy
You should now create a function to compute the eigenenergy from the expected value, `get_energy` by wrapping up previously defined functions.

The input is `exp_val`  with the shape of (2, N^2), where the first index is for spin up  \uparrow and down \downarrow, and the second index is for different k points.
First, compute the interacting term `H_int` from the expected value `exp_val``.
Second, compute the total energy from `H_int`.
Finally, diagonalize the total energy `H_total`.
The first output is the wavefunction `wf` with the shape of (2, 2, N^2). The first index is for spin up \uparrow and down \downarrow. The second index is the level index. The third index is for different k points. Namely, the wave function is arranged along each column. 
The second output is the eigenvalue, i.e., energies `en` with the shape of (2,N^2). The first index is the level index. The second index is for different k points. 

# get_exp_val
You should now create a function to compute the expected value directly from wavefunction and eigeneneries, and the filling factor, called `get_exp_val`.

The first input is the wavefunction `wf` with the shape of (2, 2, N^2). The first index is for spin up \uparrow and down \downarrow. The second index is the level index. The third index is for different k points. Namely, the wave function is arranged along each column. 
The second input is the energies, `en`` with the shape of (2,N^2). The first index is the level index. The second index is for different k points. 
The third input is the filling factor `nu`. 
First, compute the occupancy from the energies `en` and filling factor `nu`.
Then, compute the expected value from wavefunction `wf` and occupancy `occ`.
The output is the expected value `exp_val`  with the shape of (2, N^2), where the first index is for spin up  \uparrow and down \downarrow, and the second index is for different k points.

# solve
You should now create a function to self-consistently solve the problem, `solve`.

The first input is the initial ansatz for the expected value `exp_val_0`,  with the shape of (2, N^2), where the first index is for spin up  \uparrow and down \downarrow, and the second index is for different k points.
The second input is the filling factor `nu`.
To solve it self consistently, you should iterate between (1) getting the energy and wavefunction from the expected value and (2) getting the expected value from energy, wavefunction and filling factor. 
You should start with the initial ansatz for the expected value `exp_val_0``, and loop for 100 times
The first output is the wave function with the shape of (2, 2, N^2). The first index is for spin up \uparrow and down \downarrow. The second index is the level index. The third index is for different k points. Namely, the wave function is arranged along each column. 
The second output is the eigenvalue, i.e., energies `en` with the shape of (2,N^2). The first index is the level index. The second index is for different k points. 
The third output is the expected value `exp_val`  with the shape of (2, N^2), where the first index is for spin up  \uparrow and down \downarrow, and the second index is for different k points.

