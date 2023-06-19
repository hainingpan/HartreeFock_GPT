# Preamble
**Prompt:**  
You are a physicist helping me to construct Hamiltonian and perform Hartree-Fock step by step based on my instructions. 
You should follow the instruction strictly.
Your reply should be succinct while complete. You should not expand any unwanted content.
You will be learning background knowledge by examples if necessary.
Confirm and repeat your duty if you understand it.

# Hamiltonian construction

## Construct Kinetic Hamiltonian
**Prompt:**  
You will be instructed to describe the kinetic term of Hamiltonian in {system} in the {real|momentum} space in the {single-particle|second-quantized} form.   
The degrees of freedom of the system are: {dof}  
Express the Kinetic Hamiltonian {symbol} using {var} which are only on the diagonal terms, and arrange the basis in the order of {order}.

Use the following conventions for the symbols:  
{def of var}

## Define each term in Kinetic Hamiltonian
**Prompt:**  
Now you will be instructed to construct each term in the matrix, namely {var}.  
{Def of each var}
Return the expression for {var} in the Kinetic Hamiltonian, and substitute it into the Kinetic Hamiltonian {symbol}.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{def of var}

## Construct Potential Hamiltonian
**Prompt:**  
Now you will be instructed to describe the potential term of Hamiltonian {system} in the {real|momentum} space in the {single-particle|second-quantized} form.  
The potential Hamiltonian has the same degrees of freedom as the kinetic Hamiltonian {symbol}.  
{def of var}.  
Express the potential Hamiltonian {symbol} using {var}.  

Use the following conventions for the symbols (You should also remember the conventions in my previous prompts if there are no conflicts. If you have conflicts in the conventions, you should stop and let me know):  
{def of var}

## Define each term in Potential Hamiltonian
**Prompt:**  
Now you will be instructed to construct each term in the matrix {symbol}, namely, the intralayer potential {terms}.  
{def of each term}  
Return the expressions for {symbol}, and substitute it into the potential Hamiltonian {symbol}.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or have conflicts in the conventions, you should stop and let me know):  
{def of var}

## Construct interaction Hamiltonian (momentum space)
**Prompt:**  
Now you will be instructed to construct the interaction part of the Hamiltonian {symbol} in the momentum space.  
The interaction Hamiltonian is a product of four parts.
The first part is the product of four operators with two creation and two annihilation operators following the normal order, namely, creation operators are before annihilation operators. The two creation operators carry {index}, and {order of momentum}. The two annihilation operators have the same {index}. You should follow the order of $1,2,2,1$ for the {index}, $1,2,3,4$ for the { momentum}. 
The second part is the constraint of total momentum conservation, namely the total momentum of all creation operators should be the same as that of all annihilation operators. [For each operator, the total momentum is the sum of moire reciprocal lattice $b_i$ and momentum with in the first BZ $k_i$.]  
The third part is the interaction form. You should use the bare Coulomb interaction {form}, where $q$ is the transferred momentum between a creation operator and an annilation operator with the same {index}.  
The fourth part is the normalization factor, you should use {normalization factor} here.
Finally, the summation should be running over all {index}, and {momentum}
Return the interaction term {symbol} in terms of {symbol} and $V(q)$ (with $q$ expressed in terms of {momentum}).  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{def of var}

===  
EXAMPLE:  
For a Hamiltonian $H$, where $H=\begin{pmatrix} H_{a,a} & H_{a,b} \\ H_{b,a} & H_{b,b} \end{pmatrix}$ and the order of basis is (a), (b), we can construct the creation operators $\psi_a^\dagger$  and $\psi_b^\dagger$, and the annihilation operator $\psi_a$  and $\psi_b$.  
The corresponding second quantized form is $\hat{H}=\vec{\psi}^\dagger H \vec{\psi}$, where $\vec{\psi}=\begin{pmatrix} \psi_a \\ \psi_b \end{pmatrix}$ and $\vec{\psi}^\dagger=\begin{pmatrix} \psi_a^\dagger & \psi_b^\dagger \end{pmatrix}$. 

## Convert from single-particle to second-quantized form, return in the matrix
**Prompt:**  
Now you will be instructed to construct the second quantized form of the total noninteracting Hamiltonian in the {real|momentum} space.  
The noninteracting Hamiltonian in the {real|momentum} space {symbol} is the sum of Kinetic Hamiltonian {symbol} and Potential Hamiltonian {symbol}.  
To construct the second quantized form of a Hamiltonian. You should construct the creation and annihilation operators from the basis explicitly. You should follow the EXAMPLE below to convert a Hamiltonian from the single-particle form to second-quantized form.  
Finally by "total", it means you need to take a summation over the {real|momentum} space position {$r$|$k$}.   
Return the second quantized form of the total noninteracting Hamiltonian {symbol}  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{def of var}

## Convert from single-particle to second-quantized form, return in summation (expand the matrix)
**Prompt:**  
Now you will be instructed to expand the second-quantized form Hamiltonian {symbol} using {symbols}. You should follow the EXAMPLE below to expand the Hamiltonian.  
You should use any previous knowledge to simplify it. For example, if any term of {meatrix element in single-particle form} is zero, you should remove it from the summation.  
Return the expanded form of {symbol} after simplication.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{def of var}

===  
EXAMPLE:  
For a $\hat{H}=\vec{\psi}^\dagger H \vec{\psi}$, where $\vec{\psi}=\begin{pmatrix} \psi_a \\ \psi_b \end{pmatrix}$ and $\vec{\psi}^\dagger=\begin{pmatrix} \psi_a^\dagger & \psi_b^\dagger \end{pmatrix}$, we can expand it as  $\hat{H}=\sum_{i,j=\{a,b\}} \psi_i^\dagger H_{i,j} \psi_j$.  

## Convert Hamiltonian in real space to momentum space (continuum version)
**Prompt:**  
Now you will be instructed to convert the total noninteracting Hamiltonian in the second quantized form from the basis in real space to the basis by momentum space.  
To do that, you should apply the Fourier transformation to {real space creation op} in the real space to the {momentum space creation op} in the momentum space, which is defined as {def of Fourier Transformation}, where $r$ is integrated over the {entire real space|first Briiloun zone}. You should follow the EXAMPLE below to apply the Fourier transformation.  
Express the total noninteracting Hamiltonian {symbol} in terms of {momentum space creation ops}. Simplify any summation index if possible.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{def of var}

===  
EXAMPLE:  
Write a Hamiltonian $\hat{H}$ in the second quantized form, $\hat{H}=\int dr \psi(r)^\dagger H(r) \psi(r)$, where $r$ is integrated over the entire real space.  
Define the Fourier transformation $c^\dagger(k)=\frac{1}{\sqrt{N V}} \int \psi^\dagger(r) e^{i k \cdot r} dr$, where $r$ is integrated over the entire real space, $N$ is the number of unit cells, and $V$ is the area of the unit cell in the real space.  
This leads to the inverse Fourier transformation $\psi^\dagger(r) = \frac{1}{\sqrt{N \Omega}} \sum_k c^\dagger(k) e^{-i k \cdot r}$, where $k$ is summed over the extended Brillouin zone (i.e., the entire momentum space), $\Omega$ is the area of Brillouin zone in the momentum space.  
Thus, substitute $\psi^\dagger(r)$ and $\psi(r)$ into $\hat{H}$, we get  
$$\hat{H} = \int dr \frac{1}{\sqrt{N \Omega}} \sum_{k_1} c^\dagger(k_1) e^{-i k_1 \cdot r} H(r) \frac{1}{\sqrt{N \Omega}} \sum_{k_2} c(k_2) e^{i k_2 \cdot r} =\sum_{k_1,k_2} c^\dagger(k_1) \frac{1}{N\Omega} \int dr e^{-i (k_1-k_2)\cdot r} H(r) c(k_2) = \sum_{k_1,k_2} c^\dagger(k_1) H(k_1,k_2) c(k_2)$$  
, where we define the Fourier transformation of $H(r)$ as $H(k_1,k_2)=\frac{1}{N\Omega} \int dr e^{-i (k_1-k_2)\cdot r} H(r)$.

**Completion:**  


