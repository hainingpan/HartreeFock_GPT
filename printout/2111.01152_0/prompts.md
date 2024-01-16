# Preamble
## One-shot Example
###
I am reading a paper with the following abstract. I want you to first extract the essential information from it and then execute my instructions in order to create the Hamiltonian in the Hartree Fock Approximation. 

To extract information, I will provide you with a set of questions about the calculation (this type of prompt will start with **Extraction**). You should first try your best to answer them. You should first quote the relevant part from the abstract which you think can lead to an answer.
Then, you provide your rationale from this quote to the answer.
Finally, you should give a succinct final version of the answer.

To execute the calculation, I will provide a set of tasks. You should follow the instructions strictly. Your reply should be succinct while complete. You should not expand any unwanted content.
You will be learning background knowledge by examples if necessary.

A crucial part of execution is two-step: 
(1) fill in the paper-specific information from the extraction into a provided template (this prompts will start with **Execution--Core information**); 
(2) combine with the predetermined notations provided by me (this prompts will start with **Execution--Notation**), and then, you should proceed to execute the instruction. 
Note that there might be cases where no paper-specific information is needed from the abstract in certain execution tasks.

I will provide three examples (Example 1, Example 2, and Example 3) for **Extraction**, **Execution--Core information**, and **Execution**--Notation**, respectively, for you to learn.
Confirm and repeat your duty if you understand it.

Example 1

```

**Extraction**
**Abstract**
We present a theory on the quantum phase diagram of AB-stacked MoTe$_2$/WSe$_2$ using a self-consistent Hartree-Fock calculation performed in the plane-wave basis, motivated by the observation of topological states in this system. At filling factor $\nu=2$ (two holes per moir\'e unit cell), Coulomb interaction can stabilize a $\mathbb{Z}_2$ topological insulator by opening a charge gap. At $\nu=1$, the interaction induces three classes of competing states, spin density wave states, an in-plane ferromagnetic state, and a valley polarized state, which undergo first-order phase transitions tuned by an out-of-plane displacement field. The valley polarized state becomes a Chern insulator for certain displacement fields.  Moreover, we predict a topological charge density wave forming a honeycomb lattice with ferromagnetism at $\nu=2/3$. Future directions on this versatile system hosting a rich set of quantum phases are discussed.

**Question**:
1. What are the degrees of freedom that should be included in the Hamiltonian?
2. [Another question]

```

You response should be:

```

1. What are the degrees of freedom that should be included in the Hamiltonian?
**Quote**: AB-stacked MoTe$_2$/WSe$_2$ 
**Explanation**:  First the studied system is MoTe$_2$/WSe$_2$ , which is a heterostructure of bilayer. Therefore, layer is a degree of freedom that should be included in the Hamiltonian. Secod, because MoTe$_2$ and WSe$_2$   are both TMD materials which have strong Ising type spin orbit coupling. This coupling ties the electron's spin to its valley index, resulting in distinct energy levels for spins aligned parallel or antiparallel to the orbital angular momentum. Since MoTe$_2$ and WSe$_2$ usually have the topmost valence band at +K/-K valley. This means the valley index is coupled to the spin index. Therefore, only two types of degrees of freedom (layer and valley) should be considered in the Hamiltonian.
**Anwser**: layer, valley
2. [Another question]
[Another answer]

```

Example 2

```

**Execution--Core information**
You should fill in the specific information from above into the following template, and then repeat it. 
'''

You will be instructed to describe the kinetic term of Hamiltonian in {system of interest}.

The degrees of freedom of the system are:{independent degree of freedom} 
''' 
```


Your response should be:

```

You will be instructed to describe the kinetic term of Hamiltonian in AB-stacked MoTe$_2$/WSe$_2$.
The degrees of freedom of the system are: Layer and the combined spin-valley degree. 

```

Example 3

```

**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
Express the Kinetic Hamiltonian $H_T$ using $E_{+K,b}$, $E_{-K,t}$, $E_{-K,b}$, $E_{-K,t}$ which are only on the diagonal terms, and arrange the basis in the order of (+K,bottom), (+K,top), (-K,bottom), (-K,top). 

Use the following conventions for the symbols:

$\tau=\pm 1$ represents $\pm K$ valleys, 
bottom layer ($\mathfrak{b}$) and top layer ($\mathfrak{t}$)

'''

```


Your response should be:


```

**Instructions to execute**:
You will be instructed to describe the kinetic term of Hamiltonian in AB-stacked MoTe$_2$/WSe$_2$.
The degrees of freedom of the system are: Layer and the combined spin-valley degree. 
Express the Kinetic Hamiltonian $H_T$ using $E_{+K,b}$, $E_{-K,t}$, $E_{-K,b}$, $E_{-K,t}$ which are only on the diagonal terms, and arrange the basis in the order of (+K,bottom), (+K,top), (-K,bottom), (-K,top). 

Use the following conventions for the symbols:

$\tau=\pm 1$ represents $\pm K$ valleys, 
bottom layer ($\mathfrak{b}$) and top layer ($\mathfrak{t}$)

**Answer**
[The answer to the above 'Instructions to execute' ]
```

# Identify kinetic term (continuum,first-quantized)
## Extraction
###
**Extraction**
**Abstract**
We present a theory on the quantum phase diagram of AB-stacked MoTe$_2$/WSe$_2$ using a self-consistent Hartree-Fock calculation performed in the plane-wave basis, motivated by the observation of topological states in this system. At filling factor $\nu=2$ (two holes per moir\'e unit cell), Coulomb interaction can stabilize a $\mathbb{Z}_2$ topological insulator by opening a charge gap. At $\nu=1$, the interaction induces three classes of competing states, spin density wave states, an in-plane ferromagnetic state, and a valley polarized state, which undergo first-order phase transitions tuned by an out-of-plane displacement field. The valley polarized state becomes a Chern insulator for certain displacement fields.  Moreover, we predict a topological charge density wave forming a honeycomb lattice with ferromagnetism at $\nu=2/3$. Future directions on this versatile system hosting a rich set of quantum phases are discussed.

**Question**:
1. What is the system to study?
2. What are the independent degrees of freedom that should be included in the Hamiltonian, namely, the independent flavor of the creation/annihilation operator? Remember the system is spin-valley locked.


## Execution--Core information
### 
**Execution--Core information**
You should fill in the specific information from above into the following template, and then repeat it. 
'''
You will be instructed to describe the kinetic term of Hamiltonian in {system of study}.
The degrees of freedom of the system are: {independent degree of freedom} 
''' 


## Execution: Notation:
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
Express the Kinetic Hamiltonian $H_T$ using $E_{+K,b}$, $E_{-K,t}$, $E_{-K,b}$, $E_{-K,t}$ which are only on the diagonal terms, and arrange the basis in the order of (+K,bottom), (+K,top), (-K,bottom), (-K,top). 

Use the following conventions for the symbols:

$\tau=\pm 1$ represents $\pm K$ valleys, 
bottom layer ($\mathfrak{b}$) and top layer ($\mathfrak{t}$)
'''



# Define energy dispersion (continuum)
## Extraction
### 
**Extraction**
**Extraction**
**Abstract**
We present a theory on the quantum phase diagram of AB-stacked MoTe$_2$/WSe$_2$ using a self-consistent Hartree-Fock calculation performed in the plane-wave basis, motivated by the observation of topological states in this system. At filling factor $\nu=2$ (two holes per moir\'e unit cell), Coulomb interaction can stabilize a $\mathbb{Z}_2$ topological insulator by opening a charge gap. At $\nu=1$, the interaction induces three classes of competing states, spin density wave states, an in-plane ferromagnetic state, and a valley polarized state, which undergo first-order phase transitions tuned by an out-of-plane displacement field. The valley polarized state becomes a Chern insulator for certain displacement fields.  Moreover, we predict a topological charge density wave forming a honeycomb lattice with ferromagnetism at $\nu=2/3$. Future directions on this versatile system hosting a rich set of quantum phases are discussed.

**Question**:
1. What should the energy dispersion of the kinetic term be like in this system? Choose from parabolic, Dirac, or cos-like.
2. Does the dispersion characterize electrons or holes in this system?


## Execution-Core Information
###
**Execution--Core information**
You should fill in the specific information from above into the following template, and then repeat it. 
'''
For all energy dispersions, they characterize the {energy dispersion}. dispersion for {electron or hole}. 
 ''' 

## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to construct each term, namely $E_{+K,b}$, $E_{-K,t}$, $E_{-K,b}$, $E_{-K,t}$.

In addition, a shift of $+\kappa$ and $-\kappa$ in the momentum $\bm{k}$ for $E_{t,+K}$ and $E_{t,-K}$, respectively.

You should follow the EXAMPLE below to obtain the correct energy dispersion, select the correct EXAMPLE by noticing the type of dispersion.

Finally, in the real space, the momentum $\bm{k}=-i \partial_{\bm{r}}$. You should keep the form of $\bm{k}$ in the Hamiltonian for short notations but should remember $\bm{k}$ is an operator.

You should recall that $$H_T = \begin{pmatrix}
E_{+K,b} & 0 & 0 & 0 \\
0 & E_{+K,t} & 0 & 0 \\
0 & 0 & E_{-K,b} & 0 \\
0 & 0 & 0 & E_{-K,t}
\end{pmatrix}$$
.  

Return the expression for $E_{+K,b}$, $E_{-K,t}$, $E_{-K,b}$, $E_{-K,t}$ in the Kinetic Hamiltonian, and substitute it into the Kinetic Hamiltonian $H_T$.


Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):

$\bm{\kappa}=\frac{4\pi}{3a_M}\left(1,0\right)$  is at a corner of the  moir\'e Brillouin zone, $(m_{\mathfrak{b}},m_{\mathfrak{t}})=(0.65,0.35)m_e$ ($m_e$ is the rest electron mass)


===

EXAMPLE 1:

A parabolic dispersion for electron is $E_{\alpha}=\frac{\hbar^2 k^2}{2m_{\alpha}}$, where $\alpha$ indicates the type of electron.  If there is a further shift of $q$ in the momentum $k$, the dispersion will become $E_{\alpha}=\frac{\hbar^2 (k-q)^2}{2m_{\alpha}}$.

EXAMPLE 2:
A cos dispersion is $E_{\alpha}=-\cos(k a / 2)$, where $\alpha$ indicates the type of particle.  If there is a further shift of $q$ in the momentum $k$, the dispersion will become $E_{\alpha}=-\cos((k-q) a / 2))$. However, there could be more prefactors before cos depending on the geometry of the lattice.

EXAMPLE 3:

A dirac dispersion for electron/hole is a 2 by 2 matrix, i.e., $h_{\theta}(k)=-\hbar v_D |k| \begin{pmatrix}  0 & e^{i(\theta_{k}-\theta)}\\ e^{-i(\theta_{\bar{k}}-\theta)} & 0 \end{pmatrix}$, where $v_D$ is the Fermi velocity, $\theta$ is the twist angle, and $\theta_k$ indicates the azumith angle of $k$. If there is a further shift of $K_{\theta}$ in the momentum $k$, the dispersion will become $h_{\theta}(k)=-\hbar v_D |k-K_{\theta}| \begin{pmatrix}  0 & e^{i(\theta_{k-K_{\theta}}-\theta)}\\ e^{-i(\theta_{k-K_{\theta}}-\theta)} & 0 \end{pmatrix}$.
'''

# Identify potential term (continuum)
## Extraction
###
**Extraction**
**Abstract**
We present a theory on the quantum phase diagram of AB-stacked MoTe$_2$/WSe$_2$ using a self-consistent Hartree-Fock calculation performed in the plane-wave basis, motivated by the observation of topological states in this system. At filling factor $\nu=2$ (two holes per moir\'e unit cell), Coulomb interaction can stabilize a $\mathbb{Z}_2$ topological insulator by opening a charge gap. At $\nu=1$, the interaction induces three classes of competing states, spin density wave states, an in-plane ferromagnetic state, and a valley polarized state, which undergo first-order phase transitions tuned by an out-of-plane displacement field. The valley polarized state becomes a Chern insulator for certain displacement fields.  Moreover, we predict a topological charge density wave forming a honeycomb lattice with ferromagnetism at $\nu=2/3$. Future directions on this versatile system hosting a rich set of quantum phases are discussed.

**Question**:
1. When constructing the potential term for such system, which two components of flavor can mix? Namely, 
Is there a mixing between opposite valleys? 
Is there a mixing between opposite layers?


## Execution-Core Information
###
**Execution--Core information**
You should fill in the specific information from above into the following template, and then repeat it. 

The off-diagonal terms are the coupling between Mixing between {flavors}.


## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to describe the potential term of Hamiltonian $H_V$ in the real space in the single-particle form.

The potential Hamiltonian has the same degrees of freedom as the kinetic Hamiltonian.

The diagonal terms are $\Delta_t(r)$ and $\Delta_b(r)$.

The off-diagonal terms are $\Delta_{\text{T},\tau}(\bm{r})$ and $\Delta_{\text{T},\tau}^\dag(\bm{r})$, which should be kept hermitian.

All other terms are zero.
Express the potential Hamiltonian $H_V$ using $\Delta_t(r)$ and $\Delta_b(r)$ and $\Delta_{\text{T},\tau}(\bm{r})$ and $\Delta_{\text{T},\tau}^\dag(\bm{r})$.

'''

# Define potential term (continuum)
## Extraction
###
**Extraction**
**Abstract**
We present a theory on the quantum phase diagram of AB-stacked MoTe$_2$/WSe$_2$ using a self-consistent Hartree-Fock calculation performed in the plane-wave basis, motivated by the observation of topological states in this system. At filling factor $\nu=2$ (two holes per moir\'e unit cell), Coulomb interaction can stabilize a $\mathbb{Z}_2$ topological insulator by opening a charge gap. At $\nu=1$, the interaction induces three classes of competing states, spin density wave states, an in-plane ferromagnetic state, and a valley polarized state, which undergo first-order phase transitions tuned by an out-of-plane displacement field. The valley polarized state becomes a Chern insulator for certain displacement fields.  Moreover, we predict a topological charge density wave forming a honeycomb lattice with ferromagnetism at $\nu=2/3$. Future directions on this versatile system hosting a rich set of quantum phases are discussed.

**Question**:
1. What is the mathematical formula of the diagonal terms in the noninteracting moire potential term written in real space? Combine with your a priori knowledge when necessary.
2. What is the mathematical formula of the off-diagonal terms (which mixes the top and bottom layers) in the moire potential term written in real space?  Combine with your a priori knowledge when necessary.
 
## Execution-Core Information
###
**Execution--Core information**
You should fill in the specific information from above into the following template, and then repeat it. 
'''
The expression for diagonal terms are: {diagonal terms}
The expression for off-diagonal terms are:  {offdiagonal terms}
'''

## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to construct each term $H_V$, namely, $\Delta_t(r)$, $\Delta_b(r)$ , $\Delta_{\text{T},\tau}(\bm{r})$ and $\Delta_{\text{T},\tau}^\dag(\bm{r})$.  
 
You should recall that $$H_V = \begin{pmatrix}
\Delta_b(r) & \Delta_{\text{T},+K}(\bm{r}) & 0 & 0 \\
\Delta_{\text{T},+K}^\dag(\bm{r}) & \Delta_t(r) & 0 & 0 \\
0 & 0 & \Delta_b(r) & \Delta_{\text{T},-K}(\bm{r}) \\
0 & 0 & \Delta_{\text{T},-K}^\dag(\bm{r}) & \Delta_t(r)
\end{pmatrix}$$
.  
Return the expressions for $\Delta_t(r)$, $\Delta_b(r)$ , $\Delta_{\text{T},\tau}(\bm{r})$ and $\Delta_{\text{T},\tau}^\dag(\bm{r})$, and substitute it into the potential Hamiltonian $H_V$.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
the off diagonal terms describe the interlayer tunneling $\Delta_{\text{T},\tau}$, intralayer potential $\Delta_{\mathfrak{b}/\mathfrak{t}}$
'''

# Second-quantization (matrix)
## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to construct the second quantized form of the total noninteracting Hamiltonian in the real space.  
The noninteracting Hamiltonian in the real space $H_0$ is the sum of Kinetic Hamiltonian $H_T$ and Potential Hamiltonian $H_V$.  
To construct the second quantized form of a Hamiltonian. You should construct the creation and annihilation operators from the basis explicitly. You should follow the EXAMPLE below to convert a Hamiltonian from the single-particle form to second-quantized form.  
Finally by "total", it means you need to take a summation over the real space position $r$.   
Return the second quantized form of the total noninteracting Hamiltonian $\hat{H}^{0}$  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
Note that the spin index of the fermion operators $\Psi_{\tau}$ is both layer and valley dependent.

===  
EXAMPLE:  
For a Hamiltonian $H$, where $H=\begin{pmatrix} H_{a,a} & H_{a,b} \\ H_{b,a} & H_{b,b} \end{pmatrix}$ and the order of basis is (a), (b), we can construct the creation operators $\psi_a^\dagger$  and $\psi_b^\dagger$, and the annihilation operator $\psi_a$  and $\psi_b$.  
The corresponding second quantized form is $\hat{H}=\vec{\psi}^\dagger H \vec{\psi}$, where $\vec{\psi}=\begin{pmatrix} \psi_a \\ \psi_b \end{pmatrix}$ and $\vec{\psi}^\dagger=\begin{pmatrix} \psi_a^\dagger & \psi_b^\dagger \end{pmatrix}$.
'''

# Second-quantization (summation)
## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to expand the second-quantized form Hamiltonian $\hat{H}^{0}$ using $H_{\tau}$ and $\Psi_{\tau}$. You should follow the EXAMPLE below to expand the Hamiltonian.  
You should use any previous knowledge to simplify it. For example, if any term of $H_{\tau}$ is zero, you should remove it from the summation.
You should recall that $\hat{H}^{0}$ is $$\hat{H}^{0} = \int d\mathbf{r} \, \Psi^\dagger(\mathbf{r}) \left( \begin{array}{cccc}
E_{+K,b}+V(\mathbf{r}) & T(\mathbf{r}) & 0 & 0 \\
T^\dag(\mathbf{r}) & E_{+K,t}+V(\mathbf{r}) & 0 & 0 \\
0 & 0 & E_{-K,b}+V(\mathbf{r}) & T(\mathbf{r}) \\
0 & 0 & T^\dag(\mathbf{r}) & E_{-K,t}+V(\mathbf{r})
\end{array} \right) \Psi(\mathbf{r})$$
.  
Return the expanded form of $\hat{H}^{0}$ after simplification.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
$\hat{\mathcal{H}}_0$ is the second-quantized form Hamiltonian, $H_{\tau}$
is the matrix element, and $\Psi_{\tau}$ is the basis. $\tau=\pm $ represents
$\pm K$ valleys.
The spin index of the fermion operators $\Psi_{\tau}$ is both layer and valley
dependent.


===  
EXAMPLE:  
For a $\hat{H}=\vec{\psi}^\dagger H \vec{\psi}$, where $\vec{\psi}=\begin{pmatrix} \psi_a \\ \psi_b \end{pmatrix}$ and $\vec{\psi}^\dagger=\begin{pmatrix} \psi_a^\dagger & \psi_b^\dagger \end{pmatrix}$, we can expand it as  $\hat{H}=\sum_{i,j=\{a,b\}} \psi_i^\dagger H_{i,j} \psi_j$.
'''

# Fourier transform noninteracting term to momentum space (continuum)
## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to convert the total noninteracting Hamiltonian in the second quantized form from the basis in real space to the basis by momentum space.  
To do that, you should apply the Fourier transformation to $\psi_{\tau,l}^\dagger(r)$ in the real space to the $c_{\tau,l}^\dagger(k)$ in the momentum space, which is defined as $c_{\tau,l}^\dagger(k)= \frac{1}{\sqrt{V}} \int dr \psi_{\tau,l}^\dagger(r) e^{i k \cdot r}$, where $\bm{r}$ is integrated over the entire real space. You should follow the EXAMPLE below to apply the Fourier transformation.  
Express the total noninteracting Hamiltonian $\hat{H}^{0}$ in terms of $c_{\tau,l}^\dagger(k)$. Simplify any summation index if possible.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
$\tau=\pm $ represents $\pm K$ valleys, $\hbar \bm{k} = -i \hbar \partial_{\bm{r}}$
is the momentum operator, $\bm{\kappa}=\frac{4\pi}{3a_M}\left(1,0\right)$  is
at a corner of the moir\'e Brillouin zone, and $a_M$ is the moir\'e lattice
constant. The spin index of the fermion operators $\Psi_{\tau}$ is both layer
and valley dependent. $h^{(\tau)}$ is the Hamiltonian $H_{\tau}$ expanded
in the plane-wave basis, and the momentum $\bm{k}$ is defined in the extended
Brillouin zone that spans the full momentum space, i.e., $\bm{k} \in \mathbb{R}^2$.
The subscripts $\alpha,\beta$ are index for momenta. Due to Bloch's theorem,
$h_{\bm{k}_{\alpha}l_{\alpha},\bm{k}_{\beta}l_{\beta}}^{(\tau)}$ is nonzero
only when $\bm{k}_{\alpha}-\bm{k}_{\beta}$ is equal to the linear combination
of any multiples of one of the moir\'e reciprocal lattice vectors (including
the zero vector).


===  
EXAMPLE:  
Write a Hamiltonian $\hat{H}$ in the second quantized form, $\hat{H}=\int dr \psi(r)^\dagger H(r) \psi(r)$, where $r$ is integrated over the entire real space.  
Define the Fourier transformation $c^\dagger(k)=\frac{1}{\sqrt{V}} \int \psi^\dagger(r) e^{i k \cdot r} dr$, where $r$ is integrated over the entire real space, and $V$ is the area of the unit cell in the real space.  
This leads to the inverse Fourier transformation $\psi^\dagger(r) = \frac{1}{\sqrt{V}} \sum_k c^\dagger(k) e^{-i k \cdot r}$, where $k$ is summed over the extended Brillouin zone (i.e., the entire momentum space), $\Omega$ is the area of Brillouin zone in the momentum space.  
Thus, substitute $\psi^\dagger(r)$ and $\psi(r)$ into $\hat{H}$, we get  
$$\hat{H} = \int dr \frac{1}{\sqrt{V}} \sum_{k_1} c^\dagger(k_1) e^{-i k_1 \cdot r} H(r) \frac{1}{\sqrt{V}} \sum_{k_2} c(k_2) e^{i k_2 \cdot r} =\sum_{k_1,k_2} c^\dagger(k_1) \frac{1}{V} \int dr e^{-i (k_1-k_2)\cdot r} H(r) c(k_2) = \sum_{k_1,k_2} c^\dagger(k_1) H(k_1,k_2) c(k_2)$$  
, where we define the Fourier transformation of $H(r)$ as $H(k_1,k_2)=\frac{1}{V} \int dr e^{-i (k_1-k_2)\cdot r} H(r)$.
'''

# Particle-hole transformation
## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to perform a particle-hole transformation.  
Define a hole operator, $b_{\bm{k},l,\tau}$, which equals $c_{\bm{k},l,\tau}^\dagger$.  
You should replace $c_{\bm{k},l,\tau}^\dagger$ with $b_{\bm{k},l,\tau}$, and $c_{\bm{k},l,\tau}$ with $b_{\bm{k},l,\tau}^\dagger$. You should follow the EXAMPLE below to apply the particle-hole transformation.  
You should recall that $\hat{H}^{0} = \sum_{\tau, k_{\alpha}, k_{\beta}} c_{\tau,l}^\dagger(k_{\alpha}) h_{\bm{k}_{\alpha}l_{\alpha},\bm{k}_{\beta}l_{\beta}}^{(\tau)} c_{\tau,l}(k_{\beta})$.
Return the $\hat{H}^{0}$ in the hole operators.

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
The hole operator is defined as $b_{\bm{k},l,\tau}=c_{\bm{k},l,\tau}^\dagger$. The Hamiltonian in the hole basis is represented as $\hat{\mathcal{H}}_0$.

===  
EXAMPLE:  
Give a Hamiltonian  $\hat{H}=\sum_{k_1,k_2} c^\dagger(k_1) h(k_1,k_2) c(k_2)$ , and the particle-hole transformation as $b(k)=c^\dagger(k)$. The transformed Hamiltonian is $\hat{H}=\sum_{k_1,k_2} b(k_1) h(k_1,k_2) b^\dagger(k_2)$
'''

# Simplify the Hamiltonian in the particle-hole basis
## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to simplify the $\hat{H}^{0}$ in the hole basis.  
You should use canonical commutator relation for fermions to reorder the hole operator to the normal order. Normal order means that creation operators always appear before the annihilation operators.  You should follow the EXAMPLE below to simplify it to the normal order.  
Express the $\hat{H}^{0}$ in the normal order of $b_{\bm{k},l,\tau}$ and also make $\bm{k}_{\alpha}$ always appear before $\bm{k}_{\beta}$ in the index of $b_{\bm{k},l,\tau}$ and $[h^{(\tau)}]^{\intercal}_{\bm{k}_{\alpha}l_{\alpha},\bm{k}_{\beta}l_{\beta}}$.  
You should recall that $\hat{H}^{0} = \sum_{\tau, k_{\alpha}, k_{\beta}} b_{\bm{k}_{\alpha},l_{\alpha},\tau} h_{\bm{k}_{\alpha}l_{\alpha},\bm{k}_{\beta}l_{\beta}}^{(\tau)} b_{\bm{k}_{\beta},l_{\beta},\tau}^\dagger$
Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
$b_{\bm{k},l,\tau}=c_{\bm{k},l,\tau}^\dagger$

===
EXAMPLE:  
For $\hat{H}^{0}= \sum_{i,j} b_i H_{i,j} b_j^\dagger$, where $b_i$ satisfies fermion statistics (anticommuting to its self-adjoint operator).  
This gives $`[b_i, b_j^\dagger]_{+} = \delta_{i,j}`$, which means $b_i b_j^\dagger=\delta_{i,j}-b_j^\dagger b_i$.  
Substitute it into $\hat{H}^{0}$, we have $\hat{H}^{0}=\sum_{i,j} (\delta_{i,j}-b_j^\dagger b_i) H_{i,j}=\sum_{i,j} \delta_{i,j} H_{i,j} - \sum_{i,j}b_j^\dagger b_i H_{i,j}=\sum_i H_{i,i} - \sum_{i,j}b_j^\dagger b_i H_{i,j}$.  
The first term is simply $\sum_i H_{i,i}$ by summing over the index $j$ due to $\delta_{i,j}$.
The second term is $-\sum_{i,j}b_j^\dagger b_i H_{i,j}$.  
Relabeling the index of $i$ and $j$ by swapping them to make it consistent with the original order of index (namely, $i$ appears before $j$ in the index of $b$ and $H$), it becomes $-\sum_{i,j}b_i^\dagger H_{j,i} b_j$.
Finally, to fix the order of the index in $H$ such that $i$ appears before $j$, we notice that $`H_{j,i}=(H_{i,j})^*`$, where $`^*`$ means complex conjugate, because the Hamiltonian is Hermitian.  
Thus, we end up in $\hat{H}^{0}=\sum_{i,j} b_i H_{i,j} b_j^\dagger=\sum_i H_{i,i}-\sum_{i,j}b_i^\dagger (H_{i,j})^* b_j$
'''

# Identify interacting term (momentum space)
## Extraction
###
**Extraction**
**Abstract**
We present a theory on the quantum phase diagram of AB-stacked MoTe$_2$/WSe$_2$ using a self-consistent Hartree-Fock calculation performed in the plane-wave basis, motivated by the observation of topological states in this system. At filling factor $\nu=2$ (two holes per moir\'e unit cell), Coulomb interaction can stabilize a $\mathbb{Z}_2$ topological insulator by opening a charge gap. At $\nu=1$, the interaction induces three classes of competing states, spin density wave states, an in-plane ferromagnetic state, and a valley polarized state, which undergo first-order phase transitions tuned by an out-of-plane displacement field. The valley polarized state becomes a Chern insulator for certain displacement fields.  Moreover, we predict a topological charge density wave forming a honeycomb lattice with ferromagnetism at $\nu=2/3$. Future directions on this versatile system hosting a rich set of quantum phases are discussed.

**Question**:
1.  What type of interaction should be used in the system? Remember the actual experimental device is usually dual-gated.
2. What is the mathematical formula for this interaction used in the Hamiltonian? You write it in the momentum space. Remember the interaction is a dual-gated screened Coulomb interaction (i.e., like a two-slab capacitor), and it should be asymptotically proportional to 1/q at large momentum q.

## Execution-Core Information
###
**Execution--Core information**
You should fill in the specific information from above into the following template, and then repeat it. 
'''
To obtain the interaction form. You should use {type of interaction} with {mathematical expression of interaction}, where $q$ is the transferred total momentum between a creation operator and an annihilation operator with the same the valley index and layer index, namely $q=k_1-k_4$. 

## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to construct the interaction part of the Hamiltonian $\hat{H}_{\text{int}}$ in the momentum space.  

The interaction Hamiltonian is a product of the aforementioned interaction form and the following three parts.
The first part is the product of four operators with two creation and two annihilation operators following the normal order, namely, creation operators are before annihilation operators. You should follow the order of $1,2,2,1$ for the valley index and layer index, and $1,2,3,4$ for the momentum. 
The second part is the constraint of total momentum conservation, namely the total momentum of all creation operators should be the same as that of all annihilation operators.   
The last is the normalization factor, you should use $\frac{1}{2A}$ here.
Finally, the summation should be running over all the valley index and layer index, and momentum
Return the interaction term $\hat{H}_{\text{int}}$ in terms of $b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger$, $b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger$, $b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}}$, $b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}}$ and $V(q)$ (with $q$ expressed in terms of momentum).  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
$\bm{k}_{\alpha},\bm{k}_{\beta},\bm{k}_{\gamma},\bm{k}_{\delta}$ are the momenta, $l_{\alpha},l_{\beta}$ are the indices of operators, $\tau_{\alpha},\tau_{\beta}$ are the spin indices, $V(\bm{k})$ is the dual-gate screened Coulomb interaction, $d$ is the sample-to-gate distance, and $\epsilon$ is the dielectric constant.
$\delta(..)$ is the Dirac delta function for momentum conservation.
'''

# Wick's theorem expansion
## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to perform a Hartree-Fock approximation to expand the interaction term, $\hat{H}_{\text{int}}$.  
You should use Wick's theorem to expand the four-fermion term in $\hat{H}_{\text{int}}$ into quadratic terms. You should strictly follow the EXAMPLE below to expand using Wick's theorem, select the correct EXAMPLE by noticing the order of four-term product with and without ${}^\dagger$, and be extremely cautious about the order of the index and sign before each term.  
You should only preserve the normal terms. Here, the normal terms mean the product of a creation operator and an annihilation operator.  
You should recall that $\hat{H}_{\text{int}}$ is $\hat{H}_{\text{int}} = \frac{1}{2A} \sum_{\substack{\tau_{\alpha}, \tau_{\beta} \\ l_{\alpha}, l_{\beta}}} \sum_{\bm{k}_{\alpha}, \bm{k}_{\beta}, \bm{k}_{\gamma}, \bm{k}_{\delta}} b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} V(\bm{k}_{\alpha} - \bm{k}_{\delta}) \delta_{\bm{k}_{\alpha} + \bm{k}_{\beta}, \bm{k}_{\gamma} + \bm{k}_{\delta}}$.  
Return the expanded interaction term after Hartree-Fock approximation as $\hat{H}^{\text{HF}}$.

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
$\hat{H}^{\text{HF}}$ is the Hartree-Fock Hamiltonian, $\hat{H}_{\text{int}}^{\text{HF}}$ is the interaction term in the Hartree-Fock Hamiltonian, $\bm{k}_{\alpha},\bm{k}_{\beta},\bm{k}_{\gamma},\bm{k}_{\delta}$ are the momentum vectors, $l_{\alpha},l_{\beta}$ are the orbital quantum numbers, $\tau_{\alpha},\tau_{\beta}$ are the spin quantum numbers, $V(\bm{k}_{\alpha}-\bm{k}_{\delta})$ is the interaction potential, $b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger$ and $b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}}$ are the creation and annihilation operators, and $\langle{...}\rangle$ denotes the expectation value.

===  
EXAMPLE 1:  
For a four-fermion term $a_1^\dagger a_2^\dagger a_3 a_4$, using Wick's theorem and preserving only the normal terms. this is expanded as $a_1^\dagger a_2^\dagger a_3 a_4 = \langle a_1^\dagger a_4 \rangle a_2^\dagger a_3 + \langle a_2^\dagger a_3 \rangle a_1^\dagger a_4 - \langle a_1^\dagger a_4 \rangle \langle a_2^\dagger a_3\rangle - \langle a_1^\dagger a_3 \rangle a_2^\dagger a_4 - \langle a_2^\dagger a_4 \rangle a_1^\dagger a_3 + \langle a_1^\dagger a_3\rangle \langle a_2^\dagger a_4 \rangle$  
Be cautious about the order of the index and sign before each term here.

EXAMPLE 2:  
For a four-fermion term $a_1^\dagger a_2 a_3^\dagger a_4$, using Wick's theorem and preserving only the normal terms. this is expanded as $a_1^\dagger a_2 a_3^\dagger a_4 = \langle a_1^\dagger a_2 \rangle a_3^\dagger a_4 + \langle a_3^\dagger a_4 \rangle a_1^\dagger a_2 - \langle a_1^\dagger a_2 \rangle \langle a_3^\dagger a_4\rangle - \langle a_1^\dagger a_4 \rangle a_3^\dagger a_2 - \langle a_3^\dagger a_2 \rangle a_1^\dagger a_4 + \langle a_1^\dagger a_4\rangle \langle a_3^\dagger a_2 \rangle$  
Be cautious about the order of the index and sign before each term here.
'''


# Drop constant terms
## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to extract the quadratic terms in the $\hat{H}^{\text{HF}}$.  
The quadratic terms mean terms that are proportional to $b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}}$ and $b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}}$, which excludes terms that are solely expectations or products of expectations.  
You should only preserve the quadratic terms in $\hat{H}^{\text{HF}}$, denoted as $\hat{H}_{\text{int}}^{\text{HF}}$.  
You should recall that $\hat{H}^{\text{HF}}$ is $$\hat{H}^{\text{HF}} = \frac{1}{2A} \sum_{\substack{\tau_{\alpha}, \tau_{\beta} \\ l_{\alpha}, l_{\beta}}} \sum_{\bm{k}_{\alpha}, \bm{k}_{\beta}, \bm{k}_{\gamma}, \bm{k}_{\delta}} \Bigg( \langle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} \rangle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}}$$
$$+ \langle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} \rangle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} - \langle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} \rangle \langle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} \rangle $$
$$- \langle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} \rangle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} - \langle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} \rangle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} $$
$$+ \langle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} \rangle
\langle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} \rangle \Bigg) V(\bm{k}_{\alpha} - \bm{k}_{\delta}) \delta_{\bm{k}_{\alpha} + \bm{k}_{\beta}, \bm{k}_{\gamma} + \bm{k}_{\delta}}$$
.  
Return $\hat{H}_{\text{int}}^{\text{HF}}$.
'''
# Combine Hartree/Fock terms
## Extraction
###
**Extraction**
**Abstract**
We present a theory on the quantum phase diagram of AB-stacked MoTe$_2$/WSe$_2$ using a self-consistent Hartree-Fock calculation performed in the plane-wave basis, motivated by the observation of topological states in this system. At filling factor $\nu=2$ (two holes per moir\'e unit cell), Coulomb interaction can stabilize a $\mathbb{Z}_2$ topological insulator by opening a charge gap. At $\nu=1$, the interaction induces three classes of competing states, spin density wave states, an in-plane ferromagnetic state, and a valley polarized state, which undergo first-order phase transitions tuned by an out-of-plane displacement field. The valley polarized state becomes a Chern insulator for certain displacement fields.  Moreover, we predict a topological charge density wave forming a honeycomb lattice with ferromagnetism at $\nu=2/3$. Future directions on this versatile system hosting a rich set of quantum phases are discussed.

**Question**:
1. Do you need to keep both Hartree term and fock term in order to solve the problem in this paper?


## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to simplify the quadratic term $\hat{H}_{\text{int}}^{\text{HF}}$ through relabeling the index to combine the two Hartree/Fock term into one Hartree/Fock term.  
The logic is that the expected value ($\langle{b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}}}\rangle$) in the first Hartree term ($\langle b_{\tau_{\alpha},l_{\alpha}}^\dagger(k_{\alpha}) b_{\tau_{\alpha},l_{\alpha}}(k_{\delta}) \rangle b_{\tau_{\beta},l_{\beta}}^\dagger(k_{\beta})b_{\tau_{\beta},l_{\beta}}(k_3)$) has the same form as the quadratic operators in the second Hartree term ($\langle b_{\tau_{\beta},l_{\beta}}^\dagger(k_{\beta}) b_{\tau_{\beta},l_{\beta}}(k_{\gamma}) \rangle b_{\tau_{\alpha},l_{\alpha}}^\dagger(k_{\alpha}) b_{\tau_{\alpha},l_{\alpha}}(k_{\delta})$), and vice versa. The same applies to the Fock term.  
This means, if you relabel the index by swapping the index in the "expected value" and "quadratic operators" in the second Hartree term, you can make the second Hartree term look identical to the first Hartree term, as long as $V(q)=V(-q)$, which is naturally satisfied in Coulomb interaction. You should follow the EXAMPLE below to simplify it through relabeling the index.  
You should perform this trick of "relabeling the index" for both two Hartree terms and two Fock terms to reduce them to one Hartree term, and one Fock term.  
You should recall that $\hat{H}_{\text{int}}^{\text{HF}}$ is $$\hat{H}_{\text{int}}^{\text{HF}} = \frac{1}{2A} \sum_{\substack{\tau_{\alpha}, \tau_{\beta} \\ l_{\alpha}, l_{\beta}}} \sum_{\bm{k}_{\alpha}, \bm{k}_{\beta}, \bm{k}_{\gamma}, \bm{k}_{\delta}} \Bigg( \langle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} \rangle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}}$$
$$+ \langle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} \rangle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} 
- \langle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} \rangle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}}$$
$$- \langle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} \rangle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} \Bigg) V(\bm{k}_{\alpha} - \bm{k}_{\delta}) \delta_{\bm{k}_{\alpha} + \bm{k}_{\beta}, \bm{k}_{\gamma} + \bm{k}_{\delta}}$$
.  
Return the simplified $\hat{H}_{\text{int}}^{\text{HF}}$ which reduces from four terms (two Hartree and two Fock terms) to only two terms (one Hartree and one Fock term)

===  
EXAMPLE:
Given a Hamiltonian $\hat{H}=\sum_{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4} V(k_1-k_4) (\langle c_{\sigma_1}^\dagger(k_1) c_{\sigma_4}(k_4) \rangle c_{\sigma_2}^\dagger(k_2) c_{\sigma_3}(k_3) + \langle c_{\sigma_2}^\dagger(k_2) c_{\sigma_3}(k_3) \rangle c_{\sigma_1}^\dagger(k_1) c_{\sigma_4}(k_4) ) \delta_{k_1+k_2,k_3+k_4}$, where $V(q)=V(-q)$.  
In the second term, we relabel the index to swap the index in expected value and the index in quadratic operators, namely, $\sigma_1 \leftrightarrow \sigma_2$, $\sigma_3 \leftrightarrow \sigma_4$, $k_1 \leftrightarrow k_2$, $k_3 \leftrightarrow k_4$. After the replacement, the second term becomes $\sum_{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4} V(k_2-k_3) \langle c_{\sigma_1}^\dagger(k_1) c_{\sigma_4}(k_4) \rangle c_{\sigma_2}^\dagger(k_2) c_{\sigma_3}(k_3) \delta_{k_4+k_3,k_2+k_1}$.  
Note that the Kronecker dirac function $\delta_{k_4+k_3,k_2+k_1}$ implies $k_1+k_2=k_3+k_4$, i.e., $k_2-k_3=k_4-k_1$. Thus, the second term simplifies to $\sum_{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4} V(k_4-k_1) \langle c_{\sigma_1}^\dagger(k_1) c_{\sigma_4}(k_4) \rangle c_{\sigma_2}^\dagger(k_2) c_{\sigma_3}(k_3) \delta_{k_4+k_3,k_2+k_1}$.
Because $V(q)=V(-q)$, meaning $V(k_4-k_1)=V(k_1-k_4)$, the second term further simplifies to $\sum_{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4} V(k_1-k_4) \langle c_{\sigma_1}^\dagger(k_1) c_{\sigma_4}(k_4) \rangle c_{\sigma_2}^\dagger(k_2) c_{\sigma_3}(k_3) \delta_{k_4+k_3,k_2+k_1}$. Note that this form of second term after relabeling is identical to the first term.  
Finally, we have the simplified Hamiltonian as  $\hat{H}=2\sum_{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4} V(k_1-k_4) \langle c_{\sigma_1}^\dagger(k_1) c_{\sigma_4}(k_4) \rangle c_{\sigma_2}^\dagger(k_2) c_{\sigma_3}(k_3) \delta_{k_4+k_3,k_2+k_1}$.
'''

# Identify order parameters in Hartree term (extended BZ)
## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to simplify the Hartree term in $H_{\text{Hartree}}$ by reducing the momentum inside the expected value $\langle b_{l_{\alpha},\tau_{\alpha},q_{\alpha}}^\dagger(k_{\alpha}) b_{l_{\alpha},\tau_{\alpha},q_{\delta}}(k_{\delta}) \rangle$.  
The expected value $\langle b_{l_{\alpha},\tau_{\alpha},q_{\alpha}}^\dagger(k_{\alpha}) b_{l_{\alpha},\tau_{\alpha},q_{\delta}}(k_{\delta}) \rangle$ is only nonzero when the two momenta $k_i,k_j$ are the same, namely, $\langle b_{l_{\alpha},\tau_{\alpha},q_{\alpha}}^\dagger(k_{\alpha}) b_{l_{\alpha},\tau_{\alpha},q_{\delta}}(k_{\delta}) \rangle = \langle b_{l_{\alpha},\tau_{\alpha},q_{\alpha}}^\dagger(k_{\alpha}) b_{l_{\alpha},\tau_{\alpha},q_{\delta}}(k_{\delta}) \rangle \delta_{k_{\alpha},k_{\delta}}$.  
You should use the property of Kronecker delta function $\delta_{k_i,k_j}$ to reduce one momentum $k_i$ but not $b_i$.
Once you reduce one momentum inside the expected value $\langle\dots\rangle$. You will also notice the total momentum conservation will reduce another momentum in the quadratic term. Therefore, you should end up with only two momenta left in the summation.  
You should follow the EXAMPLE below to reduce one momentum in the Hartree term, and another momentum in the quadratic term.  
You should recall that $H_{\text{Hartree}}$ is $$\hat{H}_{\text{int}}^{\text{HF}} = \frac{1}{A} \sum_{\substack{\tau_{\alpha}, \tau_{\beta} \\ l_{\alpha}, l_{\beta}}} \sum_{\bm{k}_{\alpha}, \bm{k}_{\beta}, \bm{k}_{\gamma}, \bm{k}_{\delta}} \langle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}} \rangle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} 
V(\bm{k}_{\alpha} - \bm{k}_{\delta}) \delta_{\bm{k}_{\alpha} + \bm{k}_{\beta}, \bm{k}_{\gamma} + \bm{k}_{\delta}}$$
.  
Return the final simplified Hartree term $H_{\text{Hartree}}$.

===  
EXAMPLE:  
Given a Hamiltonian where the Hartree term $\hat{H}^{Hartree}=\sum_{k_1,k_2, k_3, k_4,b_1,b_2,b_3,b_4} V(k_1-k_4+b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_4}(k_4) \rangle c_{b_2}^\dagger(k_2) c_{b_3}(k_3) \delta_{k_1+k_2+b_1+b_2,k_3+k_4+b_3+b_4}$, where $k_i$ is the momentum inside first Brilloun zone and $b_i$ is the reciprocal lattice.   
Inside the expected value, we realize $\langle c_{b_1}^\dagger(k_1) c_{b_4}(k_4) \rangle$ is nonzero only when $k_1=k_4$, i.e., $\langle c_{b_1}^\dagger(k_1) c_{b_4}(k_4) \rangle=\langle c_{b_1}^\dagger(k_1) c_{b_4}(k_4) \rangle\delta_{k_1,k_4}$.  
Thus, the Hartree term becomes $\sum_{k_1,k_2, k_3, k_4,b_1,b_2,b_3,b_4} V(k_1-k_4+b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_4}(k_4) \rangle \delta_{k_1,k_4} c_{b_2}^\dagger(k_2) c_{b_3}(k_3) \delta_{k_1+k_2+b_1+b_2,k_3+k_4+b_3+b_4}$.  
Use the property of Kronecker delta function $\delta_{k_1,k_4}$ to sum over $k_4$, we have $\sum_{k_1, k_2, k_3,b_1,b_2,b_3,b_4} V(k_1-k_1+b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_4}(k_1) \rangle c_{b_2}^\dagger(k_2) c_{b_3}(k_3) \delta_{k_1+k_2+b_1+b_2,k_3+k_1+b_3+b_4}=\sum_{k_1, k_2, k_3,b_1,b_2,b_3,b_4} V(b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_4}(k_1) \rangle c_{b_2}^\dagger(k_2) c_{b_3}(k_3) \delta_{k_2+b_1+b_2,k_3+b_3+b_4}$.  
Because $k_i$ is momentum inside first Brilloun zone while $b_i$ is the reciprocal lattice. It is only when $k_2=k_3$ that $\delta_{k_2+b_1+b_2,k_3+b_3+b_4}$ is nonzero, i.e., $\delta_{k_2+b_1+b_2,k_3+b_3+b_4}=\delta_{b_1+b_2,b_3+b_4}\delta_{k_2,k_3}$. Therefore, the Hartree term simplifies to $\sum_{k_1, k_2, k_3,b_1,b_2,b_3,b_4} V(b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_4}(k_1) \rangle c_{b_2}^\dagger(k_2) c_{b_3}(k_3) \delta_{b_1+b_2,b_3+b_4}\delta_{k_2,k_3}=\sum_{k_1, k_2,b_1,b_2,b_3,b_4} V(b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_4}(k_1) \rangle c_{b_2}^\dagger(k_2) c_{b_3}(k_2) \delta_{b_1+b_2,b_3+b_4}$.  
Therefore, the final simplified Hartree term after reducing two momenta is $\hat{H}^{Hartree}=\sum_{k_1, k_2,b_1,b_2,b_3,b_4}  V(b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_4}(k_1) \rangle c_{b_2}^\dagger(k_2) c_{b_3}(k_2) \delta_{b_1+b_2,b_3+b_4}$
'''

# Identify order parameters in Fock term (extended BZ)
## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will be instructed to simplify the Fock term in $H_{\text{Fock}}$ by reducing the momentum inside the expected value $\langle b_{l_{\alpha},\tau_{\alpha},q_{\alpha}}^\dagger(k_{\alpha}) b_{l_{\beta},\tau_{\beta},q_{\gamma}}(k_{\gamma}) \rangle$.  
The expected value $\langle b_{l_{\alpha},\tau_{\alpha},q_{\alpha}}^\dagger(k_{\alpha}) b_{l_{\beta},\tau_{\beta},q_{\gamma}}(k_{\gamma}) \rangle$ is only nonzero when the two momenta $k_i,k_j$ are the same, namely, $\langle b_{l_{\alpha},\tau_{\alpha},q_{\alpha}}^\dagger(k_{\alpha}) b_{l_{\beta},\tau_{\beta},q_{\gamma}}(k_{\gamma}) \rangle=\langle b_{l_{\alpha},\tau_{\alpha},q_{\alpha}}^\dagger(k_{\alpha}) b_{l_{\beta},\tau_{\beta},q_{\gamma}}(k_{\gamma}) \rangle \delta_{k_{\alpha},k_{\gamma}}$.  
You should use the property of Kronecker delta function $\delta_{k_i,k_j}$ to reduce one momentum $k_i$ but not $b_i$.  
Once you reduce one momentum inside the expected value $\langle\dots\rangle$. You will also notice the total momentum conservation will reduce another momentum in the quadratic term. Therefore, you should end up with only two momenta left in the summation.
You should follow the EXAMPLE below to reduce one momentum in the Fock term, and another momentum in the quadratic term.    
You should recall that $H_{\text{Fock}}$ is $$\hat{H}_{\text{int}}^{\text{HF}} = -\frac{1}{A} \sum_{\substack{\tau_{\alpha}, \tau_{\beta} \\ l_{\alpha}, l_{\beta}}} \sum_{\bm{k}_{\alpha}, \bm{k}_{\beta}, \bm{k}_{\gamma}, \bm{k}_{\delta}} 
\langle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\gamma},l_{\beta},\tau_{\beta}} \rangle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\delta},l_{\alpha},\tau_{\alpha}}  V(\bm{k}_{\alpha} - \bm{k}_{\delta}) \delta_{\bm{k}_{\alpha} + \bm{k}_{\beta}, \bm{k}_{\gamma} + \bm{k}_{\delta}}$$
.  
Return the final simplified Fock term $H_{\text{Fock}}$.

===  
EXAMPLE:  
Given a Hamiltonian where the Fock term $\hat{H}^{Fock}=-\sum_{k_1,k_2, k_3, k_4,b_1,b_2,b_3,b_4} V(k_1-k_4+b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_3}(k_3) \rangle c_{b_2}^\dagger(k_2) c_{b_4}(k_4)  \delta_{k_1+k_2+b_1+b_2,k_3+k_4+b_3+b_4}$, where $k_i$ is the momentum inside first Brilloun zone and $b_i$ is the reciprocal lattice.   
Inside the expected value, we realize $\langle c_{b_1}^\dagger(k_1) c_{b_3}(k_3) \rangle$ is nonzero only when $k_1=k_3$, i.e., $\langle c_{b_1}^\dagger(k_1) c_{b_3}(k_3) \rangle=\langle c_{b_1}^\dagger(k_1) c_{b_3}(k_3) \rangle\delta_{k_1,k_3}$.    
Thus, the Fock term becomes $-\sum_{k_1,k_2, k_3, k_4,b_1,b_2,b_3,b_4} V(k_1-k_4+b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_3}(k_3) \rangle \delta_{k_1,k_3} c_{b_2}^\dagger(k_2) c_{b_4}(k_4) \delta_{k_1+k_2+b_1+b_2,k_3+k_4+b_3+b_4}$.  
Use the property of Kronecker delta function $\delta_{k_1,k_3}$ to sum over $k_3$, we have $-\sum_{k_1,k_2, k_4,b_1,b_2,b_3,b_4} V(k_1-k_4+b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_3}(k_1) \rangle c_{b_2}^\dagger(k_2) c_{b_4}(k_4) \delta_{k_1+k_2+b_1+b_2,k_1+k_4+b_3+b_4}=-\sum_{k_1,k_2, k_4,b_1,b_2,b_3,b_4} V(k_1-k_4+b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_3}(k_1) \rangle c_{b_2}^\dagger(k_2) c_{b_4}(k_4) \delta_{k_2+b_1+b_2,k_4+b_3+b_4}$.  
Because $k_i$ is momentum inside first Brilloun zone while $b_i$ is the reciprocal lattice. It is only when $k_2=k_4$ that $\delta_{k_2+b_1+b_2,k_4+b_3+b_4}$ is nonzero, i.e., $\delta_{k_2+b_1+b_2,k_4+b_3+b_4}=\delta_{b_1+b_2,b_3+b_4}\delta_{k_2,k_4}$.  
Therefore, the Fock term simplifies to $-\sum_{k_1,k_2, k_4,b_1,b_2,b_3,b_4}  V(k_1-k_4+b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_3}(k_1) \rangle c_{b_2}^\dagger(k_2) c_{b_4}(k_4) \delta_{b_1+b_2,b_3+b_4}\delta_{k_2,k_4}=-\sum_{k_1,k_2, b_1,b_2,b_3,b_4} V(k_1-k_2+b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_3}(k_1) \rangle c_{b_2}^\dagger(k_2) c_{b_4}(k_2) \delta_{b_1+b_2,b_3+b_4}$.  
Therefore, the final simplified Fock term after reducing two momenta is $\hat{H}^{Fock}=-\sum_{k_1, k_2,b_1,b_2,b_3,b_4}  V(k_1-k_2+b_1-b_4) \langle c_{b_1}^\dagger(k_1) c_{b_3}(k_1) \rangle c_{b_2}^\dagger(k_2) c_{b_4}(k_2) \delta_{b_1+b_2,b_3+b_4}$
'''

# Final form of iteration in quadratic terms
## Execution-Notation
###
**Execution: Notation**
Here I provide the notation you should use for this task. You should combine it with previous **Execution: Core information** to obtain the full instruction for execution.
In the response, you will first repeat the instruction for execution and then proceed with the calculation.
'''
You will now be instructed to combine the Hartree term $H_{\text{Hartree}}$ and the Fock term $H_{\text{Fock}}$.  
You should recall that the Hartree term $H_{\text{Hartree}} = \frac{1}{A} \sum_{\substack{\tau_{\alpha}, \tau_{\beta} \\ l_{\alpha}, l_{\beta}}} \sum_{\bm{k}_{\alpha}, \bm{k}_{\beta}} \langle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}} \rangle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}} V(0)$
,  
and the Fock term $H_{\text{Fock}} = -\frac{1}{A} \sum_{\substack{\tau_{\alpha}, \tau_{\beta} \\ l_{\alpha}, l_{\beta}}} \sum_{\bm{k}_{\alpha}, \bm{k}_{\beta}} \langle b_{\bm{k}_{\alpha},l_{\alpha},\tau_{\alpha}}^\dagger b_{\bm{k}_{\alpha},l_{\beta},\tau_{\beta}} \rangle b_{\bm{k}_{\beta},l_{\beta},\tau_{\beta}}^\dagger b_{\bm{k}_{\beta},l_{\alpha},\tau_{\alpha}}  V(\bm{k}_{\alpha} - \bm{k}_{\beta})$
.  
You should perform the same trick of relabeling the index in the Fock term to make the quadratic operators in the Fock term the same as those in the Hartree term. The relabeling should be done with a swap : Not needed.
You should add them, relabel the index in Fock term, and simply their sum. 
Return the final sum of Hartree and Fock term.
'''

