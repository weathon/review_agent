# The Geometry Of Llm Quantization: Gptq As Babai'S Nearest Plane Algorithm

Jiale Chen1, Yalda Shabanzadeh1**, Elvir Crncevi** ˇ c´
2, Torsten Hoefler3**, Dan Alistarh**1,2 1Institute of Science and Technology Austria (ISTA), 2Red Hat, Inc., 3ETH Zürich jiale.chen@ist.ac.at

## Abstract

Quantizing the weights of large language models (LLMs) from 16-bit to lower bitwidth is the de facto approach to deploy massive transformers onto more affordable accelerators. While GPTQ emerged as one of the standard methods for one-shot post-training quantization at LLM scale, its inner workings are described as a sequence of algebraic updates that obscure geometric meaning or worst-case guarantees. In this work, we show that, when executed back-to-front (from the last to first dimension) for a linear layer, GPTQ is mathematically identical to Babai's nearest plane algorithm for the classical closest vector problem (CVP) on a lattice defined by the Hessian matrix of the layer's inputs. This equivalence is based on a sophisticated mathematical argument, and has two analytical consequences: first, the GPTQ error propagation step gains an intuitive geometric interpretation; second, GPTQ inherits the error upper bound of Babai's algorithm under the assumption that no weights are clipped. Leveraging this bound, we design post-training quantization methods that avoid clipping, and outperform the original GPTQ. In addition, we provide efficient GPU inference kernels for the resulting representation. Taken together, these results place GPTQ on a firm theoretical footing and open the door to importing decades of progress in lattice algorithms towards the design of future quantization algorithms for billion-parameter models. Source code is available at https://github.com/IST-DASLab/GPTQ-Babai.

## 1 Introduction

Generative pre-trained transformers (GPT) models contain hundreds of billions of parameters and have massive computational and memory costs (Luccioni et al., 2024). Post-training quantization (PTQ) has emerged as a practical solution for reducing their footprint (Gholami et al., 2021). Among a growing family of methods, GPTQ (Frantar et al., 2023) was the first to push one-shot quantization down to the 4-bit regime, while retaining near-baseline accuracies. GPTQ is still very popular nowadays and yields state-of-the-art results in some regimes (Kurtic et al., 2025).

Despite its empirical success, the GPTQ algorithm was only presented as a sequence of greedily applied algebraic operations: the procedure picks one weight at a time, quantizes it via rounding or clipping, and then optimally updates the not-yet-quantized weights to correct for the remaining per-layer loss; it then continues with the next weight, and so on. This procedure leaves an obvious open question: why does a local greedy rule work so well globally? Current literature does not answer this question, leaving little guidance for principled extensions or failure case analysis.

Our contribution. This paper is the first1to provide a geometric interpretation for GPTQ, which implies a layer-wise global error bound. Our main theoretical results (Section 4) are (i) the GPTQ optimization problem, i.e., linear-layer quantization with the L2 objective on the output, is equivalent to the closest vector problem (CVP) w.r.t. L2 distance; (ii) the GPTQ algorithm executed from the last to first dimension is the same as Babai's nearest plane algorithm on the basis of the factorized Hessian matrix, without LLL basis reduction, and this finding holds independently of whether large weights are clipped to the quantization grid (a process known as weight clipping); and (iii) the worst-case layer-wise error in the no-clipping setting is bound tightly by the trace of the diagonal matrix of the 1The concurrent work of Birnick (2025) appeared on arXiv slightly later than our preprint (Chen et al., 2025).

1 LDL decomposition of the Hessian matrix. In addition (Section 5), we tie our theoretical findings to practical quantization by introducing new no-clipping methods of better accuracy than the original GPTQ, together with efficient GPU inference kernels for the resulting representation.

## 2 Related Work

Second-order compression (pruning and quantization). The idea of using Hessian information to guide parameter removal dates back to Optimal Brain Damage (LeCun et al., 1989) and Optimal Brain Surgeon (OBS) (Hassibi et al., 1993). Optimal Brain Compression (OBC) (Frantar & Alistarh, 2022) generalizes OBS to the post-training setting and unifies structured pruning and quantization (also called Optimal Brain Quantizer, OBQ) under a single exact solver. GPTQ (Frantar et al., 2023) inherits OBQ's error propagation method but applies it in a fixed order, so that the inverse Hessian can be shared and only needs to be computed once. GPTQ only has cubic computational complexity in the column/row dimension, making it suitable for LLMs. QuIP (Chee et al., 2023) proves an error guarantee for GPTQ and proposes the LDLQ method as an equivalent variant of GPTQ. Lattices, CVP algorithms, and hardness. The closest vector problem (CVP) is NP-complete to approximate within any constant factor under polynomial-time reductions (van Emde Boas, 1981; Micciancio & Goldwasser, 2002; Dinur et al., 2003), motivating decades of approximation algorithms. Babai's nearest plane heuristic (Babai, 1986) delivers a solution in polynomial time and, when preceded by LLL basis reduction (Lenstra et al., 1982), enjoys a 2 O(n)approximation. BKZ basis reduction (Kannan, 1987) further tightens the constant in an exponential-time solver.

## 3 Preliminaries And Notations

We use Python-style indexing inside square brackets to select elements and sub-matrices from a tensor, e.g., [j, :] selects the j-th row vector, [:, j] selects the j-th column vector, and [j :, j] selects the sub-column consisting of rows after the j-th (inclusive) row in the j-th column, [:, J] selects the column vectors indexed by set J as a sub-matrix, etc2.

## 3.1 Linear-Layer Quantization Problem

Problem. Let X = [x1*, . . . ,* xn]
⊤∈ R
n×c be the sampled calibration input data of batch size n and input dimension c with xi ∈ R
cand n ≥ c = rank (X). Let W = [w1*, . . . ,* wr] ∈
R 
c×r be the linear layer weights of input dimension c and output dimension r with wi ∈ R
c.

Let S = [s1*, . . . ,* sr] ∈ R
c×r
̸=0 be the non-zero quantization scales with si ∈ R
c
̸=0. Here we consider a general case that applies to any grouping pattern: each weight element wi[j] has its own scaling factor si[j]. Assume S is statically computed using methods like AbsMax or MSE
before any weight updates. Let Z† ⊆ Z be the quantization grid (representable integers). In the clipping setting, e.g., for INT4 format, Z† = {−8, . . . , −1, 0, 1*, . . . ,* 7}. In the no-clipping setting, Z† = Z, which allows any integer as the quantization results. Let Z = [z1*, . . . ,* zr] ∈ Z†
c×r be the (unknown) quantized integers with zi ∈ Z
c
†. Denote Q = [q1*, . . . ,* qr] ∈ R
c×ras the dequantized weights with qi = diag (si) zi ∈ R
c. The goal is to minimize the L2 error on the layer output XW ∈ R
n×r: ∥XQ − XW∥
2 F 
=Pr i=1 
∥X diag (si) zi − Xwi∥
2, i.e, finding argminzi∈Z
c †
∥X diag (si) zi − Xwi∥
2for all 1 ≤ i ≤ r.

OBQ algorithm. Let set Jiinitialized to {1*, . . . , c*} be the set of not-yet-quantized indices of wi. We denote Ji as J as a short-hand notation. For each weight vector wi, OBQ chooses

$$j\leftarrow\operatorname{argmin}_{j\in J}{\frac{\left(\mathbf{q}_{i}[j]-\mathbf{w}_{i}[j]\right)^{2}}{\left(\mathbf{X}[:,J]^{\top}\mathbf{X}[:,J]\right)^{-1}[j,j]}}$$

as the next dimension to quantize. OBQ quantizes the chosen element wi[j] as qi[j] ← si[j] ·
ROUND 
wi[j]
si[j]
, Z†
via the ROUND (·,Z†) function which rounds the inputs to the nearest values

$$(1)$$

in Z†. OBQ then optimally updates the subset of weights wi[J] via an error propagation step wi[j
′] ← wi[j
′] + ∆wi[j
′] for all j
′ ∈ J with

$$\Delta\mathbf{w}_{i}[j^{\prime}]\leftarrow\frac{\left(\mathbf{X}[:,J]^{\top}\mathbf{X}[:,J]\right)^{-1}[j^{\prime},j]}{\left(\mathbf{X}[:,J]^{\top}\mathbf{X}[:,J]\right)^{-1}[j,j]}\left(\mathbf{q}_{i}[j]-\mathbf{w}_{i}[j]\right).$$
$$\left(2\right)$$
(qi[j] − wi[j]). (2)
OBQ continues iteration with J ← J \ {j} until J is empty. GPTQ algorithm. GPTQ reduces the computational complexity of OBQ by applying the OBQ quantization and error propagation steps in a fixed dimensional order, e.g., from the first to last dimension (j ← 1 to c), instead of dynamically determined orders (Eq. 1). The fixed order is independent of the output channel i, thus the Hessian information X[:, J]
⊤X[:, J]−1[:, j] can be shared across wi for all i, without recomputation. Furthermore, the Hessian information for all j can be precomputed at once using Cholesky or LDL decomposition of the Hessian matrix X⊤X.

Algorithm 1 is the pseudocode of GPTQ. The algorithm is identical to the original GPTQ paper (Frantar et al., 2023) except for missing the blocking mechanism that only affects the memory access pattern and computational speed, but not the numerical results. Additional notations are as follows.

P ∈ {0, 1}
c×cis a permutation matrix that modifies the dimensional order of GPTQ quantization.

The default order is front-to-back (from the first to last dimension), i.e., P = I. λ ∈ R+ is a small damping factor for computing the Hessian matrix, ensuring the matrix is of full rank. A typical choice is λ =1 100c Pcj=1 X⊤X[*j, j*] = 1 100c
∥X∥
2 F. Function LDL returns the lower triangular matrix in the LDL decomposition. Symbols ∗ and / denote element-wise multiplication and division, respectively.

## Algorithm 1: Gptq

Input: original weights W ∈ R
c×r, per-coordinate scales S ∈ R
c×r
̸=0 , calibration activation X ∈ R
n×c, permutation P ∈ {0, 1}
c×c, damping ratio λ > 0, integer grid Z† ⊆ Z
Output: quantized weights Z ∈ Z
c×r
†, dequantized weights Q ∈ R
c×r 1 H ← P
⊤X⊤X + λIP // dampen and reorder Hessian 2 L ← LDL H−1// factorize (take the L matrix from the LDL decomposition) the inversed Hessian as the shared coefficients for error propagation 3 W,S ← P
−1W, P
−1S // reorder weights and scales 4 Q, Z ← W, 0 // initialize dequantized and quantized weights 5 for j ← 1 to c do 6 ζ ← W[j, :]/S[j, :] // element-wise divide current row by its scales 7 Z[j, :] ← ROUND (ζ, Z†) // quantize coefficients to the target grid 8 Q[j, :] ← Z[j, :] ∗ S[j, :] // dequantize current row back to weight space 9 ε ← Q[j, :] − W[j, :] // quantization error for current row 10 W[j :, :] ← W[j :, :] + L[j :, j]ε // propagate error to not-yet-quantized rows; broadcast over columns 11 end 12 Z, Q ← P Z, P Q // undo reorder to restore original input order; return integers and dequantized weights

## 3.2 The Closest Vector Problem (Cvp)

Problem. Let B = [b1*, . . . ,* bc] ∈ R
n×c be a set of c basis vectors of dimension n with bj ∈ R
n and n ≥ c = rank (B). Let y ∈ R
n be an external target vector to approximate. Let z ∈ Z
c be the
(unknown) integer vector representing the basis combinations of the lattice vector. The goal is to find the vector on the lattice defined by the basis B that is closest to the target vector y, i.e., finding argminz∈Zc ∥Bz − y∥
2. A visualization of a two-dimensional CVP is shown in Figure 1 (a).

Babai's nearest plane algorithm. Babai's algorithm iteratively projects the target vector onto the nearest hyperplane of an LLL-reduced lattice and rounds the corresponding coefficient. Figure 1 (b) visualizes the basis reduction step and Figure 1 (c-d) visualize the projection steps.

![3_image_0.png](3_image_0.png)

Algorithm 2 is the pseudocode of Babai's nearest plane algorithm to solve CVP. For better computational efficiency, the pseudocode uses a conceptually equivalent approach. Instead of projecting the target vector onto the nearest hyperplane, it moves the target vector along the basis direction towards the hyperplane where the origin lies. The projection error is retained in the updated target vector as it is orthogonal to the hyperplane and will not affect subsequent projections. Additional notations are as follows. Function LLL returns the transformation matrix of the LLL reduction with the parameter delta defaulting to 34
. Function QR returns the orthogonal matrix in QR decomposition, which is the same as the normalized Gram-Schmidt orthogonalization process. ⟨·, ·⟩ denotes the vector dot product. Function ROUND is defined as in the GPTQ algorithm.

## Algorithm 2: Babai'S Nearest Plane

Input: lattice basis (column vectors) B ∈ R
n×c, target vector y ∈ R
n Output: closest lattice vector's basis coefficients z ∈ Z
c 1 T ← LLL (B) // unimodular transformation matrix from LLL basis reduction 2 A ← BT // reduce the basis 3 Φ ← QR (A) // normalized Gram-Schmidt process (take the Q matrix from the QR
decomposition)
4 y
′, z ← y, 0 // initialize residual target and integer solution in reduced basis 5 for j ← c to 1 do 6 ζ ← ⟨Φ[:, j], y
′⟩ / ⟨Φ[:, j], A[:, j]⟩ // exact coefficient along the unnormalized Gram-Schmidt vector; ratio between the projections of residual and the reduced basis on the Gram-Schmidt direction 7 z[j] ← ROUND (ζ, Z) // round to the nearest plane 8 y
′ ← y
′ − A[:, j]z[j] // update the residual 9 end 10 z ← T z // map integer solution back to the original basis and return Babai's error bound. Figure 1 shows the rounding boundaries of the optimal (e), round-to-nearest (RTN) (f), and Babai's algorithm without basis reduction (g-h). Compared to RTN, Babai's algorithm generates rectangular partitions and thus has a smaller worst-case error. The error bound has been proven in Babai (1986). Formally, let Φ = [ϕ1*, . . . ,* ϕc] be the set of normalized Gram-Schmidt vectors of the LLL-reduced basis A = [a1*, . . . ,* ac]. Let A˜ = [a˜1*, . . . ,* a˜c] denote the unnormalized Gram-Schmidt vectors with a˜j = ⟨ϕj , aj ⟩ ϕj . At iteration j, the algorithm replaces the exact coefficient ζ with the closest integer, so the deviation satisfies |ζ − z[j]| ≤ 12
. Hence, the error component along a˜j has norm at most 1 2
∥a˜j∥. Because the A˜ is orthogonal, these error components add in Euclidean norm, giving a bound on the residual (error) vector y
′: ∥y
′∥
2 ≤
1 4 Pcj=1 ∥a˜j∥
2 =
1 4 Pcj=1 ⟨ϕj , aj ⟩
2. Babai's algorithm guarantees to return the center vector of the hyper-cuboid
(Figure 1 (g)) constructed by the unnormalized Gram-Schmidt vectors A˜ where the target y is located.

Equality is attained when the target y lies at the corner of the hyper-cuboid, so the bound is tight. Babai
(1986) additionally proved a relative error bound for γ with ∥Bz − y∥ ≤ γ · minz′∈Zc ∥Bz′ − y∥.

The bound is 1 ≤ γ ≤
r1 + max1≤j≤c Pjj
′=1∥a˜j′∥
2
∥a˜j ∥
2 ≤
√c + 1 · max1≤j
′≤j≤c
∥a˜j′∥
∥a˜j ∥
.

## 4 Theoretical Results

We first show that weight quantization is an instance of the classical closest vector problem (CVP) in Section 4.1, which allows us to work in a lattice defined by the Hessian. We then reinterpret OBQ's, equivalently GPTQ's, error propagation step as a nearest hyperplane projection in Section 4.2, establishing our main equivalence in Section 4.3: GPTQ, running back-to-front, coincides exactly with Babai's nearest plane algorithm. This equivalence allows us to import Babai's guarantees to obtain a tight, layer-wise error bound in the no-clipping setting in Section 4.4. Finally, we analyze how quantization order influences this bound in Section 4.5.

## 4.1 Equivalence Between L2 Quantization And Cvp

A quantization problem with the L2 objective argminzi∈Z
c †
∥X diag (si) zi − Xwi∥
2and a CVP
with the L2 distance argminz∈Zc ∥Bz − y∥
2share the same solution (z = zi) whenever the structural conditions B = X diag (si) and y = Xwi hold and the solution domain matches. To ensure the solution domain matches, we can either disable the clipping in the quantization setup
(setting Z† = Z) or enable the clipping in the CVP setup (making z ∈ Z
c
†). Table 1 is a take-away dictionary showing the correspondence between the quantization and CVP concepts.

| Quantization symbol                        | CVP interpretation                                  |
|--------------------------------------------|-----------------------------------------------------|
| Input activation X ∈ R n×c                 | Basis directions (columns are generators)           |
| Scale si ∈ R c ̸=0                          | Basis stretches                                     |
| B(i) = X diag (si) ∈ R n×c                 | Lattice basis (columns are generators)              |
| Weight wi ∈ R c                            | Floating-point coordinates on the unstretched basis |
| Integer weight representation zi ∈ Z c †   | Integer coordinates on the lattice basis            |
| Dequantized weight qi = diag (si) zi ∈ R c | Dequantized coordinates on the unstretched basis    |
| Target output activation y(i) = Xwi ∈ R n  | External target vector to approximate               |

We can introduce a factor of the Hessian matrix, X = [χ1*, . . . ,* χc] with X⊤X = X
⊤X . The loss can then be reformulated as ∥X diag (si) zi − X wi∥
2.

Theorem 1 (Quantization and CVP) The CVPs using any possible factors X *of the Hessian matrix* X⊤X *are equivalent under an orthogonal transformation (rotation and reflection) of the lattice and* external target vector.

Proof Let X and X
′be two possible factors of the Hessian matrix with X
⊤X = X
′⊤X
′. The inner products ⟨χj1
, χj2
⟩ and χ
′j1
, χ
′j2 must be equal for all 1 ≤ j1, j2 ≤ c. In other words, the lengths ∥χj1 ∥ =χ
′ j1
, and the angles ∠ (χj1, χj2) = ∠χ
′ j1
, χ
′ j2
, for all 1 ≤ j1, j2 ≤ c.

According to Theorem 1, any decomposition factor X of the Hessian matrix X⊤X can be used instead of X without changing the geometric properties of the CVP and its associated quantization problem. This is useful for reducing the computational cost, e.g., we may use a square matrix X ∈ R
c×cinstead of the rectangular matrix X ∈ R
n×c.

## 4.2 Obq'S Geometric Interpretation

![5_image_0.png](5_image_0.png)

We first demonstrate the geometric interpretation of OBQ (GPTQ's slower predecessor) to facilitate our equivalence proof of GPTQ and Babai's algorithm in Section 4.3.

Figure 2: Equivalence of OBQ's error propagation and Babai's projection. (a) 3D plot showing the target being projected onto the nearest plane. (b) 3D plot showing how the projection error is propagated. (c) 2D plot showing the vectors on the nearest hyperplane in (a-b). (d) 2D plot showing the vectors on the orthogonal projection plane in (b). Theorem 2 (Error Propagation and Babai's Projection) Babai's nearest plane algorithm iteratively projects the target vector onto the nearest hyperplane and rounds the coefficient. The OBQ error propagation step (Eq. 2) is exactly this projection on the original basis B = X diag (si) without basis reduction.

Proof Let B = [b1*, . . . ,* bc] be the basis with bj being a basis vector. Let J be the set of unprojected indices with j1, j2 ∈ J and j1 ̸= j2. Let y =Pj∈J
ζjbj be the current residual target where ζj ∈ R is a real number to be rounded to integers. Let *N HP* := ⌊ζj2⌉ bj2 + Span {bj | j ̸= j2} be the nearest hyperplane that is orthogonal to the Gram-Schmidt vector bj2 −Pj̸=j2 Projbj
(bj2).

Figure 2 (a) is a 3D plot showing the projection error vector ∆y = ProjNHP (y) − y. We focus on analyzing the error propagation in the direction of basis bj1induced by the projection of basis bj2and collapse the span of other basis vectors to a single dimension as illustrated by the hyperline HL := ⌊ζj2⌉ bj2 + Span {bj |j ̸= j1, j2}. Figure 2 (b) is a 3D plot showing the decomposition of the error ∆y =Pj∈J ∆ζjbj as the error component vectors in the basis directions. Figure 2
(c) is a 2D plot showing the vectors on plane N HP. The number ζj will be updated to ζj + ∆ζj such that ProjNHP (y) = Pj∈J
(ζj + ∆ζj ) bj . Next, let N = B−⊤ = [n1*, . . . ,* nc] be the inverse basis. Then, we have ⟨nj , bj ⟩ = 1 and nj ⊥ bj
′ , ∀j ̸= j
′. We project all the vectors in Figure 2 (b) onto the orthogonal projection plane OPP := Span {nj |j = j1, j2} that is orthogonal to the hyperline HL, and continue the proof in the 2D geometry in Figure 2 (d). Denote the angle θ = ∠ (nj1, nj2) = π−∠ (ProjOPP (bj1),ProjOPP (bj2)). Then, 
∆ζj1 ∥ProjOPP (bj1 )∥
∆ζj2 ∥ProjOPP (bj2 )∥
= cos θ =
⟨nj1
,nj2 ⟩
∥nj1 ∥∥nj2 ∥
=
∥nj2 ∥
∥nj1 ∥
⟨nj1
,nj2 ⟩
⟨nj2
,nj2 ⟩
. For j = j1, j2, ∥ProjOPP (bj )∥ ∥nj∥ =
⟨ProjOPP (bj ),nj ⟩
cos( π 2 −θ)=
⟨bj ,nj ⟩
cos( π 2 −θ)
=1 cos( π 2 −θ)
. For j, j′ ∈ {j1, j2}, ⟨nj , nj
′ ⟩ =N⊤N[*j, j*′] = B⊤B−1[*j, j*′].

Combining the above equations, ∆ζj1 =
∥ProjOPP (bj2 )∥∥nj2 ∥ ∥ProjOPP (bj1 )∥∥nj1 ∥
⟨nj1
,nj2 ⟩
⟨nj2
,nj2 ⟩
∆ζj2 =
⟨nj1
,nj2 ⟩
⟨nj2
,nj2 ⟩
∆ζj2 =

![6_image_0.png](6_image_0.png)

(B⊤B)
−1[j1,j2]
(B⊤B)
−1[j2,j2]
∆ζj2. Finally, substituting B = (X diag (si)) [:, J] and ζj =
wi[j]
si[j]
completes the Corollary 3 (OBQ Dimension Selection) *At each dimension selection step (Eq. 1), OBQ selects* the not-yet-quantized dimension j such that the nearest hyperplane of dimension j *is closest to the* target residual vector. Proof We use the same notations defined in Theorem 2. Figure 3 is a 2D plot showing the distance (projection error or quantization error) between the target residual vector y and the nearest hyperplane *N HP* of the basis bj2. For better illustration, we collapse *N HP* into a single dimension. The distance ∥∆y∥ can be expressed as ∥∆y∥ =Projnj2
(∆y)
 =
|∆ζj2| Projnj2
(bj2)
 =
|∆ζj2||⟨bj2
,nj2 ⟩|
∥nj2 ∥=
|∆ζj2| ∥nj2 ∥
. For each wi, OBQ independently selects j = argminj∈J(qi[j]−wi[j])2
(X[:,J]⊤X[:,J])−1[j,j]
= argminj∈J
(∆ζj )
2
⟨nj ,nj ⟩ 
= argminj∈J
|∆ζj | ∥nj ∥
as the next dimension to quantize, which is exactly minimizing this distance. 4.3 GPTQ AND BABAI'S ALGORITHM Originally, GPTQ (Algorithm 1) runs from the first to the last dimension (j ← 1 to c) while Babai's algorithm (Algorithm 2) runs from the last to the first dimension (j ← c to 1). This is the only (superficial) difference between the two algorithms, as formalized below. Theorem 4 (GPTQ and Babai) GPTQ and Babai's algorithm without basis reduction will have the same results if we align the dimensional order of these two algorithms, e.g., running GPTQ from the last to the first dimension. Proof We prove this theorem both geometrically and algebraically. We first present the geometric proof. Theorem 2 shows that each intermediate weight vector produced by OBQ, equivalently GPTQ, can be viewed as Babai's residual vector in the activation space. At step j (running from the last to the first dimension, j ← c to 1), GPTQ's error propagation update is exactly Babai's projection at step j, which projects the current residual of the target vector onto the hyperplane orthogonal to the j-th Gram-Schmidt vector. Alternatively, we present a more rigorous algebraic proof. Section B describes the exact quantization procedures using Babai's algorithm in more detail, with the pseudocode in Algorithm 4. Section C contains the equivalence proof, in which we proceed in three steps. First, we rewrite GPTQ to track the cumulative quantization error and show that this form is algebraically equivalent to the standard implementation. Second, we run GPTQ in the back-to-front order and replace the lower triangular factor with an upper triangular one so that each update affects only the not-yet-quantized coordinates. Third, we prove that the step-wise rounding decisions of the back-to-front GPTQ coincide with those of Babai's algorithm. Geometric interpretation of GPTQ. Theorem 4 shows that, if we regard the activations as the lattice basis and transform the floating-point weight vector to a target vector in the activation space, GPTQ performs an *orthogonal walk* through a nested sequence of affine subspaces in a pre-computed dimensional order. Ineffectiveness of composing algorithms. A seemingly appealing idea is to take the solution returned by any Babai iteration and then perform one further GPTQ-style error propagation step on the weights in the activation space, hoping to push the approximation even closer to the optimum. However, as proven in Section C.4, such an extra update vanishes: the final results of Z and Q remain unchanged. In other words, once Babai's projection has been executed, any subsequent GPTQ-style correction is algebraically redundant. This confirms that the equivalence in Theorem 4 is already tight; neither algorithm can be strengthened by composition.

## 4.4 Gptq'S Error Bound

Having established the correspondence between GPTQ and Babai's nearest plane algorithm, we can now import Babai's approximation guarantee to obtain an upper bound on the layer-wise quantization error in the no-clipping setting.

Theorem 5 (GPTQ Error Bound) Assume no clipping (Z† = Z) and let T be the permutation matrix of the reversed GPTQ quantization order (equivalently P with the reversed column order). Let D *be the diagonal matrix of the LDL decomposition of the permuted Hessian matrix* T
⊤X⊤XT . For every output channel i (1 ≤ i ≤ r) produced by Babai's algorithm, or equivalently the GPTQ algorithm executed back-to-front, the (absolute) quantization error has a tight upper bound: ∥X diag (si) zi − Xwi∥
2 ≤
1 4 T
−1si⊤ DT
−1si*. For the relative bound for* γ *with* ∥X diag (si) zi − Xwi∥ ≤ γ · minz
′
i∈Zc ∥X diag (si) z
′ i 
− Xwi∥*, we have* 1 ≤ γ ≤
r1 + max1≤j≤c Pjj
′=1 d 2 j
′
d 2 j
≤
√c + 1 · max1≤j
′≤j≤c dj
′
dj where dj =pD[*j, j*]T
−1si[j].

$$T^{-1}\mathbf{s}_{i})\ [j]$$

Proof The full proof of Theorem 5 is presented in Section D.1. If the scales si are small enough, we may assume the weights wi are nearly uniformly distributed within the hyper-cuboid constructed by Babai's orthogonalized basis vectors, the expected absolute error will be 1 3 of the worst-case bound. See Section D.2 for a proof.

## 4.5 The Role Of Quantization Order In Gptq

The quadratic form on the right-hand side of the absolute error bound in Theorem 5 is sensitive to the pivot order of the LDL decomposition of the Hessian matrix; this is the quantization order. Reordering the dimensions changes the entries of the diagonal matrix D before the scale siis "weighted" by them. A poor order may place large D entries against large si entries and hence inflate the bound.

For a batched quantization algorithm like GPTQ, the order should be independent of the output channel i. To develop a good heuristic order, a reasonable approximation to make, especially for large quantization group sizes, is that the elements of si[j] are equal for all 1 ≤ j ≤ c. Then we can focus on finding the optimal pivot order for the LDL decomposition of the Hessian matrix X⊤X to minimize tr (D). Finding the optimal order is NP-hard (Rose et al., 1976). However, heuristics often effectively reduce the trace term in practice. Even with clipping, heuristics can reduce the error. GPTQ introduces the act-order, the descending order of the Hessian diagonal, i.e., the ascending order of the Hessian diagonal when applied to Babai's algorithm. To improve upon act-order, we propose the min-pivot order, which is essentially taking the minimum diagonal entry at each LDL (or Cholesky) decomposition step. This order can be calculated by Algorithm 3, which has cubic time complexity and does not increase the overall time complexity of quantization. This order also has a geometric interpretation, as the order of the Gram-Schmidt orthogonalization process of the basis: always taking the shortest residual vector as the next one to orthogonalize, agreeing with Babai's relative error bound. Across our preliminary runs (Section D.3), min-pivot *consistently* reduces tr (D) relative to act-order, but the downstream accuracy gains are modest. We nevertheless report min-pivot as a principled choice, and view act-order as a cheap approximation that only considers the Hessian diagonal, which already captures most of the benefit when the Hessian matrix is well-conditioned. Algorithm 3: Min-Pivot Input: Hessian H ∈ R
c×c Output: order encoded as a permutation matrix T ∈ {0, 1}
c×c 1 J ← {1*, . . . , c*} // initialize the not-yet-pivoted indices 2 T ← 0 // initialize the output permutation matrix 3 for j ← 1 to c do 4 j
′ ← argminj
′∈JH[j
′, j′] // choose next index with the smallest current diagonal 5 H ← H − H[:, j′]H[j
′, :]/H[j
′, j′] // updates remaining entries with rank-1 Schur complement 6 T [j
′, j] ← 1 // record the index 7 J ← J \ {j
′} // mark pivot as used 8 end

## 5 Applications

The original GPTQ algorithm clips the overflowed integers at the rounding step, introducing large errors that violate the error bound in Theorem 5. In this section, we explore error-guaranteed variants of GPTQ that work in the no-clipping regime. We notice that enforcing no-clipping by simply increasing scales is counterproductive: larger scales enlarge the bound, and the resulting errors can exceed those of a clipped scheme such as MSE. Hence, any practical no-clipping design must account for the weight distributions that are known to have heavy outliers (Li et al., 2025). We would still like to apply small scales, but use small bitwidths for the bulk of inliers while handling the overflowed outliers with more storage budget without clipping them. We therefore propose two overflow-tolerant schemes. Scale-adjusted SpQR (SSQR). SpQR (Dettmers et al., 2024) keeps a small set of outliers in full precision, but it still leaves clipping in place: weights are grouped, the outliers and a shared scale are chosen per group before the GPTQ updates, and there is no guarantee the updated inlier weights stay within the representable range. We design SSQR with a scale-adjustment mechanism to fix this issue. For simplicity, we discard SpQR's second-level quantization for the scales. For a weight vector wi ∈ R
c, we represent the quantized weight qi ∈ R
cas diag (si) zi + ξi where z ∈ Z
c†
is the low-bitwidth integer weight vector, si ∈ R
c
̸=0 is the floating-point scale vector with each scale shared per group (only one number per group is actually stored), and ξi ∈ R
cis the sparse floating-point outlier vector (stored in the compressed sparse row format, CSR) that captures all the overflowed weights after GPTQ's error propagation. The scale-adjustment mechanism tunes the scale si until the density of ξi satisfies the specified rate. Because exhaustive trial-and-error over per-group scales is infeasible in large layers, the mechanism only proportionally changes si so that the search space reduces to one dimension. With the observation that the outlier rate is negatively related to the scales in general, this can be done via binary search: initialize si using MSE, quantize wi with the specified format using GPTQ without clipping, calculate the density of ξi, and adjust si and iterate. Section E.1 Algorithm 9 is the pseudocode. Huffman-encoded post-training quantization (HPTQ). To better align with the infinite, unconstrained lattice in CVP, we design HPTQ, which represents both inliers and outliers in a unified, equal-spaced integer grid. The idea is to use Huffman encoding, which was also explored for network compression by Choi et al. (2017). We quantize the weight matrix W ∈ R
c×ras Q = sZ
with a single scalar s ∈ R̸=0 and integers Z ∈ Z
c×r. We select s via an entropy-guided binary search: initialize a range proportional to the maximum weight, quantize to unclipped integers with GPTQ, measure the Huffman coding cost of Z, and adjust s until the encoded bits meet a target average bitwidth. This yields uneven-bitwidth representations that preserve accuracy while meeting a compression budget. Section E.1 Algorithm 11 is the pseudocode.

Experiments compare round-to-nearest (RTN), original GPTQ, HPTQ, and SSQR with 1~5% outliers. We also include Huffman-encoded RTN (HRTN) as a baseline to HPTQ, which mirrors HPTQ but replaces GPTQ with RTN (pseudocode: Section E.1 Algorithm 12). The quantization order is act-order for all methods. RTN, GPTQ, and SSQR use group size 128. RTN and GPTQ calculate the scales with the MSE method. Figure 4 (a-b) shows that HPTQ sustains low perplexity on Qwen3-8B at reduced bitwidths and scales favorably across model sizes, with 3.125-bit emerging as Pareto optimal in terms of perplexity vs compression. Further information can be found in Section E, including the experimental setup (Section E.2), additional metrics such as benchmark results (Qwen3 models: Section E.3; Llama models: Section E.4), and comparison with other methods (Section E.5).

![9_image_0.png](9_image_0.png)

Figure 4: (a) Comparison of quantization methods (RTN, GPTQ, HRTN, HPTQ, and SSQR with 1~5% outliers) on Qwen3-8B evaluated on WikiText-2. Perplexity is plotted against the average effective bitwidth per weight, with the BF16 baseline shown as a horizontal line. HPTQ has the best (lowest) perplexity. See Section E.3 for zero-shot evaluation results. (b) Scaling behavior of HPTQ across multiple model sizes (0.6B, 1.7B, 4B, 8B, 14B) and bitwidths (4.125, 3.125, 2.125). The x-axis denotes the effective model size after quantization, and the y-axis shows perplexity on WikiText-2. Each curve corresponds to a fixed bitwidth, while points along a curve represent different model scales. Using our HPTQ method, 3.125-bit stands out as the Pareto optimal bitwidth (optimal perplexity vs compression trade-offs). (c) End-to-end inference speedups of our SSQR kernel vs the PyTorch BF16 matrix multiplication kernel on NVIDIA RTX A6000 GPU. We run the Qwen3-8B model across multiple outlier rates (0%~5%) and inlier bitwidths (4, 3, 2) and measure the TPOT (time per output token) metric. Our kernel achieves about 2× speedup end-to-end. CUDA inference kernel. We implement an inference kernel for SSQR in CUDA/C++, optimized for low-batch latency, handling both dense inliers and sparse outliers while targeting the Ampere platform. The kernel supports group-quantized inlier weights in the 2-4-bit range with scales in 16 bits and support for unstructured sparsity, used to avoid weight clipping. Figure 4 (c) visualizes the end-to-end speedup in the LLM decoding phase vs the PyTorch BF16 kernel. Our kernel achieves about 2× speedup across different bitwidth and outlier rate settings when generating 128 new tokens at a batch size of 1. Technical details and layer-wise speedups are described in Section E.6.

## 6 Closing Remarks

Summary. We have shown that GPTQ, when executed back-to-front, is mathematically identical to Babai's nearest plane algorithm applied to the lattice defined by a layer's Hessian without basis reduction. Based on this theory, we propose error-guaranteed practical methods and provide optimized CUDA kernels that deliver low-latency inferences. More broadly, the lattice perspective opens a twoway channel: decades of the closest vector problem (CVP) heuristics can refine practical quantizers, while the behavior of massive neural networks may, in turn, inspire new questions for lattice theory.

Future Work. Looking ahead, extending the analysis to clipped grids and exploring (scale-aware)
basis reductions are the immediate next steps. However, we emphasize that the state-of-the-art 4-bit floating-point formats (e.g., MXFP4 and NVFP4) are essentially no-clipping (Egiazarian et al., 2025; Chen et al., 2026): since they use very small quantization groups (32 and 16, respectively), the near-optimal choice of scale is AbsMax per-group, which leads to no weight being clipped. As such, a no-clipping analysis of these formats would directly apply to actual practice. We will also extend the lattice view beyond weight-only linear layers to activation and KV-cache quantization.

## Ethics Statement

Throughout this work, we have strictly adhered to the ICLR Code of Ethics. All datasets utilized in our experiments are publicly available and widely recognized within the scientific community. We ensure that these datasets do not contain any personally identifiable information or sensitive content. Our work does not involve human subjects, animals, or any form of personal data collection. We have thoroughly considered potential dual-use concerns and do not foresee any harmful applications of our methods. There are no conflicts of interest to declare, and no external sponsorship influenced the outcomes of this research. All experiments were conducted with integrity and transparency.

## Reproducibility Statement

We are committed to ensuring that our work is transparent and reproducible. To facilitate this, clear explanations of any assumptions and a complete proof of the claims have been included in the main text and appendix. We also share the source code as part of the supplementary materials. The code is documented and includes instructions for setting up the environment, running the simulations, and reproducing the results presented in our paper. By making our resources openly available and providing detailed explanations, we aim to enable the research community to validate and build upon our findings.

## Acknowledgments

This research was supported by the Scientific Service Units (SSU) of ISTA through resources provided by Scientific Computing (SciComp). The ISTA team was supported by generous grants from Google and NVIDIA. The authors thank Vage Egiazarian for the discussions on this work.

## References

László Babai. On lovász' lattice reduction and the nearest lattice point problem. *Combinatorica*, 6
(1):1–13, March 1986. ISSN 1439-6912. doi: 10.1007/BF02579403. URL https://link.s pringer.com/article/10.1007/BF02579403.

Johann Birnick. The lattice geometry of neural network quantization - a short equivalence proof of gptq and babai's algorithm, 2025. URL https://arxiv.org/abs/2508.01077.

Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M De Sa. Quip: 2-bit quantization of large language models with guarantees. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), *Advances in Neural Information Processing Systems*, volume 36, pp. 4396–4429. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc
/paper_files/paper/2023/file/0df38cd13520747e1e64e5b123a78ef8-P
aper-Conference.pdf.

Jiale Chen, Yalda Shabanzadeh, Elvir Crncevi ˇ c, Torsten Hoefler, and Dan Alistarh. The geometry of ´
llm quantization: Gptq as babai's nearest plane algorithm, 2025. URL https://arxiv.org/ abs/2507.18553.

Jiale Chen, Vage Egiazarian, Roberto L. Castro, Torsten Hoefler, and Dan Alistarh. Wush: Nearoptimal adaptive transforms for llm quantization, 2026. URL https://arxiv.org/abs/25 12.00956.

Yoojin Choi, Mostafa El-Khamy, and Jungwon Lee. Towards the limit of network quantization. In International Conference on Learning Representations, 2017. URL https://openreview.n et/forum?id=rJ8uNptgl.

Tim Dettmers, Ruslan A. Svirschevski, Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, and Dan Alistarh. SpQR: A sparse-quantized representation for near-lossless LLM weight compression. In *The Twelfth International Conference* on Learning Representations, 2024. URL https://openreview.net/forum?id=Q1u2 5ahSuy.

I. Dinur, G. Kindler, R. Raz, and S. Safra. Approximating cvp to within almost-polynomial factors is np-hard. *Combinatorica*, 23(2):205–243, April 2003. ISSN 1439-6912. doi: 10.1007/s00493-003
-0019-y. URL https://link.springer.com/article/10.1007/s00493-003-0 019-y.

Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, and Dan Alistarh. Extreme compression of large language models via additive quantization. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 12284–12303. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/egiazarian24a.html.

Vage Egiazarian, Roberto L. Castro, Denis Kuznedelev, Andrei Panferov, Eldar Kurtic, Shubhra Pandit, Alexandre Marques, Mark Kurtz, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Bridging the gap between promise and performance for microscaling fp4 quantization, 2025. URL https://arxiv.org/abs/2509.23202.

Elias Frantar and Dan Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), *Advances in Neural Information Processing Systems*, volume 35, pp. 4475–4488. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper_files/p aper/2022/file/1caf09c9f4e6b0150b06a07e77f2710c-Paper-Conference. pdf.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. OPTQ: Accurate quantization for generative pre-trained transformers. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=tcbBPnfwxS.

Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, and Kurt Keutzer.

A survey of quantization methods for efficient neural network inference, 2021. URL https:
//arxiv.org/abs/2103.13630.

Babak Hassibi, David G. Stork, and Gregory J. Wolff. Optimal brain surgeon and general network pruning. In *IEEE International Conference on Neural Networks*, pp. 293–299 vol.1, 1993. doi:
10.1109/ICNN.1993.298572. URL https://ieeexplore.ieee.org/document/298 572.

Ravi Kannan. Minkowski's convex body theorem and integer programming. *Mathematics of* Operations Research, 12(3):415–440, 1987. ISSN 0364765X, 15265471. URL http://www. jstor.org/stable/3689974.

Eldar Kurtic, Alexandre Noll Marques, Shubhra Pandit, Mark Kurtz, and Dan Alistarh. "give me BF16 or give me death"? accuracy-performance trade-offs in LLM quantization. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), *Proceedings of the* 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 26872–26886, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1304. URL https://aclanthology .org/2025.acl-long.1304/.

Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. In D. Touretzky (ed.), Advances in Neural Information Processing Systems, volume 2. Morgan-Kaufmann, 1989. URL https:
//proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1 c7093bd25041881277658-Paper.pdf.

Arjen Klaas Lenstra, Hendrik Willem Lenstra, and László Lovász. Factoring polynomials with rational coefficients. *Mathematische Annalen*, 261(4):515–534, dec 1982. ISSN 1432-1807. doi:
10.1007/BF01457454. URL https://link.springer.com/article/10.1007/BF
01457454.

Xinlin Li, Osama Hanna, Christina Fragouli, and Suhas Diggavi. ICQuant: Index coding enables low-bit LLM quantization. In *Second Conference on Language Modeling*, 2025. URL https:
//openreview.net/forum?id=m6nBgFSMTL.