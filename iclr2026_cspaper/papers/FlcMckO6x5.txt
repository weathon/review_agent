# Separable Neural Networks: Approximation Theory, Ntk Regime, And Preconditioned Gra- Dient Descent

Yisi Luo1**, Deyu Meng**1∗
1Xi'an Jiaotong University yisiluo1221@foxmail.com, dymeng@mail.xjtu.edu.cn

## Abstract

Separable neural networks (SepNNs) are emerging neural architectures that significantly reduce computational costs by factorizing a multivariate function into linear combinations of univariate functions, benefiting downstream applications such as implicit neural representations (INRs) and physics-informed neural networks (PINNs). However, fundamental theoretical analysis for SepNN, including detailed representation capacity and spectral bias characterization & alleviation, remains unexplored. This work makes three key contributions to theoretically understanding and improving SepNN. First, using Weierstrass-based approximation and universal approximation theory, we prove that SepNN can approximate any multivariate function with arbitrary precision, confirming its representation completeness. Second, we derive the neural tangent kernel (NTK) regimes for SepNN, showing that the NTK of infinite-width SepNN converges to a deterministic (or random) kernel under infinite (or fixed) decomposition rank, with corresponding convergence and spectral bias characterization. Third, we propose an efficient separable preconditioned gradient descent (SepPGD) for optimizing SepNN, which alleviates the spectral bias of SepNN by provably adjusting its NTK spectrum. The SepPGD enjoys an efficient O(nD) complexity for n D training samples, which is much more efficient than previous neural network PGD methods. Extensive experiments for kernel ridge regression, image and surface representation using INRs, and numerical PDEs using PINNs validate the efficiency of SepNN and the effectiveness of SepPGD for alleviating spectral bias.

## 1 Introduction

Separable neural networks (SepNNs) are a class of neural architectures that represent a multivariate function as a linear combination of univariate functions, each parameterized by a lightweight factor neural network (Liang et al., 2022; Cho et al., 2023; Luo et al., 2024). A key advantage of SepNNs lies in their ability to significantly reduce computational costs by reducing network propagations. The computational efficiency makes SepNN particularly advantageous and efficient in applications such as implicit neural representations (INRs) (Liang et al., 2022; Luo et al., 2024), physical-informed neural networks (PINNs) (Cho et al., 2023; Yu et al., 2024), and neural radiance fields (Chen et al., 2022). Compared to other efficient architectures for neural networks, SepNNs hold unique efficiency advantages. Especially, a classical line of work employs tensor decomposition to factorize the weights of networks (Liu & Parhi, 2023), thereby reducing the number of parameters. This efficient architecture is the most closely related to SepNNs. However, SepNNs are motivated by a fundamentally different idea. Rather than decomposing the network weights, the principle of SepNNs is to separate the input vector into multiple smaller inputs and process each input using factor networks. This design uniquely improves efficiency in scenarios involving coordinate-based neural networks and function evaluations on grids by reusing the factor outputs. Such a structure is particularly advantageous in applications like INRs (Liang et al., 2022), which map coordinates to pixel values, and PINNs (Cho et al., 2023), which map coordinates to physical
∗Corresponding author. Code of SepPGD is in https://github.com/YisiLuo/SepPGD
1 fields. Moreover, SepNNs not only facilitate efficient training, but also offer improved interpretability and robustness by leveraging low-dimensional representations and interpretable factor modeling, demonstrating strong potential for scientific applications such as separable operator learning (Yu et al., 2024), radio map construction in wireless communication (Yuan et al., 2025), geophysical full waveform inversion in Earth science (Chen et al., 2025), and transcriptomics analysis in bioinformatics (Song et al., 2023). Therefore, the structure and efficiency of SepNNs hold important value across a variety of practical machine learning applications. We make more discussions on the efficiency advantage of SepNNs in Section A.1. Formally, a D-variable SepNN fΘ(x1, · · · , xD) can be expressed as

$$f_{\Theta}(x_{1},\cdots,x_{D})=L$$

(xD)) : R

$$\mathcal{L}(f_{\Theta_{1}}(x_{1}),\cdots,f_{\Theta_{d}}(x_{d}))$$
) : $\mathbb{R}^D\to\mathbb{R}$,
where L denotes a type of linear combination that encodes the interactions between different univariate factor functions {fΘd(xd)}
D
d=1 (the factor functions {fΘd(xd)}
D
d=1 are parameterized by
univariate neural networks, such as multi-layer perceptrons (MLPs)), and Θ = {Θ1, *· · ·* , ΘD} are
learnable parameters. In the bivariate case D = 2, the inner product is a classical (if not only) choice for L (see for example (Liang et al., 2022; Wang et al., 2025)):
$$f_{\Theta}(x_{1},x_{2})=f_{\Theta_{1}}(x_{1})^{\top}f_{\Theta_{2}}(x_{2}):\mathbb{R}^{2}\to\mathbb{R},$$
2 → R, (1)
where fΘ1(x1), fΘ2(x2) : R → R
R are factor functions mapping separated inputs to R-dimensional latent vectors, and R serves as a "rank" parameter that determines the representation capacity of the SepNN. In the multivariate case where D > 2, multiple alternatives exist for defining the linear combination L. A natural and widely-used option is the tensor canonical parafac (CP) decomposition (Kargas & Sidiropoulos, 2021), which will be the primary focus of this study:

(CP) $f_{\Theta}(x_{1},\cdots,x_{D})=\sum_{r=1}^{R}\left(f_{\Theta_{1}}(x_{1})\right)_{r}\left(f_{\Theta_{2}}(x_{2})\right)_{r}\cdots\left(f_{\Theta_{D}}(x_{D})\right)_{r}:\mathbb{R}^{D}\rightarrow\mathbb{R}$, (2)
$$(1)$$

which computes the inner product between D factor functions {fΘd(xd) : R → R
R}
D
d=1, and
(fΘd(xd))r denotes the r-th component of fΘd(xd). The CP SepNN (2) degenerates into (1)
when D = 2. In addition, other stochastic tensor decomposition formulations can be considered to construct the SepNN. For instance, the tensor-train (TT) decomposition represents multivariate functions by a sequence of lower-dimensional tensor functions (Gorodetsky et al., 2019; Zhou et al., 2025), and the tensor Tucker decomposition introduces an additional core tensor C to encode weighted multilinear relationships into SepNNs (Luo et al., 2024):

(TT) $f_{\Theta}(x_{1},\cdots,x_{D})=\sum_{r_{1}=1}^{R_{1}}\sum_{r_{2}=1}^{R_{2}}\cdots\sum_{r_{D-1}=1}^{R_{D-1}}\left(f_{\Theta_{1}}(x_{1})\right)_{1,r_{1}}\left(f_{\Theta_{2}}(x_{2})\right)_{r_{1},r_{2}}\cdots\left(f_{\Theta_{D}}(x_{D})\right)_{r_{D-1},1},$ (3) (Tucker) $f_{\Theta}(x_{1},\cdots,x_{D})=C\times_{1}f_{\Theta_{1}}(x_{1})\times_{2}\cdots\times_{D}f_{\Theta_{D}}(x_{D}):\mathbb{R}^{D}\rightarrow\mathbb{R}.$
Here, C ∈ R
R1*×···×*RD denotes the core tensor in the Tucker decomposition model,
(R1, · · · , RD) denotes the TT or Tucker rank of the model, ×d : R
R1*×···×*RD × R
Rd →
R 
R1×···×Rd−1×Rd+1*×···×*RD denotes the tensor product between a tensor and a vector, i.e., C ×d fΘd(xd) := foldd(unfoldd(C)fΘd(xd)), where fΘd(xd) ∈ R
Rd is the factor output in the Tucker model, unfoldd : R
R1*×···×*RD → R
Qd06=d Rd0×Rd denotes the unfolding operator from a tensor to a matrix, and foldd : R
Qd06=d Rd0 → R
R1×···×Rd−1×Rd+1*×···×*RD denotes the folding operator from a vector to a tensor. The (fθd(xd))rd−1,rddenotes the (rd−1, rd)-th output component of the matrixvalued univariate function fθd
(xd) : R → R
Rd−1×Rd in the TT model. The aforementioned linear combinations L can be viewed as specific instances within the generalized linear Einstein summation for multi-dimensional array (Ahlander, 2002). Furthermore, nonlinear combinations (e.g., nonlinear activations (Li et al., 2025b)) can also be considered to enhance the stochasticity among factors. The efficiency advantage of SepNNs comes from its separability nature. Especially, when training on a D-dimensional grid tensor with each dimension containing n training samples (for instance, with training inputs of the form {(x1, · · · , xD) | xd = 1, 2, · · · , n, d = 1, · · · , D}, which contains n D training samples), the SepNN only needs to query nD times of inputs by querying the separated input xd in each dimension via factor functions and then combining the outputs through the linear combination L. Hence, the computational complexity of a SepNN scales as O(nD) in an epoch, compared to O(n D) for a conventional neural network (Cho et al., 2023; Luo et al., 2024). This efficiency superiority makes SepNNs advantageous in downstream applications such as INRs (Liang et al., 2022; Luo et al., 2024) and PINNs (Cho et al., 2023; Vemuri et al., 2025). While SepNNs have enabled a variety of promising applications, their theoretical foundations remain relatively underdeveloped, hindering a deeper understanding of their representation capacity and optimization behaviors. In this work, we seek to address the following fundamental questions regarding the theoretical aspects of SepNNs: 1) Do SepNNs possess sufficient representation capacity to approximate any continuous multivariate function in Euclidean space? 2) How can we characterize the training dynamics of SepNNs and identify any inherent spectral bias during optimization? 3) How can the spectral bias of SepNNs be mitigated to further enhance training efficiency? To address these fundamental questions, this work makes the following contributions:
- Using a novel combination of Weierstrass-based approximation and universal approximation theories, we rigorously establish an approximation theorem that SepNNs possess the capacity to approximate any continuous multivariate function with arbitrary precision, thereby confirming their representation completeness.

- We derive the neural tangent kernel (NTK) regimes for SepNNs under different asymptotic conditions. The SepNN's NTK converges to a deterministic kernel under infinite width and infinite rank, and converges to a random kernel under infinite width and fixed rank, providing new insights into the spectral bias characterization and training behavior of SepNNs.

- We further propose a scalable separable preconditioned gradient descent (SepPGD) method that provably adjusts the eigenvalue distribution of NTK matrix, effectively alleviating spectral bias of SepNNs. The SepPGD achieves a significantly lower computational complexity of O(nD) for n D training samples, which is much more efficient than existing neural network preconditioning methods. Extensive experiments across various downstream tasks including kernel ridge regression, image & surface representation using INRs, and PINNs demonstrate the improved efficiency and effectiveness of our SepPGD approach for alleviating spectral bias of SepNNs.

## 2 Approximation Theory Of Sepnn

Approximation Theory. For any continuous multivariate function f(x1, · · · , xD) defined on a compact set, it is well-established that standard neural networks such as MLPs with suitable activation functions can approximate it to arbitrary accuracy (Leshno et al., 1993; Pinkus, 1999)—a result known as the universal approximation theorem. However, such a universal approximation theory remains lacking for SepNN structures. To fill this blank, our first contribution is to establish a universal approximation theorem of SepNNs (including CP, TT, and Tucker (2)-(3)) as follows.

Theorem 1 (Universal approximation theorem of multivariate SepNNs). Let X1 ⊂ R, X2 ⊂ R, · · · , XD ⊂ R and X = X1 × X2 *× · · · × X*D ⊂ R
D *be any compact sets, and let* f(x) : X → R
be a continuous multivariate function where x := (x1, · · · , xD). For any  > 0, the following statements hold, in which all MLPs are drawn in the set {W2σ(W1x + b) : W2 ∈ RW˜ ×W ,W1 ∈
RW×1, b ∈ RW , W , W ˜ ∈ N+} with σ *a non-polynomial function.*

_(CP) There exist rank $R\in\mathbb{N}_{+}$ and univariate MLPs $f_{\Theta_{d}}:\mathcal{X}_{d}\rightarrow\mathbb{R}^{R}\left(d=1,\cdots,D\right)$ such that_  $$\sup_{\mathbf{x}\in\mathcal{X}}\left|f(\mathbf{x})-\sum_{r=1}^{R}\left(f_{\Theta_{1}}(x_{1})\right)_{r}\left(f_{\Theta_{2}}(x_{2})\right)_{r}\cdots\left(f_{\Theta_{D}}(x_{D})\right)_{r}\right|<\epsilon.$$
* _There exist ranks_ $R_{1},\cdots,R_{D-1}\in\mathbb{N}_{+}$ _and univariate MLPs_ $f_{\Theta_{d}}:\mathcal{X}_{d}\to\mathbb{R}^{R_{d-1}R_{D-1}}$ ($d=1,\cdots,D$, $R_{0}=R_{D}=1$) such that_ $$\sup_{x\in\mathcal{X}}\left|f(\mathbf{x})-\sum_{r_{1}=1}^{R_{1}}\sum_{r_{2}=1}^{R_{2}}\cdots\sum_{r_{D-1}=1}^{R_{D-1}}\left(f_{\Theta_{1}}(x_{1})\right)_{1,r_{1}}\left(f_{\Theta_{2}}(x_{2})\right)_{r_{1},r_{2}}\cdots\left(f_{\Theta_{D}}(x_{D})\right)_{r_{D-1},1}\right|<\epsilon.$$
(Tucker) There exist ranks R1, · · · , RD ∈ N+*, univariate MLPs* fΘd: Xd → R
Rd (d = 1, · · · , D),
and a core tensor C ∈ R
R1×···×RD *such that*

$\sup_{\mathbf{x}\in\mathcal{X}}\left|f(\mathbf{x})-\mathcal{C}\times_{1}f_{\Theta_{1}}(x_{1})\times_{2}\cdots\times_{D}f_{\Theta_{D}}(x_{D})\right|<\epsilon$,
*viarate MLPs $f_{\Theta_{\Lambda}}:\mathcal{X}_d\to\mathbb{R}^R$*
$(d=1,\,\cdot\,\cdot\,)$
$${D}\rangle_{\mathrm{s}}$$
where ×d : (R
R1*×···×*RD × R
Rd ) → R
R1×···×Rd−1×Rd+1×···×RD denotes the mode-d specific product between a tensor and a vector.

The theorem states that any continuous multivariate function on compact sets can be well approximated by either the CP, TT, or Tucker SepNNs. The detailed proof, which is based on the combination of Stone-Weierstrass theorem (Fedorova, 2002) and universal approximation theorem (Leshno et al., 1993), is placed in Appendix Section A.5. We go through the proof sketch as follows. First, taking the CP SepNN (2) as an example, we consider the associated separable function class:

$$\mathcal{A}=\Big{\{}g:\mathcal{X}\to\mathbb{R}:g(x_{1},\cdots,x_{N})=\sum_{r=1}^{N}\left(g_{1}(x_{1})\right)_{r}\left(g_{2}(x_{2})\right)_{r}\cdots\left(g_{D}(x_{D})\right)_{r},R\in\mathbb{N},\left(g_{d}(x_{d})\right)_{r}\in C(\mathcal{X}_{d})\Big{\}}.$$

The function class A consists of all separable functions that can be expressed in the CP form, using the linear combination of vector-valued continuous univariate functions gd(xd) : Xd → R
R defined on compact sets Xd. By slightly extending the classical universal approximation theorem (Leshno et al., 1993) to vector-valued functions, we show that each gd(xd) can be approximated by an MLP with non-polynomial activation functions. Hence, for any function in A, there exists a SepNN
that approximates it up to arbitrary precision—a fact that can be formalized by bounding the total approximation error via the Cauchy-Schwarz inequality across the errors of the factor MLPs.

It therefore remains to show that A is dense in C(X ), the space of continuous functions over X . This would imply that any continuous multivariate function can be approximated arbitrarily well by an element of A, and consequently by a SepNN. To establish the density of A, we leverage the Stone- Weierstrass theorem (Fedorova, 2002), which asserts that a function class defined on a compact set X is dense in C(X ) if it: (1) contains the identity function; (2) separates points in X (i.e., for any a 6= b in X , there exists g ∈ A such that g(a) 6= g(b)); and (3) is closed under algebraic operations. We carefully examine that A meets these requirements. By combining this with standard universal approximation results, we conclude with the universal approximation capacity of SepNNs. Related Work. (Cho et al., 2023) provided the approximation theory for the bivariate SepNN (1). Their proof is based on the orthogonal basis functions construction for the tensor product function space. Compared to this prior art, our approximation theory extends to any multivariate function approximation with D ≥ 2 and more types of SepNNs such as TT and Tucker, and includes the result in (Cho et al., 2023) as a special case when D = 2 for CP SepNN (2). Furthermore, our proof offers a simpler alternative against (Cho et al., 2023) for the D = 2 case of CP SepNN. Another work (Yu et al., 2024) deduced the approximation theory for a separable physical-informed operator network, which is related to the CP SepNN (2). Their proof is based on the separability of the sine activation function using trigonometric angle addition and universal approximation theory. Compared to this work, we characterize a more general class of SepNNs using any non-polynomial activation functions and deduce the approximation error more systematically for various types of SepNNs. Our proof technique is unified and broadly applicable to these structures, such as separable operator network. The SepNN is also closely related to the Kolmogorov-Arnold network (KAN) (Liu et al., 2025), which represents learnable neurons as weighted summations of independent spline bases and serves as a fundamental architecture in scientific applications (Li et al., 2025a). It is a promising direction to extend the suggested proof technique to KAN approximation and its variants.

## 3 Ntk Regimes And Spectral Bias Characterization

NTK Regimes. While the SepNN is demonstrated to be a universal approximator for multivariate function representation, its training dynamics remain poorly understood even for classical learning problems with gradient descent. We borrow the theoretical tools of NTK (Jacot et al., 2018; Arora et al., 2019a) and aim to characterize the training process of SepNNs via associated kernel regression. Our main contributions are as follows. First, under the asymptotic regime of infinite network width and infinite rank, we prove that the NTK of a SepNN—which can be expressed as the summation over factor MLPs' NTKs (Lemma 1)—converges to a deterministic kernel (Theorem 2), analogous to known results for standard MLPs (Jacot et al., 2018). This result allows us to char-

![4_image_0.png](4_image_0.png)

acterize the convergence rate of wide SepNNs and to identify spectral bias related to the eigenvalue distribution of the NTK matrix1. Lemma 1 (NTK of CP SepNN). Let the multivariate CP SepNN be defined as fΘ(x1, · · · , xD) =
√
1 R
PR
r=1 (fΘ1(x1))r(fΘ2(x2))r*· · ·*(fΘD (xD))r: R
D → R*, where each* fΘd: R → R
R *is a* parametric MLP with parameters Θd*, and* Θ = (Θ1, · · · , ΘD) is the collection of all parameters.

Then the NTK of fΘ*, defined as* KΘ(x, x 0) := h∇ΘfΘ(x), ∇ΘfΘ(x 0)i for x = (x1, · · · , xD) and x 0 = (x 01
, · · · , x0D)*, is given by*

$$K_{\Theta}(\mathbf{x},\mathbf{x}^{\prime})=\frac{1}{R}\sum_{d=1}^{D}\mathbf{a}_{d}(\mathbf{x})^{\top}K_{\Theta_{d}}(x_{d},x_{d}^{\prime})\mathbf{a}_{d}(\mathbf{x}^{\prime}),$$
$$\quad(4)$$

where KΘd
(xd, x0d
) ∈ R
R×R is the NTK matrix of the d-th factor MLP fΘd *with elements*
(KΘd(xd, x0d))r,s =∇Θd(fΘd(xd))r, ∇Θd(fΘd(x 0 d))s
, and ad(x) *is a vector defined by* ad(x) =  Qd06=d fΘd0 (xd0 )1
,Qd06=d fΘd0 (xd0 )2
, *· · ·* ,Qd06=d fΘd0 (xd0 )R
>∈ R
R.

Theorem 2 (Deterministic NTK under infinite width and infinite rank). Let the CP SepNN be defined as fΘ(x1, · · · , xD) = √
1 R
PR
r=1 (fΘ1(x1))r
(fΘ2(x2))r
· · ·(fΘD (xD))r
: R
D → R, where each factor MLP fΘd: R → R
R *has the architecture* fΘd(xd) = √
1 W 
W2,dσ(W1,dxd+bd)*, with* W1,d ∈
RW×1, bd ∈ RW , W2,d ∈ R
R×W , and σ a differentiable activation function with derivative σ˙ *. Let* each element of W1,d, bd,W2,d be independently initialized by N (0, 1)*. Then, as both the width* W → ∞ and the rank R → ∞, the NTK of fΘ *converges almost surely to a deterministic kernel*

$$K_{\Theta}(\mathbf{x},\mathbf{x}^{\prime})\stackrel{\mathrm{a.s.}}{\longrightarrow}\sum_{d=1}^{D}k(x_{d},x_{d}^{\prime})\prod_{d^{\prime}\neq d}c_{d^{\prime}}(x_{d^{\prime}},x_{d^{\prime}}^{\prime}),$$

where cd(xd, x0d) = Ew,b∼N(0,1) (σ(wxd + b)σ(wx0d + b)) and k(xd, x0d) = cd(xd, x0d) + Ew,b∼N(0,1) ( ˙σ(wxd + b) ˙σ(wx0d + b)(xdx 0 d + 1)).

Remark 1. The proof of Lemma 1 and Theorem 2 are placed in Appendix Sections A.6-A.7. While we consider the two-layer MLP in Theorem 2, it is straightforward to extend to multi-layer MLPs or other network structures by utilizing the corresponding NTK formulations (Arora et al., 2019b). Remark 2. *Using similar arguments to standard NTK analysis (Jacot et al., 2018; Arora et al.,*
2019a), it is easy to show that the deterministic NTK of SepNN also stays fixed (in terms of a small error) during training when W, R → ∞ *(under square loss function and infinitely-small learning* rate). This can be achieved by bounding the difference of NTK at two training time points using O( √
1 R
) and O( √
1 W
) *related to the small movement of each weight. Such asymptotic analysis of* NTK during training is formally analyzed in Appendix Section A.4.

Spectral Bias Characterization. Consider training pairs {xi, yi}
n i=1 and square loss minimization:
minΘ
1 2 Pn i=1(fΘ(xi) − yi)
2 with fΘ a CP SepNN. Using the same argument as classical NTK
analysis (Jacot et al., 2018; Arora et al., 2019a), under infinitely small learning rate, the dynamics of network predictions u(t) = (fΘ(t)(x1), · · · , fΘ(t)(xn))> ∈ R
n under standard gradient descent optimizer would follow du(t)
dt = −K(t)(u(t) − y), where Θ(t) and K(t) ∈ R
n×n respectively denote the weights and NTK matrix of the SepNN at training time t over training samples {xi} (i.e., (K(t))i,j = KΘ(xi, xj )), and y = (y1, · · · , yn)
>. If the NTK of the SepNN stays fixed during training (as is expected for SepNNs with sufficiently large width and rank), the dynamics become du(t)
dt = −K(u(t) − y) where K denotes the fixed NTK matrix. Denote the eigenvalue decomposition of K as K =Pn i=1 λiviv
>
i, where λ1 *≥ · · · ≥* λn ≥ 0 are eigenvalue and v1, *· · ·* , vn are orthogonal eigenvectors. Multiplying both sides by vi, we obtain d dt v
>
i u(t)=
−v
>
i K(u(t) − y) = −λi v
>
i
(u(t) − y). This differential equation has an analytical solution:

$\mathbf{v}_{i}^{\top}\left(\mathbf{u}(t)-\mathbf{y}\right)=\exp\left(-\lambda_{i}t\right)\left(\mathbf{v}_{i}^{\top}\left(\mathbf{u}(0)-\mathbf{y}\right)\right).$
$$({\boldsymbol{\boldsymbol{\hat{s}}}})$$
. (5)
Hence, the training error (u(t)−y) and its convergence rate can be characterized by the eigenvalues of the NTK matrix. Note that {vi} form an orthogonal basis, hence (u(t) − y) = Pn i=1 v
>
i
(u(t) −
y). Each term v
>
i(u(t) − y) decays exponentially due to the exp (−λit) term in (5), driving the total training error (u(t) − y) converging to zero as well. The components of labels y that project onto the eigenvectors vi with larger eigenvalue λi converge faster than those correspond to smaller eigenvalues due to the exp (−λit) term, a property known as the spectral bias of neural networks trained with gradient descent (Geifman et al., 2024; Shi et al., 2025; 2024). This indicates that convergence is slower along directions corresponding to smaller eigenvalues, often requiring more training steps if the condition number of NTK matrix is large. The NTK-based convergence rate applies to SepNNs as well, hence SepNNs also exhibit inherent spectral bias due to the uneven eigenvalue distribution of the NTK matrix (Fig. 1(d)). In Section 4, we introduce an efficient SepPGD method to alleviate spectral bias in SepNNs by adjusting the eigenvalue distribution.

Random NTK Under Fixed Rank. In practice, the rank R of SepNNs is often chosen to be smaller compared to network width to promote low-dimensional representations and better generalization (Liang et al., 2022; Luo et al., 2024). Therefore, we further consider the fixed rank and infinite width asymptotic regime, and prove that the NTK converges to a stochastic kernel defined by Gaussian processes associated with the covariance of factor MLPs. This indicates that infinite rank R is necessary to obtain a deterministic NTK by applying the large number law to vanish the covariance.

Corollary 1 (Random NTK under infinite width and fixed rank). *Let the CP SepNN be defined as* fΘ(x1, · · · , xD) = √
1 R
PR
r=1 
(fΘ1
(x1))r
(fΘ2
(x2))r
· · ·(fΘD (xD))r
: R
D → R and the assumptions in Theorem 2 hold. Then, for a fixed rank R, as the width of the factor MLP W → ∞, the NTK of the SepNN fΘ *converges in distribution to a stochastic kernel*

$$K_{\Theta}(\mathbf{x},\mathbf{x}^{\prime})\stackrel{d}{\longrightarrow}\sum_{d=1}^{D}k(x_{d},x_{d}^{\prime})V_{d}(\mathbf{x},\mathbf{x}^{\prime}),$$

where Vd(x, x 0) = 1R
PR
r=1 Qd06=d fΘd0 (xd0 )r fΘd0 (x 0d0 )r
, in which each factor fΘd0 (xd0 )r is a Gaussian process with covariance E(fΘd0 (xd0 )r fΘd0 (x 0d0 )r
) = cd(xd0 , x0d0 ).

Remark 3. *Under the fixed rank condition, the training dynamic can not be characterized uniformly* using a fixed NTK matrix as in (5) due to the randomness. However, the random NTK can at least characterize the training dynamic within a small range of training time around t by using the potential stochastic differential equation and probability bound, which are promising future directions. We also empirically find that even with small rank, the proposed SepPGD method is effective in accelerating convergence and alleviating spectral bias in SepNNs (Appendix Table 3).

All NTK theoretical results are empirically validated in Fig. 1. Fig. 1(a) shows that under a fixed rank R, the NTK at initialization does not converge to a fixed kernel and holds randomness even with larger network width (Corollary 1). Fig. 1(b) shows that the NTK at initialization tends to converge towards a deterministic kernel with joint increase of network width and decomposition rank (Theorem 2). Fig. 1(c) shows that the NTK tends to stay fixed during training with joint increase of network width and decomposition rank (Remark 2). Finally, Fig. 1(d) illustrates the spectral bias that the eigenvalues of the SepNN's NTK matrix decay rapidly, resulting in slower convergence within components that project onto eigenvectors of these smaller eigenvalues. Another positive property of SepNN is that its NTK matrix can be computed in parallel over multiple inputs, resulting in greater efficiency compared to computing the NTK matrices of other neural architectures (Novak et al., 2022; Mohamadi et al., 2023). In fact, the SepNN's NTK (12) admits an elegant form that allows the NTK matrix over grid inputs to be expressed as a Kronecker product of smaller NTK matrices (detailed in Appendix Section A.3), thereby improving efficiency.

## 4 Efficient Separable Preconditioned Gradient Descent

Prior Arts. The NTK-based preconditioning methods are proposed in recent works (Geifman et al., 2024; Shi et al., 2025), which adjust the eigenvalue distribution of the NTK matrix, thereby modulating the convergence rate. Especially, consider a D-dimensional data grid and n D training labels sorted in a vector y ∈ R
n D, and the vector of network predictions fΘ(X) ∈ R
n D, where X ∈ R
D×n Dare batched inputs. The training residual is r = fΘ(X) − y and the square loss is krk 2 `2
. Then, a gradient descent iteration with learning rate η is given by Θ ← Θ − η∇ΘfΘ(X)
>r.

According to (5), the convergence rate is related to the spectrum of NTK matrix, denoted by K here. (Geifman et al., 2024) proposed the following PGD with a preconditioning matrix S: Θ ← Θ − η∇ΘfΘ(X)
>Sr. In this way, the convergence rate is related to the eigenvalues of KS.

Hence, the preconditioner S ∈ R
n D×n Dis constructed to modulate the spectrum of KS so as to improve convergence (Geifman et al., 2024). For n D training samples, this method has complexity O(n D) by applying S. Later, (Shi et al., 2025) leveraged this method to alleviate the spectral bias of INRs. They reduced the complexity of NTK matrix application by sampling a mini-batch, which enjoys O(n D/p) complexity with p > 1 denoting the number of mini-batches. Their preconditioning method can be expressed as Θ ← Θ − η∇ΘfΘ(Xi)
>Siri, where Xi,Si, ri are sampled mini-batches. Different from these methods, our SepPGD applies smaller preconditioners separately for factor MLPs, further reducing computational complexity. SepPGD and Properties. We first elaborate the training configuration of SepNNs for square loss optimization on grid points2. SepNNs exhibit efficiency benefits when applied to separable grid points (Liang et al., 2022; Cho et al., 2023). We therefore consider input points situated on a grid xˆ1 × · · · × xˆD = {(x1, · · · , xD) | xd ∈ xˆd, d = 1, · · · , D} ⊂ R
D, where each xˆd is a discrete set (or say vector) of points in R. For simplicity, we assume that each xˆd has cardinality n, meaning each dimension contains n training samples, resulting in a total of n D training samples. Consider a SepNN fΘ(x) = √
1 R
PR
r=1 
(fΘ1(x1))r
· · ·(fΘD (xD))r
: R
D → R. Each factor MLP fΘd
: R →
R 
R is applied in parallel to the input vector xˆd ∈ R
n, yielding a factor matrix fΘd(xˆd) ∈ R
R×n.

The corresponding optimization model is formulated as

$$\operatorname*{min}_{\Theta}{\frac{1}{2}}\sum_{i=1}^{n^{D}}(f_{\Theta}(\mathbf{x}_{i})-y_{i})^{2},\ \mathbf{x}_{i}\in{\hat{\mathbf{x}}}_{1}\times\cdots\times{\hat{\mathbf{x}}}_{D}.$$
$$(6)$$
2, xi ∈ xˆ1 *× · · · ×* xˆD. (6)
Owing to the grid structure of the training data, the labels {yi}
n D
i=1 can be naturally reshaped into a D-th order tensor Y ∈ R
n*×···×*n. Let R := (ZΘ − Y) ∈ R
n*×···×*n denote the residual tensor during training, where (ZΘ)i1,··· ,iD = fΘ((xˆ1)i1
, *· · ·* ,(xˆD)iD ) is the (i1, · · · , iD)-th SepNN output.

The motivation behind SepPGD is to perform PGD for each factor MLP separately. We first compute D factor preconditioning matrices {Sd}
D
d=1 for the D factor MLPs {fΘd}
D
d=1. This involves calculating a pseudo NTK matrix KΘd ∈ R
n×n for each fΘdon the corresponding input data xˆd ∈ R
n using sum-of-logits (Mohamadi et al., 2023), followed by eigenvalue modulation as described in
(Geifman et al., 2024; Shi et al., 2025). Specifically, we calculate Sd = I −Pk i=1(1 −
g(λi)
λi)viv
>
i
,
where {λi, vi} are eigenvalues and eigenvectors of KΘd. We set the modulation function g(λi) =
λk for i ≤ k and g(λi) = λi for *i > k* by following (Shi et al., 2025), which makes the eigenvalues of Sd more evenly distributed for the first k eigenvalues. Furthermore, denote ⊕ the concatenation of vectors. We are now ready to present the proposed SepPGD algorithm.

Definition 1 (Separable PGD). *Consider the optimization problem (6) and the SepNN* fΘ(x) =
√
1 R
PR
r=1 (fΘ1(x1))r
· · ·(fΘD (xD))r
: R
D → R. Given D symmetric factor preconditioning matrices {Sd ∈ R
n×n}
D
d=1 *for factor MLPs* {fΘd}
D
d=1*, the SepPGD iteration is given by*

$J\;d=1\;J\;d+\;J\;d\;d$ 2. 
$$\Theta\leftarrow\Theta-\eta\bigoplus_{d=1}^{D}(\underbrace{\nabla_{\Theta_{d}}\langle f_{\Theta_{d}}(\hat{\mathbf{x}}_{d}),\mathbf{M}_{d}\rangle}_{\mathrm{gradient~of~factor~MLP}}),\tag{1}$$
_where $M_{d}$ is the mode-d specific preconditioner defined by 
$\mathbf{M}_{d}$ is the induced projective precomment output by_  $$\mathbf{M}_{d}=\left(\bigoplus_{\begin{subarray}{c}\bigoplus\\ \mathbf{r}=1\text{d}\neq d\end{subarray}}^{n}f_{\mathbf{\Theta}_{d}}(\mathbf{\hat{x}}_{d})_{r_{i}}\right)\underbrace{\left(\bigoplus_{d=1}^{D}(\mathbb{R}\times\mathbf{A}\mathbf{S}_{d})\right)}_{n^{D-1}\times n}\in\mathbb{R}^{R\times n},\ d=1,\cdots,D,\tag{8}$$
$$\left(7\right)$$
where the subscript fΘd0 (xˆd0 )r,: refers to the r*-th row of the factor matrix* fΘd0 (xˆd0 ) ∈ R
R×n N
. The d06=d refers to the outer product that calculates between D − 1 *vectors* {fΘd0 (xˆd0 )r,: ∈ R
1×n}
and returns a single long vector of size 1 × n D−1, and LR
r=1 *denotes the concatenation between* R
vectors and returns an R × n D−1 *matrix. The* unfoldd : R
n1*×···×*nD → R
Qd06=d nd0×nd denotes the unfolding operator from a tensor to a matrix, and ×d denotes the mode-d tensor-matrix product, a convention in tensor decomposition literature (Kolda & Bader, 2009; Luo et al., 2024). Remark 4 (Computational complexity). The computational complexity comparison of different preconditioning methods is given in Table 1. In terms of applying the preconditioner, SepPGD scales as O(nD) by multiplying D n-by-n preconditioning matrices {Md}, while standard NTK-based method (Geifman et al., 2024) scales as O(n D) *by multiplying an* n D*-by-*n D preconditioning matrix S. Moreover, the preconditioner construction stage in SepPGD is also more efficient. Specifically, SepPGD constructs the preconditioner by calculating the NTK matrix and performing eigenvalue decomposition for D *factor NTK matrices* {KΘd
}, each of size n × n. In contrast, classical NTK-based PGD (Geifman et al., 2024) requires the same operations on a single large NTK matrix of size n D × n D*. Therefore, the preconditioner construction complexity for SepPGD scales as* O(D(n 3 + n 2P)) (note that NTK matrix calculation consumes O(n 2P) and eigenvalue decomposition consumes O(n 3))
3, where P is the number of network parameters, while the classical NTK-
based PGD method scales as O(n 3D + n 2DP). We therefore see that SepPGD is more efficient in both preconditioner construction and application. Table 1: Computational complexity comparison (in terms of applying the preconditioner) of several preconditioning methods for n D training samples using an over-parameterized SepNN fΘ : R
D →
R with P learnable parameters (n D < P). Here, r denotes training residual, H ∈ R
P ×P denotes the Hessian matrix, S ∈ R
n D×n Ddenotes the NTK-based preconditioning matrix, Si ∈ R
nD
p × nD
p denotes the mini-batch version of the NTK preconditioning matrix, and Md ∈ R
R×n denotes the proposed mode-d specific preconditioner defined in (8).

| Preconditioning method                           | Gradient formulation        | Complexity       |
|--------------------------------------------------|-----------------------------|------------------|
| Hessian-based methods                            | H−1∇ΘfΘ(X) >r               | O(P)             |
| Modified NTK spectrum (Geifman et al., 2024)     | ∇ΘfΘ(X) >Sr                 | O(n D) (n D < P) |
| Inductive gradient adjustment (Shi et al., 2025) | ∇ΘfΘ(Xi) >Siri (mini-batch) | O(n D/p) (p > 1) |
| Separable PGD (Ours)                             | LD d=1 ∇Θd hfΘd (xˆd),Mdi   | O(nD)            |

![8_image_0.png](8_image_0.png)

Lemma 2. *Let the SepNN be* fΘ(x) = fΘ1
(x1)
>fΘ2
(x2) : R
2 → R and the suppositions in Definition 1 hold. Then the SepPGD iteration in (7) is equivalent to the classical NTK-based PGD
as Θ ← Θ−η∇ΘfΘ(X)
>Sr˜ *, where* X ∈ R
2×n 2*are batched inputs,* r = vec(R>) is the vectored training residual, and S˜ = (S1 ⊗ In + In ⊗ S2) ∈ R
n 2×n 2is constructed by Kronecker product.

Specifically, this implies [∇Θ1hfΘ1(xˆ1),M1i, ∇Θ2hfΘ2(xˆ2),M2i] = ∇ΘfΘ(X)
>Sr˜ .

(a) f1*(x,y)* (b) f2*(x,y)* (a) f1*(x,y)* (b) f2*(x,y)*
The proof is provided in Appendix Section A.9. Lemma 2 establishes the connection between the proposed SepPGD method (7) and the classical NTK-based PGD (Geifman et al., 2024; Shi et al., 2025) under an appropriate choice of the large preconditioner S˜. Although the two PGD
methods are equivalent in this setting, SepPGD is computationally more efficient. This efficiency comes from the parallel computation of factor gradients. Specifically, the total gradient
∇ΘfΘ(X)
>Sr˜ can be written in the form of outer products like (C> ⊗ A)vec(B) (see Section A.9), whereas the factor gradients {∇Θd hfΘd(xˆd),Mdi}D
d=1 in SepPGD are computed via matrix products like vec(ABC) in the O(n) dimensional space. This is more efficient than evaluating the matrix product (C> ⊗ A)vec(B) in the O(n 2) dimension. The key property we are using here is the equivalence between the Kronecker product expression and the vectorized matrix product, i.e., (C> ⊗ A*)vec(*B) = vec(ABC). This allows us to decompose the large preconditioner S˜ ∈ R
n 2×n 2into smaller preconditioners Md ∈ R
n×n and perform the SepPGD in an efficient manner. Lemma 2 can also be extended to non-grid inputs. Especially, if we construct the preconditioner as S˜ =Pd Sd, then the SepPGD gradient ∇ΘdhfΘd((X)d,:),Mdi for some non-grid inputs X ∈ R
D×n (n non-grid points in D-dimensional space) is equivalent to the classical NTK-based PGD update ∇ΘfΘ(X)
TSr˜ under element-wise evaluation without Kronecker product structure.

Here, Md = einsum(d06=dfΘ0d
((X)d0,:),Pd Sdr; (R, n) × (n) → (*R, n*)) is the d-th preconditioner constructed by Einstein product, where  denotes element-wise product. We use this formulation to test SepPGD for non-grid inputs; see Section A.2. Given the equivalence between SepPGD and NTK-based PGD (Geifman et al., 2024), it therefore remains to show that the constructed preconditioner S˜ = (S1 ⊗ In + In ⊗ S2) is indeed effective in adjusting the eigenvalue distribution of KS˜ to accelerate convergence. This can possibly be verified, because the eigenvalue of a Kronecker product matrix S1⊗In is the product of eigenvalues of S1 and In. Therefore, S˜ would have better spectrum (i.e., smaller condition number) than K˜ =
(KΘ1 ⊗ In + In ⊗ KΘ2
) since Sd has better spectrum than KΘd
. Suppose that K˜ is close to the true NTK matrix K (which can be verified using the NTK matrix formulation in Lemma 3). We can ultimately show that KS˜ has better spectrum than K. Therefore, the proposed SepPGD could provably and efficiently adjust the spectrum of the SepNN's NTK matrix during training, effectively alleviating spectral bias. In practice, SepPGD allows the preconditioner {Md} to be efficiently updated every ten iterations, which is computationally expensive in previous methods. It is believed that the result in Lemma 2 (and the analysis following) can be readily extended to multivariate cases D > 2. Also, based on the convergence result of NTK-based preconditioning algorithm (Geifman et al., 2024), we can also deduce the convergence and solution consistency (w.r.t. standard gradient descent) of our SepPGD algorithm by using the equivalence between SepPGD and the corresponding kernel ridge regression with representer theorem. This is left for future research.

(a) *Peppers* (b) *Plane* (a) *Peppers* (b) *Plane*

## 5 Numerical Experiments

The numerical results are presented to verify the effectiveness of SepPGD for improving convergence of SepNNs. Examples include kernel ridge regression (KRR), image and surface representation using INRs, and PINNs. Detailed experimental settings are placed in Appendix Section A.12.

![9_image_0.png](9_image_0.png)

KRR. Following (Geifman et al., 2024), we perform KRR by using the gradients of neural network w.r.t. parameters as the feature function of kernel (see Appendix Section A.12 for detailed formulation and settings). We test both MLP and CP SepNN, and compare SepPGD with the classical NTK-based PGD, the modified spectrum kernel (MSK) (Geifman et al., 2024; Shi et al., 2025). Following (Geifman et al., 2024), we consider both noisy (standard deviation 0.01) and noiseless cases. The convergence behavior, measured by testing MSE during training, is shown in Fig. 2(a). Because the efficiency advantage of SepNN and SepPGD comes from the lower complexity in an iteration, we plot the convergence curve w.r.t. execution time rather than iteration number. In noiseless case, SepPGD achieves the fastest convergence. In the presence of noise, SepPGD remains robust, while MSK has slower convergence. SepPGD improves SepNN to a large margin as shown in Fig. 2. Image Representation and Recovery. We leverage INR for image representation and recovery (inpainting) by following the settings in (Sitzmann et al., 2020; Shi et al., 2025). Convergence curves and representation visual examples are shown in Fig. 2(b) and Fig. 3's left. SepPGD effectively improves the convergence of SepNN by alleviating spectral bias and better capturing image fine details. Moreover, SepPGD accelerates the convergence without affecting the model's generalization (in most cases improving generalization); see image inpainting results in Appendix Fig. 10. Surface Representation. We perform surface representation by representing the volumetric occupancy grids of a 3D surface using INRs (Shi et al., 2025). The results (under the same iteration number) with intersection over union (IoU) are shown in Fig. 3's right. SepPGD effectively improves the ability of SepNN to capture surface textures and details by alleviating spectral bias during training. PINNs. We use the separable PINN (Cho et al., 2023) (CP SepNN) and perform the tests on 3D diffusion, Klein-Gordon, and Helmholtz equations using grid samples. Convergence curves using testing MSE vs. time and visual examples (under the same iteration) are shown in Fig. 4. More results are shown in Appendix Figs. 13-14. The separable PINN (Cho et al., 2023) is more efficient than classical PINN, and SepPGD further enhances the convergence speed of separable PINN.

(a) Kernel ridge regression (b) Image representation (a) Kernel ridge regression (b) Image representation
(a) f1*(x,y)* (b) f2*(x,y)* (a) f1(x,y) (b) f2*(x,y)*
(a) *Peppers (b) Plane* (a) *Peppers (b) Plane*

## 6 Conclusions And Discussions

We established the universal approximation theory for SepNNs, deduced their NTK regimes, and proposed an efficient separable PGD to alleviate spectral bias of SepNNs. The algorithm was shown to be effective in various applications such as image & surface representation and PINNs. We further elaborate on the potential impact of our proposed theory and method. First, SepNNs have been attracting growing attention in various applications due to their efficient structure, such as INRs and PINNs. Numerical experiments demonstrate that SepPGD can effectively speed up convergence in applications involving INRs and PINNs (e.g., in image and surface representation and numerical PDEs). Therefore, we believe that our theory and the corresponding algorithm have the potential to benefit these fields and address related challenges. Moreover, SepNNs are also increasingly being adopted across diverse scientific domains by leveraging their efficient structure and interpretability (Cho et al., 2023; Song et al., 2023; Chen et al., 2025). Therefore, understanding the theoretical approximation ability, training behavior (e.g., NTK regime), and addressing the potential optimization challenge of SepNNs are believed to be valuable and important for these practical applications.

## Acknowledgments

This work is supported by the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (JYB2025XDXM101), the NSFC (No. 124B2029, 62476214),
and the Tianyuan Fund for Mathematics of the National Natural Science Foundation of China (Grant No. 12426105). We thank the anonymous reviewers for the constructive and valuable comments.

## References

K. Ahlander. Einstein summation for multidimensional arrays. Computers & Mathematics with Applications, 44(8):1007–1017, 2002.

Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In Proceedings of the 36th International Conference on Machine Learning (ICML), volume 97, pp. 322–332, 2019a.

Sanjeev Arora, Simon S. Du, Wei Hu, Zhiyuan Li, Ruslan Salakhutdinov, and Ruosong Wang. On exact computation with an infinitely wide neural net. In Proceedings of the 33rd International Conference on Neural Information Processing Systems (NeurIPS), pp. 8141–8150, 2019b.

Liliana Borcea, Josselin Garnier, Alexander V. Mamonov, and Jorn Zimmerling. When data driven ¨
reduced order modeling meets full waveform inversion. *SIAM Review*, 66(3):501–532, 2024.

Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. TensoRF: Tensorial radiance fields. In *17th European Conference on Computer Vision (ECCV)*, pp. 333–350, 2022.

Ruihua Chen, Bangyu Wu, Meng Li, and Yisi Luo. Full-waveform inversion with velocity model low-rank implicit neural representation. *IEEE Transactions on Geoscience and Remote Sensing*, 63:1–16, 2025. doi: 10.1109/TGRS.2025.3594184.

Junwoo Cho, Seungtae Nam, Hyunmo Yang, Seok-Bae Yun, Youngjoon Hong, and Eunbyung Park.

Separable physics-informed neural networks. In Thirty-seventh Conference on Neural Information Processing Systems, pp. 23761–23788, 2023.

Sven Dummer, Nicola Strisciuglio, and Christoph Brune. Rda-inr: Riemannian diffeomorphic autoencoding via implicit neural representations. *SIAM Journal on Imaging Sciences*, 17(4):2302– 2330, 2024.

V. P. Fedorova. *The Stone–Weierstrass Theorem and Spaces of Measures*, volume 72, pp. 417–427.

Springer, 2002.

Amnon Geifman, Daniel Barzilai, Ronen Basri, and Meirav Galun. Controlling the inductive bias of wide neural networks by modifying the kernel's spectrum. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.

Alex Gorodetsky, Sertac Karaman, and Youssef Marzouk. A continuous analogue of the tensor-train decomposition. *Computer Methods in Applied Mechanics and Engineering*, 347:59–84, 2019.

Yicheng Li Qian Lin Jun S. Liu Haobo Zhang, Jianfa Lai. Towards a statistical understanding of neural networks: Beyond the neural tangent kernel theories. *arXiv:2412.18756*, 2024.

Russell L. Herman. Generalized fourier series and function spaces. In An Introduction to Fourier Analysis. Chapman and Hall/CRC, 2016.

Arthur Jacot, Franck Gabriel, and Clement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa- Bianchi, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 31, 2018.

Nikos Kargas and Nicholas D. Sidiropoulos. Supervised learning and canonical decomposition of multivariate functions. *IEEE Transactions on Signal Processing*, 69:1097–1107, 2021.

Patrick Kidger and Terry Lyons. Universal Approximation with Deep Narrow Networks. In Proceedings of Thirty Third Conference on Learning Theory, volume 125, pp. 2306–2327, 2020.

Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

Tamara G. Kolda and Brett W. Bader. Tensor decompositions and applications. *SIAM Review*, 51
(3):455–500, 2009.

Jaehoon Lee, Yasaman Bahri, Roman Novak, Sam Schoenholz, Jeffrey Pennington, and Jascha Sohldickstein. Deep neural networks as gaussian processes. In *International Conference on Learning* Representations (ICLR), 2018.

S. Lee, G. Wolberg, and S.Y. Shin. Scattered data interpolation with multilevel b-splines. IEEE
Transactions on Visualization and Computer Graphics, 3(3):228–244, 1997. doi: 10.1109/2945. 620490.

Moshe Leshno, Vladimir Ya. Lin, Allan Pinkus, and Shimon Schocken. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. *Neural Networks*, 6(6):861–867, 1993.

Longlong Li, Yipeng Zhang, Guanghui Wang, and Kelin Xia. Kolmogorov-arnold graph neural networks for molecular property prediction. *Nature Machine Intelligence*, 7:1346–1354, 2025a.

Yanyi Li, Xi Zhang, Yisi Luo, and Deyu Meng. Deep rank-one tensor functional factorization for multi-dimensional data recovery. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 18539–18547, 2025b.

Yicheng Li, Zixiong Yu, Guhan Chen, and Qian Lin. On the eigenvalue decay rates of a class of neural-network related kernel functions defined on general domains. Journal of Machine Learning Research, 25(1):3977 - 4023, 2024.

Ruofan Liang, Hongyi Sun, and Nandita Vijaykumar. CoordX: Accelerating implicit neural representation with a split MLP architecture. In International Conference on Learning Representations (ICLR), 2022.

Chaoyue Liu, Libin Zhu, and Mikhail Belkin. On the linearity of large non-linear models: when and why the tangent kernel is constant. In Proceedings of the 34th International Conference on Neural Information Processing Systems, pp. 15954 - 15964, 2020.

Xingyi Liu and Keshab K. Parhi. Tensor decomposition for model reduction in neural networks: A
review. *IEEE Circuits and Systems Magazine*, 23(2):8–28, 2023.

Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljacic, Thomas Y. Hou, and Max Tegmark. KAN: Kolmogorov-Arnold networks. In *The Thirteenth* International Conference on Learning Representations, 2025.

Yisi Luo, Xile Zhao, Zhemin Li, Michael K. Ng, and Deyu Meng. Low-rank tensor function representation for multi-dimensional data recovery. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5):3351–3369, 2024.

Mohamad Amin Mohamadi, Wonho Bae, and Danica J. Sutherland. A fast, well-founded approximation to the empirical neural tangent kernel. In Proceedings of the 40th International Conference on Machine Learning, pp. 25061–25081, 2023.

Roman Novak, Jascha Sohl-Dickstein, and Samuel S Schoenholz. Fast finite width neural tangent kernel. In *Proceedings of the 39th International Conference on Machine Learning*, volume 162, pp. 17018–17044, 2022.

Allan Pinkus. Approximation theory of the MLP model in neural networks. *Acta Numerica*, 8:
143–195, 1999. doi: 10.1017/S0962492900002919.

Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In *Proceedings* of the 21st International Conference on Neural Information Processing Systems, pp. 1177–1184, 2007.