# Faster Gradient Methods For Highly-Smooth Stochastic Bilevel Optimization

Lesi Chen and Junru Li Tsinghua University & Shanghai Qizhi Institude, China
{chenlc23,jr-li24}@mails.tsinghua.edu.cn

## El Mahdi Chayti

Ecole Polytechnique F ´ ed´ erale de Lausanne (EPFL), Switzerland ´
el-mahdi.chayti@epfl.ch Jingzhao Zhang †
Tsinghua University & Shanghai Qizhi Institude,China jingzhaoz@mail.tsinghua.edu.cn

## Abstract

This paper studies the complexity of finding an ϵ-stationary point for stochastic bilevel optimization when the upper-level problem is nonconvex and the lowerlevel problem is strongly convex. Recent work proposed the first-order method, F
2SA, achieving the O˜(ϵ
−6) upper complexity bound for first-order smooth problems. This is slower than the optimal Ω(ϵ
−4) complexity lower bound in its singlelevel counterpart. In this work, we show that faster rates are achievable for higherorder smooth problems. We first reformulate F2SA as approximating the hypergradient with a forward difference. Based on this observation, we propose a class of methods F2SA-p that uses pth-order finite difference for hyper-gradient approximation and improves the upper bound to O˜(pϵ−4−2/p) for pth-order smooth problems. Finally, we demonstrate that the Ω(ϵ
−4) lower bound also holds for stochastic bilevel problems when the high-order smoothness holds for the lowerlevel variable, indicating that the upper bound of F2SA-p is nearly optimal in the region p = Ω(log ϵ
−1/ log log ϵ
−1).

## 1 Introduction

Many machine learning problems, such as meta-learning (Rajeswaran et al., 2019), hyper-parameter tuning (Bao et al., 2021; Franceschi et al., 2018; Mackay et al., 2019), adversarial training (Goodfellow et al., 2020), and reinforcement learning (Yang et al., 2019; Hong et al., 2023; Zeng et al., 2024; Zeng & Doan, 2024) can be abstracted as solving the following bilevel optimization problem:
min x∈Rdx φ(x) = f(x, y
∗(x)), y
∗(x) = arg min y∈R
dy g(x, y), (1)
We call f and g the upper-level and lower-level functions, respectively, and call φ the hyperobjective. In this paper, we consider the most common *nonconvex-strongly-convex* setting where f : R
dx → R is smooth and possibly nonconvex, and g : R
dy → R is smooth jointly in (x, y)
and strongly convex in y. Under the lower-level strong convexity assumption, the implicit function theorem (Dontchev & Rockafellar, 2009) indicates the following closed form of the hyper-gradient
∇φ(x) = ∇xf(x, y
∗(x)) − ∇2xyg(x, y
∗(x))[∇2yyg(x, y
∗(x))]−1∇yf(x, y
∗(x)). (2)
Following the works in nonconvex optimization (Carmon et al., 2020; 2021; Arjevani et al., 2023),
we consider the task of finding an ϵ-stationary point of φ, *i.e.*, a point x ∈ R
dx such that
†The corresponding author.

∥∇φ(x)∥ ≤ ϵ. Motivated by many real machine learning tasks, we study the stochastic setting, where the algorithms only have access to the stochastic derivative estimators of both f and g. The first efficient algorithm BSA Ghadimi & Wang (2018) for solving the stochastic bilevel problem leverages both stochastic gradient and Hessian-vector-product (HVP) oracles to find an ϵ-stationary point of φ(x). Subsequently, Ji et al. (2021) proposed stocBiO by incorporating multiple enhanced designs in BSA to improve the complexity. Yang et al. (2023b) proposes FdeHBO that uses finitedifferences to estimate HVP vectors. However, all these methods require the stochastic Hessian assumption (5) on the lower-level function, which is stronger than the standard SGD assumption.

Kwon et al. (2023) proposed the first fully first-order method F2SA that works under standard SGD
assumptions on both f and g (Assumption 2.1). The main idea is to solve the following penalty problem (Liu et al., 2022; 2023; Shen & Chen, 2023; Shen et al., 2025b; Lu & Mei, 2024; 2026):

$$\min_{\mathbf{x}\in\mathbb{R}^{d_{x}},y\in\mathbb{R}^{d_{y}}}f(\mathbf{x},\mathbf{y})+\lambda\left(g(\mathbf{x},\mathbf{y})-\min_{\mathbf{z}\in\mathbb{R}^{d_{y}}}g(\mathbf{x},\mathbf{z})\right),$$  the sufficiently large such that $\lambda=\Omega(\epsilon^{-1})$. If we interpret $\lambda$ as the $\mathbf{I}$-norm
$$({\mathfrak{I}})$$

where λ is taken to be sufficiently large such that λ = Ω(ϵ
−1). If we interpret λ as the Lagrangian multiplier, then Problem (3) can be viewed as the Lagrangian function of the constrained optimization minx∈Rdx ,y∈R
dy f(x, y),s.t. g(x, y) ≤ g(x, y
∗(x)). Thanks to Danskin's theorem, the gradient of the Problem (3) only involves gradient information. Therefore, F2SA does not require the stochastic Hessian assumptions (5). More importantly, by directly leveraging gradient oracles instead of more expensive HVP oracles, the F2SA is more efficient in practice (Shen et al., 2025a; Xiao & Chen, 2025; Jiang et al., 2025) and is also the only method that can be scaled to 32B sized large language model (LLM) training (Pan et al., 2024).

Kwon et al. (2023) proved that the F2SA method finds an ϵ-stationary point of φ(x) with O˜(ϵ
−3)
first-order oracle calls in the deterministic case and O˜(ϵ
−7) stochastic first-order oracle (SFO) calls in the stochastic case. Recently, Chen et al. (2025b) showed the two-time-scale stepsize strategy improves the upper complexity bound of F2SA method to O˜(ϵ
−2) in the deterministic case, which is optimal up to logarithmic factors. However, the direct extension of their method in the stochastic case leads to the O˜(ϵ
−6) SFO complexity (Chen et al., 2025b; Kwon et al., 2024a) , which still has a significant gap between the Ω(ϵ
−4) lower bound for SGD (Arjevani et al., 2023). It remains open whether optimal rates for stochastic bilevel problems can be achieved for fully first-order methods.

In this work, we revisit F2SA and interpret it as using forward difference to approximate the hypergradient. Our novel interpretation in turn leads to straightforward algorithm extensions for the F2SA method. Observing that the forward difference used by F2SA only has a first-order error guarantee, a natural idea to improve the error guarantee is to use higher-order finite difference methods. For instance, we know that the central difference has an improved second-order error guarantee. Based on this fact, we can derive the F2SA-2 method that solves the following symmetric penalty problem:

$$\min_{\mathbf{x}\in\mathbb{R}^{d_{\mathbf{x}},\mathbf{y}\in\mathbb{R}^{d_{\mathbf{y}}}}}\frac{1}{2}\left(f(\mathbf{x},\mathbf{y})+\lambda g(\mathbf{x},\mathbf{y})-\min_{\mathbf{z}\in\mathbb{R}^{d_{\mathbf{y}}}}\left(-f(\mathbf{x},\mathbf{z})+\lambda g(\mathbf{x},\mathbf{z})\right)\right).\tag{4}$$  $\mathbf{x}$\(\mathbf{x}  
Compared with Eq. (3), this new penalty problem perturbs the lower-level variables y and z in the opposite direction to better cancel out the approximation errors to Problem (1). The connection between bilevel optimization and finite difference approximations was recently established by Chayti & Jaggi (2024) in the context of meta-learning, but their results were limited to symmetric approximations. We extend their findings beyond meta-learning and to general finite difference approximations, addressing their conjecture about broader applicability. In this work, we show that F
2SA-2 provably improves the SFO complexity of F2SA from O˜(ϵ
−6) to O˜(ϵ
−5) for second-order smooth problems. Moreover, our idea is generalizable for any pth-order smooth problems. It is known in numerical analysis there exists the pth-order central difference that uses p points to construct an estimator to the derivative of a unitary function with pth-order error guarantee, as recalled in Lemma 3.1. Motivated by this fact, we propose the F2SA-p algorithm and show that it enjoys the improved O˜(pϵ−4−2/p) SFO complexity, as formally stated in Theorem 3.1.

Moreover, as formally stated in Theorem 4.1, it is easy to extend the Ω(ϵ
−4) lower bound for SGD
(Arjevani et al., 2023) from single-level optimization to bilevel optimization using a fully separable construction that automatically satisfies all our additional smoothness conditions in Definition 2.2.

It shows that F2SA-p is optimal up to logarithmic factors when p = Ω(log ϵ
−1/ log log ϵ
−1) (see Remark 3.4). We summarize our main results in Table 1 and discuss open problems in the following. Open problems. Our upper bounds improve known results for high-order smooth problems, but our result still has a gap between the Ω(ϵ
−4)lower bound for p = O(log ϵ
−1/ log log ϵ
−1). Recently, Kwon et al. (2024a) obtained some preliminary results towards closing this gap for p = 1, where they showed an Ω(ϵ
−6) lower bound holds under a more adversarial oracle. But it is still open whether their lower bounds can be extended to standard stochastic oracles. Another open problem is the tightness of the condition number dependency, for which the current upper and lower bounds have a gap of Ω(κ 9) as demonstrated in Table 1. Two recent concurrent works (Ji, 2025; Chen
& Zhang, 2025) proposed tighter lower bounds for p = 1: Ji (2025) showed a lower bound of Ω(κ 5/2ϵ
−4) and Chen & Zhang (2025) showed that of Ω(κ 4ϵ
−4). However, it is still open to fully close the gap in condition number dependency for both p = 1 and p ≥ 2.

| Method      | Smoothness     | Reference               | Complexity       |
|-------------|----------------|-------------------------|------------------|
| 2SA         | 1st-order      | (Kwon et al., 2023)     | O˜(poly(κ)ϵ −7 ) |
| F 2SA       | 1st-order      | (Kwon et al., 2024a)    | O˜(poly(κ)ϵ      |
| F           | −6 )           |                         |                  |
| 2SA         | 1st-order      | (Chen et al., 2025b)    | O˜(κ 12ϵ −6 )    |
| F 2SA-p     | 1st-order      | Theorem 3.1             | O˜(pκ9+2/pϵ      |
| F           | −4−2/p)        |                         |                  |
| +           |                |                         |                  |
| Lower Bound | pth-order in y | (Arjevani et al., 2023) | Ω(ϵ −4 )         |

Table 1: The SFO complexity of different methods to find an ϵ-stationary point for pth-order smooth first-order bilevel problems with condition number κ under standard SGD assumptions.

Notations. We use *∥ · ∥* to denote the Euclidean norm for vectors and the spectral norm for matrices and tensors. For any set S and functions *g, h* : S → [0, ∞) we write g = O(h) or h *= Ω(*g)
equivalently if there exists c > 0 such that g(s) ≤ ch(s) for every s ∈ S. We use O˜(·) and Ω( ˜ ·) to hide logarithmic factors in O(·) and Ω(·). We alsouse h1 ≲ h2 to mean h1 = O(h2), h1 ≳ h2 to mean h1 = Ω(h2), and h1 ≍ h2 to mean that both h1 ≲ h2 and h1 ≳ h2 hold. Additional notations for tensors are introduced in Appendix A.

## 2 Preliminaries

The goal of bilevel optimization is to minimize the hyper-objective φ(x), which is in general nonconvex. Since finding a global minimizer of a general nonconvex function requires exponential complexity in the worst case (Nemirovskij & Yudin, 1983, § 1.6), we follow the literature (Carmon et al., 2020; 2021) to consider the task of finding an approximate stationary point.

Definition 2.1. Let φ : R
dx → R *be the hyper-objective defined in Eq. (1). We say a random* variable xˆ ∈ R
dx is an ϵ-hyper-stationary point if E∥∇φ(xˆ)∥ ≤ ϵ.

Next, we introduce the assumptions used in this paper, which ensure the tractability of the above hyper-stationarity. Compared to (Kwon et al., 2023; Chen et al., 2025b), we additionally assume the high-order smoothness in lower-level variable y to achieve further acceleration.

## 2.1 Problem Setup

First of all, we follow the standard assumptions on SGD (Arjevani et al., 2023) to assume that the stochastic gradient estimators satisfy the following assumption.

Assumption 2.1. There exists stochastic gradient estimators F(x, y) and G(x, y) *such that*

$\tau$, $\xi$) = $\nabla f(\mathbf{x},\mathbf{y})$, $\tau$, $\xi$) = $\nabla g(\mathbf{x},\mathbf{y})$. 
EF(x, y; ξ) = ∇f(x, y), E∥F(x, y) − ∇f(x, y)∥
2 ≤ σ 2; EG(x, y; ζ) = ∇g(x, y), E∥G(x, y) − ∇g(x, y)∥
2 ≤ σ 2, where σ > 0 is the variance of the stochastic gradient estimators. We also partition F = (Fx, Fy) and G = (Gx, Gy) such that Fx, Fy, Gx, Gy are estimators to ∇xf, ∇yf, ∇xg, ∇yg*, respectively.*

$$\mathbb{E}\|F(\mathbf{x},\mathbf{y})$$ $$\mathbb{E}\|C(\mathbf{x},\mathbf{y})$$

Second, we assume that the hyper-objective φ(x) = f(x, y
∗(x)) is lower bounded. Otherwise, any algorithm requires infinite time to find a stationary point. Note that the implicit condition infx∈Rdx φ(x) > −∞ below can also be easily implied by a more explicit condition on the lower boundedness of upper-level function infx∈Rdx ,y∈R
dy f(x, y) > −∞.

Assumption 2.2. *The hyper-objective defined in Eq. (1) is lower bounded, and we have* φ(x0) − inf x∈Rdx

$$\varphi(\mathbf{x})\leq\Delta,$$

where ∆ > 0 is the initial suboptimality gap and we assume x0 = 0 *without loss of generality.*
Third, we assume the lower-level function g(x, y) is strongly convex in y. It guarantees the uniqueness of y
∗(x) and the tractability of the bilevel problem. Although not all the problems in applications satisfy the lower-level strong convexity assumption, it is impossible to derive dimension-free upper bounds when the lower-level problem is only convex (Chen et al., 2024, Theorem 3.2). Hence, we follow most existing works to consider strongly convex lower-level problems.

Assumption 2.3. g(x, y) is µ-strongly convex in y*, i.e., for any* y1, y2 ∈ R
dy *, we have*

$\varphi(\boldsymbol{\varepsilon})$
$\text{(a)}-$
g(x, y2) ≥ g(x, y1) + ⟨∇yg(x, y1), y2 − y1⟩ +
µ 2

$$y_{1}-y_{2}\|^{2},$$

where µ > 0 *is the strongly convex parameter.* Fourth, we require the following smoothness assumptions following (Ghadimi & Wang, 2018). According to Eq. (2), these conditions are necessary and sufficient to guarantee the Lipschitz continuity of ∇φ(x), which further ensure the tractability of an approximate stationary point of the nonconvex hyper-objective φ(x) (Zhang et al., 2020; Kornowski & Shamir, 2022). Assumption 2.4. For the upper-lower function f and lower-level function g*, we assume that* 1. f(x, y) is L0*-Lipschitz in* y. 2. ∇f(x, y) and ∇g(x, y) are L1*-Lipschitz jointly in* (x, y).

3. ∇2xyg(x, y) and ∇2yyg(x, y) are L2*-Lipschitz jointly in* (x, y).

We refer to the problem class that jointly satisfies all the above Assumption 2.1, 2.2, 2.3 and 2.4 as first-order smooth bilevel problems, for which (Kwon et al., 2024a; Chen et al., 2025b) showed the F
2SA method achieves the O˜(ϵ
−6) upper complexity bound. In this work, we show an improved bound under the following additional higher-order smoothness assumption on lower-level variable y.

Assumption 2.5 (High order smoothness in y). Given p ∈ N+, we assume that 1. ∂
q
∂yq ∇f(x, y) is Lq+1*-Lipschitz for all* q = 1, · · · , p − 1.

2. ∂
q+1
∂yq+1 ∇g(x, y) is Lq+2-Lipschitz in y for all q = 1, · · · , p − 1.

We refer to problems jointly satisfying all the above assumptions as pth-order smooth bilevel problems, and also formally define their condition numbers as follows.

Definition 2.2 (pth-order smooth bilevel problems). *Given* p ∈ N+, ∆ > 0, L0, L1, · · · , Lp+1 > 0, and µ ≤ L1, we use F
nc-sc(L0, · · · , Lp+1, µ, ∆) *to denote the set of all bilevel instances satisfying* Assumption 2.2, 2.3, 2.4 and 2.5. For this problem class, we define the largest smoothness constant L¯ = max0≤j≤p Lj and condition number κ = L/µ ¯ .

All our above assumptions align with (Chen et al., 2025b) except for the additional Assumption 2.5. A classic example of a highly smooth function is the softmax function (Garg et al., 2021, Lemma 2(3)). Therefore, many hyper-parameter tuning problems for logistic regression are provably highly smooth, such that our theory can be applied. We give two examples from (Pedregosa, 2016): the first one aims to learn the optimal weights of each sample in a corrupted training set, and the second one aims to learn the optimal regularizer of each parameter.

Example 2.1 (Data hyper-cleaning). Let x ∈ R
n parameterize the per-sample weight of a training set with n *samples via* σ(xi) = exp(xi)/Pn i=1 exp(xi) and y ∈ R
d be the parameters of a linear model. Let ℓval *be the logistic loss of the linear model on the validation set and* ℓ itr be the logistic loss on the training sample i*. The problem aims to solve*

$$\operatorname*{min}_{\mathbf{x}\in\mathbb{R}^{n}}\ell_{\mathrm{val}}(\mathbf{y}),\quad{\mathrm{s.t.}}\quad\mathbf{y}\in{\arg\operatorname*{min}_{\mathbf{y}\in\mathbb{R}^{d}}}\sum_{i=1}^{n}\sigma(x_{i})\ell_{\mathrm{tr}}^{i}(\mathbf{y}).$$
$$({\boldsymbol{5}})$$

Example 2.2 (Learn-to-regularize). Let x ∈ R
d *parameterize the regularization matrix via* Wx =
diag(exp(x))*, and* y ∈ R
d *be the parameters of a linear model. Let* ℓval and ℓtr be the logistic loss of the linear model on the validation set and training set, respectively. The problem aims to solve

$$\operatorname*{min}_{\mathbf{x}\in\mathbb{R}^{d}}\ell_{\mathrm{val}}(\mathbf{y}),\quad\mathrm{s.t.}\quad\mathbf{y}\in\arg\operatorname*{min}_{\mathbf{y}\in\mathbb{R}^{d}}\ell_{\mathrm{tr}}(\mathbf{y})+\mathbf{y}^{\top}\mathbf{W}_{\mathbf{x}}\mathbf{y}.$$

2.2 COMPARISON TO PREVIOUS WORKS

Before we show our improved upper bound, we first give a detailed discussion on other assumptions made in related works. Stochastic Hessian assumption. Ghadimi & Wang (2018); Ji et al. (2021) assumes the access to a stochastic Hessian estimator H(x, y) such that
$$\mathbb{E}H(\mathbf{x},\mathbf{y})=\nabla^{2}g(\mathbf{x},\mathbf{y}),\quad\mathbb{E}\|\mathbf{H}(\mathbf{x},\mathbf{y})-\nabla^{2}g(\mathbf{x},\mathbf{y})\|\leq\sigma^{2}.$$
2. (5)
According to (Arjevani et al., 2020, Observation 1 and 2), such an assumption is stronger than standard SGD assumptions and equivalent to the mean-squared-smoothness assumption (6) on the
lower-level gradient estimator G under the mild condition of ∇G(x, y) = H(x, y). Under this
assumption, in conjunction with Assumption 2.2, 2.3, and 2.4, Ghadimi & Wang (2018) proposed
the BSA method that can find an ϵ stationary point of φ(x) with O˜(ϵ
−6) SFO complexity and
O˜(ϵ
−4) stochastic HVP complexity. Later, Ji et al. (2021) further improved the SFO complexity
term to O˜(ϵ
−4). Compared to them, we consider the setting where the algorithms only have access
to stochastic gradient estimators, and make no assumptions on the stochastic Hessians.

Mean-squared smoothness assumption. Besides Assumption 2.1, 2.2, 2.3, 2.4 and the stochastic Hessian assumption (5), Khanduri et al. (2021); Yang et al. (2021; 2023b) further assumes that the stochastic estimators to gradients and Hessians are mean-squared smooth:

$$(6)$$
$$\begin{array}{l}{{\mathbb{E}\|F(\mathbf{x},\mathbf{y})-F(\mathbf{x}^{\prime},\mathbf{y}^{\prime})\|^{2}\leq\bar{L}_{1}^{2}\|(\mathbf{x},\mathbf{y})-(\mathbf{x}^{\prime},\mathbf{y}^{\prime})\|^{2},}}\\ {{\mathbb{E}\|G(\mathbf{x},\mathbf{y})-G(\mathbf{x}^{\prime},\mathbf{y}^{\prime})\|^{2}\leq\bar{L}_{1}^{2}\|(\mathbf{x},\mathbf{y})-(\mathbf{x}^{\prime},\mathbf{y}^{\prime})\|^{2},}}\\ {{\mathbb{E}\|\mathbf{H}(\mathbf{x},\mathbf{y})-\mathbf{H}(\mathbf{x}^{\prime},\mathbf{y}^{\prime})\|^{2}\leq\bar{L}_{2}^{2}\|(\mathbf{x},\mathbf{y})-(\mathbf{x}^{\prime},\mathbf{y}^{\prime})\|^{2}.}}\end{array}$$

Under this additional assumption, they proposed faster stochastic methods with upper complexity bound of O˜(ϵ
−3) via variance reduction (Fang et al., 2018; Cutkosky & Orabona, 2019). In this paper, we only consider the setting without mean-squared smoothness assumptions and study a different acceleration mechanism from variance reduction. Jointly high-order smoothness assumption. Huang et al. (2025) introduced a second-order smoothness assumption similar to but stronger than Assumption 2.5 when p = 2. Specifically, they assumed the second-order smoothness jointly in (x, y) instead of y only:

$\nabla^{2}f(\mathbf{x},\mathbf{y})$ is $L_{2}$-Lipschitz jointly in $(\mathbf{x},\mathbf{y})$;  $\nabla^{3}g(\mathbf{x},\mathbf{y})$ is $L_{3}$-Lipschitz jointly in $(\mathbf{x},\mathbf{y})$.  
The jointly second-order smoothness (7) ensures that the hyper-objective φ(x) has Lipschitz continuous Hessians, which further allows the application of known techniques in minimizing secondorder smooth objectives. Huang et al. (2025) applied the technique from (Jin et al., 2017; 2021; Xu et al., 2018; Allen-Zhu & Li, 2018) to show that an HVP-based method can find a second-order stationary point in O˜(ϵ
−2) complexity under the deterministic setting, and in O˜(ϵ
−4) under the stochastic Hessian assumption (5). Yang et al. (2023a) applied the technique from (Li & Lin, 2023)
to accelerate the complexity HVP-based method to O˜(ϵ
−1.75) in the deterministic setting. Chen et al. (2025b) also proposed a fully first-order method to achieve the same O˜(ϵ
−1.75) complexity.

Compared to these works, our work demonstrates a unique acceleration mechanism in stochastic bilevel optimization that only comes from the high-order smoothness in y.

$$\left(7\right)$$

## 3 The F 2Sa-P Method

To introduce our method, we first recall the prior F2SA method (Kwon et al., 2023) and establish their relationship between finite difference schemes, which further motivates us to design better algorithms by using better finite difference formulas.

## 3.1 Hyper-Gradient Approximation Via Finite Difference

The core idea of F2SA (Kwon et al., 2023) is to solve the reformulated penalty problem (3) and use the gradient of the penalty function to approximate the true hyper-gradient. To make connections of F
2SA to the finite difference method, let us introduce the extra notation gν as the perturbed lowerlever problem with y
∗ν(x) and ℓν(x) being its optimal solution and optimal value, respectively:

$$\begin{array}{r l}{{}}&{{}}\\ {{}}&{{g_{\nu}(\mathbf{x},\mathbf{y}):=\nu f(\mathbf{x},\mathbf{y})+g(\mathbf{x},\mathbf{y}),}}\\ {{}}&{{}}\\ {{}}&{{\mathbf{y}_{\nu}^{*}(\mathbf{x}):=\operatorname*{arg\,\,\operatorname*{min}_{\mathbf{y}\in\mathbb{R}^{d_{y}}}g_{\nu}(\mathbf{x},\mathbf{y}),}}\\ {{}}&{{}}\\ {{}}&{{\ell_{\nu}(\mathbf{x}):=\operatorname*{min}_{\mathbf{y}\in\mathbb{R}^{d_{y}}}g_{\nu}(\mathbf{x},\mathbf{y}),}}\end{array}$$

$$(8)$$

Then we have ∂
∂ν ℓν(x)|ν=0 = limν→0
$\ell_{\nu}(\mathbf{x})-\ell_{0}(\mathbf{x}))\,=\,\lim_{\mathbf{x}\to\mathbf{x}}\ell_{\nu}(\mathbf{x})$
ν = limν→0 f(x, y
∗ν(x))+g(x,y
∗ ν
(x))−g(x,y
∗(x))
ν.
In our notation, we can rewrite (Chen et al., 2025b, Lemma B.3) as ∂
∂ν ℓν(x)|ν=0 = φ(x). Similarly,
we can also rewrite (Kwon et al., 2023, Lemma 3.1) as
$${\frac{\partial^{2}}{\partial\nu\partial\mathbf{x}}}\ell_{\nu}(\mathbf{x})|_{\nu=0}={\frac{\partial^{2}}{\partial\mathbf{x}\partial\nu}}\ell_{\nu}(\mathbf{x})|_{\nu=0}=\nabla\varphi(\mathbf{x}).$$
ℓν(x)|ν=0 = ∇φ(x). (8)
Let ν = 1/λ in Eq. (3). Then the fully first-order hyper-gradient estimator (Kwon et al., 2023; Chen
et al., 2025b) is exactly using forward difference to approximate ∇φ(x), that is,
$$\frac{\partial}{\partial\mathbf{x}}\ell_{\nu}(\mathbf{x})-\frac{\partial}{\partial\mathbf{x}}\ell_{0}(\mathbf{x})\approx\frac{\partial^{2}}{\partial\nu\partial\mathbf{x}}\ell_{\nu}(\mathbf{x})|_{\nu=0}=\nabla\varphi(\mathbf{x}).\tag{9}$$  However, the forward difference is not the only way to approximate a derivative. Essentially, it is 
falls into a general class of pth-order finite difference (Atkinson & Han, 2005) that can guarantee an O(ν p) approximation error. We restate this known result (with generalization to vector-valued functions) in the following lemma and provide a self-contained proof in Appendix B for completeness.

Lemma 3.1. *Assume the function* ψ : R → R
d has C-Lipschitz continuous pth-order derivative.

There exist coefficients {αj} *such that*

$$\left\|{\frac{1}{\nu}}\sum_{j}\alpha_{j}\psi(j\nu)-\psi^{\prime}(0)\right\|={\mathcal{O}}(C\nu^{p}).$$

If p is even, the indices run j = −p/2, · · · , p/2. If p is odd, they run j = −(p−1)/2, *· · ·* ,(p+1)/2.

Furthermore, all the coefficients satisfy |jαj | ≤ 1 *for all* j ̸= 0 and |α0| ≤ I[p *is odd*].

The explicit formulas for αj can be found in Appendix B. When p = 1, we have α0 = −1, α1 = 1, and we obtain the forward difference estimator ψ(ν) − ψ(0)/ν; When p = 2 we have α−1 = −1/2, α1 = 1/2 and we obtain the central difference estimator (ψ(ν) − ψ(−ν))/(2ν). Lemma 3.1 tells us that in general we can always construct a finite difference estimator O(ν p) error with p points for even p or p + 1 points for odd p under the given smoothness conditions. Inspired by Lemma 3.1 and Eq. (8) that ∂
2
∂ν∂x ℓν(x)|ν=0 = ∇φ(x), we propose a fully first-order estimator via a linear combination of ∂
∂x ℓjν(x) to achieve O(ν p) approximation error to ∇φ(x) given that ∂
p+1
∂νp∂x ℓν(x)
is Lipschitz continuous in ν. It further leads to Algorithm 1 that will be formally introduced in the next subsection.

## 3.2 The Proposed Algorithm

Due to space limitations, we only present Algorithm 1 designed for even p in the main text. The algorithm for odd p can be designed similarly, and we defer the concrete algorithm to Appendix D. Algorithm 1 F
2SA-p (x0, y0), even p 1: y j 0 = y0, ∀j ∈ N
2: for t = 0, 1, · · · , T − 1 3: **parallel for** j = −p/2, −p/2 + 1, · · · *, p/*2 4: y j,0 t = y j t 5: for k = 0, 1, · · · , K − 1 6: Sample random i.i.d indexes {(ξ y j
, ζy j
)}.

7: y j,k+1 t = y j,k t − ηy jνFy(xt, y j,k t; ξ y j) + Gy(xt, y j,k t; ζ y j)

8: **end for** 9: y j t+1 = y j,K t 10: **end parallel for** 11: Sample random i.i.d indexes {(ξ x i
, ζx i
)}
S
i=1.

12: Let {αj}
p/2 j=−p/2 be the pth-order finite difference coefficients defined in Lemma 3.1.

13: Φt =
1 S
PS
i=1 Pp/2 j=−p/2 αj
 
jFx(xt, y j t+1; ξ x i) + 
Gx(xt, y j t+1; ζ x i
)
ν
!

14: xt+1 = xt − ηxΦt/∥Φt∥
15: **end for**
Algorithm 1 follows the double-loop structure of F2SA (Chen et al., 2025b; Kwon et al., 2024a) and changes the hyper-gradient estimator to the one introduced in the previous section. Now, we give a more detailed introduction to the procedures of the two loops of F2SA-p.

1. In the outer loop, the algorithm first samples a mini-batch with size S and uses Lemma 3.1 to construct Φt via the linear combination of ∂
∂x ℓjν(xt) for j = −p/2, · · · *, p/*2 every iteration. After obtaining Φt as an approximation to ∇φ(xt), the algorithm then performs a normalized gradient descent step xt+1 = xt − ηxΦt/∥Φt∥ with total T iterations.

2. In the inner loop, the algorithm returns an approximation to ∂
∂x ℓjν(xt) for all j =
−p/2, · · · *, p/*2. Note that Danskin's theorem indicates ∂
∂x ℓjν(xt) = ∂
∂x gjν(xt, y
∗
jν(xt)).

It suffices to approximate y
∗
jν(xt) to sufficient accuracy, which is achieved by taking a K-step single-batch SGD subroutine with stepsize ηy on each function gjν(x, ·).

Remark 3.1 (Effect of normalized gradient step). *Compared to (Chen et al., 2025b; Kwon et al.,* 2023), the only modification we make to the outer loop is to change the gradient step to a normalized gradient step. The normalization can control the change of y
∗
jν(xt) and make the analysis of inner loops easier. We believe that all our theoretical guarantees also hold for the standard gradient step via a more involved analysis.

## 3.3 Complexity Analysis

This section contains the complexity analysis of Algorithm 1. We first derive the following lemma from the high-dimensional Faa di Bruno formula (Licht, 2024). ` Lemma 3.2. Let ν ∈ (0, 1/(2κ)]. For any instance in the pth-order smooth bilevel problem class F
nc-sc(L0, · · · , Lp+1, µ, ∆) *as Definition 2.2,* ∂
p+1
∂νp∂x ℓν(x) is O(κ 2p+1L¯)*-Lipschitz continuous in* ν.

Our result generalizes the prior result for p = 1 (Kwon et al., 2023) to any p ∈ N+ and also tightens the prior bounds for p = 2 (Chen et al., 2025b) as we remark in the following.

Remark 3.2 (Tighter bounds for p = 2). Note that the variables x and ν play equal roles in our analysis. Therefore, our result in p = 2 *essentially implies that* ∂
3
∂ν∂x2 ℓν(x) is O(κ 5L¯)-Lipschitz continuous in ν *around zero, which tightens the* O(κ 6L¯) bound of Hessian convergence in (Chen et al., 2025b, Lemma 5.1a) and is of independent interest. The main insight is to avoid the direct calculation of ∇2φ(x) = ∂
3
∂ν∂x2 ℓν(x)|ν=0 which involves third-order derivatives and makes the analysis more complex, but instead always to analyze it through the limiting point limν→0∂
3
∂ν∂x2 ℓν(x).

Recall Eq. (8) that ∂
2
∂ν∂x ℓν(x)|ν=0 = ∇φ(x). Then Lemma 3.2, in conjunction with Lemma 3.1, indicates that the pth-order finite difference used in F2SA-p guarantees an O(ν p)-approximation error to ∇φ(x), which always improves the O(ν)-error guarantee of F2SA (Kwon et al., 2023; Chen et al., 2025b) for any p ≥ 2. This improved error guarantee means that we can set ν = O(ϵ 1/p)
to obtain an O(ϵ)-accurate hyper-gradient estimator to ∇φ(x), which further leads to the following improved complexity of our algorithm. Theorem 3.1 (Main theorem). For any instance in the p*th-order smooth bilevel problem class* F
nc-sc(L0, · · · , Lp+1, µ, ∆) *as per Definition 2.2, set the hyper-parameters as*

$$\nu\times\min\left\{\frac{R}{\kappa},\left(\frac{\epsilon}{L_{\kappa}2^{p+1}}\right)^{1/p}\right\},\ \ \eta_{x}\times\frac{\epsilon}{L_{1}\kappa^{3}},\ \ \eta_{y}\asymp\frac{\nu^{2}\epsilon^{2}}{L_{1}\kappa\sigma^{2}},\tag{10}$$ $$S\asymp\frac{\sigma^{2}}{\nu^{2}\epsilon^{2}},\ \ K\asymp\frac{\kappa^{2}\sigma^{2}}{\nu^{2}\epsilon^{2}}\log\left(\frac{RL_{1}\kappa}{\nu\epsilon}\right),\ \ T\asymp\frac{\Delta}{\eta_{x}\epsilon},$$
I would like $\mathbf{J}$ if $\mathbf{J}$. 

where R = ∥y0 − y
∗(x0)∥. Run Algorithm 1 if p is even or Algorithm 2 (in Appendix D) if p *is odd.*
Then we can provably find an ϵ-stationary point of φ(x) *with the total SFO calls upper bounded by*

$$p T(S+K)={\mathcal{O}}\left({\frac{p\Delta L_{1}\bar{L}^{2/p}\sigma^{2}\kappa^{9+2/p}}{\epsilon^{4+2/p}}}\log\left({\frac{R L_{1}\bar{L}\kappa}{\epsilon}}\right)\right).$$

The above theorem shows that the F2SA-p method can achieve the O˜(pκ9+2/pϵ
−4−2/p log(κ/ϵ))
SFO complexity for pth-order smooth bilevel problems. In the following, we give several remarks on the complexity in different regions of p.

Remark 3.3 (First-order smooth region). For p = 1*, our upper bound becomes* O˜(κ 11ϵ
−6)*, which* improves the O˜(κ 12ϵ
−6) bound in (Chen et al., 2025b) by a factor of κ*. The improvement comes* from a tighter analysis in the lower-level SGD update and a careful parameter setting. Remark 3.4 (Highly-smooth region). For p = Ω(log(κ/ϵ)/ log log(κ/ϵ)) in Definition 2.2, we can run F2SA-q *with* q ≍ log(κ/ϵ)/ log log(κ/ϵ) *and the* O(qκ9ϵ
−4(κ/ϵ)
2/q log(κ/ϵ)) complexity in Theorem 3.1 simplifies to O(κ 9ϵ
−4log3(κ/ϵ)/ log log(κ/ϵ)) = O˜(κ 9ϵ
−4), which matches the best-known complexity for HVP-based methods (Ji et al., 2021) under stochastic Hessian assumption (5). In the upcoming section, we will derive an Ω(ϵ
−4) lower bound to prove that the F2SA-p is nearoptimal in the above highly-smooth region if the condition number κ is a constant. We leave the study of optimal complexity for non-constant κ to future work. Comparison of results for odd p **and even** p. Note that by Lemma 3.1 when p is odd, we need to use p+1 points to construct the estimator, which means the algorithm needs to solve p+1 lower-level problems in each iteration to achieve an O(ν p) error guarantee. In contrast, when p is even, p points are enough since the pth-order central difference estimator satisfies that α0 = 0. It suggests that even when p is odd, the algorithm designed for odd p may still be better. For instance, the F2SA-2 may always be a better choice than F2SA since its benefits *almost come for free*: (1) it still only needs to solve 2 lower-level problems as the F2SA method, which means the per-iteration complexity remains the same. (2) Although the improved complexity of F2SA-2 relies on the second-order smooth condition, without such a condition, its error guarantee in hyper-gradient estimation will only degenerate to a first-order one, which means it is at least as good as F2SA.

## 4 An Ω(Ε −4) Lower Bound

In this section, we prove an Ω(ϵ
−4) lower bound for stochastic bilevel optimization via a reduction to single-level optimization. Our lower bound holds for any randomized algorithms A, which can be defined as a sequence of measurable mappings {At}
T
t=1 that is defined recursively by
(xt+1, yt+1) = At (r, F(x0, y0), G(x0, y0)), · · · , F(xt, yt), G(xt, yt))), t ∈ N+, (11)
where r is a random seed drawn at the beginning to produce the queries, and *F, G* are the stochastic gradient estimators that satisfy Assumption 2.1. Without loss of generality, we assume that
(x0, y0) = (0, 0). Otherwise, we can prove the same lower bound by shifting the functions.

$$(\Phi_{0})),\cdots,F(\mathbf{x}_{t},\mathbf{y}_{t}),G(\mathbf{x}_{t},\mathbf{y}_{t})))\,,\;\;t\in\mathbb{N}_{+},\tag{11}$$

The construction. We construct a separable bilevel instance such that the upper-level function f(x, y) ≡ fU (x) and its stochastic gradient align with the hard instance in (Arjevani et al., 2023), while the lower-level function is the simple quadratic g(x, y) ≡ g(y) = µy2/2 with deterministic gradients. We defer the concrete construction to Appendix E. For this separable bilevel instance, we can show that for any randomized algorithm defined in Eq. (11) that uses oracles (FU , G), the progress in x can be simulated by another randomized algorithm that only uses FU , meaning that the single-level lower bound (Arjevani et al., 2023) also holds. Theorem 4.1 (Lower bound). There exist numerical constants c > 0 *such that for all* ∆ >
0, L1, · · · , Lp+1 > 0 and ϵ ≤ c
√L1∆*, there exists a distribution over the function class* F
nc-sc(L0, · · · , Lp+1, µ, ∆) and the stochastic gradient estimators satisfying Assumption 2.1, such that any randomized algorithm A defined as Eq. (11) can not find an ϵ*-stationary point of* φ(x) = f(x, y
∗(x)) *in less than* Ω(∆L1σ 2ϵ
−4) *SFO calls.*
Below, we give a detailed discussion on the constructions in related works. Comparison to other bilevel lower bounds. Dagreou et al. (2024) proved lower bounds for finite- ´ sum bilevel optimization via a similar reduction to single-level optimization. However, the direct extension of their construction in the fully stochastic setting gives f(x, y) = fU (y) and g(x, y) =
(x − y)
2, where the high-order derivatives of f(x, y) not O(1)-Lipschitz in y and thus violates our assumptions. Kwon et al. (2024a) also proved an Ω(ϵ
−4) lower bound for stochastic bilevel optimization. However, their construction f(x, y) = y and g(x, y) = (fU (x) − y)
2 violate the first-order smoothness of g(x, y) in x when y is far way from fU (x). In this work, we use a fully separable construction to avoid all the aforementioned issues in other works.

## 5 Experiments

In this section, we conduct numerical experiments to verify our theory. Following (Grazzi et al., 2020; Ji et al., 2021), we consider the "learn-to-regularize" problem of logistic regression (Example 2.2) on the "20 Newsgroup" dataset, which provably satisfies the highly smooth assumption of any order. The dataset contains 18,000 samples, each sample consists of a feature vector in dimensional 130, 107 vector and a label that takes a value in {1, *· · ·* , 20}. We compare our proposed method F2SA-p with p ∈ {2, 3, 5, 8, 10} with both the previous best fully first-order method F2SA
(Kwon et al., 2023; Chen et al., 2025b) and other Hessian-vector-product-based methods stocBiO (Ji et al., 2021), MRBO and VRBO (Yang et al., 2021). We also include a baseline "w/o Reg" that means the training result of SGD without tuning any regularization. For all the algorithms, we search the other hyperparameters (including ηx, ηy, ν) in a logarithmic scale with base 10. We run the algorithms with K = 10 iterations in the inner loop, and T = 1000 iterations in the outer loop, and report the test loss/accuracy *v.s.* the number of outer-loop iterations t in Figure 1. To demonstrate the potential of our methods on nonsmooth nonconvex problems, we also provide additional experiments on a 5-layer multilayer perceptron (MLP) network with ReLU activation in Appendix F. The codes to reproduce our experiments are available online 1.

## 6 Conclusions And Future Works

This paper proposes a class of fully first-order method F2SA-p that achieves the O˜(pϵ−4−2/p) SFO complexity for pth-order smooth bilevel problems. Our result generalized the best-known O˜(ϵ
−6)
result (Kwon et al., 2024a; Chen et al., 2025b) from p = 1 to any p ∈ N+. We also complement our result with an Ω(ϵ
−4) lower bound to show that our method is near-optimal when p = Ω(log ϵ
−1/ log log ϵ
−1). Nevertheless, a gap still exists when p is small, and how to fill it even for the basic setting p = 1 is an open problem. Another open problem is whether our theory can be extended our theory to structured nonconvex-nonconvex bilevel problems studied by many recent works (Kwon et al., 2024b; Chen et al., 2024; 2025a; Jiang et al., 2025; Xiao et al., 2023; Xiao
& Chen, 2025). In addition, it will also be interesting to further improve the convergence rate of our methods by combining them with variance-reduction (Fang et al., 2018; Cutkosky & Orabona, 2019) or momentum techniques (Fang et al., 2019; Cutkosky & Mehta, 2020).

1https://github.com/TrueNobility303/F2BA

![9_image_0.png](9_image_0.png)

## Acknowledgments

Lesi Chen thanks Jeongyeol Kwon for helpful discussions and Sanyou Mei for pointing out related references. Jingzhao Zhang is supported by Shanghai Qi Zhi Institute Innovation Program, Tsinghua Dushi Funds, and Xiongan AI Institute.

## References

Zeyuan Allen-Zhu and Yuanzhi Li. Neon2: Finding local minima via first-order oracles. In *NeurIPS*,
2018.

Yossi Arjevani, Yair Carmon, John C. Duchi, Dylan J. Foster, Ayush Sekhari, and Karthik Sridharan. Second-order information in non-convex stochastic optimization: Power and limitations. In COLT, 2020.

Yossi Arjevani, Yair Carmon, John C. Duchi, Dylan J. Foster, Nathan Srebro, and Blake Woodworth.

Lower bounds for non-convex stochastic optimization. *Mathematical Programming*, 199(1):165–
214, 2023.

Kendall Atkinson and Weimin Han. Finite difference method. Theoretical Numerical Analysis: A
Functional Analysis Framework, pp. 249–271, 2005.

Fan Bao, Guoqiang Wu, Chongxuan Li, Jun Zhu, and Bo Zhang. Stability and generalization of bilevel programming in hyperparameter optimization. In *NeurIPS*, 2021.

Yair Carmon, John C. Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points i. *Mathematical Programming*, 184(1):71–120, 2020.

Yair Carmon, John C. Duchi, Oliver Hinder, and Aaron Sidford. Lower bounds for finding stationary points ii: first-order methods. *Mathematical Programming*, 185(1):315–355, 2021.

El Mahdi Chayti and Martin Jaggi. A new first-order meta-learning algorithm with convergence guarantees. *arXiv preprint arXiv:2409.03682*, 2024.

He Chen, Jiajin Li, and Anthony Man-cho So. Set smoothness unlocks clarke hyper-stationarity in bilevel optimization. In *NeurIPS*, 2025a.

Lesi Chen and Jingzhao Zhang. On the condition number dependency in bilevel optimization. *arXiv* preprint arXiv:2511.22331, 2025.

Lesi Chen, Jing Xu, and Jingzhao Zhang. On finding small hyper-gradients in bilevel optimization:
Hardness results and improved analysis. In *COLT*, 2024.

Lesi Chen, Yaohua Ma, and Jingzhao Zhang. Near-optimal nonconvex-strongly-convex bilevel optimization with fully first-order oracles. *JMLR*, 2025b.

Ashok Cutkosky and Harsh Mehta. Momentum improves normalized SGD. In *ICML*, 2020.

Ashok Cutkosky and Francesco Orabona. Momentum-based variance reduction in non-convex SGD.

In *NeurIPS*, 2019.

Mathieu Dagreou, Thomas Moreau, Samuel Vaiter, and Pierre Ablin. A lower bound and a near- ´
optimal algorithm for bilevel empirical risk minimization. In *AISTATS*, 2024.

Asen L. Dontchev and R. Tyrrell Rockafellar. *Implicit functions and solution mappings*, volume 543. Springer, 2009.

Cong Fang, Chris Junchi Li, Zhouchen Lin, and Tong Zhang. Spider: Near-optimal non-convex optimization via stochastic path-integrated differential estimator. In *NeurIPS*, 2018.

Cong Fang, Zhouchen Lin, and Tong Zhang. Sharp analysis for nonconvex SGD escaping from saddle points. In COLT, 2019.

Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel programming for hyperparameter optimization and meta-learning. In *ICML*, 2018.

Ankit Garg, Robin Kothari, Praneeth Netrapalli, and Suhail Sherif. Near-optimal lower bounds for convex optimization for all orders of smoothness. In *NeurIPS*, 2021.

Saeed Ghadimi and Mengdi Wang. Approximation methods for bilevel programming. arXiv preprint arXiv:1802.02246, 2018.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.

Riccardo Grazzi, Luca Franceschi, Massimiliano Pontil, and Saverio Salzo. On the iteration complexity of hypergradient computation. In *ICML*, 2020.

Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale stochastic algorithm framework for bilevel optimization: Complexity analysis and application to actor-critic. SIAM Journal on Optimization, 33(1):147–180, 2023.

Minhui Huang, Xuxing Chen, Kaiyi Ji, Shiqian Ma, and Lifeng Lai. Efficiently escaping saddle points in bilevel optimization. *JMLR*, 26(1):1–61, 2025.

Kaiyi Ji. Lower complexity bounds for nonconvex-strongly-convex bilevel optimization with firstorder oracles. *arXiv preprint arXiv:2511.19656*, 2025.

Kaiyi Ji, Junjie Yang, and Yingbin Liang. Bilevel optimization: Convergence analysis and enhanced design. In *ICML*, 2021.

Liuyuan Jiang, Quan Xiao, Lisha Chen, and Tianyi Chen. Beyond value functions: Single-loop bilevel optimization under flatness conditions. In *NeurIPS*, 2025.

Chi Jin, Rong Ge, Praneeth Netrapalli, Sham M. Kakade, and Michael I. Jordan. How to escape saddle points efficiently. In *ICML*, 2017.

Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M. Kakade, and Michael I. Jordan. On nonconvex optimization for machine learning: Gradients, stochasticity, and saddle points. *Journal of the* ACM, 68(2):1–29, 2021.

Ishtiaq Rasool Khan, Ryoji Ohba, and Noriyuki Hozumi. Mathematical proof of closed form expressions for finite difference approximations based on taylor series. Journal of Computational and Applied Mathematics, 150(2):303–309, 2003.

Prashant Khanduri, Siliang Zeng, Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang.

A near-optimal algorithm for stochastic bilevel optimization via double-momentum. In *NeurIPS*, 2021.

Tamara G. Kolda and Brett W. Bader. Tensor decompositions and applications. *SIAM review*, 51(3):
455–500, 2009.

Guy Kornowski and Ohad Shamir. Oracle complexity in nonsmooth nonconvex optimization. *JMLR*,
pp. 1–44, 2022.

Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert D. Nowak. A fully first-order method for stochastic bilevel optimization. In ICML, 2023.

Jeongyeol Kwon, Dohyun Kwon, and Hanbaek Lyu. On the complexity of first-order methods in stochastic bilevel optimization. In *ICML*, 2024a.

Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, and Robert D. Nowak. On penalty methods for nonconvex bilevel optimization and first-order stochastic approximation. In *ICLR*, 2024b.

Huan Li and Zhouchen Lin. Restarted nonconvex accelerated gradient descent: No more polylogarithmic factor in the in the O(ϵ
−7/4) complexity. *JMLR*, 2023.

Martin W Licht. Higher-order chain rules for tensor fields, generalized bell polynomials, and estimates in orlicz-sobolev-slobodeckij and total variation spaces. Journal of Mathematical Analysis and Applications, 534(1):128005, 2024.

Bo Liu, Mao Ye, Stephen Wright, Peter Stone, and Qiang Liu. Bome! bilevel optimization made easy: A simple first-order approach. In *NeurIPS*, 2022.

Risheng Liu, Xuan Liu, Shangzhi Zeng, Jin Zhang, and Yixuan Zhang. Value-function-based sequential minimization for bi-level optimization. *TPAMI*, 45(12):15930–15948, 2023.

Zhaosong Lu and Sanyou Mei. First-order penalty methods for bilevel optimization. SIAM Journal on Optimization, 34(2):1937–1969, 2024.

Zhaosong Lu and Sanyou Mei. Solving bilevel optimization via sequential minimax optimization.

Mathematics of Operations Research, 2026.

Luo Luo, Yujun Li, and Cheng Chen. Finding second-order stationary points in nonconvex-stronglyconcave minimax optimization. In *NeurIPS*, 2022.

Matthew Mackay, Paul Vicol, Jonathan Lorraine, David Duvenaud, and Roger Grosse. Self-tuning networks: Bilevel optimization of hyperparameters using structured best-response functions. In ICLR, 2019.

Arkadij Semenovic Nemirovskij and David Borisovich Yudin. Problem complexity and method ˇ
efficiency in optimization. 1983.

Rui Pan, Jipeng Zhang, Xingyuan Pan, Renjie Pi, Xiaoyu Wang, and Tong Zhang. ScaleBiO: Scalable bilevel optimization for LLM data reweighting. *arXiv preprint arXiv:2406.19976*, 2024.

Fabian Pedregosa. Hyperparameter optimization with approximate gradient. In *ICML*, 2016.

Aravind Rajeswaran, Chelsea Finn, Sham M. Kakade, and Sergey Levine. Meta-learning with implicit gradients. In *NeurIPS*, volume 32, 2019.

Han Shen and Tianyi Chen. On penalty-based bilevel gradient descent method. In *ICML*, 2023. Han Shen, Pin-Yu Chen, Payel Das, and Tianyi Chen. Seal: Safety-enhanced aligned llm fine-tuning via bilevel data selection. In ICLR, 2025a.

Han Shen, Quan Xiao, and Tianyi Chen. On penalty-based bilevel gradient descent method. *Mathematical Programming*, pp. 1–51, 2025b.