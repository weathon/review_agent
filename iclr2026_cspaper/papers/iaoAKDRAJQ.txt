# A Tale Of Two Geometries: Adaptive Optimiz- Ers And Non-Euclidean Descent

Shuo Xie1∗ Tianhao Wang2∗ Beining Wu3 **Zhiyuan Li**1 1Toyota Technological Institute at Chicago 2University of California, San Diego 3University of Chicago
{shuox,zhiyuanli}@ttic.edu, tianhaowang@ucsd.edu, beiningw@uchicago.edu

## Abstract

Adaptive optimizers can reduce to normalized steepest descent (NSD) when only adapting to the current gradient, suggesting a close connection between the two algorithmic families. A key distinction between their analyses, however, lies in the geometries, e.g., smoothness notions, they rely on. In the convex setting, adaptive optimizers are governed by a stronger adaptive smoothness condition, while NSD relies on the standard notion of smoothness. We extend the theory of adaptive smoothness to the nonconvex setting and show that it precisely characterizes the convergence of adaptive optimizers. Moreover, we establish that adaptive smoothness enables acceleration of adaptive optimizers with Nesterov momentum in the convex setting, a guarantee unattainable under standard smoothness for certain non-Euclidean geometry. We further develop an analogous comparison for stochastic optimization by introducing adaptive gradient variance, which parallels adaptive smoothness and leads to dimension-free convergence guarantees that cannot be achieved under standard gradient variance for certain non-Euclidean geometry.

## 1 Introduction

Adaptive optimizers such as Adam have been indispensable for training large-scale machine learning models (Bi et al., 2024; Dubey et al., 2024; Yang et al., 2025; Wen et al., 2025). Their dominance in training efficiency, however, has recently been challenged by the surprising effectiveness of simpler Normalized Steepest Descent (NSD)-type methods such as Muon and Lion (Jordan et al., 2024; Chen et al., 2023; Team et al., 2025; Liu et al., 2025; Shah et al., 2025). Behind this competition of two family of optimizers, a broader consensus has begun to emerge: their superior performance is critically related to their ability to exploit non-Euclidean geometry of the loss landscape (Balles et al., 2020; Xie & Li, 2024; Zhang et al., 2024; Pethick et al., 2025). Recent studies have rigorously characterized how adaptive optimizers exploit non-Euclidean geometry. For example, Maladkar et al. (2024) and Xie et al. (2025a) show that AdaGrad and Adam benefit from exploiting the ℓ∞-geometry of loss functions, and a one-sided variant of Shampoo has been shown to leverage the geometry induced by the matrix spectral norm (Xie et al., 2025b; An et al., 2025). Notably, Bernstein & Newhouse (2024) proposed a striking connection between adaptive optimizers and NSD: with exponential moving average (EMA) turned off, certain adaptive optimizers reduce exactly to their NSD counterparts. For example, without EMA, Adam coincides with NSD
under the ℓ∞ norm, and Shampoo coincides with NSD under the matrix spectral norm, which is proposed to be an independent algorithm Muon. Yet, beyond these connections, there is no formal result that systematically characterizes the relationship between the two families of algorithms. This naturally motivates the following question:
Q1. Do adaptive methods (like Adam, Shampoo) and their corresponding non-Euclidean descent (like Lion, Muon) exploit the non-Euclidean geometry of loss landscape in the same way?

To address this question, we adopt a theoretical perspective and focus on comparing different types of smoothness assumptions that underpin the analysis of these methods. In fact, even under the
∗Equal contribution.

same geometry, two distinct notions of smoothness arise. The first is the standard smoothness under a general norm (cf. Definition 2.3), which governs the convergence rate of NSD. The second is called the *adaptive smoothness* (cf. Definition 2.4), introduced by Xie et al. (2025b) and shown to govern the convergence rate of adaptive optimizers in the convex case. Indeed, a main contribution of our work is to show that adaptive smoothness also characterizes the convergence rate of adaptive optimizers in the nonconvex setting. Therefore, while both adaptive optimizers and NSD can exploit non-Euclidean geometry, they rely on fundamentally different smoothness assumptions. This difference is not merely terminological but quantitative: adaptive smoothness is always no smaller than the standard smoothness under the same geometry. In other words, from the standpoint of technical conditions, the adaptive smoothness represents a stronger assumption, which in turn motivates our second question:
Q2. *Does the stronger smoothness assumption in adaptive methods offer optimization benefit?*
We answer this question affirmatively. In particular, we show that by leveraging Nesterov acceleration, adaptive optimizers can attain an accelerated O(T
−2) rate under adaptive smoothness in the convex setting. In sharp contrast, it has been shown by Guzman & Nemirovski ´ (2015) that the convergence rate of any optimizer is no better than Ω(T
−1) under the standard ℓ∞ smoothness assumption. This establishes a clear separation: adaptive smoothness enables adaptive optimizers to achieve acceleration under non-Euclidean geometry, while the standard smoothness fails. Therefore, the stronger adaptive smoothness assumption indeed translates into concrete optimization benefits, showing its difference from the standard smoothness. In fact this difference has a direct and interesting analogy in terms of the noise assumption in the stochastic setting. When gradient noise is present, its variability can be measured in two distinct ways: the standard variance considers gradient variation under a fixed norm, whereas the *adaptive* variance (cf. Definition 4.1) measures noise in a more stringent but also more adaptive way that requires uniform control over the geometry prescribed by each preconditioner under consideration. By construction, adaptive variance is always no smaller than standard variance, directly paralleling the relationship between adaptive and standard smoothness. Analogous to adaptive smoothness that enables acceleration under a stronger requirement, adaptive variance can likewise yield benefits despite being larger. We demonstrate this through a careful analysis of NSD under two types of noise assumptions: adaptive variance enables a dimension-free rate, which is not attainable in the worst case under the standard variance condition. Taken together, our results demonstrate that adaptive smoothness and adaptive variance are different from their standard counterparts as adaptive smoothness enables an acceleration rate and adaptive noise enables a dimension-free rate. These findings reveal an intricate interplay between adaptivity and non-Euclidean geometry, deepening our theoretical understanding of adaptivity in optimization. Below we summarize our main contributions.

- In Section 3, we show the convergence rate for adaptive optimizers on nonconvex functions (Theorems D.2, D.7 and D.8), which depends on the adaptive smoothness and matches optimal O˜(T
−1/4) rate. It theoretically justifies that adaptive methods and NSD exploit the geometry through different smoothness notions in the nonconvex setting.

- In Section 4.2, we identify the benefit of the adaptive smoothness by showing it enables an acceleration rate O˜(T
−2) of adaptive optimizers equipped with Nesterov momentum (Theorem 4.3) in contrast to the convergence rate Ω(T
−1) the standard ℓ∞ smoothness.

- In Section 4, we extend the benefit of adaptive geometry to noise assumptions by introducing adaptive noise (Definition 4.1). We show that this stronger notion of noise can provide a new type of convergence rate for NSD with momentum on nonconvex functions which gets rid of dependence on parameter size d (Theorem 4.5). We complement its superiorty by providing a lower bound under the standard noise (Theorem 4.7).

- Our analysis of adaptive optimizers is carried out through a unified framework that covers a broad class of methods, including AdaGrad, AdaGrad-Norm, and one-sided Shampoo. The proof technique developed in this framework may be of independent interest.

## 1.1 Notations

Let Md be the set of all d-by-d matrices, S
d ⊂ Md be the subset of all symmetric matrices. We use S
d+ to denote the set of positive semi-definite matrices. We denote by Id ∈ Mdthe identity matrix.

For matrices A, B, we denote their inner product by ⟨A, B⟩ = Tr(A⊤B).

For H ∈ Sd+, ∥x∥H := 
√x⊤Hx is the (semi-)norm of x ∈ R
d with respect to H. For a convex set *H ⊆ S*d+, we define the induced H-norm as

$$\left\|\mathbf{x}\right\|_{\mathcal{H}}:=\sup_{\mathbf{H}\in\mathcal{H},\mathrm{Tr}(\mathbf{H})\leq1}\left\|\mathbf{x}\right\|_{\mathbf{H}}.\tag{1}$$

Throughout the paper, we reserve f for the loss function and x0 for the initialization of an optimization algorithm. For convenience, we denote the initial suboptimality as ∆0 = f(x0) − minx f(x).

## 2 From Adam/Signgd To Adaptive Smoothness

We use the example of Adam and SignGD to motivate the notion of adaptive smoothness in Section 2.1, and then present the formal definition in Section 2.2, along with some related background.

## 2.1 Adam And Signgd Can Exploit ℓ∞ Geometry, But In Different Ways

We start by discussing a specific pair of algorithms, Adam and SignGD, to illustrate the problem of interest. It is known that SignGD can be viewed as Normalized Steepest Descent (NSD) under the ℓ∞ norm and its convergence rate for deterministic nonconvex functions admits the following form (Xie et al., 2025a)

$$\operatorname*{min}_{t\in[T]}\|\nabla f(\mathbf{x}_{t})\|_{1}\leq O{\bigg(}{\sqrt{\frac{\Delta_{0}L_{\|\cdot\|_{\infty}}(f)}{T}}}{\bigg)}$$

where L∥·∥∞
(f) is the standard smoothness of f under the ℓ∞ norm (see Definition 2.3). Note that SignGD can also be viewed as a special case of Adam with β1 = β2 = 0. However, the convergence rate of Adam for general β1, β2 instead depends on a different diagonal adaptive smoothness notion, which is defined as Ldiag(f) = minH∈Dd,−H⪯∇2f(x)⪯H Tr(H) in Maladkar et al. (2024); Xie et al. (2025a). In particular, Adam with β1 = 0 (a.k.a. RMSProp) for deterministic nonconvex functions admits the convergence rate mint∈[T] ∥∇f(xt)∥1 = O˜(p∆0Ldiag(f)/T) (Xie et al., 2024). Notably, this diagonal adaptive smoothness is always no smaller than L∥·∥∞
(f) (Balles et al., 2020). This suggests that though both SignGD and Adam admit convergence guarantees for the ℓ1 norm (the dual norm of *∥ · ∥*∞) of the gradients, they achieve so under different smoothness notions. This distinction motivates the following question:
How does the diagonal adaptive smoothness Ldiag(f) emerge as an ℓ∞ *geometry?*
To address this question, let us consider the convergence rate of NSD under any norm *∥ · ∥*H for H ∈ H = Dd+ (see Theorem 4.5):

(iii.4.3).  $$\min_{t\in[T]}\|\nabla f(\mathbf{x}_{t})\|_{\mathbf{H},*}=O\bigg{(}\sqrt{\frac{\Delta_{0}L_{\|\cdot\|_{H}}(f)}{T}}\bigg{)}$$
(2)
where *∥ · ∥*H,∗ is the dual norm of *∥ · ∥*H. Minimizing both sides of (2) over H ∈ Dd+ with Tr(H) ≤ 1 yields

$$\inf_{\begin{subarray}{c}\underline{\mathrm{diagonal}}\,\mathbf{H}>0\\ \mathrm{Tr}(\mathbf{H})\leq1\end{subarray}}\min_{\mathbf{H}\in[T]}\|\nabla f(\mathbf{x}_{t})\|_{\mathbf{H}^{*}}=O\bigg{(}\sqrt{\frac{\Delta_{0}}{T}}\inf_{\begin{subarray}{c}\underline{\mathrm{diagonal}}\,\mathbf{H}>0\\ \mathrm{Tr}(\mathbf{H})\leq1\end{subarray}}L_{\|\cdot\|_{\mathbf{H}}}(f)\bigg{)}=O\bigg{(}\sqrt{\frac{\Delta_{0}L_{\mathrm{diag}}(f)}{T}}\bigg{)}.$$
(3)
where the equality can be checked by the definition of L*diag*(f). Now the right-hand side matches the aforementioned convergence rate of Adam. The adaptivity of Adam is then demonstrated by its ability to automatically identify and adapt to the best diagonal matrix-induced norm for any given loss function, without the need of knowing H.

$$(2)$$
$$({\mathfrak{I}})$$

Importantly, the left-hand side of (3) is closely related to the ℓ1 norm of the gradients because

$$\sup_{\text{diagonal}H\succeq0,\text{Tr}(\mathbf{H})\leq1}\|\cdot\|_{\mathbf{H}}=\|\cdot\|_{\infty},\qquad\inf_{\text{diagonal}H\succeq0,\text{Tr}(\mathbf{H})\leq1}\|\cdot\|_{\mathbf{H},*}=\|\cdot\|_{1}.\tag{4}$$

We illustrate this fact in Fig. 1. In words, this means that the ℓ∞ *norm is the pointwise supremum of* all weighted ℓ2 norms induced by diagonal matrices with unit trace, whereas its dual, the ℓ1 *norm, is* the pointwise infimum of all the corresponding dual norms. Also, the unit ℓ∞ ball is the intersection of all unit balls for those ℓ2 norms, and the unit ℓ1 ball is the union of all dual unit balls.

Indeed, the duality between supremum of a class of primal norms and infimum of the corresponding dual norms in (4) is not just a coincidence, but rather a special property induced by the structure of the preconditioner set H = Dd+ for Adam. This property holds more generally for any wellstructured preconditioner set and we discuss the corresponding adaptive smoothness in Section 2.2.

$$2.2\quad A$$

## 2.2 Adaptive Smoothness Associated With Well-Structured Preconditioner

The following definition of well-structured preconditioner sets is proposed by Xie et al. (2025b) to unify the analysis of a broad family of adaptive optimizers with structured preconditioners.

Definition 2.1 (Well-structured preconditioner set). H ⊆ Sd+ *is said to be a* well-structured preconditioner set if H = S
d+ ∩ K for some matrix subalgebra1 K ⊆ Md with Id ∈ K.

As will be discussed in Section 3.1, many commonly used adaptive optimizers, including Adam, AdaGrad, and their variants, can be cast into the framework of a meta-algorithm (Algorithm 1)
with well-structured preconditioner sets. A specific case is H = Dd+, the set of all diagonal PSD
matrices, which is the running example in the previous subsection. For any such well-structured preconditioner set H, we have the duality between the supremum of the primal norms and the infimum of the corresponding dual norms, formalized in the following lemma.

Lemma 2.2. Let H ⊆ Sd+ *be any well-structured preconditioner set. Recall that its induced norm* is defined as *∥ · ∥*H = supH∈H,Tr(H)≤1 ∥ · ∥H*. Then it holds that*

$||\cdot||_{\mathcal{H},*}=\inf_{\mathbf{H}\in\mathcal{H},\mathrm{Tr}(\mathbf{H})\leq1}\|\cdot\|_{\mathbf{H},*}=\inf_{\mathbf{H}\in\mathcal{H},\mathrm{Tr}(\mathbf{H})\leq1}\|\cdot\|_{\mathbf{H}-1}$.  
Based on this fact, we can generalize the discussion in Section 2.1 to any well-structured preconditioner set H, showing that NSD and adaptive optimizers with preconditioner set H can exploit the geometry induced by ∥ · ∥ = *∥ · ∥*H via two different smoothness notions, the former being the standard smoothness under *∥ · ∥*H and the latter being the adaptive smoothness defined in Definition 2.4.

We proceed to introduce the adaptive smoothness associated with any well-structured preconditioner set H. We first review the standard smoothness notion under a general norm ∥·∥.

Definition 2.3. *For a loss function* f : R
d → R and any norm ∥·∥, we will use L∥·∥(f) to denote the smoothness of f with respect to ∥·∥, i.e., the smallest positive constant L *such that*
∥∇f(x) − ∇f(y)∥∗ ≤ L∥x − y∥ *for any* x, y.

When ∥ · ∥ = *∥ · ∥*H for some well-structured preconditioner set H, L∥·∥H
(f) is then the standard smoothness of f under the norm induced by H. In contrast, the adaptive smoothness associated with H is defined as the smallest smoothness of f under all norms *∥ · ∥*H induced by H ∈ H with Tr(H) ≤ 1, as formalized below. This term is introduced as H-smoothness in Xie et al. (2025b). We rename it to highlight this notion adapts to the structure of H, in contrast to the standard smoothness.

Definition 2.4 (Adaptive Smoothness, Xie et al. 2025b). The adaptive smoothness of a function f w.r.t. a well-structured preconditioner set H is defined as the smallest smoothness of f *under all*
∥ · ∥H for H ∈ H *with* Tr(H) ≤ 1*, that is,*
$$\Lambda_{\mathcal{H}}\left(f\right):=\operatorname*{min}_{\begin{array}{c}{{H\in\mathcal{H}}}\\ {{\mathrm{Tr}(H)\leq1}}\end{array}}L_{\|\cdot\|_{H}}(f)=\operatorname*{min}_{\begin{array}{c}{{H\in\mathcal{H}}}\\ {{\forall\mathbf{x},-H\preceq\nabla^{2}f(\mathbf{x})\preceq H}}\end{array}}\mathrm{Tr}(\mathbf{H}).$$
Tr(H). (5)
$$(S)$$

Algorithm 1 General Adaptive Optimization Algorithm Hyperparam: ϵ ≥ 0, total steps T, learning rate η, convex cone *H ⊂ S*+, decay factor β Input: initialization x0, stochastic loss functions {ft}
T
t=1 : R
d → R
M−1 ← 0 for t = 0, 1, · · · , T − 1 :
gt ← ∇ft(xt)

Mt−1 + gtg
⊤
t, Cumulative variant, β Mt−1 + (1 − β) gtg
⊤
t, EMA variant, β Mt−1 + gtg
⊤
t, Weighted variant.

Vt ← arg minH∈H Mt + ϵId, H−1+ Tr(H)
xt+1 ← xt − ηV
−1 t gt return xT
Mt ←

In the deterministic convex setting, it has been shown by Xie et al. (2025b) that the convergence rate of an adaptive optimizer with any well-structured preconditioner set H is of order O(ΛH(f) *∥X ∥*2H /T). In Section 3, we extend such characterization to the nonconvex setting, demonstrating that the adaptive smoothness ΛH(f) governs the convergence behavior of any adaptive optimizer with well-structured preconditioner set H. Comparison between two smoothness notions. For any H ∈ H with Tr(H) = 1, it always holds ∥x − y∥H ≥ ∥x − y∥H and ∥∇f(x) − ∇f(y)∥H,∗ ≤ ∥∇f(x) − ∇f(y)∥H,∗
. Therefore,

$$L_{\parallel:\parallel_{H}}(f)=\sup_{\mathbf{x},\mathbf{y}}\frac{\|\nabla f(\mathbf{x})-\nabla f(\mathbf{y})\|_{H,*}}{\|\mathbf{x}-\mathbf{y}\|_{H}}\geq\sup_{\mathbf{x},\mathbf{y}}\frac{\|\nabla f(\mathbf{x})-\nabla f(\mathbf{y})\|_{\mathcal{H},*}}{\|\mathbf{x}-\mathbf{y}\|_{\mathcal{H}}}=L_{\parallel:\parallel_{\mathcal{H}}}(f).$$

Minimizing over H ∈ H with Tr(H) = 1 then yields ΛH (f) = L∥·∥H (f) ≥ L∥·∥H
(f). In other words, as a condition, the adaptive smoothness is arguably stronger than the standard smoothness.

But they can differ by at most a multiplicative factor of d, as summarized in Proposition 2.5.

Proposition 2.5. For any well-structured preconditioner set H ⊆ Sd+ *and any loss function* f :
R

d → R, it always holds that L∥·∥H
(f) ≤ ΛH (f) ≤ d · L∥·∥H
(f).

## 3 Unified Analysis In The Nonconvex Setting

In the nonconvex setting, we establish a *unified* analysis that encompasses a broad family of adaptive optimization algorithms. Our result highlights how the convergence behavior of these methods depends critically on the notion of adaptive smoothness.

## 3.1 Adaptive Optimizers With Well-Structured Preconditioner Sets

We adopt the framework in Gupta et al. (2017) and Xie et al. (2025b) to describe adaptive optimizers in a unified way, as displayed in Algorithm 1. This meta-algorithm is flexible in two aspects: the way of aggregating past gradients and the choice of preconditioner set H. First, there are three different ways to aggregate the past gradients in Algorithm 1, each of which is presented in a separate algorithm block in Appendix D.1. The cumulative and EMA variants are the most common ways, and they are indeed equivalent to the weighted variant up to hyperparameter transformations. Therefore, it suffices to study the weighted variant, and the results for the other two variants follow as corollaries. Another flexibility of Algorithm 1 comes from the choice of convex cone H. More specifically, Algorithm 1 recovers several standard optimizers by specifying H as follows:
- H = {all diagonal PSD matrices} recovers AdaGrad and Adam.

- H = {c Id | c > 0} recovers AdaGrad-Norm and AdaSGD (Wang & Wiens, 2020).

- H = S
d+ recovers full-matrix AdaGrad (Duchi et al., 2011).

- H = S
dL
+ ⊗IdR yields one-sided Shampoo/ASGO recently proposed by (Xie et al., 2025a; An et al., 2025)
In particular, based on the notion of well-structured preconditioner sets defined in Definition 2.1, Xie et al. (2025b) develops a unified convergence analysis for Algorithm 4 in the convex setting, and the convergence rate depends on the adaptive smoothness with respect to H defined in Definition 2.4.

Additional notations. We define PH(M) := arg minH∈H M, H−1+ Tr(H) for any M ∈
S
d++. Then in Algorithm 1, Vt = PH(Mt) and Lemma B.4 will show that PH(M)
2is the projection of M onto H. Specifically, when H contains all the PSD matrices, Vt is M
1 2 t.

3.2 CONVERGENCE RATE IN THE DETERMINISTIC NONCONVEX SETTING Here we only present results for the deterministic case to highlight the role of adaptive smoothness, and the complete results for the (stochastic) nonconvex setting and corresponding proofs can be found in Appendix D.2. We first present the convergence guarantee for the weighted variant of Algorithm 1 in the following theorem.

Theorem 3.1. For any ϵ ≥ 0, β ∈ (0, 1], η > 0, and T ∈ N*, let* {xt}
T
t=0 be the iterates of Algorithm 1 with well-structured preconditioner set H, where the update of Mt *follows* the weighted version*, i.e.,* Mt = βMt−1 + gtg
⊤
tfor all t ∈ [T]. Let ΛH (f) be the adaptive smoothness of the loss f according to Definition 2.4. Then when ft ≡ f*, it holds that*

$$\frac{1}{T}\sum_{t=0}^{T-1}\|\nabla f(\mathbf{x}_{t})\|_{\mathcal{H},*}\leq\frac{\sqrt{\sum_{i=0}^{T-1}\beta^{i/2}}}{T}\xi+\frac{\sqrt{d}\epsilon^{1/4}}{\sqrt{T}}\sqrt{\xi}.$$  $\frac{2\Delta_{0}}{\eta}+\eta\Lambda_{\mathcal{H}}\left(f\right)\|\mathbf{S}_{T}\|_{\text{op}}$_and_ $\mathbf{S}_{T}=\mathbb{E}\sum_{t=0}^{T-1}\mathbf{V}_{t}^{-1}(\mathbf{V}_{t}^{2}-\beta\mathbf{V}_{t-1}^{2})\mathbf{V}_{t}^{-1}$_._

 $where\;\xi=\frac{2\Delta_0}{\eta}+1$  . 
For general well-structured preconditioner set, ∥ST ∥op = O˜ (log(d)[(1 − β)*T /β* + log(d)])*. When* the preconditioner set only has diagonal matrices, ∥ST ∥op = (1 − β)T + O˜ (1).

The above result for the weight variant can be converted to guarantees for the cumulative and EMA variants. Specifically, the cumulative variant is equivalent to weighted accumulation with β = 1 while the EMA variant with learning rate η E and stability constant ϵ E produces identical iterates as weighted accumulation with ηW = η E/
√1 − β and ϵW = ϵ E/(1−β). Below we present the result for the cumulative variant in Theorem 3.2, and the result for the EMA variant is in Theorem D.8.

Theorem 3.2. For any ϵ ≥ 0, η > 0, and T ∈ N*, let* {xt}
T
t=0 be the iterates of Algorithm 1 with well-structured preconditioner set H, where the update of Mt *follows* the cumulative version*, i.e.,*
Mt = Mt−1 +gtg
⊤
tfor all t ∈ [T]. Let ΛH (f) be the adaptive smoothness of the loss f *according* to Definition 2.4. Then when ft ≡ f*, it holds that*

$$\frac{1}{T}\sum_{t=0}^{T-1}\|\nabla f(\mathbf{x}_{t})\|_{\mathcal{H},*}\leq\frac{1}{\sqrt{T}}\left(\xi+\sqrt{d}\epsilon^{1/4}\sqrt{\xi}\right).$$
$\frac{\Delta_{0}}{\eta}+\eta\cdot\Lambda_{\mathcal{H}}\left(f\right)\log^{2}d$). Moreover, when setting $\eta=\Lambda_{\mathcal{H}}\left(f\right)\log d$).  
_where $\xi=\tilde{O}\left(\frac{\Delta_{0}}{\eta}+\eta\cdot\Lambda_{H}\right)$_  $\xi=\tilde{O}(\sqrt{\Delta_{0}\cdot\Lambda_{H}\left(f\right)}\log d)$
ΛH(f) log2 d
, it holds that
 $\eta=\sqrt{\frac{\Delta_0}{\Delta y\left(f\right)\log^2}}$
At a high level, Theorem 3.2 shows that, with appropriate hyperparameters, Algorithm 1 with any well-structured preconditioner set H achieves a convergence rate of order O˜(log d·p∆0ΛH(f)/T)
on deterministic nonconvex functions where O˜(·) hides logarithmic factors in problem parameters other than the dimension d. This result illustrates that the adaptive smoothness ΛH (f) governs the convergence rate of adaptive optimizers in the nonconvex setting, complementing previous results for the convex setting in Xie et al. (2025b). Moreover, we remark that when H contains only diagonal matrices, the log d factor disappears, recovering the bounds in Xie et al. (2025a). It is worth noticing that the convergence guarantees in the above two theorems are concerned with
∥∇f(xt)∥H,∗
depending on specific H rather than ∥∇f(xt)∥2
. For the specific case of Adam where H is the set of all diagonal PSD, this becomes a guarantees in terms of ℓ1 norm of the gradients, as we discussed in Section 2.1. On the other hand, Pethick et al. (2025); Kovalev (2025a) show that NSD achieves O(
∆0L*∥·∥H*
(f)
T)
1 2in the deterministic case. Taken together, these two rates suggest that adaptive optimizers and NSD exploit different smoothness notions in the nonconvex setting.

## 3.3 Technical Contribution: A Novel Matrix Inequality

Previous theoretical results for one-sided Shampoo/ASGO (Algorithm 7) and other well-structured preconditioners primarily focus on convex objectives (Xie et al., 2025b; Kovalev, 2025a). In the nonconvex regime, existing convergence analyses apply essentially when the preconditioner set contains only diagonal matrices2(Xie et al., 2025a). In contrast, we show the first unified convergence analysis that applies to any general well-structured preconditioner set, well beyond the diagonal cases. A central difficulty in our analysis is the extension from diagonal preconditioners to a general preconditioner set H. In the diagonal case, the proof basically decomposes to entry-wise analyses, and scalar telescoping readily yields the desired bounds. However, for general H, *noncommutativity* prevents such simplification, and bounding the second-order terms requires handling delicate matrix inequalities. Our resolution of this challenge yields a key technical contribution, formalized below.

Lemma 3.3. Let ϵ ≥ 0 and β ∈ (0, 1]. For any T ∈ N, consider any sequence of vectors g0*, . . . ,* gT −1 ∈ R
d*. Let* M−1 = 0*, and recursively define* Mt = βMt−1 + gtg
⊤
tfor t = 0, . . . , T − 1. For any well-structured preconditioner set H*, define* Vt = arg minH∈H⟨Mt +
ϵId, H−1⟩ + Tr(H) for each t ∈ [T − 1]. Then for any H ∈ H ∩ Sd++*, it holds*

$$\sum_{t=0}^{T-1}\|{\mathbf{V}_{t}}^{-1}{\mathbf{g}}_{t}\|_{H}^{2}\leq\mathrm{Tr}({\mathbf{H}})\left\|{\mathbf{S}_{T}}\right\|_{\mathrm{op}}\quad\text{where}{\mathbf{S}_{T}}=\sum_{t=0}^{T-1}{\mathbf{V}_{t}}^{-1}\left({\mathbf{V}_{t}}^{2}-\beta{\mathbf{V}_{t-1}^{2}}\right){\mathbf{V}_{t}}^{-1}.$$
$=\;e\;\;1\;\;\pi r$ . 
Moreover, there exists an absolute constant C1, C2 > 0, independent of d, T, ϵ, β and H*, such that*

$$\|\mathbf{S}_{T}\|_{\rm op}\leq C_{1}\left(1+\log\left(1+\frac{d}{\epsilon}\sum_{t=0}^{T-1}\|\mathbf{g}_{t}\|_{2}^{2}+d^{2}(1-\beta)T\right)\right)\left(\frac{1-\beta}{\beta}T+\log\|\mathbf{V}_{T-1}^{2}/\epsilon\|_{\rm op}\right)+C_{2}.$$
$\square$
$\mathbf{a}=\mathbf{a}\mathbf{b}+\mathbf{a}\mathbf{b}$. 
In the special case when H *is commutative, the above bound can be further improved to*

$$\|\mathbf{S}_{T}\|_{\mathrm{op}}\leq(1-\beta)T+\log\|\mathbf{V}_{T-1}^{2}/\epsilon\|_{\mathrm{op}}.$$

Specializing gt's to be the gradients, Lemma 3.3 provides a general upper bound on the sum of second-order terms. It highlights the essential gap between diagonal and general preconditioner sets: noncommutativity introduces an additional log d factor, making the dependence strictly worse than in the diagonal case. Nevertheless, this is the first bound that applies to arbitrary well-structured preconditioner sets, and it plays a central role in extending convex analyses to the nonconvex setting.

The proof of Lemma 3.3 can be found in Appendix C. A key step is to establish a novel matrix inequality Lemma C.1 that relates the difference between two positive definite matrices to the difference between their logarithms, which may be of independent interest.

## 4 Benefit Of Adaptive Geometry

We have shown in Section 3 that the nonconvex convergence rate of adaptive optimizers relies critically on the adaptive smoothness of the loss, and the bound is worse than that of the corresponding NSD. This naturally raises the concern in Question 2: does the stronger adaptive smoothness lead to stronger results? In this section, we address this question from two complementary angles:
1. Under the adaptive smoothness assumption, adaptive optimizers can achieve faster convergence rates on convex functions via Nestrov acceleration.

2. The distinction between standard smoothness and adaptive smoothness mirrors a parallel separation in the assumptions on gradient noise.

At a high level, these two angles share the same underlying mechanism: *Under non-Euclidean* geometry, averaging might not be effective in reducing the norm, which we will explain below.

7

| Algorithm 2 Accelerated Adaptive Algorithm   | Algorithm 3 NSD with momentum   |
|----------------------------------------------|---------------------------------|
| Hyperparam: ϵ ≥ 0, total steps T, learning rate η, convex cone H ⊆ S+ Input: initial x0, constants α0, . . . , αT ∈ (0, 1] x¯0 ← x0, M−1 ← 0 for t = 0, 1, . . . , T − 1 : gt ← ∇f t (xt) where f αt,x¯t t is in (8) αt,x¯t Mt ← Mt−1 + gtg ⊤ t Mt + ϵId, H−1  + Tr(H) Vt ← arg min H∈H Hyperparam: ϵ ≥ 0, total steps T, learning rate η, norm ∥ · ∥, averaging parameter α Input: initialization x0, initialization m0, stochastic loss functions {ft} T t=1 for t = 0, 1, · · · , T − 1 : gt ← ∇ft(xt) mt ← (1 − α)mt−1 + αgt ut ← arg max ⟨mt,u⟩ ∥u∥≤1 xt+1 ← xt − ηV t gt −1 x¯t+1 ← αtxt+1 + (1 − αt)x¯t return x¯T xt+1 ← xt − ηut return xT                                              |                                 |

4.1 ADAPTIVE VARIANCE: AN ANALOGUE OF ADAPTIVE SMOOTHNESS Our main results in this section are concerned about accelerated adaptive algorithms for convex functions and NSD in the nonconvex setting. Before presenting these results, we introduce a key technical ingredient: *adaptive variance*, a quantity that serves as the analogue of adaptive smoothness for gradient noise.

Definition 4.1 (Standard and adaptive gradient variance). For an index set T , let {ft}t∈T be a set of stochastic loss functions where each ft : R
d → R.

- For any norm *∥ · ∥* on R

$\|\ \text{on}\ \mathbb{R}^{d}$, the gradient variance of $\{f_{t}\}_{t\in\mathcal{T}}$ under $\|\cdot\|$ is defined as $\sigma_{\|\cdot\|}(\{f_{t}\}_{t\in\mathcal{T}})^{2}:=\sup\limits_{t\in\mathcal{T},\mathbf{x}\in\mathbb{R}^{d}}\mathbb{E}\big{[}\big{|}|\nabla f_{t}(\mathbf{x})-\mathbb{E}[\nabla f_{t}(\mathbf{x})]\big{|}\big{|}^{2}\big{]}$
$$(\mathbf{6})$$

2(6)
- The adaptive gradient variance of {ft}t∈T *w.r.t. any well-structured preconditioner set* H is σH({ft}t∈T )
2 = min H∈H,Tr(H)≤1 sup t∈T ,x∈Rd E

-∇ft(x) − E[∇ft(x)]
2 H−1
. (7)
This adaptive variance is inspired by the noise assumption in Kovalev (2025a), both capturing the overall variation of gradient noise in the geometry induced by H. Compared with the traditional noise assumption that only characterizes ℓ2 norm variance, adaptive variance provides a more informative measure. In addition, analogous to the comparison between ΛH (f) and L∥·∥H
(f), the adaptive variance is always no smaller than ∥·∥H,∗
-variance, as formalized in Proposition B.11.

Here we can compare Definition 4.1 with bounded covariance assumption in Xie et al. (2025b); An et al. (2025) that there exists Σ ⪰ 0 such that E[∇f(xt) − ∇ft(xt)][∇f(xt) − ∇ft(xt)]⊤ ⪯ Σ.

When the covariance matrix is upper bounded by Σ, σH ≤ Tr(PH(Σ)) for general H as shown in Proposition B.10. On the other hand, Definition 4.1 doesn't require the existence of Σ that can upper bound the covariance matrix everywhere. Therefore, Definition 4.1 is a weaker assumption than the bounded covariance assumption.

## 4.2 Acceleration On Convex Functions

We follow the framework in Kovalev (2025a) to formulate a unified class of adaptive optimizers with well-structured preconditioner sets with Nesterov acceleration in Algorithm 2. Through the perspective introduced by Kovalev & Borodich (2024), the idea is to interpret each step of Nesterov acceleration as a single step of standard gradient on a modified loss, i.e., f αt,x¯tin Eq. 8. Here for a constant α ∈ (0, 1] and a reference point x¯, the corresponding modified loss is defined as

$$f^{\alpha,\bar{\mathbf{x}}}(\mathbf{x}):=\alpha^{-2}f(\alpha\mathbf{x}+(1-\alpha)\bar{\mathbf{x}}).$$
−2f(αx + (1 − α)x¯). (8)
For stochastic convex functions satisfying Assumption 4.2, we establish the convergence rate of Algorithm 2 in the following Theorem 4.3, whose proof is in Appendix E.

Assumption 4.2. Let f be a convex loss function. For all t ∈ [T] *and any* x, E[∇ft(x)] = ∇f(x).

$$({\mathfrak{s}})$$

Theorem 4.3. For a well-structured preconditioner set H, let f be a convex loss function whose H-smoothness constant is ΛH (f) ∈ (0, ∞) according to Definition 2.4. For ϵ > 0, T > 0, consider Algorithm 2 *with* αt = 2/(t + 2) for t = 0, 1, . . . , T − 1*. Suppose* x
∗*is the global minima and* maxt=0,1*,...,T* −1 ∥xt−x
∗∥H ≤ D for some D > 0 and Assumption 4.2 holds with adaptive gradient variance σH({ft}
T
t=1)
2 ≤ σ 2H for some σH ∈ [0, ∞)*. Then it holds that*

$$\mathbb{E}[f(\bar{\mathbf{x}}_{T})-f(\mathbf{x}^{*})]\leq\frac{2D^{2}\epsilon}{\eta(T+1)^{2}}\mathbb{E}\,\text{Tr}(V_{T-1}^{-1})+\left(\frac{D^{2}}{2\eta}-\frac{\eta}{2}\right)\mathbb{E}\frac{4}{(T+1)^{2}}\sum_{t=0}^{T-1}\mathbf{g}_{t}^{\top}\mathbf{V}_{t}^{-1}\mathbf{g}_{t}$$ $$+\frac{2\eta^{2}}{(T+1)^{2}}\cdot\Lambda_{\mathcal{H}}\left(f\right)\cdot\tilde{O}(\log^{2}d)+\frac{\eta}{T^{1/2}}\sigma_{\mathcal{H}}\cdot\tilde{O}(\log d).$$  $\mathbf{\alpha}$\(\mathbf{\alpha}  

_Moreover, when choosing learning rate $\eta=D$, the convergence rate becomes_  $$\mathbb{E}[f(\bar{\mathbf{x}}_{T})-f(\mathbf{x}^{*})]=\tilde{O}\bigg{(}\frac{\Lambda_{\mathcal{H}}\left(f\right)D^{2}\log^{2}d+d\sqrt{\epsilon}D}{T^{2}}+\frac{\sigma_{\mathcal{H}}D\log d}{\sqrt{T}}\bigg{)}.$$

Remark 4.4. A drawback of Algorithm 2 is that the optimal choice of learning rate η in Theorem 4.3 depends on the unknown parameter D = maxt ∥xt − x
∗∥H. To circumvent this issue, we follow the approach in Kovalev (2025a) to introduce a projected variant (see Algorithm 8 and discussion in Appendix E.2), which ensures that all iterates remain inside a ∥·∥H-ball of radius D*. The removes* the requirement for prior knowledge of D, and we establish a same convergence rate in Theorem *E.5.*
Our convergence guarantee attains a deterministic component of order O˜(ΛH (f) D2/T2), an accelerated rate governed by the adaptive smoothness ΛH (f). In comparison, Guzman & Nemirovski ´
(2015) shows that any first order optimizer can only achieve Ω( 
L*∥·∥∞*(f)
T log T
) for the specific case of ℓ∞ norm smoothness. Taken together, these results show that the adaptive smoothness is necessary to achieve the acceleration, which can't be replaced by the weaker non-Euclidean smoothness, highlighting its algorithmic benefit and answering Question 2 in the affirmative. The analysis in Kovalev (2025a) yields similar results when H contains only diagonal matrices. However, their Assumption 4, which is critical for their analysis, imposes restrictive conditions on both the loss and the gradient noise for more general H. By contrast, our approach avoids such requirements by leveraging Lemma 3.3.

## 4.3 Nonconvex Results For Nsd Under Adaptive Noise Assumption

The ineffectiveness of averaging in the dual space also leads to difficulty in reducing gradient variance via averaging. To illustrate this, consider i.i.d. random vectors x1*, . . . ,* xn with E[xi] = 0 and E[∥xi∥
2 2] ≤ σ 2, we have E[∥
1 n Pn i=1 xi∥
2 2] ≤
σ 2 n while E[∥
1 n Pn i=1 xi∥
2 1] ≤ d σ 2 n
, and the extra d factor in the latter bound can be tight. This causes the dimension-dependent convergence rate of NSD in the nonconvex setting, as shown in recent works (Pethick et al., 2025; Kovalev, 2025b).

In particular, the dimension-dependent factor ρ = supx
∥x∥H,∗
∥x∥2captures the mismatch between

∥·∥H,∗ and ∥·∥2. For diagonal H, NSD reduces to SignGD with ∥·∥H,∗ = ∥·∥1, yielding ρ = Θ(√d)
and vacuous bounds when T ≪ d. We avoid this by using the adaptive variance assumption. We
prove the stochastic rate of NSD in Theorem 4.5 and the proof is deferred to Appendix F.1. The concurrent work Kovalev & Borodich (2025) also leveraged the adaptive variance assumption to prove a dimension-free nonconvex convergence rate of NSD. However, they used a smoothness metric similar to adaptive smoothness while our Theorem 4.5 uses the standard smoothness. Our rate is strictly better because of the relationship between standard smoothness and adaptive smoothness. Theorem 4.5. Let H be a well-structured preconditioner set. For any ϵ ≥ 0, α ∈ (0, 1), η > 0,
and T ∈ N*, let* {xt}
T
t=0 be the iterates of Algorithm 3 with ∥ · ∥ = ∥ · ∥H and m0 = ∇f0(x0).
Let L∥·∥H
(f) be the smoothness of the loss f w.r.t. ∥·∥H according to Definition *2.3. Suppose*
_Assumption 4.2 holds with adaptive gradient variance $\sigma_{\mathcal{H}}(\{f_{t}\}_{t=1}^{T})^{2}\leq\sigma_{\mathcal{H}}^{2}$. Then for_  $$\mathbb{E}\frac{1}{T}\sum_{t=0}^{T-1}\|\nabla f(\mathbf{x}_{t})\|_{\mathcal{H},*}\leq\frac{\Delta_{0}}{\eta T}+\frac{2\eta}{\alpha}L_{\|.\|}(f)+\frac{2\sigma_{\mathcal{H}}}{\alpha T}+2\sigma_{\mathcal{H}}\sqrt{\alpha}.$$  _Let $a_{0}=\sqrt{\Delta_{0}L_{\|.\|}(f)}/\sigma_{\mathcal{H}}$. If $a_{0}<1$, then_
H*. Then it holds that*

* _When $T<a_{0}^{-6}$, we choose $\alpha=T^{-2/3}$ and $\eta=\sqrt{\frac{\Delta_{\alpha}}{L_{\frac{1}{2}\left(\lambda\right)}T}}\,T^{-5/12}$. The rate is $O\left(\sigma_{\mathcal{H}}T^{-1/3}\right)$._ * _When $T\geq a_{0}^{-6}$, we choose $\alpha=\frac{a_{0}}{\sqrt{T}}$, $\eta=\frac{\Delta_{\alpha}^{2/4}\eta-3/4}{L_{\frac{1}{2}\left(\lambda\right)}T^{1/4}\sigma_{\mathcal{H}}^{2/4}}$. The rate is $O\left(\frac{\left(\Delta_{\alpha}L_{\frac{1}{2}\left(\lambda\right)}(f)\right)^{1/4}\sqrt{\sigma_{\mathcal{H}}}}{T^{1/4}}\right)$. If $a_{0}\geq1$, then_
_by $a_{0}\geq1$, then_  * _When $T\leq a_{0}^{2}$, we choose $\alpha=1$ and $\eta=\sqrt{\frac{\Delta_{0}}{L_{\uparrow\downarrow_{\eta}}(T)}}T^{-1/2}$. The rate is $O\left(\sqrt{\Delta_{0}L_{\uparrow\downarrow_{\eta_{k}}}(f)}T^{-1/2}\right)$._ * _When $T\geq a_{0}^{2}$, we choose $\alpha=\frac{a_{0}}{\sqrt{T}}$ and $\eta=\frac{\Delta_{0}^{1/4}(T^{-2/4}}{L_{\uparrow\downarrow_{\eta_{k}}}(f)^{1/4}\sigma_{\eta_{k}^{1/4}}^{1/4}}$. The rate is $O\left(\frac{(\Delta_{0}L_{\uparrow\downarrow_{\eta_{k}}(f)})^{1/4}\sqrt{\sigma_{\eta_{k}}}}{T^{1/4}}\right)$._
Theorem 4.5 shows that, with appropriate choices of the learning rate η and averaging parameter α, NSD achieves a dimension-free rate depending only on the standard smoothness L∥·∥H
(f) and the adaptive variance σH, thereby avoiding the unfavorable ρ factor. Next Theorem 4.6 shows that such a dimension-free upper bound is unattainable under the standard gradient variance assumption.

Theorem 4.6. For any ϵ ≥ 0, α ∈ (0, 1), η > 0, and T ∈ N*, let* {xt}
T
t=0 *be the iterates* of Algorithm 3 with any norm ∥·∥. Suppose Assumption 4.2 *holds with the gradient variance* σ∥·∥∗
({ft}
T
t=1)
2 ≤ σ 2
∥·∥∗
for some σ∥·∥∗
∈ [0, ∞)*. Then it holds that*

$$\mathbb{E}\frac{1}{T}\sum_{t=0}^{T-1}\|\nabla f(\mathbf{x}_{t})\|_{*}\leq\frac{\Delta_{0}}{\eta T}+\frac{2\eta}{\alpha T}\sigma_{\|\cdot\|_{*}}+2\sigma_{\|\cdot\|_{*}}\cdot\min\left(1,\alpha^{1/2}\psi(\|\cdot\|_{*},\|\cdot\|_{2})\right),$$  _where $\psi(\|\cdot\|_{*},\|\cdot\|_{2})=\sup_{\mathbf{x}}\frac{\|\mathbf{x}\|_{*}}{\|\mathbf{x}\|_{2}}\cdot\sup_{\mathbf{x}}\frac{\|\mathbf{x}\|_{2}}{\|\mathbf{x}\|_{*}}$ measures the distortion between the two norms._

Here the norm distortion ψ(∥· ∥∗, *∥· ∥*2) can grow with the dimension d. Consequently, Theorem 4.6 gives two kinds of upper bound for the convergence rate of NSD:
- When T is small (e.g. *T < d*), the constant term σ∥·∥∗
dominates.

- For sufficiently large T and small α (for which the four terms are balanced), the last term in the upper bound becomes 2σ∥·∥∗ α 1/2ψ(∥ · ∥∗, *∥ · ∥*2), which depends on d.

Moreover, such a dependence on d is inevitable in the worst case as shown by Theorem 4.7.
Theorem 4.7. For any fixed ∆0, L, σ2, d, T, learning rate η, and any averaging parameter α*, there* exists a loss function f : R
d → R, a sequence of stochastic iid loss functions f0, f1, · · · , fT −1 and
an initialization x0 *satisfying the following conditions:(1)* f(x0)−infx f(x) = ∆0 and L∥·∥∞
(f) ≤
L*; (2) For any* x ∈ R
d*, it holds that* E[∇ft(x)] = ∇f(x) and E[∥∇ft(x) − ∇f(x)∥
21
] ≤ σ
2.
When running Algorithm 3 with ∥ · ∥ = ∥ · ∥∞, learning rate η, averaging parameter α and initialization x0 = 0*, it holds that*
_it holds that_  $$\mathbb{E}\Big{[}\min_{t\in[T]}\|\nabla f(\mathbf{x}_{t})\|_{1}\Big{]}=\min\{e^{-2}5^{-\frac{1}{4}}(dL\Delta_{0}\sigma^{2})^{\frac{1}{4}}T^{-\frac{1}{2}},e^{-2}5^{-\frac{1}{2}}\sigma\}$$
Theorem 4.7 also shows two kinds of lower bound we can achieve on signGD with momentum:
- When T is not large enough, we can achieve the lower bound Ω(σ), which shows the hardness induced by the stochasticity and matches the first upper bound in Theorem 4.6.

- On the other hand, if we want to achieve the error ϵ < e−25
− 12 σ, we require the number of steps T = Ω(ϵ
−2(dL∆0σ 2)
1 2 ), whose dependence on dimension d is Ω(d 1 2 ).

In conclusion, under the standard gradient variance assumption with ∥· ∥ = *∥· ∥*∞ and ∥· ∥∗ = *∥· ∥*1, the d-dependent rate in Theorem 4.6 is unavoidable. In contrast, Theorem 4.5 attains a dimensionfree rate under the adaptive gradient variance assumption, highlighting a fundamental gap.

## 5 Conclusion

We extend the unified analysis of adaptive optimizers to nonconvex functions, establishing convergence rate depending on the adaptive smoothness. It strengthens the comparison between smoothnesses that adaptive optimizers and NSD use in the convex settings. We further show the benefit of adaptive smoothness by showing the accelerated rate of adaptive optimizers with Nesterov momentum. The benefit of adaptive geometry is also justified by comparing two kinds of noise.

## References

Zeyuan Allen-Zhu and Lorenzo Orecchia. Linear coupling: An ultimate unification of gradient and mirror descent. *arXiv preprint arXiv:1407.1537*, 2014. 15 Kang An, Yuxing Liu, Rui Pan, Yi Ren, Shiqian Ma, Donald Goldfarb, and Tong Zhang. Asgo:
Adaptive structured gradient optimization. *arXiv preprint arXiv:2503.20762*, 2025. 1, 5, 8, 15 Lukas Balles, Fabian Pedregosa, and Nicolas Le Roux. The geometry of sign gradient descent.

arXiv preprint arXiv:2002.08056, 2020. 1, 3, 15 Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology. *arXiv preprint* arXiv:2409.20325, 2024. 1 Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. *arXiv preprint arXiv:2401.02954*, 2024. 1 Lizhang Chen, Jonathan Li, and Qiang Liu. Muon Optimizes Under Spectral Norm Constraints.

arXiv preprint arXiv:2506.15054, 2025. 15 Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, et al. Symbolic discovery of optimization algorithms. arXiv preprint arXiv:2302.06675, 2023. 1 Xiang Cheng, Fred Roosta, Stefan Palombo, Peter Bartlett, and Michael Mahoney. FLAG n'FLARE:
Fast Linearly-Coupled Adaptive Gradient Methods. In *International Conference on Artificial* Intelligence and Statistics, pp. 404–414. PMLR, 2018. 15 Sinho Chewi, Sebastien Bubeck, and Adil Salim. On the complexity of finding stationary points ´
of smooth functions in one dimension. In *International Conference on Algorithmic Learning* Theory, pp. 358–374. PMLR, 2023. 45 Ashok Cutkosky. Anytime online-to-batch, optimism and acceleration. In International conference on machine learning, pp. 1446–1454. PMLR, 2019. 15 Ashok Cutkosky and Harsh Mehta. Momentum improves normalized sgd. In International conference on machine learning. PMLR, 2020. 15 Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024. 1 John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. *Journal of machine learning research*, 2011. 5 Alina Ene, Huy L Nguyen, and Adrian Vladu. Adaptive gradient methods for constrained convex optimization and variational inequalities. In Proceedings of the AAAI Conference on Artificial Intelligence, 2021. 15 Kevin Frans, Pieter Abbeel, and Sergey Levine. What Really Matters in Matrix-Whitening Optimizers? *arXiv preprint arXiv:2510.25000*, 2025. 15 Vineet Gupta, Tomer Koren, and Yoram Singer. A unified approach to adaptive regularization in online and stochastic optimization. *arXiv preprint arXiv:1706.06569*, 2017. 5, 38 Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In *International Conference on Machine Learning*, 2018. 15 Cristobal Guzm ´ an and Arkadi Nemirovski. ´ On lower complexity bounds for large-scale smooth convex optimization. *Journal of Complexity*, 31(1):1–14, 2015. 2, 9 Wei Jiang, Dingzhi Yu, Sifan Yang, Wenhao Yang, and Lijun Zhang. Improved Analysis for Signbased Methods with Momentum Updates. *arXiv preprint arXiv:2507.12091*, 2025. 15 Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. 1 Pooria Joulani, Anant Raj, Andras Gyorgy, and Csaba Szepesvari. A simpler approach to accelerated optimization: iterative averaging meets optimism. In International conference on machine learning. PMLR, 2020. 15 Ali Kavis, Kfir Y Levy, Francis Bach, and Volkan Cevher. Unixgrad: A universal, adaptive algorithm with optimal guarantees for constrained optimization. *Advances in neural information processing* systems, 32, 2019. 15 Jonathan A Kelner, Yin Tat Lee, Lorenzo Orecchia, and Aaron Sidford. An almost-linear-time algorithm for approximate max flow in undirected graphs, and its multicommodity generalizations. In Proceedings of the twenty-fifth annual ACM-SIAM symposium on Discrete algorithms, pp. 217– 226. SIAM, 2014. 15 Dmitry Kovalev. SGD with Adaptive Preconditioning: Unified Analysis and Momentum Acceleration. *arXiv preprint arXiv:2506.23803*, 2025a. 6, 7, 8, 9, 34 Dmitry Kovalev. Understanding gradient orthogonalization for deep learning via non-euclidean trust-region optimization. *arXiv preprint arXiv:2503.12645*, 2025b. 9, 15, 41 Dmitry Kovalev and Ekaterina Borodich. On linear convergence in smooth convex-concave bilinearly-coupled saddle-point optimization: Lower bounds and optimal algorithms. arXiv preprint arXiv:2411.14601, 2024. 8 Dmitry Kovalev and Ekaterina Borodich. Non-Euclidean SGD for Structured Optimization: Unified Analysis and Improved Rates. *arXiv preprint arXiv:2511.11466*, 2025. 9 Tim Tsz-Kit Lau, Qi Long, and Weijie Su. PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective. *arXiv preprint arXiv:2505.21799*, 2025. 15 Huan Li, Yiming Dong, and Zhouchen Lin. On the O(
√d K1/4 ) Convergence Rate of AdamW Measured by ℓ1 Norm. *arXiv preprint arXiv:2505.11840*, 2025. 32 Elliott H. Lieb. Convex trace functions and the wigner-yanase-dyson conjecture. Advances in Mathematics, 11(3):267–288, 1973. 17 Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982, 2025. 1 Devyani Maladkar, Ruichen Jiang, and Aryan Mokhtari. Convergence Analysis of Adaptive Gradient Methods under Refined Smoothness and Noise Assumptions. *arXiv preprint* arXiv:2406.04592, 2024. 1, 3 Depen Morwani, Itai Shapira, Nikhil Vyas, Eran Malach, Sham Kakade, and Lucas Janson. A New Perspective on Shampoo's Preconditioner. *arXiv preprint arXiv:2406.17748*, 2024. 15 Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher. Training deep learning models with norm-constrained lmos. arXiv preprint arXiv:2502.07529, 2025. 1, 6, 9, 15 Maria-Eleni Sfyraki and Jun-Kun Wang. Lions and muons: Optimization via stochastic frank-wolfe.

arXiv preprint arXiv:2506.04192, 2025. 15 Ishaan Shah, Anthony M Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, Anil Thomas, Ashish Tanwer, Darsh J Shah, et al. Practical efficiency of muon for pretraining. *arXiv preprint arXiv:2505.02222*, 2025. 1 Chongjie Si, Debing Zhang, and Wei Shen. Adamuon: Adaptive muon optimizer. arXiv preprint arXiv:2507.11005, 2025. 15