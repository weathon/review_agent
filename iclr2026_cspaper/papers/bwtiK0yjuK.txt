# Change Point Localization And Inference In Dynamic Multilayer Networks

Fan Wang School of Mathematics and Statistics University of Melbourne fan.wang.2@unimelb.edu.au Yik Lun Kei Department of Statistics University of California, Santa Cruz ykei@ucsc.edu Oscar H. Madrid Padilla Department of Statistics University of California, Los Angeles oscar.madrid@stat.ucla.edu Kyle Ritscher Department of Statistics University of California, Los Angeles kritscher@g.ucla.edu Xin Ma Department of Biostatistics Columbia University xm2141@cumc.columbia.edu

## Abstract

We study offline change point localization and inference in dynamic multilayer random dot product graphs (D-MRDPGs), where at each time point, a multilayer network is observed with shared node latent positions and time-varying, layerspecific connectivity patterns. We propose a novel two-stage algorithm that combines seeded binary segmentation with low-rank tensor estimation, and establish its consistency in estimating both the number and locations of change points. Furthermore, we derive the limiting distributions of the refined estimators under both vanishing and non-vanishing jump regimes. To the best of our knowledge, this is the first result of its kind in the context of dynamic network data. We also develop a fully data-driven procedure for constructing confidence intervals. Extensive numerical experiments demonstrate the superior performance and practical utility of our methods compared to existing alternatives.

## 1 Introduction

Statistical network analysis models entities as nodes and their interactions as edges. While singlelayer networks capture pairwise interactions efficiently, many real-world systems involve multiple types of interaction among the same set of nodes. Multilayer networks address this complexity by organizing these varied interactions into distinct layers over a common node set, enabling both the capture of heterogeneity and the identification of shared latent structures. In practice, network structures often evolve over time. For instance, transportation networks may exhibit gradual diurnal variations or sudden structural changes due to accidents or road closures. Detecting such sudden shifts and providing adaptive strategies, such as dynamic traffic signal control or rerouting recommendations, is crucial for efficient transportation management. These abrupt structural shifts are referred to as change points. This naturally falls in the territory of change point analysis. Change point analysis is a well-established area in statistics concerned with detecting abrupt structural changes in ordered data. It can be broadly classified into online and offline settings, depending on whether data are analyzed sequentially as they are collected or retrospectively after the full dataset has been observed. In the context of dynamic networks, online change point detection has been studied in models such as inhomogeneous Bernoulli networks (e.g., Yu et al., 2021) and random weighted edge networks (e.g., Chen et al., 2024). Offline detection has been explored in various network models, including inhomogeneous Bernoulli networks (e.g., Wang et al., 2021), stochastic block models (e.g., Xu and Lee, 2022; Bhattacharjee et al., 2020) and random dot product graphs 1
(e.g., Padilla et al., 2022). More recently, Wang et al. (2025) investigated online change point detection in dynamic multilayer random dot product graphs (D-MRDPGs). In this paper, we study offline change point localization and inference for D-MRDPGs. Specifically, at each time point, we observe a realization of an L-layered multilayer network, where nodes are associated with fixed but latent positions, and layer-specific weight matrices capture heterogeneous interactions across layers. These weight matrices are allowed to vary over time. Our goal is to develop efficient procedures for localizing and inferring change points under this dynamic multilayer structure in the offline setting.

## 1.1 List Of Contributions

The main contributions of this paper are summarized as follows. First, to the best of our knowledge, this is the first study on offline change point detection in dynamic multilayer networks. We propose a novel two-stage procedure: (i) seeded binary segmentation with refined CUSUM statistics to generate a coarse set of candidates, and (ii) refinement via low-rank tensor estimation. We establish consistency for both the estimated number of change points and their locations. Second, we derive the limit distributions of the refined estimators, revealing two distinct regimes depending on whether the jump size is fixed or vanishes as the time horizon grows. To the best of our knowledge, these are the first such results in the network literature. We further provide a data-driven procedure for constructing confidence intervals. Finally, extensive numerical experiments demonstrate that our methods substantially outperform existing state-of-the-art algorithms.

## 1.2 Notation And Organization

For p ∈ N
+, let [p] = {1*, . . . , p*}. For sequences {an}n∈N+ , {bn}n∈N+ ⊂ R
+, write an = O(bn)
if an ≤ Cbn for some constant C > 0 and all sufficiently large n, and an = Θ(bn) if both an = O(bn) and bn = O(an). For a sequence of random variables {Xn}n∈N+ , Xn = Op(an) if limM→∞ lim supn P(|Xn| ≥ M an) = 0. For sets C and C
′, define the one-sided Hausdorff distance d(C
′|C) = maxc∈C minc
′∈C′ |c
′ − c|, with d(C
′|C) = ∞ if either set is empty.

For A ∈ R
p1×p2, let Ai and Aj denote its ith row and jth column, and σ1(A) *≥ · · · ≥*
σp1∧p2(A) ≥ 0 its singular values. For tensors M, Q ∈ R
p1×p2×p3 P
, define ⟨M, Q⟩ =
p1 i=1 Pp2 j=1 Pp3 l=1 Mi,j,lQ*i,j,l* and ∥M∥
2F = ⟨M,M⟩. The mode-s matricization of M is denoted by Ms(M) with M1(M)i1,(i2−1)p3+i3 = Mi1,i2,i3and Ms(M) ∈ R
ps×Qt̸=s pt. Tucker ranks (r1, r2, r3) are given by rs = rank(Ms(M)). For Us ∈ R
qs×ps, the marginal multiplication operator ×1 is defined as M ×1 U1 = {Pp1 k=1 Mk,j,l(U1)i,k}i∈[q1], j∈[p2], l∈[p3] ∈ R
q1×p2×p3, with
×2 and ×3 defined analogously.

The paper is organized as follows. Section 2 introduces the D-MRDPG model, the two-stage localization procedure and theoretical guarantees. Section 3 derives limiting distributions of the refined estimators and proposes a data-driven method for confidence intervals. Section 4 presents numerical experiments and Section 5 concludes. Proofs and auxiliary results are in the Appendix.

## 2 Change Point Localization 2.1 Problem Formulation

We consider the multilayer random dot product graph (MRDPG) model (Jones and Rubin-Delanchy, 2020), an extension of the random dot product graph (Young and Scheinerman, 2007) to multilayer networks. Each layer is characterized by a distinct weight matrix, while all layers share a common set of latent positions. We focus on undirected edges, noting that the directed case is analogous. Definition 1 (Multilayer random dot product graphs, MRDPGs). *Given a sequence of deterministic* matrices {W(l)}
L l=1 
⊂ R
d×d*, let* {Xi}
n i=1 ⊂ R
d *be fixed vectors satisfying* X⊤
i W(l)Xj ∈ [0, 1] for all i, j ∈ [n], l ∈ [L]. An adjacency tensor A ∈ {0, 1}
n×n×L *follows an MRDPG if*

$$\mathbb{P}\{\mathbf{A}\}=\prod_{l=1}^{L}\prod_{1\leq i\leq j\leq n}\mathbf{P}_{i,j,l}^{\mathbf{A}_{i,j,l}}(1-\mathbf{P}_{i,j,l})^{1-\mathbf{A}_{i,j,l}}$$ $$=\prod_{l=1}^{L}\prod_{1\leq i\leq j\leq n}\left(X_{i}^{\top}W_{(l)}X_{j}\right)^{\mathbf{A}_{i,j,l}}\left(1-X_{i}^{\top}W_{(l)}X_{j}\right)^{1-\mathbf{A}_{i,j,l}}.$$

We write A ∼ MRDPG({Xi}
n i=1, {W(l)}l∈[L]) *and denote the probability tensor by* P ∈ R
n×n×L.

We now extend this static model to a dynamic setting and introduce a change point framework.

Definition 2 (Dynamic multilayer random dot product graphs, D-MRDPGs). Let {Xi}
n i=1 ⊂ R
d be latent positions and {W(l)(t)}l∈[L],t∈[T] ⊂ R
d×d be a weight matrix sequence. A sequence of mutually independent adjacency tensors {A(t)}t∈[T]*follows the dynamic MRDPGs if* A(t) ∼
MRDPG({Xi}
n i=1, {W(l)(t)}l∈[L]) for t ∈ [T]*. We write* {A(t)}
T
t=1 ∼ D-MRDPGs({Xi}
n i=1,
{{W(l)(t)}l∈[L]}
T
t=1) *and write* {P(t)}
T
t=1 *as the corresponding sequence of probability tensors.*
Model 1. Let {A(t)}t∈[T] ⊂ {0, 1}
n×n×L *follow* D-MRDPGs as in Definition 2. (i) *Assume that* there exist change points 0 = η0 < η1 < · · · < ηK < T = ηK+1 *such that for* t ∈ [T − 1],
{W(l)(t)}
L
l=1 ̸= {W(l)(t + 1)}
L
l=1 if and only if t ∈ {ηk}
K
k=1*. Let* ∆ = mink∈[K+1](ηk − ηk−1) be the minimal spacing between two consecutive change points and assume ∆ = Θ(T). (ii) *For each* k ∈ [K], define the k-th jump size and normalized jump tensor as κk = ∥P(ηk+1) − P(ηk)∥F and Ψk = κ
−1 k{P(ηk+1) − P(ηk)}*, and let* κ = mink∈[K] κk *denote the smallest jump magnitude.*
Model 1 allows abrupt changes in layer connectivity (via weight matrices), while keeping latent positions unchanged over time. This framework is motivated by applications such as air transportation networks (Section 4.2), where nodes represent airports with relatively stable intrinsic attributes (e.g. geographical location and logistical capacity). In contrast, airline routing preferences, encoded in the weight matrices, may shift due to route optimization strategies or policy interventions. The framework can be further extended to allow latent positions to change at the change points; see Appendix C for details. In Model 1(i), we assume that the minimal spacing ∆ between successive change points scales with the time horizon T, essentially bounding the number of changes K. This assumption can be relaxed (see Section 5 and Appendix G.1). In Model 1(ii), the change magnitude is quantified via the Frobenius norm of the difference between expected adjacency tensors. This metric is sufficiently general to accommodate both dense changes - small but widespread deviations across many layers - and sparse changes - large deviations concentrated in a few layers. Throughout, we allow all model parameters, including the number of nodes n, number of layers L, latent dimension d, jump size κ and minimal spacing ∆ to diverge with T.

## 2.2 Change Point Localization Algorithm

In this section, we introduce a two-stage procedure for offline change point localization in dynamic multilayer networks, detailed in Algorithm 1. **Stage I** generates a coarse set of change point candidates using seeded binary segmentation and CUSUM statistics. **Stage II** refines them via localized scan statistics constructed using a tensor-based low-rank estimation technique. This approach builds on Wang et al. (2021) for single-layer networks and extends it to the multilayer setting. For **Stage I**, we begin by defining the seeded intervals (Kovacs et al., 2023) and CUSUM statistics ´ (Page, 1954) for dynamic multilayer networks in Definitions 3 and 4.

Definition 3 (Seeded intervals). Let J = ⌈CJ log2(T)⌉ *for some sufficiently large absolute constant* CJ > 0. For each j ∈ [J]*, define the collection of intervals* Jj as Jj = {(⌊(i − 1)T2
−j⌋, ⌈(i −
1)T2
−j + T2
−j+1⌉]: i ∈ [2j − 1]}*. The full collection of seeded intervals is defined as* J = 
SJ
j=1 Jj .

Definition 4 (CUSUM statistics). Given a tensor sequence {B(t)}t∈[T] and any 0 ≤ *s < t < e* ≤
T*, define the CUSUM statistics as*

$$\widetilde{\mathbf{B}}^{s,e}(t)=\sum_{u=s+1}^{e}\omega_{s,e}^{t}(u)\mathbf{B}(u),\quad\text{where}\quad\omega_{s,e}^{t}(u)=\begin{cases}\sqrt{\frac{e-s}{(e-s)(t-s)}},&\text{for}u\in[t]\backslash[s],\\ -\sqrt{\frac{t-s}{(e-s)(e-t)}},&\text{for}u\in[e]\backslash[t].\end{cases}\tag{1}$$
4
Stage I implements a modified version of seeded binary segmentation (SBS), a computationally efficient algorithm introduced by Kovacs et al. (2023). SBS leverages seeded intervals to construct a ´ multiscale collection of candidate regions for detecting multiple change points. Within each interval, the algorithm computes CUSUM statistics and retains time points where the statistic is maximized and exceeds a predefined threshold, as preliminary change point estimators We next define the refined scan statistics used in **Stage II**, based on tensor heteroskedastic principal component analysis (TH-PCA), a low-rank tensor estimation method proposed by Han et al. (2022) and detailed in Algorithm 2 in Appendix D.

Definition 5 (Refined scan statistics). Let {A′(t)}t∈[T] and {B′(t)}t∈[T] be independent sequences generated according to Definition 2. Given {(bk, sk, ek)}
Ke k=1, for any k ∈ [Ke] and t ∈ (sk, ek),
we define the refined scan statistic as Db sk,ek bk(t) =Pbsk,ek (bk)/∥Pbsk,ek (bk)∥F, Af′
sk,ek(t),
where Pbsk,ek (bk) = TH-PCABf′
sk,ek(bk),(d, d, msk,ek bk),p(ek − bk)(bk − sk)/(ek − sk), 
p(ek − bk)(bk − sk)/(ek − sk)*with* TH-PCA *detailed in Algorithm 2,* Bf′
·,·(·) defined in Definition 4 and m s,e bk defined in Assumption 1(ii).

Stage II refines each preliminary change point estimate from **Stage I** by locating the time point that maximizes the refined scan statistics within a local window around the initial estimate. This step employs the TH-PCA procedure with an additional truncation step (see Algorithm 2 in Appendix D) to more accurately estimate the local expected CUSUM adjacency tensors, yielding provably improved localization accuracy.

The assumption of mutual independence among the four sequences in Algorithm 1 is imposed for theoretical convenience. In practice (and in our numerical experiments in Section 4), **Stage I** and Stage II are implemented using the same two split tensor sequences via the odd–even splitting approach. The computational cost is O(T n2Llog2(T)) for Stage I and O(T n2Lr *log(*n)) for Stage II, where r is the maximum input rank in TH-PCA, giving an overall cost of O(T n2Lr log2(T ∨n)).

## 2.3 Theoretical Guarantees

This section establishes the theoretical guarantees of the proposed two-stage change point localization procedure (Algorithm 1). We begin by justifying the use of low-rank tensor estimation via TH-PCA (Algorithm 2) in **Stage II** through an analysis of the expected CUSUM-transformed and average adjacency tensors. While the expected averaged adjacency tensors introduced below are not used in this section, they are essential for deriving the limiting distributions in Section 3.

For any 0 ≤ *s < t < e* ≤ T, define the expected CUSUM-transformed and average adjacency tensors as

$$\widetilde{\mathbf{P}}^{s,e}(t)=\mathbb{E}\big{\{}\widetilde{\mathbf{B}}^{s,e}(t)\big{\}}\quad\text{and}\quad\mathbf{P}^{s,e}=\mathbb{E}\big{\{}\mathbf{B}^{s,e}\big{\}},\quad\text{where}\quad\mathbf{B}^{s,e}=(e-s)^{-1}\sum_{t=s+1}^{e}\mathbf{B}(t),\tag{2}$$

and Be·,·(·) is defined in Definition 4. Both tensors admit Tucker representations of the form Pes,e(t) = S×1X×2X×3Qes,e(t), and Ps,e = S×1X×2X×3Qs,e, where X = (X1*, . . . , X*n)
⊤ ∈
R

n×dand S ∈ R
d×d×d 2with S*i,j,l* = 1{l = (i−1)d+j}. The matrices Qes,e(t) and Qs,e are given

by  $$\widetilde{Q}^{s,e}(t)=\sum_{u=s+1}^{e}\omega_{s,e}^{t}(u)Q(u),\quad Q^{s,e}=(e-s)^{-1}\sum_{t=s+1}^{e}Q(t),\tag{3}$$  where $\omega_{s,e}^{t}(u)$ is define in (1) and $Q(u)\in\mathbb{R}^{L\times d^{2}}$ with rows  $$\big{(}Q(u)\big{)}_{l}=\big{(}(W_{(l)}(u))_{1}\cdots(W_{(l)}(u))_{d}\big{)},\quad l\in[L].\tag{4}$$
Algorithm 1 Two-stage change point localization for D-MRDPGs INPUT: Mutually independent sequences {A(t)}t∈[T], {A′(t)}t∈[T], {B(t)}t∈[T], {B′(t)}t∈[T] ⊂
{0, 1}
n×n×L, threshold τ ∈ R
+, collection of seeded intervals J
Initialise: s ← 0, e ← T, *C ← ∅* e Stage I: Seeded Binary Segmentation, SBS(s, e)*, τ,*J
for I = (α
′, β′] ∈ J do if I = (α
′, β′] ⊆ (*s, e*] **then**
(*α, β*] = (⌊α
′ + 64−1(β
′ − α
′)⌋, ⌈β
′ − 64−1(β
′ − α
′)⌉]
if β − α ≥ 2 **then**
bI ← arg max*α<t<β*
Ae α,β(t), Be α,β(t)⟩, aI ←Ae α,β(bI), Be α,β(bI)⟩
else aI ← −1 end if else aI ← −1 end if end for I
∗ ← arg maxI∈J aI
if aI∗ > τ **then**
C ←
e *C ∪ {* e bI∗ }, SBS(s, bI∗ )*, τ,*J, SBS(bI∗ , e)*, τ,*J
end if Stage II: Local Refinement, LR(Ce)
{bk}
Ke k=1 ← Cewith 0 = b0 < b1 < · · · < bKe < bKe+1 = T
for k = 1 to Ke do
(sk, ek] ←⌊(bk−1 + bk)/2⌋, ⌈(bk + bk+1)/2⌉
ηek ← arg maxsk*<t<e*k Dbsk,ek bk(t) ▷ See Definition 5 end for OUTPUT: {ηek}
Ke k=1 To establish the low-rank structure of Pes,e(t) and Ps,e (in terms of Tucker ranks, see Section 1.2),
and to state theoretical guarantees for Algorithm 1, we state some necessary assumptions below.

Assumption 1. *Consider* D-MRDPGs({Xi}
n i=1, {{W(l)(t)}l∈[L]}
T
t=1) *from Definition 2.*
(i) Let X = (X1*, . . . , X*n)
⊤ ∈ R
n×d*. Assume that* rank(X) = d, σ1(X)/σd(X) ≤ Cσ and σd(X) ≥ Cgap
√n with absolute constants Cσ, Cgap > 0.

(ii) For any 0 ≤ s < t < e ≤ T, let Qes,e(t) ∈ R
L×d 2be defined in (3)*. Denote* m s,e t =
rank(Qes,e(t)). Assume that σ1Qes,e(t)/σms,e tQes,e(t)≤ Cσ and σms,e tQes,e(t)≥ Cgap with absolute constants Cgap, Cσ > 0.

(iii) For any 0 ≤ s < e ≤ T, let Qs,e ∈ R
L×d 2be defined in (3). Denote ms,e = rank(Qs,e).

Assume that σ1Qs,e/σms,e Qs,e≤ Cσ and σms,e Qs,e≥ Cgap *with absolute constants* Cgap, Cσ > 0.

Assumption 1(i) imposes a full-rank condition on the latent position matrix X, requiring its smallest singular value to be at least of order 
√n, with all singular values of the same order. Since X
represents latent positions rather than observed data, the full-rankness of X can be interpreted as a condition on the knowledge of the intrinsic dimension d, ensuring that the input dimension to TH- PCA is no smaller than the true latent dimension d. Further discussion on rank selection, see Wang et al. (2025) and Section 4.1. Assumptions 1(ii) and (iii) - with (iii) for Section 3 - impose low-rank conditions on the CUSUM
and averaged forms of {Q(t)}
T
t=1, where each Q(t) comprises the weight matrices {W(l)(t)}
L
l=1. In Appendix E, we show that, with high probability, each working interval (sk, ek] or (˜sk, e˜k] contains exactly one change point ηk, implying max{m sk,ek t, ms˜k,e˜k } ≤ rank(Q(ηk)) + rank(Q(ηk+1)) for t ∈ (sk, ek). This implicitly constraints the ranks of {Q(ηk)}
K+1 k=1 . While this low-rank structure may not directly or transparently reflect the explicit model structure, such ambiguity is common in tensor-based models (e.g. Jing et al., 2021). The signal-to-noise ratio (SNR) is commonly used to characterize the inherent difficulty of change point detection. We now state the SNR condition required for our theoretical guarantees.

Assumption 2 (Signal-to-noise ratio condition). *Assume that there exists a large enough absolute* constant CSNR > 0 *such that* κ
√∆ ≥ CSNR log(T)pnL1/2 + d 2mmax + nd + Lmmax*, where* mmax = maxk∈[K+1] rankQ(ηk)with Q(ηk) *defined in* (4).

We compare Assumption 2 to its counterpart in Wang et al. (2021). When the sparsity parameter ρ = 1, their SNR condition (Assumption 3) becomes κ
√∆ ≥ CSNR log1+ξ(T)
√nd for some ξ > 0. Our assumption is consistent with this and extends it to the multilayer setting by accounting for the additional complexity from multilayers and the low-rank structure of layers' weight matrices.

Theorem 1. Let {ηek}
Ke k=1 be the output of Algorithm 1. Suppose the mutually independent adjacency tensor sequences {A(t)}t∈[T], {A′(t)}t∈[T], {B(t)}t∈[T], {B′(t)}t∈[T] ⊂ {0, 1}
n×n×L are generated according to Definition 2 and satisfy Model 1, Assumptions 1(i), (ii) and 2. Assume the threshold τ *is chosen such that* cτ,1n
√Llog3/2(T) *< τ < c*τ,2κ 2∆*, where* cτ,1, cτ,2 > 0 are sufficiently large and small absolute constants, respectively. We have that

$\mathbb{P}\Big{\{}\widetilde{K}=K\text{and}|\widetilde{\eta}_{k}-\eta_{k}|\leq\epsilon_{k},\,\forall k\in[K]\Big{\}}\geq1-CT^{-c},\quad\text{where}\epsilon_{k}=C_{\epsilon}\frac{\log(T)}{\kappa_{k}^{2}},$

and Cϵ, C, c > 0 *are absolute constants.* Theorem 1 implies that, with probability tending to 1 as T → ∞, the estimated number of change points satisfies Ke = K and the relative localization error vanishes: maxk∈[K] ∆−1|ηek − ηk| ≤
Cϵ∆−1κ
−2log(T) → 0 by Assumption 2. This establishes the consistency of Algorithm 1 in both detecting and localizing all change points. Remark 1. Compared to Wang et al. (2021), which established minimax-optimal localization rates for single-layer networks, our work extends these guarantees to more complex multilayer settings without sacrificing accuracy. In contrast, Wang et al. (2025) focused on the online setting and obtained a localization rate of order κ
−2(d 2mmax + nd + Lmmax) log(∆/α), where α controls the Type-I error rate. Our approach, by comparison, achieves a substantially sharper rate of order κ
−2 klog(T).

## 3 Limiting Distributions

Inference on change points is generally more challenging than establishing high-probability bounds on localization errors. To address this, we introduce a final refinement step, inspired by approaches such as those in Madrid Padilla et al. (2023); Xue et al. (2024); Xu et al. (2024).

Let {A(t)}t∈[T] and {B(t)}t∈[T] be independent samples as defined in Definition 2. Let {ηek}
K k=1 be the output of Algorithm 1 with 0 = ηe0 < ηe1 < *· · ·* < ηeKe < ηeKe+1 = T. For each k ∈ [Ke],
define the final estimators as

define the final commutators as  $$\widehat{\eta}_{k}=\underset{\widetilde{s}_{k}<t<\varepsilon_{k}}{\operatorname{arg\,min}}\,\mathcal{Q}_{k}(t)=\underset{\widetilde{s}_{k}<t<\varepsilon_{k}}{\operatorname{arg\,min}}\,\sum_{u=\widetilde{s}_{k+1}}^{t}\|\mathbf{A}(u)-\widehat{\mathbf{P}}^{\widetilde{u}_{k-1},\widetilde{s}_{k}}\|_{F}^{2}+\sum_{u=t+1}^{\widetilde{\varepsilon}_{k}}\|\mathbf{A}(u)-\widehat{\mathbf{P}}^{\widetilde{u}_{k},\widetilde{s}_{k+1}}\|_{F}^{2},\tag{5}$$  where $(\widetilde{s}_{k},\widetilde{e}_{k}]=((\widetilde{\eta}_{k-1}+\widetilde{\eta}_{k})/2,(\widetilde{\eta}_{k}+\widetilde{\eta}_{k+1})/2]$ and 
$$(6)$$
ηek−1,ηek,(*d, d, m*ηek−1,ηek), 1, 0), (6)
$$\hat{\bf P}^{\tilde{\eta}_{k-1},\tilde{\eta}_{k}}=\mathrm{TH-PCA}({\bf B}^{\tilde{\eta}_{k-1},\tilde{\eta}_{k}},(d,d,m^{\tilde{\eta}_{k-1},\tilde{\eta}_{k}}),1,0),$$
with TH-PCA detailed in Algorithm 2, B·,· defined in (2) and mηek−1,ηek
defined in Assumption 1(iii).
Theorem 2. Let {A(t)}t∈[T], {A′(t)}t∈[T], {B(t)}t∈[T], {B′(t)}t∈[T] ⊂ {0, 1}
n×n×L be mutually
independent adjacency tensor sequences generated according to Definition 2 and satisfying Model 1,
Assumptions 1 and 2. Let {ηbk}
Ke
k=1 *be defined in* (5) *with* {ηek}
Ke
k=1 obtained from Algorithm 1, using
a threshold τ *satisfying condition stated in Theorem 1.*
For k ∈ [K], if κk → 0, as T → ∞, then when T → ∞*, we have* |ηbk − ηk| = Op(κ

$$\eta_{k}|=O_{p}(\kappa_{k}^{-2})\;a n d$$
$\kappa_{k}^{2}(\widehat{\eta}_{k}-\eta_{k})\xrightarrow[r\in\mathbb{R}]{\mathcal{D}}\underset{r\in\mathbb{R}}{\arg\min}\,\mathcal{P}_{k}^{\prime}(r),\quad\text{where}\quad\mathcal{P}_{k}^{\prime}(r)=\begin{cases}-r+2\sigma_{k,k}\mathbb{B}_{1}(-r),&r<0,\\ 0,&r=0,\\ r+2\sigma_{k,k+1}\mathbb{B}_{2}(r),&r>0,\end{cases}$
for r ∈ Z. Here, B1(r) and B2(r) *are independent standard Brownian motions, and for any* k
′ ∈
{*k, k* + 1}, σ 2 k,k′ = Var ⟨Ψk, Ek′ (1)⟩, where Ψk *is the normalized jump tensor (Model 1), and* Ek′ (t) = Ak′ (t) − P(ηk′ ) *with* {Ak′ (t)}t∈Z
i.i.d. ∼ MRDPG({Xi}
n i=1, {W(l)(ηk′ )}l∈[L]).

Theorem 2 establishes the localization error bounds and limiting distributions for the refined change point estimators in the vanishing jump regime (κk → 0). In particular, it shows the uniform tightness κ 2 k|ηbk − ηk| = Op(1), which improves upon Theorem 1 by a logarithmic factor and guarantees the existence of limiting distributions. To the best of our knowledge, Theorem 2 is the first to derive limiting distributions for change point estimators in network data. These limiting distributions are associated with a two-sided Brownian motion. Results for the non-vanishing jump regime (κk → ρk > 0) are deferred to Appendix A.

## 3.1 Confidence Interval Construction

Using Theorem 2, we construct data-driven (1 − α) confidence intervals for ηk, k ∈ [K], in the vanishing regime, for a user-specified confidence level α ∈ (0, 1) as follows. Step 1: Estimate the jump size and normalized jump tensor. Compute the estimated jump size κˆk = ∥Pb ηek,ηek+1 − Pb ηek−1,ηek ∥F and the estimated normalized jump tensor Ψb k = ˆκ
−1 k(Pb ηek,ηek+1 −
Pb ηek−1,ηek ) where Pb·,·is defined in (6).

Step 2: Estimate the variances. For each k
′ ∈ {*k, k* + 1}, compute

$$\hat{\sigma}_{k,k^{\prime}}^{2}=\frac{1}{\tilde{\eta}_{k^{\prime}}-\tilde{\eta}_{k^{\prime}-1}-1}\sum_{t=\tilde{\eta}_{k^{\prime}-1}+1}^{\tilde{\eta}_{k^{\prime}}}\left(\langle\hat{\mathbf{\Psi}}_{k},\mathbf{A}(t)-\hat{\mathbf{P}}^{\tilde{\eta}_{k^{\prime}-1},\tilde{\eta}_{k^{\prime}}}\rangle\right)^{2}.$$

Step 3: Simulate limiting distributions. Let B ∈ N
+ and M ∈ R
+. For each b ∈ [B], let

$\mathbb{I}\;M\in\mathbb{R}^+\,.$ For 
  **Lemma limiting distributions.** Let $B\in\mathbb{R}^{n}$ and $M\in\mathbb{R}^{n}$. For each $\theta\in[B]$, let  $$\hat{u}_{k}^{(b)}=\operatorname*{arg\,min}_{r\in(-M,M)}\widehat{P}_{k}^{\prime}(r),\quad\text{where}\quad\widehat{P}_{k}^{\prime}(r)=\begin{cases}-r+\frac{2\theta_{k,k}}{\sqrt{T}}\sum_{i=\lceil T\rceil}^{-1}z_{i}^{(b)},&r<0,\\ 0,&r=0,\\ r+\frac{2\theta_{k,k+1}}{\sqrt{T}}\sum_{i=1}^{\lceil T\rceil}z_{i}^{(b)},&r>0,\end{cases}$$
with independent standard Gaussian random variables {z

$$\mathrm{bles}\,\left\{z_{i}^{(b)}\right\}_{i=-\lfloor T M\rfloor}^{\lceil T M\rceil}.$$

Step 4: Construct the confidence interval. Let qˆα/2, qˆ1−α/2 be empirical quantiles of {uˆ
(b) k
}
B
b=1.

The (1 − α) confidence interval for ηk is given by

$$\left[\widehat{\eta}_{k}-\frac{\widehat{q}_{1}{-}\alpha/2}{\widehat{\kappa}_{k}^{2}}\,\mathbb{I}\left\{\widehat{\kappa}_{k}\neq0\right\},\,\widehat{\eta}_{k}-\frac{\widehat{q}\alpha/2}{\widehat{\kappa}_{k}^{2}}\,\mathbb{I}\left\{\widehat{\kappa}_{k}\neq0\right\}\right].$$

The empirical performance of this procedure is evaluated in Section 4.1.

## 4 Numerical Experiments 4.1 Simulation Studies

To evaluate the performance of our method (Algorithm 1) for change point detection and localization, we compare it to gSeg (Chen and Zhang, 2015) and kerSeg (Song and Chen, 2024). For the competitors, we consider two input types: networks (nets.) and their layer-wise Frobenius norms (frob.). For gSeg, we construct the similarity graph using the minimum spanning tree and apply the original edge-count scan statistics. For kerSeg, we use the kernel-based scan statistics fGKCP1.

For both methods, we set the significance level α = 0.05. Our proposed method is referred to as CPDmrdpg. Following Wang et al. (2025), we use relatively large Tucker ranks as inputs to TH-PCA
(Algorithm 2) for robustness, setting r1 = r2 = 15 and r3 = L to compute the refined scan statistics (Definition 5). Based on Theorem 1, we set the threshold τ = cτ,1n
√Llog3/2(T) with cτ,1 = 0.1.

We also assess the confidence intervals constructed utilizing the procedure in Section 3.1, a capability not supported by the competitors. We set B = 500 and M = T as suggested by Xu et al. (2024).

To assess sensitivity to tuning parameters, we vary the threshold constant cτ,1 ∈ {0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25} and input ranks r ∈ {10, 15, 20}. We further conduct additional simulations to evaluate the robustness of our method under temporal dependence, high-frequency change points and randomly located changes. In addition, we compare our approach with existing dynamic multilayer network approaches (Wang et al., 2025), which are designed for online settings, as well as with deep-learning-based approaches (Li et al., 2024). All results are reported in Appendix G.1.

Performance is quantified using the following metrics: (i) Absolute error: |Kb − K| where Kb and K denote the numbers of estimated and true change points, respectively; (ii) One-sided Hausdorff distances (see Section 1.2): d(C|C 
b ) and d(C|Cb) where Cb and C denote the sets of estimated and true change points, respectively; (iii) Time segment coverage: C(G, G
′) = T
−1 PA∈G *|A| ·*
maxA′∈G′ |A ∩ A′|/*|A ∪ A*′| where G and G
′ denote the partitions of the time span [1, T] into intervals between consecutive true and estimated change points, respectively.

Throughout, we set the time horizon to T = 200 and the number of layers to L = 4, and consider node sizes n ∈ {50, 100}. Each setting is evaluated over 100 Monte Carlo trials. We consider two network models: the Dirichlet distribution model (DDM) and the multilayer stochastic block model (MSBM), with structural changes specified in each scenario. In the DDM, we generate latent positions {Xi}
n i=1∪{Yi}
n i=1 i.i.d. 

∼ Dirichlet(1d) with d = 5 and 1d ∈ R
d denoting the all-one vector.

For each time t, we sample weight matrices {W(l)(t)}
L
l=1 ⊂ R
d×d with entries (W(l)(t))u,v ∼
Uniform((ρtL + l)/(4L),(ρtL + l + 1)/(4L)). The edge probabilities are given by Pi,j,l(t) = X⊤
i W(l)(t)Yj and the adjacency entries are sampled as A*i,j,l*(t) ∼ Bernoulli(P*i,j,l*(t)). In the MSBM, the edge probability tensor P*i,j,l*(t) ∈ [0, 1]n×n×L is defined as P*i,j,l*(t) = p1,l if nodes i, j ∈ Bc for some c ∈ [Ct], and p2,l otherwise, where {Bc}c∈[Ct] partitions the nodes into Ct communities. The connection probabilities are drawn from p1,l ∼ Uniform((3L+l−1)/(4L),(3L+ l)/(4L)) and p2,l ∼ Uniform((2L + l − 1)/(4L),(2L + l)/(4L)). The adjacency tensor A(t) ∈
{0, 1}
n×n×L is then sampled A*i,j,l*(t)
ind. 

∼ Bernoulli(P*i,j,l*(t)).

Scenario 1. We consider the DDM with K = 2 change points at t ∈ {70, 140}, yielding 3 time segments {Ai}
3 i=1. We set ρt = 2 for t ∈ A1∪A3, and ρt = 3 with reversed layer order for t ∈ A2.

Scenario 2. We consider the MSBM with K = 5 change points at t ∈ {20, 60, 80, 160, 180}, resulting in 6 time segments {Ai}
6 i=1. We let {Bc(t)}c∈[Ct] be evenly-sized communities and specify the changes as follows: Ct = 4 for t ∈ A1, Ct = 2 for t ∈ A2, Ct = 4 for t ∈ A3, Ct = 4 with reversed layer order for t ∈ A4, Ct = 3 for t ∈ A5 and Ct = 4 for t ∈ A6.

Scenario 3. We consider the MSBM with K = 3 change points at t ∈ {50, 100, 150}, yielding 4 time segments {Ai}
4 i=1. The number of communities is fixed at Ct = 3 but in the first layer, the the community sizes vary across segments (0.3n, 0.4n, 0.3n) in A1 ∪ A4, (0.4n, 0.3n, 0.3n) in A2 and (0.5n, 0.3n, 0.2n) in A3. The remaining layers retain equal-sized communities.

Scenario 4. We consider the MSBM with K = 5 change points at t ∈ {20, 60, 80, 160, 180},
resulting in 6 time segments {Ai}
6 i=1. The number of communities is fixed at Ct = 4 with equal-sized partitions, while the connection probabilities vary across segments. Specifically, for ϵ = 0.1, we let p1,l ∼ Uniform (0.5 · [0.21 + δt · ϵ], 0.5 · [0.*25 +* δt · ϵ]) and p2,l ∼ Uniform (0.21 + δt · ϵ, 0.25 + δt · ϵ), where δt = 0 for t ∈ A1 ∪ A5, δt = 1 for t ∈ A2 ∪ A4 ∪ A6 and δt = 2 for t ∈ A3.

The changes in **Scenarios 1** and 4 follow Model 1, while those in **Scenarios 2** and 3 do not, allowing us to assess the robustness of our methods. Table 1 presents results all four scenarios. Across most scenarios, our method demonstrates the strongest overall performance, nearly accurately estimating both the number and locations of change points, and remaining robust even when Model 1

Table 1: Means of evaluation metrics for Scenarios 1–4.

n = 50 n = 100

Scenario Method |Kb − K| ↓ d(C|C b ) ↓ d(C|Cb) ↓ C(G, G

′) ↑ |Kb − K| ↓ d(C|C b ) ↓ d(C|Cb) ↓ C(G, G

′) ↑

1

CPDmrdpg 0.01 0.00 0.42 99.86% 0.00 0.00 0.00 100% gSeg (nets.) 1.09 Inf Inf 52.82% 1.12 Inf Inf 52.62% kerSeg (nets.) 0.10 0.00 3.12 99.13% 0.12 0.00 2.82 99.17% gSeg (frob.) 0.52 Inf Inf 90.12% 0.47 Inf Inf 88.71% kerSeg (frob.) 0.26 0.00 5.76 98.35% 0.30 0.00 6.07 98.11%

2

CPDmrdpg 0.00 0.00 0.00 100% 0.00 0.00 0.00 100% gSeg (nets.) 1.60 Inf Inf 67.68% 1.58 Inf Inf 69.24% kerSeg (nets.) 0.15 0.00 1.53 99.32% 0.16 0.00 1.81 99.31% gSeg (frob.) 0.23 Inf Inf 97.71% 0.16 0.04 1.65 99.17% kerSeg (frob.) 0.35 0.11 3.43 98.37% 0.40 0.02 4.42 97.81%

3

CPDmrdpg 0.19 9.64 0.14 95.11% 0.00 0.02 0.02 99.98% gSeg (nets.) 0.98 Inf Inf 68.93% 0.69 Inf Inf 80.10% kerSeg (nets.) 0.16 0.18 2.06 98.90% 0.17 0.00 3.26 99.16%

gSeg (frob.) 0.92 Inf Inf 66.78% 0.79 Inf Inf 72.11%

kerSeg (frob.) 0.82 48.52 5.11 73.55% 0.79 48.82 4.75 73.80%

4

CPDmrdpg 0.00 0.02 0.02 99.98% 0.00 0.00 0.00 100% gSeg (nets.) 5.00 Inf Inf 0.00% 4.98 Inf Inf 0.77% kerSeg (nets.) 0.36 0.14 2.65 98.56% 0.34 0.08 2.93 98.47% gSeg (frob.) 1.53 Inf Inf 74.92% 1.86 Inf Inf 68.57% kerSeg (frob.) 0.40 0.05 3.71 98.12% 0.42 0.06 2.93 98.63%

Table 2: The 95% confidence interval coverage (average length) for change points across all scenarios. is violated. For gSeg, Frobenius norm (frob.) inputs yield better results than networks (nets.), while kerSeg performs better with networks, benefiting from its high-dimensional kernel-based design.

Although both competitors exhibit low Hausdorff distances d(C|C 
b ), their higher reverse distances d(C|Cb) and frequent errors in estimating the number of change points suggest they often detect spurious change points. Table 2 reports the coverage and average lengths of the confidence intervals constructed via the procedure in Section 3.1 for node size n ∈ {100, 150}. The proposed method generally achieves strong coverage with reasonably narrow intervals. Coverage is lower in **Scenario 3**, where violations of Model 1 and relatively small, layer-specific changes pose greater challenges. The performance improves with larger n as the change magnitudes κk increase.

## 4.2 Real Data Experiments

Our analysis incorporates two real data sets, the worldwide agricultural trade network data set presented here and the U.S. air transport network data set in Appendix G.2. The worldwide agricultural trade network data are available from Food and Agricultural Organization of the United Nations (2022). The dataset comprises annual multilayer networks from 1986 to 2020 (T = 35), with nodes representing countries and layers representing agricultural products. A directed edge within a layer indicates the trade relation between two countries of a specific agricultural product. We use the top L = 4 agricultural products by the trade volume and the n = 75 most active countries based on import/export volume. Tuning parameters follow the setup described in Section 4.1.

Table 3 summarizes the change points detected by the proposed and competing methods for the worldwide agricultural trade network data. Notably, the gSeg method fails to detect any change points after 2010, regardless of input type. Meanwhile, the kerSeg method detects change points in 1990 and 1992, which are temporally too close. In contrast, our proposed method (CPDmrdpg) identifies four major change points (1991, 1999, 2005, and 2013) that align well with known geopolitical and policy-related events. Furthermore, confidence intervals, which may be constructed for the pro-

| n   | Scenario 1   | Scenario 2   | Scenario 3     | Scenario 4   |
|-----|--------------|--------------|----------------|--------------|
| 100 | 100% (0.003) | 100% (0.106) | 76.67% (1.528) | 100% (0.605) |
| 150 | 100% (0.001) | 100% (0.029) | 95.33% (0.653) | 100% (0.294) |

| Method         | Detected change points       |
|----------------|------------------------------|
| CPDmrdpg       | 1991, 1999, 2005, 2013       |
| gSeg (nets.)   | 1993, 2002, 2010             |
| kerSeg (nets.) | 1990, 1992, 1999, 2005, 2012 |
| gSeg (frob.)   | 1993, 2002, 2009             |
| kerSeg (frob.) | 1990, 1992, 1997, 2003, 2012 |

Table 4: Detected change point from Algorithm 1 and 95% confidence intervals via Section 3.1 for the worldwide agricultural trade network data.

| Detected change points   | Time point   | Confidence interval   |
|--------------------------|--------------|-----------------------|
| 1991                     | 6            | (5.97, 6.03)          |
| 1999                     | 14           | (13.98, 14.02)        |
| 2005                     | 20           | (17.97, 18.05)        |
| 2013                     | 28           | (25.99, 26.06)        |

posed method via the procedure from Section 3.1, are presented in Table 4 at a 95% confidence level. The 1991 change point aligns with the German reunification and the dissolution of the Soviet Union, both of which triggered major political shifts that significantly affected the trade dynamics. The 1999 change point corresponds to the World Trade Organization's (WTO) Third Ministerial Conference, a key moment in debates on globalization, particularly regarding agricultural subsidies and tariff reductions, with developing nations demanding fairer trade terms. The 2005 change point marks a WTO agreement to eliminate agricultural export subsidies, promoting greater equity in global markets. Finally, the 2013 change point corresponds to the adoption of the WTO's Bali Package, the first fully endorsed multilateral agreement, which introduced the Trade Facilitation Agreement and key provisions on food security and tariff quota administration, significantly impacting agricultural trade.

## 5 Conclusion

In this paper, we study offline change point localization and inference in dynamic multilayer networks - a setting that, to the best of our knowledge, has not been previously addressed. We propose a two-stage algorithm with consistency guarantees for estimating both the number and locations of change points. We further develop local refinement procedures, derive limiting distributions and introduce a data-driven method for constructing confidence intervals for the true change points. The current framework assumes temporal independence, but it can be extended to incorporate temporal dependence structures (e.g. Padilla et al., 2022; Cho and Owens, 2023); see Appendix B for details on the framework and corresponding adjustments to the theoretical analysis.

Several limitations of this work remain open for future research. First, the assumption ∆ = Θ(T) precludes frequent change points. This could be relaxed using alternative selection strategies such as the narrowest-over-threshold approach (Baranowski et al., 2019) instead of the greedy selection in this paper. Second, our inference procedure is limited to vanishing jumps. It would be interesting to explore practical procedures for the non-vanishing regime, potentially building on bootstrap methods (e.g. Cho and Kirch, 2022).

## References

Rafal Baranowski, Yining Chen, and Piotr Fryzlewicz. Narrowest-over-threshold detection of multiple change points and change-point-like features. Journal of the Royal Statistical Society Series B: Statistical Methodology, 81(3):649–672, May 2019. ISSN 1467-9868. doi: 10.1111/rssb.12322. URL http://dx.doi.org/10.1111/rssb.12322.

Monika Bhattacharjee, Moulinath Banerjee, and George Michailidis. Change point estimation in a dynamic stochastic block model. *Journal of machine learning research*, 21(107):1–59, 2020.

Bureau of Transportation Statistics. T-100 domestic market (u.s. carriers), 2022. URL
https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=
GDL&QO_fu146_anzr=Nv4.

Cathy Yi-Hsuan Chen, Yarema Okhrin, and Tengyao Wang. Monitoring network changes in social media. *Journal of Business & Economic Statistics*, 42(2):391–406, 2024.

Hao Chen and Nancy Zhang. Graph-based change-point detection. *The Annals of Statistics*, 43
(1):139 - 176, 2015. doi: 10.1214/14-AOS1269. URL https://doi.org/10.1214/ 14-AOS1269.

Haeran Cho and Claudia Kirch. Bootstrap confidence intervals for multiple change points based on moving sum procedures. *Computational Statistics & Data Analysis*, 175:107552, 2022.

Haeran Cho and Dom Owens. High-dimensional data segmentation in regression settings permitting temporal dependence and non-gaussianity, 2023. URL https://arxiv.org/abs/2209. 08892.

Food and Agricultural Organization of the United Nations. Food and agriculture data, 2022. URL
https://www.fao.org/faostat/en/\#data/TM.

Rungang Han, Rebecca Willett, and Anru R Zhang. An optimal statistical and computational framework for generalized tensor estimation. *The Annals of Statistics*, 50(1):1–29, 2022.

Bing-Yi Jing, Ting Li, Zhongyuan Lyu, and Dong Xia. Community detection on mixture multilayer networks via regularized tensor decomposition. *The Annals of Statistics*, 49(6):3181–3205, 2021.

Andrew Jones and Patrick Rubin-Delanchy. The multilayer random dot product graph. arXiv preprint arXiv:2007.10455, 2020.

Solt Kovacs, Peter B ´ uhlmann, Housen Li, and Axel Munk. Seeded binary segmentation: a general ¨
methodology for fast and optimal changepoint detection. *Biometrika*, 110(1):249–256, 2023.

Jie Li, Paul Fearnhead, Piotr Fryzlewicz, and Tengyao Wang. Automatic change-point detection in time series via deep learning. *Journal of the Royal Statistical Society Series B: Statistical* Methodology, 86(2):273–285, 2024.

Carlos Misael Madrid Padilla, Daren Wang, Zifeng Zhao, and Yi Yu. Change-point detection for sparse and dense functional data in general dimensions. Advances in Neural Information Processing Systems, 35:37121–37133, 2022.

Carlos Misael Madrid Padilla, Haotian Xu, Daren Wang, OSCAR HERNAN MADRID PADILLA,
and Yi Yu. Change point detection and inference in multivariate non-parametric models under mixing conditions. *Advances in Neural Information Processing Systems*, 36:21081–21134, 2023.

Oscar Hernan Madrid Padilla, Yi Yu, and Carey E Priebe. Change point localization in dependent dynamic nonparametric random dot product graphs. *Journal of Machine Learning Research*, 23
(234):1–59, 2022.

Ewan S Page. Continuous inspection schemes. *Biometrika*, 41(1/2):100–115, 1954. Hoseung Song and Hao Chen. Practical and powerful kernel-based change-point detection. IEEE
Transactions on Signal Processing, 2024.

Roman Vershynin. *High-dimensional probability: An introduction with applications in data science*,
volume 47. Cambridge university press, 2018.

Daren Wang, Yi Yu, and Alessandro Rinaldo. Optimal covariance change point localization in high dimension. *arXiv preprint arXiv:1712.09912*, 2017.

Daren Wang, Yi Yu, and Alessandro Rinaldo. Optimal change point detection and localization in sparse dynamic networks. *The Annals of Statistics*, 49(1):203–232, 2021.

Fan Wang, Wanshan Li, Oscar Hernan Madrid Padilla, Yi Yu, and Alessandro Rinaldo. Multilayer random dot product graphs: estimation and online change point detection1. *Journal of the Royal* Statistical Society Series B: Statistical Methodology, page qkaf051, 08 2025. ISSN 1369-7412. doi: 10.1093/jrsssb/qkaf051. URL https://doi.org/10.1093/jrsssb/qkaf051.

Jon Wellner et al. *Weak convergence and empirical processes: with applications to statistics*.

Springer Science & Business Media, 2013.

Cong Xu and Thomas CM Lee. Statistical consistency for change point detection and community estimation in time-evolving dynamic networks. IEEE Transactions on Signal and Information Processing over Networks, 8:215–227, 2022.

Haotian Xu, Daren Wang, Zifeng Zhao, and Yi Yu. Change-point inference in high-dimensional regression models under temporal dependence. *The Annals of Statistics*, 52(3):999–1026, 2024.

Gengyu Xue, Haotian Xu, and Yi Yu. Change point localisation and inference in fragmented functional data. *arXiv preprint arXiv:2405.05730*, 2024.

Stephen J Young and Edward R Scheinerman. Random dot product graph models for social networks. In *International Workshop on Algorithms and Models for the Web-Graph*, pages 138–149.

Springer, 2007.

Yi Yu, Oscar Hernan Madrid Padilla, Daren Wang, and Alessandro Rinaldo. Optimal network online change point localisation. *arXiv preprint arXiv:2101.05477*, 2021.

Anru R Zhang, T Tony Cai, and Yihong Wu. Heteroskedastic pca: Algorithm, optimality, and applications. *The Annals of Statistics*, 50(1):53–80, 2022.