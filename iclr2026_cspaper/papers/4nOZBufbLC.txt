# Count Bridges Enable Modeling And Deconvolving Transcriptomic Data

Nic Fishman1,∗, Gokul Gowri2, Tanush Kumar1, Jiaqi Lu1**, Valentin de Bortoli**3, Jonathan S. Gootenberg4,6 & **Omar Abudayyeh**5,6 1Harvard University; 2MIT; 3CNRS; 4Beth Israel Deaconess Medical Center; 5Brigham and Women's Hospital; 6Harvard Medical School ∗Corresponding author: njwfish@gmail.com

## Abstract

Many modern biological assays, including RNA sequencing, yield integer-valued counts that reflect the number of molecules detected. These measurements are often not at the desired resolution: while the unit of interest is typically a single cell, many measurement technologies produce counts aggregated over sets of cells. Although recent generative frameworks such as diffusion and flow matching have been extended to non-Euclidean and discrete settings, it remains unclear how best to model integer-valued data or how to systematically deconvolve aggregated observations. We introduce Count Bridges, a stochastic bridge process on the integers that provides an exact, tractable analogue of diffusion-style models for count data, with closed-form conditionals for efficient training and sampling. We extend this framework to enable direct training from aggregated measurements via an Expectation-Maximization-style approach that treats unit-level counts as latent variables. We demonstrate state-of-the-art performance on integer distribution matching benchmarks, comparing against flow matching and discrete flow matching baselines across various metrics. We then apply Count Bridges to two large-scale problems in biology: modeling single-cell gene expression data at the nucleotide resolution, with applications to deconvolving bulk RNA-seq, and resolving multicellular spatial transcriptomic spots into single-cell count profiles. Our methods offer a principled foundation for generative modeling and deconvolution of biological count data across scales and modalities.

## 1 Introduction

Integer-valued counts are a fundamental product of scientific measurements because of the discrete nature of molecules. Modern biological assays yield massive streams of count data: RNA-seq read counts, fluorescence imaging molecule counts, and mass cytometry ion counts (Klein et al., 2015; Raj et al., 2008; Bendall et al., 2011). However, these measurements are often aggregated over multiple individual units, obscuring the fine-grained patterns underlying these natural phenomena. Transcriptomics technologies exemplify this challenge, with technologies such as Visium capturing 10-50 cells per spot (Stahl et al., 2016) and bulk RNA-seq aggregating thousands to millions of cells ˚
per readout, yielding averages rather than high-resolution details. Deconvolving these aggregates into single-cell profiles is critical for the precise mapping of cellular heterogeneity, cell-cell interactions, and tissue architecture (Moses & Pachter, 2022; Armingol et al., 2021). The challenge is twofold: building generative models that respect the integer nature of counts and extending these models to infer unit-level profiles from aggregated observations. Recent developments in generative modelling only partially addresss the problem. Discrete diffusion models (Austin et al., 2021; Lou et al., 2023) treat counts as unordered categories through masking or uniform noise. Blackout Diffusion (Santos et al., 2023), the only count-specific approach, uses pure-death processes that cannot transport between arbitrary distributions. The biological deconvolution literature on the other hand focuses on deconvolving cell-type (cluster-level) proportions (Kleshchevnikov et al., 2022; Cable et al., 2022; Li et al., 2023), rather than unit-level count profiles. Thus, there is need for a framework that respects the integer and ordinal structure of counts, enables transport between arbitrary distributions, and can systematically deconvolve aggregated observations.

1 We introduce Count Bridges: a stochastic bridge process on Z
d using Poisson birth-death dynamics.

This yields closed-form conditionals for exact sampling and extends naturally to deconvolution via an EM algorithm treating unit-level counts as latent. The birth-death mechanism allows transport between arbitrary integer-valued distributions while preserving the ordinal structure, as both increments and decrements respect the natural ordering of counts. We show that Count Bridges outperform existing methods on synthetic benchmark datasets and scale more favorably to high-dimensional settings. We then showcase Count Bridges on two real-world biological applications centered on deconvolution: nucleotide-resolution single-cell RNA-sequence modeling for bulk RNA-seq deconvolution and reference-free spatial transcriptomic deconvolution. The codebase is available here.

## 2 Background On Diffusion Models

Diffusion models specify a time–indexed family of *bridge kernels* connecting X0 ∼ p0 to a simple source distribution X1 ∼ p1 (often Gaussian). There are two layers of structure: (i) an unconditional forward process (Xt)t∈[0,1] with kernels Kt|0(xt | x0) = Law(Xt | X0 = x0); (ii) for any 0 ≤ s ≤ t ≤ 1, a family of bridge kernels Ks|0,t(xs | x0, xt) = Law(Xs | X0 = x0, Xt = xt).

Diffusion models require two consistency properties. First we require a bridge consistency identity.

For any $0\leq s\leq t\leq u\leq1,\quad K_{s|u}(x_{s}\mid x_{u})=\int K_{s|t}(x_{s}\mid x_{t})\,K_{t|u}(x_{t}\mid x_{u})\,dx_{t}$.  
Thus multi-step sampling along any grid u→t→s matches the single-step u→s bridge. Second, the kernel must have a projective posterior:

$$K_{s|t}(x_{s}\mid x_{t})=\int q_{0|t}(x_{0}\mid x_{t})\,K_{s|0,t}(x_{s}\mid x_{0},x_{t})\,d x_{0},$$
$$(2)$$

where q0|t(x0 | xt) = Law(X0 | Xt = xt). This identity expresses Ks|t as a mixture over the posterior of the p0 data. It is essential for denoising: during sampling, each predicted Xt changes the posterior q0|t, so the reverse kernels must be projective under this posterior update.

Together, equation 1 and equation 2 lets us define a general diffusion approach. First we train a denoiser qθ that approximates the posterior, X˜0 ∼ qθ(· | xt, t) ≈ Law(X0 | Xt=xt), using tuples
(t, Xt, X0) drawn from the "global" bridge: sample x0 ∼ p0, x1 ∼ p1, t ∼ Unif[0, 1] and then Xt ∼ Kt|0,1(· | x0, x1). For sampling, pick a grid 1 = tK > · · · > t0 = 0, draw X1 ∼ p1, set XtK ← X1, sampling

$$\tilde{X}_{0}^{(k+1)}\sim q_{\theta}(\,\cdot\mid X_{t_{k+1}},t_{k+1}),\qquad X_{t_{k}}\sim K_{t_{k}\mid0,t_{k+1}}(\,\cdot\mid\tilde{X}_{0}^{(k+1)},X_{t_{k+1}}).$$

0, Xtk+1 ). (3)
By our consistency properties, this multi–step procedure is equivalent to sampling directly from the
(0, 1) bridge, so the model cannot drift out of the training distribution.

2.1 DIFFUSION AS A BRIDGE BETWEEN NOISE AND DATA
Let us consider the unconditional Kt|0 process (Xt)t∈[0,1] of the following form

$${}^{1)},X_{t_{k+1}}).$$
$$({\mathfrak{I}})$$
$$(4)$$

Xt = α(t)X0 + Bt, (4)
where (Bt)t∈[0,1] is a d-dimensional Gaussian process with non-decreasing standard deviation σ(t),
and α(t) a non-increasing function. Note that α(0)= 1 and σ(0)= 0.

We want to define a process that interpolates smoothly between X0 ∼ p0 and X1 given by another distribution as in Peluchetti (2023); Albergo et al. (2023); Delbracio & Milanfar (2023); Liu et al. (2022; 2023). We have the following proposition defining the global and local bridge.

Proposition 2.1. Let (Xt)t∈[0,1] be given by equation 4. For 0 < s < t ≤ 1*, consider* (Xs)s∈[0,t] conditioned on Xt = xt and X0 = x0. Then the conditional law Ks|0,t(· | x0, xt) *is Gaussian and* can be written

$$X_{s}\stackrel{d}{=}\alpha(s)(1-r(s,t))X_{0}+\frac{\alpha(s)}{\alpha(t)}r(s,t)X_{t}+\sigma(s)(1-r(s,t))^{1/2}Z,$$

where Z ∼ N (0,Id) is independent of (X0, Xt) and r(s, t) = α(t)
2σ(s)
2 α(s)
2σ(t)
2 *. In particular, the family*
{Ks|0,t}0≤s≤t≤1 defined by equation 5 satisfies equations 1 and 2.

(5)  $\frac{1}{2}$ . 
$=\;\;6\;\Rightarrow2$
 $\blacksquare$

Note that if X1 ∼ N (0,Id), α(1) = 0 and σ(1) = 1 we have Xt d= α(t)X0 + σ(t)Z. Furthermore, our equation 5 recovers the interpolation described in Albergo et al. (2023) with the identification α(t) → α(t)(1 − r(t)),
α(t)
α(1) 
r(t) → β(t) and σ(t)(1 − r(t))1/2 → γt.

## 2.2 Sampling The Posterior

In this paradigm the bridge is only the first of two choices that define the model. We also have to choose how to model the posterior X0|Xt, t. There are two core options: we can use differential equations to model the posterior in the limit of small steps or we can focus more directly on modeling the posterior. In Euclidean space, the former lets us learn a simple conditional expectation, whereas the latter always requires a distribution model. Infinitesimal. Consider a small backward step of size δ > 0. The local bridge between times t and t − δ is Gaussian, so conditioned on Xt = x we can write to first order in δ Xt−δ | Xt = x ≈ x − δ b(x, t) + 
√δ ξt, ξt ∼ N 0, Σ(*x, t*),
where b is the reverse-time drift and Σ is the diffusion covariance of the bridge.

The conditional law Xt−δ | Xt is Gaussian and can be computed in closed form:
b(x, t) = B1(t) x + B2(t) E[X0 | Xt = x] + b0(t), Σ(*x, t*) = Σ0(t).

The diffusion covariance depends only on t (from the Brownian increment), and the drift depends on the posterior Law(X0 | Xt) only through its mean. This justifies learning the mean qθ(x, t) ≈
E[X0 | Xt = x] (equivalently, a score or velocity) as in standard diffusion models (Song et al., 2020).

Distributional. Following De Bortoli et al. (2025) we can learn the conditional law qθ(· | xt, t) ≈ Law(X0 | Xt=xt), using any distribution learning approach. We can then sample and directly plug into the bridge X˜
(k+1)
0 ∼ qθ(· | Xtk+1 , tk+1), Xtk ∼ Ktk|0,tk+1 · | X˜
(k+1)
0, Xtk+1 .

to sample the posterior. The distributional perspective is particularly powerful when the infinitesimal perspective fails to admit a simplification to the conditional expectation, which motivates our use of the distributional approach for Count Bridges (see Sec. 3.2). In categorical discrete settings, all approaches are distributional since they are based on cross-entropy losses, see Campbell et al. (2022); Austin et al. (2021); Shi et al. (2024); Sahoo et al. (2024).

## 3 Count Bridges

3.1 AN INTEGER BRIDGE BETWEEN DISTRIBUTIONS Mirroring Sec. 2, we seek a bridge for integer-valued data. Instead of a Gaussian process, we use a pair of independent Poisson birth/death processes (Bt)t∈[0,1] and (Dt)t∈[0,1] that increment/decrement the counts. We define an increasing "jump-intensity" function w : [0, 1] → R≥0 with w(0) = 0, w(1) = 1, and then write the cumulative birth/death intensities as Λ±(t) = λ± w(t) for some λ± > 0 so Bt ∼ Poi(Λ+(t)) and Dt ∼ Poi(Λ−(t)). From here we can define the unconditional kernel Kt|0:
Xt = X0 + Bt − Dt. (6)
Denoting the displacement dt = Xt − X0, the total number of jumps Nt = Bt + Dt, and the slack variable Mt = min(Bt, Dt). Any two of these variables determine the third:
Nt = |dt| + 2Mt, Bt =
1 2
(Nt + dt), Dt = Nt − Bt. (7)
From the (Nt, Bt) perspective, Poisson superposition and thinning imply that, conditional on (Nt, Bt) at time t, the earlier counts (Ns, Bs) for *s < t* can be sampled by a Binomial draw for Ns and a Hypergeometric draw for Bs. Switching to (Mt, dt), a Poisson change of variables yields the slack posterior Mt | dt, whose pmf has Bessel form (see Prop. A.6 in App. A). These two ingredients together give a count analogue of Proposition 2.1; the full derivation is in App. A.

Proposition 3.1. Let (Xt)t∈[0,1] be given by equation 6. Now, consider (Xs)s∈[0,t] *conditioned by* Xt = xt and X0 = x0. Then, we have the Poisson Birth-Death bridge Ks|0,t:

$$X_{s}\ {\stackrel{d}{=}}\ X_{0}+B_{s}-D_{s},$$
d= X0 + Bs − Ds, (8)

![3_image_0.png](3_image_0.png)

where we condition on dt = Xt − X0 and sample Mt | dt ∼ Bes(|dt|; Λ+(t),Λ−(t)), changing variables to Nt and Bt to sample Bs, and Ds *which we can plug into equation 8:*

$$N_{s}\mid N_{t}\sim\text{Bin}\bigg{(}N_{t},\frac{w(s)}{w(t)}\bigg{)}\,,\ \ B_{s}\mid(N_{t},N_{s},B_{t})\sim\text{Hyp}(N_{t},B_{t},N_{s}),\ \ D_{s}=N_{s}-B_{s}.\tag{9}$$

$\textit{The fann}$
The family {Ks|0,t}0≤s≤t≤1 *defined by equation 8 satisfies equations 1 and 2.*
We visualize this process in Fig. 1 where we show the trajectories for the one- and two-step models along with the core composition property that drives bridge models. This setup enables training and sampling from a Count Bridge, see Algorithms 1 and 2. These results leverage our custom CUDA kernel implementing the fast Bessel sampler of Devroye (2002) to enable sampling at scale.

In Fig. 1 we also see that as dt grows the slack Mt concentrates near zero, so there is no slack. This means that Count Bridges are an instance of the static Schrodinger bridge problem (L ¨ eonard, ´
2013): they solve an entropy-regularized optimal transport. Let κ =pλ+λ− be the jump intensity and p κ ref(x0, x1) = p0(x0)Kκ 1|0
(x1|x0) be the joint law of (X0, X1) induced by the kernel. Over the space of couplings C(p0, p1) = {C on *X × X* : C(·, X ) = p0, C(X , ·) = p1}, Count Bridges solve

$$C_{\kappa}\in\arg\operatorname*{min}_{C\in{\mathcal{C}}(p_{0},p_{1})}\operatorname{KL}\!\left(C\,\|\,p_{\mathrm{ref}}^{\kappa}\right).$$

Letting κ → ∞ yields the independent coupling p0 ⊗ p1, but as κ ↓ 0 we obtain

$${\mathit{p u a t i o n s~I~a n d~2.}}$$
$$\mathrm{KL}\big(C\,\|\,p_{\mathrm{ref}}^{\kappa}\big)\;\approx\;\log\!\left({\frac{2}{\kappa}}\right)\mathbb{E}_{C}|X_{1}\!-\!X_{0}|-H(C),$$

so κ → 0 recovers discrete OT with cost |x1−x0| (see App. A.2).

This echoes the Gaussian case (Sec. 2) where we define σ = σ(1) and p σ ref, and as σ ↓ 0

$$\mathrm{KL}\big(C\,\|\,p_{\mathrm{ref}}^{\sigma}\big)\;\approx\;\frac{1}{2\sigma^{2}}\,\mathbb{E}_{C}\|X_{1}-X_{0}\|^{2}-H(C)$$

so σ → 0 recovers quadratic OT, while σ → ∞ again gives p0 ⊗ p1 (Shi et al., 2023). Thus the bridge parameters κ (count) and σ (Gaussian) play the same role as entropy–regularization strengths. 3.2 DISTRIBUTIONAL SCORING LOSS FOR THE DENOISER
Training requires a distributional loss due to the discrete nature of the space. As shown by Holderrieth et al. (2024), the ELBO for discrete generators (e.g., jump processes) is distributional and cannot be reduced to expectations over point estimates. This mirrors the need for cross-entropy in discrete diffusion and flow models. We can use cross-entropy with Count Bridges (we test this, see App. D.1), but it has two issues: first, it does not incorporate the lattice structure; second, cross-entropy cannot model the joint of Xs | Xt without exponential cost in dimension, so cross entropy is usually

Require: dataset (x0, x1), w(·),Λ±(·) 1: for each minibatch do 2: sample (x0, x1) ∼ (x0, x1) 3: t ∼ Unif[0, 1] 4: d1 ← x1 − x0 5: M1 ∼ Bes(|d1|; Λ+(1),Λ−(1)) 6: N1 ← |d1| + 2M1 7: B1 ← 12 (N1 + d1) 8: Nt ∼ BinN1, w(t) 9: Bt ∼ HypN1, B1, Nt 10: xt ← x1−2(B1−Bt)+(N1−Nt) 11: update θ on L(θ) 12: end for
Require: xtK = x1, model qθ, w(·),Λ±(·) 1: for k = K, K − 1, . . . , 1 do 2: sample xˆ0 ∼ qθ(· | xtk , tk) 3: dtk ← xtk − xˆ0 4: Mtk ∼ Bes(|dtk |; Λ+(tk),Λ−(tk)) 5: Ntk ← |dtk | + 2Mtk 6: Btk ← 12 (Ntk + dtk ) 7: r ← w(tk−1)/w(tk) 8: Ntk−1 ∼ BinNtk , r 9: Btk−1 ∼ HypNtk , Btk , Ntk−1  10: xtk−1 ← xtk−2(Btk −Btk−1)+(Ntk −Ntk−1) 11: end for 12: return xt0
Algorithm 1: Training Poisson–BD Bridge

$$-N_{t_{k-1}})$$

Algorithm 2: Sampling Poisson–BD Bridge factorized, modeling each coordinate of Xs | Xt independently or autoregressively. Specializing to count data we can go beyond cross-entropy by using a proper scoring rule that (i) incorporates the geometry and (ii) enables modeling of the joint.

Formally, let (X0, Xt) denote a training pair from Kt|0,1 at time t ∈ [0, 1], and let qθ(· | xt, t) be our denoiser. We train qθ using a strictly proper distributional scoring rule (Gneiting & Raftery, 2007; De Bortoli et al., 2025). Fix a negative-type semimetric ρ on Z
D (all our experiments focus on the ρ(*x, x*′) = ∥x − x
′∥
β 2 with β = 1). For any distribution p and outcome y, the energy score is

$S_{\rho}(p,y)=\frac{1}{2}\,\mathbb{E}_{X,X^{\prime}\sim p}\big{[}\rho(X,X^{\prime})\big{]}-\mathbb{E}_{X\sim p}\big{[}\rho(X,y)\big{]}$ and $\mathcal{L}(\theta)=\mathbb{E}_{X_{0},X_{t},t}\big{[}S_{\rho}\big{(}q_{\theta}(\,\cdot\mid X_{t},t),\,X_{0}\big{)}\big{]}$
which is strictly proper when ρ is characteristic. Taking m i.i.d. samples xˆ
(j) ∼qθ(· | xt, t) we can use the plugin estimator: Sbρ =1 m(m−1)
Pj̸=j
′
1 2 ρ(ˆx
(j), xˆ
(j
′)) −
1 m Pm j=1 ρxˆ
(j), x0.

## 4 Deconvolution With Count Bridges

We extend Count Bridges to handle unit–level generation when we only observe aggregates. Consider G units in the one-dimensional case where the group-level state at time t is a vector Xt ∈ Z
G
with entries Xgt for unit g at time t. Each entry evolves independently according to the bridge in Section 3. The key challenge: we observe the unit–level endpoint x1 but only the aggregate at time 0, a0 =PG
g=1 xg0 ∈ Z, not the unit–level vector x0. Our goal is to learn a count bridge qθ(x0 | xt*, t, z*)
that generates unit–level endpoints given start data at time t = 1 and side information z.

We formulate this as a generalized EM problem, similar to Rozet et al. (2024), where X0 is latent and a0 =Pg Xg0 is observed. Let A : Z
G→Z be a linear aggregate map (e.g., sums across units, block sums). For (xt*, t, z*), the denoiser qθ(· | xt*, t, z*) defines an i.i.d. product prior over X0 =
(X10*, . . . , X*G0). Conditioning on the aggregate yields

$$Q_{\theta}({\bf X}_{0}\mid a_{0},x_{t},t,z)\;\propto\;\Big[\prod_{g=1}^{G}q_{\theta}(X_{g0}\mid x_{t},t,z)\Big]\;{\bf1}\{A({\bf X}_{0})=a_{0}\}.$$

In the E-step we will generate "latent" x
≈
0using the model and in the M-step we will use these x
≈
0to train the model at the aggregate level. We summarize the overall procedure in Algorithms 3 and 4. E-Step The ideal E–step would sample from the exact aggregate–conditional law X⋆0 ∼ Qθ(· | a0, xt*, t, z*).

We could then use the sampled x
⋆0as latent variables to sample xt between (x
⋆0, x1) using the unit–level kernel Kt|0,1 from Prop. 3.1.1 Unfortunately, Qθ is generally intractable to sample from, 1The same method described here can be used with distributional diffusion on continuous space, but we focus on counts since most often when we observe aggregates we believe they are based on discrete underlying data.

Require: (x1, a0, z), w(·),Λ±(·), qθ, Π
1: for k = K, K − 1*, . . . ,* 2 do 2: Sample xˆ0,tk ∼ qθ(· | xtk
, tk, z)
3: x˜0,tk ← Π(xˆ0,tk
, a0, z)
4: Update xtk−1by running the reverse step 5: using steps 4–10 of Alg. 2, with x˜0,tk 6: **end for** 7: x
≈
0 ← sample and project xˆ0,t1 8: **return** x
≈ 0 Algorithm 3: Guided Sampling to for x

![5_image_0.png](5_image_0.png)

≈ 0 Require: (x1, a0, z), w(·),Λ±(·), qθ, Π
1: for each minibatch do 2: **E-step:** Sample latent x
≈
0from 3: x1 conditional on a0 via Alg. 3 4: **M-step:** t ∼ Unif[0, 1]
5: Sample xt via the forward bridge on 6: (x
≈ 0
, x1) using steps 4–10 of Alg. 1 7: Update θ using the gradient of −Lagg(θ)
8: **end for**
Algorithm 4: Training with Aggregate Supervision Figure 2: A scaled and rounded variant of the classic 8 gaussian to two moons task. Here we compare the trajectories of continuous flow matching, discrete flow matching, and count bridges.

CB achieves the lowest W2, MMD, and EMD, see Table 6.

given just a unit-level model, so we approximate it through the diffusion sampling process itself.

Starting from x1, we run the sampling process as in Algorithm 2, but at each timestep tk we: (1) predict xˆ0 ∼ qθ(· | xtk, tk, z), (2) project xˆ0 to satisfy the aggregate constraint (see Sec. 4), yielding x˜0, and (3) perform the sampling step using x˜0 as the predicted endpoint. This projection–guided diffusion ensures the aggregate constraint is incorporated throughout the denoising trajectory (see Alg. 3). This process produces latent x
≈
0samples that are consistent with the aggregate constraints, which we can then use in the M-step to train the model. M-Step With these unit-level samples in hand, the M–step runs the bridge process as in Section 3. But instead of computing the loss on the unit-level latents, we compute the loss with respect to the aggregates. Given the ground-truth aggregate a0, we lift the same strictly proper score to aggregates:
S
A ρ
*p, a*=
1 2 Ep-ρ(A(X), A(X
′))− Ep-ρ(A(X), a)and Lagg(θ) = EA0,Xt,thS
A
ρ qθ(· | Xt*, t, z), A*0i with the plug-in obtained by sampling Xˆ
(j)
0 ∼ qθ(· | Xt*, t, z*) and forming aˆ
(j) = A(Xˆ
(j)
0).

Approximate Sampling from the conditional distribution Given a predicted endpoint xˆ0 from our diffusion model and target aggregate a0, we need to sample from the conditional distribution Qθ(· | A(X0) = a0). While this is intractable, we can derive a principled approximation. Proposition 4.1 (First–order aggregate projection). Let A(X0) *be the aggregate, and let* p0 be the prior law of X0*. Under the regularity conditions in App. B.1, the aggregate–conditional law* Qθ(· | A0 = a0) *admits a first–order exponential tilt. The corresponding generalized KL projection* Π(x0) = arg min y0: A(y0)=a0 DKL(y0∥x0)
gives a kind of first–order approximation to P
Qθ(· | A0 = a0)*. For an elementwise sum* A(x0) =
g xg0 *this projection is the simple scaling* Π(x0)g = a0xg0/(Pg
′ xg
′0).

The proposition shows that the natural rescaling operation is not ad hoc, but can be justified as a kind of first-order approximation to the true conditional distribution in a large sample regime (see Appendix B.1). When unit-level training data exist, we can learn a projection Πψ(xˆ0*, z, a*0) that actually enables sampling conditional on the mean. See Sec. 6 where we show how to learn such a projection.

## 5 Related Works

Stochastic interpolants. Our formulation allows us to transport any integer-valued distribution p1 to another integer-valued distribution p0. In the case of Euclidean state space early works such 6 as (De Bortoli et al., 2021; Vargas et al., 2021; Chen et al., 2021) have shown how to perform such an interpolation leveraging (Entropic) Optimal transport and the concept of Schrodinger Bridges. ¨ In more recent works, ignoring the Optimal Transport constraints, several works have proposed to bridge distributions in a more relaxed formulation leveraging the concept of Markov projection, see Peluchetti (2023); Albergo et al. (2023); Delbracio & Milanfar (2023); Liu et al. (2022; 2023) for instance. Those frameworks can be shown to be strictly equivalent to diffusion models in the case where one of the end distribution is a unit Gaussian, see Gao et al. (2025). However, those works are limited to the Euclidean setting, and extension to the integer-valued setting is required. Discrete diffusion models. Recently, with the advent of language diffusion models such as Ye et al. (2025); Song et al. (2025); Sahoo et al. (2024); Shi et al. (2024); Ou et al. (2024a); Arriola et al. (2025); Nie et al. (2024); Zheng et al. (2023), discrete diffusion models have gained considerable traction. Most works rely on discrete equivalents of the original formulation of diffusion models, explicitly or implicitly replacing the continuous Gaussian noising process by a Continuous-Time Markov Chain (CTMC) (Austin et al., 2021; Campbell et al., 2022; Lou et al., 2023; Campbell et al., 2024; Kitouni et al., 2024; Sun et al., 2023). Other approaches include relying on some Euclidean relaxation (Chen et al., 2022) or modelling the space of probability (Avdeyev et al., 2023; Stark et al., 2024). Similarly, flow matching techniques have been extended to cover this paradigm (Gat et al., 2024). Most of these works focus on *categorical* data and therefore consider uninformed forward process such as uniform or masking process. In contrast, in this work, we focus on ordinal data. To the best of our knowledge, the only existing work that also deals with such a process is Blackout Diffusion (Santos et al., 2023), which considers a pure-death process where an image is taken to the all-zero limit, as opposed to an endpoint conditioned bridge. Our approach generalizes this setup in two ways: first, we allow births and deaths at every time, recovering their pure birth construction in the limit as κ → 0; second, we generalize the process to a bridge which can transport X1 to X0. Finally, we highlight that diffusion models have been extended to the very general setting where only an *infinitesimal generator* is available Benton et al. (2024); Holderrieth et al. (2024). While our work can be seen as an instanciation of this general framework, these general frameworks do not give any information regarding the design of the forward process for integer-valued data, the specific parameretization in terms of slack variables and the necessity of the distributional diffusion loss. Distributional Diffusion Models. In De Bortoli et al. (2025); Shen et al. (2025), the authors learn the conditional distribution p0|t(x0|xt) through the use of scoring rules, going beyond the classical training framework of diffusion, which approximates the conditional mean E[X0|Xt = xt]. The importance of approximating the covariance was already noted by Nichol & Dhariwal (2021) and further analyzed in (Ho et al., 2020; Nichol & Dhariwal, 2021; Bao et al., 2022a;b; Ou et al., 2024b).

In a similar flavor (Xiao et al., 2022) uses a GAN to approximate p0|t(x0|xt).

Sequence-to-expression models An ambitious goal in biology is to predict gene expression from DNA sequence information. There have been several attempts to train deep learning models for sequence-to-expression prediction tasks (Barbadilla-Mart´ınez et al., 2025), including Enformer (Avsec et al., 2021), a state-of-the-art transformer-based DNA sequence model. While powerful, Enformer, like the vast majority of sequence-to-expression models, was trained on bulk gene expression data and is not able to predict single-cell expression profiles, missing the cellular heterogeneity and fine-grained regulatory patterns that shape tissue function. Spatial transcriptomic deconvolution Spatial transcriptomics encompasses a family of recently developed techniques which measure gene expression and spatial location in tissues. The majority of these techniques are not capable of resolving individual cells, instead providing aggregate information over small neighborhoods consisting of on the order of tens of cells (Moses & Pachter, 2022). To address this limitation, a number of deconvolution methods have been developed to infer singlecell level information (Li et al., 2023). The majority of these methods, including cell2location
(Kleshchevnikov et al., 2022) and RCTD (Cable et al., 2022), require a paired non-spatially resolved scRNA-seq atlas, and output cluster-level mixture proportions rather than single cell counts. The ideal deconvolution would recover full single-cell count profiles directly from spatial data without requiring external reference atlases. DestVI (Lopez et al., 2022), which outputs count profiles but requires a reference, and STDeconvolve (Miller et al., 2022) which does not require a reference but outputs cluster-level predictions, both take steps toward this goal.

![7_image_0.png](7_image_0.png)

## 6 Applications

We evaluate with three distributional metrics: the Energy score, the Wasserstein-2 distance, and the MMD (RBF). For deconvolution, we evaluate cell-type proportion predictions using RMSE, the Jensen-Shannon Divergence (JSD), and Spearman correlation following Li et al. (2023). Synthetic tasks have std. errors over 3 training seeds; main applications have std. errors 3 over inference seeds.

## 6.1 Synthetic Distributions

Here, we benchmark count bridges (CB) against continuous flow matching (CFM) (Lipman et al.,
2022) and discrete flow matching (DFM) (Gat et al., 2024) across a range of synthetic experiments. Discrete 8-Gaussians to 2-Moons. We adapt this classic task to the integers. We plot the learned trajectories in Fig 2. Qualitatively CB achieves the best performance. DFM is much more competitive in this experiment than CFM, but DFM trajectories are decoupled from the underlying geometry, whereas CB produces OT-like trajectories similar to CFM. These qualitative evaluations are confirmed quantitatively: CB achieves the best performance across W2, Energy, and MMD (see App. D.1).

Scaling in Low-Rank Gaussian Mixtures. To test scalability to higher dimensions, we construct integer-valued datasets with fixed intrinsic dimensionality while ambient dimension d increases in powers of two from 4 to 512. Each dataset is a 5-component Gaussian mixture with latent rank r = 3, projected to Z
d. In Fig. 3 see that CB has the best scaling in dimensionality (see App. D.2 for more).

Deconvolution of Gaussian Mixtures. We extend the lowrank mixture task to evaluate deconvolution capabilities. In this experiment, each observation is an aggregate constructed by summing a group of G samples. For each group, the G samples are drawn from a group-specific Gaussian mixture whose component weights are sampled from a Dirichlet distribution with concentration parameters (α1*, . . . , α*5). The labels of the G
source components are provided as unit-level side information. We then vary the size of the group G and the extent of variation between groups by changing the concentration parameter α (see Appendix D.3 for details). In Fig. 4 we see performance degrades as groups become more uniform and larger. We explore the theoretical limits to deconvolution in Apps. B.2 and B.3, which confirm that deconvolution requires between-group heterogeneity to enable identification, which is inherently lost as groups become large. Despite these limits, we demonstrate practical deconvolution on moderately-sized groups in our spatial transcriptomics application (Section 6.3).

Figure 4: Deconvolution of the low-

![7_image_1.png](7_image_1.png) rank Gaussian mixture across different group sizes and levels of between-group heterogeneity.

6.2 MODELLING GENE EXPRESSION AT SINGLE-CELL AND SINGLE-NUCLEOTIDE RESOLUTION A central goal in biology is to understand the relationship between DNA sequence and gene expression. Many models relate sequence and expression, the most prominent of which, such as Enformer (Avsec

| Method              | Bulk MSE     | CT MSE       | Comparison    | MMD   | W2    | Energy   |
|---------------------|--------------|--------------|---------------|-------|-------|----------|
| Fine-tuned Enformer | 2.590        | 3.142        |               |       |       |          |
| Count Bridge        | 0.601 ±0.000 | 1.410 ±0.002 | Bulk mean     | 0.515 | 0.208 | 56.800   |
| Count Bridge        | 0.446 ±0.000 | 0.182 ±0.001 | 28.583 ±0.003 |       |       |          |

Table 1: Nucleotide-level MSE for bulk and bulked cell-type (CT) specific predictions.

Table 2: Gene expression count profile deconvolution error for bulk RNA sequencing data.

et al., 2021), are Transformer-based models that predict expression from sequence. More recent work has explored fine-tuning Enformer on single-cell data (Hingerl et al., 2024). On the other hand, there is a mature literature on deconvolving bulk RNA-seq (Newman et al., 2019; Wang et al., 2019). These methods operate at the gene (rather than nucleotide) level, leveraging bulk cell-type profiles or single-cell references to deconvolve bulk profiles into cell-type proportions (not count profiles). We use CBs to jointly model sequence and single-cell expression counts in scRNA-seq data, and to enable nucleotide-level deconvolution of bulk profiles. To validate CBs in this setting, we demonstrate two key results. First, we show that CBs trained on single-cell data produce meaningful count profiles and outperform a fine-tuned Enformer model on sequence-to-expression prediction. Second, we show that conditioning CBs on bulk profiles enables deconvolution of bulk gene expression into inferred single-cell gene expression profiles. We validate these deconvolved profiles distributionally and show that they achieve state-of-the-art performance relative to cell-type proportion deconvolution models.

Modeling sequence and single-cell counts We train CBs on PBMC scRNA-seq counts at nucleotide resolution using 106cells across 103 donors (Yazar et al., 2022). Each training example corresponds to a nucleotide position in a single cell, and is represented by the noisy count xt and diffusion time t from the CB forward process, a cell-type embedding, a local genomic context z obtained by encoding the surrounding DNA sequence with Enformer, and i.i.d. noise ζ for the distributional loss. These features are concatenated and passed through residual multi–head attention blocks and a final softplus head that parameterizes the conditional count distribution X0|Xt*, t, z*. The model is trained directly on unit-level (single-cell) expression profiles rather than only on aggregated counts. During training we randomly mask cell-type labels so that the model supports both unconditional and cell-type-conditional sampling at test time. Learned projection for deconvolution Since we have unit-level data we can learn a better projection operator than the simple rescaling function in Prop. 4.1. We augment the CB with a small projection module Πψ, an attention block operating on each nucleotide (represented by z) across cells in the batch. Given an initial CB prediction xˆ0, an observed aggregate a0, and the noisy state xt, the module outputs x˜0 = Πψ(xˆ0, a0, xt, z), we train this using the distributional loss to learn to sample X0 | A(X0)=a0, Xt, t. To support both unconditional and aggregate-conditioned inference, we apply the projection module only on a random 10% of training examples where a0 is provided. Bulk gene expression We first evaluate the ability of our model to predict expression from sequence, both unconditionally and conditional on cell type. As a baseline, we use an Enformer model finetuned directly on the PBMC dataset. We find that Count Bridge predictions outperform fine-tuned Enformer (Table 1, for results by cell type and further details see App. E).

Deconvolved profiles We can use this unit-level model for deconvolution tasks: we can condition on an aggregate (bulk profile) to sample single-cell profiles from the model while matching that aggregate. We next evaluate the ability of CBs to deconvolve mixtures of cell types from heldout individuals. We held out 10% of patients from our training set and synthetically bulked these patients. Since we have the ground truth data, we can then evaluate deconvolution quality. We first evaluate the distributional quality of these predictions against the bulk mean, further validating the CB count profiles (Table 2). As a more robust set of baselines, we compare to CIBERSORTx (Newman et al., 2019) and MuSiC (Wang et al., 2019). To facilitate comparison, we aggregate our nucleotide-level predictions into gene counts and assign each of our deconvolved cells to the closest Table 3: Cell-type proportion deconvolution error for nucleotide level bulk RNA sequencing data.

| Method       | JSD          | RMSE         | Spearman     |
|--------------|--------------|--------------|--------------|
| CIBERSORTx   | 0.194        | 0.109        | 0.079        |
| MuSiC        | 0.313        | 0.140        | 0.186        |
| Count Bridge | 0.113 ±0.001 | 0.073 ±0.000 | 0.267 ±0.005 |

cell type. CBs achieve better performance on JSD, RMSE, and Spearman correlation while providing nucleotide-level counts (Table 3). In App. E we plot the UMAP for qualitative comparison. 6.3 DECONVOLVING SPATIAL TRANSCRIPTOMIC SPOTS INTO SINGLE-CELL COUNTS Next, we show how CBs can be used to infer single cell gene expression profiles from spot-level aggregates in spatial transcriptomic data. In spatial transcriptomic data generated by Visium (Stahl ˚ et al., 2016), it is common to have access to side information beyond the spot-level count aggregates. In particular, many datasets include images of the cells with a nuclear stain (Palla et al., 2022). CBs provide a natural way to leverage this cell-level side information to deconvolve aggregate count data. Modeling spatial aggregates We train CBs on a MERFISH mouse brain dataset (Vizgen, 2021), which is resolved at the single-cell level, and artificially aggregate neighborhoods of cells to simulate spot-level Visium data. This synthetic dataset gives us access to spot-level aggregates and their corresponding single-cell ground truth, as well as single-cell nuclear images. Following the notation in Sec. 4, the spot-level counts can be treated as aggregates a0, and single-cell images can be treated as unit-level side information z. In this application, we never observe single-cell count profiles, only spot-level aggregates and the single-cell images. We leverage a UViT (Bao et al., 2023) extended to incorporate count and noise patches (see App. F). We use a simple source distribution X1 ∼ Poi(10).

Cell type proportions We benchmark CBs against STDeconvolve (Miller et al., 2022), a widely used spatial transcriptomic deconvolution method which is state-of-the-art among reference-free approaches Li et al. (2023) (see Appendix F for comparisons to reference-based methods). STDeconvolve outputs cell type (cluster identity) proportions for each spot rather than single cell counts. As such, we evalute the quality of deconvolution by comparing the predicted cell type proportions to the true cell type proportions per spot. For CBs, which provide single-cell count profile predictions rather than cell type proportions, we assign each predicted count profile its nearest neighbor cell type in order to compare against STDeconvolve. CBs outperforms STDeconvolve on both the JSD and the RMSE (Table 4).

STDeconvolve 0.288 0.177 0.255 Count Bridge **0.231**
±0.002**0.110**
±0.001**0.332**
±0.001 Table 4: Cell-type proportion deconvolution error for spatial transcriptomics.

| Method       | JSD          | RMSE         | Spearman     |
|--------------|--------------|--------------|--------------|
| STDeconvolve | 0.288        | 0.177        | 0.255        |
| Count Bridge | 0.231 ±0.002 | 0.110 ±0.001 | 0.332 ±0.001 |

Count profiles We next evaluate the quality of the count profiles inferred by CBs. Here, because STDeconvolve does not provide these predictions, we instead consider a simple baseline: predicting the spot-level mean (a0/G) for each cell. This baseline, while seemingly naive, is actually biologically wellmotivated. In spatial transcriptomics, cells within a spot represent local tissue organization where neighboring cells coordinate their functions (Armingol et al., 2021). As such, we expect cells in spatial neighborhoods to have correlated gene expression profiles, making the spot mean a reasonable baseline. Nonetheless, CBs outperform the spot-level mean baseline (see Table 5), showing CBs can learn meaningful unit-level distributions from real-world aggregate data. In App. F we provide a more detailed biological evaluation of the cell types and pathways in our generated data, alongside the UMAP to facilitate qualitative comparison.

Table 5: Gene expression count profile deconvolution error for spatial transcriptomics

| Comparison   | MMD          | W2           | Energy       |
|--------------|--------------|--------------|--------------|
| Spot mean    | 0.409        | 0.030        | 41.717       |
| Count Bridge | 0.203 ±0.000 | 0.017 ±0.000 | 8.903 ±0.014 |

## 7 Conclusion

Count Bridges offer a tractable, discrete-native alternative to continuous diffusion models, unifying direct count generation with deconvolution from aggregates. We demonstrate the power of Count Bridges for nucleotide-level deconvolution of bulk RNA-seq and spatial transcriptomic deconvolution.

Limitations (i) When counts are well-approximated as continuous, Euclidean models may match or exceed performance. (ii) Identifiability in pure deconvolution degrades as group sizes grow or between-group heterogeneity shrinks, so our EM procedure is most reliable at moderate aggregation. (iii) The projection step we use is a first-order surrogate and lacks serious theoretical support.

Despite these caveats, Count Bridges lay the groundwork for rigorous discrete generative modeling and invite future work on a deeper understanding of the projection-guided sampler, sharper identifiability bounds, and generally stronger guarantees for projection-guided EM. Ethics Statement. This study uses publicly released, de-identified single-cell and spatial transcriptomics datasets under their respective licenses; no new human subject data were collected, and institutional review board (IRB) approval was therefore not required. We do not foresee serious ethical implications to Count Bridges beyond the risks already posed by standard diffusion/flow matching models. Our deconvolution methods could possibly pose some additional privacy risks. We used LLMs to help draft portions of the code used in our experiments and to edit portions of this manuscript. All our models are intended for research use only, not clinical use. LLMs were not used in any way significantly outside the current norms of academic research. Reproducibility Statement. We have taken significant steps to ensure that all results presented in this work are reproducible. An anonymous source code repository is provided here, containing complete implementations of the Count Bridge framework, including model architectures, training procedures, projection-based deconvolution, and evaluation pipelines. The appendix includes full mathematical derivations and proofs of all theoretical claims. We also provide descriptions of all data preprocessing steps for synthetic benchmarks, PBMC sequence-to-expression prediction, and spatial transcriptomic aggregation, as well as architectural and hyperparameter specifications. Together, these materials are intended to allow independent researchers to fully reproduce our theoretical and empirical findings.

## References

Nist digital library of mathematical functions. https://dlmf.nist.gov/, 2025. See §10.41(ii).

Accessed 2025-09-24.

Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. *arXiv preprint arXiv:2303.08797*, 2023.

Erick Armingol, Adam Officer, Olivier Harismendy, and Nathan E Lewis. Deciphering cell-cell interactions and communication from gene expression. *Nature reviews. Genetics*, 22(2):71–88, February 2021. ISSN 1471-0056,1471-0064. doi: 10.1038/s41576-020-00292-x.

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. *arXiv preprint arXiv:2503.09573*, 2025.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, and Jian Zhou. Dirichlet diffusion score model for biological sequence generation. In *International Conference on Machine Learning*, pp.

1276–1301. PMLR, 2023.

Ziga Avsec, Vikram Agarwal, Daniel Visentin, Joseph R Ledsam, Agnieszka Grabska-Barwinska, ˇ
Kyle R Taylor, Yannis Assael, John Jumper, Pushmeet Kohli, and David R Kelley. Effective gene expression prediction from sequence by integrating long-range interactions. *Nature methods*, 18 (10):1196–1203, 4 October 2021. ISSN 1548-7091,1548-7105. doi: 10.1038/s41592-021-01252-x.

Fan Bao, Chongxuan Li, Jiacheng Sun, Jun Zhu, and Bo Zhang. Estimating the optimal covariance with imperfect mean in diffusion probabilistic models. In International Conference on Machine Learning, 2022a.

Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In International Conference on Learning Representations, 2022b.

Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22669–22679, 2023.

Luc´ıa Barbadilla-Mart´ınez, Noud Klaassen, Bas van Steensel, and Jeroen de Ridder. Predicting gene expression from DNA sequence using deep learning models. *Nature reviews. Genetics*, 26(10): 666–680, 13 May 2025. ISSN 1471-0056,1471-0064. doi: 10.1038/s41576-025-00841-2.

Sean C Bendall, Erin F Simonds, Peng Qiu, El-Ad D Amir, Peter O Krutzik, Rachel Finck, Robert V
Bruggner, Rachel Melamed, Angelica Trejo, Olga I Ornatsky, Robert S Balderas, Sylvia K Plevritis, Karen Sachs, Dana Pe'er, Scott D Tanner, and Garry P Nolan. Single-cell mass cytometry of differential immune and drug responses across a human hematopoietic continuum. Science (New York, N.Y.), 332(6030):687–696, 6 May 2011. ISSN 0036-8075,1095-9203. doi: 10.1126/science. 1198704.

Joe Benton, Yuyang Shi, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. From denoising diffusions to denoising markov models. *Journal of the Royal Statistical Society Series B:* Statistical Methodology, 86(2):286–301, 2024.

Dylan M Cable, Evan Murray, Luli S Zou, Aleksandrina Goeva, Evan Z Macosko, Fei Chen, and Rafael A Irizarry. Robust decomposition of cell type mixtures in spatial transcriptomics.

Nature biotechnology, 40(4):517–526, April 2022. ISSN 1087-0156,1546-1696. doi: 10.1038/
s41587-021-00830-w.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.

Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. arXiv preprint arXiv:2402.04997, 2024.

Tianrong Chen, Guan-Horng Liu, and Evangelos A Theodorou. Likelihood training of schr\" odinger bridge using forward-backward sdes theory. *arXiv preprint arXiv:2110.11291*, 2021.

Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. *arXiv preprint arXiv:2208.04202*, 2022.

Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrodinger ¨
bridge with applications to score-based generative modeling. *Advances in neural information* processing systems, 34:17695–17709, 2021.

Valentin De Bortoli, Alexandre Galashov, J Swaroop Guntupalli, Guangyao Zhou, Kevin Murphy, Arthur Gretton, and Arnaud Doucet. Distributional diffusion models with scoring rules. arXiv preprint arXiv:2502.02483, 2025.

Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. *arXiv preprint arXiv:2303.11435*, 2023.

Luc Devroye. Simulating bessel random variables. *Statistics & probability letters*, 57(3):249–257, 2002.

C Dom´ınguez Conde, C Xu, L B Jarvis, D B Rainbow, S B Wells, T Gomes, S K Howlett, O Suchanek, K Polanski, H W King, L Mamanova, N Huang, P A Szabo, L Richardson, L Bolt, E S Fasouli, K T Mahbubani, M Prete, L Tuck, N Richoz, Z K Tuong, L Campos, H S Mousa, E J Needham, S Pritchard, T Li, R Elmentaite, J Park, E Rahmani, D Chen, D K Menon, O A Bayraktar, L K James, K B Meyer, N Yosef, M R Clatworthy, P A Sims, D L Farber, K Saeb-Parsy, J L Jones, and S A Teichmann. Cross-tissue immune cell analysis reveals tissue-specific features in humans.

Science (New York, N.Y.), 376(6594):eabl5197, 13 May 2022. ISSN 0036-8075,1095-9203. doi:
10.1126/science.abl5197.

Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin Patrick Murphy, and Tim Salimans. Diffusion models and gaussian flow matching: Two sides of the same coin. In The Fourth Blogpost Track at ICLR 2025, 2025.

Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and Yaron Lipman. Discrete flow matching. *Advances in Neural Information Processing Systems*, 37: 133345–133385, 2024.

Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation.

Journal of the American statistical Association, 102(477):359–378, 2007.