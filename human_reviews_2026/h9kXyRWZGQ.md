# Sublinear iterations can suffice even for DDPMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
SDE-based methods such as denoising diffusion probabilistic models (DDPMs) have shown remarkable success in real-world sample generation tasks. Prior analyses of DDPMs have been focused on the exponential Euler discretization, showing guarantees that generally depend at least linearly on the dimension or initial Fisher information. Inspired by works in log-concave sampling (Shen & Lee, 2019), we analyze an integrator -- the denoising diffusion randomized midpoint method (DDRaM) -- that leverages an additional randomized midpoint to better approximate the SDE. Using a recently-developed analytic framework called the "shifted composition rule", we show that this algorithm enjoys favorable discretization properties under appropriate smoothness assumptions, with sublinear $\widetilde{O}(\sqrt{d})$ score evaluations needed to ensure convergence. This is the first sublinear complexity bound for pure DDPM sampling --- prior works which obtained such bounds worked instead with ODE-based sampling and had to make modifications to the sampler which deviate from how they are used in practice. We also provide experimental validation of the advantages of our method, showing that it performs well in practice with pre-trained image synthesis models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper present an analysis of a stochastic integrator for the DDPM SDE.
It is based on an intermediate random middle point that is used to estimate evaluate the score in place of the initial point of the time interval.
A sublinear convergence theorem is proven: the bound is in KL divergence towards an intermediary distribution that is close to the data distribution in Wasserstein distance.
Numerical experiments tend to support the superiority of the proposed sampler.

### Strengths
The proposed sampler is sound and rely on previous literature.
Theorem 3 provides sublinear complexity bound (towards a somewhat obscure intermediate probability).
The appendix material for the proof of Theorem 3 seems well-written and documented (did not check the proof).

### Weaknesses
From the abstract one can read "prior works which obtained such bounds worked instead with ODE-based sampling and had to make modifications to the sampler which deviate from how they are used in practice." 
A similar claim line 304 "This is an option that we cannot afford in this work, as our goal is to simply analyze a discretization of the vanilla DDPM reverse process without further algorithmic modifications." 
But the proposed work studies Algorithm 1 that: 
* requires two score evaluation per iteration (OK but should be hilighted)
* is proven convergent using some specific decaying step size only discussed in Appendix (Equation A.2 line 954)
* In addition, the convergence is only proven through the use of an intermediate distribution $\pi^{\mathrm{approx}}$, with a mixed role for KL divergence and $W_2$-distance (see discussion line 284).

Due to the difference in sampling schemes, the comparison experiments in Section 5 lack clarity.
For the OU process, what are the step size used for EMD and EED?
For RMD, is it the step size from Equation (A.2)? 
Why isn't this choice discussed in the main paper?
Why figure 1 stops at 64 NFEs while standard DDPM would use 1k or 2k steps (Ho et al 2020, Song et al 2021)?
Why is there no comparison with a predictor-corrector scheme that is computationally closer (two score evaluations and two Gaussian noise per iteration)?

"Figure 3: Quantitative results: Deterministic sampling:" Is RMD deterministic?

Minor remarks: 
* the presentation of Equation RMD could be made more consistent with Algorithm 1 by using $t_{k-1}+\tau_k$ instead of $t$
* DDRaM not defined in the main text (but in abstract)
* Figure 4 and 5 are in the appendix, which is not clear when reading the text line 463 and 470.
* line 744: $Y^{\mathrm{aux}}$ or $Y^{\mathrm{alg}}$ ?
* line 755: Thenn

### Questions
See questions in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces the Denoising Diffusion Randomized Midpoint (DDRaM) method, a new SDE-based integrator for diffusion models that achieves sublinear $\sqrt{d}$ computational complexity in score evaluations under smoothness assumptions. Using the shifted composition rule, the authors provide the first theoretical sublinear convergence guarantee for pure DDPM sampling.

### Strengths
(1) The paper analyzes a stochastic sampler with the random point method, and proves that the KL and $W_2$ divergence can be controlled with iteration complexity that has a sublinear dependence on the dimension $d$.

(2) It conducts experiments to demonstrate the superiority of the randomized point method.

### Weaknesses
(1) The equations for reverse SDE seem incorrect. It can be seen from related works, e.g., [1], that there is no $\gamma$ there.

(2) The keyword, DDPM, is not very accurate. Actually, it should be termed as sampling diffusion with stochastic samplers, or SDE. This is because DDPM is only a special case of score-based diffusion when taking the limit in the length of time steps. However, the denoising diffusion model introduced in this paper is more related to the score-based SDE, instead of the original DDPM paper.

(3) The argument regarding why we approximate some distribution $\pi_{approx}$ is not convincing. The readers compare it with $\pi_{\delta}$ with early stopping. However, this is because when $\delta$ is small, $\pi_{\delta}$ is at least close to $\pi_0$ in $W_2$ distance. To make your argument reasonable, at least some similar $W_2$ guarantee should be provided. In contrast, in the main text, how it is defined is never specified. When I check the proof (Lemma 6), I find that this distribution even depends on the transition kernel of $P_{alg}$. As a result, the kind of guarantee is very unusual. And the comparison with prior works under this metric is unfair.

(4) The paper misses a closely related work [2] when introducing the analysis of DDIM.

---

[1] Nearly d-linear Convergence Bounds For Diffusion Models Via Stochastic Localization Benton et al., 2024 ICLR

[2] Unified Convergence Analysis for Score-Based Diffusion Models with Deterministic Samplers ICLR

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents an important theoretical result for SDE-based diffusion samplers (like DDPMs). Until now, the theoretical complexity for ODE-based samplers was well-studied, but the limit for SDE-based samplers was still O(d), which is linear to the data dimension d. This created a large gap from practice, where O(1) steps can make good samples. This paper uses a new analysis called Denoising Diffusion Randomized Midpoint Method (DDRaM) to prove for the first time that SDE-based DDPM samplers can achieve O(sqrt(d)) sublinear complexity.

### Strengths
1. The biggest contribution is providing the first sublinear O(sqrt(d)) proof for DDPM (SDE). This is a very important step forward in diffusion model theory. Many prior works (including Li & Jiao, ICLR 2025) achieved sublinear complexity for ODE (DDIM), but they could not solve the problem for DDPM because of its stochasticity. This paper successfully fills this important theoretical gap.

2. Besides the theory, the paper shows that the proposed DDRaM method works well in practice. In experiments on AFHQv2 (Figures 1, 2), DDRaM consistently shows better performance (lower FID, FD DINOv2) than standard samplers like Euler-Maruyama (EMD) or Exponential Euler (EED). This shows the proposed analysis is not just for theory but also has practical benefits.

### Weaknesses
My only one concern is about the decreasing importance of the DDPM sampler itself. In practice, many researchers are trying to develop samplers with very small NFE (like DDIM, DPM-Solvers) to make generation faster. Or, they train the model differently from the beginning (like Consistency Models or Rectified Flow). It is clear that proving sublinear complexity for DDPM was a very difficult problem, but I am a little unsure if solving this problem is as important as prior works on DDIM, which is more widely used in practice.

### Questions
In L1355 (Appendix C.2.1), you mentioned using numerical functions like `scipy.integrate.quad` and `scipy.optimize.root_scalar` for quadrature and root finding. Doesn't this add significant latency to the sampling process at each step?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper analyzes the denoising diffusion randomized midpoint method (DDRaM) — an SDE-based integrator for denoising diffusion probabilistic models (DDPMs), inspired by log-concave sampling (Shen & Lee, 2019). Using the "shifted composition rule" framework, it shows DDRaM needs sublinear score evaluations for convergence (the first sublinear complexity bound for pure DDPM sampling. Experimental validation confirms DDRaM works well with pre-trained image synthesis models.

### Strengths
- As far as I know, this is indeed the first $O(\sqrt{d})$ order error bound for a stochastic sampler in the diffusion model area.
- The paper is clearly written, and the idea is easy to follow.

### Weaknesses
- In the paper of Shen & Lee (2019), their analysis requires the target distribution to be a log-concave one. I am not sure if the Assumptions 1,2,3 in this paper can lead to the conclusion that the marginal distribution will be log-concave. Or could you explain how you could surpass this condition?
- The experiments validate the usage of the DDRaM method, but it does not involve popular stochastic samplers like DDPM itself, EDM-stochastic, PNDM-Stochastic, and DPM-Solver-Stochastic for comparison. Actually, I suppose these high-order stochastic samplers may also possess the property of requiring only sublinear score evaluations to ensure convergence. I encourage the authors to analyze these high-order samplers in practical use.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1
