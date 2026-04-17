# Alternating Diffusion for Proximal Sampling with Zeroth Order Queries

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
This work introduces a new approximate proximal sampler that operates solely with zeroth-order information of the potential function. Prior theoretical analyses have revealed that proximal sampling corresponds to alternating forward and backward iterations of the heat flow. The backward step was originally implemented by rejection sampling, whereas we directly simulate the dynamics. Unlike diffusion-based sampling methods that estimate scores via learned models or by invoking auxiliary samplers, our method treats the intermediate particle distribution as a Gaussian mixture, thereby yielding a Monte Carlo score estimator from directly samplable distributions. Theoretically, when the score estimation error is sufficiently controlled, our method inherits the exponential convergence of proximal sampling under isoperimetric conditions on the target distribution. In practice, the algorithm avoids rejection sampling, permits flexible step sizes, and runs with a deterministic runtime budget. Numerical experiments demonstrate that our approach converges rapidly to the target distribution, driven by interactions among multiple particles and by exploiting parallel computation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Method for non-parametric diffusion based approximation of proximal sampling exploiting zeroth-order information
with particle interaction. The authors use a Gaussian mixture surrogate for denoising the particles in the alternating proximal sampling process. They avoid the computation of gradient of $f$ for guidance in the SDEs and use of wasteful rejection sampling for denoising step. This leads to a multi particle sampling algorithm with deterministic runtime and exponential convergence guarantees with accurate score function esitmation

### Strengths
The authors present a very clear motivation and methodology for their work. The background is covered in a structure manner and the theoretical analysis and the experiments support the claims in the paper. The paper is very well written and the contributions are noteworthy. 

The proposed sampling method uses simple surrogate to bypass inefficient use of rejection sampling and approximation of the score function. The method relies on noisy approximation of the score function of the denoising SDE, which derives it from zeroth order information and multi-particle interactions, improving the coverage of the disconnected modes of the distributions, and alleviating the need for projection and convex support.

### Weaknesses
The introduction stresses that the proximal algorithm should be scalable. However the experiments are limited to small $d$ scenarios. It would be informative to have additional results showing convergence time vs $d$ and compare the algorithm with the baselines.

Typo: 146 $B_t^\leftarrow$ is backward Brownian motion.

### Questions
The Monte Carlo estimation of the score function is increasing the overall computation with $M$ - But in the experiment section I didn't understand how the authors used large $M$ and compare to algorithms that don't have that many parallel threads in a fair setup. Can the authors provide a description of the fairness?

Mode coverage in the target is attributed to both the ability to use larger noise scale $h$ and the inverse weight of $x$ in high concentration regions. But the two seem to be in contention in $\hat q_{k+1/2}(\cdot|X_k)$, i.e close to $X_k$ and with small $h$ the component weights in Eq 10 will be smaller. Doesn't increasing $h$ at the same time make the weights more uniform?

Have the authors studied the convergece time vs $M$ and $N$ when $N\times M$ is fixed?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a canonical isoperimetric sampling problem via a localized reverse heat flow and develops theoretical guarantees by leveraging the proximal-sampling analysis paradigm, reflecting the close connection between proximal operators and localized reverse diffusion. Unlike standard proximal samplers, the proposed method does not require first-order (gradient) information of the target distribution. This is enabled by a Tweedie-style identity that reformulates the denoising score, thereby removing the need for explicit gradients. While related identities have been used for importance weighting in [huang2024a], the probability path considered here is distinct—combining a global proximal path with a local reverse diffusion path—and the paper analyzes a Sequential Monte Carlo (SMC) scheme, which was not studied in [huang2024a].

### Strengths
1. Although many components (proximal samplers, reverse diffusion Monte Carlo, and SMC) are drawn from prior work, the paper integrates them coherently to deliver a zero-order sampling scheme with provable convergence. The design is guided by a clear insight: Langevin-based methods inherently require gradients, whereas reverse-diffusion samplers can operate using only zero-order information.
2. The paper explicitly elucidates the close connection between proximal sampling and reverse diffusion, an important and conceptually insightful observation that may inform the design of gradient-free samplers.
3. The exposition is clear and well-structured: the intuition, assumptions, and main theorems are presented transparently, and the experiments are comprehensive, effectively demonstrating the proposed method’s efficacy.

### Weaknesses
1. The complexity analysis appears incomplete. The paper provides a one-step KL contraction bound with an additive term, but lacks an end-to-end error and cost analysis that composes these bounds over the full trajectory. A global, non-asymptotic complexity guarantee would substantially strengthen the contribution.
2. The theoretical results are largely asymptotic. In practice, performance with finite $M$ (particles) and $N$ (time steps) is crucial. The assumptions used to control the one-step error via $M$ and $N$ seem idealized; we recommend validating their plausibility empirically on synthetic data and reporting sensitivity to $M$ and $N$.
3. The comparison with RGO is unclear. What constitutes an ``iteration''? Does one pass that updates the entire sequence $\{x_i\}_{i=1}^N$ count as a single iteration or as $N$ iterations? Please clarify the accounting, and provide convergence comparisons against wall-clock time to enable fair, hardware-agnostic evaluation.

### Questions
Refer to the above parts.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The method provides an alternate implementation of the proximal sampler/RGO, based on an estimate for the scores using a particle ensemble of samples. This method yields a score estimator which appears as a mixture of Gaussians; in particular, the score estimator is implementable using zeroth order oracle queries. Theoretical analysis for this algorithm is provided, showing rates polynomial in the problem parameters. Experimental evidence then shows that this algorithm outperforms standard implementations of the RGO several benchmarks of interest.

### Strengths
The method provides a means for implementing the RGO in the proximal sampler using only \emph{zeroth} order queries, which is a more general computational model than the standard gradient oracle model. Furthermore, it does not need any convoluted tricks (MALA + underdamped) to implement the proximal sampler, compared to prior theoretical proposals.

The method appears to work extremely well in practice when compared to the standard implementation of the RGO. Generally, I suppose it is not too surprising that these particle methods can perform well in practice, although theoretical guarantees may be harder to establish.

The method seems to be easily parallelizable.

### Weaknesses
The theoretical guarantees are not particularly strong; I would be surprised if in the LSI setting this could improve upon the guarantees for the usual implementation of the RGO. Indeed, as the error scales as 1/N in the number of particles, so we should expect $N \asymp \varepsilon^{-2}$ or polynomial in the accuracy (compared to the standard implementation). Of course, there are other drawbacks to the theory.

### Questions
For this algorithm, when running the particle ensemble, do we take the entire ensemble of N particles as samples? This has the risk of inducing extra errors from correlation, but is unlikely to matter much in practice.

I am also surprised that we only see $1/N$ in the error, compared to the usual curse of dimensionality for non-parametric estimation. What is the intuition here?

Line 373 has some space formatting issues.


Notation for R\’enyi divergence switches between $\mathcal R$ and $R$.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studies proximal sampling for a distribution satisfying the log-Sobolev inequality without implementing the restricted Gaussian oracle (RGO) by rejection sampling. Instead, it uses the idea of diffusion-based Monte Carlo to simulate the backward SDE (i.e., eq (5)) for the backward step in proximal sampling, that is, the RGO. By tracking $N$ particles, the proposed method approximates $\pi^X$ by (9), and hence approximates the score function in (5) by (11). The key idea behind the approximation (9) is importance sampling. Thus, different from the exact implementation of RGO using first-order oracle of potential $f$ (for solving an optimization problem) and rejection sampling, Algorithm 1 approximately implements RGO by simulating (5) over $T$ internal steps within the stepsize $h$ of the (outer) proximal sampling.

The paper presents a complete analysis of Algorithm 1 with both discretization error and score estimation/Monte Carlo simulation error. It also provides two numerical experiments: 1) Gaussian-LASSO mixture; and 2) uniform sampling over non-convex sets. Numerical results in Section 5.1 demonstrate the advantages of the proposed method: it allows the use of a larger stepsize $h$ compared with proximal sampling based on rejection sampling, leading to faster convergence.

### Strengths
Unlike the proximal sampling based on rejection sampling, this paper approximately implements RGO via simulation backward SDE and score function using a particle system (i.e., Monte Carlo simulation). The advantage (as reflected in the experiments) is that the stepsize $h$ of the proximal sampling could be taken large (as long as $T={\cal O}(h)$). In particular, $h$ does not depend on dimension $d$ and properties of $f$ such as smoothness $L$. Moreover, the method does not require the first-order oracle of $f$, such as a subgradient; instead, it only assumes the zeroth-order oracle.

### Weaknesses
A major shortcoming is that the technical challenge addressed by the work is not very clear. Simulating SDEs with particle systems is a well-studied idea that has appeared frequently in the literature, so the methodological novelty seems limited, in view of the comparison with related works in Section 6. Moreover, the technical depth of the analysis is uncertain, as many of the proof techniques appear to be adapted from existing works, such as Vempala and Wibisono (2019).

Another notable shortcoming lies in the experiment in Section 5.2. In-and-Out finds T1 but fails to reach T2. It is claimed that the proposed method successfully explores both T1 and T2; however, based on Figure 4, it is difficult to conclude that the proposed method clearly identifies either region. The yellow points appear scattered all over, so there is no specific pattern that can be observed. The uniform sampling results are not convincing.

Finally, the comparison only with RGO in Section 5.1 might be limited.

### Questions
1. The key idea behind the approximation (9) is importance sampling. It would be nice if this insight could be made explicit in the paper.

2. Typo: line 710, a "$y$" is missing in the Gaussian.

3. Line 130, Fisher information should be relative Fisher information.

4. Some related papers might be missing:
Mixing Time of the Proximal Sampler in Relative Fisher Information via Strong Data Processing Inequality, Andre Wibisono
Proximal Oracles for Optimization and Sampling, Jiaming Liang and Yongxin Chen
Oracle-based Uniform Sampling from Convex Bodies, Thanh Dang and Jiaming Liang

5. Similar to (Kook, Vempala, and Zhang, 2024), the last paper above by Dang and Liang studies proximal sampling for uniform sampling on convex bodies. It presents two implementations of RGO using projection and separation oracles. It would be interesting to compare with this paper as well in Section 5.2.

### Soundness
3

### Presentation
3

### Contribution
2
