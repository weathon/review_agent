# A Statistical Benchmark for Diffusion-Posterior-Sampling Algorithms

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
We propose a statistical benchmark for diffusion-posterior-sampling (DPS) algorithms in linear inverse problems.
Our test signals are discretized Lévy processes whose posteriors admit efficient Gibbs methods.
These Gibbs methods provide gold-standard posterior samples for direct, distribution-level comparisons with DPS algorithms.
They can also sample the denoising posteriors in the reverse diffusion, which enables the arbitrary-precision Monte Carlo estimation of various objects that may be needed in the DPS algorithms, such as the expectation or the covariance of the denoising posteriors.
In turn, this can be used to isolate algorithmic errors from the errors due to learned components.
We instantiate the benchmark with the minimum-mean-squared-error optimality gap and posterior-coverage tests and evaluate popular algorithms on the inverse problems of denoising, deconvolution, imputation, and reconstruction from partial Fourier measurements.
We release the benchmark code at https://github.com/zacmar/dps-benchmark and invite the community to contribute and report results.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a new benchmark for evaluating different posterior sampling algorithms using diffusion models (dubbed DPS in this paper; Different from DPS [1]), where the posterior samples can be computed analytically, so that the ground truth is given. 
Previous *benchmarks* that admit analytical posterior samples were constrained to settings where the prior is a mixture of Gaussians, which largely differs from the natural data statistics. The prior distributions considered in this paper is much larger, and the authors propose methods to efficiently compute ground truth posterior distributions. Several widely established baselines are compared.

**References**

[1] Chung et al., "Diffusion posterior sampling for general noisy inverse problems", ICLR 2023

### Strengths
1. To the best of my knowledge, this is the first approach to go beyond mixture of gaussian priors when attempting to build a ground truth posterior distribution.

2. The paper is well-written and easy to follow, with sufficient background given in the appendix.

3. The method of acquiring the posterior distribution by extending Kuric et al. [1] is sound.


**References**

[1] Kuric et al., "The Gaussian latent machine: Efficient prior and posterior sampling for inverse problems", arxiv 2025

### Weaknesses
1. Being able to use different prior/posterior distributions as ground truth is, in and of itself, important. Nevertheless, the argument would be strengthened if the paper shows that the proposed distributions in this paper are closer to real-world statistics in some cases. Currently, only some references are given.

2. The authors mention that the proposed framework can be extended to higher-dimensional settings, but there are complications. It would add much value if the authors were to include experiments with $d$ that match the typical image resolutions. Currently, it seems like the experiments are conducted with low dimensionality ($d$). What's the value of $d$ chosen here?

### Questions
Is there any reason to constrain the benchmark for *diffusion* posterior sampling algorithms?

### Soundness
3

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
3

### Summary
This paper introduces a statistical benchmark for evaluating diffusion posterior sampling algorithms using discretized Lévy processes with tractable Gibbs posteriors as ground truth. While the framework enables rigorous distribution-level comparisons, the evaluation is severely limited to low-dimensional (d=64) linear inverse problems, raising serious concerns about scalability and practical relevance to realistic imaging applications.

### Strengths
Developing a benchmark for posterior sampling in high-dimensional problems is important.

### Weaknesses
• All experiments use d=64 signals with only linear operators. No evidence is provided that the framework scales to realistic dimensions (e.g., 256×256 images) or nonlinear problems, fundamentally limiting the practical applicability and making it unclear whether findings transfer to problems researchers actually solve.

• The authors cite power-law phenomena in finance and images to motivate heavy-tailed priors, but never demonstrate that their 1D discretized Lévy processes meaningfully capture structure in realistic signals. The connection to actual image statistics remains unsubstantiated.

• Table 4 shows learned denoisers often match or exceed oracle performance, undermining claims about isolating likelihood approximation errors. The paper doesn't establish whether likelihood errors dominate versus other sources (discretization, hyperparameter sensitivity), weakening the diagnostic utility argument.

• DPS algorithms are tuned with learned denoisers but evaluated with oracle denoisers using the same hyperparameters (lines 276-278). This mismatch means oracle results may be suboptimal, contradicting claims about properly isolating algorithmic errors.

• Claims of "efficient implementations" and "acceptable runtimes" (lines 231-234, 822-823) lack any quantitative evidence; no runtime comparisons, memory usage, or scalability analysis is provided to substantiate efficiency claims or assess practical feasibility at higher dimensions.

### Questions
Does this benchmark can be used for amortized diffusion sampling methods, i.e., learning the full posterior?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
Diffusion posterior sampling algorithms have become prominent methods for sampling from posterior diffusion with a denoising diffusion model prior. While many methods have been proposed in the recent years, most of the interesting benchmarks do not come with ground-truth posterior samples to which one can compare against. The aim of this paper is to close this gap by proposing a statistical benchmark that mimicks the behaviour of realistic data (power-law-like extremes as stated in the paper). To this end the authors consider the posterior associated to Lévy processes and use an efficient Gibbs sampler to obtain gold-standard posterior samples that serve as reference.

### Strengths
- This paper tackles a fundamental problem in the evaluation of diffusion posterior samplers and proposes a very useful benchmark which in my opinion could be useful to the community and should be present in all the forthcoming papers. 
- The model is general enough to contain different instantiations such as Laplace and spike and slab and thus goes beyond the existing gaussian mixture toy examples. 
- The paper is rather well-written and quite pedagogical, I enjoyed reading it.

### Weaknesses
The only weakness I see is the structuring of the main paper. For example I think that some parts of the related works (such as the first two paragraphs) could be moved to the appendix as they are slightly relevant to the content of the paper. This space could be used to provide for example more background on the GLM framework, as one needs to go to the appendix to read more interesting details about it. 
I also think that Figure 1 and 2 are misplaced as at this stage of the paper the Lévy process is not introduced and we don't know yet what St(1) means.

### Questions
I have a few suggestions and related works to be considered: 
- I think it would have been interesting to include samples from a conditional diffusion model, by either training the conditional denoiser or estimating the denoiser using Monte Carlo samples as is done for DPS methods. I believe that it could be relevant since it provides a lower bound on the performance that one hopes to achieve with DPS methods. 
- [1] considers an actual real world setting where gold standard samples can be obtained using MCMC. 
- The toy Gaussian mixture benchmark is introduced in [2, 3] 

[1] Cardoso, G.V. and Pereira, M., 2025. Predictive posterior sampling from non-stationnary Gaussian process priors via Diffusion models with application to climate data.  
[2] Cardoso, G., Idrissi, Y.J.E., Corff, S.L. and Moulines, E., 2023. Monte Carlo guided diffusion for Bayesian linear inverse problems.  
[3] Boys, B., Girolami, M., Pidstrigach, J., Reich, S., Mosca, A. and Akyildiz, O.D., 2023. Tweedie moment projected diffusions for inverse problems.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
- The authors introduce a benchmark suite for evaluating algorithms designed to solve linear inverse problems with diffusion model priors
- The benchmark is built on a synthetic setup derived from discretized Lévy processes
- It hence include setting of heavy-tailed/power-law–like distributions beyond the Gaussian case
- The key motivation lies in the fact that Lévy processes possess explicit marginal distributions and can be targeted using Gibbs sampling
- This property allows the benchmark to generate ground-truth posterior samples (from inverse problem and denoising posterior) for quantitative comparison across algorithms

### Strengths
- The paper is well-written and accompanied with concise explanations in the appendix
- The motivation of the paper is well articulated namely for principled benchmarking in diffusion-based inverse problem solvers
- The proposed benchmark is a valuable contribution, as it extends evaluation on Gaussian setup to a broader family of distributions

### Weaknesses
**Overstated or misleading claims**
The repeated use of the term "oracle", e.g., Lines 56, 129, 277, 355 is misleading.
The samples used in the benchmark are produced via Gibbs sampling—an iterative procedure—hence they are approximate, not exact. The quality of these samples depends on choices such as burn-in time, which are hyperparameters of the framework.
This issue becomes more apparent when the benchmark is applied to algorithms requiring gradients of the denoiser (Line 257-263 and equation (60)): the paper substitutes the latter with a covariance estimator of $X_0 | X_t,$ and hence further deviating from the notion of an "oracle".


**Template for posterior samplers**
The proposed benchmark template seems overly restrictive. By focusing on algorithms that use only the denoiser, it neglects methods that require the Jacobian of the denoiser.
Although the paper connects this to the covariance $Cov(X_0 \mid X_t)$, estimating this covariance is far more computationally demanding and less stable, and therefore it downgrade the claim that the benchmark offers "oracle" quantities with minimal approximation error.

**Evaluation design**
- The inclusion of learned denoisers in the evaluation is conceptually inconsistent with the paper’s stated goal of removing approximation errors (Section 1.1).
If the benchmark aims to isolate algorithmic performance, learned denoisers reintroduce training-dependent variability. While the authors justify this by citing robustness testing, the notion of robustness is loose and in practice requires hyperparameter tuning, which introduces additional confounding factors.
- The experimental comparison is limited. Only 3 algorithms are evaluated, and these do not represent the diversity of available approaches, e.g., optimization-based, variational, or midpoint-guided methods; see the literature in [1] and [2]

**Remarks and minor issues**

- In background, rephrase the statement in Line 132 about DDPM, sampling in fact depend on several parameters and it is bold to say " researchers typically use"; I would argue that frequently DDIM sampling is used with $\eta = 0$ (simulating the probability-flow ODE) for sharp samples with few diffusion steps
- The used abbreviation **DPS** is already/actually the name of a well-known algorithm in diffusion models and inverse problems [4], hence the abbreviation might be misleading using it here to refer to something else may cause confusion.
- The authors may also consider adding the following reference on inverse problems benchmarks [3]
- Line 288: The statement that DiffPIR is an extension of C-DPS is incorrect. DiffPIR follows a distinct formulation based on quadratic half-splitting with an auxiliary variable and does not rely on the VJP of the denoiser.


---

.. [1] Daras, Giannis, et al. "A survey on diffusion models for inverse problems." arXiv preprint arXiv:2410.00083 (2024).

.. [2] Oliviero-Durmus, Alain, et al. "Generative modelling meets Bayesian inference: a new paradigm for inverse problems." Philosophical Transactions A 383.2299 (2025): 20240334.

.. [3] Zheng, Hongkai, et al. "Inversebench: Benchmarking plug-and-play diffusion priors for inverse problems in physical sciences." arXiv preprint arXiv:2503.11043 (2025).

.. [4] Chung, Hyungjin, et al. "Diffusion posterior sampling for general noisy inverse problems." arXiv preprint arXiv:2209.14687 (2022).

### Questions
- I generally found the figures hard to understand and interpret, I'm referring namely to figure 1, it says that it shows reverse using the oracle denoiser, but it is not clear, similarly, for figure 3, it is hard to interpret namely to say wether the algorithms performs well or not 
- can the authors provides hints/explanation on the derivation of equation (13)
- the authors claim that introduced framework can also assess the approach where the conditional components is learned (Line: 168-170), but it is not clear how it can be achieved given that in some tasks the likelihood is not known, e.g. tasks such deraining or dehazing, see for instance [1]; yet the benchmark is built on the ability to explicitly write the posteriors/marginals and target them using Gibbs sampling

- A more broad question: did the authors think about how the benchmark can be extend to nonlinear inverse problems ?

---

.. [1] Wang, Hanting, et al. "IRBridge: Solving Image Restoration Bridge with Pre-trained Generative Diffusion Models." arXiv preprint arXiv:2505.24406 (2025).

### Soundness
2

### Presentation
3

### Contribution
3
