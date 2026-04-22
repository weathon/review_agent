# Accelerate Autoregressive Normalizing Flows Sampling with GS-Jacobi Iteration

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
AutoRegressive Normalizing Flows (abbreviated as AR Flow) enjoy extensive applications in tasks such as density estimation and image generation. However, due to the causal affine coupling blocks requiring sequential computation, the sampling process is extremely slow. 
In this paper, we demonstrate that through a series of optimization strategies, such AR Flows sampling can be greatly accelerated by using the Gauss-Seidel-Jacobi (abbreviated as GS-Jacobi) iteration method.
Specifically, we find that blocks in AR Flows have varying importance: a small number of blocks play a major role in image generation, while other blocks contribute relatively little; some blocks are sensitive to initial values and prone to numerical overflow, while others are relatively robust. Based on these two characteristics, we propose the Convergence Ranking Metric (CRM) and the Initial Guessing Metric (IGM):
CRM is used to identify whether a Flow block is "simple" (converges in few iterations) or "tough" (requires more iterations); IGM is used to evaluate whether the initial value of the iteration is good. 
The TarFlow was chosen as the main experimental subject in our study owing to its SOTA performance on several benchmarks.
Experiments on four TarFlow models demonstrate that GS-Jacobi sampling can significantly enhance sampling efficiency while maintaining the quality of generated images (measured by FID), achieving speed-ups of 4.53× in Img128cond, 5.32× in AFHQ, 2.96× in Img64uncond, and 2.51× in Img64cond without degrading FID scores or sample quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a method to accelerate sampling in autoregressive normalizing flows by reformulating the inverse transformation as a nonlinear fixed-point problem solved through a hybrid Gauss Seidel and Jacobi iteration. Two diagnostic metrics are introduced to analyze convergence behavior and guide adaptive computation. The method enables efficient parallel updates within flow blocks while preserving numerical stability. Experiments show faster sampling without loss in visual.

### Strengths
The paper addresses a clear and well-motivated problem: the slow sampling speed of autoregressive normalizing flows, which has long hindered their practical use. To my knowledge, the introduction of diagnostic metrics for convergence and initialization offer novel insights in sampling from normalizing flow models.

### Weaknesses
The major concern for me is that the method is closely tied to specific autoregressive normalizing flow architectures, mainly TarFlow. Also, the paper does not compare against alternative approaches that achieve faster sampling through model distillation [1], learned/high-order samplers [2-3], leaving unclear how the proposed iteration method performs relative to these stronger baselines. 

Additionally, the convergence ranking metric and the initial guessing metric seem to rely on heuristic choices. While the authors presented empirical study on their robustness, both metrics appear sensitive to architecture, dataset, and initialization choices (which is also related to my first point), which raises questions about stability under different models or datasets. 

Lastly, Proposition 1 does not discuss about the convergence of approximation error. How large should T be? Analysis on the convergence rate would strengthen the paper. Current manuscript does not indicate how accurate the approximated proposed method is. 


[1] Progressive Distillation for Fast Sampling of Diffusion Models

[2] DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps

[3] Learning to Discretize Denoising Diffusion ODEs

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

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
The paper proposes a sampling method for autoregressive normalizing flows. Recently, TarFlow shows that normalizing flows with autoregressive flow layers can perform comparably to other deep generative models. However, autoregressive flows are slow in inference because they must compute $x_{i}$ iteratively. The paper proposes treating the inverse process as a nonlinear system, enabling it to be solved using the fixed-point iteration method.

### Strengths
1.  The idea is simple, but it can effectively improve TarFlow’s sampling speed.

### Weaknesses
1. The significance and impact of the proposed method appear limited because this method is tailored for autoregressive normalizing flows, which represent only a small subset of deep generative models.

2. The proposed method seems to bring in new problems. That is, when we use the fixed-point iteration method, we need to recompute $\sigma$ and $\mu$ at each iteration. That means we will need to run the VIT T times for each layer. When T is greater than the number of patches, the method will be slower than the baseline.
3. Do we have an analysis of the relationship between T and the image size? How can we determine T for the proposed method?

### Questions
Please refer to the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the critical sampling bottleneck in autoregressive normalizing flows (AR flows) by introducing a parallelizable Gauss-Seidel/Jacobi iteration strategy. The authors observe that in AR flow models (e.g. TarFlow), sampling is very slow because each affine coupling block operates as a causal RNN that must be executed sequentially. The paper’s key contribution is to reformulate the AR flow sampling as solving a diagonal nonlinear system and apply a hybrid Gauss-Seidel-Jacobi iteration to solve it in parallel, dramatically accelerating generation without loss of quality. Notably, they introduce two novel metrics – Convergence Ranking Metric (CRM) and Initial Guessing Metric (IGM) – to adapt the iteration procedure to the model’s characteristics, ensuring stability and efficiency. Empirical results on state-of-the-art TarFlow models show 4.5×–5.3× speed-ups with essentially no degradation in FID (image quality), which is a significant practical improvement.

In terms of novelty and significance, the idea of using fixed-point iterations to accelerate AR flows builds on some prior work (e.g. Newton-based solvers for autoregressive inversion). However, this paper goes further by hybridizing Jacobi and Gauss-Seidel updates and introducing adaptive metrics to handle non-uniform convergence across model components, which is a fresh and non-trivial innovation.

The writing is well-structured, with a logical flow from identifying the problem to proposing the method and validating it. However, there are numerous typos, so I cannot give a high rating for the presentation.

### Strengths
The method yields dramatic improvements in sampling speed for autoregressive normalizing flows. Across multiple models, it achieves 4×–5× speedups without degrading image fidelity, as evidenced by nearly unchanged FID scores (within <1% of the baseline).

Innovative Use of Iterative Solvers in AR Flows: The paper introduces a novel hybrid of Jacobi and Gauss-Seidel iteration to parallelize what was a sequential process. This is a creative cross-disciplinary idea, applying classic numerical methods to deep generative modeling. The approach is well-grounded in theory. The authors show that their fixed-point iteration will converge to the correct solution (under the model’s triangular Jacobian structure) and provide an error propagation formula.

A major strength is the introduction of the CRM and IGM metrics to guide the sampling procedure. These metrics directly tackle the two main challenges identified: (1) different transformer blocks have non-uniform convergence behavior, and (2) naive initialization can cause instability. CRM provides a principled way to determine which coupling blocks are “tough” (slower to converge) so the algorithm can allocate more iterations or use sequential updates for those, while treating others with fast Jacobi updates. IGM allows the sampler to intelligently choose a safe starting point for the iteration, preventing the divergence (“numeric overflow”) that would otherwise occur in sensitive early blocks. The use of these metrics is empirically justified. By addressing these issues, the proposed method is robust. It converges where a naive parallel iteration would fail, and it does so efficiently by not over-investing computation in blocks that don’t need it.

### Weaknesses
The proposed solution, while effective, adds considerable complexity to the sampling process. Implementing the GS-Jacobi sampler requires computing the CRM and IGM metrics using the model’s weights and a batch of training data. This offline analysis step is unusual for generative model sampling and might need to be repeated if the model or data distribution changes. Moreover, the sampling algorithm introduces new hyperparameters (e.g. how to segment a tough block, how many Jacobi vs. Gauss-Seidel iterations to use) that are not trivial to choose a priori. The tuning was done on a case-by-case basis for each dataset/model. Such manual optimization might be necessary for new models, which is a potential drawback in terms of ease of use. The method works impressively once tuned, but the paper does not provide a simple recipe for selecting these hyperparameters automatically.

The parallel iteration helps only to the extent that many parts of the model can converge quickly in a few Jacobi steps while isolating a few slow parts. Thus, the speed-up is not guaranteed for every AR flow architecture.

This paper contains numerous typos and grammatical issues. Here are the ones I found just by skimming through it:
* L36: solution -> high-resolution
* L84: images generation -> image generation
* L84: attention mechanic -> attention mechanism
* L87: trys -? tries
* L134: denotes -> denote
* L140: can be calculate -> can be calculated
* L144: an non-linear -> a non-linear
* L154: Converge and Error Propagation -> Convergence and Error Propagation
* L196: take all $X(0) = Z$ cause -> taking all $X(0)=Z$ causes
* L215: centers in 0 -> centers at 0
* L262: matrixs -> matrices
* L290: suffer -> suffers
* L302: GUASS-SEIDEL-JACOBI -> GAUSS-SEIDEL-JACOBI
* L304: unit a time -> unit at a time
* L307: an non-decrease -> be a non-decreasing
* L308: defination -> definition
* L315: cumsum -> cumulative sum
* L361: not -> no
* L365: maximum value occur -> maximum value that occurs
* L368: attention layers parameters -> attention layers' parameters
* L369: simple -> simply
* L374: simpe -> simple
* L376: relative -> relatively
* L421: learing -> learning

### Questions
Could you elaborate on why Gauss–Seidel is superior to other alternatives for accelerating convergence in autoregressive flows? What motivates using this GS–Jacobi scheduling over a more straightforward sequential sampling or existing parallelization techniques, and why is it expected to succeed where naive parallelization fails?

Did you try Anderson acceleration, SOR, or block-Newton? How do they compare in stability and speed?

How generalizable is TarFlow to domains beyond images, e.g., audio or language, where the autoregressive structure and dependencies differ significantly?

While the paper reports up to ~5× speed-ups on moderate-size image models (e.g. 128×128 resolution in Img128cond) without quality loss, how does the method scale with increasing sequence length or model size? Is the parallel iterative scheme still efficient for substantially larger images or longer sequences, and what are the memory or computation trade-offs as these grow? It would be useful to know if any limitations or diminishing returns appear when scaling up to more complex datasets or very high-dimensional generation tasks.

### Soundness
4

### Presentation
2

### Contribution
3
