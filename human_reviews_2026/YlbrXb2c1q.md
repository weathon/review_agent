# Trajectory Optimal  Anisotropic Diffusion Models

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
We study anisotropic diffusion for generative modeling by replacing the scalar noise schedule with a matrix‑valued path $M_t$ that allocates noise (and denoising effort) across subspaces. We introduce a trajectory‑level objective that jointly trains the score network and \emph{learns} $M_t(\theta)$; in the isotropic case, it recovers standard score matching, making schedule learning equivalent to choosing the weight over noise levels. We further derive an efficient estimator for $\partial_\theta \nabla \log p_t$ that enables efficient optimization of $M_t$. For inference, we develop an anisotropic reverse‑ODE sampler based on a second‑order Heun update with a closed‑form step, and we learn a scalar time-transform $r(t;\gamma)$ that targets discretization error. Across CIFAR‑10, AFHQv2, and FFHQ, our method matches EDM overall and substantially improve few-step generation. Together, these pieces yield a practical, trajectory‑optimal recipe for anisotropic diffusion.
Code is available at \footnote{anonymous.4open.science/r/anisotropic-diffusion-paper-8738}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies diffusion models where the added noise is anisotropic and parameterized by a learned, time-varying covariance matrix using the DCT basis. They show that is equivalent to learning a noise schedule in the isotropic setting, and proposes a method for parameterizing and learning these time-varying weights. They also performed extensive experimental validation and showed that their method out-performed baselines in low NFE regimes.

Learning the optimal noise to be used in diffusion models is an important question and this paper takes promising steps towards answering it. However, there should be more justification on why some of the design choices (e.g. loss function, DCT basis) are taken, and clearer presentation of the experimental results.

### Strengths
This paper is clear, well-motivated and well-written. The method presented here for learning a noise schedule in the isotropic setting could be of independent interest. There are comprehensive experiments over multiple datasets and multiple ablations of the proposed method.

### Weaknesses
1. In this paper the basis (DCT basis) in which the noise schedule is learned is fixed, which imposes a strong assumption on the anisotropy of the noise. It would be better if this basis can be learned instead.
2. The numerical experiment results are tabulated in Table 1 for low NFEs but are plotted in Figure 2 for high NFEs. This is a strange choice for presenting the same data using two different formats, and seems to obfuscate the fact that EDM performs better than the proposed methods on CIFAR-10 and AFHQ for higher NFEs.
3. The results in Section 3.1 that isotropic TLSM is equivalent to learning an optimal weight function seems to be similar to that in [1].
4. An implicit assumption when formulating the trajectory-level score matching loss (Eq. 11) is that minimizing the loss will also also induce learning an optimal noise schedule in some sense. However, the authors only showed that $\text{net}(x, t; \phi) = \nabla \log p_t(x, \theta)$ when the score is minimized; it is unclear what $p_t(x, \theta)$ should be at optimality.

[1] Kingma and Gao 2023, Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.

### Questions
1. It would be interesting to see how this method performs on other bases used in image processing in addition to the DCT basis, for example the wavelet basis.
2. For the numerical FID results, there should also be a comparison of the best FID achieved by each method across all NFEs, as it seems like this method is only better than EDM at low NFEs.
3. As this methods sees the largest improvements on FFHQ over the baselines, do you think it is a result of this dataset being more structured (e.g. faces centered at eyes) than the others? If this is the case, perhaps anisotropic noise can be seen as a way to adapt to more structured datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to learn an optimal forward diffusion process noise schedule as well as the discretization schedule (for generation) for diffusion-based ODE generation. Specifically, the forward process noise schedule is parametrized as a matrix-valued path which admits anisotropic diffusion, and is optimized jointly with the score network to minimize the score matching loss. The discretization schedule is optimized to minimize local truncation error and the score matching loss. The authors demonstrate the proposed optimal noise schedule and discretization schedule improve sample generation quality, i.e., FID scores, on CIFAR10, AFHQ-64, and FFHQ-64 in both small NFE (number of function evaluations) and large NFE regimes.

### Strengths
- **[S1] The paper is original in the aspect that it develops a technique for directly optimizing matrix-valued noise schedule w.r.t. the score matching objective (Lemma 3).** To the best of my knowledge, prior work either optimizes a scalar-valued noise schedule [1] or diagonal matrix-valued noise schedule [2] to enhance diffusion model generation. In contrast, this work provides a technique for optimizing arbitrary matrix-valued noise schedules by deriving an expression for the gradient of the noise schedule w.r.t. score matching loss, thereby enabling direct gradient-based optimization.

[1] Variational Diffusion Models, NeurIPS, 2021.

[2] Diffusion Models With Learned Adaptive Noise, NeurIPS, 2024.

### Weaknesses
- **[W1] Learnable discretization schedule lacks novelty.** By optimizing the trajectory-level discretization loss $\hat{H}(\gamma)$, the authors propose to learn a discretization such that the difference between one-step Euler velocity and two-step Heun velocity is minimized.  However, there are multiple existing works which explore similar ideas (with better performance). For instance, [1] optimizes discretization to minimize distributional error between accurate and discretized trajectories; [2] optimizes discretization to minimize distance between low-NFE and high-NFE trajectories; [3] optimizes discretization to minimize distance between true and generated ODE trajectory endpoints.

- **[W2] There is insufficient evidence that the proposed techniques consistently improve generation quality.** Specifically, the addition of proposed techniques do not yield consistent improvements over datasets. For instance, learning scalar-valued noise schedule ($g^{iso}$ in Table 1) degrades FID on CIFAR10 and AFHQ, and learning matrix-valued noise schedule ($g^{ani}_1$,$g^{ani}_2$ in Table 1) degrades FID on AFHQ. Only learned discretization seems to provide consistent gains in FID.

[1] Align Your Steps: Optimizing Sampling Schedules in Diffusion Models, 2024

[2] Fast ODE-based Sampling for Diffusion Models in Around 5 Steps, CVPR, 2024

[3] Accelerating Diffusion Sampling with Optimized Time Steps, CVPR, 2024


**Minor Comments**

- Line 243 : Optimization Score Matching --> Optimization of Score Matching
- Line 314 : $\tilde{\theta}$ is undefined
- Line 341 : omid --> omit
- Line 345 : $\hat{x},\hat{t}$ --> $(\hat{x},\hat{t})$

### Questions
- **[Q1] How does the learned noise schedule compare to popular hand-designed schedules such as Cosine or Linear? [1]**

- **[Q2] In Table 1, I observe trends where FID increases and then decreases with increasing number of NFEs. Can the authors explain why?**

- **[Q3] Can the authors provide results combining EDM with learned discretization?**

- **[Q4] Since the proposed method involves finetuning EDM networks, how does the overall computation cost compare to other finetuning type of methods such as ReFlow [2] or Consistency Distillation [3]?**

- **[Q5] How does the proposed method compare against other fast samplers such as DPM-Solver [4], DEIS [5], AMED [6] etc.?**

- **[Q6] How does the proposed method scale to larger, e.g., text-to-image diffusion models?**

[1] Improved Denoising Diffusion Probabilistic Models

[2] Simple ReFlow: Improved Techniques for Fast Flow Models

[3] Consistency Models Made Easy

[4] DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps

[5] Fast Sampling of Diffusion Models with Exponential Integrator

[6] Fast ODE-based Sampling for Diffusion Models in Around 5 Steps

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates an alternative to the standard isotropic noise modeling in diffusion models by proposing a learnable, matrix-valued noise variants, $M_t(\theta)$, to allocate noise non-uniformly across data subspaces. The "Trajectory-Level Score Matching" (TLSM) objective is introduced to jointly learn $M_t(\theta)$ and a score network. They also propose a gradient estimator for $\theta$ and a two-stage algorithm that first learns $M_t(\theta)$ and then a separate time-reparameterization $r(t, \gamma)$ intended to reduce discretization error. Empirically, the method reports promising results in the low-NFE (few-step) sampling regime on several benchmarks, outperforming the EDM baseline on FFHQ.

### Strengths
+ The paper propose a proposal to replace the scalar schedule with a learnable, matrix-valued path is a novel direction.

+ The paper presents promising empirical results in the low-NFE regime. For example, on FFHQ at NFE=9, the reported FID is 6.05, compared to 57.28 for the EDM baseline.  

+ The paper describes a self-contained method, including the TLSM loss, a gradient derivation (Lemma 3), and a two-stage algorithm.

### Weaknesses
+ The core theoretical contribution is somewhat weak to me. As Lemma 2 reveals, the method in the isotropic case (J=1) is merely a reparameterization of existing weighted score-matching techniques, not a new framework. 

+ The entire method hinges on the complex gradient estimator (Eq. 14), yet the paper provides no analysis of its properties. The bias (from approximating the true score with "net") and the variance (scaling with data dimension $d$) are both unquantified, making the estimator a 'black box'. 

+ The decision to decouple the optimization of $M_t(\theta)$ (continuous-time loss) from $r(t, \gamma)$ (discrete-time error) is a heuristic, greedy approach. The $\theta^*$ found in Stage 1 is not guaranteed to be optimal for the true end-goal (the K-step discrete sampler), undermining the "trajectory-optimal" claim.


+ The paper reports FID vs NFE but overlooks important efficiency metrics. Each training step involves three backward passes (Lemma 3), and inference requires a second-order Heun integrator with two network evaluations per step.

### Questions
1.  How does the variance of this estimator scale with the data dimension $d$, and how was this managed in practice?

2. Decoupling $M_t(\theta)$ and $r(t, \gamma)$ is a greedy approach. The $ \theta^* $ from Stage 1 may not be optimal for the K-step sampler, questioning the "trajectory-optimal" claim. Could a joint optimization with $ \gamma$ yield a better $ \theta$?

3. The method involves $M_t^{-1}(\theta)$ (Eq. 3, Eq. 15) and $M_t^{1/2}(\theta)$. How is numerical stability handled when $M_t(\theta)$ is near-singular, especially for $t \approx 0$?

### Soundness
2

### Presentation
3

### Contribution
2
