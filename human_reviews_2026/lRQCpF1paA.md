# Mode-seeking for inverse problems with diffusion models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
A pre-trained unconditional diffusion model, combined with posterior sampling or maximum a posteriori (MAP) estimation techniques, can solve arbitrary inverse problems without task-specific training or fine-tuning. However, existing posterior sampling and MAP estimation methods often rely on modeling approximations and can be computationally demanding. In this work, we propose the variational mode-seeking loss (VML), which, when minimized during each reverse diffusion step, guides the generated sample towards the MAP estimate. VML arises from a novel perspective of minimizing the Kullback-Leibler (KL) divergence between the diffusion posterior $p(\mathbf{x_0}|\mathbf{x_t})$ and the measurement posterior $p(\mathbf{x_0}|\mathbf{y})$, where $\mathbf{y}$ denotes the measurement. Importantly, for linear inverse problems, VML can be analytically derived and need not be approximated. Based on further theoretical insights, we propose VML-MAP, an empirically effective algorithm for solving inverse problems, and validate its efficacy over existing methods in both performance and computational time, through extensive experiments on diverse image-restoration tasks across multiple datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a variational mode-seeking loss (VML) for solving inverse problems with diffusion models and derives a closed-form expression for linear operators. While the formulation is conceptually interesting and mathematically sound, the experimental validation is limited and not convincing. The paper only evaluates nearly noise-free settings and reports perceptual metrics (FID, LPIPS) without standard reconstruction measures such as PSNR or SSIM. As a result, it is difficult to assess the practical effectiveness and robustness of the proposed method. If the authors could provide stronger experimental evidence, particularly under noisy conditions and with standard reconstruction metrics, the contribution would become significantly more convincing and could merit a higher evaluation.

### Strengths
1. The derivation for linear inverse problems is clear and mathematically consistent, showing the authors’ understanding of the theoretical aspects.

2. The proposed algorithms are conceptually straightforward and easy to implement, requiring only a pre-trained unconditional diffusion model.

3. The writing and organization of the paper are generally clear, with helpful figures and tables that make it easier to follow the main idea.

### Weaknesses
1. **Limited experimental settings**:
The experiments are conducted only under an almost noise-free condition (σₓ = 1e-9). However, in diffusion-based inverse problem literature, it is standard to evaluate both noisy (e.g., σₓ = 0.05) and noise-free scenarios to assess robustness. The lack of results in noisy conditions makes it difficult to judge the algorithm’s stability and general applicability.

2. **Missing key evaluation metrics**:
While the paper reports FID and LPIPS, it omits SSIM and PSNR, which are standard quantitative metrics for measuring reconstruction fidelity in inverse problems. The absence of these metrics substantially weakens the validation of the proposed method’s effectiveness.

3. **Formatting issue**:
The submission contains blue-colored text, suggesting it was uploaded as an unclean diff version. Although minor, this formatting issue slightly affects readability and presentation quality.

### Questions
1. The paper only evaluates nearly noise-free settings (σᵧ = 1e−9). Could the authors provide results for noisy scenarios (e.g., σᵧ = 0.01 or 0.05) to test robustness?

2. Why are PSNR and SSIM not reported? These are standard reconstruction metrics for inverse problems.

3. The submission includes blue-colored text, suggesting an unclean version. Please ensure a clean and properly formatted final submission.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel inference-time guidance strategy called Variational Mode-Seeking Loss for solving inverse problems using pre-trained unconditional diffusion models. The core idea is to minimize the reverse KL divergence at each reverse diffusion step, which encourages the intermediate sample $\mathbf{x}_t$ to converge toward the Maximum a Posteriori estimate as $t \to 0$. The authors derive a closed-form expression for VML in the case of linear inverse problems and propose a simplified version $VML_S$ by omitting higher-order covariance terms. They also introduce a preconditioner to handle ill-conditioned linear operators. Extensive experiments on image restoration tasks (inpainting, super-resolution, deblurring) across multiple datasets demonstrate that and its preconditioned variant outperform existing methods in terms of LPIPS and FID metrics, often with lower computational cost.

### Strengths
- **Originality:** The formulation of VML as a mode-seeking loss is novel and well-motivated from a variational perspective.
- **Empirical Validation:** Extensive experiments on multiple tasks and datasets show consistent improvements in both perceptual quality (LPIPS) and sample fidelity (FID).
- **Generality:** The method is extended to latent diffusion models, showing promise beyond pixel-space models.

### Weaknesses
- **Complexity:** Although the simplified VML is used, the gradient computation still requires Jacobians of the denoiser, which can be computationally expensive, especially for high-resolution images or complex degradation operators.
- **Limited Non-Linear Extension:** The extension to non-linear inverse problems (via LDMs) is preliminary and suffers from optimization challenges due to the non-linearity of the decoder.

### Questions
1. In Equation (8), the term $(1 - \Sigma^+ \Sigma)$ uses "1" instead of the identity matrix $\mathbf{I}$. Is this a typo, or is it meant to be a scalar or broadcasted operation? Clarification is needed.
2. The optimization of $D_{\text{KL}}(p(\mathbf{x}_0|\mathbf{x}_t) \| p(\mathbf{x}_0|\mathbf{y}))$ is non-convex and may not guarantee convergence to the true MAP. How does the authors' gradient-based optimizer handle local minima, especially in early reverse steps?
3. The preconditioner requires SVD of $\mathbf{H}$, which may be infeasible for very large or non-linear operators. Are there scalable alternatives for such cases?

### Soundness
4

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
This paper proposes a variational mode seeking loss (VML) as a principled guidance mechanism for solving inverse problems using unconditional diffusion models. The VML objective, defined as the KL divergence between $p(x_0 | x_t)$ and the measurement posterior is minimized at each reverse diffusion step to steer samples toward the MAP estimate.

### Strengths
Main strengths of the paper:

- the paper is well written, easy to read from a technical perspective.
- mathematical details are provided and cleanly described
- a fair bit of experimental results are provided - which demonstrates the method across many scenarios

### Weaknesses
I believe the paper misses some relevant comparisons as their method is also somewhat a second-order method (at least the optimal solutions in special cases would be so). So comparisons to some relevant methods (see below) would improve the work.

I think the main weakness is that it is not clear why VML would be preferable to standard methods. Given that there are several approximations in the way the VML is used in practice, I think the claim that this is more principled is a bit weak. I'd encourage authors to both provide practical and mathematical justification of VML compared to existing works.

### Questions
- In the motivation part, I think it wasn't made clear what the *practical* motivation of this work is.

- Please define the argument of the loss when you define VML. What is the main "input variable" of VML?

- The authors say *we hypothesize that these higher-order terms may not be crucial in practice*, does this not reduce your approach to standard guidance approaches?

- The authors derive in Props 1 & 2, their VML. As noted by the authors, this takes the form that is similar to the usual guidance. Can the authors compare their approach mathematically to second-order guidance approaches such as Boys et al (2024) or other moment-matching based methods. Especially the approach based on preconditioners.

- Similarly, given the relationship, I'd also expect TMPD (Boys et al, 2024) or similar approaches to be included in the experimental comparisons.

- Is the algorithm more robust to noise compared to standard approaches? Perhaps an experiment with increasing noise levels would be helpful here. This is a bit like your Figure 7 but x axis would be increasing noise levels.

- Is there any scope or reason to use anything else than KL? 

Style comment: I think having blue text in the main text is not standard and appropriate. I wonder if these were meant to highlight this is a previous submission. In any case, I'd ask authors to get rid of text colouring (except for instructive cases like equation highlighting etc).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a new inference-time optimization method, called Variational Mode-Seeking Loss (VML), for solving inverse problems using pre-trained unconditional diffusion models. Unlike prior approaches that approximate the conditional score or solve complex ODEs for posterior sampling or MAP estimation, this work formulates a loss function derived from minimizing the Kullback–Leibler divergence between the diffusion posterior $p(x_0 \mid x_t)$ and the measurement posterior $p(x_0 \mid y)$. 

The authors show that, for linear inverse problems, this loss can be expressed in closed form, avoiding the need for approximations. Based on this, they propose VML-MAP, an algorithm that iteratively minimizes the loss during each reverse diffusion step, guiding samples toward the MAP estimate. They further propose a preconditioned variant to address optimization instability in ill-conditioned problems. Extensive experiments on several image restoration tasks—such as inpainting, super-resolution, and debluring—demonstrate improved perceptual quality and competitive computational efficiency compared to methods like DDRM, ΠGDM, and MAPGA.

### Strengths
1. The derivation of the variational mode-seeking loss is mathematically rigorous and well connected to Bayesian reasoning. The paper provides clear intuition for why minimizing the reverse KL divergence aligns the diffusion trajectory with the posterior mode.
2. By deriving an analytical expression for VML under linear degradations, the paper avoids the heuristic approximations common in posterior sampling methods. This provides both conceptual clarity and computational efficiency.
3. The evaluation spans multiple datasets and degradation operators. The results consistently show that VML-MAP and its preconditioned variant outperform or match strong baselines across tasks, confirming robustness.

### Weaknesses
The paper is well executed but the conceptual novelty feels moderate. The mode-seeking idea is mainly a reformulation of MAP estimation within the diffusion framework. While mathematically elegant, it is not entirely clear what fundamentally distinguishes VML from existing guidance-based or posterior sampling methods beyond its derivation.

1. The practical improvements over prior MAP-based solvers are relatively small in some tasks, and the results mostly show incremental gains in LPIPS or FID rather than large performance leaps. The paper could analyze why the VML-based formulation improves results—whether due to better gradient alignment, reduced variance, or implicit regularization.

2. The derivations and proofs rely heavily on the assumption of a linear degradation operator and Gaussian noise. It is unclear how VML could generalize to nonlinear or learned operators, which are becoming increasingly common in inverse problems. The brief mention of extending to latent diffusion models acknowledges this, but the results there remain limited.

Overall, the work is technically impressive and experimentally solid, but it would benefit from stronger conceptual clarity on how the proposed formulation fundamentally advances beyond reweighted MAP estimation with standard priors.

### Questions
1. Can the proposed method be interpreted as an implicit form of score correction or gradient projection? If so, is there a way to visualize or quantify how the diffusion trajectory differs from existing guided diffusion methods?
2. The derivation assumes a linear operator H and Gaussian measurement noise. Could you discuss how the approach might extend to nonlinear or non-Gaussian settings?
3. The VML objective relies on minimizing $\mathrm{KL}(p(x_0\mid x_t) \| p(x_0\mid y))$. Why is the reverse KL chosen instead of the forward direction, and how does this affect the sampling trajectory compared to posterior averaging methods?

### Soundness
2

### Presentation
3

### Contribution
2
