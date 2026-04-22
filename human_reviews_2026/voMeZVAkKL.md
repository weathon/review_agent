# FAST‑DIPS: Adjoint‑Free Analytic Steps and Hard‑Constrained Likelihood Correction for Diffusion‑Prior Inverse Problems

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Training-free diffusion priors enable inverse-problem solvers without retraining, but for nonlinear forward operators data consistency often relies on repeated derivatives or inner optimization/MCMC loops with conservative step sizes, incurring many iterations and denoiser/score evaluations. We propose a training-free solver that replaces these inner loops with a hard measurement-space feasibility constraint (closed-form projection) and an analytic, model-optimal step size, enabling a small, fixed compute budget per noise level. Anchored at the denoiser prediction, the correction is approximated via an adjoint-free, ADMM-style splitting with projection and a few steepest-descent updates, using one VJP and either one JVP or a forward-difference probe, followed by backtracking and decoupled re-annealing. We prove local model optimality and descent under backtracking for the step-size rule, and derive an explicit KL bound for mode-substitution re-annealing under a local Gaussian conditional surrogate. We also develop a latent variant and a one-parameter pixel$\rightarrow$latent hybrid schedule. Experiments achieve competitive PSNR/SSIM/LPIPS with up to 19.5$\times$ speedup, without hand-coded adjoints or inner MCMC. Code and data: [here](https://github.com/ququlza/FAST-DIPS)

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces FAST-DIPS which aims to improve both the speed and accuracy of existing diffusion-prior solvers. The core contribution is a two-stage update process applied at each step of the reverse diffusion chain: "analytic step" and "hard-constrained likelihood correction" step. The experimental results show that FAST-DIPS consistently outperforms state-of-the-art methods in terms of reconstruction quality and/or computational time.

### Strengths
1. The proposed combination of an analytic data consistency step with the hard-constrained likelihood correction is novel and appears highly effective. 
2. The paper provides compelling empirical evidence of its superiority, often delivering higher quality in a fraction of the time required by competing methods.
3. The paper is theoretically sound and clearly structured.

### Weaknesses
1. The Method section is dense and overly technical, relying heavily on acronyms and mathematical derivations with minimal intuition.

2. The "adjoint-free" claim could be misleading. The method avoids computing the adjoint operator $A^T$ at each iteration by pre-computing a term involving $(A A^T)^{-1}$. This is only advantageous for a specific class of operators where this matrix is easy to compute and invert. 

3. The choice of $\epsilon$ in the hard constraint is not well explained

4. The extension of the method to non-linear problems like phase retrieval is handled by linearizing the forward operator at each step. This is a reasonable approach, but the paper provides very little detail or justification for it. 

5. The exclusive use of FFHQ restricts the generality of the conclusions.

### Questions
1. Could the author please clarify the practical limitations of the adjoint-free formulation? For which classes of forward operators $A$ does the pre-computation of $(A A^T)^{-1} A$ become a bottleneck that outweighs the per-iteration speed-up?

2. Though the experiment section has included various inverse problems, the selection of the dataset is very limited. Can the author please add at least one more dataset other than human face to benchmark the performance?

3, The hybrid schedule introduces a switching threshold $\sigma_\text{switch}$. How is this parameter selected in practice, and how does performance vary with its value?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes FAST-DIPS, a fast and training-free solver for diffusion-prior inverse problems. The key idea is to perform a hard-constrained likelihood correction at each diffusion step through an adjoint-free ADMM scheme with analytic step size computation, thereby avoiding the need for hand-coded adjoints or inner MCMC loops. 
Experimental results across multiple linear and nonlinear inverse problems demonstrate that FAST-DIPS achieves comparable or superior reconstruction quality while reducing runtime compared to state-of-the-art training-free baselines.

### Strengths
1. **Clear and principled framework**:
The paper presents FAST-DIPS, a training-free and adjoint-free framework for diffusion-prior inverse problems, offering a clean and principled alternative to existing plug-and-play or posterior-sampling approaches.

2. **Broad applicability and ease of use**:
The method supports both pixel-space and latent-space diffusion models through an adjoint-free design, making it versatile and easy to apply to a variety of inverse problems without hand-crafted adjoints or retraining.

2. **Strong empirical performance**:
The proposed analytic-step ADMM correction eliminates the need for hand-crafted adjoints or inner MCMC loops, making the method broadly applicable to both linear and nonlinear operators while achieving 5×–25× faster inference.

### Weaknesses
1. **Presentation issue**:
Table 1 exceeds the page width and is difficult to read in its current format. The authors should consider reformatting or splitting the table across pages to improve readability and compliance with ICLR formatting guidelines.

### Questions
1.The performance of the baseline algorithms in Table 1 differs from that reported in their original papers, even under the same experimental settings. For instance, the SITCOM paper reports a PSNR of 30.68 for the SR task, whereas Table 1 reports 29.555. A similar discrepancy is also observed for the DAPS algorithm. While Figure 3 effectively demonstrates the superiority of the proposed method under the same runtime, I would appreciate clarification regarding these inconsistencies.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces FAST-DIPS, a training-free diffusion-prior inverse problem solver. The core contributions are: (1) an adjoint-free correction step that (2) enforces a hard-constrained likelihood ($||\mathcal{A}(x)-y||\le\epsilon$) using (3) an ADMM formulation, where the primal update is solved efficiently with an analytic, non-iterative step size derived from a local quadratic model. The method is further extended to latent and hybrid (pixel/latent) settings.

### Strengths
1. The adjoint-free ADMM formulation with analytic step sizes (via VJP/JVP or finite-difference approximations) is a clever way to minimize engineering overhead while ensuring efficient and deterministic updates.

2. The paper provides strong local guarantees, including exact minimization of quadratic models (Proposition 3), KKT satisfaction at fixed points (Proposition 4), and descent properties with backtracking.

### Weaknesses
1. My main concern is that, while adjoint-free, the method still requires autodiff through $\mathcal{A}$, and the latent mode incurs repeated decoder calls. In my view, the engineering benefit of avoiding adjoints is somewhat offset by the computational cost of automatic differentiation through $\mathcal{A}$.

2. The method’s hard constraint ($||\mathcal{A}(x)-y||\le\epsilon$) shifts the tuning burden from the likelihood weight to the credible set’s radius $\epsilon$. Although this is briefly acknowledged in the limitations section, the paper should more thoroughly discuss how this critical hyperparameter is selected.

3. Experiments are limited to FFHQ (faces only), which lacks diversity; no tests are conducted on broader datasets such as CelebA-HQ, LSUN, or natural images (e.g., ImageNet subsets). This raises concerns about generalization to non-face domains or higher resolutions.

4. The complete algorithm, as presented in the appendix, is somewhat complex, making the method less elegant. Moreover, the writing could be improved for better clarity and flow.

### Questions
1. How was the hyperparameter $\epsilon$ (the radius of the credible set) determined for each of the eight experiments?

2. The paper claims to use no inner MCMC, but the ADMM iterations ($K=3$–$5$) with $S=1$ descent appear to form mini-loops. Please clarify if I have misunderstood this point.

3. For the latent variant, have you analyzed the computational cost of the JVP $J_{\mathcal{A}\circ\mathcal{D}}(z)g$?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces FAST-DIPS, a training-free solver for diffusion-prior inverse problems, including those with nonlinear forward operators. The method's core is a hard-constrained proximal correction via an adjoint-free ADMM. This approach replaces costly inner MCMC loops or iterative optimization with an analytic step size, computable from one VJP and one JVP. Experiments across eight linear and nonlinear tasks demonstrate comparable or superior quality to state-of-the-art baselines, while achieving faster runtimes.

### Strengths
1.  The method’s adjoint-free design is both interesting and practical, eliminating the need for hand-coded adjoints. By relying on standard automatic differentiation (VJP/JVP), the framework is directly applicable to a broad class of nonlinear inverse problems—a setting that remains challenging for many competing methods.

2.  The framework is evaluated extensively and demonstrates a substantial speedup over baselines such as DAPS by replacing costly inner MCMC loops with an analytic step size, while maintaining—often even improving—reconstruction quality.

### Weaknesses
1.  One concern is the method's reliance on differentiable forward operators. The entire framework is built upon the availability of VJP and JVP, making it inapplicable to common non-differentiable degradations such as JPEG restoration or quantization. This may limit its utility for many real-world degradation types.

2.  Another concern is the need for task-specific hyperparameter tuning. The authors show that key parameters were set to different values for each of the eight tasks. This implies that users must perform a new, potentially expensive hyperparameter search for any novel problem, undermining its practicality as a plug-and-play solver. Moreover, I wonder if this hyperparameter selection is essential for other baselines as well. If so, we need to include the cost of hyperparameter selection to show practical speed-up in the usage of the proposed method.

3. Moreover, I am skeptical about the method's scalability to higher resolutions. The computational cost of the VJP/JVP, especially for the latent variant, which requires backpropagation through the decoder, was manageable for the $256 \times 256$ experiments. However, this computation and memory overhead is likely to become a significant bottleneck as resolution increases, potentially limiting the framework's applicability to large-scale problems.

### Questions
Please see the Weaknesses for the details.

### Soundness
2

### Presentation
3

### Contribution
3
