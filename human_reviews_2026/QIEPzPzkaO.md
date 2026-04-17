# Free Lunch for Stabilizing Rectified Flow Inversion

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Rectified-Flow (RF)-based generative models have recently emerged as strong alternatives to traditional diffusion models, demonstrating state-of-the-art performance across various tasks. By learning a continuous velocity field that transforms simple noise into complex data, RF-based models not only enable high-quality generation, but also support training-free inversion, which facilitates downstream tasks such as reconstruction and editing. However, existing inversion methods, such as vanilla RF-based inversion, suffer from approximation errors that accumulate across timesteps, leading to unstable velocity fields and degraded reconstruction and editing quality. To address this challenge, we propose Proximal-Mean Inversion (PMI), a training-free gradient correction method that stabilizes the velocity field by guiding it toward a running average of past velocities, constrained within a theoretically derived spherical Gaussian. Furthermore, we introduce mimic-CFG, a lightweight velocity correction scheme for editing tasks, which interpolates between the current velocity and its projection onto the historical average, balancing editing effectiveness and structural consistency. Extensive experiments on PIE-Bench demonstrate that our methods significantly improve inversion stability, image reconstruction quality, and editing fidelity, while reducing the required number of neural function evaluations. Our approach achieves state-of-the-art performance on the PIE-Bench with enhanced efficiency and theoretical soundness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes Proximal-Mean Inversion (PMI), a training-free technique designed to stabilize the velocity field during the inversion process of Rectified Flow (RF) models. The core idea is to apply a proximal operator that pulls the predicted current velocity towards a running average of past velocities ($\bar{v}_{t_{k}}$). This correction is constrained within a theoretically derived spherical Gaussian to prevent trajectory deviation into low-density regions. Furthermore, the paper introduces mimic-CFG for the editing stage, which uses an interpolation strategy (akin to Classifier-Free Guidance) between the current velocity and its projection onto the mean velocity direction to balance editability and structure preservation. The authors claim state-of-the-art performance in reconstruction and editing tasks on PIE-Bench with enhanced efficiency (fewer NFEs).

### Strengths
Strong Theoretical Motivation and Novelty: The identification of accumulated approximation errors leading to velocity field instability is a crucial problem in RF inversion. The proposed solution of using a running mean velocity for proximal correction is novel and intuitively sound, as it leverages the global consistency property of the Rectified Flow trajectory (near-constant velocity field).Zero NFE Overhead: The implementation of PMI as a correction step without requiring any additional Neural Function Evaluations (NFEs) is a significant practical advantage. Many existing inversion refinement methods (like FPI) achieve accuracy at the cost of increased computational load, while PMI promises improved stability for free in terms of NFE count.Theoretical Constraint and Justification: The derivation of the Stability Condition (Proposition 1), which defines the radius $r_i$ for the spherical Gaussian constraint, is a strong point. It provides a principled way to control the correction magnitude, tethering the solution to high-density regions based on instability theory (citing Zhang et al., 2025). This moves the method beyond a simple heuristic.Comprehensive Experimental Validation: The method is evaluated rigorously across four distinct solvers (Euler, Heun, RF-Solver, FireFlow) for both reconstruction and editing tasks, covering a wide range of integration complexity. The consistent quantitative improvements in PSNR, SSIM, and LPIPS across all baselines (Table 1 and Table 2) convincingly demonstrate the general effectiveness and plug-and-play nature of PMI.Effective Editing Strategy (mimic-CFG): The mimic-CFG strategy is a clever application of the PMI principle to the editing context, successfully adapting the highly effective CFG framework using only the internal, historically averaged velocity as the "unconditional" signal. This efficiently addresses the trade-off between editability and consistency.

### Weaknesses
1. Reliance on Simplistic Approximation in PMI:The core update (Eq. 13) is derived by minimizing a first-order Taylor approximation of $F(v)$ (Eq. 11) constrained by an $L_2$ sphere (Eq. 12). This reduces the complex proximal minimization to a simple gradient step, making the "Proximal" in "Proximal-Mean" a somewhat generous term. The simplicity is beneficial for efficiency, but the theoretical depth of the final step is thin.

2. The $\mathcal{O}(\Delta t_i^2)$ Error Analysis (Proposition 2) is Misleading:The paper claims the local error remains $\mathcal{O}(\Delta t_i^2)$, the same as the standard Euler method. This is expected, as PMI is a correction applied after the Euler step, and it does not change the core differential equation integration method. The proof is trivial and does not justify the benefit of PMI. The true benefit lies in correcting the accumulated errors by leveraging the running mean, not in improving the local truncation error bound, which is solely dictated by the Euler method's order. This proposition adds little value and distracts from the method's actual contribution.

3. Lack of Sensitivity Analysis for $r_{t_{k}}$ (Proposition 1):The Stability Condition is derived based on the full-trajectory Gaussian properties of $\hat{z}_1$. However, the final parameter choice for $r_i$ (Eq. 14) still includes an arbitrary $\epsilon$ term (line 793, 794) and is proportional to $\Delta t_i$. The paper provides no empirical analysis on how $\epsilon$ is chosen or how sensitive the results are to the constant factor $\sqrt{2n+3\sqrt{2n}}/T$. This makes the supposedly "theoretically derived" radius look like a complex, slightly obfuscated heuristic, undermining the strong theoretical claim.Limited 


4. Editing Evaluation:While the quantitative results are strong, the editing evaluation is primarily focused on background preservation (PSNR, SSIM on unedited regions) and CLIP similarity (a weak proxy for true edit fidelity). A critical review requires evidence that the editing is semantically better and not just more consistent with the background. More focused metrics, such as a user study or more fine-grained editing task failures, would be required to fully validate the claimed "enhanced editing quality." The qualitative results in Figure 2 show subtle improvements, but the editing changes are often minimal.

### Questions
above

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
5

### Summary
This paper proposes Proximal-Mean Inversion (PMI), a training-free, gradient-based correction method for stabilizing velocity fields during inversion in rectified-flow (RF)-based generative models. The paper further introduces mimic-CFG, a velocity correction mechanism for editing tasks inspired by classifier-free guidance. Both methods aim to address approximation errors and instability in existing inversion approaches, improving reconstruction quality and editing fidelity. Experiments illustrate the effectiveness of the proposed methods.

### Strengths
- The motivation of this paper is clear. The inaccuracy of inversion is a noticeable problem of RF-based methods, where many prior works has been attempting to eliminate this. The proposed method aims to address this problem through gradient correction, which is intuitive and promising.
- The derivation of PMI is mathematically detailed, including closed-form updates (see Proposition 1 and Appendix A.1), and analysis of error bounds are also provided (Proposition 2, Appendix A.3). 
- Experiments are comprehensive, demonstrating the effectiveness of the proposed method on various settings.
- The paper is well-written and easy to follow.

### Weaknesses
- Some ablation studies are missed. As stated in Line 238, authors use the first-order Taylor Expansion to estimate the objective in Equation (10). Authors are expected to conduct experiments to demonstrate how does the expansion order influence the performance.

- The settings of baseline methods is not clearly stated in the paper. Some hyperparameters in baseline methods are crucial to their performance (such as the feature sharing choices in FireFlow and RF-Solver). Empirically, the flaws of baseline methods in Figure 2 might be addressed through adjusting these hyperparameters. As a result, authors are expected to provide the more detailed information for fair comparion.

- Personally, I think the performance improvement of PMI is marginal, especially in the image reconstruction tasks. As illustrated in Figure 1, in the conditional case, the performance of RF-Solver is good enough. Although the unconditional image reconstruction results of PMI in Figure 1 are better, I think it is not necessary to test the performance under this setting because nowadays the description of images are easy to be obtained (for example, by MLLM), which can be served as the condition.

### Questions
See weakness

### Soundness
2

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
3

### Summary
The paper introduces Proximal-Mean Inversion (PMI), a novel, training-free gradient correction method to enhance the stability and accuracy of RF-based generative model inversion. The motivation of this paper is the problem where the existing inversion methods suffer from approximation errors that accumulate across timesteps. PMI addresses this by guiding the perturbed velocity field toward a running average of past velocities using a proximal update. The paper also proposes mimic-CFG, a lightweight velocity correction that interpolates between the current velocity and its projection onto the historical average, balancing editing effectiveness and structural consistency. The proposed method demonstrates good results from the expriments.

### Strengths
1. The proposed methods are techically sounds and supported by theoretical proofs.
2. The approach achieves state-of-the-art quality on PIE-Bench with fewer sampling steps and no additional NFEs, accelerating the model.
3. Mimic-CFG provides an efficient guidance mechanism for editing on top of CFG, balancing structural consistency and editing control according to the experiment results.

### Weaknesses
1. The performance of the proposed methods seems to be dependent on hyperparameter selection, and could have potential for overcorrection. Over-correction (small $w$) harms both background preservation and editing quality, despite better SSIM/PSNR improvements. Similarly, for the proximal operator parameter $\lambda$ in editing, large values can lead to overcorrection, compromising editing quality.
2. How sensitive is the performance to the hyper-paraetmer across different RF models?
3. While the method is integrated into several solvers (Euler, Heun, RF-Solver, FireFlow), the key ablation studies (on $\lambda$ and $w$) are primarily conducted using one base method (Fig. 3). This limits confidence in how well the optimal hyperparameters and correction strategies generalize across different flow solvers or data distributions.

### Questions
see weakness section.

### Soundness
3

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
This paper proposes Proximal-Mean Inversion (PMI) to improve the stability and accuracy of inversion in Flow Matching models. The key idea is to perform a proximal correction of the predicted velocity field by guiding it toward a running average of past velocities, thereby mitigating accumulated approximation errors during inversion. The method also constrains updates within a theoretically derived spherical Gaussian region, ensuring stability in high-dimensional latent spaces. In addition, the authors propose mimic-CFG, a lightweight editing strategy that interpolates between the predicted velocity and its projection on the average direction, effectively balancing structural consistency and editability.

### Strengths
* The paper introduces a theoretically grounded proximal correction framework supported by rigorous stability and error analyses..
* The PMI formulation is elegant and practical, addresses the instability and accumulated inversion errors in flow-based generative models.
* Extensive quantitative and qualitative evaluations on PIE-Bench demonstrate consistent improvements across multiple baselines, in both inversion and editing tasks.
* The paper is well written, clearly organized, and easy to follow.

### Weaknesses
* Although the focus is on inversion-based editing, it would strengthen the paper to compare against a broader set of diffusion and flow-based editing baselines on PIE-Bench.
* Including the full benchmark results or additional comparisons with diffusion-based methods (e.g., DDIM inversion variants) would provide clearer context.
* Missing relevant recent works such as InfEdit [1], which explores inversion-free diffusion model based editing, and FlowEdit [3] / FlowChef [2], which also employ flow steering without inversion. These could better situate PMI’s contribution in the broader editing landscape.
* The proximal objective (Eq. 9) and its averaging strategy appear empirical. It would be interesting to analyze alternative formulations. For instance, exponential moving averages or adaptive weighting of past velocities.
* While mimic-CFG’s interpolation weight (w = 0.94) is empirically validated, additional discussion on its generalization or task dependency would help establish robustness.

[1] “Inversion-Free Image Editing with Natural Language,” CVPR 2024.

[2] “FlowChef: Steering of Rectified Flow Models for Controlled Generations,” ICCV 2025.

[3] “FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models,” ICCV 2025.

### Questions
* The stabilization of inversion and editing velocities is a promising direction. Could this proximal correction mechanism be extended to other tasks, such as inverse problem solving or classifier guidance stabilization (as in [3])?
* Eq. (9) defines a proximal objective with a simple average of velocities. What would happen if this were replaced by an exponential moving average (EMA) or momentum-based scheme? Would it improve convergence or maintain better global consistency over long trajectories?
* Could the authors discuss the potential interaction between PMI and high-order solvers like Adams-Bashforth-Moulton or used by RF-Solver or FireFlow? It would be insightful for readers to understand how the entire workflow evolves.

### Soundness
3

### Presentation
3

### Contribution
3
