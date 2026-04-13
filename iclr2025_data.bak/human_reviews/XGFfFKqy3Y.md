## Human Reviewer 1

### Summary
The paper introduces SITCOM (Step-wise Triple-Consistent Sampling), a new framework for solving inverse problems using diffusion models. SITCOM enforces three consistency conditions—measurement, backward, and forward consistency—to guide the sampling process and improve the fidelity of reconstructions. The approach is validated through comprehensive experiments on tasks such as deblurring, super resolution, and in-painting, with results showing competitive performance and reduced run-time compared to baseline methods.

### Strengths
nnovative Theoretical Contributions: The concept of backward consistency (Definition 1) is an original addition that, along with measurement and forward consistency, forms a robust framework for inverse problem solvers using diffusion models. This mathematical insight is well-formulated and adds depth to the field (lines 216–269).
Optimization-Based Sampling: The detailed explanation of the optimization problem in Equation (S1) and the associated algorithm (lines 324–431) shows a strong command of integrating theoretical principles with practical implementation.
Comprehensive Experiments: The paper includes results on various tasks and datasets, demonstrating the robustness of the proposed approach (Table 1).
Efficiency Improvements: The reported reduction in run-time, while maintaining or improving performance, is a notable practical advantage (lines 486–539).

### Weaknesses
Clarity in Backward Consistency: Definition 1 (lines 216–269) could benefit from more intuitive explanations or examples to make the concept clearer to a broader audience.
Absence of Mathematical Justifications: While the theoretical contributions are solid, the paper lacks formal mathematical justifications for key aspects:
Choice of Optimization Parameters: There is no in-depth analysis on the selection of K (number of optimization steps) or λ (regularization parameter). Providing this would strengthen the confidence in the method's robustness (lines 324–431).
Necessity of Triple Consistency: The paper argues for the simultaneous use of measurement, backward, and forward consistency but does not include formal proof or theoretical analysis justifying why all three are essential for the claimed performance (lines 216–377).
Convergence Analysis: A formal convergence analysis is missing, which could help clarify the reliability and scalability of the SITCOM sampler. While the algorithm shows empirical success, having mathematical insights into its convergence properties and error bounds would offer more confidence in its reliability and scalability.
Mixed Results in Some Tasks: The improvements in tasks like ImageNet Gaussian Deblurring and Phase Retrieval are less significant compared to other baselines (Table 1), which may raise questions about the trade-offs in complexity versus marginal performance gains.
Lack of Computational Details: Detailed metrics like Number of Function Evaluations (NFE) are not presented in the main manuscript (it is located at appendix F). Including these would provide a clearer view of the computational cost. Please consider reorganize the contents.
Comparison to Deterministic Samplers: The paper does not explore whether the SITCOM method could be adapted into a deterministic version like DDIM or address how it compares to the optimized noise schedules like EDM.

### Questions
1. Could the authors provide additional intuition or examples for Definition 1 to make backward consistency more accessible?
2. Are there any performance curves that illustrate the effect of varying the number of optimization steps K and sampling steps N on both image quality and computational efficiency?
3. Can the authors elaborate on how SITCOM performs when using fewer sampling or optimization steps and if there are trade-offs between quality and computational cost?
4. Is there potential for SITCOM to be adapted into a deterministic sampler like DDIM? If so, what modifications would be needed?
5. How does the method handle noise characteristics that deviate significantly from those tested in the paper?
6. Are there theoretical justifications for the choice of optimization parameters, or is there an analysis showing the method's convergence properties?
Overall: This paper makes a meaningful contribution by introducing a theoretically rich and empirically validated approach for inverse problem-solving using diffusion models. The addition of backward consistency is an interesting theoretical insight, and the experimental results are compelling. However, the paper could benefit from clearer explanations, formal justifications for certain methodological choices, and more detailed computational metrics. Addressing these points would enhance the completeness and accessibility of the work.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper identifies the key limitations in previous DMs for IPs and summaries, referred to as the triple-consistency conditions. The newly designed optimization-based method, SITCOM, achieves comparable or even superior restoration results with significantly reduced computational time. The paper demonstrates the superiority of the proposed method across commonly used benchmarks, including five linear tasks and three nonlinear tasks.

### Strengths
1. The intuitive explanation of the triple-consistency conditions is clear and reasonable. The paper identifies the potential factors that prevent existing methods from balancing quality and efficiency, summarizing as the triple-consistency conditions.

2. The paper consistently achieves comparable or superior empirical results across commonly used linear and nonlinear tasks as shown in Table 1.

### Weaknesses
1. **Incremental Modification and Unfair Comparisons**  
   The major issue with the paper is that it presents only an incremental modification over existing work and makes unfair comparisons.

   1. *Theoretical and Motivational Perspective*  
      Although the paper is the first to propose and formulate the triple-consistency conditions formally, each has already been individually adopted and utilized in other works. Specifically, measurement consistency is used in [4, 5, 6, 7], backward consistency in [1, 2, 3], and forward consistency in [4, 7]. The paper provides only intuitive explanations for each condition in Sections 3.1, 3.2, and 3.3, lacking clear evidence of how each contributes to the final restoration performance. A straightforward ablation study—removing each consistency condition from the proposed algorithm, SITCOM, and comparing performance with the full SITCOM while keeping other parameters—would give clearer hints of the significance of each condition. Unfortunately, the experiments fail to control variables, causing the conclusions of the paper potentially misleading.

   2. *Technical Perspective*  
      The proposed method, SITCOM (Algorithm 1 in the paper), differs from DCDP only in two aspects:
      * (1) Changing the optimization variable from $\hat{x}_0$ to $\hat{v}_i$, where these variables are connected by the Tweedie-network denoiser $\hat{x}_0' = f(\hat{v}_i; t, \theta)$.
      * (2) Adding an early stopping criterion (line 5 in Algorithm 1).

      My main concern is with (1). The paper claims this change is introduced to achieve backward consistency but comes at the cost of computing backward gradients through the Tweedie-network denoiser $f$. Consequently, when keeping other parameters the same (e.g., diffusion steps $N$, number of gradient updates $K$, stopping criterion $\delta$), SITCOM is significantly slower than DCDP due to the additional backward pass through $f$. However, the paper compares SITCOM with smaller $N$ and $K$ to DCDP with much larger $N$ and $N$ and concludes that SITCOM is more efficient in achieving comparable results. I strongly suggest that the authors conduct a detailed and fair comparison between SITCOM and DCDP by fixing all other parameters and only changing the optimization variables. Otherwise, it is unclear whether the observed efficiency gains are due to backward consistency or simply fewer diffusion and gradient steps.

2. **Other Minor Issues**
   1. The paper lacks comparison and discussion of several recently proposed methods [5, 6, 8].
   2. There is no formulation or experimentation on solving inverse problems with latent diffusion models, which are included in the baseline papers used [4, 7].
3. **Non-anonymous issue.** Files in the given code link [https://anonymous.4open.science/r/SITCOM-7539/README.md](https://anonymous.4open.science/r/SITCOM-7539/README.md) (`SITCOM.py`, `SITCOM_with_noise.py` and `SITCOM_with_noise_imagenet.py`, etc.)  contain non-anonymous information.

**References**

[1] Chung, Hyungjin, et al. "Diffusion Posterior Sampling for General Noisy Inverse Problems." *ICLR 2023*.

[2] Song, Jiaming, et al. "Pseudoinverse-Guided Diffusion Models for Inverse Problems." *ICLR 2023*.

[3] Song, Jiaming, et al. "Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation." *ICML 2023*.

[4] Li, Xiang, et al. "Decoupled Data Consistency with Diffusion Purification for Image Restoration." arXiv preprint arXiv:2403.06054 (2024).

[5] Wu, Zihui, et al. "Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors." arXiv preprint arXiv:2405.18782 (2024).

[6] Xu, Xingyu, et al. "Provably Robust Score-Based Diffusion Posterior Sampling for Plug-and-Play Image Reconstruction." arXiv preprint arXiv:2403.17042 (2024).

[7] Zhang, Bingliang, et al. "Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing." arXiv preprint arXiv:2407.01521 (2024).

[8] Dou, Zehao, et al. "Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective." *ICLR 2024*.

### Questions
Please address my concerns in the Weaknesses part.

### Soundness
1

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
5

---

## Human Reviewer 3

### Summary
The authors identify limitations in current diffusion model-based inverse solvers, noting that the measurement-guidance term often pushes the trajectory toward inconsistency, leading to artifacts in intermediate samples and requiring a large number of correction steps.

To address these issues, they propose three conditions for achieving measurement-consistent diffusion trajectories: (1) standard data manifold measurement consistency, (2) forward diffusion consistency, and (3) backward diffusion consistency.

1. Measurement-consistent: Reconstruction is consistent with the measurements.
2. Forward-consistent: Intermediate samples during the diffusion trajectory should resemble in-distribution samples produced by forward diffusion.
3. Backward-consistent: The measurement-guided output, followed by Tweedie’s denoising, should also be the Tweedie-denoised output of some noisy data.

The authors introduce an algorithm that ensures all three consistencies, called Step-wise Triple-Consistent Sampling (SITCOM). The proposed method outperforms existing approaches in various image inverse problems.

### Strengths
The paper is well-written and easy to follow.
- It clearly motivates the issue of inconsistency in current models and provides an intuitive solution to address these inconsistencies.
- The proposed algorithm, which integrates these solutions, demonstrates improved performance over existing methods.

The authors showcase the applicability of their method across various image inverse problems.

### Weaknesses
The claims about consistency are not fully substantiated. It remains unclear if the proposed method genuinely addresses all three inconsistencies.
- Experimental validation is needed to quantify how effectively the method mitigates these inconsistencies. For example, could experiments be designed to measure the degree to which each inconsistency is reduced?

The impact of each individual component in the method is not verified.
- How would the results change if one component were removed?
- Additionally, could similar components be added to other algorithms, such as DPS or DDNM, to reduce inconsistencies in those methods as well? Would this approach similarly alleviate inconsistencies?

### Questions
The questions are integrated in the section above.

### Soundness
2

### Presentation
4

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper presents an optimization-based algorithm, SITCOM, to solve inverse problems with a pretrained diffusion model. The authors state three conditions to hold at each sampling step of the diffusion model. SITCOM optimizes measurement consistency on $x_t$ and uses a resampling procedure to enforce the consistency conditions. Experiments show that SITCOM performs better than existing methods in several image restoration tasks.

### Strengths
- The paper is well-presented, with a clear and structured approach.
- The proposed algorithm introduces "forward consistency" and "backward consistency," which is a novel framework that may interest a broader research community.
- Experimental results on image restoration tasks are promising.

### Weaknesses
- The paper does not provide rigorous guarantees for the output of SITCOM, whereas several prior methods offer assurances on correct sampling from the desired posterior distribution [2,3,4,7].
- The technical novelty appears limited. The optimization of $x_t$ for measurement consistency using Tweedie’s approximation directly resembles DPS, while the resampling technique is derived from ReSample. The resulting algorithm appears to be a straightforward combination of existing methods.
- This paper lacks a comparison to many relevant works [1~8]. Some of them provide asymptotic exact posterior sampling [2,3,4,7], while [1] appears to achieve the proposed "backward consistency" as well. It seems strange that SITCOM is only compared with DPS, DDNM, and two closely related algorithms (DAPS, DCDP), which seemingly have not undergone peer review.
- The stopping criterion for preventing noise overfitting needs further justification. Given that the actual noise level $\sigma_y$ is typically unknown in practice, an ablation study on the algorithm's sensitivity to $\delta$ would be valuable.

### Reference
[1] Wang et al. "A Plug-in Method for Solving Inverse Problems with Diffusion Model." In NeurIPS 2024.

[2] Dou et al. "Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective." ICLR 2023.

[3] Gabriel et al. "Monte Carlo guided Denoising Diffusion models for Bayesian linear inverse problems." ICLR 2023.

[4] Wu et al. "Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors." In NeurIPS 2024.

[5] Rout et al. "Solving linear inverse problems provably via posterior sampling with latent diffusion models." In NeurIPS 2023.

[6] Song et al. "Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency." In ICLR 2024.

[7] Wu et al. "Practical and Asymptotically Exact Conditional Sampling in Diffusion Models." In NeurIPS 2023.

[8] Rout et al. "Beyond first-order Tweedie: Solving inverse problems using latent diffusion." In CVPR 2024.

### Questions
- How is the stopping criterion determined in practice? Does different $\delta$ influence the final output of SITCOM?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 5

### Summary
This paper addresses inverse problems using a plug-and-play approach with diffusion models, focusing on reducing the low sampling speeds that are characteristic of iterative diffusion models in these settings. To this end, the authors propose a "triple consistency" framework, adding a backward consistency component to the already established data and forward consistencies. The backward consistency aims to ensure that the solution obtained after data-consistency remains a valid diffusion model solution, essentially acting as a projection onto the intersection of data-consistent and diffusion model-consistent solutions.

The paper provides experimental results on both linear and nonlinear tasks and compares the proposed method with other existing techniques, showing performance improvements for certain tasks.

### Strengths
- The paper addresses a timely and important problem in inverse problem-solving using diffusion models.

- Experiments cover a variety of linear and nonlinear tasks, giving a broad perspective on the method's applicability.

### Weaknesses
Modeling of Backward Consistency: The concept of backward consistency is interesting, but its implementation feels somewhat ad hoc. While it resembles a projection onto the intersection of data and prior-consistent solutions, this usually requires multiple iterations to reach a meaningful intersection. Here, however, only a single iteration is employed, which may be insufficient.

Alternative Approach: The backward consistency term could potentially be better represented as a fixed-point condition for the denoiser. This approach might be more systematic, aligning with methods like RED-diff, which regularize through denoising consistency.
Sufficiency of Eq. (4): Additionally, it's unclear if enforcing Eq. (4) (the Tweetie consistency) is sufficient to ensure backward consistency. This claim requires more justification.

Experimental Limitations:
- Comparison Scope: Tables 1 and 2 lack comparisons with key existing methods. Notably, the absence of comparisons with RED-diff and PGDM in Table 2 weakens the empirical analysis.

- Unfair Comparison: Simply reusing the hyperparameters from existing methods without adjusting them for the current dataset results in an unfair comparison. This could particularly impact performance for methods like DPS, RED-diff, and PGDM, which may not perform optimally without parameter tuning.

- Sampling Efficiency: A central claim of this work is the reduced number of iterations required by the proposed method. However, the paper lacks concrete experimental evidence to substantiate this, such as timing results that would demonstrate faster sampling speeds in seconds. Given the emphasis on efficiency, readers would expect clear evidence supporting this claim.

[PGDM] Song, J., Vahdat, A., Mardani, M., & Kautz, J. (2023, May). Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations.

### Questions
- How sensitive is the performance with respect to the parameter λ?

see the weakness part for more comments

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4