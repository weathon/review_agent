# ParaSolver-Turbo: Accelerating Parallel Diffusion Integrator via Intrinsic Partially Linear Structure

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
This paper explores the challenge of accelerating the sequential inference process of Diffusion Probabilistic Models (DPMs). We tackle this critical issue from a dynamic system perspective, in which the inherent sequential nature is transformed into a parallel sampling process. 
Specifically, we first reveal that the sequential integral solver of the diffusion model can be approximated by a full linear solver, enabling efficient computation for parallel integral solvers of DPMs. Based on such a linear formulation, we then introduce a unified framework that reformulates the original nonlinear sequential integral process of diffusion model as a system of partial linear equations. Moreover, we further develop an immediate update strategy to solve the system. In addition, we prove that (1) the system admits a unique root corresponding precisely to the trajectory of the sequential integral solver; (2) solving the system guarantees convergence to the trajectory of sequential integral solvers in equal or fewer iterations. 
Building on these insights, we present \textit{ParaSolver-Turbo}, a partial linear parallel integral solver to accelerate a broad class of sequential and parallel sampling methods such as DDPM and ParaSolver.
Extensive experiments validate that ParaSolver-Turbo achieves $2\times\sim50\times$ speedup in terms of wall-clock time without measurable quality degradation. The source code will be released publicly.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper considers accelerating the diffusion sampling process by solving a set of nonlinear and linear equations in a parallel manner. A so-called ParaSolver-Turbo method is proposed as an extension of ParalSolver. A sliding window approach is proposed to turn certain linear equations into nonlinear equations to improve sampling quality. Experimental results indicates the fast sampling process in terms of wall-clock in comparison to other parallel methods.

### Strengths
The main contribution of the paper is to introduce both a set of linear and nonlinear equations in the sampling process, where the set of linear equations sit at the low noise region (closer to the estimated clean image) while the set of nonlinear equations sit at the high noise region (far away from the estimated clean image). Basically, the set of linear equations are obtained by using a common estimated clean image.  I think it is because of the introduction of linear equations, which makes it more computationally efficient than other parallel sampling methods.

### Weaknesses
(1) I don't think Proposition 2 is correct. That is, for different values of ϱ, the ϱ-nonlinear equation system
in Eq. (10) is NOT equivalent. This is because the assumption that the diffusion model has a perfect noise predictor does not hold in practice. I don't think it is a widely used assumption. If the diffusion model has a perfect noise predictor, 1-step sampling would be enough. If the assumption hold, the conclusion in Proposition 2 does not provide any practical guidance.  

(2) In the paragraph of "Hyperparameter Settings" in Section 6, the authors should mention how the hyper-parameters are set for ParaSolver-Turbo, rather than saying "More details are provided in the Appendix." I would think the setup for the parameter rho is crucial. 
 
(3) In Table 1, one thing I don't understand is why for 1000 steps, the NFE needed for the new method much smaller than that of DDPM. On the other hand, for steps of 25 and 50, the NFE for the new method is larger than that of DDPM. Is it because for 1000 steps, the new method performs sampling over a set of coarse timesteps? If it is the case, then it is not a fair comparison w.r.t. the ParaSolver.

### Questions
(1) Right below Equ. (4), how come N=T where N is timestep index?

(2)  The authors only state "More details are provided in the Appendix." in a few places without specifying the section in appendix. 

(3) Please specify the number of GPUs being used in Table 1 for each method.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work studies how to accelerate the reverse process of diffusion models by designing a parallel sampling algorithm, proposes the ParaSolver-Turbo algorithm, and achieves a 71.4x speedup. To achieve this goal, this work proposes the $\varrho$-nonlinear equation system, which combines the nonlinear score and linear conditional score, to avoid a large NFEs.

### Strengths
1.	From the empirical perspective, ParaSolver-Turbo achieves better performance compared with DDPM, DDIM, DPMsolver, and ParaSolver, ParaDiGMs, with a much faster speed.

### Weaknesses
A minor concern is the assumption. In the proof of Proposition 1, Eq. (21) uses the analytic form of the condition score $\nabla \log p(X_t|X_T)$ for the unconditional score $\nabla \log p(X_t)$, which is also adopted by Song et al. It would be better to discuss this assumption in the main content.Questions:

### Questions
Please see my weakness.

1.	Since the flow-based models is popular, it would be better to do experiments in the flow-based models (such as flux or SD 3.5) to show the advantage of the algorithm. 

Comments: 

1.	It would be better to rewrite lines 712.

2.	It would be better to add the discussion of limitations and broader impacts (in the appendix if there is no space in the main content).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ParaSolver-Turbo, a method that formulates diffusion sampling as solving a system of nonlinear equations and accelerates it through parallel sampling. Building upon previous works, it reformulates the diffusion differential equation, which is traditionally solved sequentially, into a system of banded nonlinear equations. The paper then introduces a linearized version of this system by incorporating the final clean sample estimated by the neural network. By combining the linear and nonlinear systems, the proposed method strikes a better balance between computational cost and solution accuracy. Experiments demonstrate that ParaSolver-Turbo achieves greater speedup than baseline methods such as ParaDiGMS and ParaSolver.

### Strengths
1. Combining parallel sampling and diffusion solver is a relatively new idea.
2. Introducing a linear system to reduce computational cost sounds a reasonable method.

### Weaknesses
1. My major concern lies in the experimental results and settings.
    - As shown in the main table, compared with traditional solvers, the iteration compression ratio of ParaSolver-Turbo is generally smaller than its speedup ratio (e.g., for DDIM with 25 steps, the iteration compression ratio is 25/11=2.27<3.1). However, in my understanding, since parallel computation increases the computation amount per iteration, the speedup should not exceed the iteration compression ratio. Could the authors provide explanation to this?
    - From my experience, image generation is typically a computation-intensive task, where the actual computation cost dominates the inference time rather than the number of model calls, especially for large models like Stable Diffusion. Therefore, the actual speedup ratio should correlate more with NFE than with iterations. Achieving a speedup through parallel sampling seems strange to me. \textbf{I suggest reporting the batch size per GPU and GPU utilization, and adding comparison results with fewer-step solvers in Table 2.} 
        - Moreover, for video generation tasks, diffusion models contribute an even larger portion of the computation cost, so this approach appears to have limited scalability for such larger tasks.
    - Some of the baselines compared in this paper are somewhat outdated. Current traditional sequential solvers have long achieved 5–10 step sampling [1–4], and the authors should discuss or compare the proposed method with these methods. Some of the claims about speedup are still made against the earliest 1000-step DDPM, which seems overclaimed.

2. I suggest the authors add a more detailed discussion on ParaSolver-Turbo in relation to previous parallel sampling methods, such as ParaSolver, ParaDiGMS, and ParaTAA. This would help readers quickly grasp the landscape of this research area and understand the proposed method.

[1] Zheng, K., Lu, C., Chen, J., & Zhu, J. (2023). Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36, 55502-55542.

[2] Zhou, Z., Chen, D., Wang, C., & Chen, C. (2024). Fast ode-based sampling for diffusion models in around 5 steps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7777-7786).

[3] Liu, E., Ning, X., Yang, H., & Wang, Y. (2024). A unified sampling framework for solver searching of diffusion probabilistic models. In The Twelfth International Conference on Learning Representations.

[4] Liang, Y., Fang, X., Chen, H., & Wang, Y. Linear Multistep Solver Distillation for Fast Sampling of Diffusion Models. In The Thirteenth International Conference on Learning Representations.

### Questions
1. Since the linear system is introduced to reduce the number of denoised samples per step, which introduces additional errors (because the predicted clean sample is only an expectation and is different from the true clean sample), how can ParaSolver-Turbo achieve fewer iterations in some cases compared to ParaSolver?

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
3

### Summary
This paper addresses the challenge of accelerating the inference of diffusion models. The authors proposed ParaSolver-Turbo, a parallel sampling framework that exploits a partial linear structure in the diffusion integral formulation. The experimental result show that the proposed solver outperforms previous parallel sampling methods in inference time while maintaining generation quality.

### Strengths
1. The reformulation of sampling process into a partially linear system is novel.
2. Theoretical results are provided, assuming an ideal denoiser. 
3. ParaSolver-Turbo has good empirical performance. It is faster than existing parallel sampling methods while maintaining similar FID and CLIP scores.

### Weaknesses
1. Both the reformulation and theoretical analysis rely critically on the ideal denoising assumption, which is theoretically impossible. This weakens the theoretical basis of the proposed method and undermines the claimed theoretical properties in realistic settings. 

2. The paper mentions that one-step methods like diffusion distillation and consistency models reduce sample quality, but the experiments do not compare ParaSolver-Turbo with these methods. It is unclear whether the proposed approach achieves better quality.

### Questions
1. In the proof of Proposition 1, what's the justification for the first equality in (21)? It is insufficient to say following Song et al 2023b. 

2. Lines 206-208, why does it need information regarding the target data distribution if it equivalent as claimed? 

3. What's the implication of the failure of the ideal denoising assumption? 

4. Is there any rule for choosing $\rho$?

### Soundness
3

### Presentation
3

### Contribution
3
