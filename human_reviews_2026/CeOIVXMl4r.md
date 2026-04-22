# STORK: Faster Diffusion and Flow Matching Sampling by Resolving both Stiffness and Structure-Dependence

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 4, 6

## Abstract
Diffusion models (DMs) and flow-matching models have demonstrated remarkable performance in image and video generation. However, such models require a significant number of function evaluations (NFEs) during sampling, leading to costly inference. Consequently, quality-preserving fast sampling methods that require fewer NFEs have been an active area of research. However, prior training-free sampling methods fail to simultaneously address two key challenges: the stiffness of the ODE (i.e., the non-straightness of the velocity field) and dependence on the semi-linear structure of the DM ODE (which limits their direct applicability to flow-matching models). In this work, we introduce the Stabilized Taylor Orthogonal Runge–Kutta (STORK) method, addressing both design concerns. We demonstrate that STORK consistently improves the quality of diffusion and flow-matching sampling for image and video generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the challenges of non-straight velocity fields and the dependence on the semi-linear structure in diffusion model ODEs, which make fast but quality-preserving sampling difficult. STROKE tackles these issues by designing a solver that explicitly handles stiffness and structural dependency problems inherent in diffusion and flow-matching models.

### Strengths
- The authors redesign the Stable Runge–Kutta (SRK) method to align with the structural characteristics of diffusion and flow dynamics.
- They propose a general framework applicable to both noise-predicting (diffusion-based) and flow-based generative models, extending beyond the semi-linear assumption used in most prior solvers.
- The paper combines rigorous theoretical analysis (including convergence and stability proofs) with comprehensive quantitative and qualitative experiments, demonstrating consistent improvements across different datasets and models.

### Weaknesses
- Despite claiming “Stable Runge–Kutta,” the paper does not explore low-NFE (≤5) regimes, where stability advantages would be most pronounced. It remains unclear whether STROKE can outperform existing methods when the step count is drastically reduced.
- The paper lacks comparisons on larger benchmarks such as ImageNet-256 and ImageNet-512, which are standard for evaluating scalability and visual fidelity at higher resolutions.
- Although NFEs are matched across methods, a fair comparison in terms of actual computational cost is missing. The paper introduces the concept of *virtual NFEs*—intermediate approximations computed via Taylor expansion and finite-difference—but does not show how much wall-clock time, memory usage, or throughput efficiency is affected. Internal computation overhead (e.g., Taylor expansion, finite-difference updates, velocity caching) incurs additional cost even without extra NFEs. The paper abstracts this under “virtual NFEs,” but a detailed analysis of real runtime and resource usage would strengthen the claim of efficiency.

### Questions
- In Appendix B, the authors illustrate stiffness through a toy example, but can the same stiffness analysis be quantitatively applied to real diffusion or flow-matching models? A visualization or empirical stiffness profile for these models would help support the theoretical motivation.
- In Tables 6 and 7, the baseline outperforms STROKE at smaller NFEs, raising the question: how does STROKE behave in 512×512 image generation when NFE ≤ 5? It would be interesting to see whether SRK’s stability advantages appear under such extreme sampling constraints.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces STORK, a method for faster sampling from diffusion models by using the Taylor expansion version of a solver (SRK) that is more accurate for stiff ODEs.

### Strengths
1. The paper introduces a novel method for faster sampling from diffusion models.
2. The performance of STORK-4 shows improvements when NFE is small.

### Weaknesses
1. The paper motivates the proposed method by claiming diffusion models exhibit stiff dynamics, but this is not properly justified empirically or theoretically. There is insufficient evidence presented that diffusion models are actually stiff, making the motivation unclear. Table 1 shows that the STORK-4 is significantly better than SRK4. More intriguingly, even with 50 NFE of SRK4, its performance is around 6.167 for FID score which is lower than its STORK-4 at 10 NFE. This raises the question of whether SRK4 is a good solver for diffusion models and whether the stiffness assumption is valid.
2. Theorem 1 establishes the theoretical guarantee of STORK-4 based on its approximation quality to SRK4. However, it is not clear whether the theory actually supports the empirical finding that 10 NFE of STORK-4 can be better than 50 NFE of SRK4. Additionally, since the goal is to achieve fewer NFE in sampling, does the asymptotic rate (provided in Theorem 1) even matter in practice? The constant in front of the rate may matter more for the low NFE regime that this paper targets. Theorem 1 does not help understand the performance of few-step sampling, leaving the empirical gains unexplained by the theoretical analysis.
3. STORK-4 has many hyperparameters, which makes it unclear whether the observed performance gains come from genuine improvements or from more extensive hyperparameter search. Maybe fix a parameter like s=9 suggested in the paper and report the results for all experiments could help understand the performance gains.

### Questions
- Can you clarify why other methods, like the DPM solver, cannot apply to flow matching? What do you mean by the error introduced in each step when you talk about the data prediction step?

### Soundness
2

### Presentation
2

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
This paper introduces STORK, a new type of training-free numerical ODE solver for diffusion and flow models. STORK is based on SRK methods for solving stiff ODEs. To reduce NFEs, STORK further introduces virtual NFEs, which uses finite difference to approximate the derivatives of the velocity field to extrapolate intermediate velocities without extra NFEs. Experiments across various dataset and pretrained models show that STORK outperforms state-of-the-art ODE solvers such as UniPC and DPM++.

### Strengths
- The results presented in this paper are solid. Quantitative evaluations demonstrate a clear advantage of STORK over UniPC and DPM-Solver in few-step settings. Qualitative results also look promising: STORK seems to generate more details than other solvers under few-step settings, especially on video generation.
- Despite the dense math, the presentation quality of this paper is high, making it easily readable. Notably, this paper makes a connection with the notion of stiffness in classical numerical analysis, and then adapts the widely used SRK methods for flow models with a clear motivation.

### Weaknesses
- As mentioned in L300, it is stated that "naive application of SRK4 to the CIFAR-10 dataset results in very poor sampling results", so the authors propose to plug Taylor approximation to SRK4. This is an interesting observation, but the reason is not analyzed in depth. If SRK is considered a common method for solving stiff ODEs, why would it not work in this case? Using Taylor expansion and Adams-Bashforth approximation is common in previous flow ODE solvers, so plugging this into SRK seems to make it less unique.
- I wonder how the methods are categorized in Fig. 4. In particular, UniPC, which works very well in practice, is categorized as not stiff, whereas DPM-Solver is considered stiff. Yet it is clear that UniPC can be considered as an extension of DPM-Solver with an additional corrector. Please check this carefully.
- Another claimed advantage of the proposed method is structure-independency. However, in most cases, structure-dependency of ODE solvers is not really an issue in practice. It is very trivial to adopt diffusion solvers using exponential integrators for flow models by rescaling the noise schedule and reparametrizing the prediction format.
- Why is FID getting worse with more NFEs? (fig. 6, Table 6, 10, 11). This behavior is clearly different from DPM-Solver and UniPC, which often converge monotonically. Presenting images generated under higher NFEs could help analyzing this behavior.

### Questions
Please see weaknesses.

### Soundness
2

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
4

### Summary
The paper proposes an advanced ODE sampler, STORK, to handle the stiff dynamics of ODE, while remaining applicable to both score-matching models and flow-matching models. Authors tackle the limitation of prior samplers such as the famous DPM-Solvers, which rely on a semi-linear assumption, thus not natural to be directly applied to flow-matching models. STORK is a variant of stabilized Runge–Kutta (SRK), which is known to handle stiffness well, but it requires expensive intermediate model evaluations. So authors suggest Taylor approximation-based virtual NFEs to approximate internal stages. In experiments, they show strong empirical results across various tasks, including unconditional/conditional image generation and T2V generations, showing consistent improvements under low NFE setups.

### Strengths
1. Clear motivation and message: the authors clearly present what problem they aim to solve and how they approach it.
2. Conceptually sound method: Leveraging SRK to handle stiffness, and using Taylor-based virtual NFEs to reduce computational cost, finally making a viable and efficient sampler for flow-matching models
3. Comprehensive empirical evaluation: covering unconditional/conditional image generation and text-to-video tasks, with consistently strong results.
4. Well-written manuscript: the paper is clear and reader-friendly, e.g., additional explanation on Appendix B about stiffness and numerical analysis, parts like this make the paper accessible to broad readers

### Weaknesses
1. Runtime analysis: authors are only reporting NFE here, but for a more thorough analysis/comparison, they need to report wall clock time and GPU(VRAM) usage. So that readers could better understand in detail, e.g., how much of that time it takes to virtual NFE calculation?
2. On Table 1,
    - Clarification on NFE report: since they're comparing samplers, with different count evaluations such as inner NFEs for higher-order/intermediate calculation, or virtual NFEs, and so on... Table 1 is a bit hard to read thoroughly and feels unclear.
    The authors should consider explicitly separating NFEs for the super-step/sub-step for clearer comparison.
    - SRK vs STORK: so SRK is super-step (real NFE) + sub-step(real NFE), while STORK is super-step (real NFE) + sub-step(virtual NFE).
    It would be helpful to show the comparison not just in "budget-fair" regime, but in "algorithm-fair" regime, e.g., evaluate both methods with the same super/sub-step structure where SRK operates properly (i.e., with sufficient NFE budget).
    It could strongly back the message “SRK works well but is too expensive, hence we propose STORK to replace those sub-step NFEs with virtual ones.”
3. Stiffness analysis
    - Since the "stiffness" is one of the main motivations, the actual empirical stiffness analysis on pretrained diffusion/flow matching models would strengthen the argument. While calculating Jacobian could be expensive, even a toy-level experiment or approximate analysis would be highly informative and appreciated.

[minor]

1. I understand the limited page for the main part, but it would be beneficial if the ablation table for the main hyperparameter (Table 3 in the supplement), or pseudocode/algorithm (algo1, algo2) could be moved or at least mentioned in the main paper.

### Questions
This is a minor question, but Fig5a looks somewhat unusual, where it shows DPM-Solver++ outperforming UniPC, which contradicts to already reported results from other papers - why is this?

### Soundness
3

### Presentation
2

### Contribution
3
