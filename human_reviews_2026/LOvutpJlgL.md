# Practical Diffusion Planning via Temperature-Guided Reward Conditioning

- Decision: Reject
- Scores: 8, 4, 4, 6

## Abstract
Diffusion planners address sequential decision-making by framing plan generation as a generative modeling task over trajectories, mitigating compounding errors and myopic predictions typical of autoregressive methods. They sample long-horizon, globally consistent plans in a single pass, enabling parallel refinement and robust handling of multimodal futures. Reward conditioning is typically achieved through classifier guidance or classifier-free guidance (CFG), with CFG favored for its performance and flexibility but requiring extensive, task-specific hyperparameter tuning that limits scalability and generalization. Our analysis reveals that guidance performance hinges on careful adaptation to the data manifold and reward distribution, contributing to CFG's hyperparameter fragility. In this work, we propose the temperature-guided diffusion planner (TGDP), which adapts CFG for reward conditioning by self-calibrating to these task-specific characteristics. TGDP leverages temperature-based sample reweighting during training and adaptive guidance scaling at inference, yielding robust high-reward plan generation without per-task hyperparameter optimization. Across standard reward-driven benchmarks, TGDP matches performance of prior methods while maintaining a single set of default hyperparameters, establishing a practical, scalable, and generalizable approach to diffusion-based planning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
SafeFlowMatcher meaningfully advances SafeDiffuser’s core idea—using CBFs to guarantee safety in generative planning—by swapping the stochastic, per-step–constrained diffusion backbone for a deterministic flow-matching model with a prediction–correction integrator and vanishing time-scaled CBF correction. This shift strengthens the theory (deterministic forward invariance with finite-time convergence vs. SafeDiffuser’s finite-time probabilistic invariance), slashes computational load (one/few ODE passes and a single lightweight QP, rather than a QP at every denoising step), and improves practical behavior (no boundary trapping, higher task scores, and real-time feasibility). The trade-offs are modest—FM is less naturally exploratory than diffusion and both approaches still require known, differentiable safety functions—but for robotics/control where safety and latency dominate, SafeFlowMatcher is a clear, useful step beyond the original SafeDiffuser.

### Strengths
- Deterministic safety guarantees with forward invariance and finite-time convergence, rather than SafeDiffuser’s almost-sure (probabilistic) invariance across stochastic reverse steps. This yields cleaner, stronger theory for deployment.

- Much lower computational load: one/few ODE integrations plus a single lightweight CBF-QP, instead of solving a QP at every denoising step. This makes real-time online planning feasible where SafeDiffuser typically isn’t.

- Prediction–correction decoupling minimizes distributional distortion. The model first predicts with flow matching, then applies a vanishing time-scaled CBF correction that avoids boundary sticking/local traps that SafeDiffuser mitigates with relaxed or time-varying specs.

- Better empirical planning quality at equal or stronger safety: smoother rollouts, higher task scores, and zero safety violations. Deterministic dynamics reduce stochastic artifacts present in diffusion-based sampling.

- Simpler and more portable pipeline: no long reverse diffusion chain, fewer sensitivity knobs, and a safety layer that operates on executed trajectories. This makes it easier to plug into different environments/backbones than SafeDiffuser’s stepwise embedded constraints.

### Weaknesses
- Requires known, differentiable, and correctly calibrated safety sets b(x). In real systems with perception noise, contacts, or nonconvex geometry, specifying (or learning) smooth, faithful barriers is hard and errors can yield either over-conservatism or false safety.

- CBF-QP feasibility is not guaranteed, especially under tight actuation limits or when the predicted state is far inside the unsafe set. Slack-based fallbacks weaken guarantees and the paper lacks a systematic analysis of infeasibility rates and recovery.

- Limited robustness treatment to uncertainty and mismatch (estimation noise, delays, unmodeled dynamics, moving constraints). The deterministic guarantees don’t provide ISS/chance-constrained bounds, so small errors could accumulate into violations.

- Potential distributional drift from repeated corrections: even with vanishing scaling, the safety projection can bias trajectories away from the learned flow over long horizons, reducing diversity and pushing states out of the model’s training support.

- Empirical scope and baselines: comparisons focus on diffusion backbones; fewer head-to-head results versus strong control baselines (e.g., MPC + CBF-CLF-QP, Neural-ODE/CNF + CBF filters) and no on-hardware validation to substantiate real-time claims.

### Questions
- Can you quantify the practical bottlenecks of SafeDiffuser that specifically motivated your design choices (e.g., per-step QP latency, end-to-end wall clock, trap rates), and state the a priori performance targets you aimed to hit so readers can judge whether the reported gains meet those targets?

- What conditions ensure feasibility of the CBF-QP in the correction step under tight actuation limits or when the predictor proposes states deep inside the unsafe set, and what is the formal recovery strategy when the QP is infeasible (e.g., backtracking the rollout, horizon extension, or schedule adjustment)?

- How sensitive is performance and safety to the vanishing time-scaling schedule and the choice of the class-K function α(·), and can you provide ablations and practical tuning guidance that demonstrate stable behavior near boundaries without excessive conservatism or reward loss?

- To what extent do repeated corrections induce distributional drift away from the nominal flow, and can you report quantitative divergence metrics (e.g., energy distance or trajectory FID) alongside impacts on trajectory diversity and long-horizon returns?

- Could you include baselines beyond diffusion backbones—such as MPC with CBF-CLF-QP, Neural-ODE/CNF planners with a CBF filter, and diffusion with a single-shot CBF projection—to isolate how much of the gain arises from the prediction–correction plus vanishing mechanism rather than from the backbone swap alone?

- How robust are the proposed guarantees in the presence of state-estimation errors, dynamics mismatch, delays, or moving constraints, and can you provide an ISS or chance-constrained analysis (or targeted experiments) that bounds violation probability under realistic sensing and model errors?

### Soundness
4

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
4

### Summary
The author introduces the Temperature-Guided Diffusion Planner (TGDP), an approach that addresses the hyperparameter fragility of Classifier-Free Guidance (CFG). Conventional CFG in diffusion planning often necessitates extensive, task-specific hyperparameter tuning because guidance performance critically depends on adapting to the data manifold and reward distribution. TGDP mitigates this by self-calibrating to these task-specific characteristics through a proposed training and inference scheme. During training, it reweights the diffusion loss for each sample based on its return and a randomly sampled temperature. At inference, TGDP computes an adaptive guidance scale by measuring the geometric collinearity between denoising targets conditioned on high, zero, and low temperatures. The authors provide emprical evidence on D4RL locomotion, Maze2D, and Kitchen benchmarks demonstrating that TGDP consistently matches or surpasses the performance of prior diffusion planners (e.g., CFG, CG, MCSS) while utilizing a single, fixed set of default hyperparameters.

### Strengths
1. The paper tackles a highly practical and impactful problem, "hyperparameter brittleness of CFG".
2. The motivation is highly intuitive and reasonbale.
3. The suggested method is simple and novel to my understanding.

### Weaknesses
1. The method replace the tuning of CFG's guidance scale and target with its own hyperparameters. However, the maximum temperature is also a critical hyperparameter that defines the training objective and guidance targets at inference. The paper uses a fixed value but does not provide an anylsis of how this value was chosen or how sensitive the model's performance is to it. 
2. The adaptive scaling relies on the assumption that the collinearity of diffusion targets is a reliable proxy for distinguishing between intra-mode and inter-mode guidance. While this appears to hold true for the tested D4RL benchmarks, these environments, while standard, may not cover all possible return landscape complexities. The paper would be strengthened by a brief discussion on potential failure modes or types of data distributions where this geometric heuristic might be less effective.
3. Although this pepr focused on improving CFG, I suggest the author include more related works based on CG that tackles inaccurate guidance [1-3].

[1] Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning, 2023

[2] Inference-Time Policy Steering through Human Interactions, 2025

[3] Local Manifold Approximation and Projection for Manifold-Aware Diffusion Planning, 2025

Comment: I believe this work has promise. While I currently recommend borderline reject, I am willing to raise my score if the authors adequately address my concerns through revisions or clarifications.

### Questions
see weakness

### Soundness
3

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
2

### Summary
This paper introduces Temperature-Guided Diffusion Planning (TGDP), a practical and scalable framework for reward-conditioned diffusion planning. TGDP improves upon Classifier-Free Guidance (CFG), which is widely used in diffusion models but suffers from fragile, task-specific hyper-parameter tuning.

### Strengths
This work presents a Practical Diffusion Planning pipeline after DV [1] in the field of diffusion planning. I think this paper will be another attempt on improving diffusion planning, providing inspiration for robotics (especially for the manipulation field), and also offline RL community. The motivation behind this paper is solid: CFG/CG is fast but require heavy hyper-parameter tuning for target return and CFG scale. MCSS (DV), requiring no parameters tuning, provides high-quality inference, but is slow for generating since it requires more unbiased data for selection. This paper provide experiments on the classic D4RL dataset, which should be good for other researcher to have a try.

[1] What Makes a Good Diffusion Planner for Decision Making? Lu et.al ICLR 2025

### Weaknesses
I think the main weakness is the paper writing. Till now I still not very sure what is actually temperature conditioned diffusion planning. So if the author can provide a detailed answer to my question, I am considering raising my score. I will be staying online during the rebuttal period. I hope the authors can provide active response during rebuttal.

1) What is $\beta_{max}$ for? Is that a task-specific parameter? Why do we need $\beta_{max}$? Is $\beta_{max}$ also a condition that is required to feed into the network of diffusion?
2) What is the intuition behind designing eq (5)? Could the author elaborating more on the introduction of **cosine-similarity** between the 0/high/low-temperature diffusion output? Any math / intuition supported is quite encouraged. My understanding: "if predictions come from different modes → reduce scale (avoid over-guidance) If within same mode → strengthen guidance (avoid underguidance)" but this makes no sense to me, I can also say "if predictions come from different modes → you should improve scale (to quickly passing the confusion state)." I am hoping to get more insights here.
3) For Algorithm 1 line 1322, there's $\tau_{low}$, but it is never used below? This is quite confusing. Is that a typo?
4) Can the authors provide more experiments on other datasets, except for the 9 classic dataset in D4RL?
5) I do not find any codes in OpenReview or from the paper.

### Questions
If CFG can be represented as:

 `sample = D(zero) + s * (D(target_return) - D(zero))`, 

where target_return requires to be tuned.

Can TGDP be summarized in one sentence, as:

`sample = D(eps, zero) + s_adaptive * (D(eps, $\beta_{max}$) - D(eps, zero))`, 

where s_adaptive is large when D(eps, $\beta_{max}$) is similar to D(eps, -$\beta_{max}$), otherwise it is small?

Is my understanding is correct?

### Soundness
3

### Presentation
1

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
The paper introduces a Temperature-Guided Diffusion Planner (TGDP) to address the limitations of traditional reward guidance based on classifier-free guidance (CFG), which is often inflexible and not scalable due to its high sensitivity to hyperparameters. In this work, the authors incorporate a temperature parameter as an auxiliary input to the diffusion model. During training, the diffusion model is optimized with temperature-weighted sampling to capture reward-aware distributions, and during inference, the temperature is adaptively utilized to control the guidance strength in the sampling process. Experimental results demonstrate that TGDP achieves significant improvements over conventional diffusion-based planners.

### Strengths
1. The paper introduces a novel idea of adaptively adjusting the level of reward guidance, which is both meaningful and practical for diffusion-based planning.

2. The authors provide comprehensive theoretical analysis on the application of temperature in both training and guidance, presented in the appendix.

3. Extensive experimental results are included to validate the effectiveness of the proposed method, along with detailed implementation information.

### Weaknesses
1. The method section is somewhat disorganized, as parts of it include content that would be more appropriate for the experimental section (e.g. detailed descriptions of the D4RL implementation). This mixing of methodological explanation and implementation details makes it difficult to follow the core ideas of the proposed approach.

2. In addition, there are numerous formatting issues throughout the paper, including unusually large spacing between text, figures, and tables. Several figure legends overlap with the plots themselves, obscuring key details and making it hard to interpret the results clearly.

### Questions
1. How can the authors ensure that trajectory returns are effectively encoded through temperature-weighted training? Would this implicit formulation reduce scalability or generalizability compared to traditional reward-guided inference methods, especially when the reward function changes across tasks?

2. The paper states that traditional reward guidance is sensitive to the hyperparameter controlling guidance strength. However, could the maximum temperature $\beta_\text{max}$ itself become an important hyperparameter that significantly affects the performance of TGDP?

### Soundness
3

### Presentation
2

### Contribution
3
