# Scaling Multi-Agent Environment Co-Design with Diffusion Models

- Decision: Reject
- Scores: 2, 6, 8

## Abstract
The *agent-environment co-design* paradigm jointly optimises agent policies and environment configurations in search of improved system performance. With application domains ranging from warehouse logistics to windfarm management, co-design promises to fundamentally change how we deploy multi-agent systems. However, current co-design methods struggle to scale. They collapse under high-dimensional environment design spaces and suffer from sample inefficiency when addressing moving targets inherent to joint optimisation. We address these challenges by developing **Diffusion Co-Design (DiCoDe)**, a scalable and sample-efficient co-design framework pushing co-design towards practically relevant settings. DiCoDe incorporates two core innovations. First, we introduce Projected Universal Guidance (PUG), a sampling technique that enables DiCoDe to explore a distribution of reward-maximising environments while satisfying hard constraints such as spatial separation between obstacles. Second, we devise a critic distillation mechanism to share knowledge from the reinforcement learning critic, ensuring that the guided diffusion model adapts to evolving agent policies using a dense, low variance, and up-to-date learning signal. Together, these improvements lead to superior environment-policy pairs when validated on challenging multi-agent environment co-design benchmarks including warehouse automation, multi-agent pathfinding and wind farm optimisation. Our method consistently exceeds the state-of-the-art, achieving up to 39\% higher rewards in the warehouse setting with 66\% fewer simulation samples. This sets a new standard in agent-environment co-design, and is a stepping stone towards reaping the rewards of co-design in real world domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes **Diffusion Co-Design (DiCoDe)**, a framework that jointly optimizes **multi-agent policies** and **environment configurations** using **guided diffusion models**. The goal is to improve **sample efficiency** and **scalability** in high-dimensional environment design spaces. The method introduces two key components:

1. **Projected Universal Guidance (PUG)**: A sampling technique that guides diffusion models to generate **high-reward environments** while satisfying **hard constraints** (e.g., obstacle separation).
2. **Critic Distillation**: A mechanism that transfers knowledge from the **agent critic** to an **environment critic**, providing a **dense and up-to-date learning signal** for the diffusion model.

The method is evaluated on three domains: **warehouse automation (D-RWARE)**, **wind farm control (WFCRL)**, and **multi-agent pathfinding (VMAS)**. Results show **up to 39% higher reward** and **66% fewer simulation samples** compared to prior co-design methods.

### Strengths
- **Originality**:  
  DiCoDe is the **first method to apply guided diffusion models** to **multi-agent environment co-design**, and it introduces **PUG**, a novel constraint-aware sampling technique that combines **universal guidance** with **projected constraints**.

- **Quality**:  
  The paper provides **strong empirical results** across **three diverse domains**, showing **consistent improvements** over baselines. The **critic distillation** mechanism is well-motivated and addresses **policy shift**, a known issue in co-design.

- **Clarity**:  
  The method is described **systematically**, with **clear algorithms**, **pseudocode**, and **ablations**. The **visualizations** (e.g., shelf placement heatmaps) help illustrate the **learned environment structures**.

- **Significance**:  
  If scalable, DiCoDe could **fundamentally improve** how we design **real-world multi-agent systems** (e.g., warehouses, wind farms), where **environment layout** and **agent policy** are **tightly coupled**.

### Weaknesses
### W1. **Scalability vs. Algorithm Design is Unclear**
- While the paper claims **scalability**, the **computational cost** of **guided diffusion** **increases** with the **number of agents** and **environment dimensionality**.
- **No complexity analysis** is provided for **PUG sampling** or **critic distillation** w.r.t. **agent count**.
- **Missing experiment**: **Scaling curves** with **increasing agents** (e.g., 4→16→32) to **quantify** how **wall-clock time** or **memory** grows.

### W2. **Limited Benchmarking on Standard MARL Tasks**
- All experiments are on **custom or modified environments** (D-RWARE, WFCRL, VMAS).
- **No evaluation** on **widely-used MARL benchmarks** like:
  - **SMAC** (StarCraft Multi-Agent Challenge)
  - **MPE** (Multi-Agent Particle Environments)
  - **SMACv2** (stochastic version)
- This **limits generalizability**—it’s **unclear** whether DiCoDe **outperforms SOTA methods** in **standard cooperative/competitive settings**.

### W3. **Cooperation Mechanism is Under-Explained**
- The paper **asserts** that DiCoDe **improves cooperation**, but **does not explain** **how** or **why**.
- **No ablation** on **joint vs. individual rewards**, **communication**, or **emergent specialization**.
- **No visualization** of **agent behaviors** (e.g., trajectories, role separation) to **support** the **cooperation claim**.
- **Missing metric**: **cooperation-specific measures** (e.g., **joint action diversity**, **team coherence**, **inter-agent distances**) are **not reported**.


While DiCoDe introduces **novel ideas** and shows **strong results** in **custom domains**, the **lack of standard benchmark evaluation**, **unclear scalability analysis**, and **vague cooperation claims** **limit its contribution**. The **core novelty** (PUG + critic distillation) is **interesting**, but **without broader validation**, it **remains unclear** whether the method **generalizes** beyond **specific, engineered environments**. I encourage the authors to:

1. Evaluate on **SMAC/MPE** benchmarks.
2. Provide **scaling curves** with **agent count**.
3. Clarify **how cooperation is improved** with **quantitative evidence**.

### Questions
### Q1. Scalability Analysis
Can you provide **wall-clock time** and **GPU memory usage** as a function of **agent count** (e.g., 4, 8, 16, 32)?  
How does **PUG sampling time** scale with **environment dimensionality**?

### Q2. Standard Benchmark Evaluation
Why not evaluate on **SMAC** or **MPE**?  
Can you run **DiCoDe vs. QMIX/MAPPO** on **3s5z** or **simple_spread** to **validate generalizability**?

### Q3. Cooperation Mechanism
What **specific behaviors** emerge that indicate **better cooperation**?  
Can you provide **trajectory visualizations** or **role specialization plots**?  
Did you measure **joint action entropy**, **inter-agent distances**, or **task allocation patterns**?

### Q4. Ablation on Critic Distillation
What happens if you **remove critic distillation** and **train the environment critic only on returns**?  
How **sensitive** is the method to **M_distill** (number of Monte-Carlo samples)?

### Q5. Constraint Handling in PUG
How does **PUG** perform when **constraints are non-convex** (e.g., **connectivity**, **visibility**, **dynamic obstacles**)?  
Can you show a **failure case** where **PUG violates constraints** or **gets stuck in local minima**?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Diffusion Co-Design (DiCoDe), a novel framework for scalable and sample-efficient multi-agent environment co-design, where both agent policies and environment parameters are jointly optimised to maximise system performance. Existing co-design approaches suffer from combinatorial explosion and instability due to policy–environment non-stationarity. DiCoDe overcomes these issues through two key innovations: (1) Projected Universal Guidance (PUG), a constraint-aware diffusion sampling technique that enables diffusion models to generate valid, high-reward environment configurations while preserving diversity; and (2) Critic Distillation, which transfers dense value estimates from the MARL critic to a separate environment critic that provides low-variance, adaptive guidance to the diffusion process. This design allows the diffusion model to remain fixed after pretraining, while the critic dynamically steers environment generation as agents learn. Evaluated across three benchmarks — warehouse logistics (D-RWARE), wind farm control (WFCRL), and cooperative navigation (VMAS) — DiCoDe achieves up to 39% performance gains and 66% fewer samples compared to existing co-design baselines, demonstrating its scalability and stability in high-dimensional multi-agent design spaces.

### Strengths
1. The paper introduces Diffusion Co-Design (DiCoDe), a new framework for _jointly_ optimising multi-agent policies and environment parameters — a direction previously limited by scalability and sample inefficiency.
2. The paper proposes Projected Universal Guidance (PUG) that is a principled sampling technique that merges universal guidance and projected diffusion models, enabling generation of high-reward, constraint-satisfying environments. This addresses the infeasibility of standard classifier guidance and avoids invalid samples. This paper also proposes Critic Distillation that is a mechanism that distils information from the MARL agent’s critic into an environment critic, providing a dense and up-to-date learning signal. This reduces variance, mitigates policy-shift, and vastly improves sample efficiency — a key bottleneck in earlier works.
3. Across three diverse benchmarks — D-RWARE (warehouse logistics), WFCRL (wind-farm control), and VMAS (multi-agent navigation) — DiCoDe consistently surpasses baselines such as MAPPO-based co-design and domain randomisation. In general, DiCoDe achieves 39% higher task rewards and 66% fewer samples in warehouse logistics. In general, DiCoDe maintains strong performance as the number of agents/design parameters scales (e.g., from 2→8 turbines), unlike prior RL-based co-design methods that degrade sharply. More importantly, it demonstrates adaptability of DiCoDe to both discrete and continuous design spaces — a rare accomplishment among environment design algorithms.
4. The paper includes thorough ablation studies verifying the impact of both PUG and critic distillation: (1) Removing PUG or replacing it with prior sampling/guidance methods reduces performance by 26–57%; (2) Omitting critic distillation reduces stability and increases noise, validating its necessity. Qualitative results (e.g., shelf placement heatmaps) visually confirm that DiCoDe learns functionally interpretable structures.
5. This paper has provided implementation details, baselines, and hyperparameters are fully disclosed, with code promised for release. Also, nine-seed averages with confidence intervals indicate credible empirical validation. Evaluation spans multiple environment modalities, not confined to a single task class, strengthening generality claims. 
6. This paper moves the co-design field closer to real-world deployment, bridging simulation-based MARL and practical layout/dynamics optimisation. More importantly, it provides a unifying framework that could be extended to unsupervised environment design and multi-objective optimisation, as the authors note in discussion.

### Weaknesses
1. The paper lacks a formal theoretical analysis of convergence or stability for the co-design process.  While the diffusion-based sampling and critic distillation are well-motivated empirically, there is no formal justification (e.g., fixed-point or equilibrium guarantees) that the joint optimisation of agents and environment converges to any optimal co-design solution. The method’s stability arguments are heuristic — mainly relying on decoupling the generator (fixed diffusion model) from the adaptive critic, rather than a principled proof.
2. The diffusion model is pretrained independently of the MARL task using random environment samples.  This means the model can only generate designs lying within the support of the pretraining distribution. If the task’s optimal environments fall outside that distribution, DiCoDe cannot discover them.
3. The benchmarks mostly test spatial layout optimisation; dynamic environment parameters (e.g., stochastic rules, non-stationary dynamics) are not explored. Competing baselines are limited — the comparison excludes recent regret-based UED and diffusion co-design methods (like ADD [1]), making it hard to quantify progress beyond the immediate predecessors.
4. DiCoDe requires three separately trained components (diffusion model, MARL policy, and environment critic) with multi-stage coordination, which significantly increases training complexity and compute demand. The paper acknowledges this but doesn’t provide quantitative training-time comparisons or scaling analysis with respect to environment dimensionality.
5. Realistic design tasks often involve multiple competing objectives (e.g., safety, energy efficiency, human preference).  The paper briefly mentions this as future work but does not experiment with or theoretically extend to multi-objective optimisation.

[1] Chung, H., Lee, J., Kim, M., Kim, D., & Oh, S. (2024). _Adversarial Environment Design via Regret-Guided Diffusion Models_. In _Advances in Neural Information Processing Systems (NeurIPS 37)_.

### Questions
Please answer the following questions:
1. In Eq. (6), how is $\omega$ chosen, and does the resulting trade-off preserve the optimal joint solution or bias the diffusion toward high-entropy but sub-optimal regions?
2. The paper substitutes the estimated gradients $\nabla\_{\theta} u + \omega \nabla\_{\theta} V'$ into the reverse SDE. Are these gradients unbiased estimates of $\nabla\_{\theta} \log \Lambda^{*}\_{\phi}$, and what happens when $\nabla_{\theta} V'$ is inaccurate?
3. The paper claims 66 % fewer samples. Is this reduction measured in total environment evaluations, MARL steps, or wall-clock time?
4. Does the pretrained diffusion model require retraining for each task (D-RWARE, WFCRL, VMAS), or can it generalise across design domains?
5. How sensitive is Projected Universal Guidance (PUG) to the projection operator $P_\Theta$ — does its differentiability or non-convexity affect convergence? The reverse diffusion update is derived under the assumption that gradients $\nabla_{\theta_t}$ are well-defined and continuous.  If $P_\Theta$ is non-differentiable (e.g., clipping, rounding, or nearest-valid-cell projection), then the effective sampling dynamics break the continuity assumption of the SDE approximation. If $P_\Theta$​ is non-convex, the projection may not be unique — multiple feasible projections could exist.  That can make the diffusion process stochastic in unintended ways, and might bias the trajectory of the reverse process away from the true guided distribution. In diffusion training, the convergence of score matching relies on smoothness of the forward–reverse transitions. A hard projection step effectively introduces discontinuities in the score field, which can cause unstable or biased sampling, especially if used iteratively across hundreds of timesteps. Could PUG create bias if projection removes valid but rare designs?

Potential typo:

Is there any typo in Eq. (8)? I cannot find any term inside $\nabla_{x}(\cdot)$ is with respect to the variable $x$.

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
4

### Summary
This paper develops an algorithm combining diffusion models and multi-agent reinforcement learning (MARL) for finding an optimal design and agent strategy to maximize the utility. The algorithm has two main parts: the first part is the so-called projected universal guidance to generate a valid environment via projection, then they learn an environment critic by incorporating an estimate of the agent critic function for sample efficiency. Empirical results across warehouse logistics, wind farm control, and multi-agent navigation show substantial performance and scalability improvements.

Overall, I believe this paper has a good theoretical background and has good methodological improvements in terms of practical performance, and I think the equations and algorithms are clear, and the numerical results look convincing. Therefore, I recommend accepting the paper.

### Strengths
I think introducing diffusion models for optimizing the environment and policy for a multi-agent reinforcement learning problem is a good strength. The authors also introduce 2 key insights, namely the projection operator in generating the environment, and also using the agent critic estimates while optimizing for the environment critic function.

### Weaknesses
I do not see a major weakness for this paper.

### Questions
Could you explain the projection operator's operation more clearly? It may be possible that for certain applications, it is challenging to compute a feasible configuration or identify the "closest" configuration based on the problem setup.

You mentioned PDM, and stated that PUG is more efficient because it does not require the solution to be feasible at all times. How does the projection operator specifically help here?

I understand the motivation behind introducing the agent critic estimate into the cost function for optimizing the environment critic. How does that compare to the shared experience actor-critic approaches for MARL?

I would also appreciate more insight into how the set \Theta is chosen for the samples. The authors mention that it's a FIFO queue, but is it possible to have a biased estimate if the agent critic is not sufficiently converged with respect to the environment?

### Soundness
3

### Presentation
3

### Contribution
3
