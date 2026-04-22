# Mitigating Reward Hacking in Multi-Reward Optimization for Text-to-Image Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Text-to-image generation models have achieved remarkable progress in preference optimization, yet achieving robust alignment across diverse reward models remains a significant challenge. Existing multi-reward fusion approaches rely on weighted summation, which is costly to tune and insufficient for balancing conflicting objectives. More critically, optimization with reward models is highly susceptible to reward hacking. We theoretically suggest that using a unified global upper bound as the optimization target may induce reward hacking in certain samples. In addition, optimization with weak reward models is particularly prone to exacerbating this risk. To address this issue, we propose a Pareto frontier-guided optimal transport framework, which constructs a frontier for each prompt as the optimization target and maps generated samples within the same batch to their corresponding frontiers. Based on the characteristics of the reward models, we further design both online and offline optimization strategies to adapt to the distinct requirements of different reward models. Finally, we introduce the Joint Domination Rate (JDR) and Joint Collapse Rate (JCR) as more principled metrics for evaluating multi-reward optimization. Experimental results demonstrate that, compared with strong baselines, our method achieves a 10\% performance improvement, effectively mitigating reward hacking while enhancing multi-reward alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates reward hacking in multi-reward optimization for text-to-image models and argues that a single global bound across prompts can induce hacking because per-prompt reward ceilings are heterogeneous. It formalizes this observation with a property and a proposition, distinguishes strong and weak rewards, and proposes a Pareto-frontier-guided optimal transport framework. The framework has an offline stage that extracts prompt-wise Pareto fronts and uses an agent to prune weak rewards, followed by an online stage that moves samples toward dominating points on the fronts. The paper introduces Joint Domination Rate (JDR) and Joint Collapse Rate (JCR) to measure joint gains and collapses across rewards. Experiments on Parti-Prompts and a user study on DiffusionDB with SD3.5-Turbo plus LoRA and DRaFT-K report consistent improvements.

### Strengths
The paper presents a clear causal story from heterogeneous reward ceilings to reward hacking and supports it with formal definitions that motivate per-prompt targets. The Pareto-frontier plus optimal transport formulation is concrete and implementable, and the split into offline frontier extraction, agent-mediated pruning of weak rewards, and online frontier exploration is a practical recipe. JDR and JCR focus on per-sample, all-dimension consistency and align with the core thesis. The experiments cover single-reward, weighted multi-reward, reward-soup, and heterogeneous-bound baselines and the results show consistent gains with informative frontier visualizations.

### Weaknesses
1. **Positioning and ablation of the OT step:** The approach is conceptually related to Pareto-style multi-reward methods, but the paper does not isolate what the optimal transport mapping adds beyond non-dominated sorting or a simpler projection to the frontier. A small controlled study under matched base model, data, and token budget that compares a weighted sum, separate constraints, non-dominated sorting without OT, and the proposed OT loss would clarify the algorithmic delta and strengthen the claim of necessity.

2. **Operationalization of weak rewards and reliance on a proprietary agent:** The definitions of strong and weak rewards are motivated qualitatively, yet the paper does not report empirical correlations with human labels, threshold choices, or selection diagnostics. The pruning relies on a proprietary GPT-4o agent, which raises reproducibility concerns. Reporting reward–human correlations on a held-out labeled set, the decision thresholds with error analysis, and a simple non-proprietary fallback heuristic would make the procedure more transparent and reproducible.

### Questions
1. **Positioning and ablation of the OT step:** Run one minimal controlled replacement of the OT mapping with (i) non-dominated sorting + projection and (ii) a weighted-sum objective, under the same base model/data/token budget, and briefly explain what gradient signal OT introduces that the others do not? If adding runs is infeasible, a compact 2-D toy frontier with analytic gradients would still clarify when OT materially changes the solution.

2. **Weak-reward detection and reliance on a proprietary agent:** I highly recommend the authors to share a small diagnostic (e.g., ~200 held-out prompts) reporting each reward’s correlation with human labels and the pruning threshold you used, plus an ablation comparing “no pruning” vs “GPT-4o pruning” vs a simple open fallback (e.g., z-score filter + monotonicity checks)? Even approximate numbers would help assess false-pruning risk and reproducibility without proprietary tools.

### Soundness
3

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
3

### Summary
The paper introduces an innovative approach to optimizing multi-reward models for text-to-image generation, employing Pareto frontiers and optimal transport theory to guide solutions towards Pareto-optimality, significantly enhancing model performance. It features a GPT-4o decision agent for adaptive multi-reward model training, identifying and managing weak reward models prone to reward hacking. Enhanced metrics JDR and JCR are introduced for reliable multi-reward optimization assessment. Experimental results demonstrate superior efficacy in improving text-to-image models and avoiding reward manipulation compared to strong baselines.

### Strengths
1.	The paper offers a compelling insight of reward hacking: the conflict between a "unified global optimization target" and the "heterogeneity of rewards" as the core issue is a compelling insight.
2.	This paper introduces a novel method using Pareto frontiers and optimal transport theory to optimize multi-reward models in text-to-image generation tasks. This approach potentially mitigate the reward hacking issue by guiding the optimization towards Pareto-optimal solutions.
3.	As performance metrics specifically designed for multi-reward optimization, the Joint Dominance Ratio (JDR) and Joint Collapse Ratio (JCR) provide a more precise way to evaluate the effectiveness of different training strategies.

### Weaknesses
1.	The proposed method's complexity involving: 1) pre-computing a candidate set of samples, 2) extracting the Pareto frontier, 3) solving an optimal transport problem, and 4) employing a GPT-4o agent for monitoring. This introduces significant computational overhead and implementation challenges.
2.	The paper does not thoroughly examine whether the proposed method generalizes to other text-to-image models or scales with an increased number of reward models. It is not clear how this method will perform in more complex scenarios.

### Questions
1.	Could the authors elaborate on the computational cost of the proposed method? Specifically, how much does it increase training time and GPU resource consumption compared to the baselines? Is there a possibility for a "lightweight" implementation that could significantly reduce complexity and cost at the expense of some performance?
2.	While the proposed method has shown excellent results with four reward models. How do the authors foresee its performance when using a much larger set of reward models? At that scale, would the computation of the Pareto frontier and the solving of the optimal transport problem become bottlenecks?
3.	The classification of reward models is vital to the proposed strategy. An expert-driven decision requiring significant prior knowledge, could the authors provide more details regarding its robustness?

### Soundness
3

### Presentation
4

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
This paper tackles the problem of reward hacking in multi-reward optimization for T2I models. The authors identify a fundamental flaw in existing methods: they use a unified global optimization target (a single upper bound for rewards), which fails to account for the fact that different text prompts have heterogeneous reward landscapes and achievable upper bounds.

### Strengths
Pros:
1. The identification of "heterogeneous reward bounds vs. a unified global target" as the root cause of reward hacking is an interesting insight. It provides a theoretical foundation for the proposed solution.
2. The use of Pareto frontiers and Optimal Transport is a natural and mathematically sound approach for multi-objective optimization. It directly addresses the core problem of prompt-specific targets in a structured way.
3. The evaluation is comprehensive. It includes a strong set of baselines, robust quantitative analysis using both standard and their new metrics, compelling qualitative examples, a human user study, and a detailed ablation study that validates each component of their framework.

### Weaknesses
Cons:
1. The use of GPT-4o as a core component in the training loop is a potential weakness: 
    1. It makes the method less reproducible and accessible, as it relies on a costly, black-box external API whose behavior could change over time. 
    2. The feasibility of this approach for dozens or hundreds of rewards is questionable. 
    3. The paper would be strengthened by exploring or discussing simpler, fully automatable heuristics for detecting reward collapse (e.g., based on reward score variance, KL divergence from a reference distribution, or simpler image artifact detectors) as alternatives to the LLM agent.
    4. Have you explored replacing the GPT-4o agent with a more traditional, programmatic method for detecting reward collapse? For instance, could you monitor the distribution of reward scores and flag a model for pruning if its variance spikes or its mean diverges too quickly from a moving average?
2. The formal definitions rely on the correlation with an unobserved "human-perceived quality Q". In practice, the classification seems to be done post-hoc by the GPT-4o agent observing collapse during training. The link between the formal definition and the practical implementation could be tighter. An a priori method for estimating reward model strength would make the framework more efficient.
3. The paper mentions using the Sinkhorn algorithm but does not discuss the choice of hyperparameters (e.g., entropy regularization strength ε) or the sensitivity of the results to these choices.
4. Could you provide details on the scale of the offline precomputation? Specifically, how many samples per prompt (M in the paper) were needed to form a robust initial Pareto frontier, and how much did this contribute to the overall training time?
5.  In the online phase, how are new points added to the Pareto frontier? Is the entire frontier for a prompt re-calculated at each step using all historical and newly generated samples, or is there a more efficient updating mechanism?
6. How does the method perform on prompts that are stylistically or semantically very different from those used to build the initial offline Pareto frontiers? Does the online phase sufficiently adapt to out-of-distribution prompts?
7. The OT framework maps dominated samples to the frontier. How does the framework handle samples that are already very good (i.e., close to or on the frontier)? Is the loss signal for these samples near zero, effectively leaving them unchanged?

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses reward hacking in multi-reward optimization for text-to-image generation. It proposes a Pareto frontier-guided optimal transport framework with prompt-specific frontiers and online/offline strategies tailored to reward model strengths. Two new metrics, Joint Domination Rate (JDR) and Joint Collapse Rate (JCR), are introduced for better evaluation. Experimental results show a 10% performance improvement over baselines, effectively mitigating reward hacking while enhancing multi-reward alignment.

### Strengths
1. This paper is well written and easy to follow
2. This article provides sufficient theoretical analysis of reward hacking and has a clear standpoint.

### Weaknesses
1. The Pareto optimal problem is used to address situations where there are conflicts between rewards. However, in the paper, there does not appear to be a clear conflict between the two types of rewards selected. Empirically, training with text-image alignment reward tends to lead to an improvement in human preference reward, and vice versa. It is suggested that the authors conduct experiments to investigate whether the proposed method remains effective when conflicts between rewards exist. For example, as mentioned in the paper, rewards related to authenticity and artistic style may conflict.

2. Is this method applicable to any arbitrary text-to-image post-training approach? For instance, methods like ReFL, FlowGRPO, etc. Can the authors provide experiments to validate and illustrate this?

### Questions
Please see the questions in the Weaknesses. I hope these issues can be resolved, and I will reconsider my grading.

### Soundness
3

### Presentation
3

### Contribution
3
