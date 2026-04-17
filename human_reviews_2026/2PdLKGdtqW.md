# From Gradient Volume to Shapley Fairness: Towards Fair Multi-Task Learning

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 8

## Abstract
Multi-task learning often suffers from gradient conflicts, leading to unfair optimization and degraded overall performance. To address this, we present SVFair, a Shapley value-based framework for fair gradient aggregation. We propose two scalable geometric conflict metrics: VolDet, a gram determinant volume metric, and VolDetPro, its sign-aware extension distinguishing antagonistic gradients. By integrating these metrics into Shapley value computation, SVFair quantifies each task’s deviation from the overall gradient and rebalances updates toward fairness. In parallel, our Shapley value computation admits controllable complexity. Extensive experiments show that SVFair achieves state-of-the-art results across diverse supervised and reinforcement learning benchmarks, and further improves existing methods when integrated as a fairness-enhancing module.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SVFair, a gradient-based multi-task learning (MTL) method that aims to balance training across tasks in a fairness-aware manner. The authors define two gradient conflict metrics: VolDet, the volume of the parallelotope formed by task gradients (computed via the Gram determinant), and VolDetPro, a signed extension that penalizes negative cosine relations between gradients. These metrics quantify the degree of conflict or synergy among tasks. Using these as importance weights, the paper constructs a single-pass Shapley value approximation that determines each task’s contribution to the joint update. The resulting algorithm adaptively reweights task gradients to seek a Pareto-stationary point near the center of the trade-off front, improving fairness across tasks. Experiments on standard MTL benchmarks show small but consistent improvements in balance metrics compared to simple averaging and prior multi-gradient methods.

### Strengths
Geometric intuition. The gradient volume concept (Gram determinant of task gradients) provides a tangible geometric measure of task conflict. It’s an appealing visualization tool and a natural extension of cosine-similarity-based conflict measures.

Unified perspective on fairness in MTL. By linking gradient aggregation to task-level “fairness,” the paper situates multi-task balance as a fairness problem, offering conceptual coherence with broader fairness research.

Implementation simplicity. SVFair requires only first-order gradient information and scales linearly in the number of tasks. It’s easy to implement atop existing MTL optimizers and could serve as a practical baseline.

Readable and self-contained. The paper is decently written, with useful algorithm boxes and schematic figures clarifying how VolDet and VolDetPro differ. It’s easy for readers familiar with gradient manipulation to reproduce.

### Weaknesses
Lack of originality and differentiation. The key ingredients (gradient alignment metrics and Shapley weighting) are not new. Prior works such as PCGrad (Yu et al., NeurIPS 2020), IMTL-G (Liu et al., 2021), GradVac (Wang et al., 2022), and ParetoMTL (Navon et al., 2022) already addressed gradient conflict and Pareto fairness using similar geometric or cooperative-game ideas. VolDet and VolDetPro are essentially new metrics, not fundamentally new algorithms. Suggestion: The paper should clearly articulate what these new metrics provide beyond previous gradient-conflict measures and why the Shapley approximation adds value.

Weak theoretical grounding. The claim that minimizing VolDetPro “drives the solution toward a front-center Pareto point” is heuristic, no theorem or guarantee is offered. Similarly, the Shapley-value approximation is described qualitatively but without convergence or unbiasedness analysis. Suggestion: Include theoretical propositions linking VolDet/VolDetPro to Pareto-stationarity or fairness guarantees.

Empirical evaluation is thin. Experiments are small-scale and mostly incremental. Improvements over baselines are minor (often <2%), and no statistical tests are reported. There are few ablations on the role of VolDet vs. VolDetPro or on Shapley-weight computation overhead. Suggestion: Expand experiments to more diverse MTL setups (e.g., computer vision or NLP) and provide detailed comparisons to ParetoMTL, PCGrad, MGDA, and IMTL-G.

“Fairness” interpretation feels overstated. While the method balances gradients, calling it “fairness-aware” is somewhat misleading; the fairness concept here refers only to task balance, not fairness in the social or demographic sense. Suggestion: Use more neutral terminology like “balanced multi-task optimization” to avoid confusion.

Unclear benefit of signed volume (VolDetPro). The proposed “signed” determinant variant is underexplained—why should penalizing negative cosine relations via determinant sign improve convergence? Empirical evidence is insufficient. Suggestion: Provide analytical insight or controlled experiments illustrating when VolDetPro outperforms VolDet.

Relation to prior Shapley gradient methods. Shapley-based gradient allocation (e.g., Ghorbani & Zou, 2019; Yoon et al., 2022) has already been applied to task weighting and feature attribution. The “single-pass approximation” presented here is not clearly compared to these prior formulations. Suggestion: Include a subsection detailing differences in computational complexity and approximation bias.

Positioning in literature. The related work section underplays existing geometric and cooperative-game perspectives on MTL optimization. Without a clearer comparison, readers may perceive SVFair as an incremental tweak rather than a conceptual advance.

### Questions
Can you theoretically connect VolDetPro minimization to convergence to a Pareto stationary solution?

How does your Shapley approximation differ in complexity or quality from prior Shapley-MTL approaches (e.g., Yoon et al. 2022)?

Did you test the sensitivity of the method to task scaling or normalization (since determinant magnitudes can vary drastically)?

Would your framework work for a large number of tasks (the Gram determinant computation scales poorly)?

Can you clarify whether “fairness” here has any formal definition (e.g., equal loss, equal gradient norm) or is purely heuristic?

Please provide statistical significance or confidence intervals for performance differences.

Would you consider integrating ParetoMTL’s convex combination layer with VolDet weighting to strengthen the theoretical link?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the common issue of gradient conflicts in multi-task learning, which can lead to unfair optimization among tasks. The authors propose SVFair, a Shapley value–based framework designed to quantify and mitigate such unfairness. Specifically, they introduce two geometric conflict metrics: (1) VolDet, a Gram-determinant-based volume metric that measures the geometric diversity of task gradients, and (2) VolDetPro, a sign-aware extension that distinguishes antagonistic gradients. By integrating these metrics into Shapley value computation, SVFair evaluates each task’s deviation from the aggregated gradient and rebalances the updates to improve fairness. Extensive experiments across both supervised and reinforcement learning benchmarks demonstrate that SVFair achieves state-of-the-art performance and can serve as a fairness-enhancing plug-in module for existing multi-task optimization methods.

### Strengths
1. The paper proposes a novel approach that combines Shapley-value-based fairness reasoning with gradient conflict measurement in multi-task learning. 

2. The method is well-motivated and theoretically solid. The analysis is detailed and convincing, providing clear justification for the proposed metrics and their role in achieving fair optimization.

3. The experimental evaluation is extensive, covering both supervised and reinforcement learning settings. Results demonstrate strong performance improvements and fairness benefits over several competitive baselines.

4. The authors have released code, which greatly improves the reproducibility and credibility of the work.

### Weaknesses
1. The core contribution of the proposed SVFair framework lies in computing Shapley values based on the newly defined VolDet and VolDetPro conflict metrics. However, once these weights are obtained, the subsequent problem formulation and solution procedure (Sec. 4.3) largely follow prior works such as FairGrad and PIVRG, and the corresponding convergence analysis also remains very similar. This overlap somewhat weakens the theoretical novelty of the paper.

2. Although the authors conducted an ablation study on the temperature coefficient $\tau$ in Table 4, they did not provide further theoretical analysis or practical guidance for tuning this parameter. In particular, the optimal $\tau$ varies across tasks and does not show a consistent trend (for example, in the Cityscapes experiment, performance first drops and then improves as $\tau$ increases). I believe the authors should include a more thorough investigation and discussion to clarify the role of $\tau$ and how it should be selected.

3. The overall presentation could be improved. For instance, the abstract is overly concise and does not sufficiently describe the problem background or the core motivation behind introducing Shapley-value-based fairness reasoning. A slightly more detailed introduction would help readers better understand the context and significance of the proposed approach.

### Questions
1. The computational cost of Shapley value estimation grows exponentially with the number of tasks, as also acknowledged by the authors. To address this, the paper employs Monte Carlo subset sampling to approximate the Shapley values. However, increasing the number of sampled subsets should, in principle, improve the estimation accuracy. Would this lead to better downstream performance (e.g., in terms of fairness or overall task accuracy)? I encourage the authors to include an ablation study on the number of sampled subsets to clarify the trade-off between computational cost and performance.

2. Referring to Weakness 2, while the authors conducted ablation experiments on the temperature coefficient $\tau$, they did not provide further theoretical insights or practical guidance on how to choose this parameter. Could the authors elaborate on how $\tau$ influences the optimization dynamics, and whether any general guidance or heuristic can be derived from the observed trends?

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
5

### Summary
This paper proposes two novel geometric gradient conflict measures, VolDet and VolDetPro, which quantify the squared volume of a subset of normalized task gradients. It then uses Shapley values to attribute each task's contribution to gradient conflict, and this information is incorporated into an existing fairness-based MTL optimization framework. Experiments on a suite of MTL benchmarks demonstrate the effectiveness of the proposed method.

### Strengths
1. The introduced VolDet and VolDetPro metrics are novel, and they can capture the geometric misalignment across task gradients.
2. The Shapley values further leverage the VolDet/VolDetPro and lead to the overall per-task weights quantifying contribution to the misalignment. This angular-driven measure is also novel.
3. Theoretical analysis and extensive empirical studies are provided, which can demonstrate the effectiveness of the proposed SVFair framework.

### Weaknesses
1. Line 240-241 states that higher $\phi_i$ values indicate task $i$ is more misaligned with others and should be assigned a lower influence. It may be a typo. It seems that such tasks should actually be assigned higher influence, as mentioned in Eq. (7) and Line 2093–2094.
2. The volume metric is built on the normalized task gradients. It ignores the magnitude information. Although the term $g_i^\top d$ in Eq. (7) considers magnitude information, it may lead to a suboptimal solution. For example, consider two approximately aligned task gradients, but with very different magnitudes. They have similar Shapley value weights, and Eq. (7) reduces to $\arg\min \sum_i \frac{1}{g_i^\top d}$, which corresponds to the minimum potential delay (MPD) fairness in FairGrad [1].  This MPD fairness criterion may not be suitable for this case, thus leading to suboptimal solutions. 
3. The calculation of the Shapley value introduces additional cost at each training step. Though Table 5 shows that the cost can be mitigated by Monte Carlo subset sampling, an ablation study is still needed to demonstrate the necessity of using Shapley value weights. The VolDet and Shapley value are two independent and separable concepts. It would be informative to compare them with other simpler weighting strategies, such as $\phi_i^\prime=\sum_{i, j\neq i} M_{ij}$, where a larger value indicates better alignment. You do not necessarily need to use this example; what I want to express is that it would be better to provide ablation studies to show that the Shapley value provides sufficient benefit compared to its additional cost.

### Questions
See the discussion in weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes SVFair, a Shapley value-based framework for fair multi-task learning (MTL) that systematically quantifies and mitigates gradient conflicts among tasks. The authors introduce two utility functions, named as VolDet and VolDetPro, to compute Shapley values in a single training pass. These values are then used to guide gradient aggregation, promoting balanced optimization across tasks. Extensive experiments on supervised and reinforcement learning benchmarks demonstrate that SVFair achieves state-of-the-art performance and improves existing MTL methods when integrated as a fairness module.

### Strengths
++ The paper integrates Shapley values into MTL optimization for quantifying gradient deviation and conflict at a subset level. The proposed VolDet and VolDetPro metrics offer ageometric perspective on gradient interactions.

++ The method is well-motivated and theoretically grounded, with convergence guarantees to Pareto stationary points under reasonable assumptions.

++ SVFair demonstrates strong empirical performance across diverse benchmarks (e.g., NYU-v2, CelebA, MT10) and can be easily integrated into existing MTL methods, enhancing their fairness and performance. The framework is scalable and supports Monte Carlo sampling for large-task settings.

### Weaknesses
-- Although the complexity is dominated by the training cost, the exact Shapley value computation can be prohibitive for very large N (number of tasks). While Monte Carlo sampling is proposed, its effectiveness and convergence properties are not thoroughly analyzed or empirically validated across different task scales.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
