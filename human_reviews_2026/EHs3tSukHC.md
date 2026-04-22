# Off-Policy Safe Reinforcement Learning with Cost-Constrained Optimistic Exploration

- Avg Score: 3.33
- Decision: Accept (Poster)
- Scores: 0, 6, 4

## Abstract
When safety is formulated as a limit of cumulative cost, safe reinforcement learning (RL) aims to learn policies that maximize return subject to the cost constraint in data collection and deployment. Off-policy safe RL methods, although offering high sample efficiency, suffer from constraint violations due to cost-agnostic exploration and estimation bias in cumulative cost. To address this issue, we propose Constrained Optimistic eXploration Q-learning (COX-Q), an off-policy safe RL algorithm that integrates cost-bounded online exploration and conservative offline distributional value learning. First, we introduce a novel cost-constrained optimistic exploration strategy that resolves gradient conflicts between reward and cost in the action space and adaptively adjusts the trust region to control the training cost. Second, we adopt truncated quantile critics to stabilize the cost value learning. Quantile critics also quantify epistemic uncertainty to guide exploration. Experiments on safe velocity, safe navigation, and autonomous driving tasks demonstrate that COX-Q achieves high sample efficiency, competitive test safety performance, and controlled data collection cost. The results highlight COX-Q as a promising RL method for safety-critical applications.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper introduces an off-policy primal–dual safe reinforcement learning (RL) algorithm named COX-Q, aiming to improve sample efficiency and reduce estimation bias. The method integrates a cost-constrained optimistic exploration strategy to adjust the trust region for safe exploration. Additionally, the authors propose to mitigate estimation bias through the use of truncated quantile critics.

### Strengths
The empirical evaluations are comprehensive, with three applications.

### Weaknesses
1. The presentation of the theoretical results is weak. The paper does not clearly state the assumptions underlying the proposed method, leaving it unclear under what conditions the approach is valid or applicable. The policy’s action distribution is simply assumed to be Gaussian without justification. Moreover, Lemma 1 and Lemma 2 are rather trivial and do not appear to offer substantial theoretical contribution. If these are presented as lemmas, one would expect a main theorem or stronger result to follow—otherwise, the theoretical section lacks depth.  
2. Even if the lemmas are accepted, the authors fail to explain how these results contribute to improving sample efficiency or mitigating estimation bias. While the empirical evaluations suggest some performance gains, it remains unclear under what circumstances the proposed method is effective. The framework does not appear to generalize well.  
3. Although the experimental coverage is broad, the presentation quality is poor. For instance, no quantitative results are provided to support the claim of reduced estimation bias. In addition, the explanations are poorly organized and often lack logical coherence, which makes the empirical section difficult to follow.
4. There is no pseudo code for clarification and implementation.

### Questions
Given the major conceptual and presentation issues identified above, I do not have specific technical questions for the authors. The paper requires substantial clarification and restructuring.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes COX-Q, an off-policy safe reinforcement learning algorithm that achieves efficient and safety-aware exploration by integrating optimism, gradient conflict resolution, and uncertainty estimation. Building upon the optimistic actor-critic framework, COX-Q introduces a cost-constrained optimistic exploration strategy that balances reward maximization with constraint satisfaction. It employs a Policy-MGDA mechanism to align reward and cost gradients, ensuring exploration occurs only in directions that improve performance without increasing risk, and an adaptive step-length controller to prevent constraint violations during updates. Additionally, COX-Q leverages Truncated Quantile Critics (TQC) to obtain uncertainty-aware and bias-reduced value estimates.

### Strengths
1. The paper tackles an important and timely problem by addressing the challenges associated with implementing off-policy reinforcement learning algorithms in the context of safe RL, where maintaining safety during exploration remains a key concern.

2. The proposed framework integrates gradient conflict resolution with mechanisms to mitigate value estimation biases, effectively addressing both reward overestimation and cost underestimation—two critical issues that commonly affect off-policy safe RL methods.

3. The paper presents a comprehensive empirical evaluation across diverse domains, including locomotion, navigation, and autonomous driving tasks, and benchmarks the proposed method against strong and relevant baselines.

### Weaknesses
[W1] Relevant Comparisons are Missing

The concept of gradient manipulation or resolution is not entirely new in Safe RL, as several works [1–3] have proposed similar approaches. To convincingly demonstrate the advantages of the gradient resolution method presented in this paper, it is essential to compare against these baselines. While some of these methods report results in the context of on-policy RL, their underlying ideas can be readily adapted to off-policy settings.

Recent work [4] addresses the problem of cost underestimation, which is also a focus of our method. Therefore, it represents an important baseline for evaluating the effectiveness of our approach in mitigating cost underestimation.

[W2] Lack of Ablations

The proposed method comprises two key components: (i) cost-constrained optimistic exploration via Gradient Resolution, and (ii) mitigation of value estimation bias via Truncated Quantile Critics (TQC). The paper would benefit significantly from ablation studies that quantify the contribution of each component individually. Specifically, it would be valuable to demonstrate:

1. the performance of gradient resolution alone compared to prior works [1–3], and

2. the effect of addressing value estimation bias via TQC compared to [4].

Such ablations would clearly establish the incremental value of each component and provide a stronger empirical justification for the proposed approach.

References
[1] Gu et al., Balance Reward and Safety Optimization for Safe Reinforcement Learning: A Perspective of Gradient Manipulation (2024)
[2] Chow et al., Safe Policy Learning for Continuous Control (2020)
[3] Liu et al., Constrained Variational Policy Optimization for Safe Reinforcement Learning (2022)
[4] Gao et al., Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration (2025)

### Questions
My questions pertain to the weaknesses highlighted above:

1. How does the proposed gradient resolution method for optimistic exploration differ from existing approaches in the literature?

2. In the context of safe exploration, preventing constraint violations during early learning is critical. While overestimation of cost can promote safe exploration, it may also lead to overly conservative behavior. How does the proposed method better balance safety and exploration compared to existing baselines such as MICE [1]?

Reference
[1] Gao et al., Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration (2025)

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
2

### Summary
The paper tackles a key gap in off-policy safe RL: optimistic exploration improves sample efficiency but often violates cost constraints, because reward and cost gradients point in different directions and cost critics are biased. The authors propose a primal-dual off-policy algorithm that performs cost-constrained optimistic exploration, and adaptively shrinks the exploration step.

### Strengths
1. The paper clearly identifies the off-policy safety pain point, i.e., optimism without cost control, and a concrete mechanism of policy-MGDA + cost-aware step-length selection to make OAC-style exploration safe.
2. Both key steps are derived with closed-form solutions. The SMARTS results, though single-seeded, are non-trivial and show the method still works with large networks.

### Weaknesses
1. Safety still depends on cost-critic accuracy. The whole cost-bounded exploration, c.f. Sec. 4.2., assumes the lower/mean cost estimates are reliable. In sparse-cost navigation the paper shows underestimation, and COX-Q behaves like ORAC. A discussion for robustifying the cost critic is missing. I am curious how the authors would further improve this.
2. The lemmas give local, per-step solutions, but there is no theorem that the whole off-policy procedure respects the CMDP constraint under function approximation and replay. However the abstract emphasizes controlled data-collection cost. I believe this gap should be made explicit.
3. Some baselines are slightly weakened. CAL is run with UTD=1 but CAL's strength is precisely high UTD.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
