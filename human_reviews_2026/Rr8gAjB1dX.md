# AdaNav: Adaptive Reasoning with Uncertainty for Vision-Language Navigation

- Avg Score: 5.33
- Decision: Reject
- Scores: 8, 4, 4

## Abstract
Vision-Language Navigation (VLN) requires agents to follow natural language instructions by grounding them in sequential visual observations over long horizons. Explicit reasoning could enhance temporal consistency and perception–action alignment, but reasoning at fixed steps often leads to suboptimal performance and unnecessary computation. To address this, we propose AdaNav, an uncertainty-based adaptive reasoning framework for VLN. At its core is the Uncertainty-Adaptive Reasoning Block (UAR), a lightweight plugin that dynamically triggers reasoning. We introduce Action Entropy as a policy prior for UAR and progressively refine it through a Heuristics-to-RL training method, enabling agents to learn difficulty-aware reasoning policies under the strict data limitations of embodied tasks. Results show that with only 6K training samples, AdaNav achieves substantial gains over closed-source models trained on million-scale data, improving success rate by 20% on R2R val-unseen, 11.7% on RxR-CE, and 11.4% in real-world scenes.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce AdaNav, a module that allows VLN agents to select a reasoning mode at a given step. The module takes as input the history of action entropies and current state. The output is a probability to invoke one of four reasoning modes: summary, error correction, description and "no reasoning". In contrast to previous work, this setup focuses reasoning to critical points in the trajectory in order to avoid context rot. The reasoning mode policy is initialized by a heuristic based on action entropy and slowly annealed towards a fully learned policy. The reward for the reasoning mode policy takes success rate, distance-to-goal and reasoning output length into account. The reasoning module can be dropped into existing agents and experiments show consistent improvements across benchmarks. Ablation experiments show that the approach is robust to hyperparameter and design choices like heuristic prior are crucial for training.

### Strengths
- well motivated approach
- important problem of selective reasoning in VLN
- good ablations

### Weaknesses
- presentation is a bit cramped but could be solved by additional page

### Questions
Did you consider outdoor VLN tasks? If not, why?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an uncertainty-driven framework that decides when and how to reason during VLN via a lightweight Uncertainty-Adaptive Reasoning (UAR) block and a Heuristic-to-RL training scheme. Action Entropy serves as an interpretable trigger signal, then reinforcement learning refines the policy.

### Strengths
1. Clear formulation that separates navigation and reasoning policies and factors the reasoning decision into “when/which mode” and “what to say,” which makes the mechanism easy to slot into existing VLN stacks. 
2. Sensible uncertainty signal backed by a diagnostic showing higher action entropy in failed trajectories, plus a compact control vector for mode selection. 
3. Practical Heuristic-to-RL schedule that starts from uncertainty priors and anneals toward a learned policy, avoiding costly reasoning supervision.

### Weaknesses
1. The approach hinges on action entropy. Alternatives such as value uncertainty, disagreement, or temporal variance are not compared head-to-head, even though the paper shows a correlation analysis for entropy. 

2. Reward shaping penalizes reasoning by length. This may incentivize terse but unhelpful traces or gaming of minimal length; failure-case analysis for this trade-off is limited. 

3. Additional backbones or continuous-control tasks would make the “plug-in” claim feel more comprehensive.

### Questions
1.What is the end-to-end latency budget on hardware during real-world runs and how much of it is attributable to UAR vs. text reasoning vs. base policy inference. A simple table with mean/95th percentile would help practitioners. 

2. When the entropy prior disagrees with the learned selector during annealing, does the system exhibit oscillation or premature triggering. Any guardrails beyond λ-scheduling.

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
4

### Summary
This paper propose AdaNav, an adaptive navigation inference framework based on uncertainty. Combining prior and heuristic reinforcement learning training mechanisms, it significantly outperforms multiple closed-source models in benchmarks such as R2R and RxR-CE, as well as in real-world scenarios, without relying on large-scale labeled data.

### Strengths
1.An adaptive inference framework is proposed that does not require large-scale inference labeled data, addressing the key questions of "when to infer and how to infer" in VLN.

2.Innovatively, action entropy is used as an uncertainty signal, combined with historical trends, to dynamically trigger inference.

3.This paper proposed a "Heuristic-to-RL" training mechanism, combining heuristic guidance and reinforcement learning for optimization.

### Weaknesses
1. The figures need improvement. Figure 1 mainly shows the differences from traditional methods, but the description of the prior and UAR Block is very limited. Specific figures or sub-figures describing the method could be added.

2. Regarding the effectiveness of the UAR Block and the necessity of reinforcement learning, there is a lack of comparison with adaptive inference baselines, such as inference based on fixed interval changes or triggering based on the simplest confidence level.

3. I am curious about the computational efficiency metrics, such as the average inference time and memory usage of this adaptive module. It is difficult to assess its feasibility in practical deployment.

4. I am curious about the selection of the weights $W_1$, $W_2$, and $W_3$ in Equation 6, as well as b. Is this learnable? If only action entropy has a significant impact? An ablation study could be conducted.

### Questions
1. In Figure 3, the action entropies for success and failure overlap considerably, especially at the beginning and end of the episode. I'm confused about the definition of action entropy as a good measure of uncertainty in the paper. How is this difficult-to-determine uncertainty handled?

2. Figure 4b shows a relatively high frequency of actual step-interval inference. Is adaptive uncertain inference necessary? Table 7(a) uses a fixed 5-step inference; is it possible that adaptive uncertain inference has a higher actual inference frequency? What if the ablation is a fixed 3-step, 1-step inference?

3. In Heuristic-to-RL training, how sensitive is the prior distribution setting? Are progress- and distance-based prior strategies better than action entropy?

4. Table 3 uses significantly different data for 8-frame and 64-frame models, but the model results are not significantly different. Is this because the model has a limited learnable context length?

5. For failure case analysis, under what circumstances did the UAR Block fail to trigger inference correctly?

### Soundness
3

### Presentation
2

### Contribution
3
