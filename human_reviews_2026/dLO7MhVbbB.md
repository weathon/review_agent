# SafeDec: Constrained Decoding for Safe Autoregressive Generalist Robot Policies

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Recent advances in end-to-end, multi-task robot policies based on transformer models have demonstrated impressive generalization to real-world embodied AI tasks. Trained on vast datasets of simulated and real-world trajectories, these models map multimodal observations directly to action sequences for physical execution. Despite promising real-world capabilities, these models are still data-driven and, therefore, lack explicit notions of behavioral correctness. We address this gap by introducing \textbf{SafeDec}, a constrained decoding framework for autoregressive, transformer-based robot policies that enforces invariant safety specifications on candidate action trajectories. Task-specific safety rules are expressed as Signal Temporal Logic (STL) formulas and are enforced at inference time with minimal overhead. Our method ensures that generated actions provably satisfy STL specifications under assumed dynamics at runtime without retraining , while remaining agnostic of the underlying policy. 
We evaluate \textbf{SafeDec} on tasks from the CHORES benchmark for state-of-the-art generalist policies (e.g., SPOC, Flare, PoliFormer) across hundreds of procedurally generated environments and show that our decoding-time interventions are useful not only for filtering unsafe actions but also for conditional action generation. Videos are available at constrained-robot-fms.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes SafeDec, a constrained decoding framework for robot policies. The robot policies are modelled as a pre-trained multi-modal causal transformer. The proposed framework controls the transformer’s output at inference time to ensure that robot behaviors satisfy safety requirements specified in Signal Temporal Logic (STL). Under this framework, the authors propose two techniques termed Hard Constrained Decoding (HCD) and Robustness Constrained Decoding (RCD). Assuming the system dynamics are available, HCD casts the logits for actions resulting in future failures to “-inf” to rule out unsafe behaviors, whereas RCD re-weights the logits for actions based on their contribution towards the robustness score as well as their original logits to balance the trade-off between safe behaviors and achieving tasks. The authors conduct experiments on AI2-THOR indoor scenes using three pretrained robot policies (SPOC, PoliFormer, and Flare) across two STL requirements. Compared with the unrestricted transformer baseline and filtering-based baseline, the proposed framework achieved a better trade-off between safety and goal-directed behavior. Ablation studies show that SafeDec can also handle inaccurate dynamics to some extent and show how different balancing coefficients for RCD lead to different performance.

### Strengths
1. The problem this paper aims to tackle is indeed valuable. Many robot policies now use transformer backbones, and one concern is the lack of safety guarantees. An efficient solution to enforce task constraints is of utmost importance.
2. The proposed framework is easy to understand and appears flexible to different robot systems. Since it doesn’t require the system to be differentiable, the proposed framework can handle broader types of safety constraints and system dynamics.

### Weaknesses
1. The idea is a bit trivial, given that the “constrained decoding” idea for Transformers has already been discussed in [1], and the experimental findings are not significant (the filtering baseline also achieves relatively high performance).
2. The experiment section lacks essential baselines (consider gradient-based methods [2,3], “beam-search” methods and sampling-based approaches such as CEM or CMA-ES [4], since the authors assume the system dynamics are available). 
3. The proposed framework appears to work only on a discrete action space.
4. This framework is hard to generalize to STLs other than invariance properties.
5. The writing quality needs to be improved (misuse of citation types in L033-035; invalid citation in L084-085 and L301-303; wrong superscripts for “t’” used in L633-636; improper format for robustness score computation in L249-250 and L253-254, as the robustness score is on trajectories, not on states; L228-229 it should be $\hat{a}_{t+k-1}^{(i)}$) Some implementation details are missing (planning horizon T; the computation time, etc.)

References:
1. Willard, Brandon T., and Rémi Louf. "Efficient guided generation for large language models." arXiv preprint arXiv:2307.09702 (2023).
2. Leung, Karen, and Marco Pavone. "Semi-supervised trajectory-feedback controller synthesis for signal temporal logic specifications." 2022 American Control Conference (ACC). IEEE, 2022.
3. Zhong, Ziyuan, et al. "Guided conditional diffusion for controllable traffic simulation." arXiv preprint arXiv:2210.17366 (2022).
4. Kapoor, Parv, Anand Balakrishnan, and Jyotirmoy V. Deshmukh. "Model-based reinforcement learning from signal temporal logic specifications." arXiv preprint arXiv:2011.04950 (2020).

### Questions
1. I got confused for HCD introduced in L222-238. What is the k here, is it a hyperparameter defined for "look-ahead steps", or is it an index enumerated from 1 to T-t progressively? How do you get $x_{t+k-1}$ when you try to get $\hat{x}\_{t+k}$, I assume you only have $x_0,...x_t$ when you say "t is the current decision step"? Is the whole HCD process like, first from $x_t$, find possible actions $\hat{a}\_t$ that lead to STL-violation based on $\hat{x}\_{t+1}$, and then make these $z_t=-\inf$ so you never sample from them, and then sample from the rest buckets to get real $a_t$, and then use system dynamics to get $x_{t+1}$, then do the same thing to get real $a_{t+1}$ and then get $x_{t+2}$, ... till finding the final $x_{t+k}$? In this process, do you need to call the transformer forward pass multiple times? Do you need to update the transformer visual input? I guess the RCD procedure is somewhat similar. A pseudo-code algorithm or a simple example will be nice to illustrate the whole process. 
2. In L213-215 the authors mentioned "... we propose SafeDec: A constrained decoding strategy that integrates STL specifications into the foundation model action selection process itself, ensuring satisfaction without distorting the model’s distribution." I am a bit confused why SafeDec doesn't distort the model's distribution. It seems like HCD will filter out some actions, and RCD will also change the action distribution by reweighting.
3. The reason for saying "Hard to generalize to STLs other than invariance properties" is that for the invariance properties without explicit time intervals, one just needs to treat them as state constraints per timestep, which is exactly how the current HCD and RCD handle constraints right now. For this type of constraint, one does not even need to use the concept of robustness score to reweight the logits. But for more complex STLs (like "Eventually (Always ...)"), it is no longer Markovian, and the agent needs to use extra bits to keep track of the history (as in some RL works[1,2], but only for two-temporal-layer STLs.) And for general STLs, one needs to first decompose them into reachability and invariant properties [3,4], then decides how to activate the associated subtasks as time progresses. The proposed framework does not seem to adapt to these settings.

References
1. Aksaray, Derya, et al. "Q-learning for robust satisfaction of signal temporal logic specifications." 2016 IEEE 55th Conference on Decision and Control (CDC). IEEE, 2016.
2. Venkataraman, Harish, Derya Aksaray, and Peter Seiler. "Tractable reinforcement learning of signal temporal logic objectives." Learning for dynamics and control. PMLR, 2020.
3. Kapoor, Parv, Eunsuk Kang, and Rômulo Meira-Góes. "Safe planning through incremental decomposition of signal temporal logic specifications." NASA Formal Methods Symposium. Cham: Springer Nature Switzerland, 2024.
4. Liu, Ruijia, et al. "Zero-Shot Trajectory Planning for Signal Temporal Logic Tasks." arXiv preprint arXiv:2501.13457 (2025).

### Soundness
2

### Presentation
1

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
The authors present an approach for satisfying invariant properties in plans generated by transformer models. Their approach uses an output filter that leverages the underlying distribution from the model, and mask predicted tokens that violate the invariant specification based on a hard constraint or robustness-based weighting. The latter provides a trade-off between satisfaction of the constraints and satisfaction of the original task, whereas the hard constraint formulation will sacrifice performance to satisfy the invariant property.

### Strengths
The authors present a sound, reasoned approach to safety using transformer models. The paper is well-written and easy to follow. The authors present the work well in the context of existing approaches, and it represents a nice first step to this type of online safety.

### Weaknesses
This paper overstates the application of STL. Invariant properties are a (useful) but small subset of STL. I think the abstract and paper need to be much clearer about this. It is a worthwhile step forward, but it is a very limited fragment of STL. After the abstract, invariants are not mentioned until section 4. Reading the paper, it sounds as though the authors will satisfy general STL specifications, which their proposed approach is not able to do. It is very important to be clear about this, as it represents a very concrete limitation of the approach.

It would be useful to see a comparison of the results against something like SafeVLA. That model requires fine-tuning with safety constraints, as the authors note. The existing comparators show the cost of satisfaction in terms of success rate. However, SafeVLA or a similar model could help the reader understand how close SafeDec comes to an optimal safe execution. That is, if SafeVLA outperforms SafeDec in terms of task satisfaction, then it helps a reader identify the tradeoff between fine-tuning for safety compared to run-time safety constraints.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
SafeDec introduces a constrained decoding framework that enforces formal safety specifications during inference for large transformer-based robot policies such as SPOC, Flare, and PoliFormer. Instead of retraining these models to internalize safety rules, SafeDec intervenes only at decoding time to ensure that generated action sequences comply with Signal Temporal Logic (STL) constraints.

### Strengths
- **Model-agnostic generality without retraining.** The framework operates at test time and is agnostic to the underlying policy; it is demonstrated on SPOC, Flare, and PoliFormer without any additional training or fine-tuning.
- **Clear empirical trade-offs between variants.** The simulation study contrasts HCD (strict safety satisfaction) and RCD (safety–performance trade-off), making variant-specific strengths and weaknesses evident.
- **Clear writing and easy to understand.**

### Weaknesses
- **Scope mismatch (navigation vs. manipulation).** While the introduction references both navigation and manipulation, all experiments are limited to navigation tasks in AI2-THOR, creating a gap between the paper’s claimed and demonstrated scope.
- **Problem framing conflated with solution design.** The claim of “introducing the novel problem of constrained decoding for transformer-based policies under STL specifications” blends the problem statement with specific solution choices (transformers and STL). The work would be more precise if it framed the broader problem as *online safety enforcement during inference* and justified these components as its methodological instantiation.
- **Limited evaluation.** Although the motivation centers on real-world safety, results are confined to a single simulated domain with two constraint types (geofencing and avoidance). This limits the generality of the conclusions.
- **No comparison with relevant baselines** (see questions)

### Questions
- **Ambiguity around “post-hoc manipulation.”** The term is used to critique existing methods but is not clearly defined—whether referring to post-training safety alignment or post-generation filtering at inference. Moreover, no empirical evidence from robotic settings supports the assertion that post-hoc interventions necessarily lead to “degenerate or brittle behaviors.”
- **Deviation from the base model’s distribution.** The method seeks to preserve the policy’s original action preferences while ensuring safety, yet both HCD (masking unsafe actions) and RCD (reweighting logits) modify the underlying probability distribution. This change conflicts with the stated goal of maintaining distributional faithfulness to the base model (lines 195--199).
- **Limited expressivity of tested constraints.** The evaluated constraints are simple invariants (“always avoid”). Demonstrating performance on temporally scoped constraints, such as “avoid room A between time 1--time 2” or “avoid A until visiting B”, would better motivate the need for STL’s representation.
- **Omission of baselines.** The paper only compares with bound-to-fail baselines and would be strengthened by comparisons against (a) multi-sample rollout selection (choose the trajectory satisfying safety), (b) corrective optimization of predicted actions to resolve violations, (c) described “posthoc manipulation” methods, and (d) recent guided-sampling approaches (e.g., DynaGuide).
- **Writing and presentation issues.** The reference style is occasionally inconsistent, and acronyms such as HCD and RCD should be expanded upon first mention in the abstract or introduction for clarity.
- **Missing experimental and implementation details.**  All rollouts were feasible under the specified constraints, including how infeasible cases were treated, and key implementation details such as action vocabulary, decoding horizon, sampling strategy, and mapping from STL predicates to action tokens. Including these would improve clarity and reproducibility.
- **Missing discussion on failure modes.**

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents SafeDec, a constrained decoding framework that enforces formal safety specifications in Signal Temporal Logic (STL) within transformer-based robot navigation policies at inference time. The method is ​​an inference-time technique that reweights or masks candidate actions using STL satisfaction scores, i.e., decoding either through hard masking (HCD) or robust reweighting (RCD) of logits, ensuring that selected actions do not violate safety constraints under a simple dynamics model. SafeDec is evaluated on AI2THOR environments using three generalist navigation policies (SPOC, Flare, PoliFormer), demonstrating substantial improvements in STL satisfaction with limited impact on task success. The paper also compares performance when dynamics are noisy and the effect of weighting $\beta$ between robustness and base logits in RCD.

### Strengths
- SafeDec adapts constrained decoding from natural language processing to low-level action generation in robotic policies.
- The evaluation convincingly shows that SafeDec enforces simple invariants (avoidance, geofencing) with a modest performance loss across different state-of-the-art robot navigation policies.
- The paper demonstrates adaptation to small noise in dynamics and performance changes in the relative weighting of the safety–performance trade-off.

### Weaknesses
- The paper doesn’t compare performance to other similar methods mentioned (SafeVLA and SELP); instead, it is limited to simple filtering baselines.
- Implementation details are unclear, such as the setup of Simplex default actions and the decoding interface modification of the generalist policies. This affects the reproducibility. 
- The paper has limited STL specification diversity, with only the Always operator (invariant) for “always avoid” and “stay within bounds” conditions being tested.
- The writing has stray citations (line 084, page 3, and lines 302-302, page 6 )and needs to be proofread once again. The explanation/caption of Figure 1 can be expanded to detail the architecture, the robot's initial and final positions, and the graph if the authors choose to place it at the beginning of the problem formulation section. Trajectory visualizations (Figure 2) have low contrast compared to the background and lack clear legends.

### Questions
1. How are actions represented in SPOC, Flare, and PoliFormer? Can you elaborate on how you access and manipulate the last-layer logits of these policies?
2. Can the paper include comparisons to other baselines, such as SafeVLA or SELP? Both explicitly address safety constraints in transformer-based robotics or planning.
3. Can the authors elaborate more on the mechanism of the default actions in the Simplex/Filtering baseline and how they were determined?
4. Can SafeDec experiments demonstrate performance with other STL specifications that use the eventually operator or until operator? Or larger compound specifications?
5. Have you tried experiments where the error in dynamic modeling was large due to unmodeled effects?

### Soundness
2

### Presentation
2

### Contribution
2
