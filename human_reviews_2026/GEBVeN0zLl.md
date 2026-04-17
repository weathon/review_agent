# Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards

- Decision: Reject
- Scores: 0, 4, 4, 4

## Abstract
Aligning large language models to human preferences is inherently multidimensional, yet most pipelines collapse heterogeneous signals into a single optimizeable objective. We seek to answer what it would take to simultaneously align a model across various domains spanning those with: verifiable rewards (mathematical accuracy), non-verifiable subjective preferences (human values), and complex interactive scenarios (multi-turn AI tutoring dialogues). Such multi-objective reinforcement learning setups are often plagued by the individual objectives being at odds with each other, resulting in inefficient training and little user control during inference. We propose a unified framework that: (i) standardizes {process reward model} (PRM) training across both verifiable and non-verifiable settings to better supervise models' chain-of-thought reasoning; (ii) performs {multi-objective alignment} by training the LLM with our $\textbf{M}$ulti-$\textbf{A}$ction-$\textbf{H}$ead $\textbf{DPO}$ (MAH-DPO) and a vectorized reward where the dimensions of the vector correspond to the various objectives instead of a single scalar; and (iii) demonstrates how such a system provides fine-grained inference-time user control. Experiments across math reasoning, value alignment, and multi-turn dialogue show that our framework improves performance across multiple objectives simultaneously, while minimizing cross-objective trade-offs and enabling flexible inference time user control.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper’s style looks pretty strange and does not satisfy the official ICLR template. Maybe can desk reject it according to the submission policy (in standard format, I think it will exceed 9 pages).

### Strengths
None

### Weaknesses
None

### Questions
None

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a unified framework for multi-objective alignment of Large Language Models (LLMs) across both verifiable (e.g., mathematical accuracy) and non-verifiable (e.g., human values) rewards. The core contributions are threefold:

1. Standardized Process Reward Model (PRM) Training: A methodology for training PRMs across diverse domains. For verifiable domains, it uses hindsight credit assignment. For non-verifiable domains, it proposes three strategies based on process structure and rollout cost: majority voting, direct LLM-as-Judge labeling, and holistic reward approximation.

2. Multi-Action-Head DPO (MAH-DPO): An extension to DPO that uses a shared LLM backbone with separate, specialized output heads for each objective. This preserves a vectorized reward structure during training and enables flexible, inference-time control over objective trade-offs via head weighting.

3. PRM-Guided Decoding with Continuing Hidden State: An inference-time method that uses trained PRMs to guide step-wise generation while maintaining a continuous hidden state via key-value cache, avoiding the discontinuity issues of prompt re-encoding.

### Strengths
1. Originality: The fusion of a multi-head architecture with DPO to create MAH-DPO is a creative and original contribution. The unified PRM training framework also demonstrates significant conceptual synthesis.

2. Quality: The work is technically sound and executed with high quality. The experimental validation is extensive across multiple domains, and the methodology is described with sufficient detail for replication.

### Weaknesses
1. Lack of Theoretical Analysis: The paper lacks a theoretical justification for why MAH-DPO should lead to better multi-objective optimization. An analysis of the gradient dynamics (e.g., how the shared backbone balances conflicting signals from different heads) would strengthen the foundation of the method.

2. Insufficient Computational Analysis: While more efficient than training separate models, the computational overhead of MAH-DPO (training and inference) compared to a single-head DPO model is not quantified.

3. Heavy Reliance on PRM Quality: The entire framework's performance is contingent on the quality of the PRMs, which is particularly challenging in non-verifiable domains. The paper does not sufficiently address the potential failure modes or sensitivity to PRM inaccuracies.

### Questions
1. Theoretical Motivation: Could the authors provide more insight into the gradient dynamics of MAH-DPO? Specifically, how does the shared backbone's update (Eq. 9) theoretically prevent gradient interference or promote synergistic representations across objectives, rather than just being a weighted average?

2. Performance under Severe Conflict: How does MAH-DPO perform when the preference data for different objectives are in direct conflict? For instance, can you show results on a curated dataset where a response is preferred for "Helpfulness" but dispreferred for "Honesty"?

3. Computational Overhead: What is the concrete training and inference time/memory cost of MAH-DPO compared to a single-head DPO model with a similar total parameter count? Does the inference-time ensembling of heads introduce noticeable latency?

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
This paper proposes Simultaneous Multi-Objective Alignment combining verifiable (e.g., factual, task-based) and non-verifiable (e.g., subjective, human-value) objectives through a dual mechanism: (i) MAH-DPO (Multi-Attribute Head Direct Preference Optimization) for training multiple DPO heads with shared backbone parameters, and (ii) PRM-guided decoding, a reward-model–based inference scheme that detects step boundaries and reranks partial generations. The authors claim this architecture enables alignment across heterogeneous reward signals and achieves Pareto-optimal trade-offs between accuracy and subjective alignment quality. Experiments are conducted on math reasoning, question answering, and human-value benchmarks, showing moderate gains over single-objective DPO and GRPO variants.

### Strengths
1. Multi-objective alignment is crucial for combining factual correctness with value-driven behavior.
2. Separation between training (MAH-DPO) and inference (PRM-guided decoding) is conceptually clean and practically appealing.
3. Experiments span both verifiable and subjective tasks, demonstrating the generality of the framework.
4. The paper clearly situates itself within the DPO lineage and articulates the motivation for simultaneous alignment.

### Weaknesses
1. The “logit ensemble” and the equation `π_mix = Σ_i w_i π_i` are inconsistent, affecting calibration.
2. The relabeling formula `r̃_t = r_t + γ^(n−t) z` omits rollout-specific normalization and indexing.
3. The “vectorized reward” claim is overstated; implementation reduces to weighted scalar DPO.
4. The reference policy `π_ref` may drift if the backbone is shared, violating DPO assumptions.
5. Evaluation uses the same LLM family for labeling and judgment, creating circular bias.
6. Statistical tests and error bars are missing from all reported metrics.
7. The boundary detector `Q(·)` is undefined, limiting reproducibility.
8. The complexity advantage claim lacks empirical validation.

### Questions
1. Please clarify the exact relationship between the ensemble weights `w_i` and the mixture distribution `π_mix`.  Are the weights normalized per token or globally across heads? How does this affect calibration stability?

2. In the PRM relabeling step `r̃_t = r_t + γ^(n−t) z`,  what determines the rollout length `n`, and is there a normalization term to prevent scaling bias across episodes?

3. Could the authors explicitly define the objective aggregation `L_total = Σ_j λ_j L_DPO^(j)`? Are the `λ` coefficients fixed or dynamically adjusted based on gradient norms or validation metrics?

4. Under what settings of the weighting factor `λ` (e.g., normalization range or adaptive schedule) does the combined DPO objective remain approximately convex or submodular in practice?

5. Does the theoretical interpretation of MAH-DPO hold under the embedding spaces used in experiments? Specifically, how sensitive are the observed trade-offs to the backbone architecture or token-level normalization?

6. Could the authors provide reproducible runtime benchmarks — including GPU type, batch size, and wall-clock comparisons to substantiate the claimed efficiency advantage?

7. How was circularity controlled when both training and evaluation use LLM-as-Judge? Any cross-model validation or other validation would make the findings more credible.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- Proposes a unified framework that standardizes PRM training across verifiable and non-verifiable domains, combines Multi-Action-Head DPO with vectorized rewards, and adds PRM-guided decoding for controllable inference.
- Reports gains across math accuracy, human values, and tutoring engagement, with inference-time steering to navigate trade-offs.
- Highlights a practical rule: emphasize test-time PRM guidance for verifiable rewards and multi-head training for non-verifiable rewards.

### Strengths
- Unifies process supervision across objective types and connects it to controllable decoding.
- Empirical trends are consistent across domains with synergy between training-time and test-time methods.
- Practical inference control via head weighting enables user customization without retraining.

### Weaknesses
- Heavy reliance on LLM-as-judge can import bias and generate noisy labels that propagate.
- Limited theory and comparisons against strong scalarization or parameter-merging baselines; unclear how much MAH-DPO drives gains.
  - Weak theoretical justification for why multi-head with a shared backbone avoids negative transfer compared to alternatives.
- Scalability and compute/latency costs from multi-head policies plus guided decoding need fuller analysis.

### Questions
- How sensitive are outcomes to PRM label noise and can ensemble-of-judges or disagreement-aware training improve robustness?
- What is the delta over MODPO and linear scalarization when compute and data are held constant?
- Can the verifiability heuristic be formalized to choose training vs test-time alignment adaptively per task?
- How does the approach scale to 10+ objectives without negative transfer?

### Soundness
2

### Presentation
3

### Contribution
2
