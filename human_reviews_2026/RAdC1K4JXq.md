# Learning to Reason with Mixture of Tokens

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has become a leading approach for improving large language model (LLM) reasoning capabilities. Most current methods follow variants of Group Relative Policy Optimization, which samples multiple reasoning completions, scores them relative to each other, and adjusts the policy accordingly. However, these approaches invariably sample discrete tokens at each reasoning step, discarding the rich distributional information in the model's probability distribution over candidate tokens. While preserving and utilizing this distributional information has proven beneficial in non-RL settings, current RLVR methods seem to be unnecessarily constraining the reasoning search space by not using this information. To address this limitation, we investigate mixture-of-token generation (MoT-G) in RLVR. We present a unified framework that generalizes existing MoT-G approaches, including existing training-free methods that construct mixture embeddings as weighted sums over token embeddings, and extend RLVR to operate directly in this continuous mixture space for generating chain-of-thought. Evaluating two MoT-G variants on Reasoning-Gym, a suite of reasoning-intensive language tasks, we find that MoT-G methods achieve substantial improvements (5–35\% gains on 7/10 tasks) compared to standard decoding with the Qwen2.5-1.5B model, while reaching comparable accuracy with half the number of trajectories, suggesting improved training efficiency. Through comprehensive hidden-state and token-level analyses, we provide evidence that MoT-G's benefits may stem from its ability to maintain higher hidden-state entropy throughout the reasoning process and promote exploration in token space.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Mixture-of-Token Generation (MoT-G), a framework that extends Reinforcement Learning with Verifiable Rewards (RLVR) to operate in a continuous token space rather than discrete sampling. By aggregating multiple candidate tokens into mixture embeddings, MoT-G preserves uncertainty and promotes exploration during reasoning. Experiments on Reasoning-Gym show 5–35% accuracy gains on most tasks using the Qwen2.5-1.5B model, achieving comparable results with half the trajectories.

### Strengths
The proposed method itself is coherent and easy to follow.  The authors conduct experiments on several benchmarks to validate the effectiveness of the proposed method.

### Weaknesses
While MoT-G presents an effective way to extends RLVR to operate in a continuous token space, several limitations temper its overall contribution.

**(1) Model scale limitation:** all experiments are conducted primarily on the Qwen2.5-1.5B model, with brief extensions to 3B and 7B variants in the appendix. Such relatively small models may not fully capture the complex reasoning dynamics that large-scale RLVR systems (e.g., ≥30B parameters) exhibit. The scalability and stability of MoT-G in larger models remain unclear. 

**(2) Limited dataset coverage:** the evaluation focuses almost entirely on Reasoning-Gym, a procedurally generated benchmark. While diverse, it may not represent the full variety of real-world reasoning tasks such as mathematical proofs, code reasoning, or commonsense multi-hop inference. The absence of widely used public datasets (e.g., GSM8K, MATH, or ARC-Challenge) weakens the generalizability of results.

**(3) Computational complexity:** mixture-based generation introduces higher memory and compute costs per step, yet the paper provides no quantitative efficiency analysis beyond “half the trajectories.”

**(4) Performance regressions:** on deterministic tasks (e.g., self-reference, number sequence), MoT-G underperforms, indicating that the added uncertainty may harm precise reasoning rather than help exploration.

### Questions
1.	How sensitive is MoT-G’s performance to the choice of mixture size k and Dirichlet concentration parameters across different reasoning domains?
2.	Can the proposed method be combined with other exploration-enhancing techniques such as temperature scaling or entropy regularization?
3.	Does the higher hidden-state entropy correlate causally with reasoning correctness, or could it reflect noise?
4.	How would MoT-G perform when scaled to larger models (e.g., ≥14B) or used with curriculum or transfer learning setups?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces **Mixture-of-Token Generation (MoT-G)** for reinforcement learning with verifiable rewards (RLVR, e.g., GRPO). Instead of sampling a single token, the model aggregates multiple token embeddings into a continuous mixture to enhance exploration. Two variants—**Dirichlet weighting** and **Different Tokens sampling**—are evaluated on the Reasoning-Gym benchmark using Qwen2.5-1.5B. The approach achieves modest gains (5–35%) on half of the tasks and minimal improvements (≈1–2%) on math datasets. Analyses of hidden-state entropy and token diversity suggest that MoT-G improves exploration, though causal links remain unclear.

### Strengths
- Clear formulation of mixture-based generation under the RLVR framework.  
- Includes interpretability analyses (entropy, diversity, visualization).  
- Shows potential for more efficient exploration with fewer rollouts.  
- Presentation and figures are clean and understandable.

### Weaknesses
- **Low novelty:** Essentially applies prior soft-token or mixture-of-token ideas to RLVR without new objectives or theory.  
- **Limited experiments:** Only one base model (Qwen2.5-1.5B) and fixed GRPO setup; no scaling or robustness tests.  
- **Marginal gains:** Minimal improvement on core reasoning benchmarks (e.g., GSM8K, Math-500).  
- **Weak ablations:** Lacks analysis isolating the contribution of mixture components or sampling parameters.

### Questions
- How sensitive are the results to the number of sampled tokens (*k*) and the choice of aggregation (Dirichlet vs. uniform)?  
- Can the method generalize to larger models or other reasoning domains?

### Soundness
3

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
3

### Summary
This paper points out a limitation of current Reinforcement Learning with Verifiable Rewards (RLVR) methods, which rely on discrete token sampling and thus discard valuable distributional information, constraining the reasoning search space. Thus, authors propose a "Mixture-of-Token Generation" (MoT-G) framework where a weighted mixture of multiple token embeddings is generated at each reasoning step, allowing the model to maintain uncertainty and explore alternative paths. Experiments show that MoT-G achieves significant performance gains of 5-35% on 7 out of 10 tasks in the Reasoning-Gym benchmark. The paper's analysis suggests these benefits may stem from MoT-G's ability to maintain higher hidden-state entropy and promote token-space exploration during reasoning.

### Strengths
- This paper is easy to follow and has a reasonable motivation. Authors identifies the core limitations of current RLVR methods that rely on discrete token sampling which discardes distributional information and limiting the search space, and then proposes a logically clear improvement framework (named MoT-G).
- Clear experimental results. The experimental section of the paper is clearly designed, and through comprehensive testing on Reasoning-Gym, it strongly demonstrates that the MoT-G method achieved significant performance improvements of 5-35% on 7 out of 10 tasks.

### Weaknesses
- The specific computational overhead is unclear, even though the authors used fewer trajectories. The computational time and cost of MoT-G seem to be much higher than the standard method because it must aggregate $k$ tokens and maintain gradients for all of them during training, rather than just for a single token.
- The performance of MoT-G is not universally applicable. The method performs poorly on certain tasks. Moreover, why does the MoT-G method's performance degrade when increasing trajectories from 15 to 20, as shown in Table 4, contradicting the standard RL principle that more samples should improve the policy estimate and final accuracy?

### Questions
See Weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Mixture-of-Token Generation (MoT-G) for RL with verifiable rewards, replacing hard token commitments during chain-of-thought with continuous “mixture embeddings” formed by aggregating k sampled tokens per step. It unifies sampling (top-k, nucleus, min-p, k-sampling) and aggregation (probability-weighted, Dirichlet-weighted, etc.) and adapts GRPO to train in this space with single- vs. multi-token loss variants and approximate KL/log-prob handling. Two concrete instantiations, including Dirichlet (top-k + Dirichlet weights) and Different-Tokens (sample without replacement + normalized weights), are evaluated. On Reasoning-Gym with Qwen2.5-1.5B, MoT-G improves pass@1 by 5–35% on 7/10 tasks and often matches 10-chain accuracy using 5 chains; gains on GSM8K and MATH500 are modest. Analyses show higher hidden-state entropy and greater token-level diversity, suggesting better exploration; Dirichlet is more robust to higher decoding temperatures.

### Strengths
1. Strong motivation to preserve distributional information in RLVR

2. Two practical instantiations with intuitive exploration properties. Dirichlet weighting is a straightforward way to align exploration with model confidence while controlling variance.

3. Improvements on a diverse suite of reasoning tasks, including relational/inductive/cognitive tasks in Reasoning-Gym.

### Weaknesses
1. Baseline coverage is too narrow. Comparisons are primarily against single-token GRPO. Missing comparisons to strong RLVR variants (e.g., DAPO/GSPO/DR-GRPO updates), and to established test-time scaling methods or exploration enhancements. No end-to-end comparison against training-free soft-thinking at inference within the same RLVR training regime

2. Evaluation scope is limited, and the improvements are not consistent and significant. Main gains are on Reasoning-Gym; math benchmarks show small gains and two tasks (Number Sequence, Self-Reference) regress notably. Lack of harder, or widely adopted reasoning benchmarks (e.g., AIME24/25, OlympiadBench, BIG-bench subsets) under the RLVR setting limits external validity.

3. The paper highlights “half the trajectories,” but mixture steps increase per-step compute (k soft-embeddings, aggregation, KL approximations). I think the method should be more efficient. However, there is no wall-clock training time, FLOPs, or GPU-hours analysis to demonstrate these points.

4. The overall writing quality of the work can be improved. For example, in lines 128 and 129, the GRPO objective and notation contain inconsistencies, i.e., arg min vs. maximize expected return. The overall presentation is not good. It is a bit difficult for readers to directly understand and follow the work.

### Questions
1. Does the method harm the coherence of the final outputs? I noticed that there are some garbled example outputs. I also think it should be a drawback of "soft-thinking" methods.

2. How does k affect memory and speed? Any instability at larger k?

3. Clarify end criteria and whether the model learns to stop mixing vs. using a fixed heuristic. Any ablations on stopping criteria?

### Soundness
2

### Presentation
2

### Contribution
2
