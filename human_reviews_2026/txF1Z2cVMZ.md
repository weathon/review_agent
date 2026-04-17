# HiPO: Hybrid Policy Optimization for Dynamic Reasoning in LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 4, 4

## Abstract
Large Language Models (LLMs) increasingly rely on chain-of-thought (CoT) reasoning to improve accuracy on complex tasks. However, always generating lengthy reasoning traces is inefficient, leading to excessive token usage and higher inference costs. This paper introduces the Hybrid Policy Optimization (i.e., HiPO), a framework for adaptive reasoning control that enables LLMs to selectively decide when to engage in detailed reasoning (think-on) and when to respond directly (think-off). We construct a cross-domain, logically rich dataset using a hybrid multi-agent construction pipeline that provides explicit supervision for reasoning-mode selection. Then, building on this data, we introduce a hybrid reinforcement learning (RL) reward system that integrates mode-specific rewards with global bonuses to align reasoning quality with efficiency. Experiments across mathematics, coding, and general knowledge benchmarks demonstrate that HiPO can substantially reduce token length while maintaining or improving accuracy. Further analysis shows that HiPO learns fine-grained, context-sensitive reasoning behavior, activating CoT primarily on reasoning-intensive tasks and suppressing it when unnecessary.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose Hybrid Policy Optimization, which could adaptively choose to use or not to use the “thinking mode” during reasoning. Specifically, HiPO first builds hybrid training data containing both Think-on and Think-off samples, with DeepSeek-V3 generated explicit explanations to justify the mode selections. Next, the Hybrid reinforcement learning reward system begins to learn via the Judge and Answer segments, with two advantages considering both mode-level and instance-level benefits. The experimental results on Qwen series have verified the effectiveness and efficiency of the proposed HiPO.

### Strengths
-	This work introduces a straightforward and effective pipeline for building an adaptive reasoning model. The experiments show satisfactory results.
-	The authors have conducted extensive analyses on the detailed model settings in Section 4.3 and 4.4, which makes the conclusion more solid.
-	Overall, the writing is clear.

### Weaknesses
-	In the Related work part, the authors have claimed that “Despite progress, …, limited adaptation to hard cases due to monotonic shortening”. However, the existing adaptive reasoning methods (at least the adaptive baselines used in experiments) should be discussed to highlight the differences, technical novelty, and advantages of HiPO. It is not that clear which technical part is novel in this work.
-	In experiments, is the cold-start stage of HiPO the same as those of baselines (GRPO, AdaptThink, and AutoThink)? Did AdaptThink and AutoThink adopt the exact same data as HiPO’s data in both stages? These questions come from the weird results: in most cases, AdaptThink and AutoThink perform worse than the Cold-start (on or all) baselines. Fair comparisons are required.
-	HiPO has an additional segment of “Judge_analysis” that may bring in new costs for both Think-on and Think-off modes. Does the “Length” metric in Table 2 indicate the average total inference length (including all judge/think/answer/other segments)?
-	The experiments in Table 4 should include other adaptive baselines to provide a more convincing proof.
-	Typo: Page 7, Table 5 -> Figure 5.

### Questions
-	Will the data used in this work be open-sourced?
-	In Section 3.1.2, how to set the predefined threshold for different tasks with various hardness levels?

### Soundness
3

### Presentation
2

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
This paper introduces HiPO (Hybrid Policy Optimization), a training framework that enables Large Language Models to dynamically decide when to perform explicit reasoning (Think-on) or to directly answer (Think-off). The method combines a hybrid data pipeline that pairs concise and detailed responses with mode-justification annotations, and a hybrid RL reward balancing accuracy, token efficiency, and mode-selection quality. Experiments on math and code benchmarks show that HiPO reduces average output length while improving accuracy, effectively mitigating overthinking without hurting performance.

### Strengths
1. This work proposes a hybrid RL framework that learns when to reason (Think-on/off) to substantially reduce token length while maintaining or improving accuracy.

2. Extensive experiments on math/code benchmarks show consistent token reduction and accuracy gains.

3. Problem formulation, data pipeline, and two-stage training are described precisely.

### Weaknesses
1. Auto-generated judge quality unchecked: 

   The pipeline uses DeepSeek-V3 to produce “why this mode” justifications that directly enter the RL loss. No ablation measures how often these rationales are wrong or self-contradictory; noisy judge labels could mislead policy updates and inflate gains.

2. Binary gating only:
 
   Think-on/off is a single-bit decision, so the model cannot partially unroll a chain or perform staged reasoning (plan → verify → summarize). The paper does not show how to extend HiPO to finer-grained or depth-adaptive gating, limiting utility on problems of medium complexity.

3. Statistical significance unclear:
 
   Main results come from one rollout on small benchmarks (e.g. AIME 24/25) without multiple runs. Without detailed testing procedures in the main text, it is hard to tell whether the observed improvements are real or just random fluctuation.

### Questions
1. What is the "reasoning mode-difficulty" relationship (correlation between reasoning mode with performance) after HiPO training?

2. Have you considered about how to extend HiPO to finer-grained or depth-adaptive gating? If so, what architectural or reward changes are needed, and do results on medium-difficulty tasks show further token savings without accuracy loss?

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
4

### Summary
The paper proposes HiPO, a framework for adaptive reasoning in large language models (LLMs) that aims to dynamically decide when to engage in detailed chain-of-thought reasoning ("Think-on") and when to respond directly ("Think-off"). It introduces (1) a hybrid data construction pipeline generating paired Think-on/Think-off examples with mode justifications, and (2) a hybrid reinforcement learning (RL) reward system combining accuracy and efficiency rewards. Experiments on mathematical and coding benchmarks suggest HiPO reduces token usage while maintaining accuracy

### Strengths
- The motivation is clear. 
- Addresses a relevant problem in LLM reasoning efficiency.
- Includes ablation studies on bias terms and normalization factors.

### Weaknesses
- There already exists a substantial body of research on adaptive reasoning and dynamic CoT control in large language models. However, the authors reduce all prior methods to a single dismissive sentence “these methods still face coarse supervision, limited adaptation to hard cases due to monotonic shortening, and a lack of principled trade-offs between quality, token cost, and latency.” This characterization is overly simplistic and fails to accurately represent the diversity and sophistication of existing approaches. Many prior works have proposed different strategies for trade-offs, such length penalties, etc. Moreover, the authors do not convincingly explain why their proposed HiPO framework is inherently better suited to address these issues. Without a detailed comparative analysis and a clearer articulation of the unique mechanisms through which HiPO overcomes these prior limitations, the novelty of the paper remains questionable.

- It is unclear whether the authors used the Qwen3-8B base model or the Qwen3-8B instruct model during experimentation. The paper states: “Since the Qwen3 model can freely switch between inference modes, we chose it for our experiment.” However, this capability is specific to the instruct variant of Qwen3. If the authors indeed used the instruct model, their reported results should be compared to the Qwen3 technical report, particularly Table 17, which shows a MATH score of 97.4, while this paper reports only 93.6, even after applying the proposed method. This discrepancy raises concerns about experimental reproducibility and baseline validity. The authors need to clearly specify which version of the model was used, under what inference configuration, and explain why their results are significantly lower than the officially reported benchmarks.

- The paper mentions a cold-start stage before reinforcement learning. Regarding the cold-start stage, the paper mentions generating hybrid data using an external model, and Figure 1 strongly suggests that the DeepSeek model was used to construct the cold-start dataset. If that is the case, the process resembles knowledge distillation, where the target model learns from a stronger teacher rather than discovering effective strategies on its own. It would be valuable to test whether the system could train directly with RL without cold-starting the policy. Such an experiment could help determine whether the cold-start stage provides essential stability or merely serves as a redundant initialization step. If HiPO’s reward structure is indeed well-designed, a cold-start-free RL variant should, in principle, converge similarly.

- It would be highly informative to explore whether the ratio of Think-on to Think-off instances correlates with task difficulty. The reviewer suggests dividing the MATH-500 dataset according to its five officially defined difficulty levels and analyzing how HiPO adapts across them. 

- The reviewer is concerned that the model might be implicitly relocating reasoning content into the Answer segment, effectively performing internal reasoning within Think-off mode, as suggested by examples like Figure 1. If such behavior occurs, it could represent a form of reward hacking, since the current reward formulation does not explicitly penalize the length or reasoning structure of responses. As a result, the model might learn to disguise reasoning tokens within the Answer segment to maximize rewards without genuine efficiency improvement. The authors should therefore report the token distributions of Think-on and Think-off outputs separately, and analyze whether reasoning leakage occurs.

### Questions
See Weaknesses

### Soundness
3

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
4

### Summary
The paper proposes HiPO, a framework for adaptive reasoning control that lets an LLM decide when to Think-on (emit CoT) vs. Think-off (answer directly). HiPO has two parts: (i) a hybrid data construction pipeline that pairs Think-on/Think-off responses per query (preferring the shortest correct sample and adding an explanation justifying the chosen mode), and (ii) a hybrid RL reward with a bias adjustment to avoid over-reliance on verbose Think-on and mode-aware advantages that allocate signal to <judge> (mode decision) vs. <answer> tokens. On Qwen3 (1.7B/8B/32B) across math and coding benchmarks (AIME24/25, MATH-500, HumanEval, LiveCodeBench, MBPP, GPQA), HiPO reports reduced token length and lower Think-on ratio while maintaining or improving accuracy over baselines (Cold-Start±GRPO, AdaptThink, AutoThink).

### Strengths
- Clear problem focus (overthinking) with an intuitive mode-gating mechanism and token-level training signal split across <judge> and <answer>. 

- Consistent efficiency gains (shorter outputs, lower Think-on ratio) with comparable or better accuracy on multiple benchmarks and model sizes. 

- Various ablations. Removing global advantage or local normalization degrades accuracy/efficiency, and sensitivity to γ/ω/N is explored.

### Weaknesses
1. Limited statistical rigor. Tables show point improvements but lack variance/CI across seeds and tasks, making deltas (often modest) hard to assess. 

2. Heuristic bias adjustment. The ω-scaled boost to Think-off when close to Think-on can tilt optimization without a principled guarantee (risk of mode-collapse oscillations); analysis is empirical only. 

3. Data pipeline dependence. Selecting the shortest correct response may bias style and harms robustness if “shortest” correlates with template artifacts; the policy could overfit to training distributions. (Ablations help but remain narrow...) Any other variants can improve the quality for this part.

4. Generalization claims. While Qwen3 1.7B/32B are included, domains remain math/code; transfer to tool-use, multilingual tasks, or safety-critical reasoning is not shown

### Questions
1. Can you include Statistical results over seed and different temperatures?

2. Bias adjustment safety. Can you provide a convergence/monotonicity argument, or at least diagnostics showing no oscillatory gating? What happens for ω→0 and ω→0.05 across datasets? (Fig. 5 hints at trade-offs.) 

3. Failure modes. When Think-off is chosen incorrectly (hard question), does HiPO recover (e.g., via fallback) or lock into shallow answers? Any error typology?

4. Beyond math/code. Have you tried tool-augmented QA or long-context tasks where mode selection interacts with retrieval?

5. Ablations on explanation. If you remove the mode-explanation supervision, how much do gating quality and accuracy drop?

### Soundness
3

### Presentation
3

### Contribution
2
