# Staying in the Sweet Spot: Responsive Reasoning Evolution via Capability-Adaptive Hint Scaffolding

- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has advanced the reasoning capabilities of large language models (LLMs).
However, existing RLVR methods often suffer from exploration inefficiency due to mismatches between problem difficulty and model capability: overly difficult problems hinder reasoning path discovery, while overly simple problems offer little learning signal.
To address this, we first formalize the effect of problem difficulty by quantifying the relationship between loss descent magnitude and rollout accuracy.
Building on this analysis, we propose SEELE, a supervision-aided RLVR framework that dynamically adjusts problem difficulty to lie within the high-performance region.
SEELE augments each training sample by appending a hint (part of a full solution) for difficulty reduction. 
Unlike previous hint-based approaches, SEELE deliberately computes the hint length for each individual problem to achieve an optimal difficulty.
The optimal hint length is determined via multi-round rollout sampling, where an item response theory model fits accuracy–hint pairs from previous rounds to predict the next-round hint.
This instance-level, real-time difficulty adjustment aligns problem difficulty with the evolving model capability, thereby improving exploration efficiency. 
Experiments show that SEELE outperforms Group Relative Policy Optimization (GRPO) and Supervised Fine-tuning (SFT) by +10.0 and +8.4 points, respectively, and exceeds the best prior supervision-aided approach by +3.8 points on average across six math reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies how to improve training efficiency in RLVR by dynamically adjusting per-instance problem difficulty via hint scaffolding. The authors present a theoretical analysis arguing learning efficiency is upper-bounded by a quadratic function of rollout accuracy and is maximized near 50% accuracy. Building on this, they propose SEELE, a multi-round rollout scheme that fits a predictor online using an IRT/3PL model over collected (hinting rate, accuracy) pairs and appends an instance-specific hint whose length is predicted by the predictor. Experiments on multiple base models show consistent gains over GRPO, SFT, and recent hint/supervision-aided baselines.

### Strengths
1. The motivation is clear with theoretical analysis
2. The method is sound and novel.
3. Strong empirical gains across tasks and model families

### Weaknesses
1. Efficiency concerns: SEELE converts the original single-round rollout generation in GRPO into a multi-round process, which inevitably introduces additional computational overhead compared with single-round parallel generation. However, the paper does not report any efficiency metrics such as wall-clock time or overall compute cost.

2. Applicability concerns
- The multi-round rollout scheme requires a sufficient number of rollouts to fit the predictor reliably, which may limit its applicability under low-resource settings.
- The method is tightly coupled with group-based RL algorithms such as GRPO, and cannot be directly applied to algorithms like PPO that do not rely on group rollouts.
- Both the theoretical analysis and the predictor design are tailored to binary-reward settings, restricting generalization to more complex reward structures.

3. Lack of hyperparameter sensitivity analysis: 
   The paper does not study how performance varies with respect to key hyperparameters such as $k_0, v_0$, leaving the robustness of the method unclear.

4. Lack of limitation discussion

5. Missing discussion of related works:
   The idea of selecting or emphasizing prompts with intermediate success rates has appeared in prior works on Prompt Curriculum and Prompt Selection [1,2,3]. However, these connections are not discussed, which would help situate SEELE within the broader landscape of adaptive prompt and curriculum methods.

[1] Self-Evolving Curriculum for LLM Reasoning\
[2] Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning\
[3] Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models?

### Questions
see weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SEELE, a supervision-aided RLVR framework that dynamically adjusts problem difficulty. SEELE appends instance-specific partial solutions as hints and adaptively adjusts their length to maintain rollout accuracy near 50%, a theoretically justified optimum for learning efficiency. The framework employs multi-round estimation and an prediction model to infer the relationship between hint ratio and accuracy based on Item Response Theory. Experimental results show that SEELE achieves stronger generalization across multiple models and benchmarks, outperforming SFT, GRPO, and other baselines.

### Strengths
1. The paper is well-written and presents a coherent motivation for the proposed approach.
2. The choice of maintaining a 50% rollout accuracy is theoretically supported, and the integration of Item Response Theory provides a rigorous and interpretable framework for modeling task difficulty and hint rate.
3. SEELE achieves consistent performance gains across multiple model families and reasoning benchmarks. The paper also conducts thorough ablation studies on target difficulty levels and multi-round configurations, providing a convincing evaluation of the framework’s design efficacy.

### Weaknesses
1. Some notations are confusing
- The paper does not clearly specify whether $f_{\phi}$ is a global predictor shared across all instances or a local model optimized separately for each instance.
- $w$ is not defined in Algorithm 1.
2. I think the current evaluation setup does not clearly demonstrate the benefits of scaffolding compared with the original GRPO method, as the training datasets are biased toward harder examples. It would be more informative to analyze SEELE’s effectiveness across different difficulty regimes—for example, by training on (1) hard questions with hints, (2) tractable questions that the model can already solve (with or without hints), and (3) a mixed setting: training on datasets with both hard questions and tractable questions. Such a comparison would better clarify the advantages of adaptive scaffolding methods, i.e., learning from harder examples.

### Questions
1. Could you clarify the "j" in Figure 2?
2. For the partial solutions used as hints, were only correct reasoning traces included, or were all generated traces (including incorrect ones) used? Additionally, is this the same dataset used in SFT, and how many reasoning traces per question were retained?
3. What is the computational overhead introduced by the multi-round estimation compared with original GRPO? It would be helpful to include a quantitative comparison showing total wall-clock cost.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
SEELE is a supervision-aided RLVR framework that fixes a key problem in current RL-with-verifiable-rewards methods: the task difficulty often doesn’t match the model’s current ability, which makes exploration inefficient. SEELE keeps training inside that region by adding hints to each problem and dynamically choosing the hint length per instance so that the task is neither too hard nor too easy. To pick that hint length, it runs multi-round rollouts and fits an item-response-theory (IRT) model on accuracy–hint pairs to predict the best hint for the next round. The authors evaluate the method over several reasoning benchmarks.

### Strengths
1. The paper introduces SEELE, a method that leverages instance-level hints to make RLVR exploration more effective.
2. The method is evaluated on several math-reasoning benchmarks and shows consistent gains even with relatively small models.

### Weaknesses
1. The paper should clarify how the hints are produced and whether they risk leaking too much target information, effectively turning the setup into SFT; more analysis of how different hint lengths affect training dynamics would help.
2. In Figure 4, reward increases while response length drops sharply and accuracy improves only modestly, which suggests the training dynamics may not be fully understood or may be unstable.
3. The experiments are limited to small models (e.g., Qwen2.5-3B); it would be stronger to show results on larger models (e.g., Qwen2.5-32B or QwQ-32B) to confirm the method scales and is not just fixing small-model artifacts.

### Questions
Please refer the limitations.

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
The paper presents Seele, a multi-round hint framework, for RL-Finetuning of LLMs, which is adaptable to problem difficulty. The authors begin by providing a theoretical foundation for identifying how rollout accuracy correlates with learning efficiency. They then propose an adaptive multi-round framwork

### Strengths
- Theoretical contribution for relationship between learning efficiency and learning accuracy
- Empirical evaluation on a wide range of math + general domain reasoning benchmarks.

### Weaknesses
- Measuring the difficulty from the average success rate + hinting has been studied in prior work, making the contribution strictly the integration of these approaches, which seems limited from a novelty perspective.
- Using all mxn rollouts for advantage calculation seems incorrect, in particular for the baseline computation, where the hint is different per round. A value baseline (V(s)) should strictly be a function of the state, which isn't static in this instance, given that the loss is computed only on generated tokens. 
- The design decisions (e.g 3PL and multi-round sampling) aren't independently ablated, making it unclear what the contribution of each component is to the approach

### Questions
- How does the flop equivalent performance compare for Seele vs other approaches due to additional adaptable component contributing to additional computation?
- How does Seele work for LongCOT models (responses are quite short at 800 tokens) with lower scores on math reasoning benchmarks?
- See Weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2
