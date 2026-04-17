# Accurate and Diverse LLM Mathematical Reasoning via Automated PRM-Guided GFlowNets

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Achieving both accuracy and diverse reasoning remains challenging for Large Language Models (LLMs) in complex domains like mathematics. A key bottleneck is evaluating intermediate reasoning steps to guide generation without costly human annotations. To address this, we first introduce a novel Process Reward Model (PRM) trained automatically using Monte Carlo Tree Search coupled with a similarity-based data augmentation technique, effectively capturing step-level reasoning quality. Leveraging this PRM, we then adapt Generative Flow Networks (GFlowNets) to operate at the reasoning step level. Unlike traditional reinforcement learning focused on maximizing a single reward, GFlowNets naturally sample diverse, high-quality solutions proportional to their rewards, as measured by our PRM. Empirical evaluation shows strong improvements in both accuracy and solution diversity on challenging mathematical benchmarks (e.g., +2.59\% absolute accuracy on MATH Level 5 for Llama3.2-3B), with effective generalization to unseen datasets (+9.4\% absolute on SAT MATH). Furthermore, we benchmark our PRM against existing open-source reward models, demonstrating superior alignment with reasoning quality and more consistent guidance for downstream generation. Our work demonstrates the potential of PRM-guided, step-level GFlowNets for developing more robust and versatile mathematical reasoning in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper trains an automated Process Reward Model (PRM) using MCTS + similarity grouping, then uses the PRM as the step-level reward to fine-tune LLMs with GFlowNets (SubTB) at step granularity/ Reported gains: small but positive on in-domain MATH Level 5 and larger on SAT MATH; authors also claim increased diversity via lower semantic similarity of generated solutions.

### Strengths
1. Clear step-level formulation (actions = reasoning steps) and an explicit termination probability π(sf|s) within the GFlowNet policy. 
2. Self-contained PRM training pipeline (MCTS, continuous scores, rollout reuse + similarity grouping) with some validation diagnostics

### Weaknesses
1. PRM reliability and calibration are under-substantiated.
The core premise hinges on PRM accuracy (U(s′|s) ∈ [0,1]) guiding both PPO and GFlowNets. But the paper offers limited, small-scale PRM validation and mostly heuristic similarity grouping. This makes it hard to trust the PRM as an absolute reliable signal.

2. Overlap/Difference with prior work (“Flow of Reasoning”) is unclear.
The paper positions step-level GFlowNets for diverse trajectories, but related work already targets divergent reasoning with minimal examples (Flow of Reasoning, 2024/2025). What is materially new? The comparison is relegated to related-work mentions rather than a head-to-head study, and the conceptual delta is not crisply argued. 

3. Gains are modest and may be within variance.
On in-domain MATH Level 5, the 3B model improves ~+2.6pts over baseline and 8B is +0.7–0.9. GSM8K barely moves. There are no confidence intervals, no multi-seed repeats, and limited ablations teasing apart PRM vs. GFlowNet vs. decoding heuristics. It’s difficult to conclude statistical significance rather than randomness. (Table 2). 

4. Termination probability π(sf|s): underspecified in practice.
While π(sf|s) is mentioned (sink state), the actual parameterization/training signals for termination are not detailed: how is π(sf|s) learned/stabilized under SubTB at step level; what is the impact of termination on reward estimates? (Sec. 4.1–4.2). 

5. Chosen tasks don’t stress true solution multiplicity.
MATH/GSM8K often admit stylistic variation rather than structurally distinct paths. If “diversity” is central, it'd be more convincing to evaluate on domains with genuinely multiple optimal plans (e.g., Blocksworld, program synthesis with multiple implementations, theorem-proving with lemmas reorderings). The current “diversity” metric is a weak proxy and might just reward rephrasing, not distinct strategies. (Sec. 5.2).

### Questions
1. Termination behavior and reasoning depth:
You mention the use of a termination probability π(sf|s) to model when reasoning should stop. Could you discuss what qualitative patterns you observed — for instance, do longer reasoning chains correlate with higher accuracy, or does early termination sometimes produce more concise correct reasoning?
2. On PRM design and training dynamics:
How sensitive is your overall training process to the quality of the PRM? For example, if the PRM is slightly miscalibrated or trained on fewer MCTS rollouts, does the downstream GFlowNet policy still converge reliably, or do you observe instability?

### Soundness
2

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
This paper introduces a novel framework to improve both accuracy and diversity in LLM mathematical reasoning. The authors first develop an automated PRM using MCTS and a similarity-based data augmentation technique to capture step-level reasoning quality without human annotation. This PRM is then used for a step-level GFlowNet, which empirically demonstrates significant gains in accuracy and diversity over PPO baselines.

### Strengths
1. The paper tackles the highly important and challenging goal of improving both accuracy and diversity in LLM reasoning.
2. The paper is clearly presented and easy to follow.
3. The method shows strong empirical gains over the PPO baseline.

### Weaknesses
1.  The paper lacks sufficient ablation studies on the key techniques used in the proposed PRM, such as similarity-based data augmentation and continuous scoring. This makes it difficult to judge the actual effectiveness and individual contribution of each proposed component.

2.  The performance evaluation is limited to a comparison with PPO, while lacking comparisons against other methods like DPO or GRPO. Although the method demonstrates superior performance over PPO, it remains unclear whether it holds any advantage over these other techniques.

3.  The paper lacks a direct experimental comparison between the token-level GFlowNet and the proposed step-level GFlowNet. The authors only argue conceptually for the superiority of the step-level approach. To validate the step-level GFlowNet as a significant contribution, it is essential to demonstrate this superiority empirically.

### Questions
Please refer to the Weaknesses section above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a novel framework to enhance the accuracy and diversity of Large Language Models (LLMs) in mathematical reasoning. The method combines two key components: an automatically trained Process Reward Model (PRM) and step-level fine-tuning using Generative Flow Networks (GFlowNets). Empirical results on math benchmarks show that this approach improves accuracy and significantly enhances the diversity of generated solution strategies, with particularly better generalization performance.

### Strengths
1. The paper is generally well written and easy to understand.

2. The paper provides thorough experimentation, demonstrating improvements in both accuracy and a quantitatively measured diversity metric across multiple benchmarks.

### Weaknesses
1. The baseline of using GFlowNets with only final rewards seems missing, which would clarify the specific contribution of the sophisticated step-level PRM versus the GFlowNet objective itself. Also see Q1.

2. The overall framework involves complex components (MCTS, similarity-based augmentation, step-level GFlowNets with SubTB loss), which might make it sensitive to hyperparameters and potentially difficult to reproduce. The paper would be strengthened by a sensitivity analysis of key parameters (e.g., the similarity threshold).

### Questions
1. The interplay between PRM and GFLowNet is still somehow confusing. Is the GFLowNet objective generally better than PPO, or does it have specific strengths when optimizing the PRM? Could you comment on the results of using GFlowNets with a simple, binary terminal reward and compare it with PPO? This would help isolate the benefit of the step-level reward signal from the benefit of the GFlowNet's diversity-seeking objective.

2. Besides, could you elaborate on the PPO baseline? What’s the exact reward used here? Only from PRM or also combined with the final correctness? If combined, how are they combined?

### Soundness
2

### Presentation
2

### Contribution
2
