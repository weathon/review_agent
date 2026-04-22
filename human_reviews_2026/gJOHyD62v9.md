# Bridging Pairwise and Pointwise GRMs: Preference-Aware Reward Mechanism with Dynamic Rubric Adaptation

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 8

## Abstract
Reward models (RMs) are central to reinforcement learning from human feedback (RLHF), providing the critical supervision signals that align large language models (LLMs) with human preferences. 
While generative reward models (GRMs) offer greater interpretability than traditional scalar RMs, current training paradigms remain limited. Pair-wise methods rely on binary good-versus-bad labels, which cause mismatches for point-wise inference and necessitate complex pairing strategies for effective application in RLHF. On the other hand,  point-wise methods require more elaborate absolute labeling with rubric-driven criteria, resulting in poor adaptability and high annotation costs.
In this work, we propose the Preference-Aware Task-Adaptive Reward Model (PaTaRM), a unified framework that integrates a preference-aware reward (PAR) mechanism with dynamic rubric adaptation.
PaTaRM leverages relative preference information from pairwise data to construct robust point-wise training signals, eliminating the need for explicit point-wise labels. Simultaneously, it employs a task-adaptive rubric system that flexibly generates evaluation criteria for both global task consistency and instance-specific fine-grained reasoning. 
This design enables efficient, generalizable, and interpretable reward modeling for RLHF. 
Extensive experiments show that PaTaRM achieves an average relative improvement of 4.7\% on RewardBench and RMBench across Qwen3-8B and Qwen3-14B models. Furthermore, PaTaRM boosts downstream RLHF performance, with an average improvement of 13.6\% across IFEval and InFoBench benchmarks, confirming its effectiveness and robustness.
Our code is available at \url{https://anonymous.4open.science/r/PaTaRM-E779}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces PaTaRM, a Preference-Aware Task-Adaptive Reward Model that combines a preference-derived point-wise reward mechanism with dynamic rubric adaptation. PaTaRM converts relative preference data into robust point-wise training signals without requiring explicit point-wise labels, and uses a task-adaptive rubric to generate evaluation criteria that capture both global task consistency and instance-specific fine-grained reasoning. Experiments across multiple models and benchmarks indicate effectiveness.

### Strengths
1. Conducts experiments across different base models and benchmarks, suggesting cross-model applicability.

2. Leverages relative preferences to construct point-wise signals.

3. Employs dynamic, task-adaptive rubrics that can provide interpretable, fine-grained evaluation criteria aligned to each instance.

4. Framework appears modular and compatible with existing RLHF pipelines.

### Weaknesses
1. Generative reward modeling increases training and inference latency, especially during policy optimization.

2. The requirement of n generative rollouts per prompt further amplifies computational costs and may limit scalability.

3. Reward quality is limited by the evaluation LLM’s capabilities; any biases or inaccuracies in the evaluator propagate to the reward.

4. Dynamic rubric adaptation is sensitive to prompt design and may introduce prompt-induced variance or rubric drift.

### Questions
1.	How do you formally define the gap between pair-wise and point-wise reward models in your setting? 

2.	In Figure 1, beyond illustrating differences, can you quantify the discrepancy between pair-wise and point-wise signals?

3.	What is the end-to-end compute breakdown (per prompt rollouts, evaluator LLM invocations, training updates), and how does cost scale with n rollouts, model size, and sequence length?

4.	What are the wall-clock time relative to a standard non-generative reward model?

5.	How sensitive is performance to decoding settings for the generative evaluator?

6.	Does dynamic rubric adaptation mitigate or exacerbate known reward hacking patterns compared to static rubrics or standard RMs?

7.	How sensitive are results to the evaluator LLM choice and to the rubric prompt template? Do small prompt edits materially change outcomes?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces the Preference-aware Task-adaptive Reward Model (PaTaRM), a framework that introduces (1) a Preference-Aware Reward (PAR) mechanism that efficiently converts relative preference signals from widely available pairwise data into robust pointwise training signals, eliminating the need for explicit pointwise labels, and (2) a dynamic rubric adaptation system where the model generates its own flexible, instance-specific evaluation criteria for fine-grained reasoning. Qwen3-8B and Qwen3-14B are optimized via supervised fine-tuning and then reinforcement learning, and experiments show that the reward models have improved performance on reward model benchmarks and provide more effective reward signals for downstream RLHF policy alignment.

### Strengths
**S1.** The paper directly addresses the inflexibility of rubric-based GRMs that rely on rubric generation from external models.

**S2.** The models perform well on the downstream reward modeling benchmark, and the reward signals lead to policy improvements on benchmarks like IFEval and InFoBench.

### Weaknesses
**W1.**  The given formulation actually does not explicitly enforce global transitive consistency across preference chains, which is my concern, especially when you are trying to convert somewhat less information (pairwise) into richer information (pointwise). In practice, this means the model may learn locally coherent but globally inconsistent relationships (e.g., a > b, b > c, yet a<c), since it optimizes over independent pairwise comparisons. I believe this could lead to unstable global rankings and reduce the interpretability of the learned reward function.

**W2.** Since PaTaRM generates the instance-specific rubrics it uses for evaluation, there is a potential for reward self-optimization, where the model implicitly learns to produce rubrics that are easier for it to satisfy rather than faithfully reflecting nuanced user preferences. Without strong constraints or external validation, this adaptive rubric generation could risk a subtle form of “reward hacking” and the authors seem not to address this.

**W3.** It seems that the performance of PaTaRM is still behind compared to existing reward models, in particular R3 (see Q1: it seems like the reported numbers are mismatched with the original paper). However, R3 only utilizes SFT (possibly with high quality) without any further RL training, so I wonder if you would consider Q2.

### Questions
**Q1.** Is there any result for RM-Bench pairwise and comparison with the other reward models? In addition, there is a mismatch between the reported number for R3's performance on RewardBench. However, for all other numbers, I think they should be good.

**Q2.** Since the SFT seems to reduce performance, I wonder if it is because the SFT dataset is not great in terms of quality, especially since Qwen-2.5-72B-Instruct is being used. It is possible that even Qwen3-8B or Qwen3-14B already performs better than Qwen-2.5-72B-Instruct. Perhaps, you could either first distill from better models for better SFT data quality or skip the SFT training and allocate all data using RL.

**Q3.** Is there any statistical significance reported?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces PaTaRM (Preference-aware Task-adaptive Reward Model), a reward-modeling framework for RLHF that turns cheap pairwise preference data into rich, pointwise-like supervision. The method involves generating multiple judgment rollouts for the chosen and rejected responses, scores them under adaptive rubrics, and positively rewards rollouts that are consistent with the human preference, effectively extracting pointwise signals without explicit pointwise labels. Experiments show that this unified setup improves over standard pairwise and pointwise generative reward models on RewardBench and RMBench and also yields better downstream RLHF rewards, supporting the claim that PaTaRM meaningfully bridges pairwise and pointwise GRMs while keeping annotation costs low.

### Strengths
* The paper tackles a very clear gap between efficient pairwise supervision and practically useful pointwise reward models. The proposed PaTaRM combines a preference-aware reward with dynamic rubric adaptation, allowing the model to derive rich, instance-specific supervision from ordinary pairwise data, thereby improving sample efficiency.

* Empirical results on RewardBench and RMBench with two Qwen3 sizes support the claim.
* The paper is clearly-written and well formatted.

### Weaknesses
* The idea of deriving pointwise-like signals from pairwise labels are not new. In standard RLHF using PPO for example, the three stages were 1) SFT training, 2) Reward model training, and 3) RL (e.g., PPO) using the reward model trained in step 2. In this framework, the reward model training in stage 2 relied on a pairwise training where we train the model to assign a higher score to the chosen. Then, with this trained reward model, we use it as a pointwise reward during stage 3. Apart from having multiple rollout during the reward model training, could the authors clarify on the difference?

* The authors mention point-wise GRM could propagate bias. However, the current framework that the authors propose could also suffer from the pairwise data bias. If the pairwise data is noisy or biased, the whole reward inherits that bias and maybe amplifies it. 

* As an extension of previous comment, it would be nice to see how the method compares when the labeled pairwise data is noisy (maybe with injected noise such as randomly flipped labels), and also with respect to the size of the label data.

* It would be nice to see the performance on more reasoning heavy downstream tasks such as math and coding.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper applies a language model post-training pipeline for generative reward models, allowing the model to generate the rubric and evaluate based on its rubric, namely PaTaRM. Notably, every component in post-training, including SFT, DPO, and RL training, is positively contributing to improving PaTaRM as a reward model. The empirical results on the most recent open-source models like Qwen3 demonstrate strong performance on both conventional reward modeling benchmarks and verifiable tasks like instruction-following.

### Strengths
1. The paper contributes to improving generative reward models, which is one of the emerging trends in reward modeling.
2. The experimental results on the Qwen3 series are promising, outperforming a few proprietary generative reward models.
3. Ablation study on SFT and RL training phases clearly supports that applying post-training for reward modeling is a valid approach for better human preference alignment.

### Weaknesses
The experimental design of the paper is reasonable, demonstrating the performance of PaTaRM as a reward model itself, and expanding its use in the RL training. However, one key experimental result is missing: Bradley-Terry reward model vs PaTaRM, both using Qwen3 as base models.

- **Comparison against the Qwen3 Bradley-Terry reward model**: While PaTaRM demonstrates promising results across the two reward model benchmarks, the choice of scalar reward models in Table 1 is questionable. Since the baseline performance of the Qwen3 series could be the dominant factor for the performance of PaTaRM, vanilla Bradley-Terry (BT) reward models trained on top of Qwen3-8B and 14B could build a stronger baseline for the claims in Section 4.2. For example, Skywork-Reward-V2 [1] series trained on Qwen3 show very strong performance on both RewardBench 2 and RMBench, while their training dataset is unknown. Given that, baselines trained with the same base models are essential to clearly state the empirical strength of the method.

&nbsp;

**References**

[1] Liu et al., 2025, “Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy.” (Preprint)

### Questions
- There is a few literatures that claim Qwen models are exceptionally good at RL(VR) training. Would different model families, e.g., Gemma or Llama, benefit from the PaTaRM objective? Small-scale experiments like 3B scale would strengthen the effectiveness of the proposed method.

- How would the test-time scaling with thinking budget impact the final performance? Accuracy and thinking budget correlation analysis alongside the voting experiments in Section 4.7 would be interesting.

- Not a major limitation, but more samples generated from PaTaRM and the dataset for training PaTaRM in the Appendix will yield better understanding on the paper for the readers.

### Soundness
2

### Presentation
3

### Contribution
3
