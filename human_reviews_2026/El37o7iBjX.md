# Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 2

## Abstract
Despite recent rapid progress in AI safety, current large language models remain vulnerable to adversarial attacks in multi-turn interaction settings, where attackers strategically adapt their prompts across conversation turns and pose a more critical yet realistic challenge. Existing approaches that discover safety vulnerabilities either rely on manual red-teaming with human experts or employ automated methods using pre-defined templates and human-curated attack data, with most focusing on single-turn attacks. However, these methods did not explore the vast space of possible multi-turn attacks, failing to consider novel attack trajectories that emerge from complex dialogue dynamics and strategic conversation planning. This gap is particularly critical given recent findings that LLMs exhibit significantly higher vulnerability to multi-turn attacks compared to single-turn attacks. We propose DialTree, an on-policy reinforcement learning framework integrated with tree search that autonomously discovers diverse multi-turn attack strategies by treating the dialogue as a sequential decision-making problem, enabling systematic exploration without manually curated data. Through extensive experiments, our approach not only achieves more than 44.2% higher ASR across 12 target models compared to previous state-of-the-art approaches, but also effectively uncovers new attack strategies by learning optimal dialogue policies that maximize attack success across multiple turns.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DIALTREE-RPO, a novel on-policy reinforcement learning framework that automates the discovery of multi-turn jailbreak strategies against large language models. It treats red-teaming as a goal-oriented, sequential decision-making problem, using tree-structured dialogue exploration and policy optimization to learn adaptive attack strategies without human-crafted templates or data.

### Strengths
1.	First to frame multi-turn red-teaming as a sequential decision problem; introduces a tree-search + GRPO framework that eliminates the need for human-written templates.
2.	Achieves 85% average ASR across 10 models—26% better than the prior best—with only 3 queries per attack; experiments are thorough and reproducible.
3.	Exposes a systemic LLM vulnerability to strategic multi-turn attacks; releasing the methodology (not payloads) accelerates the development of multi-turn safety defenses.

### Weaknesses
1. Lack of novelty. Tree-search + RL for dialogue is not virgin territory. TAP already combines tree expansion + pruning for jailbreaks;
2. This paper did not propose any new jailbreak datasets either; instead, it reused the datasets provided by MTSA.
3. No tests have been conducted to evaluate the robustness of this jailbreaking method under different defenses.

### Questions
1. There is no report on the performance of the red team model after the cold start, nor is there any information on how much the RL has improved it.
2. No tests were conducted to evaluate the robustness of this jailbreaking method under different defenses, which will affect the practical application of the jailbreaking method.
3. The paper claims that "it autonomously discovers diverse multi-round attack strategies". Are there any data to support this statement?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DIALTREE-RPO, a reinforcement learning framework that formulates multi-turn red-teaming as sequential decision-making. The method uses tree-based exploration with quality-aware pruning to efficiently search attack trajectories, and adaptive masking to prevent format degradation during training. Trained against a single small model (Llama-3.2-1B) using on-policy GRPO, it achieves 85.3% average attack success rate across 10 diverse target models including GPT-4o and Gemini-2.0, outperforming prior methods by 25.9% with fewer queries. The framework autonomously discovers novel attack strategies like gradual escalation and cross-lingual evasion, demonstrating that multi-turn interactions expose significantly greater vulnerabilities than single-turn attacks in current LLMs.

### Strengths
Formulates multi-turn red-teaming as a goal-oriented sequential decision-making problem, extending GRPO to dialogues with non-verifiable rewards. 

Dialogue tree rollout with quality-aware pruning enables structured exploration of exponentially large attack spaces, while adaptive masking solves the critical format unlearning problem. 

Tests against diverse closed-source (GPT-4o, Gemini-2.0, o3-mini) and open-source models; includes thorough ablations on tree depth, branching factor, group size, and pruning strategies; validates automated evaluation with human agreement.

### Weaknesses
Although the paper cites previous state-of-the-art methods like X-Teaming (https://arxiv.org/abs/2504.13203
), which shows a higher ASR (94.3%) compared to this paper (86%), it lacks a comparison with prior multi-turn state-of-the-art methods such as X-Teaming and ActorAttack (https://arxiv.org/abs/2410.10700). 

The paper primarily focuses on the attack side and does not explore the defense side.

### Questions
Since previous papers like X-Teaming and ActorAttack show higher effectiveness, does DIALTREE-RPO offer any other advantages, such as improved efficiency or diversity?

Is there analysis done on Claude 3.7/4 Sonnet model, which has been considered nearly immune to
single-turn attacks (https://arxiv.org/pdf/2501.18837) 

Table 4 shows HarmAug-Guard has 84.73% accuracy with 27% false positive rate on JailbreakBench. How do reward model errors affect policy learning?

What responsible disclosure and access control practices will be implemented to prevent the malicious use of DIALTREE-RPO?

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
The paper shows an RL post-training method for multi-turn red-teaming attacks via sequential decision-making in the dialogue. The RL training pipeline leverates exploration instead of labeled data collection. Experiments show the higher ASR of the proposed method compared to baselines.

### Strengths
The paper is generally well-organized, and the figures are clear to follow. The problem of RL post-training in a multi-turn setting is sound and important. It is also interesting to regard the multi-turn setting as a form of decision-making in dialogue in which RL can be applied.

### Weaknesses
- Overall, the method is based on GRPO by applying it to the multi-turn dialogue setting, which is a little bit incremental regarding novelty and contribution. One possible way is to conduct an experiment based on other RL post-training methods, e.g., RLOO, REINFORCE++, DAPO.

- In the node pruning process, since the previous node will be pruned according to the same criteria of topic adherence, the context will be the same along all sampled trajectories along the valid tree nodes and the diversity of the trajectories will be influenced a lot. One possible way to mitigate it might be to develop an adaptive branching factor to make it more diverse. Also, the branching factor needs to be larger in Figure 3, e.g., 16, 32. 

- In the reward design, it uses a classifier to construct a verifiable reward for this non-verifiable problem, as stated in the beginning. However, it is still at risk of reward hacking if the judge is not perfect.

- The adaptive masking part needs more formal presentation, which is expected to be shown in Eq. (3), for example. Figure 2 uses running smoothing for presentation, which is different from the standard deviation or confidence interval. More experiments with different seeds are expected.

- More related work on multi-turn jailbreak defense (e.g. [1] [2]) needs to be discussed in Sec 6 for comprehensiveness, especially [2] seems to also explicitly model the multi-turn dialogue as sequential decision-making in the context of jailbreaks.

---
[1] Lu et al. X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability, arxiv, 2025

[2] Hu et al. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks, arxiv, 2025

### Questions
See Weakness

### Soundness
3

### Presentation
3

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
This paper introduces DIALTREE-RPO, a tree-based reinforcement learning framework for multi-turn red-teaming attacks against LLMs. By modeling adversarial dialogues as sequential decision making and using structured exploration with pruning and adaptive masking, the method autonomously discovers diverse multi-turn attack strategies that outperform previous baselines in attack success rate and efficiency.

### Strengths
1. The paper addresses an important gap in AI safety: the vulnerability of LLMs to adaptive multi-turn attacks.
2. The tree-based exploration and pruning mechanisms are conceptually well-motivated for discovering diverse trajectories, which is crucial for efficient red-teaming.
3. The empirical results are strong and comprehensive, showing clear improvements over previous baselines across a variety of models.
4. The proposed method has promising transferability, being able to jailbreak much larger models after training on smaller ones.

### Weaknesses
1. **Lack of ablation on SFT stage:** There is no explicit comparison of attack efficacy using only the SFT (supervised fine-tuning) attacker before RL training. The improvement brought by RL over the SFT baseline is unclear, which is an important ablation for understanding the contribution of each stage.
2. **Pruning criteria are overly simplistic:** The current pruning process only eliminates obviously problematic branches, such as those with malformed outputs or irrelevant topics. However, it does not attempt to distinguish among the valid branches to identify which ones have higher potential for successful attack. As a result, promising attack trajectories may be overlooked, since the approach only performs random subsampling among the remaining candidates rather than scoring or prioritizing those with greater promise. A more sophisticated strategy—such as utilizing PRMs to rank and select the most promising branches—could help ensure that the tree search focuses resources on the most effective attack paths.
3. **Justification for adaptive masking is unclear:** The effectiveness of adaptive masking is attributed to format retention. However, since the reward is not directly related to format, the sign of the advantage may not reflect format quality, so I'm very curious why this happens. The experimental section also lacks a thorough analysis explaining why masking format tokens on negative-advantage trajectories is beneficial.
4. **Reward and evaluation inconsistency:** The training stage uses HarmAugGuard for harmfulness evaluation, while the test stage resorts to GPT-4o as the judge—these may be misaligned.
5. **Ambiguity in tree rollout vs. GRPO comparison:** In Table 2 and line 370, the comparison between "w/o any pruning" (tree rollout only) and "w/o tree rollout" (standard GRPO) is unclear and seemingly contradictory—the results show that GRPO performs better than tree rollout-only, yet the text claims better gain from tree rollout than GRPO. The statement requires clarification.
6. **Utterance length limitation:** In Appendix C.2, you mention limiting outputs to 256 and 150 tokens. How is this enforced? Is it simple truncation? RL training can lead to longer outputs; could this be responsible for the increase in malformed outputs in Figure 2, i.e., the model may not have finished its format before truncation?

### Questions
1. **Tree rollout path:** With a maximum depth of 5 and a branching factor of 2, the maximum possible number of paths is 32. Is supplemental sampling performed when the actual number of paths falls short due to pruning?
2. **Jailbreak-R1 implementation:** Jailbreak-R1 is a single-turn method. How did you use it in your multi-turn setup? Was it modified for multi-turn, or was a single-turn evaluation used?

### Soundness
3

### Presentation
2

### Contribution
2
