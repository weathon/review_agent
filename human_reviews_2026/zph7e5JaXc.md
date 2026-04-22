# Explain in Your Own Words: Improving Reasoning via Token-Selective Dual Knowledge Distillation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Knowledge Distillation (KD) can transfer the reasoning abilities of large models to smaller ones, which can reduce the costs to generate Chain-of-Thoughts for reasoning tasks. KD methods typically ask the student to mimic the teacher's distribution over the entire output. However, a student with limited capacity can be overwhelmed by such extensive supervision causing a distribution mismatch, especially in complex reasoning tasks. We propose Token-Selective Dual Knowledge Distillation (TSD-KD), a framework for student-centric distillation. TSD-KD focuses on distilling important tokens for reasoning and encourages the student to explain reasoning in its own words. TSD-KD combines indirect and direct distillation. Indirect distillation uses a weak form of feedback based on preference ranking. The student proposes candidate responses generated on its own; the teacher re-ranks those candidates as indirect feedback without enforcing its entire distribution. Direct distillation uses distribution matching; however, it selectively distills tokens based on the relative confidence between teacher and student. Finally, we add entropy regularization to maintain the student's confidence during distillation. Overall, our method provides the student with targeted and indirect feedback to support its own reasoning process and to facilitate self-improvement. The experiments show the state-of-the-art performance of TSD-KD on 10 challenging reasoning benchmarks, outperforming the baseline and runner-up in accuracy by up to 54.4% and 40.3%, respectively. Notably, a student trained by TSD-KD even outperformed its own teacher model in four cases by up to 20.3%. The source code is available at https://github.com/kmswin1/TSD-KD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Token-Selective Dual Knowledge Distillation (TSD-KD), a student-centric framework for improving reasoning through selective and dual-mode knowledge transfer. The method integrates: Indirect distillation, the student proposes candidate tokens and receives preference ranking feedback from the teacher (similar to DPO-like weak supervision); Direct distillation, selective distribution matching on tokens with large student–teacher uncertainty gaps using JSD, Entropy regularization — confidence enhancement through minimizing entropy of the most uncertain tokens. The authors conduct comprehensive experiments on reasoning benchmarksusing Qwen2.5 and Gemma2 model families. Results show consistent improvements, with the student model occasionally outperforming its teacher.

### Strengths
1. Complete and sound framework combining distillation with entropy regularization forms a coherent pipeline.

2. Writing is good.

3. Use of token entropy for identifying important tokens aligns with recent research trends in reasoning-focused LLMs (e.g., 80/20 entropy rule, ARPO).

4. Comprehensive experiments across two model families demonstrate generalizability.

### Weaknesses
1. The core idea of TSD-KD is highly similar to <Keypoint-based Progressive Chain-of-Thought Distillation (icml 2024)> in motivation. Both approaches emphasize selective token weighting and distillation; TSD-KD replaces KPOD’s mask-learning with entropy-based selection but retains the same underlying philosophy. However, this previous work is completely neglected. 


2. At this time, it is unclear why the authors did not conduct experiments on the Qwen3 family, such as Qwen3-8B, which has become the de-facto principle for reasoning evaluation (like in <Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning>.). The paper only reports results on Qwen2.5 and Gemma2, both of which are now relatively outdated and substantially weaker in reasoning capability. Since the proposed method explicitly targets reasoning enhancement, it is essential to verify its effectiveness on more competitive and up-to-date models.

3. Limited theoretical insight are provided in this paper. For example, the paper lacks a clear theoretical justification for why entropy-based token selection truly improves reasoning robustness beyond serving as a heuristic importance measure.

### Questions
refer to Weaknesses, I will adjust my score according to the responses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Token-Selective Dual Knowledge Distillation (TSD-KD), a framework for distilling reasoning ability from large language models to smaller student models by focusing on important tokens in the reasoning chain. TSD-KD combines two main innovations: (1) indirect, preference-based distillation, where the teacher re-ranks student-generated token candidates without forcing its output distribution, and (2) direct, gated distillation that selectively applies distribution-matching to tokens where the student is uncertain and the teacher is confident. The approach is regularized with selective entropy minimization on the most uncertain tokens. Empirical evaluation across 10 challenging reasoning benchmarks demonstrates TSD-KD’s strong performance, including cases where the compressed student surpasses its teacher.

### Strengths
1.The use of student-generated candidates (preference-based indirect distillation) and selective, entropy-based token gating in direct distillation is thoughtfully motivated and distinguishes the framework from prior “teacher-forcing” approaches. The focus on letting the student “explain in its own words” resonates with cognitive insights and supports the central claim.
2.The explicit combination of indirect and direct knowledge distillation, each carefully limited to critical tokens, is well-positioned to address known weaknesses of pure distribution-matching or of-point-wise imitation.
3. The mathematical formulation is transparent, the underlying assumptions are stated, and the algorithmic components are described with appropriate rigor. 
4. TSD-KD consistently outperforms strong baselines, with substantial absolute gains. Importantly, in multiple cases, the student model trained via TSD-KD surpasses the teacher.

### Weaknesses
1. The preference-based indirect distillation encourages the student to align with the teacher’s ranking on top-$k$ student candidates. However, this assumes that the student's beam search is likely to generate candidates close to the correct reasoning trace, which may not hold for weaker students or for highly ambiguous problems. 
2. While tables and figures provide extensive quantitative results, the paper lacks qualitative or error analysis on the types of reasoning improvements the student makes with TSD-KD (beyond aggregate accuracy).
3. Even though performance improvements are noticeable, the paper does not report any statistical significance tests.

### Questions
1. How robust is the preference-based indirect distillation if the student’s top-$k$ candidates are mostly incorrect? Does the framework degrade gracefully if initial reasoning is off-policy, or does performance collapse? Are there analyses on very weak students or pathological candidate proposals?
2. Do any ablation results suggest redundancy between direct distillation with entropy gating and selective entropy minimization? Are there tasks where one suffices without the other, and can the gains be attributed to only one component in certain domains?
3. What steps are in place to detect or mitigate biases propagated from teacher to student, considering that only a subset of tokens is distilled but on the student’s own output?

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
The paper proposes Token-Selective Dual Knowledge Distillation (TSD-KD), a framework designed to efficiently transfer the reasoning abilities of a large teacher model to a smaller student model, aiming to reduce the cost of Chain-of-Thoughts (CoT) generation. Adopting a student-centric, on-policy distillation paradigm, TSD-KD applies supervision only to the most critical or uncertain tokens during the reasoning process, thereby avoiding the distribution mismatch and overwhelming issues associated with traditional Teacher-Forcing. The method integrates three key components, all guided by token selection: Indirect Distillation (teacher acts as a preference ranker for student candidates), Direct Distillation (applying GKD loss to tokens with a large uncertainty gap—student uncertain but teacher certain), and Entropy Regularization (selectively minimizing student entropy on critical tokens).

### Strengths
1) TSD-KD achieves State-of-the-Art performance across 10 challenging reasoning benchmarks. Experimental results demonstrate its significant superiority over existing baseline methods across multiple tasks.
2)  The student model, after training, even surpasses its teacher model on some reasoning tasks (with improvements up to 20.3%). This result strongly suggests the framework is not merely imitative but effectively promotes the student model in building its own, more generalizable reasoning logic.

### Weaknesses
1) The core insight of the paper—that "high-entropy/uncertain tokens are critical branching points in reasoning" and should be targeted for selective supervision—is not an original discovery. This phenomenon, which guides the model learning process, has been well-established in antecedent works (such as the RL-based methods by Wang et al. (2025) and Lei et al. (2025)). Therefore, the paper's contribution lies primarily in the engineering application and integration of this existing principle into the knowledge distillation domain for selective supervision, rather than a breakthrough in fundamental mechanism discovery or method innovation.
2) The TSD-KD methodology lacks deep theoretical innovation in distillation, being an effective combination of existing techniques and intuitive heuristic rules. Specifically, the Indirect Distillation employs the established Plackett-Luce (PL) model from RLHF, and Direct Distillation uses the known Generalized JSD (GKD) loss. While the "uncertainty gap" token selection mechanism is novel, it functions as an intuitive heuristic rule. Consequently, the paper's main contribution is the effective integration of these existing components, rather than the proposal of a new foundational distillation mechanism or a novel loss function.
3) The crucial length of the "Opener" for selective supervision is defined by an empirical hyperparameter: the c% accumulated entropy threshold (set to c=10% based on ablation studies). This fixed-ratio approach is a heuristic inherited from similar suggestions in other reinforcement learning works. The absence of a dynamic or adaptive mechanism that adjusts this threshold based on the specific complexity and depth of the reasoning task limits the theoretical generalizability of the method, as the optimal empirical value may vary significantly across different domains (e.g., mathematical vs. common-sense reasoning) and model architectures.

### Questions
1) Given that the insight "high-entropy/uncertain tokens contribute more" is highly similar to recent RL-based works (e.g., Wang et al. (2025) and Lei et al. (2025) as mentioned), how do the core innovative mechanisms of this paper (e.g., the uncertainty gap selection, the Dual Distillation design) demonstrate a theoretical or empirical advantage over the Token Importance mechanisms in the precursor works?
2) In the context of Knowledge Distillation, what specific advantages—such as increased data efficiency or stability—does selective supervision offer that cannot be achieved or are less efficient using traditional RL frameworks (i.e., penalizing/rewarding only critical tokens via sparse reward signals)?
3) Indirect Distillation is only applied during the Opener phase. How does the student model maintain reasoning consistency and quality during the subsequent unsupervised phases? If the student selects a path consistent with the teacher's preference during the Opener, to what extent does this restrict its ability to develop new, non-imitative reasoning logic "in its own words" in the subsequent steps?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Token-Selective Dual Knowledge Distillation (TSD-KD), a framework for transferring large language model reasoning abilities to smaller models. The method is designed to provide targeted supervision, focusing on high-uncertainty tokens to mitigate issues with existing KD approaches. Three key components are listed: 

1) Token-Selective Indirect Distillation: The teacher provides preference rankings over the student's top-k generated tokens in the initial sequence of reasoning (the "opener"), utilizing a Plackett-Luce model.
2) Token-Selective Direct Distillation: A JSD-based distribution matching loss is applied only to tokens where the student's uncertainty (entropy) significantly exceeds the teacher's confidence (the "uncertainty gap").
3) Token-Selective Entropy Regularization: The entropy of the student's top 10% most uncertain tokens is minimized.

### Strengths
1. The core idea of Token-Selective Direct Distillation is well motivated. 
2. Authors provide comprehensive ablation study in demonstrating effects of each component and showed strong empirical results over baselines.

### Weaknesses
W1: Hyperparameter Sensitivity: The framework relies on an extremely sensitive set of hyperparameters ($c$, $k$, $s$, $\beta$), as demonstrated by sharp performance drop-offs in the appendix analyses. This suggests the method is brittle and lacks practical generalizability. In the Table 1, authors also only report the performance from the best hyperparamter selections. I wonder how much this complex setup could transfer into new domains or tasks.

W2: Conflict Between On-Policy Learning and Entropy Minimization ($\mathcal{L}_{EM}$): The $\mathcal{L}_{EM}$ term, which minimizes entropy on the top 10% most uncertain tokens, fundamentally conflicts with the core on-policy principle of preserving and encouraging exploration. While selectivity is claimed as a mitigation, the paper does not analyze the true impact on the student's output diversity or rigorously justify that minimizing entropy is superior to simpler confidence maintenance.

W3: The paper provides insufficient analysis to attribute the performance at Token-Selective Indirect Distillation. It is unclear if the success of the indirect distillation is due to the preference ranking (teacher's subtle knowledge transfer) or simply the top-k token candidate proposal. Based on prior work, latter might be the bigger contribution. I believe authors should perform additional ablation experiments to justify that preference ranking is necessary.

W4: There is a very relevant paper Speculative KD (https://arxiv.org/abs/2410.11325). Authors should consider compare to or mention.

### Questions
At W3

### Soundness
3

### Presentation
3

### Contribution
3
