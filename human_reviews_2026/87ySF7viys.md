# RESTRAIN: From Spurious Votes to Signals — Self-Training RL with Self-Penalization

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
Reinforcement learning with human-annotated data has boosted chain-of-thought
reasoning in large reasoning models, but these gains come at high costs in labeled
data while faltering on harder tasks. A natural next step is experience-driven learning, where models improve without curated labels by adapting to unlabeled data.
We introduce REinforcement learning with Self-resTRAINt training (RESTRAIN),
a self-penalizing RL framework that converts the absence of gold labels into a
useful learning signal. Instead of overcommitting to spurious majority votes,
RESTRAIN exploits signals from the model’s entire answer distribution: penalizing
overconfident rollouts and low-consistency examples while preserving promising
reasoning chains. This self-penalization mechanism integrates seamlessly into
policy optimization methods such as GRPO, enabling continual self-improvement
without supervision. On challenging reasoning benchmarks, RESTRAIN delivers
large gains using only unlabeled data. With Qwen3-4B-Base and OctoThinker
Hybrid-8B-Base, it boosts Pass@1 by up to +140.7% on AIME25, +36.2% on
MMLU STEM, and +19.6% on GPQA-Diamond, nearly matching gold-label
training while using no gold labels. These results demonstrate that RESTRAIN establishes a scalable path toward stronger reasoning without gold labels.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes RESTRAIN, an RLVR framework that does not need ground truth answers for questions. It tackles the problem that majority-voted answer may not be correct, especially for hard labels, and designed a weight-smoothing method to consider infrequent answers (pseudo-label weighting, negative rollout penalization and prompt-level weighting). The experiments show relatively large improvements over other unsupervised baselines.

### Strengths
- The motivation is very clear (majority voting != correct) and the designed soft weighting of infrequent answers seem reasonable and directly tackle the problem of majority-voting.
- The negative rollout penalization is also convincing in down-weighting too infrequent/long-tailed answers (where the model is unsure about).
- From the experiment results, it seems the proposed RESTRAIN can largely improve over existing unsupervised RLVR methods like TTRL. Besides, they test on both Qwen3-4B-Base model and Octothinker Hybrid 8B, which demonstrate that the method is applicable to recent SOTA base models and various model sizes.
- They also provided very detailed ablation studies and useful details for reproduction.

### Weaknesses
- Not sure if the threshold kappa will be sensitive to rollout numbers and sampling temperature. Is it fixed over different datasets (which have different difficulty)?
- Typo in Table 1: w/ access to gold labe -> w/ access to gold label

### Questions
- How does the gap between pass@1 and majority voting change after RESTRAIN training?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work aims to solve the training instability and collapse caused by unreliable majority-vote signals in unsupervised RL methods like TTRL. The paper proposes RESTRAIN, a self-penalizing framework that replaces TTRL's reward scheme. Its novel loss function stabilizes training by using soft, frequency-based weighting for all potential answers, actively penalizing low-consistency outputs, and down-weighting unreliable prompts identified by a frozen model. Experiments show RESTRAIN significantly outperforms TTRL, avoids training collapse, and nearly matches the performance of a fully supervised, gold-label baseline.

### Strengths
1. The work directly addresses the critical and well-documented stability problem of majority-vote-based unsupervised RL.

2. The three components are well-motivated and logically designed to counteract the specific failure modes of TTRL, such as signal smoothing and noise penalization.

3. The empirical results are a significant strength, demonstrating performance that not only avoids collapse but also approaches the upper bound of fully supervised gold-label training.

4. The paper is supported by a thorough set of ablation studies that demonstrate the necessity of each component for stable training and final performance.

### Weaknesses
1. High Hyperparameter Sensitivity: The method's performance appears highly sensitive to its key hyperparameters, including the weighting skewness, the penalty threshold, and the penalty offset. The ablation studies show that performance drops sharply outside a narrow range of these values, which may hinder its general applicability and reproducibility.

2. Limited Novelty: The contribution appears to be more of a successful systems-level engineering effort than a fundamentally new paradigm. The core components (soft weighting, curriculum learning, and negative penalization) are all established concepts. Overall, this work is more like an incremental "patch" on TTRL.

### Questions
1. The hyperparameters are clearly critical. How were the optimal values selected?

2. Computational Cost: This method inherits the high computational cost of TTRL while introducing additional computation. Are these additional overheads significant compared to the TTRL baseline?

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
4

### Summary
This paper proposes a new GRPO-based loss for self-training. Specifically, it does not rely solely on majority-voted answers but instead treats all answers as potentially correct, assigning different weights and penalizing low-consistency examples. Through extensive experiments on math and science datasets and two different models, the authors demonstrate that their self-training method outperforms previous approaches.

### Strengths
1. The proposed method is intuitive, well-motivated, and clearly described. It is simple yet effective.
2. The method achieves significant improvements over comparative self-training approaches across different models and two task domains.

### Weaknesses
1. The proposed loss and experiments are too closely tied to GRPO. It remains uncertain whether the method is compatible with other RL algorithms, such as PPO and PRIME, as compared by TTRL.
2. The tested datasets and models are relatively limited. Although the proposed method shows promising results in this paper, it is unclear whether it is biased toward inherently stronger models with certain specialized skills (both are base models), which may not generalize well in practice.

### Questions
1. How can TTRL be adapted to other RL framework and what are the results?
2. Could you provide results using Llama-3.1-8B-Instruct and Qwen2.5-Math-1.5B or 7B models?

### Soundness
3

### Presentation
4

### Contribution
3
