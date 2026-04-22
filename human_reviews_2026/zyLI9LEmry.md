# Preference-based Policy Optimization from Sparse-reward Offline Dataset

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Offline reinforcement learning (RL) holds the promise of training effective policies from static datasets without the need for costly online interactions. However, offline RL faces key limitations, most notably the challenge of generalizing to unseen or infrequently encountered state-action pairs. When a value function is learned from limited data in sparse-reward environments, it can become overly optimistic about parts of the space that are poorly represented, leading to unreliable value estimates and degraded policy quality. To address these challenges, we introduce a novel approach based on contrastive preference learning that bypasses direct value function estimation. Our method trains policies by contrasting successful demonstrations with failure behaviors present in the dataset, as well as synthetic behaviors generated outside the support of the dataset distribution. This contrastive formulation mitigates overestimation bias and improves robustness in offline learning. Empirical results on challenging sparse-reward offline RL benchmarks show that our method substantially outperforms existing state-of-the-art baselines in both learning efficiency and final performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents a preference-based RL algorithm, which trains policies by contrasting successful demonstrations with failure behaviors present in the dataset. Experiments on offline benchmarks with sparse rewards validate the effectiveness of the proposed method.

### Strengths
- This paper proposes a contrastive preference learning framework to bypass direct value function estimation.
- This paper provides both empirical and theoretical analyses.
- The proposed approach outperforms baselines in various benchmarks.

### Weaknesses
- The motivation is not adequately supported by evidence. The authors claim that existing methods are sensitive to support mismatches and prone to high variance or instability, particularly when data are limited or rewards are sparse, but no empirical results or references are provided to support these statements.
- The paper does not provide results with other competitive baselines on MetaWorld, such as PREFORL [1] and CPL [2].
- There is no sensitivity analysis on the representative segment length $k$ and the contrastive bias $\lambda$.

References:

[1] Tarasov et al. "Revisiting the Minimalist Approach to Offline Reinforcement Learning", NeurIPS, 2023.

[2] Hejna et al. "Contrastive Preference Learning: Learning from Human Feedback without RL", ICLR, 2024.

### Questions
How is MetaWorld configured to use sparse rewards? By default, MetaWorld provides dense reward settings.

### Soundness
2

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
This paper proposes a novel method named PREFORL, which utilizes a contrastive learning framework to learn preferences from successful trajectories and synthetically degraded trajectories, aiming to address the value overestimation problem in sparse-reward offline reinforcement learning.

### Strengths
1) The idea is novel, introducing the concept of preference learning into offline RL to mitigate overestimation.

2) It designs a scheme for generating negative trajectories and validates the algorithm's effectiveness through extensive experiments.

### Weaknesses
1) The underlying theory and mechanism explaining why introducing the preference learning framework alleviates overestimation are unclear.

2) The method for generating negative trajectories and the selection of the contrastive bias parameter vary across different tasks, and their impact on performance remains unknown.

3) The proof for Lemma 3.1 is not rigorous, as the approximation between $\hat{A}$ and $A^*$ is unreasonable.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents PREFORL (PREFerence-based Optimization for Offline RL), a novel contrastive preference learning framework designed to train robust policies from sparse-reward offline datasets. The method aims to bypass the core challenges of conventional offline RL, specifically the extrapolation error and overestimation bias that plague value-based methods in data-limited, sparse-reward settings.

### Strengths
1.  The key idea is fundamentally original: avoiding direct, unstable value function estimation (the common bottleneck in sparse-reward offline RL) by transforming the task into a more robust contrastive learning problem.

2. The introduction of two controlled degradation operators ($\mathcal{D}^{\perp a}$ and $\mathcal{D}^{\perp s}$) is a highly creative mechanism for generating meaningful synthetic negative examples.

### Weaknesses
1. For state-based degradation ($\mathcal{D}^{\perp s}$), the computation overhead associated with Nearest Neighbor Search is acknowledged to be non-negligible and potentially time-consuming for large-scale datasets. A brute-force, exact search is computationally prohibitive.

2. The performance of the action-based degradation method is highly sensitive to the choice of the noise variance ($\sigma$), which limits its robustness. This requires manual tuning per environment (or environment domain) to find the reasonably small number (e.g., $1\%$ to $2\%$) that maximizes the success rate.

### Questions
The CPL baseline is excluded from the Maze2D evaluation, with the justification that the dataset lacks unsuccessful trajectories. Given that PREFORL's core novelty is to augment these negative examples, can you make a direct comparison showing that CPL fails while PREFORL succeeds due to the synthetic degradation? It would have been a more powerful demonstration.

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
4

### Summary
This paper proposes PREFORL, a preference-based offline reinforcement learning method that addresses value overestimation in sparse-reward settings. The approach trains policies via contrastive learning between successful demonstrations and synthetic degraded trajectories generated through action perturbation or state-based substitution. Core contributions include: A degradation framework augmenting sparse offline datasets, (2) A preference optimization loss bypassing explicit value estimation, and (3) Theoretical analysis linking the loss to policy imitation. Evaluations on Adroit, Sparse-MuJoCo, Maze2D, and MetaWorld benchmarks show PREFORL outperforms offline RL/imitation baselines in success rates and normalized scores.

### Strengths
- Novel degradation framework towards both action and state level.
- Comprehensive experimental validation.
- Complete theoretical analysis.
- Good writing for easy understanding.

### Weaknesses
- More navigation tasks (e.g. Antmaze-umaze/medium/large-diverse/replay), as well as offline RL baselines, should be performed. 
- The ablation study of the degraded dataset size is lacking.

### Questions
- The proposed paradigm includes both a plug-in data-augmentation pipeline and a corresponding contrastive training pipeline. Can this paradigm be implemented on other BC-based offline-RL methods like Decision Transformer[1]?  I am glad to see how this paradigm performs on the DT backbone with different scales of data and models.
- Why is the state-degradation dataset constructed with the nearest neighbor state instead of directly adding noise like the action-degradation? I think there should be a comparison study of these two different state-degradation manners.

[1] Decision Transformer: Reinforcement Learning via Sequence Modeling

### Soundness
2

### Presentation
3

### Contribution
2
