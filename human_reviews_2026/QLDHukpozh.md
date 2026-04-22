# OPRIDE: Efficient Offline Preference-based Reinforcement Learning via In-Dataset Exploration

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Preference-based reinforcement learning (PbRL) can help avoid sophisticated reward designs and align better with human intentions, showing great promise in various real-world applications. However, obtaining human feedback for preferences can be expensive and time-consuming, which forms a strong barrier for PbRL. In this work, we address the problem of low query efficiency in offline PbRL, pinpointing two primary reasons: inefficient exploration and overoptimization of learned reward functions. In response to these challenges, we propose a novel algorithm, Offline PbRL via In-Dataset Exploration (OPRIDE), designed to enhance the query efficiency of offline PbRL. OPRIDE consists of two key features: a principled exploration strategy that maximizes the informativeness of the queries and a discount scheduling mechanism aimed at mitigating overoptimization of the learned reward functions. Through empirical evaluations, we demonstrate that OPRIDE significantly outperforms prior methods, achieving strong performance with notably fewer queries. Moreover, we provide theoretical guarantees of the algorithm's efficiency. Experimental results across various locomotion, manipulation, and navigation tasks underscore the efficacy and versatility of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces OPRIDE, an offline PbRL framework designed to improve query efficiency and mitigate overoptimization. It comprises two key components: In-dataset exploration (IDE), which selects queries maximizing differences among value functions trained on bootstrapped reward models, and variance-based discount scheduling (VDS), which dynamically reduces the discount factor for high variance state-action pairs to mitigate overestimation and stabilize learning. The authors provide theoretical guarantees on query efficiency, and experiments on Meta-World and AntMaze benchmarks demonstrate that OPRIED achieves superior performance with only 10 queries compared to prior offline PbRL baselines.

### Strengths
1. The IDS mechanism is intuitive and well-motivated. Unlike disagreement-based methods that focus on reward uncertainty, IDS selects queries that maximize information gain about optimal policy by measuring value function difference. Table 3 empirically validates that IDS significantly outperforms random and disagreement-based selection.
2. The method achieves compelling performance with only 10 queries across diverse tasks (Tables 1 & 2, Figure 2), significantly outperforming baselines.
3. The paper provides thorough ablations examining IDS vs. baselines (Table 3), VDS effectiveness (Table 9), and sensitivity to hyperparameters (Tables 7, 10). These studies clearly demonstrate the contribution of each component.

### Weaknesses
1. This paper only compares two-stage methods such as OPRL and PT. However, recent advances in offline PbRL have introduced a distinct line of one-stage framework that learns policies without reward learning, including IPL [1] and CPL [2]. These approaches are generally more robust and often outperform two-stage reward learning frameworks, as they avoid the approximation errors introduced by learning a separate reward estimator. it does not suffer from approximation error in reward estimator learning. The authors should include comparisons with these one-stage methods to more convincingly demonstrate the superiority of the proposed approach.

2. Please also refer other two-stage offline PbRL studies, such as CLARIFY [3] and LiRE [4], and discuss how they related to the proposed method in the Related Works section.

3. Experiments with non-ideal and noisy annotator (real human) are missing. To validate the practicality of the proposed method in real-world scenarios, the authors should include experiments involving actual human feedback [3, 4].

4. The experiments are conducted only in vector-based environments. While most offline PbRL studies do not include experiments with proprioceptive (pixel-based state) inputs, evaluating such setting is essential for real-sensory and visual domains. Please refer to recent online PbRL studies that demonstrated effectiveness in pixel-based environments for this setting [5,6].

References

[1] Hejna, J., & Sadigh, D. (2023). Inverse preference learning: Preference-based rl without a reward function. Advances in Neural Information Processing Systems, 36, 18806-18827.

 [2] Hejna, J., Rafailov, R., Sikchi, H., Finn, C., Niekum, S., Knox, W. B., & Sadigh, D. Contrastive Preference Learning: Learning from Human Feedback without Reinforcement Learning. In The Twelfth International Conference on Learning Representations.

[3] Mu, N., Hu, H., Hu, X., Yang, Y., XU, B., & Jia, Q. S. CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries. In Forty-second International Conference on Machine Learning.

[4] Choi, H., Jung, S., Ahn, H., & Moon, T. (2024, July). Listwise Reward Estimation for Offline Preference-based Reinforcement Learning. In International Conference on Machine Learning (pp. 8651-8671). PMLR.

[5] Park, J., Seo, Y., Shin, J., Lee, H., Abbeel, P., & Lee, K. SURF: Semi-supervised Reward Learning with Data Augmentation for Feedback-efficient Preference-based Reinforcement Learning. In International Conference on Learning Representations.

[6] Metcalf, K., Sarabia, M., Mackraz, N., & Theobald, B. J. (2023, December). Sample-Efficient Preference-based Reinforcement Learning with Dynamics Aware Rewards. In Conference on Robot Learning (pp. 1484-1532). PMLR.

[7] Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 30, No. 1).

[8] Nikishin, E., Schwarzer, M., D’Oro, P., Bacon, P. L., & Courville, A. (2022, June). The primacy bias in deep reinforcement learning. In International conference on machine learning (pp. 16828-16847). PMLR.

[9] Coste, T., Anwar, U., Kirk, R., & Krueger, D. Reward Model Ensembles Help Mitigate Overoptimization. In The Twelfth International Conference on Learning Representations.

[10] Nauman, M., Bortkiewicz, M., Miłoś, P., Trzcinski, T., Ostaszewski, M., & Cygan, M. (2024, July). Overestimation, Overfitting, and Plasticity in Actor-Critic: the Bitter Lesson of Reinforcement Learning. In International Conference on Machine Learning (pp. 37342-37364). PMLR.

### Questions
1. How high variance in value estimation directly relates to overestimation? Overestimation can instead be measured explicitly as the gap between predicted Q-values and actual returns, or indirectly detecting spikes in streaming critic outputs [7,8].

2. If overoptimization and overestimation need to be mitigated, alternative strategies such as using minimum prediction among ensemble reward models, spectral normalization and periodic resets can serve as direct remedies [9, 10]. How does the proposed VDS compare to these existing techniques?

3. How would the proposed framework behave under noisy or inconsistent human feedback, which is common in real-world preference learning settings?

### Soundness
3

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
4

### Summary
This paper introduces OPRIDE, a novel algorithm designed to enhance query efficiency in offline PbRL. The authors demonstrate that OPRIDE achieves statistical efficiency by analyzing the suboptimality gap. Comprehensive experiments on Meta-World and D4RL benchmarks validate the superior performance of the proposed algorithm.

### Strengths
- The query efficiency is an important problem in offline PbRL.
- OPRIDE achieves SOTA in most tasks across Meta-World and Antmaze benchmarks.
- The proposed method is solid with theoretical guarantees.

### Weaknesses
- There are differences between the theoretical Algorithm 2 and the practical Algorithm 1; however, I fully understand that this simplification is necessary for theoretical analysis.

### Questions
- I'm intrigued by one observation: adding as little as 0.1% to 0.2% of additional data consistently improved performance without signs of leveling off. Could you clarify at what point this trend starts to plateau? For example, does the effect diminish around 1%, 3%, 5%, or even higher? At what point does adding more data stop producing meaningful gains? Additionally, could you provide any insight into why such a small fraction of data has such a significant impact?

Please include this experiment in the paper.

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
This paper proposes OPRIDE to improve query efficiency in offline PbRL through two main components: (1) in-dataset exploration that selects queries by maximizing value function disagreement between ensemble members, and (2) variance-based discount scheduling (VDS) that adjusts the discount factor based on ensemble prediction variance to mitigate reward overoptimization. The experimental evaluation demonstrates strong query efficiency with comprehensive ablation studies.

### Strengths
1. Proposed method demonstrates substantially higher query efficiency than prior work, achieving compelling results with only 10 queries across diverse domains.
2. The paper identifies and addresses reward overestimation in offline PBRL, an issue that has been relatively overlooked in previous studies. 
3. The use of critic-based query sampling, instead of reward model-based sampling, is intuitive and directly targets policy-relevant queries.

### Weaknesses
1. Missing baselines: While the paper compares OPRIDE with methods such as OPRL and PT, it does not include more recent and orthogonal approaches that bypass reward modeling (IPL [1], DPPO [2], and CPL [3]) or enhance query sampling with sequential ranked lists (LiRE[4]). The lack of these comparisons weakens the experimental contribution.

2. Concerns about iterative learning: Unlike conventional two-stage offline PbRL methods that learn V and Q only once during policy extraction, OPRIDE iteratively trains V and Q during reward learning. This may introduce additional computational overhead compared to standard two-stage approaches. Furthermore, plasticity loss and capacity loss of neural networks may prevent both reward and corresponding value estimators from properly adapting to updated reward estimates as new preference data arrive [5, 6].

3. Limited experimental scope: All experiments rely on vector-based state inputs, which limits practical applicability in pixel-based environments. In addition, considering the focus of PbRL, the paper also should provide actual human-in-the-loop experiments.

4. Scalability concerns with mandatory ensembles: The method depends on ensemble reward models to compute value difference and measure variance. While ensemble can improve robustness, this requirement become impractical when reward models are very large, limiting scalability. The paper uses only M=2, but even this may be prohibitive in certain real-world settings.

[1] Hejna, J., & Sadigh, D. (2023). Inverse preference learning: Preference-based rl without a reward function. Advances in Neural Information Processing Systems, 36, 18806-18827.  
[2] An, G., Lee, J., Zuo, X., Kosaka, N., Kim, K. M., & Song, H. O. (2023). Direct preference-based policy optimization without reward modeling. Advances in Neural Information Processing Systems, 36, 70247-70266.  
[3] Hejna, J., Rafailov, R., Sikchi, H., Finn, C., Niekum, S., Knox, W. B., & Sadigh, D. Contrastive Preference Learning: Learning from Human Feedback without Reinforcement Learning. In The Twelfth International Conference on Learning Representations.  
[4] Choi, H., Jung, S., Ahn, H., & Moon, T. (2024, July). Listwise Reward Estimation for Offline Preference-based Reinforcement Learning. In International Conference on Machine Learning (pp. 8651-8671). PMLR.  
[5] Lyle, C., Rowland, M., & Dabney, W (2022). Understanding and Preventing Capacity Loss in Reinforcement Learning. In International Conference on Learning Representations.  
[6] Lyle, C., Zheng, Z., Nikishin, E., Pires, B. A., Pascanu, R., & Dabney, W. (2023, July). Understanding plasticity in neural networks. In International Conference on Machine Learning (pp. 23190-23211). PMLR.

### Questions
1. Interpretation of survival instinct: The authors argue that using datasets with only ~5% perturbed trajectories makes their setup more challenging than [1]. However, this seems contradictory: survival instinct tends to strengthen when expert-like trajectories dominate, making learning easier under misspecified rewards. Thus, a low perturbation ratio likely makes the task easier, not harder. Clarifying this conceptual mismatch would improve the paper’s experimental claims.

2. Connection between Eluder dimension and proposed method: What is the specific connection between Eluder dimension in Theorem 4 and the practical algorithm (Algorithm 1)?

3. Variance and overestimation relationship: How does high variance in value estimation directly lead to overestimation? Overestimation is typically measured as the gap between predicted Q-values and actual discounted sum of rewards under the same reward function, or judged via expected Q-values. High variance reflects uncertainty but does not necessarily imply overestimation.

4. Discount factor adjustment and overestimation: How does adjusting the discount factor in VDS specifically mitigate overestimation? Section 3.2 refers to pessimism, but the mechanism is unclear. Reducing the discount factor may induce more myopic value estimates rather than truly mitigating overestimation. Does this adjustment simply reduce horizon length, or is there a more fundamental connection?

5. Heuristic nature of m% threshold: Using a static threshold (Top m%) to identify high-variance samples for discount adjustment is quite heuristic. What principled improvements could be made?

6. Scalability concern (related to Weakness 4) : What solutions exist when for cases where only a single reward model can be used due to computational constraints?

[1] Li, A., Misra, D., Kolobov, A., & Cheng, C. A. (2023). Survival instinct in offline reinforcement learning. Advances in neural information processing systems, 36, 62062-62120.

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
2

### Summary
This paper proposes Offline PbRL via In-Dataset Exploration (OPRIDE), a novel algorithm designed to systematically enhance the query efficiency of offline PbRL. OPRIDE introduces a principled exploration strategy that identifies the most informative queries by analyzing value differences between trajectories, ensuring that each query maximally contributes to learning the optimal policy. Additionally, to prevent overoptimization of the learned reward function, particularly in regions with high uncertainty, OPRIDE incorporate a discount factor scheduling mechanism that dynamically adjusts the discount based on the variance in the reward estimation. Experiments are performed on diverse locomotion and manipulation tasks, including AntMaze and Meta-World. Results compared to existing baselines demonstrates improvements in query efficiency.

### Strengths
1. The method shifts query generation from purely reward-based uncertainty to policy-level uncertainty, focusing on the difference of value differences. This provides a more direct way to improve policy-relevant information efficiency compared to standard reward uncertainty sampling.
2. Using multiple reward and value function estimates improves robustness and helps quantify epistemic uncertainty in offline settings where exploration is limited.
3. Dynamically adjusting the discount factor based on value variance provides a practical, adaptive mechanism to mitigate reward overestimation and stabilize learning in noisy preference data.
4. Authors conducted comprehensive experimental evaluations, which demonstrate overall significant improvements compared to baselines.

### Weaknesses
1. Potential instability: Adaptive discount factor modulation could introduce nonstationarity in learning dynamics, especially when variance estimates fluctuate across batches.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2
