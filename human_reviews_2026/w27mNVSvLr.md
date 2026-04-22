# Cross-domain Offline Policy Adaptation with Dynamics- and Value-aligned Data Filtering

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Cross-Domain Offline Reinforcement Learning aims to train an agent deployed in the target environment, leveraging both a limited target domain dataset and a source domain dataset with (possibly) sufficient data coverage. Due to the underlying dynamics misalignment between the source and target domain, simply merging the data from two datasets may incur inferior performance. Recent advances address this issue by selectively sharing source domain samples that exhibit dynamics alignment with the target domain.  However, these approaches focus solely on dynamics alignment and overlook \textit{value alignment}, i.e., selecting high-quality, high-value samples from the source domain. In this paper, we first demonstrate that both dynamics alignment and value alignment are essential for policy learning, by examining the limitations of the current theoretical framework for cross-domain RL and establishing a concrete sub-optimality gap of a policy trained on the source domain and evaluated on the target domain. Motivated by the theoretical insights, we propose to selectively share those source domain samples with both high dynamics and value alignment and present our Dynamics- and Value-aligned Data Filtering (DVDF) method. We design a range of dynamics shift settings, including kinematic and morphology shifts, and evaluate DVDF on various tasks and datasets, as well as in challenging extremely low-data settings where the target domain dataset contains only 5,000 transitions.  Extensive experiments demonstrate that DVDF consistently outperforms prior strong baselines and delivers exceptional performance across multiple tasks and datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Cross-domain offline RL aim to train an agent for the target domain using data from both source and target domains. Previous methods selectively sharing source domain samples that exhibit dynamic alignment with the target domain. This paper argues that focusing solely on dynamics alignment overlooks value alignment, the selection of high-quality, high-value samples from the source domain. It proposes a method to selectively utilize these source domain samples with both high dynamic and value alignment, using a combination of a dynamic score and an advantage function. This approach results in the Dynamics- and Value-aligned Data Filtering (DVDF) method. Theoretical analysis and experiments are conducted to validate the effectiveness of this method.

### Strengths
- The paper is clearly motivated and well-written, with a logical structure that guides readers through the problem setup and proposed solution.

- The proposed method is reasonable, experiments demonstrate the effectiveness of the method.

### Weaknesses
- The practical implementation introduces several approximations to the analysis, which prevents the value misalignment term from being controlled in practice. Additionally, the introduction of $\hat{A}$ and SQL is not very clear to me. For instance, it is unclear why an additional advantage function is needed, and why SQL outperforms IQL.

- In principle, a more accurate advantage is better. While the advantages of SQL and IQL shown in Figure 2(a) indicate that SQL’s advantage is higher than IQL’s, this does not mean SQL achieves a more accurate advantage. Furthermore, Figure 2(b) exhibits high variance with only 5 seeds, making the results unconvincing.

- The score function introduces a parameter lambda that trades off value alignment and dynamic alignment. However, the experiments in Figure 3 do not reflect the importance of value alignment. There are no significant differences in the results (ranging from ~90 to ~110, and ~60 to ~70), and high variance with only 5 seeds further obscures these minor differences.

- The ablation study lacks some important components that would help better understand the method. For example, Figure 2 omits Solely dynamics, and Figure 3 omits the cases of lambda=1 and lambda=0. Moreover, in my view, Figure 2(b) and Figure 3(a) could be combined into a single figure.

### Questions
- The proposed method includes two terms for selecting source-domain data. I am curious how differences in dynamics affect the importance of these two terms. Furthermore, is there a relationship between parameter selection and dynamics differences?

- As a plug-in module, integrating DVDF into IGDF and OTDF leads to performance degradation on certain tasks—such as half-r and ant-mr. What is the reason for this?

- The value misalignment term only relates to the source domain. Does this mean the expert dataset will have more samples selected than the random and medium datasets? Moreover, if the dataset is expert-level, does this imply that reinforcement learning algorithms (whether IQL or SQL) are unnecessary, and supervised learning alone is sufficient?

- In my understanding, VGDF is an implicit combination of value alignment and dynamic alignment. How should we interpret the paper’s claim that "VGDF still only addresses dynamics mismatch"?

- Why do SQL provide more reliable advantage estimates than IQL? Additionally, how do other offline RL methods such as extreme Q-learning and in-sample actor-critic perform in this regard?

- See Weaknesses.

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
This paper proposes a Dynamics- and Value-aligned Data Filtering (DVDF) method, which can selectively share those source domain samples with both high dynamics and value alignment.

### Strengths
* From a theoretical perspective, this paper reveals that the existing theoretical framework that tightens the performance discrepancy of a given policy between the source and target domains misaligns with the RL objective, and fails to guarantee learning a well-performing target policy.

* DVDF trades off the dynamics and value misalignment and selectively shares source domain samples to train the policy.

* DVDF can be generally treated as a plug-in module and seamlessly integrated with recent methods like IGDF and OTDF.

### Weaknesses
* The paper does not clearly explain how the advantage function is obtained.

* The experimental evaluation lacks results on lower-quality datasets.

### Questions
* It would be helpful to clarify whether the advantage function is affected by overestimation and how accurately it can be estimated in practice.

* Have the authors considered validating the proposed approach through real-world robot experiments?

* Could the authors include comparisons and a discussion with recent cross-domain offline RL studies (e.g., PSEC, DmC)? Incorporating these baselines and analysing the differences would strengthen the paper and better position the proposed method within the current literature.

Reference:

Liu, T., Li, J., Zheng, Y., Niu, H., Lan, Y., Xu, X., Zhan, X. Skill expansion and composition in parameter space. In International Conference on Learning Representations, 2025.

Van, L. L. P., Nguyen, M. H., Kieu, D., Le, H., Tran, H. T., & Gupta, S. DmC: Nearest Neighbor Guidance Diffusion Model for Offline Cross-domain Reinforcement Learning. arXiv preprint arXiv:2507.20499.

### Soundness
3

### Presentation
2

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
This paper studies cross-domain offline RL. They showed that solely focusing on dynamics alignment as existing works do is not efficient, and illustrated the importance of value alignment. Theoretical analysis derives an upper bound on cross-domain sub-optimality, showing the necessity of considering both alignments. They presented a method (DVDF) that combine both dynamics and value alignment in the policy, and empirically, DVDF improves performance over prior filtering methods.

### Strengths
The paper presents a solid theoretical framework that decomposes the sub-optimality gap into dynamics misalignment and value misalignment terms. To the best of my knowledge, this idea is novel, and the analysis is mathematically sound. The propositions and proofs extend previous results to a more general setting. Overall, the paper is clearly written and well organized.

### Weaknesses
1. The theoretical decomposition is elegant; however, the proposed algorithm does not appear to directly optimize or approximate the theoretical quantities introduced.
2. While the theory claims to address both dynamics and value misalignment across domains, the experiments are conducted on fixed D4RL datasets with nearly identical dynamics. As a result, the method effectively filters samples based on behavioral or value similarity rather than demonstrating adaptation across genuinely different MDPs. Prior works [1, 2] introduced synthetic dynamics perturbations to evaluate such cross-domain robustness.
3. The experimental design is relatively simple, and the ablations for each component are limited to only one or two examples. A more systematic empirical analysis would strengthen the paper’s claims.

> [1] Cross-Domain Offline Policy Adaptation with Optimal Transport and Dataset Constraint (Lyu et al., 2025)  
> [2] Contrastive Representation for Data Filtering in Cross-Domain Offline Reinforcement Learning (Wen et al., 2024)

### Questions
1. Can you clarify what specific notion of “value alignment” the paper refers to, and why this concept is theoretically distinct from existing objectives such as minimizing policy or Q-function discrepancy?
2. Equation 6 (overload notations): the $s'\in s_{tar}'\cup s_{src}'$ term looks weird. Why use the same small s to denote if s’ is a sampled state and the latter are state sets?
3. Section 5.2, line 305: g_{\xi%} is not explicitly defined. Is it the the top $\xi$-quantile of batch source sample set?
4. Table 2: the notation of data selection ratio may be inconsistent: l%, $\xi$.

### Soundness
3

### Presentation
3

### Contribution
3
