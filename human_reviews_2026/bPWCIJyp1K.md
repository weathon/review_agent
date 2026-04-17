# Efficient Offline Reinforcement Learning via Peer-Influenced Constraint

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Offline reinforcement learning (RL) seeks to learn an optimal policy from a fixed dataset, but distributional shift between the dataset and the learned policy often leads to suboptimal real-world performance. Existing methods typically use behavior policy regularization to constrain the learned policy, but these conservative approaches can limit performance and generalization, especially when the behavior policy is suboptimal. We propose a Peer-Influenced Constraint (PIC) framework with a ``peer review" mechanism. Specifically, we construct a set of similar states and use the corresponding actions as candidates, from which we select the optimal action to constrain the policy. This method helps the policy escape local optima while approximately ensuring the staying within the in-distribution space, boosting both performance and generalization. We also introduce an improved version, Ensemble Peer-Influenced Constraint (EPIC), which combines ensemble methods to achieve strong performance while maintaining high efficiency. Additionally, we uncover the Coupling Effect between PIC and uncertainty estimation, providing valuable insights for offline RL. We evaluate our methods on classic continuous control tasks from the D4RL benchmark, with both PIC and EPIC achieving competitive performance compared to state-of-the-art approaches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new regularizer for policy optimization in offline RL called PIC. Unlike conventional approaches that constrain the policy toward the specific action observed for a given state, PIC encourages the policy to align with actions drawn from a set of similar (peer) states, which can be interpreted as a relaxation of the conventional approaches. The experiment results show the proposed method achieves state-of-the-art results on D4RL benchmarks.

### Strengths
- The proposed method is intuitive and tackles a valid limitation of the commonly used policy regularizer in offline RL.
- The paper provided a theoretical justification of the proposed approach.
- The paper achieves strong empirical results across multiple tasks in the D4RL benchmark.
- The proposed method can be integrated into other (offline) RL methods like TD3 or IQL.

### Weaknesses
- The peer identification relies on norm distance in the raw state space, which may not align with semantic similarity in high-dimensional (for example, pixel) environments. The proposed method should be tested in such environments.
- Figure 4 (c) seems to show that the proposed method is somewhat sensitive to the hyperparameter delta.
- Table 5 shows the proposed method used different hyperparameters for each task in D4RL Mujoco. However, as far as I know, the baselines like CQL and IQL used the same hyperparameters for the Mujoco tasks. Therefore, this is not a fair comparison.

### Questions
- In Equation (7), it seems when j=1, \hat{s}_j will always be s. Is this correct?
- In Equation (9), why are alpha and delta both required? To my understanding, only one of the two is required to adjust the importance between the two terms.

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
The paper proposes Peer-Influenced Constraint (PIC) and its ensemble extension EPIC for efficient offline reinforcement learning. The key idea is to exploit peer states within the offline dataset to form candidate action sets, allowing the policy to learn from similar states and select the highest-valued in-distribution action. When combined with ensembles, the method exhibits a Coupling Effect between peer constraint strength and uncertainty estimation, enabling robust OOD suppression with smaller ensembles.

Importantly, EPIC can be viewed as an evolution of ensemble-based offline RL methods such as EDAC. While EDAC enhances SAC-N by diversifying ensemble critics to reduce overestimation bias, it still requires large ensembles (often 10–50 critics) and substantial compute.
EPIC inherits EDAC’s ensemble foundation but integrates a PIC that reuses dataset structure — effectively reducing the reliance on large ensembles and improving generalization, **leading to performance enhancement in larger state-action domain**. This synergy between behavioral regularization and value uncertainty represents a notable advancement in the design of efficient, uncertainty-aware offline RL.

### Strengths
**Strengths**

* **Clear conceptual advance over EDAC**: achieves comparable or better uncertainty calibration with significantly fewer critics through the Coupling Effect, offering a well-grounded improvement in efficiency without sacrificing performance.

* **Innovative peer-state reuse**: effectively leverages intra-dataset structure to improve generalization beyond strict behavior cloning. Moreover, the study convincingly demonstrates that such peer-state reuse directly contributes to reducing the required ensemble size while maintaining stable OOD suppression.

* **Insightful analysis of the PIC–ensemble interaction**: I particularly appreciate the experiment in Figure 4(d), which clearly visualizes how the effect of PIC varies with ensemble size. The results show that while PIC’s contribution may saturate under very large ensembles, it plays a critical role in the small-ensemble regime—precisely where efficiency and computational savings matter most. This ablation provides strong empirical evidence that the proposed Coupling Effect is both interpretable and practically beneficial, reinforcing the paper’s claim of achieving “efficient” offline RL.

### Weaknesses
**Weakness**

While the motivation experiment in Figure 1 illustrates the behavioral difference between TD3+BC and PIC, the chosen environment—a small and discrete gridworld—may not be fully representative of the setting where the proposed peer-influenced constraint operates most effectively.
The key assumption behind PIC is that nearby states tend to share similar optimal actions, which implicitly presumes a degree of continuity and smoothness in both the state and action spaces.
Such an assumption is reasonable in continuous and high-dimensional control domains (e.g., MuJoCo tasks) but may not hold in discrete or highly non-smooth environments, where small state changes can correspond to qualitatively different optimal actions.
Clarifying this assumption early in the paper would help readers understand the scope and applicability of PIC.

### Questions
**Questions relative to weakness mentioned earlier**

- Could the authors elaborate on the assumptions underlying the peer-influenced constraint—specifically, the degree of state–action continuity required for the method to remain valid? My understanding is that PIC implicitly assumes that nearby states tend to share similar optimal actions, which is reasonable in continuous and smooth control domains. However, this assumption may not hold in small or discrete environments such as the one illustrated in Figure 1. Especially, this could have negative impression of selecting OOD action which is prohibited in offline RL setting. If this interpretation is correct, clarifying such assumptions in the paper would strengthen its scope and applicability. Otherwise, please correct me if I am mistaken in my understanding of Figure 1 and the intended domain of applicability.

- How would PIC behave in discrete or non-smooth domains where neighboring states may not share similar optimal actions?

 - Would the method need to adapt (e.g., via learned similarity metrics or representation smoothing) to remain effective in such cases?

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
This paper addresses the distributional shift problem in offline reinforcement learning, where policies trained on fixed datasets often fail to generalize. It proposes a novel policy regularizer called Peer-Influenced Constraint (PIC), which guides the learned policy using actions from similar "peer" states found within the dataset. The paper also identifies a "Coupling Effect" between this policy constraint and the uncertainty estimation of ensemble-based methods. Based on these ideas, it introduces EPIC (Ensemble Peer-Influenced Constraint), a method that integrates PIC into an ensemble framework to achieve strong performance with higher computational efficiency. The methods are evaluated on standard D4RL benchmarks, where they demonstrate state-of-the-art results.

### Strengths
S1: The core concepts of Peer-Influenced Constraint (PIC) and the "Coupling Effect" are novel, intuitive, and provide a valuable new perspective on balancing policy and value regularization in offline RL.
S2: The paper is supported by extensive and rigorous experiments on standard D4RL benchmarks (Gym-MuJoCo, AntMaze, Adroit), demonstrating strong, state-of-the-art performance across multiple task suites.   
S3: The inclusion of targeted experiments on generalization, computational efficiency, and offline-to-online fine-tuning provides a comprehensive evaluation of the proposed methods.
S4: The paper is clearly written, and the motivation is well-illustrated, particularly with the introductory gridworld example that effectively contrasts the proposed approach with baselines like TD3+BC.

### Weaknesses
W1: The core mechanism of PIC relies on finding nearest neighbors using Euclidean distance in the raw state space. This approach is known to be ineffective in high-dimensional spaces (e.g., image-based observations) due to the "curse of dimensionality," where distances between points become less meaningful. This severely limits the method's scalability and general applicability. 
W2: The method appears to be highly sensitive to hyperparameters, which undermines its claim as a simple "plug-in" module. In particular, the hyperparameter η is set to 200 for all Adroit tasks but 1 for all others—a 200x difference that is not explained or analyzed in the parameter study. This suggests that achieving good performance may require extensive, task-specific tuning, a known issue that can obscure the true robustness of an algorithm.

### Questions
1.Could the authors please explain the rationale for the 200x difference in the hyperparameter η between the Adroit tasks (η=200) and all other tasks (η=1)? why do the dexterous Adroit tasks require such extreme critic diversity regularization?Why was this parameter excluded from the sensitivity analysis in Section 4.4 and F.6?   
Could the authors provide more intuition on the mechanism behind the "Coupling Effect"? Specifically, why does constraining the policy to a high-value in-distribution action cause the Q-function ensemble to exhibit higher variance on OOD actions？

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Peer-Influenced Constraint (PIC), a plug-in policy regularizer for offline RL that, for a query state, retrieves K nearest “peer” states from the dataset, collects their associated actions as candidates, and selects the best candidate by a critic (min over critics) to constrain the actor toward high-value in-distribution actions. The authors further introduce EPIC (Ensemble PIC), showing a claimed “Coupling Effect” between PIC strength and uncertainty estimation that purportedly enables smaller ensembles without losing performance. Experiments on D4RL (Gym-MuJoCo, AntMaze, Adroit) report competitive or SOTA results and improved efficiency vs. large-ensemble methods. Key ingredients include a KD-Tree for fast peer retrieval, a formal performance-gap bound under Lipschitz assumptions, and ablations over K, δ (PIC strength), and N (ensemble size).

### Strengths
1. Practical, plug-and-play idea. PIC cleanly reuses cross-state action candidates to relax over-conservative behavior-cloning constraints while staying in-distribution; integrates with TD3/SAC/IQL. 

2. Clear algorithmic specification. Definition of PIC distance and candidate selection with min-critic action choice is straightforward; EPIC generalizes this to N critics.  

3. Empirical coverage & results. Extensive D4RL results show PIC-TD3 competitive with ensemble-free baselines and EPIC surpassing baselines on many tasks (Gym-MuJoCo, AntMaze, Adroit).  

4. Efficiency angle. Claimed ability to use smaller ensembles while maintaining performance addresses a common pain-point in offline RL. Parameter studies (K, δ, N) are helpful for practice.

### Weaknesses
1. State-space similarity & representation. PIC hinges on Euclidean nearest neighbors in the raw state space (KD-Tree). In many high-dimensional or poorly scaled domains, Euclidean distance can be misleading; the paper lacks experiments comparing learned/state-normalized metrics vs. raw features, and ablations on feature scaling or representation robustness. 

2. Coupling-effect clarity. The section discussing how δ interacts with uncertainty occasionally reads inconsistently (e.g., whether increasing δ “raises” uncertainty vs. “reduces” overestimation leading to “lower” uncertainty). This needs a tighter, causal explanation and clearer metrics (Qmin/Qstd/Qclip) across datasets/time. 

3. Theoretical assumptions are strong/generic. The Lipschitz assumptions and the bound provide qualitative reassurance but do not uniquely characterize PIC’s effect beyond standard smoothness arguments. No finite-sample or function-approximation analysis is provided to justify behavior under approximate critics. 

4. Scalability of neighbor search. KD-Tree is efficient in moderate dimensions but can degrade with very large |D| and high-dimensional S; the paper acknowledges this limitation without proposing practical approximations (e.g., ANN indices, learned retrieval).

### Questions
1. Distance metric & normalization. What exact preprocessing/normalization is used before KD-Tree (per-dimension scaling, whitening, PCA)? Have you tried learned embeddings (e.g., representation pretraining) for neighbor search, and how does that affect performance vs. raw states? 

2. Coupling effect mechanics. Please reconcile the narrative around δ’s effect on uncertainty. Under fixed N, does increasing δ increase or decrease Q-uncertainty on OOD actions in practice? Can you provide a controlled analysis (same seeds, same checkpoints) that tracks policy action-distribution shift and critic variance step-by-step?  

3. Critic dependence & bias. Since a* is chosen via min over critics, how sensitive is PIC/EPIC to critic underestimation bias? Provide an ablation using single-critic and target-action noise or double Q with clipped targets to test robustness.  

4. Candidate-set composition. Beyond nearest states, did you explore wider but weighted candidate pools (e.g., radius-based retrieval with distance-weighted selection), and what is the trade-off vs. compute? Any results on K beyond 50 and on adaptive K per state? 5. Generalization diagnostics. In AntMaze/Adroit, can you report how often the selected a* comes from (i) the current state vs. (ii) peers, and the distance of chosen peers? This would concretely show when PIC escapes local optima rather than re-selecting the behavior action.

### Soundness
2

### Presentation
3

### Contribution
2
