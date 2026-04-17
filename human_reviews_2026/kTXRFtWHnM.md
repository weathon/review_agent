# Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Cross-domain reinforcement learning (CDRL) is meant to improve the data efficiency of RL by leveraging the data samples collected from a source domain to facilitate the learning in a similar target domain. Despite its potential, cross-domain transfer in RL is known to have two fundamental and intertwined challenges: (i) The source and target domains can have distinct state space or action space, and this makes direct transfer infeasible and thereby requires more sophisticated inter-domain mappings; (ii) The transferability of a source-domain model in RL is not easily identifiable a priori, and hence CDRL can be prone to negative effect during transfer. In this paper, we propose to jointly tackle these two challenges through the lens of \textit{cross-domain Bellman consistency} and \textit{hybrid critic}. Specifically, we first introduce the notion of cross-domain Bellman consistency as a way to measure transferability of a source-domain model. Then, we propose $Q$Avatar, which combines the Q functions from both the source and target domains with an adaptive hyperparameter-free weight function. Through this design, we characterize the convergence behavior of $Q$Avatar and show that $Q$Avatar achieves reliable transfer in the sense that it effectively leverages a source-domain Q function for knowledge transfer to the target domain. Through experiments, we demonstrate that $Q$Avatar achieves favorable transferability across various RL benchmark tasks, including locomotion and robot arm manipulation. Our code is available at https://rl-bandits-lab.github.io/Cross-Domain-RL/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies cross-domain transfer RL: the source and the target MDPs could differ in both state and action spaces. The transfer could be infeasible in that good inter-domain mappings may not exist. It introduces QAvatar, an algorithm that uses a convex combination of the Q-values from the source and the Q-values from the learned critic in the target to update the target policy. They show the efficacy of their method through locomotion, manipulation, and goal navigation tasks.

### Strengths
- The paper is mostly well-written.
- Cross-domain transfer in RL is an interesting problem, and the authors tackle it with a novel approach.
- The algorithm, QAvatar, is principled and is motivated by theoretical insights.

### Weaknesses
- There are discrepancies between what the theory suggests and what the practical version of QAvatar does: while $\alpha_t$ suggested by the theory involves norm defined w.r.t. visitation distributions of the policy at time t (line 332), empirically, the authors use the entire replay buffer for it (lines 1315 - 1318). 
- In the experiments, other CDRL baselines considered by the authors perform worse than learning from scratch (SAC) and direct finetuning (FT).  Did the authors optimize the hyperparameters for these baselines?
- While QAvatar is theoretically motivated, the method as such does not come with any guarantees. In this setting, one would be interested in sample complexity guarantees: the proposed algorithm is provably more sample efficient than learning from scratch. The authors do not provide guarantees of this form. But I understand that this might be beyond the scope of the paper.

### Questions
See weaknesses.

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
4

### Summary
The paper proposes an interesting and novel approach with solid mathematical proof and promising empirical performance.
The paper proposes cross‑domain Bellman consistency $\delta$ and QAvatar method. By learning a internal mapping,  the $Q_{src}$ aligns with the target-domain transitions and use a hybrid critic that interpolates target and mapped‑source Q values with a closed‑form weight $\alpha$. The authors prove an average sub‑optimality bound and show empirical improvements on tasks with different state/action dimensionalities, including locomotion, manipulation, and navigation.

### Strengths
1. Clean problem framing for CDRL with mismatched spaces; explicit transferability notion via a Bellman‑style residual
2. Hybrid critic that can down‑weight a poor source, limiting negative transfer
3. A formal tabular NPG analysis that separates learning progress and approximation error
4. The experimental evaluation is comprehensive and convincing, covering diverse tasks and challenging transfer scenarios

### Weaknesses
1. The theory-practice gap is substantial: the convergence analysis assumes on-policy, tabular NPG, while the implementation is off-policy, deep RL with replay buffers and twin Q-networks. The critical assumptions of the bound (e.g., on-policy error norms) do not hold in the implemented algorithm.

2. The claim of being “hyperparameter-free” is overstated: the update frequency $N_\alpha$ for the adaptive weight acts as a hidden hyperparameter, and no ablation study is provided.

3. The loss to update the mapping $\phi, \psi$ Eqn 5 (unquared) and Eqn 73 (squared) are mismatch

4. mior format issue: The title of tables houle be ahead of the table

### Questions
Same as weakness

additional questions:

Can you add ablation test regards to the update coefficient $N_\alpha$? Is the performance sensitive to the udpate frequency?

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
This paper addresses the problem of cross-domain reinforcement learning (CDRL) where the source and target domains have different state and action spaces. The authors identify two key challenges: (1) learning the mapping between these distinct spaces, and (2) avoiding "negative transfer" when the source-domain model is not actually beneficial for the target task. The core ideas include cross-domain Bellman Consistency, hybrid critic, hyperparameter-free weighting, and mapping via Bellman Loss.

The authors provide a theoretical analysis in the tabular NPG setting to show that QAvatar achieves a good sub-optimality bound that gracefully degrades to the standard NPG bound in cases of negative transfer. They then present a practical implementation using SAC and normalizing flows. Experiments on MuJoCo, Robosuite, and Safety-Gym tasks show that QAvatar significantly outperforms other CDRL baselines.

### Strengths
- The central concept of using a hybrid critic weighted by a measure of "transferability" is very clever. The automatic, hyperparameter-free weighting scheme, which pits the target TD error against the cross-domain Bellman error, is an elegant solution to the problem of negative transfer.
- Learning the state-action mappings by minimizing the cross-domain Bellman loss (a reward- and dynamics-aware objective) is a strong alternative to unsupervised, dynamics-agnostic methods like cycle-consistency. The toy example in Appendix D.1 provides a good motivation for this.
- The method shows consistent and significant sample efficiency improvements over all baselines (SAC, FT, PAR, CAT, CMD) across a good variety of cross-domain tasks (locomotion, manipulation, navigation) with differing state/action spaces.

### Weaknesses
- The practical implementation of QAvatar is very complex. It requires running a full SAC algorithm while also training two normalizing flow models for the state and action mappings, and also calculating two separate Bellman errors (target and cross-domain) at every step (or every $N_\alpha$ steps) to compute the weight. The paper notes this results in "about twice the training time of SAC," which is a major practical limitation and cost.
- The paper uses normalizing flows for the mappings, which works for the low-dimensional state/action spaces in MuJoCo and Robosuite. It is not clear how this approach would scale to high-dimensional state spaces, such as images, where learning a normalizing flow is significantly more challenging.
- Stability of Mapping Loss: The mapping functions $(\phi, \psi)$ are trained to minimize a crossdomain Bellman loss (Eq. 5) that depends on both the fixed $Q_{s r c}$ and the changing target policy $\pi^{(t)}$. This creates a complex, non-stationary optimization problem for the mappings. It's unclear how stable this is, especially if $Q_{s r c}$ is sub-optimal or the policy is exploring.

### Questions
- The computational cost of doubling training time (per Section 6) is a significant drawback. Could you discuss the trade-off? At what point does the 2x computational cost per step outweigh the ~2x sample efficiency gain?
- The mapping functions are trained to minimize a Bellman loss dependent on $Q_{\text {src }}$. What happens if $Q_{s r c}$ is sub-optimal? Does the loss find the "correct" physical mapping, or does it find a distorted mapping that simply makes the sub-optimal $Q_{\text {src }}$ look consistent with the target dynamics?

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
3

### Summary
The paper introduces QAvatar, a hybrid-critic framework for cross-domain reinforcement learning (CDRL), which targets transfer scenarios where source and target domains possess distinct state and/or action spaces mismatch and the transferability of source models is hard to measure. Experiments demonstrate substantial empirical improvements of QAvatar on diverse RL benchmarks with both positive and negative transfer cases.

### Strengths
This paper proposes the formal definition of cross-domain Bellman consistency and its incorporation into the analysis of transferability represents a strong conceptual advance over prior methods. 

The paper presents a detailed theoretical analysis with proofs supporting the soundness of QAvatar’s adaptive weighting scheme.

### Weaknesses
Although the hybrid critic can theoretically downweight poor source Q-functions, the effectiveness of knowledge transfer fundamentally depends on the quality of the state/action mapping functions $\phi$ and $\psi$. When mapping is highly ambiguous or misaligned, both the source-supplied and hybrid Q-values may be misleading.

While this paper classifies major approaches and discusses adversarial alignment and cycle consistency, the comparison or discussion with alternative alignment strategies (e.g., [1-3]) is lacking. 

It is hard to tell how the proposed method would perform in high-dimensional practical tasks. 

There is little discussion regarding the challenges posed by partial observability or scaling to dozens of source-target pairs.

[1] Cross Domain Policy Transfer with Effect Cycle-Consistency.

[2] Cross-Domain Policy Transfer by Representation Alignment via Multi-Domain Behavioral Cloning.

[3] Cross-Domain Knowledge Transfer for RL via Preference Consistency.

### Questions
In Figure 1, while QAvatar outperforms in terms of sample efficiency, the gap with FT or CAT-SAC is at times moderate in locomotion (HalfCheetah task). A deeper dissection of where relative benefits are highest would be useful.

Can the authors provide empirical results or qualitative diagnostics assessing when the learned mappings $\phi$ and $\psi$ are well aligned?  What if the mapping is misleading, would the proposed method avoid collapse?

Have the authors tested QAvatar’s hybrid critic and mapping approach on image-based RL or other high-dimensional observation settings? If not, do they foresee major obstacles?

### Soundness
3

### Presentation
3

### Contribution
2
