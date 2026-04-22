# Preference Conditioned Multi-Objective Reinforcement Learning: Decomposed, Diversity-Driven Policy Optimization

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Multi-objective reinforcement learning (MORL) aims to optimize policies in environments with multiple, often conflicting objectives. While a single, preference-conditioned policy offers the most flexible and efficient solution, existing methods often struggle to cover the entire spectrum of optimal trade-offs. This is frequently due to two underlying challenges: destructive gradient interference between conflicting objectives and representational mode collapse, where the policy fails to produce diverse behaviors. In this work, we introduce  $D^3PO$, a novel algorithm that trains a single preference conditioned policy to directly address these issues. Our framework features a decomposed optimization process to encourage stable credit assignment and a scaled diversity regularizer to explicitly encourage a robust mapping from preferences to policies. Empirical evaluations across standard MORL benchmarks show that $D^3PO$ discovers more comprehensive and higher-quality Pareto fronts, establishing a new state-of-the-art in terms of hypervolume and expected utility, particularly in complex and many-objective environments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work introduced a so-called Decomposed, Diversity Driven Policy Optimization (D3PO) method for scalarized multi-objective PPO. It proposed Late-Stage Weighting for better learning signal, and Scaled Diversity Regularization for representational mode collapse prevention

### Strengths
1. Good empirical results for the tested environments.
2. The method design is generally reasonable

### Weaknesses
1. While representational mode collapse is indeed a key factor limiting full coverage of the optimal trade-off spectrum, linear scalarization (LS) methods inherently suffer from incomplete Pareto front discovery [1,2,3], even with a perfect representational mode. Therefore, since D3PO is also an LS-based method, its performance on FruitTree is unlikely to surpass that of GPI-LS.

2. The additional KL-divergence objective might serve as a regularization tern that making the Pareto front more convex[1], which is an interesting point to analyze but ignored by this paper.
 
3. KL-divergence is not commutative, meaning that $D_{KL}(\pi_A||\pi_B)!=D_{KL}(\pi_B||\pi_A)$, then how can you deal with the relation among $D_{KL}(\pi_A||\pi_B),D_{KL}(\pi_B||\pi_A),|w_A-w_B|$

4. The issue of mitigating destructive gradient interference is not novel. This work only partially addresses it through Late-Stage Weighting, whereas prior studies [3] have proposed more principled conflict-avoidance mechanisms that are studied in optimization literatures [4,5]

5. The theoretical contributions appear straightforward and somewhat redundant. For instance, Proposition 3 states that globally optimal solutions of the policy diversity loss can prevent strict mode collapse; however, (1) it focuses only on global optima, and (2) although a positive preference L1 distance induces positive policy KL divergence, the assumption of a fixed proportion between KL divergence and L1 distance is questionable.


[1] https://openreview.net/forum?id=TjEzIsyEsQ6

[2] https://arxiv.org/abs/2208.07914

[3] https://openreview.net/forum?id=49g4c8MWHy

[4] https://inria.hal.science/inria-00389811v2/document

[5] https://arxiv.org/abs/2110.14048

### Questions
It is unclear why the computation time is significantly improved. Is the gain primarily due to the Late-Stage Weighting mechanism, or does it result from parallel training?

### Soundness
3

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
4

### Summary
in multi-objective reinforcement learning (MORL) where the utility function of the decision maker is unknown, one needs to learn the complete coverage set of optimal trade-offs, such that the decision maker can select their preferred trade-off a posteriori. In case the utility function can be modeled as a weighted sum over objectives, we can equivalently apply the weights on the individual rewards, on the episodic return, or on the objective-specific Q-values. In this work, the authors advocate for late-stage weighting (i.e., weighting on objective-specific Q-values), claiming that naively combining conflicting objectives into one learning signal produces opposing gradients, and thus hampers learning. They propose a weight-conditioned PPO algorithm, called D^3PO, that learns Q-values per objective and uses the weights in the per-objective losses. Additionally, they incorporate a regularizer that promotes diversity over the weight-conditioned policies to avoid "mode collapse" where most policies are similar regardless of the weight-conditioning.

### Strengths
- The paper is clear and provides an algorithm that is well-grounded
- The experiments are extensive, performed on multiple benchmark environments for discrete action-spaces and continuous action-spaces, with multiple relevant baselines
- The results are competitive or outperform the baselines on multiple multi-objective metrics (hypervolume, expected utility, sparsity)

### Weaknesses
My concerns are that the algorithm claims 3 contributions for their algorithm: 1) multi-head critic, 2) late-weighting loss, 3) diversity reguralization.
 1. using a multi-head critic is common in many multi-policy algorithms (MORL-baselines [1] uses a multi-head critic in MOPPO, and this was already used in early deep MORL work [2]), and is thus not a contribution specific from this paper.
 2. as far as I understand (I would appreciate the authors correcting me otherwise), the late-weighting loss for the policy is specific to the PPO clipping objective, and only has an effect clipping occurs. Using late-clipping on the policy gradient objective should result in the same gradients as early clipping. 
 3. diversity reguralization seems very effective. It compares the distance between action distributions with the distance between the preferences themselves. But do these distances scale in the same fashion? For $w = w'$, the distance is indeed 0, but I do not believe $\pi_\theta(.|s_t,w') - \alpha||w-w'||_1$ results in a valid probability distribution. Def7 (line 803) mentions *mode collapse* when 2 policies with distinct preference weights are the same. But there exist many cases where difference weights lead to the same policy. A canonical example is the Deep Sea Treasure environment, where the coverage set is concave, and only the extrema's are optimal for linear scalarization functions. In that case, there are 2 optimal policies, and any weight combination maps to either of them.

### Questions
- even though late weighting stabilizes the learning process, by applying the weighting after the clipped per-objective losses, is the policy still maximizing the decision maker's utility function?
 - it seems to me that the most impactful component of D^3PO (with respect to performance) is the diversity regularizer. Even though this is not in the scope of the paper, do you think it would be possible to apply this reguralization on the baselines to obtain similar performance gains? 

 [1] Felten, F., Alegre, L. N., Nowe, A., Bazzan, A., Talbi, E. G., Danoy, G., & C da Silva, B. (2023). A toolkit for reliable benchmarking and research in multi-objective reinforcement learning. Advances in Neural Information Processing Systems, 36, 23671-23700.
 [2] Abels, A., Roijers, D., Lenaerts, T., Nowé, A., & Steckelmacher, D. (2019). Dynamic weights in multi-objective deep reinforcement learning. In International conference on machine learning (pp. 11-20). PMLR.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes D3PO (Decomposed, Diversity Driven Policy Optimization), a preference-conditioned multi-objective reinforcement learning framework. D3PO is an PPO extension that introduces multi-head critic with Late-Stage Weighting (LSW), which preserves raw per-objective signals and applies preferences only after PPO stabilization. A scaled diversity regularizer that provides a formal guarantee against mode collapse. The method is evaluated on multi-objective benchmarks and shows improved Pareto front quality.

### Strengths
The paper presents a technically sound extension of PPO for multi-objective reinforcement learning through a preference-conditioned framework.

The proposed multi-head critic with Late-Stage Weighting (LSW) and scaled diversity regularization are well-motivated and supported by reasonable theoretical intuition.

The experimental results are generally good, covering both discrete and continuous multi-objective tasks and showing improvements.

### Weaknesses
The proposed ideas are promising, but the conceptual structure and method description in Section 4 could be clearer. While Figure 1 appears to illustrate the overall framework, it is never explicitly referenced or discussed in the paper, which makes it harder to connect the algorithmic details to the visual explanation. The section would benefit from clearer guidance and stronger linkage between the conceptual figure and the mathematical formulation to help readers follow the proposed mechanisms.

While the paper focuses on preference-conditioned MORL, decomposition-based approaches remain a strong and active direction in the MORL field. It would be helpful if the authors could discuss the connection between D3PO and decomposition-based methods or provide experimental comparison to clarify their relative advantages. 

The comparison after line 291 is interesting, but needs a clarification of its generality.

The claim in the paragraph title (line 395) is not clear from Fig. 2, where for Hopper C-MORL has a clearly better coverage (and mostly better performance) while in Ant and Humanoid the proposed algorithm does not explore some obvious regions of the Pareto place. Reported results are here very good in comparison with the considered candidates, but single-objective approaches (e.g. Dreamer v3) tend to do better, which should be considered as a base line of the single-objective boundary case. 

The value of $\alpha$ is not given nor discussed, although it seems difficult to find a meaningful constant for the trade-off between a KL divergence and a weight vector difference.

### Questions
Beyond the current benchmarks, at what type of applications would the D3PO’s advantages be most beneficial?

### Soundness
2

### Presentation
2

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
In this paper, the authors look into the destructive gradient interference between conflicting objectives and representational mode collapse of Multi-objective RL. The paper proposes to train a single policy that can adapt to different user preferences. This is achieved by computing per-objective advantages and applying preference weights only after PPO’s stabilization step. They also enforce proportional behavioral diversity across different preference vectors to prevent collapse.

### Strengths
The paper identifies two challenges, the mode collapse and the conflicting objectives clearly. 

The proposed solutions are technically sound.

Provided simulations demonstrate the effectiveness of the proposed method.

### Weaknesses
There are limited discussions on the hyperparemeters. Also, the method needs to adopt many parameters, which may be hard to scale in real MORL tasks. 

Examples of policy behaviors (e.g., different strategies emerging for different preferences) would help validate “behavioral diversity” more intuitively.

The performance gain might be marginal compared to previous methods.

### Questions
Can the authors plot the evolution of reward across training? It would be good to validate how fast the algorithm converges.

Some conflicting objectives can essentially cause optimizing one objective will reduce performance under the other objective. Have the authors observed such scenarios? How the proposed method can tackle that?

In Hopper, the performance of proposed method is worse. Is it possible to initialize MORL policies using other methods, then applying proposed approach?

### Soundness
3

### Presentation
3

### Contribution
3
