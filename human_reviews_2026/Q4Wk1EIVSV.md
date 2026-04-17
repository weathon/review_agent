# Addressing Exogenous Variability in Cooperative Multi-Agent Reinforcement Learning

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Multi-agent reinforcement learning (MARL) has advanced control of many cooperative multi-agent systems. However, most approaches are trained against a single fixed adversarial strategy, leaving teams fragile to adversarial strategy shifts at test time. To handle such limitations, in this paper, we recast cooperative MARL from a new perspective into an Exogenous Dec-POMDP, separating agent-controllable endogenous and environment-driven exogenous dynamics in order to learn policies that adapt to exogenous shifts while preserving coordination. Our framework is composed of two main components: (i) learning exogenous dynamics and (ii) updating policy with two complementary goals - coordination to achieve high team return and causal influence on future exogenous evolution.
We implement the framework under centralized training with decentralized execution into a practical algorithm, named Learning Exogenous Influence for Coordination and Adaptation (LEICA), and evaluate it on SMAX with distinct train/test adversarial strategies. Experimental results show that our approach drastically improves performance in test time with unseen opponents' strategies while achieving high training-time performance, demonstrating its ability to handle exogenous shift and improve training stability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses generalization in multi-agent reinforcement learning (MARL) and proposes Exogenous Decentralized POMDPs (ED-POMDP) to model the separate endogenous influences of the agents' actions and the exogenous influences by the environment (transition). Based on a modeled causal chain, Learning Exogenous Influence for Coordination and Adaptation (LEICA) is proposed to shape rewards with respect to the separated influences. An endogenous and exogenous state predictor based on variational autoencoders are trained to estimate causal influences via a Jacobi matrix of the predictors. These influence estimates are transformed into counterfactual intrinsic rewards and added to the extrinsic reward for standard MARL algorithms like MAPPO to train on. LEICA is evaluated on a Jax-based variant of StartCraft Multi-Agent Challenge (SMAX) and shown to generalize to randomized variations of the original micromanagement tasks, where initial states are varied to different degrees.

### Strengths
The paper focuses on an important and interesting topic. Generalization in MARL, especially regarding distribution shifts, are open challenges that need to be addressed.

The paper is well-written and mostly easy to follow.

### Weaknesses
**Novelty**

While the addressed problem is well-motivated, the approach is merely a mixture of existing techniques that are already known/established:
1. Multi-task reinforcement learning [1,2,3]
2. Initial state variation [4,5,6]
3. Reward shaping with counterfactuals/causal inference [7,8,9]

While 3. seems to be a theoretically sound approach to solving the addressed problem, 1. and 2. are known to merely shift the actual problem without sufficiently solving it in a causal manner [10], as the learning algorithms depend on i.i.d. samples (either data point-wise or problem-wise).

To improve the paper, a discussion and experimental comparison to these works is required to justify the proposed approach.

The paper also ignores prior work on robust MARL, which addresses adversarial behavior of separate agents [11,12,13].

**Soundness**

The paper focuses on a causal chain that determines the effect of the endogenous context on the next exogenous context. However, according to Fig. 1, both contexts form a colliding node in the causal graph, which could interfere with the general concept. Since the paper does not provide any theoretical analysis, e.g., regarding identifiability, I cannot confirm the validity of the proposed approach from a causal perspective.

The paper proposes to consider "baseline actions" for the counterfactual rewards, which resemble the "default actions" for the difference rewards or aristrocat utilities introduced in [7].

**Quality**

While the paper is generally well-written and presented, the proposed approach introduces a list of hyperparameters, such as $\alpha$ (trade-off exogenous influence and value-sensitivity), $\tau$ for sharpness of the weighting vector, and $\beta$ for decaying the weighting vector. $\lambda$ for the influence-based reward, indicating a tuning-intensive approach.

Of these hyperparameters, only $\alpha$ is ablated. For revision, I recommend:
1. A more thorough ablation study on all of these hyperparameters, e.g., for the appendix.
2. An intuition, how to set these parameters, e.g, regarding domains different from SMAX
3. Ideally, finding an (adaptive) approach that can set these hyperparameters automatically.

**Significance**

The experiments are conducted on some well-known SMAC maps and compared to standard MARL baselines, such as MAPPO and QMIX. The only variation tested in the paper is the initial state, which has been investigated for SMAC in [4,5,6], where MAPPO and QMIX have already been demonstrated to perform poorly.

To improve the significance of the work, especially regarding generalization, I suggest the following:
- Vary the rewards and other functions, as defined in Section 3.1
- Compare with the methods introduced in [4,6]
- Compare with adversarial (test) methods introduced in [11,12,13]
- Test other domains beyond StarCraft II, such as multi-agent MuJoCo and Google Research Football

**Literature**

[1] Omidshafiei et al., "Deep Decentralized Multi-task Multi-Agent Reinforcement Learning under Partial Observability", ICML-17

[2] Li et al., "Multi-task Reinforcement Learning in Partially Observable Stochastic Environments", JMLR-09

[3] Hessel et al., "Multi-Task Deep Reinforcement Learning with PopArt", AAAI-19

[4] Lyu et al., "On Centralized Critics in Multi-Agent Reinforcement Learning", JAIR-23

[5] Ellis et al., "SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning", NeurIPS-23 Benchmarks

[6] Phan et al., "Attention-Based Recurrence for Multi-Agent Reinforcement Learning under Stochastic Partial Observability", ICML-23

[7] Wolpert et al., "Optimal Payoff Functions for Members of Collectives", Advances in Complex Systems 2001

[8] Li et al., "Automatic Reward Shaping from Confounded Offline Data", ICML-25

[9] Jaques et al., "Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning", ICML-19

[10] Schölkopf, "Causality for Machine Learning", 2019

[11] Li et al., "Robust Multi-Agent Reinforcement Learning via Minimax Deep Deterministic Policy Gradient", AAAI-19

[12] Phan et al., "Resilient Multi-Agent Reinforcement Learning with Adversarial Value Decomposition", AAAI-21

[13] Li et al., "Byzantine Robust Cooperative Multi-Agent Reinforcement Learning as a Bayesian Game", ICLR-24

### Questions
1. According to the problem statement in Section 3.1: What is the difference to multi-task learning, given that the reward function can vary as well?
2. In Definition 1: What does the letter $I$ in the definition of $S^{e}_{t}$ stand for? Is it the same $I$ (mutual information) as in Equations 1 and 2?
3. Why is the test considered to be adversarial? The experimental setting description implies that the initial states are merely randomized (without any value-minimizing intentions).

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
Most MARL approaches are trained against a single fixed adversarial strategy, leaving teams vulnerable to adversarial strategy shifts at test time. In this work, the authors recast cooperative MARL from a new perspective into an Exogenous Dec-POMDP. It separates agent-controllable endogenous and environment-driven exogenous dynamics in order to learn policies. It consists of VAEs to learn endogenous and exogenous dynamics and influence-based reward design. Through experiments in a modified SMAX, the authors demonstrate the effectiveness of their proposed approach.

### Strengths
1. This paper views MARL cooperation from a somewhat new perspective.
2. The experimental results demonstrate the effectiveness of the LEICA in a modified SMAX (more challenging with changing opponent strategies) 
3. The paper is well-written, especially the introduction.

### Weaknesses
1. It seems that decoupling the controllable and non-controllable idea is similar to the idea of DRIMA, which considers environmental risk and cooperation risk. 
2. It seems that dividing the state/observation into endogenous and exogenous parts depends on the environment. In SMAX/SMAC, the state/observation can be decomposed in such a way due to the data structure design. However, it is unclear whether it is suitable for other environments.
3. The reward design is similar to the design of COMA. Moreover, formulas (5)-(8) do not provide much insight regarding endogenous and exogenous parts.

REFERENCE 

[1] Disentangling Sources of Risk for Distributional Multi-Agent Reinforcement Learning, ICML 22.

### Questions
1. line 66, why "a new scalable SMAX"? Do you show that your new SMAX is more scalable through experiments?
2. Figure 1, please describe the blue line and the black line in detail.
3. line 165-166, is the training process divided into multiple stages? The first stage learns the VAEs?
4. line 173, “decomposed as S_t = (...”. Does this rely on the data structure of environments? 
5. What are the MARL insights regarding (5)-(8)?
6. line 278-280, There are many MARL value-based approaches published after 2020.

### Soundness
3

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
4

### Summary
This paper targets cooperative multi-agent reinforcement learning under exogenous variability—e.g., unseen opponent strategies at test time. The authors propose an Exogenous Dec-POMDP (ED-POMDP) that factorises the global state into an endogenous component (controllable by the team) and an exogenous component (driven by the environment/opponent). Under this formalism they derive LEICA, a CTDE algorithm that (i) learns separate variational predictors for endogenous and exogenous transitions, and (ii) shapes a counterfactual, influence-weighted intrinsic reward which encourages actions that simultaneously improve team return and shift future exogenous states in a favourable direction. Extensive experiments on a new SMAX benchmark with 63 opponent strategies show large gains over MAPPO, QMIX, LAIES and SHAQ in both training and zero-shot generalisation regimes.

### Strengths
Novel conceptual framing: ED-POMDP explicitly disentangles controllable vs. uncontrollable dynamics, giving a principled way to reason about robustness to non-stationary, non-learning opponents.
Practical algorithm: LEICA retains the scalability of CTDE actor-critic methods; the additional VAE predictors and Jacobian-based reward are cheap to compute and easy to plug into MAPPO.
Strong empirical results: Across 15 train/test splits (Small / Medium / Large) LEICA consistently outperforms strong baselines, often doubling the win-rate on unseen strategies while maintaining highest training performance.

### Weaknesses
Manual state partition: The endogenous/exogenous split is hand-crafted using domain knowledge of SMAX; the method could fail if the partition is misspecified or unavailable in new domains.
Single-step influence: The reward uses only the one-step Jacobian ∂sˆx_{t+2}/∂s^e_{t+1}; long-horizon influence or multi-step planning is not considered.

### Questions
1. Could the partition be learned from data using, e.g., conditional independence tests, sparsity priors, or causal discovery?
2. Why restrict influence to one-step? Would a multi-step rollout (even short) improve the credit assignment, especially in sparse-reward tasks?
3. What is the relation between the world model in model-based RL and the proposed exogenous dynamic model?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a framework for cooperative MARL under exogenous variability. The authors reformulate MARL as an Exogenous Dec-POMDP that explicitly separates team-controllable and environment related state components. They propose a CTDE-based algorithm, called LEICA, that combines two variational inference for exogenous/endogenous factors with an influence-weighted intrinsic reward that encourages both coordination and adaptability. Experiments on the SMAX benchmark demonstrate gains over standard MARL baselines.

### Strengths
- Authors is proposing an interesting approach to separate the learning of agent controlled states and environment controlled states to improve the robustness of MARL
- The paper is well written and easy to follow.

### Weaknesses
- No comparison to robust MARL baselines: my main concern with this paper is that it omits comparison to established minimax or distributionally-robust algorithms (e.g.,M3DDPG, “Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning”). This limits the strength of the robustness claim.
- Hand-crafted state partition: The endogenous/exogenous split is assumed given; learning this decomposition automatically would improve generality.
- Computational complexity: Two VAEs per agent and Jacobian-based influence estimation introduce significant training overhead compared to MAPPO/QMIX.

### Questions
- how do you apply this type of method to environments where it is hard to separate the exogenous and indigenous states (e.g. image-based inputs)
- Can the intrinsic reward be generalized to other environment other than SMAX, such as the ones that has on opponents. It makes sense for SMAX as we are trying to reduce the opponent’s health, so influence is desired. What happens to the other environment (e.g.cooperative navigation)?

### Soundness
3

### Presentation
3

### Contribution
3
