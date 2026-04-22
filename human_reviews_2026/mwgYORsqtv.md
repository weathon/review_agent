# Demystifying The Mechanisms Behind Emergent Exploration in Goal-Conditioned RL

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
In this work, we take a first step toward elucidating the mechanisms behind emergent exploration in unsupervised reinforcement learning. We study Single-Goal Contrastive Reinforcement Learning (SGCRL) (Liu et al., 2025), a self-supervised algorithm capable of solving challenging long-horizon goal-reaching tasks without external rewards or curricula. We combine theoretical analysis of the algorithm’s
objective function with controlled experiments to understand what drives its exploration. We show that SGCRL maximizes implicit rewards shaped by its learned representations. These representations automatically modify the reward landscape to promote exploration before reaching the goal and exploitation thereafter. Our experiments also demonstrate that these exploration dynamics arise from learning low-rank representations of the state space rather than from neural network function approximation. Our improved understanding enables us to adapt SGCRL to perform safety-aware exploration.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to demystify the mechanisms behind emergent exploration in Single-Goal Contrastive Reinforcement Learning (SGCRL). The authors combine theoretical analysis and controlled experiments both in continuous and tabular settings to argue that SGCRL implicitly maximizes a representation-based reward defined by goal similarity. This implicit reward gives rise to an automatic exploration–exploitation curriculum: states along unsuccessful trajectories gradually lose goal similarity (discouraging revisits), while those along successful paths gain similarity (encouraging exploitation). The authors show that these dynamics persist even without neural network function approximation, suggesting they arise from the low-rank structure of contrastive representations rather than network inductive biases. They further propose that understanding these dynamics enables safety-aware exploration by manipulating representations.

### Strengths
* The paper provides a clear and well-structured theoretical account of why SGCRL exhibits emergent exploration despite lacking explicit rewards.
* I think the use of tabular environments to isolate the contrastive representation mechanism is elegant and persuasive.
* The paper is well written, and the figures effectively communicate the representation dynamics.
- The demonstration that altering representation geometry can control exploration pathways is an intriguing direction for safe RL.

### Weaknesses
- I feel the the proposed interpretation, exploration as maximisation of a representation-based implicit reward, is closely aligned with existing frameworks such as successor representations, potential-based reward shaping, and eigenoptions (e.g. (Machado et al)). The connection to these earlier ideas is not fully acknowledged, giving the impression of a new theory rather than a reinterpretation.
- The framing suggests spontaneous behaviour arising from complex dynamics, but the mechanism can be straightforwardly understood as representation-induced reward shaping.
- The intervention experiments are promising but not systematically evaluated or compared to existing safety-aware exploration methods.

### Questions
1. How does the proposed “implicit reward” interpretation differ formally from the successor feature formulation of value functions? 
        
2. Would the same dynamics appear if the critic used a non-contrastive predictive loss (e.g., temporal difference learning or policy gradient)? One could argue that with the policy gradient once the agent is near convergence a similar patterns appears, i.e., that the described phenomenon in the paper might not be exclusive to the contrastive approach
        
3. How sensitive are the observed behaviours to representation normalisation or embedding dimensionality?  I would assume normalisation/compression plays a big role in shaping the geometry of the reward function.

### Soundness
3

### Presentation
4

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
This work analyses the origins of implicit exploration behaviors observed in Single-Goal Contrastive Reinforcement Learning (SGCRL), a model that actively aligns state-action representations with the representations of a single goal. The paper presents a theoretical and experimental analysis of the dynamics of state and goal representations learned by the model. A long list of experiments characterizes the desiderata to make SGCRL induce a form of decaying optimism that boost exploration.

The paper is well-written and the work provides a sound and novel analysis that helps understand the emergence of implicit exploration behaviors in a RL setting. Thus, I recommend acceptance. 


I'd be happy to further increase my score:
Section 4.2 kicks in an open door. Given the infoNCE loss function, it is not surprising that the similarity between phi(g) and phi(s) rules the behavior. The paper decides to focus on Single-Goal RL and shows in Appendix that uniform goal sampling does not work. The experimental section could provide an analysis of how the number of fixed goals, along with their goal-conditioned policy, affect the representations and the exploration behaviors. Single-Goal RL is a limited setting, extending the analysis may bring more general results or significant insights for future work.


Minor comments: 
- Figure 5a), please, bring the star forward.
- The benefits of intervening on the representations rather than with a direct reward is unclear.

### Strengths
The paper is well-written and the work provides a sound and novel analysis that helps understand the emergence of implicit exploration behaviors in a RL setting.

### Weaknesses
The analysis focuses on a constrained setting, Single-Goal contrastive learning.

### Questions
Please, see above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper provides a mechanistic analysis of SGCRL [1], aiming to explain its “emergent exploration.” It formalizes the actor’s behavior as optimizing a shaped return based on similarity metrics, making explicit the intuition that policies are guided by goal-similarity in the learned representation rather than raw state-space distance. Building on this, it analyzes how InfoNCE training shapes the representation such that explored regions with low similarity to the goal are effectively “pruned”. The authors validate these predictions via controlled representation interventions and via a tabular reconstruction to show the exploratory behavior does not come from the use of function approximators. Experiments in grid worlds (e.g., Four Rooms) and toy tasks (e.g., Tower of Hanoi) empirically match the theory.

[1] Liu. et. al. A Single Goal is All You Need: Skills and Exploration Emerge from Contrastive RL without Rewards, Demonstrations, or Subgoals 2025

### Strengths
1. Clear mechanistic story linking the actor objective to an implicit ψ-similarity reward. Completing the intuition that SGCRL left as future work.

2. The use of interventions (attract/repel via embedding edits) and tabular models gives a good ablation study and nicely visualizes the behaviors.

3. Empirical trends in grid worlds align with the theory, making the presentation generally clean.

### Weaknesses
1. I'm concerned mostly about the scope mismatch: Framed as broadly “goal-conditioned RL,” but almost all analysis/evidence is for one algorithm (SGCRL) and mostly single-goal settings; multi-goal evidence is limited/toy and lacks HER-style goal-conditioned algorithms.

2. One of the main contributions, Theorem 1 helps with the formalization, but conceptually modest. It largely reframes the intuition as a theorem, but only under the alignment assumption.

3. Assumption of fixed ψ(g) limits the evidence strength. It does not hold using neural encoders (ψ(g) will drift). The paper shows robustness empirically, but the theoretical scope could be clearer.

4. Cognitive-science in the conclusion part is questionable. It reads as methodological borrowing (rational analysis, interventions, small model), not really a real "cognitive" experiment. I suggest this should be toned down or made more precise.

### Questions
1. Please explicitly scope when ψ(g) can be treated as fixed (tabular lookup, imaginary goal, frozen goal head) and discuss how in practice, the predictions degrade when ψ(g) drifts under shared encoders.

2. Can you provide multi-goal experiments with HER-style baselines (sampling goal from future states) to support the framing?

3. The sentence “drawn to high ψ-similarity even when far from the goal…” is confusing. Please state explicitly that attraction/repulsion is governed by similarity in embedding space, not raw state-space distance.

4. The role of the LogSumExp could potentially also be discussed in the represetation learning as it's so crucial to the contrastive RL. How does that impact the represetation/exploration?

### Soundness
3

### Presentation
2

### Contribution
2
