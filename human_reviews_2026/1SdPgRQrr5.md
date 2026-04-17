# Dual-Objective Reinforcement Learning with Novel Hamilton-Jacobi-Bellman Formulations

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
Hard constraints in reinforcement learning (RL) often degrade policy performance. Lagrangian methods offer a way to blend objectives with constraints, but require intricate reward engineering and parameter tuning. In this work, we extend recent advances that connect Hamilton-Jacobi (HJ) equations with RL to propose two novel value functions for dual-objective satisfaction. Namely, we address: 1) the Reach-Always-Avoid (RAA) problem – of achieving distinct reward and penalty thresholds – and 2) the Reach-Reach (RR) problem – of achieving thresholds of two distinct rewards. In contrast with temporal logic approaches, which typically involve representing an automaton, we derive explicit, tractable Bellman forms in this context via decomposition. Specifically, we prove that the RAA and RR problems may be rewritten as compositions of previously studied HJ-RL problems. We leverage our analysis to propose a variation of Proximal Policy Optimization (DO-HJ-PPO), and demonstrate that it produces distinct behaviors from previous approaches, out-competing a number of baselines in success, safety and speed across a range of tasks for safe-arrival and multi-target achievement.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel RL framework that integrates Hamilton-Jacobi equations from optimal control theory to handle dual-objective tasks involving safety and goal achievement. 
The authors introduce two new value formulations: Reach-Always-Avoid, which requires reaching a goal while maintaining safety thereafter, and Reach-Reach, which requires achieving two distinct goals sequentially. 
These value functions are defined via min-max extremal operators, enabling direct reasoning about best- and worst-case outcomes.
The authors show that the Reach-Always-Avoid and Reach-Reach problems can be decomposed into simpler subproblems (Reach, Avoid, and Reach-Avoid), each corresponding to known HJ-Bellman equations. 
To make these formulations tractable, they augment the MDP state with historical extrema to restore the Markov property and propose an RL algorithm DO-HJ-PPO, a variant of PPO that jointly learns decomposed and composed actor-critic pairs. 
The algorithm employs a Reach-Avoid GAE to approximate the extremal Bellman differences.
Experiments on discrete and continuous control tasks demonstrate that DO-HJ-PPO outperforms baselines.

### Strengths
1. The paper builds on HJ optimal control theory and connects it elegantly to RL. The proposed formulations (Reach-Always-Avoid and Reach-Reach) are mathematically rigorous and clearly grounded in control-theoretic Bellman principles.

2. The theoretical decomposition of Reach-Always-Avoid and Reach-Reach problems into subproblems is clean and provable, allowing modular reuse of existing HJ-RL methods.

3. By directly modeling safety and goal satisfaction in the value function, the approach avoids the delicate tuning required by Lagrangian or constrained RL methods.

4. Experiments span both discrete and continuous control tasks. The method consistently outperforms strong baselines, showing robustness and good generalization.

### Weaknesses
1. While the theoretical model is elegant, many experiments (e.g., Hopper-RAA, Hopper-RR) use geometrically simple rewards that do not fully reflect the difficulty of real-world control problems. Thus, the results may overstate performance in realistic scenarios.

2. The algorithm involves multiple coupled actor-critic networks (for decomposed and composed objectives), which can significantly increase computational and memory costs. The paper lacks ablation studies or discussion on scalability or efficiency.

3. The baselines (e.g., CPPO, RESPO) are standard but may not be fully optimized under the same hyperparameter or reward conditions, making the fairness of some comparisons uncertain.

### Questions
1. The paper extends the HJ-Bellman framework to stochastic and learned environments, but most derivations assume deterministic dynamics. How robust is the theoretical foundation when applied to stochastic MDPs?

2. The RAA and RR formulations are proven decomposable, but the algorithm jointly learns all sub-critics with bootstrapped targets.
Is there any theoretical guarantee that this joint training converges to the same fixed point as the original HJ decomposition?

3. The continuous-control tasks use simplified geometric rewards.
Would DO-HJ-PPO still outperform baselines if trained with the dense, physically meaningful rewards used in standard MuJoCo benchmarks?

### Soundness
4

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
2

### Summary
The paper extends Hamilton–Jacobi-style RL beyond the classic Reach (R), Avoid (A), and Reach–Avoid (RA) objectives to two harder, compositional objectives: Reach–Always–Avoid (RAA) (reach a goal and keep avoiding hazards forever afterward) and Reach–Reach (RR) (reach two distinct goals, in any order). The key theoretical result is that both RAA and RR can be decomposed into already-solved HJ-RL subproblems (avoid + reach–avoid for RAA; three reach problems for RR), yielding explicit Bellman forms (Thm. 1–2). Because optimal actions depend on history (e.g. “have I already reached goal 1?”), the paper introduces a specific state augmentation that tracks max-reward and min-penalty over time and shows this augmentation is sufficient (Thm. 3). On top of this, they build DO-HJ-PPO, an on-policy method that learns the decomposed critics alongside the composed RAA/RR critic and couples rollouts so subproblems stay relevant. Experiments on gridworld, Hopper, F-16, SafetyGym, and HalfCheetah show higher “both-objectives achieved” rates and faster completion than Lagrangian and prior HJ-RL baselines.

### Strengths
1. The decompositions are stated cleanly, proved in the appendix, and tied directly to implementable Bellman operators; the need for history and the exact augmentation are justified, not hand-waved.
2. Many safe / task-spec RL problems really are “reach X and stay safe” or “reach X then Y”; having explicit Bellman forms and a working PPO variant for those is useful, especially since baselines mostly get only “partial success.”

### Weaknesses
1. Assumptions are narrow: Main theory is for deterministic, finite MDPs, yet real tasks are stochastic/continuous, but the paper only gives a heuristic stochastic variant (SRABE) without matching guarantees. A short discussion of what breaks in stochastic dynamics is needed.
2. State augmentation cost: The proposed augmentation grows the state with running max/min signals; for higher-dim tasks (real robots, multi-goal specs) this could be heavy, and the paper doesn’t study scalability.
3. The paper lacks sample-complexity or contraction analysis for the composed operators. Prior HJ-RL work leans on discounted versions to get contraction, and here the composed PPO version is justified mostly by analogy.

### Questions
Please see weaknesses.

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
The paper defines two dual-objective, extremal value functions in reach-avoid optimal control problems: "**reach and always avoid**" (RAA) and "**reach and reach**" (RR),  and shows each can be decomposed into known HJ-RL primitives of Reach (R), Avoid (A), Reach-Avoid (RA), yielding explicit Bellman targets and corresponding algorithms that train a composed actor–critic together with additional critics/actors for the sub-objectives.
The paper argues history is necessary and introduces a minimal augmentation (running max/min) with a sufficiency result.
The proposed value function decompositions was combined with deep Q-learning (DQN) and proximal policy optimization (PPO) and tested in a simple grid-world and modified continuous control environments (Hopper, F16, SafetyGym, and HalfCheetah).

### Strengths
- **Originality and significance**: moderate. The problems studied in this paper are classical in logic terms, but the decomposition is novel and practically attractive.
- **Quality and correctness**: the reductions using min/max identities and distributivity/commutativity under deterministic transition and policy assumptions (relaxed later) seem sound and reasonable for the stated setting. Proofs were not carefully checked.
- **Clarity**: This paper is clear, easy to follow, and contextualized well. Section 2 related works is comprehensive yet concise, very useful for readers. The most relevant works (e.g., `Fisac et al. (2019)`, `Hsu et al. (2021)`) were cited throughout and explained well. Figure 1 is a bit confusing (Figure 2 is more understandable for me) and dense, but it illustrates the problem setting to some extent.
- **Compositional formulation**: converting RAA and RR to R, A, and RA can be helpful for compositionality and generalization.

### Weaknesses
## Formal relation to temporal logic

I believe the value function decompositions are based on some algebra/logic implicitly.
It would help readers if the paper states the specs more explicitly, potentially using temporal logic, even if the proposed method does not explicitly use automata.

My feeling is that this paper is just doing quantitative semantics for temporal modal operators $F$ **f**inally, $G$ **g**lobally/always, and $U$ **u**ntil.

Let $r, p, q$ be temporal propositions.
Then,
- R is $F r$ "finally reach r"
- A is $G \lnot p$ "always avoid (not reach) p"
- RA is $\lnot p U r$ "avoid p until reach r"
- RAA is $F r \land G \lnot p$ "(finally reach r) and (always avoid p)"
- RR is $F p \land F q$ "(finally reach p) and (finally reach q)"

The first three have quantitative semantics:
- R: $F r \leftrightarrow \max_t r_t$
- A: $G \lnot p \leftrightarrow \min_t -p_t$
- RA: $\lnot p U r \leftrightarrow \max_t \min\\{r_t, \min_{t' \leq t} -p_{t'}\\}$
which were shown in Section 4.

It is possible to prove
- RAA: $F r \land G \lnot p \equiv (\lnot p U r) \land (G \lnot p)$  = "(avoid p until reach r) and (always avoid p)" (?)
- RR: $F p \land F q \equiv F((p \land F q) \lor (q \land F p))$ "finally ((reach p and finally reach q) or (reach q and finally reach p))"

The LHSs have quantitative semantics given in Section 3:
- RAA: $\min\\{\max_t r_t, \min_t -p_t\\}$
- RR: $\min\\{\max_t p_t, \max_t q_t\\}$

The RHSs have quantitative semantics given in Section 6. (small gaps)

The point is, there exists a more principled approach to derive these value functions and Bellman equations than the lengthy *10+ page proofs* in the appendix, which are more error-prone.

The author stated that "*While the problems we attempt to solve (e.g. reaching multiple goals) can be thought of as specific instantiations of LTL specifications, our approach to solving these problems is fundamentally different from those in this line of work.*," but being "fundamentally different" doesn't mean it's better, more efficient, or more generalizable.
There's no evidence that LTL cannot express RAA or RR.
Even if it's true, there alternatives such as metric temporal logic (MTL) or signal temporal logic (STL).
I'm not an expert in temporal logic, but the concept of **monitoring** is very relevant to the state augmentation issue.
The author is encouraged to further connect this paper with the existing logic literature;
if they decide not to do so, at least justify the "min/max algebra" approach by spelling out the axioms and operations (commutativity, associativity, distributivity, idempotence, absorption, monotonicity, etc.).

## Complexity: why RA?

The second issue is pertinent to the first one.
This paper decided to convert RAA and RR in to R, A, and RA to "leverage new HJ-based methods on the subproblems", because these HJ-RL "primitives" have been previously studied.
However, after scrutinizing Sections 3, 4 and 6, it seems to me this approach unnecessarily complexifies the problem?
Why not directly derive the value functions and Bellman equations from the specifications in Section 3?
Especially, the use of RA in RAA seems strange.
Why not just R and A?

"Focusing on extremal values rather than discounted sums", "directly specifying desired behavior", and composing task specs, values, and policies is a nice direction.
However, the RL community has developed several methods for general learning objectives `Wang et al. (2020)`, `Cui & Yu (2023)`, `Tang et al. (2025)`.
Please compare this work with them, and explain why the proposed method is preferred for the stated tasks.

## MDP without rewards and need for augmented states

Figure 2 and Section 5 might be misleading because each node seems to be considered as a state, and whether a node has been reached isn't part of the state.
"Non-augmented" MDPs/policies are flawed because this paper is actually using a "**MDP without rewards**" setting: the MDP $M = (S, A, f)$ is a triple and only consists of the transition function, and the reward function is not on the environment side but designed by the agent. 
Although I believe this setting is indeed more natural and should be used, but since the more widely used formulation considers the reward function as a part of MDP, it's incorrect to consider a node as a state, otherwise the reward function is not well defined.

"Node not reached/with an item" and "node reached/without an item" should be two distinct states;
"Reaching an unvisited node/collecting an item" and "reaching a previously visited node" should generate different observations/rewards (e.g., locations + a bunch of flags).
The flaw is not due to the limitations of "memoryless" policies; it's because of the wrongly chosen states.

## Stochastic reach-avoid

This paper extended reach-avoid problem formulations using min/max to the stochastic setting in Section 7.1.
However, the value function and Bellman equation (SRABE) were given without sufficient justification.

Caveat: *expected maximum* and *maximum expectation* are two different objectives, and consequently their value functions are also different.
One may argue that both are meaningful objectives, but the corresponding value functions and Bellman equations should not be *defined* but *derived* from the objective.
The error of interchanging max and expectation in `Quah & Quek (2006)` was noticed by `Gottipati et al. (2020)` and `Cui & Yu (2023)` and some fixes were proposed.
The author is encouraged to explicitly state the learning objective in Section 5 like what they did in Section 3 and confirm the Bellman equation truly corresponds to this objective.

## References

- Kian Hong Quah and Chai Quek. **Maximum reward reinforcement learning: A non-cumulative reward criterion.** Expert Systems with Applications, 31(2):351–359, 2006. https://doi.org/10.1016/j.eswa.2005.09.054.
- Sai Krishna Gottipati, Yashaswi Pathak, Rohan Nuttall, Raviteja Chunduru, Ahmed Touati, Sriram Ganapathi Subramanian, Matthew E Taylor, and Sarath Chandar. **Maximum reward formulation in reinforcement learning.** arXiv preprint, 2020. https://arxiv.org/abs/2010.03744.
- Ruosong Wang, Peilin Zhong, Simon S Du, Russ R Salakhutdinov, and Lin Yang. **Planning with general objective functions: Going beyond total rewards.** In Neural Information Processing Systems, 2020. https://proceedings.neurips.cc/paper/2020/hash/a6a767bbb2e3513233f942e0ff24272c-Abstract.html
- Wei Cui and Wei Yu. **Reinforcement learning with non-cumulative objective.** IEEE Transactions on Machine Learning in Communications and Networking, 1:124–137, 2023. https://doi.org/10.1109/TMLCN.2023.3285543.
- Yuting Tang, Yivan Zhang, Johannes Ackermann, Yu-Jie Zhang, Soichiro Nishimori, Masashi Sugiyama. **Recursive Reward Aggregation.** In Reinforcement Learning Conference, 2025. https://openreview.net/forum?id=13lUcKpWy8

### Questions
Minor issues
- The introduction employs strong intensifiers (e.g., “performing far more safely”) without quantitative context. This reads like forward-referencing results.
- Apostrophes for plurals. "SBE’s and RABE’s" $\to$ SBEs and RABEs (no apostrophes for plural acronyms)
- l.049: user?
- l.050: "Figure 1, top middle-left" is awkward. Consider using subcaptions (e.g., Figure 1a).
- l.050-051: Consider avoiding F16 and Hopper, because readers may not know them if they are not familiar with these environments.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper investigates two dual-objective satisfactory problems in RL: the reach–reach problem and the reach–always–avoid problem. These problems extend the previously studied reach, avoid, and reach–avoid problems analyzed from the Hamilton–Jacobi perspective of dynamic programming. The authors derive the corresponding Bellman equations for both problems and demonstrate that each can be decomposed into compositions of the previously studied HJ-based reach, avoid, and reach–avoid problems. Furthermore, the paper extends the framework to handle stochastic policies and provides a formulation for implementing it using PPO. Experiments conducted across several environments validate the effectiveness of the proposed method.

### Strengths
- The RR and RAA problems are two interesting dual-objective satisfactory problems, and the paper reveals their unique structures that require specialized treatment.

- The paper presents a solid theoretical analysis by deriving the Bellman equations for the RR and RAA problems and elucidating their connections to the previously studied reach, avoid, and reach–avoid problems.

- Empirical results across multiple environments show that the proposed PPO-based method outperforms baseline approaches.

### Weaknesses
I do not have major concerns about the paper (possibly due to limited familiarity with the related literature). Some minor points are as follows:
- The paper considers a deterministic transition function. It is unclear whether the proposed method can be extended to the stochastic transition case.
- For stochastic policies, it remains uncertain whether the proposed PPO-based update can guarantee convergence to an optimal policy.

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3
