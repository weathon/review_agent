Now I have all the information I need. Let me compose the final review.

## Summary

The paper introduces CORMAB, a class of restless multi-armed bandit problems with combinatorial action constraints that prevent arm-level decoupling, and SEQUOIA, an algorithm that embeds a ReLU Q-network into a mixed-integer linear program (MILP) to perform combinatorial action selection at each timestep of sequential planning. Four concrete CORMAB instantiations are formulated (multiple interventions, bipartite matching, capacity constraints, path constraints), and SEQUOIA is evaluated against myopic and iterative baselines, reporting an average 24.8% improvement.

## Strengths

- **Novel and well-motivated problem formulation:** The CORMAB formulation (Section 2–3) formally identifies a meaningful gap between standard RMABs (which admit Whittle-index decoupling) and real-world resource allocation problems where combinatorial constraints prevent arm-by-arm decomposition. The four instantiations are concrete and practically motivated, making a genuine contribution in defining this problem class.

- **Principled algorithmic integration of DQN and MILP:** The core idea of embedding a ReLU Q-network into a MILP for combinatorial action selection (Section 4.1, Equations 3–6, Algorithm 1) is conceptually clean. It directly leverages the MILP-representability of ReLU networks to turn intractable enumeration into a mathematical program — a principled alternative to heuristic decomposition.

- **ITERATIVE DQN ablation demonstrates value of MILP action selection:** The 12.1% gap between SEQUOIA and ITERATIVE DQN on the capacity-constrained setting (Section 5) is meaningful evidence that the MILP-based action selection — not just Q-network training — contributes substantially. ITERATIVE DQN shares the same Q-learning training but uses greedy heuristic action selection, isolating the effect of the MILP solver.

- **Practical computational enhancements:** The warm-start strategy (myopic pre-training, perturbed/infeasible action sampling for diversity, memoization in Section 4.2) addresses the real computational bottleneck of O(EHM) = ~640,000 MILP solves with concrete, reusable ideas. The insight that infeasible actions produce valid per-arm transitions in RMABs is clever.

- **Same architecture works across all four domains without per-domain tuning** (Section 5), demonstrating general-purpose applicability of the method within the CORMAB family.

## Weaknesses

### Fatal

None.

### Major

- **Insufficient baselines to establish superiority over plausible alternatives.** The paper's experimental baselines fall into two categories: (a) myopic methods that do not address sequential planning at all (MYOPIC, SAMPLING, ITERATIVE MYOPIC, RANDOM, NO ACTION), and (b) ITERATIVE DQN, which is a deliberately weakened ablation of SEQUOIA (greedy heuristic action selection instead of MILP). No baseline simultaneously addresses sequential planning *and* combinatorial action selection in a non-trivial way. Plausible alternatives include: policy-gradient methods (PPO/SAC) with action masking or feasibility penalties; MCTS combined with a learned value function (which the paper mentions in the context of AlphaGo in Section 6 but does not evaluate); or approximate Whittle index policies adapted to coupled settings. The paper's claim that "We are not aware of additional RL algorithms from the literature that could be applied to CORMAB" (Section 5) does not justify omitting adapted versions of standard approaches. Without at least one strong baseline that addresses both challenges, the 24.8% improvement claim is hard to interpret — improvement over trivially weak alternatives does not establish that SEQUOIA is a *good* approach, only that it is better than methods that are not designed for this setting.

- **Generality claim is unsupported by evaluation.** The paper claims SEQUOIA "generalizes to other sequential planning problems with per-timestep combinatorial actions" (Section 4, Section 7), but this claim rests entirely on the CORMAB domain. A key training strategy — training on infeasible actions (Section 4.2) — explicitly relies on the RMAB-specific property that "we can simulate valid state transitions even for infeasible actions because the state transitions are defined independently per-arm." This property does not hold in general MDPs, where infeasible actions may produce undefined transitions. The paper does not evaluate on any non-RMAB domain, making the generality claim entirely conjectural.

### Minor

- **Scalability to truly exponential action spaces is not demonstrated.** The evaluated instances use N ∈ {5, 10, 20} actions and J ∈ {20, 40, 100} arms. For the budget-constrained case, the largest action space is C(20,5) = 15,504 — small enough for exhaustive enumeration. The paper frames the problem as having "exponentially large" action spaces, and while J=100 with B=20 yields C(100,20) ≈ 5.4×10²⁰ actions in principle, the MILP does not enumerate them and its wall-clock solve time is only reported in the appendix. The conclusion's admission that an IPOPT-based continuous relaxation "finds significantly better actions than Gurobi, if we allot the same amount of time" (Section 7) raises the question of whether the MILP approach would actually be practical at scales where enumeration is truly infeasible. This is addressable in rebuttal with runtime analysis.

- **The "24.8% improvement" headline is presented without sufficient context.** The abstract claims this improvement across "existing methods," but since none of the baselines simultaneously address sequential planning and combinatorial action selection, this percentage overstates the practical significance. The positioning "for the first time, sequential combinatorial settings" (p.2) is also too strong given that Delarue et al. (2020), which the paper itself cites, considers sequential combinatorial action selection (albeit in a simpler, mostly deterministic setting).

- **Bipartite matching and capacity-constrained formulations are described too tersely in the main text.** Section 3 devotes one paragraph of prose to each, with MILP formulations entirely deferred to Appendix C. This makes it difficult to assess the difficulty or structure of these problems from the main text alone.

### Trivial

None.

## Nice-to-Haves

- A comparison against at least one policy-gradient or MCTS-based baseline adapted to combinatorial action spaces would substantially strengthen the empirical case.
- Evaluation on one non-RMAB domain (e.g., stochastic scheduling) would substantiate the generality claim.
- An ablation isolating the MILP action selection (e.g., using random/ Sampling action selection during training but MILP only at evaluation) would more cleanly separate the contribution of the MILP from the Q-network training procedure.
- Analysis of Q-function quality on feasible vs. infeasible actions would clarify whether training on infeasible actions introduces systematic bias.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Harsh critic: "IPOPT outperforming Gurobi undermines the MILP motivation"** — This is overstated. The paper presents IPOPT as a practical *alternative for real-time applications*, not as a replacement. The MILP approach guarantees optimality over the Q-network; IPOPT with randomized rounding is a heuristic that performs well under equal time budgets. This is a useful practical observation, not a fatal flaw. Kept as a minor concern about scalability rather than a fatal one.
- **Harsh critic: "SAMPLING does not mirror He et al. (2016)"** — The paper states the SAMPLING baseline "mirrors the method proposed by He et al. (2016)" (Section 5). He et al. propose action elimination with DQN, which is not simply random sampling of k actions. However, this is a minor characterization issue that does not affect the validity of the experiments or the main claims.
- **Harsh critic: "Normalization by RANDOM reward makes differences visually small"** — This is a presentation preference. The normalization is clearly explained (R_RANDOM = 1) and makes the lower-bound methods comparable. Not a substantive weakness.
- **Harsh critic: "claim of novel formulations is likely overblown"** — The paper states "To the best of our knowledge, the following problem formulations are all novel for restless bandits" (Section 3), which is carefully scoped. Whether combining standard CO constraints with RMABs is "novel" is a judgment call; the paper provides specific instantiations with full MILP formulations.
- **Strength finder: "Generality demonstrated without per-domain tuning" as a supporting strength** — While true within the CORMAB family, this does not substantiate the broader generality claim to non-RMAB domains. Kept as a strength but its weight is reduced by the major weakness on unsupported generality.

## Novel Insights

The most insightful observation is the tension between SEQUOIA's two contributions: the *algorithmic* contribution (DQN+MILP integration) is well-supported by the ITERATIVE DQN ablation, while the *problem-formulation* contribution (CORMAB) is novel but the empirical evaluation does not yet establish that CORMABs require the DQN+MILP approach specifically — it only establishes that methods not designed for this setting perform poorly. The paper would be significantly strengthened if the experimental design had a "fair fight" between SEQUOIA and an adapted standard RL method.

## Suggestions

- Add at least one baseline that does both sequential planning and combinatorial selection: e.g., PPO with action masking, or a simple MCTS rollout using the learned Q-network as a value estimator. Even a negative result (showing these don't work) would strengthen the paper.
- Consider rephrasing the "24.8% improvement" claim to specify the comparison class, e.g., "24.8% improvement over methods that do not address both sequential planning and combinatorial selection simultaneously."
- Move the IPOPT comparison from the conclusion footnote to a proper discussion in the experiments section, with analysis of when exact MILP vs. continuous relaxation is preferred.

## Score and Decision

**Calibration anchors:**

- **High (avg >7):** DNC (7.33, Accept poster) — similar topic (RL with combinatorial/large discrete action spaces), strong baselines and ablations; Neur2RO (6.67, Accept poster) — neural network embedded in optimization, strong empirical improvements with limited baselines but reviewers still noted baseline gaps; Sequential resource allocation A3YUPeJTNR (8.0, Oral) — sequential planning in restless-arm-adjacent setting with strong theory.
- **Medium (4–6):** TreeDQN (4.25, Reject) — DQN+MILP for branch-and-bound, limited generalization and missing baselines; Big Learning (5.25, Reject) — novel formulation but insufficient baselines; DNC (80wh3jjCZf) at 7.33 is the closest topic match and was accepted despite some baseline concerns but had 3 SOTA baselines and ablations.
- **Low (<3):** Hyperion (3.0, Reject) — overclaimed improvements against weak/outdated baselines; Vision-free grammar induction (2.33, Reject) — unfair comparisons. These papers claimed large improvements against inadequate baselines.

This paper's situation shares elements with both the medium and low anchors: it has a genuinely novel problem formulation and a principled algorithm, but the empirical case is undermined by the absence of any strong baseline that addresses the same problem. The TreeDQN paper (4.25) was rejected despite having some SOTA baselines because of generalization/scalability concerns. The Hyperion paper (3.0) was rejected primarily for weak baselines and overclaimed results. This paper is better than Hyperion (the problem formulation is genuinely novel and the algorithm is well-motivated), but worse than Neur2RO (which demonstrated dramatic computational speedups against well-established solvers) and worse than DNC (which had 3 SOTA baselines and action spaces up to 10^73). The paper's core contribution — the CORMAB formulation — is valuable, but the experimental validation does not meet the bar for confidence in the empirical claims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>