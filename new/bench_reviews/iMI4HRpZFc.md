## Summary

This paper introduces "delusions" in target-directed RL agents—systematic false beliefs about target reachability arising from improper coordination between a target generator and estimator. It provides a taxonomy classifying delusions into generator-side types (G.1: nonexistent targets; G.2: temporarily unreachable targets) and estimator-side types (E.0/E.1/E.2 misevaluations), demonstrates these failure modes in a custom MiniGrid environment (SwordShieldMonster), and proposes two new hindsight relabeling strategies ("generate" and "pertask") plus hybrid mixtures that tailor training data for generator vs. estimator, showing reduced delusional behavior and improved OOD generalization for Skipper and LEAP agents.

## Strengths

- **Real and under-discussed failure mode identification**: The paper surfaces a genuine problem—target-directed agents systematically misestimate reachability of impossible or temporarily unreachable goals due to biased training distributions. This is a concrete and practically important issue that existing HER and goal-conditioned RL work has not systematically addressed. The distinction between G.1 (permanently unreachable) and G.2 (temporarily unreachable) targets is particularly valuable, as G.2 arises from irreversible dynamics that are common in real settings but rarely analyzed.

- **Principled diagnostic evaluation**: Rather than reporting only OOD success rates, the paper traces a causal chain from estimation errors (E.0/E.1/E.2) → delusional behavior frequencies → OOD generalization (Fig. 3). The SSM environment enables ground-truth distance computation and clear delusion classification, making this dissection possible. This is better diagnostic practice than typical RL papers.

- **The 2-slotted hybrid approach is a useful design insight**: The observation that generators and estimators have conflicting training data needs—generators benefit from clean data, estimators need exposure to problematic targets—and the corresponding suggestion to separate their relabeling processes is simple but principled. This insight extends naturally beyond HER.

- **Concrete empirical gains**: The hybrid strategies (notably F-(E+P+G)) achieve substantially better OOD success rates than any single atomic strategy (Fig. 3h), and the improvement is clearly attributable to reduced delusional behaviors rather than generic sample efficiency.

## Weaknesses

### Major:

- **Limited empirical breadth undermines the general applicability claim**: Despite repeated claims of broad applicability ("strategies applicable to general target-directed agents," "adapting beyond HER should be straightforward"), the main paper presents detailed results for only one environment (SSM) and one method (Skipper). Both Skipper and LEAP use explicit distance estimators and HER-based training—architectures where the proposed strategies are straightforwardly implementable. The paper claims "4 sets of experiments align in terms of conclusions" but relegates 3/4 to the appendix. No results on continuous control, high-dimensional observations, or non-HER-based target-directed methods (e.g., Director, model-based planners) are provided. This is a significant gap between the scope of the claims and the scope of the evidence.

- **Narrow baseline comparisons for proposed relabeling strategies**: The practical contribution is two new HER sampling strategies and their mixtures, yet the experimental comparison is limited to atomic HER variants (future/episode/pertask/generate) and hand-tuned mixtures. Existing HER mixture strategies from prior work (e.g., the 3-strategy mixtures in Nasiriany et al., 2019; Yang et al., 2021a's similar "generate"-like approach) are discussed in related work but not implemented as baselines. There is also no comparison against alternative approaches to handling unreachable goals (e.g., reachability classifiers, uncertainty-based rejection), which would help establish whether the specific relabeling strategies are necessary rather than any coverage-expanding mechanism being sufficient. The paper acknowledges Yang et al. (2021a) used a "similar" mixture to "generate" but claims the delusion impact was not explored—yet does not compare against that exact mixture as a baseline.

- **Mixture proportions are hand-tuned without sensitivity analysis**: The hybrid strategies use specific ratios (F-(E+G): 50/50; F-(E+P+G): 50/25/25) with no principled justification or sensitivity sweep. If these proportions need careful per-environment tuning, the practical value of the guidelines in Section 7 and the claim of "autonomous" delusion avoidance is weakened. No analysis of performance robustness to mixing ratio changes is provided.

### Minor:

- **The "delusion" framing adds terminology more than analytic precision**: The taxonomy maps cleanly to existing concepts—G.1 corresponds to unreachable states in the MDP, G.2 to reachability conditioned on the current state equivalence class, and E.0/E.1/E.2 to estimation errors over different subsets of source-target pairs. The psychiatric analogy (delusion vs. hallucination) does not yield new formal properties, metrics, or predictions beyond what standard notions of coverage, reachability, and off-policy bias already provide. The taxonomy is descriptively useful, but the framing overstates the conceptual novelty.

- **Computational cost of "generate" is unquantified**: Section 4.1.1 acknowledges that "generate" requires running the generator at every training step, incurring "additional computational burden, depending on the complexity of target generation processes." No wall-clock time comparison or FLOPs analysis is provided, making it difficult for practitioners to assess the cost-benefit tradeoff.

- **The "pertask" strategy's tradeoffs are empirically underexplored**: Table 1 notes that pertask "can cause extensive G.2 targets if used to train generators" and has "low efficiency in learning close-proximity source-target relationships," and Fig. 3(d) shows F-P has significantly worse short-distance estimation. These downsides are noted but their quantitative impact on the hybrid strategies is not explicitly isolated. Understanding when pertask's benefits outweigh its costs is critical for the practical guidelines in Section 7.

- **Ground-truth distance computation relies on privileged access unavailable in realistic domains**: The estimation error metrics rely on computing exact shortest-path distances between all state pairs, which is only feasible in small discrete environments. The paper claims general applicability, but its diagnostic tools would not scale to domains where delusions might be most consequential. This mismatch should be explicitly acknowledged.

### Trivial:

- The psychiatric framing occasionally obscures rather than illuminates—terminology like "belief formation" and "belief evaluation systems" adds overhead without analytical payoff.

## Nice-to-Haves

- Validation on at least one environment with continuous state space or high-dimensional observations (e.g., robotic manipulation with irreversible state changes) to test whether G.2 delusions arise and whether pertask/general scale.
- Comparison against an explicit reachability/feasibility classifier as a baseline, to isolate whether the benefit comes from the specific relabeling strategies or from broader data coverage.
- Ablation or sensitivity analysis on mixture proportions to establish robustness.

## Removed Points

- **Criticism that the paper overclaims novelty relative to prior work on goal misgeneralization, HER failures, etc.**: The paper does cite Di Langosco et al. (2022), Jafferjee et al. (2020), Zhao et al. (2024), and others in related work, and positions its contribution as identifying *delusions as a systematic subclass* of these failure modes with targeted mitigations. The contribution claim is about the taxonomy and targeted strategies, not about discovering these failures from scratch. Removed as it mischaracterizes the paper's novelty claim.

- **Criticism that safety claims are unsupported because no concrete safety metrics appear in experiments**: The introduction mentions "safety catastrophes" as motivation, and the introduction's E.1 discussion notes "potentially catastrophic if the G.1 targets are beyond safety constraints." The experiments focus on demonstrating the phenomenon and mitigation, not on safety per se. Criticizing the lack of safety metrics is scope creep—the paper's stated scope is identifying and addressing delusions, not benchmarking safety.

- **Criticism that initial state distributions are designed to amplify G.2 risk, making the setting unfair**: The paper is explicit about this design choice (Section 5.1: "This change increases risks of E.2") and it is an appropriate stress test for the proposed methods. A stress test that amplifies the failure mode being studied is standard methodology.

- **Criticism about lack of non-target-directed baselines**: The paper's scope is improving target-directed agents; comparing against non-planning methods (e.g., flat policy) would not address whether the proposed strategies fix delusions within that framework. This is scope creep.

- **Demand for theoretical proofs of convergence**: This is an empirical methods paper studying a failure mode and proposing practical mitigation strategies. Demanding formal convergence guarantees is outside the paper's stated scope and community norms.

## Novel Insights

The distinction between G.1 (permanently nonexistent targets) and G.2 (temporarily unreachable targets) is a genuinely useful categorization that maps onto different training data deficiencies and requires different mitigations. G.2 delusions are particularly insidious because the targets are valid states that the estimator may have correctly learned about from other initial conditions, creating a false sense of competence. The insight that generators and estimators have *conflicting* training data needs—generators should avoid problematic targets, estimators should be exposed to them—and the corresponding 2-slotted approach is a clean architectural principle that transcends the specific HER instantiations.

## Suggestions

1. **Include at least one additional experiment set in the main text** (e.g., LEAP on SSM), with a clear table summarizing all 4 experiment sets. This would substantiate the "all 4 sets align" claim visible to readers without appendix access.
2. **Run a simple sensitivity sweep on mixing proportions** (e.g., ±10% around the chosen ratios for F-(E+P+G)) and report whether OOD performance is robust or sensitive.
3. **Add wall-clock time comparisons** for each strategy variant to quantify the computational overhead of "generate" and the memory cost of "pertask."
4. **Explicitly acknowledge the scalability limitations** of ground-truth distance computation and state equivalence class identification in the conclusion, and discuss how diagnostics could be approximated in realistic domains.

## Score and Decision

**Calibration**: Compared to similar papers in goal-conditioned RL and HER augmentation:
- *Skipper (Zhao et al.)*: Scores 5-6, accepted as poster. Similar gridworld focus, identified a real failure mode, limited environment diversity.
- *Bad Habits (Policy Confounding)*: Scores 5-6, rejected. Identified a real problem with toy-environment diagnostics but limited breadth.
- *HInt (Null Counterfactual)*: Scores 5-8, accepted. HER augmentation with similar scope concerns but stronger theoretical grounding.
- *Breadth-First Subgoal Exploration*: Scores 5-6, rejected. Grid-world-centric, limited baselines.

This paper identifies a genuine, under-examined failure mode and provides principled mitigations, but its empirical evidence is narrow (one environment-method pair in the main text), comparisons are limited within-family, and the conceptual framing overclaims relative to its technical contribution. It sits above the purely diagnostic "Bad Habits" paper (which had weaker mitigations) but below HInt (which had stronger theoretical backing and broader experiments). Roughly comparable to Skipper in contribution level.

**Score: 5** — The delusion taxonomy and targeted mitigation strategies are a useful contribution, but the narrow empirical evaluation and limited baseline comparisons prevent strong confidence in the claimed generality. The conceptual novelty is somewhat overstated relative to the technical contribution (two new HER sampling strategies plus mixtures).

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>