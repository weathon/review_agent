## Summary

The paper identifies and categorizes failure modes of target-directed RL agents, where agents blindly chase unreachable or unsafe self-generated targets due to miscalibrated estimators. It introduces a taxonomy of "delusions" — generator-side (G.1: nonexistent targets, G.2: temporarily unreachable targets) and estimator-side (E.0, E.1, E.2) — and proposes two relabeling strategies ("generate" and "pertask") plus a 2-slotted hybrid approach that tailors training data separately for generators and estimators. Experiments on the SwordShieldMonster grid-world environment with the Skipper method (and additional sets in the appendix with LEAP and a second environment) show that hybrid strategies reduce estimation errors and delusional behaviors, leading to improved OOD generalization.

## Strengths

- **Systematic diagnostic taxonomy with causal structure**: The G.1/G.2/E.0/E.1/E.2 categorization (Sections 3.1–3.2) connects generator pathologies to estimator failures with clear causal links. The G.2/E.2 distinction — valid targets that are temporarily unreachable due to irreversible state transitions — is a particularly useful contribution, as this temporal dependency is underappreciated compared to simple hallucination (G.1).

- **Rigorous evaluation methodology decomposing errors by delusion type**: Section 5.2's approach of measuring estimation errors split by E.0/E.1/E.2, delusional behavior frequencies, and OOD success separately provides mechanistic evidence rather than just outcome-level metrics. This decomposition reveals the tradeoffs between strategies (e.g., F-P's decent E.2 accuracy vs. poor short-distance non-delusional accuracy in Figure 3d,h).

- **2-slotted hybrid approach resolves generator-estimator conflict**: Section 4.3 identifies that generators benefit from learning only viable targets while estimators benefit from exposure to problematic targets — a conflict that single-strategy training cannot resolve. The hybrid strategies (e.g., F-(E+P+G) using "future" for the generator slot and a mixture for the estimator slot) explicitly accommodate these divergent needs, which is a practical architectural insight.

- **"Pertask" strategy is a well-motivated intervention for E.2 delusions**: The connection between the diagnosed failure mode (estimators lack exposure to cross-episode targets that are unreachable from the current equivalence class) and the intervention (relabeling with targets from other episodes within the same task) is clear and directly validated in Figure 3f,g.

- **Experimental pipeline shows delusion reduction maps to OOD improvement**: The chain from lower E.2 estimation errors (Figure 3f) → fewer delusional behaviors (Figure 3g) → better OOD success (Figure 3h) for F-(E+P) and F-(E+P+G) provides coherent evidence for the paper's central claim.

## Weaknesses

### Fatal
None.

### Major

- **Limited experimental scope for claimed generality**: The paper claims its strategies "should be expected to be applicable generally" (Section 4.1, line 132) and its title promises broad insights for "target-directed decision-making," yet all four experiment sets use grid-world environments built on MiniGrid (12×12 for SSM) with two methods (Skipper, LEAP) that both use HER-based training and involve discrete state spaces with hand-crafted irreversible dynamics. The "pertask" strategy in particular requires storing all past observations (Section 4.1.2, line 154: "requires algorithmic designs recording all past observations"), which may face scaling issues in continuous or high-dimensional settings. Without testing on any standard continuous-control or higher-dimensional benchmark, the general-applicability claim rests on argument rather than evidence.

- **"Generate" strategy novelty is overstated relative to its attribution**: Section 4.1.1 explicitly credits Zhao et al. (2024) for the core idea ("proposed to train the estimator additionally with candidate targets proposed by the generator") and presents the JIT HER reformulation as the paper's contribution. However, Table 1 lists both "generate" and "pertask" as "proposed in this paper" without distinguishing the different levels of novelty. Since "generate" constitutes half of the two proposed strategies, the paper's methodological novelty is narrower than presented — the primary novel interventions are "pertask" (cross-episode relabeling) and the 2-slotted hybrid framework, not "generate" itself.

### Minor

- **Hybrid strategy mixing proportions are unmotivated and untested**: The three hybrid strategies use specific ratios — F-(E+G) at 50/50, F-(E+P) at 50/50, F-(E+P+G) at 50/25/25 (Section 5.4) — with no theoretical or empirical justification, and no ablation over alternative proportions. Since these hybrids represent the paper's main practical contribution, the absence of sensitivity analysis means we cannot tell whether the reported improvements are robust or sensitive to this hyperparameter choice.

- **The psychiatric "delusion" framing adds terminology more than analytical depth**: While the taxonomy organizes real phenomena, the G.1/G.2 distinction maps to well-known concepts in goal-conditioned RL (goal feasibility / reachability), and the E.0/E.1/E.2 categories follow directly from what the estimator is misevaluating. The solutions (diversify training data, expose estimator to generated goals) follow from standard reasoning about these problems. The delusion framing does not lead to interventions that would not be designed without it. This does not invalidate the contributions but means the conceptual novelty is more organizational than substantive.

- **Guidelines in Section 7 are partially generic**: Point 2 ("Use proper update rules and try to maximize the diversity of the training data") is advice that predates this paper. Point 4 ("Analyze the state structure... identify temporary unreachabilities") assumes access to the environment's reachability structure, which may not be available in practice. Points 1 and 3 are more specific and actionable.

### Trivial
None.

## Nice-to-Haves

- Experiments on at least one continuous or higher-dimensional environment (e.g., robotic manipulation) to validate that the strategies and the "pertask" scaling approach transfer beyond grid worlds.
- Ablation over mixing proportions (e.g., 75/25 vs. 50/50 for F-(E+P)) to establish robustness of the hybrid approach.
- Trajectory-level case studies showing the behavioral difference between delusional and non-delusional agents on the same OOD task, providing intuitive understanding beyond error metrics.
- Comparison to alternative delusion-mitigation approaches (e.g., modifying the generator's training directly to reduce G.1/G.2 proposals, or using safety/constraint filters distinct from the estimator).

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Critic's claim that "the relationship between estimation accuracy and downstream performance is more complex than the paper acknowledges"**: The paper explicitly discusses this in Section 5.5, noting that F-P's "low accuracy in short-distance non-delusional estimation (d)) devastated the baseline to the lowest performance in h)." The paper acknowledges the tradeoff between delusion-correction and non-delusional accuracy; the critic's claim that the paper ignores this is a misread.

- **Critic's demand for experiments on D4RL or robotic manipulation benchmarks as a fatal flaw**: Treating this as a fatal issue overstates the case. The paper's primary contribution is diagnostic — identifying and categorizing failure modes with controlled environments — and simple environments are appropriate for this purpose. The concern about generality is legitimate and kept as a Major weakness, but it is not fatal to the paper's diagnostic contribution.

- **Critic's claim that the SSM environment "simply validates the authors' own design choices"**: The SSM environment is designed to exhibit specific failure modes, which is a standard practice in diagnostic work. The paper also tests on a second environment (in the appendix) that is "dominantly haunted by G.1" (different from SSM's G.2 dominance), providing some variation in failure mode coverage.

- **Critic's demand for comparison to alternative delusion-mitigation approaches**: This is a reasonable suggestion but overreaches as a required weakness. The paper's contribution is identifying and categorizing the problem, and proposing targeted strategies. Comparing to all possible alternative approaches would be a nice-to-have, not a methodological requirement.

- **Critic's demand for analysis of how the delusion taxonomy changes intervention strategy**: This is a valid point for deepening the paper but is more of a nice-to-have. The paper does show that different delusion types require different strategies (Table 1, Section 4.1), which is evidence that the taxonomy has practical value even if the interventions could have been designed without it.

- **Critic's claim that "the abstract claims significant improvements based on experiments limited to simple grid worlds"**: The paper does report significant improvements on its tested environments. The concern about generalizability is captured in the Major weakness about experimental scope; the abstract's claim is accurate for what was tested.

- **Critic's note about missing appendix experiments**: The appendix was stripped by the parser. The original submission contains the additional 3 experiment sets. Criticizing the paper for appendix content that exists but was not parsed is not valid.

## Novel Insights

The paper's most insightful observation is the generator-estimator conflict in training data needs (Section 4.3): generators benefit from learning only viable targets (to reduce hallucination), while estimators benefit from exposure to problematic targets (to learn to reject them). This conflict makes single-strategy training fundamentally suboptimal and motivates the 2-slotted approach — a structural insight that goes beyond just proposing new relabeling strategies and explains why existing HER strategies inevitably produce delusional agents even when they seem to work well on in-distribution tasks.

## Suggestions

- Add at least one experiment on a non-grid-world environment (e.g., a continuous control task with irreversible dynamics) to support the general-applicability claim, even if simplified.
- Run a small ablation over mixing proportions for at least one hybrid strategy (e.g., F-(E+P) at 75/25 vs. 50/50 vs. 25/75) to demonstrate that improvements are not brittle to this choice.
- Clarify in Table 1 that "generate" builds on Zhao et al. (2024) to avoid the impression that both strategies have equal novelty.

<context>
**Paper summary**: The paper introduces a taxonomy of "delusions" in target-directed RL — classifying generator-side failures (G.1: nonexistent, G.2: temporarily unreachable) and estimator-side failures (E.0, E.1, E.2) — and proposes two HER-based relabeling strategies ("generate" adapted from Zhao et al. 2024, and the novel "pertask" cross-episode strategy) plus a 2-slotted hybrid approach that tailors training data separately for generators and estimators. Experiments on the SSM grid-world (and 3 additional sets in the appendix) with Skipper and LEAP show that hybrid strategies reduce estimation errors and delusional behaviors, leading to improved OOD generalization.

**Original reviewer signal**: Harsh Critic views the paper as incremental with overclaimed generality and novelty (limited to grid worlds, "generate" adopted from prior work, taxonomy adds terminology more than depth). Strength Finder views the taxonomy as a systematic contribution with clear causal structure, the 2-slotted hybrid as a practical architectural insight, and the experimental pipeline as providing coherent mechanistic evidence.

**What was dropped and why**: (1) Critic's claim that the paper doesn't acknowledge the complex relationship between estimation accuracy and OOD performance — verified against Section 5.5, which explicitly discusses the F-P tradeoff. (2) Critic's framing of limited experiments as a fatal flaw — kept as Major but not fatal, since the diagnostic purpose justifies controlled environments. (3) Critic's claim that the SSM environment merely validates the authors' own design choices — this is standard practice for diagnostic work. (4) Demand for alternative approach comparisons — moved to nice-to-have. (5) Demand for D4RL/robotic manipulation as required — kept as major concern about generality claims but not treated as fatal.

**Cross-checks performed**: (1) Verified that "generate" is explicitly credited to Zhao et al. (2024) in Section 4.1.1 but Table 1 claims both strategies as "proposed in this paper" — confirmed overclaim. (2) Verified that mixing proportions (50-50, 50-25-25) are presented without ablation — confirmed. (3) Verified that the paper does discuss the F-P estimation/performance tradeoff in Section 5.5 — critic's claim of non-acknowledgment is wrong. (4) Verified the "applicable generally" claim in Section 4.1 line 132 — confirmed. (5) Verified Section 7 guidelines content — partially generic but not entirely.

**Review construction notes**: The paper's primary contribution is diagnostic (identifying and categorizing failure modes with controlled environments), and simple environments are appropriate for this purpose. The generality concern is real but should not be treated as fatal. The "generate" novelty overclaim is real but the paper does credit the source. The 2-slotted hybrid framework and "pertask" strategy are the genuine novel contributions. The psychiatric framing is engaging but adds limited analytical depth beyond organizing known concepts under new terminology.
</context>