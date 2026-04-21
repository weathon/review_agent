Now I have a thorough understanding of the paper and calibration data. Let me compose the final review.

## Summary

The paper introduces a taxonomy of "delusions" in target-directed RL agents, distinguishing generator-side (G.1: nonexistent, G.2: temporarily unreachable) and estimator-side (E.0, E.1, E.2) failure modes, where agents hold false beliefs about targets. It proposes two HER relabeling strategies—"generate" (from prior work) and "pertask" (novel)—and a 2-slotted hybrid architecture that provides different training data to the generator and estimator. Experiments on the purpose-built SwordShieldMonster (SSM) environment and an additional environment, across two frameworks (Skipper, LEAP), demonstrate that hybrid strategies reduce delusion-related errors and improve OOD generalization.

## Strengths

- **The G.2/E.2 taxonomy distinction is genuinely insightful.** The identification that temporarily unreachable targets (due to irreversible state transitions) constitute a distinct and overlooked failure mode is a real conceptual contribution. Section 3.1.2 defines G.2 with concrete SSM examples (e.g., after acquiring the sword, class ⟨1,0⟩ cannot reach ⟨0,0⟩), and Section 3.2.3 shows how E.2 delusions arise. This is a principled distinction absent from prior GCRL literature.

- **The 2-slotted hybrid architecture is a well-motivated design principle.** Section 4.3 identifies that generators benefit from avoiding exposure to problematic targets while estimators need such exposure, and proposes separate relabeling processes for each. Table 1 makes the tradeoffs explicit (e.g., "pertask" helps the estimator against E.2 but causes excessive G.2 targets if used for generator training). This insight generalizes beyond HER.

- **Clear causal chain in experimental results.** Figure 3 traces a consistent path from reduced E.2 estimation errors (subplot f) → reduced delusional behavior frequencies (subplot g) → improved aggregated OOD performance (subplot h). The decomposed evaluation metrics (estimation errors by delusion type, behavior frequencies) enable targeted diagnosis that standard success-rate reporting cannot provide.

- **Empirical guidelines are actionable.** Section 7 provides concrete, step-by-step recommendations for practitioners (e.g., "Analyze the state structure of the target tasks and identify temporary unreachabilities"), making the theoretical contributions readily applicable.

## Weaknesses

### Fatal
None.

### Major

- **Validation primarily on a purpose-built environment designed to exhibit the predicted failure modes.** The SSM environment explicitly constructs the sword-and-shield mechanic to instantiate irreversible transitions and equivalence classes (Section 2), directly embedding G.1 and G.2 failures. While this is a valid controlled-experiment approach for isolating the phenomenon, it provides limited evidence that the proposed solutions transfer to settings where these failure modes arise organically. The second environment and the LEAP framework help, but no experiment is conducted on a standard goal-conditioned RL benchmark where irreversible dynamics (e.g., robotics manipulation with dropped objects) arise without being designed by the authors. Demonstrating the G.2/E.2 phenomenon and its mitigation on even one established benchmark would substantially increase confidence in the generality of the claims.

- **Baselines limited to HER relabeling variants within the same framework.** All baselines are HER strategies (F-E, F-P, F-G) applied to Skipper/LEAP. There is no comparison with fundamentally different approaches that could address similar problems—e.g., goal-conditioned RL with feasibility estimation, model-based planners with reachability analysis, or even goal-filtering modules. While the paper's scope is HER-focused, the claims about generality ("applicable generally," Section 4.1) would be better supported by at least one comparison with an alternative approach. The absence makes it unclear whether the proposed relabeling tricks are competitive with, subsumed by, or complementary to other architectural solutions.

### Minor

- **Mixture proportions are hand-tuned without sensitivity analysis.** The hybrid strategies use specific ratios (50/50 for F-(E+P); 50/25/25 for F-(E+P+G), Section 5.4) with no justification or sensitivity analysis. Since the main experimental argument is that hybrids outperform pure strategies, the fragility of this improvement to the mixing proportions matters. A few alternative mixture configurations would strengthen confidence in the conclusion.

- **Overclaimed generality of proposed strategies.** The assertion that "our proposed ideas do not rely on additional assumptions" (Section 4.1) is not fully accurate: "pertask" implicitly assumes task-relevant states have been visited in the training buffer, and "generate" assumes the generator's training-time distribution approximates its decision-time distribution. These are reasonable assumptions, but they exist.

- **The "delusion gap" is promised but not cleanly quantified.** Section 5.2 introduces the "delusion gap" concept ("the amount of performance degradation caused by delusions") but the paper never provides a clear numerical characterization of this gap. A controlled experiment comparing an oracle estimator with ground-truth reachability against learned estimators would cleanly establish this quantity.

- **E.0 is a residual category.** E.0 is defined as "misevaluating non-delusional targets" (Section 3.2.1), which captures all estimation error that isn't E.1 or E.2. This makes E.0 more of a catch-all than a distinct delusion type. The taxonomy would be sharper with a more precise characterization of E.0 or its decomposition.

### Trivial
None worth noting.

## Nice-to-Haves

- Validation on at least one standard GCRL benchmark with naturally irreversible dynamics (e.g., robotic manipulation where a dropped object cannot be recovered), to establish that delusions are endemic rather than an artifact of SSM's design.
- Comparison with a model-based baseline that performs reachability estimation, to situate the proposed approach within broader alternatives.
- Trajectory-level visualizations showing a hybrid agent correctly rejecting a G.2 target that a baseline agent pursues, complementing the aggregate statistics.
- Sensitivity analysis on the mixture proportions, testing at least 2-3 alternative ratios per hybrid strategy.

## Removed Points

These points were flagged for removal and should be treated with caution:

- **"Psychiatric framing adds no technical content"** — While the "delusion" terminology is borrowed from psychiatry, the paper uses it as a framing device, not a technical claim. This is a stylistic choice common in many papers and does not constitute a substantive weakness. Removed as a formatting/style nitpick.

- **"Three of four experiment sets in the appendix"** — The appendix contents exist in the original submission but were stripped by the parser. This is not an author error. The paper does summarize the appendix results (Section 5.6) and states that all 4 sets align. Removed as a parser artifact concern.

- **"Zhao et al. (2024) may share authors — self-referential"** — This is speculation about author identity in a double-blind submission. The citation is to related prior work that is properly attributed. Removed as it questions the existence/validity of a cited reference.

- **"F-P devastates OOD performance despite better E.2 accuracy — deserves deeper analysis"** — The paper actually explains this tradeoff clearly in Section 5.5: "pertask" biases training toward long-distance pairs at the expense of short-distance non-delusional estimation, which destroys basic function. This is a core design insight, not an unexplained mystery. Removed as a misunderstanding of the paper's content.

- **"Combine strategies with methods outside the HER family"** — The paper scopes itself to HER-based training and explicitly acknowledges this. Requesting experiments outside the stated scope is a nice-to-have at best. Removed as a scope-creep demand elevated to a weakness.

## Novel Insights

The most underappreciated insight in this paper is the diagnosis that generators and estimators have *conflicting training data needs*: generators benefit from clean, achievable targets, while estimators need exposure to problematic targets to learn to reject them. This architectural tension is not obvious and motivates the 2-slotted hybrid as a structural rather than merely algorithmic solution. The G.2 distinction—semantically valid targets that are temporarily unreachable due to irreversible dynamics—is also a genuinely overlooked failure mode that has practical relevance in many real-world RL settings (e.g., resource depletion, irreversible actions in robotics).

## Suggestions

- Add at least one experiment on a standard GCRL benchmark with natural irreversibility (even a simple robotic manipulation task) to demonstrate that G.2/E.2 delusions arise and are mitigated organically, not just by construction in SSM.
- Conduct a sensitivity analysis on the mixture proportions (e.g., test 30/70, 50/50, 70/30 for F-(E+P)) to show the hybrid improvement is robust.
- Provide a quantitative definition of the "delusion gap" with an oracle estimator baseline, turning an intuitive but vague concept into a measurable quantity.
- Narrow the generality claims in Section 4.1 to acknowledge the implicit assumptions in "pertask" and "generate."

## Score and Decision

**Calibration summary:**

- **High-scoring anchors (>7):** Papers with novel taxonomies/frameworks for RL failure modes validated in controlled settings scored 7.0–7.5 (Curse of Diversity: 7.5; Correlated Proxies/Reward Hacking: 7.2; Mechanistic Interpretability Identifiability: 7.0; CEB Fairness Benchmark: 7.5). These papers had both strong conceptual contributions and broader empirical validation or were in more mature domains.

- **Medium-scoring anchors (4–6):** Papers with narrow baselines within their own framework scored 4.2–5.8 (OpenStereo: 4.5; TGForecaster: 5.0; DA-Bench: 5.75; ITM: 4.2). These typically had limited external validation and methodological scope concerns.

- **Low-scoring anchors (<3):** Truly circular evaluations (LOLAMEME: 3.0; QualEval: 2.33) where the evaluation was self-referential and no insight was gained from experiments.

This paper sits between the medium and high bands. It has a genuinely novel and useful conceptual contribution (G.2/E.2 taxonomy, 2-slotted design) with meaningful controlled experiments. However, it lacks validation on standard benchmarks and has no baseline comparisons beyond HER variants—weaknesses that place it below the 7+ anchors. It is substantially better than the low-scoring circular-evaluation papers because SSM is a controlled environment for isolating a phenomenon (valid experimental methodology), not an arbitrary benchmark designed to make the method look good. The paper is also better than many medium-band papers because its conceptual contribution is more novel and the evaluation more thorough within its scope. I place it in the 5.5–6 range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>