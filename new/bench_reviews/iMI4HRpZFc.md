Now I have a comprehensive understanding of the paper. Let me compose the final review.

## Summary

The paper introduces a taxonomy of "delusions" in target-directed RL agents—distinguishing generator-side issues (G.1: nonexistent targets, G.2: temporarily unreachable targets) from estimator-side failures (E.0, E.1, E.2)—and proposes two new hindsight relabeling strategies ("generate" and "pertask") plus a 2-slotted hybrid approach that separately serves the conflicting training needs of generators and estimators. The strategies are evaluated on the custom SSM gridworld environment and a second environment, across two agent frameworks (Skipper and LEAP), showing that hybrid strategies reduce estimation errors and delusional behaviors, leading to improved OOD generalization.

## Strengths

- **Systematic identification and categorization of delusion types**: The taxonomy (G.1/G.2 for generator-side, E.0/E.1/E.2 for estimator-side) provides a structured vocabulary for analyzing failure modes in target-directed RL. The distinction for G.2 (temporarily unreachable targets) is particularly notable—Section 3.1.2 explicitly notes "G.2 targets are often overlooked in literature, since hallucinations are mostly discussed in contexts without temporal progression," identifying a real and underappreciated failure mode.

- **Strategies explicitly grounded in the taxonomy**: "Generate" directly addresses E.1 by exposing estimators to generator-proposed candidates (Section 4.1.1); "pertask" directly addresses E.2 by providing cross-episode source-target pairs (Section 4.1.2). Each strategy's motivation clearly maps to a specific delusion type, unlike prior mixture strategies that combined relabeling for sample efficiency alone.

- **Multi-faceted evaluation**: The paper evaluates estimation errors broken down by delusion type (Figures 3b,d,f), delusional behavior frequencies (Figures 3c,g), and OOD success rates (Figure 3h), enabling more fine-grained attribution than standard success-rate-only evaluations. The paper also shows that individual strategies have specific tradeoffs (e.g., "pertask" improves E.2 but hurts short-distance estimation in Fig 3d), motivating hybrids rather than naive diversity.

- **Principled 2-slotted design insight**: Section 4.3 observes that generators benefit from avoiding problematic targets while estimators need exposure to them, and proposes separate relabeling processes—a design insight not present in prior HER work.

- **SSM environment enabling precise diagnosis**: The SSM gridworld's 4 equivalence classes from sword/shield possession create a controlled setting where G.2 delusions arise naturally and can be precisely attributed (Section 2).

## Weaknesses

### Fatal
None.

### Major

- **Confound between delusion reduction and general estimator quality improvement**: The proposed strategies ("generate" and "pertask") increase training data diversity for the estimator. The hybrid strategies combine multiple data sources, which improves general estimator quality (visible in Fig 3d where hybrids also improve non-delusional short-distance estimation). The paper attributes OOD gains primarily to delusion reduction (Section 5.5: "less frequent G.2 generation and lower E.2 errors… lead to less frequent delusional behaviors… which in turn improves the OOD performance"), but this causal chain is not isolated. Ablations that increase diversity without specifically targeting delusional source-target pairs (e.g., a random relabeling strategy that adds diversity but not delusional pairs) would be needed to disentangle these mechanisms. Without this, the central claim that delusion reduction is the causal driver—rather than improved general estimator calibration—remains plausible but unsubstantiated.

- **Limited visibility of empirical scope for the breadth of claimed generality**: The paper frames its contributions as broadly applicable to "target-directed agents" generally (Section 4: "applicable to general target-directed agents coming from various training procedures"; Section 1: strategies "should be expected to be applicable generally"). However, the full main text presents results for only one method (Skipper) on one environment (SSM), with 3 of 4 experimental sets in the appendix. The strategies themselves are HER relabeling variants that are inherently tied to a specific training paradigm. While the *ideas* behind "generate" and "pertask" may generalize, the empirical evidence supporting this claim is confined to HER-based methods on custom gridworlds, and a reader cannot evaluate the other 3 experimental sets from the main text alone.

### Minor

- **Informal delusion framework lacking formal precision**: The taxonomy is presented as a "framework" (Section 3), but the categories (G.1, G.2, E.0–E.2) are informal descriptors—G.1 is "nonexistent targets," G.2 is "temporarily unreachable targets," and the E-types classify estimation errors by which target type they misjudge. The "necessary conditions" (estimator, update rules, training data) are asserted without formal argument. The paper claims G.1 and G.2 are "disjoint" (Section 3.1), but under partial observability (acknowledged in Section 2), a target could be classified as both nonexistent and temporarily unreachable depending on the agent's belief state. The taxonomy provides useful vocabulary and structure, but the psychiatric framing adds nomenclature more than analytical power.

- **Overstated usage of "autonomous"**: The abstract and conclusion claim agents can "address delusions autonomously and preemptively" (lines 15, 280). In practice, the strategies require deliberate practitioner choices (relabeling selection, mixing proportions, 2-slot design). "Autonomous" is misleading if interpreted as the agent itself discovering and correcting delusions; it more accurately describes that the trained agent can reject problematic targets at decision-time once the practitioner has configured proper training data.

- **Vague empirical guidelines**: Section 7 offers guidelines like "analyze the state structure" and "inspect candidate targets if possible," which, while reasonable, lack the specificity needed for practitioners to diagnose delusion types in new environments without the ground-truth access available in SSM.

### Trivial
None.

## Nice-to-Haves

- Results on an established RL benchmark (e.g., FetchReach, AntMaze with goal-conditioned setups) to strengthen the generality claim beyond SSM.
- Side-by-side trajectory visualizations showing delusional vs. corrected behavior under the proposed strategies.
- Ablation isolating delusion-specific error reduction from general estimator improvement.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Insufficient experiments because appendix is not visible**: The harsh critic's claim that "3 of 4 experimental sets are relegated to the appendix" is a structural critique of the main text layout. The paper explicitly states results exist across 2 environments × 2 methods, and that "All 4 sets of experiments align in terms of conclusions" (line 255). The appendix experiments exist in the original submission—this is a parser limitation, not an author error. Removed.

- **Missing related works**: The harsh critic requests comparison with standard benchmarks like FetchReach/AntMaze and extension beyond HER to diffusion-based goal generators. These are reasonable suggestions but are scope extensions, not flaws in what the paper does present. Moved to Nice-to-Haves.

- **Statistical significance testing concern**: The paper uses 20 seed runs with 95% CIs and aggregated evaluation over 80 tasks. While the CIs in Fig 3h may overlap, the relative ordering is consistent across the other metrics (estimation errors, behavior frequencies). Formal significance tests are not the norm in this subfield for large-scale RL evaluations. Moved to minor consideration, though not emphasized.

- **"Fairer comparison" criticism regarding generator always using "future"**: The harsh critic argued this design choice "may understate the severity of delusions under other generator strategies." However, the paper justifies this (line 237) as creating a fairer comparison for the estimator strategies, and the paper does present generator-level comparisons separately (Figs 3a,e). Using "future" for the generator is a conservative choice that actually makes it harder to show delusion-related improvements—a deliberate and reasonable design decision. Removed as unfair comparison criticism that would favor the authors' method.

- **Computation cost of "pertask"**: The harsh critic noted "pertask" samples "across the entire memory" and may be computationally expensive. The paper does acknowledge this: "'pertask'… biases the training data distribution, making the agent spread out its efforts into learning the source-target pairs potentially far away" (line 152) and Table 1 notes "low efficiency in learning close-proximity source-target relationships." The paper also acknowledges computational cost of "generate" explicitly (line 140, 142). Removed as a concern already partially addressed.

- **Psychiatric terminology critique**: While the informal nature of the framework is a valid minor concern, the specific complaint that "psychiatric framing adds vocabulary without adding analytical power" is partially an aesthetic judgment. The delusion vs. hallucination distinction (belief evaluation vs. belief formation) maps cleanly onto the estimator-generator decomposition, which is genuinely clarifying. Kept a softened version as a minor concern.

- **Formatting/typos**: Removed per instructions.

## Novel Insights

The paper's most insightful contribution is the observation that generators and estimators have *conflicting* training data requirements—the generator benefits from avoiding problematic targets (to learn good proposals) while the estimator needs exposure to them (to learn to reject them). This motivates the 2-slotted hybrid approach (Section 4.3), which goes beyond simply mixing relabeling strategies and instead recognizes that different architectural components may need different data distributions. This is a genuinely structural insight about target-directed agent design. Separately, the G.2 category (temporarily unreachable targets) identifies a failure mode that standard goal-conditioned RL discussions of "hallucination" overlook, since hallucination framing typically assumes static reachability.

## Suggestions

- Add an ablation that increases training diversity without specifically targeting G.1/G.2 source-target pairs (e.g., random cross-episode relabeling), to isolate whether OOD gains stem from delusion reduction specifically or from general estimator improvement.
- Either soften the "autonomous" claim to clarify it means "the trained agent can reject targets at decision-time without further intervention" (not "the agent discovers and corrects its own delusions during training"), or provide evidence that the benefits extend beyond HER to at least one non-HER method.
- Consider making the G.1/G.2 disjointness claim conditional on full observability, since the paper acknowledges partial observability exists.

<context>
**Original reviewer signal:** Harsh Critic assessed the paper as workshop-level, citing thin empirical evidence, a confound between delusion reduction and estimator quality, and an informal framework. Strength Finder highlighted the systematic taxonomy, grounded strategies, 2-slotted hybrid insight, and multi-faceted evaluation with OOD validation.

**What was dropped and why:** (1) The "only 1/4 experiments visible" complaint—the 3 missing experimental sets exist in the appendix but were stripped by the parser; the paper explicitly claims consistent results across all 4 sets. (2) The "fairer comparison" criticism about using "future" for generators—this is a conservative design choice that makes it harder, not easier, to show delusion improvements, so it's not an unfair advantage. (3) The "pertask computational cost" concern—the paper already discusses this tradeoff. (4) Missing related works/benchmarks like FetchReach—these are scope extensions, not flaws. (5) Formal significance testing—20 seed runs with 95% CIs and 80 evaluation tasks is standard for the subfield.

**Cross-checks performed:** Verified the paper explicitly acknowledges the 2-env × 2-method scope (lines 191, 253-255); verified the "autonomous" claim language (lines 15, 25, 280); verified the G.1/G.2 disjointness claim and absence of formal argument (line 78); verified the confound claim by checking Fig 3d showing hybrids improve non-delusional estimation too (line 239-243); verified the paper discusses pertask tradeoffs (line 152); verified paper acknowledges generator "future" is a deliberate choice (line 237).

**Severity read:** The two major weaknesses—(1) the confound between delusion reduction and general estimator quality, and (2) limited empirical scope for the breadth of generality claims—are substantive but not fatal. The confound does not invalidate the empirical results; it undermines the specific causal attribution of OOD gains to delusion reduction. The empirical scope limitation is real but mitigated by the appendix results. No single weakness threatens the core contribution, which is the taxonomy and the practical strategies it motivates.

**Anything else load-bearing:** The paper's core contribution is framing + practical strategies, backed by controlled experiments that clearly demonstrate the strategies work. The contribution is more conceptual (the delusion taxonomy and its implications for training data design) than empirical (scalable benchmarks). Evaluators should weight the insight quality of the taxonomy against its informality and the moderate empirical scope.
</context>