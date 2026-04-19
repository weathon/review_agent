## Summary
This paper introduces a taxonomy of "delusions" in target-directed RL agents (categorizing generator issues G.1/G.2 and estimator issues E.0/E.1/E.2), proposes hybrid hindsight relabeling strategies ("generate", "pertask", and mixtures), and validates them empirically on the Skipper agent in a custom SSM environment. The hybrid strategies demonstrate reduced estimation errors and improved OOD generalization compared to exclusive baselines.

## Strengths
- **Clear, actionable taxonomy**: The G.1/G.2 (generator) and E.0/E.1/E.2 (estimator) categorization (Section 3) provides a structured framework for diagnosing failure modes in target-directed agents, particularly distinguishing semantically invalid targets from temporarily unreachable ones—a distinction often conflated in prior work.
- **Environment designed for ground-truth analysis**: The SSM environment (Section 2) enables exact reachability computations via its fully observable grid structure and irreversible item acquisitions, allowing precise measurement of E.1/E.2 errors that standard benchmarks cannot provide.
- **Systematic empirical validation of hybrid strategies**: Figure 3 demonstrates that hybrid relabeling mixtures (e.g., "F-(E+P+G)") simultaneously reduce E.2 estimation errors (Fig 3f) and achieve superior OOD success rates (Fig 3h) compared to exclusive baselines, with 20-seed runs providing statistical grounding.
- **Practical guidelines for practitioners**: Section 7 offers a concrete 4-step checklist for identifying E.1/E.2 risks and selecting appropriate relabeling strategies, making the theoretical contributions directly applicable.

## Weaknesses

### Fatal
None

### Major
- **Narrow empirical base limits generalizability claims**: The main text presents results only for Skipper on SSM (Section 5.5). While Section 5.6 claims "4 sets of experiments align" including LEAP and another environment, these are deferred to the appendix without concrete metrics in the main text. This is a significant evidential gap for a paper claiming "general guidelines for target-directed methods." Calibration anchors like iPWxqnt2ke (6 domains) and ruv3HdK6he (Game AI only, scored 5-6) show that narrow evaluation is a common weakness that reviewers weigh against acceptance.

- **Causal mechanism not cleanly isolated**: The paper claims hybrids "address delusions" specifically, but the experiments conflate (a) improved coverage of source-target distance distributions, (b) exposure to G.1/G.2 pairs, and (c) the specific relabeling mechanism. There is no ablation where "pertask" is applied only to non-problematic targets matched in distance distribution to "episode" to isolate whether temporal mismatch (cross-episode pairing) versus simply more long trajectories drives E.2 reduction. This weakens the core thesis that delusions are a distinct failure mode requiring targeted mitigation rather than just better coverage.

### Minor
- **Overclaiming in abstract/introduction**: The abstract claims "safety catastrophes" due to delusions, but no safety-critical scenarios are evaluated—only success rates in grid worlds. Similarly, the introduction frames delusions as a "neglected failure mode" qualitatively distinct from standard generalization/coverage issues, but the technical content (unreachable states, misestimated Q-values on out-of-support pairs) is expressible in standard RL language. This framing overstates novelty without demonstrating qualitatively new phenomena (e.g., persistence of wrong beliefs under corrective data).

- **Evaluation metrics tied to specific planner architecture**: The "delusional behavior frequencies" (Section 5.2) depend on Skipper's graph-based candidate selection and value-iteration planning. How candidate sets are constructed affects measured frequencies, and the paper provides no guidance for approximating G.1/G.2 detection in continuous or high-dimensional domains where exact reachability is unavailable. This limits the step from "we fix Skipper on SSM" to "general guidelines for target-directed agents."

### Trivial
- **Confidence interval presentation**: Figure 3c,g use 50% CIs due to "chaotic overlap" rather than standard 95%, which is unusual and suggests high variance. Full 95% CI plots would strengthen statistical claims.
- **Appendix-dependent claims**: Several key claims (e.g., LEAP results, second environment, update rule details) are stated in the main text but only substantiated in the appendix, making the main argument incomplete without it.

## Nice-to-Haves
- Compare against simpler coverage fixes (e.g., prioritized replay by TD-error, longer MELs, curriculum-based initial state distributions) to test whether "generate"/"pertask" are minimal necessary interventions or just one effective option among many.
- Provide qualitative trajectory visualizations showing typical delusional plans before vs. after mitigation to ground statistical claims in interpretable behavior.
- Discuss computational overhead of "generate" (JIT generator calls) versus baselines, as this affects practical adoption.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Critic's claim about "not yet released" or "unverifiable" models/benchmarks**: The paper cites SSM, Skipper, LEAP, and various HER strategies. Per instructions, all cited entities are assumed to exist as of 2026-04-19. Criticisms questioning their existence or release status are removed.
- **Critic's claim about missing appendix/proofs**: The parser strips appendix sections from all submissions; criticisms about missing appendix content are removed as they reflect parser behavior, not author errors.
- **Critic's claim about "unfair comparison" favoring baselines**: The asymmetry in comparisons (e.g., testing hybrids against exclusive baselines) is intentional to prove a stronger point; this is not a valid weakness.
- **Generic reproducibility nitpicks**: Undisclosed hyperparameters or implementation details that are trivial are removed per hard rules.
- **Critic's claim about related work gaps**: Per instructions, missing related works cannot be verified and are removed.

## Novel Insights
The paper's taxonomy provides genuine value by explicitly distinguishing G.2 (temporarily unreachable due to irreversible dynamics) from G.1 (semantically invalid), and linking E.2 errors specifically to trajectory-level relabeling biases. This framing clarifies why "episode" relabeling—widely used for sample efficiency—systematically fails on tasks with irreversible state transitions, a failure mode that standard HER literature treats as generic generalization error. The hybrid 2-slotted approach (separate relabeling for generator vs. estimator) is a practical insight: generators benefit from avoiding problematic targets while estimators need exposure to them, a tension that single-distribution training cannot resolve.

## Suggestions
1. **Broaden empirical validation**: Include at least one standard HER benchmark (e.g., FetchReach, AntMaze) with a dual-component planner to demonstrate that hybrid strategies improve OOD performance beyond SSM-style irreversible grids. Even negative results would clarify scope boundaries.
2. **Add coverage-matched ablation**: Compare "pertask" against a baseline that samples long-distance targets from the same episode distribution (matched in distance spectrum) to isolate whether cross-episode pairing specifically—not just more long trajectories—drives E.2 reduction.
3. **Tone down safety claims**: Replace "safety catastrophes" with more precise language about OOD failure modes, or add an experiment demonstrating safety-relevant delusions (e.g., agents pursuing targets that violate constraints).
4. **Clarify extensibility**: Add a subsection discussing how G.1/G.2 detection might be approximated in continuous domains (e.g., via learned reachability classifiers or constraint predictors) to address the "toy planner" concern.

## Score and Decision
**Calibration anchors compared:**
- **4gaySj8kvX** (JaxGCRL, GCRL benchmark): Scores 6,8,8,8 (avg 7.5, Spotlight). Stronger empirical breadth (8 environments) and clearer contribution (codebase + benchmark), but similar empirical focus.
- **iPWxqnt2ke** (Policy Gradient Subspaces): Scores 6,6,6,8 (avg 6.5, Poster). Limited to 6 domains but accepted; empirical existence proof with solid analysis.
- **ruv3HdK6he** (ALIGN-GAP, Game AI only): Scores 6,6,6,5 (avg 5.75, Poster). Narrow environment scope (Game AI) noted as weakness but accepted.
- **ZdvI91pInB** (Logic-Informed IRL): Scores 6,5,6,6 (avg 5.75, Reject). Conceptual framework with limited evaluation; rejected due to missing reward discovery evaluation.
- **s9SVlWOcLt** (Proto Successor Measure): Scores 5,8,6,8 (avg 6.75, Reject). Only 2 environments despite claiming "a number"; rejected partly due to narrow evaluation and overclaiming.

**Reasoning**: This paper sits between iPWxqnt2ke (empirical existence proof, accepted) and s9SVlWOcLt (narrow eval + overclaim, rejected). The taxonomy is more novel than iPWxqnt2ke's gradient subspace observation, but the empirical base is narrower (1 main environment vs. 6 domains). The overclaiming ("safety catastrophes," "neglected failure mode") mirrors s9SVlWOcLt's weakness. Compared to ruv3HdK6he (Game AI only, accepted at 5.75), this paper has stronger analysis but similar scope limitations. The causal isolation weakness is more severe than anchors' typical "limited domains" complaints.

Positioning relative to anchors: Stronger conceptual contribution than iPWxqnt2ke but weaker empirically; similar scope issues to ruv3HdK6he but with more overclaiming. The 5.5-6 range is appropriate—borderline accept/poster, but the overclaiming and causal isolation gap push toward the lower end.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>