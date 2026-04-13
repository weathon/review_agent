=== CALIBRATION EXAMPLE 21 ===

# Final Consolidated Review
## Summary
This paper studies whether a model-free DRC agent trained on Sokoban performs internal decision-time planning, using a concept-based interpretability pipeline. The main evidence is threefold: linear probes decode two future-directed square-level concepts from hidden state, qualitative analyses suggest these decoded representations are iteratively formed and revised in a search-like manner, and activation interventions along probe directions can steer behavior toward alternative routes on specially constructed levels. The paper is novel and thought-provoking, but its strongest claims are stated more definitively than the evidence fully supports.

## Strengths
- **The paper contributes a concrete interpretability workflow aimed at planning, not just representation discovery.** Section 3.1 clearly lays out a three-step procedure—probe for planning-relevant concepts, inspect plan formation, then test behavioral dependence via interventions. That integrated methodology is a substantive contribution beyond isolated probe plots.
- **The chosen concepts are highly interpretable and operationalized at the right granularity for Sokoban.** The square-level concepts \(C_A\) (future agent approach direction) and \(C_B\) (future box push direction) produce board-level decoded structures that are easy to inspect and often correspond to coherent routes linking boxes and targets (Figures 1 and 5).
- **The probe results show genuinely nontrivial future-directed information in hidden state beyond the immediate observation.** In Figure 4, hidden-state probes substantially outperform observation-only baselines for both \(C_A\) and \(C_B\), supporting the claim that the recurrent state contains structured information about future interaction patterns rather than merely reflecting the current board.
- **The intervention experiments are the paper’s strongest evidence.** In Section 6.1, adding learned concept directions to hidden states reliably redirects the agent toward suboptimal-but-valid alternative routes, with large gaps over random-direction baselines (especially for Agent-Shortcut). This goes beyond correlational probing and demonstrates meaningful behavioral influence of the identified directions.
- **The paper links representational emergence to a behavioral planning-like signature.** Figure 9 shows that stronger decodability of \(C_A\)/\(C_B\) across training correlates with greater ability to exploit extra test-time compute, tying the interpretability findings to a distinctive behavioral phenomenon rather than presenting them in isolation.
- **The claim that internal computation is iterative is supported by multiple pieces of evidence.** Section 5 and Figure 6 show that decoded future-directed structure improves over internal ticks and additional forced “thinking steps,” which is a more compelling story than a single static probe snapshot.

## Weaknesses
###: Fatal
- None.

### Major:
- **The central claim is overstated relative to what is actually shown.** The paper claims “the first mechanistic evidence that model-free reinforcement learning agents can learn to plan” and says it shows the agent “is indeed internally planning.” The evidence is clearly suggestive and interesting, but it does not cleanly rule out a narrower interpretation in which the recurrent policy encodes and iteratively refines future trajectory predictions or intentions without fully demonstrating online plan *evaluation* in the stronger sense defined in Section 2.1. This matters because the paper’s contribution is judged primarily by this headline claim.
- **The key probed concepts are defined from the agent’s own realized future behavior, which limits what can be inferred from probe success.** Section 3.2 explicitly defines \(C_A\) and \(C_B\) retroactively from “the next time the agent moves onto” or “pushes a box off” each square, or `NEVER` if it never does so. As a result, strong decoding primarily establishes that hidden state contains information predictive of the agent’s eventual trajectory. That is interesting, but by itself it does not distinguish planning from commitment to a future course of action, predictive state tracking, or a sophisticated policy representation. This is a substantive methodological limitation, not a nitpick.
- **Evidence for the “evaluation” component of planning is mostly qualitative and weaker than the evidence for future-directed representation.** The paper’s own characterization of planning requires that the agent “evaluate plans by predicting their consequences,” but Section 5 supports this mainly with hand-selected qualitative examples and cautious language such as “appears to” and “suggestive of.” Figure 6 shows increasing decodability over ticks, but the jump from better future-label decodability to plan evaluation/search is not fully established quantitatively.
- **The intervention results demonstrate steerability, but the causal claim should be narrower.** Section 6.1 shows that adding probe weight directions to hidden states can redirect behavior on handcrafted Agent-Shortcut and Box-Shortcut levels. This is strong evidence that these directions are behaviorally meaningful. However, it does not fully establish that the probed variables are the native computational substrate of planning in general Sokoban play, as opposed to high-leverage directions that can bias behavior in specially designed settings. The paper should be more careful in moving from “causally influences behavior” to “is the mechanism of planning.”
- **The “parallelized bidirectional search” interpretation is intriguing but under-validated.** The claim rests largely on qualitative visualizations in Figure 1 and appendix examples. The paper does not provide a quantitative characterization showing, across many episodes, that plans systematically extend forward from boxes and backward from targets, or that these fronts meet in a way diagnostic of bidirectional search. As written, this part reads as an appealing hypothesis rather than a demonstrated mechanistic conclusion.
- **Generalization of the main conclusion is limited by the empirical scope.** The core evidence in the main paper is for one DRC configuration in one environment, Sokoban. The paper is careful in many places to discuss “the agent we study,” but the framing at the abstract/introduction level is broader (“model-free RL agents can learn to plan”). Given the narrow case study, the broader wording should be tempered.

### Minor
- **The asymmetry between intervention types deserves more analysis.** In Table 1, Agent-Shortcut interventions are very successful across layers, while Box-Shortcut is notably weaker, especially at Layer 1 (56.2%). The paper acknowledges the result but does not analyze what it implies about the distinct causal roles or reliability of \(C_A\) versus \(C_B\).
- **Some interpretive statements are stronger than the direct evidence.** For example, Section 4 says the small gain from 1x1 to 3x3 probes “confirms” localized representations; this is better described as evidence consistent with locality, not definitive confirmation.
- **The decoded plans are not always complete or perfectly aligned with successful solutions.** The paper itself notes that plans “often contain flaws (like the lack of one necessary arrow in Figure 5c).” This does not invalidate the approach, but it weakens the degree to which single decoded visualizations can be taken as direct readouts of the underlying planning process.

### Trivial
- None.

## Nice-to-Haves
- Quantify the “evaluation” story more directly: e.g., how often decoded plans change from infeasible/suboptimal to feasible/better across ticks.
- Add disruption-style interventions, not only redirection: ablate or scramble \(C_A\)/\(C_B\)-aligned structure at specific ticks and test whether plan quality and success degrade.
- Test interventions on ordinary Boxoban levels, not only shortcut constructions, to assess whether the identified representations matter in naturalistic play.
- Include at least one stronger control agent in the main text, such as a variant that does not benefit from extra test-time compute, to better separate planning-related representations from generic future-trajectory encoding.
- Quantify reliability of decoded plans by reporting plan/behavior mismatch rates across episodes rather than relying mainly on positive visual case studies.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper is weakened by parser/formatting issues / duplicated figure captions.”** Removed as a formatting artifact of the extracted text, not a paper issue.
- **Criticisms implying the paper failed to release or justify cited tools/models/benchmarks.** Removed per instruction; all cited entities are assumed to exist and be available.
- **Generic praise such as “the paper is well-written” or “the topic is important.”** Removed because these are not paper-specific strengths.
- **Any complaint that the paper should have covered many more agents/environments as a requirement for validity.** Kept only in weakened form as a scope/generalization limitation, not as a fatal flaw, because the paper’s core contribution is a case study and methodology rather than a universal empirical survey.

## Novel Insights
The strongest synthesis across the reviews is that the paper’s real contribution is not a definitive proof of internal planning, but a well-designed bridge from behavioral signatures to mechanistic evidence. The results are most convincing when read as showing that the DRC agent learns structured, future-directed internal representations that are iteratively refined and behaviorally actionable, with a planning interpretation that is plausible and perhaps likely, but not uniquely established. In that framing, the work is still quite strong: it provides a concrete template for how to study emergent planning mechanistically, and its intervention results materially raise the standard above purely descriptive probing.

## Suggestions
- **Narrow the headline claim.** Reframe the contribution as strong mechanistic evidence for *planning-like internal computation* or *future-directed search-like computation* in a model-free agent, unless stronger quantitative tests are added for evaluation/search.
- **Directly address the circularity of \(C_A\)/\(C_B\).** Add at least one probe target that captures predicted consequences not defined by the agent’s realized trajectory, ideally something closer to environment dynamics or counterfactual structure.
- **Quantify search-like signatures.** Measure, over many episodes, whether decoded plans expand from boxes and targets in the proposed directions, and whether changes across ticks are consistent with revision/evaluation rather than mere stabilization.
- **Strengthen the causal section with broader controls.** Complement shortcut redirection with disruption experiments and, if feasible, tests on standard levels.
- **Analyze the failure modes.** Report when decoded plans mismatch behavior, and discuss what those mismatches imply about probe faithfulness and the limits of the interpretation.
- **Discuss alternative interpretations more explicitly.** The paper would be stronger if it squarely acknowledged that predictive-state and policy-commitment explanations remain viable competitors, then argued carefully why the full body of evidence favors planning.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 8.0]
Average score: 8.0
Binary outcome: Accept
