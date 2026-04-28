## Summary
This paper introduces a diagnostic framework for "delusions" in target-directed reinforcement learning, distinguishing between generator hallucinations (proposing bad targets) and estimator false beliefs (valuing bad targets). The authors propose two new hindsight relabeling strategies ("pertask" and "generate") and demonstrate their effectiveness in hybrid combinations using a controlled environment (SSM) designed to make delusional behaviors ground-truth verifiable.

## Strengths
- **Controlled environment design (SSM) enabling ground-truth delusion measurement**: The SwordShieldMonster environment segments states into equivalence classes based on item possession, creating verifiable cases of temporary unreachability (G.2 delusions). This allows direct measurement of estimator errors on unreachable states—something impossible in standard black-box benchmarks (Section 2, lines 58-59).
- **Clear decomposition of failure modes**: The explicit separation of Generator errors (G.1/G.2) from Estimator errors (E.0/E.1/E.2) provides a useful diagnostic framework. Figure 2 visualizes specific E.1 and E.2 behaviors, making the abstract concept concrete.
- **Hybrid 2-slotted training architecture**: The proposal to use different data distributions for generators and estimators (Section 4.3, lines 178-186) is architecturally sound. Generators need to learn feasibility while estimators need to learn infeasibility—conflating these needs in a single buffer is correctly identified as a bottleneck.
- **Empirical validation showing reduced estimation errors**: Figure 3(f) demonstrates that hybrid strategies like "F-(E+P)" significantly lower E.2 estimation errors compared to baseline "F-E", and Figure 3(h) shows corresponding OOD performance improvements.

## Weaknesses

### Fatal
None

### Major
- **Missing comparison to standard HER 'random' strategy undermines novelty claims for "pertask"**: The paper claims "pertask" (sampling goals from the entire memory/buffer) as a novel contribution, but this closely resembles the standard HER 'random' strategy (sampling goals uniformly from the episode/buffer), which is not included as a baseline. Table 1 (line 169) describes "pertask" as augmenting training data with "targets that were experienced," but the paper does not clarify how this differs functionally from existing buffer-wide goal sampling. Without this comparison, it is unclear whether the observed gains come from the proposed strategy or from well-known distributional correction techniques. This directly affects the novelty assessment.

### Minor
- **Heavy reliance on appendix for experimental evidence**: The paper explicitly states (line 191-192) that "3 out of 4 sets of experiments are presented in the Appendix." The main text focuses almost exclusively on "Skipper on SSM" (Set 1/4), while generalization claims across methods (Skipper vs. LEAP) and environments rest on appendix material. While common in conference submissions, this limits immediate verifiability of the headline contribution that strategies "generalize across methods and environments."
- **Compute overhead of "generate" strategy not quantified**: The paper acknowledges (line 142-143) that "generate" incurs "additional computational burden" but provides no wall-clock time, FLOP counts, or training efficiency comparisons. In RL, a 20-50% increase in training time for marginal OOD gains is a critical trade-off that affects practical adoption.

### Trivial
- **Mixing proportions appear heuristic without ablation**: The specific ratios (e.g., 50% episode, 25% pertask, 25% generate in F-(E+P+G), line 233) are not justified theoretically or empirically in the main text. A sensitivity analysis would strengthen the design choices.

## Nice-to-Haves
- Visualization of value landscape heatmaps comparing baseline vs. hybrid estimators would make the "delusion correction" concept more intuitive.
- Discussion of whether these delusions persist in continuous control tasks where "unreachable" is a matter of dynamics constraints rather than discrete state segmentation.
- Automatic ratio tuning based on detected delusion rates rather than fixed hyperparameters.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Criticism about "pertask" not being novel relative to negative sampling/contrastive goal-conditioning**: While the concern about missing HER-'random' baseline is valid (kept as Major), broader claims about "contrastive goal-conditioning" or "negative sampling in UVFAs" are outside the paper's stated scope (HER-based target-directed agents). The paper explicitly focuses on hindsight relabeling strategies, not contrastive methods.
- **Criticism about psychiatric "delusion" analogy conflating standard value estimation errors**: The paper explicitly defines delusions as "obviously wrong beliefs... inability to reject false beliefs" (line 23) and maps this to convergence failure on out-of-support states. This is a terminological framing choice, not a mathematical error.
- **Criticism about G.1/G.2 distinction blurring in continuous control**: The paper explicitly scopes to discrete environments (SSM) where this distinction is clear (line 58-59). Criticizing absence of continuous control validation is scope creep.
- **Criticism about Figure 3 error bars inconsistency**: The caption (line 249) explicitly notes "95% for all subfigures except c) & g), which used 50% due to the chaotic overlap." This is documented, not hidden.
- **Criticism about theoretical justification for convergence**: The paper is an empirical systems contribution (Section 5), not a theoretical paper. Demanding convergence proofs is outside community norms for this contribution type.
- **Strength about "Ground-Truth Based Evaluation Metrics"**: This is partially redundant with the SSM environment strength and overstates the novelty—measuring estimation error against ground truth is standard in controlled environments.

## Novel Insights
The paper's core insight—that HER's positive-only bias (relabeling only with achieved/future goals) systematically blinds estimators to unreachable targets—is genuinely under-discussed in the GCRL literature. The SSM environment design, which creates ground-truth-verifiable cases of temporary unreachability through item-possession state segmentation, is a useful contribution for the community. However, the proposed "pertask" strategy's relationship to existing buffer-wide goal sampling remains unclear, which limits the novelty contribution.

## Suggestions
1. Add HER-'random' as a baseline to clarify whether "pertask" offers benefits over standard uniform goal sampling from the buffer.
2. Include a compute overhead table comparing wall-clock training time across all strategies.
3. Add a mixing ratio ablation (e.g., varying pertask from 0% to 50%) to justify the chosen proportions.
4. Consider moving one additional experiment set (e.g., LEAP results) into the main text to strengthen generalization claims.

## Score and Decision

**Calibration anchors retrieved:**
- `/home/wg25r/review_agent/human_reviews_2026/mwgYORsqtv.md` (avg 6.00, Accept): Empirical GCRL study with controlled experiments; reviewer requested HER-style baselines but paper still accepted.
- `/home/wg25r/review_agent/human_reviews_2026/VoKut0M4bI.md` (avg 5.00, Reject): Diagnostic/auditing framework for RL with empirical validation; appendix-deferred details noted.
- `/home/wg25r/review_agent/human_reviews_2026/7N0ugLE17t.md` (avg 5.00, Accept): Continuous-time MARL with missing discrete-time baselines criticized but still accepted.
- `/home/wg25r/review_agent/human_reviews_2026/ZgCCDwcGwn.md` (avg 7.00, Accept Oral): Framework paper with comprehensive empirical evaluation across 27 tasks; missing ablations noted but strong results carried it.
- `/home/wg25r/review_agent/human_reviews_2026/lIIJDDxBrg.md` (avg 3.50, Reject): Missing critical baselines and incomplete evaluation (2 of 5 tasks) led to rejection.
- `/home/wg25r/review_agent/human_reviews_2026/kzRWbQgady.md` (avg 3.33, Reject): Diagnostic framework for RL benchmarks; concerns about novelty and statistical rigor led to rejection.

**Comparison:** This paper sits between the 5.0-6.0 range anchors. It has stronger controlled environment design than VoKut0M4bI.md (5.00) and clearer empirical validation than kzRWbQgady.md (3.33), but weaker baseline coverage than mwgYORsqtv.md (6.00). The missing HER-'random' baseline is similar to the missing discrete-time baselines in 7N0ugLE17t.md (5.00, Accept), but the appendix-heavy experimental presentation is a additional weakness. The paper's empirical contributions are real but the novelty claim for "pertask" is undermined by the missing baseline. Relative to anchors, this warrants a **5.5**.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>