Now I have enough context to write the final review. Let me synthesize all the reviewer inputs, verify claims against the paper, and calibrate against similar papers.

Key calibration anchors:
- "Is Memorization Actually Necessary for Generalization?" (re-analysis/critique paper): Scores 3,3,3,6 and 3,3,6,5,5 → Reject. Similar single-case study format, similar critique-style, but had methodological weaknesses in its own re-analysis.
- "Unprocessing Seven Years of Algorithmic Fairness": Scores 8,6,8,6 → Accept (oral). This is a strong re-evaluation paper with novel methodology (unprocessing) and systematic evaluation across many methods/datasets.
- "The Effectiveness of Curvature-Based Rewiring and the Role of Hyperparameters in GNNs Revisited": Scores 6,5,6,6 → Accept (poster). Similar critique/re-evaluation with hyperparameter control, but more focused on a specific claim domain.
- "Joint Training Does Not Transfer" (pure rebuttal): Scores 1,1,1,5,5 → Withdrawn/Reject.
- "Re-evaluating Retrosynthesis": Scores 8,3,6,5 → Mixed, but interesting.

This paper is a detailed forensic critique of a specific high-visibility paper, plus an attempt to generalize lessons. It's somewhere between "Is Memorization Actually Necessary" (rejected) and "Unprocessing Seven Years" (accepted oral). The key question is whether the concrete forensic audit is strong enough and whether the methodological contribution (Best-of-N hyperparameter comparison) is novel enough.

The paper's actual strengths are very strong: it uncovers concrete, verifiable issues (omitted data, incorrect statistics, selective reporting, retracted claims). The Best-of-N analysis is a genuine methodological contribution. The blueprint claims are somewhat overreaching but the case study itself is thorough and important.

Compared to "Unprocessing Seven Years" which had a novel methodology AND broad evaluation, this paper is narrower (one target) and the methodology is less novel (Best-of-N is adapted from existing ideas). But the service of correcting the scientific record on a high-visibility ICLR Oral is significant.

Let me now write the review.</think>Now I have sufficient calibration context. Let me write the consolidated review.

## Summary

This paper provides a detailed forensic re-analysis of "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs" (Nguyen et al., 2024), an ICLR 2025 Oral paper. Through systematic re-examination of all four lines of evidence—human evaluations, NLP benchmarks, LLM-as-a-Judge evaluations, and community adoption claims—the authors demonstrate that min-p's claimed superiority collapses under proper statistical analysis and fair comparison. From this case study, they distill six general lessons for more rigorous empirical ML research, including a novel Best-of-N methodology for controlling hyperparameter search volume across methods.

## Strengths

1. **Concrete, verifiable findings that invalidate the original paper's central claims.** The paper identifies specific, consequential issues: one-third of human evaluation data was omitted without justification (§2.1); the original paper's pooled t-test contradicts its own "consistently outperforms" claim—proper per-condition tests with Bonferroni correction show significance in only 1 of 12 comparisons (Table 1); the higher of two scores was selectively reported for min-p while the lower was reported for top-p (§4.3); and community adoption claims of 54k GitHub repositories and 1.1M stars were retracted by the original authors themselves (§5). These are not interpretive disagreements—they are factual discrepancies documented with the original paper's own data.

2. **Rigorous statistical re-analysis.** The application of Bonferroni correction for 12 simultaneous comparisons and the use of an Intersection-Union Test (IUT) to assess the "consistently outperforms" claim are statistically well-motivated. The IUT is particularly apt: if the claim is that min-p is better *everywhere*, then one must reject the null in all conditions, and the largest p-value of 0.378 makes this clearly impossible. This goes beyond simply noting the original statistics were wrong—it provides the correct analysis.

3. **Substantial original experimentation via Best-of-N hyperparameter analysis.** The GSM8K CoT sweep across 9 models, 2 stages, 4 samplers, 31 temperatures, 6 hyperparameters per sampler, and 3 seeds (~6000 A100-hours) is a genuine empirical contribution. The methodology of subsampling equal numbers of hyperparameters per sampler and comparing maximum achieved performance directly addresses the widespread problem of unfair hyperparameter comparisons—a methodological contribution that extends beyond this specific case study.

4. **An important service to the scientific community.** Demonstrating that an ICLR 2025 Oral paper's claims do not hold up under proper scrutiny—and that the paper's own data, properly analyzed, contradicts its conclusions—is valuable for correcting the scientific record. The finding that 3 of 4 original reviewers and the Area Chair cited inflated community adoption numbers as a main justification for their endorsement (§5) is a striking meta-scientific observation.

## Weaknesses

### Major:

1. **The "blueprint" framing overreaches beyond what the evidence supports.** The paper's title and abstract claim to provide "a blueprint for more rigorous science in empirical machine learning research," but the actual evidence base is a single case study. Five of the six "general lessons" (multiple comparisons correction, data transparency, scrutinizing qualitative summaries, methodological clarity, and consistent reporting) are well-established principles in statistics and empirical methodology, not novel insights. Only the Best-of-N hyperparameter analysis represents a genuinely new methodological contribution. The paper would be stronger if it scoped its claims more precisely—for example, "lessons from a detailed case study"—rather than presenting well-known best practices as a novel "blueprint." This does not undermine the forensic audit's value but tempers the paper's broader aspiration.

2. **Benchmark re-analysis is limited to GSM8K CoT, leaving the quality-diversity tradeoff claim under-tested.** The original paper's core claim was about creative and coherent outputs—a quality-diversity tradeoff—yet the re-analysis uses only mathematical reasoning benchmarks (GSM8K CoT) where diversity is largely irrelevant. The paper acknowledges that 2 of 12 models showed min-p advantages under corrected prompt formatting (Appendix C) but does not analyze whether this pattern is meaningful. Without testing on a creative generation or diversity-focused benchmark, the claim that min-p "offers no apparent advantage over previously existing samplers" extends beyond what the evidence directly shows for its intended use case.

3. **Some of the paper's own analyses do not fully meet the methodological standards it advocates.** While the paper rightly criticizes Nguyen et al. for omitted details, some of its own re-analyses are incompletely specified:
   - §2.2–2.3: The human evaluation re-analysis does not report effect sizes or confidence intervals on differences, nor does it state which specific condition produced the single significant result after Bonferroni correction. No power analysis is provided to interpret null results.
   - §4.2: The finding that min-p received ~2× and ~10× more hyperparameter tuning is stated qualitatively without exact counts or a table of LLM-as-a-Judge results with confidence intervals.
   - The qualitative response annotation in §2.3 is described without inter-annotator agreement, annotation criteria, or whether multiple annotators were used.

   These gaps do not invalidate the conclusions, but they create an asymmetry: the paper holds the original work to high standards (correctly) while not fully meeting those same standards in its own reporting. For a paper whose central thesis is about rigor, this is a meaningful shortcoming.

### Minor:

1. **Unclear epistemic status of extra-paper evidence.** Several key findings depend on communications outside the original paper: public confirmations with authors about omitted data (§2.1), Telegram logs showing selective reporting (§4.3), and retracted GitHub claims (§5). While these are important and damning, their archival permanence is uncertain. The paper mitigates this by posting annotations publicly, but integrating more of this evidence directly into appendices would better satisfy its own transparency standards.

2. **The new human evaluation re-analysis (§2.4) is less thorough than the original data re-analysis.** The discovery of the apparently incorrect value (7.80 vs. 5.80) is important, but the paper does not apply the same rigorous statistical testing (Bonferroni, IUT, effect sizes) to the new evaluation data. The conclusion that "min-p offers no apparent advantage" rests primarily on a figure rather than a clearly walked-through statistical analysis.

3. **Limited engagement with potential counterarguments.** The paper does not seriously consider whether the Best-of-N analysis could systematically disadvantage a sampler whose good configurations are rarer but much better (heavy-tailed performance distribution), or whether some of the original authors' corrections might reflect standard (if still problematic) community practices rather than egregious error.

### Trivial:

- The paper's limitation statement (§6) is admirably honest: "Conclusions here are based on that evidence. We emphasize that new evidence might lead to different conclusions." This is appropriate and should be retained.

## Nice-to-Haves

- Extend the benchmark analysis to at least one creative/diversity-focused task or dataset to directly test min-p's intended use case.
- Add a second brief case study (even on a different paper) to support the "blueprint" generality claim.
- Report effect sizes and CIs alongside the hypothesis tests in the human evaluation re-analysis.
- Provide a practical reviewer checklist derived from the six lessons, transforming abstract principles into concrete, actionable guidance.
- Discuss potential limitations of the Best-of-N subsampling methodology (e.g., sensitivity to hyperparameter grid choice, implications of heavy-tailed performance distributions).

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"The paper relies on non-public communications and evolving artifacts" (Harsh Critic point 3):** While the concern about archival permanence is minor and worth noting, the harsh critic overstates this as a structural problem. The paper cites public posts and communications, and the key findings (omitted data, selective reporting) are verifiable from the original paper's own released data and tables. The retracted claims are a matter of public record in the ICLR Camera Ready. This is not a methodological gap—it is standard practice in scientific critique.

- **"The paper conflates methodological errors with intentional misconduct" (Human Finder/Sparke point):** The paper is actually quite measured in its language. It says "our comprehensive re-examination demonstrates that its conclusions are invalidated by its own data" and discusses "methodological ambiguity" and "inconsistent reporting"—it does not allege fabrication or fraud. The harsh critic's point about not distinguishing inadvertent errors from bias is partially valid but the paper's language is already relatively careful.

- **"The blueprint lessons are largely well-known best practices" (Neutral Reviewer point 2):** This is partially valid and reflected in Major Weakness 1 above. However, the characterization is too dismissive—demonstrating that well-known best practices were violated in a high-profile ICLR Oral, and that their violation materially changed the paper's conclusions, is itself a contribution. The point is kept but repositioned as overclaiming in the "blueprint" framing rather than the lessons being valueless.

- **"Missing comparison on creative/diversity benchmarks" (Spark point 1):** This is valid but overstated as a missing *experiment*. The paper's re-analysis of the original paper's own diversity evaluations (human evals and AlpacaEval) does address quality-diversity, even if the new GSM8K sweeps do not. I keep this as Major Weakness 2 but reframe it—the paper does address diversity through the original evidence, just not through its own new experiments.

- **"Insufficient distinction between inadvertent errors and intentional bias" (Neutral Reviewer point 6):** The paper's language is already fairly measured. It says "appeared to have reported results inconsistently, favoring min-p" and "potentially selective reporting"—not alleging fraud. This concern is overblown relative to the actual paper text.

## Novel Insights

The paper makes a striking meta-scientific observation that goes beyond its own stated conclusions: 3 of 4 ICLR reviewers and the Area Chair explicitly cited unsubstantiated community adoption numbers (54k repos, 1.1M stars) as a main justification for strong endorsement—numbers that were subsequently retracted. This reveals a structural vulnerability in peer review where claims of practical adoption can override assessment of methodological soundness, even at top venues. The Best-of-N methodology for controlling hyperparameter search volume is the paper's most transferable contribution and addresses a widespread but under-discussed problem in ML evaluation.

## Suggestions

1. **Rescope the framing** from "blueprint" to "lessons from a detailed case study" to better match the evidence base. The forensic audit is genuinely valuable; the overclaiming on generality dilutes the paper's credibility.

2. **Run at least one creative-generation benchmark** (even AlpacaEval with direct head-to-head comparisons) through the Best-of-N analysis framework to test the quality-diversity claim where it matters most.

3. **Report effect sizes** (e.g., Cohen's d) and confidence intervals alongside p-values in the human evaluation re-analysis to fully practice the rigor the paper advocates.

4. **Provide exact counts and a summary table** for the LLM-as-a-Judge analysis (number of configurations per sampler, win rates with CIs), since the current text relies on qualitative figure descriptions.

5. **Briefly discuss limitations of the Best-of-N methodology** (assumptions about uniform hyperparameter grids, heavy-tailed performance distributions) to strengthen the methodological contribution.

## Evaluation

**Originality:** Moderate. The forensic re-analysis approach is thorough but follows well-established critique methodology. The Best-of-N hyperparameter comparison is genuinely novel and transferable. The "blueprint" lessons themselves are mostly not new, though their demonstration through a high-visibility case study is.

**Importance of research question:** High. Correcting the scientific record on a prominent paper and exposing methodological failures in ML evaluation practices addresses a genuine and growing concern in the community.

**Whether claims are well supported:** Mixed. The forensic audit findings are extremely well-supported with verifiable evidence. The broader "blueprint" claim is less well-supported, as it extrapolates from a single case study to general principles. The benchmark re-analysis is limited to one task type.

**Soundness of experiments:** The human evaluation re-analysis is sound in its statistical approach but incomplete (no effect sizes). The GSM8K sweep is extensive and well-designed but constrained to mathematical reasoning. The LLM-as-a-Judge re-analysis relies partly on ongoing/external work and qualitative descriptions rather than the same statistical rigor applied elsewhere.

**Clarity of writing:** Good. The paper is well-structured, clearly argued, and transparent about its goals and limitations.

**Value to research community:** High for the forensic audit and the Best-of-N methodology; moderate for the general lessons, which largely restate known best practices.

## Score and Decision

**Calibration reasoning:**

- "Is Memorization Actually Necessary for Generalization?" (single-case re-analysis, methodological critique): Scores 3,3,3,6 and 3,3,6,5,5 → Reject. This paper had similar scope (re-analyzing one target paper) but weaker methodological contribution and its own re-analysis errors.
- "Unprocessing Seven Years of Algorithmic Fairness": Scores 8,6,8,6 → Accept (oral). Broader evaluation (1000s of models, multiple datasets), novel methodology (unprocessing), systematic comparison across an entire subfield. This paper is not at that level of breadth or methodological novelty.
- "The Effectiveness of Curvature-Based Rewiring..." (GNNs hyperparameter critique): Scores 6,5,6,6 → Accept (poster). Similar in spirit—showing a prominent method's advantages disappear with fair hyperparameter tuning. More focused but also more modest in its claims. This paper is comparable.
- "Joint Training Does Not Transfer" (pure rebuttal): Scores 1,1,1,5,5 → Withdrawn/Reject. Much weaker—pure negative rebuttal with no positive methodology.

This paper is above the "Memorization Necessary" level because its forensic findings are concrete and verifiable (not interpretive), it adds substantial new experimentation (6000 A100-hours of sweeps), and its Best-of-N methodology is a genuine positive contribution. It is below the "Unprocessing" level because it derives from a single case study and its methodological novelty is more limited—a clever application of subsampling rather than a new mathematical framework.

The paper's core contribution—the forensic audit—is strong and important. Its secondary contribution—the "blueprint"—is somewhat overclaimed but contains one genuinely novel methodological idea. The main limitation is the scope mismatch between the single-case evidence and the generalizing claims, plus the benchmark re-analysis not covering the original paper's intended use case (creative/diverse generation).

MY FINAL SCORE: 6
MY FINAL DECISION: Accept