Now let me search for calibration papers.Now let me search for a few more calibration papers with similar themes (empirical negative results, ablation studies on LLMs for MT).Now I have all the information I need. Let me produce the consolidated review.

---

## Summary

This paper investigates the source of translation performance in *Machine Translation from One Book* (MTOB), specifically asking whether LLMs learn to translate Kalamang (and other low-resource languages) from grammatical explanations or from the parallel examples embedded in grammar books. Through a carefully designed ablation that manually filters grammar books into parallel (BOOK_para) and non-parallel explanatory (BOOK_non-para) subsets, the authors demonstrate that almost all MT gains stem from parallel examples. They further show that fine-tuning NLLB-1.3B on the parallel data matches the expensive long-context Gemini setup, and introduce a typological feature prompt that achieves leading results on linguistically-appropriate tasks (grammaticality judgment and IGT prediction).

---

## Strengths

- **Central ablation is exactly the right one and is cleanly executed.** Manually filtering BOOK_para vs. BOOK_non-para directly tests the headline question. Table 2 is unambiguous: BOOK_p matches or outperforms BOOK_all in most settings, while BOOK_¬p alone plummets by 7–8 CHRf++ points relative to BOOK_p. This is the paper's load-bearing result and it is clearly supported.

- **Statistical analysis strengthens the core claim.** The regression of CHRf++ on test-set type coverage (Figure 2) is significant in both directions (p < 0.005, F-test; significant Pearson correlations), and all prompt settings fall within the 95% confidence interval of the regression line. Crucially, total token count is *not* a significant predictor (p = 0.997), isolating vocabulary coverage as the operative variable rather than sheer prompt length.

- **NLLB comparison makes the conclusion practically actionable.** Table 4 shows fine-tuned NLLB-1.3B achieves CHRf++ competitive with Gemini given the same parallel data (34.2 vs. 26.6 eng→kgv; 28.6 vs. 33.1 kgv→eng), in a fraction of the time and cost. This is a genuine practical takeaway for the XLR MT community: standard MT fine-tuning with extracted parallel sentences suffices.

- **Use of CHRf++ over CHRf is both principled and important.** The paper correctly addresses a methodological weakness flagged in prior reviews of MTOB. Using CHRf++ incorporates word order and aligns with XLR MT evaluation norms.

- **Constructive positive contribution beyond the critique.** The paper does not stop at negation—it identifies where grammar *does* help (grammaticality judgment and IGT prediction) and introduces typological feature prompting, which consistently performs best on these linguistically-motivated tasks. This provides a nuanced, actionable message: task-appropriate data matters.

- **Generalization to Nepali and Guaraní.** Table 3 consistently shows BOOK_¬p and BOOK_all hurt performance relative to 0-SHOT for seen low-resource languages (up to −7 CHRf++), while BOOK_p has a neutral or small positive effect. This extends the negative finding beyond a single unseen language.

---

## Weaknesses

### Fatal
*None.*

### Major

- **The regression supporting the core claim is built on approximately 12–15 data points.** Figure 2 is based on the set of distinct prompt configurations tested with Gemini, which is a small sample for a regression model making significance claims. While the F-test is reported as p < 0.005, regression with this few observations is sensitive to leverage from individual points. Bootstrapped confidence intervals or permutation tests would substantially strengthen this analysis, which is central to the paper's argument that grammar explanations add *no statistically significant advantage* beyond vocabulary coverage.

- **The positive typological prompting claim is overstated relative to the evidence.** The abstract claims typological prompting "achieves leading results" on grammaticality judgment and IGT prediction, and §5 headlines this as a firm positive finding. However:
  - In Figure 1, TYP + BOOK_p beats BOOK_p by 2%, 0%, and 3% on the three judgment settings (65% vs. 63%; 76% vs. 76%; 83% vs. 80%).
  - In Table 5, TYP + BOOK_para beats BOOK_para by only 0.7 morpheme accuracy (46.1 vs. 45.4) and is *worse* on word accuracy and CHRf++.
  - No variance estimates, confidence intervals, or significance tests are reported for these tasks, despite the paper using significance language ("leading results").
  - These differences are plausibly within measurement noise given the 100-example and 97-example test sets used. The positive claim should be softened to an encouraging trend rather than an established finding.

### Minor

- **The 100-example Kalamang test set is small for drawing robust conclusions.** While a clear improvement over MTOB's 50 examples, a 100-example test set means reported CHRf++ differences of ~1–2 points carry substantial uncertainty. This is particularly relevant for the comparisons between BOOK_all and BOOK_para, where the margin (e.g., 34.4 vs. 30.8 eng→kgv for Gemini) is somewhat narrow. The paper acknowledges this indirectly but does not provide bootstrapped confidence intervals.

- **The Gemini-ft underperformance is relegated to an appendix without satisfying analysis.** The observation that fine-tuned Gemini underperforms NLLB and in-context Gemini by 6–12 CHRf++ is interesting and somewhat contradicts common expectations. Attributing this to Gemini being "already extensively instruction-tuned" is speculative and deserves at least brief empirical probing.

- **The grammaticality judgment task has an unverified validity assumption.** As the authors acknowledge (§3.4), local word swaps and shuffles may not always produce ungrammatical sentences in Kalamang. This limits how strongly Figure 1 can be read as evidence of grammatical knowledge acquisition—surface-level anomaly detection could partially drive performance. The limitation is appropriately disclosed but means the positive result must be taken with caution.

### Trivial

- The Nepali and Guaraní settings are qualitatively different from Kalamang (the models likely have prior knowledge of these languages; 0-SHOT is already high for npi). The paper appropriately hedges when discussing these as "seen low-resource" languages, but the introduction could make this asymmetry more prominent rather than grouping these findings equally with the Kalamang result.

---

## Nice-to-Haves

- **Test the extremely low parallel data regime (10–50 examples + grammar).** The paper tests grammar alongside 1,200+ parallel examples. In the truly data-starved scenario (≤50 examples), grammar explanations might matter more. This is the most consequential missing experiment for the generality of the "grammar doesn't help MT" conclusion.

- **Ablate which typological features drive the linguistic task improvements.** The TYP prompt is tested only as a complete block. Knowing whether word order, case marking, or tense features are the primary contributors would make the positive result much more interpretable and actionable.

- **Test with a pedagogical grammar book** (one designed for language learning rather than linguistic description). The kgv grammar is a descriptive theoretical text; a pedagogical grammar might provide clearer LLM-exploitable structure. The paper mentions this limitation but does not test it.

- **Scale analysis for parallel data.** Reporting how MT performance scales with 100 / 500 / 1,000 parallel examples vs. grammar book size would make the practical recommendations more actionable for data collection efforts.

- **Per-phenomenon error analysis.** Aggregate CHRf++ could mask that grammar explanations help with specific morphological or word-order phenomena even when overall scores do not improve. A small-scale breakdown would enrich the analysis.

---

## Removed Points

*These points are flagged to be removed; treat them with caution. They were raised in the input reviews but do not survive verification against the paper or the hard/soft rules.*

- **"Gemini API dependency limits reproducibility"** (Neutral Reviewer, Weakness 3): The paper explicitly justifies using Gemini-1.5-Flash due to context-length requirements (1M token window), also uses open-weight Llama-3.1-8B for comparison where context allows, and discloses all settings. This is a standard reproducibility concern rooted in the use of a closed API and falls under the removable nitpick category. The model exists and is accessible.

- **"Cross-language generalization claim is overstated"** (Harsh Critic, Concern 2): The paper does not claim Nepali and Guaraní support the same mechanistic conclusions as Kalamang. It explicitly states the npi effect sizes are smaller, attributes this to prior model competence, and limits the claim to generalizing the *negative* finding ("no evidence that LLMs can effectively exploit grammatical explanations for translation"). The framing in the paper is consistent with the evidence.

- **"Request for human evaluation beyond the appendix analysis"** (Neutral Reviewer, Suggestion 4): The paper notes the infeasibility of engaging proficient Kalamang speakers in a footnote, which is a legitimate constraint. Demanding human evaluation for XLR translation is non-standard and outside the paper's scope given this constraint.

---

## Novel Insights

The paper's most genuinely novel observation is that the apparent strength of grammar-book-prompted LLMs for XLR translation can be almost entirely accounted for by the *vocabulary coverage* of the parallel examples embedded in the book—not by any exploitation of the grammatical explanations. The regression in Figure 2 operationalizes this sharply: all prompt configurations, including those with full grammar books, sit on the same linear relationship between type coverage and CHRf++, meaning the explanatory text adds no measurable signal once coverage is controlled. The secondary insight—that the *form* of grammatical knowledge matters more than its *presence* (typological features help linguistic tasks; raw prose explanations do not help translation)—opens a productive direction for thinking about what kinds of linguistic resources are actually useful to LLMs in different task settings.

---

## Suggestions

1. Add bootstrapped confidence intervals to the CHRf++ scores in Table 2, and a permutation test or bootstrap-based significance test to the regression in Figure 2. With ~12 data points, additional robustness checks are essential for the paper's central statistical claim.
2. Soften the abstract and §5 framing of the typological prompting positive results (e.g., "promising preliminary evidence" rather than "achieves leading results") until significance can be established.
3. Add a brief experiment or discussion on the low-parallel-data regime—even showing 0-SHOT + TYP vs. 10-SHOT + TYP would partially address this.
4. Move the Gemini fine-tuning failure (currently appendix) into the main text with at least one attempted explanation beyond speculation about instruction tuning.

---

## Score and Decision

**Calibration against anchor papers:**

- **MTOB** (tbVWug9f2h.md): Scores 6, 8, 8 → accepted as spotlight. This is the paper being directly critiqued. MTOB introduced the benchmark; the paper under review delivers the ablation MTOB's reviewers explicitly asked for (CHRf++, NMT baseline, disentangling signal sources). The present paper is methodologically cleaner on the ablation but less novel as a standalone contribution.

- **ASROB** (sjvz40tazX.md): Scores 8, 3, 5, 6 → rejected. ASROB extended MTOB to speech with limited additional insight. The paper under review is substantively different: it directly challenges MTOB's core claim with a principled ablation and new analysis, rather than merely extending the task modality.

- The paper's weaknesses (small test set, regression on ~12 points, overclaimed typology results) are comparable to what MTOB's 6-scoring reviewer flagged (narrow modelling, insufficient interpretable evaluation), but the paper actually addresses several of those concerns (NMT baseline, CHRf++). The two weaknesses that remain unaddressed (small-sample regression, inflated positive claim) push back toward the lower end of acceptability.

**Assessment by axis:**
- *Originality*: Moderate — the ablation idea is straightforward but necessary and has not been done before for this benchmark.
- *Importance of research question*: High — directly tests whether a spotlight paper's central claim holds.
- *Claims well-supported*: MT negative claim: yes, strongly. Typology positive claim: weakly.
- *Soundness of experiments*: Good, but the regression sample size is a concern.
- *Clarity of writing*: Good — well-organized, honest about limitations.
- *Value to the research community*: Meaningful — the practical implication (parallel data > grammar text for XLR MT; NLLB fine-tuning competes) is actionable.

**Final score: 6.5** — positioned below MTOB's average (~7.3) because it is an analysis paper rather than a benchmark creator, and because the positive claim is overclaimed; but comfortably above ASROB's median because the central negative finding is rigorous, practically important, and well-executed.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>