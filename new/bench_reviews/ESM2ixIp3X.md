Now let me do calibration searches.Now I have enough to write the final review. Let me also quickly read the most comparable anchor paper.Now I have enough information for a complete review. Let me compile it.

---

## Summary

The paper revisits similarity-based metrics for summary factual consistency detection. The authors propose SBERTScore, which applies the BERTScore precision/recall formulation at the sentence level using sentence-transformer (SBERT) embeddings, and compare summaries against source documents rather than reference summaries. The paper demonstrates that this input-text change dramatically improves performance (BERTScore from 0.500 to 0.759 balanced accuracy) and that SBERTScore is competitive with trained NLI/QA metrics on the CNN/DM split of a nine-dataset benchmark. A secondary finding is that different metrics capture different error types, and that AND-combining metrics can improve aggregate performance.

---

## Strengths

- **Source vs. reference comparison (Table 3):** The empirical magnitude of the effect is striking: BERTScore goes from random-chance (0.500) to 0.759, and SBERTScore from 0.499 to 0.779, simply by switching comparison text. While Bao et al. (2023) previously fed source documents to BERTScore, this paper is the first to compare this setting systematically against the full NLI/QA landscape, providing actionable guidance for practitioners.

- **Granularity ablation (Table 4):** The Sent-Sent configuration is best (0.779), and the analysis isolates the contributions of architecture and granularity: Word-Word SBERT (0.767) vs. Word-Word BERT (0.759) shows the architecture gain; Sent-Sent SBERT (0.779) shows the further gain from granularity. The finding that 45.76% of source documents are truncated at Doc level clearly motivates sentence-level segmentation.

- **CNNDM competitiveness (Table 7a):** SBERTScore (0.720) outperforms the zero-shot NLI baseline SummaC_ZS (0.686) and even matches trained methods like DAE (0.696) and QuestEval (0.670) without any fine-tuning. This is a genuine and practically significant result for the zero-shot setting.

- **Error-type differentiation (Table 8):** SBERTScore achieves the highest recall on correct summaries (0.522 on CNN/DM vs. next-best 0.436), while NLI/QA metrics have higher recall on intrinsic errors. This characterisation is useful for downstream metric selection and motivates the combination approach.

- **Computational efficiency (Section 3.1):** The O(N+M) vs. O(NM) complexity argument is valid, and the empirical 3× speedup over SummaC and 30× over QuestEval are concrete practical contributions.

---

## Weaknesses

### Fatal
None.

### Major

- **The abstract overclaims competitive performance without qualification.** The abstract states SBERTScore "can compete with existing NLI and QA-based factuality metrics on the benchmark." Table 7b directly contradicts this on the XSum split: SBERTScore scores 0.605 balanced accuracy, behind BERTScore (0.695), QuestEval (0.665), and QAFactEval (0.705), and comparable to the weakest NLI baselines. XSum (2353 samples, single-sentence, highly abstractive summaries) is the largest and arguably hardest domain in the benchmark—it is not a corner case to be footnoted. The competitive claim should be scoped to CNN/DM, or the abstract should explicitly acknowledge the XSum failure. The paper does explain this in Section 5.5 (single-sentence summaries preclude averaging over sentence pairs), which shows awareness of the issue, but that explanation does not appear in the abstract or conclusions.

- **Limited method novelty.** SBERTScore (Equations 1–2) is literally the BERTScore precision/recall formula with cosine similarity over SBERT sentence embeddings instead of contextualised word embeddings. The conceptual distance from BERTScore is small. The paper's own Section 2.3 notes that Bao et al. (2023) already fed source documents to BERTScore and attempted sentence-level extension (albeit unsuccessfully with a different approach), and the NLI sentence-level idea was explored by SummaC. The contribution is more empirical than methodological, and the paper would be more honest to frame itself as an empirical investigation rather than a novel metric proposal.

### Minor

- **The "unfair experimental settings" framing mischaracterises the original design intent.** Section 1 and Section 5.2 frame prior similarity-based metric failures as arising from "unfair experimental settings." But reference-based BERTScore was designed as a generation quality metric against references—comparing to references was not an error in prior evaluations, it was the intended use. Using source documents instead creates a different metric, not a "fairer" version of the same one. A more accurate framing is that the paper repurposes similarity-based metrics for the factuality task, which is a legitimate contribution but a different claim.

- **The AND-combination result is metric-pair dependent and the claim should be qualified.** The paper states AND combinations "improve the balanced accuracy." This is true for many pairs (e.g., AND(DAE, QAFactEval) = 0.828 > both 0.807 and 0.817), but AND(SBERTScore, QAFactEval) = 0.801 is actually below QAFactEval alone (0.817). The paper should clarify that AND only reliably improves results when the two metrics have sufficiently different error profiles and similar individual accuracy levels—combining a weaker metric with the strongest can hurt.

- **No comparison to LLM-based factuality metrics.** The competitive baselines are all from 2020–2022 (SummaC, DAE, QAFactEval, QuestEval). Since ~2023, prompt-based LLM evaluators (e.g., G-Eval, AlignScore) have become the reference point in factuality evaluation. The paper's competitive claim is made against a somewhat dated field. The argument that SBERTScore is valuable as a zero-shot, low-cost alternative could be strengthened or weakened significantly by including at least one modern LLM-based reference.

### Trivial

- None material.

---

## Nice-to-Haves

- Performance breakdown within XSum by summary length would directly test the authors' hypothesis that single-sentence summaries are the root cause of SBERTScore's degradation.
- A precision-recall curve or ROC comparison between SBERTScore and BERTScore would clarify whether SBERTScore's high recall on correct summaries is a genuine threshold effect or a calibration artifact (the threshold-at-consistent-bias hypothesis in the error analysis).
- A controlled ablation swapping SBERT weights for standard BERT weights at the sentence level would isolate whether the improvement comes from the sentence-transformer training objective (NLI/paraphrase) or purely from the sentence-level aggregation.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: "Bao et al. already discovered the source-vs-reference finding."** Partially misreads the paper. Section 2.3 explicitly states Bao et al. did not compare across metric paradigms. The paper's cross-paradigm benchmark comparison (Table 7) is incremental but real.
- **Harsh critic: "SBERTScore AND QAFactEval improves over QAFactEval."** Actually the harsh critic correctly identified this does not hold, though the paper itself doesn't claim this specific combination—it claims AND generally improves (which is true for most pairs). Kept but placed in Minor above.
- **Harsh critic: Section 3.1 complexity argument is misleading (SummaC runs in same O(N+M)).** Partially valid—SummaC_ZS does run sentence-pair comparisons in the same complexity class, and the actual speedup is likely implementation-dependent. However, the empirical runtime comparison is provided (Appendix A) so the claim is at least empirically grounded. Weakened to trivial note.
- **Harsh critic: Size-weighted aggregate results missing.** This is a reasonable methodological nicety but not a core flaw given that balanced accuracy is the standard in this benchmark evaluation literature; removed as not substantively undermining claims.
- **Harsh critic: SBERTScore "biased toward predicting consistent."** The paper explicitly frames this high recall-on-correct-summaries pattern as a strength and motivates the complementarity discussion. The criticism mischaracterises the paper's own analysis as neglectful when it is in fact the central framing of Section 5.6. Removed.

---

## Novel Insights

The most genuinely informative contribution of the paper is not SBERTScore itself but the empirical demonstration that the long-standing perceived failure of similarity-based metrics in factuality evaluation is an artifact of input-text choice, not a fundamental limitation of the paradigm. The magnitude of improvement from simply switching comparison text (random-chance → competitive with trained systems) suggests the field has been systematically miscalibrating its baselines. Additionally, the error-type analysis revealing that similarity-based metrics have high recall on *correct* summaries (low false-positive rate) while NLI/QA metrics have higher recall on *error types* suggests a principled asymmetry that practitioners can exploit for high-precision factuality pipelines. The AND-combination framework operationalises this complementarity, though the gains depend substantially on which pair is combined.

---

## Suggestions

1. **Qualify the abstract.** State that SBERTScore is competitive on the CNN/DM split but underperforms on the XSum split, with a brief explanation (single-sentence summaries prevent sentence-pair averaging). This makes the claim honest without downgrading the contribution.
2. **Reframe from "unfair experimental settings" to "off-label use of reference-based metrics."** The paper's contribution is repurposing similarity metrics for the source-conditioned factuality task, not correcting an experimental error. This framing is more accurate and arguably stronger.
3. **Add a note on which metric combinations are beneficial vs. harmful.** Clarify that AND helps when both metrics have similar individual accuracy but different errors; when one metric is substantially stronger (QAFactEval), AND with a weaker metric may degrade.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Human Score | Comparison to paper under review |
|---|---|---|
| `/home/wg25r/review_agent/human_reviews/rYyu3jpk8z.md` | 4.80 | Open-domain text evaluation paper, also incremental method with good empirical work, rejected. Most topically similar. Paper under review is roughly comparable—slightly more focused scope, similarly limited novelty. |
| `/home/wg25r/review_agent/human_reviews/Rry1SeSOQL.md` | 6.75 | Reference-free MT evaluation, reformulated as ranking problem, achieves SOTA across three benchmarks. Stronger novelty, stronger results. Paper under review is weaker on both dimensions. |
| `/home/wg25r/review_agent/human_reviews/Im2neAMlre.md` | 7.33 | Systematic evaluation of T2I metrics with >100K annotations; genuinely large-scale and comprehensive. Much stronger scope than paper under review. |
| `/home/wg25r/review_agent/human_reviews/xN6z16agjE.md` | 3.00 | Arabic hypernymy evaluation—narrow, weak contribution. Paper under review is clearly better. |
| `/home/wg25r/review_agent/human_reviews/kDakBhOaBV.md` | 4.00 | Data diversity metric paper with limited novelty, rejected. Paper under review is somewhat comparable—both are metric papers with incremental contribution. |

**Assessment:** The paper sits squarely near the borderline between the rYyu3jpk8z cluster (avg 4.80) and the kDakBhOaBV cluster (avg 4.00). The practical value is real—the source-vs-reference finding is actionable, the error-type analysis is informative, and the zero-shot computational advantage matters. However, the method is a near-trivial substitution, the central competitive claim is only half-supported, and the paper does not engage with modern LLM-based evaluation methods. Compared to accepted metric papers in the 6–7 range (MT-Ranker, Gecko), it lacks the scope, novelty, and empirical robustness to reach acceptance. The rYyu3jpk8z paper (4.80, rejected) is the closest anchor—the paper under review is in the same quality tier. I score it **4.5**: the empirical contributions are solid and practically useful, but the methodological thinness and partially unsupported abstract tip it below the acceptance bar.

**Originality:** Low-medium — the core idea (source input + SBERT) is straightforward; the cross-paradigm comparison adds some value.
**Importance of research question:** Medium — factuality evaluation is important; this is a meaningful sub-problem.
**Claims well-supported:** Partially — CNN/DM claim is well-supported; XSum claim is not.
**Soundness of experiments:** Good — nine datasets, appropriate baselines, significance tests, ablations.
**Clarity of writing:** Good — well-organized and easy to follow.
**Value to community:** Limited but real — practitioners get actionable guidance at low cost.

**Decision: Reject** — below acceptance threshold due to insufficient methodological novelty and a partially unsupported central claim.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>