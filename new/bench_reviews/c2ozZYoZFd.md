Now I have all the information needed to write the final review. Let me synthesize everything.

## Summary

This paper provides a comprehensive re-examination of the ICLR 2025 Oral paper on min-p sampling, demonstrating across four lines of evidence that the original paper's claimed superiority is undermined by its own data. The authors discover omitted human evaluation data, incorrect statistical tests, unfair hyperparameter comparisons, selective reporting, and unsubstantiated adoption claims—all confirmed by the original authors' corrections and retractions. The paper's most novel methodological contribution is a "Best-of-N" analysis that controls for hyperparameter search volume when comparing methods with unequal tunable parameter spaces, showing min-p's advantage vanishes under fair comparison.

## Strengths

- **Novel "Best-of-N" methodology for fair hyperparameter comparison (Section 3.1):** The subsampling-and-max approach—repeating 150 times per N—directly exposes how methods with more tunable hyperparameters can appear superior simply by having more configurations to search over. This addresses a widespread but rarely discussed problem and generalizes beyond this case study. Figures 4–5 demonstrate the method's effectiveness concretely.

- **Rigorous statistical re-analysis (Section 2.2, Table 1):** The Bonferroni correction (1 of 12 significant at α=0.05, 0 of 12 at α=0.01) and Intersection-Union Test (largest p-value 0.378) directly address the logical structure of the "consistently outperforms across all settings" claim. The IUT is particularly appropriate and underused in ML—claiming superiority *everywhere* requires evidence *everywhere*.

- **Specific, verifiable findings confirmed by original authors:** The omitted one-third of data (Section 2.1), retracted adoption claims (Section 5), and incorrect prompt formatting (Section 3) were all acknowledged or corrected by the original authors, making these findings unassailable.

- **Selective reporting discovery (Section 4.3):** Identifying that the higher of two scores was reported for min-p (52.01 at p=0.05 vs. 50.14 at p=0.01) while the lower was reported for top-p (50.07 at p=0.9 vs. 50.43 at p=0.98) is a specific, numerical finding that directly demonstrates biased presentation.

- **Extensive computational sweep (Section 3):** ~6000 A100-hours sweeping 9 models × 2 stages × 4 samplers × 31 temperatures × up to 6 hyperparameters × 3 seeds provides strong empirical coverage that the original paper lacked.

- **Adoption claims debunked with concrete evidence (Section 5):** The comparison of claimed 1.1M stars against the combined 453k stars of major LM repositories is a concrete sanity check, and the observation that 3 of 4 reviewers and the AC cited these (now retracted) numbers as primary justification shows how unsubstantiated claims directly distort peer review.

## Weaknesses

### Fatal
None.

### Major

- **Abstract overclaims in a way that mirrors the overclaiming being criticized:** The abstract states "min-p sampling improves neither quality, nor diversity, nor the trade-off between quality and diversity," but Section 3 acknowledges "min-p does produce higher scores for 2 of 12 language models" with standard formatting. The correct claim is that min-p does not *consistently* or *robustly* outperform baselines—not that it improves *neither*. A paper that positions itself as a blueprint for rigorous science should be scrupulously precise in its own claims; the absolute language in the abstract undermines this posture. The body text ("largely indistinguishable") is more measured, but the abstract is what most readers will cite.

- **Single-benchmark scope for the paper's most novel empirical contribution:** The Best-of-N analysis—the paper's most novel methodological contribution—covers only GSM8K CoT due to compute constraints (explicitly acknowledged: "Due to our compute budget, we only evaluated GSM8K CoT"). The original paper evaluated both GSM8K and GPQA. While the methodological point about hyperparameter volume fairness is general, the specific empirical counter-evidence that "min-p's advantage vanishes when controlling for hyperparameter tuning" is established on one benchmark. This limits the empirical generality of the finding.

### Minor

- **Best-of-N pool size asymmetry unaddressed:** Basic sampling has only 31 configurations (temperatures only), while top-k, top-p, and min-p each have 186 (31 temperatures × 6 hyperparameters). When subsampling N=100, basic sampling saturates its pool at N=31, making its Best-of-N curve constant beyond that point. The paper does not address this edge case. This does not affect the central comparison of min-p vs. other tunable samplers (all with 186-sized pools), but as the paper positions Best-of-N as a general "blueprint" methodology, the unaddressed asymmetry is a gap that future users should be warned about.

- **Low-diversity data excluded from human evaluation re-analysis:** The paper critiques the original authors for omitting 1/3 of data, then itself excludes the low-diversity setting. The three justifications are reasonable (authors said to ignore it; min-p's claimed advantage is high diversity; top-p's hyperparameter was poorly chosen), but a paper centered on data transparency should ideally present the full analysis alongside the focused analysis, enabling readers to independently verify the exclusion is warranted.

### Trivial
None.

## Nice-to-Haves

- Running the Best-of-N analysis on at least one additional benchmark (GPQA) would substantially strengthen the empirical generality of the central finding.
- Formalizing the Best-of-N methodology as a standalone procedure with explicit assumptions, pool-size guidance, and choice of N range would increase its impact beyond this case study.
- Section 2.4 documents that the new human evaluation changed so many variables simultaneously that it is essentially a different experiment; the paper could state this more forcefully rather than just listing the changes.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic's "scandals list is editorial"**: This is a formatting/presentation nitpick. The list effectively establishes context and is within the norms of an introduction.

- **Strength Finder's "uncovering unsubstantiated adoption claims with external verification" as a separate strength**: This overlaps heavily with the already-listed strength about Section 5. Consolidated into the existing strength.

- **Strength Finder's "identifying a likely incorrectly reported numerical value" (7.80 vs. 5.80)**: This is a minor observation in Section 2.4, not a core strength. The paper phrases it cautiously ("we believe one value is incorrectly reported") and it's a single number discrepancy, not a systematic finding. Moved to trivial/nice-to-have level.

- **Harsh critic's "qualitative annotation is inevitably subjective"**: The paper explicitly publishes annotations to mitigate subjectivity, which the harsh critic acknowledges. This is adequately addressed.

- **Demand for formalization of Best-of-N as standalone procedure**: Moved to Nice-to-Have. The current embedded presentation is adequate for the paper's scope.

- **Request for saturation analysis plot**: This is a nice visualization improvement but not a substantive weakness.

## Novel Insights

The paper reveals a striking asymmetry in how methodological rigor is applied: the original paper's claim of min-p "consistently outperforming across all settings" was structurally protected from falsification by its own methodology—pooling across settings masked individual failures, omitting baseline data removed unfavorable comparisons, and unequal hyperparameter tuning volume created an inherent advantage. The Intersection-Union Test insight is particularly important: when a claim is universal ("consistently outperforms everywhere"), the burden of proof is also universal, requiring evidence for every sub-claim rather than just a favorable aggregate. This structural observation—that the logical form of a claim determines the appropriate standard of evidence—has implications far beyond this case study and should be more widely recognized in ML evaluation.

## Suggestions

- Moderate the abstract's absolute language: replace "improves neither quality, nor diversity, nor the trade-off" with "does not consistently or robustly improve quality, diversity, or their trade-off" to align with the body text and avoid mirroring the overclaiming being criticized.
- Add a brief paragraph in Section 3.1 addressing the pool-size asymmetry: when a sampler has fewer configurations than N, its Best-of-N curve saturates. Recommend that future applications of this methodology either limit N to the smallest pool size or analyze samplers with comparable pool sizes separately.
- If possible in a revision, run the Best-of-N analysis on GPQA to establish the finding across the same benchmarks the original paper used.

## Calibration

**Anchor papers compared:**

- **jNiEMDsRgc.md** (avg 7.33, Accept Poster): LLM ranking fragility. More focused and polished statistical analysis; min-p paper is broader in scope with more confirmed findings but less clean methodologically. Slightly below this anchor.

- **YhgBy6jTR8.md** (avg 7.0, Accept Poster): Debunking generative classifier alignment claims by revealing omitted confound. Very similar type of contribution—revealing a confound invalidates prior claims. The low-pass filtering paper has a cleaner mechanistic explanation; the min-p paper has broader evidence and a more novel methodology. Roughly comparable.

- **EUAXc9Hlvm.md** (avg 7.0, Accept Poster): Context parroting debunking. Simpler, cleaner story with theoretical insight. Min-p paper is broader but has more gaps (single benchmark). Slightly below.

- **GLPmZhhCAE.md** (avg 5.5, Accept Poster): Fairness benchmark with extensive hyperparameter tuning showing well-tuned ERM beats debiasing. Very related theme. Min-p paper has a more compelling narrative and verified findings, clearly above.

- **H6PLJnnK6e.md** (avg 5.0, Accept Poster): MCTS hyperparameter tuning case study. More of an engineering contribution. Min-p paper is clearly superior with verified findings and a generalizable methodology.

- **zEJd3JXVxb.md** (avg 5.0, Reject): Dataset distillation debunking. Min-p paper is stronger with confirmed retractions and a novel methodology.

- **hOF6s8Yfxs.md** (avg 2.67, Reject): Hyperparameter search on test set critique. No clear contribution, no concrete solution. Min-p paper is vastly superior.

- **1CR1MTIgmq.md** (avg 0.00, Reject): Pure attack paper. Min-p paper is in a completely different category with genuine methodological contributions.

The paper sits above the medium-scoring debunking/reanalysis anchors (5.0–5.5) due to its confirmed findings, novel methodology, and impact (retractions), but below the highest-scoring ones (7.0–7.33) due to its single-benchmark limitation and abstract overclaim. Score: 6.5.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>