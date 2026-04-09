## Summary

This paper presents a detailed re-analysis of Nguyen et al. (2024), a high-visibility ICLR 2025 Oral paper that introduced min-p sampling for LLM decoding. Through systematic re-examination of the original paper's four lines of evidence—human evaluations, NLP benchmarks, LLM-as-a-Judge evaluations, and community adoption claims—the authors demonstrate that min-p's claimed superiority vanishes when methodological flaws are corrected (omitted data, improper statistical testing, unequal hyperparameter tuning, selective reporting). From this case study, the authors derive a general "blueprint" for more rigorous empirical ML research, centered on fair hyperparameter comparisons, proper statistical testing, data transparency, and scrutiny of qualitative claims.

## Strengths

- **Rigorous statistical re-analysis that invalidates the original paper's core claims.** The application of Bonferroni correction across 12 comparisons (Table 1) and the Intersection-Union Test for the "consistently outperforms" claim shows that min-p's statistical significance collapses from 5/12 to 1/12 at α=0.05 after correction. This is a methodologically sound and impactful demonstration of how incorrect statistical practice can manufacture false conclusions.

- **Novel "Best-of-N" hyperparameter control methodology.** Section 3 develops a principled framework for comparing methods that receive different volumes of hyperparameter tuning. By subsampling equal numbers of hyperparameters per sampler and computing maximum achievable performance, the analysis reveals that min-p's apparent advantage on GSM8K is an artifact of unequal search budgets (Fig. 4–5). This is a genuinely useful methodological contribution that addresses a widespread confound in empirical ML comparisons.

- **Comprehensive, multi-evidence re-analysis covering all four lines of original evidence.** Rather than cherry-picking one dimension, the paper systematically addresses human evaluations (omitted data, incorrect statistics, mischaracterized qualitative feedback), NLP benchmarks (unequal tuning), LLM-as-a-Judge (under-specified methodology, selective reporting favoring min-p), and adoption metrics (retracted claims). The breadth strengthens the overall case substantially.

- **Full data transparency enabling independent verification.** All re-analyses link to publicly available data, annotations, and code repositories, practicing what the blueprint preaches and making the critique itself reproducible.

## Weaknesses

### Major:

- **The blueprint's generalizability rests on a single case study.** The paper derives six general lessons from one paper's failures. While the authors state "the errors made in evaluating min-p are common in empirical machine learning research," this claim is supported only by the min-p analysis and a list of scandals in the introduction, not by systematic evidence that these specific failure modes (omitted data, incorrect multiple comparison correction, selective reporting of favorable hyperparameters) are prevalent across the field. The paper would be significantly strengthened by showing that even one additional high-profile paper exhibits similar issues under the same analytical framework, or by surveying the literature for prevalence of these specific errors.

- **GPQA benchmark claims remain unchallenged.** The original paper claimed min-p achieves "superior performance across benchmarks and temperatures" on both GSM8K and GPQA. The current re-analysis only sweeps GSM8K due to compute budget constraints. This leaves the GPQA portion of the original paper's benchmark claims unaddressed, creating a gap in the refutation. Even acknowledging the compute constraint, the paper should discuss whether there is reason to expect GPQA results to differ, or note this as an explicit limitation.

### Minor:

- **Statistical power is not discussed for the human evaluation re-analysis.** With n=53 participants and a Bonferroni-corrected α of 0.05/12 ≈ 0.004, the analysis may be underpowered to detect moderate effect sizes. The paper correctly applies the correction but does not address whether the study had sufficient power to detect the original paper's claimed effects under the corrected threshold. A brief power discussion would clarify whether "no significant difference" could reflect limited sensitivity rather than genuine equivalence.

- **The new human evaluation (Section 2.4) was conducted by the original authors in response to feedback, introducing potential confounds.** The paper documents that the original authors changed multiple factors simultaneously (temperature application order, participant pool, hyperparameters, stimuli, rubric), making it difficult to isolate why the new results differ. The paper's conclusions do not rely solely on this new study—the original data re-analysis is sufficient—but the discussion of Section 2.4 should more explicitly acknowledge these confounds.

- **The LLM-as-Judge critique identifies serious issues but does not run a corrected experiment.** The paper demonstrates that the original study used indirect comparisons (each sampler vs. basic), had 2–10× more hyperparameter tuning for min-p, and selectively reported favorable scores. However, the paper stops at critique rather than executing a direct pairwise comparison experiment that would definitively settle whether min-p matches baselines under fair conditions. Given the compute investment in Section 3, this seems like a tractable addition.

- **The 7.80 vs. 5.80 numerical discrepancy claim (Section 2.4) is stated with high confidence but limited documentation.** The paper asserts a value in Nguyen et al.'s Table 15 is incorrect based on "the authors' publicly posted data," but does not show the recalculation or provide a table comparing the reported vs. recomputed values. Given that this is a data integrity accusation, the verification should be presented with the same transparency the paper demands of others.

### Trivial:

- None of substance.

## Nice-to-Haves

- **Formalize the "Best-of-N" framework into a reusable tool or checklist.** The hyperparameter volume control methodology is the paper's most transferable contribution. Providing a simple script or checklist that researchers can apply before submission would convert the case-study lessons into an immediately actionable community resource.

- **Run a direct pairwise LLM-as-a-Judge comparison** (min-p vs. top-p, min-p vs. basic) under equal hyperparameter budgets to conclusively demonstrate the absence of min-p's advantage in that evaluation paradigm as well.

- **Add a hyperparameter sensitivity/variance analysis** to the Best-of-N results—analyzing not just maximum achievable performance but the width of high-performing regions for each sampler, which would indicate whether some samplers are more brittle or harder to tune.

- **Include GPQA in the benchmark sweep**, even with a more limited model/hyperparameter set, to close the gap in refuting the original paper's "across benchmarks" claim.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Citation stacking in the introduction** (harsh critic): Listing many references is standard in position papers arguing a field-wide problem. This is a style preference, not a substantive weakness.

- **Figure 1 being garbled** (harsh critic): This is a PDF extraction artifact, not a paper problem.

- **Title/framing mismatch between "blueprint" and case study** (harsh critic): The paper does deliver a blueprint (Section 6's six lessons), and the case study is the evidence from which it's derived. The framing is reasonable.

- **Missing creative writing benchmark** (spark finder): The paper re-analyzes the original paper's evidence; the original paper used GSM8K and GPQA for benchmarks, and human evaluations for creative writing. Requesting new experiments on WritingPrompts is scope creep beyond re-analysis.

- **Compute-adjusted tuning comparison** (spark finder): Controlling for number of hyperparameters rather than compute hours is a reasonable and simpler framework. This is a nice-to-have, not a flaw.

- **Telegram link as evidence source** (positive reviewer): The Telegram link is cited alongside a publicly accessible GitHub repository. The source is verifiable and the claim (selective reporting of scores) is also supported by the repository data. This does not undermine the paper's rigor.

## Novel Insights

The most striking insight from this re-analysis is the "Best-of-N" hyperparameter volume control concept: when methods are compared at equal search budgets, apparent advantages can evaporate entirely. This reframes the common empirical ML practice of giving new methods extensive hyperparameter sweeps while comparing against baselines at default or few settings—not just as "unfair," but as a mechanism that can manufacture false claims of superiority. The fact that min-p's advantage on GSM8K disappears under equal search budgets (Figs. 4–5) despite being presented as a fundamentally better sampling algorithm is a powerful illustration that hyperparameter tuning volume is itself a confounding variable, and that the field needs explicit accounting for it in all empirical comparisons.

## Suggestions

- Explicitly acknowledge the GPQA gap in the paper and discuss whether results are expected to generalize; even a small additional sweep on GPQA would significantly strengthen the completeness of the refutation.

- Add a brief statistical power analysis for the human evaluation study to clarify whether the "no significant difference" finding reflects genuine equivalence or limited sensitivity.

- Present the 7.80 vs. 5.80 verification with explicit recalculation details, consistent with the paper's own transparency standards.

- Consider formalizing the six lessons into a short "rigor checklist" that could be directly usable by authors and reviewers, transforming the case study's impact from "this specific paper was wrong" to "here is how to prevent this class of errors going forward."