Now I have a thorough understanding of the paper. Let me compose the final review.

## Summary

This paper provides a detailed forensic re-analysis of the ICLR 2025 Oral paper "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs," demonstrating through four lines of evidence that the original paper's claims of min-p's superiority are unsupported by its own data. The paper identifies omitted human evaluation data (1/3 of scores), incorrect statistical analysis (no multiple-comparison correction), mischaracterized qualitative feedback, selective reporting in LLM-as-a-Judge evaluations (reporting the higher of two scores for min-p, the lower for top-p), and retracted community adoption claims. It also introduces a "Best-of-N" methodology for equalizing hyperparameter search volume across methods, showing min-p's benchmark advantage vanishes under fair comparison.

## Strengths

1. **The forensic re-analysis is thorough, rigorous, and devastating.** The paper systematically dismantles each of the original paper's four evidence lines with concrete, verifiable findings: omitted data (Section 2.1), improper statistical tests (Section 2.2, including the IUT argument with maximum p-value 0.378), mischaracterized qualitative responses (Section 2.3), and selective metric reporting (Section 4.3). The selective reporting finding—reporting 52.01 (p=0.05) for min-p but 50.07 (p=0.9) for top-p while suppressing 50.14 (p=0.01) and 50.43 (p=0.98)—is particularly damning and hard to catch without data access.

2. **The Best-of-N hyperparameter volume methodology is a genuine and transferable methodological contribution.** The subsampling approach that equalizes the number of hyperparameters evaluated per sampler (N=1 to 100, averaged over 150 repetitions) provides a principled, implementable protocol for fair comparison of methods that require different amounts of tuning. This addresses a real gap in empirical ML methodology.

3. **The documented impact on peer review is important.** Section 5 documents that 3 of 4 reviewers and the AC cited the now-retracted 54K repositories / 1.1M stars claims as primary justification—connecting methodological flaws directly to real-world review outcomes.

4. **The paper practices what it preaches on transparency.** The re-annotations, data, and analysis are all publicly posted, and the paper explicitly acknowledges its key limitation ("Conclusions here are based on that evidence. We emphasize that new evidence might lead to different conclusions.").

## Weaknesses

### Fatal
None.

### Major

- **The "blueprint" framing overreaches relative to what is actually novel.** The title and abstract promise "a blueprint for conducting more rigorous science," but five of six lessons (correct for multiple comparisons, release all data, scrutinize qualitative summaries, ensure methodological clarity, watch for selective reporting) are well-established principles in statistics and experimental methodology. Only the Best-of-N hyperparameter volume methodology (lesson #1) is genuinely new. The paper's primary contribution is the forensic case study itself, not a general blueprint—the framing as such oversells the constructive contribution. This is a structural issue because it positions the paper as providing a novel methodology for the field, when the methodology contribution is narrower than claimed.

- **The broad conclusion that min-p "offers no apparent advantage" extends beyond what the NLP benchmark evidence alone supports for the creative-writing domain.** The paper's NLP benchmark extension (Section 3) uses only GSM8K CoT—a mathematical reasoning benchmark—while min-p's original claim was about creative and coherent text generation. The human evaluation re-analysis (which did use creative writing tasks) already shows min-p doesn't consistently win, so the overall conclusion is still well-supported. However, the NLP benchmark results add evidential weight primarily for reasoning tasks, not for the creative-writing use case that motivated min-p. The paper acknowledges the compute limitation ("Due to our compute budget, we only evaluated GSM8K CoT") but does not qualify the generality of its null finding accordingly.

### Minor

- **The 7.80 vs. 5.80 discrepancy is presented with hedging ("we believe") that could be stated more conclusively.** The paper says "we believe the correct numerical value should be 5.80" based on the authors' publicly posted data. If the data is publicly available, this should be verifiable and stated as a confirmed error rather than a belief.

- **The paper does not distinguish between "min-p is not consistently superior" and "all samplers are equivalent."** The finding that 2 of 12 models showed min-p advantages even after prompt format correction deserves more discussion—what explains those exceptions, and do they matter for practitioners choosing a sampler?

- **The Best-of-N methodology lacks formal characterization.** The paper demonstrates the approach on one benchmark (GSM8K) with 6 hyperparameters per sampler. Properties like variance, bias, and sensitivity to the choice of hyperparameter grid are not analyzed. This does not undermine the GSM8K finding but limits immediate generalization as a standard tool.

### Trivial
None.

## Nice-to-Haves

- Evaluate min-p on creative-writing benchmarks (e.g., AlpacaEval creative writing) using the Best-of-N methodology to directly test the original use case.
- Formalize the statistical properties of the Best-of-N subsampling estimator (variance, bias, failure modes).
- Show full Best-of-N curves for each sampler individually, not just the difference plot, to help readers assess whether any sampler consistently leads at specific budgets.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh critic's claim about the Best-of-N analysis "conflating equal search budget with equally easy to tune."** The paper's conclusion is appropriately narrow: "min-p does not outperform other samplers when controlling for hyperparameter volume." It does not claim all samplers are equally easy or hard to tune. The question of tuning difficulty is interesting but outside the paper's stated scope, which is about fair comparison. **Removed because it demands the paper address a concern outside its scope.**

- **Harsh critic's claim about "appeal to mob dynamics" in the introduction's list of prior scandals.** This is a pure style/presentation nitpick. The list of prior rigor scandals establishes legitimate context for the work. **Removed as a formatting/presentation nitpick.**

- **Harsh critic's suggestion about LLM-as-a-Judge indirect comparison being "necessarily worse."** The paper does not claim direct comparison is the only valid design; it notes the indirect comparison introduces confounds due to known non-transitivity. This is a reasonable concern, not an overclaim. **Removed as a misunderstanding of the paper's argument.**

- **Strength finder's claim that the paper's single most important evidence is the Best-of-N analysis.** This overstates the contribution. The forensic findings (omitted data, selective reporting, incorrect statistics) are more important and more damning than the benchmark re-analysis. **Removed because it conflicts with the verified major weakness about overclaiming the methodology contribution.**

- **General formatting and presentation nitpicks flagged by the harsh critic.** Removed per the hard rule on formatting artifacts.

## Novel Insights

The paper reveals a striking pattern where multiple independent flaws (omitted data, incorrect tests, mischaracterized qualitative summaries, selective metric reporting, unsubstantiated adoption claims) all directionally favored the proposed method. While the paper doesn't explicitly characterize this pattern, the convergence of errors in one direction—which collectively enabled a high-visibility oral paper and influenced multiple reviewers—is more concerning than any single error alone. The selective reporting in Section 4.3 (higher-of-two for min-p, lower-of-two for top-p) is particularly diagnostic: this is the kind of bias that is almost impossible to catch without raw data access, making a strong case for mandatory data release in empirical ML papers.

## Suggestions

- Reframe the contribution more honestly: the paper's primary value is the forensic case study exposing specific flaws in a high-visibility paper, with one novel methodological tool (Best-of-N volume control) as a secondary contribution. The "blueprint" framing should be toned down to reflect that most lessons are established principles newly illustrated rather than novel.
- Qualify the "no apparent advantage" conclusion to specify that it holds for mathematical reasoning benchmarks with equal hyperparameter search budgets; the creative-writing domain is supported by the human evaluation re-analysis but not directly by the NLP benchmark extension.
- Verify and conclusively state (rather than hedging with "we believe") the 7.80 vs. 5.80 discrepancy, since the data is publicly accessible.

## Calibration

Comparing against retrieved anchors:

**High-scoring papers (≥6):**
- *Never Train from Scratch* (avg 8.0, Oral): Fair comparison of methods revealing overestimated differences—conceptually similar to this paper's equal-hyperparameter-volume point. That paper was more constructive (proposing SPT), scored 8.
- *Realistic Evaluation of Deep PLL Algorithms* (avg 7.5, Spotlight): Forensic re-evaluation exposing unfair prior comparisons, proposing a fair benchmark. Very similar contribution profile to this paper. Scored 7.5.
- *Dataset Bias* (avg 8.0, Oral): Revisiting prior experiment showing persistent bias. Scored 8 despite limited novel methodology.
- *SS Unseen-Class re-evaluation* (avg 6.0, Poster): Re-evaluation with controlled variables showing prior claims were flawed. Scored 6.

**Low-scoring papers (≤4):**
- *Is Memorization Actually Necessary?* (avg 3.75, Reject): Rebuttal of prior work with methodological errors, but reviewers found the analysis had its own flaws and didn't provide constructive alternatives.
- *Joint Training Does Not Transfer* (avg ~2.6, Withdrawn): Pure rebuttal with no constructive contribution and poor presentation.
- *Does Deep Active Learning Work in the Wild?* (avg 3.4, Withdrawn): Shows methods fail when hyperparameters aren't cherry-picked. Conceptually related to this paper's Best-of-N point. Rejected for limited constructive contribution.
- *Do Think Tags Really Help LLMs Plan?* (avg 4.0, Withdrawn): Critical evaluation of ReAct claims, showing performance driven by exemplar similarity. Similar premise but weaker execution.

**Medium (≈5):**
- *Best-of-N / majority voting for sampling* (avg 5.75, Poster): Proposes inference-time strategies, related methodology but not a critique paper.

This paper is substantially stronger than the low-scoring anchors (which either had their own analysis flaws or lacked constructive contribution). It is comparable to *Realistic Evaluation of Deep PLL Algorithms* (7.5, Spotlight): both expose unfair prior comparisons through careful re-analysis and propose a fair evaluation framework. The current paper's forensic findings are more devastating (omitted data, selective reporting) but its constructive methodology contribution is narrower (one technique, not a full benchmark). It is somewhat weaker than *Never Train from Scratch* (8.0) which proposed both a diagnosis and a concrete solution. Given this calibration, a score of 7.0 is appropriate: strong forensic contribution with a real but narrower methodology contribution, slightly oversold as a "blueprint."

**Originality:** The forensic re-analysis approach and Best-of-N methodology are novel. The "blueprint" lessons are mostly well-established principles, but one (hyperparameter volume control) is genuinely new.  
**Importance:** High—exposing methodological flaws in a high-visibility ICLR Oral paper that influenced reviewers has significant community value.  
**Claims support:** The forensic claims are extremely well-supported with specific, verifiable evidence. The "no apparent advantage" conclusion is well-supported overall, though slightly overgeneralized from GSM8K.  
**Experimental soundness:** Excellent—the statistical re-analyses (Bonferroni, IUT) are appropriate and the Best-of-N methodology is sound, though not formally characterized.  
**Clarity:** Well-written and well-organized, though the "blueprint" framing could be more precise.  
**Value:** High—the community needs this kind of careful auditing, regardless of whether the "blueprint" is as novel as claimed.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>