## Summary

This paper investigates under what conditions exact benchmark contamination in pretraining data affects final evaluation performance. Through controlled retraining experiments scaling along three axes—model parameters (up to 1.6B), training tokens (up to 40B / 15× Chinchilla), and repetition count (up to 144×)—the authors find that contamination effects diminish monotonically with training tokens and can vanish entirely with sufficient clean data. They complement this with a theoretical analysis showing that AdamW's weight-decay mechanism implies an exponential attenuation of past gradient updates, providing an upper bound on forgetting that they use to estimate forgetting dynamics in larger public models such as OLMo-7B and Llama 3 405B.

---

## Strengths

- **Direct measurement via controlled retraining, not post-hoc detection.** Rather than using inference-time proxies, the paper explicitly injects contamination and measures accuracy gaps—a methodologically cleaner approach that avoids the approximation errors of influence functions and the assumption failures of n-gram detection methods. This is a genuine design contribution.

- **Systematic characterization of three independent scaling axes.** The paper disentangles the effects of model size, token count, and repetition count in well-controlled ablations (Figure 2). Most prior contamination work conflates these; cleanly showing that contamination effects are monotone in all three dimensions and that token count can fully counteract the other two is a substantive quantitative contribution.

- **Novel empirical finding: novel data is necessary for forgetting.** Figure 3a vs. 3d directly shows that repeated training on the same 100M tokens stabilizes the contamination effect at a nonzero level, while novel data drives it to zero. This is a crisp mechanistic observation not previously isolated in the contamination literature.

- **Spaced repetition finding.** Figure 3e/3f shows that benchmark questions distributed uniformly throughout training exhibit *more* overfitting than questions clustered at the end—a counterintuitive result with practical implications for how contamination timing matters.

- **Quantitative AdamW forgetting theory.** Proposition 1 and the derivation in Section 5.1 provide a closed-form upper bound on example forgetting as a function of optimizer hyperparameters alone. This allows the framework to be applied without retraining, which is the main value of the theoretical contribution. The empirical validation that actual forgetting is faster than the bound strengthens (rather than weakens) the practical implications.

- **OLMo-1B validation: 96% of contamination effect decays in 1% of remaining training.** Figure 4 demonstrates extremely rapid forgetting in a realistic model—the magnitude (15 pp increase → nearly zero within ~1700 steps out of 739,328 total) is striking and quantitatively concrete.

---

## Weaknesses

### Fatal

None.

### Major

- **Scale gap between experiments and claims about frontier models.** The core controlled experiments use models up to 1.6B parameters. The extrapolation to Llama 3 405B (Section 5.1, Figure 5c) relies entirely on the theoretical bound applied to assumed hyperparameters (weight decay = 0.1, inferred from public reports), with no empirical validation at an intermediate scale such as 7B. Because the theory is an upper bound—and empirical forgetting is shown to be faster—the bound is safe in one direction, but the key question of whether the scaling trends from 124M–1.6B actually hold at 7B+ is empirically unanswered. The paper explicitly acknowledges this gap, but the language in the abstract and introduction (e.g., "many LLMs, including Llama 3, have forgotten the data seen at the beginning of training") still implies empirical certainty that is not warranted. These claims should be phrased as theoretically-bounded estimates, not factual conclusions.

- **Only exact contamination is studied; conclusions are not transferable to realistic contamination scenarios.** The entire empirical setup uses exact string insertion. Real-world contamination concerns typically involve paraphrases, benchmark-style leakage, answer-only leakage, or web-scraped content that closely paraphrases evaluation items. Approximate contamination may generalize better to clean evaluation examples and thus resist forgetting differently from exact copies. The Discussion acknowledges this limitation, but the abstract and conclusion do not adequately hedge the scope of the findings. A paper making recommendations about when evaluations remain valid must be clearer that these results apply to exact contamination only.

- **Uncertainty quantification is incomplete.** Bootstrapped confidence intervals are computed over evaluation examples, not over independent training runs. Each data point in Figure 2 appears to come from a single training run. For a paper whose central conclusions ("12x contamination becomes insignificant at 15× Chinchilla") rest on gaps falling within confidence bands, the absence of training-run variance is a genuine validity concern. At minimum, the paper should explicitly state how many training runs were performed per condition and discuss whether the run-to-run variance is expected to be small relative to the example-level bootstrap intervals.

### Minor

- **"Completely forgotten" is stronger than what the metric supports.** Section 4.2 defines complete forgetting as "no longer any accuracy difference between contamination and holdout benchmark questions." This is accuracy-level forgetting. The model may still assign higher log-likelihood to contaminated examples, retain internal representational traces, or differ on more sensitive metrics. Figure 3a tracks CE loss differences and shows they narrow but does not show them reaching exactly zero. The terminology should be tightened: "no detectable accuracy gap" or "accuracy-level forgetting" avoids overstating the result.

- **Pooled benchmark reporting may mask heterogeneous behavior.** All main results aggregate across seven benchmarks. Contamination effects plausibly differ across HellaSwag (long-context completion), MMLU (knowledge questions), BoolQ (binary QA), etc. The claim that "12 repetitions become insignificant at 15× Chinchilla" is more convincing if it holds benchmark-wise. A per-benchmark analysis (even in the appendix) is important for assessing robustness.

- **Gradient alignment as a confound for the theoretical bound.** The theory treats gradient updates from contaminated examples as independent of clean-data gradients. The paper acknowledges in the main text that orthogonality is needed for the stronger guarantee, but if contaminated benchmarks cover knowledge also present in FineWeb-Edu, later clean gradients might reinforce rather than cancel contaminated ones—meaning neither weight-decay decay nor gradient cancellation operates as assumed. This is not just a theoretical footnote; it affects whether the bound is conservative in both directions.

- **OLMo-1B section samples a single temporal point.** Contamination is inserted at step 369,000 and the subsequent forgetting is tracked. This demonstrates forgetting at one specific point in training but cannot characterize whether the same rate holds at earlier or later points in the training trajectory. Given that the spaced-repetition finding (Section 4.2) shows position matters, a single injection point weakens the generality of the OLMo results.

### Tiny

- The 1× repetition regime is practically the most relevant (sparse real-world contamination) but is excluded from main experiments because effects "fall within the margin of confidence intervals" (footnote 1). The paper infers a ~3 pp effect for 1× from a linear extrapolation from 4×, which assumes linearity in a potentially nonlinear regime. This should be flagged more clearly as a rough inference.

- The y-axis calibration in Figure 6c (theoretical curve anchored to empirical peak) is informal. It makes the qualitative fit visually compelling but weakens the quantitative comparison. A brief note on what the calibration does and does not imply would clarify the status of the comparison.

---

## Nice-to-Haves

- **Run one forgetting experiment at 7B scale** (OLMo-7B is open-source with known hyperparameters and public checkpoints). This is the single most impactful missing experiment—it would directly test whether the 1.6B findings hold at a scale closer to the models discussed in Section 5, and would validate whether the theoretical bound is conservative in the right direction at larger scales.

- **Test approximate/paraphrased contamination.** Even a small-scale experiment with manually paraphrased benchmark questions (or MT-generated variants) would substantially expand the applicability of the findings and directly address the scope limitation.

- **Per-benchmark forgetting curves.** Show whether the forgetting dynamics are consistent across all seven benchmarks or whether certain task types (e.g., factual MMLU questions vs. commonsense HellaSwag) resist forgetting differently. Even in an appendix, this would significantly strengthen the robustness of the pooled claims.

- **Discuss late-training contamination risk explicitly.** Figure 3e shows that end-of-training contamination is remembered less than uniformly distributed contamination, which might seem reassuring, but contamination in the final training stage (e.g., preprocessing occurring late) may not have sufficient remaining training to decay. A brief analysis of "minimum cool-down tokens needed" as a function of repetition count would be actionable for practitioners.

- **Sensitivity analysis on Llama 3 hyperparameter assumptions.** The analysis in Figure 5c assumes weight decay = 0.1. Showing the forgetting bounds for weight decay ∈ {0.01, 0.1, 0.3} would make the Llama 3 conclusions more robust to uncertainty in the actual optimizer configuration.

- **Gradient alignment visualization** (cosine similarity between contaminated-data and clean-data gradients over training) would empirically test whether the orthogonality assumption underlying the stronger version of Proposition 1 is approximately satisfied in practice.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Title overreach (Harsh Critic):** The paper's stated research question ("how does the presence of a text in training data influence performance on that same text?") is well-scoped in Section 3, and the experimental design matches this scope. The broader framing in the title is standard academic presentation, not overclaiming.

- **Duplicate filtering details must be in the main text (Harsh Critic):** The paper explicitly notes the filtering is documented in Supplement A.3 with an experiment to verify validity. Supplement is an appropriate venue for this level of detail; the main text describes the procedure sufficiently for understanding.

- **N-times Chinchilla is not the right normalization (Harsh Critic):** The paper is not claiming this is theoretically optimal—it is using a widely-adopted empirical anchor to contextualize training scales. This is appropriate and not a methodological error.

- **"Individual examples don't matter is too broad" (Harsh Critic):** The paper's claim is explicitly scoped to performance on that same example at end-of-training evaluation. Privacy-style memorization and fine-tuning sensitivity are explicitly carved out in the Discussion. This is reasonable scope management, not overclaiming.

- **Harsh Critic's concern that exact contamination "favors detectability" and is therefore too optimistic:** This is actually a strength of the design—exact contamination is the hardest case for forgetting to occur. If even exact contamination is forgotten, approximate contamination is presumably also forgotten. The experiment is appropriately conservative.

- **Missing comparison to specific related works (from any reviewer):** Per instructions, missing related work citations are not raised as per the evaluation rules.

---

## Novel Insights

The most genuinely novel mechanistic insight is the connection between optimizer hyperparameters and forgetting rates that emerges from Equation (3) and Proposition 1: weight decay and learning-rate schedule jointly determine an upper bound on how long any past gradient update can persist in the final model weights, and this bound is analytically computable for any public training run with known hyperparameters. This transforms the contamination question from "did this data appear in training?" to "is there still any optimizer-preserved trace of it in the final weights?"—a shift with broader implications for data attribution and evaluation validity that the paper begins to develop but does not fully explore. A second underappreciated insight is the spaced-repetition finding (Section 4.2): uniform distribution of contaminated examples throughout training produces *stronger* final overfitting than end-clustered repetition, which inverts the naive recency-weighted intuition and has concrete implications for how contamination timing should be reported and interpreted in practice.

---

## Suggestions

1. **Rewrite all "LLMs have forgotten early data" claims in Section 5 and the abstract to explicitly qualify them as theoretical upper-bound estimates dependent on assumed hyperparameters.** Phrases like "indicates that many LLMs... have forgotten" should become "our theoretical bound implies that, assuming standard AdamW hyperparameters, gradient updates from early training data have decayed below ε."

2. **Add a paragraph to Section 3.2 or Section 4 explicitly stating** that all empirical conclusions apply to exact-match contamination only, and that applicability to approximate or semantic contamination is an open question. This should appear before the first main result, not only in the Discussion.

3. **Report the number of training runs per experimental condition** and discuss expected variance. If only single runs were used (as seems likely given compute constraints), add a brief robustness discussion noting which conclusions would require multiple runs to verify statistically.

4. **Move the orthogonality condition from the supplement to a prominent box or paragraph in Section 5.1**, clarify what it means intuitively (clean-data gradients should not re-encode contaminated information), and discuss whether it is expected to hold when clean data shares a knowledge domain with contaminated benchmarks.

5. **Add per-benchmark forgetting curves** (accuracy gap by benchmark vs. token count) to the appendix and reference them from Section 4.1 to let readers assess whether any single benchmark is driving the pooled results.

6. **Provide a concrete "practical guide" table** summarizing: given N repetitions and model scale M, approximately how many additional training tokens are needed before contamination effects fall below some threshold. This synthesizes the empirical findings into an actionable form for dataset curators and evaluation designers.