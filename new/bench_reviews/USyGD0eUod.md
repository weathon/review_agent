## Summary

This paper applies a sanity-check methodology to SAE evaluation metrics by comparing SAEs trained on trained transformers against those trained on randomly initialized variants (with multiple randomization schemes that isolate different factors). The central finding is that auto-interpretability AUROC scores—widely used to evaluate SAE quality—produce similar or even higher values for randomized models than for trained models across five Pythia model sizes (70M–6.9B), while only a Gaussian-embedding Control sits at chance level. The paper also introduces token distribution entropy as a preliminary metric that does distinguish trained from random models, and presents toy model experiments suggesting random networks may preserve or amplify superposition.

## Strengths

- **The core empirical finding is important and timely.** Showing that auto-interpretability AUROC scores fail to distinguish trained from randomly initialized transformers (Figure 1: trained at 0.79, random variants at 0.87–0.88 for Pythia-6.9b; Figure 2: AUROC rows across five model sizes) is exactly the kind of rigorous sanity check the mechanistic interpretability field needs, following the spirit of Adebayo et al. (2020). This result should change how the community uses and reports these metrics.

- **Multiple well-motivated randomization variants isolate different factors.** The distinction between Re-randomized incl. embeddings, Re-randomized excl. embeddings, Step-0, and Control (Section 3) allows disentangling the contributions of embedding structure, parameter scale, and trained computation. The finding that norm-preserving randomization (Re-randomized) more closely resembles the trained model than Step-0 initialization (Figure 2, L1 norm row) is a concrete, interpretable observation that suggests parameter scale matters for SAE metrics.

- **The token distribution entropy metric successfully discriminates trained from random models.** Figure 2 (last row) shows trained models have increasing entropy across layers (features becoming more abstract), while randomized variants maintain consistently low entropy (features remain token-specific). This demonstrates the gap is measurable—just not captured by standard metrics—and provides a constructive proof-of-concept for better evaluation.

- **Scale of experiments lends weight to the findings.** Testing across five model sizes (70M to 6.9B), multiple layers per model, and 100M training tokens (with 1B-token robustness check in Appendix C) makes the failure pattern credible and demonstrates it worsens with model size.

## Weaknesses

### Fatal
None.

### Major

- **The paper's framing softens a more alarming finding: random models *outperform* trained models on AUROC for large models.** Figure 1 shows trained Pythia-6.9b at AUC=0.79 while random variants reach 0.87–0.88. The paper frames this as metrics that "do not distinguish" (abstract, Section 1, Section 3 text), but the data shows the metric actively *reverses* the correct ranking. The distinction matters: "doesn't distinguish" implies the metric is merely uninformative; "gets it backwards" implies it is actively misleading. The paper briefly mentions that "randomized variants are more similar to the trained model than the variant at initialization" (Section 3) and speculates about parameter norms and SAE size, but does not seriously investigate *why* random models score higher on a metric that ostensibly measures interpretability of learned features. If the metric rewards features that are simpler or more token-specific—and random models produce such features more reliably—this has very different implications than mere non-discrimination. The paper's own token entropy analysis (which shows random models learn simpler, token-specific features) actually supports this interpretation, but the connection is not made explicit.

- **No statistical uncertainty is reported for the central "no distinction" claim.** Only 100 latents are sampled per SAE for auto-interpretability (Section 3), with no confidence intervals, standard errors, or significance tests anywhere in the main text. With thousands of latents per SAE, 100 is a potentially unrepresentative sample. This matters especially for smaller models (e.g., Pythia-70m in Figure 2), where visible gaps between trained and random variants do exist in AUROC—the claim that metrics "do not distinguish" is a quantitative claim that requires quantitative support. Without variance estimates, it is impossible to assess whether the observed similarities are statistically meaningful or driven by sampling noise.

### Minor

- **SAE size is confounded with model size across the main experiments.** Larger Pythia models have wider residual streams, so with fixed R=64, the SAEs for larger models have more latents. The paper speculates that "features become more specific as SAE size increases" (Section 3), meaning the observed narrowing trained-vs-random gap could be driven by SAE width rather than any property of the underlying model. The expansion factor ablation on Pythia-160m (Figure 18) partially addresses this at one model size but does not resolve the cross-size confound. A simple ablation—training SAEs with the same latent count on different model sizes—would clarify this.

- **The "AUROC (Pruned)" label in Figure 2 is not defined in the main text.** The text discusses "fuzzing" and "detection" scores but Figure 2 uses "AUROC (Pruned)" and "AUROC (Detection)" as row labels. The relationship between "Pruned" and "fuzzing" is unclear from the main text alone, which affects interpretation of the central results.

- **The toy model section (Section 4) provides hypotheses but no mechanistic explanation for the main empirical findings.** The connection between superposition preservation/amplification in toy MLPs and the central finding—that auto-interpretability AUROC is similar or higher for random transformers—is tenuous, since the toy model measures sparsity/reconstruction tradeoffs, not auto-interpretability scores. The paper honestly defers this to future work (Section 4), but this leaves the core "why" question unanswered.

- **Only TopK SAEs are tested.** Standard ReLU SAEs, JumpReLU SAEs, and other architectures may behave differently. The paper states its SAE choice clearly (Section 3) but the generalizability of the findings across SAE architectures is unknown.

### Trivial
None.

## Nice-to-Haves

- Per-latent scatter plots of entropy vs. AUROC (referenced as Appendix H) could appear in the main text to make the argument more concrete.
- Testing additional SAE architectures (ReLU, JumpReLU, Gated) to establish architecture-generality.
- Cross-model SAE evaluation (train on trained-model activations, evaluate on random-model activations and vice versa) to distinguish "SAEs always learn equally interpretable features" from "the activation distributions are genuinely similar in decomposability."
- Disaggregated per-latent analysis of the AUROC reversal to reveal whether it is driven by a few high-scoring trivial features or is a systematic pattern.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"CE loss score should be explicitly discussed as a metric that passes the sanity check"** — The paper already discusses this explicitly at line 110: "the CE loss score only makes sense for the trained variant: for any of the randomized variants, the loss is very poor." The paper's logic is that CE loss score is meaningless for random models (they have terrible loss regardless), so it doesn't truly "pass" a sanity check in the same framework—it's trivially different for a trivial reason. This is a presentation preference, not a substantive gap.

- **"The Karvonen et al. contrast is underexplored"** — The paper does discuss the chess transformer finding and provides a clear hypothesis about why language data differs (sparsity of language vs. board games, Section 2). Demanding experimental tests of this hypothesis (e.g., training SAEs on chess transformers with random weights) is scope creep beyond the paper's stated contribution.

- **"Bricken et al. discrimination for one-layer transformers; depth not ruled out as variable"** — The paper notes the size dependency (Section 2, line 70). While depth could matter, this is speculative and asking for experiments to disentangle depth from size is beyond the paper's scope.

- **"Token entropy is underdeveloped and deserves more prominence"** — The paper presents it as "preliminary" and "not a direct measure of 'abstractness'" which is honest. While more analysis would strengthen the paper, the current treatment is reasonable for a proof-of-concept. Demanding full development of an alternative metric is scope creep.

- **"Toy model visual demonstration (Figure 3) is not compelling"** — The paper uses Figure 3 as a visual illustration of a concept, not as primary evidence. The Pareto frontier analysis (Figure 5) provides the quantitative support. Visual demonstrations are standard supporting material in toy model sections.

- **"GloVe experiment is single-seed and underpowered"** — The paper states this limitation explicitly (Section 4.3). Single-seed toy experiments are common in hypothesis-generating sections. This is not a core claim requiring rigorous statistical support.

- **"Missing cross-model SAE evaluation experiment"** — This would be an interesting experiment but is not necessary to support the paper's core claim that metrics fail to distinguish trained from random models. The paper demonstrates this already with its current experimental design.

## Novel Insights

The most underappreciated implication of this paper's results is that the AUROC reversal (random > trained) may reveal something structural about what auto-interpretability scoring actually measures. The paper's token entropy analysis suggests that random models learn simpler, token-specific features, and these features are *easier* for an LLM to explain and classify precisely because they are simple. This creates a perverse incentive structure: the metric rewards simplicity and specificity, which random models provide more reliably than trained models (whose features become more abstract and harder to classify). This is analogous to how simpler models can achieve higher accuracy on trivially decomposable tasks—the metric conflates "easy to explain" with "meaningful to explain." The paper hints at this but doesn't make the connection explicit.

## Suggestions

- Rewrite the framing to explicitly acknowledge the AUROC reversal (random > trained for large models) and discuss its implications: the metric doesn't just fail to distinguish, it actively rewards the wrong features. This is a stronger and more actionable finding than "metrics do not distinguish."
- Report confidence intervals or bootstrap distributions for AUROC scores, especially for the smaller models where gaps are visible. Even with 100 latents, bootstrap CIs are straightforward to compute and would transform the central claim from a visual judgment into a statistical finding.
- Define "AUROC (Pruned)" in the main text when Figure 2 is introduced, clarifying its relationship to the "fuzzing" scoring method.

## Score and Decision

**Calibration anchors:**

- **High-scoring (avg > 7):** `syThiTmWWm.md` (null models cheat LLM benchmarks, avg 7.75, Oral) — very similar null-model-breaks-evaluation theme, but that paper had cleaner execution and more thorough analysis. `PBjCTeDL6o.md` (gradient baselines harmful for interpretability, avg 8.0, Oral) — negative result for interpretability with a proposed fix. `5IWJBStfU7.md` (MI identifiability fails, avg 7.0, Poster) — fundamental negative result for MI. The paper under review has a similarly important finding but less thorough analysis than these.

- **Medium-scoring (4–6):** `vWRwdmA3wU.md` (similarity scores don't guarantee task-relevance, avg 6.25, Poster) — similar "metrics are misleading" theme but in brain-model comparison, less timely. `HpUs2EXjOl.md` (SAE metrics miss semantic quality, avg 5.75, Poster) — adjacent topic but less impactful finding. The paper under review is stronger than these due to its clean null-model design and more alarming finding.

- **Low-scoring (< 3):** `TJU9J8iQXL.md` (fairness metrics flawed, avg 2.33, Reject) — superficially similar "metrics are broken" theme but poorly executed and lacking the clean experimental methodology. `wJVZkUOUjh.md` (EXAGREE, avg 2.0, Reject) — weak contribution with no clear methodology. The paper under review is far above these.

The paper under review sits between the medium and high anchors. Its core finding is as important as the high-scoring papers, but the analysis depth (unexplored reversal, no statistical testing, SAE size confound) falls short. Compared to `syThiTmWWm` (7.75), this paper has a weaker mechanistic explanation and less thorough exploration of why the failure occurs. Compared to `5IWJBStfU7` (7.0), this paper has a more practical and actionable finding but similar limitations in depth. I place it at 6.5 — a paper with an important finding that the community needs to hear, held back by insufficient analysis of the reversal and lack of statistical rigor.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>