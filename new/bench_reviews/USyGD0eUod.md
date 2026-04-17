Now I have sufficient context from calibration papers. Let me write the final review.

## Summary

This paper investigates whether commonly used sparse autoencoder (SAE) quality metrics — including auto-interpretability scores (fuzzing/detection AUROC), reconstruction metrics (cosine similarity, explained variance, L1), and CE loss recovery — can distinguish trained transformers from randomly initialized ones. Across multiple Pythia model sizes (70M–6.9B) and randomization schemes, the authors find that these aggregate metrics are often surprisingly similar between trained and random models, particularly for larger models. The paper also introduces token distribution entropy as a proof-of-concept measure of feature "abstractness" that does differentiate trained and random models, and presents toy model experiments suggesting random networks preserve or amplify superposition in their inputs.

## Strengths

- **Timely and important question.** The paper applies a sanity-check paradigm (analogous to Adebayo et al., 2020 for saliency maps) to SAE evaluation — a rapidly growing sub-field that currently lacks strong null-model baselines. This is exactly the kind of validation the mechanistic interpretability community needs.

- **Well-designed experimental setup with multiple controls.** The five variant comparison (trained, step-0, re-randomized incl./excl. embeddings, and Gaussian control) is thoughtfully constructed and allows isolating the contribution of training vs. architecture vs. embedding structure. Testing across the full Pythia suite (70M–6.9B) and across layers adds breadth.

- **Important finding that the trained–random gap narrows with model scale.** The observation that smaller models (Pythia-70m) show clearer separation between trained and random, while larger models (Pythia-6.9b) show substantial overlap, is a new and concerning finding.

- **Token distribution entropy as a constructive contribution.** The entropy-based measure distinguishing "abstract" (high-entropy, deep-layer) features in trained models from "simple" (low-entropy, token-specific) features in random models provides a useful diagnostic beyond aggregate AUROC. The depth trend — entropy increasing with layer depth for trained models but remaining flat for randomized models — is informative and consistent with intuition.

- **Honest framing in key places.** The paper explicitly acknowledges limitations: "we do not claim that SAEs fail to capture information from trained Transformers above and beyond randomly initialized transformers; only that aggregate auto-interpretability measures do not necessarily indicate the existence of interesting underlying features" (Section 5). The toy model section is presented speculatively rather than as definitive mechanistic explanation.

## Weaknesses

### Major:

- **Quantitative gap between claims and evidence.** The paper's title and core claim state that metrics "do not distinguish" trained from random transformers, but the evidence relies primarily on visual overlap of curves (Figures 1–2) without quantitative effect sizes, confidence intervals across seeds, or formal discriminability analysis. As acknowledged in Appendix E, multiple seeds show variance, yet the main figures present single runs. The paper would be substantially strengthened by reporting: (a) mean differences and standard errors between trained and random across seeds, (b) an explicit discriminability test (e.g., can one classify a model as trained vs. random from its SAE metrics?), and (c) per-feature AUROC distributions, not just aggregates. Without this, "do not distinguish" overstates what the data show — they arguably show "insufficiently distinguish, especially at scale."

- **Limited model diversity restricts generality.** All experiments use only the Pythia suite. The paper acknowledges this limitation but the conclusions are stated broadly. Pythia has specific architectural choices (parallel attention/MLP, particular training regimes) that could interact with the findings. Even one experiment on a different architecture (e.g., GPT-2, a non-parallel-attention model) would substantially increase confidence. This is particularly important given the scale-dependent findings — whether the gap closure at larger scales generalizes to other architectures is an open and important question.

- **Fuzzing AUROC as the primary metric without robustness checks.** The paper relies almost entirely on fuzzing AUROC as its autointerpretability metric, noting that simulation scoring is too expensive. But it does not validate whether fuzzing and simulation scoring behave similarly in the random-model setting (they were only shown to correlate in the trained setting by Paulo et al. 2024). It is plausible that fuzzing AUROC, which tests whether an explanation can classify activating vs. non-activating tokens, is particularly susceptible to simple token-frequency features — exactly the kind of features that random models produce. Testing at least one additional metric (e.g., simulation scoring on a subset of features, or detection scoring more thoroughly) would clarify whether the finding is about auto-interpretability metrics broadly or about fuzzing specifically.

### Minor:

- **The Gaussian embedding control is extreme but rarely contextualized.** The control condition (i.i.d. Gaussian per token occurrence) deliberately destroys token identity, making it nearly guaranteed to score at chance level. While this establishes a useful floor, the paper's framing sometimes implies that the "similarity" of trained and random to each other (compared to this control) is the key finding, when the more meaningful comparison is directly between trained and random. The paper does include step-0 and re-randomized variants, making this a minor framing issue rather than a methodological one.

- **Toy models (Section 4) remain speculative.** The linear algebra argument and MLP experiments provide intuition but are not quantitatively connected to the main transformer results. The Pareto frontier analysis shows random MLPs "sparsify" inputs, but this does not directly explain fuzzing AUROC scores. The paper appropriately defers conclusions, but Section 4 reads more as a research direction than as explanatory support.

- **Token entropy is a useful but crude proxy for abstractness.** The entropy of latent activations over token IDs conflates several notions — a feature activating on many syntactically-related tokens could have high entropy but low "abstractness." The paper acknowledges this, but the depth-trend finding is presented without validation against any ground-truth or qualitative analysis of whether high-entropy features from trained models are genuinely more "abstract."

## Nice-to-Haves

- Per-feature AUROC histograms comparing trained vs. random (revealing whether a subset of high-scoring features in trained models drives the aggregate).
- Activation steering or ablation experiments on features from both trained and random models, to test whether causal efficacy differs.
- Results on at least one non-Pythia architecture.
- Correlation analysis between fuzzing AUROC and token entropy, to test whether entropy-based composite metrics improve discriminability.
- Results at intermediate training checkpoints (e.g., Pythia step-100, step-1000) to establish whether the metric failure is specific to the trained/random binary or reflects a deeper insensitivity.

## Removed Points

- **Overclaiming that the paper asserts metrics "completely fail" when the paper actually uses hedged language.** The harsh critic's framing is somewhat overstated — the paper's abstract says "in many settings" and the conclusion says "can be surprisingly similar," both appropriately qualified. The title "do not distinguish" is strong but is tempered by the body text.

- **Demand for behavioral/causal evaluation as a prerequisite.** The spark reviewer suggests causal steering experiments as missing, but this goes beyond the paper's stated scope. The paper's claim is specifically that *aggregate auto-interpretability metrics* are insufficient — it does not claim to solve the problem. Requiring a full causal evaluation framework is scope creep.

- **Demand for a formal normative definition of what metrics should approximate.** While the paper would be stronger with a clear formalization, sanity-check papers in the interpretability literature (including Adebayo et al., 2020) generally do not provide formal definitions of what metrics should measure. The paper's framing — treating random networks as a null model analogous to saliency map sanity checks — is a reasonable and established approach.

- **Criticisms about the "extreme" control being misleading.** The paper's main comparison is trained vs. randomized variants (step-0, re-randomized), not trained vs. control. The control is used to establish a floor. Removing or downplaying it would not change the core finding.

- **Formatting and writing nitpicks** from reviewers.

- **Reproducibility concerns** about number of seeds, hyperparameters, etc. The paper reports results across multiple model sizes, layers, and randomization schemes, and Appendix E reports multiple seeds. The implementation is based on publicly available frameworks (Sparsify, Delphi).

## Novel Insights

The finding that aggregate auto-interpretability metrics are increasingly similar between trained and random transformers as model scale grows is a genuinely concerning result for the mechanistic interpretability community. It suggests that scaling up SAE evaluations alone will not solve the problem of distinguishing meaningful learned features from architectural artifacts, and that the community needs metrics that specifically capture feature abstractness or causal efficacy — qualities that scale-dependent aggregate scores can mask.

## Suggestions

- Provide quantitative discriminability analysis: mean AUROC differences, confidence intervals across seeds, and ideally a simple classifier that attempts to distinguish trained from random using SAE metric vectors.
- Soften the title and abstract to say "insufficiently distinguish" or "fail to reliably distinguish" rather than "do not distinguish," which is more strongly supported by the data.
- Test the core finding on at least one additional model architecture to assess generality.
- Validate whether the fuzzing AUROC findings hold for simulation-based scoring, even on a small subset of features.
- Correlate token entropy with AUROC within each model variant to test whether combining these measures improves discriminability.

## Score and Decision

**Calibration comparison:**

- *Interpretability Illusions in Simplified Models* (v675Iyu0ta) — similar theme of "interpretability methods can be misleading," but on toy Dyck language data only. Received scores of 3,6,6,8,5 (mean ~5.6), decision: Reject. The current paper has stronger empirical scope (real LLMs, multiple sizes) but similar limitations in generality and overclaiming.

- *A Principled Evaluation Framework for Neuron Explanations* (todLTYB1I7) — shows evaluation metrics fail sanity checks. Received scores 5,6,6,3,5,5 (mean ~5), decision: Reject. That paper had a more formal framework but less practical impact; the current paper has more practical impact but less formal rigor.

- *Don't trust your eyes* (OZWHYyfPwY) — questions reliability of feature visualizations. Received scores 8,6,6,8 (mean ~7), decision: Reject (but at a more competitive venue). That paper had adversarial methodology and theoretical grounding; the current paper has less rigorous methodology but arguably more practical import.

- *Correcting Flaws in Common Disentanglement Metrics* (hv8l922Ad7) — identifies flaws in metrics and proposes fixes. Received scores 3,3,5,3,3 (mean ~3.4), decision: Reject. The current paper is substantially stronger than this, as it targets a more impactful problem and has much more extensive experiments.

The current paper identifies a genuinely important problem and provides substantial empirical evidence, but the evidence does not firmly support the claim in its title. The lack of quantitative analysis, single model family, reliance on one auto-interpretability metric, and speculative toy model section are significant weaknesses. However, the core finding is real, important, and timely. This is a solid contribution that would be significantly improved with quantitative rigor and broader evaluation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>