Now I have sufficient information to write the final review. Let me synthesize everything.

## Summary

This paper applies a sanity-check methodology (à la Adebayo et al. 2020) to sparse autoencoder (SAE) evaluation: it asks whether commonly used SAE quality metrics can distinguish features from trained transformers versus randomly initialized ones. Over Pythia models from 70M to 6.9B parameters and four randomization schemes, the authors find that aggregate auto-interpretability (fuzzing AUROC) scores are surprisingly similar between trained and randomized models, with the gap narrowing for larger models. The paper introduces token distribution entropy as a novel metric that does capture the trained/randomized distinction—features from trained models become more "abstract" (higher entropy) in deeper layers, while randomized model features remain token-specific.

## Strengths

- **Excellent and overdue sanity-check experimental design.** Directly applying the Adebayo et al. (2020) framework to SAE metrics is a natural but critically important contribution. The finding that fuzzing AUROC curves for trained (gray) and randomized (colored) variants substantially overlap (Figures 1–2), while only the Gaussian control (black) falls to chance, is a striking and actionable result for the field.

- **Multiple randomization variants provide useful discriminative granularity.** The four schemes—Re-randomized incl./excl. embeddings, Step-0, and Gaussian control—allow the reader to assess how much of the similarity is driven by different aspects of model structure. The Control producing chance-level scores validates that the auto-interpretability pipeline can reject truly unstructured data, making the overlap for the other variants more meaningful.

- **Token distribution entropy successfully reveals a dimension that aggregate AUROC misses.** The last row of Figure 2 shows that entropy increases across layers for the trained variant (reflecting more abstract features) but remains low for randomized variants (single-token features). This is a genuine and insightful finding—it shows that the metrics fail to capture a qualitatively important distinction that does exist.

- **Results hold across a wide range of model scales and SAE hyperparameters.** Figure 2 reports results for Pythia-70m through 6.9b, and Figure 18 confirms robustness to expansion factors (16–128) and sparsities (16, 32). Appendix C confirms similar results with SAEs trained on 1B tokens.

- **The paper is forthright in its limitations.** Section 5 states: "we do not claim that SAEs fail to capture information from trained Transformers above and beyond randomly initialized transformers; only that aggregate auto-interpretability measures do not necessarily indicate the existence of interesting underlying features." This careful framing is commendable.

## Weaknesses

### Fatal
None.

### Major

- **The title and abstract overgeneralize from auto-interpretability failure to all SAE metrics, while the paper's own evidence shows the failure is concentrated in auto-interpretability.** The title appropriately says "Automated Interpretability Metrics," but the abstract claims "SAEs trained on randomly initialized transformers produce auto-interpretability scores **and reconstruction metrics** that are similar to those from trained models" and recommends "treating common SAE metrics as useful but insufficient proxies." However, the paper's own evaluation section (Section 3) notes that: (1) cosine similarity and explained variance "are often far lower for the random control than the other models," and reconstruction errors "increase across layers" for the control while "the remaining variants decrease"; (2) CE loss score "only makes sense for the trained variant"—meaning it intrinsically discriminates. The paper's evidence shows that reconstruction-based metrics carry discriminative signal for the Gaussian control, and CE loss score is outright inapplicable to random models. The correct recommendation is narrower than stated: aggregate *auto-interpretability scores* are insufficient, while reconstruction metrics retain signal. This matters because the broader framing dilutes the paper's genuine and important finding about auto-interpretability specifically.

- **The model-size dependence of the core finding is acknowledged but not reflected in the title or conclusions.** The paper explicitly states (Section 2): "we found that auto-interpretability scores for randomized models were relatively low for smaller models (e.g., Pythia-70m) but that the gap was narrowed for larger models (e.g., Pythia-6.9b)." This means the categorical title claim "Do Not Distinguish" is false for Pythia-70m by the paper's own evidence—the metrics *do* distinguish at smaller scales. The paper does not investigate *why* this scaling occurs, nor moderate its conclusions accordingly. A more accurate title would reference the scaling-dependent nature of the failure, and the conclusions should explicitly scope the claim to larger models.

### Minor

- **Results are not disaggregated by randomization variant for the headline AUROC finding.** Figure 1 shows four colored lines that "overlap" with the trained model, but the reader cannot assess which randomization scheme contributes most to the overlap. The "Re-randomized excl. embeddings" variant preserves trained embeddings and unembeddings—the very matrices that encode token identity—so high auto-interpretability scores for this variant are less surprising. The paper's strongest tests are "Step-0" and "Re-randomized incl. embeddings" (which randomize embeddings), and it would significantly strengthen the paper to report whether these variants alone still produce the overlapping AUROC curves. The paper notes that "the randomized variants (blue and orange lines) are more similar to the trained model than the variant at initialization (green line)" for L1 norm values, suggesting the variants do differ, but this analysis is not extended to AUROC.

- **The toy model section (Section 4) is underpowered and inconclusive, occupying ~25% of the paper without advancing the core argument.** Section 4.1 shows linear maps preserve superposition, which is straightforward. Section 4.2 finds that random MLPs make superposed and Gaussian inputs more similar in their sparsity profiles (Figure 5a)—this actually suggests random networks *reduce* the distinguishability of structured vs. unstructured inputs, rather than amplifying superposition. Section 4.3 uses a single random seed and a fixed dataset size. The section concludes with "we defer conclusions... to future work," raising the question of whether it warrants its current prominence.

- **The entropy metric, while promising, is only a proxy for "abstractness" and its relationship to computational relevance is not demonstrated.** A latent activating on many synonymous tokens would have high entropy without being computationally "abstract" in a mechanistically relevant sense. The paper acknowledges this ("While the token distribution entropy is not a direct measure of 'abstractness'"), but does not test whether high-entropy features are actually causally implicated in model computations (e.g., via activation patching or steering).

### Trivial
None.

## Nice-to-Haves

- Correlate per-latent fuzzing AUROC with token-distribution entropy to test whether the auto-interpretability pipeline systematically rewards token-specific (low-entropy) features—if so, the problem lies in the evaluation pipeline's reward structure, not SAEs per se.
- Investigate why the trained-random gap closes at larger model sizes—does the gap close because random-model features get better scores, or because trained-model features get worse?
- Provide a small qualitative comparison table (top-5 features from trained vs. random models with explanations and activating examples) in the main text to make the failure mode concrete.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"CE loss score only making sense for trained variant is itself a strong distinguishing signal that the paper dismisses too quickly."** (Harsh Critic) — The paper does NOT dismiss this; it explicitly states "Importantly, the CE loss score only makes sense for the trained variant" (Section 3). The paper uses this as a genuine distinguishing feature. The critic misrepresented the paper's treatment.

- **"Section 4.2 contradicts the amplification hypothesis."** (Harsh Critic) — The paper itself does not commit to the amplification hypothesis; it presents both hypotheses (preservation vs. amplification) as speculative and explicitly "defer[s] conclusions as to the mechanism responsible to future work." The critic is treating a speculative section as making claims the paper does not make.

- **"Statistical significance and variance not reported."** (Harsh Critic) — This is a generic reproducibility nitpick. For a paper reporting 100 sampled latents per SAE across multiple model sizes, layers, and randomization schemes, single-run evaluation is standard in this field. Demanding confidence intervals for this scale of benchmarking is not the community norm.

- **"Missing related works."** (Harsh Critic) — Per instructions, I do not flag missing related works as I cannot confirm their existence.

- **"Evaluation on a metric that directly measures computational relevance (activation patching/steering)."** (Harsh Critic) — This demands the paper address a problem outside its stated scope. The paper is about evaluating *current* SAE metrics, not proposing a complete solution. The entropy analysis is presented as a "proof-of-concept" for better metrics.

- **Strength claim: "Robustness to SAE hyperparameters and training data scale"** — This is a valid strength but is partially redundant with other stated strengths. Kept in supporting evidence for the main strengths.

## Novel Insights

The paper identifies an important asymmetry in how SAE metrics fail: aggregate auto-interpretability scores collapse trained and random models together, while token distribution entropy reveals that these models produce qualitatively different features (abstract vs. token-specific). This suggests that the auto-interpretability pipeline may systematically reward token-specific features—features that are easiest to explain with simple textual descriptions—over abstract features that may be more computationally relevant. The model-size scaling of the failure (worse at larger scales) is particularly concerning for the field's trajectory toward larger models.

## Suggestions

- Revise the abstract to clearly scope the reconstruction-metric claim: specify that the similarity finding applies primarily to auto-interpretability scores, while reconstruction metrics distinguish the Gaussian control and CE loss score is inapplicable to random models.
- Add a subtitle or qualifying phrase to the title reflecting model-size dependence, e.g., "Automated Interpretability Metrics Fail to Distinguish Trained and Random Transformers at Scale."
- Report AUROC curves separately for the strongest randomization tests ("Step-0" and "Re-randomized incl. embeddings") to show the headline result holds even when embeddings are randomized.
- Consider trimming the toy model section or restructuring it as an appendix, since its current contribution is speculative and inconclusive.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| `/home/wg25r/review_agent/human_reviews_2026/qyVzZsrsnS.md` (Model diffing / ADL) | 7.5 | Stronger than paper under review—clearer methodology, more complete evaluation, no overclaiming |
| `/home/wg25r/review_agent/human_reviews_2026/Ml8t8kQMUP.md` (SAE causal effects) | 7.0 | Stronger—proposes and validates a method, not just identifies a failure |
| `/home/wg25r/review_agent/human_reviews_2026/HBcgLe6NZD.md` (GNN explanations failure) | 6.0 | Comparable—exposes evaluation metric failure, proposes fix; similar contribution profile |
| `/home/wg25r/review_agent/human_reviews_2026/EjInprGpk9.md` (SAE feature instability) | 5.5 | Slightly weaker—identifies instability but less actionable |
| `/home/wg25r/review_agent/human_reviews_2026/Q4ooLNOFeR.md` (SAE interpretability vs utility) | 4.5 | Weaker—identifies gap but with narrower scope and more contested methodology |
| `/home/wg25r/review_agent/human_reviews_2026/XPm8t1J1g7.md` (Random masking vs saliency) | 4.0 | Weaker—finding not novel (ROAR), just a new framework |
| `/home/wg25r/review_agent/human_reviews_2026/tWe5owhOyU.md` (SALVE) | 2.0 | Much weaker—no real evaluation, purely qualitative |

The paper under review sits between the GNN explanations failure paper (6.0) and the model diffing paper (7.5). Its core finding is more impactful than the GNN paper because it identifies a fundamental evaluation failure in a widely-used tool (SAEs), with cleaner experimental design. However, the overgeneralization in the abstract and the lack of model-size-dependent qualification in the title hold it back from the 7+ range. The entropy metric is promising but preliminary. Overall, this is a solid, important contribution with meaningful but addressable weaknesses.

**Originality:** High — the sanity-check framing for SAE metrics is novel and timely.
**Research question importance:** High — directly impacts how the field evaluates SAE quality.
**Claims support:** Moderate — the auto-interpretability claim is well-supported; the broader reconstruction-metrics claim is overgeneralized; the title is too categorical given model-size dependence.
**Experiment soundness:** Good — multiple model sizes, randomization variants, hyperparameters; but lacking per-variant disaggregation.
**Clarity:** Good — well-written with careful limitations section.
**Community value:** High — the finding will change how researchers interpret auto-interpretability scores.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>