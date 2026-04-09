## Summary

This paper applies sanity-check methodology from saliency map validation (Adebayo et al., 2020) to sparse autoencoder (SAE) evaluation in mechanistic interpretability. The authors train SAEs on both trained and randomly initialized transformers (Pythia, 70M–6.9B) and find that aggregate auto-interpretability scores (fuzzing/detection AUROC) and reconstruction metrics are surprisingly similar across both settings for larger models. They propose token distribution entropy as a diagnostic for feature "abstractness" and recommend routine randomized baselines.

## Strengths

- **Rigorous null model design with multiple randomization variants.** The paper goes beyond a single random baseline by including Step-0 initialization, re-randomization with/without embeddings, and a Gaussian control. The control variant consistently achieves ~0.50 AUROC (Figures 1, 6–14), validating that the auto-interpretability pipeline is not trivially gullible—only when structured activations exist (even from random weights) does it produce high scores. This multi-variant design substantially strengthens the causal attribution of the failure to the metrics rather than the pipeline.

- **Compelling qualitative evidence that closes the intuition gap.** Appendix J/L provides randomly sampled feature dashboards with generated explanations and activating examples. The Control features (Appendix L.2) produce vague, nonsensical explanations ("various tokens including articles, conjunctions..."), while randomized variants (L.3/L.4) yield superficially coherent single-token explanations ("the token 'record' often refers to..."). This makes the abstract quantitative finding concrete and inspectable—a practice most papers in this area neglect.

- **The core empirical finding addresses a genuine validation gap.** The observation that auto-interpretability AUROC for Pythia-6.9b trained (0.79) overlaps with randomized (0.87–0.88) in Figure 1 is striking. The field has been relying on these metrics to justify SAE quality claims; demonstrating their insensitivity to whether the underlying model has learned anything is an important negative result.

## Weaknesses

### Major:

- **Section 4's toy models explain SAE trainability, not auto-interpretability scores—a mechanistic gap.** The toy models (Section 4) demonstrate that random networks preserve or amplify superposition, which explains why SAEs can achieve low reconstruction error on random networks. However, the paper's headline claim is about auto-interpretability scores—an LLM-based measure of feature explainability. The leap from "random networks produce sparse, reconstructible activations" to "LLMs can generate plausible explanations for those activations" is not established. The more direct explanation for high auto-interpretability scores is the "single-token feature" hypothesis (Section 3, "Latent explanation complexity"), which is supported by the token entropy analysis but is not connected to the superposition framework. These two narratives (superposition preservation → SAE trainability vs. simple token-specific features → LLM explainability) need synthesis, or the paper should acknowledge that Section 4 addresses a different question than the one the title raises.

- **The scaling transition from distinguishable (70M) to indistinguishable (6.9B) is the most consequential finding but receives no dedicated analysis.** Figure 6 (Pythia-70M) shows trained AUC ≈ 0.63 vs. randomized ≈ 0.50—a visible gap. Figure 1 (Pythia-6.9B) shows near-complete overlap. This transition is arguably the paper's most important empirical result: it tells us *where* the metrics break down. Yet there is no single summary visualization (e.g., AUROC gap vs. parameter count or model size) and no substantive hypothesis for *why* larger random models better mimic the metrics. The authors speculate that "features become more specific as SAE size increases" (Section 3), but this is counter-intuitive (larger models should learn more abstract features) and under-developed. A dedicated scaling analysis would significantly strengthen the paper.

- **The claim about "computationally relevant features" is asserted but not functionally validated.** The paper's central conclusion is that "high aggregate auto-interpretability scores do not, by themselves, guarantee that learned, computationally relevant features have been recovered." The term "computationally relevant" implies features that causally influence model behavior. While the qualitative evidence (Appendix J) strongly suggests trained features are more abstract, no causal intervention (e.g., steering, activation patching) tests whether trained features actually affect downstream behavior while random features do not. The authors note that CE loss score (Figure 2, row 5) is only meaningful for trained models, which is indirect evidence, but a direct ablation—showing that intervening on a high-AUROC random feature has no behavioral effect—would transform this from a suggestive finding to a definitive one. The paper explicitly scopes itself to evaluation metrics rather than functional validation, so this is not a fatal omission, but it leaves the strongest version of the claim unproven.

### Minor:

- **Statistical power of 100-feature sampling for large SAEs.** For Pythia-6.9b with expansion factor R=64, the SAE latent space is large (potentially hundreds of thousands of latents). Sampling 100 features represents <0.04% of the dictionary. Appendix E shows variance across 5 training seeds for Pythia-70M, but does not isolate variance due to feature sampling specifically. If the distribution of interpretable features is heavy-tailed, the aggregate AUROC could be noisy. This concern is partially mitigated by the consistency of results across layers and model sizes, but a brief analysis of sampling variance (or a note on why 100 is sufficient) would strengthen the quantitative claims.

- **Token distribution entropy conflates feature simplicity with feature type.** The entropy metric (Section 3) successfully distinguishes trained from random features in aggregate, but it can penalize legitimately specific learned features. For example, a trained feature that selectively fires on a single technical term (e.g., a specific gene name) would have low entropy despite being a genuinely learned feature. The paper acknowledges this ("the token distribution entropy is not a direct measure of 'abstractness'"), but does not quantify the false positive rate. A brief discussion of when this metric would mislead would be valuable for practitioners considering its adoption.

- **Title overstates the result for small models where discrimination succeeds.** The title claims automated interpretability metrics "DO NOT DISTINGUISH" trained and random transformers. However, for Pythia-70M (Figure 6), the gap is meaningful (trained 0.63 vs. randomized ~0.50). The failure is scale-dependent. A more precise title reflecting this (e.g., "...Are Insufficient for Large Transformers") would better represent the contribution.

### Trivial:

- Figure 2 is a multi-panel figure where rows are referenced by number in the text but not labeled in the caption, making navigation slightly inconvenient.

## Nice-to-Haves

- **Causal intervention ablation:** Even a small-scale experiment (e.g., on Pythia-70M) showing that steering with trained SAE features modifies behavior while steering with random SAE features does not would provide the strongest possible validation of the "computationally relevant" claim.
- **Single summary plot of AUROC gap vs. model size:** A plot showing how the trained-vs-randomized AUROC gap shrinks with scale would make the scaling finding immediately visible and citable.
- **Cross-architecture validation on one non-Pythia model** (e.g., Gemma-2 or Llama-3) to assess generalizability beyond the Pythia family.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Explainer LLM sycophancy/hallucination concern** (Harsh Critic, Spark Finder): The Control variant consistently achieves AUROC ≈ 0.50 (chance), demonstrating that the 70B explainer does *not* find plausible explanations in pure noise. This directly addresses the concern—the explainer is not hallucinating coherence where none exists; it is correctly identifying structure in activations that happen to arise from random weights processing structured inputs.

- **Missing SAE architecture ablation** (transferred from WCRQFlji2q.md): The paper tests expansion factors 16–128 and sparsity k=16,32 (Appendix F), and uses TopK SAEs, a current standard. Demanding tests across all SAE architectures (Gated, JumpReLU, etc.) is scope creep for a paper focused on evaluation metrics rather than SAE design.

- **Computational cost of randomized baselines as a limitation** (Harsh Critic): This is a practical concern about a recommendation, not a flaw in the experimental methodology. The Step-0 variant is available for Pythia at no additional training cost, and the re-randomization procedure is a one-time operation.

- **Embedding-only SAE baseline** (Spark Finder): The "Re-randomized excl. embeddings" variant already isolates the role of embeddings by preserving them while randomizing all other weights. Training SAEs on raw embeddings is a different experiment that would not address the paper's question about whether transformer processing (even random) produces interpretable features.

- **Missing related work citations** (transferred weakness): Cannot verify existence of specific references; removed per hard rules.

- **Formatting/style nitpicks** (OCR artifacts in references, figure caption numbering): Removed per hard rules.

## Novel Insights

The scaling dimension of this result is underappreciated even by the reviewers: the transition from distinguishable (70M) to indistinguishable (6.9B) suggests that larger random matrices are increasingly effective at preserving input data structure through their transformations. This is consistent with random matrix theory (random projections approximately preserve distances in high dimensions via the Johnson-Lindenstrauss lemma), but the implication for interpretability is novel—*the very mathematical property that makes large random networks useful for dimensionality preservation also makes them dangerous as interpretability nulls*, because they produce activations that are superficially structured enough to be "explained" by an LLM. This reframes the problem: the issue is not that auto-interpretability metrics are broken per se (they correctly reject pure noise), but that they lack a notion of *computational depth*—they cannot distinguish "the network preserved this input structure" from "the network learned to compute this structure."

## Suggestions

- Add a single plot showing the AUROC gap (trained − randomized) as a function of model size, potentially faceted by layer, to make the scaling finding immediately visible.
- In Section 4, explicitly acknowledge the gap between the superposition/sparsity explanation (why SAEs train well on random nets) and the auto-interpretability result (why LLMs can explain the resulting features), and clarify that the token-entropy analysis addresses the latter while the toy models address the former.
- Consider running steering/patching on a small model (e.g., Pythia-70M) with trained vs. random SAE features to provide direct functional validation of the "computationally relevant" claim, even if only as a preliminary experiment.

---

**Axis evaluations:**

- **Novelty:** High. The systematic application of randomized baselines to auto-interpretability evaluation is novel and addresses a real validation gap. The scaling finding is particularly new.

- **Technical soundness:** Moderate-to-good. The experimental design is strong (multiple null variants, control, robustness checks). The main gap is the disconnect between the toy model mechanism (sparsity) and the headline result (interpretability), and the lack of functional validation for the "computationally relevant" claim.

- **Empirical support:** Good for the core negative result (metrics fail to distinguish at scale). Weaker for the proposed solution (token entropy) and for the mechanistic explanation. The scaling trend is well-documented but under-analyzed.

- **Significance:** High. If the field has been relying on auto-interpretability scores as evidence of meaningful feature discovery, this paper's demonstration that these scores are largely insensitive to training is a foundational challenge. The practical implications for how SAE research is evaluated are substantial.

- **Clarity:** Good. The paper is well-organized and the appendices provide valuable qualitative evidence. The multi-panel Figure 2 could benefit from row labels in the caption.