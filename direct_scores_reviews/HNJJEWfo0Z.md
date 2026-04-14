## Summary

This paper investigates statistical inconsistencies in perturbation-based fidelity metrics for saliency maps by formally decomposing the underlying assumptions into two testable points: [P1] perturbing a pixel should cause a probability drop, and [P2] the pixel importance ranks induced by different perturbation types should be consistent. The authors propose two conformity measures, DROP and PSim, to quantify violations of these assumptions, and conduct an empirical study spanning 9 perturbation types, 5 models (including adversarially trained variants), and 3 datasets. The consistent finding is that both assumptions are violated in practice, raising questions about the reliability of widely used fidelity metrics such as AOPC and AD%.

---

## Strengths

- **Formal decomposition of fidelity metric failure modes**: The paper clearly separates the inconsistency problem into two distinct assumptions ([P1] and [P2]) and builds dedicated metrics (DROP, PSim) that map directly onto each assumption. This decomposition is more principled than Tomsett et al. (2020), which observed inconsistency empirically without isolating which assumption was responsible.
- **Breadth of empirical coverage**: The study covers 9 perturbation types (including two inpainting methods, three Gaussian blur widths, and four value-replacement strategies), 5 models, and 3 datasets, with ~75 million model forward passes. Results are consistent across all configurations, providing robust evidence that the DROP ≈ 0.5 and PSim ≈ 0.3–0.6 findings are not artifacts of a specific setup.
- **Actionable observation on Gaussian blur**: The consistent finding that Gaussian blur achieves markedly higher DROP and PSim scores across datasets and models (Figure 2, 3, 4, 5) is practically useful, giving practitioners a concrete—if not yet theoretically explained—basis for preferring blur-based perturbations over value-replacement approaches.
- **Adversarial model analysis as a negative result**: The finding that adversarially trained models (ResNet50 L2 and Linf) still exhibit low DROP and PSim scores (Table 2) is a genuine negative result that refutes a plausible prior hypothesis that robust training would regularize pixel-importance behavior.

---

## Weaknesses

- **Conceptual gap between DROP and actual fidelity metric operation**: Fidelity metrics such as AOPC and AD% perturb the *top-K most salient pixels* in cumulative fashion and measure the aggregate probability change. DROP, by contrast, measures whether *randomly sampled individual pixels*, perturbed in isolation, cause any probability drop. These are not equivalent conditions. Specifically, it is expected—and benign—that a randomly chosen, individually perturbed pixel often does not cause a probability drop (due to softmax normalization, feature redundancy, and local receptive-field structure in CNNs), without this implying that fidelity metrics give wrong saliency rankings. The paper never formally bridges the gap from "single random pixel DROP ≈ 0.5" to "AOPC rankings of saliency methods are unreliable." This is the most substantive weakness: the conformity measure does not directly test the condition that fidelity metrics actually rely on.

- **Absence of downstream validation**: No experiment links low DROP or PSim scores to actual discordance in saliency method rankings. If two saliency methods (e.g., GradCAM vs. LIME) are ranked the same by AOPC regardless of perturbation type, then the low conformity scores, while interesting, do not constitute a practical failure. Without demonstrating that low DROP/PSim predicts ranking disagreements, the proposed measures' utility as "pre-conditional checks" remains unsubstantiated.

- **No explanation for the Gaussian blur advantage**: Section 5.3 concludes that Gaussian blur exhibits relatively higher conformity but provides no mechanism. The likely explanation—that blur preserves local spatial context and produces less out-of-distribution (OOD) inputs than value-replacement perturbations—is absent. Without this, the recommendation to prefer Gaussian blur is an empirical heuristic rather than a principled guideline, and practitioners are left without guidance on *why* or *when* to apply it.

- **No concrete actionable threshold or protocol**: The paper recommends running DROP/PSim as a "pre-conditional check" before reporting fidelity scores, but provides no pass/fail threshold for acceptable conformity. Without operationalizable criteria, the recommendation is difficult to adopt in practice.

- **Typographic errors that affect interpretability**: (a) Equation (9) writes `PSim = (1/|K|) Σ PSim`, which is circular—the summand is evidently intended to be `PSim_M^k` but is written as `PSim`, making the equation formally undefined. (b) The final sentence of Section 5.3 reads "indicating low conformity to Point [P1]," but Section 5.3 analyzes PSim, which corresponds to Point [P2]. These errors are minor individually but collectively suggest imprecise formulation.

- **Minor factual error regarding VOC 2007**: The paper states that "PASCAL VOC 2007 did not have train and test splits" and therefore uses all 4,952 images. In fact, VOC 2007 provides standard trainval and test splits. Because no model is trained on this data, the practical impact is minimal, but the stated justification is incorrect.

---

## Nice-to-Haves

- A correlation analysis between DROP/PSim scores and variance in AOPC/AD% rankings across perturbation types would substantially strengthen the motivation for using these measures as diagnostic tools.
- An ablation on perturbation magnitude (e.g., varying ε for FR perturbation) would help disentangle whether inconsistency is driven by the perturbation *type* or the perturbation *strength*.
- A mechanistic analysis explaining why Gaussian blur yields higher conformity (e.g., OOD-ness of the perturbed image as measured by feature distribution distance) would elevate the empirical heuristic into a principled recommendation.
- Visualization of per-image DROP scores overlaid spatially on input images could reveal whether inconsistency is localized to texture-heavy regions or background areas, connecting the conformity measures to the structure of the image and the saliency map.
- A discussion of computational feasibility: running DROP/PSim as described requires roughly 15 million forward passes per model, which is substantial. Even an approximate protocol (e.g., fewer random pixels, single perturbation pair for PSim) would make the workflow more practical.

---

## Removed Points

*These points are flagged to be removed—treat them with caution.*

- **"Contribution list internally redundant"** (Harsh Critic): This is a style and presentation nitpick. The three bullets have different emphases (framework, measures, empirical scope) and do not constitute a factual or methodological flaw.
- **"Assumption invariance in Eq. 5 is too strong"** (Harsh Critic): The requirement that `rbo(R(φ), R(ψ)) ≈ 1` across perturbation types is, in fact, *exactly what the paper tests*. The paper's point is that existing fidelity practice implicitly assumes perturbation-invariance of PIR; the critic's objection confuses the paper's criterion with a normative prescription.
- **"Adversarial model results should not be in a full table since authors refrain from conclusions"** (Harsh Critic): Including a full result table while being appropriately cautious in interpretation is standard practice; the data is informative even without strong conclusions. Not a flaw.
- **"50-pixel subsampling is unreliable"** (Harsh Critic): The paper cites Appendix S2 for a proof that a subset of a ranked list preserves the ranking structure. This is a reasonable justification; the concern about sparse sampling is not unreasonable but the paper has explicitly addressed it and the choice is defensible given the computational scale already involved.
- **"The title does not mention the single-pixel restriction"**: Purely presentational; not a methodological flaw.
- **"Imagenette vs. full ImageNet is unmotivated"**: The authors explicitly state the goal is "not to be exhaustive with different datasets and models but to understand the impact of perturbations." Evaluating at this scale is a scope decision, not a flaw.

---

## Novel Insights

The most genuinely novel observation arising from the synthesis of all three reviews is the following: the paper's finding that DROP ≈ 0.5 for value-replacement perturbations (U0, U1, FR) is arguably a *predictable consequence* of the out-of-distribution nature of those perturbations—replacing a pixel with image min, max, or a random value pushes the input off the data manifold, and the model's output in that region of input space is essentially arbitrary. Gaussian blur, by contrast, keeps the perturbed image closer to natural image statistics, which is precisely why it yields higher conformity. This mechanistic connection—OOD-ness of the perturbed image as the root cause of low conformity—is implied by the data but never stated. If confirmed, it would re-frame the problem from "fidelity metrics are inconsistent" to "value-replacement perturbations produce OOD inputs on which neural network outputs are non-informative," which is a more precise and actionable diagnosis that connects to the existing literature on OOD robustness in neural networks.

---

## Suggestions

1. **Provide downstream validation**: Run AOPC/AD% on the same model-dataset-perturbation combinations and correlate the metric rankings with DROP/PSim scores. Even a small-scale experiment (one model, one dataset, a few saliency methods) showing that low PSim predicts ranking reversals would substantially validate the proposed measures.
2. **Connect to OOD literature**: Measure the distributional distance (e.g., FID or feature-space distance) between original and perturbed images for each perturbation type, and show that this correlates with DROP/PSim scores. This would provide the mechanistic explanation for the Gaussian blur finding.
3. **Fix Equation (9)**: Change the summand from `PSim` to `PSim_M^k` to make the equation non-circular.
4. **Fix Section 5.3 typo**: The sentence "indicating low conformity to Point [P1]" should read "Point [P2]."
5. **Define operational thresholds**: Propose a concrete threshold (e.g., DROP < 0.7 or PSim < 0.5 should trigger a warning) to make the pre-conditional check actionable rather than open-ended.
6. **Correct VOC 2007 statement**: Acknowledge that VOC 2007 has standard splits, and clarify that all images are used because the analysis does not involve model training.

---

**Evaluation summary:**

- **Novelty**: Moderate. The problem and the direction are known from Tomsett et al. (2020); this paper adds formal measures and broadened empirics, but does not significantly advance the theoretical understanding.
- **Technical soundness**: Weak-to-moderate. The core conceptual gap—DROP measures per-pixel random perturbations while AOPC uses cumulative top-K perturbations—is a genuine mismatch that is not reconciled. Typographic errors in key equations further reduce confidence in precision.
- **Empirical support**: Moderate. The scale and consistency of results across models and datasets is commendable, but the missing downstream validation leaves the practical relevance of the conformity measures in doubt.
- **Significance**: Moderate for XAI practitioners; limited for the broader ICLR audience. The paper identifies and formalizes a real problem but does not offer a remedy.
- **Clarity**: Adequate overall; undercut by the errors in Eq. 9 and Section 5.3.

MY FINAL SCORE: <pineapple>4.2</pineapple>