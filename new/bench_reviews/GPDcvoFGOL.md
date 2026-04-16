## Summary
This paper proposes a new “second-order lens” for interpreting CLIP MLP neurons by tracing a neuron's contribution through later attention value/output paths into the final CLIP image representation, then decomposing the resulting output-space direction into sparse text directions. Empirically, the paper finds these effects are concentrated in late layers, highly selective across images, and often well-approximated by a dominant direction, and it uses this machinery for two applications: semantic adversarial example generation and zero-shot segmentation.

## Strengths
- **Clear motivation for why standard views are insufficient for CLIP neurons.** The paper convincingly explains that direct effects are negligible for CLIP MLP neurons and that indirect effects are obscured by self-repair; Table 1 supports this with a much larger performance drop from ablating second-order effects than indirect effects, and higher variance captured by the first PC.
- **Interesting empirical characterization of the proposed quantity.** Figure 3 and Section 3.3 provide meaningful observations: the strongest effects are in moderately late layers, they are sparse/selective across images, and a one-direction approximation preserves downstream accuracy surprisingly well.
- **Good use of CLIP’s shared text-image space.** Decomposing neuron-associated directions into sparse text representations is a natural and elegant idea in this setting, and Table 2 / Figure 5 provide plausible qualitative evidence that the recovered phrases track salient visual concepts.
- **The applications are not mere toy demos.** The zero-shot segmentation result is strong enough to be noteworthy: Table 4 beats prior attribution-style baselines across all reported metrics. The adversarial-generation pipeline is also creative and shows nontrivial gains over the included baselines in Table 3.
- **Strong clarity overall.** The paper is generally well written, well structured, and candid about some limitations in Section 6, which makes the core technical narrative easy to follow.

## Weaknesses
###: Fatal
- **The paper’s central framing overstates what the proposed “second-order effect” captures.** The text repeatedly describes it as the neuron’s “total contribution to the output, flowing via all the consecutive attention heads” (e.g., Abstract, Section 1, Section 3.2), but the actual quantity in Eq. (5) only propagates through later attention OV/value paths. The paper itself later states in Section 6: “we ignored the effect of neurons on consecutive queries and keys” and “did not analyze the mutual effects between neurons.” This does not make the method invalid, but it does mean the main object is a **restricted path-based proxy**, not the full downstream contribution suggested by the wording. Since the main empirical findings and both applications are built on this quantity, the overclaim matters and should be corrected explicitly.

### Major:
- **The core interpretability claim is only partially validated.** The main quantitative validation in Section 4 is reconstruction-oriented: the paper shows that sparse text combinations can approximate the neuron-associated direction well enough to preserve ImageNet classification accuracy (Figure 4). That demonstrates that CLIP text embeddings can span useful CLIP-space directions, but it is weaker than showing the selected phrases are faithful descriptions of neuron function. The paper also claims in the introduction that “these concepts correctly track which inputs activate a given neuron,” but the main text does not provide a direct quantitative activation-prediction or retrieval-style evaluation for this claim.
- **The adversarial-example application is substantially confounded by the external generation pipeline.** The attack uses neuron scores, sparse phrase extraction, an LLM to build prompts, a text-to-image model to generate images, and then manual filtering of failures (“manually remove images that include \(c_2\) objects or do not include \(c_1\) objects”). The included baselines are helpful, so this is not a strawman criticism; however, because so much of the final outcome depends on prompt quality, image generation fidelity, and manual curation, the results in Table 3 do not cleanly isolate how much of the success comes from the proposed neuron interpretation itself. This weakens one of the headline demonstrations.
- **The segmentation improvement is real but only weakly tied to the specific second-order interpretation.** The segmentation method selects neurons by similarity between \(r_n^l\) and the class text embedding, averages their activation maps, and thresholds the result. Table 4 is promising, but there is no direct ablation against obvious alternatives such as first-order-based neuron directions, simpler neuron-selection heuristics, or matched random/activation-based selections. As written, the section shows a useful CLIP-based segmentation heuristic derived from the learned directions, but it does not fully establish that the gain is specifically due to the second-order lens.

### Minor
- **The “approximately rank-1” claim would benefit from more complete analysis.** Table 1 reports that the first PC explains 48.2% of variance for second-order effects, which is meaningful but far from exhaustive. Figure 3 shows that replacing effects with a PC1-based approximation preserves accuracy, which is interesting, but the paper does not explain where the remaining variance goes, nor provide per-neuron or per-layer distributions showing when the rank-1 approximation is strong versus weak.
- **The selectivity claim is under-specified in the main text.** The paper states that each neuron is significant for “less than 2% of images,” but the operational definition of significance is only indirectly conveyed through ablation on the “100 images with the largest norms” in Figure 3. The intuition is reasonable, but the claim would be stronger with a clearer thresholded definition and sensitivity analysis.
- **Several important design choices are not systematically ablated.** For example: using layers 8–10 for applications, top-100 neurons for adversarial prompting, top-200 neurons for segmentation, and the threshold of 0.5 for segmentation binarization. These choices are plausible and not unusual for this style of paper, but some sensitivity analysis would help establish robustness.
- **Ignoring layer normalization in the main derivation is a nontrivial simplification.** The paper acknowledges this in footnote 2 and says Appendix A.6 addresses it. That is acceptable as a derivational simplification, so this is not a fatal flaw, but because the paper’s method depends on linearized path contributions, the main text would benefit from a clearer quantitative reassurance that the simplification does not materially alter the conclusions.

### Trivial
- **The segmentation gains over the strongest prior baseline are modest in absolute magnitude.** Table 4 improves over TextSpan from 58.1 to 59.0 mIoU and 84.1 to 84.9 mAP. These are still real gains, but the paper should frame them as incremental rather than dramatic.
- **The adversarial success rates, while better than the baselines, remain modest in absolute terms for several tasks.** This is enough for a proof-of-concept, but not yet evidence of a broadly reliable “mass production” pipeline.

## Nice-to-Haves
- Provide a direct quantitative evaluation of description quality, e.g., whether the recovered phrases predict high-activation images for a neuron better than baselines.
- Add ablations comparing second-order-based neuron selection against first-order, indirect-effect, activation-only, and random selection for segmentation.
- Break down failure modes of the adversarial pipeline: prompt-generation failures, image-generation failures, and CLIP-classification failures after successful generation.
- Report per-neuron/per-layer distributions of variance explained by the first PC, and test whether rank-2/3 approximations materially improve fidelity.
- Include a brief robustness analysis for key hyperparameters such as number of selected neurons and layer choices.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The layerwise mean-ablation experiments do not isolate per-neuron importance, so conclusions about individual neurons are invalid.”**  
  This is too strong. The paper does include neuron-level structure via \( \phi_n^l(I) \), and the layerwise ablations in Section 3.3 are used mainly to characterize where these effects matter globally. It is fair to ask for finer-grained statistics, but not to dismiss the neuron-level conclusions outright.

- **“The paper should compare against sparse autoencoders or additional outside related work.”**  
  Per instruction, missing-related-work complaints should be removed unless directly verifiable from the paper’s own scope.

- **“The method should be evaluated on many more datasets/models or on non-CLIP architectures.”**  
  This is mostly scope creep. Broader validation would strengthen the work, but the paper is explicitly about interpreting CLIP neurons, and it does include at least some appendix evidence beyond the main ViT-B-32 setting.

- **Pure reproducibility nitpicks about omitted hyperparameters or release details.**  
  Not substantive enough for the main review.

## Novel Insights
The key synthesis here is that the paper is strongest when read as introducing a **useful restricted path-based interpretability proxy** for CLIP neurons, rather than a complete account of neuron function. Under that interpretation, the empirical findings are quite interesting: late-layer MLP neurons seem to have sparse, image-conditional value-path effects that are much easier to model in CLIP space than either direct effects or indirect ablations. The paper’s main weakness is therefore not that the method fails, but that the framing sometimes upgrades a partial but effective mechanism into a fuller causal account than the evidence supports. If the claims were narrowed and the interpretation quality were validated more directly, the work would look considerably stronger.

## Suggestions
- **Tighten the central claim.** Replace “total contribution through later attention heads” with wording that explicitly describes Eq. (5) as a value/OV-path contribution proxy, and discuss its scope earlier, not only in limitations.
- **Add a direct neuron-description evaluation.** For each neuron, test whether the sparse text decomposition predicts which images yield high second-order norms or high activations better than alternative descriptions or random text.
- **Strengthen the segmentation section with targeted ablations.** Compare selection by second-order directions against first-order directions, activation magnitude, and random matched neurons.
- **Disentangle the adversarial pipeline.** Quantify how much of the final success rate is lost at each stage and report results before/after manual filtering.
- **Report richer statistics for the rank-1 story.** Show the distribution of PC1 explained variance across neurons/layers and examine whether higher-rank approximations improve downstream performance or interpretability.

## Score and Decision
**Assessment across axes:**  
- **Originality:** High. The second-order lens is a meaningful and novel extension of prior CLIP interpretability ideas.  
- **Importance:** Good. Understanding neuron-level mechanisms in CLIP is valuable, and the work targets a genuine blind spot in existing methods.  
- **Support for claims:** Mixed. The empirical evidence for the proposed proxy is solid, but the strongest interpretability claims and the “total contribution” framing are overstated relative to what is actually computed and validated.  
- **Experimental soundness:** Reasonably strong but incomplete. Core characterizations are interesting; application evaluations are promising but undercontrolled.  
- **Clarity:** Strong. The paper is generally clear and well organized.  
- **Community value:** Positive. Even with caveats, this is a useful interpretability direction for CLIP-style models.

**Calibration against human-reviewed anchors:**  
- Compared to **TextSpan / “Interpreting CLIP's Image Representation via Text-Based Decomposition”** (/home/wg25r/review_agent/human_reviews/5Ca9sSzuDp.md, all 8s, accepted oral), this submission is somewhat weaker: it has similarly appealing CLIP-space interpretability ideas, but the central object is more restricted than the framing suggests, and the validation of neuron descriptions is less direct.  
- Compared to **INViTE** (/home/wg25r/review_agent/human_reviews/5iENGLEJKG.md, mixed 5/5/8/3, accepted poster), this paper is stronger in empirical characterization and clearer in technical motivation, though it shares some undercontrolled application evidence.  
- Compared to **Describe-and-Dissect** (/home/wg25r/review_agent/human_reviews/Rnxam2SRgB.md, 3/6/5/5, rejected), this paper is much stronger: it is more mechanistically grounded, less dependent on a stack of black-box modules for the core method, and provides stronger quantitative analyses.  
- Compared to **PatchSAE** (/home/wg25r/review_agent/human_reviews/imT03YXlG2.md, 6/6/6/8, accepted poster), this paper feels roughly comparable in overall quality: both are interesting interpretability papers with real contributions and some overreach/under-validation. I place this submission slightly above PatchSAE on clarity and conceptual sharpness, but below TextSpan on completeness and support.

Overall, this looks like a **borderline-to-moderately positive accept**: a real contribution with notable strengths, but one that should substantially narrow its claims and strengthen validation.

**Score: 6.5 / 10**  
**Decision: Accept**

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>