Let me verify the specific claims from the paper text that I already have. Key things to check:

1. The paper says "total contribution to the output, flowing via all the consecutive attention heads" in the abstract/intro
2. The manual filtering of adversarial images
3. The layernorm handling

Let me re-read the paper content more carefully for these specific points.

From the paper text I have:

**On "total contribution"**: The abstract says: "we present the 'second-order lens', analyzing the effect flowing from a neuron through the later attention heads, directly to the output." And the intro says: "its total contribution to the output, flowing via all the consecutive attention heads." Equation 5 explicitly sums over l' = l+1 to L, over attention heads h, with the path being neuron → OV matrices → projection to output. The paper does acknowledge in Section 6 that they "ignored the effect of neurons on consecutive queries and keys in the attention mechanism" and "we did not analyze the mutual effects between neurons in the same layer or across different layers."

**On manual filtering**: Section 5.1 says "We repeat the experiment 3 times and manually remove images that include c_2 objects or do not include c_1 objects."

**On layernorm**: Footnote 2 says "Throughout the paper, we ignore layer-normalization terms to simplify derivations. We address layers-normalization in detail in Appendix A.6."

Now let me assess each major criticism:

**Harsh Critic Point 1 (Definition correctness of second-order effect)**: The critic argues that Eq. 5 doesn't capture the "total contribution" because it ignores later MLPs, query/key effects, and layernorm. The paper states "its total contribution to the output, flowing via all the consecutive attention heads." The key question is whether "flowing via all the consecutive attention heads" already scopes the claim to just the path through attention heads. Looking at the actual wording: "the effect flowing from a neuron through the later attention heads, directly to the output" — this does seem to carefully specify the path through attention heads. The phrase "total contribution" in the intro is more problematic: "its total contribution to the output, flowing via all the consecutive attention heads." This is somewhat ambiguous — "total" modifies "contribution" but the scope is "flowing via all the consecutive attention heads." The paper is transparent in Section 6 about what it ignores. However, the abstract says "analyzing the effect flowing from a neuron through the later attention heads, directly to the output" which seems to correctly scope it. The real question is whether the paper's claims implicitly assume this captures a neuron's full function. The paper does say "we present the 'second-order lens', analyzing the effect flowing from a neuron through the later attention heads" which is clear about the path. But some phrasing like "interpreting the function of individual neurons in CLIP" in the abstract is broader. This is a legitimate concern about claim scope — the method captures a specific path contribution, not the neuron's full functional role. However, the paper does address this in limitations, and the second-order effect clearly captures more than first-order effects. I'll keep this as a major (not fatal) weakness about overclaiming.

**Harsh Critic Point 2 (Evidence for text decompositions)**: The critic argues that the evidence for text decompositions describing individual neurons is weak. The validation through classification accuracy is a layer-level metric, not neuron-level. No quantitative per-neuron evaluation is done. The qualitative examples are cherry-picked. This is a valid and important concern. However, the paper does show that the text descriptions match the top activating images (Figure 5 and Table 2). The lack of a systematic per-neuron evaluation metric is a real gap. I'll keep this as a major weakness.

**Harsh Critic Point 3 (Adversarial pipeline confounds)**: The manual filtering is indeed problematic. The paper says "manually remove images that include c_2 objects or do not include c_1 objects" but doesn't specify whether baselines get the same treatment. This is a valid concern. I'll keep this as a major weakness but downgrade it from what the harsh critic implies (the method still shows clear relative gains).

**Harsh Critic Point 4 (Segmentation: attribution vs segmentation)**: The criticism that this is really "attribution-based" segmentation rather than "zero-shot segmentation" has some merit, but is somewhat semantic. The paper does frame it as "zero-shot segmentation" and it is evaluated on a segmentation benchmark. The modest improvement over TextSpan is a concern. I'll include this as a minor weakness.

Let me now synthesize the final review.

**Calibration Summary:**
- TextSpan (5Ca9sSzuDp): 8,8,8,8 = Oral — This is the predecessor paper from the same group, analyzing attention heads in CLIP
- PatchSAE (imT03YlG2): 6,6,6,8 = Poster — SAE for CLIP interpretability
- INViTE (5iENGLEJKG): 5,5,8,3 = Poster marginal — CLIP token interpretation with text
- NeuronPath (WQQyJbr5Lh): 6,6,6,6,6 = Poster — Influential neuron paths in ViT
- Describe-and-Dissect (Rnxam2SRgB): 3,6,5,5 = Reject — Neuron description with LLMs
- Incidental Polysemanticity (OeHSkJ58TG): 6,6,5 = Reject — Theory on polysemantic neurons

This paper is clearly stronger than Describe-and-Dissect (rejected) and the polysemanticity paper (rejected). It has more rigorous methodology, actual applications, and clear technical contributions. However, it has more limitations than TextSpan (8/8/8/8). The key question is whether it's closer to TextSpan or to the 6-range papers (PatchSAE, NeuronPath).

Compared to TextSpan, this paper:
- Extends the analysis from attention heads to neurons (a harder problem)
- Introduces the second-order lens concept (novel)
- Has two applications (adversarial + segmentation)
- However, has weaker evaluation: no per-neuron quantitative validation, small-scale adversarial eval, manual filtering issues
- The predecessor was cleaner and more complete

Compared to PatchSAE/NeuronPath (6 range):
- This paper has stronger novelty (second-order lens concept)
- Better applications (two concrete downstream uses)
- But similar issues with evaluation rigor and some overclaiming

I would place this in the 5.5-6.5 range. It's a solid contribution with real novelty, but the evaluation gaps and overclaiming hold it back from the 7+ range.</think>Now I have enough calibration data. Let me synthesize the final review.

## Summary

This paper introduces the "second-order lens" for interpreting individual neurons in CLIP-ViT, analyzing how neuron activations propagate through subsequent attention head OV matrices to affect the final output representation. The authors show that these second-order effects concentrate in late layers, are significant for only ~2% of images per neuron, and can be approximated by a single direction in CLIP's joint text-image space. By decomposing these directions into sparse sets of text representations via orthogonal matching pursuit, they reveal polysemantic neuron behavior and leverage it for two applications: automatic generation of semantic adversarial examples and zero-shot segmentation.

## Strengths

- **Well-motivated problem formulation with clear empirical justification.** The paper convincingly demonstrates why existing approaches fail: first-order effects of MLP neurons are negligible in CLIP, and indirect effects are obscured by self-repair mechanisms. Table 1 provides strong evidence — mean-ablating second-order effects drops accuracy to 29.6% versus 52.3% for indirect effects, and the first PC explains 48.2% of variance versus 11.0%. This cleanly motivates the need for the second-order lens.

- **Elegant mathematical formulation.** The second-order effect in Equation 5 cleanly decomposes into an input-dependent term (attention-weighted activations) and an input-independent term (PV-weighted output directions), enabling both the empirical characterization and the text decomposition pipeline. The factorization is principled and draws on prior OV-circuit analysis (Elhage et al., 2021).

- **Interesting empirical characterization of second-order effects.** The three key findings — late-layer concentration, sparsity (~2% of images), and approximate rank-1 structure (Figure 3) — are non-obvious properties of CLIP's internal computation that enable the downstream approach. The rank-1 reconstruction preserving near-baseline accuracy is particularly compelling.

- **Two diverse and creative applications.** The adversarial example pipeline (finding spurious concept overlaps via sparse decomposition, then using LLMs + text-to-image models) is conceptually novel. The zero-shot segmentation results (Table 4) show improvements over prior CLIP-based attribution methods across all metrics.

- **Honest discussion of limitations.** Section 6 explicitly acknowledges unanalyzed mechanisms (neuron effects on queries/keys, neuron-neuron interactions) and failure modes of the adversarial pipeline, which strengthens credibility.

## Weaknesses

### Major:

- **Overstated claims about interpreting "neurons' function."** The paper's central claim is to "interpret the function of individual neurons in CLIP" (Abstract), but the second-order effect (Eq. 5) captures only a specific path: neuron → value matrices in subsequent attention heads → projection → output. It ignores contributions through later MLPs, modulation of attention patterns via queries/keys affected by the neuron, and the nonlinear effects of layernorm (relegated to Appendix A.6). The paper acknowledges these limitations in Section 6, but the framing throughout (Abstract, Introduction, Section 3.2) presents the second-order effect as the neuron's "total contribution" or full "function," which is misleading. A more accurate framing would be: "the second-order lens captures the dominant pathway by which neurons influence the output, via subsequent attention head value circuits." The current framing risks practitioners misunderstanding what the method actually analyzes. The paper would be substantially strengthened by rescoping its claims.

- **Insufficient validation that sparse text decompositions describe individual neurons.** The primary quantitative check is layer-level classification accuracy after replacing all neurons' second-order contributions with sparse text reconstructions (Figure 4). This only shows that the decompositions preserve the layer's aggregate effect on classification — it does not demonstrate that for any given neuron, the chosen texts predict *which images* activate it, or that the signs and magnitudes of coefficients align with where the neuron is important. The qualitative evidence (Table 2, Figure 5) shows only four cherry-picked neurons. There is no systematic analysis of how often decompositions are interpretable across the neuron population, nor comparison to baselines like random text combinations or SAE-based decompositions. Without per-neuron quantitative evaluation, the central interpretability claim is inadequately supported.

- **Adversarial evaluation suffers from methodological confounds.** The pipeline manually removes "images that include c₂ objects or do not include c₁ objects" (Section 5.1), but the paper does not report how many images are discarded per method, nor whether baselines receive identical filtering. If the second-order method generates higher-quality images that require less filtering, that is a legitimate advantage — but it must be documented to allow fair interpretation. Additionally, evaluation on only 5 CIFAR-10 binary classification pairs with 100 images each (and high variance, e.g., ±4.5 for several tasks) is too narrow a benchmark to support the strong "mass-production" claim. No comparison to existing semantic/unrestricted adversarial attack methods (e.g., diffusion-based attacks) is provided.

### Minor:

- **Segmentation improvements are modest and lack robustness analysis.** The zero-shot segmentation improvement over TextSpan (the authors' prior method) is small: 59.0 vs 58.1 mIoU, 84.9 vs 84.1 mAP. The paper provides no sensitivity analysis for the number of top neurons selected (200), the layer choice (8–10), or the threshold (0.5). No analysis of how masks behave when the queried class is absent from the image, a critical failure mode. Furthermore, some baselines (Partial-LRP, rollout, Chefer et al.) are attribution methods originally designed for classification explanation rather than segmentation — it is unclear whether they benefit from equivalent post-processing.

- **Rank-1 approximation is central but its failure modes are uncharacterized.** The aggregate statistic (first PC explains 48.2% variance) masks per-neuron variation. There is no distribution showing how many neurons are well-approximated by a single direction versus poorly approximated, and no analysis of which types of neurons or concepts lose information in the rank-1 projection. This matters because secondary concepts a neuron responds to could be lost in the principal component.

- **Text pool dependence is acknowledged but not analyzed.** The paper shows pool choice matters (Figure 4), but does not analyze what concepts are systematically missed when a concept is absent from the pool. Neurons encoding concepts absent from the vocabulary cannot be properly described.

### Trivial:

- The 2% sparsity figure is mentioned prominently but the exact quantile thresholding and consistency across neurons is not documented in the main text (only shown via the "w/o large norm" vs "w/o small norm" experiment in Figure 3).

## Nice-to-Haves

- Comparison to sparse autoencoder (SAE) baselines for neuron decomposition, which is the dominant approach in mechanistic interpretability and the most natural benchmark for this work.
- Broader adversarial evaluation on ImageNet-scale multi-class classification rather than 5 binary CIFAR-10 tasks.
- Per-neuron quantitative evaluation of description quality (e.g., predict neuron activations on held-out images using the text decomposition).
- Distribution of per-neuron variance explained by the first PC to reveal how many neurons are poorly approximated.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Claim that the second-order effect definition is incorrect or mathematically wrong.** The harsh critic argues Eq. 5 is "not the actual contribution" and implicitly assumes linear propagation. However, Eq. 5 is precisely defined as a specific path contribution through OV matrices, and the paper is transparent about which paths it captures vs. ignores. The equation is mathematically correct for what it computes. The issue is one of *framing* (overclaiming scope), not mathematical error.

- **Demand for layernorm to be handled in the main text.** The paper acknowledges the layernorm simplification in Footnote 2 and addresses it in Appendix A.6. For an empirical paper using path-based attribution, this is standard practice — logit lens and OV-circuit analysis in the mechanistic interpretability literature also commonly simplify layernorm. Deferring details to an appendix is reasonable.

- **Demand for comparison to dedicated zero-shot segmentation methods (MaskCLIP, GroupViT).** The paper's segmentation section is a downstream application demonstrating that interpretability yields capabilities, not a claim to advance segmentation SOTA. Comparing to attribution baselines is the appropriate comparison group given the framing.

- **Concern about method only applying to CLIP.** The paper explicitly leverages CLIP's shared text-image space, which is a feature, not a bug. Criticizing a method designed for a specific model family for not working on non-text-aligned models is scope creep.

- **Demand for confidence intervals for segmentation results.** Single-run evaluation is the norm for this type of attribution/segmentation work in the vision interpretability community.

## Novel Insights

The key insight that emerges from this work — distinct from its own stated contributions — is that the "second-order effect" provides a natural decomposition point between two failure modes of neuron interpretability in CLIP: first-order effects are too small (neurons don't directly drive the output), while indirect effects are too noisy (self-repair obscures individual roles). The second-order path through attention OV matrices hits a productive middle ground precisely because the attention heads, not the MLPs, serve as the primary routing mechanism in late CLIP layers. This suggests a more general principle: for transformers where certain layer types dominate information routing (here, attention layers), path-based attribution should follow those dominant pathways rather than attempting to capture all routes. Whether this principle generalizes beyond CLIP to other vision-language models or LLMs would be a valuable direction.

## Suggestions

- Rescope the central claims: replace "interpreting the function of individual neurons" with "analyzing the second-order path contribution of neurons through attention head value circuits" throughout the abstract and introduction. This is more accurate and still impactful.
- Add a per-neuron evaluation metric: measure whether the text decomposition for a neuron predicts its activation patterns on held-out images (e.g., rank correlation between predicted and actual activation magnitudes across images). This would directly validate the interpretability claim at the neuron level.
- For adversarial evaluation, report the fraction of generated images discarded per method (to quantify the manual filtering asymmetry) and test on at least 10-15 class pairs beyond CIFAR-10 to establish generalizability.
- Show a histogram of per-neuron variance explained by the first PC across all neurons in the analyzed layers, to characterize how universally the rank-1 approximation applies.

## Score and Decision

**Calibration anchors:**
- **TextSpan/Gandelsman et al. 2024** (5Ca9sSzuDp): 8,8,8,8 = Oral. Same group, predecessor work analyzing attention heads in CLIP. Stronger validation, cleaner claims, well-established applications.
- **PatchSAE** (imT03YXlG2): 6,6,6,8 = Poster. SAE for CLIP interpretability with similar concerns about design choices, ad-hoc hyperparameters, and evaluation depth.
- **INViTE** (5iENGLEJKG): 5,5,8,3 = Poster. CLIP token interpretation with text, similar scope limitations and CLIP-dependency concerns.
- **NeuronPath** (WQQyJbr5Lh): 6,6,6,6,6 = Poster. Influential neuron paths in ViT, similarly limited in scale but clear methodology.
- **Describe-and-Dissect** (Rnxam2SRgB): 3,6,5,5 = Reject. Neuron description with LLMs, lacking per-neuron quantitative validation — this paper is clearly stronger.

This paper sits between the 6-range papers (PatchSAE, NeuronPath, INViTE) and the predecessor TextSpan (8-range). It has genuine novelty (the second-order lens concept), interesting structural findings (rank-1, sparsity, polysemy), and creative applications. However, it overclaims about interpreting "neurons' function" (when it actually analyzes a specific pathway), lacks per-neuron quantitative validation of the text decompositions, and the adversarial evaluation is small-scale with methodological confounds. These are not fatal issues but they meaningfully weaken the paper's contribution compared to TextSpan, which made more carefully scoped claims with cleaner evaluation. The paper is stronger than the marginal accept papers (INViTE) because of its more principled formulation and concrete structural findings, but weaker than robust 6-range papers due to the evaluation gaps and overclaiming. I place it at the lower end of the accept range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>