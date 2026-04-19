## Summary

The paper introduces a "second-order lens" for interpreting CLIP neurons by tracing their contributions through subsequent attention heads to the output representation. The key empirical finding is that these second-order effects are functionally significant (mean-ablation drops ImageNet accuracy to 29.6% vs. 52.3% for indirect effects) and compressible (first PC explains 48.2% of variance vs. 11.0%), enabling a rank-1 approximation and sparse text decomposition that reveals polysemantic neuron behavior. The paper demonstrates downstream applications in semantic adversarial example generation and zero-shot segmentation.

## Strengths

- **Clear motivation for a genuinely novel interpretability lens.** The paper establishes why first-order effects are near-constant in CLIP and why indirect effects are obscured by self-repair, then proposes a well-defined alternative: path attribution through attention OV values. Table 1 provides clean quantitative evidence that second-order effects are more consequential and structurally simpler than indirect effects — the core empirical contribution is solid.

- **Elegant use of CLIP's shared text-image space for automated neuron description.** Rather than relying on human annotations or expensive training, the paper decomposes each neuron's rank-1 direction into a sparse sum of existing text embeddings. Table 2 and Figure 5 show that the resulting text descriptions align with the top-activating images, and that polysemy (e.g., neuron #2914 encoding both "yacht" and "cabriolet") is captured.

- **Creative applications that demonstrate actionable understanding.** The adversarial example pipeline (Section 5.1) and zero-shot segmentation (Section 5.2) go beyond anecdotal neuron visualizations. The segmentation improvement over TextSpan (59.0 vs. 58.1 mIoU, 84.9 vs. 84.1 mAP in Table 4) and the adversarial success rates in Table 3 (consistently beating baselines) show the interpretability yields practical utility.

- **Honest limitations section.** Section 6 candidly acknowledges omission of query/key-mediated effects and neuron-neuron interactions, which is rare in this subfield.

## Weaknesses

### Major

- **The "total contribution" framing is inconsistent with what is actually computed.** The introduction describes the second-order effect as the neuron's "total contribution to the output, flowing via all the consecutive attention heads" (line 25), and the abstract similarly describes it as "analyzing the effect flowing from a neuron through the later attention heads, directly to the output." Yet Equation (5) treats attention weights as fixed — the neuron's effect is propagated only through OV/value paths. The effect on downstream queries, keys, and subsequent attention patterns is entirely omitted. Section 6 acknowledges this ("ignored the effect of neurons on consecutive queries and keys"), but this is not a minor omission: a neuron can meaningfully alter downstream computation by changing which tokens attention attends to, which is a qualitatively different causal pathway. The paper should present this as a *partial* path attribution from the start, not as the neuron's total downstream contribution. As written, the claimed object and the computed object differ.

- **The rank-1 approximation is validated only indirectly through downstream accuracy, not through direct reconstruction fidelity.** The central claim that each neuron's effect can be approximated by a single direction (enabling the sparse text decomposition) relies primarily on the observation that replacing all effects with their first-PC reconstruction preserves ImageNet accuracy (Figure 3, "rec. from PC #1"). But zero-shot classification is insensitive to substantial per-neuron reconstruction errors due to redundancy across thousands of neurons. Table 1 reports variance explained for one layer comparison only, and no distribution of per-neuron cosine similarity, logit agreement, or reconstruction error is provided in the main text. Since the entire Section 4 text decomposition pipeline rests on this rank-1 reduction, direct per-neuron validation is essential — not nice-to-have.

### Minor

- **The adversarial-example application is confounded by manual filtering and low absolute success rates, weakening the headline "mass production" claim.** The pipeline (Section 5.1) generates 100 images per task through LLM prompt writing and text-to-image generation, then "manually remove[s] images that include c_2 objects or do not include c_1 objects" (line 219). Success rates are low in absolute terms (5–23 per 100 on binary tasks), and the evaluation is restricted to a narrow CIFAR-10 pairwise setup. While the baselines are run through the same pipeline (providing controlled relative comparison), the absolute claims about "mass-producing semantic adversarial examples" overstate what the experiment demonstrates. The result is better characterized as a proof of concept.

- **The segmentation evaluation lacks a robustness analysis of hyperparameter choices.** The method selects top 200 neurons from layers 8–10 and binarizes at threshold 0.5 after standardization. It is unclear whether these choices were tuned on the benchmark, whether baselines had equivalent access to class-conditioned component selection, or how sensitive results are to varying these parameters. This does not invalidate the Table 4 results, but it weakens confidence in the claimed improvement.

### Trivial

- **Figure 3 is overloaded**, simultaneously used to demonstrate concentration in late layers, sparsity across images, and rank-1 reconstructability. These are distinct claims that would benefit from disentangled visualizations.

## Nice-to-Haves

- Report per-neuron cosine similarity and logit agreement distributions to directly validate the rank-1 approximation quality, rather than relying solely on aggregate accuracy preservation.
- Ablate the adversarial pipeline while keeping LLM/T2I components fixed and varying only the source of mined concepts (neuron-based vs. random/similarity-matched) to more cleanly isolate the interpretability contribution.
- Extend analysis beyond OpenAI ViT-B-32 to additional architectures (the paper mentions ViT-L-14 in the appendix but should include key results in the main text given the broad claims about "CLIP neurons").

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Weakness: The adversarial comparison does not isolate whether neuron polysemy is the true driver of success."** The paper does include a "similar words" baseline that controls for concept similarity without neuron mining, and an "indirect effect" baseline that controls for the path attribution object. While a stronger control (matching prompt length/diversity identically) would be better, the existing baselines provide reasonable isolation.
  
- **"Weakness: Selectivity of <2% is anecdotal, depending on an arbitrary top-100 cutoff."** The paper's definition of "significant" uses largest-norm examples, and the sparsity is demonstrated via the accuracy drop when ablating only large-norm effects vs. small-norm effects (Figure 3). This is a standard operationalization in mechanistic interpretability.

- **"Weakness: Layernorm omission makes Eq. (5) an approximation without clear status."** The paper acknowledges this in footnote 2 ("we ignore layer-normalization terms to simplify derivations") and refers to Appendix A.6. For a path-attribution lens where the focus is attention-value pathways, this is an acceptable simplification.

- **"Weakness: Segmentation threshold tuning and unfair baseline comparison."** The baselines in Table 4 are standard published methods (GradCAM, TextSpan, etc.) that are compared as fixed baselines; this is standard practice. The neuron selection uses the learned text-embedding similarity, which is a principled criterion, not an arbitrary hyperparameter sweep.

- **"Weakness: Missing broader quantitative semantic validation of neuron descriptions beyond Table 2's cherry-picked examples."** Table 2 is qualitative, and the reconstruction evaluation uses accuracy (Figure 4), but the qualitative alignment between text descriptions and top-activating images (Figure 5) is reasonably convincing for an initial interpretability lens. More systematic validation is desirable but not a fatal gap.

- **"Weakness: Figure 3 and other figures are overloaded."** This is a minor presentation point, already moved to Trivial.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

1. **Reframe the core claim:** Change "total contribution" to "second-order value-path contribution" or similar throughout the paper. Acknowledge upfront (in the abstract/introduction, not only Section 6) that query/key-mediated effects are excluded, and characterize the method as a partial path attribution lens. This is a fixable framing issue that directly addresses the most important criticism.

2. **Add per-neuron validation metrics:** Report the distribution of cosine similarities between true φₙˡ(I) and its rank-1 approximation across neurons and layers, and optionally per-neuron logit-agreement distributions. This would strengthen the empirical foundation of the entire text-decomposition pipeline.

3. **Clarify the adversarial evaluation:** Report how many images were manually removed and what fraction remained, and include a breakdown of attack success by task on *all* generated images (including those that would be filtered), to give a more honest picture of the pipeline's reliability.

## Score and Decision

I calibrated against several anchors:
- The prior CLIP-text-decomposition paper by the same lead author (5Ca9sSzuDp) received 8,8,8,8 (Accept Oral) — that paper had cleaner experiments on head-level interpretation with no framing inconsistencies.
- Papers with novel interpretability ideas but validation gaps (OCqyFVFNeF: 8,6,5; L7jtdGhWzT: 6,5,3) provide a reference for this tier.
- Borderline papers (4-6 range) typically have unclear methodology or insufficient validation, while this paper's core empirical results (Table 1, Figure 3) are actually quite strong.

The paper under review has a genuinely novel contribution (second-order path attribution for CLIP neurons is harder and more interesting than head-level analysis) with solid empirical backing for the core claim. The major issues — overstated framing ("total contribution") and indirect validation of the rank-1 approximation — are significant but not fatal; both are addressable with reframing and additional metrics. The applications are creative and demonstrate real utility. Compared to the 8,8,8,8 anchor, this paper has more framing inconsistency and weaker per-neuron validation. Compared to the borderline (4-6) papers, it has stronger core empirical results and more creative applications. I place it solidly above the borderline cluster.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>