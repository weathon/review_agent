=== CALIBRATION EXAMPLE 1 ===

# Final Consolidated Review
## Summary

This paper introduces a "second-order lens" for interpreting MLP neurons in CLIP-ViT models, analyzing effects flowing from neurons through subsequent attention heads to the output representation. The authors show these effects concentrate in late layers, are significant for <2% of images, can be approximated by rank-1 directions in the joint embedding space, and can be decomposed into sparse text representations revealing polysemantic neuron behavior. They apply this interpretation to generate semantic adversarial examples and improve zero-shot segmentation.

## Strengths

- **Methodologically grounded motivation:** The second-order lens directly addresses documented failures of direct effects (MLP contributions to residual stream are nearly constant) and indirect effects (obscured by self-repair mechanisms, Table 1 shows 52.3% accuracy retained vs. 29.6% for second-order). The formalism builds appropriately on Elhage et al.'s attention decomposition.

- **Useful empirical characterizations:** The findings that second-order effects concentrate in late layers (Figure 3), exhibit sparsity (<2% of images have significant effects), and are approximately rank-1 (reconstruction from PC#1 preserves accuracy) enable tractable analysis. Table 1 quantifies the advantage over indirect effects (48.2% vs 11.0% variance explained by PC1).

- **Actionable applications:** The zero-shot segmentation results improve over TextSpan across all metrics (Table 4: 78.1 vs 76.5 Pix.Acc, 59.0 vs 58.1 mIoU). The adversarial example generation demonstrates that interpretability can expose model vulnerabilities.

## Weaknesses

- **Low and incompletely reported adversarial success rates:** Table 3 shows success rates of 5-23 images per 100, which undercuts claims of "mass production." More critically, the paper states "we manually remove images that include $c_2$ objects or do not include $c_1$ objects" but never reports what fraction of generated images was retained. If 50% of images were filtered, the effective success rate differs substantially from the reported numbers.

- **Limited adversarial evaluation scope:** Only 5 class pairs are tested from 45 possible CIFAR-10 pairs, with no pre-registered selection criterion. The binary classification setup is easier than realistic multi-class scenarios. High variance in some tasks (8.0 ± 4.5) with only 3 repetitions raises reliability concerns.

- **Text description validation is indirect:** The paper validates sparse decompositions via downstream classification accuracy (Figure 4), not semantic meaningfulness. Two different word sets could produce similar reconstruction accuracy if they span similar directions in CLIP's embedding space. No human evaluation or systematic analysis of description quality is provided.

- **Cherry-picked qualitative examples:** Table 2 and Figure 5 show only 4 selected neurons. No aggregate statistics report what fraction of neurons have coherent, interpretable descriptions versus uninterpretable ones.

- **Aggregate statistics mask individual variation:** The 48.2% variance explained (Table 1) is a layer-level aggregate; the paper does not show the distribution of per-neuron variance explained. Individual neurons could vary widely around this mean.

- **Hyperparameter choices lack justification:** The "top 200 neurons from layers 8-10" for segmentation is stated without ablation. The sparsity threshold of 100 images (2%) is ad hoc with no sensitivity analysis.

- **Segmentation gains are marginal:** The improvements over TextSpan (+1.6 Pix.Acc, +0.9 mIoU, +0.8 mAP) are small, with no statistical significance testing. Given that TextSpan is from the same authors, a stronger baseline comparison would be appropriate.

## Nice-to-Haves

- **Individual neuron causal validation:** Ablating specific neurons and measuring disproportionate degradation on images matching their text descriptions would strengthen causal claims about interpretation quality.

- **Attention head attribution:** Analyzing which specific attention heads carry the second-order signal would strengthen the mechanistic claim that information flows through specific pathways.

- **Failure case analysis:** Showing examples where top text descriptions do not match top activating images would reveal limits of the method.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Layer normalization approximation:** The paper acknowledges this in Footnote 2 and defers to Appendix A.6. This is standard practice in mechanistic interpretability work and does not constitute a missing contribution in the main text.

- **Generalization beyond CLIP:** The paper explicitly scopes itself to CLIP-ViT. Demanding analysis of other architectures is scope creep.

- **Notation inconsistencies ($r_n^l$ vs $r_n^d$):** This is a minor formatting issue, not a substantive problem.

- **Computational complexity analysis:** While useful, this is not a standard requirement for interpretability papers. The method involves a straightforward summation over layers, heads, and tokens.

- **Cross-architecture results in main text:** ViT-L-14 results are provided in Appendix A.1, which is appropriate for supplementary material.

- **Comparison to unrelated semantic attack methods:** The paper compares to appropriate baselines (random neurons, indirect effects, similar words). Demanding comparison to spatial attacks or adversarial patches addresses a different research question.

## Novel Insights

The rank-1 approximability of second-order effects is genuinely surprising and practically useful—it suggests that despite the complexity of the attention pathway, each neuron's contribution can be captured by a single direction. The sparsity finding (<2% of images) suggests that CLIP neurons implement sparse, selective computations rather than dense feature mixing. The discovery that polysemantic neurons can be exploited for semantic adversarial attacks is a concrete demonstration of why interpretability matters for safety: spurious internal correlations create attack surfaces that would be invisible without mechanistic analysis.

## Suggestions

- **Report manual filtering rates:** State explicitly what fraction of generated adversarial images was retained after filtering in each task, and report success rates on the actual denominator.

- **Add per-neuron variance distribution:** Include a histogram or summary statistics showing the distribution of variance explained by PC1 across individual neurons, not just the layer aggregate.

- **Expand adversarial evaluation:** Test on all 45 CIFAR-10 pairs or a randomly selected subset with pre-registered selection, and report results with proper statistical characterization.

- **Provide quantitative interpretability assessment:** Report what fraction of neurons have text descriptions that match their top activating images, or conduct human evaluation of description quality.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 5.0]
Average score: 6.8
Binary outcome: Accept
