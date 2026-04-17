Now I have enough context from calibration papers. Let me synthesize the final review.

## Summary

This paper introduces the "second-order lens" for interpreting individual neurons in CLIP-ViT models, arguing that direct (first-order) and indirect (ablation-based) effects fail to reveal neuron function due to constant direct effects and self-repair mechanisms respectively. The authors define the second-order effect as a neuron's contribution flowing through subsequent attention head OV circuits to the output, finding these effects are selective (significant for <2% of images) and approximately rank-1. They decompose these rank-1 directions into sparse text representations, revealing polysemantic neuron behavior, and apply this understanding to mass-produce semantic adversarial examples and zero-shot segmentation.

## Strengths

- **Principled motivation for the second-order lens**: The observation that first-order effects are nearly constant and indirect effects are obscured by self-repair (Table 1: 48.2% vs 11.0% variance explained by PC1; mean-ablation drops accuracy to 29.6% vs 52.3%) provides a clear and well-grounded reason to study second-order effects. This is a genuine advance over the prior TextSpan work (Gandelsman et al., 2024), which explicitly noted that MLP neuron effects were not interpretable via direct effects.

- **Novel and concrete mathematical formulation**: Equation 5 cleanly decomposes the second-order effect into input-dependent (attention-weighted activations) and input-independent (OV-projected neuron directions) terms, enabling tractable analysis and the subsequent sparse decomposition approach.

- **Two compelling applications grounded in interpretability insights**: The generation of semantic adversarial examples by exploiting polysemantic neuron overlap (Table 3: significantly higher success rates than baselines) and the zero-shot segmentation outperforming prior methods (Table 4: 59.0 mIoU vs 58.1 for TextSpan) demonstrate that the mechanistic understanding yields actionable downstream capabilities.

- **Interesting empirical findings**: The selectivity finding (significant for <2% of images), the rank-1 structure, and the concentration in late layers are non-trivial observations that advance understanding of CLIP's internal computation.

## Weaknesses

### Fatal
None.

### Major

- **The second-order effect construct ignores neuron influence on attention patterns (Q/K), creating an incomplete account of neuron function**: The paper acknowledges in Section 6 that it "ignored the effect of neurons on consecutive queries and keys," but then presents φ_n^l as capturing the neuron's contribution "to the output." The derivation in Eq. (5) only accounts for the flow through OV matrices, treating attention weights as fixed rather than as functions that neurons can modify. The paper provides no empirical analysis of how significant the Q/K-mediated effects might be. This matters because all downstream conclusions—selectivity, rank-1 structure, text decomposition, polysemy—are derived from φ_n^l as if it captures the neuron's functional role. Without a causal validation (e.g., replacing neuron outputs and comparing the full forward pass change vs. the second-order effect), the claim that this captures what a neuron "does" is only partially justified. The acknowledgement of this limitation is present but understated relative to the strength of the claims made elsewhere.

- **Evidence for the rank-1 approximation is thin for interpretive purposes**: The claim that each neuron's second-order effect is "approximately rank-1" (Section 3.3) underpins the entire text decomposition pipeline, yet the primary validation is preservation of classification accuracy after replacing φ_n^l with its PC1 approximation (Fig. 3). Table 1 reports 48.2% variance explained by PC1 for second-order effects—leaving more than half the variance unexplained. This is a substantial gap. Classification accuracy is an extremely forgiving metric: even substantial distortions to representation directions orthogonal to class boundaries would not affect top-1 accuracy. The paper does not report per-neuron distributions of explained variance, making it impossible to know whether most neurons are well-captured by rank-1 or whether the aggregate is inflated by a few neurons. If many neurons have multi-directional effects, the sparse text decomposition becomes a lossy compression whose faithfulness as an interpretability tool is questionable.

- **The adversarial example evaluation is small-scale and subject to manual curation without transparency**: Table 3 evaluates only 5 binary CIFAR-10 tasks with 100 images each (repeated 3×), and the authors "manually remove images that include c₂ objects or do not include c₁ objects" without reporting how many images were removed per method and per task. This creates an uncontrolled selection variable that could differentially benefit the proposed method. The success rates themselves are modest (5–23 out of 100 for the best method) and highly variable across tasks. No comparison to standard adversarial attack methods (e.g., PGD, AutoAttack) is provided to contextualize whether this constitutes meaningful "model deception," nor is there evaluation on target models other than the same CLIP backbone used to derive neuron explanations.

### Minor

- **Polysemanticity is asserted primarily through cherry-picked examples**: Table 2 and Figure 5 show a handful of neurons with qualitatively diverse text tokens and top-activating images. There is no systematic quantification—how many neurons are polysemantic, how many concepts per neuron, how semantically distant the concepts are—making it difficult to assess how prevalent and meaningful this phenomenon truly is versus arising from the rank-1 approximation blending unrelated signals.

- **The segmentation improvement over TextSpan is marginal (59.0 vs 58.1 mIoU, 84.9 vs 84.1 mAP) without error bars or significance tests**: Without ablations isolating whether the improvement comes from the second-order lens specifically versus simply using more neurons or different neuron selection criteria, it is hard to attribute the gain to the proposed mechanism.

- **Experiments are limited primarily to ViT-B-32**: While Appendix A.1 mentions ViT-L-14 results, the main paper overwhelmingly focuses on one model, and it remains unclear how well findings generalize to other CLIP variants (e.g., ResNet encoders, different patch sizes, OpenCLIP models) or other vision-language models entirely.

### Trivial
None.

## Nice-to-Haves

- A per-neuron explained variance analysis for the rank-1 approximation would significantly strengthen confidence in the text decomposition approach.
- Reporting the fraction of generated images that were manually filtered out in the adversarial evaluation, broken down by method and task, would make the results more transparent.
- An ablation comparing neuron selection by indirect effects vs. second-order effects for the segmentation task would clarify the specific contribution of the proposed lens.
- Comparing against at least one standard adversarial attack method would contextualize the effectiveness of the semantic adversarial examples.

## Removed Points

- **"The adversarial attack is not compared to optimization-based methods like PGD"**: The paper explicitly positions its adversarial examples as *semantic*—they lie on the natural image manifold, which is fundamentally different from pixel-level perturbation attacks. Comparing to PGD would be an apples-to-oranges comparison against a different threat model. *(Removed: the paper's contribution is a different type of adversarial attack, and criticizing the lack of comparison to a different paradigm is scope creep.)*

- **"The text pool may not be diverse enough to capture 'general' image properties"**: This is a generic concern that applies to any dictionary-based decomposition method. The paper already studies three pools (10k words, 30k words, and ~28k ImageNet descriptions) and shows the gap narrows with more descriptions (Fig. 4), providing direct evidence. *(Weakened to nice-to-have from major, as the paper partially addresses this.)*

- **"No comparison to modern zero-shot segmenters like MaskCLIP or Grounded-SAM"**: The paper evaluates on ImageNet-Segmentation using attribution-based methods, which is the appropriate comparison category for *explainability-based* zero-shot segmentation. Grounded-SAM and MaskCLIP use different paradigms (additional supervision, different architectures). *(Removed: inappropriate comparison across paradigms.)*

- **"The derivation ignores layer normalization"**: The paper explicitly addresses this in a footnote and Appendix A.6. This is a standard simplification in mechanistic interpretability work that studies transformer circuits. *(Removed: the paper already addresses this.)*

- **"Polysemanticity of neurons is not novel"**: While the phenomenon itself is known (cited: Elhage et al., 2022), the specific contribution here is revealing polysemanticity *through second-order effects* in CLIP neurons and exploiting it for adversarial attacks. The novelty claim is in the method and application, not the phenomenon. *(Removed: the paper doesn't claim to have discovered polysemanticity.)*

- **"Manual filtering of adversarial images introduces confirmation bias"**: The manual removal of images that don't satisfy semantic constraints (containing c₂ or not containing c₁) is a reasonable quality control for *semantic* adversarial evaluation—you can't claim an image is a semantic adversarial example if it literally contains the wrong object. The concern about transparency (not reporting how many were removed) is kept as a minor point above. *(Removed as a major concern; the filtering is semantically justified, though transparency about counts is still desirable.)*

- **"Neuron-neuron interactions are not modeled"**: The paper acknowledges this as future work. This is a genuine limitation but is standard for a first analysis—no single paper can model all interactions. *(Removed: acknowledged limitation that is standard to defer.)*

## Novel Insights

The second-order lens reveals an interesting structural property of CLIP's late-layer neurons: their influence on the output is channeled almost entirely through a small number of attention heads and can be well-approximated by a single direction in the shared text-image space. This suggests that late-layer neurons in CLIP operate as "write heads" that inject sparse, directionally coherent signals into the attention circuit—functionally similar to key-value memories—which attention then routes to the output. This structural insight—that neurons are both more interpretable (rank-1) and more selective (<2% of images) than first-order or indirect analysis would suggest—has implications for how we think about MLP layers in vision transformers more broadly.

## Suggestions

- Report per-neuron explained variance distributions for the rank-1 approximation to validate that the sparse decomposition is not merely a lossy projection.
- Add an ablation for the segmentation task using neurons selected by indirect effects rather than second-order effects to isolate the contribution of the proposed lens.
- For the adversarial evaluation, report the total number of generated images, how many passed manual filtering, and success rates both before and after filtering.

## Score and Decision

**Calibration reasoning:** The closest comparison paper is TextSpan (Gandelsman et al., 2024), which received all 8s and was accepted as an oral. That paper had a cleaner formulation (first-order effects only), stronger segmentation results, and more thorough evaluation, but did not address neuron interpretability. The current paper extends that work to neurons via second-order effects—a harder and more novel problem—with solid qualitative results but weaker quantitative validation. The SAE-for-CLIP paper (PatchSAE) scored 6s and was accepted as a poster despite concerns about insufficient quantitative evaluation. The NeuronPath paper scored 6s across the board. The INViTE paper (similar CLIP interpretability work with applications) scored 5s and 8s, accepted as poster.

This paper sits between TextSpan (8) and the weaker interpretability papers (5-6). It has genuinely novel ideas (second-order lens, rank-1 neuron effects, polysemy exploitation for adversarial attacks) but with significant gaps in validation: the rank-1 approximation is under-justified for interpretive purposes (only 48.2% variance explained, validated via classification accuracy alone), the adversarial evaluation is narrow and manually curated with limited transparency, and the segmentation improvement over the prior work is marginal. These are real but not fatal weaknesses—they don't invalidate the core findings, but they do limit confidence in the strength of the conclusions.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>