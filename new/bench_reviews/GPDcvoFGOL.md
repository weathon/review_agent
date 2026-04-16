## Summary

The paper introduces the "second-order lens" for interpreting individual neurons in CLIP-ViT by analyzing how their effects propagate through subsequent attention heads to the output, rather than through direct or indirect effects. The authors find that these second-order effects are concentrated in late layers, highly selective (significant for <2% of images), and approximately rank-1 in the joint text-image space. They decompose each neuron's direction into a sparse set of text representations, revealing polysemantic neuron behavior, and demonstrate two applications: mass-producing semantic adversarial examples by exploiting spurious concept overlaps, and zero-shot segmentation that outperforms prior methods.

## Strengths

1. **Well-motivated and principled formulation.** The second-order lens directly addresses the failure modes of existing approaches: direct effects are negligible for MLP neurons, and indirect effects are obscured by self-repair (Table 1 clearly shows 48.2% variance explained by PC1 for second-order effects vs. 11.0% for indirect). The decomposition in Eq. (5) cleanly separates attention-weighted activations from input-independent terms, enabling tractable analysis.

2. **Useful empirical characterization of second-order effects.** The three key properties—late-layer concentration, sparsity, and rank-1 approximation—are each supported by targeted ablation experiments (Figure 3). The "w/o large norm" vs. "w/o small norm" comparison effectively demonstrates selectivity, and the reconstruction-from-PC1 experiment shows negligible accuracy drop, providing initial evidence for the rank-1 structure.

3. **Compelling applications that go beyond characterization.** The semantic adversarial pipeline is creative and novel—exploiting polysemanticity to find spurious concept overlaps and generating on-manifold adversarial images. The method significantly outperforms all baselines on every task (Table 3), including the ship→truck task where no baseline succeeds at all. The zero-shot segmentation also outperforms all prior methods across all metrics (Table 4).

4. **Clear demonstration of polysemanticity.** Table 2 and Figure 5 effectively illustrate how single neurons correspond to multiple unrelated concepts (e.g., Neuron #2914 encodes both "yacht" and "cabriolet"), corroborating and concretely demonstrating superposition theory in a real production model.

## Weaknesses

### Major:

- **The rank-1 approximation is central to the method but only weakly evidenced.** The claim that each neuron's second-order effect is approximately rank-1 (≈ x_n^l(I) · r_n^l + b_n^l) underpins the entire sparse text decomposition pipeline. However, the evidence is limited to: (a) negligible accuracy drop when reconstructing from PC1 at the layer level (Figure 3), and (b) 48.2% variance explained by PC1 on aggregate for second-order effects (Table 1). The paper never reports per-neuron variance explained, so it is impossible to tell whether many neurons have genuinely rank-1 effects or whether the aggregate metric masks significant deviations. Additionally, the pre-selection of S_n^l as "images with largest second-order effect norms" biases toward finding a single dominant direction. Without per-neuron variance analysis, it is unclear how lossy the rank-1 approximation is for individual neurons or whether the extracted text descriptions capture the full behavior of poorly-approximated neurons.

- **No systematic evaluation of neuron description quality.** The paper claims to "interpret the function of individual neurons" and that sparse text descriptions "reveal" polysemantic behavior, but the only quantitative evaluation is ImageNet accuracy under collective reconstruction (Figure 4), which validates functional preservation of the *aggregate* of all neuron directions—not whether individual neuron descriptions are faithful or meaningful. One could rotate the text basis across neurons and achieve similar downstream accuracy with entirely different per-neuron semantics. The qualitative examples in Table 2 are hand-picked (4 neurons out of thousands), and there is no analysis of what fraction of neurons yield coherent vs. degenerate decompositions. A human evaluation or automated metric (e.g., correlation between predicted and actual top-activating images) would substantially strengthen the core interpretability claim.

- **Limited scope of adversarial evaluation.** The adversarial experiments cover only 5 binary CIFAR-10 tasks with 100 images per task, and success rates are modest (5.3–22.7%). There is no controlled experiment showing that images of class c₁ *without* the mined spurious words are classified correctly at higher rates—the causal role of the discovered cues is asserted but not independently verified. The baselines (random neurons, indirect effects, similar words) are all constructed through the same pipeline, and no comparison is made to existing semantic or on-manifold adversarial attack methods. The manual filtering of generated images introduces potential experimenter bias into what remains a small-sample evaluation.

### Minor:

- **Hyperparameter choices are not ablated.** Key design decisions—the number of top neurons (100 for adversarial, 200 for segmentation), the binarization threshold (0.5), the specific layers (8–10)—are stated but not systematically varied. The segmentation improvement over TextSpan is small (+0.9 mIoU, +0.8 mAP), making it unclear whether these gains are robust to hyperparameter changes.

- **Evaluation limited to CLIP ViT architectures.** All experiments use CLIP ViT-B-32 (with ViT-L-14 in the appendix). It is unknown whether the rank-1 and sparsity properties hold for ResNet-based CLIP, other vision-language models, or even different ViT configurations. The second-order lens relies on the class-token output mechanism and residual stream structure, which may not transfer straightforwardly.

- **Ignoring layer normalization and query/key effects.** The derivation explicitly ignores layer normalization (deferred to Appendix A.6, not available in the main paper) and the effects of neurons on queries and keys in subsequent attention layers. The authors acknowledge these limitations but do not quantify their impact; if higher-order effects through query/key modifications are substantial, the second-order lens may miss important neuron functions.

## Nice-to-Haves

- **Human evaluation of text descriptions:** Have annotators rate whether the extracted text descriptions match top-activating images for a random sample of neurons. This is standard in the interpretability literature and would directly validate the core claim.

- **Sparse autoencoder (SAE) baseline:** SAEs are the dominant approach for decomposing polysemantic neurons. Comparing second-order lens + OMP against SAE decomposition on the same model would clarify the relative merits of the approach.

- **Analysis of rank-1 approximation quality per neuron:** Reporting the distribution of PC1 variance explained across neurons, not just the aggregate, would clarify how universally the rank-1 assumption holds and identify where it breaks down.

- **Multi-class adversarial evaluation:** Extending adversarial experiments beyond 5 binary CIFAR-10 tasks to ImageNet-scale classification would better contextualize the attack's practical significance.

## Removed Points

- **"The second-order effect is an ad-hoc linear proxy, not rigorously defined" (Harsh Critic #1):** While the derivation simplifies away layer normalization and holds attention weights fixed, the paper explicitly acknowledges these simplifications and provides empirical validation through ablation studies (Figure 3, Table 1). The approach follows established mechanistic interpretability methodology (Elhage et al., 2021), and the paper demonstrates that the construct has practical utility. Calling it "ad-hoc" overstates the concern; it is a well-defined path-specific attribution, though incomplete. Downgraded to a minor concern about ignored terms.

- **"No comparison to SAE baselines" (Spark):** While a natural baseline, SAEs operate on activation patterns directly rather than on second-order output effects. The paper's contribution is specifically about neuron *contributions to the output representation through attention*, not about activation decomposition per se. SAEs would address a different question. Moved to Nice-to-Have.

- **"No causal validation linking text concepts to neuron behavior" (Spark):** This is essentially asking for intervention-based verification, which is a high bar for an empirical interpretability paper. The paper already provides some causal evidence through the adversarial attack pipeline, which demonstrates that mined concepts can change model behavior. Moved to Nice-to-Have.

- **"Self-repair claim under-investigated — only one layer comparison" (Neutral Reviewer #6):** The paper presents Figure 3 showing ablation across all layers and Table 1 as a summary comparison. The comparison focuses on layer 9 because that's where effects concentrate (as shown by the full sweep in Figure 3). This is a standard presentation choice, not a methodological gap.

- **"Manual filtering introduces experimenter bias" (Harsh Critic #4):** The filtering removes images that visually contain the wrong class or lack the correct class—this is a reasonable quality control step that makes the evaluation cleaner, not a source of bias toward success. The paper reports results after filtering and is transparent about it. The concern is noted but does not constitute a methodological flaw.

- **"Segmentation improvement is marginal / may be due to model-specific tuning" (Harsh Critic #5, Neutral #5):** The improvement over TextSpan (0.9 mIoU, 0.8 mAP) is admittedly small, but the method outperforms all baselines across all metrics. Whether the gain is due to "heavier model-specific tuning" is debatable—TextSpan also uses CLIP's internal structure. The concern about hyperparameter sensitivity is valid but better addressed as a minor weakness about missing ablations.

- **"Overclaiming in the abstract that direct/indirect effects 'fail to capture neurons' function'" (Harsh Critic):** The paper demonstrates in Table 1 and Figure 3 that mean-ablating indirect effects causes only a 52.3% accuracy drop (vs. 29.6% for second-order effects) and that PC1 explains only 11.0% variance for indirect effects. The claim that these other lenses fail to capture neuron function is supported by this evidence, even if the word "fail" is slightly strong. This is a matter of emphasis, not a factual error.

- **"Using IN class descriptions introduces implicit supervision" (Harsh Critic, Section 4 notes):** The paper tests multiple pools including generic common words (10k, 30k) and shows that all pools can reconstruct well (Figure 4). The IN descriptions pool is one variant, and the paper is transparent about it. This does not undermine the interpretability framework.

## Novel Insights

The paper's most interesting insight is that MLP neurons in CLIP can be understood through their *routing* via subsequent attention heads, rather than through their direct residual stream contribution or through ablation-based indirect effects. The finding that this routing is highly selective (<2% of images) and approximately one-dimensional in the output space is non-obvious and practically useful—it enables a tractable decomposition into text directions that would be infeasible for a high-rank effect. The adversarial application cleverly exploits the polysemantic nature revealed by these decompositions: by finding spuriously overlapping concepts in individual neurons, one can generate semantic adversarial examples that are on the natural image manifold, rather than relying on pixel-level perturbations. This provides concrete evidence that mechanistic interpretability can reveal exploitable model vulnerabilities, not just descriptive insights.

## Suggestions

1. **Report per-neuron variance explained by PC1** across all neurons and layers, not just the aggregate. This directly addresses the central claim and could be a simple histogram or percentile table in an appendix.

2. **Conduct a random-sample evaluation of text decompositions.** Instead of cherry-picked examples, show decompositions for a randomly sampled set of neurons and assess coherence, or compute a quantitative metric like rank correlation between predicted and actual top-activating class labels.

3. **Add a simple controlled experiment for the adversarial attack:** Generate images of class c₁ *without* any spurious words and compare classification rates. This would directly establish the causal role of the mined concepts.

## Score and Decision

I compared this paper against the following calibration anchors:
- **TextSpan (same authors, direct predecessor)**: Scores 8,8,8,8 (Oral). Strong, well-validated work on CLIP attention head interpretation. The current paper extends to neurons, adds two applications, and provides a novel lens. Weaker in evaluation rigor (no per-neuron validation, limited adversarial scope) but conceptually more ambitious.
- **PatchSAE**: Scores 6,6,6,8 (Poster). Similar domain (sparse decomposition of CLIP features) with similar evaluation concerns about lacking quantitative validation of interpretability quality.
- **INViTE**: Scores 5,5,8,3 (Poster). Similar concern about limited baselines and vocabulary dependence.
- **Describe-and-Dissect**: Scores 3,6,5,5 (Reject). Weaker paper with similar neuron description ambitions but less methodological rigor.

This paper is clearly above the Describe-and-Dissect tier—it has a well-defined analytical framework, strong motivation, two working applications, and clear empirical characterization. It is slightly below TextSpan in evaluation thoroughness (TextSpan had more extensive per-component validation) but adds genuine novelty. It is roughly comparable to PatchSAE in overall quality: a solid contribution with meaningful but incomplete evaluation. The main weaknesses—the thin rank-1 evidence, lack of per-neuron validation, and narrow adversarial evaluation—are real but do not undermine the core contribution of introducing the second-order lens and demonstrating its utility.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>