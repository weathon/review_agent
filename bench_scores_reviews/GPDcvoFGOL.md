## Summary
This paper introduces a "second-order lens" for interpreting individual neurons in CLIP-ViT by analyzing the flow of their activations through subsequent attention heads to the output, rather than relying on direct effects (negligible in CLIP's MLP layers) or indirect ablation effects (obscured by self-repair). The authors show that these second-order effects are sparse (significant for <2% of images per neuron), approximately rank-1 in CLIP's joint text-image space, and can be decomposed into sparse sets of text descriptions via orthogonal matching pursuit — revealing polysemantic behavior. The method is applied to two downstream tasks: mass-generation of semantic adversarial examples exploiting spurious neuron polysemy, and zero-shot image segmentation via neuron activation aggregation.

---

## Strengths

- **Principled motivation for second-order effects.** The paper makes a compelling, specific case for *why* both direct and indirect effects fail for CLIP neurons: direct effects are near-constant (citing Gandelsman et al. 2024), and self-repair obscures ablation-based indirect effects. Table 1 quantifies this concretely — mean-ablating indirect effects causes only a 52.3% → ~52% accuracy drop, while second-order ablation causes a 60% → 29.6% drop, making the choice of lens empirically well-motivated rather than arbitrary.

- **The rank-1 approximation is a striking empirical finding.** The observation that $\phi_n^l(I)$ can be replaced by $x_n^l(I)r_n^l + b_n^l$ (a scalar × fixed direction + bias) with negligible classification accuracy loss (Figure 3, "rec. from PC #1") is non-trivial. It implies that each neuron's output-relevant effect lives in a one-dimensional subspace across the images it affects — a property that significantly simplifies subsequent text decomposition. The fact that first PC explains 48.2% variance for second-order effects vs. only 11.0% for indirect effects (Table 1) further distinguishes the lens.

- **Text decomposition faithfully tracks neuron activation.** The qualitative alignment between top-activated images (Figure 5) and their sparse text decompositions (Table 2) is convincing: neurons activate on semantically coherent visual concepts (e.g., Neuron #4 fires on snowy/wintry images and is decomposed into "snowy," "frost," "advent," "closings"). The decomposition generalizes across two conceptually different text pools (common words vs. ChatGPT-generated ImageNet descriptions), suggesting it captures genuine representational structure.

- **Polysemantic neuron behavior linked to a downstream application.** Unlike most interpretability papers that stop at qualitative descriptions, this work directly leverages polysemy for adversarial example generation. The pipeline (neuron decomposition → LLM scene composition → text-to-image generation) is novel and demonstrates that mechanistic understanding can operationally inform adversarial analysis. This is a concrete step beyond qualitative neuron labeling.

- **Zero-shot segmentation improvement via neuron ensembling.** Using the second-order neuron directions to identify class-relevant neurons and averaging their spatial activation maps yields measurable improvements over TextSpan across all three metrics (78.1% vs. 76.5% pixel accuracy, 59.0% vs. 58.1% mIoU, 84.9% vs. 84.1% mAP in Table 4), and the qualitative comparison in Figure 7 shows the method captures more of the full object shape rather than isolated discriminative parts.

---

## Weaknesses

### Fatal
None.

### Major

- **Unresolved tension between 48.2% explained variance and near-lossless reconstruction.** Table 1 shows the first PC of the second-order effect explains only 48.2% of variance, yet Figure 3 shows replacing each neuron's contribution with the rank-1 approximation causes a negligible accuracy drop. This is the central methodological claim of the paper (rank-1 approximation enables text decomposition), and the result is unexplained. The most plausible explanation is that the remaining 51.8% of variance is orthogonal to all classification-relevant directions, but the paper does not verify or even discuss this. Without this explanation, a reader cannot trust that the rank-1 direction faithfully captures the *functionally relevant* component of the neuron, as opposed to accidentally preserving accuracy through a different mechanism. This should be addressed directly — e.g., by showing the residual variance is approximately orthogonal to the class-text directions used in downstream tasks.

- **Missing critical ablation for segmentation: second-order selection vs. direct-effect selection.** The segmentation result (Section 5.2) selects neurons by $|\langle r_n^l, M_{\text{text}}(c_i) \rangle|$ — i.e., using the second-order direction $r_n^l$. But it is unclear whether the improvement over TextSpan comes from using the *second-order effect* to identify neurons vs. simply using the same second-order directions as a better neuron selector. An ablation comparing (a) neurons selected by their second-order direction, (b) neurons selected by their first-order direction, and (c) random neuron selection, all using the same aggregation procedure, is essential to attribute the gain to the second-order lens specifically.

- **Fraction of neurons with interpretable decompositions is unreported.** Table 2 and Figure 5 show 4 neurons out of thousands per layer. There is no analysis of what proportion of neurons yield semantically coherent decompositions versus noisy or degenerate ones. If only a small fraction of neurons are cleanly interpretable, the method's scope as a general neuron interpretability tool would be significantly limited. Even a histogram of top-text cosine similarities, or a categorization of neuron decompositions into interpretable vs. incoherent, would substantially strengthen the claim of generality.

### Minor

- **Low absolute adversarial success rates with limited statistical power.** The best case is 22.7/100 images ("dog→deer"); most tasks are in the 5–8 range. The paper selects 5 out of 45 possible CIFAR-10 class pairs with no justification for representativeness, and only 3 experimental repetitions, yielding standard deviations that span 0 to ±4.5 images — inadequate for such small counts. The manual removal of images (containing $c_2$ objects or lacking $c_1$ objects) is not quantified, so the effective denominator is opaque. These issues together make the adversarial application feel preliminary.

- **No hyperparameter ablation for segmentation.** The design choices "top 200 neurons from layers 8-10" are central to the segmentation result but are not ablated. Why layers 8-10? Why 200? Without this ablation, it is unclear whether the result is robust to these choices or tuned to the benchmark.

- **Layer-norm approximation not theoretically justified (partially addressed).** Footnote 2 and Appendix A.6 address this empirically by showing near-lossless reconstruction, but the derivation's interpretive value formally assumes the approximation holds. The authors should clarify in the main text that the empirical reconstruction accuracy (Figure 3) serves as the primary justification, and briefly characterize the approximation error.

- **The 2% sparsity claim is partially circular.** The threshold is operationalized as the "top 100 images from ~5000 training images" (which is 2% by construction). While the validation-set analysis is distinct, the paper should show a distribution of second-order effect norms across images to confirm 2% is a meaningful natural cutoff rather than a definitional artifact.

### Tiny

- **The framework's restriction to models with a shared text-image space is not acknowledged as a limitation.** The method's key steps (decomposing $r_n^l$ into text directions, selecting class-relevant neurons via text similarity) require that the output space be jointly accessible to a text encoder. This means the approach as presented does not generalize to DINO, MAE, or vision-only transformers. This is worth a sentence in the limitations section to accurately scope the contribution.

- **Computational cost of the adversarial pipeline is unreported.** The attack requires LLaMA3 and DeepFloyd IF inference. A brief runtime or resource statement would help readers assess scalability.

---

## Nice-to-Haves

- **Stability analysis of text decompositions across different text pools.** The paper evaluates accuracy vs. pool size (Figure 4) but does not check whether the *same* words appear across pools for the same neuron. Showing cross-pool stability would strengthen the claim that descriptions reflect intrinsic neuron functionality, not artifacts of the specific vocabulary.

- **Visualization of failure cases in text decompositions.** Cases where a neuron has high second-order norm but the text descriptions are semantically incoherent or irrelevant would help users understand the method's limits.

- **Scaling analysis to ViT-L-14 and larger models.** The paper reports ViT-L-14 in an appendix but does not discuss how the rank-1 approximation quality or neuron interpretability scales with model size. Larger models have more neurons per layer and richer residual streams, and whether second-order effects remain rank-1 there is non-obvious.

- **Quantification of the Q/K pathway exclusion.** The paper excludes neuron effects on attention queries and keys (Section 6). A rough estimate of how large this excluded pathway is, relative to the value pathway, would help calibrate how much mechanistic signal remains uncharacterized.

- **Causal validation of text decompositions.** Showing that ablating specific text concepts reduces the neuron's response on the predicted top images would move the decomposition from correlational to causal.

---

## Removed Points

*These points are flagged as removed; treat them with caution.*

- **Notation inconsistency (superscript $d$ vs. $l$).** The harsh reviewer noted that Section 4 uses superscript $d$ while Section 3 uses $l$. Removed as a pure formatting/style nitpick.

- **Claim that "outperforming previous methods" in the abstract is an overstatement.** The paper does outperform across all three metrics in Table 4. The margin being small is addressed under Major weaknesses; the abstract claim itself is factually accurate.

- **"Similar words" baseline should be replaced by sparse autoencoders (SAEs).** Demanding SAE comparison for the adversarial experiment is scope creep — the existing baselines (random neurons, indirect-effect-based decomposition, direct text-similarity matching) form a coherent ablation of the paper's specific contributions. SAE comparison is a nice-to-have for the interpretability community but is not a standard requirement for an applications paper.

- **Binary CIFAR-10 evaluation should be replaced by 1000-class ImageNet.** The paper's adversarial application is a demonstration of the interpretability framework rather than a standalone attack paper. Demanding ImageNet-scale evaluation imposes requirements outside the paper's stated scope.

- **Segmentation evaluated on only one benchmark (ImageNet-Segmentation) should add PASCAL VOC or COCO.** This is the standard benchmark for zero-shot foreground/background segmentation of ImageNet classes, and is the same benchmark used by all baselines (Chefer et al. 2021, TextSpan). Adding COCO semantic segmentation would be a meaningful extension but goes beyond the paper's stated scope.

- **Text pool diversity/bias analysis.** Criticism about potential biases in ChatGPT-3.5 descriptions is removed as speculative without specific evidence of harm to results; the cross-pool accuracy comparison in Figure 4 provides sufficient validation.

---

## Novel Insights

The most genuinely novel observation across all three reviews is the **functional significance of the rank-1 finding in CLIP's second-order effects**. The first PC of $\phi_n^l$ across $S_n^l$ captures only 48.2% of geometric variance yet preserves essentially all classification-relevant information. This suggests that CLIP's neurons write functionally to a one-dimensional subspace of the joint text-image space, with the remaining geometric variance being "noise" orthogonal to all semantically meaningful directions. If confirmed (see the unresolved tension in Major Weaknesses), this would imply that CLIP's MLP neurons act as rank-1 updates to the shared semantic manifold, which has implications for understanding the distributed code in CLIP beyond this paper's direct contributions. A second insight, less developed but promising: the polysemantic spurious co-activations discovered through this lens (e.g., "dog" neurons that also activate on "elephant" and "value"-sign contexts) represent a qualitatively different vulnerability surface than pixel-space adversarial examples — they are semantic, human-interpretable, and generated without any gradient-based optimization.

---

## Suggestions

1. **Explain the rank-1 variance tension (highest priority).** Add a direct analysis or discussion explaining why 48.2% geometric variance captured by PC#1 nonetheless preserves near-100% of classification accuracy. A simple check: compute the cosine similarity between the residual (φ − rank-1 approximation) and all class-text directions; if this residual is consistently near-orthogonal to the classification-relevant subspace, say so explicitly.

2. **Add the second-order vs. direct-effect neuron selection ablation for segmentation.** Rerun the segmentation pipeline selecting neurons by their first-order direction $Pw^{l,n}$ instead of $r_n^l$, keeping everything else identical. This is the most important single experiment needed to attribute the segmentation gain to the second-order lens specifically.

3. **Report the fraction of neurons with semantically coherent decompositions.** Even a coarse categorization (e.g., >50% top images semantically aligned with top text label = "interpretable") across all neurons in a single layer would substantially strengthen the claim of generality.

4. **Quantify the manual image filtering in the adversarial experiment.** Report how many images were removed per task and run, so that reported success rates are interpretable as fractions of a well-defined denominator.

5. **Explicitly state the CLIP-joint-space requirement as a limitation.** One sentence in Section 6 noting that the method requires a shared text-image representation space (and thus does not directly extend to vision-only models) would accurately scope the contribution.

6. **Ablate the segmentation hyperparameters.** Provide a brief sensitivity analysis of performance to the number of neurons (e.g., 50/100/200/500) and layer range (e.g., layers 7-9 vs. 8-10 vs. 9-11) to establish robustness of the reported result.