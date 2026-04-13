=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
## Summary
This paper proposes a Disentangled Representation Learning (DRL) framework for microscopy image classification that transfers a weakly-supervised VAE (Ada-GVAE) trained on synthetic data (Texture-dSprites) to real target datasets. The key methodological choice is using pretrained DINO ViT features as input to the VAE rather than raw RGB pixels. The approach is evaluated on four microscopy benchmarks (plankton, yeast vacuoles, human cells), showing improved classification accuracy over raw-pixel baselines and providing qualitative interpretability through correlation with hand-crafted features.

## Strengths
- **Effective input representation choice:** The empirical finding that DINO features (Φ) consistently outperform raw RGB across all four datasets is meaningful (e.g., Lensless: 77.46% → 94.62% MLP with finetuning; Table 1). The pretrained features appear to provide a representation space more amenable to disentanglement transfer.
- **Practical interpretability demonstration:** Section 3.6 provides a concrete example of how disentangled factors can distinguish between novel classes and environmental perturbations in plankton anomaly detection. The analysis showing Arcella samples differ from Eupotes in Texture-Shape space but overlap in Color-Scale space (Figure 7) illustrates potential utility.
- **Correlation with hand-crafted features (Lensless):** Figure 5 shows the learned "Scale" dimension correlates strongly with hand-crafted scale (r=0.86), providing some empirical validation that transferred factors retain semantic meaning on target data.

## Weaknesses
- **Disentanglement evaluation on Source rather than Target:** The disentanglement metrics (MIG, DCI, OMES) in Section 3.5 are computed on Texture-dSprites after finetuning on target data, not on the target microscopy data itself. The paper acknowledges: "Since the real-world Target Datasets do not have any labels of the FoV, we evaluate the disentanglement on Texture-dSprites." This evaluates whether the model *retains* source disentanglement, not whether it *acquires* meaningful disentanglement on microscopy data. The Figure 5 correlation analysis (for Lensless only) partially addresses this but does not substitute for proper target-domain evaluation.
- **Missing modern baselines:** The paper compares only against the RGB-input variant of the same pipeline and against hand-crafted features from original dataset papers. There are no comparisons against: (1) fine-tuned DINO/DINOv2 with linear probes, (2) other interpretable methods (prototype networks, concept bottleneck models), or (3) standard XAI methods. Without knowing what accuracy is achievable without disentanglement, the "good trade-off" claim is difficult to evaluate.
- **Cost of disentanglement not shown in main text:** The Discussion acknowledges an ablation (Appendix A.2.5) showing that "for WHOI15, the disentanglement degrades the classification performances" when using raw Φ features directly. This critical comparison—quantifying the accuracy cost of enforcing disentanglement—is relegated to the appendix rather than presented prominently.
- **Underperformance on Sipakmed:** The best result (72.98%, Table 4) is below the 2018 hand-crafted-feature baseline (78.92%). The authors attribute this to FoV mismatch, noting "an ad-hoc Source dataset to take into account the FoVs of the separated parts of the cell would be useful." This highlights that performance depends heavily on source-target alignment, which is not controllable when target FoVs are unknown.
- **Overstated novelty claim:** The introduction claims "this work represents the first application of DRL to real-world datasets." DRL has been applied to real datasets (CelebA, Cars3D, SmallNORB) in prior literature. The qualifier "with unknown FoVs" would make this accurate, but as stated it is incorrect.
- **Hyperparameter choices not justified:** The latent dimension is fixed at 10 across all datasets (which have 4-15 classes with varying complexity), β is only varied in {1, 2}, and the threshold for pruning "inactive" dimensions (Section 3.3) is never specified.

## Nice-to-Haves
- Latent traversals on real microscopy images to visually demonstrate that individual dimensions correspond to interpretable factors (a standard practice in DRL papers)
- Correlation analysis between latent dimensions and hand-crafted features for all four target datasets, not just Lensless
- Comparison with fine-tuned DINO as a simple classification baseline to establish the accuracy upper bound

## Removed Points
*These points are flagged to be removed, treat them with caution.*

- **Label leakage concern in WHOI15:** The reviewer suggests that stratified splitting "maintaining a balanced number of samples per class" might introduce distribution shift. This is standard stratified splitting practice and not a valid concern.

- **No statistical significance tests:** While the paper reports mean ± std over 20 models without formal significance testing, the standard deviations are small and differences appear substantial. This is a minor methodological concern at ICLR standards.

- **Domain gap between Source and Target not quantified:** The paper discusses this qualitatively (e.g., for Sipakmed). While quantitative analysis would strengthen the paper, this is beyond the stated scope.

- **"No discussion of failure modes in medical context":** This criticism asks for broader impact analysis beyond the paper's scope. The method is presented as a research contribution, not a deployed clinical tool.

## Novel Insights
The central tension in this work—evaluating disentanglement on a synthetic source dataset because target FoVs are unknown—reveals an underappreciated problem in transfer-based DRL: when FoVs are not annotated on real data, validating that the transferred representation is genuinely disentangled *with respect to target structure* becomes circular. The paper partially addresses this via correlation with hand-crafted features, but this only works when such features exist and capture relevant factors. For domains without established morphological descriptors, this validation approach would not generalize.

## Suggestions
- Rename Section 3.5 to clarify that disentanglement is evaluated on Source as a proxy, and explicitly state this as a limitation of the evaluation protocol.
- Move the ablation comparing Φ features directly (without disentanglement bottleneck) to the main paper, showing the accuracy-disentanglement trade-off explicitly for all datasets.
- Correct the "first application of DRL to real-world datasets" claim to specify "with unknown or partially known factors of variation."
- Report the threshold for pruning inactive latent dimensions in Section 3.3.
- Add a comparison with a fine-tuned DINO linear probe baseline to establish the accuracy cost of the disentanglement bottleneck.

# Actual Human Scores
Individual reviewer scores: [3.0, 1.0, 3.0, 3.0]
Average score: 2.5
Binary outcome: Reject
