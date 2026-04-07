## Summary

This paper proposes ARU-GD+MCD, which integrates Monte Carlo Dropout into an Attention Residual U-Net with Guided Decoder for brain tumor segmentation on MRI scans. The model generates both segmentation predictions and uncertainty maps, evaluated on BraTS 2019 with four MRI modalities. The primary contribution is adding uncertainty estimation capability to an existing architecture, reporting improved Dice scores for Tumor Core (TC) and Enhancing Tumor (ET) regions while generating pixel-wise uncertainty heatmaps.

## Strengths

- **Well-motivated clinical problem**: The paper correctly identifies that clinicians must manually verify predicted tumor boundaries without knowing which regions are unreliable. Uncertainty maps that highlight low-confidence regions address a genuine gap in clinical deployment of segmentation models.

- **Comprehensive architectural description**: Section 3 provides detailed specifications including layer dimensions, activation functions, and dropout placements. The guided decoder mechanism with intermediate outputs (out1, out2, out3) is clearly explained, enabling potential reproduction.

- **Demonstrated improvements in challenging regions**: Table 1 shows TC Dice improving from 0.876 to 0.899 and ET Dice from 0.801 to 0.856 when MCD is added. These regions are clinically important and technically harder to segment due to their smaller size and complex boundaries.

## Weaknesses

- **No quantitative uncertainty evaluation**: The paper's primary claimed contribution is uncertainty estimation, yet uncertainty maps are evaluated solely through visual inspection of Figure 2. There is no Expected Calibration Error (ECE), AUROC for error detection, Brier Score, or reliability diagrams to demonstrate that uncertainty actually correlates with segmentation errors. For a paper centered on uncertainty quantification, this is a critical gap.

- **Unexplained Whole Tumor regression**: ARUNet+GD+MCD achieves 0.886 WT Dice compared to ARUNet+GD's 0.911—a 2.5 point decrease. The paper describes this as "comparable," which mischaracterizes the result. Why stochastic inference improves TC/ET but degrades WT remains unexplained and requires mechanistic analysis.

- **Model selection on training loss**: Section 3.5 states "The best model was saved based on training loss." This non-standard practice risks selecting an overfit checkpoint and may bias reported results. Standard practice uses validation loss for model selection.

- **Missing state-of-the-art baselines**: The comparison includes UNet, Res-UNet, and AG Res-UNet but excludes nnU-Net, which consistently outperforms hand-designed architectures on BraTS and is cited in Section 2.1. No comparison with other uncertainty estimation methods (Deep Ensembles, Test-Time Augmentation) is provided to contextualize the MC Dropout approach.

- **No statistical significance testing**: All results are single-run point estimates. Segmentation models exhibit run-to-run variance, and the claimed improvements (e.g., TC: 0.876→0.899, ET: 0.801→0.856) lack confidence intervals or significance tests over multiple seeds.

- **No ablation study on core hyperparameters**: The dropout rate (0.2), number of MC passes (T=20), and dropout placement (decoder-only) are stated without justification. A proper ablation should verify these choices.

- **2D slice-only approach with limited data**: Using only 25 out of 155 slices (indices 50-98) discards approximately 84% of volumetric data. This may exclude apical and basal tumor extent and deviates from the 3D volumetric evaluation standard for BraTS, limiting clinical relevance and comparability to literature.

## Nice-to-Haves

- **Update to BraTS 2021/2023**: While BraTS 2019 remains valid for methodology validation, newer datasets with ~5× more cases would strengthen impact and relevance.

- **Inference latency analysis**: Quantify the computational cost of 20 MC passes versus the uncertainty benefit, since clinical deployment requires real-time operation.

- **Clinical user study**: Verify whether uncertainty maps actually reduce clinician verification time as claimed in the introduction.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Citation completeness for Maji et al. (2020)**: The harsh critic flagged the citation as "details not specified." While the citation format is incomplete, the reference exists and is verifiable. This is a formatting issue, not a validity concern.

- **Aleatoric vs. epistemic uncertainty criticism**: The paper correctly notes in Section 2.2 that MC Dropout captures epistemic uncertainty. Criticizing the absence of aleatoric uncertainty estimation is scope creep—the paper never claims to address both.

- **Demand for newer BraTS dataset as requirement**: While dated, BraTS 2019 remains a valid benchmark for uncertainty methodology validation. This is a reasonable suggestion but not a fatal flaw for the paper's stated scope.

## Novel Insights

The observation that MC Dropout improves TC and ET performance while degrading WT is intriguing and underexplored. One potential explanation is that dropout's stochastic regularization preferentially benefits smaller, harder-to-segment classes where overfitting is more likely. The uncertainty maps appearing at tumor boundaries align with intuition (boundary regions are inherently ambiguous), but without quantitative correlation analysis, this remains a visual claim rather than a validated finding. The guided decoder's intermediate outputs could theoretically provide multi-scale uncertainty information, but the paper does not explore whether aggregating uncertainty across out1, out2, out3 improves calibration over using only the final output.

## Suggestions

1. **Add quantitative uncertainty metrics**: Compute AUROC for distinguishing correct vs. incorrect predictions using uncertainty, and report ECE or Brier Score. This is essential for a paper claiming uncertainty estimation as its primary contribution.

2. **Explain the WT regression**: Analyze why adding stochastic inference improves small-class performance but degrades large-class performance. Consider whether dropout placement or rate affects classes differently.

3. **Add ablation experiments**: At minimum, test dropout rates (0.1, 0.2, 0.3) and MC pass counts (10, 20, 50) to justify hyperparameter choices.

4. **Use validation loss for model selection**: Re-train and select models based on validation loss to ensure fair comparison.

5. **Report uncertainty-error correlation**: Quantitatively demonstrate that high-uncertainty regions overlap with segmentation errors (false positives/negatives), not just visually.