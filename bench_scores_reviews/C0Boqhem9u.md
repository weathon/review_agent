## Summary

LinBridge is a learnable framework for interpreting nonlinear neural encoding models by factorizing their Jacobian matrices into a stable **linear inherent component** (JM_inherent) and sample-selective **nonlinear mapping biases** (ΔJM). Using a self-supervised contrastive learning strategy (InfoNCE), LinBridge is trained on the Jacobian matrices of the test set of a nonlinear encoding model. Validated on visual fMRI data (NSD dataset) with CLIP-ViT features, the authors show that (1) JM_inherent achieves R² patterns highly correlated with those of the nonlinear model, and (2) the ΔJM-based AFD metric reveals hierarchically increasing nonlinearity from primary to tertiary visual cortex.

---

## Strengths

- **Novel decomposition concept for NeuroAI interpretability.** The idea of factorizing sample-specific Jacobians into a shared "inherent" linear component and per-sample nonlinear bias is a concrete and principled approach to bridging the interpretability-performance trade-off in neural encoding — something most prior work handles only by sticking to fully linear models or leaving nonlinear models as black boxes.

- **Strong neuroscientific face validity via a novel metric.** The AFD-based finding that nonlinearity increases from PVC → SVC → TVC (Figures 5b–d) is not merely confirmatory; it provides a quantitative, voxel-level characterization of nonlinear encoding gradients within a unified framework, which goes beyond the qualitative statements in prior literature (e.g., Güçlü & van Gerven, 2015).

- **Demonstrated stability of JM_inherent across batch sizes.** Figure 4(a) shows Pearson correlations approaching 1 even for small batch sizes (down to 16), which is a concrete and falsifiable validation that the extracted component is reproducible and not an artifact of the full test set.

- **Large-scale, well-controlled experimental setup.** The NSD dataset with 8,000 training / 1,000 test samples from 4 subjects who completed the full protocol provides a solid empirical grounding. The use of FDR-corrected significance testing (P < 0.05, 200 bootstrap iterations) follows established neuroscience norms.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Validation limited to a 2-layer MLP encoding model, which severely understates the framework's claimed generality.** The paper positions LinBridge as applicable to "various nonlinear encoding models," yet the only encoding architecture tested is two fully connected layers with a single ReLU (Section 3.2), which has trivially bounded nonlinearity. It is not clear that insights or Jacobian structures from this shallow model generalize to deeper CNNs, transformers, or the multi-layer ResNets that are standard in modern neural encoding literature. The paper itself acknowledges this in Section 5 as a limitation, but deferring this to future work is insufficient at ICLR given that the generality claim is central. This concern is especially acute because the method's novelty depends on there being meaningful, complex nonlinearity to extract.

- **Optimization soundness of the contrastive objective requires justification.** ΔJM is defined as JM − JM_inherent (Eq. 3), making it a function of the very variable being optimized. The InfoNCE loss (Eq. 7) simultaneously pulls JM_inherent toward JM (positive pairs) and pushes it away from ΔJM (negative pairs). As JM_inherent converges toward JM, ΔJM → 0, and the negative samples vanish — creating a potentially degenerate optimization landscape. The L1 regularization on ΔJM (Eq. 8) could counteract collapse, but this interaction is not analyzed. No ablation study or convergence analysis is provided to rule out trivial solutions (e.g., JM_inherent collapsing toward zero, or toward the mean of JM without meaningful separation).

- **The mechanism by which JM_inherent generates R² predictions is not explained.** Section 4.2 states that JM_inherent "achieves comparable or even superior performance in specific brain regions" with R² correlations ~0.997 vs. the nonlinear model. But JM_inherent is a Jacobian (a derivative), not a function — using it to predict brain activity requires treating it as a set of linear weights (i.e., ŷ ≈ JM_inherent · x). This implicit use is never made explicit. Without clarification of this forward-pass computation, the reader cannot assess whether the "comparable performance" claim is methodologically sound or conflates different quantities. This matters because if JM_inherent is simply the mean Jacobian used as a linear map, the result should be compared against a directly trained ridge regression baseline to isolate LinBridge's contribution.

### Minor

- **AFD metric is dependent on t-SNE ordering of stimuli.** The absolute value of the first derivative (AFD) is computed from polynomial fits over samples sorted by their 1D t-SNE projection of CLIP features. While the paper shows the semantic ordering is intuitive (Figure 14–16), AFD computed under different orderings (e.g., random permutation, or sorted by low-level features like luminance) could yield different nonlinearity estimates. No sensitivity analysis or comparison across orderings is provided. This makes it difficult to distinguish whether AFD measures *neural* nonlinearity or *stimulus-space* structure.

- **Cross-subject statistics in the main text are insufficient.** All main-text figures and R² values are reported for Subject 2 only, with other subjects relegated to appendices. For neuroscience claims about hierarchical nonlinearity to be credible, the main text should at minimum report mean ± SD across all 4 subjects for key metrics (e.g., AFD distributions, R² correlations).

- **No ablation comparing contrastive learning to simpler Jacobian aggregation.** The contrastive framework is the methodological core of LinBridge, but no baseline is tested (e.g., mean Jacobian across samples, PCA of Jacobians). Without this, it is unclear whether the contrastive training adds value over straightforward averaging, which would significantly weaken the methodological contribution.

### Tiny

- The notation $\mathbf{JM}_k \in \mathbb{R}^{d \times 1 \times p}$ (Eq. 2) is non-standard for Jacobians (conventionally $\mathbb{R}^{p \times d}$). While internally consistent, the transposition and extra dimension should be explicitly justified to avoid confusion.

- Figure 4(a)'s x-axis caption shows batch sizes {256, 512, 1024, 2048} but the text describes {16, 32, 64, 128, 256, 512}. This inconsistency between the figure and text should be corrected.

---

## Nice-to-Haves

- **Receptive field / weight visualization for V1 voxels.** Displaying the spatial structure of JM_inherent weights for early visual cortex (e.g., checking for Gabor-like tuning) would provide strong biological plausibility support for the interpretation claims, grounding "inherent linear component" in known neuroscience.

- **Synthetic ground truth experiment.** Generating synthetic data with a known ground-truth linear + nonlinear decomposition would allow direct quantitative verification of whether LinBridge recovers the correct factorization, rather than relying solely on face validity.

- **Semantic category breakdown of ΔJM.** Showing how the nonlinear bias ΔJM varies across semantic categories (e.g., faces vs. scenes) in higher visual areas would connect the methodological finding to concrete neuroscientific content, making the contribution more interpretable.

- **Application to a deeper or pre-trained encoding model (e.g., ResNet or fine-tuned ViT).** Even a single demonstration on a deeper architecture would substantially strengthen the generality claim without requiring a full new set of experiments. This is noted as a limitation but would transform the paper from a proof-of-concept to a general tool.

- **Comparison to ridge regression baseline.** Testing whether JM_inherent-derived predictions are meaningfully better than a directly-fitted ridge regression (which is also a linear model of CLIP-ViT features) would clarify the added value of LinBridge over the existing linear encoding baseline.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Contradictory findings" (Harsh Critic, Section 4.1 vs. 4.2):** The critic argues that if JM_inherent achieves R² "comparable" to the nonlinear model, this contradicts the premise that nonlinearity is needed. This is a misreading: Figure 4(c) reports the *correlation between per-voxel R² maps* (≈0.997), not that JM_inherent achieves the same absolute R². The nonlinear model achieves higher absolute R² and activates more voxels (Section 4.1). JM_inherent is derived *from* the nonlinear model's Jacobians and inherits its learned structure — it is not a standalone linear model trained from scratch. The two findings are not contradictory.

- **"Ambiguity in Training Procedure / Data Leakage" (Review 2):** LinBridge is trained on Jacobians computed from the test set in a self-supervised manner — no ground-truth fMRI labels are used during LinBridge training. This is an interpretability post-hoc analysis tool, not a predictive model being evaluated on hold-out data. The concern about "test set usage" conflates standard supervised overfitting with a self-supervised factorization task. The generalization question (does JM_inherent transfer to unseen batches?) is addressed by Figure 4(a)'s batch-size stability analysis.

- **Request for theoretical convergence proofs (Spark Finder):** Demanding formal convergence guarantees for a contrastive learning objective in an empirical neuroscience systems paper is not a standard expectation in this field. Moved to Nice-to-Haves if anything.

- **Criticism of Jacobian dimension notation as an "error":** The notation $\mathbb{R}^{d \times 1 \times p}$ is non-standard but consistent with the tensor structure used throughout; it is not a mathematical mistake. Downgraded to a tiny stylistic note.

---

## Novel Insights

The most genuinely novel observation that emerges across the three reviews — but is not fully articulated in the paper itself — is the following: the near-perfect R² correlation between JM_inherent and the full nonlinear model implies that the dominant *spatial structure* of visual encoding is essentially linear even when using a nonlinear model, and that the nonlinearity primarily acts as a sample-by-sample *gain modulation* captured by ΔJM. If this interpretation is correct, LinBridge is not merely an interpretability wrapper but a tool that reveals a specific functional form of nonlinearity in visual cortex: linear spatial filters modulated by stimulus-specific nonlinear weights. This would connect naturally to divisive normalization and gain-field models in computational neuroscience, a connection that would substantially elevate the paper's contribution if made explicit and tested.

---

## Suggestions

1. **Add one deeper encoding model** (e.g., a 4–6 layer CNN or a fine-tuned ViT head) to Section 4 experiments, even if briefly, to empirically support the "various nonlinear encoding models" claim in the abstract. This is the single most impactful change the authors could make.

2. **Clarify the JM_inherent-to-R² pipeline explicitly** (Section 4.2): state precisely how JM_inherent is used to generate voxel-level predictions on the test set, and add a ridge regression baseline using the same CLIP-ViT features for direct comparison.

3. **Add an optimization sanity check for the contrastive loss**: plot ΔJM norm, cosine similarity between JM_inherent and JM, and InfoNCE loss value over training iterations to verify the optimization does not collapse or trivially satisfy the objective.

4. **Include a sensitivity analysis for AFD** using at least one alternative sample ordering (e.g., random permutation or luminance-based) to demonstrate the metric is driven by neural structure rather than t-SNE artifact.

5. **Report cross-subject statistics** (mean ± SD across 4 subjects) for the AFD histograms and the R² correlation results in the main text, with the per-subject breakdowns remaining in the appendix.

---

**Overall assessment:** LinBridge introduces a conceptually appealing and technically coherent framework for interpretable nonlinear neural encoding, with genuine methodological novelty in the Jacobian factorization approach and a novel neuroscientific finding on hierarchical nonlinearity. However, the paper is currently a solid proof-of-concept rather than a fully validated framework: the exclusive use of a 2-layer MLP encoder, the unexplained contrastive loss dynamics, and the ambiguity in the R² comparison mechanism collectively leave the core claims empirically under-supported at ICLR standards. Novelty is moderate-to-good; technical soundness has identifiable gaps; empirical support is narrow; significance is high *if* the framework generalizes; clarity is generally adequate but has key missing details in the methodology.