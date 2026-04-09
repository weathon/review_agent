## Summary

This paper proposes Classifier-Constrained Alternating Training (CCAT) to mitigate modality imbalance in multimodal learning. The key insight is that existing alternating training methods, while reducing encoder-level gradient conflicts, fail to prevent classifier bias toward faster-converging modalities. CCAT addresses this via a two-stage framework: (1) pretraining a shared classifier with bidirectional cross-attention and a modality-contribution regularization term, then (2) freezing this classifier during alternating modality-specific encoder training, with lightweight LoRA modules providing modality-specific adaptation and a sample-level secondary update mechanism for severely imbalanced samples.

## Strengths

- **Well-motivated problem framing:** The paper identifies a concrete, previously underexplored failure mode of alternating training—classifier-level bias entrenchment—and supports it with empirical evidence (Figure 1 showing persistent contribution disparity under MLA). The analogy to class imbalance (Section 3.1) provides a useful conceptual lens that justifies the fixed-classifier design.
- **Principled architectural design with clear ablation support:** Each component (classifier freezing, alternating training, secondary updates, LoRA modules) is systematically ablated in Table 2, demonstrating that all elements contribute to the final performance. The ablation is clean and the degradation patterns are interpretable (e.g., removing LoRA drops CREMA-D from 85.89% to 84.68%).
- **Strong empirical gains on challenging benchmarks:** The reported improvements are substantial, particularly +6.76% on Kinetic-Sound and the large margin on CREMA-D, suggesting the method meaningfully addresses modality imbalance rather than offering marginal tuning benefits.

## Weaknesses

### Major:

- **Missing SOTA baselines in main results table:** Section 4.1 explicitly lists MLA, MMPareto, and LFM as baselines, and Section 4.2 observation (iv) references their unimodal results, yet **Table 1 contains no rows for these methods**. MLA (Zhang et al., 2024) is the most critical omission—it is the direct predecessor that CCAT extends, and its absence makes it impossible to assess the specific contribution of the classifier constraint beyond the alternating mechanism itself. This is a significant gap in the experimental validation for a paper claiming SOTA performance.
- **Disconnect between theoretical analysis and actual method:** Section 3.1 derives the modality imbalance mechanism under the assumption that the fused feature is a linear combination $\mathbf{f} = \gamma_1 \mathbf{f}^{(1)} + \gamma_2 \mathbf{f}^{(2)}$ (Eq. 3), establishing a "theoretical isomorphism" with class imbalance. However, the actual pretraining stage uses **bidirectional cross-attention** (Appendix A.1, Eqs. 14–22), which produces context-aware representations—not a scalar-weighted sum. The paper does not acknowledge or discuss this simplification. While the linear model provides intuition, calling it a "theoretical isomorphism" and a "new theoretical framework" (contribution i) overstates what is essentially an illustrative analogy.

### Minor:

- **Unusually large performance gap on CREMA-D raises baseline fidelity questions:** CCAT achieves 85.89% vs. OGM-GE's 68.14%—a +17.75% absolute improvement. For a well-studied benchmark and a method that modifies training strategy rather than backbone capacity, this gap is anomalous. The paper states encoders are "ResNet18 across all datasets" but does not explicitly confirm that all baseline numbers were re-implemented with identical encoders, training budgets, and preprocessing. If baseline numbers come from original papers with different architectures, the comparison is unfair; if re-implemented, the poor baseline performance requires explanation. Either way, clarification is needed.
- **Insufficient justification for LoRA over alternatives:** LoRA modules are applied to transform features ($\text{LoRA}_m(\mathbf{z}_i^m) = \mathbf{B}_m \mathbf{A}_m \mathbf{z}_i^m$, Eq. 9) before the frozen classifier. The ablation (Table 2) shows LoRA helps (+1.21% on CREMA-D), but does not compare against the natural alternative: simply unfreezing the classifier and allowing full fine-tuning during alternating training. If unfrozen fine-tuning performs comparably, the low-rank constraint and the entire freezing+LoRA design would lose justification. This comparison is essential for validating the core architectural claim.
- **No variance or error bars reported:** Results are averaged over three random seeds (footnote of Table 1) but no standard deviations or confidence intervals are provided. Given the magnitude of claimed improvements, reporting variance is important for assessing robustness.
- **Algorithm 1 notation is confusing regarding contribution estimation:** Line 10 references Eq. (6) for estimating contributions, but Section 3.3 explicitly states that during alternating training "the computation of $c$ follows the same decision-level fusion used in the inference stage," not the cross-attention fusion of Eq. (6). The algorithm and the prose contradict each other, harming reproducibility.

### Trivial:

- **MI estimator lacks bias/variance discussion:** The contribution estimator (Eq. 5) uses an InfoNCE-style formulation with cosine similarity but no learnable temperature or projection heads. While this serves as a regularization term rather than a contrastive objective, the paper provides no discussion of estimator quality or its sensitivity to feature space geometry across modalities.

## Nice-to-Haves

- **Computational cost analysis:** The two-stage training plus sample-level secondary updates increases wall-clock time. Reporting training time and FLOPs relative to single-stage baselines would help practitioners assess the cost-benefit trade-off.
- **Tri-modal or larger-scale evaluation:** The method is evaluated only on bimodal datasets. Testing on a trimodal dataset or a larger-scale benchmark (e.g., AudioSet or full Kinetics) would strengthen claims of scalability.
- **Sensitivity analysis for the pretraining stage:** An ablation comparing the pretrained+unbiased classifier initialization versus a randomly initialized frozen classifier would isolate whether gains come from the quality of initialization or simply from the freezing constraint itself.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"How do gradients propagate with frozen classifier and no LoRA?" (from Spark Finder):** Table 2 Row 4 (Fix ✓, LoRA ✗) achieved 84.68%. The concern that gradients cannot flow without LoRA reflects a misunderstanding of backpropagation—a frozen linear layer still passes gradients to its inputs (the encoders), even though its own weights do not update. This is standard behavior and not a problem.
- **"Missing related works" (from Harsh Critic via transferable weaknesses):** Removed per hard rule—cannot confirm existence of specific uncited works.
- **"Missing reproducibility statement / code availability" (from transferable weaknesses):** Removed per hard rule—nitpicks about code availability and reproducibility artifacts are excluded.
- **"'faithfully' editing artifact in contributions" (from Harsh Critic):** Removed as a formatting nitpick per hard rule.
- **"Class imbalance of datasets requires balanced metrics" (from transferable weaknesses):** While worth noting, this is a generic concern that could apply to nearly any classification paper and doesn't specifically target a flaw in this work's core claims. The paper's focus is modality imbalance, not class imbalance.

## Novel Insights

The parallel between class imbalance and modality imbalance at the gradient-dynamics level (Section 3.1) is a genuinely useful reframing that suggests a family of techniques from the class-imbalance literature—fixed classifiers, re-weighting, delayed re-sampling—could be ported to multimodal learning. However, the current analysis is more illustrative than rigorous; the most valuable insight is the empirical observation (Figure 1) that alternating training alone does not resolve classifier-level bias, which directly motivates the freezing strategy and could independently inform future work even beyond this specific method.

## Suggestions

- **Add MLA, MMPareto, and LFM results to Table 1** (or a supplementary table). MLA is the most critical comparison since CCAT directly extends it—include it to quantify the specific gain from the classifier constraint.
- **Add an ablation row for "unfrozen classifier with full fine-tuning"** to Table 2, establishing that LoRA + freezing outperforms the simpler alternative of allowing the classifier to adapt freely during alternating training.
- **Explicitly acknowledge the linear-fusion simplification** in Section 3.1 and soften the "theoretical isomorphism" language to "illustrative analogy" or "conceptual parallel," noting that the actual architecture uses cross-attention. This preserves the motivational value without overclaiming.