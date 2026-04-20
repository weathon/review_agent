## Summary
This paper introduces Dual-Stream Adapters (DSA), a parameter-efficient adapter architecture for anomaly segmentation that explicitly separates in-distribution and out-of-distribution features through an anomaly prior module, dual-stream feature refinement, and an uncertainty-based hyperbolic loss. The method freezes the ViT backbone and achieves competitive results with 38% fewer training parameters than prior state-of-the-art methods while preserving in-distribution semantic segmentation accuracy.

## Strengths
- **Parameter-efficient SOTA performance with strong empirical results**: DSA-Large achieves the best average AuPRC (81.3) and competitive FPR across five anomaly segmentation benchmarks while using 133M trainable parameters — 38% fewer than Mask2Anomaly (216M). This is supported by Table 3 (average column) and visually by Figure 1 (right), where DSA-Large occupies a favorable position on the AuPRC vs. FPR95 trade-off.
- **Ablation validates dual-stream architecture necessity**: Table 5(a) shows that removing the anomaly prior module drops AuPRC from 59.6 to 18.5 on SMIYC RO-21, and removing dual-stream feature refinement drops it further to 12.0. The ablation confirms the two-stream specialization is essential, not an architectural convenience.
- **Hyperbolic loss outperforms alternatives**: Figure 6(b) demonstrates L_ubhl (AuPRC 47.8, FPR95 40.2) substantially outperforms both binary cross-entropy (AuPRC 20.0) and contrastive loss (AuPRC 41.7), validating the design choice of using Poincaré ball geometry for feature separation.
- **In-distribution segmentation preserved or improved**: Table 4 shows DSA-Large achieves the highest mIoU on Cityscapes (83.71 without outlier exposure, 82.58 with), even exceeding vanilla Mask2Former (83.37), demonstrating the method does not sacrifice core segmentation accuracy for OOD detection.
- **Practical use of void labels as OOD supervision**: The method leverages existing void/background labels from Cityscapes as OOD supervision without requiring curated outlier datasets, making the approach more practical than methods needing additional data collection.

## Weaknesses

### Major
- **Disconnect between the dual-stream training mechanism and the inference-time scoring function**: The paper's core novelty is a dual-stream adapter trained with a hyperbolic loss to separate ID and OOD features, but the final anomaly score used for all reported results is standard MSP over the decoder's class logits (Eq. 2: s(M,C) = 1 - max(softmax(C)^T · sigmoid(M))). The OOD stream (F_ood) and hyperbolic uncertainty are never explicitly used in the scoring function at inference time. The dual-stream design thus functions primarily as a training regularizer that shapes learned representations, not as the explicit mechanism producing anomaly predictions. The abstract and introduction heavily imply the dual streams are responsible for performance gains, but the inference pipeline does not leverage them directly. The ablation (Table 5a) shows the streams matter during training, but the paper should clarify that the "uncertainty-based" aspect is training-only and not incorporated into the final anomaly score.

- **Missing specification of how dual-stream features are fused for the Mask2Former decoder**: Section 4 and Figure 2 describe two parallel feature streams (F_id and F_ood) refined through DSFR blocks, then state these "are then processed by a Decoder to produce an 'Output Image'" (Fig. 2 caption). However, the paper never specifies how the two streams are combined, concatenated, or routed into the Mask2Former pixel decoder and transformer decoder. Since Mask2Former expects a single set of features for its decoders, the fusion operation is essential for reproducibility. The paper references ViT-Adapter as the base architecture but does not detail its modifications for dual-stream inputs.

### Minor
- **Confounded evaluation comparing methods with mixed supervision protocols**: Table 3 compares DSA-Large against a heterogeneous pool of methods with different training paradigms — some use outlier exposure (✦ ✦✦), some use synthetic negatives, and DSA uses void labels (▽) without outlier exposure. The paper does note these protocol differences with markers, but does not stratify comparisons or provide controlled apples-to-apples baselines. The performance gap may partially reflect training data richness rather than architectural superiority alone.

- **The "38% fewer parameters" efficiency claim is inherent to the adapter paradigm, not the dual-stream design**: Freezing the ViT backbone and training only adapter modules achieves the parameter reduction regardless of whether the adapter is single-stream or dual-stream. This efficiency advantage would be shared by any adapter-based method, not unique to DSA. The paper should clarify this distinction to avoid overstating the novelty of the efficiency gain.

### Trivial
- **Inconsistent FPR thresholds across tables** (FPR_0.5 in Table 1 vs. FPR_0.1 in Tables 2 & 3) without cross-dataset normalization or justification, making trend comparison across sections less intuitive.
- **Ablation in Table 5(b) reports optimal DSFR count on a single dataset** (SMIYC RO-21) while using it to justify configuration across all experiments; the configuration may not be globally optimal.

## Nice-to-Haves
- Feature-space visualization (t-SNE/UMAP of F_id vs. F_ood) would strengthen the claim that the hyperbolic loss actually separates features as intended.
- Reporting a single-stream adapter baseline with identical frozen backbone and decoder setup would help isolate the dual-stream contribution from the general benefit of fine-tuning adapters vs. full finetuning.
- Including confidence intervals or error bars for benchmark results would follow best practices, though single-run evaluation is common in this subfield.

## Removed Points
These points are flagged to be removed — treat them with caution:

1. *"Harsh Critic: 'The dual-stream design is functionally irrelevant at test time'"* — Weakened to a Major weakness instead of Fatal. The paper does use MSP at inference, and the OOD stream doesn't appear in the scoring formula, but the ablation results (Table 5a) confirm the dual-stream training does contribute significantly to learned representations. Removing streams causes catastrophic performance drops (AuPRC 59.6 → 12.0), so they are not "functionally irrelevant" — they just operate implicitly through training rather than explicitly through inference.

2. *"Harsh Critic: 'Non-reproducible due to undefined fusion mechanism'"* — Weakened to Major instead of Fatal. While the fusion details are missing, the paper builds on ViT-Adapter whose structure is documented, making this a clarity gap rather than a fundamental reproducibility blocker. The method produces strong results, which implies the authors found a working fusion.

3. *"Harsh Critic: 'Confounds training regimes, invalidating SOTA claims'"* — Weakened to Minor. Mixing protocols (outlier exposure vs. void labels) is noted with markers and is standard practice in the anomaly segmentation literature. The paper does not claim these comparisons are apples-to-apples; it presents a comprehensive benchmark. A cleaner controlled comparison would strengthen the paper but doesn't invalidate it.

4. *"Harsh Critic: '38% lower training parameters claim is misleading'"* — Weakened to Minor. The efficiency does come from freezing the backbone (standard for adapters), but the paper's framing ("reducing training parameters by 38% w.r.t. the previous state-of-the-art") is factually correct for the comparison made (DSA-Large 133M vs. Mask2Anomaly 216M). It's more about framing emphasis than incorrectness.

5. *"Human Strength Finder: 'Large and consistent improvement over general-purpose vision adapters'"* — Verified through Tables 1 & 2 but moved to supporting evidence for the parameter-efficient SOTA strength. These tables compare DSA-Tiny vs. Side/ViT adapters and show consistent gains, but they're less impactful than the main SOTA comparison (Table 3).

## Novel Insights
The paper tackles a practical constraint — large transformer backbone costs for mask-based anomaly segmentation — with a frozen-backbone + dual-stream adapter that leverages existing void labels for OOD supervision rather than requiring curated outlier data. While the dual-stream/OOD mechanism doesn't explicitly drive the inference-time anomaly score (relying instead on standard MSP), the ablation confirms it meaningfully regularizes training-time feature learning. The key insight that this approach achieves SOTA anomaly segmentation with fewer parameters while preserving in-distribution accuracy is useful for deployment constraints, though the paper would benefit from more transparently framing the dual streams as training regularizers rather than as an explicit inference mechanism.

## Suggestions
1. **Clarify the inference-time role of the dual-stream/OOD features** in the introduction and method sections. Explicitly state whether the uncertainty-based aspect is training-only or whether the OOD stream contributes to the anomaly score, even implicitly through decoder conditioning.
2. **Specify the tensor fusion operation** between F_id and F_ood before the Mask2Former decoder (concatenation, averaging, summation) to enable reproduction.
3. **Add a controlled baseline** with a single-stream adapter under identical frozen-backbone conditions to isolate the dual-stream contribution from general adapter benefits.
4. **Consider an ablation or variant** that incorporates OOD stream features directly into the scoring function to test whether explicit uncertainty-based scoring further improves results.

## Score and Decision
I calibrated against several anchor papers:
- **High-score anchor** (mUXdysoxEP, feature separation for OOD): 8, 8, 6, 5 — Accept. This paper had cleaner methodology with explicit feature-space separation that directly drives detection, unlike DSA's implicit training-only separation.
- **Mid-range adapter anchors** (oVCVCo3laS, gWw0NjTQRg): 3-6 range — papers with adapter architectures but lacking strong empirical validation or facing clarity issues.
- **Low-score anchor** (uK4TYkVBJG, SAM adaptor with missing details): 3, 5, 3, 3 — Reject. This paper had missing architecture and loss details similar to DSA but weaker empirical validation overall.

DSA sits notably above the low-score anchor because its empirical results are strong and comprehensive across multiple benchmarks, and the ablation validates the core design. It falls below the high-score anchor because the disconnect between the dual-stream training mechanism and the MSP inference scoring, plus the missing fusion specification, prevents the methodological clarity seen in the stronger comparison. The paper is comparable to solid adapter papers with good results but methodological gaps that could be addressed in revision.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>