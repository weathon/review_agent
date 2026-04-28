## Summary

This paper presents FreeSplatter, a feed-forward transformer framework for pose-free sparse-view 3D Gaussian reconstruction that jointly estimates camera parameters. The method predicts pixel-wise Gaussian maps in a unified reference frame, enabling both high-fidelity novel view synthesis and instant pose estimation via PnP solvers. Two variants are trained for object-centric and scene-level scenarios, demonstrating strong empirical results on multiple benchmarks.

## Strengths

- **Strong empirical performance on reconstruction quality**: Table 2 shows FreeSplatter-O achieving 31.929 PSNR on OmniObject3D, substantially outperforming pose-dependent baselines LGM (24.852) and InstantMesh (24.077). Figure 3 visually confirms sharper detail preservation compared to baseline artifacts.

- **Unified architecture for both object and scene reconstruction**: The single-stream transformer design handles both scenarios with ~306M parameters, avoiding the pair-wise processing and global alignment overhead of methods like DUST3R (Section 3.2, lines 99-103).

- **Effective geometric supervision via pixel-alignment loss**: Table 3 demonstrates the necessity of L_align (Equation 6), showing PSNR drops from 30.443 to 26.684 on GSO when removed, validating this design choice for stabilizing training.

- **Transparent acknowledgment of limitations**: Section 5 (line 330) honestly admits the depth supervision dependency prevents training on datasets without depth labels (e.g., RealEstate10K), which is appropriate scientific practice.

## Weaknesses

### Fatal

None identified.

### Major

- **Confounded comparison isolates depth supervision, not pose-free architecture**: The central claim that FreeSplatter "outperforms previous pose-dependent large reconstruction models by a large margin" (Section 4.2, line 252) is not rigorously supported. Section 3.3 (lines 111-116) explicitly states that L_pos using ground-truth depth is "essential to model's convergence." However, the compared baselines (LGM, InstantMesh) use only RGB rendering loss without depth supervision. The ~7 dB PSNR gap likely reflects the depth prior advantage rather than the pose-free architecture's value. Without an ablation removing depth supervision from FreeSplatter, or comparison against a pose-dependent model also trained with depth, the claim that "camera poses may not be essential" (line 254) remains unproven. The more accurate conclusion is that "depth supervision is more valuable than camera poses," which is a known result. This strikes at the validity of the paper's primary contribution.

### Minor

- **Pose estimation comparison conflates training data alignment with methodological superiority**: Table 1 shows FreeSplatter-O vastly outperforming MAS3R on OmniObject3D (RRE 11.5° vs 96.7°), but Section 4.3 (line 298) acknowledges MAS3R's poor performance stems from "significant domain gap" (trained on natural images, tested on rendered objects). FreeSplatter-O is trained on Objaverse (rendered), so this comparison evaluates training data alignment rather than pose estimation methodology. Cross-domain robustness or baselines trained on the same distribution would better validate the method.

- **No quantitative inference time benchmarks**: The Abstract claims recovery of camera parameters in "mere seconds" and describes the framework as "highly scalable," but no actual inference times (seconds per scene, FPS) or GPU memory usage are reported in the main text to substantiate these efficiency claims.

- **Reference view sensitivity not analyzed**: Section 3.2 (line 79) states the first image is taken as the reference view with learnable reference/source embeddings. However, there is no analysis of how reconstruction quality degrades if the first view is occluded, blurry, or low-quality—a critical consideration for real-world deployment.

### Trivial

- **Minor typographical inconsistency in tables**: Table 1 and Table 2 show "FreeSplatte" instead of "FreeSplatter" in some rows. This appears to be a minor drafting error that should be corrected.

## Nice-to-Haves

- **Ablation on depth supervision**: Train FreeSplatter without L_pos (using only rendering loss) to quantify how much of the performance gain comes from the depth prior vs. the pose-free architecture.

- **Fair baseline with depth supervision**: Compare against a pose-dependent LRM also trained with ground-truth depth supervision to isolate the "pose input" variable from "depth supervision."

- **Failure case visualizations**: Show examples where pose estimation fails or the unified frame misaligns (e.g., symmetric objects, textureless regions). Current figures only show successes.

- **Reference to self-supervised training pathways**: Discuss potential approaches to remove depth supervision requirement (e.g., photometric consistency, teacher-student distillation) to enable scaling to unlabeled data.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic - "Persistent typo reduces confidence"**: The "FreeSplatte" vs "FreeSplatter" inconsistency appears to be a parser artifact or minor typo. The paper text consistently uses "FreeSplatter." This is a trivial formatting issue, not a substantive weakness.

- **Strength Finder - "Simplified downstream pipeline"**: While Section 4.5 describes eliminating alignment steps for 3D content creation, this is a secondary application benefit rather than core evidence supporting the main pose-free reconstruction claim.

- **Strength Finder - "Competitive camera pose estimation accuracy"**: This conflicts with the verified weakness that the pose estimation comparison is confounded by domain mismatch. The weakness takes precedence.

- **Harsh Critic - "Abstract claims without quantitative time"**: This is valid but already captured in Minor weaknesses; the detailed section-by-section note is redundant.

## Novel Insights

The paper's core tension reveals an important distinction in the pose-free reconstruction literature: "pose-free at inference" does not equal "calibration-free at training." FreeSplatter demonstrates that depth supervision during training can compensate for the lack of camera poses at inference, but this shifts the bottleneck from pose estimation to depth annotation availability. This is a meaningful contribution to understanding what enables pose-free reconstruction, even if the framing overclaims the architecture's role relative to the supervision signal. The honest acknowledgment of this limitation in Section 5 distinguishes it from papers that obscure such dependencies.

## Suggestions

1. **Reframe the core claim**: Instead of claiming pose-free architecture surpasses pose-dependent methods, explicitly state that "depth-supervised pose-free reconstruction surpasses RGB-only pose-dependent reconstruction." This is still a valuable result.

2. **Add critical ablation**: Remove L_pos during training (use only rendering loss + L_align) to show how much performance degrades. This quantifies the depth prior's contribution.

3. **Report inference metrics**: Add a table with inference time (seconds per scene), GPU memory usage, and Gaussian count to substantiate efficiency claims.

4. **Analyze reference view robustness**: Test reconstruction quality when different input views are designated as the reference (first image), especially for occluded or low-quality views.

5. **Add cross-domain pose evaluation**: Test FreeSplatter-O (trained on Objaverse) on real-world object datasets to demonstrate generalization beyond the training domain.

## Score and Decision

**Calibration anchors retrieved:**

| Paper | Avg Score | Comparison to FreeSplatter |
|-------|-----------|---------------------------|
| **Surf3R** (91GG9IUOCh.md) | 2.00 | Similar unfair baseline issue (Dust3R comparison protocol unclear, missing baselines). FreeSplatter is more transparent about limitations. |
| **Feedforward 4D Reconstruction** (cdvppYbBE1.md) | 4.00 | Unfair baseline comparison (different input frame intervals vs STORM). Similar confound severity. |
| **NOVA3R** (c0QRZMKwSb.md) | 5.50 | Pose-free reconstruction with solid experiments but some missing comparisons. FreeSplatter has stronger results but similar confound issues. |
| **YoNoSplat** (ImRhA9xmay.md) | 6.50 | Pose-free + pose-dependent unified model with thorough ablations. More rigorous experimental design than FreeSplatter. |
| **Pi3** (DTQIjngDta.md) | 8.00 | Permutation-equivariant architecture with comprehensive evaluation across tasks. Gold standard for this category. |
| **Depth Anything 3** (yirunib8l8.md) | 7.00 | Strong geometry prediction with clear ablations. Better justification of design choices than FreeSplatter. |

**Scoring rationale:** FreeSplatter falls between the 4.0-range papers (unfair comparisons that undermine claims) and 5.5-6.5 range papers (solid methods with minor experimental gaps). The depth supervision confound is significant but less severe than Surf3R's missing baselines because FreeSplatter explicitly acknowledges the limitation. However, it lacks the ablation rigor of YoNoSplat (6.5) and Pi3 (8.0). The empirical results are genuinely strong, and the transparency about limitations is commendable, but the core claim about pose-free superiority is not isolated from the depth prior advantage. Relative to anchors, this warrants a **5.0** (borderline accept/reject territory).

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>