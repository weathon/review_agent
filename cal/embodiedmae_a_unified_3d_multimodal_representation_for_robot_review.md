=== CALIBRATION EXAMPLE 48 ===

# Final Consolidated Review
## Summary

EmbodiedMAE presents a unified 3D multi-modal masked autoencoder for robot manipulation that integrates RGB, depth, and point cloud modalities through stochastic masking and cross-modal decoder fusion. The authors construct DROID-3D by processing the full DROID dataset (76K trajectories) with ZED SDK to extract high-quality metric depth maps and point clouds, and demonstrate that their model consistently outperforms existing vision foundation models across 70 simulation tasks and 20 real-world tasks on two robot platforms.

## Strengths

- **DROID-3D is a genuine resource contribution.** Processing the complete 76K trajectories with ZED SDK's temporal fusion (vs. SPA's subset of ~1/15 of DROID with AI-estimated depth) provides temporally consistent, metric-scale depth and point clouds. This addresses a real data gap in the embodied 3D learning community.
- **Compelling cross-modal fusion evidence.** The re-coloring experiment (Figure 3, column 12) demonstrates that the model has implicitly learned object-level semantic segmentation—when a single RGB patch's color is altered during depth-to-RGB prediction, only the corresponding object adopts the new color while surrounding elements remain unchanged. This is a non-trivial emergent property that goes beyond what standard MAE training typically produces.
- **Thorough evaluation scope.** 90 total tasks across two simulation benchmarks (LIBERO, MetaWorld) and two real-world platforms with fundamentally different hardware profiles (low-cost SO100 at ~$250 vs. high-precision xArm with LiDAR) provides meaningful evidence of generalizability, not just incremental gains on a single benchmark.
- **Strong scaling behavior with practical distillation.** Figure 6 shows monotonic improvement from Small→Giant, and the distillation pipeline (Section 2.4) with feature alignment at three network depths enables smaller models to approach Giant performance, making deployment feasible on resource-constrained systems.

## Weaknesses

### Major:

- **DROID-3D depth quality claims rely solely on qualitative evidence.** Figure 2 shows visual comparisons between ZED SDK output and AI-estimated depth, but provides no quantitative metrics (e.g., temporal consistency error, depth accuracy against known geometry, or reprojection error). Since the dataset quality underpins the entire pre-training pipeline, this is a significant gap—especially when claiming superiority over SPA's CrocoV2-Stereo approach. Even a small validation subset with ground truth would substantially strengthen this core contribution.

- **Insufficient ablation on core architectural design choices.** While Table 4 ablates masking ratio, feature alignment, and loss ratio during distillation, several fundamental design choices lack justification through ablation: (1) Why is stochastic Dirichlet-based mask allocation across modalities better than fixed ratios? The concentration parameter α is never ablated. (2) What is the individual contribution of cross-modal decoder fusion versus independent per-modality reconstruction? (3) How critical is initializing from DINOv2 weights versus training from scratch? The ablation of the distillation phase (Table 4) is useful but does not address these pre-training design questions, leaving the "why it works" partly unexplained.

### Minor:

- **Point cloud findings are fragmented across main text and appendix.** Section 3.4 states that "PC-based policies even underperform RGB-only inputs" due to sensor noise, presenting this as a fundamental limitation. However, Appendix B.3 (Table 9) shows that with enhanced preprocessing, EmbodiedMAE-PC achieves 82.1% on xArm—a substantial improvement. The main text should better integrate this finding: the limitation is not inherent to the PC modality but rather to input quality, which is addressable. As presented, a reader could easily miss the positive PC result buried in the appendix.

- **RQ1 evaluation (Section 3.2) is purely qualitative.** The cross-modal prediction visualizations in Figure 3 are compelling, but no quantitative reconstruction metrics (e.g., SSIM/PSNR for RGB, Chamfer distance for point clouds) are reported under varying masking ratios. Since RQ1 asks whether the model "learns features that integrate information across different modalities," objective metrics would provide stronger evidence than visual inspection alone.

### Trivial:

- The DINOv2-RGBD baseline (Appendix A.3) uses a simple summation-based fusion. While this follows Zhu et al. (2024), a more sophisticated fusion (e.g., cross-attention) could serve as a stronger ablation point, though the comparison against SPA and DP3 partially mitigates this.

## Nice-to-Haves

- **Attention/fusion pattern visualization** showing how RGB, depth, and point cloud tokens interact in the decoder. This would provide mechanistic evidence for the cross-modal fusion claim beyond reconstruction quality, addressing the "how" of RQ1.
- **Language integration roadmap.** The limitation of no native language support is honestly acknowledged (Section 5), but even a brief discussion of how DROID-3D's language annotations could be incorporated (e.g., via contrastive alignment or VLA-style training) would strengthen the paper's positioning against increasingly language-driven VLA models.
- **Cross-embodiment evaluation** beyond tabletop manipulators (e.g., mobile manipulation, different kinematic structures) to test the generality of the 3D representations.
- **Quantitative domain gap analysis** between DROID-3D and the evaluation benchmarks, to substantiate the claim that DROID-3D provides domain-compatible pretraining data.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Latency budget concern for real-time control**: The harsh critic argued that EmbodiedMAE-Large's ~32ms inference leaves insufficient time for the policy network at 30Hz. This misunderstands the action chunking mechanism: the policy predicts action chunks (length 16–64) and executes subsets open-loop, meaning the perception+policy pipeline runs at a much lower effective frequency (~4–15Hz depending on chunk/exec ratio), giving ample time even for the Large model. **Removed because it is factually wrong—it misunderstands the action chunking architecture described in Section A.2.**

- **Dataset availability / DROID-3D hosting concerns**: Questions about whether the processed DROID-3D will be hosted or require re-running the 500-hour pipeline. **Removed per hard rule: do not question the existence, release status, or availability of any cited dataset.**

- **Carbon footprint / computational cost of Giant pre-training**: **Removed as a reproducibility nitpick—the paper provides detailed hyperparameters in Table 8 (8×L40, 200K steps) sufficient for reproducibility assessment.**

- **Missing comparisons with Octo/OpenVLA with 3D inputs or other 3D VFMs (Point-BERT, etc.)**: **Removed per rules: the paper's stated scope is vision foundation models as backbones for manipulation policies, not full VLA systems; demanding comparison outside this scope is scope creep. Also, do not mention missing related works without confirming their existence.**

- **Statistical significance of 10 real-world trials**: While 10 trials per task is limited, this is the accepted standard in the robotics community for real-world experiments (CoRL, RSS). **Moved to minor concern rather than major, as demanding 30–50 trials would be outside community norms for this type of work.**

- **CLS token removal concern**: The policy network uses cross-attention over patch tokens, making CLS unnecessary. This is standard practice in many ViT-based policy architectures. **Removed as factually unproblematic.**

- **Table formatting issues**: **Removed per formatting nitpick rule.**

## Novel Insights

The most striking emergent finding is the re-coloring result (Figure 3, column 12): when a single RGB patch with altered color is injected during depth-to-RGB reconstruction, the model propagates that color change to the entire corresponding object while preserving the appearance of other objects. This suggests EmbodiedMAE develops implicit object-level semantic segmentation through multi-modal masked reconstruction alone—without any segmentation supervision. This is a non-trivial property that distinguishes the learned representation from simple feature aggregation and hints that the cross-modal reconstruction objective may serve as an indirect form of object discovery. The paper mentions this but underplays its significance; it deserves more emphasis as evidence that the multi-modal MAE objective is doing something fundamentally more interesting than per-pixel reconstruction.

## Suggestions

- Add quantitative depth quality metrics for DROID-3D (even on a small validation set with known geometry) to substantiate the ZED SDK superiority claim over AI-estimated depth.
- Ablate the Dirichlet concentration parameter α and compare stochastic vs. fixed mask allocation ratios to justify the masking strategy.
- Move the key Appendix B.3 finding (enhanced PC quality → strong performance) into the main text discussion of real-world results, reframing the PC limitation as addressable through preprocessing rather than inherent.
- Report quantitative reconstruction metrics (SSIM, Chamfer distance) for the RQ1 cross-modal predictions to complement the visual evidence.

---

**Axis evaluations:**
- **Novelty**: Moderate. The architecture synthesizes existing components (MAE, MultiMAE-style masking, DINOv2 initialization, DP3 encoder) into a new domain-specific combination. The primary novelty lies in the application scope (embodied 3D), the DROID-3D resource, and the empirical demonstration that this combination works well where prior 3D VFMs have struggled.
- **Technical soundness**: Generally sound, with the main gap being insufficient ablation on pre-training design choices and lack of quantitative validation for the dataset quality claims.
- **Empirical support**: Strong in breadth (90 tasks, 4 benchmarks, 2 real platforms) but with some depth issues (qualitative-only RQ1, limited ablations on core design choices, 10-trial real-world evaluations).
- **Significance**: High. If the claims hold, EmbodiedMAE provides both a practical VFM and a valuable dataset resource for the embodied 3D learning community, addressing a well-recognized gap.
- **Clarity**: Good overall. The paper is well-structured with clear research questions. The main clarity issue is the fragmented PC findings between main text and appendix.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 4.0]
Average score: 5.0
Binary outcome: Reject
