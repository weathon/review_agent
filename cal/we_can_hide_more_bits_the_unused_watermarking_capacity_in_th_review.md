=== CALIBRATION EXAMPLE 34 ===

# Final Consolidated Review
## Summary

This paper challenges the assumption that image watermarking capacity has plateaued near 100–200 bits by deriving geometric upper bounds on message-carrying capacity under PSNR and linear robustness constraints, showing theoretical capacities are orders of magnitude larger than current methods achieve. Through controlled experiments on simplified setups (single gray image, PSNR-only), the authors demonstrate that existing architectures like Video Seal fail to approach even greatly reduced theoretical limits, while simple linear and handcrafted models succeed—pointing to architectural rather than fundamental limitations. As proof of concept, the authors train Chunky Seal, a 90×-scaled version of Video Seal, achieving 1024-bit capacity with comparable quality and robustness.

## Strengths

- **Novel geometric framework for watermarking capacity** that departs from classical information-theoretic approaches (Gaussian channels, mutual information) by counting lattice points within PSNR balls intersected with the valid image cube. This yields concrete, computable bounds (Bounds 1–9) that are directly tied to the discrete, quantized nature of digital images—a meaningful advance over prior theory that could not be instantiated for real images.

- **Compelling diagnostic experiments that isolate architectural failure.** The gray-image experiments (Section 3.1) are a methodological highlight: by stripping away all real-world complexity and showing Video Seal still fails at 1024 bits while a single linear layer succeeds at 2048 bits, the paper cleanly separates model limitations from problem difficulty. The finding that Video Seal at 256×256px performs identically to 32×32px (Table 1) is particularly striking evidence that the architecture fails to exploit spatial resolution.

- **Achievability validation closes the theory-practice loop for the PSNR-only case.** The handcrafted embedder reaching 456,509 bits at 42 dB (Table 1)—close to the theoretical bound—confirms the bounds are not vacuous artifacts. This rules out hypothesis D and strengthens the paper's core argument that models, not theory, are the bottleneck.

- **First practical demonstration of 1024-bit robust watermarking.** Chunky Seal's 4× capacity increase over Video Seal while maintaining ≥98% bit accuracy on most standard augmentations (Table 3) establishes a new practical benchmark.

## Weaknesses

### Major:

- **Robustness bounds are heuristic, not formal, with large uncertainty.** Bounds 10–12 are explicitly acknowledged as heuristics "near-exact only for axis-aligned transformations" (Appendix G.2). Figures 8–9 demonstrate they can both over- and under-estimate true capacity. The conservative Bound 13 gives 904 bits for 256×256px under aggressive cropping (Table 2), while heuristic bounds suggest ~98,000 bits at 40 dB—over 100× discrepancy. Since the paper's central claim—"robustness constraints reduce but cannot fully explain the low watermarking capacity"—rests on these bounds, this uncertainty significantly weakens the theoretical contribution for the practically relevant (robust) setting. The PSNR-only bounds are rigorous; the robust bounds are suggestive but not conclusive.

- **Chunky Seal's trade-offs undermine the "comparable robustness and quality" claim.** The LPIPS degradation (0.0085 vs. 0.0019, Table 3) is 4.5×, and on COCO, JPEG robustness drops sharply: JPEG Q40 yields only 65.86% bit accuracy vs. Video Seal's 97.79% (Table 5). Even on SA-1B, Rotation 10° shows 97.26% vs. 98.31%, and JPEG Q50 shows 98.35% vs. 99.64% (Table 4). The claim of "comparable robustness" in the main text (page 9) is overstated given these gaps. Without a Pareto front analysis (capacity vs. robustness and capacity vs. quality curves at multiple operating points), it is unclear whether Chunky Seal shifted the trade-off frontier or simply moved along it by accepting worse LPIPS and some robustness degradation in exchange for higher capacity.

- **Achievability evidence is limited to the non-robust, gray-image setting.** The strongest results showing the theory-practice gap is closable—linear model at 2048 bits, handcrafted at 456K bits, tiled 32×32 at 32K bits—all operate on a single gray image with no robustness constraints. For the robust setting, the only evidence is Chunky Seal's 4× improvement via massive scaling (1.8B parameters). This leaves open the possibility that the robust bounds are substantially over-estimated and that the true robust capacity is much closer to current practice than the heuristic bounds suggest. The paper's argument would be much stronger with even a simple achievability experiment under robustness constraints.

- **No mechanistic explanation for Video Seal's architectural failure.** The paper demonstrates that Video Seal fails at 1024 bits on gray images and that linear models succeed, attributing this to "structural limitations" and referencing identity mapping difficulty (He et al., 2016; Hardt and Ma, 2017). But this is a high-level analogy, not an analysis. Is the bottleneck in the encoder (information bottleneck in the embedding dimension), the decoder (insufficient capacity to invert the embedding), the optimization landscape, or gradient flow? Understanding which component fails would directly inform the paper's own call for "new architectural designs" and make the contribution more actionable.

### Minor:

- **No ablation study on Chunky Seal.** Multiple changes are made simultaneously: embedding dimension (256→2048), U-Net channel multipliers ([1,2,4,8]→[4,8,16,32]), all three channels instead of luma-only, ConvNeXt depth/dimension changes, and gradient clipping. Without isolating which changes drive the 4× capacity gain, the paper cannot distinguish between "scaling works" and "specific architectural fixes (e.g., using all channels) work." This matters because the conclusion advocates for "architectural innovation" but the evidence only supports "brute-force scaling."

- **bpp resolution-invariance asserted but not verified for robustness bounds.** For PSNR-only bounds, the mathematical form (capacity = cwh · log₂(q)) ensures per-pixel capacity is resolution-independent. However, the robustness bounds involve singular value structure of transformation matrices that may not decompose cleanly across resolution, and no empirical verification of this invariance at practical resolutions (256×256) is provided.

- **Training convergence for Video Seal at 1024 bits is not verified.** While the hyperparameter sweep makes convergence failure at 1024 bits plausible, no learning curves or convergence diagnostics are shown. If the model simply needed longer training or different optimization, the "architectural limitation" conclusion would be weakened.

### Trivial:

- The discontinuity between Bounds 7 and 8 in Figure 3 (right panel) could confuse readers; a brief note in the caption explaining it arises from the volume approximation undercounting boundary points would help.

## Nice-to-Haves

- Evaluation against adversarial watermark removal attacks (not just standard augmentations), which is the practical threat model for provenance systems.
- A capacity scaling law: plot achievable capacity as a function of model size and training compute to reveal whether Chunky Seal is near saturation or further scaling yields diminishing returns.
- Pareto front curves (capacity vs. PSNR, capacity vs. robustness) at multiple operating points for both Video Seal and Chunky Seal.
- Real-world compression pipeline evaluation (social media platforms) rather than only standard JPEG.
- Multi-watermark coexistence demonstration, which the discussion mentions as feasible but does not validate.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Weakness: "progress has stagnated" claim needs more evidence.** Figure 1 directly shows multiple recent methods clustering at similar capacities on a log-scale plot, and the paper cites specific bit counts for recent methods. The claim is well-supported.

- **Weakness: Practical importance of higher capacity not motivated in introduction.** This is a style/organization nitpick; the introduction clearly states the question and the discussion provides use cases.

- **Weakness: Appendices too long (14 pages).** Pure formatting complaint. The derivations are necessary for reproducibility.

- **Weakness: Privacy implications of high-capacity watermarks not discussed.** This is scope creep; the paper is about capacity analysis, not deployment policy.

- **Weakness: Missing related works.** Per rules, not evaluated.

- **Weakness: Reproducibility concerns about Chunky Seal training resources.** Per rules, large artifacts impractical to include are not a valid criticism.

- **Weakness: Unfair comparison with linear model on gray image.** Per rules, criticisms about unfair comparison when asymmetry favors the baseline (Video Seal is a stronger baseline than the linear model) should be removed. The linear model comparison is designed to show that even the simplest architecture succeeds where a sophisticated one fails—a stronger point for the paper's argument.

- **Weakness: Need for statistical significance testing / confidence intervals.** Moving to nice-to-have per soft rules—single-run evaluation is the norm in this area.

## Novel Insights

The paper reveals a striking disconnect between watermarking and representation learning: deep architectures that excel at learning complex features apparently cannot learn what amounts to a high-dimensional identity mapping for watermarking. The fact that Video Seal performs identically at 256×256px and 32×32px (Table 1) suggests the architecture has an effective "capacity bottleneck" independent of spatial resolution—likely related to the information bottleneck in the embedding dimension or the decoder's limited receptive field for bit extraction. This mirrors the identity mapping problem identified in ResNet but manifests differently: the network can learn to embed 512 bits but cannot scale to 1024, suggesting a sharp phase transition rather than gradual degradation. This observation—that watermarking capacity may exhibit threshold behavior in neural architectures—is not explicitly discussed in the paper but has important implications for architecture design.

## Suggestions

- **Add Pareto front analysis:** Train Video Seal and Chunky Seal at multiple capacity targets (64, 128, 256, 512, 1024 bits) and plot capacity vs. PSNR, LPIPS, and robustness curves. This would definitively show whether Chunky Seal shifts the frontier or trades off quality for capacity.

- **Perform a minimal ablation on Chunky Seal:** At minimum, isolate the effect of (1) using all 3 channels vs. luma-only and (2) the embedding dimension increase, as these are the most architecturally meaningful changes beyond simple width scaling.

- **Test achievability under at least one robustness constraint:** Even a simple experiment—e.g., training the linear model on a gray image with JPEG augmentation—would reveal whether the gap between theory and practice persists under robustness or is an artifact of the PSNR-only analysis.

- **Diagnose Video Seal's failure mode:** Analyze gradient norms, intermediate representations, or information flow through Video Seal at 512 vs. 1024 bits on the gray image task. Even a simple analysis of whether the encoder or decoder is the bottleneck would make the "architectural limitations" claim more actionable.

---

**Axis Evaluations:**

- **Novelty:** High for the geometric capacity framework and the diagnostic experimental methodology; moderate for Chunky Seal (explicitly described as simple scaling).

- **Technical soundness:** Strong for the PSNR-only bounds (formal, validated by achievability experiments); notably weaker for robustness bounds (heuristic, large gap between heuristic and conservative estimates).

- **Empirical support:** Strong for the core diagnostic argument (gray image experiments are well-designed); moderate for the practical claim that robust high-capacity watermarking is achievable (Chunky Seal's mixed robustness results, especially on COCO JPEG, and the 1.8B parameter cost).

- **Significance:** Potentially field-shaping if the geometric framework is adopted and the robustness bounds are tightened; however, the practical demonstration (4× improvement via 90× scaling) only partially delivers on the theoretical promise.

- **Clarity:** Dense but well-structured theoretical sections; clear experimental setup; the main argument flows logically from theory → diagnosis → proof of concept.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Reject
