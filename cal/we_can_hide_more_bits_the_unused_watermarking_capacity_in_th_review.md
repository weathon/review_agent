=== CALIBRATION EXAMPLE 54 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title "We Can Hide More Bits: The Unused Watermarking Capacity in Theory and in Practice" is direct and reflects the paper's core contribution. The abstract clearly states the problem, the approach (theoretical bounds and empirical validation), and the key finding: a large gap exists between theoretical capacity and current methods, demonstrated by scaling up a model to 1024 bits. All abstract claims are substantiated in the paper.

**Introduction & Motivation:** Excellent. It effectively sets the stage by describing the perceived stagnation in watermarking capacity and posing the critical, unanswered question: "Have we already reached the theoretical ceiling?" The contributions (i, ii, iii) are stated clearly and map directly to the paper's structure. The framing as a hunt for the cause of the performance gap (hypotheses A-E) is a strong narrative device.

**Method / Approach (Theoretical Bounds - Sections 2, Appendices):** The geometric modeling of images as a finite grid and the derivation of capacity bounds under PSNR constraints (Bounds 1-9) are rigorous and well-explained. The progression from absolute capacity to PSNR-only to robustness constraints is logical. The use of volume approximations and exact counting (Mitchell's algorithm) is appropriate.

However, the section on robustness constraints (Section 2.5) is the weakest part theoretically. Bounds 10-12 are explicitly labeled as "heuristic" and rely on an intuitive product of singular values factor (Eq. 6). The paper is transparent about their limitations (Figs. 8, 9 show they are neither strict upper nor lower bounds) and provides a separate, extremely conservative bound (Bound 13). While this honesty is commendable, for an ICLR paper, the lack of formal guarantees or tighter analysis for the robustness case is a notable gap. The construction of LinJPEG is clever and useful for analysis.

**Experiments & Results (Sections 3 & 4):** This is the strongest part of the paper. The experimental design is excellent for isolating the cause of the capacity gap.
- **Section 3.1:** The experiment training Video Seal on a single gray image with only a PSNR constraint is a brilliant minimal test. Its failure to reach 1024 bits convincingly rules out hypotheses A, B, and C (advanced robustness/perceptual constraints, data distribution).
- **Section 3.2:** The subsequent experiments are conclusive. The success of the linear model and the handcrafted model shows the bounds are achievable (ruling out hypothesis D), directly implicating hypothesis E (models are underperforming). The tiling experiment further highlights the architectural under-utilization of resolution.
- **Section 4 (Chunky Seal):** This is the crucial "proof of concept" that higher capacity is feasible under real-world constraints. Scaling up Video Seal to achieve 4x capacity while maintaining comparable quality and robustness is a solid result. Table 3 shows a comprehensive evaluation. The fact that this was achieved with simple scaling and no hyperparameter tuning strongly supports the core thesis.

**Potential Experiment Concerns:**
- The comparison in Figure 1/Table 4 includes many older methods (HiDDeN, MBRS). It would be beneficial to explicitly state that the most relevant SOTA comparisons are Video Seal, TrustMark, and WAM, as the older baselines set a lower bar.
- The Chunky Seal model is enormous (~1B parameter embedder). While this successfully demonstrates feasibility, it raises practical concerns about efficiency, which are only briefly mentioned in the conclusion. A brief discussion on parameter counts or FLOPs relative to capacity gain would be useful context.

**Writing & Clarity:** The paper is generally well-written but dense, especially in the theoretical sections and appendices. The use of figures (Fig. 2, 3, 4, 5) is very effective. Some parts, like the explanation of the heuristic robustness factor (Eq. 6) and the zonotope over-approximation for Bound 13, are complex and could benefit from additional intuitive explanation. The flow from theory to controlled experiments to real-world model is logical and clear.

**Limitations & Broader Impact:** The discussion section (Sec. 5) adequately covers key limitations: theoretical bounds are for tractable setups, robustness bounds are heuristic, and Chunky Seal's size/latency. A significant limitation under-emphasized is the **computational cost and practicality** of the scaling approach. While the paper correctly states the goal was to explore feasibility, the community might misinterpret this as an endorsement of naive scaling. The proposed "sanity checks" are a valuable contribution. Broader impact is positively framed (enabling embedded manifests) without delving into potential misuse, which is reasonable for this work.

### Overall Assessment

This is a strong, well-executed paper that makes a significant contribution to the watermarking field. It successfully answers its core question: current models are far from theoretical capacity limits, and this gap is due to architectural/training limitations, not inherent problem complexity. The combination of novel theoretical bounds, cleverly designed controlled experiments, and a scaled-up practical model provides compelling, multi-faceted evidence. The main weakness is the non-rigorous treatment of robustness bounds, but the paper is transparent about this and supports its conclusion with strong empirical results (Chunky Seal). The work is insightful, provides clear directions for future research, and meets the bar for ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents a theoretical and empirical analysis of the capacity limits of image watermarking. It establishes geometric upper bounds on the achievable bits per pixel under PSNR and linear robustness constraints, showing these bounds are orders of magnitude higher than current deep learning-based methods achieve. The authors empirically demonstrate that even simple linear models can outperform state-of-the-art architectures in simplified settings, and they propose Chunky Seal, a scaled-up version of Video Seal, which embeds 1024 bits while maintaining comparable quality and robustness, proving higher capacities are practically attainable.

### Strengths
1. **Rigorous Theoretical Framework**: The paper develops a novel geometric framework for deriving capacity bounds under PSNR and linear robustness constraints (e.g., Bounds 1-13). This approach is more aligned with the discrete, quantized nature of digital images compared to classical information-theoretic models based on Gaussian noise. The derivations for corner cases, robustness to transformations, and linearized JPEG are thorough and well-supported.
2. **Compelling Empirical Evidence**: The controlled experiments (e.g., training on a single gray image) effectively isolate the cause of the capacity gap. The results showing Video Seal's failure to embed 1024 bits in a simple setting, while linear and handcrafted models succeed (Table 1, Figure 5), strongly argue that architectural limitations, not inherent problem complexity, are the primary bottleneck.
3. **Demonstration of Practical Improvement**: The proposed Chunky Seal model achieves a 4× increase in capacity (1024 bits) over the strong Video Seal baseline while preserving image quality (PSNR ~45 dB) and robustness across a wide range of perturbations (Table 3). This concrete result validates the paper's core claim that significant performance gains are possible.

### Weaknesses
1. **Heuristic and Loose Robustness Bounds**: The theoretical capacity bounds under robustness constraints (Bounds 10-12) are explicitly labeled as heuristics. The authors show these bounds can be both over- and underestimates (Figures 8-9), and the conservative alternative (Bound 13) is extremely loose. This limits the precision of the theoretical predictions for real-world, robust watermarking.
2. **Limited Practicality of Chunky Seal Solution**: While Chunky Seal proves higher capacity is feasible, it achieves this via massive model scaling (90× larger embedder, 23× larger extractor), resulting in high computational cost. The paper does not address efficiency or explore more parameter-efficient architectural innovations, which is critical for real-world deployment.
3. **Simplified Robustness Model**: The theoretical analysis is restricted to linear transformations and a linearized approximation of JPEG (LinJPEG). While many common perturbations are linear or approximately linear, the analysis may not fully capture complex, non-linear, or adversarial attacks prevalent in real-world scenarios.

### Novelty & Significance
**Novelty**: The geometric modeling of capacity for digital images is a significant departure from classic information-theoretic approaches and provides a fresh, practical perspective. The systematic empirical deconstruction showing architectural underperformance in simplified settings is highly insightful.
**Significance**: The work convincingly argues that the watermarking field is far from its fundamental limits, challenging a potential stagnation narrative. It provides both a theoretical lens (new bounds) and a practical proof-of-concept (Chunky Seal) to guide future research toward higher-capacity methods. The proposed "sanity checks" for new watermarking methods are a valuable contribution.

### Suggestions for Improvement
1. **Tighten Robustness Bounds**: Future work should aim to derive tighter, non-heuristic bounds for capacity under common non-linear transformations (e.g., standard JPEG, gamma correction). Exploring connections to lattice packing or coding theory could be fruitful.
2. **Pursue Architecturally Efficient Scaling**: Instead of advocating for pure model scaling, the authors should suggest and explore specific architectural inductive biases (e.g., frequency-aware layers, structured embeddings) that could achieve high capacity more efficiently. An ablation study on Chunky Seal's components would be informative.
3. **Broaden the Attack Spectrum in Evaluation**: Evaluate Chunky Seal against a more comprehensive suite of attacks, including state-of-the-art adversarial removal techniques and non-linear, learned distortions, to better assess its practical robustness.
4. **Improve Presentation and Reproducibility**:
   * Provide full training details (hyperparameters, computational resources) for Chunky Seal to ensure reproducibility.
   * Add a summary table in the main text listing all key bounds and their regimes of validity for easier reference.
   * Consider streamlining the highly technical appendices (e.g., on zonotopes) for improved readability, focusing on intuitive explanations in the main text.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Test Chunky Seal's capacity in the simplified gray-image setting.** The core claim is that capacity is far from theoretical limits. To validate that Chunky Seal represents progress, it must be evaluated on the same controlled task (single gray image, PSNR-only) where Video Seal failed. Does it approach the handcrafted baseline (456k bits) or at least surpass the linear model (2048 bits)? Without this, it's unclear if scaling addresses the fundamental limitation or just works on a different, easier distribution.

2. **Perform an ablation study on Chunky Seal's scaling factors.** The model is scaled up in multiple ways (channels, depth, etc.). Which components are necessary for the capacity gain? A simple scaling sweep (e.g., 256, 512, 1024 bits) with controlled increases in parameters would show whether capacity scales predictably with model size, strengthening the architectural argument.

3. **Compare against a robust, high-capacity non-neural baseline.** The handcrafted and linear baselines show high capacity without robustness. To isolate the impact of robustness constraints, design a simple scheme (e.g., the handcrafted method combined with error-correcting codes) and test it under the same augmentations used for Chunky Seal. If it outperforms deep models, the architectural critique is stronger.

4. **Conduct a fair, capacity-matched comparison with prior work.** Chunky Seal (1024 bits) is compared to Video Seal (256 bits). To claim a Pareto improvement, train Video Seal at 1024 bits (or a similarly scaled version) on the same dataset and training budget. Otherwise, gains may be due to increased capacity budget rather than better efficiency.

### Deeper Analysis Needed (top 3-5 only)
1. **Diagnose why Video Seal fails on the gray-image task.** The claim is "structural limitations." Analyze the learned embeddings: do they occupy a low-dimensional subspace? Test if increasing the width/depth of Video Seal (without other changes) improves capacity. This would pinpoint whether the architecture is fundamentally limited or just undertrained.

2. **Quantify the capacity gap under robustness constraints in practice.** The theoretical bounds suggest high capacity remains under augmentations. For Chunky Seal, measure the *maximum* capacity achievable (via rate-distortion curves) under each augmentation (e.g., by training models at different bit lengths) and compare to the theoretical predictions. This shows whether the gap persists in realistic settings.

3. **Validate the heuristic robustness bounds empirically.** Bounds 10-12 are unproven approximations. For small images (e.g., 8x8), compute the exact capacity under a linear transformation (via enumeration) and compare to the heuristic. This is essential to trust the central claim that robustness doesn't explain the low empirical capacity.

4. **Analyze the bit error patterns under strong attacks.** When accuracy drops (e.g., rotation >10°), are errors random or structured? This reveals if the model fails due to insufficient invariance or because the message is destroyed. It informs architectural improvements.

### Visualizations & Case Studies
1. **Visualize the spatial and channel distribution of the watermark signal.** For Chunky Seal and Video Seal, plot the watermark energy per pixel/channel. Does Chunky Seal utilize more pixels or channels effectively? This would show if scaling leads to better spatial utilization.

2. **Show qualitative examples of failure under cropping/rotation.** Display watermarked images before and after severe cropping (e.g., 25%) and the corresponding extracted bits. This reveals whether failures are due to content loss or decoder inadequacy.

3. **Case study: Embed and recover a realistic C2PA manifest.** The discussion suggests high capacity enables embedding full manifests. Demonstrate this by encoding a ~1KB manifest into an image using Chunky Seal and recovering it after mild perturbations. This proves practical utility beyond bit accuracy metrics.

### Obvious Next Steps
1. **Implement and evaluate a tiled version of Video Seal on real images with robustness.** The tiling strategy worked on the gray image. Apply it to robust watermarking: train a small model on patches and tile it across a full image. This is a straightforward high-capacity baseline that should be tested; if it works well, it challenges the need for monolithic scaling.

2. **Provide the "sanity checks" as quantitative metrics.** The conclusion proposes sanity checks (e.g., capacity linear in image size). Define these as concrete, measurable properties and evaluate Chunky Seal and prior work against them. This would provide a clear framework for future work.

3. **Discuss the practical trade-offs of Chunky Seal's scale.** The model is huge (~1B parameters for embedder). Analyze its inference latency, memory footprint, and training cost compared to Video Seal. For real-world deployment, efficiency matters; ignoring this weakens the contribution.

4. **Test the method on a wider range of image distributions and resolutions.** The evaluation uses SA-1B and COCO at 256px. Test on higher resolutions (e.g., 1024px) and diverse content (art, text, low-light). This ensures the capacity gains generalize beyond the training distribution.

# Final Consolidated Review
## Summary
This paper establishes geometric upper bounds on the message-carrying capacity of image watermarking under PSNR and linear robustness constraints, showing that theoretical limits are orders of magnitude higher than current deep learning methods achieve. Through controlled experiments, it demonstrates that state-of-the-art architectures fail to approach these bounds even in simplified settings, implicating structural limitations. As proof of feasibility, the authors scale up a existing model to create Chunky Seal, which embeds 1024 bits—4× higher than prior work—while preserving image quality and robustness.

## Strengths
- **Novel geometric capacity bounds** – The paper develops a practical framework modeling images as a discrete grid and derives a family of bounds (Bounds 1-13) for capacity under PSNR and linear transformations (e.g., cropping, rotation, linearized JPEG). This provides a fresh, analytically tractable alternative to classical information-theoretic models that rely on unrealistic Gaussian assumptions.
- **Compelling minimal experiments** – By training a state-of-the-art model (Video Seal) on a single gray image with only a PSNR constraint, the authors isolate architectural limitations: Video Seal fails to embed 1024 bits where a simple linear model succeeds at 2048 bits and a handcrafted method reaches ~456k bits (Section 3.1, Table 1). This cleanly rules out hypotheses that data distribution, perceptual constraints, or robustness requirements explain the capacity gap.
- **Practical proof-of-concept** – Chunky Seal, a scaled-up version of Video Seal, demonstrates that significantly higher capacities are achievable under real-world constraints: it embeds 1024 bits while maintaining PSNR ~45 dB and robust bit accuracy across common perturbations (Table 3). This validates the core claim that current methods are far from saturation.

## Weaknesses
- **Heuristic robustness bounds** – The capacity bounds under linear transformations (Bounds 10-12) are explicitly heuristic, based on a product of singular values, and the paper shows they can both over- and under-estimate true capacity (Figures 8-9). While a conservative bound (Bound 13) is provided, the lack of formal guarantees limits the precision of theoretical predictions for robust watermarking.
- **Incomplete empirical gap analysis under robustness** – Chunky Seal is not evaluated on the same controlled gray-image task (PSNR-only, no robustness) where the capacity gap was first demonstrated. Without this, it remains unclear whether the scaled architecture actually narrows the gap to theoretical limits or merely improves performance on a more complex, distributed task.
- **Practical efficiency concerns** – Chunky Seal achieves higher capacity via massive model scaling (~90× larger embedder, ~23× larger extractor), but the paper only briefly notes the resulting size and latency implications. For real-world deployment, computational cost is critical, and the work does not explore parameter-efficient alternatives or provide a thorough efficiency analysis.

## Nice-to-Haves
- An ablation study identifying which components of Chunky Seal (e.g., channel multipliers, depth) are most critical for capacity gains.
- A capacity-matched comparison by training Video Seal at 1024 bits with proportional scaling to ensure fair evaluation of architectural improvements.
- Extension of the theoretical analysis to derive tighter, non-heuristic bounds for common non-linear transformations like standard JPEG.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism that the comparison includes older methods (e.g., HiDDeN)** – The paper provides a comprehensive benchmark (Table 4) that includes recent state-of-the-art methods (Video Seal, TrustMark, WAM); the inclusion of older baselines offers historical context and does not invalidate the core comparison.
- **Request for analysis of bit error patterns under attacks** – While insightful, this is a suggestion for deeper analysis rather than a flaw in the current contribution.
- **Demand for evaluation on a wider range of image distributions and resolutions** – The paper evaluates on SA-1B and COCO at 256px, which is standard for the field; extending further is a natural next step but not required for the stated scope.

## Novel Insights
The paper’s central insight is that the stagnation in watermarking capacity is not due to fundamental limits or the complexity of real-world constraints, but to architectural inefficiencies in current deep learning models. This is revealed by deriving achievable geometric bounds and showing that even simple baselines outperform state-of-the-art neural networks in minimal settings. The finding that models fail to utilize available resolution (e.g., tiling a low-resolution model achieves higher capacity) underscores a specific structural shortcoming, redirecting the field from incremental tuning to rethinking model design.

## Suggestions
- Evaluate Chunky Seal on the single gray-image PSNR-only task to quantify how close it comes to the handcrafted baseline and theoretical bounds.
- Provide quantitative metrics for Chunky Seal’s inference latency, memory footprint, and training cost relative to capacity gains to frame the efficiency trade-off clearly.
- Explore architectural innovations beyond naive scaling, such as frequency-aware layers or structured embeddings, that could achieve high capacity with better parameter efficiency.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Reject
