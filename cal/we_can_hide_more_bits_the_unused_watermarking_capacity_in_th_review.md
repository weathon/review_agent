=== CALIBRATION EXAMPLE 58 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly captures the paper's central claim: watermarking capacity is far from exhausted. The abstract succinctly summarizes the problem, the theoretical and empirical analysis, and the demonstration of higher capacity via Chunky Seal. The abstract's claims—that theoretical capacities are "orders of magnitude larger" and that a simple scaled model achieves 4× higher capacity—are supported by the paper's results. The only minor note is that the phrase "orders of magnitude" is most accurate for the PSNR-only case; with robustness constraints, the gap is still large but may be less extreme.

### Introduction & Motivation
The introduction effectively frames the stagnation in watermarking capacity and poses the critical question of whether we are near the theoretical limit. The three contributions are clearly stated and logically set up the rest of the paper. The motivation is strong and well-justified.

### Method / Approach (Bounds on Watermarking Capacity)
This section is a core contribution, deriving capacity bounds under PSNR and robustness constraints. The geometric approach is novel compared to classical information-theoretic models.

**Strengths:** The derivation is systematic, moving from absolute capacity to PSNR constraints and then robustness. The use of volume approximations and exact lattice counting for small radii is appropriate. The analysis of arbitrary cover images (corner case) shows the penalty is at most 1 bpp, which is insightful.

**Weaknesses:**
1. **Heuristic robustness bounds (Bounds 10–12)** are not rigorously justified. The paper acknowledges they are heuristic and shows examples where they can be inaccurate (Figures 8 and 9). This is a significant limitation because the paper's claim that robustness cannot explain the low empirical capacity relies on these heuristics. The conservative bound (13) is extremely loose, leaving uncertainty about the true capacity under practical transformations.
2. **Linearized JPEG (LinJPEG)** is an approximation that may not capture the full non-linearity and perceptual effects of real JPEG compression. The impact on capacity bounds for compression robustness is therefore not fully established.
3. **Data distribution argument** uses VQ-VAE/VQGAN to estimate the number of perceptually distinct images. While plausible, this is an approximation and not a fundamental limit.

Despite these weaknesses, the theoretical analysis convincingly shows that, even under conservative assumptions, capacity should be much higher than current methods achieve.

### Experiments & Results (Empirical Performance is Much Lower than Predicted)
This section tests five hypotheses to explain the gap between theory and practice. The experiments are well-designed and controlled.

**Strengths:** 
- The simplified setup (single gray image, only PSNR constraint) effectively isolates architectural limitations from other complexities.
- Video Seal's failure to embed 1024 bits, while linear and handcrafted models succeed, strongly suggests structural underperformance (hypothesis E).
- The tiling experiment shows that Video Seal does not utilize full resolution, and higher capacities are achievable even with the same architecture.

**Weaknesses:**
- The handcrafted and linear baselines are not robust, so they only demonstrate potential in a non-robust setting. However, the tiling experiment shows that capacities far beyond current models are feasible even with a trained model (though still without robustness).
- The experiment does not fully rule out hypothesis D (bounds are unrealistic) for the robust case, since the robustness bounds are heuristic. However, the conservative bound (Table 2) still suggests capacities >900 bits for aggressive crops, which is much higher than current models.

Overall, the experiments robustly support the claim that models underperform relative to what is achievable, even in simple settings.

### Better Performance in Practice is Possible: Chunky Seal
This section scales up Video Seal to achieve 1024 bits with comparable quality and robustness.

**Strengths:** Chunky Seal demonstrates that 4× higher capacity is attainable with a straightforward scaling of a known architecture. The comprehensive evaluation (Table 3) shows maintained quality and robustness across many transformations, validating the feasibility of higher capacities.

**Weaknesses:**
- The model size increases dramatically (embedder 90×, extractor 23×), making it impractical for deployment. The paper appropriately notes this is a proof of concept, not a practical solution.
- No ablation study is provided to disentangle the effects of increased dimensions, use of all channels, and training tricks like gradient clipping.
- The robustness results, while comparable, show slight degradations (e.g., higher LPIPS, lower bit accuracy on some augmentations). A more detailed analysis of these trade-offs would be helpful.

Nevertheless, Chunky Seal successfully proves that higher capacity is achievable without sacrificing much quality or robustness, reinforcing the paper's main argument.

### Discussion and Conclusions
The discussion appropriately highlights the implications (e.g., embedding full manifests) and limitations. The proposed sanity checks for future watermarking methods are a valuable contribution. The limitations section is honest, covering the heuristic nature of robustness bounds, the tractability of theoretical analysis, and the impractical size of Chunky Seal.

**Missing element:** The broader impact section could briefly discuss potential negative societal impacts (e.g., watermarking for surveillance or censorship), but this is not critical for the paper's technical contribution.

### Writing & Clarity
The paper is generally well-written and logically structured. The mathematical derivations are detailed, and the figures effectively illustrate the bounds and results. Some formatting artifacts from PDF parsing are present but do not impede understanding. The paper is dense but appropriate for an ICLR audience.

### Overall Assessment
This paper makes a significant and timely contribution by challenging the assumption that watermarking capacity is near saturation. It provides compelling theoretical bounds (though partially heuristic) and rigorous experiments showing that current models are far from optimal, even in simplified settings. The demonstration of higher capacity via Chunky Seal, despite its impractical size, solidifies the claim that architectural and training innovations are needed. The weaknesses—most notably the lack of tight robustness bounds and the simplicity of the empirical setups—are acknowledged by the authors and do not undermine the core message. For ICLR, which values novel, impactful ideas that open new research directions, this paper is strong and likely to be accepted. The proposed sanity checks provide a concrete path forward for the community.

# Neutral Reviewer
## Balanced Review

### Summary
This paper systematically investigates the theoretical limits of image watermarking capacity under perceptual quality (PSNR) and robustness constraints. It establishes upper bounds indicating capacities are orders of magnitude higher than what current deep learning-based models achieve. To validate this gap, the authors conduct controlled experiments showing even state-of-the-art models like Video Seal fail in simplified settings, and they propose Chunky Seal, a scaled-up model that embeds 1024 bits (4x the baseline) while maintaining comparable quality and robustness.

### Strengths
1. **Clear and Important Research Question:** The paper directly addresses a perceived stagnation in watermarking capacity (100-200 bits) by asking how close we are to fundamental limits. This is a well-motivated and significant question for the field.
2. **Systematic Theoretical Analysis:** The derivation of capacity bounds is methodical, starting from the absolute capacity of the image space, adding PSNR constraints (considering gray/central vs. arbitrary images), and then incorporating robustness to linear transformations (e.g., crop, rotation, linearized JPEG). The use of geometric (ball-cube intersection) rather than information-theoretic arguments makes the analysis more accessible and tied to practical constraints.
3. **Rigorous Empirical Validation:** The authors effectively bridge theory and practice. They strip down the problem to a single gray image with only a PSNR constraint, demonstrating that Video Seal cannot embed 1024 bits where simple linear and handcrafted models can reach >450k bits (Table 1, Fig. 5,6). This elegantly isolates architectural/training limitations from real-world complexities.
4. **Demonstrated Practical Improvement:** Chunky Seal successfully increases capacity to 1024 bits with quality/robustness on par with Video Seal (Table 3). This serves as concrete proof that higher capacities are practically achievable, supporting the paper's core thesis.

### Weaknesses
1. **Heuristic and Incomplete Theoretical Bounds:** The robustness bounds (Bounds 10-12) are explicitly labeled as heuristics, relying on singular-value products which the authors note can both under- and over-estimate true capacity (Figs. 8,9). The provided "conservative" bound (Bound 13) is extremely loose. A more formal treatment of capacity under non-linear quantization (Q) is missing, leaving the robustness analysis as the least rigorous part of the theory.
2. **Limited Architectural Innovation:** Chunky Seal is presented as a "simple scale-up" of Video Seal (90x larger embedder, 23x larger extractor). While it proves the feasibility of higher capacity, it does not introduce novel architectural principles. The paper acknowledges this and calls for future innovation, but the proposed model itself is not a significant algorithmic advance.
3. **Sparse Discussion of Efficiency and Scalability:** The paper focuses on capacity but gives limited attention to the computational and memory costs of scaling models. Chunky Seal's massive size (∼1.8B params total) is noted, but its implications for training cost, inference latency, and practical deployment are not analyzed. Efficiency is a key concern for real-world watermarking.
4. **Parser Artifacts Obscure Details:** While not the authors' fault, the extracted text contains numerous garbled tables (e.g., Fig. 1, Table 1), broken equation references, and misplaced text, which hinders a complete assessment of the experimental details and numerical results. This forces the reviewer to infer some content.

### Novelty & Significance
**Novelty:** The primary novelty lies in the comprehensive geometric framework for deriving actionable capacity bounds for digital image watermarking, moving beyond classical Gaussian channel models. The empirical demonstration that SOTA models fail catastrophically in simplified, analyzable settings is also a novel and impactful critique.
**Significance:** The work is highly significant for the watermarking community. It shifts the narrative from incremental improvements on a saturated Pareto front to recognizing a vast, unexploited capacity potential. The proposed "sanity checks" (scaling with image size, outperforming linear baselines, etc.) provide valuable guidance for future research. Successfully embedding 1024 bits opens doors for applications like embedding full manifests.

### Suggestions for Improvement
1. **Strengthen the Robustness Theory:** A major weakness is the heuristic nature of the robustness bounds. Future work (or a revision) should aim for more formal guarantees, perhaps using tools from lattice theory or quantization analysis to better characterize the capacity reduction under linear+Q transformations.
2. **Explore Architectural Inductive Biases:** The paper convincingly argues that current architectures are the bottleneck. To elevate the contribution, the authors could propose and test a novel architectural component or training strategy specifically designed to better utilize the available pixel-space dimensions, moving beyond naive scaling.
3. **Include an Efficiency-Aware Analysis:** A discussion or small experiment on the trade-off between capacity, model size, and inference speed would greatly increase the practical relevance of Chunky Seal. Could a more efficient architecture achieve similar gains?
4. **Clarify Experimental Details Amidst Artifacts:** Given the parsing issues, the authors should ensure the camera-ready version has perfectly clear tables and figures, especially for key results like Table 1 (handcrafted model capacities) and Fig. 5 (training sweeps). Explicitly stating the hyperparameters that yielded the best Chunky Seal results would aid reproducibility.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison to recent high-capacity watermarking methods.** The paper only compares Chunky Seal to a scaled version of Video Seal and a few older methods (HiDDeN, MBRS). It must be compared against other recent works claiming higher capacity (e.g., InvisMark, RoSteALS, WAM, MuST) to substantiate the claim of pushing the Pareto frontier.
2. **Robustness evaluation under stronger, composite, or adversarial attacks.** The tested augmentations (cropping, rotation, JPEG) are standard but mild. The claim of comparable robustness is undermined without testing against state-of-the-art watermark removal attacks, strong diffusion model edits, or realistic pipelines (e.g., social media processing chains combining multiple transforms).
3. **Ablation study on Chunky Seal's architectural changes.** The model is a scaled-up Video Seal. The contribution is unclear without ablations: which scale-up factor (channels, depth, using all color channels) is critical for the capacity gain? A component-wise analysis is needed to attribute improvements.
4. **Capacity scaling experiments with image resolution.** The theory predicts capacity should scale linearly with pixel count. The paper should test this empirically by training/evaluating Chunky Seal or baselines at multiple resolutions (e.g., 128x128, 512x512) to see if the observed bpp remains constant or drops.

### Deeper Analysis Needed (top 3-5 only)
1. **Validation of heuristic robustness bounds.** The paper admits Bounds 10-12 are heuristic and not proven lower/upper bounds. The core claim that "robustness constraints cannot explain the low capacity" relies on these. They must provide empirical validation (e.g., via constructive coding schemes) to show these bounds are approachable, or at least discuss their potential looseness.
2. **Root-cause analysis of architectural failure.** The paper identifies that Video Seal fails on a simple gray-image task, but only hypothesizes "structural limitations." A deeper analysis is needed: e.g., examining gradient flow, loss landscape, or representational capacity of the U-Net/ConvNeXt for learning the identity mapping required for the residual.
3. **Analysis of the trade-off curve (capacity vs. quality/robustness).** The paper claims Chunky Seal increases capacity 4x while preserving quality/robustness. This is a single point. To argue that the Pareto front can be pushed, they should show the trade-off curve: how does capacity scale with model size or training, and what is the associated cost in PSNR or robustness for a fixed training budget?

### Visualizations & Case Studies
1. **Visualizations of failure modes and capacity saturation.** Show cases where Chunky Seal fails to embed 1024 bits correctly (e.g., on complex textures, high-frequency areas). Visualizing the per-pixel residual pattern for different messages would reveal if the model uses the full spatial capacity or relies on specific patterns.
2. **Case studies on images that stress the bounds.** Test the handcrafted and linear baselines on non-gray, natural images to see if the theoretical PSNR-only capacity is approachable in practice. This would bridge the gap between the simplified theory and real data.

### Obvious Next Steps
1. **Implement and test a baseline matching the theoretical setup.** The handcrafted embedder (Eq. 2) is for a gray image. They should design and test a *nonlinear* but capacity-optimal model (e.g., a learned projection) for the PSNR-only case on natural images to see how close one can get to the bound.
2. **Explore architectural innovations, not just scaling.** The conclusion states that new architectures are needed, but the paper only presents scaled-up Video Seal. They should have at least proposed and tested one novel architectural modification (e.g., different residual pathways, frequency-domain processing) informed by the geometric analysis.
3. **Extend theory to non-linear and non-blind scenarios.** The theoretical analysis is limited to linear transforms and a blind decoder. A discussion or initial bounds for common non-linearities (e.g., non-linear color adjustments, neural compression) and non-blind decoding would strengthen the claim of generality.

# Final Consolidated Review
## Summary
This paper investigates the fundamental limits of image watermarking capacity under perceptual quality (PSNR) and robustness constraints. It establishes theoretical upper bounds showing that achievable capacities are orders of magnitude higher than current deep learning methods, and through controlled experiments and a scaled-up model, demonstrates that this gap stems from architectural limitations rather than theoretical ceilings.

## Strengths
- **Novel geometric framework for capacity bounds:** The systematic derivation moves from absolute limits to PSNR constraints and linear robustness transformations using a ball-cube intersection model, providing a practical and actionable alternative to classical information-theoretic approaches.
- **Clever empirical isolation of architectural bottlenecks:** By simplifying the task to watermarking a single gray image with only a PSNR constraint, the paper shows that state-of-the-art models (Video Seal) fail to embed 1024 bits while simple linear and handcrafted models achieve up to 456,509 bits, proving that current architectures severely underutilize available capacity.
- **Proof-of-concept demonstration of higher capacity:** Chunky Seal, a scaled-up version of Video Seal, successfully embeds 1024 bits (4× the baseline) while maintaining comparable image quality and robustness across standard transformations, validating that significantly higher capacities are practically attainable.

## Weaknesses
- **Heuristic robustness bounds introduce uncertainty:** The capacity bounds for linear transformations with quantization (Bounds 10–12) are explicitly heuristic and not rigorously derived; while the conservative bound (Bound 13) still suggests higher capacity than current methods, the lack of tight guarantees leaves the exact capacity under practical robustness constraints less firmly established.
- **Practical limitations of the demonstrated improvement:** Chunky Seal achieves higher capacity primarily through massive model scaling (90× larger embedder, 23× larger extractor) without architectural innovation, and the paper does not analyze the efficiency trade-offs (e.g., inference latency, memory footprint), making it a proof-of-concept rather than a deployable solution.

## Nice-to-Haves
- A deeper investigation into why current architectures fail to utilize available capacity (e.g., through gradient flow or representational capacity analysis) could guide future architectural designs.
- Exploring the capacity-quality-robustness trade-off curve by varying model size or training budget would further substantiate the claim that the Pareto front can be pushed.

## Novel Insights
The paper’s central insight is that the observed stagnation in watermarking capacity is not due to fundamental limits but to suboptimal model architectures. By deriving achievable bounds and showing that even simple models outperform state-of-the-art neural networks in controlled settings, it shifts the community’s focus from incremental improvements to rethinking architectural inductive biases and training strategies. The proposed sanity checks (e.g., scaling capacity with image size, outperforming linear baselines) provide concrete guidance for future work.

## Suggestions
- In the camera-ready version, ensure that tables and figures (e.g., Table 1, Figure 5) are clearly presented to avoid ambiguity from parsing artifacts, and explicitly state the hyperparameters used for Chunky Seal’s best results to aid reproducibility.
- Consider adding a brief discussion on the computational costs and potential efficiency optimizations for Chunky Seal to address practical deployment concerns.

# Actual Human Scores
Individual reviewer scores: [4.0, 6.0, 6.0]
Average score: 5.3
Binary outcome: Reject
