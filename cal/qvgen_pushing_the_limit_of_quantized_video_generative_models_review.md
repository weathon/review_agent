=== CALIBRATION EXAMPLE 68 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:** The title "QVGEN: PUSHING THE LIMIT OF QUANTIZED VIDEO GENERATIVE MODELS" accurately reflects the core contribution: a new QAT framework for video DMs. The abstract clearly states the problem (ineffective low-bit quantization for video DMs), the solution (auxiliary modules Φ and rank-decay), and the key results (first to achieve full-precision comparable 4-bit quality, significant gains on VBench). Claims are supported by the presented evidence.

**Introduction & Motivation:** The introduction effectively motivates the problem. The computational demands of SOTA video DMs are well-documented, and quantization is correctly identified as a key solution. The gap is clear: prior quantization methods (designed for images) fail dramatically for video DMs at ≤4 bits, as shown in Fig. 1. The contributions are explicitly listed and align with the paper's structure.

**Method / Approach (Section 3):**
*   **Theoretical Analysis & Auxiliary Modules (Φ):** The convergence analysis (Thm. 3.1) provides a principled motivation for reducing the gradient norm \(\|\mathbf{g}_t\|_2\). The assumption of convexity is a simplification, but the authors responsibly address this by providing a nonconvex analysis in Appendix C, strengthening the theoretical grounding. The introduction of trainable Φ modules, initialized with the weight quantization error, is a natural and logical step to reduce quantization-induced error and stabilize training. Figure 3 provides strong empirical validation, showing Φ leads to lower gradient norms and training loss compared to Q-DM.
*   **Rank-Decay Strategy:** The core innovation is the progressive removal of Φ via *rank-decay*. The observation that the singular value spectrum of \(\mathbf{W}_\Phi\) becomes increasingly dominated by small values during training (Fig. 4) is insightful and provides a clear rationale for a low-rank removal strategy. The procedure—iterative SVD decomposition, application of a rank-based regularization \(\gamma\) to decay low-impact components, and truncation—is well-described and visualized in Fig. 2 and Alg. 1. A minor concern is the computational overhead of performing SVD on \(\mathbf{W}_\Phi\) for every linear layer during each decay phase. While likely manageable (as \(\mathbf{W}_\Phi\) is low-rank), the paper does not quantify this cost or discuss potential approximations.
*   **Reproducibility:** The method description is sufficiently detailed. The algorithm (Alg. 1) clarifies the training loop. Key hyperparameters (initial rank \(r\), shrinking ratio \(\lambda\), annealing schedule for \(u\)) are stated and their choices are justified through ablations (Tabs. 4, 5, I).

**Experiments & Results (Section 4):**
*   **Scope & Baselines:** The experimental scope is **excellent** and a major strength. Testing across four different SOTA video DMs (CogVideoX-2B/5B, Wan-1.3B/14B) covering a wide parameter range (1.3B to 14B) demonstrates generality and scalability. The chosen baselines are comprehensive, including recent SOTA PTQ (ViDiT-Q, SVDQuant) and QAT (LSQ, Q-DM, EfficientDM) methods adapted for video.
*   **Evaluation Metrics:** Using VBench and VBench-2.0 (for huge models) is appropriate and standard. Reporting all 8 dimensions of VBench provides a nuanced view of performance across different video qualities (imaging, motion, consistency). The addition of traditional metrics like PSNR/SSIM/LPIPS in Appendix J is a good supplement.
*   **Main Results:** Table 1 presents compelling results. QVGen consistently and significantly outperforms all baselines at both W4A4 and W3A3. The claim that it is the "first to reach full-precision comparable quality under 4-bit settings" is supported, especially for CogVideoX-2B where most metrics are within 2% of the BF16 model. The 3-bit results, while showing a larger gap, still represent a substantial advance over prior art. Results on huge 5B and 14B models (Tab. 2, Fig. 5) successfully demonstrate scalability, with W4A4 showing minimal drops on VBench-2.0.
*   **Ablation Studies:** The ablations are thorough and effectively isolate the contribution of each component (Tab. 3). The studies on shrinking ratio \(\lambda\) (Tab. 4) and initial rank \(r\) (Tab. 5) are informative. The comparison of decay strategies in Tab. 6 is particularly strong, showing the superiority of the proposed *rank-decay* over intuitive alternatives like linear decay, magnitude pruning ("Sparse"), or residual quantization ("Res. Q.").
*   **Efficiency Analysis:** The discussion of inference speedup (1.21-1.44× on A800) and memory reduction is honest, noting that current gains are limited by kernel implementation and projecting higher gains with fusion. The training cost analysis (Tab. 7) is fair, showing QVGen adds only ~1.02× time and ~1.01× memory over Q-DM, a reasonable trade-off for its large performance gains.
*   **Minor Points:** The combination experiment with SVDQuant (Appendix I) is a nice addition, showing QVGen can build on strong PTQ initialization for further gains. The analysis of gradient norms in video vs. image QAT (Appendix H.1) provides valuable insight into why video quantization is uniquely challenging.

**Writing & Clarity:** The paper is generally well-written and logically structured. The figures are effective. There are occasional minor grammatical awkwardness or parser artifacts (e.g., "iter ~~p~~ er ~~d~~ ecay ~~p~~ hase"), but these do not impede understanding.

**Limitations & Broader Impact:**
*   **Limitations:** The stated limitation (focus on video, potential future extension to NLP) is accurate but somewhat narrow. Other limitations are implicit but could be stated: 1) The method currently quantizes only linear layers, leaving attention and normalization layers in full precision; quantizing these is a future challenge. 2) The 3-bit quantization, while improved, still shows significant quality degradation on some metrics (e.g., Scene Consistency for large models), indicating a remaining frontier. 3) The training cost, while reasonable relative to Q-DM, is still non-trivial (up to ~182 GPU days for Wan 14B).
*   **Broader Impact:** The paper lacks a broader impact statement. Quantization enables more efficient deployment, which has positive environmental implications (lower energy use) and democratization potential (running on consumer hardware). However, it also lowers the barrier to generating high-quality synthetic video, which could exacerbate risks related to misinformation and deepfakes. A brief discussion would be appropriate.

### Overall Assessment
This is a strong paper that makes a clear and significant contribution. It is the first to demonstrate effective ultra-low-bit (4-bit and 3-bit) quantization for state-of-the-art video diffusion models via a novel QAT framework. The method is theoretically motivated, empirically well-validated across multiple models and scales, and thoroughly ablated. The work meets the high bar for ICLR in terms of novelty, technical depth, and experimental rigor. The most important concerns are relatively minor: a more explicit discussion of computational overhead from SVD and a broader impact statement. The core contribution stands solidly.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes QVGen, a novel quantization-aware training (QAT) framework for video diffusion models that enables effective 4-bit and 3-bit quantization. The core idea is to introduce auxiliary modules (Φ) to reduce quantization errors and stabilize training by lowering gradient norms, and then progressively remove these modules via a rank-decay strategy to eliminate inference overhead. Extensive experiments on multiple state-of-the-art video DMs (up to 14B parameters) show that QVGen achieves full-precision comparable quality at 4-bit and significantly outperforms existing quantization methods.

### Strengths
1. **Strong Empirical Results**: The paper provides comprehensive experiments across four SOTA video DMs (CogVideoX and Wan families, 1.3B–14B parameters), demonstrating that QVGen is the first to achieve full-precision comparable performance at 4-bit and shows substantial improvements over existing PTQ and QAT methods. For instance, 3-bit CogVideoX-2B achieves +25.28 in Dynamic Degree and +8.43 in Scene Consistency on VBench (Table 1).
2. **Theoretical Foundation**: The paper includes a theoretical analysis (convex and nonconvex) linking gradient norm reduction to improved convergence in QAT, motivating the use of auxiliary modules Φ. This analysis is supported by empirical measurements of gradient norms and training loss (Figure 3).
3. **Innovative Rank-Decay Strategy**: To avoid inference overhead from Φ, the authors propose a novel rank-decay strategy that progressively shrinks Φ by repeatedly applying SVD and a rank-based regularization to decay low-contributing components. Ablations (Tables 3-6) validate its effectiveness over naive decay or pruning strategies.
4. **Comprehensive Evaluation**: The evaluation uses multiple metrics (VBench, VBench-2.0), includes both quantitative scores and qualitative examples (Figure 1, Appendix P), and reports efficiency gains (inference latency, model size reduction). The method also scales to large models (Wan 14B) with minimal performance drop (Table 2, Figure 5).

### Weaknesses
1. **Limited Analysis of 3-bit Performance**: While 4-bit results are strong, 3-bit quantization still shows significant degradation, especially on challenging metrics like Scene Consistency (Table 2). The paper does not deeply analyze the causes or propose targeted improvements for 3-bit, leaving a gap for ultra-low-bit quantization.
2. **Training Efficiency Overhead**: Although inference overhead is eliminated, the training process introduces additional cost (Table 7 shows ~1.02× GPU-days and ~1.01× memory vs. Q-DM). The paper acknowledges this but does not explore ways to reduce training cost (e.g., combining with parameter-efficient methods like LoRA).
3. **Lack of Comparison to Some Relevant Works**: The paper focuses on comparing to general QAT/PTQ methods for image DMs, but does not compare to recent video-specific quantization methods like Q-DiT (Chen et al., 2024b) or QVD (Tian et al., 2024) in the main experiments (only mentioned in related work). Including these would strengthen the claims of being state-of-the-art for video.
4. **Theoretical Assumptions May Not Hold**: The convergence analysis assumes convexity (or smoothness) of the loss, which may not strictly hold for deep non-convex networks. While the nonconvex extension is provided, the practical impact of gradient norm reduction on convergence in such complex models is not fully validated beyond empirical trends.

### Novelty & Significance
**Novelty**: The paper presents the first dedicated QAT framework for video diffusion models, introducing auxiliary modules and a novel rank-decay strategy. The theoretical link between gradient norm and QAT convergence for video DMs is also new.

**Clarity**: The paper is well-structured, with clear motivations, method descriptions, and experiments. Figures and tables are informative, though some formatting artifacts from PDF parsing exist (e.g., broken references like "~~9~~5@outlook.com", misplaced figure references, garbled tables). These do not detract from understanding.

**Reproducibility**: The paper provides implementation details (Section 4.1, Appendix D), training hyperparameters, and code/model availability. However, the training cost for large models (e.g., Wan 14B requires ~182 GPU-days) may limit reproducibility for some researchers.

**Significance**: Efficient deployment of video DMs is a critical challenge. QVGen enables high-quality 4-bit quantization with no inference overhead, pushing the Pareto frontier for accuracy vs. model size. This could significantly impact real-world applications and inspire further work on video model compression.

### Suggestions for Improvement
1. **Deeper Analysis of 3-bit Limitations**: Investigate why 3-bit quantization struggles with metrics like Scene Consistency and explore techniques (e.g., mixed-precision, better initialization) to close the gap.
2. **Reduce Training Cost**: Integrate parameter-efficient fine-tuning (e.g., LoRA) with QVGen to lower training memory and time, as suggested in Section 4.4, and report results.
3. **Include More Video-Specific Baselines**: Compare QVGen directly with recent video quantization works (e.g., Q-DiT, QVD) to better establish state-of-the-art.
4. **Expand Theoretical Validation**: Provide more empirical evidence linking gradient norm reduction to convergence improvement in the nonconvex setting, e.g., by analyzing loss landscape or gradient variance.
5. **Discuss Broader Applicability**: While the paper shows preliminary results on image generation (Appendix M), a more thorough evaluation on other tasks (e.g., NLP) would strengthen the claim of generalizability.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison with video-specific PTQ methods at the same low bit-width (e.g., W4A4).** The paper compares against ViDiT-Q and SVDQuant, but these are evaluated at higher bit-widths (W4A6) or with per-group quantization. A direct W4A4 comparison is absent, undermining the claim that QAT is necessary and superior for 4-bit video DMs.
2. **Ablation isolating the effect of gradient norm reduction from the added capacity of Φ.** The theory links lower gradient norm to better convergence, but no experiment shows the improvement is due to gradient norm reduction and not simply the extra parameters. A control where Φ is added but gradient norm is artificially kept high (e.g., via gradient clipping) would test the core claim.
3. **Quantization of attention layers and its impact.** The paper quantizes only linear layers, leaving attention at full precision. Since attention dominates compute in video DMs, ignoring its quantization limits the practical speedup claims and leaves the method's applicability to the full model incomplete.
4. **Evaluation on held-out video datasets beyond VBench prompts.** The training uses OpenVidHQ and testing uses VBench's unaugmented prompts. Testing on independent datasets (e.g., Kinetics captions) is needed to demonstrate generalization and avoid overfitting to the evaluation benchmark.
5. **Exploration of 2-bit quantization or mixed-precision schemes.** The paper focuses on 3/4 bits but does not explore 2-bit or mixed precision (e.g., 4-bit weights, 8-bit activations). Given the claim of "pushing the limit," probing lower bit-widths is a logical next step to understand the boundary.

### Deeper Analysis Needed (top 3-5 only)
1. **Why is video QAT harder than image QAT?** The paper shows video DMs have larger gradient norms (Table C) but does not investigate the root cause. Analysis of temporal modeling's effect on the loss landscape (e.g., Hessian spectra across frames) would provide deeper insight into the unique challenges.
2. **Per-component breakdown of quantization error.** Scene Consistency is notably difficult to maintain. Analyzing whether errors stem more from spatial vs. temporal components, or specific layers (e.g., cross-attention), would guide future improvements.
3. **Sensitivity analysis of hyperparameters (rank \(r\), decay ratio \(\lambda\)) across different model architectures.** Ablations are done on Wan 1.3B only. Showing robustness across model sizes and families (CogVideoX vs. Wan) is necessary to claim generality.
4. **Convergence dynamics during rank-decay phases.** The paper claims decay has minimal impact, but no plots show how loss and gradient norm behave when rank is reduced. Monitoring these during decay would validate stability and inform scheduling.

### Visualizations & Case Studies
1. **Visualization of per-frame quantization error over time.** Plotting MSE or activation divergence between full-precision and quantized models across frames would reveal if errors accumulate temporally, explaining drops in temporal metrics.
2. **Case studies of failure modes, especially for Scene Consistency and 3-bit models.** The paper shows successful samples. Highlighting where the method fails (e.g., scene collapse, motion artifacts) with analysis would clarify limitations and direct future work.
3. **Layer-wise singular value distribution before and after rank-decay.** Figure 4 aggregates singular values; showing how distributions differ across layer types (attention vs. FFN) and how decay affects them would validate the rank-decay strategy's design.

### Obvious Next Steps
1. **Quantize attention layers.** Extending QVGen to include attention quantization (even at higher bits initially) is a necessary step to achieve full-model acceleration and should have been explored.
2. **Combine with efficient attention methods (e.g., SVG) for end-to-end speedup.** The paper mentions orthogonality but does not present results combining QVGen with sparsity or efficient attention. A joint experiment would demonstrate practical synergy.
3. **Explore 2-bit quantization or mixed precision.** Given the focus on ultra-low bits, attempting 2-bit weight/activation quantization is a clear next step to truly push the limit.
4. **Apply to other video generation tasks (e.g., video prediction, interpolation).** Testing on tasks beyond text-to-video would show broader applicability and should be included to strengthen the contribution.
5. **Release pre-trained quantized models and detailed code.** For reproducibility and impact, providing easy-to-use checkpoints and training scripts is essential; the paper only mentions code will be available.

# Final Consolidated Review
## Summary
This paper introduces QVGen, a quantization-aware training framework for video diffusion models that enables effective 4-bit and 3-bit quantization. The method introduces auxiliary modules to reduce the gradient norm and improve training convergence, then progressively removes them via a novel rank-decay strategy to eliminate inference overhead. Experiments across four state-of-the-art video DMs (1.3B to 14B parameters) show that QVGen achieves full-precision comparable quality at 4-bit and significantly outperforms existing quantization methods.

## Strengths
- **Strong and extensive empirical validation.** QVGen is evaluated across four different SOTA video diffusion models (CogVideoX and Wan families, 1.3B–14B), demonstrating it is the first to achieve full-precision comparable performance at 4-bit weights and activations. It consistently and substantially outperforms prior PTQ and QAT methods adapted from image generation, e.g., achieving gains of +25.28 in Dynamic Degree and +8.43 in Scene Consistency for a 3-bit model on VBench.
- **Innovative and well-motivated method.** The core contributions—using auxiliary modules (Φ) to reduce the gradient norm (theoretically motivated via convergence analysis) and the novel *rank-decay* strategy to progressively eliminate Φ without harming performance—are novel and effectively address the unique challenges of video DM quantization. Ablation studies (Tables 3-6) clearly validate each component's contribution and the superiority of *rank-decay* over intuitive alternatives.
- **Demonstrated scalability and practical efficiency.** The method scales successfully to huge models (Wan 14B), with W4A4 quantization showing only a ~1% drop on the comprehensive VBench-2.0 suite. The final quantized model has zero inference overhead, adheres to standard uniform quantization for kernel compatibility, and achieves measured speedups (1.21–1.44× on an A800 GPU).

## Weaknesses
- **Incomplete comparison against video-specific quantization baselines.** While the paper extensively compares against image-based QAT/PTQ methods, it does not include a direct, low-bit (W4A4) comparison with recent video-specific PTQ works like Q-DiT or QVD, which are only mentioned in the related work. This omission slightly weakens the claim of being state-of-the-art *for video*, as the strongest prior art in the same domain is not quantitatively benchmarked.
- **Limited analysis and results for 3-bit quantization.** Although 3-bit results represent a significant advance over prior methods, the performance gap to full-precision remains substantial, especially on demanding metrics like Scene Consistency for large models (Table 2). The paper does not deeply analyze the root causes of these remaining failures or propose targeted improvements, leaving the 3-bit frontier less explored.
- **Training cost, while reasonable, is non-trivial.** QVGen adds ~1.02× training time and ~1.01× GPU memory compared to the Q-DM baseline (Table 7), and training the largest model (Wan 14B) requires ~182 GPU days. While the paper acknowledges this and suggests future work with parameter-efficient fine-tuning, the current training overhead may limit accessibility for some researchers.

## Nice-to-Haves
- Extending the quantization to attention layers, which are currently kept in full precision, would provide more comprehensive acceleration and is a natural next step.
- A more detailed investigation into why video QAT exhibits higher gradient norms than image QAT (beyond the initial observation in Appendix H.1), perhaps via analysis of the temporal loss landscape, could yield deeper insight.
- Including an experiment that more directly isolates the effect of gradient norm reduction from the added parameter capacity of Φ (e.g., by controlling the gradient norm via clipping while keeping Φ) could further solidify the theoretical motivation.

## Novel Insights
The paper provides two key novel insights. First, it theoretically and empirically establishes that reducing the gradient norm is critical for stabilizing QAT convergence in video diffusion models, explaining why prior image-based QAT methods fail. Second, it observes that the auxiliary modules' weight matrices naturally become low-rank during training (with an increasing proportion of small singular values), which motivates the design of the *rank-decay* strategy: progressively removing low-impact components via SVD and a rank-based regularization is more effective than naive pruning or linear decay. These insights are central to the method's success.

## Suggestions
- Conduct and report a direct W4A4 comparison with recent video-specific PTQ methods (e.g., Q-DiT, QVD) to firmly establish the state-of-the-art for video quantization.
- For the camera-ready version, ensure all figure and table references in the text are correct (some appear garbled in the provided PDF extract).

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0, 8.0, 6.0]
Average score: 6.8
Binary outcome: Accept
