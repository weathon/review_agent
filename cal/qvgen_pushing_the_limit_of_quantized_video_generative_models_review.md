=== CALIBRATION EXAMPLE 76 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- The title accurately reflects the paper's goal of advancing quantization for video generative models.
- The abstract clearly summarizes the problem, proposed solution, and key results. Claims of being "the first" to achieve full-precision comparable quality under 4-bit are strong and need to be backed by the experimental data. The abstract also highlights concrete improvements (e.g., +25.28 in Dynamic Degree) which are supported in the results.

### Introduction & Motivation
- The problem is well-motivated: video diffusion models are computationally prohibitive, and existing quantization methods fail under ultra-low bits (≤4-bit). Fig. 1 provides compelling visual evidence of the shortcomings of prior art.
- Contributions are clearly stated: a novel QAT framework, theoretical analysis linking gradient norm to convergence, auxiliary modules Φ, a rank-decay strategy to remove Φ without inference overhead, and extensive experiments.

### Method / Approach
- **Theoretical analysis (Sec. 3.1):** The connection between gradient norm reduction and improved convergence is well-reasoned. The convexity assumption in Thm. 3.1 is relaxed in Appendix C with a nonconvex analysis, which is appropriate. The empirical validation in Fig. 3 (showing reduced ∥g_t∥₂ and loss for the Φ-enhanced method) strongly supports the theory.
- **Auxiliary modules Φ:** The idea of adding a learnable compensation term is intuitive. Initializing Φ with the weight quantization error is sensible. The ablation (Tab. 3) confirms its large benefit.
- **Rank-decay strategy (Sec. 3.2):** The insight that Φ's singular values become increasingly small during training (Fig. 4) motivates a progressive, low-rank removal. The strategy of iterative SVD and rank-based regularization γ is novel and well-explained. Algorithm 1 in Appendix A makes the procedure reproducible.
- **Potential gaps / questions:**
  1. **Hyperparameter sensitivity:** The choices of initial rank r=32 and shrinking ratio λ=1/2 are justified via ablations (Tabs. 4, 5), but a deeper sensitivity analysis (e.g., how performance varies with r across models) would strengthen the method's generalizability.
  2. **Computational overhead of SVD:** Performing SVD on every Φ repeatedly during training could be costly, especially for huge models. The paper should discuss the practical overhead (e.g., time per decay step) or note that SVD is applied per layer only occasionally during phase transitions.
  3. **Assumptions about singular values:** The claim that small singular values correspond to "low-contributing components" is supported by prior work, but it would be helpful to verify that decaying these components indeed has minimal impact on the training loss during the decay phases.

### Experiments & Results
- **Setup:** Extensive experiments across 4 SOTA video DMs (1.3B to 14B) and multiple bit-widths (3-bit, 4-bit). Training details (dataset, optimizer, epochs) are sufficiently detailed for reproducibility.
- **Baselines:** Appropriate selection of PTQ and QAT methods from image diffusion literature, adapted to video. The adjustments for fair comparison (e.g., per-group quantization for SVDQuant) are clearly explained.
- **Quantitative results (Tabs. 1, 2):** QVGen consistently outperforms all baselines, often by large margins, especially at 3-bit and 4-bit. The claim of "full-precision comparable" performance for W4A4 is largely supported, though some metrics (e.g., Scene Consistency for CogVideoX-2B) still show a gap. Results on huge models (Tab. 2) demonstrate scalability.
- **Ablations (Sec. 4.3):** Thorough and convincing. Each component (Φ and rank-decay) is shown to be essential (Tab. 3). The comparison of decay strategies (Tab. 6) strongly favors the proposed rank-decay. Hyperparameter studies (λ, r) provide practical guidance.
- **Efficiency (Sec. 4.4):** Inference speedups (1.21×–1.44×) and model size reduction are reported. The speedups are modest, and the authors correctly note that kernel fusion could improve them. Training cost relative to baselines (Tab. 7) is reasonable (slightly higher than Q-DM but much better performance).
- **Qualitative results (Fig. 1, Appendix P):** Visual comparisons clearly show QVGen's superiority over baselines and near-BF16 quality at 4-bit.
- **Possible improvements:**
  1. **Additional metrics:** Evaluation relies heavily on VBench. While VBench is comprehensive, including standard metrics like FID, IS, or user preference studies would further solidify the quality claims.
  2. **Generalization to other tasks:** The paper focuses on text-to-video generation. A brief experiment on image-to-video or video editing tasks would demonstrate broader applicability.
  3. **Combination with other accelerations:** The synergy with sparse attention (SVG) is mentioned (Fig. 6(c)) but not quantitatively explored in the main results.

### Writing & Clarity
- The paper is generally well-written, with clear explanations, figures, and tables. The method is described step-by-step, and the appendix provides necessary details (algorithm, proofs, additional experiments).
- Minor issues (likely due to PDF parsing artifacts) do not hinder understanding: e.g., "~~" in some text, missing y-axis labels in Fig. 3. These are not the authors' fault.

### Limitations & Broader Impact
- **Limitations:** Section 6 briefly mentions the focus on video generation and plans to extend to NLP. Other limitations (e.g., training cost for huge models, hyperparameter tuning) could be discussed more explicitly.
- **Broader Impact:** The paper lacks a broader impact statement. Quantization enables more efficient deployment, which can democratize video generation, but potential misuse of generative models (deepfakes, misinformation) should be acknowledged.

### Overall Assessment
This paper presents a significant and novel contribution: the first effective QAT framework for ultra-low-bit quantization of video diffusion models. The theoretical analysis is sound, the method (auxiliary modules + rank-decay) is innovative and well-motivated, and experiments are extensive and convincing. The work clearly advances the state of the art, pushing quantized video generation to 4-bit with minimal quality loss. While some areas could be strengthened (e.g., hyperparameter sensitivity, additional metrics), the core contribution is substantial and meets ICLR's high standards. I recommend acceptance, provided the authors address the following points in a revision.

**Key points for revision:**
1. **Clarify the "first" claim:** Ensure the results fully support the claim of "full-precision comparable" quality at 4-bit, especially for metrics like Scene Consistency where gaps remain.
2. **Discuss hyperparameter sensitivity more thoroughly:** How critical are the choices of r and λ across different models? Could guidelines be provided?
3. **Add a broader impact statement:** Acknowledge potential societal implications (positive and negative) of efficient video generation.
4. **Consider adding more evaluation metrics** (e.g., FID, user studies) to further validate quality claims.
5. **Briefly discuss the computational overhead of the SVD steps** in the rank-decay strategy, especially for large models.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces QVGen, a novel quantization-aware training (QAT) framework designed for video diffusion models (DMs) under extremely low-bit quantization (e.g., 4-bit or below). The method enhances convergence by adding auxiliary modules to reduce gradient norms and then progressively removes them via a rank-decay strategy to avoid inference overhead. Extensive experiments on multiple SOTA video DMs (CogVideoX, Wan families) demonstrate that QVGen achieves full-precision comparable quality at 4 bits and significantly outperforms existing methods at 3/4 bits.

### Strengths
1. **Novelty**: First work to explore QAT for video DMs, introducing auxiliary modules and a rank-decay strategy tailored to the unique challenges of video generation quantization.
2. **Theoretical Foundation**: Provides a theoretical analysis linking gradient norm reduction to improved convergence in QAT, motivating the design of auxiliary modules.
3. **Comprehensive Evaluation**: Extensive experiments across four SOTA video DMs (1.3B to 14B parameters) and comparisons with multiple PTQ/QAT baselines show consistent and substantial gains, especially in 3/4-bit settings. Includes both quantitative metrics (VBench) and qualitative visual results.
4. **Ablation Studies**: Thorough ablations validate each component (auxiliary modules, rank-decay) and explore design choices (shrinking ratio, initial rank, decay strategies), demonstrating robustness.
5. **Efficiency Analysis**: Reports practical benefits: model size reduction, inference speedup (1.21–1.44× on A800), and training costs, highlighting deployment potential.
6. **Clarity and Reproducibility**: Well-structured paper with clear figures, tables, and algorithm description. Code and models are promised, and appendix provides extensive implementation details.

### Weaknesses
1. **Limited Comparison with Video-Specific Quantization**: While general QAT/PTQ baselines are included, direct comparison with recent video-specific quantization works (e.g., QVD, Q-DiT) is missing from the main tables, making it harder to assess relative advancement in the video domain.
2. **In-Depth Analysis of Video-Specific Challenges**: The paper attributes quantization difficulty to higher gradient norms but could provide more analysis on why video DMs (e.g., temporal dynamics) are particularly sensitive compared to image DMs.
3. **Training Overhead**: The rank-decay strategy involves repeated SVD computations and multiple training phases, which increase training time and complexity. The trade-off between training cost and final performance could be discussed more critically.
4. **Generalization Beyond Video**: Though the method is motivated as general, experiments are limited to video generation. Preliminary image results in the appendix are promising but not comprehensive; extension to other tasks (e.g., language modeling) would strengthen the claim.
5. **Hyperparameter Sensitivity**: While ablations explore some design choices, the impact of key hyperparameters (e.g., decay schedule, rank initialization) on different model architectures is not fully analyzed, which may affect reproducibility.

### Novelty & Significance
The work is highly novel as the first effective QAT framework for video DMs under extremely low-bit quantization. It addresses a significant practical challenge in deploying large video generation models. The proposed techniques (auxiliary modules, rank-decay) are innovative and well-motivated. The results set a new state-of-the-art for 3/4-bit video generation, pushing the Pareto frontier of efficiency versus quality. This is timely and relevant to the growing demand for efficient video generation, meeting ICLR's expectations for impactful research.

### Suggestions for Improvement
1. Include comparisons with video-specific quantization methods (e.g., QVD, Q-DiT) in the main results to better contextualize the advancement.
2. Provide deeper analysis on why video DMs are harder to quantize (e.g., temporal modeling effects) and how QVGen specifically mitigates these issues.
3. Discuss the limitations of the rank-decay strategy more explicitly, such as computational overhead from SVD and sensitivity to hyperparameters like shrinking ratio and initial rank.
4. Extend experiments to more diverse video generation settings (e.g., longer sequences, higher resolutions) and other domains (e.g., image generation, NLP) to demonstrate broader applicability.
5. Ensure all critical hyperparameters (e.g., learning rate schedules, exact decay phases) are clearly documented in the main paper or appendix to facilitate reproduction.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **Fair comparison with EfficientDM under the same parameter update budget.** The paper notes that EfficientDM updates only LoRA parameters, greatly reducing optimizer memory. To claim superiority, QVGen must compare against a variant of EfficientDM that fine-tunes all parameters (or conversely, apply QVGen's strategy to only LoRA parameters) with matched training FLOPs. Without this, the claimed gains could be due to a much larger effective parameter update capacity.
2.  **Long video generation and temporal coherence evaluation.** The core challenge of video generation is maintaining consistency and motion quality over time. The paper evaluates on fixed short clips (49/81 frames). To demonstrate robustness, it must test on longer generation sequences (e.g., 120+ frames) and report metrics like Temporal FID or frame-wise consistency scores (e.g., via optical flow) to show the method does not degrade over longer durations.
3.  **Ablation on different quantization granularities.** The work uses *per-channel* weight and *per-token* activation quantization. Prior PTQ work (e.g., SVDQuant) uses *per-group* for better performance. To claim pushing the limit, the authors should ablate their method with finer-grained *per-group* quantization and show it still outperforms or matches these stronger baselines. Its absence leaves open whether the gains are from the QAT framework or the choice of a simpler, potentially suboptimal, quantization scheme.
4.  **2-bit or mixed-precision (W4A2, W2A4) results.** The paper claims to push the limit of "extremely low-bit quantization (i.e., 4-bit or below)" and mentions "initial exploration as setting a direction... for future work on 2-bit." For a paper making this claim, showing results on W2A2 or mixed-precision settings is critical to demonstrate the method's boundary and where it fails, which is key information for the community.

### Deeper Analysis Needed (top 3-5 only)
1.  **Theoretical justification under non-convex assumptions is superficial.** The core convergence theorem (Thm. 3.1) relies on convexity, which the authors acknowledge is unrealistic. The appendix provides a non-convex analysis, but it's a generic smoothness result that does not incorporate the *specific dynamics of QAT* (e.g., gradient estimation error from STE, the effect of Φ). A more convincing analysis would bound the gradient norm in terms of the quantization error mitigated by Φ, directly linking the proposed module to the theoretical improvement.
2.  **Direct causal link between reduced gradient norm and final performance is missing.** Figure 3 shows QVGen has lower gradient norm and loss, but this is correlation. An ablation is needed: if one artificially clips gradients in the baseline (Q-DM) to match QVGen's norm, does performance recover? If not, the benefit may come from Φ's architectural role, not merely gradient stabilization. This distinction is crucial for understanding the method's mechanism.
3.  **Analysis of what Φ actually learns and why rank-decay works.** The paper states Φ is initialized with quantization error **W - Q_b(W)**. An analysis is missing: does Φ stay close to this error, or does it learn something else? Showing the evolution of the *correlation* between Φ and the instantaneous quantization error during training would clarify if it's merely an error compensator or a more complex adaptor. This would strengthen the rationale for the rank-decay strategy.

### Visualizations & Case Studies
1.  **Visualization of failure modes for 3-bit models.** The paper shows good qualitative results. To establish credibility, it must also show representative *failure cases* for its own 3-bit models, especially on metrics where scores drop sharply (e.g., Scene Consistency). This would honestly delineate the current limits and guide future work.
2.  **Frame-by-frame error maps or temporal inconsistency heatmaps.** For video generation, *where* and *when* quantization hurts most is vital. Generating error maps (e.g., LPIPS/SSIM difference per frame vs. FP16) across a generated video would visually reveal if artifacts are sporadic, accumulate over time, or affect specific regions (e.g., moving objects vs. background).
3.  **t-SNE/Similarity matrix of frame features.** To analyze temporal consistency, visualize the latent features of frames from a generated video. A t-SNE plot or a frame-to-frame similarity matrix would concretely show whether the quantized model produces coherent feature trajectories compared to the FP16 model, or if features jump erratically.

### Obvious Next Steps
1.  **Quantization of attention operations.** The paper quantizes only linear layers, using FP16 attention (via FlashAttention). A major source of compute/memory in video DiTs is the 3D attention. The obvious next step, which should have been explored, is applying QVGen to quantize the Q/K/V/O projections *within* the attention mechanism itself, or exploring low-bit attention kernels, to claim a more comprehensive acceleration.
2.  **A more thorough combination with SVDQuant.** The paper briefly shows QVGen combined with SVDQuant initialization brings gains. This should be a major subsection, not an appendix note. A full analysis is needed: Does SVDQuant's low-rank representation interact synergistically with the rank-decay schedule? Can the two methods be fully integrated (e.g., using SVDQuant's decomposed matrices as the initial Φ)?
3.  **Training efficiency improvements should be part of the method.** The paper identifies training cost as a limitation (QVGen adds overhead vs. Q-DM) and suggests combining with EfficientDM's LoRA strategy for future work. For a complete contribution, a variant of QVGen that integrates parameter-efficient fine-tuning (e.g., only training Φ and quantization parameters, freezing **W**) should be developed and tested to address this critical deployment hurdle.

# Final Consolidated Review
## Summary
QVGen introduces a quantization-aware training framework for video diffusion models under extremely low-bit quantization (e.g., 4-bit or below). It proposes auxiliary modules to improve convergence by reducing gradient norms and a rank-decay strategy to eliminate these modules without inference overhead. Extensive experiments show that QVGen achieves full-precision comparable quality at 4 bits and significantly outperforms existing methods at 3/4 bits.

## Strengths
- **Novel and effective approach**: First quantization-aware training method tailored for video diffusion models, introducing auxiliary modules (Φ) to stabilize training and a rank-decay strategy to remove them without inference cost.
- **Theoretically grounded**: Provides a convergence analysis linking gradient norm reduction to improved QAT performance, with empirical validation showing lower ∥g_t∥₂ and training loss (Fig. 3).
- **Comprehensive validation**: Evaluated on four state-of-the-art video DMs (1.3B to 14B parameters) across 3-bit and 4-bit settings, demonstrating substantial gains over multiple PTQ/QAT baselines on VBench metrics (Tables 1, 2).
- **Ablation studies**: Thorough ablations confirm the contribution of each component (Table 3) and explore design choices like shrinking ratio λ and initial rank r (Tables 4, 5), ensuring robustness.
- **Practical efficiency**: Achieves model size reduction and inference speedups (1.21–1.44× on A800) using standard quantization kernels, with training costs comparable to baselines (Table 7).

## Weaknesses
- **Missing broader impact statement**: The paper does not discuss societal implications, such as the democratization of video generation or potential misuse, which is expected for responsible research.
- **Correlation vs. causation for gradient norm**: The link between reduced gradient norm and improved performance is correlational; an ablation (e.g., gradient clipping in baselines) is needed to establish a direct causal mechanism.
- **Limited analysis of hyperparameter sensitivity across models**: While ablations on λ and r are provided for one model, their impact across different architectures (e.g., from 1.3B to 14B) is not thoroughly explored, affecting generalizability.
- **Attention operations remain unquantized**: The method quantizes only linear layers, leaving attention computations in FP16; quantizing attention is a significant opportunity for further acceleration that is not addressed.
- **Short video sequences only**: Evaluation is limited to clips of 49 or 81 frames; testing on longer sequences would better assess temporal consistency, a core challenge in video generation.

## Nice-to-Haves
- Including additional evaluation metrics like FID or user preference studies to complement VBench scores.
- Extending experiments to image generation or other tasks (e.g., NLP) to demonstrate broader applicability beyond video.
- Visualizing failure cases for 3-bit models to honestly depict current limitations and guide future work.
- Deeper analysis of what Φ learns during training, such as its correlation with instantaneous quantization error.
- Combining with parameter-efficient fine-tuning (e.g., LoRA) to reduce training overhead while maintaining performance.

## Removed Points
These points are flagged to be removed, treat them with caution.
- **Criticism about singular value assumptions**: The paper cites prior work (Zhang et al., 2015; Yang et al., 2020) and provides empirical evidence in Fig. 4 to support that small singular values correspond to low-contributing components.
- **Demand for direct comparison with video-specific quantization methods (e.g., QVD, Q-DiT) in main tables**: While valid, the paper compares with adapted general methods and cites these works in related work; this is more of a scope consideration than a core flaw.
- **Request for fair comparison with EfficientDM under identical parameter update budget**: EfficientDM uses a different design (updating only LoRA parameters), and the paper reports training costs separately; demanding equal budgets conflates distinct approaches.
- **Suggestion that 2-bit results are essential**: The paper focuses on 3/4-bit as "extremely low-bit" and mentions 2-bit as future work; requiring it here overreaches the stated contribution.

## Novel Insights
The paper identifies that reducing gradient norm is key to stabilizing QAT for video diffusion models, a challenge exacerbated by temporal dynamics. It further observes that the singular values of the auxiliary modules decay during training, enabling a progressive rank-decay removal strategy that preserves performance while eliminating inference overhead. These insights are novel and advance the understanding of quantization in video generation.

## Suggestions
- Add a broader impact section to discuss societal implications, both positive (democratization) and negative (misuse potential).
- Conduct an ablation study artificially clipping gradients in baselines (e.g., Q-DM) to test whether matching QVGen's gradient norm recovers performance, clarifying the causal role.
- Explore quantization of attention operations (Q/K/V projections) to further accelerate inference, as attention is a major computational bottleneck.
- Evaluate on longer video sequences (e.g., 120+ frames) to ensure temporal coherence is maintained beyond short clips.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0, 8.0, 6.0]
Average score: 6.8
Binary outcome: Accept
