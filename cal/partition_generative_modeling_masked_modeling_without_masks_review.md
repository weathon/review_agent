=== CALIBRATION EXAMPLE 74 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly captures the core contribution: replacing masking with partitioning. The abstract succinctly states the problem (MGMs process full sequences including [MASK] tokens, ARMs process only clean tokens) and the solution (PGMs partition tokens into non-interacting groups, eliminating [MASK] tokens). Key claims—5–5.5× higher throughput on OpenWebText and 7.5× on ImageNet with comparable quality—are bold and clearly stated. One minor inconsistency: the abstract says “5–5.5×” but Figure 2 mentions “5.3×”. Otherwise, the abstract is well-structured and supported by the paper.

### Introduction & Motivation
The introduction effectively motivates the inefficiency of MGMs (processing [MASK] tokens at every step) and contrasts them with ARMs. It clearly positions prior work (distillation, block diffusion) as not making individual steps cheaper while preserving any-order generation. The contributions are listed explicitly and align with the paper’s content.

### Background (Section 2)
Thorough coverage of sequence modeling, MGMs (MDLM, MaskGIT), classifier-free guidance, and self-distillation through time. The equations are standard but assume familiarity with diffusion models. No major issues.

### Partition Generative Modeling (Section 3)
- **Core idea**: Partition tokens into two groups that cannot attend to each other; each group predicts the other. This eliminates [MASK] tokens and allows processing only clean tokens during inference.
- **Training objective (Eq. 7)**: Derived from the MDLM variational bound by treating each group as clean/masked interchangeably. The weighting function \(w^{\text{PGM}}\) (Eq. 8) uses \(w(t)\) and \(w(1-t)\). This is clever and theoretically grounded.
- **Sampling**: Described in text and algorithms (Appendix). PGM can use uniform (fixed tokens per step) or MDLM-equivalent schedules. The connection to MGM samplers is preserved.
- **Variance reduction**: Claimed because PGM computes loss at all positions (two complementary signals). However, the paper later shows (Sec. 5.3) that complementary masking alone (without the PGM architecture) yields a 1.95 perplexity reduction on LM1B, but the full PGM underperforms that. This suggests the architecture does not fully capture the benefit; this nuance should be clarified in the main text.
- **Potential gap**: The claim that PGMs “eliminate [MASK] tokens entirely” is true in terms of not using a special token, but the training still uses a form of conditional masking (groups). The innovation is architectural, not purely objective-based.

### The Partition Transformer (Section 4)
- **Architecture**: Encoder (group-wise self-attention), GroupSwap layer (cross-attention to exchange information), Decoder (group-wise cross-attention, no self-attention). Figure 3 is helpful.
- **GroupSwap**: Two query initialization methods (data-independent and data-dependent). The paper uses data-independent for simplicity. However, there is no ablation study on the importance of GroupSwap; what happens if it is removed or simplified? The architecture’s complexity may affect parameter efficiency.
- **Decoder lacks self-attention**: This is justified by the conditional independence assumption but might limit modeling capacity. The paper shows balanced encoder/decoder layers work best (Table 5), but more architectural ablations would strengthen the design choices.
- **Inference efficiency**: Because groups do not interact, only clean tokens are processed. This is the key to speedup, but the overhead of GroupSwap and the dual-group design during training is not free. The paper acknowledges training is slower due to current attention implementation.

### Experiments (Section 5)
- **Language Modeling (LM1B, OpenWebText)**:
  - **Validation perplexity**: On LM1B, PGM (6/6) achieves 26.80 vs. MDLM’s 27.67 (0.87 reduction), but the abstract/introduction claim of “1.95 reduction” refers to complementary masking alone (MDLM with complementary masking gets 25.72). This is misleading and should be clarified. On OWT, PGM requires more parameters (larger dim or more layers) to surpass MDLM (Table 1, Table 5).
  - **Throughput**: Substantial gains (5–5.5×) are demonstrated (Table 1, Table 6). Latency measurements are rigorous.
  - **Downstream tasks**: PGM slightly outperforms MDLM on most tasks before/after distillation (Table 2, Table 4). This shows no quality sacrifice.
  - **Distillation**: PGM is compatible with SDTT. After distillation, with nucleus sampling, PGM remains ~4.6× faster with comparable perplexity/entropy (Fig. 4, Table 7). However, nucleus sampling reduces the speed advantage.
- **Image Modeling (ImageNet)**:
  - Results are strong: PGM matches MaskGIT FID with 7.5× throughput (Halton sampler, 32 steps). With 64 steps, FID improves to 4.56 while remaining 3.9× faster (Table 9).
  - Missing details: How is partitioning applied to 2D images? Are tokens partitioned randomly? Does this affect spatial coherence? This should be explained.
- **Isolating Complementary Masking (Sec. 5.3)**:
  - A valuable ablation: training a standard transformer with double batch size and complementary masks shows gains on LM1B (1.95 perplexity reduction) but smaller on OWT. This indicates the training objective itself is beneficial, but the PGM architecture does not fully realize these gains (PGM underperforms complementary masking by ~1 point on LM1B). The paper acknowledges this and suggests architecture improvements.

### Limitations (Appendix A)
Well-acknowledged: PGM requires slightly more parameters on OWT to match MDLM; training is slower due to attention implementation; future work includes efficient kernels and multimodal extensions. The limitations are honest and reasonable.

### Writing & Clarity
Overall clear and well-structured. Figures are informative. Some minor issues:
- The abstract’s “5–5.5×” vs. Figure 2’s “5.3×” should be consistent.
- The claim of “1.95 reduction in validation perplexity” should be explicitly tied to complementary masking, not the full PGM, to avoid misinterpretation.
- More details on image partitioning (2D grid) would help reproducibility.
- Sampling algorithms are in the appendix; a high-level description in the main text would improve readability.

## Overall Assessment
This paper introduces a novel and promising approach to improve the inference efficiency of masked generative models. The core idea—partitioning tokens instead of masking—is clever and well-motivated. The proposed Partition Transformer enables processing only clean tokens during sampling, yielding substantial throughput gains (5–7.5×) while maintaining sample quality on text and image benchmarks. The paper is thorough, with strong experiments across domains, ablations, and compatibility demonstrations with existing samplers and distillation.

However, there are notable concerns:
1. The claimed “1.95 perplexity reduction” is misleading; it applies to complementary masking, not the full PGM model, which underperforms complementary masking by about 1 point on LM1B.
2. On OpenWebText, PGM requires more parameters to surpass MDLM, indicating potential parameter inefficiency.
3. The architecture (GroupSwap) adds complexity, and training is currently slower.
4. Details on image partitioning are sparse.

Despite these issues, the contribution is significant: a new architecture that dramatically speeds up inference while retaining the flexibility of MGMs. The paper is well-written and the experiments are convincing. For ICLR, the paper likely meets the acceptance bar, provided the authors clarify the perplexity claim and address the minor concerns in a revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Partition Generative Models (PGMs), a novel class of generative models that replace the masking mechanism in masked generative models (MGMs) with a partitioning scheme. The core idea is to split tokens into two non-interacting groups and train the model to predict each group conditioned on the other, eliminating the need for [MASK] tokens entirely. This allows PGMs to process only clean tokens during sampling (like autoregressive models) while retaining the parallel, any-order generation capability of MGMs, leading to substantial inference speedups (5–7.5×) on language and image generation benchmarks with competitive sample quality.

### Strengths
1. **Significant Inference Speedup with Minimal Quality Loss**: The paper provides strong empirical evidence that PGMs achieve 5–5.5× higher sampling throughput than MDLM on OpenWebText and 7.5× higher throughput than MaskGIT on ImageNet, while maintaining comparable or better sample quality (Generative Perplexity, FID). The speedup is clearly attributed to the elimination of [MASK] token processing during sampling (Figures 1, 2, Tables 1, 6, 8, 9).
2. **Elegant Architectural Innovation and Theoretical Connection**: The proposed Partition Transformer, with its group-wise attention and GroupSwap layer, is a clever architectural solution that enforces the required separation between groups while enabling conditioning. The paper convincingly shows the connection to the MDLM variational bound and the variance reduction from complementary masking (Section 3.1, 5.3).
3. **Comprehensive Evaluation and Compatibility**: The work includes extensive experiments on both language (LM1B, OpenWebText) and image (ImageNet) domains, evaluating likelihood, sample quality, downstream task performance, and distillation compatibility. The demonstration that PGMs work with existing MGM samplers (Halton) and distillation methods (SDTT) is a practical strength (Sections 5.1, 5.2, Figures 1, 4, Table 2).

### Weaknesses
1. **Increased Model Complexity and Parameter Count for Parity**: On OpenWebText (context length 1024), the PGM requires more parameters (increased layers or embedding dimension) to match the validation perplexity of the MDLM baseline. This suggests the current architecture may be less parameter-efficient for longer sequences, and the GroupSwap layer introduces overhead (Table 1, Table 5, Section 5.1).
2. **Incomplete Ablation on Training Stability and Precision Sensitivity**: The paper notes training loss spikes with complementary masking and different precision requirements (FP32 vs BF16) for PGMs versus MDLMs (Appendix D.4, Figure 6). This raises concerns about the robustness and ease of training PGMs at scale, which is not thoroughly investigated.
3. **Limited Analysis of the "Any-Order" Generation Benefit**: While PGMs retain the any-order generation capability of MGMs, the paper does not provide experiments or analysis demonstrating the practical advantages of this property (e.g., for fill-in-the-middle, controlled generation). The focus is primarily on throughput, leaving the flexibility claim somewhat under-explored.

### Novelty & Significance
**Novelty**: The core idea of replacing masking with a hard partition to avoid processing [MASK] tokens at inference is novel. The Partition Transformer architecture, specifically the GroupSwap layer, is a new and non-obvious design to enforce group separation while enabling cross-group conditioning.
**Clarity**: The paper is generally well-written. The conceptual shift from masking to partitioning is clearly explained, and the architecture is described with helpful diagrams (Figures 2, 3). Some parts of the supplementary algorithms (e.g., Algo 3-5) are complex and could be better integrated or explained in the main text.
**Reproducibility**: The paper provides substantial experimental detail (architectures, hyperparameters, datasets) in the main text and appendix. The code is promised to be released under an MIT license. The reliance on external codebases (MDLM, HaltonMaskGIT) is acknowledged. Reproducibility is generally high, though the noted training instability might require careful tuning.
**Significance**: The work addresses a critical limitation of MGMs—their inference inefficiency—with a principled method that offers order-of-magnitude speedups. This could make parallel-token generative models much more practical for real-time applications. The results are compelling enough to influence future research in non-autoregressive generation.

### Suggestions for Improvement
1. **Provide a Deeper Analysis of Parameter Efficiency**: Investigate and discuss why PGMs require more parameters to match MDLM on longer sequences. Could the GroupSwap mechanism be simplified? An analysis of the computational cost (FLOPs, memory) during training and inference would strengthen the comparison.
2. **Expand on the Benefits of Any-Order Generation**: Include a targeted experiment (e.g., infilling, selective regeneration) that showcases the advantage of generating tokens in any order, contrasting PGMs with block-based methods like Block Diffusion that sacrifice this property.
3. **Mitigate Training Instability and Clarify Best Practices**: Propose and evaluate solutions for the training loss spikes (e.g., gradient clipping adjustments, schedule modifications). Clearly recommend precision settings (FP32/BF16) for training PGMs in different scenarios to aid future researchers.
4. **Streamline the Supplementary Material**: The sampling algorithms in Appendix E are complex and somewhat disconnected from the main narrative. Consider integrating a clearer, high-level description of the sampling process into Section 3.2, moving the detailed pseudocode to an appendix while improving its readability.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare against strong autoregressive models (ARMs).** The paper claims PGMs combine ARM and MGM strengths, but only compares to MGMs. To validate this claim, include a comparison with a similarly-sized GPT-2 in terms of validation perplexity, downstream task accuracy, and sampling throughput (with and without KV caching). Without this, it's unclear if PGMs are a practical alternative to the dominant paradigm.
2. **Ablation study on the Partition Transformer architecture.** The gains could stem from the complementary masking objective or the novel architecture (group-wise attention, GroupSwap). An ablation is needed that trains a standard bidirectional Transformer with the same complementary masking objective but without the architectural constraints (e.g., allowing cross-group attention). This would isolate the contribution of the architecture itself.
3. **Demonstrate any-order generation capability.** A key claimed advantage over ARMs is the ability to generate tokens in any order. The paper should show concrete experiments (e.g., infilling, generating from the middle, or varying the decoding order) and compare the quality of such generations with MGMs. Without this, the "any-order" claim is not substantiated.
4. **Evaluate on longer context lengths and larger scales.** Experiments are limited to context lengths of 1024 for text and a single image resolution. To show the method's robustness and scalability, include results at longer contexts (e.g., 4096) and with larger model sizes (e.g., more parameters). The appendix mentions a latency table for context 4096, but no quality evaluation is provided.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze the trade-off between speed and sample diversity/quality.** The paper reports higher throughput and similar FID/Perplexity, but does not deeply analyze diversity. For text, compute metrics like self-BLEU or pairwise token diversity; for images, report precision/recall or diversity scores. This is crucial to ensure speed gains do not come from reduced mode coverage.
2. **Investigate why PGMs require more parameters to match MDLM on OpenWebText.** On OWT, PGM needed increased width or depth to surpass MDLM. The brief analysis in the appendix is insufficient. A systematic study (varying context length, dataset, tokenizer) is needed to understand the limitations and when PGM is parameter-efficient.
3. **Analyze the cause and impact of training loss spikes.** Figure 6 shows complementary masking introduces loss spikes. The paper should investigate whether these affect convergence, final performance, or indicate instability. This is important for reproducibility and scaling.

### Visualizations & Case Studies
1. **Visualize attention patterns in the Partition Transformer.** Show heatmaps of attention weights in the encoder and GroupSwap layer to confirm that groups are isolated and information is swapped as intended. This would validate the core architectural mechanism.
2. **Show qualitative examples of generated text and images.** Provide side-by-side samples from PGM and baselines (MDLM, MaskGIT) to allow visual assessment of quality and artifacts. For text, show excerpts highlighting coherence and fluency; for images, show a grid of samples.
3. **Visualize the sampling process over time.** For images, show intermediate reconstructions at different steps, highlighting which tokens/patches are being decoded. This would illustrate the "any-order" generation in practice and how the partition evolves.

### Obvious Next Steps
1. **Implement and benchmark efficient block-diagonal attention kernels.** The paper notes training is slower due to dense attention masks. Implementing optimized kernels for the block-diagonal pattern (after grouping tokens) is an obvious engineering step that would make training more efficient and strengthen the practical contribution.
2. **Develop a distillation method tailored for PGMs.** The paper uses SDTT designed for MGMs by treating one group as masked. Designing a distillation method that leverages the partition structure (e.g., distilling both groups simultaneously) could yield further speed or quality gains and is a natural extension.
3. **Explore integration with KV caching for further speedup.** The paper mentions KV caching is possible but does not implement it. Integrating KV caching (since clean tokens are fixed during sampling) would be a straightforward way to boost throughput further and should have been explored.

# Final Consolidated Review
## Summary
This paper introduces Partition Generative Models (PGMs), a novel class of generative models that replace the masking mechanism in masked generative models (MGMs) with a partitioning scheme. Tokens are split into two non-interacting groups, and the model learns to predict each group conditioned on the other, eliminating [MASK] tokens entirely. This enables PGMs to process only clean tokens during sampling (like autoregressive models) while retaining the parallel, any-order generation of MGMs, achieving 5–7.5× higher inference throughput on text and image benchmarks with comparable sample quality.

## Strengths
- **Substantial inference speedup with minimal quality loss.** On OpenWebText, PGMs achieve 5–5.5× higher sampling throughput than MDLM while producing samples with lower generative perplexity. On ImageNet, PGMs reach comparable FID to MaskGIT with a 7.5× throughput improvement, and with more steps achieve better FID (4.56) while remaining 3.9× faster (Figures 1, 2; Tables 1, 6, 9).
- **Elegant architectural innovation with theoretical grounding.** The proposed Partition Transformer, with its group-wise attention and GroupSwap layer, is a clever design that enforces separation while enabling conditioning. The training objective is derived from the MDLM variational bound, and the method of complementary masking provides a variance reduction, yielding a 1.95 perplexity improvement on LM1B in an ablation (Section 3.1, Section 5.3, Table 1).
- **Practical compatibility and comprehensive evaluation.** PGMs are shown to be compatible with existing MGM samplers (e.g., Halton) and distillation methods (SDTT). The paper includes extensive experiments on language (perplexity, downstream tasks, distillation) and image generation (FID, IS), demonstrating the approach works across domains without sacrificing flexibility (Sections 5.1, 5.2; Tables 2, 4, 7).

## Weaknesses
- **Parameter inefficiency on longer sequences.** To match the validation perplexity of the MDLM baseline on OpenWebText (context length 1024), PGMs require either more layers or a larger embedding dimension (Table 1, Table 5). This indicates the current architecture, particularly the GroupSwap layer, may not be as parameter-efficient for longer contexts, though it still provides large throughput gains.
- **Insufficient demonstration of the "any-order" generation benefit.** While PGMs retain the any-order generation capability of MGMs, the paper does not provide experiments or analysis showcasing this flexibility (e.g., for infilling or controlled generation). The focus is primarily on throughput, leaving a core claimed advantage underexplored.
- **Training instability and precision sensitivity.** The paper notes that complementary masking introduces occasional loss spikes (Figure 6) and that PGMs require different numerical precision (FP32 vs. BF16) for optimal loss computation compared to MDLMs (Appendix D.4). This raises concerns about robustness and ease of training at scale, though all runs converged.

## Nice-to-Haves
- A deeper analysis of sample diversity (e.g., self-BLEU for text, precision/recall for images) to ensure speed gains do not come at the cost of mode coverage.
- Implementation of optimized kernels that exploit the block-diagonal attention pattern to improve training efficiency, as noted in the limitations (Appendix A).
- A targeted experiment illustrating the practical advantage of any-order generation, such as infilling or selective regeneration.

## Novel Insights
The paper provides a clear insight: by replacing masking with a hard partition, the model can avoid processing uninformative [MASK] tokens at inference, which is the primary source of MGM slowness. The complementary masking objective itself provides a variance reduction benefit, as shown in the ablation. The architectural innovation—the GroupSwap layer that exchanges information between isolated groups—is a non-obvious solution to enable conditioning without cross-group attention. The work reveals a trade-off: this architecture currently introduces some parameter overhead on longer sequences, but the inference speed gains are substantial and the framework remains fully compatible with the MGM ecosystem.

## Suggestions
- Investigate architectural simplifications or more parameter-efficient designs for the GroupSwap mechanism to close the gap with complementary masking on longer sequences.
- Include a concrete experiment (e.g., text infilling or image inpainting) that demonstrates and quantifies the advantage of any-order generation compared to sequential or block-based methods.
- Provide clear best practices for mitigating training instability (e.g., gradient clipping adjustments) and recommended precision settings to aid reproducibility.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 6.0]
Average score: 7.0
Binary outcome: Accept
