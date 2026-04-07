=== CALIBRATION EXAMPLE 71 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly indicates the focus on an efficient block-diffusion LLM, building on prior work (Fast-dLLM). The abstract succinctly states the problem (sequential AR decoding), proposed solution (data-efficient block-diffusion adaptation with hierarchical caching), and claims (500× less training data than Dream, up to 2.5× speedup without quality loss). The claims are bold and, if fully supported, would represent a significant advance. However, the abstract lacks specifics: which AR baselines are used for speed comparison? Under what hardware and batch size conditions? "Lossless adaptation" is asserted but not quantified; the experiments later show competitive but not identical accuracy.

### Introduction & Motivation
The introduction effectively frames the limitations of AR decoding and the promise of diffusion models, while noting the practical drawbacks of existing dLLMs (bidirectional attention, lack of KV cache, fixed length). It positions block diffusion (e.g., BD3-LM) as a middle ground but notes its limited scale. The motivation for Fast-dLLM v2—scalable, data-efficient adaptation of pretrained AR models—is clear. The three contributions are well-articulated: (1) data-efficient post-training adaptation (~1B tokens), (2) hierarchical caching + parallel decoding for inference acceleration, (3) large-scale experiments demonstrating speedup and preserved quality.

### Method / Approach
The methodology builds sensibly on prior work (BD3-LM, Fast-dLLM) and introduces key innovations: complementary masking, token shift for prediction, and a hierarchical caching mechanism. The training process (block-wise organization, complementary views, shifted-label prediction) is described, though some details are relegated to the appendix (e.g., attention mask design, justification for omitting 1/t factor in the loss). The inference pipeline combines block-level autoregressive decoding with intra-block parallel refinement using confidence thresholds and DualCache.

**Concerns and questions:**
- **Reproducibility:** The description of the attention mask (Appendix A.2) is complex; a clearer schematic or pseudocode in the main text would help.
- **Confidence definition:** The parallel decoding uses a "confidence threshold"; how is confidence computed? (Presumably from softmax probability, but this should be specified.)
- **Sub-block cache:** While referenced as DualCache from Fast-dLLM, a brief explanation of how it operates in this hierarchical setting would improve clarity.
- **Training objective:** The loss in Section 3.2 lacks the usual 1/t normalization; the justification (complementary masking ensures all tokens are covered) appears in Appendix A.3. This key point should be highlighted in the main text to avoid confusion.

Overall, the method appears sound and novel, but the presentation could be streamlined for better understanding.

### Experiments & Results
The experiments are extensive, evaluating 1.5B and 7B models on a diverse suite of benchmarks (code, math, knowledge, instruction following). Table 1 shows that Fast-dLLM v2 performs competitively with AR baselines (Qwen2.5-Nemo-FT) and outperforms other dLLMs (Dream, LLaDA) on average scores. Speedup results (Figure 1a, Figure 5) demonstrate up to 2.5× higher throughput than AR decoding on A100/H100 GPUs across batch sizes. Ablation studies (Tables 2-4, Figures 4,6) validate design choices: complementary masking improves performance, sub-block size 8 is optimal, and mismatched block sizes degrade accuracy (highlighting the need for sub-block decoding).

**Major concerns:**
- **Clarity of speedup comparisons:** The 2.5× speedup claim needs precise delineation. In Figure 4, the 2.6× speedup is relative to Fast-dLLM v2's own non-parallel decoding (threshold=1.0). In Figure 1a, the 2.54× speedup is relative to Qwen2.5-7B-Instruct (AR). However, the exact configurations (threshold, caching, batch size) are not consistently reported. A clear table comparing throughputs under fixed conditions (e.g., batch size=1, seq length) would strengthen the claim.
- **Limited text generation evaluation:** The benchmarks focus on accuracy (multiple-choice, code correctness, math). There is no evaluation of generated text quality (e.g., perplexity, fluency, coherence) on open-ended tasks (summarization, dialogue). The case studies in Appendix B are anecdotal; systematic human or automatic metrics are needed to fully assess "generation quality."
- **Ablation scale:** Ablations are conducted only on the 1.5B model; it is assumed trends hold for 7B, but direct verification would be more convincing.
- **Throughput reporting:** The garbled figures (due to parser issues) make it difficult to extract exact numbers. The authors must ensure clear presentation in the final version.

Despite these issues, the experimental evidence is substantial and generally supports the claims.

### Writing & Clarity
The paper is well-structured and logically organized. However, due to parser artifacts, many figures and tables are poorly rendered (e.g., Figure 1, Table 1), obscuring key data. The method description is technically dense and sometimes relies on appendix references; integrating critical details (e.g., loss normalization, attention mask intuition) into the main text would improve readability. The writing is otherwise clear and professional.

### Limitations & Broader Impact
The paper lacks an explicit limitations section. Important limitations to acknowledge include:
- The need to choose block/sub-block sizes, which may be task- or hardware-dependent.
- Evaluation limited to models up to 7B parameters; scalability to larger models (e.g., 70B) is untested.
- The trade-off between speed and accuracy with confidence thresholds may vary across tasks.
- No thorough evaluation of text generation fluency and coherence beyond accuracy-based benchmarks.

Broader impact is briefly noted in Appendix D (LLMs used only for language polishing). The work aims to reduce computational costs, which is positive, but potential misuse of efficient LLMs is a generic concern.

## Overall Assessment

This paper presents a compelling and well-executed approach to adapting pretrained AR LLMs into efficient block-diffusion models. The core contributions—data-efficient fine-tuning (~1B tokens) and a hierarchical caching mechanism enabling 2.5× speedup without significant quality loss—are substantiated by extensive experiments on models up to 7B. The methodology is novel and builds thoughtfully on prior work. However, the paper would be strengthened by clearer reporting of speedup conditions, systematic evaluation of text generation quality, and a dedicated limitations discussion. For ICLR, which values technical innovation and rigor, this paper represents a solid contribution that could be accepted with minor revisions addressing these concerns.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Fast-dLLM v2, a method to efficiently adapt pretrained autoregressive (AR) large language models into block diffusion language models (dLLMs) for parallel text generation. The core contributions are a data-efficient fine-tuning recipe requiring only ~1B tokens (a 500× reduction compared to prior dLLMs like Dream), a hierarchical caching mechanism for inference acceleration, and a parallel decoding pipeline. The method achieves up to 2.5× speedup over standard AR decoding while maintaining competitive performance across diverse benchmarks.

### Strengths
1. **High Data Efficiency**: The paper clearly demonstrates a major reduction in fine-tuning data requirement (~1B tokens vs. 500B for Dream). This is a significant practical advantage for adapting existing LLMs, supported by experiments on 1.5B and 7B Qwen models (Section 3.2, Table 1).
2. **Well-Designed Inference Optimization**: The proposed hierarchical caching (block-level and sub-block DualCache) and parallel decoding pipeline are technically sound and directly address the KV cache incompatibility problem in bidirectional dLLMs. The ablation studies (Figure 6, Tables 3-4) effectively show the impact of sub-block caching and size on throughput/accuracy.
3. **Extensive Empirical Validation**: Experiments cover multiple model scales (1.5B, 7B), diverse benchmarks (code, math, knowledge), and thorough analysis of efficiency-accuracy trade-offs (Figure 4,5). The results show the method matches or surpasses AR baseline accuracy while providing speedups.

### Weaknesses
1. **Limited Theoretical Justification**: The paper lacks a theoretical analysis explaining *why* the block-wise attention design enables such dramatic data efficiency (500× less data). The claims about "AR-friendly" design remain intuitive rather than analytically grounded.
2. **Incomplete Baseline Comparisons for Efficiency Claims**: While throughput is compared to AR baselines and some dLLMs (Dream, LLaDA), the claim of "state-of-the-art efficiency among dLLMs" (Abstract) is not fully substantiated against all contemporary acceleration methods for dLLMs (e.g., DPad, EB-Sampler, WINO from Section 2.3). A more comprehensive latency/throughput comparison on equal hardware is needed.
3. **Evaluation Depth and Failure Analysis**: The benchmark evaluation, while broad, is mostly standard accuracy reporting. There is no analysis of generation quality beyond accuracy scores (e.g., coherence, fluency, long-form text) or failure modes, which is important for assessing real-world viability. The case studies (Appendix B) are anecdotal.

### Novelty & Significance
**Novelty**: The work is an incremental but valuable advancement. The core block diffusion concept builds on BD3-LM and SDAR. The primary novelty lies in the specific integration of complementary masking, token shift, and hierarchical caching into a scalable, data-efficient adaptation recipe. The 500× data reduction is the most distinctive empirical result.
**Significance**: The practical significance is high. Reducing dLLM training cost and achieving speedups over AR decoding addresses critical deployment barriers. If the results hold at larger scales, this could influence how efficient LLM inference systems are built. The work meets ICLR's emphasis on technically sound, empirically rigorous research with clear potential impact.

### Suggestions for Improvement
1. **Add Theoretical Analysis or Motivating Experiments**: Include a controlled ablation or analysis (e.g., probing representations) to explain the data efficiency gain. Why does block-wise causal attention enable near-lossless adaptation with so few tokens compared to full-attention dLLMs?
2. **Strengthen Efficiency Comparisons**: Conduct a unified efficiency comparison on fixed hardware including more recent dLLM acceleration baselines (e.g., DPad, EB-Sampler) and speculative decoding methods. Report not just throughput but also latency breakdowns for different sequence lengths.
3. **Deepen Evaluation and Discuss Limitations**: Evaluate on longer-form generation tasks (e.g., summarization, story generation) to assess quality beyond multiple-choice/score-based benchmarks. Honestly discuss limitations: e.g., sensitivity to block size, potential quality degradation in highly creative tasks, and any trade-offs observed but not highlighted.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison to concurrent block-diffusion methods (e.g., SDAR, BD3-LM) on speed and data efficiency.** The paper only compares to Dream and AR baselines. Without head-to-head comparison with methods sharing the same core idea (AR-to-block-diffusion adaptation), the claim of state-of-the-art efficiency among dLLMs is not fully substantiated.
2. **Ablation of the hierarchical caching mechanism's components.** The paper claims significant speedup from block-level and sub-block caches but does not isolate their individual contributions (e.g., throughput/ latency with only block-cache vs. only sub-block cache vs. both). This omission makes it impossible to verify which component is critical for the reported gains.
3. **End-to-end latency (time-to-first-token, total generation time) benchmarks, especially for long sequences.** Throughput (tokens/sec) is reported, but for real-world deployment, latency under different block sizes and sequence lengths is equally important. The absence of latency plots for varied generation lengths (not just fixed context scaling) weakens the practicality claim.
4. **Evaluation on tasks requiring long-range coherence and discourse (e.g., summarization, multi-turn chat).** Benchmarks like GSM8K, MMLU, and code generation do not test the model’s ability to maintain narrative consistency over long blocks, which is a key concern for diffusion-based generation. Missing this fails to validate the "without compromising generation quality" claim for real text.

### Deeper Analysis Needed (top 3-5 only)
1. **Error analysis: When and why does Fast-dLLM v2 underperform the AR baseline?** Table 1 shows performance drops on some tasks (e.g., GPQA, MATH for 7B). A qualitative analysis of failure cases is needed to understand if errors stem from the block-diffusion formulation, masking strategy, or parallel decoding.
2. **Sensitivity analysis of the parallel decoding threshold (0.9) across different task types.** Figure 4 shows a threshold sweep only on GSM8K. The chosen threshold may not be optimal for other tasks (e.g., code generation where confidence may differ). Without this analysis, the 2.6× speedup claim is not generalizable.
3. **Analysis of the trade-off between block size, sub-block size, and generation quality for varying sequence lengths.** The ablation studies fix context length. A systematic study of how optimal block/sub-block sizes shift with target output length would reveal limitations in flexible generation.
4. **Quantification of "data efficiency": A learning curve showing performance vs. fine-tuning tokens (from 0 to 1B).** The claim of "lossless adaptation with ~1B tokens" requires showing that performance plateaus at this point and that using more data does not help. A missing learning curve makes the 500× reduction claim anecdotal.

### Visualizations & Case Studies
1. **Visualization of the parallel decoding process across timesteps for a single example, highlighting when and which tokens are unmasked.** This would concretely show how the confidence-based refinement works, and reveal failure modes (e.g., early commitment to incorrect tokens, cascading errors within a block).
2. **Side-by-side comparison of generated text from AR baseline, Fast-dLLM v2, and a full dLLM (e.g., Dream) on prompts requiring complex reasoning or narrative flow.** The provided case studies are successful examples. Contrastive examples where diffusion-based decoding leads to repetitions, contradictions, or incoherence would better illustrate the method's boundaries.
3. **Attention map visualizations for the block-wise causal mask during inference.** This would verify the claimed "bidirectional within block, causal across blocks" behavior and show if any unintended cross-block attention leaks occur, which could explain performance drops.

### Obvious Next Steps
1. **Compare inference memory footprint and FLOPs against AR and full dLLM baselines.** For deployment, efficiency is not just speed but also memory. The paper only reports throughput; showing memory usage would strengthen the efficiency claims.
2. **Integrate and evaluate with speculative decoding frameworks (like DiffuSpec).** The paper mentions this as future work, but given the claimed 10× speedup over Dream, a preliminary experiment using Fast-dLLM v2 as a draft model for a larger AR model would dramatically elevate its impact.
3. **Validate the method on a larger-scale model (e.g., 13B or 70B parameters).** The paper only tests up to 7B. ICLR expects scaling laws or evidence that the method holds for larger models, which is critical for the claim of being "built to scale to large LLMs."
4. **Provide a pseudo-code or algorithm box for the core inference pipeline.** The description in Section 3.3 is textual; a clear, step-by-step algorithm would improve reproducibility and clarify the interaction between block decoding, caching, and parallel refinement.

# Final Consolidated Review
## Summary
Fast-dLLM v2 introduces a data-efficient method to adapt pretrained autoregressive LLMs into block-diffusion language models (dLLMs) for parallel text generation. It requires only ~1B tokens of fine-tuning (a 500× reduction compared to prior dLLMs like Dream) and achieves up to 2.5× speedup over standard AR decoding while maintaining competitive accuracy across diverse benchmarks. The method combines a block-diffusion training recipe with a hierarchical caching mechanism and parallel decoding pipeline.

## Strengths
- **Remarkable Data Efficiency**: The paper demonstrates a dramatic 500× reduction in fine-tuning tokens (from ~500B for Dream to ~1B) required to adapt a pretrained AR model into a dLLM, while preserving performance. This is a significant practical advance for making dLLMs viable, evidenced by competitive results on 1.5B and 7B Qwen models (Table 1).
- **Effective and Well-Validated Inference Optimization**: The proposed hierarchical caching (block-level and sub-block DualCache) and confidence-based parallel decoding directly address the KV cache incompatibility of bidirectional dLLMs. Ablation studies (Tables 2-4, Figures 4,6) rigorously validate the design choices and show clear throughput gains (up to 2.5-2.6×) without major accuracy loss.
- **Extensive and Scalable Empirical Evaluation**: The work provides comprehensive experiments on models up to 7B parameters across a diverse suite of tasks (code, math, knowledge). The results show the method matches or exceeds AR baseline accuracy (Table 1) while delivering consistent speedups across batch sizes and hardware (Figure 1, Figure 5), substantiating its scalability claim.

## Weaknesses
- **Lack of Theoretical or Mechanistic Explanation for Data Efficiency**: The paper claims the block-wise attention design is "AR-friendly" and enables near-lossless adaptation with vastly fewer tokens, but offers only empirical results. A theoretical analysis or probing experiments to explain *why* this architecture enables such extreme data efficiency would strengthen the contribution and provide deeper insight.
- **Incomplete Efficiency Benchmarking Against Contemporary dLLM Accelerators**: While throughput is compared to AR baselines and some dLLMs (Dream, LLaDA), the claim of "state-of-the-art efficiency among dLLMs" is not fully substantiated against other recent dLLM acceleration methods (e.g., DPad, EB-Sampler, WINO mentioned in Section 2.3). A more comprehensive comparison on fixed hardware would solidify this claim.
- **Evaluation Primarily Focused on Accuracy, Lacking Text Quality Assessment**: The benchmarks (GSM8K, MMLU, code generation) measure task accuracy but do not evaluate core text generation qualities like fluency, coherence, or long-form consistency. The claim of maintaining "generation quality" is therefore only partially validated, as performance drops on some tasks (e.g., GPQA, MATH) and no open-ended text evaluation is provided.

## Nice-to-Haves
- Direct head-to-head comparison with concurrent block-diffusion adaptation methods (e.g., SDAR, BD3-LM) on both speed and data efficiency metrics.
- Ablation study isolating the individual contribution of the block-level cache versus the sub-block (DualCache) to the overall speedup.
- Analysis of the optimal confidence threshold for parallel decoding across different task types (beyond GSM8K) to generalize the speedup claim.
- Preliminary exploration of using Fast-dLLM v2 as a draft model in a speculative decoding framework (as suggested in Section 2.3) to demonstrate further efficiency gains.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength (Generic)**: "The paper is well-structured and logically organized." (This is a generic comment that applies to many papers.)
- **Weakness (Misreading)**: "The abstract's claim of 'lossless adaptation' is not quantified." (The experiments in Table 1 show competitive performance, which substantiates the claim.)
- **Weakness (Formatting Nitpick)**: "The garbled figures due to parser issues obscure key data." (This is an artifact of the review extraction process, not a flaw in the paper itself.)
- **Weakness (Scope Creep / Non-Standard)**: "The method description is complex and should include pseudo-code in the main text." (While helpful, detailed algorithmic description is often in appendices; the textual description in Sections 3.2-3.3 is sufficient for reproducibility.)
- **Weakness (Overly Demanding)**: "Ablations are only on the 1.5B model; verification on 7B is needed." (The trends are clearly demonstrated, and scaling the ablations is not required for the core claim.)
- **Weakness (Unsubstantiated by Paper)**: "There is no analysis of failure modes when the model underperforms AR." (The paper does note performance drops in Table 1 but does not analyze them; however, this is a deeper analysis point, not a fundamental flaw invalidating the results.)

## Novel Insights
The primary novel insight is the demonstration that a block-diffusion architecture with a block-wise causal attention mask is remarkably compatible with pretrained AR models, enabling efficient adaptation with orders of magnitude less data than full-attention dLLMs. This compatibility, combined with a hierarchical caching strategy, allows diffusion-based decoding to actually surpass the throughput of its AR progenitor—a significant step toward making dLLMs a practical alternative for low-latency deployment. The work shows that the trade-off between AR and diffusion LLMs is not fixed and can be favorably shifted through careful architectural design.

## Suggestions
- Add a dedicated limitations section discussing the sensitivity to block/sub-block size hyperparameters, the lack of open-ended text quality evaluation, and the untested scalability beyond 7B parameters.
- Strengthen the efficiency claim by including a comparison table of throughput/latency against a wider set of dLLM acceleration baselines (e.g., DPad, EB-Sampler) under controlled, fixed hardware settings.
- Consider a simple analysis or motivating experiment (e.g., probing the similarity of representations before/after adaptation) to provide intuition for the dramatic data efficiency gain.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
