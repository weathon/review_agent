=== CALIBRATION EXAMPLE 71 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Abstract
The abstract clearly states the problem (AR decoding inefficiency), the proposed solution (Fast-dLLM v2, a block-diffusion LLM adapted from AR models), and key claims: 1) data-efficient adaptation (~1B tokens, 500× less than Dream), 2) a hierarchical caching mechanism, and 3) up to 2.5× speedup over AR decoding without quality loss. The claims are bold and, if substantiated, would be a significant contribution. However, the abstract does not mention any limitations or trade-offs, which should be noted in the main paper.

### Introduction & Motivation
The introduction effectively motivates the need for parallel decoding alternatives to AR models and identifies the limitations of existing diffusion LLMs (KV cache incompatibility, latency, fixed lengths). It clearly positions the work as scaling up block-diffusion (BD3-LMs) to large LLMs with a data-efficient fine-tuning recipe. The three contributions are reiterated. A minor point: the related work on block diffusion (BD3-LM, SDAR) is mentioned, but the precise novelty relative to these concurrent works could be delineated more sharply. For instance, SDAR also fine-tunes AR models; what specifically makes Fast-dLLM v2's framework more "systematic" or "robust"?

### Method / Approach
The methodology is generally well-described, but several areas require clarification or raise concerns:

1.  **Novelty vs. Prior Work:** The method synthesizes ideas from BD3-LM (block diffusion), Fast-dLLM (DualCache, parallel decoding), and complementary masking. The core novel contributions appear to be the specific training recipe (block-aligned packing, complementary masking, token shift) and the *combination* of block-level caching with DualCache into a "hierarchical" system. The paper should more explicitly distinguish which components are adopted from prior work and which are novel contributions of "v2".
2.  **Training Objective and Masking (Section 3.2 & Appendix A.3):** The description of the training objective and the justification for omitting the 1/t normalization factor due to complementary masking is present but somewhat buried in the appendix. The logic (because two complementary masks ensure all tokens are covered, making the total loss per sample constant) should be integrated into the main methodology for clarity. The attention mask design (Appendix A.2) is complex and suffers from severe parsing/formatting issues in the submitted text, making it difficult to assess. The authors must ensure the final version has a clean, understandable diagram and explanation of `M_full`.
3.  **Reproducibility:** Key hyperparameters are provided (block size D=32, context length L=2048). However, details on the exact mask sampling distribution (how is `m` sampled? Is it a fixed mask ratio `t` per block?) are missing. The "flex-attention" implementation is mentioned but not described; this could be a barrier to reproduction.
4.  **Inference Pipeline (Section 3.3):** The description of parallel refinement within a block using a "confidence threshold" is brief. It references Fast-dLLM's strategy but does not detail how confidence is computed (e.g., probability of the top token) or how the threshold is chosen (0.9 is used later, but why?). The interplay between block-level caching and DualCache needs a clearer algorithmic description or pseudocode.

### Experiments & Results
The experimental evaluation is extensive, covering multiple model sizes and benchmarks. However, several critical issues must be addressed:

1.  **Baseline Fairness and Comparison Focus:** The primary speed comparison is against the original AR model (Qwen2.5). This is valid. However, to claim "state-of-the-art efficiency among dLLMs," more direct comparisons with other efficient dLLM inference methods (e.g., dKV-Cache, DPad, Sparse-dLLM) are needed, not just Dream and LLaDA. Figure 1 shows a throughput advantage over Fast-dLLM-Dream, but what about other recent accelerators? The claim of being "nearly 10× faster than Dream-7B" (Section 2.3) is dramatic but only appears in a parenthetical comment; it needs proper benchmarking.
2.  **Data Efficiency Claim:** The claim of 500× less data than Dream is highlighted. However, this comparison is potentially misleading. Dream (Ye et al. 2025a) trained a diffusion model *from scratch* or performed extensive adaptation. Fast-dLLM v2 *fine-tunes* a high-quality pretrained AR model. The massive data reduction is arguably due to starting from a strong pretrained foundation and using an AR-friendly block-wise attention structure, not solely the merit of the proposed training recipe. This should be acknowledged as a limitation or the claim should be reframed.
3.  **Quality Trade-offs:** Table 1 shows that while the average score improves, there are notable performance drops on specific benchmarks for the 7B model vs. the Qwen2.5-7B-Nemo-FT baseline (e.g., GPQA: 31.9 vs. 34.2; MATH: 61.6 vs. 72.0; MMLU: 66.6 vs. 68.6). The paper states it "matches or exceeds" AR baselines, but these regressions should be discussed. Is there a task-dependent sensitivity?
4.  **Ablation Study:** The ablation (Table 2) is good but limited. It shows the gain from complementary masking and padding. However, an ablation on the hierarchical caching mechanism itself is missing. What is the individual contribution of block-level cache vs. DualCache (sub-block cache) to the overall speedup? Figure 6b shows caching helps at batch size 32, but what about the latency-critical batch size 1 scenario?
5.  **Speedup Conditions:** The 2.5-2.6× speedup (Figures 4, 5) is achieved with a confidence threshold of 0.9, which incurs a slight accuracy drop on GSM8K. This is a reasonable trade-off, but it must be clearly presented as such. What is the speedup under the more conservative threshold=1.0 (no parallel decoding) which matches training? Figure 5 shows diffusion is faster than AR even at threshold=1.0 (implied by batch size scaling), but the magnitude of the speedup is unclear.
6.  **Evaluation of "Block Diffusion":** A key motivation is flexible sequence length. The experiments should include a test generating sequences longer than the training block size to demonstrate this flexibility. The current evaluations seem to use fixed output lengths aligned with benchmarks.

### Writing & Clarity
Despite parser artifacts, the core technical content is understandable. However, some sections are dense (Appendix A.2 on attention masks is particularly garbled and needs complete revision). The figures and tables are essential but some captions are incomplete (e.g., Figure 1's axes). The paper would benefit from a clearer high-level algorithm box for both training and inference.

### Limitations & Broader Impact
The paper briefly mentions compatibility with speculative decoding as future work. **Major limitations are not sufficiently discussed:**
1.  The method assumes a pretrained AR model as a starting point. The performance and efficiency of training a block-diffusion model from scratch are unexplored.
2.  The impact of block size hyperparameter (`D`) on quality and speed needs more analysis. Table 4 shows performance degrades if inference block size mismatches training, but what is the optimal `D`? How does it scale with model size or task?
3.  Potential failure modes: What happens when intra-block dependencies are very strong? Does the parallel refinement within a block sometimes lead to coherence issues or repetitive text compared to fully sequential AR?
4.  Broader impact is not discussed. The work aims to make LLM inference faster and cheaper, which has widespread societal implications (positive: accessibility; negative: lower cost could amplify misuse). A brief statement is needed.

### Overall Assessment
This paper presents a well-engineered system, Fast-dLLM v2, that effectively combines several recent ideas (block diffusion, complementary masking, hierarchical caching) to achieve a practical goal: speeding up LLM inference via parallel decoding while preserving quality. The demonstrated data-efficient fine-tuning (from AR models) and the empirical speedups are compelling and relevant for ICLR. However, the presentation sometimes overclaims novelty, and the experimental analysis has gaps, particularly in direct comparisons with other efficient dLLM methods and in probing the limitations and trade-offs of the approach. The contribution is significant but would be strengthened by a more rigorous and honest assessment of its positioning, a clearer ablation of its components, and a discussion of its constraints. With major revisions to address these concerns, the paper has the potential to be a strong contribution.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents Fast-dLLM v2, a method for efficiently adapting pretrained autoregressive (AR) large language models into block-diffusion language models (dLLMs) to enable parallel text generation. The core contributions are a data-efficient fine-tuning recipe requiring only ~1B tokens (a 500× reduction compared to prior dLLMs like Dream), a complementary masking strategy with block-wise attention to preserve AR capabilities, and a hierarchical caching mechanism for accelerated inference. The method achieves up to 2.5× speedup over standard AR decoding while maintaining competitive accuracy across diverse benchmarks.

### Strengths
1. **Data Efficiency**: The paper demonstrates successful adaptation of pretrained AR models (Qwen2.5-1.5B/7B) into block-diffusion models using only ~1B tokens of fine-tuning, a substantial reduction from the 580B tokens required by Dream. This is evidenced in Section 1 and supported by training details in Section 4.1 and Appendix A.1.
2. **Comprehensive Evaluation**: The authors conduct extensive experiments across multiple benchmarks (GSM8K, MATH, MMLU, HumanEval, etc.) for both 1.5B and 7B models, showing that Fast-dLLM v2 matches or surpasses AR baselines in accuracy (Table 1) while improving throughput (Figure 1, 5).
3. **Ablation Studies and Design Analysis**: The paper includes thorough ablations (Tables 2-4, Figure 6) that validate key components like complementary masking, padding, sub-block decoding, and caching, providing clear evidence for design choices and their impact on performance and efficiency.

### Weaknesses
1. **Limited Model Scale and Task Scope**: Experiments are limited to 7B models and primarily English benchmarks. The paper does not explore larger models (e.g., 70B) or more diverse tasks (e.g., long-context generation, multilingual evaluation), leaving open questions about scalability and general applicability (Section 4.1).
2. **Incremental Novelty**: While the integration of block diffusion, complementary masking, and hierarchical caching is well-executed, many core ideas (block diffusion, DualCache) build directly on prior work (BD3-LM, Fast-dLLM). The paper could better articulate the distinct advances over these predecessors (Sections 2.2, 2.3).
3. **Incomplete Comparison with Concurrent Work**: The related work mentions several concurrent methods (e.g., SDAR, D2F, Set Block Decoding) but does not include direct empirical comparisons with them, making it difficult to assess the relative advantage of Fast-dLLM v2 (Section 2.2).

### Novelty & Significance
The primary novelty lies in the highly data-efficient adaptation strategy and the integration of a hierarchical caching mechanism with block-wise parallel decoding. The significance is substantial: if the results hold at scale, this approach could make diffusion-based parallel generation practically deployable by drastically reducing fine-tuning costs and achieving speedups without quality loss. The work aligns well with ICLR's emphasis on efficient and scalable machine learning methods.

### Suggestions for Improvement
1. **Scale Experiments to Larger Models**: To strengthen claims of practicality and scalability, include adaptation results for at least one model in the 30B+ parameter range, even if preliminary, and discuss any challenges encountered.
2. **Direct Comparison with Concurrent Methods**: Add comparisons with recent concurrent works like SDAR and Set Block Decoding on common benchmarks (e.g., GSM8K, HumanEval) to clearly demonstrate the advantages of Fast-dLLM v2 in data efficiency and inference speed.
3. **Deeper Analysis of Data Efficiency**: Provide more analysis or intuition for why the proposed adaptation requires so few tokens compared to prior dLLMs. An ablation studying the contribution of the complementary masking and block-wise attention to data efficiency would be insightful.
4. **Clarify Limitations and Failure Cases**: Discuss scenarios where the method may underperform (e.g., very long sequences, specific task types) and any trade-offs between block size, speed, and quality not covered in the ablations.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison with contemporaneous block diffusion methods (e.g., SDAR, D2F, Set Block Decoding) on throughput and quality.** The claim of "state-of-the-art efficiency among dLLMs" is unsubstantiated without head-to-head benchmarks against these directly comparable works cited in the related work.
2. **Scalability experiments beyond 7B parameters (e.g., 13B, 70B).** The paper's contribution is about adapting "large LLMs," but only shows results up to 7B. Without evidence on larger models, the claim of practical scalability and effectiveness for "state-of-the-art LLMs" is not convincing.
3. **End-to-end latency measurements (time to first token, time per output token) in addition to throughput.** For real-world deployment, latency is critical. High batch-size throughput gains (Figure 5) do not prove low-latency advantages, which is a key promise of parallel decoding.
4. **Evaluation of generation quality/diversity under different sampling methods (e.g., temperature sampling, top-k) with parallel decoding.** All benchmarks use greedy decoding. The performance and potential degradation of the proposed confidence-based parallel decoding under standard sampling techniques is unknown and crucial for real use.

### Deeper Analysis Needed (top 3-5 only)
1. **Ablation study or analysis justifying the core claim of 500x data efficiency.** The paper attributes efficiency to the "AR-friendly" block-wise design but provides no experimental analysis (e.g., comparing training loss curves, attention pattern analysis, or ablations with full attention) to isolate why ~1B tokens suffices versus Dream's 500B. Without this, the adaptation mechanism is not well understood.
2. **Analysis of the trade-off between speed (threshold <1.0) and generation quality across diverse tasks.** Figure 4 shows the trade-off only for GSM8K. The selected threshold of 0.9 may harm performance on other tasks (e.g., creative writing, open-ended QA). A multi-task analysis is needed to trust the proposed configuration.
3. **Breakdown of where the speedup comes from: block caching vs. intra-block parallel decoding vs. hierarchical caching.** The 2.5x speedup is claimed from the combined system. An ablation separating the contribution of each component (block cache, sub-block cache, parallel decoding) is necessary to validate the design and identify bottlenecks.

### Visualizations & Case Studies
1. **Visualization of the parallel decoding process for a challenging example, showing iterations of masking/unmasking and confidence scores.** This would reveal whether the method works as intended (refining uncertain tokens) or fails (e.g., getting stuck in loops, making confident but wrong predictions early).
2. **Side-by-side comparison of generated sequences (and their token-level probabilities) from AR baseline and Fast-dLLM v2 for the same prompt.** This would expose subtle quality differences, coherence issues, or hallucinations that aggregate benchmarks might miss, especially when using parallel decoding with threshold <1.0.
3. **Failure case analysis: Provide examples where the model's accuracy or coherence drops significantly compared to the AR baseline, especially in longer, multi-step reasoning.** The case studies only show successes. Showing and diagnosing failures is critical for understanding the method's limitations.

### Obvious Next Steps
1. **Include latency (e.g., time per output token at batch size 1) as a core metric alongside throughput.** For a paper focused on "efficient inference" and "practical deployment," omitting latency in favor of batch throughput is a major oversight for ICLR.
2. **Benchmark against speculative decoding techniques (e.g., using Fast-dLLM v2 as a draft model) as suggested in the text.** The paper mentions this as future work but given the claimed 10x speedup over Dream, a comparison with speculative decoding (a dominant acceleration method) is essential to contextualize the practical contribution.
3. **Evaluate the impact of varying block size during inference on performance and speed for a fixed training block size.** The ablation (Table 4) only tests mismatched sizes *without* sub-block decoding. A systematic study of optimal train/test block sizes is needed for practitioners.

# Final Consolidated Review
## Summary
This paper introduces Fast-dLLM v2, a method for adapting pretrained autoregressive large language models into block-diffusion models to enable parallel text generation. The key claims are data-efficient fine-tuning requiring only ~1B tokens, a hierarchical caching mechanism for inference acceleration, and up to 2.5× speedup over standard AR decoding while maintaining competitive accuracy across diverse benchmarks.

## Strengths
- **Data-efficient adaptation from pretrained AR models**: The method fine-tunes models like Qwen2.5-7B with only ~1B tokens, a substantial reduction compared to prior diffusion LLMs such as Dream, while preserving performance on standard benchmarks (Sections 1, 4.1, Table 1).
- **Comprehensive empirical validation**: Experiments on 1.5B and 7B models across multiple benchmarks (GSM8K, MATH, MMLU, HumanEval, etc.) demonstrate that Fast-dLLM v2 matches or surpasses AR baselines in aggregate accuracy and achieves significant throughput improvements (Table 1, Figures 1, 5).
- **Ablation studies supporting design choices**: Ablations confirm the importance of complementary masking, padding, and sub-block decoding for performance and efficiency, providing clear evidence for the proposed training recipe (Tables 2-4, Figure 6).

## Weaknesses
- **Incomplete benchmarking against contemporary efficient dLLM methods**: The paper claims "state-of-the-art efficiency among dLLMs" but only compares with Dream and LLaDA, omitting direct empirical comparisons with other recent accelerators like dKV-Cache, DPad, or Sparse-dLLM. This undermines the SOTA efficiency claim and leaves the relative advantage unclear.
- **Performance regressions on specific tasks without analysis**: For the 7B model, accuracy drops on GPQA (31.9 vs. 34.2), MATH (61.6 vs. 72.0), and MMLU (66.6 vs. 68.6) compared to the AR baseline (Table 1). These degradations are not discussed, leaving open questions about task-dependent sensitivity.
- **Lack of latency analysis for practical deployment**: Throughput metrics are provided, but critical latency measures such as time to first token or per-token latency at batch size 1 are absent. For a paper focused on inference efficiency, this omission limits the assessment of real-world utility.
- **Limited evidence of scalability beyond 7B parameters**: Experiments are confined to models up to 7B, yet the paper claims applicability to "large LLMs" and "practical deployment." Without results on larger-scale models (e.g., 30B+), the scalability of the method remains unverified.

## Nice-to-Haves
- Ablation study separating the contributions of block-level caching and sub-block caching to the overall speedup.
- Evaluation of generation quality under different sampling strategies (e.g., temperature sampling) to assess the robustness of the confidence-based parallel decoding.
- Testing the flexible sequence length capability by generating sequences longer than the training block size.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add direct comparisons with contemporaneous block-diffusion methods (e.g., SDAR, Set Block Decoding) on common benchmarks to substantiate efficiency claims.
- Include latency metrics (e.g., time per output token at batch size 1) in addition to throughput for a more complete efficiency analysis.
- Discuss the performance drops on specific tasks (e.g., GPQA, MATH) and explore potential reasons or mitigations.

# Actual Human Scores
Individual reviewer scores: [8.0, 4.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept
