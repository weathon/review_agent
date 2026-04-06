=== CALIBRATION EXAMPLE 50 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is clear and reflects the core contribution. The abstract succinctly states the problem (KV cache overhead in reasoning LLMs), the failure of existing methods, the proposed solution (RL-guided head identification), and the key result (20-50% cache reduction with near-lossless performance). All claims are supported in the main text.

### Introduction & Motivation
The motivation is strong and well-presented. The paper clearly establishes that existing KV cache compression methods (token-dropping, head-reallocation) degrade significantly on reasoning models due to extended CoT generation. The controlled comparison between reasoning and instruct variants (Figure 1b) effectively isolates CoT length as the cause. The analysis of error modes (repetitive vs. over-extended generation) provides a clear rationale for the work. The hypothesis of functional heterogeneity among heads is compelling and sets the stage for the method.

### Method / Approach
The methodology is innovative and described in sufficient detail for reproducibility. The core idea—using RL to optimize sparse gating adapters that mix full and local attention—is novel for head identification.
- **Assumptions:** The operational definition of "reasoning heads" (heads that degrade under local cache) is reasonable but implies a binary, static partitioning. The paper does not explore whether head importance could be input- or sequence-dependent.
- **Logical Gaps:** The stabilization techniques (adaptive penalty weighting, self-distillation sampling) are necessary and well-motivated by the sparse reward vs. dense penalty conflict (Figure 4). However, the choice of the reward threshold `τ` and the exponential scaling in Eq. 4 seem heuristic; a brief sensitivity analysis or justification would strengthen this section.
- **Reproducibility:** The method relies on integrating MixedAttention into the AReaL and SGLang frameworks. While the hyperparameters and training details are provided (Section 3.1, Appendix A.2), releasing code would be essential for full reproducibility. The training cost (40-36 GPU-hours) is non-trivial but arguably reasonable for the target application.
- **Proofs:** No formal proofs are required; the method is empirically driven.

### Experiments & Results
The experimental design is comprehensive and rigorous, aligned with ICLR standards.
- **Models & Datasets:** Evaluating on two mainstream reasoning models (Llama-3.1-8B-R1, Qwen-2.5-7B-R1) and four diverse benchmarks (GSM8K, Math500, AIME24, MBPP) provides strong evidence. The addition of Qwen-3-4B-Thinking and MMLU-Pro subsets in the appendix further validates generalization.
- **Baselines:** The authors thoughtfully adapt baselines (H2O, R-KV, DuoAttention) for a fair comparison by augmenting token overhead and converting fixed budgets to dynamic allocation (Appendix A.2, A.7-A.8). The discussion on fixed vs. dynamic budgets is a valuable methodological contribution.
- **Main Results:** Figure 5 and Tables 1, 3, 4 show RLKV consistently outperforms baselines, especially at high sparsity (0.4-0.6). The counter-intuitive result where RLKV sometimes surpasses the full KV cache baseline (e.g., on AIME24) is noted but could be analyzed deeper. Is this due to noise reduction, or does it indicate overfitting to the training distribution?
- **Ablations:** The ablation studies (Figure 8) effectively validate the importance of adaptive penalty weighting and self-distillation sampling. The analysis of the L1 penalty weight `β` is useful.
- **Analysis:** The analysis comparing reasoning heads vs. retrieval heads (Figure 6) is insightful and supports the core hypothesis. The error mode analysis (Figures 7, 11, 12) provides a nuanced understanding of failure patterns, showing that compressing reasoning heads leads to repetitive errors, distinct from retrieval head compression.
- **Missing Analyses:** 
    1. **Head Distribution Analysis:** Figure 3 shows the adapter distribution but lacks a discussion of whether reasoning heads cluster in specific layers or have recognizable patterns. A qualitative analysis of these heads (e.g., via attention visualization) could strengthen the interpretability claim.
    2. **Generality of Identified Heads:** The heads are identified using a math-only curriculum (DeepScaleR). While results on MBPP and MMLU-Pro show some generalization, an experiment retraining on a coding dataset would clarify if reasoning heads are task-specific or model-intrinsic.
- **Statistical Significance:** The paper uses Pass@1 but does not report standard deviations or confidence intervals across multiple runs. Given the stochastic nature of RL training, some measure of variance would be helpful.

### Writing & Clarity
The paper is well-written and logically structured. Figures are clear and support the narrative. Some minor notes:
- The parsing artifacts (e.g., stray "FIX" tags, broken table formatting in the text) are distracting but, as per instructions, are not considered weaknesses.
- Section 2.3, "Stabilization for RL Training," is critical but slightly dense. A more intuitive explanation of the "vicious cycle" before diving into the equations would improve flow.
- The distinction between "reasoning heads" and prior work on "retrieval heads" is made but could be emphasized earlier and more sharply.

### Limitations & Broader Impact
The conclusion and future work sections acknowledge key limitations: variability of head distributions across models/tasks, unexplored functional roles, and challenges at extreme compression (>80%). The societal impact is positive (efficient inference) and minimal negative impact is foreseen. Two limitations could be more explicitly stated:
1. **Training Overhead:** The RL training, while modest, adds a non-zero cost and complexity barrier compared to training-free methods like R-KV.
2. **Static Allocation:** The identified heads are fixed after training; an adaptive mechanism that allocates full cache dynamically based on the ongoing generation could be more efficient but is not explored.

## Overall Assessment
This is a strong, novel paper that makes a clear contribution to efficient inference for reasoning LLMs. The core idea—using RL with sparsity pressure to identify reasoning-critical heads—is well-motivated by the failure modes of existing compression methods. The experimental evaluation is thorough, covering multiple models, benchmarks, and careful baseline comparisons. The analyses of error modes and head importance provide valuable insights beyond mere performance metrics. The main concerns are the computational cost of RL training and the need for further analysis on the generality and interpretability of the identified heads. However, these do not undermine the paper's primary contribution. For ICLR, this paper meets the bar for acceptance, provided the authors can address the requested clarifications and analyses in a revision.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces RLKV, a reinforcement learning (RL) method to identify "reasoning-critical" attention heads in Large Language Models (LLMs) for efficient Key-Value (KV) cache compression. The core insight is that reasoning models (e.g., CoT generators) rely on a specific, sparse subset of heads to maintain reasoning consistency; compressing others has minimal impact. RLKV trains lightweight gating adapters via RL with a sparsity penalty, directly optimizing for reasoning performance. The method achieves 20-50% KV cache reduction with near-lossless accuracy on mathematical and coding benchmarks.

### Strengths
1. **Novel and Well-Motivated Insight**: The paper provides a clear, empirically supported motivation: existing KV cache compression methods (token-dropping, head-reallocation) fail for reasoning models because they disrupt critical reasoning states. The hypothesis of "reasoning heads" is compelling and grounded in controlled experiments (Fig. 1, Sec. 1).
2. **Innovative Methodology**: Using RL to directly optimize the relationship between head-level cache allocation and end-task reasoning quality is a novel and fitting approach. The integration of gating adapters, L1 sparsity, and stabilization techniques (adaptive penalty weighting, self-distillation sampling) is technically sound and well-explained (Sec. 2).
3. **Extensive and Rigorous Evaluation**: Experiments cover two reasoning models (Llama-3.1-8B-R1, Qwen-2.5-7B-R1), four diverse benchmarks (GSM8K, MATH, AIME24, MBPP), and multiple compression rates. RLKV consistently outperforms strong baselines (H2O, R-KV, DuoAttention), especially at higher sparsity (Fig. 5, Tables 1,3,4). Additional analyses (error modes, head importance, ablation studies) deepen the understanding (Sec. 3).
4. **Practical Significance**: The work addresses a pressing deployment bottleneck for reasoning LLMs. The reported 20-50% memory reduction with minimal performance loss is practically valuable. The appendix includes detailed implementation notes, latency measurements, and discussion on fair evaluation budgets, aiding reproducibility (App. A.2, A.8, A.9).

### Weaknesses
1. **Limited Analysis of "Reasoning Heads"**: While the method identifies heads, it provides limited analysis of what makes them "reasoning-critical." A deeper interpretability study (e.g., analyzing attention patterns, layer/head distribution, or functional roles) would strengthen the core claim and provide more general insights.
2. **Generalization Beyond Mathematical Reasoning**: Training and primary evaluation are on mathematical reasoning. Results on MMLU-Pro subsets (App. A.6) show performance drops in some domains (e.g., Law, Physics), suggesting the identified heads may be task-specific. Broader evaluation on diverse reasoning types (e.g., commonsense, planning) is needed.
3. **Computational Overhead of RL Training**: Although the paper notes training takes "several hours" on 2 A100s (Sec. A.2), a clearer breakdown of the RL sample complexity, wall-clock time, and comparison to the inference savings would help assess the overall efficiency trade-off. The need for a separate RL training phase per model is a non-trivial cost.
4. **Comparison to Baselines on Non-Standard Grounds**: The paper uses a "dynamic budget" for fair per-sample comparison, which is justified (App. A.7). However, this differs from the fixed-budget evaluation in some baseline papers (e.g., R-KV). While the fixed-vs-dynamic analysis is provided (App. A.8), the performance advantage of RLKV is less pronounced under the original fixed-budget scheme for R-KV at lower sparsity.

### Novelty & Significance
**Novelty**: The paper introduces a novel concept ("reasoning heads") and a novel RL-based method to identify them for KV cache compression. This differs fundamentally from prior head-reallocation methods that target "retrieval heads" using static proxies. Using RL to directly optimize for reasoning quality during multi-step generation is a key innovation.
**Significance**: The work is highly significant for the efficient deployment of reasoning LLMs. It offers a new, effective solution to a major memory bottleneck, enabling larger batch sizes or deployment on memory-constrained hardware. The findings also contribute to the understanding of attention head specialization in LLMs.

### Suggestions for Improvement
1. **Conduct a Deeper Analysis of Identified Heads**: Perform a post-hoc analysis of the top "reasoning heads" (e.g., their layers, attention patterns on reasoning traces, similarity across tasks/models) to provide mechanistic insights into why they are critical and if they align with known head functionalities.
2. **Expand Evaluation to Non-Mathematical Reasoning**: Test RLKV on a wider array of reasoning benchmarks (e.g., LogiQA, StrategyQA, Big-Bench Hard) to better establish the generality of the "reasoning head" concept and the robustness of the compression method.
3. **Clarify the Computational Trade-off**: Provide a more detailed cost-benefit analysis comparing the RL training overhead (GPU-hours, number of samples) to the inference-time memory and latency savings across different deployment scales. Discuss the feasibility of amortizing this cost.
4. **Strengthen the Baseline Comparison**: Include a main experiment table or figure that directly compares RLKV with baselines under their originally reported *fixed-budget* settings (alongside the dynamic-budget results). This would provide a more complete picture and address potential concerns about evaluation fairness.
5. **Improve Writing Clarity**: Some sections, particularly the stabilization technique (Sec. 2.3) and the RL objective (Eq. 2), could be explained more clearly. A step-by-step algorithmic pseudo-code in the appendix would enhance reproducibility.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare RLKV’s head selection to simple gradient-based or output deviation baselines.** The paper lacks an ablation showing that RL is necessary. A baseline that identifies heads by measuring output deviation when compressing each head individually (on training problems) would test if the expensive RL framework is justified.
2. **Evaluate on true long-context reasoning tasks, not just long generations.** The benchmarks involve long CoT outputs, but the context (prompt) is short. The claim that head-reallocation methods fail on reasoning models is based on retrieval head identification; testing on tasks requiring long-context reasoning (e.g., multi-document QA) would strengthen the claim.
3. **Include a baseline that combines token-dropping and head-reallocation.** The paper argues the two strategies are distinct, but a hybrid method (e.g., apply token-dropping only to non-reasoning heads) could outperform both and should be compared to.
4. **Test on a wider range of model families and scales (e.g., 70B models).** Experiments are limited to three relatively small models (≤8B). To claim generality for reasoning LLMs, results on larger models (e.g., Llama-3.1-70B-R1) are needed, as head importance may scale differently.

### Deeper Analysis Needed (top 3-5 only)
1. **Analyze the actual function of identified “reasoning heads” beyond operational definition.** The paper defines reasoning heads as those that degrade performance when compressed, but provides no analysis of their attention patterns, layer distribution, or relationship to known mechanistic features (e.g., induction heads, mathematical reasoning circuits). Without this, the claim of discovering a new head type is weak.
2. **Explain why RLKV sometimes outperforms the full model.** The paper notes RLKV surpasses full KV cache on AIME24 but offers only a speculative one-sentence explanation. A rigorous analysis (e.g., attention entropy, noise reduction) is needed to validate whether this is a consistent beneficial effect or a statistical fluctuation.
3. **Provide a sensitivity analysis of the RL training stability.** The stabilization techniques are crucial, but the paper shows only one training curve. Reporting variance across multiple runs or different random seeds is necessary to trust that RL training reliably converges to a good solution.
4. **Quantify the relationship between head importance (α value) and performance degradation.** Figure 6 shows aggregate degradation when replacing top heads, but does not validate that individual head α values correlate with their actual importance. A per-head analysis would confirm RLKV’s identification accuracy.

### Visualizations & Case Studies
1. **Visualize attention patterns of a few reasoning vs. non-reasoning heads during CoT generation.** Showing attention maps for a solved problem would concretely demonstrate that reasoning heads attend to critical prior reasoning steps, while non-reasoning heads have local or noisy patterns.
2. **Show side-by-side CoT generations for the same problem under full cache, RLKV, and baselines at high sparsity.** The paper describes error modes categorically but does not show concrete examples. Displaying actual text would make the failure modes (repetition, over-extension) tangible and validate that RLKV preserves coherence.
3. **Plot the layer-wise distribution of identified reasoning heads.** A simple histogram (layer vs. count of reasoning heads) would reveal if reasoning heads cluster in specific layers (e.g., later layers), offering architectural insights.

### Obvious Next Steps
1. **Integrate RLKV with quantization methods.** The paper focuses on spatial compression (cache size), but combining with KV cache quantization (e.g., 4-bit) is a direct next step for further memory reduction. Results showing additive gains should be included.
2. **Test adaptive compression based on problem difficulty or generation length.** The current method uses a fixed sparsity per head. A dynamic scheme that adjusts the number of reasoning heads based on predicted complexity (e.g., from prompt) could improve the sparsity-performance trade-off.
3. **Evaluate the transferability of identified heads across tasks without retraining.** The paper trains on math data and tests on coding and knowledge QA. An analysis of whether heads identified on math are sufficient for other domains, or if task-specific tuning is needed, is a natural and important extension.
4. **Measure end-to-end throughput/latency with an optimized kernel.** The latency analysis (Appendix A.9) uses a naive PyTorch implementation. Implementing a fused kernel for the head-reallocation attention and reporting speedups in a realistic serving setting (with continuous batching) is critical for claiming practical efficiency.

# Final Consolidated Review
## Summary
This paper proposes RLKV, a reinforcement learning method to identify reasoning-critical attention heads in large language models (LLMs) for efficient KV cache compression. The method trains sparse gating adapters that mix full and local attention, allowing a subset of heads to retain full cache while others use compressed cache. Experiments show 20-50% KV cache reduction with minimal performance loss on mathematical and coding reasoning benchmarks, outperforming existing token-dropping and head-reallocation baselines.

## Strengths
- **Well-motivated problem and clear failure analysis**: The paper demonstrates that existing KV cache compression methods (token-dropping, head-reallocation) degrade significantly on reasoning models due to long chain-of-thought generations, and analyzes distinct error modes (Figure 1).
- **Novel methodology**: Using RL to directly optimize head-level cache allocation for reasoning quality is innovative. The integration of gating adapters with sparsity regularization and stabilization techniques (adaptive penalty weighting, self-distillation sampling) is technically sound and well-explained (Section 2).
- **Extensive and rigorous evaluation**: Experiments cover two reasoning models (Llama-3.1-8B-R1, Qwen-2.5-7B-R1), four benchmarks (GSM8K, MATH, AIME24, MBPP), and multiple compression rates, showing consistent superiority over strong baselines (Figure 5, Tables 1,3,4). Additional analyses (error modes, head importance, ablation studies) provide deeper insights.

## Weaknesses
- **Limited interpretability of identified "reasoning heads"**: The paper operationally defines reasoning heads as those that degrade performance when compressed, but provides no analysis of their attention patterns, layer distribution, or functional roles. This weakens the claim of discovering a distinct head type and misses an opportunity for mechanistic insight.
- **Insufficient analysis of performance improvements over full cache**: The occasional outperformance of the full model (e.g., on AIME24) is noted but only given a speculative one-sentence explanation. A deeper analysis (e.g., attention entropy, noise reduction) is needed to validate whether this is a consistent beneficial effect or an artifact.
- **Generalization concerns beyond mathematical reasoning**: While tested on coding and knowledge QA, performance drops on some MMLU-Pro subsets (e.g., Law, Physics) suggest the heads identified on mathematical data may be task-specific. The method's robustness to diverse reasoning types is not fully established.
- **Computational overhead without clear trade-off analysis**: The RL training requires non-trivial resources (several hours on two A100 GPUs). A clearer cost-benefit analysis comparing this overhead to the inference-time memory and latency savings across deployment scales is missing, making the practical trade-off unclear.

## Nice-to-Haves
- Sensitivity analysis for the stabilization techniques (e.g., reward threshold τ and exponential scaling in Eq. 4).
- Visualizations of attention patterns for reasoning vs. non-reasoning heads and a layer-wise histogram of reasoning heads.
- Comparison to a simpler, non-RL baseline for head identification (e.g., based on gradient or output deviation) to further justify the use of RL.
- Reporting variance across multiple RL training runs to assess stability.

## Novel Insights
The paper introduces the concept of "reasoning heads" as a sparse set of attention heads critical for maintaining chain-of-thought consistency in reasoning LLMs, distinct from previously studied "retrieval heads." By using RL to directly optimize for reasoning performance under sparsity constraints, the method reveals that only a small fraction of heads require full KV cache access. The error mode analysis further shows that compressing reasoning heads leads to repetitive generation errors, while compressing retrieval heads results in more varied failures, highlighting a functional difference.

## Suggestions
- Conduct a post-hoc analysis of the top reasoning heads (e.g., their layers, attention patterns on reasoning traces) to provide interpretability and validate their role in reasoning.
- Perform a more thorough cost-benefit analysis of the RL training, breaking down the sample complexity, wall-clock time, and comparing it to the inference savings across different deployment scales.
- Extend evaluation to a wider array of reasoning types (e.g., commonsense, planning) to better establish the generality of the reasoning head concept.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0, 8.0]
Average score: 4.5
Binary outcome: Reject
