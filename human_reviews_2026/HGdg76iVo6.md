# Which Heads Matter for Reasoning? RL-Guided KV Cache Compression

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Reasoning large language models exhibit complex reasoning behaviors through the extended chain-of-thought generation, creating unprecedented Key-Value (KV) cache overhead during the decoding phase.
Existing KV cache compression methods underperform on reasoning models: token-dropping methods break reasoning integrity by discarding critical information, while head-reallocation methods mistakenly compress reasoning-critical heads since they are designed for retrieval tasks, resulting in significant performance degradation as compression rates increase.
We hypothesize that KV heads exhibit functional heterogeneity in reasoning models-some heads are critical for chain-of-thought consistency while others are compressible.
To validate and exploit this insight, we propose RLKV, a novel reasoning-critical head identification method, which uses reinforcement learning to directly optimize the relationship between each head's cache usage and reasoning quality.
As RLKV produces rewards from actual generated samples during training, it naturally identifies heads relevant to reasoning behaviors.
We then allocate full KV cache to these heads while applying compressed constant KV cache to others for efficient inference.
Our experiments reveal that only a small fraction of attention heads is essential for reasoning, enabling our KV compression approach to outperform baseline methods while achieving 20-50% cache reduction with near lossless performance compared to uncompressed results.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses a head-level KV cache compression method specialized for reasoning tasks. The authors observe that previous approaches, such as the reasoning-specialized token-level KV dropping method (R-KV) and the factuality-focused head-level compression method (DuoAttention), exhibit significant performance degradation on reasoning tasks. To address this issue, the authors adopt DuoAttention’s optimization framework and apply their method to reasoning task datasets with reward signals. Since a naïve application of the method leads to unstable optimization, the authors propose several training stabilization techniques, which play a critical role in achieving strong compression performance. The effectiveness of the proposed head-level KV compression method is demonstrated on three math and one coding-based reasoning benchmark using the LLaMA3.1-8B-R1 and Qwen2.5-7B-R1 models. Benchmark results show that the proposed method consistently outperforms DuoAttention and R-KV, achieving a 20–50% reduction in KV cache memory for the evaluated benchmarks and models.

### Strengths
- The paper addresses a timely and relevant problem that is likely to attract the interest of the ICLR community.
- To resolve the training instability and degeneration issues, the authors introduce several training stabilization techniques based on their observations.
- The experimental results compare the proposed method with recent approaches and show consistent, albeit modest performance improvements.

### Weaknesses
- Misleading Figure 1 and main motivation
  - The figure refers to “Token-dropping” and “Head-reallocation”, but neither the figure nor the paper provides a fundamental explanation of these approaches. Instead, the paper focuses only on a specific implementation, including R-KV and DuoAttention. I recommend revising the figure and text to more accurately reflect the statement, as the current analysis refers not to general methodologies but to a specific implementation.

- Lack of conceptual justification for “reasoning heads”
  - The paper extends the concept of “retrieval heads” to propose the existence of “reasoning heads”, but this conceptual leap is not sufficiently justified. It is unclear (1) what exactly defines a “reasoning head”, and (2) the paper lacks fine-grained analysis to support its existence. 
  - (1) Can we truly claim that specific heads perform a distinct reasoning function? For instance, in Figure 6, the so-called “reasoning heads” may be playing more general roles, such as supporting standard decoding or controlling the continuation of responses, rather than performing reasoning-specific functions. The paper does not sufficiently rule out these alternative explanations. 
  - (2) Compared with previous studies such as “Retrieval Head Mechanistically Explains Long-Context Factuality” (2024), this paper does not provide multi-level (example-level, token-level) analyses for attention mechanism interpretation or comprehensive evaluations across diverse models and datasets. Without such detailed examination, the claim of “reasoning heads” remains unconvincing.

- Limited methodological or analytical novelty
  - Conceptually, the paper feels like a straightforward extension of DuoAttention to reasoning models, as it largely adopts DuoAttention’s architecture and optimization framework. The contribution does not substantially expand my understanding of attention mechanisms or yield surprising results.

### Questions
- In Figure 5, the results on LLaMA3.1–8B–R1 (GSM8K, math) and Qwen2.5–7B–R1 (MBPP, code) show very similar performance trends between DuoAttention and the proposed method. Why do these two methods behave similarly? This could indicate that the observed effects are not tied to reasoning-specific heads, but rather to more general/complex mechanisms or dataset-specific factors.

- Line 83: What is the meaning of “useless steps”? How are “useless” steps defined and identified in the experiments?

- In real-world applications, reasoning and factual retrieval often occur in a mixed manner. How would the proposed method handle such mixed scenarios? What level of lossless compression could be achieved in those cases?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies KV-cache compression for reasoning LLMs and argues that heads differ functionally in this regime: a small subset of “reasoning heads” must retain full KV cache to preserve chain-of-thought (CoT) integrity, while other heads can use a compressed cache. It proposes RLKV, which inserts per-head gating adapters that mix full attention with streaming/local attention; the adapters are trained by group-relative PPO/GRPO on verifiable reasoning tasks, with an L1 sparsity penalty to push most heads toward compressed access.

### Strengths
1.The problem formulation is right and fresh. It clearly shows why long CoTs break existing compression: token dropping induces repetition/loops; retrieval-head reallocation preserves some integrity but derails reasoning steps.

2. The performance of the proposed method is impressive. It shows consistent gains over H2O, R-KV, DuoAttention across tasks and sparsities; even beats full KV on AIME24 at certain budgets, suggesting noise from non-reasoning heads.

3. The memory reduction is clear. The paper reports 20–50% KV savings at near-lossless quality

### Weaknesses
1. The head identity stability is not discussed. The importance of head may depend on prompt style/length and reasoning mode; the paper shows global head rankings, but does not analyze input-conditional head selection or per-layer diversity across tasks.

2. The end-to-end acceleration is not reported. We get memory reductions, but not end-to-end speedups under realistic batching/servers with paged KV, quantized KV, etc. The head-wise design may not improve the end-to-end latency, since the latency can be blocked by other heads in the same layer.

### Questions
Corresponding to the weaknesses, the following questions are not answered in the paper:

1. If you train gates on math, how much zero-shot transfer do you get to coding or to multi-hop QA? Conversely, do gates trained on MBPP hurt math?

2. To accelerate the inference speed, is it possible to compress the KV in a layer-wise? or the current head-wise design already accelerated the inference speed greatly?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work addresses the critical challenge of KV cache compression for LLMs specialized for reasoning. The authors posit that existing compression techniques (both token-dropping and head-reallocation) fail on these models because the extended CoT generation process introduces a new functional requirement not previously addressed. The core hypothesis is that reasoning models exhibit "functional heterogeneity", and the paper introduces the concept of "reasoning heads". To identify these heads, the authors propose the RLKV framework, which uses RL to directly optimize downstream reasoning quality. The RL agent is trained using a verifiable reward signal (final answer correctness on math problems) and an $L1$ penalty to encourage sparsity. Empirically, the authors claim SOTA performance, achieving 20-50% KV cache reduction on models like Llama-3.1-8B-R1 and Qwen-2.5-7B-R1 with "near-lossless" performance, significantly outperforming baselines.

### Strengths
1.  The paper insightfully identifies that the primary bottleneck in modern reasoning models is not just long-context *ingestion* (which many KV cache methods target) but long-context *generation* from CoT. The motivational analysis in Figure 1, which contrasts instruct vs. reasoning models on the same task, is a strong validation of this problem statement.
2.  The main conceptual contribution is the shift in optimization objective. Instead of relying on proxy metrics like next-token loss or output deviation, RLKV directly optimizes the true downstream task metric—reasoning accuracy—via an RL reward. Optimizing for semantic correctness rather than syntactic similarity is a significant idea.
3.  The authors identify the inherent conflict between a sparse, unstable task reward and a dense, stable $L1$ regularization penalty. The proposed solutions in Section 3.3 (adaptive penalty weights and curriculum-based data sampling) are clever, well-reasoned, and appear effective for making the RL training viable, as demonstrated in the ablation studies.

### Weaknesses
1.  The paper's claims of generalization are not supported by the data and reflect a misunderstanding of the benchmark landscape. RLKV is trained on 3,000 mathematical reasoning problems from the DeepScaleR dataset. The authors then claim that MBPP (a code generation benchmark) serves as an evaluation of "out-of-distribution generalization". This may not be correct; mathematics (like AIME, DeepScaleR) and code (like MBPP, HumanEval) are potentially considered to be *related, in-domain reasoning tasks*. Therefore, the paper should provide more out-of-distribution generalization tests (e.g., to narrative summarization, logical planning, or RAG-based QA). 

2.  For a paper proposing an *efficiency* method, the absence of a training cost analysis is a critical omission. 
- The paper proposes a new, expensive RL training phase. RL for LLMs is known to be computationally intensive and complex to scale.
- This method is benchmarked against alternatives that are either training-free (H2O, R-KV) or have significantly cheaper, non-RL training (e.g., DuoAttention's single-pass proxy task or PruLong's next-token loss).
- The authors have shifted a large, undisclosed amount of computation from inference to this new training stage. Without a "Total Cost" analysis (Training FLOPs + Inference FLOPs), it is impossible to assess if RLKV provides practical efficiency gains.

3.  As admitted in Appendix A.2, the authors "convert fixed budgets to dynamic allocation" for H2O and R-KV. But these methods are explicitly designed as fixed-budget algorithms. This modification means the authors may not be comparing against the SOTA baselines as published, invalidating the empirical claims in Figure 5 and Table 1.

4.  The text states "...primarily due to quadratic attention computation and expanding KV cache." This is incorrect. The KV cache memory grows *linearly* ($O(n)$) with sequence length, not quadratically. 

5.  Typo: "AIME25" in the appendix vs. "AIME24" in the main body.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper presents RLKV, an RL–based framework for reasoning-aware KV cache compression in LLMs.  The key insight is that attention heads in reasoning models are functionally heterogeneous, with only a subset being essential for maintaining CoT. RLKV introduces mixed attention with learnable gating adapters and optimizes them via GRPO using reasoning accuracy as the reward and L1 sparsity regularization to identify which heads truly matter. The method adaptively allocates full KV cache to reasoning-critical heads and compressed cache to others, achieving 20–50% memory reduction while maintaining near-lossless or even improved reasoning performance on multiple benchmarks such as GSM8K, Math500, AIME24, MBPP, and models such as Llama-3.1-8B-R1 and Qwen-2.5-7B-R1. Extensive analyses show that reasoning heads are more crucial than other heads, and the proposed adaptive penalty and self-distillation techniques stabilize RL training. Overall, RLKV provides a novel and interpretable approach to efficient reasoning inference.

### Strengths
1. The motivation that only a subset of attention heads is critical for reasoning is very interesting. It's also well supported by empirical evidence. The observation that near-lossless reasoning performance can be maintained even when 50–80% of KV heads are compressed makes this hypothesis convincing.
2. The method does not require model retraining and instead learns a small number of gating parameters to mix full and compressed attention. This design choice makes the approach easy to adopt in practice.
3. The method achieves strong results across multiple reasoning benchmarks, maintaining almost full accuracy with 20–50% KV cache reduction.
4. The paper provides clear evidence for the functional role of reasoning heads, with ablation and masking studies showing they are substantially more important than retrieval heads.

### Weaknesses
1. As mentioned in the paper, some heads are more important for reasoning than others, but their original functional roles in the base models (e.g., Qwen2.5-Math-7B-Base) remain unclear. It would strengthen the paper to analyze whether these reasoning-critical heads are specialized in math or coding-related attention patterns in the base model. A suggested experiment is to apply the same compression scheme to the base model and evaluate its performance on reasoning and general tasks (as in Figure 6), which could reveal whether reasoning heads emerge during post-training or pre-exist in base architectures.
2. In Figure 3, most gating coefficients (α) in Qwen-2.5-7B-R1 appear close to 1, suggesting the model heavily favors full attention across heads. This trend seems inconsistent with the sparsity objective.
3. The current method freezes model weights. Is it possible/feasible to explore joint training of gates and model parameters, which may yield stronger compression performance?

### Questions
Please see weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4
