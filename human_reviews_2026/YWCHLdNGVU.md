# LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6, 4

## Abstract
The proliferation of long-context large language models (LLMs) exposes a key bottleneck: the rapidly expanding key-value cache during decoding, which imposes heavy memory and latency costs. While recent approaches attempt to alleviate this by sharing a single set of crucial tokens across layers, such coarse-grained sharing undermines model performance by neglecting the functional diversity of attention heads. To address this, we propose LycheeDecode, an efficient decoding method centered on a fine-grained hybrid-head attention mechanism that employs a hardware-efficient top-$k$ selection strategy. Specifically, the novel HardKuma-based mechanism partitions attention heads into a small subset of retrieval heads that dynamically identify crucial tokens and a majority of sparse heads that reuse them for efficient computation. Through extensive experiments on leading models like Llama3 and Qwen3 across diverse benchmarks for long-context understanding (e.g., LongBench, RULER) and complex reasoning (e.g., AIME24, OlympiadBench), we demonstrate that LycheeDecode achieves generative quality comparable to, and at times surpassing even the full-attention baseline. Crucially, this is accomplished with up to a 2.7x speedup at a 128K context length. By preserving the functional diversity of attention heads, our fine-grained strategy overcomes the performance bottlenecks of existing methods, providing a powerful and validated pathway to both efficient and high-quality long-context LLM inference. The implementation code, kernels, and models will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes LycheeDecode, a head-wise sparse attention mechanism with head sampling using HardKuma probability. During training, the HardKuma takes the input of $\alpha, \beta$, which are trainable parameters, to generate the probability focused at 0 and 1 for head selection between full attention and sparse attention. The pattern of sparse attention is propagated across layers from the full attention in shallow layers to the sparse attention in deep layers. The training loss of LycheeDecode composes of two parts: the distillation loss (between the sparsified model and the original dense model) and the regularization loss (to limit the number of full attention heads). LycheeDecode is benchmarked on various datasets, including general-purpose long-context evaluation such as LongBench and RULER, and complex reasoning datasets such as AIME and Olympiad-Bench. LycheeDecode is able to outperform the baselines on all tested benchmarks.

### Strengths
1. The paper is well written. The statement of the problem and the description of the methods, especially the HardKuma part, are clear and easy to follow.

2. The design of LycheeDecode is reasonable, efficient, and effective. In DuoAttention, head selection is conducted between the full attention head and the sliding window head with an attention sink. Therefore, the ability to capture long-context dependencies is reduced, since the sliding window has a relatively weaker ability to memorize context history. However, in LycheeDecode, this problem is well solved by using sparse attention instead of the sliding window. The selection process is performed by the head in the same position in previous layers, so no significant overhead for generating sparse patterns is introduced. This design is reasonable and makes a valuable contribution.

3. Mainstream evaluation benchmark datasets are included in this paper, and LycheeDecode achieves promising results on these benchmarks.

### Weaknesses
1. Since DuoAttention [1] is an important preliminary work for this paper, comparisons between LycheeDecode and DuoAttention should be conducted on long-context benchmarks such as RULER and LongBench.

2. "By optimizing the distributional parameters of HardKuma during training, our model learns a near-binary selection mechanism directly, thus mitigating the train-inference discrepancy and leading to a more stable and robust head specialization." (line 211-213) During inference, a head is identified as a retrieval head if $\mathbb{E}[z_h] \geq 0.5$. In this case, since $z_h$ is generated from a probability distribution, there is still a chance that this head will be treated as a non-retrieval head during training. Could the authors provide more explanation on why such a design can eliminate the train–inference discrepancy? If possible, a formal proof would help clarify this point.

---

[1] DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces LycheeDecode, a novel method for accelerating long-context inference by leveraging a hybrid sparse attention mechanism. The core idea is to specialize attention heads into retrieval heads and sparse heads". To enable end-to-end training and robust head specialization, the authors propose the HardKuma distribution, a reparameterizable method for near-binary head selection that mitigates the train-inference discrepancy common in discrete optimization. Results on several datasets demonstrates effectiveness.

### Strengths
1. The paper is easy to follow.
2. The authors give thorough experiments to validate the proposed methods.

### Weaknesses
1. The proposed method does not appear novel to me. For example, the idea of parameterizing discrete variables has been explored in prior work. Techniques such as Gumbel softmax and STE have been widely used.
2. Regarding the motivation in Figure 1, I question the significance of showing the overlap rate between corresponding heads in adjacent layers, as attention heads have no inherent ordering, and heads have no direct correspondence across layers.
3. I am uncertain about the necessity of the HardKuma distribution to reduce the train-inference gap. As I understand it, its parameters are optimized during training, but head types are determined at inference using a fixed threshold—this still introduces a train-inference discrepancy.
4. The method assigns fixed head types regardless of input, which may be suboptimal. A more robust approach would make head specialization adaptive to the specific input during inference.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes LycheeDecode, a sparse attention mechanism to accelerate long-context LLM inference by partitioning heads into a small set of "Retrieval Heads" that perform full attention and a majority of "Sparse Heads" that efficiently reuse the critical tokens identified by the retrieval heads. To learn this specialization, it introduces the HardKuma distribution, which mitigates the train-inference discrepancy common in discrete optimization. Experiments show LycheeDecode achieves up to 2.7x speedup at 128K context lengths while matching or even exceeding the performance of the full-attention baseline.

### Strengths
1. The paper's primary innovation is the hybrid-head decoding mechanism. This mechanism establishes a head-indexed pipeline, where a Retrieval Head in one layer selects tokens specifically for its corresponding Sparse Head in the next layer to reuse. 

2. The use of the HardKuma distribution directly targets a known weakness (train-inference discrepancy) in prior training-based specialization methods, leading to a more stable and direct optimization of the discrete head roles.

### Weaknesses
1. Missing Key Baseline Comparisons: The paper's central claim is that its cooperative head-specialization architecture is superior. However, it fails to provide any end-to-end performance or speed comparisons against the most direct SOTA competitors in the head-specialization sub-field (e.g., DuoAttention, RazorAttention). It only compares against a layer-sharing method (TidalDecode). This makes the SOTA claim unsubstantiated, as we cannot see how it performs against other architectures with the same design philosophy.

2. Weak Evidence: The claim of surpassing the full-attention baseline is remarkable. The paper's only explanation is a vague hypothesis about "filtering out irrelevant context that may act as noise." This claim requires rigorous qualitative proof (e.g., visualizations, case studies), which is absent.

3. Lack of Sensitivity Analysis: The method's performance hinges on two key hyperparameters: the retrieval head budget ($N_{target}$) and the token budget ($k$). The paper provides no ablation studies on how varying these budgets affects the performance/latency trade-off. The $N_{target}$ value was just set to "match... TidalDecode," which is arbitrary and likely not optimal.

### Questions
1. Missing Baselines (Duo/Razor): Why are there no end to end comparisons against DuoAttention and RazorAttention? The paper's premise is that its cooperative architecture is a key advantage over DuoAttention's isolated head roles, but this central claim is not experimentally validated.

2. Confounding Variable (Cache Correction): In Table 2, the "Cache Correction" (CC) strategy provides a large performance boost. Did you test a "Full Attention + CC" baseline? Your current results could be interpreted as the CC method being the main source of the performance gain over the baseline, not LycheeDecode.

3. Evidence for "Surpassing Full Attention": Can you provide a specific, qualitative example (e.g., visualizing attention) that proves the full-attention model failed due to "noise" while LycheeDecode succeeded by filtering it?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces LycheeDecode, which learns to partition attention heads into a few retrieval heads (full attention with top-k token selection) and many sparse heads (reusing selected tokens). The method uses HardKuma to mitigate train–inference mismatch and employs a custom TileLang block-sparse kernel. On long-context benchmarks (LongBench, RULER) and reasoning tasks (AIME24, OlympiadBench), it achieves competitive accuracy with up to 2.7× speedup at 128k context length.

### Strengths
1. The paper provides compelling motivation that layer-level token sharing ignores significant functional diversity among heads, with empirical evidence showing highly variable top-k overlap across adjacent layers.
2. Trainable retrieval and sparse head assignment, HardKuma offers a principled, differentiable relaxation that tends toward binary outcomes without rounding.
3. The custom hybrid-head kernel yields substantial kernel-level improvements, this strengthens the practicality claim.

### Weaknesses
1. Lack of comparison with training-based long-context inference, DuoAttention is arguably the most directly comparable prior work, and the omission makes it difficult to isolate novelty beyond the use of top-k propagation.
2. Head role consistency and interpretability are not evaluated, how stable head assignments are across different settings (random setting), whether specialization generalizes to unseen domains.
3. Training complexity is under-characterized. The method computes both full and sparse attention per head during training, doubling attention paths. FLOPs, KV bandwidth, and convergence cost are not reported. Scaling to deeper models remains unclear.
4. Propagation across layers may accumulate error.

### Questions
1. Do retrieval heads discovered by LycheeDecode overlap with previous methods' identified retrieval heads? (such as DuoAttention or QR head https://arxiv.org/abs/2506.09944)
2. What is the training FLOPs overhead vs. DuoAttention or RazorAttention?
3. Will head roles transfer to long-form summarization or narrative QA?
4. How does training complexity grow with more layers and longer sequences?
5. What is the Jaccard overlap between retrieval heads discovered by LycheeDecode and DuoAttention? Is the retrieval-head model-dependent or training method related?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LycheeDecode, a sparse attention method that categorizes attention heads into "retrieval heads" and "sparse heads" to accelerate long-context LLM inference. The retrieval heads perform full attention to identify critical tokens, which are then reused by sparse heads for efficient computation.

### Strengths
The paper presents a well-motivated approach. The experimental results demonstrate that LycheeDecode achieves performance comparable to full attention baselines on complex reasoning tasks.

### Weaknesses
1. Recent work has demonstrated that trainable sparse attention can also achieve efficient decoding [1-3]. The paper lacks discussion and empirical comparison with these methods.

[1] Native sparse attention: Hardware-aligned and natively trainable sparse attention
[2] Minicpm4: Ultra-efficient llms on end devices
[3] SeerAttention-R: Sparse Attention Adaptation for Long Reasoning 

2. While the paper shows kernel-level speedup for different sparse head ratios, there is no corresponding analysis of how varying the proportion of retrieval heads affects model performance. An ablation study examining the performance-efficiency trade-off across different retrieval head budgets would strengthen the paper.

3. The efficiency evaluation focuses on synthetic settings with fixed context lengths. For math reasoning tasks that involve long chain-of-thought generation, where the sequence length grows dynamically during decoding, end-to-end latency measurements would be more convincing.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
