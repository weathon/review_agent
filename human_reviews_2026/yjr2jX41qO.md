# Channel-Aware Mixed-Precision Quantization for Efficient Long-Context Inference

- Decision: Accept (Poster)
- Scores: 4, 2, 6

## Abstract
The key-value (KV) cache plays a vital role in accelerating autoregressive inference for large language models (LLMs). However, its linear memory growth with sequence length poses significant memory bottlenecks, especially in long-context scenarios.
Quantization offers a promising solution for memory efficiency. While existing methods typically apply channel-wise quantization to the key cache and token-wise quantization to the value cache, they suffer from severe performance degradation under low-bit configurations.
Our analysis reveals that quantization sensitivity varies across individual KV channels, presenting an opportunity for non-uniform bit allocation. Following this finding, we propose ChanMix, a mixed-precision quantization framework that supports channel-wise quantization on 2-bit setting with custom Triton kernels implementation. To improve low-bit quantization performance, we introduce a channel-aware bit reallocation strategy, which allocates bits across channel sensitivity.
Through extensive evaluation, ChanMix demonstrates superior performance across the NIAH, RULER, and InfiniteBench benchmarks for the Llama, Mistral, and Qwen model families, achieving improvements of at least 5 absolute percentage points on RULER compared to all baseline methods. Additionally, ChanMix enables a 2.3× increase in batch size and supports a 1.5× longer context length during inference. 
Our code is available at https://github.com/cxiliao/ChanMix.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies KV cache compression, which is important for LLM serving in real-world applications, especially in long-context scenarios. The authors claim that uniform bit allocation is the source of error for methods applying channel-wise quantization of keys and token-wise quantization of values. The authors propose ChanMix, a mixed-precision outlier-aware framework specifically designed for KV cache quantization. They evaluate their method on several benchmarks and develop a custom Triton kernel implementation to support their claims.

### Strengths
1.	The paper is well written. The tables and figures are well presented.
2.	The numerical results (e.g. Table 1) are promising: ChanMix outperform several existing methods in most cases.
3.	The efficiency analysis is clear, and a kernel implementation is useful for the community.

### Weaknesses
1. Lines 17–19 and 53 overstate the findings. While the data is compelling, it does not support the claim that all quantization error stems from uniform bit-width allocation. I recommend softening the language to allow for nuance. Mixed-precision quantization is not new, and sensitivity variation across channels is expected. That said, this appears to be the first application of mixed-precision to KV, which is appreciated. The paper is well-motivated, but the novelty and breadth of the claims should be tempered.
2. While ‘outlier’ is a widely used term in quantization literature, retrieval channel is a more recent, niche idea emerging from interpretability studies of attention heads, not a canonical term in the context of quantization. The corresponding part in section 4.1 should be expanded to provide more definition and background as retrieval channel is important in the motivation and implementation of the method proposed in this paper.

### Questions
1. How are outliers and subnormal channels defined? The first usage is Section 4.1 without proper definition.
2. Can you please provide more explanation for the statement ‘Channel reordering ensures efficient 8-bit aligned storage of the quantized cache’? It is also not clear to me how this reordering is done.
3. The largest models the paper has experiments with are Llama-3-8B and Qwen-2.5-14B. It would be good to test on larger models, such as Qwen3-32B, to see if the method scales well. Since sensitivity-aware bit allocation is a key contribution, I think the paper would benefit from adding QAQ into the comparison. Have the authors done such experiments?

I am open to increasing my rating if the weaknesses and questions are resolved.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes ChanMix, a channel-aware mixed-precision quantization framework for compressing the key-value cache to enable efficient long-context inference. It identifies asymmetric sensitivities across KV channels, categorizing them as outlier, subnormal, and retrieval-sensitive, and allocates bits adaptively (1-4 bits) to minimize quantization error while maintaining low memory usage. The framework supports 2-bit to FP8 precision, includes channel reordering for 8-bit-aligned packing, and uses custom Triton kernels for implementation (but the code is not provided). Evaluations on Llama, Mistral, and Qwen models across NIAH, RULER, and InfiniteBench benchmarks show at least 5% improvement on RULER over baselines, with 2.3× larger batch sizes and 1.5× longer contexts.

### Strengths
- The paper's originality stems from its novel three-way channel sensitivity categorization (outlier, subnormal, retrieval) and adaptive bit allocation (4 bits for retrieval, 3 for outliers, 1 for subnormals; Figure 3, lines 290-294).
- This assessment is based on the methodology (Section 4, lines 162-323) and claims in the abstract/introduction, emphasizing how it removes limitations from prior quantization works (e.g., uniform bits in KIVI, line 119) while the setting is very limited.

### Weaknesses
- Code is not provided, resulting in a lack of reproducibility.
- The experimental setup does not clearly demonstrate the effectiveness of the proposed method. In `L-323`, a group size of 32 and a residual length of 128 are mentioned. The group size is relatively small, while the residual length is large. This configuration could be used to show the method’s effectiveness under less restrictive settings.
- The retrieval-head detection hyperparameters (n, t) are fixed “by experience.” However, stability across different values and architectures is not explored

### Questions
- Please address the items mentioned under Weaknesses. For example:
a. Lack of reproducibility
b. Use of limited settings for group size and residual length

- Your evaluation focuses mainly on long-context input-short-context output datasets. What is the performance of your method on long-context output tasks?
- There is ambiguity in the “≥5% improvement” claim. The abstract states “improvements of at least 5% on RULER,” but the tables show absolute percentage-point gains compared to the baselines (e.g., +5-8 points). Please clarify whether “%” refers to percentage points or relative percent.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the KV cache memory bottleneck for long-context inference. It identifies that the performance degradation of uniform low-bit quantization stems from asymmetric channel sensitivity. The authors propose ChanMix, a mixed-precision quantization framework that allocates bits based on a three-way channel classification: retrieval-sensitive channels (e.g., 4-bit), outlier channels (e.g., 3-bit), and robust/subnormal channels (e.g., 1-bit). The paper introduces an efficient one-shot method to identify these critical retrieval channels. 

ChanMix is implemented with custom Triton kernels for efficient, 8-bit-aligned storage and dequantization. Experiments on Llama, Mistral, and Qwen show state-of-the-art results on RULER, NIAH, and InfiniteBench, significantly outperforming prior quantization methods.

### Strengths
- The paper's core strength is the novel insight that "retrieval-sensitive" channels are a distinct category from magnitude-based "outlier" channels, and both require higher precision . This three-way sensitivity (retrieval, outlier, subnormal) is well-motivated by analysis and validated by strong ablation studies . 
- The proposed one-shot method for identifying retrieval heads is simple and efficient . The SOTA results (particularly the >5% gain on RULER) are significant. 
- I like the custom Triton kernel implementation part, which fuses channel reordering with (de)quantization, makes the method highly practical and efficient

### Weaknesses
- The primary weakness is the heuristic nature of the bit allocation policy (1, 2, 3, 4 bits). The paper does not provide an analysis of how this policy was derived or its optimality. 
- The generalizability of the one-time, offline channel profile is not thoroughly tested. It is unclear if a profile from Wikitext and a synthetic prompt  holds for all downstream tasks. 
- The paper asymmetrically analyzes the K cache (channel-wise) but not the V cache (token-wise), lacking justification for this design choice.

### Questions
- How was the specific bit allocation (1, 2, 3, 4 bits) determined, and how sensitive is the model to changes in this policy?
- How robust is the offline channel profile? Does a profile generated on Wikitext transfer to specialized domains (e.g., code, math), or is reprofiling required for optimal performance?
- Why is the V cache not analyzed for channel-sensitivity and quantized channel-wise, similar to the K cache? Is the V cache less sensitive, or is this a design choice for simplicity?

I will raise my score if authors give a good rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3
