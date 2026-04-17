# GlowQ: Group-Shared LOw-Rank Approximation for Quantized LLMs

- Decision: Accept (Poster)
- Scores: 4, 4, 6

## Abstract
Quantization techniques such as BitsAndBytes, AWQ, and GPTQ are widely used as a standard method in deploying large language models but often degrades accuracy when using low-bit representations, e.g., 4 bits. Low-rank correction methods (e.g., LQER, QERA, ASER) has been proposed to mitigate this issue, however, they restore all layers and insert error-correction modules into every decoder block, which increases latency and memory overhead. To address this limitation, we propose GlowQ, a group-shared low-rank approximation for quantized LLMs that caches a single shared right factor per input-sharing group and restores only the groups or layers that yield the highest accuracy benefit. 
GlowQ computes the high-precision projection once per input-sharing group and reuses it across its modules, reducing parameter and memory overhead, and retaining the expressivity of layer-specific corrections. We also propose a selective variant, GlowQ-S, that applies the cached shared module only where it provides the largest benefit. Compared with strong baselines, our approach reduces TTFB by \(5.6\%\) and increases throughput by \(9.6\%\) on average, while reducing perplexity on WikiText-2 by \(0.17\%\) and increasing downstream accuracy by 0.42 percentage points. The selective model GlowQ-S further reduces latency, cutting TTFB by \(23.4\%\) and increasing throughput by \(37.4\%\), while maintaining accuracy within 0.2 percentage points on average.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes GlowQ, a group-shared low-rank correction method for quantized LLMs that improves both accuracy and efficiency by sharing a single right-factor matrix across input-sharing modules (e.g., Q/K/V), caching the projection to avoid redundant computation, and aligning the correction subspace with data usage via covariance-weighted optimization. Combined with a selective restoration strategy (GlowQ-S), it reduces TTFB by up to 23.4% and increases throughput by 37.4% with minimal accuracy loss, outperforming existing baselines in perplexity and downstream task performance while remaining compatible with standard PTQ pipelines.

### Strengths
1. Efficient Group-Shared Correction via Caching: GlowQ reduces redundant computation by sharing a single low-rank right factor across input-sharing modules (e.g., Q/K/V projections), enabling one-time computation and reuse of the projection R=BsharedXR=Bshared​X, which significantly lowers computational overhead during inference.
2. Data-Aware Subspace Alignment: By incorporating input covariance into the low-rank approximation objective, GlowQ aligns the correction subspace with directions that are most frequently activated in practice, enhancing recovery accuracy without increasing rank or parameters.
3. Scalable and Deployment-Friendly Design: The method combines a QR-reduced randomized SVD solver for efficient training with a selective restoration strategy (GlowQ-S), achieving substantial latency reduction (up to 23.4% lower TTFB) and throughput improvement (up to 37.4%) while maintaining compatibility with existing post-training quantization pipelines.

### Weaknesses
1. While the group-sharing idea is technically sound and well-executed, the central concept is an adaptation of well-known collective matrix factorization and SVD-sharing across blocks, now framed in the context of quantized LLMs. The extension to covariance alignment is also a known trick, though its synergy with group correction is effective.
2. While the paper compares GlowQ extensively to prior PTQ and error-corrected LLM quantization baselines (e.g., AWQ, GPTQ, L2QER, QERA, ZeroQuant), it omits direct comparison or numerical discussion of very recent approaches, such as rotation-based saliency-aware quantization methods (e.g., ROSAQ), vector quantization for KV cache (CommVQ, AnTKV), and loss-guided PTQ (GuidedQuant).
3. This method requires an additional calibration phase to estimate the compensation matrix, but the paper does not seem to list the time and memory costs for the training/calibration phase. Additionally, storing intermediate activations and error matrices for larger models (e.g., 30B+) could lead to significant memory pressure.
4. The paper claims the use of "custom CUDA W4A16 kernels," but it does not describe the implementation details or whether key techniques such as operator fusion were used in the low-rank correction modules. Therefore, it is unclear whether the comparisons between different methods are made at the same level of optimization.
5. Critically Limited and Potentially Misleading Experimental Scope (W4A16 only): The paper's most significant flaw is that all experiments are conducted in a W4A16 setting. This is an unrealistic and overly forgiving scenario that ignores the primary challenge in modern PTQ: activation quantization. By using FP16 activations, the main computation path and the input to the correction module remain noise-free, which dramatically simplifies the problem. The paper's claims of improved accuracy and efficiency are therefore unsubstantiated in any practical low-bit setting (e.g., W8A8, W4A4). It is entirely possible that the proposed group-sharing benefits would vanish or even become detrimental once activation quantization noise is introduced.
6. format error: line339 -> BitsAndBytes ?, line218,268,290,293 -> Missing formula number.

### Questions
1. The performance of GlowQ-S is highly dependent on the importance scoring functions (such as gec and gner). I am curious if there are other metrics that can indirectly reflect the effectiveness of these scoring functions. In other words, can these two metrics be proven to be globally optimal across a range of models?
2. The method relies on the assumption that modules within an input-sharing group can share a common right factor ( B_{\text{shared}} ), but it does not systematically verify the validity of this assumption across all layers or architectures. For example, when the functions of matrix projections like Q and V are different, it is unclear whether enforcing this sharing introduces bias.
3. The confinement to W4A16 is a major limitation. Can you provide any results or analysis for a W4A8 or W4A4 setting? How does your group-sharing approximation hold up when the input X to the correction path is also quantized, introducing another layer of error?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces GlowQ, a group shared low-rank approximation method, that a single rightside projection is used for a group with same input. A BX caching method is proposed to make it deployable. A selective method is proposed to further reduce the memory and latency.

### Strengths
- This paper is technically sound.
- The experimental results show the effectiveness of the proposed method.

### Weaknesses
- Writing and presentation need refinement. For instance, the symbol E in Eq. 1 is identical to E_cat in Eq. 2—consistent notation should be used. In Fig. 1 (inference-path sub-figure), the chosen colors are too similar to be distinguished; a more discernible palette is required.
- For GlowQ-S, the fraction of restored groups are not provided, should be provided for each experiment

### Questions
- In Fig.3, as PPL is lower the better, what is the definition of the percentage of PPL?
- In Tab.4, No Caching means GlowQ without caching? Or a layerwise method? If it means GlowQ without caching, the authors should provide performance comparisons with layerwise methods.
- Modern GPUs already natively support other narrow-bit formats such as MXFP4, NVFP4 and MXFP6, whose peak throughput is significantly higher than that of INT4. The paper should therefore clarify:
a) Is GlowQ directly applicable to these FP4/FP6 formats, or does its low-rank projection rely on integer-only operators?
b) If applicable, how does GlowQ’s accuracy compare with simply running the network in MXFP4/NVFP4/MXFP6 without any additional compression?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work describes an efficient low-rank compensation technique for weight-quantized LLMs. 
The method finds an optimal shared down-projection factor in a data-driven manner, with whitening and grouping treatment.

### Strengths
+ The idea is sound, with adequate proof. 
+ The method is relatively well described.  
+ There is potential practical significance.

### Weaknesses
- The empirical, data-driven, low-rank correction of quantization error is dependent on the data type and its precision.  As most of the results presented are with INT4, there is a lack of demonstration on the effectiveness for different data types and precisions--do the data statistics differ qualitatively under those conditions?  
- As activation quantization is also important in practice, it is not clear how quantization of the activations, in combination with weight quantization, change the story. 
- Quantization errors tend to accumulation over time across tokens as well, which is particularly relevant in long-context reasoning settings.  Reporting accuracy in language modeling perplexity is insufficient in addressing this issue.

### Questions
I have listed major questions in the Weaknesses section above.  Here are a few minor questions in addition.  

* The empirical statistical analysis of the quantization error is essential to this paper.  Do you have any results on any empirical scaling laws of the statistics, in addition to simple qualitative descriptions such as long tail?  It might be helpful to apply logarithmic scale in Figure 2 to expose certain power laws.  
* Is there any results on how and how well this method could be applied to MoE FFN layers?
* In addition to full-precision error correction, other orthogonal methods use mixed-precision, how is the choice among these methods subject to practical tradeoff?

### Soundness
3

### Presentation
2

### Contribution
3
