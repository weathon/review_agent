# Beyond Homogeneous Attention: Memory-Efficient LLMs via Fourier-Approximated KV Cache

- Decision: Reject
- Scores: 4, 8, 4, 2

## Abstract
Large Language Models struggle with memory demands from the growing Key-Value (KV) cache as context lengths increase. Existing compression methods homogenize head dimensions or rely on attention-guided token pruning, often sacrificing accuracy or introducing computational overhead. We propose FourierAttention, a training-free framework that exploits the heterogeneous roles of transformer head dimensions: lower dimensions prioritize local context, while upper ones capture long-range dependencies. By projecting the long-context-insensitive dimensions onto orthogonal Fourier bases, FourierAttention approximates their temporal evolution with fixed-length spectral coefficients. Evaluations on LLaMA models show FourierAttention achieves the best long-context accuracy on LongBench and Needle-In-A-Haystack (NIAH). Besides, a custom Triton kernel, FlashFourierAttention, is designed to optimize memory via streamlined read-write operations, enabling efficient deployment without performance compromise.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work is based on an interesting observation that, within attention heads, some dimensions focus on local context while others capture long-range dependencies. To exploit this heterogeneity, the paper introduces FourierAttention, which compresses the locally focused dimensions of the KV cache into fixed-length states using the HiPPO framework combined with a translated Fourier transform, achieving efficient and training-free KV-cache compression.

### Strengths
* **Novel Observation.** This work presents an interesting finding that within a single attention head, some dimensions focus on local context while others capture global dependencies. This insight provides a new perspective on KV-cache compression, distinct from prior token-wise, head-wise, or precision-wise (quantization-based) methods.

* **Solid System Implementation.** The paper adapts kernels from FlashAttention and FlashDecoding to build FlashFourierAttention, effectively integrating Fourier-based compression into the attention computation and avoiding extra memory movement during inference.

### Weaknesses
* **Limited Latency Evaluation.** The experiments only report prefill latency, while the majority of KV-cache read and update operations occur during decoding. Including TPOT (Time per Output Token) or throughput comparisons would provide a more complete performance evaluation.
* **Limited Model Generalization.** FourierAttention is built upon the finding of heterogeneous dimension roles within attention heads, yet the evaluation is limited to LLaMA-3.1-8B and LLaMA-3.2-3B. Models with different head sizes (e.g., LLaMA-3.2-1B with only 64 dimensions per head) may not exhibit the same behavior, raising questions about general applicability.
* **Lack of Per-Head Analysis.**
  Prior works such as MInference [1], RazorAttention [2], DuoAttention [3], and HeadKV [4] have shown that sparsity and attention patterns vary across heads. This paper provides limited analysis on whether the observed dimension heterogeneity holds consistently across all attention heads, leaving this aspect unclear.



[1] Jiang, Huiqiang, et al. "Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention." *Advances in Neural Information Processing Systems* 37 (2024): 52481-52515.

[2] Tang, Hanlin, et al. "Razorattention: Efficient kv cache compression through retrieval heads." *arXiv preprint arXiv:2407.15891* (2024).

[3] Xiao, Guangxuan, et al. "Duoattention: Efficient long-context llm inference with retrieval and streaming heads." *arXiv preprint arXiv:2410.10819* (2024).

[4] Fu, Yu, et al. "Not all heads matter: A head-level kv cache compression method with integrated retrieval and reasoning." *arXiv preprint arXiv:2410.19258* (2024).

### Questions
* Regarding the main finding — why can the locally focused and globally focused dimensions be separated by contiguous index ranges? Wouldn’t one expect them to be more interleaved or randomly distributed across dimensions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes FourierAttention, an online compression method for the KV cache. FourierAttention uses Hippo-FourierT to compress the KV cache, significantly reducing the memory overhead of the KV cache while maintaining better performance compared to baseline methods.

### Strengths
+ This paper discovers that the different dimensions of Q and K in attention computation play different roles. Initially, this finding seemed counterintuitive to me, as I typically assumed that different dimensions were homogeneous—this was because I had overlooked the effect of ROPE. The paper innovatively leverages this insight by applying different compression strategies to different dimensional ranges, achieving better compression efficiency.

+ The paper also implements the proposed method's corresponding FlashFourierAttention operator, which is an important contribution to the open-source community.

+ The experiments provide thorough comparisons in terms of performance and latency, effectively demonstrating the superiority of the proposed method.

### Weaknesses
+ The paper only conducts experiments on LLaMA. It would be better to include comparisons with other open-source models as well.

### Questions
See above

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
4

### Summary
This paper proposes FourierAttention, a training-free KV cache compression method. The idea is that KV cache tokens have long-context-insensitive and long-context sensitive channels -- the former can be projected onto translated Fourier bases, while the latter can be ;eft uncompressed. Authors also propose a Triton kernel for on-the-fly reconstruction of the tokens. Experiments on long-context benchmarks with LLaMA-3.* models show improvements over baseline KV cache compression methods like SnapKV and PyramidKV. One concern I have is that results in Tab. 1 often show very marginal improvements, and results are often worse than those of SnapKV and PyramidKV -- are those statistically significant? Also, authors only experiment with Llama-3.1-8B and Llama-3.2-3B -- what happens on models not in the Llama3 family (e.g. any of the recent Qwens)? On the other hand, the baselines are very competitive (according to e.g. KVPress, https://github.com/NVIDIA/kvpress) so any improvements on those is more than welcome.

Tiny typo: line 189 -- "We denode"

### Strengths
- Interesting analysis about how latent dimensions can be characterised into long-context-sensitive/insensitive (also from a mechinterp point of view)
- Strong results compared with competitive baselines like SnapKV, PyramidKV, and Palu at comparable budgets on LongBench and NIAH
- Interesting custom FlashFourierAttention kernel

### Weaknesses
- Absolute gains are *very* modest -- are they statistically significant?
- I was not able to find quantitative results on latency, please let me know if I missed those
- Experiments only on Llama3-based backbones

### Questions
- What happens on other families on models?
- Are results statistically significant?
- How sensitive are the results to errors in identifying the two types of latent dimensions?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a training-free KV cache compression method that exploits heterogeneous roles across transformer head dimensions. The authors observe that lower dimensions capture local context while upper dimensions capture long-range dependencies, and compress the former using Fourier basis functions from the HiPPO framework. A custom Triton kernel (FlashFourierAttention) is implemented for efficient deployment. The observation and the custom kernel are nice, but the evaluation is not comprehensive enough. Both the choice of the baselines (compression vs quantization) and the range of performed experiments and models selected are not enough. Also, the paper would benefit from a better framing of the method with respect to the literature about KV Cache compression.

### Strengths
- The empirical finding that different head dimensions serve distinct roles (local vs. global context) is interesting and well-validated through ablation studies
- Leveraging the HiPPO framework provides mathematical rigor, and the adaptation from complex to real-valued representations is sensible for practical implementation.
- The custom Triton kernel shows practical engineering effort and is very nice.

### Weaknesses
- (General) The title and abstract claim this is "KV cache compression," but the actual mechanism is better described as lossy approximation or dimensionality reduction. For this reason, while comparing to Palu (dimensionality reduction) makes sense, comparing to SnapKV and PyramidKV is kind of strange as these perform KV Cache eviction in a different setting. The authors should (a) discuss the differences between KV Cache compression, quantization, reduction in a clear way and (b) include quantization in the baselines.
- (Method) The observation about dimension heterogeneity is a bit over-claimed and not entirely new, as prior work has noted similar patterns. This is cited by the authors themselves.
- In Table 1. The comparison shows that the methods have been tested with different compression ratios. Why not use the same compression ratio for fair comparison? 
- In Table 1 and in general across the paper, for a fair and comprehensive evaluation one should consider a range of compression rations for all methods instead of a specific one.
- The choice of L-init=4, L-local=1024, N=512 appears arbitrary with no ablation study or principled justification. How sensitive is performance to these choices?
- The method is only evaluated on Llama models. This strongly hinders the generalization of the method to different architectures.
- I am not sure I understood how is the dimension selection actually performed? The authors say "we directly compress and decompress all KV caches, prioritizing dimensions with smaller mean-squared error" but doesn't specify the algorithm. Is this done once offline or adaptively?

### Questions
- Could you frame the contribution wrt KV Cache compression, Quantization and Low Rank reduction.
- Could you provide exact numerical results for latency and memory consumption rather than just plots?
- Why aren't other architectures included in the evaluation ? Could you include them ? 
- Could you provide results in Table1 for different compression ratios ?

### Soundness
2

### Presentation
3

### Contribution
2
