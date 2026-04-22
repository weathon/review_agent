# CommonKV: Compressing KV Cache with Cross-layer Parameter Sharing

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 6, 2

## Abstract
Large Language Models (LLMs) confront significant memory challenges due to the escalating KV cache with increasing sequence length. As a crucial technique, existing cross-layer KV cache sharing methods either necessitate modified model architectures with subsequent pre-training or incur significant performance degradation at high compression rates. To mitigate these challenges, we propose CommonKV, a training-free method for cross-layer KV cache compression through adjacent parameters sharing. Inspired by the high similarity observed in cross-layer hidden states, we utilize Singular Value Decomposition (SVD) to achieve weight sharing across adjacent parameters, resulting in a more easily mergeable latent KV cache. Furthermore, we also introduce an adaptive budget allocation strategy. It dynamically assigns compression budgets based on cosine similarity, ensuring that dissimilar caches are not over-compressed. Experiments across multiple backbone models and benchmarks including LongBench and Ruler demonstrate that the proposed method consistently outperforms other low-rank and cross-layer approaches at various compression ratios. Moreover, we find that the benefits of CommonKV are orthogonal to other quantization and eviction methods. By integrating these approaches, we can ultimately achieve a 98% compression ratio without significant performance loss.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors note the similarity in hidden representation across layers -- however, there is a high dissimilarity in crosslayer KV pairs due to differences in W_k and W_v. To alleviate KV-footprint growth, concatenate adjacent layer weight matrices W_k, W_v and run SVD, leading to a mergeable latent KV cache,. Further, the authors use cosine similarity to adapt the budget allocation. This is a training-free method with RoPE reapplied upon reconstructions.

### Strengths
- an interesting observation regarding the KV-generator being compressible is utilized to effectively reduce inference overhead.
- no pretraining/fine-tuning makes it quite practical
- strong performance with good  latency analysis. also orthogonal to other optimizations (eviction/quantization)
- discusses SVD cost of xKV well, does not have the same limitations, addresses a good set of baselines.

### Weaknesses
- Several figure references are broken, TODO -- should be removed.

### Questions
- Comments on comparison with / Applicability to MLA would be useful
- Online SVD becomes less than 10% of prefill at 128K, but this approach requires significant changes at deployment (reparametrize attention, changes for latent cache, rope, etc.) -- is it trivial to switch between this method and vanilla attention at inference? i.e., at lower sequence lengths can we do vanilla attention where prefill cost is high? What is the prefill cost at lower sequence length, and the overhead as the conversation starts at low token count but increases over multi-turn?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
* The paper proposes an SVD-based method to enable parameter sharing across KV Cache layers along with an adaptive budget strategy.  Experiments across multiple models and benchmarks show up to 98% compression ratio without significant performance loss.

### Strengths
* Neat insight into using parameter sharing across layers to compress based on similarity between adjacent layers of the KV Cache.
* Allows dynamic adaptation across different groups of KV layers for better performance.
* Training-free method of KV merging leads to lower offline compute overhead
* Results are very impressive compared to all the baselines.

### Weaknesses
* *To reduce computational overhead, we only use the cosine similarity between the first and last layers within each group as its score* It needs further justification that it is sufficient to use just the first and last layer within each group without sacrificing quality.
* It is not clear why the latent cache is more easily mergeable. I see the claim that it has more consistent hidden states, but this claim needs further explanation. This is especially important as there is a two-way overhead of constructing the latent cache and reconstructing the original KV cache. Secondly, it is not clear why parameter sharing based on just "cosine similarity" would not degrade quality.
* NIT: References to Figures are broken in the text.

### Questions
* Table 1 says *Due to significant performance loss, some methods do not report results for all compression ratios*. I think it would still be useful to see the comparison as even the case which is included (MiniCache for CR 0.3, the performance loss is very significant)

### Soundness
3

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
This paper proposes CommonKV, a training-free KV cache compression method. It addresses the dissimilarity of KV caches across layers by using offline Singular Value Decomposition (SVD) to share parameters between adjacent layers. This creates a highly consistent "latent KV cache" from similar hidden states, which can be merged with significantly less performance loss than merging the original KV caches directly.

### Strengths
The method's core strength is its novel idea of creating a more consistent latent cache via parameter sharing, which directly addresses the root cause of poor performance in direct KV cache merging. Its training-free nature makes it highly practical and easy to apply to existing models. CommonKV demonstrates superior empirical performance over baselines at high compression ratios with minimal inference latency overhead. Furthermore, its ability to be combined with quantization and eviction methods to achieve up to 98% compression is a significant advantage .

### Weaknesses
The paper would be stronger with a more detailed sensitivity analysis of the SVD rank hyperparameter. Additionally, the handling of GQA models feels like a workaround, and a deeper analysis of the architectural interaction would be beneficial.

### Questions
The idea of low rank decomposition and weight sharing sounds very related to "LORC: Low-Rank Compression for LLMs KV Cache
with a Progressive Compression Strategy", where attention weights and caches in the same layer are shared.

### Soundness
3

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
3

### Summary
This paper proposes CommonKV. CommonKV is a KV cache compression method for long-context LLM inference that is training-free and leverages cross-layer compression. The key idea is to make adjacent layers' KV caches more mergeable by sharing part of the K/V projection parameters across nearby layers. To be more specific, this is done by performing a concatenated SVD over groups of layers, which yields a shared matrix and layer-specific matrices. Evaluation on major long-context benchmarks (LongBench and RULER) with 7/8B LLMs show that CommonKV preserves accuracy while enabling significant compression ratio.

### Strengths
Thank you for submitting this paper to ICLR! KV cache compression is one of the most important and popular topics in efficient LLM inference. I appreciate the authors' efforts on the analysis of limitations in existing methods (not training free, and performance degradation under aggressive compression). Cross-layer parameter sharing is a very reasonable idea (somehow explored before). The evaluation baselines are strong and timely methods in the field as well. In particular, section 6 gives a very comprehensive overview of methods in this area, which is a unique contribution beyond the proposed method itself.

### Weaknesses
1. As compared to other KV cache compression papers in the community, the evaluation is flawed in many aspects, including but not limited to context window length of 8K, model sizes (7/8B), rationale of hyperparameter choices, system performance metrics, etc. Please refer to "questions" for a comprehensive list. 

2. There are quite a few unfilled question marks (??) and TODOs in the current draft. 

3. Figures 1, 3, 4 are hard to read --- Please consider enlarging the fonts.

### Questions
Please justify the design choice of setting the context window of all models to be 8K. As an example, LongBench has an average input length of 6.7K and a lot of queries of length > 10K. It'd be great to see whether the proposed methods scales to longer contexts, especially given that KV cache compression could target ultra-long-context scenarios. 

It is unclear whether the proposed approach scales to larger LLMs, especially those with > 8B parameters. Do you have intuitions on whether larger models with preserve the performance gain? 

Could you give some sensitivity analysis of group size and SVD rank? I'm a bit confused by the current rationale of picking values for these two hyperparameters. 

End-to-end inference latency is not a typical metrics that KV cache compression researchers use for system performance measurement. Please consider adding TTFT/throughput numbers.

### Soundness
2

### Presentation
2

### Contribution
2
