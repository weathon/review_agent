# Joint Encoding of KV-Cache Blocks for Scalable LLM Serving

- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Modern large language models (LLMs) drive interactive AI systems but are bottlenecked by the memory-heavy growth of key–value (KV) caches, which limits real-time throughput under concurrent loads. Existing KV-cache compression methods rely on rigid heuristics, disrupt tensor layouts, or require specialized compute, hindering scalability and deployment. 

We propose joint encoding of KV-cache blocks, which fuses similar blocks across requests and input chunks into shared representations while preserving standard cache structure. This alleviates the KV-cache memory bottleneck, supporting high-concurrency serving without specialized hardware. Theoretically, we analyze the rate–distortion tradeoff of fused cache blocks under a Poisson process model. Empirically, our method achieves up to 4.38× KV-cache compression with negligible accuracy loss across diverse LLMs and benchmarks, outperforming recent structured and adaptive compression baselines. Our results establish a scalable, plug-and-play pathway for memory-efficient, high-throughput autoregressive inference. Code is available at \href{https://anonymous.4open.science/r/kv_joint_encoding-55B0/}{\nolinkurl{kv_joint_encoding-55B0}}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a novel method to fuse similar KV blocks across requests and develop a shared representation to reduce the memory bottleneck of KV Caches. Their method achieves up to 4.38× KV-cache compression with negligible quality loss across many LLMs and benchmarks.

### Strengths
* The paper provides a theoretical proof of the distribution of above-threshold similarities, which is key to the fusion algorithm design.
* The paper provides a mechanism to fuse KV blocks both across chunks of the same request and across requests, making the memory compaction more generalizable.

### Weaknesses
*  The paper says methods like prefix caching are sub-optimal due to the lack of an exact match between prefixes. However, there is related work in the area of semantic caching (where KV caches are shared across semantically similar requests based on embedding similarities, e.g. cosine similarity ). Given the paper's motivation, it is important to compare with a technique that shows whether the KV Cache can be reused across semantically similar requests with comparable accuracy degradation. Such a baseline might be more efficient than the paper's method.

* Table 1 shows that the accuracy drop is significant (around 5%), which weakens the claim that the tradeoff leads to a negligible drop in accuracy. This leads me to the concern that the BFF technique is not as generalizable or robust as claimed.

### Questions
* It is a little unclear how the thresholds and similarity values are set and used in practice. Is there an offline calibration phase for this?
* Is there a reason why the motivating Figure 3 uses a different model than that from the evaluation? It would be good to see the motivation (which explores the threshold) and the results for the same model.
* I'd encourage the authors to compare their method with a stronger semantic caching baseline for KV Cache reuse.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new method of KV cache compression that proposes joint encoding of KV-cache blocks so that similar blocks across requests or across input chunks can share a fused representation. The authors present two complementary approaches (for decode and prefill respectively): Batch Fast-Fusion (BFF), across concurrent requests during decoding, and Chunks Fast-Fusion (CFF), across chunks during prefill. Experiments report up to 4.38 times KV-cache compression with negligible accuracy drop on several models and benchmark datasets.

### Strengths
Thank you for submitting this paper to ICLR! 

This paper targets a timely topic: KV cache compression for memory-efficient, long-context LLM inference. The introduced method can be used directly without modifying paged attention design of serving engines like vLLM, since it avoids operation on tensor / layer level of KV cache. I like your two complementary algorithms, which can cover both prefill and decode (most KV cache compression methods just cover prefill as a one-shot compression technique). The rate-distortion analysis gives in-depth insights on the accuracy trade-off space. Finally, evaluation numbers are good: 4.38 times compression with minimal accuracy drop.

### Weaknesses
This paper has three critical problems. It is very difficult to draw any scientific, grounded conclusions without resolving them: 

1. The proposed method is in its essence a "lossy" approach of KV cache sharing, since similar (not identical!) blocks are fused. This would inevitably lead to accuracy degradation, which could be negligible or severe depending on the particular inference scenario. Furthermore, this could cause issues like positional encoding misalignment, etc., which is a critical issue being widely discussed in the community when it comes to lossy or partial KV cache reuse. The authors need to conduct some analysis or stress test to see when their "lossy" approach of sharing could cause consistency problems. 

2. The evaluation section has essentially no proper baselines. Though the authors claim multiples times that their approach "outperforms the baseline", the only baseline result appears to be "no KV cache compression" results, as told from Table 5 and 6. Rigorous evaluation of a KV cache compression paper should include proper baselines featuring other methods in KV cache compression, including but not limited to methods in quantization (KIVI), token dropping (H2O), layer merging, etc. If you compare to "no compression" baseline, it is very difficult to convince the readers that your proposed method is worth adopting in the research space. 

3. The authors emphasize terms like "scalable serving" and "memory-efficient, high-throughput serving" multiple times throughout the paper, but the evaluation section does not have any end-to-end numbers on important system metrics related to LLM serving, including but not limited to time-to-first-token, throughput (token/s), time-between-tokens, GPU memory footprint, etc. 

Other minor issues:

1. Typos, like "KV=Cache" in Figure 1 caption.

### Questions
Please refer to "weaknesses".

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Summary：
Improving the performance of LLM inference systems has attracted significant attention from both academia and industry. Batched inference has become a de facto technique adopted by state-of-the-art open-source LLM inference frameworks, such as vLLM and SGlang, to enhance system throughput. However, batching alone does not alleviate the memory-bound nature of the decoding phase, underscoring the potential benefits of introducing KV cache compression methods. 

The most relevant prior work to this paper, exact prefix matching, enables KV cache sharing by identifying identical token prefixes across requests. While this is a lossless method, its compression ratio is inherently limited by the strict prefix matching requirement. This work addresses this limitation by relaxing the constraint and introducing a similarity-based, lossy KV cache sharing mechanism. A central focus of the design is to balance the trade-off between compression ratio and model accuracy, which is thoroughly evaluated across various benchmarks and model sizes.

### Strengths
Strength：
This paper presents a novel lossy KV cache compression method that effectively generalizes previous lossless approaches based on exact prefix matching. Given that the LLM decoding phase is intrinsically memory-bound, reducing the effective KV cache size per token—assuming the compression overhead remains negligible—represents a promising direction for fundamentally enhancing LLM inference performance. 

The experimental evaluation focuses on the fundamental rate-distortion trade-off between compression ratio and model accuracy. Results demonstrate that the proposed method achieves an F1 score that is comparable to, and in some cases even surpasses, the baseline at certain compression ratios.

### Weaknesses
Weakness:
As articulated in the paper’s introduction and motivation, the ultimate objective is to improve the performance of LLM inference systems. One major concern, however, is the absence of experimental results demonstrating that the proposed KV cache joint-encoding scheme actually leads to improved system performance—in terms of throughput and/or latency. Moreover, the paper does not sufficiently justify that such performance gains can be reliably expected.
The compression algorithm itself introduces non-negligible overhead: it iterates over N chunks or requests per layer, requiring additional computation and memory accesses. This process may lead to resource contention—both in computational units and memory bandwidth—between the compression algorithm and the ongoing LLM inference tasks, potentially offsetting any theoretical gains from KV cache compression.

### Questions
Questions:
What is the end-to-end performance impact of enabling the proposed KV cache compression? A comprehensive evaluation comparing the enabled and disabled states is needed, specifically reporting the throughput (token/sec), Time-To-First-Token (TTFT), and Time-Between-Tokens (TBT).
The paper states that the compression ratio of the proposed method is lower-bounded by that of the exact prefix method. In a scenario where both methods achieve an identical compression ratio, which one would deliver superior overall system performance? A comparison factoring in the computational and latency overhead of each approach would be highly informative.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper focuses on the problem that the KV cache becomes a memory and bandwidth bottleneck during the decode phase in multi-tenant high-concurrency LLM serving scenarios. Unlike methods that only perform quantization/low-rank/eviction on single-request KV, the authors propose a joint encoding method for KV blocks across requests and across chunks under a paged-attention layout. This method reuses similar blocks without altering the native KV tensor structure, thereby reducing the overall KV cache usage and allowing larger decode batches, it also reduce network bandwidth for KV migration in disaggregated architectures.

### Strengths
The system's predictions are very accurate.

The method's concept is simple but insightful.

The layout-preserving design is a big plus.

There is some theoretical support rather than it being purely heuristic.

The experimental results are convincing in direction.

### Weaknesses
The presentation of actual system benefits is insufficient. 
The paper emphasizes scalable, high-throughput serving, but the key results are mostly offline F1 + CR curves.
There are no clear end-to-end metrics, such as tokens/s, QPS, P95 latency, or the reduction in cross-machine KV migration traffic on actual vLLM / DistServe / Mooncake-like clusters. The paper even mentions that the current evaluation setup does not directly reflect acceleration effects, which somewhat mismatches the title.

The analysis of the numerical and semantic impacts of fusion operations is relatively shallow. Fast Fusion essentially involves averaging the K/V of multiple similar blocks into one direction and each block's scale, which changes the attention energy distribution, similar to mixing together multiple contexts. Currently, evaluation mainly relies on F1 / overall accuracy, without detailed analysis of output quality/drift for individual requests and discussion on whether extreme cases might be incorrectly fused.

Safety is not discussed at all, but the issue actually exists. In a multi-tenant environment, cross-request KV merging means cross-user state mixing, which may lead to prompt leakage and membership inference risks. Compliance concerns. There is already work in the literature showing that KV sharing can lead to leaks, yet this paper does not mention it at all.

Poisson / independence assumption is somewhat idealized. Rate-distortion analysis is based on similarity approximately Gaussian; Exceedance is sparse and can be modeled as Poisson; Samples are sufficiently independent; In practice, KV blocks are highly correlated, and the distribution changes with workload.

The entire method heavily relies on the similarity threshold: If the threshold is too low → aggressive reduction but accuracy may suffer; If too high → almost no fusion. Currently, only a few sweep curves are used to show that there is a reasonable range, but there is no clear tuning procedure and discussion on robustness across different models/tasks.

### Questions
In a real vLLM / DistServe style service framework, how much does BFF/CFF improve end-to-end throughput and latency? Can you provide at least a single-machine experiment?

In multi-tenant/multi-user scenarios, how should we view the risk of information leakage caused by cross-request block fusion? Is it possible to limit fusion within the same tenant/same security domain?

How transferable are the thresholds of Fast Fusion across different models/datasets? Is there an automatic or adaptive threshold solution?
How robust is it against similarity misjudgment? Have any noticeable error cases been observed?

Can it be combined with methods like prefix sharing / LeanKV / CacheGen? If combined, is it possible to achieve additive compression rates?

### Soundness
2

### Presentation
2

### Contribution
3
