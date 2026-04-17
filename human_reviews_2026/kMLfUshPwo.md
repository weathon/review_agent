# TAKE: Task-Aware Chunked KV Cache Eviction for Efficient Long-Context LLM Prefill

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
The rapid development of large language models (LLMs) enhances various language generation applications, but it remains a serious memory usage challenge in long-context inference. Existing global pruning aims to reduce memory in the decoding process, ignoring the prefill peaks to delay the time-to-first-token. In this paper, we present Task-Aware Chunked KV Cache Eviction (TAKE), a training-free framework to optimize KV cache memory during the prefill stage of LLM inference.
TAKE partitions long sequences into chunks and incrementally performs task-aware KV fusion and eviction, thereby avoiding full-sequence processing and reducing memory and compute overhead.
To preserve task-relevant information, we introduce lightweight task-aware probe tokens to identify salient tokens within each chunk and accumulate semantic information across chunks.
Furthermore, we propose a delayed eviction strategy that protects shallow transformer layers from early pruning, mitigating representation degradation and improving performance stability.
Extensive experimental results show that TAKE achieves superior performance, reduces the peak GPU memory usage for the KV cache and activation to about 8.9\% of the baseline model, and lowers first-token latency by over 60\% for sequences up to 128k tokens.
It also enables stable inference with arbitrary length contexts on 24GB consumer GPUs without quantization or KV offloading, while maintaining model quality.
Our code is available at https://anonymous.4open.science/r/TAKE-6B21.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TAKE, a training-free framework designed to reduce memory usage and latency during the prefill stage of long-context LLM inference. The core idea is to process the input sequence in chunks, interleaving attention computation with a novel KV cache eviction strategy. TAKE utilizes two main components: an "accumulated task-aware probe" to identify and retain task-relevant tokens across chunks, and a "delayed eviction" strategy to protect shallow layers from premature information loss. Experiments show that TAKE significantly lowers peak memory usage and time-to-first-token (TTFT) while maintaining strong performance on long-context benchmarks.

### Strengths
- The paper is well-structured and easy to follow. The figures are effective at illustrating the core concepts of chunked processing and the overall pipeline, which helps in understanding the proposed method.
- The ablation studies are relatively comprehensive. The authors have made an effort to validate the effectiveness of the two main components of their method (the task-aware probe and delayed eviction), which provides valuable insights into why TAKE works.

### Weaknesses
- The proposed method appears to be a chunked version of an importance-based eviction method like SnapKV. For a more direct and convincing comparison, it would be beneficial to adapt SnapKV to a similar chunked processing framework. This would help isolate and verify the true effectiveness of the proposed probe mechanism over existing attention-based scoring functions in a chunked setting.

- The method's reliance on using only the probe to score keys seems to create a high dependency on the accuracy of probe selection.

- The paper lacks references to some relevant prior work on chunk-based processing for KV cache management. SARATHI[1] proposed the concept of chunked-prefill. OmniKV[2] also employs a chunking strategy to accelerate prefill and minimize the KV cache for H2O.

[1] SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

[2] OmniKV: Dynamic context selection for efficient long-context LLMs

### Questions
1.  In the ablation study (Table 2), for the "w/o probe" configuration, what scoring method was used to select which keys to evict? The paper states that this configuration performs poorly, but it is unclear what alternative scoring function was used in the absence of the probe. 

2.  The current probe selection strategy uses the last few tokens of the input, which works well for benchmarks like Needle-in-a-Haystack. However, for other tasks like style transfer, where instructions might appear at the beginning (e.g., `{Reference Style}{Instruction}{Article to Convert}`), would it be more robust to scan the context for explicit task instructions instead of relying on token position?

3.  For a complex multi-hop question where the probe itself contains multiple clauses and dependencies, how confident are the authors that this method can still effectively identify and retain all the necessary pieces of evidence scattered across different chunks?

4.  Could the analysis in Figure 7b be expanded along the "Warm-up Layers" dimension? It would be interesting to see a clearer trend of how performance changes as the number of warm-up layers increases or decreases for the LongBench tasks.

### Soundness
2

### Presentation
3

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
This paper proposes TAKE, a KV cache–centric optimization that performs KV cache eviction together with chunked prefill to reduce peak GPU memory usage during LLM inference. TAKE primarily introduces two mechanisms: accumulated task-aware probing and delayed eviction. The accumulated task-aware probe estimates the relevance between each KV pair and the task query by applying an exponential moving average (EMA) over the query (Q) vectors during chunked prefill. The probe queries are drawn from the trailing tokens of the entire input sequence, as these tokens typically contain task-related information. They are smoothed during chunked prefill because query tokens are often strongly biased toward the current chunk. The delayed eviction mechanism maintains a relaxed eviction budget for shallow layers, and reduces it to the target budget when processing the final chunk. The positions preserved in shallow layers are determined by the first layer that enables eviction. TAKE is evaluated on multiple benchmarks, including NIAH and LongBench, and compared with various baselines. Empirical results show that TAKE achieves nearly lossless performance while providing the fastest TTFT among all baselines.

### Strengths
1. This paper is well written and easy to follow. The description of the method and the two major contributions are clear. The benchmark results demonstrate the promising performance and efficiency of TAKE.

2. The design of TAKE is reasonable, and the ablation studies explain the effectiveness of the proposed mechanisms. Conducting KV cache eviction along with chunked prefill can significantly reduce peak memory usage without notably harming performance.

3. TAKE provides a promising method to host a long-context LLM in GPU-memory-constrained scenarios, e.g., consumer-grade devices. The contribution of this paper is sufficient.

### Weaknesses
1. Although combining chunked prefill with KV cache eviction is an efficient design, the novelty remains limited. There are many related papers that work in this area. For example, InfiniPot [1] and Locret [2] also integrate eviction with chunked prefill, and they both achieve good performance on downstream long-context tasks. These baselines should be compared with TAKE as the main baselines, as they address the same problem.

2. NIAH is a relatively simple long-context retrieval task, and the input sequence length of LongBench is limited (most entries are less than 40K). Therefore, a more challenging benchmark is expected to be tested. RULER [3] is an appropriate benchmark for evaluating TAKE. Since query-aware KV selection is introduced in TAKE, it should (or is expected to) achieve nearly lossless performance on RULER. Please include this experiment in your paper, and discuss the potential reasons if there is a large performance degradation.

3. Multi-turn conversation is also an important task for long-context processing. Since query-irrelevant tokens are evicted in the design of TAKE, is it able to process multi-turn conversations without re-prefilling the previous chat turns? A brief discussion should be included in the paper, and if possible, discuss the potential limitations of scenarios that TAKE cannot handle. Empirical results are also welcome to clarify this issue. Since multi-turn conversation is mentioned in the introduction section (line 37), such discussion is especially necessary.

4. One possible way to reduce peak memory is to run SnapKV and conduct layer-wise eviction. In other words, the model first prefills a certain layer, then performs eviction using SnapKV before executing the computation of the next layer. Such baseline methods should also be discussed. One advantage of TAKE is that it can process extremely long sequences, where even a single-layer KV cache can exceed GPU memory constraints—for example, conducting retrieval tasks at a 10M-token input scale. Discussing such scenarios can further demonstrate the superiority of TAKE compared with vanilla methods.


---

[1] InfiniPot: Infinite Context Processing on Memory-Constrained LLMs

[2] Locret: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads

[3] RULER: What's the Real Context Size of Your Long-Context Language Models?

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TAKE, a dynamic KV-cache Eviction during the chunked-prefilling process to reduce memory usage and TTFT (time-to-first-token) and enable 128K inference on 24GB GPUs. The idea is novel in chunk-wise KV eviction, while a similar idea has been deployed on decoding, including SnapKV and R-KV. The Training-Free design and will-design experiment on Llama3 and Mistral, while the missing reasoning model experiments of DeepSeek or Qwen are also a big question as to whether TAKE is also effective on the reasoning model.

### Strengths
Unlike global pruning methods that prune after full prefill, such as SnapKV, TAKE prunes during prefill to avoid high memory peaks. it would help for long-context prefill tasks for both throughput and memory usage.

It uses smart semantic preservation to retain necessary tokens while maintaining accuracy. The accuracy looks good, and it is important for real-world application that requires high accuracy.

We believe training-free is essential for effective KV Cache Eviction, which this paper achieves.

### Weaknesses
Although it shows novelty compared to chunked prefill, we have observed a similar idea on decoding and gradually pruning the KV cache during decoding. If we set a chunk size during decoding and raise a token pruning every chunk size, the idea can also be deployed on a reasoning model and decoding process, which would significantly improve the impact of the pruning as reasoning models like DeepSeek and Qwen are becoming more and more important now.

An essential concept of chunk prefill is the selection of the chunk size; different chunk sizes affect the prefill process's performance. And more importantly, it should also affect the performance of TAKE Task-Aware Chunked KV Cache Eviction. However, I didn't see any discussion of the chunk size in the paper (even worse, I can't see the chunk size number in the main paragraph; there's only one discussion in Appendix A about setting the chunk size Z = 4096). This is an important issue that changed my rating from 6 to 4. I would like to see the ABLATION STUDY on how the chunk size affects memory usage, prefill time, and the accuracy of TAKE.

LongBench can be updated to LongBench v2 for state-of-the-art experiments.

Typo:

line 53: remaining -> remain

line 367:  time-to-fist-token -> time-to-first-token

### Questions
what is the maximum context length TAKE can support on RTX-4090, 128K or higher?

### Soundness
2

### Presentation
2

### Contribution
3
