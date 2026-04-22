# CaliDrop: KV Cache Compression with Query-based Calibration

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
Large Language Models (LLMs) require substantial computational resources during generation. While the Key-Value (KV) cache significantly accelerates this process by storing attention intermediates, its memory footprint grows linearly with sequence length, batch size, and model size, creating a bottleneck in long-context scenarios. Various KV cache compression techniques, including token eviction, quantization, and low-rank projection, have been proposed to mitigate this bottleneck, often complementing each other.
This paper focuses on enhancing token eviction strategies.
Token eviction leverages the observation that the attention patterns are often sparse, allowing for the removal of less critical KV entries to save memory. However, this reduction usually comes at the cost of notable accuracy degradation, particularly under high compression ratios. 
To address this issue, we propose CaliDrop, a novel strategy that enhances token eviction through calibration. Our preliminary experiments show that queries at nearby positions exhibit high similarity. Building on this observation, CaliDrop performs speculative calibration on the discarded tokens to mitigate the accuracy loss caused by token eviction.
Extensive experiments demonstrate that CaliDrop significantly improves the accuracy of existing token eviction methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces an incremental calibration strategy for token-wise KV cache eviction. It leverages the attention outputs of previous, similar queries to calibrate the attention results derived from the evicted KV cache, while selectively recomputing when necessary. This approach effectively reduces the error introduced by token eviction and improves the accuracy of existing token-wise KV cache compression methods.

### Strengths
* **Novel Insight and Theoretical Foundation.** This work identifies that queries at nearby positions produce similar attention outputs, enabling the use of historical queries and attention results to calibrate future outputs under a pruned KV Cache. It further provides a formalized attention decomposition theorem and proof, offering a solid theoretical grounding for the proposed calibration mechanism.

* **Generalizable and Plug-and-Play.** The proposed idea is simple yet versatile, and can be seamlessly integrated into many existing token-wise KV Cache eviction methods—including StreamingLLM, H2O, and SnapKV, serving as a training-free add-on that consistently enhances accuracy.

### Weaknesses
* **Lack of Ablation on Calibration vs. Recomputation.** Besides the proposed calibration mechanism, the implementation also performs recomputation approximately every eight decoding steps using the evicted and offloaded KV caches, which contributes notably to the accuracy improvement. However, the paper does not separately analyze the impact of calibration and recomputation on attention error and end-to-end accuracy. An ablation study is needed to isolate the effect of the calibration mechanism itself and validate its claimed contribution.

*  **Limited Performance and Efficiency Analysis.** CaliDrop’s behavior depends on semantic query similarity, dynamically deciding whether to recompute, calibrate, or skip each step. Yet, the performance evaluation is conducted only on fixed 1024-token inputs and 128-token outputs, without reporting the ratio or cost of these different branches. The study should include more diverse, real-world prompts (e.g., ShareGPT [1]) and provide a breakdown of latency metrics such as TTFT and TPOT, to better quantify the runtime overhead of recomputation and calibration during both prefill and decoding phases.

[1] shareAI. (2023). shareGPT-Chinese-English-90k Bilingual Human-Machine QA Dataset. Hugging Face Repository. Retrieved from https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k

### Questions
* Regarding the proposed CaliDrop as a token-wise eviction method, recent studies such as RazorAttention [2], DuoAttention [3], and HeadKV [4] have explored head-wise KV cache compression. It would be valuable to discuss whether the proposed calibration mechanism can adapt to head-wise eviction, and how the implementation overhead might increase as the eviction granularity becomes finer.



[2] Tang, Hanlin, et al. "Razorattention: Efficient kv cache compression through retrieval heads." *arXiv preprint arXiv:2407.15891* (2024).

[3] Xiao, Guangxuan, et al. "Duoattention: Efficient long-context llm inference with retrieval and streaming heads." *arXiv preprint arXiv:2410.10819* (2024).

[4] Fu, Yu, et al. "Not all heads matter: A head-level kv cache compression method with integrated retrieval and reasoning." *arXiv preprint arXiv:2410.19258* (2024).

### Soundness
3

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
3

### Summary
The paper introduces CaliDrop, a query-based calibration strategy for KV cache compression in large language models (LLMs). Existing token eviction methods reduce memory but harm accuracy under high compression. CaliDrop leverages the high similarity between nearby queries to estimate the contribution of evicted tokens through calibrated attention recomputation, thereby recovering lost accuracy. Experiments on LongBench, RULER, and Needle-in-a-Haystack benchmarks show consistent accuracy gains across Mistral-7B and LLaMA-3 models, with minimal throughput cost. Overall, the work presents a simple yet effective improvement for token eviction–based cache compression.

### Strengths
•	CaliDrop introduces a query-level calibration that compensates for evicted tokens using nearby historical queries, supported by the “attention decomposition theorem” (Eq. 1–2) and L1-loss reduction evidence in Fig. 1 (Sec. 3.1.3).
•	Experiments cover multiple models (Mistral-7B, LLaMA-3-8B/70B) and benchmarks (Tables 1–2), showing consistent performance gains across KV sizes 64–512 (Secs. 4.2–5.1). This breadth supports the robustness and generality of the method.
•	The calibration mechanism requires only lightweight recomputation (every ≈8 steps; Fig. 3b) and maintains comparable throughput to SnapKV (Fig. 3a), highlighting its applicability in long-context inference.

### Weaknesses
•	CaliDrop is applied only in the prefilling phase (Sec. 4.1.2); no evidence is provided for dynamic or streaming decoding. Memory overhead from offloaded KV caches and detailed latency breakdowns are also missing (Secs. 5.2–5.3).
•	Beyond the exploration of $\theta_{1}$ and $\theta_{2}$ (Table 2), the paper provides limited investigation into other critical factors such as calibration size, per-layer contribution, or offload-cache management. Moreover, the absence of statistical validation (e.g., variance or significance testing) makes it difficult to assess the robustness of the reported improvements.
•	The approach extends existing token-eviction techniques through a supplementary calibration step but does not establish a new compression framework. The theoretical analysis mainly reiterates standard properties of attention mechanisms (Secs. 3.1–3.2) without offering new learning formulations.

### Questions
1.	Could the authors provide per-layer or per-head ablations to analyze where calibration contributes most across transformer depth?
2.	How does CaliDrop perform in real-time or streaming decoding settings, where query similarity varies more rapidly?
3.	What is the quantitative GPU-memory overhead of storing offloaded KV caches for calibration at different sequence lengths?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper identifies the two fundamental limitations in the existing KV cache compression methods: 1) discarding tokens that can become crucial later, and 2) the accumulated effect of discarding tokens is overlooked. To this end, the paper proposes CaliDrop, which compensates for evicted tokens by recomputing attention outputs for queries at nearby positions, alleviating memory pressure while maintaining model accuracy. The experimental results show that CaliDrop can be applied to different KV cache compression methods and improve their performance while introducing little computation overhead.

### Strengths
The method is well motivated, and the paper is well written.
The two observations are interesting: queries at nearby positions are similar, and the historical attention outputs can be used to predict future attention outputs.
The experiments on different KV cache compression methods and models across various tasks demonstrate the effectiveness of the proposed method.
The analysis of throughput and recomputation frequency shows the efficiency of the proposed method.

### Weaknesses
The hyperparameters $\theta_1$ and $\theta_2$ require manual tuning and may have different optimal values in different tasks.
The recomputation introduces a memory peak. What are the possible impacts of it, e.g., what is the maximum length of context/evict-KV CaliDrop can handle?
The recomputation cost and frequency may increase in larger models. It would be better to include the throughput and recomputation frequency in larger models.

### Questions
see in weaknesses

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a technique that complements KV cache eviction techniques and improves their ability to retain information from evicted KV entries. The method consists in offloading the evicted KV cache, and using a historical query that helps trigger a calibration step where the attention output is adjusted based on the evicted cache.
The authors conduct extensive experiments on the LongBench and RULER datasets for several models of 7-8B and 70B parameters. They also explore threshold parameter choices and discuss the impact of their method on latency and accuracy.
Overall, this paper copes with the important question of LLM efficiency and tackles the problem of lost information in token eviction methods.

### Strengths
- The proposed method is straightforward and sensible. It is well-presented and the long-context experiments are well-designed. It is also sound from a theoretical point of view.
- The method can complement any KV cache eviction method and shows noticeable benefits for every tested method (Streaming-LLM, H2O, SnapKV).
- The method is not very sensitive to threshold choices according to Table 2, which could have been a concern in such threshold-based methods.

### Weaknesses
I have concerns about the practical efficiency of the method that are not addressed in the current state of the paper. I also think that the experiments do not cover more extreme use cases where KV cache compression is crucial.
- **Doubts about practical efficiency**: The CaliDrop method relies on offloading and reloading past KV entries. It is not clear if the KV entries are offloaded to disk (in which case the computational overhead may be heavy for long sequences) or to CPU (in which case the CPU RAM can be saturated for long sequences).  These questions and the overhead they imply are not handled properly in Section 5, which only reports the throughput in relatively short-sequence setups (1024 with 128 KV cache budget), and shows how latency scales with batch size. It would have been more relevant to see the effect of sequence lengths and compression ratios on both memory (offloaded and on GPU) usage and latency. My main concerns are 1- that the latency gains may decrease with longer sequences as more KV items need to be offloaded, reloaded and the corresponding attention map needs to be computed for each recomputation step; 2 - that the VRAM can be saturated earlier than with the raw compression methods because the recomputation steps are dependent in the total sequence length. It would be insightful to at least report latency and memory usage statistics in the benchmark evaluations to show the overhead that is traded for better performance with CaliDrop.
- **Lack of long-context experiments**: In its current state, the paper lacks a discussion of the evolution performance gains when increasing sequence length. The NIAH results in Figure 2 are only conducted with an 8K context length when similar experiments are usually conducted with 32k to 128k context lengths. A study of perplexity evolution for long sequences similar to what is done in Devoto et. al could also be relevant.

### Questions
- What is the impact of offloading on RAM usage and how does it scale as sequence length increases?
- Did you try your method with metrics other than cosine similarity for query comparison? Is the query taken before or after positional encoding?
- The direct role of \theta_1 on latency is not exposed in experiments. What is the empirical impact of \theta_1 on memory usage and latency?

### Soundness
2

### Presentation
3

### Contribution
3
