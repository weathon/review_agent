# xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Large Language Models (LLMs) with long context windows enable powerful applications but come at the cost of high memory consumption to store the key and value states (KV-Cache). Recent studies attempted to merge KV-Caches from multiple layers into shared representations, yet these approaches either require expensive pretraining or rely on per-token cosine similarity across layers, which may not always be observed in practice. We find that the dominant singular vectors are remarkably well-aligned across multiple layers of the KV-Cache. Exploiting this insight, we propose xKV, a post-training compression method that applies Singular Value Decomposition (SVD) on the KV-Cache of grouped layers. xKV consolidates the KV-Cache of multiple layers into a shared low-rank subspace, significantly reducing KV-Cache sizes. Through extensive evaluations on the RULER long-context benchmark with widely-used LLMs (e.g., Llama-3.1 and Qwen2.5), xKV achieves up to 8× KV-Cache compression rate while keeping the accuracy gap within 2–3 percentage points of the non-compressed baseline over a set of representative long-context tasks, and remains robust in multi-turn settings. Coupled with the designed Selective Reconstruction (SR) at decode time, xK-SR (keys only, values offloaded to CPU memory) yields 2.53% higher accuracy than the state-of-the-art system that combined token selection with single-layer SVD and delivers up to 3.23× end-to-end speedups over full attention on an A100 GPU. At a similar accuracy level, xKV-SR (keys and values on GPU) achieves up to 4.23× faster speedups. These results highlight xKV as a versatile, plug-and-play solution to alleviate both the memory footprint and accelerate inference in long-context LLM inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a KV caching mechanism that uses low-rank compression across layers in an LLM. The method does not require training and provides significant compression with relatively little performance degradation on long context tasks. Their observation on measuring the similarity with CKA is interesting and may serve as inspiration for future work. Overall, I think the algorithm is strong and well-motivated but I have some questions on generalizability.

### Strengths
1) Observation on cross-layer similarity is interesting and useful.
2) Sizable latency improvement (up to 4x) and reduction in memory (8x)
3) Inference-time algorithm that doesn't need training.
4) Paper is well written

### Weaknesses
1) While there are extensive experiments on long context tasks, it remains unclear how this method would perform for longer generation tasks like ones that use chain of thought. Since many models are now moving towards having robust long generation, it would be beneficial to evaluate on such scenarios.
2) It is unclear how this method would work for models that have different attention mechanisms interleaved (e.g., vanilla attention -> sliding window attention -> vanilla attention -> ...).
3) Little diversity in model size. It would be nice to see the effect on smaller or larger models (if compute is available) on a few tasks since different sized models can have different rates of compression.
4) For completeness, it would be good to include in your background low-rank + sparse compression methods that merge multiple tokens together into a low-rank component like RNNs e.g., [1,2]. 

[1] Nawrot, et al, Dynamic memory compression: Retrofitting llms for accelerated inference, 2024.

[2] Dong, et al, Get more with less: Synthesizing recurrence with kv cache compression for efficient llm inference, 2024.

### Questions
1) It makes sense for nearby layers to have strong CKA but in Fig 2b, there are also cases where an initial layer’s cache is similar to a much later layer’s caches but dissimilar to most of the closer layers’. Why is this?
2) Do you have throughput metrics, or is this method solely to optimize latency?
3) The SVD operations are fairly quick in comparison to the total prefill time. Why is this? Are you using randomized algorithms?

### Soundness
3

### Presentation
4

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
xKV compresses KV caches by grouping adjacent layers, performing a cross-layer SVD to build a shared basis, and optionally applying Selective Reconstruction (SR) to reduce decode-time reconstruction. Evaluations include RULER/LongBench and a multi-turn NIAH setup; latency plots report up to 4.3× attention-kernel speedups at 128K with xKV-SR.

### Strengths
It works with GQA and shows reasonable accuracy on RULER/LongBench.

### Weaknesses
**Q1.** Comparisons against several SOTA are missing, including  RocketKV,  KVzip,and  SeerAttention.  The authors also ignored some recent works, such as Lache, Lexicode.

**Q2.**  The core idea—leveraging inter-layer alignment for compression—overlaps conceptually with MiniCache (depth-wise KV merging) and ShadowKV (low-rank key + selective value reconstruction). Clarify what is fundamentally new beyond: (i) single-layer SVD, (ii) depth-wise merging, (iii) low-rank key + on-the-fly reconstruction. 


**Q3.**  Appendix C.1 reports SVD time fractions (e.g., 11.8% at 32K; 2.05% at 256K with G=4) and absolute SVD times up to 8.74s at 256K on A6000, but the truncation method (randSVD vs. power iteration) is unspecified. State the algorithm, flop/memory cost, and report prefill wall-clock with/without SVD and end-to-end latency.

**Q4.**  xKV appears focused on compressing the initial context; clarify how generated tokens are handled. If decode-time KV grows uncompressed, long generations can negate memory wins. Quantify memory composition over time and show stable compression under long decoding traces. 

**Q5.** Current results fix ranks (e.g., rK=384, rV=576) and stop group-size ablation at G=4. Provide full trade-off curves for rank vs. accuracy vs. effective memory reduction, and extend group-size beyond 4 (e.g., 8, 16) to identify saturation/instability points.

**Q6.** CKA indicates depth-dependent alignment; static contiguous grouping may be sub-optimal. Probably add an adaptive strategy (e.g., CKA-driven grouping) and compare to fixed G under the same compression budget. (Quantify gains, if any.)

**Q7**.  The paper doesn't provide accuracy at tighter budgets (e.g.,  128–256 token compressed contexts or small SR budgets), since recent head-/query-agnostic methods emphasize robustness at very small KV sizes.

### Questions
I mentioned in the Weakness section.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
xKV is a post-training KV-cache compressor that finds adjacent layers share highly aligned singular vectors, then runs a single cross-layer SVD to learn a shared low-rank token basis with layer-specific reconstructions; at decode, it uses Selective Reconstruction to rebuild only the needed tokens.

### Strengths
- The proposed xKV method is technically sound, with solid mathematical grounding (via SVD and CKA analysis), and extensive empirical evaluation across multiple models and benchmarks (RULER, LongBench, NIAH) supporting its claims.
- The paper is clearly written, with strong conceptual motivation, well-structured figures
- The paper introduces a cross-layer perspective on KV-cache compression, discovering that dominant singular vectors of adjacent layers are highly aligned

### Weaknesses
- the method relies on performing on-the-fly SVD during prefill, which, though amortized at long contexts, still adds non-trivial latency and may hinder deployment in low-latency or streaming settings
- the proposed approach is evaluated primarily on instruction-tuned models and long-context benchmarks; broader testing on reasoning-heavy or real-world interactive workloads is limited, leaving questions about generalization.
- The paper reports impressive attention-level speedups (up to 4.3×) but does not provide end-to-end inference latency improvements, which are what matter most in deployment. Without reporting full pipeline timing (including prefill, reconstruction, and I/O), it’s unclear how much wall-clock speedup users would actually observe.

### Questions
- Since the current study focuses on compressing the prefill context, how would xKV perform under long-generation scenarios where the KV-cache continues to grow? Could online incremental compression or periodic re-SVD be viable?
- What is the overall latency reduction per generated token compared to FlashAttention or ShadowKV?
- How much does the SVD or reconstruction overhead contribute to total runtime?
- Are the reported speedups measured only for the attention kernel, or for full inference throughput (prefill + decode)?

### Soundness
3

### Presentation
3

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
This paper addresses the memory and latency bottlenecks of KV cache in long-context reasoning, focusing on the dimension of cross-layer redundancy, and proposes a post-training pluggable method called xKV. 
Through CKA analysis, it is found that although the per-token cosine similarity of KV across adjacent layers is not high, their principal singular vectors are highly aligned between layers.
Horizontally concatenate multiple adjacent layer KVs, and perform an SVD on the concatenated matrix.
During reasoning, Selective Reconstruction is proposed: only a small number of tokens related to the current query are reconstructed at each step, avoiding the huge FLOPs of dense reconstruction.
On models such as Llama-3.1-8B and Qwen2.5-7B/14B, evaluations on long-context benchmarks like RULER, LongBench, and multi-turn NIAH show that accuracy drops can be kept within 2–3 points under approximately 8× compression; combined with SR, compared to FlashAttention-2, it can achieve up to around 4.3× attention speedup on 128k sequences, and outperforms representative methods such as MiniCache, Single SVD, SnapKV, and ShadowKV.

### Strengths
Clear new perspective: Cross-layer alignment of singular vectors
  
Method is clean and can be plugged in post-training  

Selective Reconstruction design is practical  

Experiments are fairly comprehensive and baselines are solid  

System-level perspective is prominent

### Weaknesses
The 'Aligned basis' argument is somewhat anecdotal and lacks deeper analysis.
The current experiments with CKA + rank curves are sufficient to motivate the method, but there is no discussion on whether this alignment is universal across more model architectures like MoE, encoder-decoder, multimodal; There is no visualization or statistical analysis between the bases, nor is there an explanation of which layers are more suitable for grouping.

There are still gaps in coverage compared to the latest extreme compression methods. Mainly compared with MiniCache, single-layer SVD, KIVI, SnapKV, ShadowKV, etc., which are already quite good; However, there is still a lack of direct comparison under the same configurations with methods such as SVDq, KVQuant, CSR, Lexico, etc.

The theory and guarantees are relatively weak. 
Most analyses remain at the level of empirical CKA & eigen decomposition + discussions on complexity scale; 
There is a lack of upper bounds or more formal explanations for the approximation errors introduced by selective reconstruction.

The application scenarios and multi-request settings are not explained thoroughly. The main focus is on the single-request case of long prefill + relatively short decode; For multi-tenant multi-request scenarios, how cross-layer SVD is reused when multiple streams run simultaneously, whether SR states are shared, it is not elaborated on.

### Questions
Has the CKA alignment and rank advantage been validated on more models (such as 70B scale, MoE, multimodal)?  Does it hold equally for higher layers / lower layers?

If the landmark selector used in SR is replaced (for example, using a simpler heuristic), to what extent can the advantage of xKV be maintained?

How does xKV perform when combined with extremely low-bit quantization (such as SVDq, KVQuant)? Does it still have an advantage under the same total compression ratio?

Is there a more automatic strategy for configuring group size, rather than a fixed G=4 + fixed rK, rV?

For the NIAH and LongBench results, could you provide some error cases, indicating in which tasks cross-layer sharing of the subspace fails?

### Soundness
2

### Presentation
2

### Contribution
2
