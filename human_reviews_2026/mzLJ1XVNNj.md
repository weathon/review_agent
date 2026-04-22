# PatternKV: Flattening KV Representation Expands Quantization Headroom

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
KV cache in autoregressive LLMs eliminates redundant recomputation but has emerged as the dominant memory and bandwidth bottleneck during inference, notably with long contexts and test-time scaling.
KV quantization is a key lever for reducing cache cost, but accuracy drops sharply as the native KV distribution lacks flatness and thus maintains a wide quantization range. 
Prior work focuses on isolating outliers, which caps their error but fails to flatten the overall distribution, leaving performance fragile under low-bit settings.
In this work, we show that the K cache maintains a stable structure that evolves gradually with context, while the V cache carries latent semantic regularities. Building on these insights, we propose **PatternKV**, a pattern-aligned residual quantization scheme. It mines representative pattern vectors online, aligns each KV vector to its nearest pattern, and quantizes only the residual. This reshaping of the KV distribution flattens the quantization target and narrows its range, thereby improving the fidelity of low-bit KV quantization.
Across long-context and test-time scaling settings on multiple backbones, PatternKV delivers consistent 2-bit gains, with a 0.08\% average 4-bit drop relative to FP16, improves test-time scaling accuracy by 10\% on average, and raises throughput by 1.4× while supporting 1.25× larger batches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes the distribution of the KV cache across layers and input tokens. Based on these insights, it presents PatternKV, a method that mines cluster centroid vectors to decompose the KV cache, making it more amenable to quantization. To validate its efficacy, the authors conduct evaluations on a range of long-context tasks using popular LLMs. Benchmark results demonstrate that the method achieves strong accuracy at 4-bit precision and consistent gains at 2-bit. Additional inference experiments demonstrate improved throughput and memory efficiency.

### Strengths
- The numerical analysis of the KV cache distribution is insightful. The theoretical foundation, based on variance decomposition, effectively motivates and drives the design of PatternKV.
- The accuray experimental evaluation is comprehensive, demonstrating the method's strong performance across a variety of scenarios, models, and benchmarks.

### Weaknesses
- The performance analysis of throughput and memory footprint is conducted on only a single model (Llama-3.1-8B) under a specific context length. To strengthen the claims, it would be beneficial to extend these efficiency tests to more models and a wider range of scenarios, such as longer input (e.g., 16k, 32k tokens).
- The paper lacks a detailed analysis of the computational overhead introduced by the online pattern mining, matching, and update processes. Quantifying this overhead would provide a more complete picture of the method's efficiency.

### Questions
- How would this method be applied, or how would it perform, in models that use Multi-Head Latent Attention (MLA) or other non-standard attention architectures?
- What is the memory overhead associated with storing the pattern vectors, and how does this scale with the number of layers, attention heads, and the chosen pattern set size?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes PatternKV, a pattern-aligned residual quantization scheme: (1) mining representative pattern vectors online, (2) aligning each KV vector to its nearest pattern, and (3) quantizing only the residual. The first 2 steps (flattening KV representation) benefits the third step (quantization) by narrowing the quantization range; in other words, the fidelity of low-bit KV quantization can be improved.

### Strengths
1. PatternKV, based on (1) mining representative pattern vectors online, (2) aligning each KV vector to its nearest pattern, and (3) quantizing only the residual, differs fundamentally from prior work focusing on outlier mitigation. Outlier mitigation suffers possible performance loss with low-bit quantization, while PatternKV is okay (mostly outperforms) with low-bit quantization.
2. 1.4x throughput increase seems promising (if the baseline setting is fair -- please also see "Questions" below).

### Weaknesses
1. The overheads for clustering, pattern selection, and Chebyshev updates may be significant and should be discussed.
2. V pattern utilization rate may affect the efficacy of PatternKV -- this is discussed in Section 3.3 but not thoroughly (and experimentally) analyzed.
3. Lack of direct comparisons of speed (latency or throughout) and memory consumption against related works. Comparing to the FP16 case is indirect and does not seem fair.
4. The font size of figures, especially Figure 1 - Figure 3, is way to small. I can barely see the words.

### Questions
My questions and suggestions are basically from "Weaknesses" as aforementioned.
1. From Weakness 1: Please discuss the overheads for clustering, pattern selection, and Chebyshev updates.
2. From Weakness 2: Please thoroughly and experimentally analyze the effect of V pattern utilization rate on PatternKV's efficacy.
3. From Weakness 3: Please use KIVI (or representative counterpart) with INT2/INT4 quantization as the baseline(s) for more fair comparisons of throughput and memory consumption. Comparing to the FP16 case is indirect and does not seem fair.

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
The paper proposes PatternKV, an online KV cache quantization method that first mines a small set of pattern vectors (via K-means during prefill; Chebyshev-center updates during decode), aligns each KV vector to its nearest pattern using a custom min-max distance and then quantizes only the residual. 

The stated motivation is that residualization around prototypical patterns flattens the KV distribution, narrowing its quantization range and improving low-bit fidelity. The authors also add a one-sided z-test based adaptive gate for values (mid-layers) to decide when flattening helps (`Eq. 10`).

Empirically, the paper reports modest gains over online baselines on LongBench at INT2, very small differences at INT4 (claimed average drop 0.08% vs FP16), and some improvement under long CoT test-time scaling.

### Strengths
- Online and plug-and-play design. No calibration data or fine-tuning is required.
- The analysis and theory sound good. e.g., Principled v-gate, even if approximate, the one-sided z-test provides a crisp, checkable inequality (Eq. 10) that decides when to skip flattening.

### Weaknesses
- Code is not provided (lack of reproducibility).
- The most common data type used in inference is bf16, but the baseline in the paper is FP16.
- The experimental setup needs more clarification, e.g., what dataset/prompt is used in Figs. 2 and 3?
- In Sec. F, why is the residual size different? Doesn’t it hurt your comparison?
- The figures are not readable (especially Figs. 1 and 5).

### Questions
- Please address the mentioned items in Weaknesses.
- What do you infer from Figure 3 and Figure 8? 
- In `Sec. 4.4`, throughput/memory profiling compares to FP16 only. Please provide a component breakdown (KV I/O, pattern selection over |M|, Chebyshev updates, quant/dequant) and head-to-head throughput/peak-memory vs. KIVI/ZipCache/OTT/SKVQ under identical hardware/batch/sequence settings.
- Since you select patterns after RoPE and attribute K-drift to RoPE, please report a minimal ALiBi/no-RoPE experiment or a reasoned analysis, also explain why INT2 gains shrink with larger models if observed.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces PatternKV, a lightweight and training-free method for compressing the key–value (KV) cache in large language models. Instead of directly quantizing raw KV vectors, PatternKV discovers a small set of representative pattern vectors online and aligns each new KV vector to its nearest pattern before quantizing the residual. This alignment reshapes the overall KV distribution into a flatter form with reduced variance, enabling more effective low-bit quantization. The approach builds on two key observations: (1) the key cache exhibits a stable internal structure that evolves smoothly with context, and (2) the value cache carries latent semantic regularities that can be clustered. PatternKV maintains these patterns dynamically during inference using Chebyshev-center updates to track gradual distribution shifts, and employs a statistical gate to ensure that flattening is applied only when it provably reduces quantization error.

While the method is conceptually simple and empirically effective, the reported memory reduction appears modest compared to other quantization approaches. It remains unclear how much of the potential gain is offset by the additional storage of pattern vectors and related bookkeeping overhead.

### Strengths
* The paper is well written and clearly structured, making it easy to follow and understand.

* The evaluation includes challenging mathematical and reasoning benchmarks, and the proposed method even achieves improved accuracy on some of these tasks.

* The method itself is simple yet demonstrates consistently superior results compared to prior KV-cache quantization approaches.

### Weaknesses
* The method requires maintaining a separate set of centroid (pattern) vectors for each attention head, which introduces additional memory and computational overhead. However, the paper does not report the memory footprint of storing these centroids, nor does it quantify the latency incurred by pattern mining, nearest-pattern search, and residual computation. These costs could become non-trivial as the context length grows.

* The memory reduction reported in Figure 6 appears modest compared to standard quantization methods, suggesting limited practical savings.

* The paper offers little discussion on how to implement or optimize the system for efficient inference, despite the additional components required by PatternKV.

### Questions
* In Figure 5, what quantization bit-width is used for the reported results?

* What is the additional memory overhead introduced by storing the pattern (cluster) vectors? From Figure 6, the overall memory reduction appears modest compared to standard quantization; could you clarify how much of this is due to the extra pattern storage?

* What is the latency overhead of PatternKV during inference? How does its throughput compare with other online KV-cache quantization methods such as KIVI or ZipCache?

* What is the computational cost of the clustering procedures—specifically, the K-means clustering during prefill and the Chebyshev-center updates during decoding?

* How does the input sequence length influence the inference latency or throughput scaling of PatternKV?

### Soundness
3

### Presentation
4

### Contribution
2
