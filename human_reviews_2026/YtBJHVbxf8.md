# HEX: Merging Heavy-Hitters and Expanders for Adaptive KV Cache Optimization in Long-Context Inference

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 4, 4, 0

## Abstract
Key–Value (KV) caching accelerates large-language model inference but grows linearly with sequence length, quickly exhausting GPU memory. Existing compression strategies such as quantization, pruning, or sparsification shrink this footprint, but often degrade performance. Most pruning methods discard crucial connections and disrupt information flow, while dynamic heuristics often lack theoretical basis. We propose HEX, a cache compression strategy that is both structurally efficient and adaptive. HEX constructs a sparse backbone using expander graphs with spectral guarantees on connectivity, and augments it with heavy-hitter and recent tokens to capture input-specific context. The selected entries are stored in full precision, while the remaining cache is quantized to retain information at low cost. The expander masks are precomputed and static, thus significantly reducing computational overhead and aiding sparse implementations. 
Experiments on GSM8k, CoQA, TruthfulQA, and LongBench across models of varying sizes show that HEX consistently outperforms existing methods at higher compression rates without retraining. These results illustrate how principled eviction layouts grounded in graph structure and input dynamics can yield stronger accuracy–efficiency trade-offs for long-context inference even for limited cache budgets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper addresses the KV cache memory bottleneck in long-context LLM inference by proposing HEX, a hybrid cache compression method that merges expander-based structured sparsity with H2O and KIVI.  The motivation lies in balancing theoretical connectivity guarantees with adaptive token retention to achieve accurate cache compression. HEX statically constructs d-regular expander graph masks to preserve connectivity among token and channel dimensions, then augments them with H2O, while quantizing the remainder using coarse-grained asymmetric KCVT quantization (KIVI).
Experiments on various benchmark settings and models  demonstrate the performance of HEX. However, all evaluations are performed on older model families, and latency/throughput analysis is missing.

### Strengths
1. addresses KV cache optimization, a key bottleneck in long-context inference.
2. comprehensive experiments on multiple reasoning, QA, and long-context benchmarks show consistent gains over baselines.
3. well-designed ablations demonstrate the necessity and synergy of each component.

### Weaknesses
1. This paper focuses on accuracy and memory savings but does not report real end-to-end inference speed, which is critical for KV cache work. (key point)
2. The method is only evaluated on benchmarks with short-output scenarios. It would be valuable to test its effectiveness in long-output settings, such as AIME.
3. It conceptually overlaps with GEAR and KIVI, differing mainly in the use of expander-based structural masking. It provides incremental novelty relative small.

### Questions
see in weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes HEX, a hybrid KV cache compression method for long-context LLM inference. The core idea is to combine a static, sparse backbone with a dynamic, input-aware selection policy. The static backbone is constructed using Bipartite Expander graphs, which theoretically guarantee connectivity and information flow. The dynamic policy uses H2O to identify heavy-hitters and recent tokens. Entries selected by either method are stored in full FP16 precision, while all remaining entries are compressed using KCVT quantization.

### Strengths
The primary contribution is the novel application of Bipartite Expander graphs for KV cache pruning. This introduces a theoretically-grounded, structured sparsity approach, moving beyond common heuristics or magnitude-based pruning.

### Weaknesses
- The paper's central narrative is that the Bipartite Expander is the key innovation that solves the information flow problem. However, the paper's own ablation study in Table 4 (3-bit, GSM8k) seems to largely refute this claim. (1)Full HEX (Expander + H2O): 55.42% accuracy. (2) H2O only: 53.82% accuracy. (3) Expander only: 34.95% accuracy. This data strongly suggests that the H2O component is responsible for the vast majority of the performance, while the expander's actual contribution is marginal at best. 

- The most significant flaw is the complete absence of any inference speed measurements. The paper only reports on accuracy and compression ratios. For a paper on inference optimization, this is a fatal omission.
(1) This method relies on KCVT (KIVI) for compressing the vast majority of the cache. It is well-established that such low-bit quantization methods suffer from extremely high quantization and dequantization (quant/dequant) overhead. This overhead often negates any memory savings, leading to higher latency and lower throughput than the FP16 baseline. The authors must provide evidence that HEX is not subject to this same critical flaw.

(2) Sparsity Acceleration: The authors claim the structured sparsity is attractive for hardware, but provide no proof. Hardware support for N:M sparsity is extremely limited (e.g., NVIDIA 2:4 support). There is no evidence that the d-regular pattern generated by an expander graph can be accelerated on current hardware.

(3) On-the-fly Generation Cost: The paper states that if a mask is not cached, it is "generated on the fly." This generation of a complex graph structure would introduce significant computational overhead, directly increasing latency and lowering throughput. This cost is not analyzed or measured.

The authors' promise to "incorporate [latency results] in the revised version" is not acceptable. Performance data is not a minor addition; it is the central proof required to validate the method's practicality. It must be included in the initial submission.

- Figure 1 is confusing and poorly explained. The orange (FP16) and green (quantized) matrices are depicted as square matrices. This is misleading, as the KV cache is an $n \times d$ (sequence length $\times$ dimension) matrix, which is almost never square. The diagram should be revised to clearly represent what these matrices symbolize.

- Minor: Typo in line 200: "addding" should be "adding".

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes HEX, a hybrid cache compression framework for LLM inference that combines expander graph-based structured sparsity with dynamic heavy-hitter token selection (H2O). The method aims to reduce KV-cache memory footprint without retraining or losing accuracy. HEX first precomputes expander-based masks that guarantee strong spectral connectivity, ensuring information flow under sparsity. It then dynamically augments the static mask with influential and recent tokens to adapt to input-specific importance. The remaining cache is quantized using coarse-grained asymmetric quantization. Experiments on GSM8k, CoQA, TruthfulQA, and LongBench show that HEX achieves competitive or better accuracy than FP16 and state-of-the-art baselines like GEAR and KIVI under high compression ratios.

### Strengths
1.	The combination of expander-based sparsity (with spectral guarantees) and dynamic token preservation is innovative and theoretically motivated. This bridges the gap between static graph-based connectivity and input-aware token selection.
2.	HEX consistently outperforms baselines under aggressive compression (3–4 bits) on reasoning and long-context benchmarks, sometimes even exceeding FP16 accuracy. The ablation studies convincingly demonstrate the complementary effects of the expander and H2O components.
3.	The use of precomputed expander graphs gives provable structural properties and supports regular sparsity patterns favorable for hardware acceleration, addressing an important practical concern in LLM inference.

### Weaknesses
1.	Although memory savings are well documented, the paper does not report actual inference speed or wall-clock efficiency. Since KV compression aims to reduce both memory and latency, this omission weakens the empirical validation.
2.	While spectral guarantees are discussed, the paper lacks quantitative analysis of how expander degree, sparsity, or spectral gap correlate with accuracy or efficiency. The method may be sensitive to these hyperparameters in practice.
3.	The individual components (expander sparsity, heavy-hitter selection, coarse quantization) have each been explored in prior works. The main contribution lies in their integration, which might be viewed as a well-engineered hybrid rather than a fundamentally new algorithmic principle.

### Questions
1.	A schematic diagram showing how HEX integrates static and dynamic components during streaming inference would improve clarity.
2.	The evaluation spans multiple model sizes and datasets, which is commendable. It would help to include standard deviations or confidence intervals to assess the robustness of the reported gains.
3.	The discussion of spectral guarantees is solid but could be deepened by providing quantitative comparisons (e.g., empirical spectral gaps) or intuition on how these properties translate to robustness in attention computation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
Table 1, 3 & 5, which presents key experimental results, appears to be placed in a way that causes the manuscript to exceed the page limit for the main paper body. This may represent a violation of the conference's submission guidelines.

### Strengths
N/A

### Weaknesses
N/A

### Questions
N/A

### Soundness
1

### Presentation
1

### Contribution
1
