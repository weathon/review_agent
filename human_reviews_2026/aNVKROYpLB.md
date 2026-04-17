# KV Cache Transform Coding for Compact Storage in LLM Inference

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Serving large language models (LLMs) at scale necessitates efficient key-value (KV) cache management. KV caches can be reused across conversation turns via shared-prefix prompts that are common in iterative code editing and chat. However, stale caches consume scarce GPU memory, require offloading, or force recomputation. We present KVTC, a lightweight transform coder that compresses KV caches for compact on-GPU and off-GPU storage. Drawing on classical media compression, KVTC combines PCA-based feature decorrelation, adaptive quantization, and entropy coding. It requires only a brief initial calibration and leaves model parameters unchanged. By exploiting redundancies in KV caches, KVTC achieves up to 20x compression while maintaining reasoning and long-context accuracy, and 40x or higher for specific use cases. We test KVTC with Llama 3, Mistral NeMo, and R1-Qwen 2.5 models across benchmarks
including AIME25, GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper and RULER.
It consistently outperforms inference-time baselines such as token eviction, quantization, and SVD-based methods, while achieving higher compression ratios. These results support KVTC as a practical building block for memory-efficient LLM serving with reusable KV caches.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces KVTC, a lightweight method for compressing KV caches in LLM inference. Inspired by classical transform coding, it combines PCA-based feature decorrelation, adaptive quantization via dynamic programming, and entropy coding. KVTC achieves up to 20× compression with negligible accuracy loss and scales across models such as Llama 3.1, Mistral-NeMo, and Qwen 2.5. Experiments on multiple benchmarks show that it outperforms prior quantization and SVD-based baselines while reducing inference latency.

### Strengths
1. KVTC effectively reduces memory footprint by up to 20x with minimal degradation in model accuracy (<1 point), demonstrating strong practical value.

2. The combination of PCA-based transform coding, adaptive quantization via dynamic programming, and entropy coding is well-grounded in classical signal compression theory, yet thoughtfully adapted to the KV-cache structure in modern Transformers.

3. Evaluation spans multiple model families (Llama 3.1, Mistral-NeMo, Qwen 2.5) and a wide range of tasks (GSM8K, MMLU, Qasper, AIME25, MATH500, etc.), showing strong and consistent results.

### Weaknesses
1) The practicality of kvtc hinges critically on whether SVD components computed on calibration data can be effectively transferred to out-of-domain inference contexts. While the authors provide some analysis in Appendix C.6 on the effect of calibration data domain, I remain concerned about the potential error introduced when there is a significant distributional shift between calibration and actual input sequences. For instance, in Figure 8, the relative reconstruction error for "Value Open R1 to FineWeb" is notably higher than "Value Open R1 to Open R1".

2) Table 4 reports the TTFT comparison between recomputation and decompression, which is informative. However, it would be valuable to also understand the latency in a more typical scenario where the KV cache is already present in GPU HBM (i.e., a cache hit). In such cases, what is the TTFT overhead introduced by kvtc’s decompression pipeline?

3) It would be helpful to clarify the key technical distinctions of kvtc compared to prior SVD-based KV cache compression methods such as xKV in Introduction.

4) xKV leverages inter-layer redundancy, whereas kvtc concatenates all attention heads across layers before compression. Given that Figure 2 shows block-wise cosine similarity patterns in value states rather than uniform cross-layer similarity, the rationale for global concatenation, rather than structured, layer-aware compression, needs further justification.

5) Figure 3 appears largely illustrative and less directly tied to the core technical contributions of the paper.

6) Across Tables 2 and 3, accuracy does not consistently decrease with higher compression ratios, in some cases, KVTC even outperforms the vanilla models. This counter-intuitive result is not analyzed.

7) Baselines are compared using each method’s best reported configuration, which is reasonable to ensure strong baseline performance.However, this also makes it difficult to assess performance at equivalent compression ratios, as accuracy may not scale linearly with compression strength.

8) Figure 8 demonstrates that cross-domain evaluation leads to larger reconstruction errors, yet no mechanism for adaptive or domain-agnostic calibration is provided.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes kvtc, a transform-coding pipeline for KV-cache compression that borrows from classic media codecs: PCA-based decorrelation, dynamic-programming bit-allocation for adaptive quantization, and Deflate entropy coding, with a one-time calibration and no weight changes. It targets the prefill↔decode boundary and multi-turn reuse, also handling practicalities like removing RoPE, skipping a recent-token window, and excluding early sink tokens. Empirically, kvtc attains ~16–20× (and up to ~40×) compression with accuracy within ~1 point of vanilla across GSM8K, MMLU, Qasper, RULER/VT, AIME/LCB, and MATH500, while reducing TTFT by up to 8× and outperforming or matching KIVI/GEAR (quantization), H2O/TOVA (eviction), and xKV/SVDq baselines in most settings.

### Strengths
This paper is well written and easy to follow. The paper’s main strengths lie in its practicality, simplicity, and effectiveness. It introduces a lightweight, system-friendly transform-coding approach (kvtc) that achieves up to 20× compression of KV caches with minimal accuracy loss and no model modifications.

### Weaknesses
The proposed method leverages the traditional transform-coding framework, achieving high compression ratios with minimal performance degradation. However, its limitation appears to be decompression latency. How does the decompression time compare with other methods listed in Table 2? In Table 4, the authors report only the compression and decompression times of the proposed approach, which seems insufficient for a fair comparison.

### Questions
1. In Figure 1, the authors state that $T$ denotes the time dimension. Could the authors clarify what the “time dimension” specifically refers to in this context? Does it correspond to the token dimension or another notion of temporal ordering within the sequence? 

2. The authors propose using a calibration set to derive a generalizable transformation matrix $V$. However, an important question remains: how are the calibration samples selected to ensure representativeness across diverse input distributions? Moreover, is the calibration process sensitive to the choice or domain of these samples, and how does performance vary under distribution shift? In comparison to SVD-based methods that compute a separate decomposition for each prompt, how much performance degradation does this one-time calibration introduce? 

3. Some results in Table 2 are confusing. The proposed method occasionally outperforms the vanilla model, even under compression. Could the authors provide an explanation or analysis for this unexpected improvement?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a method called KV Transform Coding (KVTC), which compresses the KV cache using a transform-coding pipeline inspired by classical image compression. The proposed approach does not modify model parameters or architectures and can be integrated seamlessly into existing inference systems. Experiments demonstrate compression up to 20$\times$ with negligible accuracy loss across multiple models and benchmarks.

### Strengths
1. The KV cache is a key bottleneck for efficient LLM inference. The paper identifies and effectively tackles this real-world challenge.
2. KVTC uses well-established techniques (PCA, quantization, entropy coding) in a novel context. It requires no retraining and can be plugged into existing frameworks.
3. The authors test on diverse LLMs (Llama 3.1, Mistral-NeMo, R1-Distilled Qwen2.5) and datasets (MMLU, GSM8K, RULER, MATH500, etc.). The paper reports improvements in both latency and memory usage.

### Weaknesses
1. **Limited algorithmic novelty**:  Transform coding with PCA+quantization is classical, and several SVD/quant methods exist (e.g., SVDq, xKV). The core components are classical in signal processing. The main contribution is an effective adaptation of known techniques, rather than a fundamentally new algorithm.

2. **Dependence on calibration data**: The PCA basis and bit allocation depend on a representative calibration dataset. When model structure changes, recalibration is required. There is limited end-to-end serving results (e.g., cross-node throughput, bandwidth savings, hit-rate improvements, multi-tenant scenarios).

3. **Unfair latency comparison setup**: The reported "8× faster TTFT" compares KVTC against a full recomputation baseline, which is not representative of how production LLM systems operate. A fairer comparison should include: 1) recomputing only the KV cache for new tokens 2) other competitive caching strategies.

4. The experimental protocol seems not align with the primary stated use case (compressing stale caches for storage between conversation turns). This mismatch, coupled with latency benchmarks from a "simple implementation" rather than a high-performance serving framework, makes it difficult to evaluate the true system-level overhead and viability of kvtc in its intended deployment.

5. Some comparison choices could bias CR: CR computed only on compressed tokens (excluding sliding window), while Deflate gains are content-dependent.  The link between Frobenius reconstruction error and downstream accuracy is only partially validated; no perplexity/CE analysis.  More ablations on calibration set composition/size and GQA specifics would help.

6. RoPE “undo” and reapplication details, and cross-layer/head concatenation vs. blockwise streaming claims, could use clearer algorithmic specification.  

7. The assumption of a single, global low-rank subspace shared by all concatenated layers and heads is a strong and maybe potentially suboptimal design choice. The paper lacks a crucial ablation study to justify this global approach over more granular alternatives (e.g., per-layer or layer-block PCA), which might yield a better rate-distortion trade-off.

8. SVD-related scheme and the bit allocation applied directly to KV Cache, inherently minimizes the **cache reconstruction loss**($\|\mathbf{K} - \widehat{\mathbf{K}}\|_F^2$), which is Q-agnostic. However, the actual objective is minimizing the **attention output loss**, where Q acts as a dynamic projection (and maybe or maybe not can be approximated by the calibration process). Considering that the theoretical derivation and implementation may be more difficult after the introduction of Q, a brief discussion will make the paper more rigorous.

### Questions
- How exactly are RoPE rotations inverted and reapplied for keys during compression/decompression in mixed prefill/decode settings?  
- Is the projection matrix V block-structured per layer/head to enable true layer-by-layer decompression, or is selective multiplication by submatrices used? Any accuracy/cost trade-offs?  
- Can you report end-to-end CR including the uncompressed sliding window to better reflect practical storage savings?  
- Which nvCOMP codec is used (Deflate vs. GDeflate/LZ4/Zstd)? How sensitive are results to codec choice and content?  
- How robust is a single global V across domains? Any cross-domain generalization or per-domain calibration vs. global calibration ablation?  
- What is the memory/latency overhead of storing and applying V, group scales/shifts, and metadata for large models (e.g., 70B) under pipeline/tensor parallelism?  
- Can you provide perplexity/per-token CE vs. CR curves, and a stronger correlation study between reconstruction error and task accuracy?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces kvtc, a lightweight transform coder designed to compress KV cache for efficient storage in LLM inference. Drawing on classical media compression techniques, kvtc integrates PCA-based feature decorrelation, adaptive quantization, and entropy coding, requiring only brief initial calibration without modifying model parameters . It achieves up to 20× compression ratio on KV cache of models like Llama 3.1, Mistral NeMo, and Qwen 2.5 R1, while maintaining reasoning and long-context accuracy across benchmarks such as GSM8K, MMLU, and LiveCodeBench.

### Strengths
+ The paper presents an interesting approach by using a transform-coding framework for KV cache compression. Experiments demonstrate that this method can maintain model performance even with a high compression rate.

+ The ablation experiments are very thorough.

### Weaknesses
+ The paper uses a limited number of evaluation benchmarks. Recent related works typically conduct comprehensive experiments on benchmarks like LongBench and RULER, whereas this paper only uses one dataset from each (Qasper and VT).

+ The "Related Work" section should also include content related to transform coding, as it is central to the proposed approach in this paper.

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
3
