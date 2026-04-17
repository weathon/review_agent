# OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Large language models (LLMs) with extended context windows enable powerful applications but impose significant memory overhead, as caching all key–value (KV) states grows linearly with sequence length and batch size. Existing cache eviction methods address this by exploiting attention sparsity, yet they typically rank tokens heuristically using accumulated attention weights without considering their true impact on attention outputs. We propose Optimal Brain Cache (OBCache), a principled framework that formulates cache eviction as a layer-wise structured pruning problem. Building on Optimal Brain Damage (OBD) theory, OBCache quantifies token saliency by measuring the perturbation on attention outputs induced by pruning tokens, with closed-form scores derived for isolated keys, isolated values, and joint key–value pairs. Our scores account not only for attention weights but also for information from value states and attention outputs, thereby enhancing existing eviction strategies with output-aware signals. Experiments on LLaMA and Qwen models show that replacing the heuristic scores in existing works, which estimate token saliency across different query positions, with OBCache's output-aware scores consistently improves long-context accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces OBCache, which formulates the KV cache eviction process as a pruning problem. Through an analysis based on output perturbation, they enhance existing scoring metrics that rely on attention weights. By integrating OBCache, current methods can achieve higher accuracy by better determining which tokens to discard.

### Strengths
1. The paper is well-written and easy to follow.

### Weaknesses
1. The analysis based on output perturbation has been explored in prior work, both in KV cache eviction  [1] and merge [2]. Notably, [1] has already pointed out that attention weights alone are not a sufficient scoring metric in kv cache eviction. Furthermore, the empirical finding that the value norm serves as a good indicator has also been proposed [3].  A more detailed discussion is needed to differentiate the theoretical analysis and empirical results from these existing studies. 

[1] Feng, Yuan, et al. "Identify critical kv cache in llm inference from an output perturbation perspective." arXiv preprint arXiv:2502.03805 (2025).

[2] Tian, Yuxuan, et al. "KeepKV: Eliminating Output Perturbation in KV Cache Compression for Efficient LLMs Inference." arXiv preprint arXiv:2504.09936 (2025).

[3] Guo, Zhiyu, et ak. "Attention score is not all you need for token importance indicator in kv cache reduction: Value also matters." arXiv preprint arXiv:2406.12335 (2024).

2. The improvement brought by OBCache appears to be limited, especially on the SnapKV baseline. Given that SnapKV is the SOTA method compared to H2O and TOVA, it is unclear whether the gains offered are significant enough to advance the state of the art in this field.

3. The computational overhead introduced by the improved scoring metric itself is not evaluated. I suggest that the authors provide a detailed analysis of this overhead.

4. The paper's evaluation focuses heavily on high compression ratios where performance degradation is substantial. For practical applications, it would be more instructive to focus the evaluation on a realistic range of performance loss (e.g., less than 5%) to better guide real-world implementation.

5. The experimental setup described in the appendix contains an inconsistency in the perturbation window size. A fixed size of 16 is used for SnapKV and H2O on the NIAH tasks, whereas a dynamic size of 5% of the prompt length is used for LongBench. This raises questions about the sensitivity to this hyperparameter. It is important to clarify how this window size should be set in practical scenarios and justify the chosen settings.

### Questions
See weaknesses.

### Soundness
2

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
This paper proposes **OBCache**, a principled framework that formulates the KV-cache eviction task as a structured pruning problem inspired by the Optimal Brain Damage/Surgeon paradigm. Instead of relying on heuristic accumulation of attention weights, the method estimates each token’s true contribution by approximating its second-order effect on the loss function. Experiments across multiple LLMs and long-context benchmarks demonstrate strong empirical gains, with OBCache preserving model fidelity under tight memory budgets. The paper is clearly written, well-motivated, and presents a meaningful advance in cache management for long-context inference.

### Strengths
The key strength lies in the elegant adaptation of classic pruning theory to dynamic KV management. The proposed importance metric provides a theoretically grounded and interpretable measure of token salience, effectively bridging a gap between structured pruning and attention-based heuristics. Implementation is straightforward and modular, and the method delivers consistent improvements in perplexity and zero-shot accuracy across different architectures. The ablations and visualizations are thorough and clearly communicate the method’s behavior.

### Weaknesses
1. While the theoretical formulation is sound, the **practical efficiency** of OBCache remains unclear. The computation of second-order–inspired importance scores is nontrivial, yet the paper does not provide a quantitative breakdown of end-to-end inference latency or throughput compared to lighter heuristics such as H2O or StreamingLLM. A latency–quality trade-off analysis would strengthen the empirical credibility of the work.

2. Since the importance score relies on a local second-order approximation, its **long-term stability** over very long contexts (e.g., 100k+ tokens) is not well discussed. It would be valuable to analyze whether cumulative approximation error leads to score drift or suboptimal pruning in extended sequences.

3. The **layer-wise pruning strategy** raises a practical question: is there a global control mechanism (e.g., a unified sparsity target) that governs cache size across all layers, or must each layer’s threshold be tuned manually? Clarifying this would improve the method’s usability for large-scale deployment.

### Questions
1. Could the authors quantify the per-token latency or throughput overhead introduced by OBCache on standard hardware (e.g., A100/H100) relative to attention-based heuristics?

2. How stable are the OBCache importance scores over extremely long contexts? Have the authors observed cumulative error or bias toward earlier tokens?

3. Is there a simple way to enforce a global cache-budget constraint across layers, rather than tuning each layer’s threshold individually?

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
3

### Summary
This manuscript introduces OBCache, a novel framework for optimizing inference efficiency in large language models (LLMs) with long context windows through principled key–value (KV) cache eviction. The method formulates cache eviction as a structured pruning problem grounded in Optimal Brain Damage (OBD) theory. It computes closed-form token saliency scores—value-based, key-based, and joint key–value pairs—using second-order Taylor approximations of the perturbation in attention outputs due to pruning. This framework generalizes and improves upon existing attention-weight-based heuristics (e.g., H2O, TOVA, SnapKV). The paper provides both theoretical analysis and extensive empirical evaluation across long-context tasks (Needle-in-a-Haystack, LongBench, PG19), showing consistent improvements in performance when integrating OBCache into existing cache eviction strategies.

### Strengths
1. The idea to evict tokens by measuring the impact of keys, values and key-value pairs is novel. This approach better capture token importance compared to prior attention-only heuristics.
2. Experiments that cover a range of models (LLaMA-3.1, Qwen-2.5) and tasks (retrieval, QA, summarization, perplexity). Results clearly show consistent performance improvement under tight KV budgets.

### Weaknesses
1. The paper does not quantify the computational or memory overhead associated with computing perturbation-based saliency scores during inference. 
2. The quality of the perturbation approximation is not explicitly evaluated. It would strengthen the work to analyze the accuracy of the estimation framework (e.g., how well the derived scores correlate with the true eviction-induced error). Furthermore, quantifying the upper-bound performance achievable with oracle (ground-truth) saliency would clarify the potential headroom for improvement.
3. Experiments are reported only up to 32K context length. It remains uncertain how OBCache scales to ultra-long contexts, where accumulated approximation errors may degrade performance.

### Questions
1. What is the runtime and memory overhead introduced by computing OBCache saliency scores compared to purely attention-based heuristics such as H2O or TOVA?
2. How accurate is the proposed perturbation-based estimation of token importance? If the true eviction impact (oracle ground truth) were used instead, what would be the maximum achievable performance? In that case, would the joint key–value pruning strategy yield the best results?
3. How does OBCache perform at very long context lengths (e.g., 64K–128K tokens)? Does the perturbation-based approximation remain stable, or does accuracy deteriorate with context depth?
4. The perturbation window is manually chosen. How do we select the window size in real-world use cases? Could the model learn to adjust this window online during inference?

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
This paper presents Optimal Brain Cache, a theoretically grounded framework for KV cache eviction in LLMs, aiming to reduce memory and latency overheads in long-context inference. Unlike prior heuristic approaches that rely mainly on attention weights, OBCACHE formulates cache eviction as a structured pruning problem inspired by the Optimal Brain Damage theory. This formulation leads to output-aware saliency scores that integrate attention weights, value states, pre-softmax logits, and attention outputs to better reflect each token's contribution. Extensive experiments across LLaMA-3.1 and Qwen-2.5 models on long-context benchmarks, including Needle-in-a-Haystack, long-sequence perplexity, and LongBench, demonstrate that integrating OBCACHE consistently improves generation quality.

### Strengths
1. The paper proposes three optimization approaches for existing scoring methods based on the Optimal Brain Damage theory.
2. The method enhances the generation quality of existing cache eviction approaches in both ruler and longbench benchmark.

### Weaknesses
1. The theoretical presentation is poorly organized, with most details relegated to the appendix, which compromises readability. It is recommended to formalize the theoretical content using clear Theorems and Proofs to improve clarity.

2. This paper omits a closely related work [1], which similarly analyzes cache eviction from an output perturbation perspective. Notably, both papers adopt a similar objective of minimizing output perturbation and found the previous attention-weight-based methods as special cases under this formulation. A detailed discussion and comparison between the two methods would be beneficial.

3. The paper lacks an evaluation of the computational overhead introduced by the compression algorithm itself.

4. The reported improvements on real-world benchmark tasks appear marginal.

5. The paper does not provide sufficient analysis or comparison among the three variants-Value, Key, and Joint-in terms of their effectiveness and trade-offs.

### Questions
1. How does the proposed method support GQA?
2. Is the dynamic cache eviction compatible with FlashAttention? If so, how much decoding speedup does it achieve compared to standard FlashAttention?
3. Can the proposed approach be applied to enhance sparse attention methods such as Quest[2] or ShadowKV[3]?
4. In [1], the L1 norm is used to define perturbation theoretically, and experiments show little empirical difference between the L1 and L2 norms. It would be valuable to clarify the motivation for adopting the squared Frobenius norm in this paper and to discuss how it differs from the L1 norm in evaluating perturbations in practice.


Reference:
1. Identify Critical KV Cache in LLM Inference from an Output Perturbation Perspective
2. Quest: Query-aware sparsity for efficient long-context llm inference
3. Shadowkv: Kv cache in shadows for high-throughput long-context llm inference

### Soundness
3

### Presentation
2

### Contribution
2
