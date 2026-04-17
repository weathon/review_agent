# Measure Once, Mask Once: Delta Refined Block Sparse Attention

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Long context inference poses a problem for large language models (LLMs) due to the high cost of quadratic attention with long input lengths. Efficient long context inference is a necessity in order to provide low-cost, low-latency LLM serving endpoints. Sparse attention is one way to mitigate the high cost of long context prefills. Many recent state-of-the-art sparse attention methods can be applied on top of pretrained quadratic transformers without any specific finetuning regimen, however, the main obstacle to overcome when designing sparse attention method lies in deciding which parts to compute and which parts to ignore during the sparse computation. Previous works generally make this decision based on heuristics derived from recurring patterns in the attention matrix or pooled block statistics to select a key-sparse attention mask. We show that these methods result in a suboptimal capture of total attention score mass. In another line of work, key-sparse attention has been shown to induce a distributional shift in attention outputs that can be mitigated by mixing query-sparse attention with existing key-sparse attention masks and combining the outputs. In order to save computation, we propose fusing the query-sparse attention and sparse attention mask generation process, resulting in a novel, dynamic, and query-dependent sparse mask generation. Our method calculates a key-sparse block mask while computing query-sparse attention, and then uses this dynamic attention mask to perform key-sparse attention before combining the outputs. Our method delivers a 2.5x speedup over Flash Attention 3 at 1M tokens and results in a total attention capture which is within 1.5\% of the oracle block top-k attention.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Measure Once, Mask Once (MOMO), a method for efficient long-context inference in large language models. MOMO builds on Delta Attention by fusing query-sparse attention (QSA) and sparse mask generation into a single pass. During QSA, the method collects block-level attention statistics via an online top-k algorithm to determine which key blocks to retain in the subsequent key-sparse attention (KSA) step. The fused process aims to eliminate redundant computation from Delta Attention, which previously required a separate mask-selection stage. The authors implement several kernel-level online top-k algorithms (exact, tournament-tree, and approximate) and demonstrate up to 2.5× speedup over FlashAttention-3 at 1M-token contexts

### Strengths
1. Introduce and solidly implement three online top-k query sparse attention block selection algorithm.
2. The acceleration gain is good.

### Weaknesses
1. The proposed method primarily reuses the Delta Attention pipeline (query-sparse + key-sparse + delta correction). The only new component is the reuse of QSA statistics to select key blocks, i.e., performing mask generation during QSA rather than afterward. This represents a minor engineering optimization rather than a substantive conceptual advance in attention modeling or theory. 
2. The paper does not provide theoretical justification or formal analysis showing why the online top-k selection during QSA approximates the oracle mask or preserves attention mass. From the evaluation, the original delta attention can also achieve a comparable accurate performance. It seems the major advance of this paper is to reduce the overhead by reusing the QSA statistics to generate mask.
3. Lack discussion or approaches to determine how to combine or set the number of k and k_exact.
4. Critical hyperparameters (γ for query skip size, χ for block size) are not explored. These are likely to affect both performance and accuracy, yet are fixed without justification.
5. Detailed latency breakdown is helpful for understanding the overhead and source of acceleration. 
6. Algorithm 1 is listed but not discussed.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a dynamic block sparse attention method called MOMO (Measure Once, Mask Once). Building upon the recent "Delta Attention" framework, this method combines key sparse attention and query sparse attention to mitigate the distribution shift caused by switching between sparse and dense attention, thus addressing the inefficiency of standard Delta Attention.

MOMO solves this problem through "measure once, mask once":
1. Merge measurement and mask: During the QSA step, an efficient online top-k algorithm is deployed simultaneously, retaining the indices of the K highest-scoring key blocks seen by the current query row.

2. Block Mask Union and Trimming: A single, efficient block sparse attention mask is obtained, which will be used for all queries.

3. Compute Block Sparse Attention (KSA): A fast but distribution-biased sparse attention output is obtained.

4. Apply Delta correction.

### Strengths
1. The computational redundancy in the Delta Attention was identified. Integrating the mask generation process with the QSA step is a very clever design that extracts high-quality dynamic sparse masks "freely" from a necessary computational step.
2. The paper reports a 2.5x speedup on 1M tokens (relative to FA3). This is a very significant engineering achievement and also garnered a good evaluation score.
3. The paper designs and compares three online top-k algorithms with different time complexities, and analyzes their latency and recall under different k values, demonstrating a very thorough investigation of top-k algorithms.

### Weaknesses
1. The proposed method is complex. This is significantly more complex to implement and maintain than a single sparse attention method or a dense attention method.
2. This method introduces a large number of hyperparameters that require tuning. Although the paper ablates some parameters, it lacks a comprehensive analysis of the sensitivity of these parameters and how they interact.
3. Regarding the Block Mask Union and Top-k Trimming sections, the description could be clearer, providing a complete computational process within a specified region.

### Questions
1. How much would change in accuracy and latency if Delta correction were removed? This would help differentiate the benefits of MOMO from the Delta Attention framework itself.
2. Could you please explain in detail the implementation details of step 2 (Union and trim) in Algorithm 1?
3. Equation 6 assumes that S(j) follows a Gaussian distribution. Is this assumption empirically valid or is there any relevant analysis?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Delta Refined Block Sparse Attention (DRBSA), a novel method to address the quadratic computational cost of self-attention during long-context inference (prefill) in Large Language Models (LLMs). DRBSA computes a static, block-level sparsity mask based solely on the pre-trained model's weight parameters, using a proposed metric, $\delta$. This "Measure Once, Mask Once" approach eliminates the need for expensive dynamic computation per-input sequence, significantly reducing FLOPs and achieving up to 4.9x speedup in long-context prefill latency while maintaining minimal perplexity degradation compared to the full model.

### Strengths
The mask is computed once offline, removing the runtime overhead common in dynamic sparse attention methods.

Can be applied directly to existing pre-trained quadratic attention models without any fine-tuning requirement.

### Weaknesses
Focus on Prefill: The primary benefit is demonstrated for the prefill stage; the method's value or applicability during the single-token decoding phase is not thoroughly explored.

### Questions
The method is highly effective for the prefill stage. How can DRBSA, or the general concept of a static, parameter-derived sparsity mask, be extended or adapted for the decoding stage, especially in the context of reasoning models (e.g., Chain-of-Thought) where selective attention to specific past tokens is increasingly critical for generating the next step?

Since modern GPUs (like NVIDIA Ampere or Hopper) are highly optimized for dense matrix multiplication, what is the actual observed overhead (in terms of clock cycles and memory stalls) of managing the irregular block-sparse computations compared to a highly optimized dense kernel like FlashAttention? 

Given the static nature of the mask, does DRBSA show any performance degradation when applied to contexts that exhibit very different attention patterns from the original pre-training data (e.g., complex code generation)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes MOMO, which fuses query sparsity and sparse mask generation within the same kernel. The authors claim that under million-token contexts, MOMO achieves about 2.5× speedup in TTFT relative to FA3.

### Strengths
The paper is well organized, and both the reported speedups and accuracy results are solid.

### Weaknesses
How does the method perform in terms of efficiency and accuracy for 4–32k context lengths? Since the reported results show no acceleration at 32k, what is the overall sparsity across different sequence lengths?

When comparing speedups, you should clearly specify what implementation you used and what the baseline implementation was, e.g., Triton, CUDA, or PyTorch.

Similar fused sparse-mask kernels have already been implemented in SeerAttention and NSA. What are the key differences between MOMO and those methods?

### Questions
see above weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
