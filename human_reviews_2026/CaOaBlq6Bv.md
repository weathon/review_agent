# Towards Multiplier-Free Transformers with Stochastic Attention

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 4, 0

## Abstract
In standard attention, a substantial fraction of compute comes from multiplying softmax weights by high-precision value vectors — even in ternary models such as BitNet, which remove multipliers elsewhere. We present Stochastic Additive No-mulT Attention (SANTA), a drop-in inference-time replacement that eliminates these value-stage multiplications. For each query, SANTA samples from the post-softmax distribution, gathers and sums selected values, and applies a single bit-shift normalization, with no expensive multipliers on the value path. SANTA’s compute scales as $O(n_{queries} \cdot S \cdot d_k)$: linear in the number of queries during prefill and linear in the sample budget $S$ during decode, while exhibiting sparse, index-based memory access. SANTA is an unbiased Monte Carlo estimator of dense attention and is orthogonal to upstream efficiency techniques (ternary quantization, low-rank kernels, sparsity, pruning). Combined with existing 1-bit/ternary quantizers, SANTA moves Transformers toward fully multiplier-free, energy-efficient inference.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SANTA, an additive attention variant that serves as a drop-in replacement for the standard attention mechanism. SANTA approximates the attention output by sampling value vectors according to the post-softmax distribution and then computing their mean. The authors prove that this method provides an unbiased estimate and further propose S²ANTA, which leverages stratified sampling to reduce variance. In comparison to top-k attention, the authors demonstrate that SANTA is more power-efficient by completely eliminating multiplications in the value stage. The experiments show that SANTA achieves promising results, with performance comparable or even superior to the top-k method across various benchmarks.

### Strengths
1. This paper is well-written and easy to follow.
2. To the best of my knowledge, the method is novel.
3. SANTA effectively eliminates a key computational bottleneck while maintaining competitive model accuracy.
4. The finding that a simple, unbiased averaging of values can achieve performance comparable to the strong top-k baseline is both surprising and compelling.

### Weaknesses
1.  While the authors argue SANTA is more power-efficient by eliminating multiplications, they dismiss sampling costs as "lightweight relative to the V matrix multiply" (§3.4) without providing empirical measurements. The computational cost of sampling from categorical distributions over long sequences should be benchmarked against top-k's partial sorting overhead to substantiate the efficiency claims.
2. Although the authors claim the method is suitable for edge devices, the paper provides no actual hardware deployment experiments, nor does it discuss the implementation on edge devices.

### Questions
1. How does SANTA perform on actual edge hardware? While top-k can be fused with online softmax via heap-based algorithms with efficient rescaling, can SANTA similarly avoid full softmax I/O? Is sampling truly faster than partial sorting on resource-constrained devices?
2. What is the fundamental advantage of weighted Monte Carlo sampling over simply averaging the top-k values with uniform weights (also multiplication-free)? This ablation would clarify whether the stochastic framework provides benefits beyond just selecting a sparse subset.

Would raise my score if authors solve my questions.

### Soundness
2

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
3

### Summary
This paper proposes SANTA (Stochastic Additive No-mulT Attention), a novel attention mechanism that aims to eliminate multiplication operations from Transformer inference.
Instead of computing the standard attention product 
softmax(𝑄𝐾⊤)𝑉, SANTA samples keys from the attention distribution and approximates the output via Monte Carlo averaging. The resulting estimator 𝐴^𝑉A^V is shown to be unbiased, with variance decreasing proportionally to 1/𝑆, where 𝑆 is the number of samples.

An enhanced version, S²ANTA, applies stratified or systematic sampling to further reduce variance. Because the sampling process involves only additions, indexing, and bit shifts, the model achieves a completely multiplier-free inference pipeline when combined with low-bit quantized feed-forward layers (e.g., BitNet).

### Strengths
1. The stochastic sampling approach provides an unbiased, theoretically grounded estimator of the attention output.
2. Addresses a key limitation of efficient Transformers, the heavy reliance on multiplications in attention.

### Weaknesses
1. Lack of hardware validation: The claimed multiplier-free and energy-efficiency benefits are purely theoretical; no real measurements or FPGA/GPU latency/energy analysis are provided.
2. Potential implementation inefficiency: Random sampling and irregular memory access could make GPU execution slower than dense attention in practice.
3. Missing baselines: No comparison against kernelized linear attention or FlashAttention energy-profiling to substantiate the energy-efficiency claim.

### Questions
1. Can you provide any empirical runtime or energy measurements on real hardware (e.g., A100/H100 or FPGA) to substantiate the multiplier-free claim?
2. If the current hardware is not a good candidate for this algorithm, can the authors provide a clearer discussion on what type of hardware architecture would be suitable for such multiplier-free computation — for instance, what memory access patterns, parallelism model, or instruction primitives (e.g., stochastic sampling units or bit-level adders) would be required to efficiently support SANTA on future accelerators?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduced Stochastic Additive No-multiplication Attention (SANTA), a drop-in, inference-time, replacement for the standard attention layer. SANTA approximates the attention operator without performing any score-value multiplications, thus offering a multiplier-free replacement. This is achieved through a Monte-Carlo estimation by sampling from the attention softmax distribution, thus only using additions, as well as a bit-shift operation instead of division (given the number of samples is a power of two).

### Strengths
The paper is easy to follow and clearly introduces the problem and its proposed attention method.

- Figure 1 is clear and is good representation of the algorithm.
- The mathematical notation is clearly introduced and the propositions and derivations are easy to follow.
- The authors showcase a good performance of SANTA compared to the sparse top-$k$ attention.

### Weaknesses
One of the main motivations of the paper seems to be getting rid of multipliers for transformer inference, complimenting techniques such as BitNet (that removes multipliers in matrix multiplications), and NoMAD-Attention, that removes multiplications in the $QK^T$ part of attention. The impact of this is not fully clear however, especially as, like the authors note, current hardware is indeed optimised for fast matrix multiplications. Unlike BitNet, SANTA is applied at inference time, and its implications to training are not explored. Furthermore, the authors hint at potential synergy of the method with BitNet-like architectures in order to fully get rid of multiplications in the forward calculation, but this is not backed by any experimental data (i.e., it is not clear if the method would work well with these architectures) — a demonstration of a fully multiplication-free transformer would benefit this argument.

- The underlying sampling idea is not new, and has been used in previous work to approximate attention (e.g., https://arxiv.org/abs/2410.16179)
- The paper does not provide any practical measurements on effective speed-ups that could be achieved. Although the algorithm might benefit future hardware to a greater extent, the decrease in FLOPs and memory transfers should still offer a benefit (and indeed top-$k$ techniques are used for this). Having a practical analysis could give a stronger case for using SANTA-like replacement of attention during inference.

### Questions
- The authors note that the number of unique keys accessed can be significantly fewer than the number of samples; it would be useful to get a sense of the implication of this to a practical speed-up.
- Although authors mention that sampling/sorting costs can be ignored, it would be helpful to get some sense of their relative cost vs. the other operations within the module. This could be especially helpful in order to understand the advantage of SANTA vs. top-$k$- as a drop-in replacement of attention during inference.
- Minor: It would be useful to also have perplexity results on a standard dataset (such as WikiText).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
This paper sets out to remove the need for multiplication in attention networks by using what it calls stochastic attention.  The

### Strengths
The paper identifies a worthy goal: Reducing computational cost of neural networks that employ attention would be helpful.

### Weaknesses
While the introduction and related work sections (1 and 2) are readable, the main contribution section (3) was not in this reviewer's estimation actually human readable.  The paper says "LLMs were used to polish the presentation and writing of this contribution" on lines 843-844 but it's not clear what parts of the paper that applies to.  If this paper was written in a language other than English, then passed to an LLM to translate, I think the LLM did a poor job on the main technical part.

The results section appears to only compare against sparsity (top-k).  It is unclear why no other quantization approaches are compared against.

Regarding Remark 3.2 that if S = $2^m$ then division can be implemented as a bit shift, I am concerned what this paper is proposing in Equation 2 is to replace multiplication of A times B by summing B copies of A.   That hardly seems like the right way to achieve efficiency.

### Questions
Are you just proposing to implement multiplication by summing a bunch of copies of the multiplicand by the multiplier?  (That is what Equation 2 seems to be doing to me.)

I'm not sure what it means to "treat each row $A_q$ as a categorical distribution over keys and sample $S$ values i.i.d. from it".   What is $S$ and how is it set or selected?  What does it mean to sample a row of a matrix?  What is meant by categorical distribution?  How does any of this approximate a multiplication?

According to what distribution is the sampling of one-hot rows governed in the example in Equation 2?

For the "illustrative example" in Equation 2, what is the corresponding value of A?

It is unclear how $V_i$ is obtained in Equation 3.  What does it mean to stack rows from V in forming $V_i$?

What is "Categorical($A_q$)" on line 158?  What is $V_i_{q,s}$ in the equation on Lines 159-160?

### Soundness
1

### Presentation
1

### Contribution
1
