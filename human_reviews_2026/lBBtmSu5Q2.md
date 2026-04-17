# On Fine-Grained I/O Complexity of Attention Backward Passes

- Decision: Reject
- Scores: 2, 4, 8, 6

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in processing long-context information. However, the quadratic complexity of attention computation with respect to sequence length poses significant computational challenges, and I/O aware algorithms have been proposed. This paper presents a comprehensive analysis of the I/O complexity for attention mechanisms, focusing on backward passes by categorizing them into small and large cache scenarios. Using the red-blue pebble game framework, we establish tight bounds on I/O complexity across all cache sizes. We confirm that the de facto standard I/O aware algorithm FlashAttention is optimal for both forward and backward passes for the large cache size scenario. For small cache sizes, we provide an algorithm that improves over existing methods and achieves tight bounds. Additionally, we extend our analysis to sparse attention, a mainstream speeding-up approach, deriving fine-grained lower bounds for both forward and backward passes and both small and large caches. Our findings complete the theoretical foundation for I/O complexity in attention mechanisms, offering insights for designing efficient algorithms of LLM training and inference.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors consider the I/O complexity of Attention gradient computation. In hardware, data is typically arranged hierarchically, with data stored in an unbounded memory, and computation occurring in a bounded cache. To compute, data is moved into the cache, computation occurs, and the result is saved in memory. Since data movement is typically more expensive that computation, I/O complexity measures only the data movements. The goal of I/O complexity is to design algorithms minimizing I/Os. Given the prevalence of attention and the success of the FlashAttention algorithm, it is a practically important question to understand whether the training process can be optimized w.r.t. I/O complexity.

The authors give I/O optimal bounds for the computation of attention gradient when restricted to algorithms using standard matrix multiplication. The authors also consider sparse attention, and give lower bounds for algorithms using standard matrix multiplication in this setting. 

While the statement of the main result is interesting, the techniques are identical to prior work, and in fact the main result can be obtained immediately from the lower bound for the forward pass. Furthermore, the lower bound on sparse attention is not well substantiated without a matching upper bound (or at least some improvement over the naive algorithm). Thus I recommend reject.

### Strengths
The authors study a practically interesting problem, and give tight results. 

They initiate the study of sparse I/O attention.

### Weaknesses
The main result (lower bound for attention gradient computation) is essentially immediate from prior work. In particular, a previous paper proves that any algorithm that computes the attention matrix already requires the FlashAttention lower bound. Since attention gradient computation involves a n x d and d x n matrix product, this immediately implies the desired lower bound. Similarly, the new upper bound for gradient computation in the small cache setting is a consequence of the equivalence with matrix multiplication (the easy direction - using matrix multiplication we can compute attention gradients).

The sparse attention lower bound is not well motivated if there is no matching upper bound, or at least some improvement on the trivial algorithm. Even if this is hard to prove, there should be some discussion towards what the obstacles are.

### Questions
What are the main obstacles towards designing I/O efficient algorithms for sparse attention?

### Soundness
4

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper extends the analysis of I/O complexity of exact attention appearing in [Dao, 2022] and [Saha & Ye, 2024], specifically providing tight bounds on the I/O complexity of the attention backwards pass using the red-blue pebble game framework. The results suggest that the popular FlashAttention algorithm is optimal in both forwards and backwards modes in the large cache regime (most practically relevant), while providing an improved algorithm in the small cache regime. The authors also extend the analysis to the sparse attention regime.

### Strengths
- The paper's derivations seem to be solid and rigorous, to the best of my understanding.
- The paper extends the results appearing in the previous work, thus completing the I/O complexity analysis for both forwards and backwards passes, small and large cache regimes, as well as dense and sparse attention.
- The paper is well-written and easy to follow.

### Weaknesses
Overall, the paper seems to be a direct extension of [Saha & Ye, 2024], adding tight bounds for the I/O complexity of attention backwards pass. However, the results seem to directly mirror the prior work; the authors utilise the same framework, and provide similar asymptotic bounds and conclusions. Due to this, my impression is that the work, although mathematically solid, seems to be incremental. The small-cache algorithm, as well as theoretical derivations seem to follow directly from [Saha & Ye, 2024], and from the practical perspective do not offer a significant contribution (as noted in the paper, the large- cache regime is more practically relevant, and FlashAttention is proven to be optimal). Due to this, my impression is that the scope of the paper is not quite sufficient for publication in ICLR.

### Questions
- Could the authors clarify how their small-cache algorithm differs/complements the similar proposition from [Saha & Ye, 2024]?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper analyzes the I/O (cache ↔ memory) complexity of the backward pass of exact softmax attention under standard GEMM, using the red–blue pebble framework. It proves *matching upper and lower bounds across all cache sizes*, with a phase transition at ($M = \Theta(d^2)$) ($M$ is the cache size and $d$ is attention head dimension). 

In the large-cache regime ($M=\Omega(d^2)$), the bounds match FlashAttention’s behavior and establish optimality; in the small-cache regime ($M=o(d^2)$), the paper gives a strictly better algorithm (and matching lower bound) than FlashAttention. It also gives lower bounds for sparse attention, recovering the dense case as a special case.

### Strengths
### Originality

* Provides the first matching upper and lower bounds for the backward pass of exact attention for all cache sizes with a clean phase transition at ($M=\Theta(d^2)$) (Theorem 1.1).  
* Extends to sparse attention with lower bounds that recover the dense case as a special instance. 

### Quality

* Uses the red–blue pebble framework rigorously and states Theorem 1.1 with an explicit formula covering both regimes.  
* Gives matching bounds in each regime: large-cache upper (Thm 4.1) and lower (Thm 4.2), small-cache upper via Algorithm 6 (Thm 4.3) and lower (Thm 4.4).    

### Clarity.

* Figure 1 clearly contrasts the paper’s tight bound (red) with FlashAttention’s upper bound (blue dashed) and marks the cross-point ($M=\Theta(d^2)$). 
* Theorems in §4 are presented as informal versions which helped readabillity.  

### Significance

* In the large-cache regime, results match FlashAttention and establish optimality; in the small-cache regime, Algorithm 6 is provably better than FlashAttention.

### Weaknesses
1. **Positioning vs prior work could be tighter.** The paper clearly cites Dao et al. (FlashAttention) and Saha & Ye for forward-pass tightness; it mentions Addanki et al. (streaming/approximate attention) in related work, but a compact comparison table clarifying different problem settings (exact vs approximate, streaming vs two-level memory) would help readers situate novelty. 

2. **Practical relevance narrative.** The paper *does* discuss when small-cache arises (e.g., per-SM caches on older GPUs) and even gives A100 vs GTX1060 examples; expanding this with a short table of device-level (M) estimates and typical head sizes (d) would strengthen the “why it matters” section.

### Questions
1. **Scope vs Addanki et al. (2023).** Please add a small table clarifying the differences (objective: exact vs approximate; model: two-level I/O vs streaming; bounds reported) and why your results are not directly comparable numerically. 

2. **Multi-head attention.** Your bounds are given per head; what changes (if any) under (H) heads computed in parallel. Does tiling across heads alter the asymptotics or only the constants?

3. **Device checklist.** Consider adding a table (SM/L1 size, datatype, typical ($d$)) for a few GPUs/edge devices to show where ($M \lessgtr d^2$) actually falls.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The original FlashAttention paper provides upper I/O complexity bounds for the backward pass of the exact attention computation, but does not provide lower bounds. This raises the question: what is the optimal I/O complexity of the attention backward pass? This paper provides a lower bound as a function of cache size. Interestingly, they show that the lower bound changes at a crossover point where the cache size if $o(d^2)$.

### Strengths
- The authors show that there is room at small cache sizes, to potentially provide a speedup over FlashAttention by reducing I/O complexity.
- The paper is pretty easy to follow and does quite a good job situating itself with respect to prior work.

### Weaknesses
- The authors do not provide an implementation of their algorithm, and so they cannot demonstrate that it actually provides a speedup over FlashAttention. The claim that the “algorithm designed for small cache sizes would become relevant and useful”, is speculative. In my view, this is the most significant limitation of this work.
- The result is only applicable for very small cache sizes, and does not apply to modern GPUs typically used for training (A100s, H100s, B200s).
- This paper (like prior work before it) assume a two-level memory hierarchy. This may limit the applicability of the results, especially since newer chips include more complex memory hierarchies including

### Questions
- Does Algorithm 6 increase the FLOPs required — even if only by a constant factor?
- Can the authors provide an implementation of their algorithm and demonstrate that it can provide a speed up on GPUs with small cache sizes?

### Soundness
3

### Presentation
3

### Contribution
1
