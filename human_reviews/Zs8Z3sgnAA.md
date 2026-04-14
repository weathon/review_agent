# Fine-grained Attention I/O Complexity: Comprehensive  Analysis for Backward Passes

- Decision: Reject
- Scores: 6, 5, 8

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in processing long-context information. However, the quadratic complexity of attention computation with respect to sequence length poses significant computational challenges, and I/O aware algorithms have been proposed.  
This paper presents a comprehensive analysis of the I/O complexity for attention mechanisms, focusing on backward passes by categorizing into small and large cache scenarios. 
Using the red-blue pebble game framework, we establish tight bounds on I/O complexity across all cache sizes. We confirm that the de facto standard I/O aware algorithm FlashAttention is optimal for both forward and backward passes for the large cache size scenario. For small cache sizes, we provide an algorithm that improves over existing methods and achieves the tight bounds. 
Additionally, we extend our analysis to sparse attention, a mainstream speeding-up approach, deriving fine-grained lower bounds for both forward and backward passes and both small and large caches. 
Our findings complete the theoretical foundation for I/O complexity in attention mechanisms, offering insights for designing efficient algorithms of LLM training and inference.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper comprehensively analyzes the I/O complexity associated with attention mechanisms in large language models. The paper identifies a critical point with respect to the cache size in which the I/O complexity changes and analyzes the I/O complexity upper bounds in both cases. The paper shows that in the small cache case, FlashAttention is not optimal and provides an algorithm that improves existing methods. And for the large cache case, the paper confirms the optimality of FlashAttention. The paper further explores the lower bound of sparse attention forward and backward passes. Overall, this paper provides insights for efficient LLM training and inference.

### Strengths
1. Comprehensive study of the I/O complexity for attention mechanism. The separation of cases by the cache size seems appropriate. 
2. The paper completes the theoretical I/O complexity in attention, which is a very important component in transformer architectures.

### Weaknesses
1. On modern GPUs such as H100 or A100, it is not clear if it will fall under the case of small cache sizes. With older generations of GPUs, the primary concern might not be the I/O complexity but other hardware constraints such as total GPU memory. 
2. The paper could have an implementation of their algorithm on attention with the small cache case and demonstrate that they can outperform FlashAttention. The paper mentions this as a future work but it is still less convincing. If possible, please provide some preliminary results for your algorithm.

### Questions
1. Under what hardware and model will the attention computation fall in different cache cases?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper provides a comprehensive analysis of the I/O complexity for attention mechanisms, focusing on dense backward and sparse forward/backward by categorizing them into small and large cache scenarios. The paper confirms that FlashAttention, the standard in industry, performs optimally for large cache sizes but is suboptimal in smaller cache environments. The paper introduces an improved algorithm for small caches, achieving tighter bounds on I/O complexity than existing solutions. Additionally, the paper extends this analysis to sparse attention mechanisms, establishing fine-grained lower bounds for both forward and backward passes. Such analysis could guide the design of efficient algorithms for LLM training and inference.

### Strengths
1. The paper is well structured. It clearly defines the problem, notations, and the proofs are clearly explained.
2. The paper provides a concrete algorithm with a computation graph that provides practical guidance for achieving optimality.

### Weaknesses
1. The sparse attention only considers the number of nonzero entries, which is inadequate for modeling sparse computation. Therefore, the bounds can potentially be further improved. Other factors like sparsity patterns and data format might also affect the I/O complexity bounds.
2. The dividing point of small and large caches for sparse attention is vaguely defined. L1830 says "It depends on whether all the necessary values for computing each output entry can be stored in the cache during the computation." Based on this definition, the cache is whether small or large also depends on the computation. A more precise mathematical definition might help.

### Questions
1. What are the practical metrics that distinguish small and large caches? In the paper's notation, the small cache size grows strictly slower than $d^2$. However, it is unclear that given a specific $d$, what size $M$ should the cache have to be counted as a small cache. Could you provide specific examples of small caches in current AI accelerators and discuss how they relate to typical values of $d$?
2. The paper claims that Algorithm 6 achieves the optimal for small cache sizes. Could you provide some practical evidence for this optimality? The comparison between the IO of Algorithm 6 and the IO of FlashAttention implemented in Triton running on a GPU with a small cache will improve the confidence of the claim.
3. The "slow memory" of a real GPU is an L2-Cache, not high-bandwidth memory (L052 "(e.g., GPU high-bandwidth memory)" is not accurate). Are the theorems of this paper still applicable to this setting? Could you clarify how your cache-memory maps onto real GPU architectures?

### Soundness
3

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
4

### Summary
In this paper the authors consider the problem of optimizing the I/O complexity of implementing attention mechanisms in LLMs. They consider both forward and backward passes and offer matching lower and upper bounds. All the analyses are done using the red-blue pebble game framework.

### Strengths
This is a solid theory paper that settles the I/O complexity for attention mechanisms. They fill some of the gaps that existed in the literature with regard to this I/O complexity. They confirm that FalshAttention, a prior algorithm, is optimal for both forward and backward passes for large cache sizes. For small cache sizes they fill a gap by presenting new algorithms that are optimal. The authors also offer lower bounds on the I/O complexity of sparse attention in the context of forward and backward passes and for both large and small cache sizes.

### Weaknesses
Some practical implementation results could enhance the paper.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3
