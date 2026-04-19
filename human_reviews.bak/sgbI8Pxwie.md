# Beyond Linear Approximations: A Novel Pruning Approach for Attention Matrix

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 5

## Abstract
Large Language Models (LLMs) have shown immense potential in enhancing various aspects of our daily lives, from conversational AI to search and AI assistants. However, their growing capabilities come at the cost of extremely large model sizes, making deployment on edge devices challenging due to memory and computational constraints. This paper introduces a novel approach to LLM weight pruning that directly optimizes for approximating the attention matrix, a core component of transformer architectures. Unlike existing methods that focus on linear approximations, our approach accounts for the non-linear nature of the Softmax attention mechanism. We provide theoretical guarantees for the convergence of our Gradient Descent-based optimization method to a near-optimal pruning mask solution. Our empirical results demonstrate the effectiveness of our non-linear pruning approach in maintaining model performance while significantly reducing computational costs, which is beyond the current state-of-the-art methods, i.e., SparseGPT and Wanda, by a large margin. This work establishes a new theoretical foundation for pruning algorithm design in LLMs, potentially paving the way for more efficient LLM inference on resource-constrained devices.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a method for unstructured pruning of the query and key projection matrices in attention. The paper provides theoretical guarantees for the convergence of the pruning method. It also provides some experiments on synthetic data demonstrating that the pruning method outperforms other pruning methods.

### Strengths
*Original problem.* The paper highlights an interesting, niche problem in the pruning literature, which, to my knowledge, has not been analyzed theoretically. 

*Clear presentation.* The paper is well-written and the structure is easy to follow. 

*Extensive theoretical analysis.* The authors provide an extensive theoretical analysis on the convergence of their pruning algorithm.

### Weaknesses
*Limited potential for speedup on large scale models.* I have two concerns around the potential for this method to provide significant speedups for large language models. (1) It is typically difficult to leverage fine-grained, unstructured sparsity for speedups on GPUs because standard vector and matrix instructions are not compatible with sparsity (though, recent changes to Nvidia GPUs could begin to mitigate this problem [Nvidia Blog](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)). (2) The technique is applicable to a small fraction of the parameters in the full model. In large models, the MLPs dominate the total parameter and FLOP counts, but this method is only applicable to the Q and K projections, which account for only 9.4% of parameters in Llama-405B (excluding embeddings, [source](https://huggingface.co/meta-llama/Llama-3.1-405B/tree/main)). So even if the method could achieve 100% sparsity, the relative reduction in FLOPs/memory consumption would be limited. 

*Experiments limited to synthetic data.* Because experiments are only performed on synthetic data and a single layer, it remains unclear how the performance of real language models would degrade with this method. Without experiments on real language models it is hard to assess the improvement of this method relative to prior work (Wanda and SparseGPT).

### Questions
How does the convergence rate in Theorem 4.1 compare to the rates one can achieve with naive baselines?

### Soundness
3

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
2

### Summary
The paper introduces a novel approach to pruning weights in large language models (LLMs) to improve computational efficiency, particularly for deployment on resource-constrained devices. This method focuses on optimizing the approximation of the attention matrix, a critical component in transformer architectures, by accounting for the non-linear properties of the Softmax attention mechanism. The authors present a Gradient Descent-based optimization technique, offering theoretical guarantees for its convergence toward a near-optimal pruning solution. Preliminary results indicate that this approach maintains model performance while significantly reducing computational costs. This work provides a new theoretical basis for designing pruning algorithms for LLMs, potentially enabling more efficient inference on devices with limited resources.

### Strengths
1. The presented technique enhances convergence guarantees for non-convex functions, crucial for optimization challenges in areas like deep learning, through the application of the Polyak-Łojasiewicz inequality. It showcases mathematical rigor with detailed derivations and validations of key conditions, supported by established theorems, notably Theorem 5.2 from Frei & Gu (2021). Practical implications include improved computational efficiency via attention weight pruning without notable performance trade-offs, assured by the method’s convergence mechanisms. Additionally, its framework offers potential applicability across various non-convex optimization scenarios, expanding its usefulness in the optimization toolkit.
2.  The experimental results suggest that the proposed algorithm outperforms existing methods under certain conditions, the above weaknesses highlight areas where the study could be improved.

### Weaknesses
1. The convergence guarantees depend on strict conditions like the boundedness and positive definiteness of matrices (e.g., $\( X^T X \succeq \beta I \))$. These conditions may not always hold in real-world applications, limiting the practical relevance of the results. Additionally, the analysis does not consider the stochasticity and noise common in real-world data and optimization, which could affect gradient descent’s performance.

2. The experimental section lacks clarity on whether the tested input sequence lengths cover the full range required for real-world corpus. Given that modern llm often deals with lengthy texts like entire documents or extended conversations, the study does not explore the algorithm’s performance in such scenarios, raising concerns about its practical utility.

### Questions
Conside the weakness

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
The paper proposes to prune the projection weight matrices of the Softmax attention conditioned on the similarity between the pruned and the original attention matrices. The proposed approach leverages gradient decent to solve the constraint opimitzation problem and has theoretical gurantees for near-optimality with polynormal time.

### Strengths
1. Considering the similarity between attention matrices during pruning of the projection weight is novel and interesting.
2. The assumptions in theoretical analyses are mild and sensible.

### Weaknesses
1. The applicability of the approach is limited. The proposed pruning approach needs to fuse the query and the key projection matrices into a single matrix through matrix multiplication. This means that it will eliminate the inference efficiency benefits of popular LLMs, such as Llama 3, which use Grouped Query Attention that shares the key-value pairs between queries. This is because after the proposed fushion, the original input X needs to be saved during decoding, while for GQA, only much smaller key-value vectors are needed which has much smaller memory I/O footprint.
2. Lacks of empirical results on pruning real-world LLM. The low error rates on synthetic data may not translate to real world dowstream performances.

### Questions
1. How to revert the pruned fused matrix back to the query and key projection matrices respectively?
2. How does the proposed attention pruning affect length extrapolation ability?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a novel approach to pruning QK projection matrices by directly optimizing for an approximation of the attention matrix, bypassing traditional linear approximations applied before the Softmax operation. By addressing the non-linear nature of the attention mechanism, the authors introduce a gradient descent-based optimization method with theoretical guarantees on convergence, aiming to produce a near-optimal pruning mask. Preliminary empirical results, although limited, demonstrate that this approach can maintain model performance while significantly reducing computational costs.

### Strengths
- The paper offers a robust theoretical analysis, including convergence guarantees for the gradient descent algorithm, which enhances confidence in the method’s foundational soundness.
- The proposed pruning method consistently outperforms baseline approaches in terms of relative error, effectively recovering the attention matrix activations. This suggests its potential for maintaining model performance while reducing computation.

### Weaknesses
- The empirical analysis relies heavily on synthetic data, specifically a full-rank Gaussian random matrix, with matrix dimensions (64x128) that are small relative to the scale of practical LLMs. This somewhat limits confidence in the method’s real-world applicability.
- I understand the main contributions are theoretical, but I think the paper would benefit from experiments with actual LLM activations and weights, even small-scale LLMs. Results from real models would give a stronger sense of how effective this method might be in practice. 
- One big limitation is that the approach only targets $W_Q$ and $W_K$ matrices in the attention mechanism, while the majority of parameters in transformers come from the MLP layers. It would be interesting to see if this technique could be extended to cover the MLP parameters as well, which could have a more substantial impact on reducing model size.

### Questions
- I assume the cost of pruning itself is very small given the closed-form gradient, but it would be helpful to see an analysis of this cost. Is this approach lightweight enough for large models?
- The assumptions regarding the Lipschitz continuity and PL inequality of the gradient are central to the theoretical guarantees. Are there empirical checks or practical insights into how these assumptions hold up in real LLMs?
- The paper uses synthetic data by generating $W_Q$ and $W_K$ through SVD on Gaussian random matrices. A bit more explanation for this choice—or testing on real model weights—would clarify whether the setup could translate to practical applications.

### Soundness
2

### Presentation
3

### Contribution
2
