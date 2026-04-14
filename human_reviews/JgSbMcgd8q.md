# Multi-Layer Transformers Gradient Can be Approximated in Almost Linear Time

- Decision: Reject
- Scores: 5, 6, 6, 6

## Abstract
The computational complexity of the self-attention mechanism in popular transformer architectures poses significant challenges for training and inference, and becomes the bottleneck for long inputs. Is it possible to significantly reduce the quadratic time complexity of computing the gradients in multi-layer transformer models? This paper proves that a novel fast approximation method can calculate the gradients in almost linear time $n^{1+o(1)}$ where $n$ is the input sequence length, while it maintains a polynomially small approximation error $1 / \mathrm{poly}(n)$ across the entire model. 
Our theory holds for general loss functions and when the multi-layer transformer model contains many practical sub-modules, such as residual connection, casual mask, and multi-head attention. 
By improving the efficiency of gradient computation, we hope that this work will facilitate more effective training and deployment of long-context language models based on our theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents a novel method to approximate the gradient computation in multi-layer transformer models with almost linear time complexity. Traditionally, the self-attention mechanism in transformers has a quadratic time complexity with respect to the input sequence length, which becomes a bottleneck during training, especially for models with long input contexts. The authors propose a fast approximation algorithm that calculates gradients in time proportional to $𝑛^{1+o(1)}$ , where n is the input length, and achieves a polynomially small approximation error. This method is applicable to general loss functions and transformers with practical components such as residual connections, multi-head attention, and causal masks. The results hold significant potential for improving the training efficiency of large language models without sacrificing much accuracy.

### Strengths
*  The paper introduces a method to approximate gradient computation in multi-layer transformers in almost linear time, breaking the quadratic time complexity barrier. This efficiency boost is crucial for training large language models, especially for long-context tasks, and could reduce both computational and energy costs.

* The proposed algorithm is versatile, supporting general loss functions and working with commonly used transformer components such as residual connections, multi-head attention, and causal masks. This ensures that the method can be applied to a wide range of practical transformer architectures and tasks without major adjustments.

### Weaknesses
* While the paper provides a strong theoretical foundation for the proposed method, there is no empirical evidence or experimental results presented to validate its performance in real-world transformer models. This leaves a gap in understanding how well the method performs in practice compared to theoretical expectations.

* Although the approximation error is stated to be polynomially small, the paper does not deeply explore how this approximation might affect model performance in different tasks. There could be edge cases where the accuracy trade-offs are more significant, especially in critical applications requiring precise outcomes.

* The paper acknowledges that practical implementation, especially on GPUs, may face coding and optimization challenges, such as needing to redefine tensor operations and re-implement certain backpropagation functions. This could limit the immediate adoption of the method by practitioners looking for a ready-to-use solution.

### Questions
* Could the authors provide empirical results or benchmarks to demonstrate the practical efficiency and effectiveness of the proposed gradient approximation method on real-world large language models? How does it compare to state-of-the-art methods in terms of both speed and accuracy?

* The paper claims that the approximation error is polynomially small, but how does this error affect downstream tasks such as text generation or classification? Are there scenarios where this approximation might lead to a noticeable degradation in performance, especially in sensitive applications?

* The authors mention potential coding challenges for implementing this method, particularly on GPUs. Have the authors considered these challenges in detail, and can they provide more concrete guidance or solutions for implementing this algorithm efficiently in real-world systems, especially for large-scale models like GPT-4 or LLaMA?

### Soundness
2

### Presentation
2

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
In this paper, the authors provide a number of algorithms that enable one to approximately compute the gradient (and forward pass) of a multi-layer Transformer in time that is nearly-linear in $n$, the input sequence length. The key challenge is in dealing with the self-similarity matrix, which has $O(n^2)$ entries. The dimension $d$ is assumed to be $O(log(n))$.

In Theorem 4.1, authors establish nearly-linear time guarantees for computing the gradient of a single-layer Transformer.

In Theorem 4.2, the authors generalize this result to a Transformer with up to $m=n^{o(1)}$ layers.

In both Theorem 4.1 and 4.2, the guarantee is roughly $1/poly(n)$ error in $n^{1+o(1)}$ time. 

In section 6, the authors discuss further extensions of their results to various settings, including multi-headed attention, causal mask, and prompt tuning.

### Strengths
Overall, I think being able to compute the attention (both forward and backward) is a question of great theoretical and practical interest.

I think the extensions to a $m$-layer Transformer (Theorem 4.2) is interesting, well-motivated, and proof is non-trivial. (However, I do have trouble verifying some parts of the proof, see below).

### Weaknesses
My main criticism of the paper is that, while the results are interesting and important, I am uncertain about the contributions of the present paper, given the existing papers on this topic. 

Theorem 4.1 shows fast gradient computation for a single layer. This result seems identical (at least in big-O notation) to the result in Theorem 1.6 of [1]. On line 446, authors claim that they generalize the results in [1] in two ways: (a) accomodate general loss function L(X), and (b) generalize their result to both $W_i$ and $W_{V_i}$. However, I do not see why (a) is meaningful -- for the purpose of computing gradients, one can wlog consider a linear $L(X)$, which is exactly handled by [1]. I also do not see why (b) is meaningful -- [1] already handles derivative wrt $W_{i}$, whereas the derivative of $W_{V_i}$ is just a minor extension of existing results (see Remark 1.3 of [1]).

I feel that the generalization of 4.1 to the multi-layer setting is potentially quite nontrivial -- the error from a single-layer's grradient computation can blow up over layers, and thus handling this is non-trivial. However, I had a bit of trouble verifying the part of the proof that bounds this error propagation. The main result seems to be Lemma 5.5 (Theorem H.4). In this proof, the key seems to be bounding error accumulation in the gradient computation over $m$ layers, whose proof I have some trouble verifying. (see item 2 under questions below)

Finally, the authors discuss a number of extensions of their work to handling things like multi-headed attention, and causal masks. I think the handling of causal mask is also interesting, but the result seem to be almost immediate following [2].

### Questions
1. In Theorem 1.6 of [1], authors show that there is a $n^{1+o(1)}$ time algorithm that computes the gradient up to 1/poly(n) accuracy. Can the authors please comment on the difference between this result and Theorem 4.1? Both in terms of guarantees, as well as in terms of proof technique. Does the present paper's proof use Theorem 1.6 in any way?
2. Can the authors elaborate on the proof of Theorem H.4 for the multi-layer analysis? Specifically, I refer to line 3646-3650: how did $n/poly(n)$ end up becoming $1/poly(n)$? Since the authors are using an inductive proof, I don't think one can rigorously prove the inductive step with explicitly writing down the degree in $poly(n)$. Related to this problem, I don't see where the authors use the assumed bound on $m$, surely, the inductive proof has to use this fact?
3. Can the authors please comment on what are the key challenges and theoretical contributions in proving their result on causal masks? In particular, please highlight what additional work was done on top of the results by [2].

**minor suggestions that did not affect score**
1. I suggest to remove mention of $m =n^{o(1)}$ in Theorem 4.1 if you are going to assume $m=1$ in the same theorem.
2. line 401-410 seems to be a duplication of 409-416. 

[1] [The Fine-Grained Complexity of Gradient Computation for Training Large Language Models](https://arxiv.org/pdf/2402.04497), Alman and Song 2024

[2] [Conv-basis: A new paradigm for efficient attention inference and gradient computation in transformers.](https://arxiv.org/pdf/2405.05219) Liang et al 2024a

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
This paper proposes a method to compute gradients in multi-layer transformers in almost linear time, addressing the quadratic complexity bottleneck of traditional self-attention mechanisms. By leveraging a low-rank approximation technique, the approach aims to make large language models more efficient, particularly for long-context tasks, while maintaining a small, polynomial approximation error. The algorithm is versatile, supporting general loss functions and widely used transformer components such as residual connections, multi-head attention, and causal masks, making it theoretically applicable to a broad range of transformer-based architectures.

### Strengths
Strengths:

1.Innovation in Complexity Reduction: The primary contribution of an almost linear time approximation method for gradient computation addresses a significant bottleneck in transformer models, advancing efficiency in training large language models.
2.Comprehensive Theoretical Support: The paper offers a well-developed theoretical foundation, demonstrating that the proposed method maintains accuracy within a polynomially small approximation error.
3.General Applicability: The approach is compatible with various sub-modules (e.g., residual connections, multi-head attention) and loss functions, enhancing its utility across diverse transformer-based applications.
4.Potential Impact on Long-Context Models: Given the paper’s focus on scaling efficiency for long input sequences, the findings are likely to benefit future developments in large language models that need to handle extensive contexts effectively.

### Weaknesses
1.Lack of Empirical Validation: Although the paper establishes a strong theoretical foundation, it lacks empirical evaluation to validate the proposed method's performance in real-world applications. While there is discussion of related methods, a deeper quantitative or empirical comparison (e.g., speed, memory usage) with current acceleration techniques could further support the claims of the paper. Without experiments or benchmarks, it is difficult to assess the practical effectiveness of the method, especially in comparison with other acceleration techniques.
2.Unclear Approximation Error Impact: While the approximation error is stated to be polynomially small, the paper does not explore how this error might affect model performance in downstream tasks. In applications requiring high precision, even small errors could be significant; this is particularly relevant in domains where precision is critical, such as healthcare or finance.
3.Limitations of Low-Rank Approximation: The proposed approach relies on low-rank approximation of the attention matrix. However, if the attention matrix is effectively full-rank, this assumption may not hold, potentially negating the efficiency gains of the method. Further clarification on the handling of full-rank attention maps would improve the paper.
4.Potential Implementation Challenges: The paper notes that implementing the approach, particularly on GPUs, may involve re-defining tensor operations and re-implementing specific backpropagation steps. This could pose a barrier to immediate adoption by practitioners, as it requires significant engineering effort. This is a good paper and I would like to raise my score if the author can provide more empirical supports.

### Questions
1. Could the authors provide examples or scenarios where this method outperforms existing quadratic complexity methods in practical applications?
2. For cases where the multi-layer approximation results in slight errors, how do these impact tasks with high sensitivity to precision (e.g., in medical or financial text processing)?
3. Can the authors clarify the compatibility of this method with common pre-training setups in large language models? Specifically, how would this method perform in tandem with models like LLaMA or GPT-4?
4. What happens if the attention matrix is full-rank? Does the proposed method lose its advantage in such cases?
5. A more detailed, quantitative comparison with recent works on transformer efficiency, such as FlashAttention[1], hyperattention[2], or tensor-based approximation methods, would enhance the positioning of the proposed method.

[1] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
[2] HyperAttention: Long-context Attention in Near-Linear Time

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
The paper addresses the important problem of reducing the quadratic computational complexity of gradient calculation in transformers, proposing an approximation algorithm that computes gradients in almost linear time with a bounded polynomial error. The work builds on recent work in reducing the quadratic complexity of the forward pass and gradient computation for single-layer transformers to linear complexity.

### Strengths
The work focuses on a highly relevant topic, given the growing context length in the training of LLMs, and highlights the need for algorithms with solid theoretical backing to improve the quadratic complexity of gradient computations in the attention mechanism.

The results seem broadly applicable, as they hold across different loss functions (unlike previous studies) and can incorporate important elements of multi-layer transformers, such as MLPs, residual connections, and causal masking.

### Weaknesses
The work is very interesting, but I have a few questions and points to raise.

In general, I feel that there are missing intuitions about the approach in the main body. For example, the technical overview section mentions that the attention matrix can be approximated by low-rank matrices. The related work section (line 157) states that the work uses polynomial approximation techniques (Aggarwal and Alman, 2022), but there is no further discussion about these techniques in the main body. I strongly believe that a small discussion section should be included, as this addresses the heart of the challenge.

Similarly, for Algorithm 1, all the steps reference lemmas in the appendix, but there is no discussion of these steps in Section 5. While Section 5 discusses accelerating the gradient for individual variables, there should be a connection and reference to Algorithm 1 in this discussion. Furthermore, I strongly suggest including a small proof sketch for the key lemma, Lemma 5.1, which shows the linear time complexity of the gradient computation with respect to $T_i$.

Lastly, paragraphs 401-407 and 409-413 essentially say the same thing with minor rewording—please address this repetition.

Some questions:

—What assumptions do the results of Theorem 4.2 (linear time complexity and $1/poly(n)$ error) require to hold?

—The results operate in a very practical regime of $d≪n$, but $d=O(log⁡ (n))$ seems too small, considering practical values like 1024–4096. I understand that such assumptions are often necessary for theoretical results, but how sensitive do you think the results are to this choice? The same question applies to the sub-polynomial choice for the number of layers $m$ and the logarithmic bit precision. Does this imply a practical limit on the number of layers for the algorithm to remain efficient?

—It would be useful to see the algorithm in practice and compare it to quadratic attention. However, I feel that the benefits will only be apparent when n is very large (e.g., at least 100k, which requires significant computational resources and may be difficult to run. Have the authors had a chance to experiment with this?

### Questions
See Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2
