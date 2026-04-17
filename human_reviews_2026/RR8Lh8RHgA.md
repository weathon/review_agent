# RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training

- Decision: Accept (Poster)
- Scores: 6, 4, 2

## Abstract
Softmax Attention has a quadratic time complexity in sequence length, which becomes prohibitive to run at long contexts, even with highly optimized GPU kernels. For example, FlashAttention-2/3 (exact, GPU-optimized implementations of Softmax Attention) cannot complete a single forward–backward pass of a single attention layer once the context exceeds $\sim 4$ million tokens on an NVIDIA GH200 (96 GB). We introduce **R**epeated **A**rrays-of-**C**ount **E**stimators (RACE) Attention, a kernel-inspired alternative to Softmax Attention that is strictly linear in sequence length and embedding size. RACE Attention replaces the exponential kernel with a sharpened angular similarity, and approximates attention outputs via Gaussian random projections and \emph{soft} Locality-Sensitive Hashing (LSH), avoiding construction of the full attention matrix. Across language modeling, masked language modeling, and text/image classification, RACE Attention matches or outperforms strong baselines up to $64$K seqeuence length while reducing wall-clock time and memory usage. In addition, we conduct a controlled scaling study on a single attention layer and demonstrate processing of up to 12 million tokens on an NVIDIA GH200 GPU and 75 million tokens on an Intel Xeon® Gold 5220R CPU in a single forward–backward pass, which is well beyond the capabilities of current state-of-the-art attention implementations. RACE Attention thus offers a practical and theoretically grounded mechanism for long-context training on today’s hardware.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper describes RACE attention as a linear-time alternative to softmax attention for very long contexts. The main idea is to replace softmax with powers of angular similarity, and then approximate this term using RACE sketches. To do this, the algorithm uses soft LSH so that its differentiable. This achieves far reduced complexity versus quadratic for standard attention, as is common in most methods for self-attention approximation. What is nice is that the experiments are broad and cover language modeling, masked LM, and classification. In this context, scaling experiments show processing of tens of millions of tokens on CPU and GPU for a single attention layer's forward-backward pass. This will be the main highlight of this work for most readers.

### Strengths
1. The scaling experiments are quite impressive. Regardless of my other comments below, this is a good practical contribution. Also, it is interesting that CPU-based RACE is viable and in some regimes can do better than FlashAttention. This point about algorithmic efficiency versus hardware acceleration could really be a main message of the paper (more on this below). In any case, reaching 50M/75M tokens is definitely a strength (but in the current version of the paper, this comes with some disclaimer).

2. The experimental breadth is very good. Both CPU and GPU kernels with OpenMP are mentioned. This is a strong engineering effort and if code is provided, it can benefit many groups working in this area. 

3. Experimental verification of how increasing degree can mimic exponential behavior in this setting is useful. Some analysis is included for the bias-variance to guide the choices in the sketching component. This is all good.

### Weaknesses
1. I am a bit confused by the numerous instances of "stress test" and therefore it unclear what the scaling experiments actually show. When stress testing 1 forward-backward pass with the multi-head attention layer, is this timing a single layer, not end-to-end model training? If so, the 75M token claim is for one attention operation, not training the full model? Is this paper only describing benchmarking the primitive or does any model work at these lengths? The reason for this question is the title "outrageously large context windows" -- is this only for the stress tests? The most reasonable reading of the title suggests full model capability.

2. I am having trouble understanding the tables on page 8. Is angular expected to be better than RACE? 

3. The paper https://proceedings.mlr.press/v139/zeng21a.html uses related ideas and also seems motivated by similar upstream papers. Another one is https://aclanthology.org/2022.iwslt-1.4.pdf. The positioning of this work on page 4/5 should at least describe how they differ.

### Questions
1. Minor: Is adapting the analysis to causal masking relatively easy (but hasn't been worked out yet) or does one run into problems?
2. check some of the references above. There may be others.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces RACE Attention, a method to address the quadratic time and memory complexity of standard softmax attention. The authors propose replacing the exponential softmax kernel with a high-degree monomial of an angular (cosine) similarity kernel. This specific kernel choice allows them to leverage Locality Sensitive Hashing (LSH) and Repeated Arrays-of-Count Estimators (RACE) sketches to compute the attention output in linear time and space complexity.

### Strengths
1.  The primary contribution and strength of this paper are the scaling results. Figure 5 shows that RACE on a CPU can outperform FlashAttention on a high-end GPU at massive sequence lengths, is a compelling demonstration of the algorithm's effect over hardware acceleration.

2. The paper is well-written and easy to follow.

3. The theoretical result also provides a nice bias-variance trade-off of their approach.

### Weaknesses
1. The paper seems to be lacking some important baselines. The authors compare their result to FlashAttention, however, at the moment FlashAttn 2 and 3 are also available that performs much faster and are not included in the comparison. Moreover, the paper focuses on alternatives to softmax and is for example lacking a comparison to Sigmoid Attention which also provides a simple kernel implementation.

2. The paper is a bit vague and ambiguous in their main algorithm. The authors argue that they use cosine kernel to prevent the exponential of softmax and be able to use RACE sketch. However, it seems that Algorithm 1 is still trying to implement softmax. Am I misunderstanding this? Technically, it seems that the connection between the features $\phi$ and the angular attention is never clearly made.

### Questions
1. Can authors elaborate on how to choose $\gamma$? Would it be through a hyperparameter search or is there a principled way of approximating a good value for it?

2. Once more question on $\gamma$, could authors provide any sensitivity analysis of how the final result changes with respect to the small changes in $\gamma$? Perhaps another useful figure would be to use the data from Fig 2 and plot the distribution of the attention distances between softmax and the angular attention to see how it varies as $\gamma$ is changed.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a novel linear-time attention mechanism. The approach replaces the exponential softmax kernel with a monomial of cosine similarity raised to a power, enabling approximation through randomized projections. By leveraging angular similarity, Locality-Sensitive Hashing, the authors propose an efficient that enables outrageously large context windows  up to 75 million tokens on CPUs and 12 million on GPUs.

### Strengths
1. This method enables linear-time and memory-efficient attention that scales to tens of millions of tokens on standard hardware, which is impressive. 
2. The algorithm is simple, differentiable, and can serve as a drop-in replacement for softmax attention.

### Weaknesses
1. **This paper is very similar to YOSO [1] (for example, the finding the similarity between equation (1) and (2) in the text, the use of LSH in estimating the similarity function, the algorithm of estimating attention outputs via hashtables), but this paper does not discuss and contrast with [1].**
2. The experiments only show model accuracy on short sequence lengths (< 8K). What about longer sequences? 
3. The efficiency results in Figure 3 are not very meaningful as any linear attentions can be extremely efficient by tuning their hyperparameters. For example, for $\phi(Q) \phi(K)^T$ type attention, by setting the output dimension of $\phi$ to be 1, its efficiency can beat any other methods. To show efficiency, the runtime and memory results should be coupled with the corresponding accuracy results. 
4. Figure 5 has the same issue, what about the accuracy? 

**If the authors can address my concerns, I am willing to raise my score.**

[1] Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh. You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling. ICML 2021.

### Questions
see weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
