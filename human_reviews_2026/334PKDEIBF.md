# Accelerating Attention with Basis Decomposition

- Decision: Reject
- Scores: 8, 0, 4

## Abstract
Attention is a core operation in large language models (LLMs) and vision-language models (VLMs). We present BD Attention (**BDA**), the first *lossless algorithmic reformulation* of attention. BDA is enabled by a simple matrix identity from Basis Decomposition (**BD**), which restructures multi-head projections into a compact form while preserving exact outputs. Unlike I/O-aware system optimizations such as FlashAttention, BDA provides a mathematically guaranteed acceleration that is architecture-agnostic. On DeepSeek-V2-Lite (16B, FP16), BDA requires only **4s** of offline preparation **with no retraining required** and, on modern GPUs, achieves **32\% faster** key/value projections and **25\% smaller** weights, while increasing end-to-end perplexity (PPL) by just **0.02\%** (FP16) or **0.0004\%** (FP32)—a negligible effect on model performance. These results position BDA as the first theoretically exact method for lossless attention acceleration that is complementary to existing engineering-level optimizations. Our code is available at https://anonymous.4open.science/r/Basis-decomp-57B8.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces BD Attention (BDA), a lossless algorithmic reformulation of multi-head attention. Unlike approximate methods (e.g., linear attention, pruning, quantization) that trade accuracy for efficiency, or system-level optimizations (e.g., FlashAttention) that primarily optimize memory/I/O, BDA leverages a Basis Decomposition (BD) identity to restructure projection matrices. This reduces redundant parameters and computations while exactly preserving outputs. The authors theoretically derive a framework for the decomposition and empirically demonstrate a speedup in the projection calculation.

### Strengths
- novel idea with foundational application, which impact can spread over all applications of transformer.
- The paper is clearly written and is easy to follow
- Strong theory supporting the claim
- Integrates smoothly with pruning and compression techniques, and supports existing query–key similarity–based methods (e.g., KV-cache compression).
- Might be used as a drop-in replacement for deployed models.

### Weaknesses
- Limited evaluation: While results on DeepSeek models and IWSLT’14 are promising, broader evaluation on more diverse and higher scale benchmarks (e.g., large-scale pretraining)
- My biggest concern about the paper is how well it connects with generally applied optimizations of attention projections, for example I am not sure if the benefits would prevail if applied with MQA/GQA.

### Questions
- Beyond compatibility with KV-cache, could BDA be used to reduce KV-cache size itself, for instance, by caching only the basis vectors?
- How does BDA interact with MQA and GQA, which are now standard in most large LLMs?
- The reported gains focus on projection efficiency. What is the measured speedup for the full forward pass of the model?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This work proposes a method for replacing a low-rank matrix of rank r by storing r columns as a basis and corresponding coefficients to reconstruct the full matrix. The method is applied to reduce computational and memory requirements in transformer language and translation models, where query/key and value/output matrix products are inherently low-rank, and low-rank linear adaptation or pruning is common.

### Strengths
The reviewer found no notable strengths in this submission.

### Weaknesses
The quality of this submission is well below acceptable standards on all fronts:

* Method: The proposed method is extremely incremental---it represents a purely engineering-oriented effort with limited algorithmic contribution. An excessive amount of space (Pages 3-6 inclusive and half of Page 7, i.e., 4.5 pages) is devoted to describing basic concepts that could be summarized in a few sentences, including the fundamentals of transformer multi-head attention and the mathematical background (Theorem 3.1), neither of which constitute contributions of this work.

* Experiments: While the reviewer appreciates the authors’ effort in providing three sets of experiments involving language modeling and machine translation, these are highly insufficient. For example, IWSLT’14 is considered a toy task in today’s machine translation research. Instead of inflating Table 2 by allocating a separate row for each learning rate scale, results should be reported on more realistic benchmarks such as WMT. Regarding the LLaMA-2 pruning experiments (Sec. 4.3), both the baseline pruning (Low rank 80%) and the proposed method lead to large degradations in perplexity; they clearly do not represent a viable approach to improving efficiency without performance loss. 

* Presentation: The writing quality is overall poor. In addition to the space-management issues noted above, some unusual terminology is used (e.g., "end-to-end perplexity"), and the introduction refers to the BLEU metric without specifying the task.

Overall, the method is too incremental, and the reported improvements fall within the realm of minor engineering or optimization tweaks rather than a generally impactful algorithmic contribution.

### Questions
The reviewer has no further questions and considers it unlikely that this work will become acceptable after any rebuttal or discussion.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper suggests an alternative parameterization, called Basis Decomposition (BD), which can be applied to low-rank layers for lossless and I/O architecture-agnostic Transformer inference acceleration. BD is based on the intuition that a rank-$r$ weight matrix $W = UV^T$ can be expressed with $(m + n)r - r^2$ parameters, resulting in $r^2$ fewer parameters than storing $U$ and $V$. The basis matrix $B \in \mathbb{R}^{n \times r}$ consists of the first (or last) $r$ rows of $W$, and the coefficient matrix $C \in \mathbb{R}^{r \times (m - r)}$ is found based on $W$ and $B$, yielding the form $W = \begin{bmatrix} I \\\\ C \end{bmatrix}B$. This parameterization reduces the number of computations in low-rank matrix multiplications. As the reconstructed weight is mathematically identical to the original low-rank formulation, the BD version of $W$ is lossless and shares the same properties during inference, which allows the use of additional compression techniques without problems. In the experimental results, the QK and VO weight matrices in the attention modules are replaced with BD layers, as they are originally low-rank. BD shows negligible perplexity changes and achieves the theoretical speedup with a custom Triton kernel.

### Strengths
1. The proposed idea is conceptually simple and well-motivated. The construction of the basis and coefficient matrices is presented with clear explanations and sufficient intuitive justification.
2. The proposed method demonstrates tangible improvements in both memory usage and throughput for low-rank models. Furthermore, the implementation appears straightforward, assuming the Triton kernel is available.
3. The paper thoughtfully addresses several practical concerns that may arise from rotating the bases of the Q, K, and V weight matrices, such as the validity of positional embeddings, training stability, query–key similarity, and compatibility with other compression techniques.

### Weaknesses
Although the paper has several notable strengths as stated above, I am slightly inclined toward rejection for the following reasons: (1) the claimed I/O architecture agnosticism of Basis Decomposition is insufficiently substantiated, and (2) the practical utility of the proposed method is confined to narrow cases where the relative rank is high, limiting its general impact. The detailed comments are as follows.

1. While the paper claims that BD is agnostic to the memory hierarchy and I/O architecture, the reported speedup is achieved using a custom Triton kernel. As the Triton implementation is not included in the submission, it is unclear whether the code has been optimized for a specific device. I am concerned that the Triton code may require tuning for particular memory or I/O characteristics, in which case the claim of architectural agnosticism would no longer hold. I suggest two possible approaches: either reporting the speedup achieved using a standard PyTorch implementation or demonstrating that the Triton kernel’s performance is indeed independent of the underlying memory and I/O architecture.
2. The reported improvements in memory usage and FLOPs are only by a factor of $r^2$, rendering the reduction in complexity negligible when $r \ll \min(m, n)$. In other words, BD remains effective only when the low-rank matrix is not very low rank. This limitation restricts the applicability of the method to general attention layers, as in many cases the head dimension is significantly smaller than the embedding dimension. For instance, in Llama-2-7B, $m = n = 4096$ and $r = 128$, yielding a relative rank of $r/m = 0.03125$, where BD offers only a $\frac{r}{m + n} = 1.5$% reduction in computational complexity for the original $QK^T$ operation. The DeepSeek-V2-Lite model in the experimental results shows some speedup primarily because it has a higher relative rank of $r/m = 128/512 = 0.25$ compared to Llama-2-7B. Similarly, if the low-rank models in Table 3 had lower rank values, both the memory savings and the speedup would likely diminish accordingly.

Overall, the method appears to be of narrow applicability and lacks sufficient validation to warrant acceptance at this stage.

### Questions
1. Is the Triton kernel independent to the memory I/O architecture?
2. How much speedup does BD make when the rank is very small compared to the full rank (e.g., $m=n=4096$ and $r=128$)?

### Soundness
1

### Presentation
2

### Contribution
2
