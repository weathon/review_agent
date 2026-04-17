# Approximate Multi-Matrix Multiplication for Streaming Power Iteration Clustering

- Decision: Reject
- Scores: 8, 4, 4, 2

## Abstract
Given a graph, accurately and efficiently detecting the communities present is one of the main challenges in network analysis. In this era, where datasets routinely exceed terabytes in size, many classical algorithms for solving this problem become computationally prohibitive. We address this challenge in the context of the Stochastic Block Model (SBM), which allows for a rigorous analysis. Our approach is a sublinear, updateable, and single-pass approximation to a classic power iteration algorithm \citep{mukherjee2024detecting}. We introduce two sketching-based variants: (1) a \emph{streaming algorithm} for single-pass processing of edge streams, and (2) an \emph{$r$-pass algorithm} that achieves a smaller space embedding at the cost of additional passes equal to the power $r$ of the matrix to be approximated. We show that both methods produce vertex embeddings that guarantee the recovery of the largest cluster when performing single-linkage clustering with an appropriate \emph{separation scale} cut threshold.

Our key contribution is a new theoretical analysis of Approximate Multi-Matrix Multiplication (AMMM), which guarantees that the error from repeated compression remains manageable. This framework extends the stable-rank-based approximate matrix multiplication (AMM) guarantees of \citep{cohen2016optimal} to arbitrarily many conforming matrices. We prove that both algorithms preserve the geometric structure needed to identify the largest community using sublinear space in practice. The streaming algorithm (1) scales with the stable rank of the graph matrix for the streaming algorithm, which we show is sublinear in practice. The $r$-pass algorithm achieves the optimal $O(\varepsilon^{-2}\log n)$. Experiments on synthetic graphs confirm that our methods can recover the largest community as effectively as the exact, expensive algorithm, across both balanced and unbalanced communities, but with dramatically lower memory and runtime.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces a novel framework for efficient clustering of large graphs in dynamic streaming environments by combining power iteration with randomized sketching techniques. Its main contribution is the development of Approximate Multi-Matrix Multiplication (AMMM), a generalization of approximate matrix multiplication that supports interleaved sketching across multiple matrix products. Building on this, the authors propose two streaming algorithms for power iteration clustering:

1.	Interleaved Sketching, which inserts independent sketch matrices between successive multiplications of the centered adjacency matrix, enabling single-pass processing of edge streams with sublinear memory.

2.	End-Sketching, which applies a single sketch after full power iteration, suitable when the graph can be stored but computing high matrix powers is expensive.

The authors provide theoretical guarantees for their algorithms. The empirical results confirm that the proposed algorithms achieve clustering performance comparable to standard power iteration while significantly reducing memory and computational overhead, especially in unbalanced and large-scale graph settings.

### Strengths
1.	The paper makes an important contribution by extending power iteration clustering to the dynamic streaming (turnstile) setting. This generalization is highly relevant for large-scale, evolving graphs where data arrives as a stream of edge insertions and deletions.
2.	The authors provide strong theoretical guarantees for their algorithms. 
3.	The empirical results support the efficiency of their proposed algorithm.

### Weaknesses
The main weakness of the paper is the readability. It is difficult for me to follow. 

1.	The theoretical analysis is dense and closely intertwined with results from Mukherjee & Zhang (2024), making it difficult for readers unfamiliar with that paper to follow the proofs and assumptions. Key steps and intuitions are often deferred to appendices or omitted.

2.	While the paper claims to address the streaming setting where edges arrive as insertions and deletions, it offers little explanation of how the algorithm is actually implemented in such a scenario. The paper focuses almost entirely on the theoretical analysis. Readers not well-versed in sketching techniques are left without a clear understanding of how the sketches are updated incrementally or how memory is managed over the stream.

### Questions
1.	Is there any intuition that the dimension $m$ of sketch is related to the stable rank of matrix B?
2.	Does the algorithm need to know the values of p and q?
3.  Could you explain how we implement Algorithm 1 in a streaming scenario (where edge insertions and deletions occur)? For example, after sampling $S_1, ..., S_r$, do we need to explicitly store these sketch matrices?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies cluster recovery in the stochastic block model. Building on the work of Mukherjee and Zhang (SODA~2024), which used a power-iteration method to recover the largest cluster. in this work, the authors incorporate matrix sketching into this approach. In particular, it approximates the power iteration $B^r$ in two ways: (1) the interleaved variant $\widetilde{B}^{(r)} = B S_1^{\top} S_1 B S_2^{\top} S_2 B \cdots B S_{r-1}^{\top} S_{r-1} B S_r^{\top}$, which is also useful in the streaming model and (2) the end-sketch variant $B^r S^{\top}$. 

The paper next gives a theoretical analysis which show the required embedding dimensions are $\widetilde{O}(sr(B)/\varepsilon^2)$ and $O(\log n/\varepsilon^2)$ for the above two versions of sketches. Finally, the paper gives an empirical evaluation on synthetic dataset that demonstrates a practical sublinear embedding dimension for largest-cluster recovery, validating their theorem.

### Strengths
- The paper presents a complete theoretical analysis. For both approaches, it gives theoretical bounds indicating the required dimension.

- The paper also gives an empirical evaluation to show the performance of their algorithms.

- The presentation of this paper is clear and easy to follow.

### Weaknesses
- From a technical perspective. It is natural to apply matrix sketching to the power-iteration approach, and the target dimension bounds largely follow from standard sketching results and concentration inequalities.

- The empirical evaluation is conducted on synthetic data. I think a real-world dataset would strengthen the paper.

### Questions
1. Sketching can also accelerate the algorithms. The authors might consider using sparse sketching matrices (e.g., CountSketch or sparse JL) and reporting their performance in the experimental evaluation.

2. I think one interesting question is whether the given target dimension is also necessary?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a single-pass streaming sketching approach for approximating product of multiple matrices by using independent subspace embeddings multiplied together to form a product.  Proposed method, which extends approximate matrix-multiplication (AMM) to the case of multi-matrix multiplies, is termed approximate multi-matrix multiplication (AMMM).  The paper focuses on the special case of sketching $B^r$ for identifying “communities” and detecting largest community cluster based on connectivity matrix B.  For this special case paper also presents an iterative “end sketch” approach where a sketch matrix is appended to the power iteration for approximating $B^r$.  Theoretical results are presented for extending AMM to AMMM, and results that give bounds on sketching dimensions for high probability guarantees on community separability and largest cluster detection for streaming sketching as well as iterative end-sketch approaches.

### Strengths
* Extending approximate matrix multiply to sketching multiplication of multiple matrices in streaming setting
* Application to the task of largest cluster detection in graphs by sketching B^r
* Strong bounds on sketch dimensions and high probability guarantees for cluster separability and largest cluster recovery

### Weaknesses
* While the paper claims that the streaming sketch matches performance of iterative end-sketch with slightly larger dimension (e.g. abstract says “The results show that the streaming method achieves recovery performance comparable to the iterative approach, though it requires a slightly larger embedding dimension”), this is not evident at all from Figure 1.
* Related to above, from Figure 1 we do not see the smallest dimension that results in perfect F1/Precision/Recall score for the iterative approach.  Results with lower dimensions (m < 100, perhaps down to 1) need to be included.
* AMMM is meant to be general, but most of the analysis and experimentation focuses on B^r scenario.  This is not a large drawback, but including other scenarios would significantly strengthen the paper.  
* Paper in its current form is hard to follow.  Main contributions of the paper also need to be called out more clearly.

### Questions
n/a

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the problem of recovering the largest clusters in a graph generated by the stochastic block model. Building on the result of [Mukherjee & Zhang (2024)], which established a clear separation threshold between adjacency matrix rows corresponding to nodes inside and outside the largest cluster after power iteration, the authors explore how oblivious sketching (e.g., using random Gaussian matrices) can be applied to this setting. They propose two variants of sketching-based approaches and demonstrate that, after sketching, the input matrix can be reduced to sublinear size while approximately preserving the separation threshold with high probability following power iteration. This enables efficient largest cluster recovery in scenarios where the adjacency matrix is too large to fit in memory, or in turnstile streaming and distributed environments where memory and communication efficiency are critical. Complementary experiments validate some discussions involved in the paper.

### Strengths
The problem of solving largest cluster recovery with applying sketching to power iteration could potentially be interesting.

### Weaknesses
1. $\textbf{Poor Presentation}$. 

The main weakness of the paper lies in its poor presentation and lack of clarity, which makes it difficult for readers to grasp the core problem and the actual contributions. As currently written, the problem formulation and proposed improvements become clear only after reading the algorithm and main results, rather than from the introduction or background sections. Unfortunately, Sections 1 and 2 on introduction and the background add confusion instead of providing context.

1.1 Ambiguous Problem Statement and Settings

The paper never clearly defines the studied problem or the computational settings (turnstile streaming and distributed). There are no formal statements or precise definitions, even in the Appendix. The abstract claims to “guarantee cluster recovery,” while the introduction describes the problem as “clustering of graph vertices.” Later, the contributions section shifts focus to “proving a recovery criterion for the largest community.” These inconsistencies leave readers to infer that the actual goal is largest-cluster recovery, since the proposed method only applies to that case.

Similarly, the discussion of settings is vague. For example, line 67 mentions that the data “can be received as a turnstile (dynamic) stream with insertions and deletions,” and that the approach extends to “distributed memory.” However, it is unclear what “streaming” means here. Does it refer to one edge insertion or deletion per iteration? How is the data distributed across machines? These are left unspecified. Moreover, the main body of the paper develops the algorithm in a local (single-machine) setting, with only brief and informal comments about how sketching might generalize to streaming or distributed environments. While one-pass streaming and one-round distributed protocols are often related, this equivalence must be explicitly stated for the problem at hand. Otherwise, the mention of multiple computational settings feels disconnected and confusing.

1.2 Weak and Misleading Introduction

The introduction is only a single paragraph and fails to set up the context properly. It starts by emphasizing the importance of parallelism in clustering graph vertices, then abruptly motivates the use of power iteration. This is confusing, since (1) power iteration is inherently iterative rather than parallel, and (2) its role in estimating the largest eigenvalue of a PSD matrix is not clearly linked to the clustering objective. The result is a disjointed narrative that obscures both motivation and technical relevance.

1.3 Inconsistent and Unsubstantiated Claims

The claimed contributions are inconsistent with the presented results. The contributions section asserts that the proposed method uses sparse Johnson–Lindenstrauss (JL) transforms and reduces “pass complexity” (line 52). However, Theorem 5.3 focuses on memory reduction, without showing how the sparsity of the JL transform (or the number of nonzeros in the sketching matrix) affects the sketch size or complexity.

The paper mentions at multiple places that the result achieves an embedding of size $\epsilon^{-2}$ in the abstract and main contributions. However, the meaning of this $\epsilon$ is never introduced, which makes it hard to interpret the result. The reader needs to guess that this is somehow related to the quality of the output, or the estimated largest cluster. Indeed, the paper uses distance in the row norms between a sketched adjacency matrix and an unskethced one after performing power iteration as a proxy to measure how good the recovered largest cluster is. However, this connection is never explicitly stated in the paper! 

1.4 Poorly Explained Transitions and Logical Flow

The relationship between Sections 5.3 and 5.4 is unclear. The paper introduces two sketching variants, Algorithms B and C, but their connection is never properly explained. Section 5.3 presents an upper bound for Algorithm B. In contrast, Section 5.4 discusses a lower bound for Algorithm C, but the transition between these sections, particularly lines 346–350, requires further clarification to help readers understand how the two results relate.

1.5 Notational Inconsistencies

The paper suffers from numerous undefined or inconsistent notations. For instance, the graph model is referred to as SBM in the abstract but as SSBM elsewhere. Matrices $B^{(r)}$ and $B^r$ are both used for the matrix in the $r$-th iteration of power iteration. The notation for matrix rows alternates between $B_{i,\cdot}^r$ (line 160) and $R_i$ (line 178) without introduction. The symbol $C$ denotes a constant in line 396 but a cluster in line 401. 

-----

2. $\textbf{Limited Novelty in the Theoretical Analysis}$

The second major concern is that the theoretical analysis appears to offer limited novelty. In line 60, the paper claims as its “primary contribution” an analysis of a generalization of Approximate Matrix Multiplication (AMM), referred to as Approximate Multi-Matrix Multiplication (AMMM), where multiple conforming matrices are independently sketched in a single pass and then multiplied to form a product with bounded error. However, upon examining Theorem 4.2, AMMM seems to be a straightforward extension of AMM, obtained by applying the standard AMM argument repeatedly to a sequence of matrices. Furthermore, the main theoretical result (Theorem 5.3 in Section 5.2) appears to be a direct combination of existing results from [Cohen et al. 2016] and [Mukherjee & Zhang (2024)], without introducing any fundamentally new analytical techniques or overcoming significant technical challenges.

-----

3. $\textbf{Inadequate Experimental Evaluation}$

The third major concern is that the experimental evaluation is insufficient to support the paper’s claims. The experiments do not include a direct comparison between the proposed sketching-based power iteration algorithm and the unsketched baseline from [Mukherjee & Zhang (2024)], either in terms of memory efficiency, runtime improvement, or the accuracy of the recovered largest cluster. Without such comparisons, it is difficult to assess the practical advantages of the proposed method. Moreover, the experimental setup using a graph with $10^4$ nodes may be a bit small to show the benefits of sketching.

### Questions
1. What is the lower bound on the sketch size for Algorithm B? If such a lower bound cannot be established, what are the main technical obstacles preventing it? (This would be somewhat surprising though)

2. How do Algorithm B and Algorithms C compare in theory? Specifically, what are the trade-offs and differences in memory / runtime / quality of the output of using different sketches per iteration and using the same one? 

3. In Lemma 5.1, regarding the computation of the stable rank of $B$, the paper states that if $(p-q)s_{*} \geq \sigma \sqrt{n}$, 

then in the balanced case where $s_l = s_{*} = s$, 

we have $sr(B) =\Theta(K)$. 

However, based on Eq. (12), it appears that 

$\frac{(p-q)^2 K s^2 + n (p-q)^2 s^2 + C_1 n}{\text{constant} (p-
q)^2 s^2} = \Theta(K + n)$, 

since $K \leq n$, ignoring logarithmic factors. 

Similarly, in the highly imbalanced case, Eq. (12) seems to imply 

$\frac{(p-q)^2 s_{*}^2 + n (p-q)^2 s_{*}^2 + C_1 n}{\text{constant} (p-q)^2 s_{*}^2} = \Theta(n)$ 

rather than $\Theta(1)$. 

Would it be possible to clarify the computation of the bounds here?

### Soundness
2

### Presentation
1

### Contribution
2
