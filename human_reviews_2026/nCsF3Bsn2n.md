# Probabilistic Kernel Function for Fast Angle Testing

- Avg Score: 8.00
- Decision: Accept (Oral)
- Scores: 8, 8, 8, 8

## Abstract
In this paper, we study the angle testing problem in the context of similarity search in high-dimensional Euclidean spaces and propose two projection-based probabilistic kernel functions, one designed for angle comparison and the other for angle thresholding. Unlike existing approaches that rely on random projection vectors drawn from Gaussian distributions, our approach leverages reference angles and adopts a deterministic structure for the projection vectors. Notably, our kernel functions do not require asymptotic assumptions, such as the number of projection vectors tending to infinity, and can be theoretically and experimentally shown to outperform Gaussian-distribution-based kernel functions. We apply the proposed kernel function to Approximate Nearest Neighbor Search (ANNS) and demonstrate that our approach achieves a 2.5x--3x higher query-per-second (QPS) throughput compared to the widely-used graph-based search algorithm HNSW. Our code and data are available at https://github.com/KejingLu-810/KS.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles the angle testing problem as part of similarity search in high-dimensional Euclidean spaces. The key idea is to depart from conventional random projection methods based on Gaussian distributions, and instead propose probabilistic kernel functions with deterministic structures that leverage reference angles. Unlike prior approaches, the proposed method does not rely on asymptotic assumptions and theoretically and experimentally outperforms Gaussian-based methods. Applied to approximate nearest neighbor search (ANNS), the proposed approach achieves 2.5–3× higher query throughput (QPS) compared to HNSW.

### Strengths
This paper's strengths are as follows.

(1) The paper presents a theoretically rigorous approach to the angle testing problem without asymptotic assumptions, proposing probabilistic kernel functions with deterministic structures.

(2) By designing appropriate projection vector structures, the method improves estimation accuracy and further demonstrates that Gaussian structures are suboptimal.

(3) The proposed method can be easily integrated into existing algorithms such as HNSW, suggesting strong practical applicability.

(4) Extensive experiments on multiple datasets show that the proposed method achieves both high speed and high accuracy.

### Weaknesses
This paper's weaknesses are as follows.

(1) Constructing the projection vector structures is computationally expensive and could become a bottleneck for extremely large-scale datasets.

(2) Minor typographical issue found: Line 153: “ZS(·))” → “ZS(·)”.

### Questions
My questions about this paper are as follows.

(1) Since the proposed method is based on probabilistic kernel functions, it presumably involves stochastic errors. Is it possible to predict or estimate the error rate analytically?

(2) How does the error rate depend on the projection vector structure? Can this relationship be analyzed theoretically?

(3) How should the number of projection vectors m and the number of subspaces L be optimized? These parameters likely depend on the data distribution, but what practical optimization strategy do the authors envision for real-world applications?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The most commonly used metrics in approximate nearest neighbor, i.e. Euclidean distance, cosine similarity, and inner product, can crucially be reduced to just inner product computation with appropriate preprocessing. The paper proposes two probabilistic kernel functions that approximate (1) the comparison of two inner products $\langle q, v_1\rangle$ and $\langle q, v_2\rangle$ and (2) whether $\langle q, v\rangle$ exceeds a given threshold. These approximations are much faster than the corresponding computations, making it possible to significantly accelerate approximate nearest neighbor search algorithms.

The proposed kernel functions are based on reference angles. Based on these functions, the authors introduce the KS1 test for CEOs tasks such as maximum inner product search and the KS2 test as a routing test in graph-based approximate nearest neighbor search. In the authors' experiments, the KS1 test yields slight improvements, while the KS2 test applied to the HNSW algorithm yields significant improvements.

### Strengths
The paper proposes an interesting approach, angle testing, which has a central role in approximate nearest neighbor search. The authors propose methods for both angle comparison and thresholding. The provided theory is neat and well-motivated.

The proposed KS2 test can be generally applied to many different graph-based approximate nearest neighbor methods, and is amenable to an efficient SIMD implementation that yields a significant improvement in throughput when combined with the popular HNSW algorithm. This combination is also more efficient than combining HNSW with the earlier PEOs approach.

The experiments provided by the authors are comprehensive and use standard benchmark datasets.

### Weaknesses
In practice, the improvement provided by KS1 over CEOs is very minor. The improvement of HNSW+KS2 over the earlier HNSW+PEOs is slightly larger but still relatively small.

The state-of-the-art methods for approximate nearest neighbor search combine graphs and quantization, e.g. Glass combines graphs with scalar quantization and SymphonyQG [1] combines graphs with RaBitQ, yet these are not included in the comparisons. The authors mention that they do not compare to e.g. Glass as it was deemed less efficient than PEOs in the corresponding paper, but experimental results in the PEOs paper do not seem to align with e.g. the results of ANN-benchmarks [2]. The KS2 test is not applicable to quantized vectors in e.g. uint8 or binary precision which are increasingly popular due to the high dimensionality of modern embedding datasets.

A weakness in presentation is that the numerous references to analysis and explanations in the PEOs paper make it difficult to understand the paper without a thorough reading of the PEOs paper.

[1] Gou et al. SymphonyQG: Towards Symphonious Integration of Quantization and Graph for Approximate Nearest Neighbor Search. Proceedings of the ACM on Management of Data. 2025.

[2] Aumüller et al. ANN-benchmarks: A benchmarking tool for approximate nearest neighbor algorithms. Information Systems, 2020.

### Questions
- Could the authors point out where in the provided code release the SIMD implementation of the KS2 test is located? I tried briefly looking in the code but was unable to figure out where in the code it was.

- The integration of KS2 into HNSW moderately increases the index size and indexing time. How do these compare to HNSW+PEOs?

- Have the authors considered testing HNSW+KS2 by integrating it to a standard benchmark such as ANN-benchmarks such that it is easier to reliably compare results?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes two projection-based probabilistic kernel functions for fast angle testing in the context of similarity search in high-dimensional Euclidean spaces. Its core observation is that the reference angle can determine the estimation accuracy of angle comparison and testing, which in turn can be determined by the structure of the projection vectors. A detail theoretical analysis is presented to support the effectiveness of the proposed probabilistic kernel functions. Then, algorithms are proposed to compute the projection vectors, together with an analysis of the computational complexity. The proposed probabilistic kernel functions are integrated into high-dimensional similarity search algorithms (Maximum Inner Product Search and Graph-based ANN Search) to enhance their effectiveness and efficiency, which are verified with experiments on six commonly used benchmark vector datasets.

### Strengths
1. This paper studies an important problem - high-dimensional similarity search.

2. Detail theoretical analysis is presented to show the effectiveness and correctness of the proposed probabilistic kernel functions.

3. Experimental results are presented to show the empirical effectiveness of the proposed probabilistic kernel functions and algorithms.

4. Source code has been released.

### Weaknesses
The empirical performance of the proposed kernel functions is not as strong. It produces marginally higher recall than the CEOs technique (Pham, 2021) as shown in Table 1, while HNSW+KS2 is only 1.1 to 1.3 times faster than HNSW+PEOs. Also, why are Tiny, GIST, and SIFT omitted from Table 1?

It would be good to discuss if the proposed kernel functions are guaranteed to lead to more accurate (and/or more efficient) similarity search than Gaussian-distribution-based kernel functions given the same number of projection vectors. If this is not guaranteed, it would be good to tune down the claim: "can be both theoretically and experimentally shown to outperform Gaussian-distribution-based kernel functions".

Presentation issues:

- It would be good to add figures to help illustrate the key concepts and proposed algorithms. 

- The statement "On the other hand, $v^\top u_{max}$ can be computed beforehand during the indexing phase and can be easily accessed during the query phase" needs further clarification. How can this value be easily accessed given that $u_{max}$ depends on $q$? 

- Typo: "with i.i.d. Gaussian entries)" => "with i.i.d. Gaussian entries."; "and Additional experimental results" => "and additional experimental results"

### Questions
There are no further questions.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper reframes angle estimation from stochastic Gaussian projections to deterministic reference-angle-based kernels. Authors first derive theoretical bounds and apply them to accelerating graph-based retrieval. Specifically authors design (what seems to be) a pruning algorithm that eliminates certain search paths. Although the pruning approach is not new in principle, the resulting approach HNSW-KS2 outperforms a previously proposed variant (HNSW-PEO) by 10% – 30%, along with a 5% reduction in index size.

### Strengths
1. Important topic and well motivated-problem 

2. Theoretically grounded approach that achieves substantial practical gains. 

3. The paper is well-written 

4. Source code is provided.

### Weaknesses
1. Gains over HNSW-PEO are relatively modest (yet non-trivial!) and the method requires extra space, which is non-trivial in some cases. For example, it is >= 40% in the case of the SIFT dataset. 
2. Evaluation is only single-threaded.


**Detailed comments:** 

**Please, do not respond to these, all questions are rhetorical. If suggested correction is not valid, just ignore it** 

Eq. (2) Shouldn’t Z_{HS} be Z_S? 

 
L341 This is not understandable without a basic explanation of what a routing test is. 

L410 there is a missing dot after and HNSW+PEOs 

L427-428 On the other hand, in the high-recall region for Word, ScaNN outperforms HSNW+KS2 due to the connectivity issues of HNSW. -> This requires justification.

### Questions
**Detailed comments:** 

 

1. Eq (1) does it really come from Theorem 3.1 in “Pham, Simple yet efficient algorithms for maximum inner product search via extreme order statistics.” It looks very different. 

2. Due to its ease of implementation, CEOs has been employed in several similarity search tasks (Pham, 2021; Andoni et al., 2015; Xu & Pham, 2024) -> CEO came after Andoni et al. Do you mean it was used in FALCON++? 

3. L354-356 you claim that you skip a distance computation. However, this doesn’t seem to be correct according to Alg 6. I think you also do not add a node to the queues. Basically, this seems to be search pruning approach, not the the approach to reduce the number of distance computations. Please, clarify. 

4. L376 Why didn’t you test multi-threaded retrieval as well? 

 

5. Is L the number of projections? It will be great to clarify in the experimental section. 

6. Does Figure 1 compare PEO and KS2 using the same L? 

7. Gains over CEO are marginal. For example (see Table 1), Probe@100: 6.98 vs 6.9, which is 1% relative. PEO improves upon reverse CEO and you improve upon PEO by double digits percentage points (e.g., 30%). How can you explain this discrepancy? What makes new kernel functions to be more effective? Is it due to the introduction of the threshold approximation? I think it is an important clarification to add to the paper as well. 

8. Thank you for sharing the code: which part of the code benchmarks HNSW-PEO though? I think it benchmarks just HNSW-K1/K2.

### Soundness
3

### Presentation
3

### Contribution
3
