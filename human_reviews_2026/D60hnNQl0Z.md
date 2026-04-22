# Panorama: Fast-Track Nearest Neighbors

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 8, 4, 2

## Abstract
Approximate Nearest-Neighbor Search (ANNS) efficiently finds data items whose embeddings are close to that of a given query in a high-dimensional space, aiming to balance accuracy with speed. Used in recommendation systems, image and video retrieval, natural language processing, and retrieval-augmented generation (RAG), ANNS algorithms such as IVFPQ, HNSW graphs, Annoy, and MRPT utilize graph, tree, clustering, and quantization techniques to navigate large vector spaces. Despite this progress, ANNS systems spend up to 99% of query time to compute distances in their final refinement phase. In this paper, we present PANORAMA, a machine learning-driven approach that tackles the ANNS verification bottleneck through data-adaptive learned orthogonal transforms that facilitate the accretive refinement of distance bounds. Such transforms compact over 90% of signal energy into the first half of dimensions, enabling early candidate pruning with partial distance computations. We integrate PANORAMA into SotA ANNS methods, namely IVFPQ/Flat, HNSW, MRPT, and Annoy, with out index modification, using level-major memory layouts, SIMD-vectorized partial distance computations, and cache-aware access patterns. Experiments across diverse datasets—from image-based CIFAR-10 and GIST to modern embedding spaces including OpenAI’s Ada 2 and Large 3—demonstrate that PANORAMA affords a 2-30x end-to-end speedup with no recall loss.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes PANORAMA, a framework to accelerate the refinement phase in Approximate Nearest Neighbor Search (ANNS). The key idea is to use a learned orthogonal transformation that concentrates the norm of the data in the first few dimensions, allowing for early candidate pruning via tight lower bounds (LB) on the L2 distance. The authors integrate PANORAMA into existing ANNS systems (IVFPQ, HNSW, MRPT, and Annoy), demonstrating empirical speedups of 2-30× while maintaining recall.

The method builds upon the idea that a well-chosen orthogonal transform T can compact signal energy, thereby yielding tighter cumulative distance bounds (via Cauchy–Schwarz inequalities). A Cayley parameterization is used to learn T, and partial distances are computed incrementally to prune candidates when the lower bound exceeds the top-k distance threshold.

### Strengths
- The motivation is relevant: reducing the verification cost in ANNS is practically important.

- The integration into multiple ANNS backends is technically solid and shows strong engineering effort.

- Experimental results are extensive, and speedup claims are clearly reported.

### Weaknesses
**W1. Incorrect runtime accounting for the learned transformation**

The learned transform T is applied both to database vectors and queries (Section 4), but in experiments, the query time does not include the transformation cost—as visible from the released code (I checked simple_benchmark.py and see transformed queries are stored separately from original queries) and described pipelines (eq (1)). Since T(q) must be computed for every query, omitting this step underestimates query latency. For large d, the transformation cost (a dense matrix–vector multiply) can dominate the supposed savings from partial distance pruning.

**W2. Theoretical–empirical mismatch in the use of the lower bound**

The theoretical lower bound relies on decomposing ∥q−x∥ = ∥T(q)∥ + ∥T(x)∥ − 2 ⟨T(q),T(x)⟩ (Eq. 1). However, even if T compacts the energy of
x and q individually, it does not guarantee that the energy of their difference ∥q−x∥ is similarly front-loaded. Thus, the learned transform may tighten bounds on norms of x, but not necessarily on distances, breaking the link between the “energy compaction” assumption (A1) and the pruning efficacy. The experiments show speedups mainly from implementation optimizations rather than LB tightness. Fig 6 seems to reflect the concentration of the norm of data points only, not the distance. I checked evaluate_all_transformed_datasets.py in the released code, which seems to confirm my findings.

**W3. Novelty is limited, and the lack of relevant competitors**

The idea of leveraging partial or bounded L2 distances has been explored in several recent works, including Gao & Long (2023) and Yang et al. (2025), both cited by the authors. Those methods also used partial distance estimation or orthogonal projections to accelerate refinement. The proposed contribution proposes a new learning transformation that potentially gives tight lower-bound formulations. Hence, the improvement is primarily an engineering optimization (cache layout, batching, SIMD) rather than a new algorithmic insight. Also, the competitors are ANNS solvers without the use of lower bounds. They do not include relevant competitors (e.g. Gao & Long (2023) or standard methods: ANNS with DCT/FFT transformation) that use LB for speeding up the verification.

**W4. Empirical gains not well-attributed**

It is unclear how much of the reported 2–30× speedup originates from the theoretical contribution (learned transformation and bound) versus from memory layout changes (level-major batching) or engineering effort. The lack of ablation on the cost of the learned transform further blurs this distinction.

**W5. Unrealistic distributional assumptions**

Assumption A3 in Section 3 states that the squared distances ∥q−x∥ follow a Gaussian distribution. This is not true for high-dimensional candidate sets where all candidates are close to q. In practice, kNN candidates are not independent random samples—they cluster around
q. This invalidates parts of the theoretical derivation and weakens claims of Theorem 2.

**Minor:**
- Notations need to be consistent in the whole paper, i.e. candidate size (N' in Problem 1, N in Theorem 2), C in Theorem 1 and Theorem 2 and 3; so(d) vs SO(d)
- Theorem 1 is trivial.
- I think Subsections 4.1 and 4.2 should be significantly improved. I could not see the link between the learning objective in (6) and 4.1 subsection.
- Text font in Algorithm is smaller than the main text.

### Questions
Please address the raised weaknesses above, and some further questions

**Q1. Query-time transformation overhead.**

PANORAMA applies the learned orthogonal transformation T(q) to each query (Appendix B.5). What is the average per-query overhead when this transformation is included in the full pipeline timing (not isolated)? How significant is this cost relative to the total query latency, especially on high-dimensional datasets such as GIST or CIFAR-10?

**Q2. Tightness of the lower bound.**

How tight is the proposed PANORAMA lower bound on **distance** compared to other dimension-wise partial-distance bounds such as those obtained via FFT or DCT transforms. Could you quantify the actual distance-ratio gap (i.e., LB/∥x−q∥) using the same number of reduced dimensions to better interpret the bound’s tightness beyond speedup metrics?

**Q3. Threading and cache sensitivity.**

Are all query experiments executed in single-threaded or multi-threaded mode? Since the proposed lower-bound (LB) pruning mechanism relies on early termination within distance accumulation, concurrent threads may interfere due to cache sharing. How sensitive is the observed speedup to the degree of threading and CPU cache hierarchy?

**Q4. Training cost and data scale.**

Could you specify the runtime to learn the transformation compared to the cost of building indexes? Appendix B.4 mentions training times under 20 minutes (≈ 1 hour for SIFT and Large/CIFAR-10), which seem to be too large compared to building indexes used by Faiss.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
ANNS is a highly practical research direction with extensive applications in industry. The authors of this paper propose a data-adaptive transformation method that compresses feature discriminablity (energy) into the leading dimensions without losing structural relationships. Combined with pruning, this approach reduces the computational cost of feature similarity calculations. The method can be jointly applied with all four existing classic ANNS algorithms to further enhance retrieval efficiency.

### Strengths
1）The research problem has very significant practical application value.
2）The theoretical proofs are comprehensive.
3）The proposed method is compatible with all four existing classic approaches, further improving retrieval efficiency.
4）It not only introduces an algorithm but also provides optimization solutions at the system level.

### Weaknesses
1）Lack of large-scale experiments. The largest dataset used in the paper is only 1 million in scale, which is relatively small for practical industrial applications. For instance, the datasets in previous NeurIPS competitions have already reached the billion scale.
2）Generalizability of the data-driven transformation matrix. The quality of this transformation matrix is likely highly dependent on the training dataset, raising concerns about its generalization capability.
3）Substantial reconstruction overhead when integrated with methods like IVFFlat. The underlying reconstruction effort required for integration is non-trivial.

### Questions
Please refer to the weaknesses section.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the problem of optimizing the *refinement* phrase of approximate near neighbor (ANN) search, which the authors define as identifying the top $k$ elements from an initial match set of $|\mathcal{C}| > k$ items. The authors introduce a novel framework called Panorama to solve the refinement problem by leveraging orthogonal transforms. In particular, the authors introduce data-adaptive learned orthogonal transforms based on the Cayley transform over the Stiefel manifold that aims to concentrate vector distances in the initial dimensions and thus reduce the computational complexity of distance calculations during the refinement phase. In addition to theoretical guarantees, the authors also implement their approach by carefully considering the memory layout of the underlying ANN algorithm and report substantial speedups on the order of 2-30x for the refinement phase.

### Strengths
1. The proposed method introduced in the paper of energy-based learned orthogonal transforms is very creative and a novel application within this particular problem domain. 

2. The authors work in combining rigorous theoretical guarantees with high-performance implementations that consider low-level memory layout details is a strong effort in bridging theory and systems work, which is a rare combination in the retrieval literature.

3. The ideas in the paper have the potential to inspire further work along this direction. 

4. The authors proposed approach is very general and can be applied to virtually any ANN algorithm regardless of its inner workings.

### Weaknesses
In my opinion, the biggest weakness in the current version of the paper is an insufficient discussion of related work on this topic. The notion of refinement in retrieval and similarity search is very well studied and goes by various names such as "reranking" and "approximate distance computations." I think it would be very helpful if the authors included a dedicated related work section that discussed prior approaches to refinement. Moreover, I believe it is critical for the authors to compare against previously published techniques in their experimental evaluation and thereby consider more rigorous baselines than naive refinement. In particular, prior works that I think are very relevant to this paper and should be discussed include: (1) [Finger](https://arxiv.org/pdf/2206.11408), (2) [Probabilistic Kernel Function for Angle Testing](https://arxiv.org/pdf/2505.20274), and (3) [A Bi-metric Framework for Fast Similarity Search](https://arxiv.org/pdf/2406.02891) (plus the broader literature on reranking techniques). I believe that addressing and experimentally evaluating against this prior literature is critical for positioning this new work appropriately. 

In addition, I think it would be very helpful if the authors considered additional large-scale benchmark datasets at the 100M or 1B vector scale, such as those from Big ANN Benchmarks.

### Questions
1. Can the authors provide a more thorough discussion of prior published work in the refinement literature, including perhaps the papers listed above (if they are in fact relevant)? I think it is critical to include this discussion in the paper in a standalone section. 

2. Can the authors also provide an experimental comparison with previously published refinement algorithms that go beyond naive refinement? Additional experiments on large-scale datasets, such as those from Big ANN benchmarks, might also be very helpful in supporting the claims made in the paper.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This work studied a method to improve the efficiency of approximate nearest neighbor search with no performance loss. The approximate nearest neighbor search algorithms are fast, but this work combines with naive kNN L2Flat (Douze et al., 2024) which performs
a brute-force kNN search over the entire dataset. The computation cost of kNN grows fast when the feature dimension of dataset goes large. The authors propose a multiple level, multiple batches method to index the dataset, the level 1 (leaf level) has M dimension which equals the number of features of the dataset.  

According to the Algorithm 1, that the computation is linear with parameters. However, it is not sure if the memory cost is same as Problem 1 2^|D|, $|D|$ is the number of samples. For retrieval question, the challenge also comes from large number of dataset pool. If memory cost is $2^|D|$, it is not desirable. 

The lower bound and upper bound as shown in Equation 3 and 4 are not informative enough. In a sense, the bound do not contribute a lot to the theoretical guarantee. 

Also the main contribution is the speedup without recall loss. If the number of retrieved samples is too big, the retrieved results are not useful. The number of returned samples is not specified in the experimental results. If authors could help solve the concerns, that would be helpful.

### Strengths
The problem is interesting and with practical use. The introduction introduce the most recent works that this work is most related. That is helpful to understand the main contribution of the work.

There are multiple dimensions of experiments to validate the speedup contribution of the work.

### Weaknesses
The lower bound and upper bound are loose, they do not show contributions to the theoretical guarantee. 

The memory cost is 2^|D|, $|D|$ is the number of samples which is huge in retrieval question. The multiple-level indexing method as shown in Figure 3 seems computational expensive to me, since the retrieval requires inner product computation from tree root to leaf. It does not make use of any correlation between samples in batches, batches.

The recall loss claim does not specify the number of returned samples, if the recall loss has the trade of the number of retrieved samples, the retrieval results are not informative enough.

### Questions
1. According to the Algorithm 1, that the computation is linear with parameters. However, it is not sure if the memory cost is same as Problem 1 2^|D|, $|D|$ is the number of samples.

2. How does the lower bound and upper bound as shown in Equation 3 and 4 contribute to the theoretical guarantee?

3. What is the average number of retrieval for each query for different datasets in the experiments?

### Soundness
3

### Presentation
3

### Contribution
2
