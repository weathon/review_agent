# Fast unsupervised ground metric learning with tree-Wasserstein distance

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
The performance of unsupervised methods such as clustering depends on the choice of distance metric between features, or ground metric. Commonly, ground metrics are decided with heuristics or learned via supervised algorithms. However, since many interesting datasets are unlabelled, unsupervised ground metric learning approaches have been introduced. One promising option employs Wasserstein singular vectors (WSVs), which emerge when computing optimal transport distances between features and samples simultaneously. WSVs are effective, but can be prohibitively computationally expensive in some applications: $\mathcal{O}(n^2m^2(n \log(n) + m \log(m))$ for $n$ samples and $m$ features. In this work, we propose to augment the WSV method by embedding samples and features on trees, on which we compute the tree-Wasserstein distance (TWD). We demonstrate theoretically and empirically that the algorithm converges to a better approximation of the standard WSV approach than the best known alternatives, and does so with $\mathcal{O}(n^3+m^3+mn)$ complexity. In addition, we prove that the initial tree structure can be chosen flexibly, since tree geometry does not constrain the richness of the approximation up to the number of edge weights. This proof suggests a fast and recursive algorithm for computing the tree parameter basis set, which we find crucial to realising the efficiency gains at scale. Finally, we employ the tree-WSV algorithm to several single-cell RNA sequencing genomics datasets, demonstrating its scalability and utility for unsupervised cell-type clustering problems. These results poise unsupervised ground metric learning with TWD as a low-rank approximation of WSV with the potential for widespread application.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes an interesting variation of Wasserstein singular vectors by embedding samples and features on a tree. By doing so, they claim to have achieve a cubic complexity as opposed to quintic complexity of the standard method. While the authors show interesting results, I believe the manuscript can be accepted after a major revision.

### Strengths
I think the paper is proposing a novel idea for estimating ground metric learning of unlabeled data set, that reduces the complexity of each iteration from $\mathcal{O}(N^5)$ to $\mathcal{O}(N^3)$.

### Weaknesses
Some important details of the proposed method are missing in the paper. This includes convergence rate, well-posedness, and memory consumption of the proposed method. See my questions/comments below.

### Questions
**Major**:

- How is the condition number of eq. 5? Does the system become ill-conditioned as matrix becomes large?
- Section 2.3, what is the convergence rate of the proposed iterative method? How it is affected by the data set size?
- What is the complexity of Cluster Tree used here?
- Please add details of memory consumption for your proposed iterative algorithm and approximation to SVD.
- I think the authors should make more effort to show the claimed complexity $\mathcal{O}(N^3)$. Consider one of the data set, and test the method against benchmark for a range of $N$.

**Minor**:

- In abstract, "...a fast and recursive algorithm..." and not "a fast, recursive algorithm…”
- In abstract, what does "low-compute application" mean?
- P2, l82: "... to have a solution" and not "to have solution"
- P3, l108: Is there a guarantee that the shortest path between any two nodes on a tree is unique?
- Figure 1: Caption refers to b5, but I don’t see b5 in the graph.
- In the references, make sure to use capital letters when needed, e.g. use “Wasserstein” instead of “wasserstein”

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a novel approach for unsupervised ground metric learning using Tree-Wasserstein Distance (TWD) as a low-rank approximation of the computationally intensive Wasserstein Singular Vector (WSV) method. The proposed method embeds samples and features in tree structures, reducing the computational complexity from O(n⁵) in traditional WSV to O(n³) by learning distances between data points as TWD on trees. Empirical results indicate that the method achieves similar or better clustering accuracy compared to Sinkhorn singular vectors (SSV) while maintaining much faster runtimes.

This paper is the improved version of the workshop paper “Unsupervised Ground Metric Learning with Tree Wasserstein Distance”. The primary innovation of this work is adding recursive basis set computation for tree-based WSV.

### Strengths
1. The paper offers rigorous theoretical support for the TWD method, with proofs on the uniqueness and existence of solutions within specific tree configurations. 
2. The empirical results are presented clearly, with comparative metrics that directly illustrate the computational runtime saving and clustering performance.
3. The paper provides a solid background review on optimal transport theory and the tree-Wasserstein distance.

### Weaknesses
1. Although the ClusterTree algorithm plays a significant role in the tree structure initialization, there is limited background provided on how it operates, what assumptions it makes, or its typical applications. I reviewed both references—Le et al. (2019) and Indyk & Thaper (2003)—but did not find any mention of a ClusterTree. Could the author be referring to the ‘Partition_Tree_Metric’ described in Le et al. (2019)?
2. The algorithm section mentions differences in handling ‘large’ and ‘small’ datasets but does not specify the boundary between the two. What happens if either m or n is very large while the other meets the ‘small’ criteria?
3. The paper’s notation is sometimes inconsistent, making it challenging to reference equations or terms precisely. For example, on line 201, a_i and a_j are bold, but on line 205, a is not bold. On line 211, what does the cost matrix B represent?
4. Figure 3 could be better organized. The paper does not provide a comparison of how other metrics perform on these datasets.
5. Line 557, The URL is invalid.

### Questions
Refer to weaknesses.

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
3

### Summary
The authors apply the unsupervised metric learning algorithm from Huizing et al. (2022) (which used Wasserstein and entropy regularized Wasserstein) to learn ground metrics for spaces of histograms based on the Tree Wasserstein distance.

### Strengths
The idea is sound, and there is a significant element of originality, especially in the development of the algorithm in Appendix C.

### Weaknesses
The paper seems rushed overall, there are a large number of typos and a number of results that should have been presented as Theorems are merely stated informally.

1. l.238 – 241 these statements require a proof. Especially the convergence, since it was already somewhat delicate in Huizing et al. (2022).
2. l.263 “Wasserstein” -> “Tree Wasserstein” or else requires a proof.
3. I find Theorem 2.2 hard to interpret. Can the authors rephrase the interpretation in the next paragraph (l. 256-l.260) as a Theorem and include the current Theorem 2.2 as a Lemma?
4. l.354 The previous work used n=100, m=80 and not n=80, m=60 as this paper states.
5. l.367 Why are you using a different metric (Frobenius norm) than the original work? How does your method compare when using the same metric d_h(B, B’) = ||log(B/B’)||_V?

### Questions
List of typos. I suggest that the authors give the manuscript a thorough proof-reading.

* l.39 confusing, why is the optimal sample distance the Wasserstein distance?
* l.106 in what sense is SWD a “geometric embedding”?
* l.111 definition of pi?
* l.116 what does “good approximation” mean here?
* l.140 precise the meaning of normalized.
* l.142 distribution -> “probability distribution”
* l.150 R and tau are undefined
* l.150 not clear what \Phi_A is. Huizing et al. (2022) describe Phi_A as ”lifts a ground metric to a pairwise distance matrix”. The authors need to explain the definition of Phi_A.
* l.151 equivalence to (3) assumes tau = 0. The authors need to be a bit more careful in their recap of Huizing et al. (2022).
* l.157 not sure that the remark in parentheses is correct, according to Huizing et al. (2022) a single Wasserstein iteration is n^3 log n. Also how many distances do we compute when we compute “m^2, n^2 W. distances”? Is is m^2 + n^2, m^2 * n^2, something else?
* l.201 W_{T_B} instead of W_B ?
* l.205 z_i^(A) undefined
* l.205 Z^{(B)} or Z_B?
* l.214 W_{TB} -> W_{T_B}

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
4

### Summary
The paper introduces Tree-WSV, integrating TWD with WSV (Huizing et al., 2022).

### Strengths
The proposed method demonstrates efficiency compared to WSV and SSV presented in Huizing et al. (2022).

### Weaknesses
- The authors missed important related works [1, 2, 3, 4, 5] that consider the relationships between the samples are informed by the relationships between the columns, and vice versa, for Wasserstein distance and specifically in tree-related settings [2,5]. Specifically, the setup of randomly permuted dataset rows and columns in the toy datasets was one of the important tasks in these works.
- The explanation of how the Wasserstein distance can serve as a tree distance in Proposition 2.1 is unclear. It’s also not evident whether there exists a tree for which the tree distance would correspond to a Wasserstein distance.
- The real-world application is restricted to single-cell RNA sequencing data, despite the introduction citing various data types as motivation.
- The experiment section includes only two competing methods without considering the baselines used in WSV or other distance metric learning approaches.
- $n$ is not yet defined in the Abstract. It’s unclear what does it represent here. In addition, the computational complexity for WSV reported by the author differs from that presented in the original WSV paper.
- In Section 1.1, the authors mention the sliced Wasserstein distance. However, it’s unclear what it is here and how SWD is a special case of TWD. Also, it’s unclear why SWD is not considered an alternative for efficient computation for Wasserstein distance in the WSV framework.
- It’s unclear $\mathbf{w}$ and $\mathbf{Z}$ in line 109 are vector or matrix. Additionally, it’s unclear what is the connection between $\mathbf{x}$, $\mathbf{y}$, $\mu$, and $\nu$ in line 111.
- The notation of Wasserstein distance $\\mathcal{W}\_C$ in line 92 and the notation of TWD $\\mathcal{W}\_\\mathcal{T}$ in line 111 are confusing. In the formal $C$ is a pairwise distance matrix, and in the later $\\mathcal{T}$ is a tree.
- It’s unclear what are $\mathbf{a}$ and $\mathbf{b}$ in Eq.(2). In addition, it’s unclear what size($\mathbf{w}$) represents in line 116.
- More details are needed for how “TWD is a good approximation of the full 1-Wasserstein distance”. What do the authors refer to as “approximation”? What is the relation between TWD and full 1-Wasserstein distance?
- It’s unclear what is $R$ in line 151.
- The authors keep using the term “basis” throughout the paper, e.g.,  tree parameter basis set, the set of basis vector, matrix’s basis set. However, it’s unclear what does it represent in these contexts.
- It’s unclear what "full WSV" represents. Is it different from WSV?
- It’s unclear why the size of the vectors of edge weights is less than the number of nodes in the tree in Section 2.1.
- It’s unclear what $\mathcal{W}_A$ and $\mathcal{W}_B$ represent in Proposition 2.1. Here, $A$ and $B$ are the sets, which are not pairwise distance matrix nor tree as in previous notations.
- It’s unclear what $\circ$ denote in Proposition 2.1. Also, it’s unclear what are $\\lambda\_A$, $\\mathbf{z\_i}\^{(\\mathbf{A})}$, and $\\mathbf{Z}\^{\\mathbf{B}}$.
- The proof for Proposition 2.1 in Appendix A is very hard to follow. The notations used are not consistent with those used in the main texts. The newly defined notation is very dense. Also, it’s unclear what are $\\mathbf{W}\_{\\mathbf{A}}$, $\\Phi\_{\\mathbf{S}}$
- More details and explanations are needed for how Theorem 2.2 supports unique and non-zero solutions in Proposition 2.1.
- Algorithm 1 is very hard to follow. For example, it’s unclear what the line “$\\mathbf{Z}\_{\\mathbf{diff}}$ …… “ represents. It’s unclear what are $\\mathbf{A}\_{leaf}$, $\\mathbf{w}\_{\\mathbf{B}}$(prev)
- The reference style is inconsistent: some entries lack publisher information, some links are not official paper links, and "Wasserstein" is sometimes written with a lowercase "w."
- The notation style is inconsistent: vectors and matrices are inconsistently represented, with a mix of boldface and regular type. The notation for the tree parameter in Section 1.1 is different than in Section 2.1

## Minor
- Missing “-” for tree-Wasserstein distances in line 077 and line 102
- The acronym "OT" is used without being defined first
- Missing punctuations in equations
- It’s unclear what is $TB$ in line 214

[1] Ankenman, J.I., 2014. Geometry and analysis of dual networks on questionnaires. Yale University.

[2] Mishne, G., Talmon, R., Cohen, I., Coifman, R. R., & Kluger, Y. (2017). Data-driven tree transforms and metrics. IEEE transactions on signal and information processing over networks, 4(3), 451-466.

[3] Gavish, M. and Coifman, R.R., 2012. Sampling, denoising and compression of matrices by coherent matrix organization. Applied and Computational Harmonic Analysis, 33(3), pp.354-369.

[4] Shahid, N., Perraudin, N., Kalofolias, V., Puy, G. and Vandergheynst, P., 2016. Fast robust PCA on graphs. IEEE Journal of Selected Topics in Signal Processing, 10(4), pp.740-756.

[5] Yair, O., Talmon, R., Coifman, R.R. and Kevrekidis, I.G., 2017. Reconstruction of normal forms by learning informed observation geometries from data. Proceedings of the National Academy of Sciences, 114(38), pp.E7865-E7874.

### Questions
- In line 194, what does “$a_i$ defined as for WSV” mean?
- What is the size of $\mathbf{U}$ in line 259?
- How does the choice of tree construction affect the proposed method? If a different tree construction method were used instead of ClusterTree, would it impact the method's outcome?
- How to decide whether the algorithm reaches convergence in Algorithm 1?
- How fast does the proposed algorithm converge? How does the performance change across the iteration?
- What are the hyperparameters of the proposed method? How sensitive are they in the experiments?
- What is the Euclidean metric baseline in Table 1?
- Why is WSV not considered as a baseline in Table 1?
- Why are other TWD methods not considered baselines in Table 1?

### Soundness
3

### Presentation
1

### Contribution
2
