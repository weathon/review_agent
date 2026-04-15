# Efficient Sparse PCA via Block-Diagonalization

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 5, 6

## Abstract
Sparse Principal Component Analysis (Sparse PCA) is a pivotal tool in data analysis and dimensionality reduction. However, Sparse PCA is a challenging problem in both theory and practice: it is known to be NP-hard and current exact methods generally require exponential runtime. In this paper, we propose a novel framework to efficiently approximate Sparse PCA by (i) approximating the  general input covariance matrix with a re-sorted block-diagonal matrix, (ii) solving the Sparse PCA sub-problem in each block, and (iii) reconstructing the solution to the original problem. Our framework is simple and powerful: it can leverage any off-the-shelf Sparse PCA algorithm and achieve significant computational speedups, with a minor additive error that is linear in the approximation error of the block-diagonal matrix. Suppose $g(k, d)$ is the runtime of an algorithm (approximately) solving Sparse PCA in dimension $d$ and with sparsity constant $k$. Our framework, when integrated with this algorithm, reduces the runtime to $\mathcal{O}\left(\frac{d}{d^\star} \cdot g(k, d^\star) + d^2\right)$, where $d^\star \leq d$ is the largest block size of the block-diagonal matrix. For instance, integrating our framework with the Branch-and-Bound algorithm reduces the complexity from $g(k, d) = \mathcal{O}(k^3\cdot d^k)$ to $\mathcal{O}(k^3\cdot d \cdot (d^\star)^{k-1})$, demonstrating exponential speedups if $d^\star$ is small. We perform large-scale evaluations on many real-world datasets: for exact Sparse PCA algorithm, our method achieves an average speedup factor of 100.50, while maintaining an average approximation error of 0.61%; for approximate Sparse PCA algorithm, our method achieves an average speedup factor of 6.00 and an average approximation error of -0.91%, meaning that our method oftentimes finds better solutions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes to address the one-component sparse PCA (SPCA) problem by first approximating the target covariance matrix by a block-diagonal matrix (by thresholding its entries and permuting rows and columns), and then looking for solutions within each one of the blocks. These smaller SPCA problems associated with each block can be (exactly or approximately) solved by means of existing SPCA algorithms, and are expected to require (potentially much) less compute in comparison with the standard approach. The authors state and prove some results bounding the suboptimality of this scheme (in terms of objective function values) in terms of the infinity norm of the matrix approximation error, the sparsity level and the largest size among the obtained blocks, including in their analysis the case in which an approximate algorithm with additive and multiplicative errors is used. Simulation results are given, comparing the proposed scheme with two other approaches: exact solution by branch-and-bound, and approximate solution by Chan's algorithm (reference Chan et al., 2015 cited by the authors).

### Strengths
S1) The proposed scheme is simple and intuitively well-motivated.

S2) The theoretical results are also fairly simple and support the intuition.

S3) The analysis includes approximate algorithms, which have great practical relevance since the problem is computationally hard.

S4) The simulation results demonstrate a significant speedup in large problem instances, most often with only a small relative suboptimality gap. Hence, the proposed algorithm is clearly useful for practitioners.

### Weaknesses
W1) Some of the technical contents, including the problem formulation, are not properly formalized, with some important aspects completely omitted. Notably, it should be stated that matrix $A$ in (SPCA) is necessarily symmetric positive semidefinite, implying the same for the blocks in the block-diagonal approximation. The absence of this information and the fact that throughout the paper $A$ is simply referred to as an "input matrix" rather than a covariance matrix may mislead the reader into thinking that the problem is more general than it actually is.

W2) The presentation of the simulation results is somewhat superficial, focusing only on presenting and briefly commenting the two quantitative criteria used for comparison, without much discussion or insight into what is going on. Specifically:
- Separate values for the different used $k$ should be reported (see point W2 below).
- It would be interesting to report the threshold value $\varepsilon$ effectively chosen by the algorithm (relatively to the infinity norm of the input matrix), as well as the proportion of zero entries after the thresholding. 
- It would also be interesting to compare the support of the solution obtained by the proposed scheme with that obtained by the baseline methods (e.g., using a Jaccard index).

W3) Reporting average results with respect to $k$ is not a reasonable choice in my opinion, as the statistics of the chosen metrics probably behave very differently for different values of $k$. 

W4) As it is, I don't see the utility of Section 4.1. First, this model is not applied to any of the datasets used in the experiments. This leads one to suspect that in practice it is quite hard to come up with estimates of the parameters required by Algorithm 4. Second, the authors do not even generate synthetic data following such a model (which is one typical use of a model) in order to illustrate the obtained results. In my view results of this kind should be added, or else the contents of 4.1 should be moved to the appendices as they're not really important (or both).

W5) Though generally well-written, the paper lacks some clarity at times, and the notation is not always consistent/clear. In particular:
- The sentence "This result is non-trivial: while the support of an optimal solution could span multiple blocks, we theoretically show that there must exist a block that contains the support of an optimal solution, which guarantees the efficiency of our framework." seems to be contradicatory. I believe that the authors mean the following: *one could think* that the solution could span multiple blocks, but they show this is not true. The same goes for Remark 1.
- What is the difference between $A^\varepsilon$ and $\tilde{A}$ in Theorem 1? It seems that two different symbols are used to denote the same object.
- The constraint $\|x\|_0 \le k$ in the last problem that appears in the proof of Theorem 2 is inocuous, since the size of each block $\tilde{A}_i'$ is at most $k$ anyway. This should be commented.

### Questions
I suggest that the authors take the above stated weaknesses into account to improve their manuscript. Apart from this suggestion:

Q1) How far from the bound implied by Theorem 1 are the measured suboptimality gaps in practice?

Q2) How realistic and how restrictive is the bound on $d^{1-\alpha}$ of Proposition 1? How can it be interpreted? A discussion should be presented on this point, as it is not obvious.

Q3) How hard is it to extend your results to the multi-component case?

### Soundness
3

### Presentation
2

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
This paper presents a block-diagonal-based approach to speed up Sparse PCA, preserving an average approximation error. The authors provide mathematical proof for this algorithm, and extensive testing on real-world datasets demonstrates consistent improvements that align with theoretical predictions.

### Strengths
1. This paper is the first to apply a block-diagonal-based method to the Sparse PCA problem, enhancing the balance between runtime efficiency and average approximation error.  
2. The authors present theoretical analysis, providing guarantees for both approximation error and runtime complexity.  
3. This approach is evaluated across many real-world datasets, demonstrating consistent improvements in runtime and approximation error.

### Weaknesses
1. In your algorithm, you noted that the input matrix could be solved using $p$ sub-matrices, but you did not provide a method to determine $p$ or explain the relationship of $d_i$ (besides stating that $\sum d_i = d$).
2. There is not much discussion about the impact of hyperparameters.
3. In your notation, $p$ in $p$-norm should not represent the same value as $p$ in diag($A_1, \cdots, A_p$).
4. Some descriptions in your Theorem, Proposition, and Proof are not rigorous.
5. Some proofs are not completely correct.

### Questions
1. In your algorithm, you noted that the input matrix could be solved using $p$ sub-matrices. Could you please give more details on determining $p$ or explaining the relationship of $d_i$ (besides stating that $\sum d_i = d$)?
2. Could you explain why $|| E ||_\infty$ is the estimation in Model 1?
3. In algorithm 4, why $m=\lfloor (2C+2)d^{1+\alpha}  \rfloor$?
4. I'm curious about the inequality in Line 71. Can you prove it?
5. The proof of Theorem 1 is incomplete. Could you complete it?
6. In the proof of Theorem 3, how do you come up with $k || A-A^\epsilon ||_\infty \leq \epsilon $?
7. Could you prove why the g() function in Eq.(2) is convex?
8. Do you think the inequality in Line 41-45 is too relaxed?
9. Some descriptions are not rigorous. For example, "an absolute constant $c > 0$". Apparently, an absolute value is greater than 0 in this case.
10. Including additional details in the proof of Theorem 4 would improve clarity. For instance, explaining how the running time is derived and whether the relaxation is reasonable would be helpful.
11. I noticed that you frequently use expressions like $ \max \max $ or $ \inf \inf \inf $ in your proof, which appears unprofessional. Could you consider rephrasing these using constraints for clarity?
12. Some sentences are too long for readers to understand. For example, Line 51-53. Could you rephrase them?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper introduces a new framework for efficiently approximating Sparse Principal Component Analysis (Sparse PCA) by transforming the input covariance matrix into a block-diagonal structure, allowing for computationally manageable sub-problems.  The proposed method involves three key steps: creating a block-diagonal approximation of the matrix, solving Sparse PCA for each block, and then reconstructing an approximate solution for the entire matrix. By focusing on smaller blocks, this approach achieves substantial speedups with minimal loss in accuracy. The framework is adaptable, integrating well with existing Sparse PCA algorithms and reducing overall complexity theoretical and emperically.

### Strengths
The idea is well-motivated, and the problem is relevant to the community. Despite the NP-hardness of sparse PCA (SPCA), the authors propose addressing it through matrix block diagonalization. This framework demonstrates advantages in time complexity over traditional methods, both theoretically and empirically. Additionally, the authors discuss how to determine the appropriate SNR threshold, $\epsilon$, within a statistical model $A = \widetilde{A} + E $ using the proposed algorithm.

### Weaknesses
The authors investigate the recovery of the first individual eigenvector and ensure correctness by establishing an upper bound on the gap between the corresponding eigenvalues in Theorem 1. However, the situation changes when considering the principal subspaces of the covariance matrix $\Sigma$, which are spanned by sparse leading eigenvectors. When leading eigenvalues are identical or close to each other, individual eigenvectors may become unidentifiable. Could the analysis in Theorem 1 be extended to handle the aformentioned case?

### Questions
See above.

### Soundness
3

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
4

### Summary
This paper solves approximate Sparse PCA with matrix block-diagonalization. The proposed algorithm has 3 steps:
approximate a given matrix by a block diagonal, 
solve separate problems and then, combine them. 
 
The matrix preparation step re-arranges the rows and columns so that most of the energy of the entries lies in a block diagonal structure and zeros out the rest of the entries. This part is what I think is the hard part to compute and the hardness of this approximation ( possibly harder than Sparse PCA to start with) is one confusing thing. 

The central theoretical innovation of the paper is the following structural property of an SPCA: The support of the optimal solution must always be contained within one block.  I think this is a cool observation (but not some deep theorem) and I have a few questions about it that I discuss later. 

The combination of solutions, after this observation is realized, is clear: you only pick the solution from the correct sub-block, since there are no interaction terms to worry about across blocks. This is cool, and all subsequent running time results and combinations with other algorithms are clear.

Edit after reading replies and discussion: I think the authors sort of addressed my concern, I give a small upgrade in my score

### Strengths
The paper has a cool observation and I think is well written. 
I have some concerns about the complexity of the approximation of a matrix by a block diagonal that I would really like clarified. 
The experimental section is reasonable some it misses additional experiments that I would like to see.

### Weaknesses
I have issues with the complexity of the block-diagonal matrix approximation and also with the experimental section, as I clarify below. 

Also some minor typos here:
Definition 1, I assume Aij and \tilde{A}_ij are the entries of the matrix, would be good to mention. I know its mentioned later in additional notation but would be better to include in this definition. 

typo: On the other hand, there also exist a number of algorithms that takes* polynomial runtime

typo: The axies* are indices of A; 

typo: th3, positive integer, and donote*

### Questions
q1) I was very confused about the matrix preparation part of the algorithm (i.e. best approximation by block -diagonal). I would think that finding the best block-diagonal matrix that approximates a given matrix A (using any reasonable metric of distance) is NP-hard. Are the authors claiming to achieve this in polynomial time for any matrix using their L_inf distance? 
I was a little lost on the notation and didn't understand where the search over all permutations of rows and columns is addressed. 
Please clarify the computational complexity of this problem and how the algorithm gets around checking all row and column permutations. 


q2) I do not see why theorem 2 is so challenging? I think its a cool observation but I think of it simply as: There is no reason for the principal component to put any energy on positions where the off-diagonal entries are zero, since it gets no benefit from that. Therefore it will pick the block with the most energy and put all its l2 mass there. For example if a matrix is just diagonal, it will put all its energy on the largest diagonal entry in absolute value. 

q3) Theorem 2 doesn't really use sparsity anywhere right? Its really a result about PCA for block-diagonal matrices, or do i misunderstand something?

Remark: Approximating matrices with block-diagonal is indeed a very interesting topic.See this interesting question and answer by Gowers on discovering near-block diagonal structure of a matrix by re-ordering and its connections to combinatorics: 
https://mathoverflow.net/questions/68041/showing-block-diagonal-structure-of-matrix-by-reordering
Another flavor is the "Bandwidth minimization problem". There instead of blocks one tries to re-arrange columns and rows so that all the non-zeros are on a few bands around the diagonal. This is known to be np-hard even for 3 bands, see: 
Complexity Results for Bandwidth Minimization
M. R. Garey, R. L. Graham, D. S. Johnson, D. E. Knuth

q4) On experimental evaluation. 
Why don't the authors compare to other algorithms as done in Papailiopoulos et al. ICML 2013 and the algorithms compared from that paper? There is (FullPath) of (d’Aspremont et al., 2007b) and  
the truncated power method (TPower) of (Yuan & Zhang, 2011). I believe Tpower is usually the fastest in practice. 
I think the Chan paper may have the best theoretical performance but it's not clear how it performs in practice. I did not see any performance plots in that paper. 

I'm willing to improve my score if the authors manage to convince me on these concerns.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a novel framework for approximating Sparse PCA by decomposing the original large-scale problem into smaller subproblems through matrix block-diagonalization. The method involves three key steps: (1) transforming the input covariance matrix into a block-diagonal form by thresholding and grouping non-zero entries, (2) applying any existing Sparse PCA algorithm to each block independently, and (3) reconstructing the overall solution from the subproblem results. This approach aims to achieve significant computational speedups while maintaining high solution quality, supported by theoretical guarantees and extensive empirical evaluations on various large-scale datasets.

### Strengths
1. The framework significantly reduces runtime by decomposing the original problem into smaller, more manageable subproblems.

2. The paper provides approximation guarantees and time complexity analyses, ensuring that the method preserves solution quality.

3. Extensive experiments on diverse, large-scale datasets demonstrate the framework’s effectiveness in reducing computational time with minimal approximation errors.

### Weaknesses
1. The core idea of decomposing a large-scale optimization problem into smaller subproblems and then combining their solutions has been previously explored in the literature (e.g., [1,2,3]). The paper does not sufficiently acknowledge or discuss these existing approaches, missing an opportunity to position its contribution within the broader context of optimization techniques.

2. While each subproblem is solved under a $k$-sparsity constraint, it is not explicitly clear how the framework ensures that the combined solution across all subproblems maintains the overall $k$-sparsity required by the Sparse PCA problem. This raises concerns about the validity of the final solution's sparsity, which is crucial for interpretability and efficiency.

[1] "Exact covariance thresholding into connected components for large-scale graphical lasso." Journal of Machine Learning Research, 13(27):781−794, 2012.

[2] "Graphical lasso and thresholding: Equivalence and closed-form solutions." Journal of Machine Learning Research, 20(10):1−44, 2019.

[3] "Learning large-scale MTP2 Gaussian graphical models via bridge-block decomposition." Advances in Neural Information Processing Systems 36:73211-73231, 2023.

### Questions
Please address the concerns outlined in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
