# Reweighted Solutions for Weighted Low Rank Approximation

- Decision: Reject
- Scores: 5, 8, 5, 5

## Abstract
The weighted low rank approximation problem is an important yet computationally challenging primitive with applications ranging from statistical analysis, model compression, and signal processing. To cope with the NP-hardness of this problem, prior work either considers heuristics or bicriteria algorithms to solve this problem. In this work, we introduce a new relaxed solution to the weighted low rank approximation which outputs a matrix that is not necessarily low rank, but can be stored using very few parameters and gives provable approximation guarantees for this problem when the rank matrix has low rank. Our central idea is to use the weight matrix itself to reweight the low rank solution. Our algorithm is extremely simple to implement and achieves remarkable empirical performance in applications to model compression. Our algorithm also gives nearly optimal communication complexity bounds for a natural distributed algorithm  associated with the low rank approximation problem, for which we show matching communication lower bounds. Together, our communication complexity bounds show that the rank of the weight matrix provably parameterizes the communication complexity of weighted low rank approximation. We also obtain the first feature selection guarantees for weighted low rank approximation.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the weighted low rank approximation problem (WLRA) when the weight matrix is also of a low rank. The authors propose a new relaxed solution to this problem which outputs a matrix that can be well stored. As a corollary, the authors also give nearly optimal communication complexity bounds for another distributed problem.

### Strengths
- The studied problem of WLRA is important.
- The proposed algorithm is efficient and can be implemented.
- The writing is clear.

### Weaknesses
- The output of the main algorithm is not guaranteed to be of low rank, which is inconsistent with the original goal of WLRA. It may be better to discuss more about this type of output, specifically, why it can replace a low-rank matrix in practice.
- There is no theoretical guarantee for Algorithm 2, which makes it less interesting.

### Questions
- On page 2 before Algorithm 1, the authors claim that the storage space $O((n+d)rk)$ is nearly optimal for $r=O(1)$. Does there a matching lower bound of space for WLRA when $r=O(1)$?
- Is rank $r$ known in advance? If not, suppose we run Algorithm 1 with $r > rank(W)$, what happens?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposed a new approximate low rank weighted recovery problem, which is very interesting.

### Strengths
1. New $\kappa$ approximate framework
2. Both theoretical and practical algorithms are proposed, which are actually simple to use.
3. Theoretical guarantees are offered to ensure the quality of the solution in certain cases.
4. Experiments conducted are convincing.

### Weaknesses
1. Writing could be clearer with the different notations, and the overall objective the paper wants to achieve.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Weighted low rank approximation (WLRA) is an important and fundamental problem in numerical linear algebra, statistics, and machine learning. Specifically, given two matrices $\mathbf{A}$, $\mathbf{W} \in \mathbb{R}_{\ge 0}$, and a parameter $k$, our goal is to minimize $|| \mathbf{W} \circ (\mathbf{A} - \widetilde{\mathbf{A}})||_F$ subject to $\mathrm{rank}(\widetilde{\mathbf{A}}) \le k$. This paper considers one of its relaxed versions: solving a matrix $\widetilde{\mathbf{A}}$ such that $|| \mathbf{W} \circ (\mathbf{A} - \widetilde{\mathbf{A}}) ||_F \le \kappa \cdot \min _{\mathrm{rank}(\mathbf{A}')\le k} ||\mathbf{W} \circ (\mathbf{A} - \mathbf{A}')||$, which assumes $\mathrm{rank}(\mathbf{W}) = r$ and removes the rank bound for matrix $\widetilde{\mathbf{A}}$.   
For the above problem, this paper proposes a simple algorithm and proves its correctness. As a corollary, this paper obtains the first relative error guarantee for unsupervised feature selection with a weighted $F$-norm objective. In addition, this paper researches the communication complexity for WLRA and gives the almost matched upper bound and lower bound.

### Strengths
(1) This paper proposes a simple algorithm for one relaxed WLRA problem and proves its correctness.  
(2) It extends to unsupervised feature selection with a weighted $F$-norm objective.    
(3) It explores the communication complexity of the WLRA problem and gives the almost matched upper bound and lower bound.  
(4) The experimental results indicate the strengths of the proposed algorithm with respect to the approximation loss and running time, compared with the existing methods.  
(5) This paper is well-written and easy to understand.

### Weaknesses
(1) This paper relaxes the classical WLRA problem with two conditions: 1) removing the low-rank requirement for matrix $\widetilde{\mathbf{A}}$; 2) assuming the weight matrix $\mathbf{W}$ is low rank. Given these two conditions, the problem becomes much easier, and the proposed algorithm is kind of trivial. Furthermore, if one discarded the low-rank requirement for matrix $\widetilde{\mathbf{A}}$, the relaxed WLRA problem would be kind of insignificant.  
(2) The unsupervised feature selection with a weighted $F$-norm objective directly follows Theorem 1.2. Also, although the bounds for communication complexity of WLRA problem are almost tight, the results and processes are kind of straightforward. Therefore, the contributions in this paper are quite limited.  
(3) In Related Work, the following reference is missing. $\textit{Recovery guarantee of weighted low-rank approximation via alternating minimization}$.

### Questions
See Weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In their manuscript, the authors propose an algorithm for the well-known weighted low-rank approximation problem, which is known to be NP-hard, but which has various applications in statistics, signal processing and machine learning. If W is the (non-negative) weight matrix, the authors consider taking the partial SVD of W \circ A, if A is the matrix to be approximated, and then multiply the entrywise inverse of W to the resulting matrix. They provide approximation guarantees for the algorithm as well as a supposedly more practical variant of the algorithm, Algorithm 2, which is meant to avoid a computational pitfall of the original method. Furthermore, they provide an analysis of the resulting communication complexity, and relate that to a lower bound. Finally, they conduct numerical simulations comparing the algorithm's performance on model compression datasets with other state-of-the-art weighted low-rank approximation algorithms. The authors also shed light on the appropriateness of a low-rank assumption on Fisher information matrices in the context of neural network loss functions.

### Strengths
The proposed algorithms is conceptually simple and appears to be new. The approximation guarantee of Theorem 1.2 is reasonable and simple. The experiments suggest that the resulting weighted low-rank approximation is in the range of the state-of-the-art in terms of approximation quality.  The presentation of the results is relatively clear and many relevant papers and methods are cited.

### Weaknesses
The main motivation of the algorithm as well as the theoretical results are tailored to the case where the weight matrix W is low-rank. However, the fundamental problem in this setting is that there is no reason why the entrywise inverse matrix W^{\circ -1} is low-rank, which leads to the necessity of computing a dense matrix in Algorithm 1, as the authors state, making the algorithm rather inpractical in a large-scale setting. With the practical variant of Algorithm 2, only the storage issue of the resulting approximation is mitigated, but not not the fact that W^{\circ -1} needs to be computed in a dense manner, which requires O(nd) of storage in intermediate calculations.
While the communication complexity discussion of Section 1.1.2 is interesting, it is unclear whether and how the algorithm can be implemented efficiently in a distributed manner. While the authors claim that their result is the first about communication complexity for weighted low-rank approximation problems, it seems that communication complexity for such results was already previously discussed, e.g., in Musco, Cameron, Christopher Musco, and David Woodruff. "Simple Heuristics Yield Provable Algorithms for Masked Low-Rank Approximation." _Innovations in Theoretical Computer Science Conference (ITCS 2021)_. 2021.

### Questions
- Can you discuss the time complexity of the resulting algorithms and intermediate space complexity of Algorithm 1 and Algorithm 2?
- Please discuss your result in the context of previous communication complexity results.
- I did not find the code for the timing experiments in the provided code submission. Furthermore, it would be good if the hyper parameter choices of the reference algorithms are provided in the manuscript, e.g. of adam and em.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
