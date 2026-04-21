# Sparse hyperbolic representation learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 3

## Abstract
Minimizing the space complexity of entity representations without the loss of information makes data science procedures computationally efficient and effective.
For the entities with the tree structure, hyperbolic-space-based representation learning (HSBRL) has successfully reduced the space complexity of representations by using low-dimensional space.
Nevertheless, it has not minimized the space complexity of each representation since it has used the same dimension for all representations and has not selected the best dimension for each representation.
This paper, for the first time, constructs a sparse learning scheme to minimize the dimension for each representation in HSBRL.
The most significant difficulty is that we cannot construct a well-defined sparse learning scheme for HSBRL based on a coordinate system since there is no canonical coordinate system that reflects geometric structure perfectly, unlike in linear space.
Forcibly applying a linear sparse learning method on a coordinate system of hyperbolic space causes a non-uniform sparsity.
Another difficulty is that existing Riemannian gradient descent cannot reach a sparse solution since the algorithm oscillates on a non-smooth function, which is essential in sparse learning.
To overcome the above issue, for the first time, we geometrically define the sparseness and sparse regularization in hyperbolic space, to achieve geometrically uniform sparsity.
Also, we propose the first optimization algorithm that can avoid the oscillation problem and obtain sparse representations in hyperbolic space by the geometric shrinkage-thresholding idea.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper researches the sparse representation learning in hyperbolic space. It defines the sparsity of one point geometrically in the Cartan-Hadamard manifold and avoids the difficulty of defining a unique coordinate system in a general manifold. In order to ensure sparsity, this paper introduces a novel sparse regularization term hyperbolic 1-norm for continuous optimization. Further, since the defined optimization problem becomes non-smooth because of the $\ell_1$ norm, the existing Riemannian gradient descent method would oscillate around the sparse solutions on a non-smooth function, while the proposed hyperbolic iterative shrinkage-thresholding algorithm (HISTA) successfully avoids the oscillation issue. Finally, the numerical experiments prove that the proposed HISTA is effective and outperforms the existing hyperbolic-space-based representation learning methods with respect to space complexity and sparse representation quality.

### Strengths
(1) This paper defines the sparsity of a point geometrically in Cartan-Hadamard manifold, which is non-trivial in hyperbolic space.  
(2) This paper introduces a novel sparse regularization term hyperbolic 1-norm on a Cartan-Hadamard with an origin and orthonormal bases, which is non-trivial in a coordinate system of hyperbolic space.  
(3) This paper proposes the HISTA that avoids the oscillation issue that occurs in the existing Riemannian gradient descent method.  
(4) Experimental results indicate that the proposed algorithm outperforms the existing method in terms of oscillation issue, space complexity, and sparse representation quality.

### Weaknesses
(1) The proposed method only works for Cartan-Hadamard manifolds.  
(2) There is no convergence analysis for the proposed HISTA.  
(3) Currently the applications of the proposed sparse learning method are kind of limited.  

Some typos:  
an CHMOO -> a CHMOO;  
Page 9, $\epsilon = 10^3$ -> $\epsilon = 10^{-3}$;  
Page 9, delete "sufficient" in Section 8 Limitation.

### Questions
See Weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper innovatively proposes a sparse learning scheme for hyperbolic representation learning. By defining sparseness and 1-norm via the Cartan-Hadamard manifold, this paper derives a novel hyperbolic iterative shrinkage-thresholding algorithm (HISTA) to obtain sparse representations. With theoretical analysis and experimental verification, HISTA successfully achieves geometrically uniform sparsity and avoids oscillation issues.

### Strengths
1. Ideas presented in this paper have novelty. Existing hyperbolic methods lack the complete statement of sparse learning.
2. Theoretical analysis is organized and clear. Adequate definitions and remarks enhance the credibility of this paper.
3. Limitations and future studies are given from different perspectives, showing the fundamental contributions of this paper in sparse hyperbolic learning.

### Weaknesses
1. There lacks a graphical presentation of experimental results in the main body. Please optimize the article architecture for better readability.
2. Symbols denoted in this paper are ambiguous sometimes. For example, the italic upper letter T represents the tangent space in the preliminary section but represents the final iteration number in Algorithm 1. Please check the uniqueness of notations.
3. Several mistakes.
a) Sections 2, 3, 7 and 8 are missed at the end of the introduction.
b) Why Definition 2 and Example 2 start in new lines? They should be consistent with others.
c) SHP mentioned in Definition 3 should be italic-bolded.
d) There is an incorrect spelling of Poincar\'e at the end of section 6.

### Questions
Please refer to the weaknesses part.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper extends the sparsity regularization framework of Euclidean representations to more general manifolds of nonpositive curvature called Cartan-Hadamard manifold (CHM). CHMs include the Euclidean geometry and hyperbolic geometries. The main idea is to study the orthonormal basis of the tangent space at some point of the manifold and apply l0- or l1-norm regularization via differential geometry tools such as the logarithmic map. Since l1-norm regularization might not converge to sparse solutions in practice, the paper proposes to extend an iterative shrinkage-threshold algorithm to CHMs to solve this practical issue.

### Strengths
Sparsity is an important tool in machine learning to avoid overfitting but has mainly been studied in the Euclidean context. Hyperbolic representations have been shown to outperform Euclidean representations in low-dimensional space. This paper presents an elegant regularization framework to promote sparsity CHMs in general.

The paper is well-written and its motivation is clear. Since the Euclidean space is a CHM, the connection and extension to CHMs are well explained.

### Weaknesses
Although the paper proposes a way to promote sparsity in CHMs, the family of considered Riemannian manifolds is still limited (i.e. complete Riemannian of nonpositive curvature). The paper illustrates a nice example where l1-norm regularization can be extended to some Riemannian manifolds, but it assumes the existence of exponential and logarithmic maps whose closed-form is often unknown in practice. 

 The practical use of the proposed framework in real world applications is also unclear. In particular, Figure 6 in the Appendix shows that no regularization at all is competitive with the proposed method in terms of accuracy (sometimes even better). This limits the practicality of the proposed approach. 

Another limitation is that the paper considers nonparametric embeddings, not neural networks. Although sparsity has been studied for embeddings in the signal processing community, the sparsity criterion in machine learning often focuses on the output representation of linear or nonlinear models such as Support Vector Machines and Multiple Kernel Learning.

### Questions
Could you explain the practical relevance of the approach?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor
