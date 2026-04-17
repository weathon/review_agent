# sparseGeoHOPCA: A Geometric Solution to Sparse Higher-Order PCA Without Covariance Estimation

- Decision: Reject
- Scores: 2, 2, 6, 4, 2

## Abstract
This paper proposes sparseGeoHOPCA, a geometric framework for sparse higher-order principal component analysis (SHOPCA). 
 The method unfolds the input tensor along each mode and reformulates the resulting subproblems as binary linear programs, transforming the nonconvex sparse objective into a tractable geometric form. 
 This eliminates covariance estimation and iterative deflation, leading to improved efficiency and interpretability in high-dimensional and unbalanced settings. 
 Theoretical equivalence with the original SHOPCA formulation is established, and error bounds linked to PCA residuals are derived, providing data-dependent guarantees. 
 The algorithm has total complexity $O(\sum_{n=1}^{N} (k_n^3 + J_n k_n^2))$  per  iteration, scaling linearly with tensor size.
 Extensive experiments demonstrate accurate sparse support recovery, stable classification under 10× compression, high-quality ImageNet reconstruction, and semantic reduction, highlighting robustness and versatility.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This manuscript proposese sparseGeoHOPCA for sparse higher-order principal component analysis on tensors. It incooperates a geometry-aware method in Tucker decomposition with sparse factors. Several theoretical properties of the algorithm are analyzed. Experiments on synthetic data, MNIST, StarPlus fMRI, and several images from ImageNet show the effectiveness.

### Strengths
1. A new model for sparse tensor PCA incooperating geometry-aware method. 

2. A algorithm for solving the model with theoretical analysis on several perspects.

### Weaknesses
1. The model is a simple extension of the geometry-aware Sparse PCA by Bertsimas & Kitane (2022) to the Tucker setting, which limits it novelty in modeling. The authors even did not explicitly refer to Bertsimas & Kitane (2022) in the modeling section. 

2. The algorithm and the theoretical analysis are also straight-forward extensions of Tucker decomposition papers and geometry-aware Sparse PCA. Thus, the novelty in algorithm design and theoretical analysis are also limited. 

3. The experiments are very weak. On synthetic data, the rank is 1 which is very special. The datasize is also very limited. The authors only compares with sparsePCAChan and sparsePCABD.

### Questions
1. Modeling: The modeling novelty beyond Bertsimas & Kitane (2022) and unfolding. 

2. Theorey: The novelty in theoretical analysis beyond Bertsimas & Kitane (2022)  and traditional Tucker-based methods.

3. Experiments: Why not test on rank greater than 1? Why not compare with more baselines?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposes sparseGeoHOPCA, a geometry-aware framework for sparse higher-order PCA that reformulates the nonconvex sparse optimization into tractable geometric subproblems.

### Strengths
1. The paper provides a worst-case upper bound for the model.

### Weaknesses
1.   The contribution of this work is incremental. Both the extension of sparse PCA to tensors [1] and  the use of column selection [2], have been well explored in prior literature.

2. The experiments are underdeveloped. The paper compares against only two baseline methods, and the real-data evaluation includes merely four color images for quantitative comparison. As presented, the empirical section is insufficient to substantiate the proposed approach.

3. The discussion on recent tensor recovery methods within the past five years remains inadequate.



[1] Sparse higher-order principal components analysis,Genevera I. Allen

[2] Exact top-k feature selection via l20-norm constraint, C. Xiao, F. Nie, and H. Huang

### Questions
See the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a geometric framework for sparse higher-order principal component analysis without covariance estimation. Inspired by the study (Bertsimas & Kitane, 2022), the proposed method unfolds the input tensor along each mode and reformulates the resulting subproblems as binary linear programs, which can be efficiently solved. Both rigorous theoretical analysis and extensive experiments are carried out to demonstrate the merits of the proposed framework.

### Strengths
1. A novel geometric framework is  introduced to deal with the SHOPCA problem, which has not been considered in the literature.
2. The proposed method could significantly reduce both computational and memory overhead in high-dimensional regimes.
3.  Both rigorous theoretical analysis and extensive experiments are provided to support the merits of the proposed framework.

### Weaknesses
1. Considering that the geometry-aware method was originally introduced for matrix sparse PCA, the novelty of this work seems to be limited.
2. For practical implementation, both the tensor ranks and the sparsity level of the proposed need to be carefully tuned.
3. It appears that the proposed framework is limited to handling cases with Gaussian noise.

### Questions
1. How to determine the tensor ranks and the sparsity level in real applications?
2. How about the performance of the proposed method beyond Gaussian noise settings?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes sparseGeoHOPCA, a geometry-aware framework for sparse higher-order PCA (SHOPCA). The key idea is to unfold the tensor by mode, reduce each mode’s subproblem to a sparse matrix task, and then reformulate that task as a binary linear optimization (BLO) with geometric exclusion constraints; after selecting supports, PCA/SVD on the chosen columns builds the factors and core.

### Strengths
1.The framework sidesteps constructing huge covariance matrices in high-dimensional, unbalanced regimes—practically useful.
2.Theorems provide a worst-case reconstruction bound using PCA residuals; the statement for the overall error bound is straightforward to implement.

### Weaknesses
1.Thin novelty relative to geometric sparse PCA on matrices. The core technical move—geometry-driven column selection via BLO—is essentially transplanted to mode-unfolded matrices; the “equivalence” and bounding arguments live at the matrix subproblem layer and do not advance guarantees for the global SHOPCA objective.
2. The worst-case bounds are feasibility-type approximations (via PCA residuals). There is no support-recovery or statistical-consistency guarantee, and the “sum of residual energies” bound ignores cross-mode coupling and the core.

### Questions
1.Please compare against modern tensor sparse methods (not only matrix SPCA and thresholded Tucker) at larger scale, reporting time and memory on matched hardware. The current main-text image study is too small to substantiate broad claims.
2.Can you prove support-recovery or error-rate bounds (e.g., minimum signal conditions) in a controlled setting (single-rank, single sparse mode)? Current results don’t ensure correct sparsity identification.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces sparseGeoHOPCA, a framework designed to address the Sparse Higher-Order Principal Component Analysis (SHOPCA) problem. The authors correctly identify that existing SHOPCA methods are often bottlenecked by the explicit computation and manipulation of large covariance matrices, which is computationally prohibitive in high-dimensional, unbalanced settings.

The core proposal is to unfold the input tensor along each mode and reformulate the resulting sparse matrix PCA subproblems from a "geometric perspective."

### Strengths
The paper tackles a well-defined and highly relevant problem. Efficiently computing sparse, interpretable components for tensor data is a critical task in many machine-learning domains, and the non-convex nature of SHOPCA makes it a challenging research frontier. The primary motivation bypassing the covariance matrix bottleneck is clear and well-supported by prior literature. A "covariance-free" approach is a highly desirable contribution to the field.

### Weaknesses
The manuscript, while promising in its motivation, suffers from a fundamental logical contradiction in its core claims, as well as an incomplete analysis and a critical lack of supporting experimental evidence.

1. The authors' entire argument rests on a flawed premise. They motivate the work by stating that SHOPCA is NP-hard, but then claim their method transforms this into "tractable" geometric subproblems, which are then explicitly identified as Binary Linear Programs (BLPs).

This is a contradiction. Binary Linear Programming is a classic, well-known NP-hard optimization problem. One cannot claim to have found a "tractable" solution by reformulating one NP-hard problem into another. This ambiguity is present throughout the paper.

2. The claimed complexity of O(P(k^3 + Jk^2)) is highly suspect and appears to be incomplete. This analysis seems to only account for the tensor preparation and solution construction phases. It conspicuously omits the computational cost of solving the N mode-wise BLPs.

The authors must provide a complete complexity analysis that includes the cost of the BLP solver or the approximation algorithm used. The cost of solving a BLP is, in the worst case, exponential. The claim of linear scaling with tensor size P is unproven and likely incorrect until the cost of the "geometric solver" stage is fully incorporated.

3. The paper's novelty is repeatedly framed as a "geometric perspective". However, this term is never formally defined. The text and Figure 1 immediately pivot from "geometric solver" to "binary linear program."

What makes this formulation "geometric"? Does the method involve convex hulls, projections, or other specific geometric operations that are being abstracted away? Or is "geometric" simply a non-standard descriptor for the BLP reformulation? This ambiguity obscures the core technical contribution and must be clarified.

4. The paper's raison d'être is the computational and memory efficiency gained by avoiding covariance matrices. However, the summary of experimental results focuses almost exclusively on accuracy and robustness (support recovery, classification).

The experimental section must include direct, quantitative comparisons of wall-clock runtime and peak memory usage against the very baselines the paper critiques (e.g., Allen 2012, Lai et al. 2014, etc.). These comparisons must be conducted in the high-dimensional, unbalanced settings where the method claims to have "notable advantages". Without this data, the paper's primary claims of efficiency are entirely unsubstantiated.

### Questions
The paper addresses an important and well-motivated problem. However, the manuscript in its current form is not suitable for publication. It is built on a central logical contradiction: claiming to solve an NP-hard problem by reformulating it into another NP-hard problem, which is then called "tractable." This core flaw, combined with an incomplete complexity analysis and a complete lack of experimental evidence for the claimed efficiency gains, invalidates the paper's primary contributions. A Major Revision is required to fundamentally restructure the paper's claims, clarify its methodology (exact vs. approximate), and provide the necessary experimental data to support its (currently unsubstantiated) claims of computational superiority.

### Soundness
2

### Presentation
2

### Contribution
2
