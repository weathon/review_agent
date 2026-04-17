# An efficient, provably optimal algorithm for the 0-1 loss linear classification problem

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 4, 6

## Abstract
Algorithms for solving the linear classification problem have a long history, dating back at least to 1936 with linear discriminant analysis.
For linearly separable data, many algorithms can obtain the exact
solution to the corresponding 0-1 loss classification problem efficiently,
but for data which is not linearly separable, it has been shown that
this problem, in full generality, is NP-hard. Alternative approaches
all involve approximations of some kind, such as the use of surrogates
for the 0-1 loss (for example, the hinge or logistic loss), none of
which can be guaranteed to solve the problem exactly. Finding an efficient,
rigorously proven algorithm for obtaining an exact (i.e., globally
optimal) solution to the 0-1 loss linear classification problem remains
an open problem. 

By analyzing the combinatorial and incidence relations between hyperplanes and data points, we derive a rigorous construction algorithm, incremental cell enumeration (ICE),
that can solve the 0-1 loss classification problem exactly in $O\left(N^{D+1}\right)$---exponential
in the data dimension $D$. To the best of our knowledge, this is
the first standalone algorithm---one that does not rely on general-purpose
solvers---with rigorously proven guarantees for this problem. Moreover,
we further generalize ICE to address the polynomial hypersurface classification
problem in $O\left(N^{G+1}\right)$ time, where $G$ is determined by both the data dimension $D$ and the polynomial degree
$K$ defining the hypersurface. The correctness of our algorithm is
proved by the use of tools from the theory of hyperplane arrangements and
oriented matroids.

We demonstrate the effectiveness of our algorithm on real-world  datasets, achieving optimal training accuracy for small-scale  datasets and higher test accuracy on most  datasets. Furthermore, our complexity analysis shows that the ICE algorithm offers superior computational efficiency compared with state-of-the-art branch-and-bound algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the traditional NP-hard 0-1 loss linear classification problem by introducing a new algorithm called Incremental Cell Enumeration (ICE). The authors first introduce a new theorem showing that to find a globally optimal classifier, it suffices to consider hyperplanes defined by any combination of $D$ data points (where $D$ is the feature dimension).  

This result formalizes an observation by Nguyen & Sanner (2013) and explains why their combinatorial search could find the exact solution. The authors further introduce a “symmetry fusion” theorem (Theorem 5) to avoid redundant checks of both orientations. The paper also extends the theory to non-linear classification by polynomial hypersurfaces. The authors introduce a practical algorithm based on these insights. Instead of naive enumeration of all $D$-combinations of $N$ points, which is computationally inefficient, the paper improves this through an incremental combination generation strategy that efficiently traverses the space of combinations without heavy repetition.

The paper demonstrates ICE on several real-world datasets, showing that it achieves higher training accuracy than SVM, LDA, and logistic regression, and requires significantly less computational time than SOTA BnB algorithms.

### Strengths
- The paper highlights an interesting conflict from previous works' observation and provides new theoretical insight to address such conflicts (Theorem 3), bridging the gaps between three combinatorial analysis works from past literature. This has a high potential impact on both theory and practice.
- The paper introduces a novel Symmetry Fusion Theorem how the misclassification error of a hyperplane’s reverse orientation can be obtained analytically.
- The majority of claims in this paper are supported with formal mathematical proof or a lemma, and the complexity analysis is transparent: the paper clearly derives the worst-case runtime and memory use.
- The paper practical algorithmic solution based on these insights for small-to-moderate problem sizes, with experiments to demonstrate that ICE offers higher training accuracy than other surrogate optimization methods and lower computational time compared to the SOTA BnB method.

### Weaknesses
Some weaknesses in the methodology need more clarification:
- In proof, the authors assume that increasing the sign‐difference increases the 0–1 error (line 718). However, it's unclear to me how sign differences between sign vectors of dual cells can directly translate into classification errors on true labels.  For example, a candidate $g$ differing in three signs might flip one of the errors to correct and flip two previously correct points to wrong. Thus, flipping signs could increase or decrease misclassification depending on the labels. The authors did not make an alignment between the sign and the true label. Thus, the "non-conformal vertices must have strictly higher 0–1 loss" is not correctly proven, which leads to a flaw in the major claim made in this paper.

 - Almost every result assumes the data is in "general position". This is a strong assumption that seems like it would often be violated in Real datasets (e.g., consider duplicate points or points that only differ by a very small epsilon). It's not clear whether ICE would still guarantee good solutions in this case.

- The ICE algorithm has worst-case polynomial time in N only for fixed D, and it remains exponential in D. The paper also acknowledges that the runtime grows exponentially with D, which is a shortcoming of ICE. Therefore, framing this as a “polynomial” algorithm is misleading, since $D$ is part of the input dimension.

- There is no rationale for the initial SVM ordering in Algorithm 1 is heuristic and does not guarantee any bound; it is not analyzed theoretically at all. I may have missed some part, but a better clarification can be helpful.

Also, the experimental results can be improved:

- **SOTA Baseline** Nguyen & Sanner (2013) proposed multiple algorithms, not just a straightforward BnB. The paper only discusses the basic BnB worst-case. A more comprehensive benchmark against all relevant methods from Nguyen & Sanner (2013) (like prioritized combinatorial search, PCS, and other heuristics) would give a better understanding of ICE’s performance.  The release their code as a zip file I found linked on this [page](https://users.cecs.anu.edu.au/~ssanner/publications.html).

- **MILP Baseline** The paper omits mixed-integer programming (MILP) baselines that were introduced in the introduction. Prior research has formulated 0-1 loss minimization as an MILP (e.g., Tang et al. 2014 for maximum-margin under 0-1 loss, Brooks 2011), and a MILP solver could serve as an exact baseline on small datasets. Including a MILP-based method in the experiments (even if only feasible for small N, D) would have been informative.

- **Missing Evaluation** The paper only evaluates the runtime comparison between the SOTA BnB method, but not its accuracy. Similarly, the paper evaluates only the accuracy of other methods, but not their runtime complexity. The paper does not provide a valid reason why these aspects were excluded from the experiments.

- **Generalization** Although the authors claim that optimal 0-1 loss often generalizes better than suboptimal solutions, this is only demonstrated through observations from cross-validation. In these small datasets, the optimal linear model did well on the test set, but is that always true? What if there’s label noise? The experimental evaluation does not explicitly test robustness to label noise or outliers. Also, class balance is not explicitly discussed, and while some UCI datasets may have imbalanced classes, the authors did not highlight performance under severe class imbalance.

- **Dimensionality of the Data** The paper does not report empirical runtime as D grows. For the more practical usage claimed in the title, the authors should have included synthetic analyses of ICE’s performance against existing methods across different numbers of D.
- **Non-linear** The authors claim that the ICE algorithm is generalizable to finding optimal 0-1 loss classifiers in non-linear hypothesis spaces. However, the paper does not demonstrate any experiments for $K>1$. All empirical results are for the linear case.

**Reference**

Tang, Y. et al. (2014). Mixed-integer programming for 0-1 loss classification

Brooks, J. Paul (2011). "Support vector machines with the ramp loss and the hard margin loss.

### Questions
- In Algorithm 1, the input data points are reordered by an initial SVM weight vector ($w^$) and sorted by $|w^{\top}x|$. Could the authors clarify the purpose of this step?

- How does ICE compare against Nguyen & Sanner’s other proposed algorithms? How does ICE compare against MILP-based formulations?

- Fixing or clarifying the rationale for these partial comparisons: runtime-only versus BnB and accuracy-only versus surrogates, would help assess whether any conclusions about performance tradeoffs are generalizable.

- Empirical results on scaling with D, even in synthetic settings, would clarify the claim of its usability.

- Additional insight into how ICE handles label noise or imbalance would help evaluate whether the generalization claims hold in a more realistic setting.

- Experiment in ICE in polynomial feature spaces (e.g., $K = 2$) would better understand the method’s contribution and whether non-linear extensions are viable in practice.

- The theoretical results assume the data has no special degeneracies like more than $D$ points exactly on one hyperplane. In practice, datasets might violate this (e.g., duplicates or correlated features). How does the ICE algorithm handle such cases?

### Soundness
2

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
In this paper, by analyzing the combinatorial and incidence relations between hyperplanes and data  points, the authors derive a rigorous construction algorithm, incremental cell enumeration (ICE), that can solve the 0-1 loss classification problem exactly in $O(N^D)$—exponential in the data dimension D. ICE is then generalized  to address the polynomial hypersurface classification problem.

The effectiveness of ICE is demonstrated on UCI datasets, achieving optimal training accuracy for small-scale datasets and higher test accuracy on most datasets. Furthermore, the complexity analysis shows that the ICE algorithm offers superior computational efficiency compared with state-of-the-art BnB algorithm.

### Strengths
1. Sound theoretical results.
2. Extension to polynomial hypersurface classification problem using K-tuple Veronese embedding.
3. Better training accuracy than baselines.
4. Much faster than BnB in the worst-case.

### Weaknesses
1. The complexity of ICE is exponential, which makes it unsuitable for large-scale or high-dimensional problems. 
2.  The sequential generator in the ICE algorithm, which is core of ICE that enumerates all liner classification decision hyperplanes, is the
 introduced by He&Little(2025).
3. There are no experimental evaluations for nonlinear cases (K>1) for the extension to polynomial hypersurface classification problem.
4. line 125, "which rigorously explains why the PCS algorithm of for solving the linear classification problem" is redundant.

### Questions
1. In Table 2, the "training accuracy" should be "testing accuracy"?
2. The testing accuracy of harder problems, such as Ai4i etc., is given in Table 2. Why are the training accuracy of them not reported?

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
4

### Summary
The paper presents Incremental Cell Enumeration (ICE), an algorithm that exactly solves the 0–1 loss linear classification problem. The authors build on results from combinatorial geometry and oriented matroid theory to provide a constructive, provably correct enumeration method for all linear dichotomies in $O(N^D)$ time. They also extend the approach to polynomial hypersurfaces via Veronese embeddings.
Empirical results on real datasets show that ICE achieves globally optimal training accuracy and often exhibits good generalization performance, while outperforming prior exact algorithms such as branch-and-bound in runtime.

### Strengths
1. Theoretical rigor and clarity: The paper offers a clean, self-contained, and mathematically rigorous development of the theory behind exact 0–1 loss classification. The exposition is clear and logically structured.

2. Conceptual contribution: Establishing a provably correct standalone algorithm for exact 0–1 loss optimization is intellectually meaningful and addresses a long-standing question in the theory of linear classifiers.

3. Technical soundness: The results and proofs appear correct and well motivated. The connections between hyperplane arrangements, dichotomies, and duality are carefully worked out.

4. Empirical validation: Experiments are convincing within the scope of small- to medium-scale datasets and demonstrate that the exact solutions generalize well.

5. Overall presentation: The paper is very well written, with precise notation and a transparent link between theoretical and empirical parts.

### Weaknesses
1. Scope and scalability: The proposed method has exponential dependence on the feature dimension D. Although this is inherent to the problem, it limits practical applicability. The discussion of computational limits and possible parallelization or approximation schemes could be expanded.

2. Empirical breadth: The experimental section, while solid, remains narrow. It would strengthen the work to include more challenging benchmarks or comparisons to small-scale MILP-based or neural surrogate solvers.

3. Accessibility of theory: The theoretical sections are dense, and readers outside geometry or matroid theory may struggle. A concise figure or schematic illustrating the relationships between the primal and dual spaces could improve readability.

### Questions
1. How does ICE scale beyond low-dimensional data (e.g., D > 10) in practice?
2. Could the framework be extended to multi-class or structured output settings?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an algorithm called incremental cell enumeration (ICE) which solves the classification problem directly with the 0-1 loss. It is claimed that the proposed algorithm is significantly faster than the existing PCS-type algorithm, and outperforms the baseline of linear classifiers such as the SVM and logistic regression in terms of classification accuracy. The proposed method is claimed to have exact optimality supported by a geometric interpretation. The implementation of the ICE algorithm is standalone and does not depend on mixed integer solvers.

### Strengths
1. The proposed method has clear geometric framing. The dual view has an intuitive explanation of the candidate enumeration.
2. The reported speedup over PCS is significant.
3. The paper has a self-contained algorithm, so it is easy to run in practice.

### Weaknesses
1. The paper has limited practice use. The method is restricted to very low dimensions and scalability is unclear.
2. Although the paper claims the exact optimality, it is not clearly shown that the global 0-1 minimizer must be included on some D-point hyperplane so the ICE algorithm miss still miss that.
3. In Table 2, many rows show test accuracy better than train, which is atypical. Hence performance claims seems to be not trustworthy.

### Questions
1.  As stated in Theorem 5, the 0-1 loss is calculated as $l$ and $N - l - D$. Does it mean that the $D$ points where the hyperplane goes through are counted as correctly classified? 
2. How was SVM tuned (grid over the parameter $C$) and how is the class weight set?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents a novel, provably optimal algorithm called Incremental Cell Enumeration (ICE) for exactly solving the NP-hard 0-1 loss linear classification problem. The authors derive the algorithm by analyzing combinatorial and incidence relations through the lens of hyperplane arrangements and point-hyperplane duality, resulting in a method with a worst-case complexity of $O(N^{D})$, which is polynomial in the number of data points (N) but exponential in dimension (D). The paper formally proves that the globally optimal solutions lie within the set of hyperplanes defined by D data points and demonstrates ICE's superior computational efficiency and ability to achieve better generalization compared to state-of-the-art approximate and exact Branch-and-Bound methods on real-world datasets.

### Strengths
This paper introduces the Incremental Cell Enumeration (ICE) algorithm and makes significant advancements in the field of exact classification, particularly for the challenging 0-1 loss objective. The paper provides clear definitions and a structured narrative from the problem definition to the complex geometrical analysis and final algorithmic implementation. The empirical results provide a critical empirical insight on generalization, showing that achieving optimal training accuracy often leads to stronger generalization (higher test accuracy) compared to approximate methods.

### Weaknesses
1. The paper successfully generalizes the theoretical framework using the K-tuple Veronese embedding to solve the 0-1 loss polynomial hypersurface classification problem. However, the empirical section explicitly states that experiments were restricted to the linear case (K=1). To fully support the significance and utility of the theoretical extension, empirical results for low-dimensional data (D≤3) with K>1 must be included.

2. The tractability is limited to low dimensions which undermines real-world applicability. It would be preferable to incorporate a formal analysis (or reference) that provides a rigorous approximation bound for the 0-1 loss achieved by the ICE-coreset relative to the true global optimum. This would bridge the gap between the theoretical exactness of ICE and the practical approximation needed for real-world data sizes and dimensions.

### Questions
1. The reported worst-case time complexity seems to vary slightly across the paper: it is initially stated as $O(N^{D})$, later as $O(N^{D+1})$, and the general hypersurface complexity in Algorithm 1 is $O(N^{G}×G^{3})$. It would be better to standardize and clarify the exact worst-case run-time complexity for the linear case (K=1).

2. The paper convincingly demonstrates superior theoretical and empirical performance for linear classification (K=1), but how does it extend to the polynomial hypersurface classification (K>1), and how does the high-dimensional results rely on an approximation wrapper? To demonstrate the practical viability of this major theoretical contribution, please provide empirical validation for small D and K>1 (e.g., quadratic boundaries, K=2, on D=2 or D=3 data).

### Soundness
3

### Presentation
3

### Contribution
3
