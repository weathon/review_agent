# Pairwise Worst-Case Ratio Analysis for Discriminative Dimensionality Reduction via Minorization-Maximization

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
In this paper, we investigate a novel discriminative dimensionality reduction method based on maximizing the minimum pairwise ratio of between-class to within-class scatter. This objective function enhances class separability by providing critical, adaptive control over the variance within each class pair. The resulting max-min fractional programming problem is non-convex and notoriously challenging to solve. Our key contribution is a provably convergent, two-level iterative algorithm, termed GDMM-QF (generalized Dinkelbach-minorization-maximization for quadratic fractional programs), to find a high-quality solution. The outer loop employs a generalized Dinkelbach-type procedure to transform the fractional program into an equivalent sequence of subtractive-form max-min subproblems. For the inner loop, we develop an efficient minorization-maximization (MM) algorithm that tackles the non-convex subproblem by iteratively solving a simple quadratic program (QP), which we derive from the dual of a convex surrogate. The proposed GDMM-QF framework is computationally efficient, guaranteed to converge, and requires no hyperparameter tuning. Experiments on multiple benchmark datasets confirm the superiority of our method in learning discriminative projections, consistently achieving lower classification error than state-of-the-art alternatives.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work proposed a new linear discriminant type dimensionality reduction algorithm. The main idea is to maximize the worst pairwise separation ratio between different classes. Compared to the many existing variants, the proposed approach takes class dispersion into account. A slight generalization of the Dinkelbach algorithm is proposed, and a linearization based sequential minimization algorithm is proposed to solve each step. Numerical experiments on small datasets were conducted, where the proposed algorithm seems to perform slightly better than the baselines.

### Strengths
- the proposed pairwise separation ratio takes the dispersion of each class into account

- a slight generalization of the Dinkelbach algorithm

### Weaknesses
- The proposed method, being linear in nature, seems overly complex. It is not clear if the efforts put into its optimization are worthwhile. The authors' limited experiments are small scale, with no mentioning of the complexity and no comparison against (the many) nonlinear reduction algorithms (e.g., supervised version of tSNE). 

- The overall contribution seems moderate: the proposed ratio objective and the generalization of the Dinkelbach algorithm, while being perhaps technically new, are straightforward extensions of prior works. The rest of the algorithmic details in Section 3.2 is a direction application of known techniques. Overall, it is not clear how significant the contributions are and how relevant the proposed algorithm is to current practice in deep learning. 

- The experiments are of such a small scale that makes its conclusions questionable. For example, in Table 1, on the COIL-20 dataset, on first glance, it seems impressive that the proposed method improved the 1-NN accuracy of LDA (0.0046) to 0.0026. However, since the test set is of size 720, the improvement amounts to about an additional 1.44 test samples being classified correctly... 

- Some Theorem and Lemma statements have gaps (that are fortunately) fixable: 
  
  - Theorem 3: convergence to the global optimum is proven under the assumption that $\lambda_k$ converges in **finite** steps. This is a big assumption that should at least be part of the theorem's assumption. 

  - Lemma 1 as currently stated is not true: a simple counterexample is when $\tilde S_{ij} = [1, 0; 0, -1]$. For the Lemma to hold, one has to explicitly translate $\tilde S_{ij}$ so that it is PSD, as mentioned on Line 206-207.

### Questions
My biggest concern of this work is on the significance of the contributions and the relevance to current practices of the ICLR audience. While linear discriminant analysis was popular in ML about 20 years ago, it is not clear if we still need another algorithm for performing linear dimensionality reduction on small scale datasets. Could the authors demonstrate the significance of their algorithm on much larger and newer benchmarks that are relevant to current applications?

Other comments: 

- On Line 50-51, the authors cited many works for different separation metrics and on Line 67 the authors attributed the max-min criteria to Bian & Tao (2011a). Is Bian & Tao (2011a) the first to propose the max-min criteria in linear discriminant analysis?

- One obvious drawback of the max-min criteria is that it is easily dominated by an outlying class. Comparison with the sum criteria, with and without outlying classes, would likely strengthen the experiments. 

- $\tilde S_{C_{ij}}$ in Eq (5) should be $\tilde S_{ij}$? 

- The final algorithm is built on the linearization of a nonconvex function. While there is no guarantee the algorithm will converge to the global solution, how much will initialization affect the final result?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies discriminative dimensionality reduction, based on maximizing the minimum pairwise ratio of between-class to within-class scatter. The main technical development is on the optimization procedure of the objective defined in (2), which involves maximization over the linear projection and minimization over pairs of classes for finding the worst pair. The paper treats the problem as max-min fractional program and applies the generalize Dinkelbach’s algorithm.

### Strengths
The authors provide proofs for each statement.

### Weaknesses
1. I might be wrong and am willing to re-evaluate the paper if the authors can address my question:  for equation (4), after finding out the worse pair (i,j) for the inner minimization (e.g., by enumeration), cannot we solve the outer maximization problem, which is a generalized eigenvalue problem, with efficient numerical procedures? If that is the case, I find it hard to justify the convex relaxation on the constraint of $T$.

2. The datasets are tiny and are rarely used in modern ML papers. It is hard to evaluate the effectiveness and efficiency of methods based on these datasets.

### Questions
See weakness above. Also the particular approach seems too complicated to generalize the method to the deep learning framework.

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
This paper proposes a novel discriminative dimensionality reduction method, GDMM-QF, based on maximizing the minimum pairwise ratio of between-class to within-class scatter (PWCRA). Specifically, in the outer loop, a generalized Dinkelbach-type procedure to transform the challenging maxmin fractional objective into an equivalent max-min subtractive problem. In the inner loop, the Minorization-Maximization (MM) framework is adopted to construct a tight, convex surrogate that lower-bounds the true objective, resulting in a semidefinite program (SDP) at each inner step. Finally, extensive experiments on multiple datasets demonstrate the superiority of the proposed method in learning discriminative projections.

### Strengths
1. The paper introducing the Pairwise Worst-Case Ratio Analysis (PWCRA) objective. This is a improvement over existing worst-case methods because it uses a pairwise-adaptive within-class scatter matrix, making it more robust to heteroscedastic data where class variances differ significantly.
2.  The theoretical foundation is strong. The paper provides rigorous proofs for all key claims, including the equivalence of the original problem to a root-finding problem (Theorem 1), the convergence of the outer loop to the global optimum (Theorem 3), and the validity of the convex relaxation (Lemma 1). 
3. The proposed GDMM-QF algorithm addresses a fundamental challenge in discriminant analysis: ensuring robust separation even for the most difficult class pairs.

### Weaknesses
1.	The author only provided the mean and standard deviation (based on 20 repetitions) but did not perform any significance tests (such as paired t-test) to demonstrate whether the improvement was statistically significant.
2.	The paper mentions that the complexity of the QP method is O(C⁶ + d³). However, the paper does not validate this on larger datasets. Could you provide such validation?

### Questions
1. The results in Table 1 show impressive performance gains. Could the authors perform a statistical significance test (such as paired t-test) to confirm that the differences in error rates between GDMM-QF and the next best method are statistically significant?
2. It would be better to have validation with a larger dataset.

### Soundness
3

### Presentation
3

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
This paper presents GDMM-QF, a novel discriminative dimensionality reduction method that maximizes the minimum pairwise ratio of between-class to within-class scatter, thereby enhancing class separability through adaptive control of class-pair variances. GDMM-QF addresses the resulting non-convex max-min fractional programming problem. GDMM-QF is hyperparameter-free, computationally efficient, and guarantees convergence. Extensive experiments on benchmark datasets demonstrate its effectiveness in learning highly discriminative low-dimensional representations, consistently outperforming existing state-of-the-art methods in classification accuracy.

### Strengths
1.	The proposed Dinkelbach–Minorization–Maximization framework is novel and contributes a fresh perspective to the field of Artificial Intelligence.
2.	The paper provides a thorough and rigorous mathematical analysis of the GDMM-QF method.
3.	The method consistently outperforms baselines (LDA, MMDA, MMRA, etc.) across multiple benchmark datasets.
4.	The algorithm’s parameter-free design is a notable practical advantage, reducing the need for extensive hyperparameter tuning and enhancing reproducibility in real-world applications.

### Weaknesses
1.	The experiments are conducted on only three datasets. Evaluating the method on larger-scale or more complex datasets would strengthen the claims regarding scalability and generalizability.
2.	Although the paper claims efficiency improvements over SDP-based methods, the empirical section lacks detailed runtime comparisons or scalability analyses as the dataset dimensionality increases.
3.	The paper does not analyze the impact of key components (e.g., pairwise normalization, MM approximation) individually. Including ablation studies could help clarify the source of the observed performance gains.
4.	The paper lacks visual illustrations such as framework or pipeline diagrams and primarily focuses on theoretical analysis. Incorporating such visualizations would improve clarity and accessibility.

### Questions
1.	It might be helpful if the authors could include empirical comparisons of runtime and memory usage on higher-dimensional or large-scale datasets to further support the claimed efficiency and scalability.
2.	The paper introduces a new optimization framework, but some parts of the motivation and intuition behind the method could be clearer. Could the authors provide more intuition or illustrative examples to help readers understand why this approach works better than existing ones?
3.	It appears that the proposed method does not rely on neural networks. So it could be useful to discuss whether there are related neural network–based approaches addressing the same problem, and how the proposed method compares with them in terms of advantages and limitations.
4.	To ensure reproducibility, will the authors release the source code and experimental setup? If not, additional implementation details—such as initialization strategy, stopping criteria, and computational resources—would be appreciated.

### Soundness
2

### Presentation
2

### Contribution
2
