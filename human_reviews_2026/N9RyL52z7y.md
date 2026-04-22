# Efficient Ensemble Conditional Independence Test Framework for Causal Discovery

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Constraint-based causal discovery relies on numerous conditional independence tests (CITs), but its practical applicability is severely constrained by the prohibitive computational cost, especially as CITs themselves have high time complexity with respect to the sample size. To address this key bottleneck, we introduce the Ensemble Conditional Independence Test (E-CIT), a general-purpose and plug-and-play framework. E-CIT operates on an intuitive divide-and-aggregate strategy: it partitions the data into subsets, applies a given base CIT independently to each subset, and aggregates the resulting p-values using a novel method grounded in the properties of stable distributions. This framework reduces the computational complexity of a base CIT to linear in the sample size when the subset size is fixed. Moreover, our tailored p-value combination method offers theoretical consistency guarantees under mild conditions on the subtests. Experimental results demonstrate that E-CIT not only significantly reduces the computational burden of CITs and causal discovery but also achieves competitive performance. Notably, it exhibits an improvement in complex testing scenarios, particularly on real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper is related to causal discovery and proposes a generic Ensemble Conditional Independence Test (E-CIT), which uses a divide-and-aggregate strategy to reduce the typical superlinear numerical complexity to a linear one. The approach is compatible with various existing CIT methods. Several numerical experiments are performed to illustrate the performance of the ensemble approach. Results for different baseline setups look promising. Type I error and statistical power remain competitive while saving computational time.

### Strengths
S1 The considered problem is interesting and relevant.
S2 The paper is technically sound and overall well-written.
S3 To obtain linear complexity is advantageous.
S4 The evaluation and comparison against baselines is convincing.

### Weaknesses
W1 It remains unclear whether hyper parameters have to be tuned.
W2 Limitations of the paper’s methods could be better discussed.

### Questions
Does it make a difference how the data is partitioned into the K subsets?

How to automatically choose the hyperparameter \alpha in general applications? How to choose the number of samples K?

Are the results worse in Table 2 and 3 if alpha is chosen smaller or larger?

The left windows in Figure 2 for the Type I Error do not have a good y-axis scaling. The key differences cannot be seen. Please use a e.g. range between 0.0 and 0.08.

Under which conditions the approach performs not good (e.g. regarding the data partitioning)?

Minor comments:

- line 391,408,410: Why are some numbers italic? What does it mean?

### Soundness
3

### Presentation
3

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
This paper proposes E-CIT, a framework for scalable conditional independence testing (CIT) that divides data into subsets, applies any base CIT independently, and aggregates p-values through a stable-distribution–based combination. The goal is to reduce computational cost while preserving Type I error control and statistical stability. The paper provides theoretical proofs of validity and convergence, and empirical results on synthetic and real datasets showing comparable accuracy to full-sample CITs but much better efficiency.

### Strengths
1. The paper addresses a practical bottleneck in causal discovery with a general framework that can wrap around existing tests such as without retraining.

2. The use of stable-distribution aggregation is mathematically sound and provides more stable inference than classical methods, with clear theoretical foundation.

3. Experiments show that E-CIT achieves similar Type I and power performance as full CIT while reducing computation time by up to an order of magnitude on large datasets.

### Weaknesses
1. The proposed framework is mainly an adaptation of existing ensemble ideas; it does not introduce a fundamentally new conditional independence test or inference principle.

2. Although the theoretical proofs are solid, they mostly re-establish known results from p-value combination theory in the CIT context, making the methodological novelty limited.

3. The evaluation focuses mostly on synthetic data and standard causal discovery tasks; there is little evidence on whether E-CIT maintains robustness on high-dimensional or real-world noisy domains.

### Questions
Since E-CIT combines subset-level p-values, how sensitive is it to the number of partitions. does too few or too many subsets affect power consistency?

The theoretical analysis assumes independence between subsets; in real data where partitions can still share dependence structure, how much does this violate the validity guarantees?

Could the authors compare their stable-distribution aggregation directly against the more recent Cauchy Combination Test (Liu & Xie 2020) to clarify relative advantages?

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
This paper introduces E-CIT, a divide-and-aggregate framework to solve the high computational cost of conditional independence (ci) tests. E-CIT partitions the data into subsets, applies base CI tests to each subset, and combines the resulting p-values. Its core contribution is a p-value combination method based on stable distributions, which provides theoretical consistency. This framework reduces the complexity of base CITs to linear in sample size (when subset size is fixed) and demonstrates competitive or superior power, especially in heavy-tailed scenarios.

### Strengths
1. The paper is well-structured, and the proposed framework is also intuitive.
2. The framework can be combined with different CI methods. 
3. Table 1 is clear and useful for unfamiliar readers.

### Weaknesses
1. It is intuitive to divide the sample set into several subsets to reduce computational burden. However, I find it difficult to understand why this simultaneously enhances statistical power in experiments. The theoretical justification throughout the paper does not focus on or support this point. Consequently, we are still unclear under which conditions the E-CIT framework outperforms single-test execution.
2. The paper's core theoretical guarantees depend on the p-values from the subtests being well-defined and effective. To ensure the subtest p-values are accurate, $n_k$ should be sufficiently large and tend to infinity. On the other hand, the paper claims that the framework reduces time complexity to $O(n)$ when $n_k$ is treated as a fixed constant, which raises a contradiction. $n_k$ cannot be simultaneously treated as a small, fixed constant to justify the $O(n)$ complexity claim, and as an asymptotically large number to justify the theoretical validity and consistency of the theorems. 
3. The paper states that the median heuristic is used for all methods to ensure a fair comparison. However, bandwidth is sample-size dependent. To isolate the effect of the ensemble strategy, I suggest the authors attempt to find the optimal bandwidth for each test in its respective setting (e.g., via bandwidth optimization [1]). 
4. The framework does not provide a principled or automated method to select $n_k$. 
5. The authors claim in the contribution that the p-value combination method proposed is novel. It may be slightly overstated. The method shares a related technical foundation with existing works on p-value combination (e.g., Ling & Rho (2022), Liu & Xie (2020)), which are already mentioned in the paper's related work. 

References:   
[1]  W. Wang et al, Practical Kernel Selection for Kernel-based Conditional Independence Test. NeurIPS 2025.

### Questions
1. Is $n_k=400$ still a good choice with a larger sample size?
2. Please elaborate more details on the differences and the relation with Stouffer et al. (1949).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an efficient ensemble independence testing approach that reduces computational cost by first partitioning the data into subsets, then applying a base conditional independence test (CIT) on each subset, and finally aggregating the resulting p-values. This strategy reduces the complexity of a base CIT to be linear in the sample size. Empirical results show that the proposed method improves efficiency while maintaining comparable accuracy.

### Strengths
-	The paper presents a strong motivation and a practical approach for improving the efficiency of conditional independence tests, and consequently, constraint-based causal discovery.
-	The paper is well-written and clearly presented.

### Weaknesses
-	This paper provides limited evaluation of generality; see Questions 5, 6, and 7 for details.
-	The paper makes strong claims of optimality but lacks sensitivity analysis for violations of the theorem’s assumptions, some of which appear overly strong.

### Questions
1.	Could the authors provide references to support the claim that ensemble learning can address high computational complexity (lines 131–134)?
2.	Is this the first work to propose an ensemble-based conditional independence testing approach? 
3.	What happens if the assumptions in Definition 1 are violated? Does the derived ensemble test still hold under such conditions?
4.	What are the implications if the unbiasedness assumption in Theorem 1 is violated? For instance, subtests may be biased in practice if the selected data subsets are not representative. While I understand there is a theoretical–practical gap, it would be helpful to analyze how sensitive the proposed algorithm is to such violations, given that the paper emphasizes its asymptotic optimal properties.
5.	Why do the authors assume nonlinear causal functions? It would be informative to also report results for linear causal functions. Additionally, can the proposed algorithm be applied to data with Gaussian noise? If so, does it still demonstrate advantages in accuracy and efficiency compared to state-of-the-art methods?
6.	The experiments on real data are not entirely convincing. The paper uses synthetically generated interventional data rather than directly collected real-world data. A more rigorous and widely accepted evaluation strategy would rely solely on the 783 observational samples. Otherwise, it is difficult to substantiate the claim that the proposed method performs well on real-world data.
7.	The sample size may not be the dominant factor affecting causal discovery efficiency. The primary challenge for constraint-based approaches lies in the accuracy and scalability of conditional independence tests when the conditioning set is high-dimensional. How does the proposed approach perform under such settings?

### Soundness
3

### Presentation
3

### Contribution
2
