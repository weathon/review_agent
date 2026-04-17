# Optimal Sub-data Selection for Nonparametric Function Estimation in Kernel Learning with Large-scale Data

- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
This paper considers estimating nonparametric functions in a reproducing kernel Hilbert space (RKHS) for kernel learning problems with large-scale data. Kernel learning with large-scale data is computationally intensive, particularly due to the high cost and complexity of tuning parameter selection. Existing sampling methods for scalable kernel learning, such as the leverage score-based sampling method and its variants, are designed to sketch the kernel matrix to minimize the expected global (in-sample or out-of-sample) prediction error. In complement to existing methods, this paper proposes an optimal informative sampling method to estimate nonparametric functions pointwisely when the subsample size is potentially small. Our method is tailored for scenarios where computational resources are limited, yet accurate pointwise prediction at each test location is desired. It aims to select an informative sub-data, which is different from sketch methods that aim to select the columns of a kernel matrix. Theoretical studies compare the efficiency of the proposed method to that based on the full data with optimally selected tuning parameters. Furthermore, integrating our sub-data selection with sketching methods can improve computation time of full-data based sketching methods while maintaining the same statistical efficiency.
Numerical experiments demonstrate the statistical and computational efficiency of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper deals with the problem of estimating a nonparametric function from a given reproducing kernel Hilbert space, when using large-scale (possibly with large sample size) training data.
This is a recurring problem in the field.
Specifically, the work focuses on a kernel Ridge Regression (KRR) problem where the RKHS is assumed given. Then, the goal is to estimate the kernel expansion coefficients and regularization weight (called tuning parameter in the paper) in a data-scalable manner.
For that, the proposed method looks for a fixed number of clustered data subsets whose centers act as a representative values of all the data samples in the cluster, and then solves the KRR problem over the resulting (smaller) data.
Finally, a theoretical discussion about the rate of convergence of the proposed method and experimental results are presented.

### Strengths
The presented approach seems novel and is well motivated.

### Weaknesses
**W1**: To me, the biggest weakness of this paper is an inconsistent, and sometimes vague, mathematical notation. 
This makes following the main claims of the paper a tedious task.
Overall, this paper would benefit substantially from a consistent and properly introduced mathematical notation.

To give some examples: \
**W1.1**: The equation in line 132 defines the optimal argument as $\hat{f}\_{N,\lambda\_T}$ . 
In the ensuing sections the comma in the subindex disappears. \
**W1.2**: In the expression in line 140, the authors use brackets [ ] to denote a vector. Later, vectors are denoted with parenthesis ( ), e.g., line 177, or curly braces { }, e.g., line 195, instead. Same for the transpose symbol, sometimes is an apostrophe $'$, e.g., line 266, and sometimes a superscript $^T$, e.g., line 268. \
**W1.3**: Some notation is introduced without definition, such as $\hat{f}\_{s\lambda}\^\*$ in line 151, and $\theta_i$ and the subscript $\_{n*n}$ in line 226.

I believe there are also some minor typos/mistakes such as: \
**W1.4**: The sentence spanning lines 158 to 161 defines the centers of the clusters redundantly. \
**W1.5**: In line 192, $\bar{\epsilon}_i$ is using the definition intended for $\bar{y}_i$. \
**W1.6**: In line 196 the $(i,j)$-th element is a set. It should likely be $K(C_i,C_j)$ or a singleton. \
**W1.7**: In line 277 $\mathscr{S}(x_0)$ is defined as a set of $x^*_i$ values; however, in line 279 it is used as the set of indices of those values. \
**W1.8**: There is typo in line 476, ``can effectively use~~s~~''.

**W2**: A discussion of the memory and time complexity comparing the proposed method to alternatives (e.g., Nyström and FALKON) would have been appreciated.
For instance, in a comparison table.

### Questions
**Q1**: In table 3, I assume that ``centers-time'' refers to the execution time of the clustering algorithm. Taking that into account, the computation time of the proposed method is still one, or even two, orders of magnitude larger than FALKON and Nyström. 
Can you elaborate on that?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a subset selection method for large-scale nonparametric regression in reproducing kernel Hilbert spaces (RKHS). The main goal is to improve prediction accuracy when computational resources limit the use of all available data. The proposed approach first partitions the dataset using clustering methods such as k-means to identify representative centers, then assigns optimal sampling weights to clusters in order to minimize the pointwise mean squared error of the resulting kernel estimator. Weighted subsampling is performed multiple times, kernel ridge regression models are trained on each subset, and their predictions are averaged. The authors claim to achieve better MSE error compared with basic Nyström or FALKON (but its first version in 2017 in Matlab).

### Strengths
The topic is interesting and the paper targets the well-known bottleneck of kernel methods, i.e. scalability. The clustering and sampling procedure with optimal weights can be interesting, but a clearer and more precise exposition, and deeper experiments are needed to evaluate it.

### Weaknesses
### **Critical Observations**

1. **Outdated comparison**  
   The paper compares only against the *original* **FALKON (2017)** and a basic **Nyström** baseline, but not against more recent or optimized implementations — for example, *“Kernel methods through the roof: handling billions of points efficiently”* (Meanti et al., NeurIPS 2020), which provides a **modern, fast version of FALKON**. As a result, the experimental comparison does not reflect the current state of the art.

2. **Misleading comparison and trivial findings**  
   The comparison is conceptually weak. Works such as *“Less is more: Nyström computational regularization”* (Rudi et al., 2015) and *“FALKON: An optimal large scale kernel method”* (Rudi et al., 2017) have already shown that Nyström-based methods can **achieve optimal learning rates** while maintaining **computational efficiency**, provided that a **sufficient number of landmarks** are used (e.g., √n in kernel ridge regression).  
   It is therefore **expected** that using too few subsampled points degrades accuracy, and that more sophisticated (but slower) sampling strategies can improve MSE. Even a fully greedy point-selection scheme — which maximizes accuracy at each step — would outperform uniform sampling, but at a much higher computational cost.  
   -> Consequently, the reported “superior accuracy” is **trivial and somewhat misleading**: it arises from sacrificing scalability, missing the main purpose of FALKON, which is to remain **fast while preserving accuracy** once enough centers are used.

3. **Computational inefficiency**  
   The proposed method is **consistently slower** than FALKON (often by an order of magnitude) while achieving only **moderate MSE improvements**.  
   -> The key insight of FALKON — maintaining accuracy while being computationally efficient — is not addressed or appreciated in this work.

4. **Clarity and presentation issues**  
   The paper is **poorly structured and very difficult to follow**. Among all, definitions and notations are often unclear or inconsistent:  
   - The meaning of \( y_{ij} \) (double index) is never explained.  
   - The notation \( \hat{w}_{x,i,C} \) contains an unexplained hat (possibly a typo).  
   - \( K_{xi} \) is defined *after* Equation (2) but used *before* it.  
More than that, theoretical assumptions are **scattered and vaguely presented**, often embedded directly within theorem statements without being clearly introduced and discussed.  How do these assumptions compare with the rest of the literature (which is **largely** missing in the entire paper)?

   Overall, the exposition lacks precision and logical flow, making the paper unnecessarily hard to read and evaluate.

### Questions
Most perplexities are already expressed above under Weaknesses. 
- Why is the most recent literature (both theory and algorithms) not considered?
- in Results at pag. 8 it's said that the method "remains efficient", but it is 100x slower than old version of FALKON, how can be considered efficient when datasets contain billions of points?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the computational challenge of large-scale nonparametric function estimation in Reproducing Kernel Hilbert Spaces (RKHS) by proposing a weighted sub-data selection method. Theoretical results show that the proposed method achieves a faster convergence rate than simple random sampling (SRS).

### Strengths
S1. The paper provides a theoretical analysis, demonstrating that its proposed estimator achieves a convergence rate superior to Simple Random Sampling.

S2. The method demonstrates superior performance in MSE compared to established benchmarks.

### Weaknesses
W1. The technical components of the proposed method rely on established algorithms and principles that lack innovation. For instance, the core clustering step relies on k-means, a classical algorithm initially proposed more than six decades ago (e.g., by Stuart Lloyd in 1957). Similarly, the use of variance minimization to derive optimal weights is a long-standing principle in classical optimization and statistics. While the integration of these elements for point-wise prediction in RKHS is applied to a specific setting, the overall methodology constitutes a recombination of existing techniques rather than a conceptually novel contribution. Furthermore, the problem being addressed, scaling kernel methods,does not align with pivotal frontier challenges in contemporary AI research, thereby limiting the broader impact and relevance of this work.

W2. The paper suffers from several issues in scholarly rigor and exposition that impact its professionalism and readability, for example:
1. In Section 2, the theoretical foundation lacks references to crucial theorems like Mercer's theorem and the Representer Theorem. Their absence makes the theoretical setup incomplete.
2. The meaning of notation is sometimes unclear. For instance, in Section 2.1, the expression {K(x, C_1), ...} does not specify whether the braces {} denote a set or a vector, creating unnecessary confusion.
3. Incorrect formatting of references is present throughout the manuscript, which diminishes the work's professionalism and adherence to standard academic conventions.

### Questions
Q1. In the first part of Section 2, the parameter $\lambda_T$ is introduced. Could you please clarify the relationship and distinction between $\lambda_T$ and the regularization parameter lambda used later in Section 2.1? Specifically, what does the subscript $T$ denote, and what is the conceptual or mathematical difference between these two symbols?

Q2.The proposed sampling strategy appears to rely on a clustering step. Could you please discuss the potential sensitivity of your method's final performance to the specific clustering algorithm chosen? Have you experimented with different clustering methods?

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies kernel ridge regression (KRR) in reproducing kernel Hilbert spaces (RKHS) in the context of large sample sizes (N) and proposes a sub-data selection scheme aimed at matching or improving the statistical accuracy while simultanesouly reducing computational complexity. The method clusters inputs into L groups (via $k$-means or variants), chooses a tuning parameter using only the cluster centers, and then, for each test location $x$, assigns sampling probabilities to training points proportional to $ | K_x|$ (a transformed kernel vector that depends on $x$, the chosen kernel, the center kernel matrix, and the tuning parameter $\lambda$). The estimator averages KRR fits over \$B\$ resampled subsets of size $n$ much smaller than the sample size $N$. The paper provides a clean derivation of KRR rates and optimal tuning parameter value $\lambda_T\$ under eigenvalue decay (Theorem 1) and an asymptotic IMSE rate for the proposed estimator (Theorem 2) showing improvement over simple random sampling (SRS) under assumptions on the informativeness profile based weights $\omega_{x_{0},j}$. Simulations and real-life data study show competitive IMSE vs. full-data KRR and improved test MSE versus Nyström method and FALKON at comparable budgets.

### Strengths
The consitional MSE‑minimizing weights $w_{x,i,C}(x)\propto |K_{xi}|$ derived from the cluster‑center surrogate model are simple, somewhat interpretable, and targeted at the desired loss (pointwise prediction at a given $x$). Theorem 1 recovers the optimal full‑data KRR rate of convergenceunder different choices of target RKHS function class and eigendecay rate, while Theorem 2 formalizes how an informative sampling profile (captured by $\omega_{x_{0},j}$) yields an IMSE rate faster than SRS for fixed $n$. The paper shows the value of using information in the full input data to determine sub‑data selection.

### Weaknesses
Theorem 2 makes the assumption : $\omega_{x_0,j}\asymp j^{2\beta}$ with $0\le 2\beta<k$ (and $2\beta\le k\le 4\beta$). However, it is not shown that the proposed $k$‑means + $|K_x|$ weighting mechanism induces such a condition, for any $\beta>0$, even under standard kernels and benign marginal input disribution $\pi(x)$. As written, the theorem establishes rates for a class of informative samplers rather than for the concrete algorithm. A lemma connecting cluster geometry and $K_x$ to eigenfunction mass can strengthen the result. Also, using Euclidean $k$‑means to define centers may not align with the RKHS geometry for non‑RBF kernels. A short discussion (or a kernel‑$k$‑means variant such as $k$‑medians) would clarify when the centers faithfully represent $K(\cdot,\cdot)$. In the definition of Relative efficiency (RE) on Lines 316-318, the choice of the exponents of IMSE and Time is not well-motivated.

### Questions
Pleae see Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3
