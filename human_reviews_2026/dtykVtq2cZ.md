# Variance Reduced Distributed Non-Convex Optimization Using Matrix Stepsizes

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Matrix-stepsized gradient descent algorithms have been shown to have superior performance in non-convex optimization problems compared to their scalar counterparts. The det-CGD algorithm, as introduced by Li et al. (2023), leverages matrix stepsizes to perform compressed gradient descent for non-convex objectives and matrix smooth problems in a federated manner. The authors establish the algorithm’s convergence to a neighborhood of a weighted stationarity point under a convex condition for the symmetric and positive-definite matrix stepsize. In this paper, we propose two variance-reduced versions of the det-CGD algorithm, incorporating MARINA and DASHA methods. Notably, we establish theoretically and empirically, that det-MARINA and det-DASHA outperform MARINA, DASHA and the distributed det-CGD algorithms in terms of iteration and communication complexities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes two variance-reduced distributed algorithms, det-MARINA and det-DASHA, for nonconvex finite-sum optimization in federated learning settings. Building on the det-CGD method, the authors incorporate variance reduction techniques from MARINA and DASHA to eliminate the convergence neighborhood issue caused by stochastic compressors in det-CGD. Under matrix smoothness assumptions, they prove convergence with improved iteration and communication complexities compared to scalar counterparts and det-CGD. Experiments on LIBSVM datasets show improved iteration/communication metrics over baselines.

### Strengths
1. The integration of matrix stepsizes with variance reduction addresses a key limitation of det-CGD by removing the non-vanishing error term, leading to better theoretical bounds.

2. The authors provide clear convergence guarantees under matrix Lipschitz assumptions, highlighting the advantages of the proposed methods.

3. Experiments validate the effectiveness of the proposed method, showing det-MARINA and det-DASHA outperform baselines in communication and iteration efficiency.

### Weaknesses
1. Considering matrix smoothness is not very standard in general stochastic optimization problems, the authors should give more discussion about this assumption, including how it can be satisfied in practical problems and the comparison with the standard smoothness assumption.

2. Although the authors provide many experimental results, more complicated and real-world scenarios or tasks would largely strengthen the experimental part.

3. How do the authors recommend choosing or approximating the matrix stepsize D in practice, especially when the full Hessian or smoothness matrices are expensive to compute? Are there heuristics or approximations beyond $D = \gamma w W$? More discussions here may be helpful.

4. It is well-known that the optimal complexity for finite-sum optimization is $\mathcal{O}(\sqrt{n} \epsilon^{-2})$. How can we understand the order of $n$ in the convergence rate for matrix stepsize methods?

### Questions
See the Weakness part.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies federated learning optimization methods with matrix stepsizes. Two new FL algorithms were proposed, det-MARINA and det-DASHA. These works extend the existing det-CGD algorithm. The two new algorithms mainly address the variance reduction aspect. As a result, the proposed algorithms are able to exhibit a superior convergence bound, exceeding the neighborhood limitation presented in det-CGD and other SGD-style methods. Experiments show that the two proposed algorithms require less iteration complexity while being as communication efficient as existing algorithms such as det-CGD.

### Strengths
The paper introduces variance-reduced variants of det-CGD, Although MARINA and DASHA have been relatively well-studied, the reviewer believes it is still a nontrivial and meaningful extension. The proposed det-MARINA and det-DASHA algorithms display noticeable improvement compared to previous algorithms.

The theoretical analysis seems solid and rigorous to the best of my knowledge. The analysis is well explained and mostly intuitive. The statements regarding iteration and communication complexity are of interest to potential readers.

The experiments utilize synthetic objective functions that satisfy the function assumptions, and the numerical experiments verify the effectiveness of the proposed algorithms. The authors also provided a comparison between algorithms in terms of communication bytes.

### Weaknesses
While the experiments effectively recover the technical assumptions on the objective function and serve as verification of the analysis, they are, for the most part, still synthetic toy examples. Since the paper addresses federated learning problems, which are largely motivated by practical applications, it would be helpful if the authors also provided further practical results with real-life FL tasks.

The writing of this paper has much room for improvement. 
- The authors provided a mathematically sound introduction and related works; however, they did not explain the motivation or the necessity of this work. What are the properties of CGD? Why is the previous method named det-CGD? More explanations should be added for this paper to be accepted as a conference paper.
- The author also used terms such as det-CGD and det-CGD1/2 interchangeably with the assumption that readers have read the "original paper", which the reviewer assumes to be Li et al.(2024).
- In terms of notations, the authors have switched notation from D to W, and introduced matrices L, S, D in Section I, while the det-CGD algorithm is introduced two pages later.
- Many mathematical definitions, such as matrix smoothness, should be provided in the manuscript.

### Questions
I wonder how the heterogeneity across agents affects the convergence of distributed det-CGD and the two proposed algorithms in this setting? Does the variance from CGD and the variance from FedAvg compound in practice and theory?

How does the computational complexity of Compressed Gradient Descent affect the general cpu wall time of the proposed algorithms? 

In practice, are the gradients first calculated in full and then compressed, or are they directly estimated as compressed signals?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces two algorithms, det-MARINA and det-DASHA, aimed at federated non-convex optimization with communication compression. These methods extend the matrix-stepsize algorithm det-CGD by incorporating variance reduction techniques (MARINA and DASHA, respectively). The primary contribution is the theoretical demonstration that these variance-reduced versions overcome the main limitation of det-CGD, which is convergence only to a neighborhood of a stationary point, and instead achieve convergence to a true stationary point with an O(1/K) rate, measured in a determinant-normalized gradient norm (Theorems 4.1 and 5.1). The paper derives conditions for the matrix stepsize D and proposes a practical relaxation by setting D = γW, providing explicit formulas for the scalar γ. Complexity analyses suggest improvements over scalar MARINA/DASHA and det-CGD. Empirical results on logistic regression problems confirm faster convergence in terms of communication bytes compared to baselines.

### Strengths
1.  The paper removes the residual error present in det-CGD convergence analyses and proves clear O(1/K) convergence to a stationary point under the matrix Lipschitz assumption.

2. The theoretical results appear correct and consistent, with checks showing that the framework recovers known results for scalar MARINA/DASHA and matrix gradient descent.

3. The relaxation D = gamma*W and explicit formulas for gamma make the method easier to use in practice.

### Weaknesses
1. The paper’s novelty is a theoretical synthesis of matrix stepsizes and variance reduction, which successfully removes the convergence
neighborhood of det-CGD. While the analysis is careful, many steps of the proof are direct translations of variance-reduction proofs to the matrix norm setting. The manuscript does not isolate what specific technical challenges (if any) required new analytical techniques.

2. The method’s reliance on knowing or accurately estimating the matrix L (or local Li) remains a major practical limitation, The paper acknowledges this (”Availability of L”, Sec 5.1) but lacks a convincing practical strategy or robustness analysis regarding estimation errors. This significantly undermines the applicability of the results.

3. The experiments do not sufficiently justify the method’s added O(d2) complexity. The evaluated logistic regression
tasks on LibSVM datasets neither reflect moderate-scale non-IID federated benchmarks nor include ill-conditioned settings where matrix stepsizes should provide clear benefits. The observed gains over scalar methods are modest, offering limited empirical justification for the added complexity.

### Questions
1. What parts of your analysis are genuinely new or rely on arguments that differ from scalar variance reduction proofs? Clarifying this would help readers understand the technical novelty.

2. How sensitive are det-MARINA and det- DASHA to errors in estimating L or Li? Showing results with controlled under- and over-estimation would make the practical stability clearer.

3. For W = diag−1(L), what efficient methods can estimate or update diag(L) in large-scale federated settings?

4. The experiments use small logistic regression problems. Can you include larger benchmarks such as CIFAR-10 or CIFAR-100 with non-IID clients and moderate neural networks to show that the approach scales and remains effective in realistic scenarios?

5. Could you test a problem with strong ill-conditioning where matrix stepsizes give much larger speedups over scalar methods? This would better justify their added complexity of your approach.

### Soundness
3

### Presentation
3

### Contribution
2
