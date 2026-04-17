# Conquer the Quantile: Convolution-Smoothed Quantile Regression with Neural Networks and Minimax Guarantees

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Quantile regression provides a flexible approach to modeling heterogeneous effects and tail behaviors. This paper introduces the first quantile neural network estimator built upon the \textbf{con}volution-type smoothing \textbf{qu}antil\textbf{e} \textbf{r}egression (known as \textit{conquer}) framework, which preserves both convexity and differentiability while retaining the robustness of the quantile loss. Extending the conquer estimator beyond linear models, we develop a nonparametric deep learning framework and establish sharp statistical guarantees. Specifically, we show that our estimator attains the minimax convergence rate over Besov spaces up to a logarithmic factor, matching the fundamental limits of nonparametric quantile estimation, and further derive general upper bounds for the estimation error in more general function classes. Empirical studies demonstrate that our method consistently outperforms existing quantile networks in both estimating accuracy and computational efficiency, underscoring the benefits of incorporating conquer into deep quantile learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a quantile regression framework, where an ReLU neural network is used to approximate the quantile function, and a convolution-type smooth quantile loss is used to train the network. Experimental results on synthetic data show that the proposed framework outperforms ReLU networks trained with quantile loss in terms of MSE evaluated at different quantile levels and runtime.

### Strengths
- The paper is well-written and easy to follow
- The paper provides strong theoretical results, where it achieves the optimal minimax convergence rate over Besov spaces up to a (log n)^3 factor. Nonasymptotic generalization bounds are also provided for nonsmooth quantile functions. This analysis has not been done for ReLU networks + convolution-type smooth quantile loss
- While the framework itself — namely the use of a convolution-type smooth quantile loss combined with ReLU networks — is not novel, I think this is not a drawback, given the accompanying theoretical results

### Weaknesses
- While I understand that a key contribution is the theoretical analysis, a main concern is that the experiments are limited to synthetic data. It is unclear how the method would perform in more realistic settings
- Only MSE loss at different quantile levels are evaluated in the experiments, while other common metrics such as MAE loss and pinball loss are missing
- The results reported in the tables are not clearly highlighted. It appears that bold font is used when the proposed method performs the best, but it is not used when a baseline performs the best

### Questions
- In the tables, what does bold font represent?
- Where does the training-time improvement of the proposed method over the baseline come from?
- It is claimed that "the results highlights the effectiveness of conquer to modern neural network architectures" Can the theoretical analysis and results be extended to more commonly used modern architectures, for example, MLPs with skip connections?

### Soundness
2

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
**Summary**

This paper extends the convolution-smoothed quantile regression (Conquer) framework to deep neural networks.  
It replaces the non-differentiable pinball loss with a kernel-convolved, smooth surrogate, allowing gradient-based optimization while preserving quantile consistency.

Main contributions:
- Defines the **ConquerNN estimator**, minimizing a smoothed quantile loss over ReLU networks.
- Provides **non-asymptotic risk bounds** and proves **minimax-rate optimality** over Besov spaces (up to logarithmic factors).
- Derives a **generalization bound** depending on architecture parameters (depth, width, sparsity).
- Presents **synthetic experiments** under heavy-tailed noise showing improved MSE and training stability vs. standard pinball loss networks.

### Strengths
1. Strong theoretical rigor and correct proofs.  
2. Achieves minimax-rate optimality over Besov spaces.  
3. Convincing argument for smooth loss improving optimization stability.  
4. Clear synthetic benchmarks and transparent experimental design.  
5. Full reproducibility with provided code and documentation.

### Weaknesses
1. **Empirical limitation:** Only synthetic data tested; no real-world validation.  
2. **Incremental novelty:** Combination of known ideas rather than new conceptual insight.  
3. **No joint-quantile or non-crossing analysis:** Key for realistic quantile regression.  
4. **Bandwidth selection heuristic:** No adaptive or data-driven rule proposed.  
5. **Limited connection to ICLR topics:** The work is statistically oriented, with weak links to deep learning challenges such as distributional modeling or uncertainty quantification.

### Questions
1. Can you propose a **data-driven rule** for selecting the bandwidth \(h\)?  
2. How does the smoothing behave when training **multiple quantiles jointly** (non-crossing constraints)?  
3. How sensitive are convergence and optimization to **misspecified \(h\)** or near-zero densities around the quantile?  
4. Have you evaluated the method on **real regression datasets** or higher-dimensional tasks?  
5. Could the smoothing principle be extended to **CRPS/Wasserstein objectives** for broader applicability?

### Soundness
3

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
This paper analyzes the integration of the conquer estimator in quantile regression with neural networks. The theoretical results mainly show that a ReLU-activated neural network that minimizes the conquer estimator performs optimally in $L^2$ norm, up to a logarithmic factor. An error bound is also established for a general neural network. Empirical studies demonstrates the benefits of applying the conquer estimator in neural networks.

### Strengths
* The paper addresses an important question of using the smoothed conquer estimator for quantile regression. The theoretical result is promising in that it shows the near-optimality of the conquer loss function.

* Theorem 3.2 extends to the setting of a general neural network, relaxing the shape of the network. This makes the discussion more applicable.

### Weaknesses
* The empirical studies lack some generality. It is hard to argue why the three scenarios are representational. In addition, the empirical benefit is decoupled from the theoretical analysis in the paper: that is, the author(s) attribute the empirical advantage to the better training stability and generalization of the neural network trained with conquer, but this is not the major theoretical claims (which only shows conquer leads to a reasonably good optimum). In order to make the argument more complete, the paper would benefit from:

   * Some empirical studies (maybe a numerical experiment) to directly corroborate Theorem 3.1/2.
   * Some further stability and generalizability analysis of conquer-trained neural network.

* The paper mainly analyzes the global optimum of a neural network with respect to the conquer objective, defined in (2.2). However, training a neural network hardly ever yields a global minimum. In order to see the benefit of conquer in neural network, a more practical piece of analysis would be related to the training dynamics, showing how conquer improves the training loss landscape. I understand that this is beyond the scope of the paper, but lacking it restricts the paper to a "expressivity-type" analysis.

* The presentation of the paper and main results can be improved. While the major claims made in the theorems can be understood, some notations are hard to keep track and are not explained anywhere in the paper. Please see some questions below.

### Questions
1. In (2.1), should we assume $A^{(1)} \in \mathbb{R}^{W \times d}$ instead of $\mathbb{R}^{W \times W}$?

2. On line 136, do we assume that the $\infty$-norm is defined on a compact domain?

3. On line 138, I cannot see how the function $f_n$ depends on $n$. Is it a typo?

4. For Assumption 3, it is useful to show how $c_1$ and $c_2$ reflects in the upper bound in Theorem 3.1. A related question: it is intuitively hard to understand why the lower bound is needed: if the probability density is not supported on a subdomain, it does not appear in the expectation in the definition of $\|\cdot\|_{\ell_2}$ either, so why would that be a problem?

5. In Theorem 3.1, does $L$ need to be equal to the quantity on the right-hand side, or is it more of an inequality?

6. In Theorem 3.1, what does "with probability approaching $1$" mean exactly? Also, is it possible to rephrase the theorem into something like "with probability no less than $1-\delta$" and have $\delta$ in your upper bound?

7. In section 4, have you tested residual-based neural networks? They are known to have better training stability and it is interesting to see if they can help avoid the issues in the baseline.

### Soundness
3

### Presentation
2

### Contribution
2
