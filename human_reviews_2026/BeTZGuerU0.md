# Kernel-based Robust Markov Subsampling for Regularized Nonparametric Regression with Contaminated Data

- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Large-scale data with contamination are ubiquitous in biomedicine, economics and social science, but
its statistical learning often suffers from computational bottlenecks and robustness.
Subsampling offers an efficient solution by sampling a representative subset of uncorrupted data from full dataset, thereby reducing computational costs while enhancing robustness. Existing subsampling methods, like leverage- and gradient-based approaches,
focus on parametric models and fail under nonparametric models or severe contamination.
To address these limitations, we propose a kernel-based robust Markov subsampling (KRMS) method for nonparametric regression with
contaminated data in reproducing kernel Hilbert space (RKHS). By dynamically adjusting Markov sampling probabilities based on
the ratio of residuals to kernel norms of predictors, our method simultaneously suppresses contaminated observations
and prioritizes informative observations, enabling robust learning from contaminated datasets. Theoretically, we establish the asymptotic properties of the estimators, including consistency and asymptotic normality, and generalization bounds under RKHS regularization, providing the first unified framework for robust subsampling in nonparametric settings.
Simulations and real-data applications demonstrate KRMS’s superiority over existing methods,
particularly for high contamination levels. Our approach bridges a critical gap in scalable and
robust statistical learning, with broad applicability to large-scale.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a penalized nonparametric kernel regression method under data contamination and introduce a Markov sub-sampling method, specifically Algorithm 1: Robust Kernel-based Markov Sub-sampling. The core of the method relies on a Metropolis-Hastings rejection scheme based on the residual kernel-norm score in (3).

### Strengths
- The authors are commended for providing a comprehensive theoretical analysis, including convergence rates, asymptotic distributions, and generalization error analysis.
- The proposed sub-sampling method effectively reduces the proportion of contamination in the data from $\theta$ to $\theta^\prime$, where $0 \le \theta \le \theta^\prime$, thereby enhancing the robustness and effectiveness of the regression.

### Weaknesses
- The experimental analysis, while covering both synthetic datasets (linear and nonlinear) and real-world datasets (financial and air quality), lacks a clear baseline comparison. Specifically, experiments under uncontaminated conditions are missing, which are essential to validate the effectiveness of the proposed method under varying contamination probabilities $\theta$.
- The concept of distribution $P^\prime$ being a "cleaner" version of the initially contaminated distribution $P$ is not sufficiently quantified. It remains unclear how much "cleaner" $P^\prime$ is compared to $P$, and whether there is any theoretical guarantee regarding the value of θ′ achieved by the algorithm.
- There is no analysis of the computational complexity of the proposed algorithm 1, which is critical for assessing its scalability and practical applicability.

### Questions
1.  Are there theoretical guarantees regarding the extent of contamination reduction, i.e., the value of $\theta^{\prime}$ achieved by the algorithm?
2. Could the authors provide a theoretical or empirical analysis of the time complexity of the proposed algorithm?
3. The experimental section is placed in the appendix, likely due to space constraints. However, given the heavy reliance on experimental validation in this work, would it be possible to restructure the paper to integrate the experiments into the main body for better readability and emphasis?

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
4

### Summary
This paper introduces a novel Kernel-based Robust Markov Subsampling (KRMS) method for nonparametric regression with contaminated data, addressing critical challenges in computational scalability and statistical robustness for large-scale datasets. The authors propose a residual kernel-norm scoring mechanism within a Reproducing Kernel Hilbert Space (RKHS) framework, which dynamically adjusts Markov sampling probabilities to suppress contaminated observations while prioritizing informative ones. Theoretically, the work establishes asymptotic properties including consistency, asymptotic normality, and generalization bounds under Huber's contamination model $P = (1-\theta)F + \theta Q$. Empirical evaluations demonstrate KRMS's superiority over existing subsampling methods, particularly under high contamination levels. This research bridges a significant gap in scalable robust nonparametric learning, offering a unified framework with broad applicability to contaminated, non-i.i.d. data in scientific domains.

### Strengths
Methodological Innovation: KRMS represents the first subsampling approach specifically designed for contaminated data in nonparametric regression settings, leveraging RKHS geometry to effectively separate contaminated observations that would be indistinguishable in original feature spaces.

Theoretical Rigor: The paper provides comprehensive theoretical guarantees including uniform ergodicity of the Markov chain (Theorem 1), consistency of estimators (Theorems 2-3), functional Bahadur representation (Theorem 4), asymptotic normality (Theorems 5-6), and generalization bounds (Theorem 7), establishing a solid foundation for the method.

### Weaknesses
The algorithm lacks convergence guarantees for the parameter estimation procedure, particularly for the recursive updating of $\alpha^{(\kappa)}$ in Algorithm 1, which is especially concerning under high contamination where initial estimates may be poor. The initialization $\alpha^{(1)} = \alpha^{(0)} + 0.2$ is arbitrary without justification, potentially affecting reproducibility and convergence behavior. Experimental limitations include insufficient parameter selection guidelines for critical values like subsample size $n_0$, burn-in period $t_0$, maximum iterations $T_0$, and stopping criterion $\xi_0$, with no sensitivity analysis provided. The exclusive use of Gaussian kernel $K(x, t) = \exp\{-(x-t)^2/4\}$ without exploring alternatives or analyzing bandwidth parameters limits understanding of kernel dependence. The experimental scale with only $N = 10,000$ observations and $p = 4$ dimensions fails to demonstrate scalability to truly massive datasets or high-dimensional settings where contamination effects would be more pronounced. Additionally, the method is restricted to continuous responses without discussion of extensions to classification problems.

Comparisons lack state-of-the-art robust nonparametric regression methods and deep learning approaches, with existing comparisons primarily against parametric methods creating an unfair advantage. No systematic sensitivity analysis is provided for contamination level $\theta$, subsample size $n_0$, regularization parameter $\lambda$, or kernel parameters. Theoretical assumptions present several concerns: Condition 1's uniform ergodicity requirement may not hold in practice with complex dependencies; Conditions 3-4's smoothness assumptions are overly restrictive for real-world applications; and the contradiction between Condition 1's Markov dependence and Condition 2's i.i.d. errors is not addressed. The distributional gap between theoretical analysis (assuming $P'$) and algorithm operation (on $\tilde{D}$) lacks rigorous justification. Proof validity issues include the unjustified convergence rate $O_p(\ln n/n^2)$ in Theorem 4 under contamination, and the impractical simultaneous conditions $\lambda = o(1)$ and $(-\ln \lambda)^{1/2}/\omega \sim n^{1/(2m+1)}$ in Theorems 5-6. Notational inconsistencies (e.g., $H_\omega(s,t)$ vs. $K_\omega(s,t)$ in Appendix B) and ambiguous definitions (e.g., initial $H(X)$ specification) further reduce clarity.

### Questions
Refer to the weakness.

### Soundness
2

### Presentation
3

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
This paper proposes KRMS, a robust subsampling framework designed for non-parametric regression with contaminated data. Specifically, KRMS introduces a residual kernel-norm score, which operates in the reproducing kernel Hilbert space and effectively identifies outliers. Moreover, this paper provides theoretical guarantees for the KRMS estimator, establishing its consistency, asymptotic normality, and generalization bounds.

### Strengths
- The presentation is clear and well-organized.
- The paper presents comprehensive theoretical analyses for the proposal.
- The paper provides both simulated and real-world experiments, demonstrating superior performance.

### Weaknesses
- While the theoretical analysis is solid and thorough, the methodological contribution appears limited, relying mainly on the kernel-trick-based residual score.
- Introducing the residual score into kernel space is an interesting idea. Nevertheless, it would be much better if the paper included an empirical ablation study comparing the kernel-trick version with its linear-space counterpart.

### Questions
Please refer to the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes KRMS, a Metropolis–Hastings (MH) subsampling scheme for kernel ridge regression under Huber contamination. The authors prove uniform ergodicity of the induced chain (finite state space), establish consistency and pointwise asymptotic normality in a symmetric periodic Gaussian RKHS, and derive a generalization bound for u.e.M.c. samples. Simulations and two “real‑data” illustrations are reported.

### Strengths
The target problem is an important regime: robust, scalable learning in RKHS under contamination.

The paper provides clear, intuitive heuristic: prefer points with small residual relative to their “kernel similarity magnitude”.

### Weaknesses
1. The main weaknesses lie in the “cleaner” stationary distribution $\mathcal{P}'$, which lacks clear explanation. And the paper states \mathcal{P}' has less contamination $\theta'<\theta$. However, this statement is not proved.  The chain with acceptance $\min \\{1,w(z_t)/w(z^*) \\}$ has stationary distribution proportional to $1/w(\cdot,\alpha)$ (with $\alpha$ frozen), not obviously to a mixture $1-\theta')F+\theta'Q$ with $\theta'<\theta)$ The paper asserts convergence to a distribution with reduced contamination yet provides no identification of the MH target beyond ergodicity, nor any argument that $1/w$ upweights uncontaminated draws in the sense of a reduced mixture weight. Theorem 1 (irreducible+aperiodic on a finite set ⇒ uniform ergodicity) does **not** characterize the limit distribution. Consequently, Theorem 7’s bound—with a leading $48M^2\theta'$ term—remains vacuous.

2. In Algorithm 1, the paper sets $\alpha^{(1)} = \alpha^{(0)} + 0.2$. Please justify the logic here.

3. In Algorithm 1, to evaluate $w(z,\alpha)$, the paper needs to compute the weighted sum of $n$ $K(x_i,x_j)$, which results a complexity of $n$. This contradicts the claim of the complexity $O(T_0(n_0^2 p+n_0^3))$. 

4. It is hard to comprehend the numbers in Tables 1-6. It is better to emphasize key numbers.

### Questions
na

### Soundness
2

### Presentation
2

### Contribution
1
