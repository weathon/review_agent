# Adaptive Accelerated Gradient Descent Methods for Convex Optimization

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
We propose A$^2$GD, an adaptive accelerated gradient method for convex and composite optimization. Drawing inspiration from stability analysis in ODE solvers, the method updates smoothness and convexity constants through Lyapunov-based formulas and invokes line search only when accumulated perturbations turn positive, an event that is empirically rare. This yields a dramatic cut in gradient evaluations while preserving strong theoretical guarantees. By integrating adaptive step size and momentum acceleration, A$^2$GD outperforms existing first-order methods across diverse problem settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The article proposes an adaptive (accelerated) gradient descent method that estimates smoothness and strong-convexity parameters on the fly. There are two key ideas: i) designing a suitable energy function that measures descrease, ii) relaxing the decrease condition, such that sufficient decrease is not necessarily imposed at every step, but only in a suitable avergage sense. The ideas are underlined with a few small scale experiments including regularized logisitic regression, a stylized semidefinite programming problem, and a simple nonconvex regularized least-squares problem.

Overall the article is well written and fun to read.

### Strengths
The algorithmic approach is relatively simple and the numerical examples (albeit small-scale and idealized) show substantial improvement compared to baselines. The content is presented in a coherent and mostly clear manner.

### Weaknesses
- The originality of the article is limited. Adaptive-step size selection is a theme that has been studied since the very early days of optimization (~50ies/60ies). The extension from enforcing decrease of a certain quality-function at every iteration towards enforcing sufficient decrease over multiple iterations is relatively minor.

- In the accelerated situation, which is arguably the less standard one, there are a few algorithmic details that seem rather ad-hoc. These include an ad-hoc lower bound on \mu_k (estimate of strong-convexity), ensuring monotonic descent of function values in accelerated gradient descent (I have some concerns about this, as non-monotonic decrease seems important for achieving acceleration from a theoretical point of view), and a warm-up phase. These ingredients are mentioned quickly, not included in the pseudo-code of Algorithm 1, and is stated that omitting these might cause instability. Hence, the effect of these ad-hoc ingredients seems important and should therefore be mentioned in the pseudo-code and corresponding ablation studies should be made.

- The experiments are rather limited and include relatively stylized problems (although three fundamentally different problem types). In particular, acceleration is claimed and for this sort of article I would expect to have seen numerical results that indicate how the convergence rate scales with O(\sqr(kappa) log(1/eps)) in the strongly convex regime, i.e., studies with varying kappa and empirically fitting the corresponding convergence rate, and results that show how O(1/k^2) is achieved in a non-strongly-convex setting. Some of this weakness could be addressed if the author could run their algorithm e.g. on problem (14) with varying \lambda, and even for \lambda=0 and using the same hyper-parameters in their algorithm for all problem instances.

### Questions
Is there a direct relationship between number of gradient evaluations and execution time? If so, I would recommend the authors to point this out. It seems that additional gradient evaluations are needed to carry out the line search, since the error criteria depends on successive iterates. Are these gradient evaluations accounted for in the numerical results?

Would the method also work in the context of stochastic gradient descent/mini-batching?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies gradient descent methods with line search, aiming to reduce the number of line search steps. In the standard method, line search is performed in every iteration where the error term is positive. In the new method, an accumulation of additive error is maintained and line search is only performed when the sum of error terms is positive. As long as the sum of error terms is negative, the objective value goes down exponentially with the observed step sizes.

### Strengths
The proposed method is relatively simple and exploits a natural observation that there are several error terms in the standard analysis and forcing all terms to be negative might be too conservative and costly. The proofs build on largely the standard arguments with one new argument for keeping track of the sum of the error terms.

The experiments show strong performance compared with other theoretical adaptive methods and un-tuned non-adaptive methods.

### Weaknesses
The accelerated result has two new hyperparameters R and epsilon compared with the standard method, which negates the adaptivity advantage of the method.

There are a lot of heuristics in the new method (heuristic line search, warm-start to set mu) and several extra hyperparameters. In the experimental comparison, it looks like the non adaptive methods are not tuned at all, leading to oscillating behavior. This is already observed in the literature e.g. O'Donoghue, Candes. Adaptive Restart for Accelerated Gradient Schemes FoCM 2013. This is a very pessimistic view and not aligned with any practical deployment of these methods. What if we also run these methods briefly/on smaller data and tune the step sizes (or restart) to avoid the oscillation?

Another weakness is that it is unclear how the new method based on line search is going to generalize to the stochastic setting. The introduction of the paper suggests that an important motivation is to understand and improve upon Adam, AMSgrad, etc. These methods are generally applied in the stochastic setting and do not use line search at all. The experimental comparisons also do not involve any of these methods.

Other comments: in figures 5,6,7, the red dots are not visible.

### Questions
Could you please comment on the comparison between this work and the works on adaptive algorithms without line search, for both deterministic and stochastic settings such as
Online Adaptive Methods, Universality and Acceleration. NeurIPS 18.

### Soundness
3

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
This paper presents a "line search-reduced" adaptive acceleration scheme for convex and composite optimization. The key idea is to track an accumulated perturbation variable and trigger line search only when it becomes positive. This mechanism adaptively updates both the smoothness and strong-convexity estimates, achieving $O(1/k^2)$ for convex problems and accelerated linear convergence for strongly convex ones.

### Strengths
- The paper proposes an interesting "line search-reduced" adaptive acceleration method through the elegant use of a running perturbation balance $p_k$ to trigger line search only when the Lyapunov-stability condition $p_k \leq 0$ is violated. 

- The paper provides clear convergence guarantees for both convex and strongly convex objectives. The way $L_k$ and $\mu_k$ are adaptively controlled makes sense and the Lyapunov-based analysis is well presented. 

- It's nice to see a clean algorithmic idea that really tries to reduce line search overhead while achieving provable acceleration. That said, it'd be even more convincing to see how sensitive the method is to the various heuristics and hyperparameters (see Weaknesses).

### Weaknesses
- There's no theoretical upper bound on the number of line search activations. While each backtracking loop is shown to finish in $O(\log L)$ steps, the paper doesn't say how often these activations happen overall. In the worst case, frequent triggers could blow up the total gradient evaluations and weaken the claimed overall complexity. 

- The experimental evaluation is somewhat limited. Results are shown only in terms of gradient evaluations; no wall-clock time, memory or cost breakdown (e.g., line-search triggers, eigen-decomp/prox calls). Especially for the composite MLE task, where each prox involves an eigen-decomp, the claimed runtime savings aren't quantified. Also, all tests are mid-scale convex problems. 

- The method uses several heuristics, such as the accept/reject rule, restart, AdProxGD warm-up, but the paper doesn't analyze how much these actually matter.

### Questions
- Can you provide a theoretical or empirical upper bound on the number of line search activations? Is the frequency roughly sublinear or constant in practice?

- Could you report wall-clock time or a cost breakdown (gradient, prox, line search) to show real runtime benefit?

- How does the proposed method scale to larger problems? How does it perform on ill-conditioned problems, e.g., by varying $L/\mu$ to test its robustness?

- Can you share ablation results for the heuristics (accept/reject rule, restarts, warm-up) to see how much they impact convergence and stability?

- Have you tested parameter sensitivity ($\mu_0, L_0, R, \varepsilon$, etc.) to check robustness of the adaptive behavior?

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
This paper proposes a new line-search-based accelerated gradient method for smooth (strongly) convex minimization. Traditional line-search-based methods typically require multiple function evaluations per iteration, whereas the proposed Lyapunov-based approach triggers line search only when the accumulated perturbation becomes positive. The analysis shows that the method achieves a near-optimal convergence rate while significantly reducing the number of line-search operations. Experimental results demonstrate that the proposed method outperforms existing line-search-based and adaptive accelerated methods.

### Strengths
Because line search can be computationally expensive, there has been considerable interest in developing line-search-free adaptive methods. In this context, proposing a method that triggers line search only occasionally, when it is truly needed, is novel and worth investigating.

### Weaknesses
Although I found the analysis and the results in Figures 1 and 2 for the gradient descent interesting, the proposed Algorithm 1 introduces several additional components, such as initialization with ten iterations of gradient descent and restarting, which make it difficult to isolate the source of improvement, especially in the experimental results. In other words, I would like to see a fair comparison focusing solely on line-search-free aspect, without incorporating other auxiliary components. For example, all existing methods in Figure 3 exhibit oscillatory behavior, and it is well known that restarting alone can substantially mitigate such oscillations, as observed in your method.

### Questions
- Line 207: Is it correct that an approximation $\mu_k=\min_{1\le i\le k} L_k$ is used to compute $\delta_k$? This choice seems like a potential overestimation, especially since $L_k$ is rarely updated. Are you plotting $p_k$ in Figure 1 using this $\mu_k$? If yes, how would the results differ if the exact $\mu$ were used in computing $p_k$?

- Line 220: I am not sure what is meant by "in a weighed $\ell_2$ sense". What exactly is the weight in this context?

- Line 253: Could you clarify what is meant by "improving efficiency"? It appears that, although line search is not triggered, the method still employs the adaptive step size $\alpha_{k+1}$, which is commonly used in line-search-free approaches. If this is correct, I believe the paper's claimed contribution should be reconsidered. When $\alpha_{k+1}$ is already adaptively chosen, it is not clear why line search remains necessary.

- Line 257: Do you also use this approximation when computing $b_k^{(1)}$?

- Line 273: Could you explain the motivation for considering the non-standard Hessian-based Nesterov accelerated gradient in (11) ?

### Soundness
2

### Presentation
3

### Contribution
2
