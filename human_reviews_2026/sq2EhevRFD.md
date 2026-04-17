# Eliminating Steady-State Oscillations in Distributed Optimization and Learning via Adaptive Stepsize

- Decision: Reject
- Scores: 0, 4, 6, 2

## Abstract
Distributed stochastic optimization and learning is gaining increasing traction due to its ability to enable large-scale data processing and model training across multiple agents without the need for centralized coordination. However, existing distributed stochastic optimization and learning approaches, such as distributed SGD and their variants, generally face a dilemma in stepsize selection: a small stepsize leads to low convergence speed, whereas a large stepsize often incurs pronounced steady-state oscillations, which prevents the algorithm from achieving stable convergence accuracy. In this paper, we propose an adaptive stepsize approach for distributed stochastic optimization and learning that can eliminate steady-state oscillations and ensure fast convergence. Such guarantees are unattained by existing adaptive stepsize approaches, even in centralized optimization and learning. We prove that our proposed algorithm achieves linear convergence with respect to the iteration number, and that the convergence error decays sublinearly with the batch size of sampled data points. In the specific case in terms of deterministic distributed optimization with exact gradients accessible to agents, we prove that our proposed algorithm linearly converges to an exact optimal solution. Moreover, we quantify that the computational complexity of the proposed algorithm is on the order of $\mathcal{O}(\log(\epsilon^{-1}))$, which matches the existing results on adaptive stepsize approaches for centralized optimization and learning. Experimental results on machine learning benchmarks confirm the effectiveness of our proposed approach.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper studies adaptive algorithms in federated learning where there is no central server involved in the stepsize scheduling. They call this a fully decentralized setting. Then, they provided an adaptive stepsize algorithm which is claimed to eliminate steady-state oscillations. Numerical experiments are performed to validate the results.

### Strengths
The fully decentralized stepsize scheduling seems to be a new setting. In this setting, the authors provided a method and showed linear convergence under some circumstances.

### Weaknesses
First of all, I find the motivation of this setting quite weak. In the stepsize scheduling, the number of bits to communicate is usually a small constant (compared to communicating a full gradient). Therefore, even in a decentralized network, this usually takes much less time than the gradient gossipings. I cannot see any practical reasons why the (global) stepsize coordination takes too much of communication time. 

Moreover, the theorem statements in the paper did not provide good convergence guarantees. For instance, in eq. (2) in Theorem 1, this is not even a convergence result with a non-vanishing term of $O (\frac {\sigma^2} {|\mathcal B|})$. 

I feel very suspicious about the correctness of the main technical results. For instance, in Corollary 1, the authors claimed to get an $\epsilon$-solution in $O ((|\mathcal B| + M) \log \frac 1 \epsilon)$ gradients. This claim also does not follow from Theorem 1.

### Questions
See the “weakness” paragraphs above. I believe that the setting of fully decentralized stepsize scheduling is not well motivated and the main technical claims in the paper are wrong.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work solves the problem of stepsize selection in distribution optimization. For this purpose, the authors propose Algorithm which automatically sets the value of stepsize and prove the convergence guarantee for this algorithm. Algorithm 1 achieves comparable convergence rate with the centralized optimization. Experimental results confirm the effectiveness of Algorithm 1.

### Strengths
1. This work studies an important problem of selecting stepsize to avoid both slow convergence and steady-state oscillations.

2. The authors provide stong theoretical guarantee for the proposed algorithm.

3. The experiment results (such as Figure 1) demonstrate the claims of this work.

### Weaknesses
1. Hyperparameter tuning: While automatically setting the stepsize, Algorithm 1 introduces additional parameters including $\beta$ $r$ and $M$, requiring futher tuning. The experiments in Figure 3 cannot solve this problem, as training with a batch size of 128 on the MNIST dataset is a very stable process

2. Lack of large-scale experiments: The experiments use five nodes and a specific data distribution, which may be not practical. It is hard to see if the results can generalize to other setups.

3. If I understand correctly, Theorem 1 requires $M > M_0$  inner-consensus-loop iterations for convergence. It is hard to estimate the value of $M_0$. If this value is very large, the theoretical gurantee does not hold in practice.

### Questions
1. How to select the values of $\beta$ $r$ and $M$ in applications?

2. How do Algorithm 1 compare with baselines in a large-scale setup and different data distributions?

3. How to select coupling weight W? Is the choice in Section 6 manually tuned or randonly set?

4. In Figure 1(f), why does the Stepsize decrease suddenly? Is this phenomenon met in every running of CIFAR10?

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
5

### Summary
This paper proposes an adaptive stepsize approach for distributed stochastic optimization and learning, which can eliminate steady-state oscillations and ensure fast convergence. For the deterministic gradients, the authors prove convergence to an exact optimal solution, whereas for stochastic gradients, they establish linear convergence with respect to the iteration number and prove that the convergence error decreases sublinearly with the batch size of sampled data points. The convergence results are reasonable to me.

In my opinion, the most interesting property of the proposed adaptive stepsize approach is that it allows the stepsize to be large in the early stages of the algorithm to accelerate convergence, while rapidly decreasing to a small value near the optimal solution to ensure stable convergence performance. This property appears to provide a "stop signal" for distributed optimization algorithms. If so, it would be useful in real-world distributed optimization applications, as most existing algorithms lack a clear criterion for determining when to stop.

Nevertheless, I have some concerns about this paper, which are summarized below:

(i) Lack of discussion on nonconvex settings (see Weakness 1 for details); 

(ii) Limited variety of experimental tasks (see Weakness 2 for details);

### Strengths
The paper is well written and easy to follow. The convergence proofs are complete and rigorous. In addition, the proposed adaptive stepsize approch provides an easily identifiable stop signal for distributed algorithms (for example, when the stepsize falls below a given threshold), which is of practical significance.

### Weaknesses
1. Lack of discussion on nonconvex settings: Although the experimental results demonstrate the effectiveness of the proposed approach in nonconvex settings, it would be helpful if the authors could elaborate on how the current theoretical results might be extended to nonconvex objective functions.

2. Limited variety of experimental tasks: Aside from logistic regression (in Appendix C), the experiments mainly focus on image classification tasks. To more comprehensively evaluate the effectiveness of the proposed approach, it would be beneficial to test it on other types of machine learning tasks, e.g., distributed training of recurrent neural networks (RNNs) for next-character prediction tasks.

If the authors could address Weaknesses 1 and 2, I would consider raising the score.

### Questions
See the weaknesses above. In addition, I have the following questions:

1. In Algorithm 1, the parameter $\beta$ is set as $\beta \in (0, 1.36)$. However, if $\beta < 1$, the stepsize in Algorithm~1 Line 16 would decay exponentially, which may prevent the algorithm from achieving convergence. I also observed that Eq. (43) in Appendix uses $\beta \in (1, 1.36)$. Please clarify the range of $\beta$ to ensure consistency.

2. In the CIFAR-10 experiment, Algorithm 1 does not appear to achieve stable convergence. Therefore, the authors should increase the number of iterations to more clearly show the convergence trend.

3. The use of the iterates $y_{i,2}^{t}$ requires further explanation. It seems to be an additional variable compared with conventional gradient-tracking approaches. 

4. There are some typos, e.g.,  (i) under Eq. (7) in Appendix, $||\frac{1}{m}\sum_{i=1}^m a_i||^2 \leq \frac{1}{m}\sum_{i=1}^m ||a_i||^2$ should hold for any vector $a_i \in \mathbb{R}^n$ rather than $a_i \in \mathbb{R}$; and (ii) in Eq.(75), the closing bracket "]'' should be replaced by "\}''.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a new algorithm with an adaptive step size for decentralized optimization problems. The authors provide convergence theorems and experimental results for this algorithm.

This paper should be rejected for the following reasons: (1) It contains false theorems and (2) It uses mathematical statements without proof.

### Strengths
The authors research the interesting scientific problem - the adaptive step size for decentralized optimization.

### Weaknesses
To prove the rejection, I provide the following arguments.

**Main arguments**

1. Theorem 2 contains a false statement about the convergence of the proposed Algorithm ($ \mathbb{E}||x_i^T - x^*||^2 \leq \mathcal{O}(\gamma^T)$, where $\gamma = \max (1 - \frac{\mu^2}{4L}, \frac{91}{92})$), because if this statement were true, we would get any convergence by rescaling. Moreover, this result contradicts standard lower bounds for $L$-smooth and $\mu$-strongly convex functions with $n=1$ and $\sigma = 0$.

2.  P. 16 line 858. The authors used the fact that $L_i^t \leq L$, where $L_i^t$ is defined in Algorithm 1 and $L$ is defined in Assumption 1. Can you provide proof for this fact?
  
3. P. 20 line 1034. I can't understand this transition. Can you clarify it?

4. p. 20 last inequality. It is not obvious why we can always choose such parameters for all $i$.

5. Also, the relationship (27)  is not obvious.

**Minor comments**

1. How do we define $L_{i}^{t+1}$ if $x_{i}^{t+1} = x_{i}^t$?

2. P. 17 eq. (14). It seems like this equation contradicts Assumption 1.

### Questions
I provided questions in the weakness section.

**Things to improve the paper that did not impact the score:**

Provide theoretical guarantees for each statement in this work.

### Soundness
2

### Presentation
3

### Contribution
2
