# Time-Varying Bayesian Optimization Without a Metronome

- Avg Score: 4.00
- Decision: Reject
- Scores: 8, 4, 2, 2

## Abstract
Time-Varying Bayesian Optimization (TVBO) is the go-to framework for optimizing a time-varying, expensive, noisy black-box function $f$. However, most of the asymptotic guarantees offered by TVBO algorithms rely on the assumption that observations are acquired at a constant frequency. As the GP inference complexity scales with the cube of its dataset size, this assumption is unrealistic in the long run. In this paper, we relax this assumption and derive the first upper regret bound that explicitly accounts for changes in the observations sampling frequency. Based on this analysis, we formulate practical recommendations about dataset sizes and stale data policies of TVBO algorithms. We illustrate how an algorithm (BOLT) that follows these recommendations performs better than the state-of-the-art of TVBO through experiments on synthetic and real-world problems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper analyses the problems setting of Time-Varying Bayesian Optimisation, where the time intervals between observations are potentially not identical. This is a problem that is likely to occur in real-world applications, as GP is famously slow to fit on large datasets due to its N^3 cost, and if the underlying function changes quickly, waiting too long might mean previously gathered data becomes irrelevant. At the same time, discarding old data too fast will be suboptimal for slowly varying function. The paper attempts to find a solution to this problem. In Theorem 3.6, author derive a novel regret upper bound that explicitly ties the algorithm's regret to the waiting time and pace of change of the time-varying component of the kernel function. Then authors describe, how based on that criterion, the optimal size of dataset can selected in practice. Once deciding on the size of dataset to keep, authors leverage the existing Wasserstein-distance criterion for deciding which points to remove, but provide new insights on it by introducing novel theoretical analysis. Lastly, authors benchmark the proposed algorithm against alternatives.

### Strengths
- The problem studied in the paper is very relevant and almost guaranteed to occur in any real-world application of time-varying BO. At the same time, the problem seems to be overlooked in existing literature. As such, this paper fills an important gap in existing research

- The theoretical results authors propose are interesting and insightful. The fact that we can derive practical criterions for selecting dataset sizes to keep based on them makes them very relevant

- The additional insights on the WasserStein-distance point removal heuristic authors provide in Theorem 3.8 nicely complement previous literature

- It looks like the proposed alternative clearly outperforms existing solutions, showing there is a clear practical value in the derived algorithm

- The paper goes beyond merely reporting the regret results and provide a comprehensive analysis of the experiments, showing how the behaviour of their algorithm changes depending on the smoothness of the underlying function

### Weaknesses
- The presentation of the paper could be slightly improved. I believe the authors never exactly write what kernel function do they use in their experiments. Such details should be clearly provided, ideally in main body. Also, I believe in the bound in Theorem 3.6, it would be nice to explain why $(1 - C_2||\mathbf{u}_n||_2^2)$ is always greater than zero. Of course, this can be deduced knowing that the temporal kernel decays to zero for large time lags, but to make the paper more accessible, I believe a sentence or two explaining this in the main body would be appropriate.

### Questions
- What spatial and temporal kernel functions did you use?
- Can you expand on why solving the optimisation problem in (5) is difficult? It seems to be that you have some objective you need to optimise over natural numbers (up to some maximum horizon length H). As such, you just need H function evaluations, which does not seem like a lot?
- For the relaxed problem you derive, can you quantify the gap between the solution you find and the exact, analytical solution to (5)? Even an empirical comparison would be appreciated.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a method for time-varying Bayesian optimization (TVBO), in which an iteration of BO proceeds with cost R(n) for dataset size n. The regret analysis of the UCB acquisition function is provided for general R(n), which has not been revealed. The proposed algorithm reduces the dataset size into n, specified by the minimization of the regret bound, because the response time diverges if n diverges.

### Strengths
The problem setting seems interesting and the theoretical analysis for general R(n) is shown first by the paper according to the authors.

### Weaknesses
The problem setting should have been explained in more detail, while the authors mention that it has not been widely studied.

For me, it is currently unclear how theoretical analysis justifies the proposed model. The rate itself of the regret bound is not largely different from existing studies, and the novelty is claimed for revealing the relation between the bound and response time R(n). However, since the bound is quite loose, I do not think the derived bound reveals some essential relation between R(n) and the regret. The data selection algorithm is a kind of simple greedy-like strategy (discard samples having a least effect on the model difference wrt Wasserstein dist at every iteration), and this strategy itself is not fully justified.

### Questions
I don't fully understand problem setting (experimental setting). According to appendix, horizon H is 600 sec. Does this mean that BO continues until the sum of the response time becomes 600? The definition of the response time itself is a bit vague. What is 'the time that separates two consecutive iterations'? Usually, BO usually assumes high observation cost for objective function. During the time for waiting observation, 't' proceed?  Does the response time and/or horizon H contain this observation cost? If it contains, what is the observation cost in the experiments? If it does not contain, why? 

It is well-known that the standard regret bound by Scrinivas et al 2012 is quite loose, because of which the regret bound (4) should also be quite loose. Therefore, I am not fully sure if the relation between the bound and R(n) has a substantial meaning. Further, in my current understanding, terms removed during deriving the bound also may depend on R(n) (eg., \Delta seemingly depends on t_n), which also makes a substantial meaning of analyzing the relation between the regret bound and R(n) through (4) unclear. As a result, I also do not fully understand how the minimization of (4) can be justified to select n.

Why minimizing (4) is reduced to the minimization of |u_n|^2? Why does the other 'n' in (4) not have any effect?

How is integrated 2-Wasserstein distance calculated? What is the computational complexity? In Bardou et al. (2024b), some approximation is introduced. The same approximation is used? The bound (7) is seemingly quite loose, and so, a quite loose bound is further approximated. How can it be a justification for using integrated 2-Wasserstein distance?

In the paragraph after (5), is the statement 'Starting from n_0 ... converges to n*' proven?

An observation is discarded by W_2 at every time when an additional observation is obtained. Comparing difference between the reduced dataset and the 'full' dataset, can you provide any justification? I guess Theorem 3.8 only (weakly) justifies one time application of the W2 minimization. When the W_2 based selection is applied multiple times, it does not mean that the resulting dataset corresponds to the minimization of the upper bound (7). Further, even if the upper bound (7) is minimized, it is also a quite loose bound.

Theorem 3.6 does not consider how the size n reduced dataset is created, which is actually selected through W_2, i.e., a data-driven manner. Therefore, in the actual process of the proposed algorithm, for example, n and u_n can be seen as a random quantity. Is the inequality in Theorem 3.6 still hold even for those random n and u_n (and C_2?)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers the time-varying Bayesian optimization, also known as non-stationary Gaussian process (GP) bandits or kernelized bandits. For this problem, this paper focuses on the computational time of the GP model and proposes a policy to discard the obtained dataset that is far away from the current time. Furthermore, this paper discusses the theoretical result and provides comparisons with several baseline methods.

### Strengths
Overall, this paper is well written, and I can understand that the GP model's computational time may be slow, since it is $O(n^3)$.

### Weaknesses
Regarding the problem setup and motivation:
- If we adopt the rank-one update of the GP model, the computational time in each step is $O(t^2)$. Bayesian optimization generally considers the case where the objective function evaluation is more dominant compared with the computational time $O(t^2)$. Are there any specific examples where $O(t^2)$ can be a severe bottleneck?
- If the computation of the GP model can be a bottleneck, we can consider other surrogate models or other optimization approaches. Is there no need to discuss and compare such other frameworks?
- Many studies derive efficient approximation techniques for GPs. Is simply using such approximation techniques, such as sparse GPs, not sufficient?

Regarding the theoretical results:
- There is no definition of the maximum information gain $\gamma\_T$. In the time-varying Bayesian optimization, the definition of $\gamma\_T$ is slightly changed and larger than that of the usual setup since the time step is additionally considered. Therefore, to discuss the order of the regret upper bound explicitly, its definition and the order should be described explicitly in the main paper.
- Theorem 3.6 does not seem to be suggestive since it is inconsistent with the actual algorithm. In Eq. (19), it is assumed that the observations after $n$-th one are discarded. However, in the actual algorithm, the observations before $(t - n^\star)$-th one are discarded. Thus, it is inappropriate to choose $n^*$ based on Theorem 3.6.

Others:
- There is a lack of important related works, such as [1].

[1] Shogo Iwazaki, Shion Takeno, Near-Optimal Algorithm for Non-Stationary Kernelized Bandits, Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, vol. 258, pp. 406-414, PMLR, 2025.

- The authors describe that the existing and proposed results are asymptotic. However, since they hold for any $T$, they are not asymptotic.

### Questions
Please answer the above questions.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates Bayesian optimization in time-varying settings and irregular sampling. Specifically, the paper bounds the cumulative regret of the irregularly observed locations and connects the regret to the sampling frequency. Based on these results the paper derives the optimal size of the dataset and proposes an TVBO algorithm that leverages these results.

### Strengths
- The proposed algorithm comes with regret guarantees and  convincing empirical evaluations.
- I found the results on optimal dataset sizes especially intriguing.

### Weaknesses
- Unfortunately, the biggest weakness of the paper is that the problem formulation is nonsensical in the sense that a trivial algorithm can achieve constant regret.

 The problem lies in the definition of regret that sums only the regret at the observed locations $R_T = \sum r_i$ and letting the algorithm decide on the sample times $t_i$. A trivial algorithm that will always achieve $\Theta(1)$ regret samples once and never again achieves $R_T = r_1$. Clearly, this solution is non-sensical in the context of time-varying optimization.
 
 A relevant problem formulation in the irregular sampling setting would need to sum the regret occurred at the last chosen $x$ at a fixed interval despite the algorithm not choosing a *new* $x$. 

- Table 1 is misleading. Regret bounds require regularity assumptions on the objective. Claiming BOLT places no assumption on $f$ is wrong.

This is a bit of a nitpick, but without placing restrictions on $f$ we also allow adversarial objectives. For example, for the function 
$$
  f(x,t)=\begin{cases}
    2^t, & \text{if $x = x_i$}.\\
    0, & \text{otherwise}.
  \end{cases}
$$
there exists no algorithm that can achieve linear regret. Indeed, the paper places many assumptions on $f$ despite stating otherwise in Table 1.
Still, I appreciate that the regret bounds allow for a wider variety of temporal correlations than those in previous work.

### Questions
Questions:

- I think an interesting consequence of Theorem 3.6 is the fact that for many kernels decreasing the response time will lead to a lower regret in the fixed data set setting. Can we recover sublinear regret by sampling fast enough and letting $||u_n||^2_2$ approach 1?

Suggestions:
- I would encourage the authors to look at the setting I described in the weakness section. I think the presented results and the proposed algorithm carry over to the setting where "unobserved" evaluations still incur regret. Albeit some modification to the theory are necessary. Unfortunately, the required changes are to substantial for a rebuttal.
- Minor: Add citation keys to Table 1.

Notes:
- GP inference does not scale as $\mathcal{O}(n^3)$ in the streaming setting (line 330). Adding a new point scales as $\mathcal{O}(n^2$). [1]

[1] Osborne, Michael, and Michael Alan Osborne. _Bayesian Gaussian processes for sequential prediction, optimisation and quadrature_. Diss. Oxford University, UK, 2010.

### Soundness
3

### Presentation
3

### Contribution
1
