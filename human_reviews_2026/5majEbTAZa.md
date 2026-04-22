# Laplacian Kernelized Bandit

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
We study multi-user contextual bandits where users are related by a graph and their reward functions exhibit both non-linear behavior and graph homophily. We introduce a principled joint penalty for the collection of user reward functions $\\{f_u\\}$, combining a graph smoothness term based on RKHS distances with an individual roughness penalty. Our central contribution is proving that this penalty is equivalent to the squared norm within a single, unified _multi-user RKHS_. We explicitly derive its reproducing kernel, which elegantly fuses the graph Laplacian with the base arm kernel. This unification allows us to reframe the problem as learning a single "lifted" function, enabling the design of principled algorithms, LK-GP-UCB and LK-GP-TS, that leverage Gaussian Process posteriors over this new kernel for exploration. We provide high-probability regret bounds that scale with an _effective dimension_ of the multi-user kernel, replacing dependencies on user count or ambient dimension. Empirically, our methods outperform strong linear and non-graph-aware baselines in non-linear settings and remain competitive even when the true rewards are linear. Our work delivers a unified, theoretically grounded, and practical framework that bridges Laplacian regularization with kernelized bandits for structured exploration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies multi-user contextual bandits where users are connected by a known graph and have nonlinear reward functions. It builds a joint RKHS kernel combining (i) graph smoothness via the Laplacian and (ii) nonlinear structure over arms, and then runs GP-UCB / GP-TS on this kernel. It proves regret bounds that depend on an effective dimension of the graph+kernel, not just number of users, and shows strong performance in experiments.

### Strengths
- Gives a clean unified kernel that fuses graph structure and nonlinear rewards in one theory.
- Provides GP-UCB / GP-TS algorithms that share information across similar users automatically.
- Regret scales with an effective dimension (can be much smaller than #users).
- Shows consistent empirical gains, especially on nonlinear graph-smooth tasks.

### Weaknesses
- The paper omits key related work. The problem setting is very close to prior work on graph-structured / collaborative contextual bandits [1][2], but these are not adequately discussed in the introduction or positioned against the proposed approach.

- The baseline comparison is incomplete. Modern neural contextual bandits can learn richer representations than fixed kernels, e.g. neural GP / neural UCB style methods [3]. These should be discussed and, ideally, included as baselines.

- The UCB/TS exploration terms depend on problem-dependent quantities (like the RKHS norm bound and noise level). These are not observable in practice, so the theoretical guarantees rely on tuning parameters you don’t actually know.

- While the bound is written in terms of an “effective dimension” $\tilde{d}$, the asymptotic rate is the same $\sqrt{T}$ scaling as standard kernelized bandits. The paper does not prove matching lower bounds or show regimes where $\tilde{d}$ is provably small, so the improvement can be marginal.

### Questions
See weakness.

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
3

### Summary
**Summary**

This paper addresses the multi-user contextual bandit problem where users are related by a graph and reward functions are both non-linear and exhibit graph homophily (i.e., connected users have similar rewards). The authors introduce a joint penalty that combines a graph smoothness term (based on RKHS distances between users) with an individual roughness penalty for each user's function. The central contribution is a proof that this intuitive, additive penalty is mathematically equivalent to the squared norm within a single, unified multi-user RKHS. The authors explicitly derive the reproducing kernel for this new space, which fuses the graph Laplacian and the base arm kernel. This unification allows them to reframe the problem as learning a single "lifted" function, enabling the development of GP-based algorithms. They provide regret bounds that depend on the effective dimension of this new kernel, rather than on the number of users or ambient dimension, and show empirically that their methods outperform strong baselines in non-linear settings.




**Advantages**

* The theoretical unification of the graph smoothness penalty and the individual RKHS penalties into a single, valid multi-user RKHS is the strength of this paper, providing a foundation for the proposed algorithms.

* By deriving the explicit multi-user kernel, the paper bridges the gap between Laplacian regularization and kernelized bandits, allowing the direct application of powerful GP bandit machinery (UCB and TS) to the graph-based problem.

* The resulting regret bounds replace a direct dependency on the number of users ($n$) with a dependency on the "effective dimension" of the new kernel, which captures the spectral properties of both the graph and the arm kernel.




**Shortcoming and Questions**

* The practical implementation of the proposed algorithms requires inverting a $t \times t$ matrix at each step $t$, leading to a computational complexity that scales poorly with the number of rounds, a common issue for GP-based methods.Given that the paper suggests recursive updates as a solution, could the authors provide an experiment comparing the cumulative regret of the exact GP-UCB update versus a more scalable approximation (like the recursive formulas or a sparse GP approach) as the time horizon $T$ becomes very large?


* The derived kernel requires the inversion of the $n \times n$ regularized Laplacian $L_{\rho}$, which could be computationally prohibitive if the number of users ($n$) is extremely large (e.g., millions of users). Would it be possible to conduct an experiment analyzing the scalability of the method with respect to the number of users $n$, and perhaps explore whether an approximation of $L_{\rho}^{-1}$ (e.g., using graph sparsification or spectral methods) could maintain competitive regret?


* The theoretical regret bounds depend on the effective dimension $\tilde{d}$, which is data-dependent and defined based on the entire sequence of actions up to time $T$, making it an a-posteriori quantity that is not known in advance. Could the authors provide an empirical analysis plotting the growth of the effective dimension $\tilde{d}$ relative to $T$ in the synthetic experiments, to give a clearer intuition for how this value behaves in practice?

### Strengths
Please see above.

### Weaknesses
Please see above.

### Questions
Please see above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies multi-user contextual bandits with a known user graph. The authors show how individual RKHS-based reward modeling and graph homophily (via the graph Laplacian) come together by deriving the reproducing kernel of a unified multi-user RKHS. Using this kernel, they design GP-UCB and Thompson Sampling algorithms and analyze their regret.

### Strengths
1. Theorem 2.1 provides a clean theoretical connection between Laplacian regularization across users and RKHS-based regularization on arm features. This connection is elegant and makes the overall framework principled and well-founded.

2. The proposed GP-based UCB algorithm are grounded on solid theoretical foundations. The corresponding regret bounds based on the effective dimension are sharp and well-justified, making the theoretical contribution of the paper clear and convincing.

### Weaknesses
1. The formulation mainly builds on previous work on linear Laplacian bandits by extending the idea from linear models to general RKHS functions. While the conceptual novelty is moderate, I think this extension is meaningful and technically non-trivial, as it requires re-deriving the RKHS characterization and associated regret analysis.


2. I am a bit confuse about the result for TS, as described below in detail.

### Questions
I am a bit surprised by the reported TS result. With a naive TS design, the paper claims a frequentist regret of order $d\sqrt{T}$. However, up to my knowledge, even in the simpler linear bandit setting, the naive version of TS typically achieves only $d^{3/2}\sqrt{T}$ frequentist regret, and stronger results usually require additional design modifications such as the feel-good TS algorithm [1]. Could the authors clarify what differs in their analysis that allows this improved bound? Or are there extra assumptions that effectively bypass the limitations known in previous TS literature?



[1] Zhang, Tong. "Feel-good thompson sampling for contextual bandits and reinforcement learning." SIAM Journal on Mathematics of Data Science 4.2 (2022): 834-857.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the problem of multi-user contextual bandits and proposed a framework based on a novel penalty that explicitly captures the relations between the user reward functions. The authors went on to provide a justification of this penalty by linking it to a multi-user RKHS, which allows for the development of two Gaussian process based algorithms. Theoretical result on  the regret bound has been derived for both algorithms, and experimental results validate their  effectiveness and superiority over existing baselines.

### Strengths
- The paper tackles an under-explored but interesting problem.
- The formulation of the penalty and its link to a RKHS is interesting and provides guidance on how relational information can be taken into account in learning.
- Theoretical results on regret bounds are a plus.
- The paper is generally well presented with clear motivation and technical descriptions.

### Weaknesses
**Connection to existing literature.** I believe it would be helpful for the authors to make a greater effort connecting their approach to similar attempts in the literature:
- First, the link between kernels and regularisations, as well as its extension to the graph case via the graph Laplacian, has been well-documented in the literature [1], and in my view this should be properly discussed in the derivation of the proposed approach.
- Second, the way the graph structure is incorporated into learning bears similarly with recent work in the space of multi-output and graph-based GPs, where this is done either implicitly [2] or explicitly [3,4]. I feel additional discussion and comparison against those would make the contribution of the paper conceptually clearer. 

**Technical framework.** The paper only focuses on a smoothness assumption, which is a reasonable starting point, however I was left wondering whether this can be generalised to the case where the user reward relations are more complex? I feel this should be possible to address by following for example the spectral learning approach in [4]. 

**Experiments.** Experimental results, while convincing, are relatively brief:
- Can the authors demonstrate the impact of different graph topology on learning performance, for example by using graphs generated from the stochastic block model? I think this can connect nicely to the discussion on effective dimension. 
- Can the authors provide ablation studies that demonstrate the utility of both the graph-based kernel and the classical kernel in the Kronecker product? 
- All experiments are synthetic at the moment, can the authors think about potential real-world experiments? This can also help a reader appreciate the practical utility of the proposed approach.

[1] Smola and Kondor, “Kernels and Regularization on Graphs,” COLT, 2023.

[2] Alvarez et al., “Kernels for vector-valued functions: A review,” Foundations and Trends in Machine Learning, 2012.

[3] Venkitaraman et al., “Gaussian processes over graphs,” IEEE ICASSP, 2020.

[4] Zhi et al., “Gaussian Processes on Graphs Via Spectral Kernel Learning,” IEEE TSIPN, 2023.

### Questions
See weaknesses above for the specific points I would like the authors to address or discuss.

### Soundness
3

### Presentation
3

### Contribution
3
