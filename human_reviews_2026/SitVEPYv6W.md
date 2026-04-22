# Asynchronous Policy Gradient Aggregation for Efficient Distributed Reinforcement Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
We study distributed reinforcement learning (RL) with policy gradient methods under asynchronous and parallel computations and communications. While non-distributed methods are well understood theoretically and have achieved remarkable empirical success, their distributed counterparts remain less explored, particularly in the presence of heterogeneous asynchronous computations and communication bottlenecks. We introduce two new algorithms, Rennala NIGT and Malenia NIGT, which implement asynchronous policy gradient aggregation and achieve state-of-the-art efficiency. In the homogeneous setting, Rennala NIGT provably improves the total computational and communication complexity while supporting the AllReduce operation. In the heterogeneous setting, Malenia NIGT simultaneously handles asynchronous computations and heterogeneous environments with strictly better theoretical guarantees. Our results are further corroborated by experiments, showing that our methods significantly outperform prior approaches.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies distributed reinforcement learning (RL) with asynchronous policy gradient methods. The authors propose two new algorithms: Rennala NIGT for homogeneous environments and Malenia NIGT for heterogeneous environments. Both methods extend normalized policy gradient techniques with asynchronous aggregation, achieving improved computational and communication complexity compared to prior work, e.g., AFedPG. The authors also establish a new lower bound and validate the methods on MuJoCo tasks, showing robustness to heterogeneity and communication delays.

### Strengths
**Originality.** Proposes new algorithms that generalize normalized gradient techniques to asynchronous distributed RL, with contributions in both homogeneous and heterogeneous settings.

**Quality.** Strong theoretical analysis, including new upper bounds, support for AllReduce, and a new lower bound. Experimental results align with the theory.

**Clarity.** The paper is well-structured, with clear algorithm pseudocode and comparisons.

**Significance.** Distributed RL is critical for scaling up to large systems. The improved communication and computation guarantees, especially with AllReduce support, are practically valuable.

### Weaknesses
**Optimality gap** There remains a gap between the presented upper bounds (κε⁻²) and the lower bound (κε⁻¹²/⁷), leaving open whether the methods are near-optimal.

### Questions
In Malenia NIGT, the mean-like dependency on agent speeds is less favorable than the harmonic dependency in Rennala NIGT. How does this affect performance in highly heterogeneous environments?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates distributed policy gradient (PG) methods in asynchronous and parallel reinforcement learning (RL) settings. The authors introduce two new algorithms for homogeneous environments and heterogeneous environments. Both algorithms are based on normalized policy gradient methods and asynchronous aggregation schemes inspired by recent work (e.g. Lan et al. 2025)

### Strengths
* I briefly checked the proofs and they seem correct. This means the paper improves over the communication complexity of existing methods. 

* They also show a lower bound for the homogenous setting in the appendix which seems to show there is room for improvement.

### Weaknesses
* There is a lack of novelty in the algorithmic components, while one may argue this particular combination is done for the first time.

* On the theoretical side, the proofs are highly reliant on/similar to existing results in federated RL and stochastic optimization. For instance, Lemma D.2 is a standard result (similar to SGDm with normalization and also a corresponding lemma from Lan et al. 2025). What is a new technical result proposed by the paper?

* I also have a concern about the heterogenous experiments in the Appendix H.3. To test Malenia NIGT in a scenario with environment heterogeneity (different $J_i$), the paper uses $n=2$ agents, where one agent receives state $s$ and the other receives state $-s$. To distinguish these, the paper states, "we concatenate the value 0 to $s_{t+1}$... In the case of the second worker, we redirect $(-s_{t+1}, 1)$". In my view, however, this is not a heterogeneous problem. The algorithm is not learning an average policy $\frac{1}{2}(J_1(\theta) + J_2(\theta))$. It is learning a single policy $J'(\theta)$ for a single environment whose state space is $\mathcal{S} \times \{0, 1\}$. The policy network can trivially learn to behave differently based on the appended bit.

### Questions
Please see the Weakness section.

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
4

### Summary
This paper introduces two asynchronous distributed policy-gradient algorithms for reinforcement learning:

* Rennala NIGT for the *homogeneous* setting (all agents share the same environment), and
* Malenia NIGT for the *heterogeneous* setting (each agent may have a different environment).

Both algorithms combine the NIGT policy gradient update (Fatkhullin et al., 2023) with asynchronous aggregation mechanisms adapted from recent federated SGD literature (Tyurin & Richtárik, 2023). The paper provides wall-clock convergence guarantees (in terms of computation and communication time) to reach an ε-stationary point under standard policy gradient assumptions. A lower bound for twice-smooth stochastic objectives is also presented, and empirical results on MuJoCo tasks illustrate the wall-clock speedups.

### Strengths
* The problem setting (efficient RL under asynchronous environments even with heterogeneity) is important and highly relevant.
* The paper is clearly written and the proofs are easy to follow.

### Weaknesses
**Novelty.** Conceptually, I struggle to see what is new here. The algorithm appears to essentially be a direct combination of two known components:

* Rennala/Malenia from federated SGD (Tyurin & Richtárik, 2023), and
* NIGT-based policy gradient from non-distributed RL (Fatkhullin et al., 2023).

The analysis likewise seems to be a straightforward combination. In particular, the core convergence argument (Section D, also F) closely follows the standard NIGT analysis, and relies on the fact that Rennala/Malenia provide unbiased policy gradient estimates with bounded variance (∝ 1/M). While it is normal for new work to build on prior components, here it is unclear where the main technical challenge lies (if any). 

**Lower bound and Missing Comparisons.** The paper’s main theoretical focus is ε-stationary convergence and also provide a lower bound for this rate (which is assuming access to only stochastic gradient estimates and no other information). The setting for the lower bound feels very artificial especially since there is a substantial body of RL papers that obtain improved ε-stationary rates by exploiting RL-specific structure (see [A],[B],[C] and also Hessian aided PG in (Fatkhullin et al 2023)). It is very odd that these are not even mentioned, let alone discussed or compared against. 


Finally, although this is less critical, the paper should be better contextualized relative to existing federated PG work. Apart from (Lan et al. 2025), the empirical comparisons focus on non-federated PG methods, which are not the most natural baselines here. There are federated PG methods with ε-stationary guarantees in the synchronous homogeneous case [C], federated PG based on Hessian aided PG in the same regime [D], and heterogeneous PG methods using softmax policies [E]. Including (or at least discussing) such baselines would better position this work within the federated PG literature.



---
[A] Xu, Gao, Gu. Sample Efficient Policy Gradient Methods with Recursive Variance Reduction. ICLR 2020.

[B] Xu, Gao, Gu. An Improved Convergence Analysis of Stochastic Variance–Reduced Policy Gradient. UAI 2020.

[C] Fan, Ma, Dai, Jing, Tan. Fault–Tolerant Federated Reinforcement Learning with Theoretical Guarantee. NeurIPS 2021.

[D] Ganesh, Chen, Thoppe, Aggarwal. Global Convergence Guarantees for Federated Policy Gradient Methods with Adversaries. TMLR 2024.

[E] Labbi, Mangold, Tiapkin, Moulines. On Global Convergence Rates for Federated Policy Gradient under Heterogeneous Environment. arXiv 2025.

### Questions
Please see weaknesses listed.

### Soundness
3

### Presentation
2

### Contribution
2
