# Robustness in the Face of Partial Identifiability in Reward Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 2, 8, 8, 8

## Abstract
In Reward Learning (ReL), we are given feedback on an unknown target reward, and the goal is to use this information to recover it in order to carry out some downstream application, e.g., planning. When the feedback is not informative enough, the target reward is only partially identifiable, i.e., there exists a set of rewards, called the feasible set, that are equally plausible candidates for the target reward. In these cases, the ReL algorithm might recover a reward function different from the target reward, possibly leading to a failure in the application. In this paper, we introduce a general ReL framework that permits to quantify the drop in "performance" suffered in the considered application because of identifiability issues. Building on this, we propose a robust approach to address the identifiability problem in a principled way, by maximizing the "performance" with respect to the worst-case reward in the feasible set. We then develop Rob-ReL, a ReL algorithm that applies this robust approach to the subset of ReL problems aimed at assessing a preference between two policies, and we provide theoretical guarantees on sample and iteration complexity for Rob-ReL. We conclude with some numerical simulations to illustrate the setting and empirically characterize Rob-ReL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a unified framework for robust reward learning in partially identifiable settings. Given some form of feedback (e.g., preferences, demonstrations), the method constructs a feasible set of reward functions consistent with observed data and then optimizes the worst case over this set (for a downstream decision such as a policy). The formulation aims to generalize across multiple feedback modalities and tasks.

### Strengths
+ The framework unifies various feedback types (e.g., pairwise comparisons, demonstrations) and downstream objectives under one minmax formalism.
+ The worst-case formulation in (7) is intuitive and could be more realistic than assuming a point estimate of the reward.

### Weaknesses
- The general minmax structure resembles many existing approaches in IRL, robust MDPs, and RLHF (e.g., distributionally robust RL, reward uncertainty sets). The paper does not sufficiently delineate what is new in (2) beyond framing and unification.

- The paper needs sharper theoretical distinctions (e.g., identifiability analysis, sample complexity bounds, or formal generalization of previous frameworks). Without this the contribution risks being incremental.

- The approach requires estimating or approximating policy-induced dynamics and state-action visitation measures, which can be computationally prohibitive and may not be scalable to realistic domains.

- The proposed method replaces one set of unknowns (the reward) with multiple estimated quantities (transition probabilities, occupancy measures, feasible reward bounds), which may not improve robustness in practice.

- The experiments are limited to small toy domains (even with the ones in the appendix). The paper needs results on more standard RL or preference-learning benchmarks (e.g., MuJoCo or D4RL tasks) to demonstrate practical value. The lack of scalability experiments weakens the claimed generality.

- It is unclear whether ablation or sensitivity studies were conducted to assess dependence on feasible set or uncertainty size.

- The paper does not discuss how many trajectories or feedback samples are required to characterize the feasible reward set. It remains also unclear how the method handles poor coverage or unobserved states which is an important concern when identifiability is partial.

- The framework implicitly assumes access to accurate simulators or transition estimates, which limits realism in practical RLHF settings.

### Questions
1) How does (2) differ theoretically or algorithmically from robust IRL or minimax RLHF formulations?
2) How sensitive is performance to the feasible reward set?
3) Can the approach handle partially observed dynamics or limited coverage?
4) What is the empirical complexity compared to baselines? how does it scale with state or trajectory length?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces a framework for reward learning from preferences and demonstrations. The authors try to quantify the effect of  identifiability issues of the reward, and propose a minimax approach to optimize the worst-case scenario over the feasible reward set. They provide Rob-ReL, an algorithm for policy preference assessment, providing theoretical guarantees on sample and computation complexity.

### Strengths
- Clear formalization of the problem. 
- The paper is well written and organized in a clear progression
- Clear sample-complexity results for their algorithm with illustrative numerical results

### Weaknesses
- A big limitation of this work is the limited number of applications. The authors provide a general framework, but then they limit themselves to a single scenario. This leaves the overall framework not properly tested, and severely limits the contribution of the paper.
- Another limitation, is that while the authors provide a general framework, the theoretical insights seem to be limited. For example, when solving a minimax game the equilibrium may not be a pure equilibrium; however, the authors do not seem to discuss this issue. Depending on the properties of ${\cal X}_g$, the loss, etc...we may have different situations. In the application they propose the solution they get seems a pure one (because it's estimating a scalar), but i would expect mixed solutions to appear depending on the problem (e.g., when ${\cal X}_g=\Pi$). While the authors do an effort at quantifying the error (eq 3), it is not very clear what are the properties of this minimax problem (which depends on the set of rewards, loss, etc.). 

- Theorem 5.3: the dependency on $\xi$ seems quite large
- It is not clear why the chosen application (estimating policy-value differences) is an interesting one

- While the paper is well written, there is still lot of notation and it is hard to follow the proofs.

- The method is only tested on a simple problem, while larger problems are untested.

### Questions
Please see the weaknesses above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the issue of partial identifiability in reward learning (ReL), a consequence of when the provided feedback is insufficient and does not allow for the identification of the target reward $r^\*$. Instead of traditional approaches, the authors propose a new framework that incorporates robust optimization to minimize the loss tied to the worst possible value for $r^*$ within some feasible set.

### Strengths
- The formalization, explanation, and clarity of the ReL problem as a pair $(\mathcal{F},g)$ is very well written, general, relevant, and well positioned against related work. A key piece is how the ReL problem is reformulated to finding the optimal object $x\in\mathcal{X}_g$ for some application $g$, when given the uncertainty set of rewards $\mathcal{R_F}$.
- The metric used to calculate the loss, $\mathcal{I}_{\mathcal{F},g}$, or how "uninformative" $\mathcal{F}$ is for application $g$, is very useful as it effectively allows for the principled comparison of various feedback sets and/or applications.
- The algorithm instantiation and associated analysis is novel. A key piece that facilitates this is in Proposition 5.1 where the minimax problem is simplified into two convex problem.

### Weaknesses
- While the authors present a powerful, general framework, they do not address tractability concerns. The limitations section significantly downplays this fact and speaks broadly on it.
- The experiments, though useful in demonstrating/verifying the author's theoretical, do not have any baselines to compare against. For example, in section 4, existing approaches are discussed. It would be interesting to see how the proposed algorithm compares against these non-robust baselines.

### Questions
- Can you expand on what you mean on lines 240-241 and/or provide a reference speaking towards a practical example?
- You make the assumption "that the feasible set $\mathcal{R}_\mathcal{F}$ contains a strictly feasible reward $\bar{r}.$ How realistic is this in practice?
- You make the assumption "that the feasible set $\mathcal{R}_\mathcal{F}$ contains a strictly feasible reward $\bar{r}.$ How realistic is this in practice?
- Consider changing the colored text in equation 7 so that it matches the blue in Lemma E.2.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper addresses the fundamental problem of partial identifiability in Reward Learning (ReL), where limited feedback makes it impossible to uniquely recover the target reward function. The authors propose a quantitative framework that allows measuring performance degradation due to identifiability issues, moving beyond prior qualitative approaches. They introduce a robust minimax approach that optimizes for the worst-case reward in the feasible set and develop Rob-ReL, an algorithm for policy preference assessment problems with provable sample and iteration complexity guarantees. The work combines demonstrations, trajectory comparisons, and newly introduced policy comparison feedback in a mixed offline-online setting. Theoretical analysis shows polynomial complexity in relevant problem parameters, and numerical experiments illustrate the approach on a low-dimensional navigation task.

### Strengths
**Rigorous Theoretical Analysis.** Theorem 5.3 provides polynomial sample and iteration complexity bounds under reasonable assumptions (Slater's condition). The proof technique combining visitation distribution estimation errors with primal-dual subgradient convergence is sound. The use of RF-Express for minimax-optimal reward-free exploration is appropriate.

**Clear Presentation and Organization.** The paper is well-structured with motivation, framework, approach, algorithm, and theory presented logically. The illustrative example in Section 6 effectively conveys the main ideas visually.

### Weaknesses
**Limited treatment of function approximation.** The tabular setting with explicit state-action representation limits applicability to high-dimensional problems. While the authors mention neural network parameterization in related work, Rob-ReL does not incorporate function approximation

### Questions
How does the method scale to continuous state-action spaces with function approximation? 
Is there a path toward extending Rob-ReL to deep RL settings, perhaps using neural network reward parameterization?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper develops a new framework, robust reward learning, that tackles the known problem of partial identifiability in reward learning. The framework incorporates several important forms of data for reward learning (including trajectories from an optimal policy, trajectory comparisons, and policy comparisons) and considers several potential downstream applications (including imitation, finding an optimal policy, and learning to compare preferences). The paper proposes to use worst-case performance in a downstream application, across all rewards compatible with the training data, as a metric for success.

The paper provides an algorithm for solving robust reward learning problems and provides theoretical guarantees on the sample and computational complexities. They also run some experiments in a small RL setting to demonstrate their algorithm works.

### Strengths
I think the paper has a number of strengths:
1. The paper addresses an *important problem*:  improving the safety of reward learning, which is a salient problem for modern AI deployments.
2. The *novel formulation* of robustness for reward learning adds upon prior work addressing this problem. The uninformativeness measure is an interesting way to quantify the difficulty of doing reward learning.
3. The formulation is *very general* and explicitly considers three important kinds of reward learning feedback. (Prior work, e.g. Skalse et al. (2023) only considered two kinds).
4. A *(somewhat) tractable algorithm* for solving the novel problem is introduced (Rob-Rel), and theoretical guarantees on sample complexity and time complexity (at least in terms of the number of iterations) are given. Further, by using duality, the paper avoided any dependence of time complexity on the size of the feasible reward set.
5. The paper *thoroughly, clearly, and fairly considers existing work*. In particular, tables 1 and 2 give clear relationships to broad prior work, and appendix *A.2* gives a thorough comparison to Skalse et al. (2023) which helped me to understand its contribution.

### Weaknesses
### Scalability
The proposed algorithm is polynomial in the size of the state space. This is a limited weakness, as it is common to some of the prior literature.

However, some reward learning methods have been shown to work in realistic and large or continuous state spaces. For example, Christiano et al.'s (2017) reward learning has been applied to text settings (large, discrete state spaces) and physics simulations (continuous state spaces). Laidlaw et al. (2025) applied CIRL to a large game.

I think the impact of this work would be significantly improved if evidence was given to suggest the framework and algorithm were scalable. If the algorithm is not scalable, it would be helpful to describe these limitations.

### Practicality of worst-case return
The proposed algorithm maximizes worst-case robustness relative to a feasible set. As the authors themselves suggest on line 241, there may be some reward learning problems where it is infeasible to get an acceptable worst-case reward. (In these cases, I think the uninformativeness could be a helpful metric for measuring this infeasibility, as noted in the strengths above). I expect that, in many real-world applications with large feasible reward spaces, many reward learning problems will be infeasible.
While it is not reasonable for the paper to solve this problem, I do believe it is a notable limitation, relative to e.g. a Bayesian approach that only tries to minimize expected loss over a posterior reward distribution.  Evidence that robust reward learning is feasible in real-world environments might improve the paper.


### Clarity
I found the paper to be quite difficult to parse in a number of places, although I think the paper is relatively clear given its technical density. Three potential points for improvement might be:
1. Reduce the introduction of abbreviations: "ReL", "IL", "PBIRL", and other abbreviations could be far easier to parse if set out plainly.
2. Reduce the introduction of notation, or where possible, explain in plain English what terms mean. Section 5, in particular, introduces a whole range of new notation that is hard to keep track of and could be better supported with natural language explanations.
3. (Minor) Throughout the paper, citations are given without bracketing, where bracketing would be much clearer. For example, lines 214, 121 and 105.


### References
* Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. arXiv preprint arXiv:1706.03741.
* Laidlaw, C., Bronstein, E., Guo, T., Feng, D., Berglund, L., Svegliato, J., Russell, S., & Dragan, A. (2025). AssistanceZero: Scalably solving assistance games. arXiv preprint arXiv:2504.07091.

### Questions
Can the authors provide any insight about whether worst-case reward learning can scale as well as alternative approaches (such as cooperative inverse RL, or reinforcement learning from human feedback)?

### Soundness
3

### Presentation
3

### Contribution
4
