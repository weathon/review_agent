# PILOT: An $\mathcal{O}(1/K)$-Convergent Approach for Policy Evaluation with Nonlinear Function Approximation

- Decision: Accept (spotlight)
- Scores: 8, 6, 8, 6

## Abstract
Learning an accurate value function for a given policy is a critical step in solving reinforcement learning (RL) problems. So far, however, the convergence speed and sample complexity performances of most existing policy evaluation algorithms remain unsatisfactory, particularly with non-linear function approximation. This challenge motivates us to develop a new path-integrated primal-dual stochastic gradient (PILOT) method, that is able to achieve a fast convergence speed for RL policy evaluation with nonlinear function approximation. To further alleviate the periodic full gradient evaluation requirement, we further propose an enhanced method with an adaptive-batch adjustment called PILOT$^+$. The main advantages of our methods include: i) PILOT allows the use of {\em{constant}} step sizes and achieves the $\mathcal{O}(1/K)$ convergence rate to first-order stationary points of non-convex policy evaluation problems; ii) PILOT is a generic {\em{single}}-timescale algorithm that is also applicable for solving a large class of non-convex strongly-concave minimax optimization problems; iii) By adaptively adjusting the batch size via historical stochastic gradient information, PILOT$^+$ is more sample-efficient empirically without loss of theoretical convergence rate. Our extensive numerical experiments verify our theoretical findings and showcase the high efficiency of the proposed PILOT and PILOT$^+$ algorithms compared with the state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a new algorithm (PILOT) to estimate value function with non-linear function approximation. Under appropriate choices of step sizes, the algorithm achieves O(1/K) error after K iterations.

### Strengths
* The paper is well-written and easy to follow. The literature review is quite complete.

* The results are interesting and the author adequately motivate the problem.

* The numerical experiments are convincing (although they could be polished, see below).

### Weaknesses
* No significant weakness. See my questions below.

### Questions
* What is K in the paragraph before the questions “ can we develop … ” on page 2? Suppose this the K defined after the question. In that case, the phrasing “ more than O(K) number of iterations to achieve the convergence ” seem to suggests that the algorithm terminates with the exact value function after O(K) iterations, which is probably not the case. Please clarify.

* Point (ii), paragraph (2) of the literature reivew: “our algorithms only require the stepsizes to be sufficiently small, which is easier to tune in practice” —this is not very clear, what does sufficiently small  ? In the previous paragraph a method with O(1/M) step size is criticized so the authors need to be more accurate here.

* Paragraph 1, Section 3: The notation $\pi: \mathcal{S} \rightarrow \mathcal{A}$ is only for deterministic policies. Please introduce randomized policies properly.

* Algorithm 1 refers to Eq. (4) and Eq. (5) who appear a page later.

* Why do the authors use arrows+name for Figures 1-4 and not just a legend?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces two algorithms, PILOT and PILOT+, designed for Policy Evaluation (PE) with non-linear function approximation. Theoretical analysis shows that PILOT, a single-timescale algorithm using VR techniques, achieves an $O(1/K)$ convergence rate with constant step sizes. PILOT+ enhances sample efficiency by adapting batch sizes based on historical stochastic gradient information. Experimental results validate the theoretical findings regarding convergence and sample complexity.

### Strengths
1. The paper is fairly well organized and the results provided are impressive.

2. The related work and comparison with other algorithms and techniques are comprehensive.

3. The theoretical guarantee is solid and the paper also demonstrates the effectiveness of the algorithm in practice.

### Weaknesses
1. Since the paper claims the algorithms they propose achieve the first $O(1/K)$ convergence rate ($K$ is the number of
iterations) with constant step-sizes for PE with nonlinear function approximation, it is better to give some brief statistical intuition of achieving this result.  

2. The paper proposes a new metric for convergence performance, and gives the abundant explanation for using this concept, but it is hard to be applied to other performance metric, for example, $|\widehat{V}^{\pi}(s) - V^{\pi}(s)|$, where $\widehat{V}^{\pi}$ is the estimate you get from a policy evaluation problem.

### Questions
1. From Corollary 1, the sample complexity of PILOT is $O(\sqrt{M}\kappa^3\epsilon^{-1} +M)$, where $M$ is the number of state-action pairs sampled by the evaluated policy, and $\epsilon$ measures the error under the new metric of convergence performance in this paper. Intuitively, to get a accurate estimate of the $V^{\pi}$, we need a sufficient number of samples (a big $M$). However, it seems that this $\epsilon$ has nothing to do with $M$. Additionally, I have noticed that the Assumption 1 needs $M$ to be sufficiently large. Can authors give the detailed justification of this explanation? I would like to see how $M$ will affect the accuracy of the policy evaluation.


2. For the sample complexity of the algorithm, the first term $O(\sqrt{M}\kappa^3\epsilon^{-1})$ seems redundant, because the total number of samples is fixed, i.e., $M$, and in this paper the first term of sample complexity comes from sampling from this dataset, and this operation does not increase the sample complexity (it will not interact with the environment). Therefore, I think PILOT+ accually saves the computational cost, not sample complexity of PILOT. Can authors explain this concept more carefully?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors develop a single-timescale version of the non-linear policy evaluation algorithm (PILOT). The algorithm allows for instance-independent stepsize choice and relies on the primal-dual optimization with additional variance reduction for gradient computations. The authors also provide a modification of their approach with adaptive batch size selection in order to avoid the preiodic full gradient computations.

### Strengths
The introduced algorithm, PILOT, provides an explicit convergence rate under the choice of small and constant step sizes. Moreover, the obtained convergence rate matches the previous minimax bound. Convergence rates are studied under the assumptions, which are classical in the optimization literature. The authors present a comprehensive framework for their theoretical analysis, and the exposition of the paper is accessible.

### Weaknesses
The only weakness is the experimental section, which contains rather simple scenarios. At the same time, for the paper which is primarily theoretical, the proposed illustration is sufficient.

### Questions
a)If time permits, I would suggest the authors to complement the experimental section. 

b) The authors could also complement the bibliography on linear policy evaluation methods (TD(0) with modifications) by the following papers:
1. Li, Tianjiao, Guanghui Lan, and Ashwin Pananjady. "Accelerated and instance-optimal policy evaluation with linear function approximation." arXiv preprint arXiv:2112.13109 (2021) - https://arxiv.org/abs/2112.13109
2. Patil, Gandharv, et al. "Finite time analysis of temporal difference learning with linear function approximation: Tail averaging and regularisation." International Conference on Artificial Intelligence and Statistics. PMLR, 2023. - https://arxiv.org/abs/2210.05918
The first one concerns instance-optimal guarantees for Polyak-Ruppert averaged iterates of TD(0). The authors also use SPIDER-type variance reduction. The second paper concerns TD(0) with realizable step size.

c) I am also a bit surprised by the fact that the mixing time of the original chain $(s_1,a_1,s_2,\ldots)$ does not pop up explicitly in the bounds. This would be a typical behavior for the optimization problems with dependent data. What is the explanation?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work investigates policy evaluation with nonlinear function approximation. The work proposes path-integrated primal-dual stochastic gradient (PILOT), and PILOT+. It shows that PILOT converges to stationary point with a convergence rate of $O(1/K). The theoretical results are based on the assumption of strong concavity and bounded variance. With adaptive batch size, PILOT+ achieves some empirical results that demonstrates some sample efficiency. The algorithms demonstrate some efficiency on simple simulation tasks like Mountain Car and Cartpole.

### Strengths
The work proposes a pair of algorithms, PILOT and PILOT+, which are good in theory and in practice, respectively. It is nice to have variants of the algorithm to excel in both perspectives. The results on stationary point convergence and a convergence rate of $O(1/K)$ seem relevant.

### Weaknesses
The work is based on very strong assumptions, which do not hold in RL tasks in general. It is of course valid to argue that some previous works are also based on such assumptions, but with the assumptions the work offers less relevance and less technical contribution to the community.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
