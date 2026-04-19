# DAS$^2$C: A Distributed Adaptive Minimax Method with Near-Optimal Convergence

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 3, 6

## Abstract
Applying adaptive methods directly to distributed minimax problems can result in non-convergence due to inconsistency in locally computed adaptive stepsizes. To address this challenge, we propose DAS$^2$C, a $\underline{\text{D}}$istributed $\underline{\text{A}}$daptive method with time-scale \$\underline{\text{S}}$eparated $\underline{\text{S}}$tepsize $\underline{\text{C}}$ontrol for minimax optimization. The key strategy is to employ an adaptive stepsize control protocol involving the transmission of two extra (scalar) variables. This protocol ensures the consistency among stepsizes of nodes, eliminating the steady-state errors due to the lack of coordination of stepsizes among nodes that commonly exists in vanilla distributed adaptive methods, and thus guarantees exact convergence. For non-convex-strongly-concave distributed minimax problems, we characterize the specific transient times that ensure time-scale separation and quasi-independence of networks, leading to a near-optimal convergence rate of $\tilde{\mathcal{O}} \left( \epsilon ^{-\left( 4+\delta \right)} \right)$ for any small $\delta > 0$, matching that of the centralized counterpart. To the best of our knowledge, DAS$^2$C is the $\textit{first}$ distributed adaptive method guaranteeing exact convergence without requiring to know any problem-dependent parameters for nonconvex minimax problems.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper delves into distributed minimax optimization problems, specifically addressing a min-max problem with costs allocated across network-connected nodes. The authors introduce an adaptive stepsize distributed method, notably agnostic to the problem's inherent parameters.

### Strengths
The paper introduces a novel adaptive step size approach tailored for distributed optimization, aiming to ensure consistency among all nodes. Leveraging this strategy, the authors put forth a distributed method, demonstrating its convergence at near-optimal rates.

### Weaknesses
The paper's assumption that every stochastic gradient remains bounded presents a restrictive condition.
The overall presentation and structure of the paper need refinement. The paper initiates with equations without offering adequate motivation or a streamlined introduction. This lack of organization is particularly concerning given the dense notation utilized in the work. Additionally, there's a notable absence of discussions surrounding the principal results and the definitions of each parameter. Delving into these results, understanding their implications, and drawing clear comparisons with other works are essential steps that should not be overlooked.

### Questions
Regarding the primary result, does it imply that the method converges for all step sizes $\gamma_{x,y}$, provided the iteration count is sufficiently large?

What drove the development of the proposed method? How did you derive those specific updates for the step size?

Is it possible to do away with the bounded gradient assumption? Generally, to demonstrate the convergence of analogous decentralized methods, only the conditions of bounded variance and bounded gradient disagreement are required.

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studied the adaptive minimax method in distributed problems. They proposed a distributed adaptive method DAS2C with time-scale separated stepsize control for minimax optimization. By leveraging the transmission of two extra (scalar) variables, non-convergence issue is solved. For nonconvex-strongly-concave distributed minimax problems, it gets a near-optimal convergence rate of $\tilde{O}  (\epsilon^{4+ \delta})$.

### Strengths
1. Minimax is an important optimization problem in machine learning and the study of minimax in a distributed setting is necessary. 

2. The paper is organized well. it presents a counterexample to show how to design the algorithm.   

3. The strategy is simple to use and extra transmission variables are scalar.

### Weaknesses
1. The motivation behind this paper is not clear. What is the advantage of the adaptive method and why do we need these types of methods in the distributed setting?   Although few papers study the application of adaptive methods in distributed minimax problems, are adaptive methods necessary in distributed or federated learning compared with non-adaptive algorithms? 

2. Some recent related distributed minimax works are missing.

[1] A faster decentralized algorithm for nonconvex minimax problems, NeurIPS 2021

[2] Taming Communication and Sample Complexities in Decentralized Policy Evaluation for Cooperative Multi-Agent Reinforcement Learning, NeurIPS 2021

[3] FedNest: Federated bilevel, minimax, and compositional optimization. ICML 2022

[4] Decentralized Riemannian Algorithm for Nonconvex Minimax Problems. AAAI 2023

[5] Solving a Class of Non-Convex Minimax Optimization in Federated Learning. NeurIPS 2023.


3. For the convergence results, if this paper focuses on the nonconvex-strongly-concave, results should include the depends on $\kappa$. 

4. The baselines in experiments seem to only solve the issues about the design of the adaptive method in distributed learning. It does not present why we need adaptive  algorithms in (minimax) distributed problems.

### Questions
1. "$DAS^2C$ is the first distributed adaptive method guaranteeing exact convergence without requiring to know any problem-dependent parameters for nonconvex minimax problems". Could you explain what are the  "problem-dependent parameters "?

2. The convergence is $\tilde{O}  (\epsilon^{4+ \delta})$. What is the result of its centralized counterpart? It seems that related works does not include their term and this result is not as tight as others.

3. In the eq. (4), first equality should be correct. But for the second equality, if you add a projection operator, is the equality still valid?

4. Why $\min _x \max _y 1 / n \sum_{j=1}^n f_i\left(x ; \xi_i+y\right)-\eta\|y\|^2,$ in Robust training of neural network tasks and Generative Adversarial Networks is NC-SC question?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper investigates the decentralized minimax optimization problem. It developed an adaptive algorithm. However, it missed some important references and introduced strong assumptions and have errors in convergence analysis.

### Strengths
1. The problem studied in interesting. 

2. The counterexample is good.

### Weaknesses
1. This paper missed some state-of-the-art literature. 


[1] A Faster Decentralized Algorithm for Nonconvex Minimax Problems
[2]  Taming Communication and Sample Complexities in Decentralized Policy Evaluation for Cooperative Multi-Agent Reinforcement Learning
[3] Decentralized stochastic gradient descent ascent for finite-sum minimax problems
[4] Jointly Improving the Sample and Communication Complexities in Decentralized Stochastic Minimax Optimization

2. This paper introduces strong assumptions so that the proof is simplified too much. In particular, it assumes the gradient is upper-bounded in Assumption 4, which is not used in the original TiAda paper. In addition, assuming that the function is strongly concave and has a bounded gradient is not common because the simple quadratic function does not satisfy this assumption. 

3. There are some errors in the proof. $\mathcal{P}$ is not a linear operator so eq (36) is not correct.

4. This paper didn't compare with the aforementioned SOTA algorithms.

### Questions
1. This paper missed some state-of-the-art literature. 


[1] A Faster Decentralized Algorithm for Nonconvex Minimax Problems
[2]  Taming Communication and Sample Complexities in Decentralized Policy Evaluation for Cooperative Multi-Agent Reinforcement Learning
[3] Decentralized stochastic gradient descent ascent for finite-sum minimax problems
[4] Jointly Improving the Sample and Communication Complexities in Decentralized Stochastic Minimax Optimization

2. This paper introduces strong assumptions so that the proof is simplified too much. In particular, it assumes the gradient is upper-bounded in Assumption 4, which is not used in the original TiAda paper. In addition, assuming that the function is strongly concave and has a bounded gradient is not common because the simple quadratic function does not satisfy this assumption. 

3. There are some errors in the proof. $\mathcal{P}$ is not a linear operator so eq (36) is not correct.

4. This paper didn't compare with the aforementioned SOTA algorithms.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studied the decentralized distributed nonconvex-strongly-concave minimax problems, and proposed an efficient adaptive decentralized algorithm to solve these problems. Theoretically, it proved that the proposed algorithm obtain a near-optimal convergence rate. Experimentally, it provided some experimental results to demonstrate the efficiency of the proposed algorithms.

### Strengths
This paper studied the decentralized distributed nonconvex-strongly-concave minimax problems, and proposed an efficient adaptive decentralized algorithm to solve these problems. Theoretically, it proved that the proposed algorithm obtain a near-optimal convergence rate. Experimentally, it provided some experimental results to demonstrate the efficiency of the proposed algorithms.

### Weaknesses
Although the proposed DAS2C algorithm is the first decentralized distributed adaptive method for nonconvex minimax problem, it basically extends the existing adaptive method [1] to decentralized distributed settings. Meanwhile, the DAS2C algorithm can address the issue of inconsistent stepsizes across different nodes by communicating the adaptive step-sizes, which basically follows the same trick in the adaptive federated learning.  


[1] Li, X., YANG, J., and He, N. (2023). Tiada: A time-scale adaptive algorithm for nonconvex minimax optimization. In The Eleventh International Conference on Learning Representations.

### Questions
1)	In DAS2C  algorithm, why use the exponential factors satisfying $\beta < \alpha$ ?

2)	The DAS2C algorithm needs some stricter assumptions (e.g., $f_i$ is second-order Lipschitz continuous for $y$) than the existing decentralized minimax optimization methods. 

3)	In the experiments, the authors should add some existing  decentralized  minimax optimization algorithms such as the DPOSG of [2] as the comparison methods.

[2] Liu, M., Zhang, W., Mroueh, Y., Cui, X., Ross, J., Yang, T., and Das, P. (2020). A decentralized parallel algorithm for training generative adversarial nets. Advances in Neural Information Processing Systems, 33:11056–11070.

4) Some related references are missing. E.g.,

[a] A Simple and Efficient Stochastic Algorithm for Decentralized Nonconvex-Strongly-Concave Minimax Optimization

[b] Jointly Improving the Sample and Communication Complexities in Decentralized Stochastic Minimax Optimization

[c] A faster decentralized algorithm for nonconvex minimax problems

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
