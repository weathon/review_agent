# Sample-efficient Learning of Infinite-horizon Average-reward MDPs with General Function Approximation

- Decision: Accept (poster)
- Scores: 6, 6, 8, 5

## Abstract
We study infinite-horizon average-reward Markov decision processes (AMDPs) in the context of general function approximation. Specifically, we propose a novel algorithmic framework named Local-fitted Optimization with OPtimism (LOOP), which incorporates both model-based and value-based incarnations. In particular, LOOP features a novel construction of confidence sets and a low-switching policy updating scheme, which are tailored to the average-reward and function approximation setting. Moreover, for AMDPs, we propose a novel complexity measure --- average-reward generalized eluder coefficient (AGEC) --- which captures the challenge of exploration in AMDPs with general function approximation. Such a complexity measure encompasses almost all previously known tractable AMDP models, such as linear AMDPs and linear mixture AMDPs, and also includes newly identified cases such as kernel AMDPs and AMDPs with Bellman eluder dimensions. Using AGEC, we prove that LOOP achieves a sublinear  $\tilde{\mathcal{O}}(\mathrm{poly}(d, \mathrm{sp}(V^*)) \sqrt{T\beta} )$ regret, where $d$ and $\beta$ correspond to  AGEC and log-covering number of the hypothesis class respectively,  $\mathrm{sp}(V^*)$ is the span of the optimal state bias function, $T$ denotes the number of steps, and $\tilde{\mathcal{O}} (\cdot) $ omits logarithmic factors. When specialized to concrete AMDP models, our regret bounds are comparable to those established by the existing algorithms designed specifically for these special cases.  To the best of our knowledge, this paper presents the first comprehensive theoretical framework capable of handling nearly all AMDPs.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies sample efficiency in infinite-horizon averaged MDP setting with general function approximation.
The authors propose average-reward generalized eluder coefficient to characterize the exploration difficulty in learning, and contribute an algorithm called Fixed-Point Local Optimization and establish sublinear regret with polynomial dependence on AGEC coefficient and span of optimal state bias, etc.

### Strengths
The paper writing is clear. The notations, definitions are clearly stated. Comparison with previous works are also clearly summarized in Table 1.

### Weaknesses
1. It seems to me the main contribution of this paper is to transfer and generalize some existing techniques and results (especially in finite horizon setting) to infinite horizon averaged return setting. The definition of AGEC Complexity Measure, FLOP algorithm and the techniques for analyzing lazy update rules seem share much similarity with previous literature like [1], [2]. I didn't find much novelty in algorithm design or technique analysis.

2. There is no conclusion section. I would suggest the authors at least have a few works to summarize the paper and discuss the future works. 


[1] Zhong et. al., Gec: A unified framework for interactive decision making in mdp, pomdp, and beyond

[2] Xiong et. al., A general framework for sequential decision- making under adaptivity constraints


## Post Rebuttal 

Thanks for the detailed reply. I think my main concerns are addressed, and I'm willing to increase my score.

### Questions
* What are the novelties in technical or algorithmic level in this paper? What are the new challenges for exploration in infinite horizon averaged reward setting?

* Is the "lazy policy update" really necessary? Although the authors explain the motivation for low policy switching is because of the additional cost in regret analysis. I'm curious whether it can be avoidable or it reveals some fundamental difficulty.

* In Theorem 3, the definition of $\beta$, is $sp(v^*)$ inside or outside of the log?

* Why the algorithm is called "Fixed-Point ..."? I'm not very understand why Eq. 4.1 is a fixed-point optimization problem.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper considers reinforcement learning for infinite-horizon average-reward MDPs under function approximation, and (i) generalizes the concept of eluder dimension to average-reward MDPs as a complexity measure, and (ii) proposes a new algorithm named FLOP to solve average-reward MDPs with low complexity in the sense of the generalized eluder dimension defined in the paper.

### Strengths
The problem addresses the challenging problem of infinite-horizon, average-reward MDPs in the function approximation setting. The idea of extending eluder dimension to this class of MDPs is a good idea. Also, the proposed algorithm that achieves sublinear regret seems to be a promising extension of the fitted Q-iteration algorithm, which takes lazy policy change into account.

### Weaknesses
The definition of AGEC (Definition 3), which is the central object and contribution in this paper, lacks clarity:
- Big-O notation is used in the definition, where the defined quantities $d_G$ and $\kappa_G$ (which also appear in $\mathcal{O}$) are smallest numbers that satisfy the inequalities that involve $\mathcal{O}$. This does not make much sense as a mathematical definition, with $\mathcal{O}$ being asymptotic.
- The set of discrepancy functions $\{l_f\}_f$ abruptly appears in Definition 3 without any proper definition. In later sections, we observe that it is an important quantity.
I would suggest a clear, mathematical definition of the complexity measure that constitutes one of the major contributions of this paper.

### Questions
In addition to the clarification of the definition of AGEC, I have the following questions:
- The function approximation error can be critical in RL, as its multiplying factor usually depends on the exploration performance in various forms. If we remove the realizability assumption, how does the additional term depend on the complexity measure defined in this paper?
- In Equation 2.1, what is $V^*(s,a)$? Should it be $Q^*$?
- In the abstract and in multiple places in the paper, $sp(v^*)$ appears with $v^*$. Should it be $sp(V^*)$? In the paper, it is assumed that $sp(V^*)$ is known. Is this knowledge necessary?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies infinite-horizon average-reward MDPs (AMDPs) with general function approximation. It extends the generalized eluder coefficient to average-reward generalized eluder coefficient (AGEC) under infinite-horizon MDPs. After showing that the low AGEC captures most existing structured MDPs, the paper develops an algorithm called FLOP to solve AMDPs with sublinear regret $O(\sqrt{T})$.

### Strengths
1. The paper provides a more general complexity measure AGEC that captures a large class of MDPs.
2. The design of confidence set is new, and the lazy update of policy is a good feature of algorithm design, which might be helpful for real implementations.

### Weaknesses
1. While the paper states that method covers most existing works, the detailed comparisons in terms of the regret performance are missed.

### Questions
Can authors provide a brief comparison between this work and existing works in terms of the regret results?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper explores the infinite-horizon average-reward Markov Decision Process (MDP) setting and introduces a comprehensive function approximation framework, accompanied by a corresponding complexity measure (AGEC) and an algorithm (FLOP). When compared with other work addressing AMDPs, the proposed framework covers the widest range of settings including both model-based and value-based, in addition, the theoretical analysis is based on the Bellman optimality assumption and the sample complexity is dependent on finite span, which is weaker compared with previous work (compared with communicating AMDP assumption and finite diameter dependence).

### Strengths
- This paper proposes a general framework for AMDP, encompassing both model-based and value-based. 

- The authors propose a new complexity measure that is larger than the Eluder dimension.

- The authors propose a new algorithm that matches the state-of-the art sample complexity results.

### Weaknesses
- The sample complexity is based on GEC and is not entirely novel. Specifically, Definition 3 (Bellman Dominance) in this paper is of the same form with Definition 3.4 of Zhong et al. (2022b), where $d$ is defined such that the sum of the Bellman error being less than the in-sample training error plus the burn-in cost.

- The authors introduce the discrepancy function in the definition of AGEC, and shows a simple example of discrepancy function being the Bellman error for the value-based case, and $(r_g+P_g V_{f'})(s_t, a_t)-r(s_t, a_t)+V_{f'}(s_{t+1})$ for the model-based case.  However, there seems a lack of discussion regarding alternative choices for the discrepancy function, such as the Hellinger distance-based discrepancy in Zhong et al. (2022b).

- The algorithm is based on upper confidence bound, which the confidence region chosen based on the discrepancy function. This approach is closely related to Algorithm 1 in Chen et al. (2022b) and Algorithm 1 in Jin et al. (2021).

- The proposed algorithm is usually impractical to implement, since it involves solving a global constrained optimization.

### Questions
- In this paper, the authors directly assume the transferability of the discrepancy function, a concept closely related to Lemma 41 in Jin et al. (2021). Could the authors elaborate on the primary technical challenges they encountered while deriving their theoretical results, when adapt the GEC in Zhong et al. (2022b) to the infinite-horizon average-reward setting, with a constrained algorithm, which seems to be a generalization of the framework established by Jin et al. (2021)?

- Can the authors elaborate the optimality of the regret in Theorem 3, when restricted to each specific instances? i.e. linear mixture AMDP, linear AMDP, etc.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
