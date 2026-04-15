# Resolving Partial Observability in Decision Processes via the Lambda Discrepancy

- Decision: Reject
- Scores: 3, 3, 3, 6

## Abstract
We consider the reinforcement learning problem under partial observability, where observations in the decision process lack the Markov property. To cope with partial observability, first we must detect it. We introduce the $\lambda$-discrepancy: a measure of the degree of non-Markovianity of system dynamics. The $\lambda$-discrepancy is the difference between TD($\lambda$) value functions for two different values of $\lambda$; for example, between 1-step temporal difference learning (TD($0$)), which makes an implicit Markov assumption, and Monte Carlo value estimation (TD($1$)), which does not. We prove that this observable and scalable value-based measure is a reliable signal of partial observability. We then use it as an optimization target for resolving partial observability by searching for memory functions---functions over the agent's history---to augment the agent's observations and reduce $\lambda$-discrepancy.  We empirically demonstrate that our approach produces memory-augmented observations that resolve partial observability and improve decision making.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
For partial observable reinforcement learning problems, the paper introduces $\lambda$-discrepancy as the difference between the TD($\lambda$) returns of two different $\lambda$ values. Several properties of $\lambda$-discrepancy are analyzed and they suggest the discrepancy might serve as a measure of partial observability for a POMDP. Based on the idea, the paper further proposes memory optimization methods by minimizing $\lambda$-discrepancy. Empirical results show performance improvement with memory functions learned by the proposed methods compared with the memoryless case.

### Strengths
- The idea of using the difference between TD($\lambda$) returns as a partial observability measure is very interesting, and it has the potential to help construct practical efficient algorithms for partial observable problems.

- The paper attempts to analyze when the $\lambda$-discrepancy would be zero, and the analysis suggests that $\lambda$-discrepancy is zero when the observation has some Markovian properties.

- The paper proposes an approach to learn a memory function for POMDP by minimizing $\lambda$-discrepancy, and it also provides two algorithms based on the approach with different levels of oracles available. Experiments show that learned memory function can indeed provide performance improvement over the memoryless case.

### Weaknesses
- The notations are very confusing, and many parts of the technical presentations are very difficult to follow. The products of tensors are presented as regular matrix multiplications without specifying which dimensions are used in the multiplications. Many definitions are not precise, like effective policy over latent state, and some augment representations of some tensors. In the proofs, it is very difficult to follow when the indices are meaningless $i, j, k, l, ...$ without specifying their domains.

- There are issues in many definitions in Section 3.1. As mentioned in the background section, the value functions of partial observable problems generally depend on the agent's entire history. But in Section 3.1, the Q-function is given as a function of the observation and action without any justification. This observation-action Q-function may be some useful approximation, but it is used throughout the paper as the correct value estimation without any discussion. Several variables of conditional probabilities like Pr$(s|\omega)$ are presented as given constants, but they are not provided by the observation model like Pr$(\omega|s) = O(\omega|s)$; the probability Pr$(s_t|\omega_t)$ depends on the policy and the time instant $t$ in general.

- The theoretical analysis in Appendix A seems incorrect. Below the first big equation in Appendix A, it is claimed that one can bootstrap by replacing the last line with a term with $Q(\omega_2, a_2)$ when $n=2$, but the claim means that 
$$\sum_{w_2, a_2} P(\omega_2 |s_2) P(a_2|\omega_2) Q(\omega_2, a_2) = \sum_{w_2, a_2} P(\omega_2 |s_2) P(a_2|\omega_2)
\sum_{s_3, r_2} P(s_3|s_2, a_2)P(r_2|s_2, a_2, s_3) r_2$$
This equation does not look correct. Even if what the authors mean is to include all the later terms in $...$, the bootstrap does not seem to hold given the time-homogenous definition of the Q-function unless with further justification. 

- There some other likely errors in Appendix A. There is a missing summation over $l$ in the second big equation in Appendix A. In the equation for $Q^\lambda$, $Q_n$ is replaced by the equation derived above, but the inner $Q_n$ in that equation is directly replaced by $Q^\lambda$ without any argument.

- Definition 1 defines $\lambda$-discrepancy by an unspecified norm. It is revealed in Appendix E.2 that the norm is l2, but this is not consistent with the proof in Appendix B where the $\lambda$-discrepancy is a weighted difference between the two Q-functions without squares.

- Even if the some of the analysis is true, they only provide properties when $\lambda$-discrepancy is zero. They still cannot explain whether decreasing $\lambda$-discrepancy can reduce the issue of partial observability in some sense. This kind of property may be explored via numerical experiments, but its not done in the paper.

- In the numerical experiments, the proposed method achieves the optimal performance only in the cheese environment. Since increasing the memory size might lead to optimal policy in principle, the inability to achieve optimal performance with more memory in many environments makes the ability of the proposed methods questionable.

### Questions
- Can the authors address the technical issues mentioned above, particularly in Section 3.1 and Appendix A?

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper discusses solving POMDP RL problems through memory augmentation. The authors introduce a $\lambda$-discrepancy, which captures the degree of non-Markovian systems. Based on this property, the authors utilize it as an optimization target to augment the agent's observations based on the memory functions to reduce such a discrepancy. Empirical results are included in the paper to verify the performance.

### Strengths
This paper proposes the $\lambda$-discrepancy to measure the discrepancy between two Q-value functions under the same policy but different λ. They further show that this measure is useful for detecting and mitigating partial observability in a POMDP. They then prove several theoretical properties that make the $\lambda$-discrepancy a reasonable optimization objective for learning effective memory functions, which can be used to reduce such a discrepancy. Simulation results also verify the performance of the proposed algorithms.

### Weaknesses
This paper mentions that the POMDP problem is inherently complex. While memory augmentation is shown to improve performance, it doesn't directly address the computational complexity of solving POMDPs. The efficiency of the proposed approach in more challenging POMDP scenarios is not fully explored.

It is still not clear to me why we need to reduce the $\lambda$-discrepancy, and at least an improved result should be shown by using this technique compared to using a single $\lambda$.

### Questions
- POMDPs are hard to solve due to the large space of the belief states, and thus the algorithms are usually computationally and time-inefficient. However, the complexity of the algorithms is not discussed, and no baselines are included in the experiments.

- Does the algorithm require the full information of the POMDP? If so, we can get the optimal solution with a POMDP solver.

- The condition in Lemma 2 is a sufficient condition; can a necessary condition be provided? That may help in designing the memory function.

- It seems that you approximate Equation (7) with Equation (10), what is the justification for that?

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes the $\lambda$-discrepancy, a theoretical measure for determining the Markovness of observations or memory-estimated Markov states in POMDPs. The measure is based on TD$(\lambda)$. 

As a brief summary, TD(0) is based on one-steps returns (i.e. bootstrapping) $Q(s_t,a_t) = r_t + \gamma Q(s_{t+1} a_{t+1})$). TD(1), also known as Monte Carlo estimation, is written as $Q(s_t, a_t) = \sum_{t=0}^\infty \gamma^t r_t$. TD($\lambda$) is a generalization of TD(0) and TD(1). The $\lambda$ parameter in TD($\lambda$) interpolates between TD(0) when $\lambda = 0$ and TD(1) when $\lambda = 1$.

The authors propose that using error between two Q approximations of differing $\lambda$ to measure how non-Markovian a state is. The intuition is that an agent with poor memory will have differing estimations of TD($\lambda_1$), TD($\lambda_2$). The authors' optimization objective is to minimize the discrepancy.

The authors demonstrate their approach on classical POMDPs, showing they can get close-to-optimal performance on certain tasks.

### Strengths
- The paper is generally well written
- The paper is well motivated -- a metric denoting the Markovness of a POMDP is a very useful tool
- The approach is novel

### Weaknesses
- The experiment section is lacking, only comparing against a memory-free baseline
- The tested environments and models are very low dimensional, and it's not clear such a method would scale to more interesting problems
- If I am understanding this correctly, the proposed metric is flawed in that errors in the Q function (not the state estimator) will result in a $\lambda$-discrepancy. For example, given a perfect state estimator $M^*(o_1, o_2, ..., o_n) = s_n$ and two imperfect Q functions $Q_\theta, Q_\phi$: $ \lVert Q_{\theta}^{\lambda_1}(s_n) - Q_{\phi}^{\lambda_2}(s_n) \rVert > 0$. In this case, I do not see how $\lambda$-discrepancy is a better metric of Markovness than the traditional TD(0) Q learning error.

### Questions
- What makes the $\lambda$-discrepancy specific to partial observability? A policy trained on a fully-observable MDP will also have a $\lambda$-discrepancy if the Q function is even slightly less-than-optimal.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper studies resolving partial observability through learning memory in non-Markov Decision Processes. The authors propose to use the discrepancy between TD($\lambda$) with different $\lambda$s to measure if the extracted states are Markovian. This discrepancy, termed $\lambda$-discrepency, is then used as an objective to learn memory states. The authors analyze the cases where the $\lambda$-discrepancy vanishes. Lemma 1 introduces a simplest case. Lemma 2 analyzes a particular correlation between the reward and the state-action distribution. Theorem 1 examines that such a specific correlation is very rare. Together they show that it is possible in practice to test with one policy whether the $\lambda$-discrepancy is nonzero for all policies, hereby prove that $\lambda$-discrepency can be an effective learning objective. Concrete algorithms are designed for both analytical estimation and deep learning. Empirical results show that the proposed method can successfully learn memory states in a series of classic POMDP tasks, and achieve better performance than TD methods.

### Strengths
Markovianizing nMDPs is a classic problem in the literature of sequential decision-making. The authors propose a novel method to tackle this problem using well-developed tools such as TD($\lambda$).

The quality of this work is good. The strongly analytical narrative makes it convincing. The logic flow is smooth. Proves are solid. The authors did a great job balancing the theoretical formulation and intuition. 

The proposed problem and solutions for nMDPs are both relevant. Relaxing the Markovian assumption in sequential decision-making is an active topic recently, see Abel et al. 2021, Janner et al. 2021, Chen et al. 2021., Qin et al. 2023. The author may consider citing these prior works if they haven't done that yet. 

Abel et al. On the expressivity of markov reward. NeurIPS 2021. 
Janner et al. Offline reinforcement learning as one big sequence modeling problem. NeurIPS 2021. 
Chen et al. Decision transformer: Reinforcement learning via sequence modeling. NeurIPS 2021.
Qin et al. Learning non-Markovian Decision-Making from State-only Sequences. NeurIPS 2023.

### Weaknesses
I am a bit concerned with the scalability of the proposed method. The empirical results do not include a gradient-based method for model-free implementation of $\lambda$-discrepency. I wonder if the authors would like to discuss the concrete challenges they met in such experiments.

### Questions
I happened to have reviewed an earlier version of this work. I am very glad to see the authors have taken some suggestions from the reviewers into account and reframed the paper. However, I also find the authors haven't resolved one of those questions in this revised version. 

In the previous version, the proposed method was shown to be struggling to acquire satisfying performances in two tasks. In this version, the authors have shown their success in one of them, Network. However, it seems the result of the other, Hallway, is omitted. I hope the authors would like to provide an explanation in their response.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
