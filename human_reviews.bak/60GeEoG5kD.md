# Doubly Optimal Policy Evaluation for Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 8, 5, 6

## Abstract
Policy evaluation estimates the performance of a policy by (1) collecting data from the environment and (2) processing raw data into a meaningful estimate. Due to the sequential nature of reinforcement learning, any improper data-collecting policy or data-processing method substantially deteriorates the variance of evaluation results over long time steps. Thus, policy evaluation often suffers from large variance and requires massive data to achieve the desired accuracy. In this work, we design an optimal combination of data-collecting policy and data-processing baseline. Theoretically, we prove our doubly optimal policy evaluation method is unbiased and guaranteed to have lower variance than previously best-performing methods. Empirically, compared with previous works, we show our method reduces variance substantially and achieves superior empirical performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the problem of efficient policy evaluation: designing a behavioral policy to collect data in order to estimate a target policy s.t. It requires as little as possible policy rollouts in order to evaluate the target policy performance for a given confidence level. 

The authors cast this problem as a double step minimization problem of the variance of the importance sampling corrected reward traces (termed the G^PDIS), where the components to minimize are the behavioral policy \mu and a baseline function b that can reduce the overall variance of G^PDIS, while maintaining unbiasedness s.t. E[G^PDIS] converges to the target policy performance.

The authors develop an algorithm, named DOPT, inspired by two established methods for policy evaluation: (1) DR which uses the optimal baseline function and (2) ODI which do not take into account a baseline function, but solves for the optimal behavioral policy that minimizes the variance of G^PDIS (again without inserting a baseline function).

The authors show theoretically that if one adds both the optimal baseline function and solves for the optimal policy given the baseline we obtain a stronger estimator that minimizes the variance terms with respect to both DR and ODI (which are both superior over the on-policy case, where one uses rollouts from the target policy to approximate its performance).

The paper then continues with the practical implementation of how to estimate the theoretically justified behavioral policy and it concludes with an experiment section where it is shown that DOPT outperforms both ON (on-policy), DR and ODI in terms of evaluation steps with respect to the relative error (the difference between the estimated performance to the real performance scaled by this difference for on-policy evaluation after one episode). The experiments are executed in a synthetic grid-world and in the Mujoco physics simulator.

### Strengths
The related work section is comprehensive, contextualizing the paper within current literature and algorithms. The paper is generally clear and accessible, a crucial feature given the depth of theoretical content. The theoretical analysis rigorously establishes DOPT's optimality over on-policy, DR, and ODI baselines.

### Weaknesses
While optimal theoretically, in practice DOPT requires off-policy estimation of (1) the Q-function q(s,a), (2) the next state value variance \ni(s, a) and (3) the amplification factor u(s,a) (how much should the probability of an action be amplified for a given s,a pair). Specifically, u is learned based on a Bellman operator with \ni as a reward and with an important sampling correction term (Lemma 3). Both \ni and u are not required by previous schemes: DR and ODI and both can introduce errors. Specifically, u which uses Bellman operator that can potentially add bootstrapping errors and importance sampling term with high variance. Although empirical results support DOPT’s advantage, further analysis of these considerations is warranted for a complete evaluation. The authors should do a better job in analyzing and presenting these considerations.

### Questions
1. In your formulation your assume that Q_{\pi} is learned from the offline data, one would expect that a basic estimator of V(S0) in this case would be \sum_a \pi(a|S0) Q(a, S0) you should explain why it is not an appropriate estimator and add it to some of your empirical evaluations.

2. Based on your arguments in Section 4 (Lines 190-230) you require unbiasedness in every state not just s0, how does it reconcile with your argument that you enlarge the state space?

3. Figure 1 shows similar patterns for sample counts of 1000 and 27000. Can the authors clarify why similar accuracy requires the same number of samples for differently scaled problems?

4. Including a background on DR and ODI in the background section could enhance reader comprehension, as these are central to the analytical comparisons made.

5. Details on estimating \ni in practical settings should be incorporated into the main text. Without this information, readers may struggle to understand the algorithm’s practical execution.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper aims to enhance the sample efficiency of online reinforcement learning algorithms by reducing the variance of estimation of the total rewards. To achieve this, the authors introduce a bi-level optimization method with both optimal data-collecting policy and data-processing baseline. The paper provides detailed proof showing a guaranteed lower variance of the proposed method than those from the state-of-the-art methods. The empirical results on MiniGrid and tasks from the Mujoco environment conducted in this paper further demonstrate the effectiveness of the proposed method, outperforming existing methods in these tasks.

### Strengths
+ The paper is quite well-written and easy to follow. The motivation and the key idea of the method are clearly presented. The differences between the proposed method and prior work have been clearly discussed in the related work section.
+ The authors provide a detailed and comprehensive theoretical proof to support the proposed method, though there seem to be some errors in the proof of theorem 1.
+ The experiments show performance improvement over existing methods.

### Weaknesses
- A potential error in the proof:

In equation (23), the second equal sign does not make sense. By equation (12) that 

${\mu}^{*}_t (a|s) \propto \pi _ t (a|s) \sqrt{u _ { \pi, t} (s,a)}$.

We should have 

$\sum_{a} \frac{\pi _ t(a|S_t)^2}{\mu^{*}_t (a|S_t)} u _ {\pi, t}(S_t,a)$

$= \sum_{a} \pi _ t (a|S_t)\sqrt{u _ {\pi, t} (S_t, a)}$

However, in the paper the author seems to give the wrong result as 

$\sum_{a} \frac{\pi _ t(a|S_t)^2}{\mu^{*}_t (a|S_t)} u _ {\pi, t}(S_t,a)$

$=\sum_{a}\pi_t(a|S_t)\sqrt{u _ {\pi, t}(S_t, a)} \sum_{b}\pi_t(S_t, b)\sqrt{u _ {\pi, t}(S_t, b)}$.

Unless I'm missing something, this seems to be an error that may invalidate the optimality of the proposed behavior policy, as the results is used in subsequent analysis and proofs. The proof also contains typos and minor issues, such as the following. All of these cast doubt on the soundness of the analysis.
	
After the second equal sign in equation (18), there's an extra ")" after $\bar{b_t}(S_t)$; 

At the start of equation (20), $\nu_t(S_t, A_t) =$ should be removed; 

In page 17, the $\mu^{*}_t$ after the third equal sign should be $\mu_t$.

- Figure 3 looks exactly the same as the Figure 3 in the following paper [1], however without citing the source. This is a major concern.

[1]Liu, S. and Zhang, S. (2024). Efficient policy evaluation with offline data informed behavior policy design. In Proceedings of the International Conference on Machine Learning.

- While the experiments show good improvement of baseline algorithms, I would suggest the experiments be expanded to more environments such as Atari (RL with visual input) and more tasks in MuJoCo, where the sampling are generally less efficient.

### Questions
1. Please address my concerns in the weakness.

2. According to algorithm 1, will the offline dataset affect the performance? In the experiments these offline datasets are collected by random policies. I wonder what if it is collected by an expert policy or other more sophisticated behavior policies, like in most RL settings? This is important to demonstrate the applicability of the proposed method on datasets with different expertise levels.

### Soundness
2

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
This paper studies the policy evaluation problem. Given a policy $\pi$ to evaluate, the authors adopted the classic importance-weighting estimator with baseline function. By optimizing the variance of this estimator, they derived an optimal behavior policy and baseline function. They show that the variance of the importance weighting estimator equipped with their optimal behavior policy and baseline function is smaller than three existing estimators including the on-policy estimator and the doubly robust estimator. Additionally, they implemented experiments on practical tasks to show the effectiveness of their doubly optimal evaluation method.

### Strengths
1. It is an interesting problem to study that to evaluate a given policy, what is the best behavior policy to collect samples.

### Weaknesses
The pipeline of their proposed method is that: First, calculating an optimal behavior policy and baseline function by solving an optimization problem; Second, using the derived behavior policy to collect samples and use importance-weight estimator on these samples. 
1. However, in the first stage, in order to calculate the optimal behavior policy and baseline function, lots of samples are needed and I think the samples needed in this stage are much more than the second stage (since the definition of $u(s,a)$ needs lots of information e.g. the transition function). In this case, it is meaningless to reduce the variance of the estimator.
2. And the optimal baseline function is found to be the q-value function of the target policy. If we already figure out an optimal baseline function, i.e. the q-value of the target policy, then it is done. No need to do importance-weighting any more. Therefore, again it is meaningless to reduce the variance of the estimator.

The above is the biggest weakness. It is not clear what is achieved by their method theoretically or practically. 

Besides, there are also other problems which make their theory fragile.

3. The set $\Lambda$ of feasible behavior policies seems weird to me. It contains behavior policy $\mu$ such that $\mu(a|s)=0$ while $\pi(a|s)>0$ ($q(s,a)=0$). In this case, the importance weighting estimator goes to infinity, let alone control the variance. One solution is to assume non-negative rewards, since with non-negative rewards $q(s,a)=0$ indicates zero rewards by playing $a$ at state $s$ which can make the importance weighting estimator be zero. However, in the paper, they don't discuss it and make any assumptions.

### Questions
My main question is what I have listed in the weaknesses part: what is the pipeline of the proposed method? If the pipeline is same as what I write in the weaknesses part, then what is meaning to reduce the variance of the estimator.

### Soundness
3

### Presentation
3

### Contribution
2
