# Deployment Efficient Reward-Free Exploration with Linear Function Approximation

- Decision: Reject
- Scores: 6, 5, 3, 5

## Abstract
We study deployment efficient reward-free exploration with linear function approximation, where the goal is to explore a linear Markov Decision Process (MDP) without revealing the reward function, while minimizing the number of exploration policies used during the algorithm. We design a new reinforcement learning (RL) algorithm whose sample complexity is polynomial in the feature dimension and horizon length, while achieving nearly optimal deployment efficiency for linear MDPs under the reward-free exploration setting. More specifically, our algorithm explores a linear MDP in a reward-free manner, while using at most $H$ exploration policies during its execution where $H$ is the horizon length. Compared to previous algorithms with similar deployment efficiency guarantees, the sample complexity of our algorithm does not depend on the reachability coefficient or the explorability coefficient of the underlying MDP, which can be arbitrarily small for certain MDPs. Our result addresses an open problem proposed in prior work. To achieve such a result, we show how to truncate state-action pairs of the underlying linear MDP in a data-dependent manner, and devise efficient offline policy evaluation and offline policy optimization algorithms in the truncated linear MDP. We further show how to implement reward-free exploration mechanisms in the linear function approximation setting by carefully combines these offline RL algorithms without sacrificing the deployment efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies deployment-efficient RL in linear MDPs. The authors contribute a reward-free exploration algorithm and prove it only takes $O(H)$ deployments to learn a near-optimal policy. That result does not rely on additional reachability assumption, and therefore, close the gap between the $\Omega(H)$ lower bound established in previous work.

### Strengths
The deployment efficient setting indeed reflects practical concerns and it is important.

The algorithm proposed in this paper has non-trivial and interesting ingredients.

The discussion in Section 4 is detailed and explain the insights clearly.

### Weaknesses
1. My main criticism would be the paper writing. Especially, the proof sketch in Section 6 does not provide too much information to me. It involves a lot notations and the authors did not explain the insights of the key steps intuitively. For example, in the proof sketch for Lemma 6, it is just mentioned which lemmas in appendix are used. Given the page limit has not been reached yet, It would be better to explain the intuition with more details.

    Besides, I'm not sure whether it is a good idea, but it may be easier for reader to understand if the authors can re-organize Section 4-6 to make connections among the explanation in Section 4, the algorithm design in Section 5 and the technique lemma in Section 6. Now the authors discuss them separately, which makes it hard to track which part of the algorithms correspond to which technique innovation.

2. I also have a few questions about the proofs. Please check Question part.

3. There are also quite a few typos. For example, line 350, 465, 475, the sub-scription of $\phi$. Besides, some notations in the appendix seem not clear to me. See Question part.

4. Given there are a lot notations, it would be better to have a table of frequently used notations in appendix for reference.

### Questions
1. Lemma 7, what do you mean by "maps a context $X$ to a distribution over $X$"? Why not just consider $\pi^G = \mathcal{K}$? Or it should be "context $x$"?

2. Lemma 8, can you explain why the last inequality in Eq.(10) holds?

3. Lemma 11 and 14, what do the $\mu$ and $\theta(v)$ denote, and how should we interpret them?

4. Lemma 15, it says "$\hat{F}\_0(\Lambda)$ be the value computed by line 12-24 of Algorithm 3". However, in 12-24 of Algorithm 3, $\hat{F}$ is defined on $\tilde{s}\_{h-1,i}$, which one of them do you refer to?

### Soundness
3

### Presentation
2

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
The paper studies deployment efficient reward-free exploration with linear function approximation. The proposed algorithm achieves nearly optimal deployment efficienc and does not depend on the reachability coefficient or the explorability coefficient of the underlying MDP as previous literature.

### Strengths
The paper is clearly written. The algorithm design and theoretical analysis is non-trivial, especially the idea of quantification/design of the truncated state-action pairs in the underling MDP.

### Weaknesses
The algorithm is heavily engineered with artifical tricks, which makes the algorithms not tractable (correct me if I am wrong). While the initial goal was to inspire practical algorithms that enhance deployment efficiency in RL applications, one might question how effectively this approach serves that purpose.

### Questions
1. Can you provide some discussions on the tractability of the algorithm? Can the current algorithm motivate some practical algorithms? Is any experiment possible to show the performance?

2. The algorithm required $2m^2+1$ independent datasets, and $m$ is set to be $32000d^2H^3/\epsilon$, which is insanely large. This requirement alone could make the 'deployment effciency' vacuous. I'd like some explanation on this point.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
1

### Summary
This paper studies deployment efficient reward free exploration problem in linear MDPs. They proposed an algorithm which achieved optimal deployment complexity $O(H)$ and polynomial sample complexity while no assumption on the reachability of the underlying MDP is needed. This result provides a positive answer to the open problem whether people can get rid of the reachability assumption under the linear MDPs setting.

### Strengths
1. The problem of studying deployment-efficient exploration policy is well-motivated both from a practical and theoretical perspective.

### Weaknesses
I admit that I cannot assess the correctness of their method so I will set my confidence score as 1. However, I hold the opinion that even though assuming the theory is correct, I don't think the current version is ready to be published. The presentation and organization of this paper need to be improved a lot. It is really hard to understand their idea even qualitatively. For example, Section 4 should be the place where they explain their high-level idea, however, after reading this section, people still have no idea on what exact technical problem will emerge if there is no assumption on reachability and how they handle these technical problems.

### Questions
Can the authors briefly explain what exact technical problem will emerge if there is no assumption on reachability (by some lines of equations) and how they handle these technical issues?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper investigates deployment-efficient reward-free exploration with linear function approximation in reinforcement learning. The goal is to explore a linear Markov Decision Process (MDP) without revealing the reward function, while minimizing the number of exploration policies. A reinforcement learning algorithm is proposed with polynomial sample complexity, achieving near-optimal deployment efficiency for linear MDPs.

### Strengths
1. The paper provides a thorough and precise theoretical justification for the proposed algorithm.

2. The proposed algorithm demonstrates a promising sample complexity bound that is polynomial in both the feature dimension and horizon length.

### Weaknesses
1. While the algorithm is theoretically sound, the paper does not provide a detailed comparison with other existing methods in terms of empirical performance.  

2. The paper introduces a reward-free exploration strategy but does not provide an in-depth analysis of the exploration-exploitation trade-off inherent in the algorithm. Specifically, it lacks a clear explanation of how the exploration policy adapts over time, especially in the presence of uncertainties in the reward-free setting.

### Questions
1. How does the proposed algorithm handle cases where the linear function approximation is not perfect or when the feature dimension is large? What are the potential implications for the performance bounds under these circumstances?

2. The exploration strategy in the proposed algorithm minimizes exploration policies, but how does the method balance this trade-off with the inherent uncertainty in large-scale environments? Are there conditions where minimizing the exploration policies might hinder the learning process in practice?

3. How does the proposed reward-free exploration strategy balance the exploration of state-action pairs with the need to minimize exploration policies?

### Soundness
3

### Presentation
2

### Contribution
2
