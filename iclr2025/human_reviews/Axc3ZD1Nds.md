## Human Reviewer 1

### Summary
This paper makes three key contributions to the field of cumulative-prospect-theory (CPT)-based reinforcement learning, a framework where the agent aims to maximize not the raw cumulative reward but a transformed version of it. This transformation involves both a utility function and a probability weighting function, introduced to better reflect human value assessments of rewards (e.g., monetary amounts converted into perceived value). The objective is to find a policy that maximizes this transformed cumulative reward.

CPT-based reinforcement learning generalizes various prior approaches, including: (1) the standard RL framework, (2) risk-sensitive RL with an exponential utility function (e.g., as in Noorani et al., 2022), (3) optimization based on VaR/CVaR of the total reward, and (4) the case studied by Kahneman and Tversky (1992).

The first contribution presents new insights into the characteristics of optimal policies within CPT-based reinforcement learning. These findings include: (1) deterministic, history-dependent optimal policies may not always exist; (2) in the absence of weighting functions, a deterministic, non-stationary optimal policy exists that depends on the current state and past cumulative reward; and (3) without weighting functions, a Markovian, non-stationary optimal policy exists if and only if the utility function is either affine or exponential.

The second contribution introduces a policy gradient theorem and a novel algorithm for CPT-based reinforcement learning.

The third contribution is an empirical evaluation of the proposed algorithm. This study includes (1) a performance comparison with CPT-SPSA-G, an existing RL algorithm for CPT problems, across various grid-world environments, demonstrating the proposed algorithm's advantages, particularly in larger environments; and (2) an exploration of different behaviors under varying utility and weighting functions.

### Strengths
The theoretical and algorithmic contributions are original and clear. If one believes that CPT-based RL problems are important, then these contributions are also significant. 

The paper did a very nice job relating its studied problem with other related problem settings studied previously, making it easy for readers to understand the position of this paper in the literature.

The paper's presentation is clear.

### Weaknesses
While I appreciate the contributions the authors made to CPT-based RL problems, I am not convinced of the importance of CPT-based RL problems by reading this paper. The paper cites another paper "... this is particularly important in applications directly involving humans
in the loop such as e-commerce, crowdsourcing and recommendation to name a few" to argue that this class of problems has important applications. However, it is not clear from the cited paper whether other formulations that take into account human behavior can also handle these applications well. In fact, compared with CPT, I feel that a more principled way that considers human behavior is trajectory-based reward RL problems (one reward for the entire trajectory) with human preferences, like RLHF in large language models. While CPT assumes a structure of the final reward (the final metric to optimize involves a weight function, a utility function, and a summation of rewards), these weight functions are unknown and must be learned from data. In contrast, in trajectory-based reward RL problems, there is no assumption on the structure of the metric being optimized. And the metric is learned with human preference data. I wonder how authors think about the pros and cons of CPT in terms of its applications compared to trajectory-based reward RL problem settings.

Another weakness is related to the empirical study of the paper. The experiments only demonstrate the effectiveness of the proposed algorithm but do not include those justifying the necessity of the CPT problem. It would be much better if the authors could demonstrate that in some applications, the data shows that human behavior can be accurately predicted with CPT but not other theories.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper is a follow-up work to [L.A. et al., 2016] on Cumulative Prospect Theory (CPT) value optimization in reinforcement learning problems. The original paper utilized the simultaneous perturbation stochastic approximation (SPSA) method to update the policy, while this paper provides a policy gradient method. The policy gradient is evaluated and compared with the SPSA method in several domains.

### Strengths
+ Discussion on the optimal policies under CPT, showing that optimal policies are generally stochastic and non-Markovian when CPT is applied.
+ Policy gradient algorithm for CPT value optimization, compared to SPSA methods in the original paper.

### Weaknesses
+ The algorithm is only validated in a few small domains, making it difficult to assess the performance of this policy gradient in more complex tasks.
+ The computation of the integral for the CPT value appears complex and time-consuming due to the use of Monte Carlo sampling.

Suggestions:

+ The authors may consider demonstrating the benefits of using CPT over standard risk-neutral RL in a concrete reinforcement learning problem at the beginning of the paper to help readers better understand CPT.

+ This paper considers the problem in a non-discounted setting, i.e., $\gamma=1$ as shown in Line 181. The claim in line 261 does not apply to the discounted case for exponential utilities. Discussion may be needed. see [Entropic Risk Optimization in Discounted MDPs, AISTATS 2023]

+ As mentioned in line 309, Proposition 7 in the paper is different from the Proposition 6 in [L.A. et.al. 2016], authors may need to provide a comparison and justification.

### Questions
+ $\phi(R(\tau))$ in line 295, 303, 305 should be $\varphi(R(\tau))$ in Theorem 6? 

+ How to choose the proper utility function for different problems?

+ The policy gradient calculation depends on the property that the function $u$ is non-decreasing. Can we work with other $u$ where non-decreasing is not guaranteed?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper introduces a novel approach to reinforcement learning (RL) by leveraging Cumulative Prospect Theory (CPT) to account for human decision-making biases, moving beyond the traditional expected utility framework. It focuses on the policy optimization problem, where the objective is the CPT value of the random variable recording the cumulative discounted rewards (CPT-PO). Additionally, it considers the particular case of the expected utility objective, where only returns are distorted by the utility function (EUT-PO). This work derives a policy gradient (PG) theorem for CPT-based RL, presenting a model-free policy gradient algorithm for optimizing a CPT policy.

### Strengths
This paper addresses an objective that encompasses a broad class, including Conditional Value at Risk (CVaR), distortion risk measures, and expected utility.

The paper effectively explains various policy classes associated with different objectives, helping to clarify how these classes relate to CPT-RL versus standard RL approaches.

The derivation of a policy gradient theorem for CPT objectives is a significant extension of the traditional PG results in reinforcement learning, broadening its applicability in human-centered decision contexts.

### Weaknesses
The experimental section provides limited insight due to its use of relatively straightforward examples (traffic control and electricity management), which may not fully illustrate the complexities that arise in more realistic, high-stakes environments, such as finance or healthcare.

In Section 5(a), the observation that different weight functions lead to different policies in the grid environment could be strengthened by assessing these policies against risk measures (such as CVaR) beyond expected return. Without this, it is unclear if the CPT-RL-PO policy outperforms standard RL-PO under any specific risk-sensitive criteria.

The grid environment results are overly simplified and provide little substantive information. It would be beneficial to evaluate whether the derived policies' performance aligns meaningfully with their respective objectives.

Limited applicability: The experiments did not demonstrate the advantages of using the proposed algorithm. It is unclear what the benefits of using CPT learning are over risk-sensitive RL or distributional RL. Although CPT is theoretically valuable, the empirical advantages of CPT over other risk-sensitive measures remain ambiguous in the presented results. The authors may consider including direct comparisons with risk-sensitive RL or distributional RL methods on the same tasks.

### Questions
Minor comment: On page 4, under Problem formulation: CPT-RL, the cumulative discounted rewards variable $X$ is referenced, but the definition provided does not discount the rewards. It would be clearer to either update the variable's definition to include discounting or adjust the notation accordingly to prevent confusion.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
3