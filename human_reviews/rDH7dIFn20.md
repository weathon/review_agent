# Variance-aware Regret Bounds for Stochastic Contextual Dueling Bandits

- Avg Score: 6.00
- Decision: Accept (poster)
- Scores: 5, 6, 5, 8

## Abstract
Dueling bandits is a prominent framework for decision-making involving preferential feedback, a valuable feature that fits various applications involving human interaction, such as ranking, information retrieval, and recommendation systems. While substantial efforts have been made to minimize the cumulative regret in dueling bandits, a notable gap in the current research is the absence of regret bounds that account for the inherent uncertainty in pairwise comparisons between the dueling arms. Intuitively, greater uncertainty suggests a higher level of difficulty in the problem.  To bridge this gap, this paper studies the problem of contextual dueling bandits, where the binary comparison of dueling arms is generated from a generalized linear model (GLM). We propose a new SupLinUCB-type algorithm that enjoys computational efficiency and a variance-aware regret bound $\tilde O\big(d\sqrt{\sum_{t=1}^T\sigma_t^2} + d\big)$, where $\sigma_t$ is the variance of the pairwise comparison at round $t$, $d$ is the dimension of the context vectors, and $T$ is the time horizon. Our regret bound naturally aligns with the intuitive expectation — in scenarios where the comparison is deterministic, the algorithm only suffers from an $\tilde O(d)$ regret. We perform empirical experiments on synthetic data to confirm the advantage of our method over previous variance-agnostic algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper addresses the problem of dueling bandits with a variance-aware regret bound. It introduces an algorithm for contextual dueling bandits, which accounts for uncertainty in pairwise comparisons between arms and provides a variance-aware regret bound. The paper highlights the importance of considering uncertainty in decision-making scenarios and proposes an efficient algorithm with a regret bound that depends on the variance of comparisons, dimensionality of context vectors, and the time horizon. The authors demonstrate the effectiveness of their method through empirical experiments.

### Strengths
The paper is well written and easy to follow.

The proposed algorithm is the first algorithm with a variance-aware regret bound. It has some novelty, and the symmetric arm selection is new.

The authors also support their theoretical claims with empirical experiments on synthetic data.

### Weaknesses
I feel this paper is an extension to [1], with similar algorithm proposed and regret analysis. The main algorithm has identical structure with the SAVE proposed in [1] based on the SupLin methodology, and it is not new and novel for the variance-aware contextual bandit problem. Under the generalized linear contextual dueling bandit setting used in the paper, we can regard $x_t - y_t$ as the contextual information, which would then be degeneralized to the ordinary generalized linear contextual bandit problem. I think the high similarity with the existing literature is the major weakness of this work.

Typos and minor issues: 
1. title of subsection 4.3 (sysmmetric -> symmetric)
2. I may overlook. What is $\kappa$ in line 20 Algorithm 1 and Eqn. (4.3)?

[1] Variance-dependent regret bounds for linear bandits and reinforcement learning: adaptivity and computational efficiency. Zhao et al., COLT.

### Questions
In addition to my concerns in the above Weaknesses section,

1. Is the variance of the noise $\epsilon_t$ equal to $p_t(1-p_t)$ under the Bernoulli setting? In that case I feel the variance would be fully dependent on the arms selected in each round.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper addresses the problem of contextual dueling bandits, where the feedback is based on pairwise comparisons between arms. The authors focus on scenarios where the binary comparisons are generated from a GLM. The authors proposed VACDB algorithm that adapts to the variance of the pairwise comparisons, leading to potentially better performance in scenarios with varying levels of uncertainty. The variance-aware regret bound of order $O\left(\sqrt{d\sum_{t=1}^{T} \sigma_t^2} + d\right)$, which also aligns with intuitive expectations, reducing to $O(d)$ in deterministic scenarios. The authors validate their approach through experiments on synthetic data, demonstrating the advantages of VACDB over previous variance-agnostic algorithms.

### Strengths
- The primary contribution of the paper is the introduction of a new algorithm VACDB (Variance-Aware Contextual Dueling Bandits), which incorporates a SupLinUCB-type approach to handle the contextual information and provide a variance-aware regret bound. 
- The regret bound $O\left(\sqrt{d\sum_{t=1}^{T} \sigma_t^2} + d\right)$ proposed by the authors provides a more nuanced performance measure that reflects the difficulty of the decision-making problem.
- Beyond the specific algorithm and regret bound, the paper also contributes to the theoretical understanding of generalized linear bandits, correcting an issue in the existing analysis of the MLE estimator.

Overall, I think the paper enhances our understanding of decision-making in dueling bandits scenarios by explicitly accounting for the uncertainty in pairwise comparisons, providing both theoretical insights and practical algorithms to address the challenge on variance-awareness algorithms.

### Weaknesses
- The paper primarily conducts experiments on synthetic data to validate the proposed algorithm. While this is a common practice, the performance of the algorithm in real-world scenarios might differ. To strengthen the paper, the authors could include experiments on real-world datasets, particularly those related to the applications mentioned like ranking, recommendation systems, or any human-interactive system, ensuring the practicality and robustness of the algorithm in diverse settings.
- The concept of a layered design for bandit algorithms, which the authors adopt in the VACDB algorithm, has been previously explored in the literature. Layered or epoch-based approaches are widely used in bandit algorithms to balance exploration and exploitation in stochastic settings. The paper could benefit from a more thorough discussion on how the layered design in VACDB specifically contributes to or differs from existing approaches.
- The VACDB algorithm relies on the computation of regularized MLE for parameter estimation. The computation of these estimators is known to grow in complexity with the number of iterations, potentially leading to scalability issues, especially in settings with a large number of arms, contexts, or iterations.

### Questions
- There is a minor typo on the second line of Introduction: "arm" -> ``arm"


================================================

**After Rebuttal:**

I appreciate the response from the authors, and I am going to keep my original rating.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work studied the stochastic contextual dueling bandits. It proposed the VACDB algorithm, which is a new SupLinUCB-type algorithm. It provided a variance-aware bound on the regret of the proposed algorithm. This work presented a detailed review on existing literature and clarified the novelty of the proposed algorithm. It also evaluated the performance of the algorithm with numerical experiments.



==============

I appreciate the response from the author(s). I may increase the score if they further resolve my concern.

### Strengths
1. This paper is in general well organized and easy to follow.
1. The variance is considered in the dueling bandit setting.
1. I appreciate the detailed review on existing literature, and the clarification on the novelty of the proposed algorithm and the differences from existing algorithms (especially SupLinUCB-type ones).

### Weaknesses
1. The variance $\sum_{t=1}^T \sigma_t$ in the regret bound is a random variable. I think it would be much better to involve a term that indicates the variance of the instance in some sense but is not random in an expected bound.
    1. The appearance of $\sum_{t=1}^T \sigma_t$ indicates that even if we know $X_t$ for all arms and $\theta^*$, we may not know the value of the derived upper bound.
1. I wonder if it is possible to derive a lower bound for the problem. If not, may the author(s) clarify the analytical challenge?

### Questions
Please refer to the **Weaknesses** section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies contextual dueling bandits, where binary comparison is generated from a generalized linear model. The work proposes a new framework to obtain variance aware regret guarantees. This work further provides empirical results comparing the algorithm against baseline.

### Strengths
The problem setup is well laid out and easy to follow with the precisely required assumptions.

### Weaknesses
The paper is not self contained and requires reader to go through multiple papers for example in section 4.2.

### Questions
No questions.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
