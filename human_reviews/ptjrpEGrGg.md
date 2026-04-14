# Learning from Imperfect  Human Feedback: A Tale from Corruption-Robust Dueling

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6, 6

## Abstract
This paper studies Learning from Imperfect Human Feedback (LIHF), addressing the potential irrationality or imperfect perception when learning from comparative human feedback. Building on evidences that human's imperfection decays over time (i.e., humans learn to improve), we  cast this problem as a concave-utility continuous-action dueling bandit but under a restricted form of  corruption:  i.e., the corruption scale is decaying over time as $t^{\rho-1}$ for some ``imperfection rate''  $\rho \in [0, 1]$.  
With $T$ as the total number of iterations, we establish a regret lower bound of $ \Omega(\max\{\sqrt{T}, T^{\rho}\})$ for LIHF, even when $\rho$ is known. For the same setting, we develop the Robustified Stochastic Mirror Descent for Imperfect Dueling (RoSMID) algorithm, which achieves nearly optimal regret $\tilde{\mathcal{O}}(\max\{\sqrt{T}, T^{\rho}\})$. Core to our analysis is a novel framework for analyzing gradient-based algorithms for dueling bandit under corruption, and we demonstrate its general applicability by showing how this framework can be easily applied to obtain corruption-robust guarantees for other popular gradient-based dueling bandit algorithms. Our theoretical results are validated by extensive experiments.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the problem of corrupted dueling bandits with strictly concave reward functions. It also considers a specific assumption (motivated by empirical observations on human behaviors), that the corruption budget at each time step decreases as $t^{\rho-1}$ for $\rho\in(0,1)$.

For that problem, the authors propose a lower bound in $\Omega(\max(\sqrt{T},T^{\rho}))$ and a stochastic mirror descent algorithm matching this bound, up to logarithmic terms.

### Strengths
It is the first work treating the problem of bandits with (strictly) concave utility functions and continuous actions spaces and it manages to do so with a quite complete picture and nice results :
- a lower bound, similar to linear bandits with corruption is proven: its proof nicely adapts to the fact that the corruption budget is individually bounded at each time step, rather than only having a bounded sum
- an algorithm with a nearly optimal (up to logarithmic terms) bound is proven. Moreover, its proof nicely constructs on the assumption that the corruption budget decay with time, making this assumption necessary for its good functioning

Overall, I find the problem interesting and the paper well written: it nicely discusses all the limitations of their results and why extending them is not obvious

### Weaknesses
My main concern is on the assumption of the link function. More precisely, it seems to me that the only link function $\sigma$ satisfying the conditions given between lines 143 and 153 is the constant function $1/2$.
I guess the reason for this is mainly due to the fact that either (or both) $\sigma$ should not be defined on $\mathbb{R}$ (but on an interval), nor that it is concave. 

For instance, I would have liked that these conditions include the simoid function (which is the mostly used link function in the literature) and would suggest the authors to provide at least one example of link function that satisfies their conditions. Actually, I am quite surprised of the concavity assumption of $\sigma$. I guess a monotonicity assumption would be more desirable and natural here.

Additionally, this is not explicitly written, but it seems that the authors rely on the fact that $l_1^{\sigma}>0$ in the whole paper. 

-------

Here are some minor remarks:
- I think the quantifiers in the lower bound results (Theorem 3.1, Lemma 3.1, Proposition) are given in a wrong order. It should typically be instead "for any learner, there exists an instance ..."

### Questions
- Can you relax the assumptions on the link function $\sigma$, so that it includes natural functions?
- Your algorithm requires a prior knowledge of the corruption budget $\rho$. Do you have any idea if we could design an algorithm with reasonable regret guarantees in the scenario where $\rho$ is unknown?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies a new problem of LIHF under the dueling bandit setting with a continuous action space. This paper proposed a regret lower bound for this problem and then introduced a new algorithm based on gradient-based mirror descent that attains this lower bound. Experimental results are further conducted to validate the high efficiency of the proposed algorithm.

### Strengths
1. I didn't check all the proof in Appendix due to limited bandwidth, but after taking a close look at the theoretical results and the proof outlines, I feel the technical results are sound and reasonable.
2. The paper is clearly written and easy to understand.
3. This paper studies a timely topic of robust bandits to adversarial corruptions.
4. Experimental results are presented after theoretical analysis.

### Weaknesses
1. I feel there should be more literature review to help readers better understand your problem setting. I am not familiar with robust bandits, so after reading some existing literature, I suggest the authors to add the following discussion:

1.1 It's better to highlight the starting point of this line of work (Lykouris et al. Stochastic bandits robust to adversarial corruptions; Gupta et al. Better Algorithms for Stochastic Bandits with Adversarial Corruptions, etc.) and their formulation of this problem.

1.2 Since this paper consider the continuous action space/Lipschitz concave utility function under adversarial corruptions, it is better to mention some similar existing work on continuous action space such as robust kernelized bandit (Bogunovic et al. A robust phased elimination algorithm for corruption-tolerant Gaussian process bandits, etc.) and robust Lipschitz bandit (Kang et al. Robust Lipschitz bandit to adversarial corruptions, etc.), and discuss the connections with these existing works. It seems that this line of work still uses the classic bandit theory framework which is different from your gradient-based one, but I feel this point should be highlighted in your main paper.

1.3 More existing literature on gradient-based mirror descent should also be presented. It is important to let readers know how you modify the gradient-based idea to deal with the adversarial corruptions.

1.4 For the assumptions made in line 143-152, it is better to highlight where these assumptions come from one by one.

2. Since you studied a very new problem of decaying corruptions and argue that this is a practical problem, I feel it is better to add some real-world experimental results in your study. Simulations can hardly validate the high efficiency of your problem setting.

### Questions
1. Why do you think the assumptions made in line 143-152 are reasonable? I feel they are very specific.

2. Can you report the running time of your proposed algorithm compared with baselines? Since you study a practical problem and hence the algorithmic efficiency is also an important factor.

### Soundness
3

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
3

### Summary
This paper studies This paper studies Learning from Imperfect Human Feedback, where human's imperfection decays over time. The authors begin by establishing a theoretical lower bound, suggesting that learning from imperfect human feedback can be as challenging as learning under arbitrary adversarial corruption. They then propose a new algorithm RoSMID, designed for continuous action spaces, which achieves a near-optimal regret bound under decaying corruption.

### Strengths
- Learning from imperfect human feedback with decaying imperfection is interesting and well-motivated. This provides a valuable extension to traditional dueling bandit frameworks by allowing for dynamic, time-decaying corruption in feedback, aligning well with real-world applications.
- The authors establish a rigorous lower bound for regret under imperfect feedback, showing that even when corruption decays, learning remains as challenging as in adversarially corrupted settings. This theoretical insight highlights the inherent challenges in dealing with imperfect human feedback.
- RoSMID is a novel algorithm that achieves a near-optimal regret bound in the presence of decaying corruption. The algorithm's reliance on gradient-based techniques in the dueling bandit setting is an innovative approach that bridges traditional bandit learning with more advanced optimization methods.
- Extensive experiments validate the theoretical bounds. These experiments also underscore the efficiency-robustness trade-off, an important consideration for the settings where feedback corruption is inevitable.

### Weaknesses
- RoSMID requires prior knowledge of the decay rate $\rho$ of the human feedback, which may be an unrealistic assumption in practical settings. A discussion of this limitation and insights into possible relaxations would be beneficial.
- Though the theoretical result is new, it is not entirely surprising given the similarities between decaying corruption and adversarial corruption. A more detailed discussion on the difference between decaying corruption and adversarial corruption and how it informs the design of RoSMID would enhance the paper's contribution.
- While RoSMID's theoretical performance is strong, the empirical evaluation could be more comprehensive. For example, it would be useful for the authors to include insights into RoSMID's computational efficiency relative to other methods and its scalability with increasing problem size.

### Questions
- Could RoSMID use an upper bound on $\rho$ if the exact decay rate is unknown? How would this affect the performance of RoSMID?
- Is the continuous action space setting inherently more challenging to address than discrete action spaces under imperfect feedback? Could the proposed algorithm be adapted to discrete action spaces?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the problem of dueling bandits with continuous actions when the outcomes are corrupted. In particular, the authors assume that the comparison at time is perturbed by an amount $c_t(a_t, a_t')$ which is $t^{\rho - 1}$. First, the authors demonstrate a lower bound of $\Omega(\max( \sqrt{T}, T^\rho ) )$ on the regret even when $\rho$ is known. The authors complement this result by proving a matching upper bound by robustifying stochastic mirror descent algorithms proposed in prior work on dueling bandits literature. Furthermore, they also study the setting where the total amount of corruption is bounded i.e. $\sum_t c_t(a_t, a_t') \le T^\rho$ and need not follow the decaying pattern. For this setting, robust versions of stochastic mirror descent and bandit gradient descent are studied.

### Strengths
1. The proposed algorithm (robust version of stochastic mirror descent) is simple and elegantly generalizes prior algorithm proposed in the literature on dueling bandits.

2. The paper is well-written and the authors have clearly explained the main difficulties in proving the regret upper bound. Moreover, it is also interesting to see that the regret lower and upper bounds match (up to logarithmic factors).

### Weaknesses
1. The model of corruption assumes that the amount of corruption decays over time. Therefore, when $t$ is large, the corruption encountered by the algorithm is quite small. I believe this is a strong assumption and severely restricts the adversary. Although the authors have results when the total amount of corruption is bounded (proposition 2 and 3) the bounds are suboptimal in such a setting.

2. Although the authors have motivated the use of decaying corruption, I believe human labelers often display persistence bias and therefore it is not clear that the corruption will decay to zero over time. Perhaps another setting could be that the feedbacks display some non-negligible bias over time.

3. The authors proposed several assumptions on the link function. For example, the derivatives are bounded (both below and above). This is not satisfied for many link functions e.g. logistic functions. Furthermore, the setting assumes that the utility function is strongly concave which is also a strong assumption.

4. The result is subsection 4.1 is presented as efficiency-robustness tradeoff in learning from imperfect feedback. However, the results are specific to two learning algorithms and it is not clear whether such a trade-off exists for any learning algorithm. In particular, can the authors provide an algorithm independent lower bound for this setting?

### Questions
1. Can you generalize your results to consider the case when the link function might have an unbounded derivative or non-smooth? This covers a large class of link functions. 

2. What is the regret upper bound when the utility function is just concave (and not strongly concave)?

3. Finally, can you explain the weaker regret upper bounds for the setting of bounded corruption (subsection 4.1)? In particular, does there exist some algorithm that provides a matching bound (as stated in theorem 3.1), or is the lower bound even stronger in this case?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studied the dueling bandit problem with constraint corruption. It first proved that the problem is not easier with the constraint on the corruption with a lower bound. Next, it showed the efficiency of the proposed RoSMID algorithm with theoretical upper bound and numerical experiments. It also discussed the application of its analytical techniques.

==========

After rebuttal: I would like to keep the score.

### Strengths
1. The paper is generally easy to follow.
1. The proposed RoSMID algorithm is nearly optimal when the gap between its upper bound and the lower bound is only $\log T$.

### Weaknesses
1. The paper repeatedly emphasize the novelty of the analysis. It would be appreciated to summarize the novelty in a few words.
1. Lemma 3.1: I am confused why the author(s) discussed the direct reward feedback setting here.

### Questions
Please refer to the 'Weaknesses' section.

### Soundness
3

### Presentation
3

### Contribution
3
