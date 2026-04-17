# Predictive CVaR Q-learning

- Decision: Accept (Poster)
- Scores: 2, 8, 6, 6, 4

## Abstract
We propose a sample-efficient Q-learning algorithm for reinforcement learning with the Conditional Value-at-Risk (CVaR) objective. Our algorithm is built upon predictive tail value function, a novel formulation of risk-sensitive action value, that admits a recursive structure as in the conventional risk-neutral Bellman equation. This structure enables the Q-learning algorithm to utilize the entire set of sample trajectories rather than relying only on worst-case outcomes, enhancing the sample efficiency. We further derive a Bellman optimality equation and a policy improvement theorem, which provide theoretical foundations of our algorithm and remedy inconsistencies that have existed in the literature. Empirical results demonstrate that our method consistently improves CVaR performance while maintaining stable and interpretable learning dynamics.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose a novel CVaR recursive structure based on predictive tail value functions and predictive tail probability functions. This recursive structure, along with a risk-sensitivity exploration strategy, leads to superior empirical performance when compared to the usual CVaR recursive structure.

### Strengths
The proposed CVaR recursive structure based on predictive tail value functions and predictive tail probability functions is innovative and clever. The theoretical work done to support this recursive structure is sound (albeit with some typos; see weaknesses). The empirical results are encouraging.

### Weaknesses
Overall, I would consider this paper to be sound work. However, I do have some concerns that would need to be addressed prior to publication. 

My biggest concern is that the work performed does not validate the authors’ claims. In particular, the authors mention numerous times that the primary benefit of their proposed method is sample efficiency, yet there is no theoretical work done in support of this claim, and the empirical results do not provide any evidence of such claims. The closest measure of sample efficiency that can be made from the empirical results is in Figures 2/3b), however in Figure 2b), none of the methods converge to the same solution so a proper evaluation of sample efficiency cannot be made. In Figure 3b), there is no statistical difference between the methods. 

Moreover, I am not convinced (i.e., it was not rigorously argued by the authors) that value-based CVaR methods have the same sample efficiency issues that policy-gradient methods do. In particular, the claim that value-based methods lead to significant sample inefficiency in lines 165-166 is not proved by the authors, nor do they provide a citation to back up these claims. In fact, a key piece of theoretical work that would greatly enhance the paper is some result that shows why the regular CVaR decomposition (Equation 2) is sample inefficient.

Another concern is that the empirical analysis lacks focus. In particular, in addition to the concerns mentioned above, the use of function approximation for the simple experiments considered in this paper seems unnecessary. I would argue that if the authors want to include the work related to function approximation, they need to provide a compelling experiment that makes proper use of it. Furthermore, it seems odd that for the simple experiments included in the paper, that neither method can find the optimal solution. 

It is also not clear whether the shown gain in performance is entirely due to the risk-sensitivity exploration strategy. In particular, was the same exploration strategy used with the baseline algorithm as well? If so, then why even include it in the first place when making the comparison (i.e., would it not be a cleaner comparison without the exploration strategy)? If not, then this is not a proper evaluation of the proposed algorithm's performance, and an ablation study would be needed.

Overall, although I see a lot of merit in the work performed by the authors, the current draft of the paper makes it seem like a ‘forced’ adaptation of prior work done in the policy-gradient domain into the value-based domain, rather than a purposeful, adequately-motivated endeavour. 

Accordingly, in order to increase my score, the authors would need to: 1a) provide theoretical and/or empirical results that support their claims related to sample efficiency, or 1b) remove the claims of sample efficiency and find a more compelling narrative for the paper. 2) The authors would also need to address my concerns related to the empirical analysis.

**Minor Comments:**
- The introduction is unfocused and hard to read. In particular, the constant switching between policy-gradient and value-based methods is hard to follow. Overall, I do not see a reason to mention policy-gradient methods at all in this paper. 
- Line 31: I would argue that CVaR has a lot of tractability issues (which is why it is such a difficult objective to optimize) and that the primary reason that it is valued is because it is a coherent risk measure.
- Lines 46-51: The discussion in this paragraph completely ignores the notion of dynamic risk measures, which would need to be mentioned to make a proper argument.
- Section 2 would greatly benefit by having more citations related to the methods that the authors are building upon.
- Appendix C needs equation numbers.
- Lemma 2 is filled with several copy/paste errors from Lemma 1 (e.g. c) is not needed in this proof)

### Questions
Lines 183-184: can the authors expand on why the choice of $\eta$ is arbitrary? This seems counterintuitive to me.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper addresses the challenges in risk-sensitive reinforcement learning (RL) using the Conditional Value-at-Risk (CVaR) objective, which focuses on optimizing the expected return in the worst-case quantile of the return distribution (e.g., for safety-critical applications like autonomous driving or finance). Standard CVaR RL methods are sample-inefficient due to two key issues: (1) noisy policy evaluation from treating CVaR as a non-decomposable, terminal objective, which delays learning signals and hinders temporal credit assignment; and (2) "blindness to success," where the agent ignores high-return trajectories outside the risk tail, leading to premature convergence to overly conservative, suboptimal policies.

### Strengths
1. The paper proposed CVaR recursive structure based on predictive tail value functions and predictive tail probability functions is innovative and clever. 
2. The theoretical work done to support this recursive structure is sound (albeit with some typos; see weaknesses). 
3. The empirical results are encouraging.

### Weaknesses
Overall I would consider this paper to be sound work. However, I do have some concerns that would need to be addressed prior to publication. 

My biggest concern is that the work performed does not validate the authors’ claims. In particular, the authors mention numerous times that the primary benefit of their proposed method is sample efficiency, yet there is no theoretical work done in support of this claim, and the empirical results provided do not provide any evidence of such claims. The closest measure of sample efficiency that can be made from the empirical results is in Figures 2-(b) and  3-(b), however in Figure 2-(b), none of the methods converge to the same solution so a proper evaluation of sample efficiency cannot be made. In Figure 3-(b), there is no statistical difference between the methods. 

Moreover, I am not convinced (i.e., it was not rigorously argued by the authors) that value-based CVaR methods have sample efficiency issues in the same way that policy gradient methods do. In particular, the claim that value-based methods lead to significant sample inefficiency in lines 165-166 is not proved by the authors, nor do they provide a citation to back up these claims. In fact, a key piece of theoretical work that would greatly enhance the paper is some results that show why the regular CVaR decomposition (Equation 2) is sample inefficient.

Another concern is that the empirical analysis lacks focus. In particular, in addition to the concerns mentioned above, the use of function approximation for the simple experiments considered in this paper seems unnecessary. I would argue that if the authors want to include the work related to function approximation, they need to provide a compelling experiment that makes proper use of it. Furthermore, it seems odd that for the simple experiments included in the paper, neither method can find the optimal solution. 

It is also not clear whether the shown gain in performance is due to the risk-sensitivity exploration strategy. In particular, was the same exploration strategy used with the baseline algorithm as well? If so, then why even include it in the first place when making the comparison (i.e., would it not be a cleaner comparison without the exploration strategy)? If not, then this is not a proper evaluation of the proposed algorithm's performance.

Overall, although I see a lot of merit in the work performed by the authors, the current draft of the paper makes it seem like a ‘forced’ adaptation of prior work done in the policy-gradient domain into the value-based domain, rather than a purposeful, adequately-motivated endeavor. 

Accordingly, in order to increase my score, the authors would need to: 1a) provide theoretical and/or empirical results that support their claims related to sample efficiency, or 1b) remove the claims of sample efficiency and find a more compelling narrative for the paper. The authors would also need to address my concerns related to the empirical analysis.

### Questions
Lines 183-184: can the authors expand on why the choice of $\eta$ is arbitrary? This seems counterintuitive to me.

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
This paper proposes a sample-efficient Q-learning algorithm (PCVaR-Q) to optimize the Conditional Value-at-Risk (CVaR) target. Its core contribution is two key innovations: First, the "predictive tail value function" is proposed, which constructs a novel recursive structure suitable for CVaR targets, similar to traditional risks. The neutral Bellman equation aims to solve the problem of noise strategy evaluation caused by the indecomposition of the target. Second, introduce a "two-way exploration" strategy, which explores the risk sensitivity of action space and intelligent bodies at the same time, so as to alleviate the "blindness to success" phenomenon.

### Strengths
1. The core innovations lie in the proposal of the predictive tail value function $f^\chi$ and the predictive tail probability function $g^\chi$ for policy kenel $\chi$. Based on the newly defined tail functions, Theorem 1 & 2 gives the Bellman equation and the Bellman optimality equation. 
2. Combined with the newly developed tail value function and probability function, the proposed "two-way randomized exploration" approach explicitly solves the known problem of "blindness to success" in CVaR learning, and encourages intelligent bodies to explore strategies with different risk preferences. This is achieved by samplingys around the risk budget $\eta$.
3. The experimental results (Figures 2 and 3) strongly support the author's argument. Compared with the CVaR-Q baseline, PCVaR-Q shows higher stability and lower variance during the training process.

### Weaknesses
1. The entire theoretical framework (especially Theorem 1 & 2) depends on Assumption 1, which states that the distribution of the residual return $R_{t:T}$ has no probability mass. However, many standard reinforcement learning environments (including discrete rewards or deterministic rewards) would violate this assumption. Moreover, I found that the experiements environment considered (such as the. Sequential decision tree setting) clearly violates Assumption 1. The author did not discuss the impact of the violation of thies assumption.
2. TThe pre-training step is crucial for success, as all models that did not undergo pre-training (Figure 4a-d) failed to learn the optimal path. This seems to weaken the argument that the algorithm has a robust exploration strategy and "sample-efficient," as exploration seems to fail in cold-start case.
3. Currently, the experiments in the main text (Section 5) have failed to clearly disentangle the contributions of the three main contributions—(1) the new Bellman equation, (2) pre-training, and (3) two-way random exploration—to the performance improvement.

### Questions
1. For the augmented state $(s, y, a)$, how is the "table-based function approximator" implemented? Considering that $y$ is a continuous variable, is discretization based on the grid $H$ used?
2. Without using pre-training, to what extent can your proposed new Bellman equation (Eq. 5) and bidirectional exploration itself solve the issues of "blindness" and learning instability?


---
Typos

1. Line 369, "through risk-nuetral Q-learning.": nuetral -> neutral.
1. Line 441, "experimental environment and and the distinct policies": there are two "and" here.
2. Line 472, "a novel novel CVaR Q-learning framework": double "novel".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Predictive CVaR Q-Learning (PCVaR-Q), which makes CVaR optimization TD-learnable by defining predictive tail value and predictive tail probability functions that satisfy a new Bellman recursion. This enables step-wise CVaR learning and supports a proven policy-improvement guarantee.

### Strengths
- Clear theoretical innovation: a Bellman-consistent, value-based formulation for CVaR.
- Strong proofs and solid connection to policy improvement.
- Well-written and conceptually clear.

### Weaknesses
- Limited empirical scope: Experiments are small-scale and tabular; results demonstrate feasibility but not scalability.
- Comparison gaps: The paper could benchmark against more recent risk-sensitive or distributional RL algorithms (e.g., D4PG, IQN with tail weighting).
- Exploration heuristic: The “risk-level exploration” scheme is sensible but empirically underexplored.

### Questions
1. How sensitive is learning stability to the sampling of risk levels during exploration?
2. Can the predictive-tail recursion extend naturally to actor–critic or deep function-approximation settings?
3. Does the algorithm handle non-stationary return distributions or stochastic environments robustly?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new method for RL with a Conditional Value-at-Risk objective. To address the challenges posed by nonlinearity and non-decomposability, the authors introduce predictive value/probability functions and develop a new RL algorithm based on them.

### Strengths
The motivation and mathematical development are clearly presented. While the derivations seem fairly standard, they provide a clear motivation for the proposed algorithm.

### Weaknesses
The paper's contribution requires further justification.

1. After formulating objective (1), one could straightforwardly apply an actor–critic approach to optimize $ \mathbb{E}^{\chi,\eta}[-(\eta - R_{1:T})^{+}]$ (e.g., REINFORCE). It is not evident that the proposed method is superior to such actor-critic methods. While it may be true that only trajectories with non-zero effective reward are informative for actor-critic, the same limitation appears to affect the proposed algorithm: since g models the tail probability, when only a small subset of trajectories has non-zero effective reward, the estimate of g is likely to be noisy.

2. The experiments are limited to toy settings and do not include comparisons against actor-critic methods or stronger baselines. A more comprehensive comparison would also address the weakness noted above.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2
