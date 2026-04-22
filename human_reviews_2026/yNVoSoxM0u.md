# Efficient Adversarial Attacks on High-dimensional Offline Bandits

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 6, 4, 2

## Abstract
Bandit algorithms have recently emerged as a powerful tool for evaluating machine learning models, including generative image models and large language models, by efficiently identifying top-performing candidates without exhaustive comparisons. These methods typically rely on a reward model---often distributed with public weights on platforms such as Hugging Face---to provide feedback to the bandit. While online evaluation is expensive and requires repeated trials, offline evaluation with logged data has become an attractive alternative. However, the adversarial robustness of offline bandit evaluation remains largely unexplored, particularly when an attacker perturbs the reward model (rather than the training data) prior to bandit training. In this work, we fill this gap by investigating, both theoretically and empirically, the vulnerability of offline bandit training to adversarial manipulations of the reward model. We introduce a novel threat model in which an attacker exploits offline data in high-dimensional settings to hijack the bandit's behavior. Starting with linear reward functions and extending to nonlinear models such as ReLU neural networks, we study attacks on two Hugging Face evaluators used for generative model assessment: one measuring aesthetic quality and the other assessing compositional alignment. Our results show that even small, imperceptible perturbations to the reward model’s weights can drastically alter the bandit's behavior. From a theoretical perspective, we prove a striking high-dimensional effect: as input dimensionality increases, the perturbation norm required for a successful attack decreases, making modern applications such as image evaluation especially vulnerable. 
Extensive experiments confirm that naive random perturbations are ineffective, whereas carefully targeted perturbations achieve near-perfect attack success rates. 
To address computational challenges, we design efficient heuristics that preserve almost 100\% success while dramatically reducing attack cost. In parallel, we propose a practical defense mechanism that partially mitigates such attacks, paving the way for safer offline bandit evaluation. Finally, we validate our findings on the UCB bandit and provide theoretical evidence that adversaries can delay optimal arm selection proportionally to the input dimension. Code is available at the anonymous repository: [https://anonymous.4open.science/r/offline-bandit](https://anonymous.4open.science/r/offline-bandit).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work represents a breakthrough contribution to the field of adversarial attacks on bandit algorithms, with profound theoretical, methodological, and practical significance. By systematically revealing the extreme vulnerability of offline multi-armed bandit algorithms to reward model perturbations in high-dimensional scenarios, the authors construct a rigorous and cohesive research framework integrating theory, algorithms, and experiments. The novel threat model, clear theoretical characterization of high-dimensional vulnerability, efficient attack strategies, and comprehensive validation on real-world reward models address a critical gap in the community—especially amid the growing reliance of generative model evaluation on bandit methods. However, the work faces non-trivial limitations in threat model assumptions, defense robustness, experimental depth, theoretical generalization, and ethical discussion.

### Strengths
Offline bandits have been widely adopted in generative model evaluation, but existing adversarial research mostly focuses on online settings or direct reward tampering. This paper pioneers the systematic study of a new threat model—"attackers only perturb pre-trained reward model weights"—filling a critical gap in the field. The problem definition is highly relevant to real-world evaluation scenarios, providing valuable insights for the community.

### Weaknesses
The paper assumes the attacker has full access to the same offline dataset as the victim, can modify the reward model weights before the victim trains the bandit, and the victim will use the tampered weights for evaluation. In practical evaluation scenarios, data and hyperparameters are often not fully disclosed to attackers, which limits the work’s practical applicability. The proposed data shuffling defense is only effective under the ideal setting where "the attacker fully knows the original sample order." If the attacker constructs perturbations based on statistical properties rather than sample order, the defense may fail. Additionally, there is no theoretical quantification of the utility-robustness tradeoff for the defense. Theorem 3.4 assumes the data distribution is a product measure with identity covariance matrix, and the mean of the optimal arm is orthogonal to that of the second-best arm. This idealized setting may not hold for high-dimensional natural images, limiting the theoretical results’ generalization.

### Questions
Extend the analysis to "partial information" (e.g., attacker has access to a subset of offline data) and "black-box reward model" scenarios, discussing the feasibility of attacks under these more realistic settings. Conduct sensitivity analysis on data sampling errors and model parameter uncertainty, quantifying how deviations from the ideal assumption affect attack performance. Relax the idealized assumptions in Theorem 3.4, extending the theoretical results to non-identity covariance matrices or correlated features.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studied offline adversarial attacks against linear UCB algorithm in high-dimensional space. The attacker has knowledge of the offline dataset associated with each arm, and has the ability to perturb the reward function. There are two scenarios considered in this paper - linear reward function and neural network-based reward. The paper studied 3 different attack goals. Theoretical results are derived to show attack feasibility. Interestingly, the authors showed that when the problem dimension is higher, the attack is easier to achieve. Experiments are done to demonstrate the effectiveness of the proposed attacks.

### Strengths
The paper extended traditional attacks on multi-armed bandits to high-dimensional space, and studied the problem from a different perspective. Instead of focusing on analyzing the attack cost, this paper targets analyzing the behavior of attack as the problem dimension grows. This is a new angle and may bring interesting topics to the community.

Both theoretical analysis and empirical study are performed the analyze the behavior of the proposed attacks, as the dimension of the bandit grows. An interesting observation is that the attack is easier as the dimension increases.

### Weaknesses
The problem setup is hard to justify in the following sense.

1. The attacker can perturb the reward function. This is too strong power. Traditional attacks only require attackers to perturb instantiated rewards, rather than the underlying reward mechanism. This is saying the attacker need to be able to change the underlying environment completely, which is too demanding.

2. Even if the bandit algorithm is forced to follow certain behaviors under attack, the data observed in each time step is still drawn from an pre-specified offline dataset. This is very weird setup, because attacker should have changed the learner's behavior, and the data should no longer be clean and fresh. This is a critical caveat in the problem setup.

3. The attack feasibility is hard to satisfy. The dimension must upper bound KT, which is a very strict constraint.

### Questions
Please help clarify my 3 concerns in the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies reward poisoning for stochastic multi-armed bandits under two reward function classes. It proposes three attack strategies and derives bounds on the maximum per-round corruption required to force errors against upper-confidence-bound style algorithms. The empirical section illustrates the practical impact of these attacks.

### Strengths
1. The problem setting is stated clearly, and the three corruption procedures are precisely specified.

### Weaknesses
1. The related work on corruption robust multi-armed bandits is not discussed at all, only a few mentioned in Appendix. I suggest the authors to include important works such as [1, 2, 3] and directly compare them by total amount of corruption ( $\mathbb{E}[ \sum_t  \delta^\top X_t] $).

2. The proposed algorithms is mostly examined against vanilla UCB, and briefly on greedy algorithms, both of which are not designed to be adversarially robust. What would be the minimum $\delta$ required against such robust MAB algorithms (e.g. [1, 2])?

3. Figure 1 is unclear. The x-axis uses “attack time (seconds)” while the rest of the paper uses discrete rounds. Either convert everything to rounds or explain the mapping between wall-clock time and rounds.

[1] Lykouris, T., Mirrokni, V. and Paes Leme, R., 2018. "Stochastic bandits robust to adversarial corruptions". In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing.
[2] Gupta, A., Koren, T. and Talwar, K., 2019. "Better algorithms for stochastic bandits with adversarial corruptions". In Conference on Learning Theory.
[3] Hajiesmaili, M., Talebi, M.S., Lui, J. and Wong, W.S., 2020. "Adversarial bandits with corruptions: Regret lower bound and no-regret algorithm". Advances in Neural Information Processing Systems.

### Questions
1. How does the proposed poisoning perform against robust variants of UCB such as [1]?
2. Theorem 3.4 appears to imply that as $d \rightarrow \infty$, the amount corruption needed goes to zero which is contradictory as at least $\Delta_{min}$(reward gap between best and second best arm) per round corruption is needed to induce suboptimal play for $T \ge \frac{K}{\Delta_{min}^2}$. Could you elaborate on how $d$, $\Delta_{min}$, and $T$ are scaled in the theorem to avoid this apparent discrepancy?

[1] Niss, L. &amp; Tewari, A.. (2020).  What You See May Not Be What You Get: UCB Bandit Algorithms Robust to $\varepsilon$-Contamination. Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence.

### Soundness
2

### Presentation
1

### Contribution
1
