# Q-learning with Posterior Sampling

- Decision: Accept (Poster)
- Scores: 2, 6, 8, 2

## Abstract
Bayesian posterior sampling techniques have demonstrated superior empirical performance in many exploration-exploitation settings. However, their theoretical analysis remains a challenge, especially in complex settings like reinforcement learning.
In this paper, we introduce Q-Learning with Posterior Sampling (PSQL), a simple Q-learning-based algorithm that uses Gaussian posteriors on Q-values for exploration, akin to the popular Thompson Sampling algorithm in the multi-armed bandit setting. We show that in the tabular episodic MDP setting, PSQL achieves a regret bound of $\tilde O(H^2\sqrt{SAT})$, closely matching the known lower bound of $\Omega(H\sqrt{SAT})$. Here, S, A denote the number of states and actions in the underlying Markov Decision Process (MDP), and $T=KH$ with $K$ being the number of episodes and $H$ being the planning horizon. Our work provides several new technical insights into the core challenges in combining posterior sampling with dynamic programming and TD-learning-based RL algorithms, along with novel ideas for resolving those difficulties. We hope this will form a starting point for analyzing this efficient and important algorithmic technique in even more complex RL settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Posterior Sampling Q-Learning (PSQL), a tabular method that maintains Gaussian posteriors over Q-values and uses posterior samples for exploration; to obtain a $\tilde{O}(H^{2}SAT^{1/2})$ regret bound, the analyzed variant makes the bootstrapped target optimistic by taking the maximum over multiple posterior draws for the next-state value. While the vanilla single-sample version exhibits strong empirical performance, the optimistic variant required for the theory is not competitive, lagging behind RLSVI and offering no gains over Staged-RandQL on the authors’ own benchmarks.

### Strengths
- Provides a near-optimal $\tilde{O}(H^{2}SAT^{1/2})$ regret analysis for a posterior-sampling variant of tabular Q-learning.
- The vanilla practical instantiation consistently outperforms classical baselines such as UCB-QL and Staged-RandQL on the reported tasks, showing promise for posterior-sampling driven exploration in tabular settings.

### Weaknesses
- The empirically evaluated “vanilla” algorithm differs from the theoretically analyzed optimistic variant; the latter is not competitive with RLSVI or Staged-RandQL on the reported benchmarks, leaving the practical relevance of the theory unclear. This mismatch makes it difficult to understand the paper’s positioning: the theoretically grounded algorithm underperforms in practice and does not improve existing regret bounds, yet the practical algorithm—which shows promise—is not benchmarked against other practical exploration baselines on more demanding tasks.
- Experimental coverage is very limited (two toy tabular environments) and omits stronger practical baselines (e.g., recent posterior-sampling or optimistic methods); if the practical implementation departs from the analyzed algorithm, broader comparisons to other exploration strategies become essential, but they are absent here.
- Presentation can be clarified: the introduction currently reads as an unstructured list of related work rather than positioning the contribution, and some statements about model-based methods could be refined.
- The “first Bayesian posterior” claim is confusing: several prior works already use posterior-sampling within Q-learning, so the novelty of this phrasing is unclear and should be clarified.

### Questions
- Given that the analyzed optimistic variant performs worse than RLSVI and Staged-RandQL, how do the authors reconcile the theoretical guarantees with practical usefulness? Can they provide intuition or evidence that the analyzed algorithm offers benefits beyond these baselines?
- If the practical PSQL* deviates from the analyzed variant, do the authors plan to benchmark it against more practical exploration algorithms such as [1] on more challenging domains (e.g., Atari, Maze2D) to establish empirical competitiveness?
- In lines 118–119 the paper says model-based methods directly model the rewards and transitions “instead of” the implied value function or policies; could the authors clarify this phrasing? In practice, model-based RL often uses learned models to improve the value function and/or policy.
- What exactly are the authors claiming as “first” in the first contribution point, “provided by the Bayesian posterior”? Could they clarify how their notion of “Bayesian posterior” differs from earlier posterior-based Q-learning efforts (e.g., [1]) and what concrete novelty they intend to highlight?

[1] Ishfaq et al., Provable and Practical: Efficient Exploration in Reinforcement Learning via Langevin Monte Carlo, ICLR 2024.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors proposed a model-free exploration method based on introducing the idea of Randomized Least-Squares Value Iteration (RLSVI) into the Q-learning framework. This algorithm, called Posterior Sampling Q-learning (PSQL), uses a Gaussian posterior estimate of Q-values and achieves a regret guarantee of $\tilde{O}(H^2 \sqrt{SAT})$, which is the same as Q-learning with UCB bonuses or Staged Randomized Q-learning. Additionally, the algorithm shows strong empirical performance on standard low-dimensional benchmarks.

### Strengths
- Interesting alternative explanation of the UCB-Q-learning learning rate, that appears from the additional entropy regularization in the variational approximation, with a clear intuition of "collapse avoidance" with entropy due to bias in the estimate;
- Strong empirical performance as well as theoretical regret guarantees;

### Weaknesses
- Lack of empirical comparison with a usual RandQL. Although this method does not offer the same rigorous guarantees as its staged version, it would be interesting to compare PSQL* and a usual RandQL without stages.
- The regret bound does not match the regret bound of a variance-reduced version of Q-learning (Li et al. 2021);

### Questions
- It would be beneficial to discuss an RLSVI-style model-based algorithm that achieves the minimax optimal regret guarantee (Xiong et al. 2022).
- What prevents you from using a standard RSLVI analysis with a single sample there?
- What prevents you from extending the sketch of the proof in Appendix F to a complete proof for a variance-shaped noise? What is a main challenge there?
- Is it possible to provide a deep-learning version of your method?

### References

Xiong, Z., Shen, R., Cui, Q., Fazel, M., & Du, S. S. (2022). Near-optimal randomized exploration for tabular Markov decision processes. Advances in neural information processing systems, 35, 6358-6371.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents Q-Learning with Posterior Sampling (PSQL), a model-free reinforcement learning algorithm that employs Gaussian posteriors on Q-values for exploration. By interpreting Q-learning as a Bayesian inference problem with a regularized ELBO objective, the authors design a conceptually simple algorithm that combines single-sample posterior sampling for action selection with multiple-sample optimism for target computation. This design allows them to prove a near-optimal regret bound of O(H^2 \sqrt(SAT)), while preliminary experiments show competitive or superior empirical performance compared to UCBQL, RLSVI, and Staged-RandQL.

### Strengths
The work is theoretically grounded, algorithmically simple, and provides new insights into the Bayesian interpretation of Q-learning. The regret guarantee is strong, and the analysis tackles key challenges in combining posterior sampling with TD learning.

### Weaknesses
Using Gaussian posteriors on Q-values may destroy important structural properties of Q-functions (e.g., boundedness or Bellman consistency), since Gaussian distributions are unbounded. The choice of posterior variance is subtle and strongly affects performance, requiring careful tuning. Moreover, the use of multiple posterior samples for target computation increases the algorithm’s computational complexity, and the theoretically unanalyzed single-sample variant (PSQL*) outperforms the analyzed one in practice, indicating a gap between theory and implementation.

### Questions
1. Can alternative posterior distributions preserve Q-value structure more faithfully than Gaussians?

2. Is there a principled or adaptive way to select the posterior variance to avoid heuristic tuning?

3. Can the multiple-sampling step for optimism be replaced with a cheaper or more elegant alternative?

4. How does the approach scale to function approximation or deep RL settings?

5. How does posterior optimism interact with TD bootstrapping bias in long horizons?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce PSQL, a novel model-free method that utilizes Gaussian Bayesian inference on Q-values. This approach incorporates a different target value based on the optimism principle, though this optimistic target value does not influence the actual decision-making process. The paper provides regret bounds that are nearly optimal and comparable to those achieved by other posterior sampling methods. Additionally, the authors establish a modified version, PSQL, which uses the target value chosen as in standard Q-learning. The results demonstrate that PSQL outperforms several baseline methods in tabular environments.

### Strengths
Authors discuss the limitations of the analysis of the vanilla PSQL algorithm

### Weaknesses
- In my opinion, the empirical results are not sufficiently extensive. It would be interesting, for example, to consider a comparison with the PSRL algorithm, which was shown to outperform Staged-RandQL in a recent study (Tiapkin et al., 2023). Furthermore, the paper lacks a comparison in more complex environments, specifically those with a continuous state space;
- Another interesting direction would be to extend this algorithm to more practical scenarios with a general state space. If this is possible, what quantity should be chosen for the variance in that setting?
- What regret bound can be achieved in the "vanilla version" of the PSQL algorithm? Could a non-trivial polynomial bound be established
- It appears there is a potential issue with the inequality  $\mathbb{E}[\tilde{X}^{\text{alt}}| \bar{\mathcal{O}}^{\text{alt}}] \leq \underline{X}$ in Lemma E.1. This is because the statement assumes $\mathbb{E}[\tilde{X}]\geq \underline{X}$, but the inequality seems to require the opposite for the proof to hold.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
