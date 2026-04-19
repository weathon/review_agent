# Provable and Practical: Efficient Exploration in Reinforcement Learning via Langevin Monte Carlo

- Decision: Accept (poster)
- Scores: 8, 6, 6

## Abstract
We present a scalable and effective exploration strategy based on Thompson sampling for reinforcement learning (RL). One of the key shortcomings of  existing Thompson sampling algorithms is the need to perform a Gaussian approximation of the posterior distribution, which is not a good surrogate in most practical settings. We instead directly sample the Q function from its posterior distribution, by using  Langevin Monte Carlo, an efficient type of Markov Chain Monte Carlo (MCMC) method. Our method only needs to perform noisy gradient descent updates to learn the exact posterior distribution of the Q function, which makes our approach easy to deploy in deep RL.  We provide a rigorous theoretical analysis for the proposed method and demonstrate that, in the linear Markov decision process (linear MDP) setting, it has a regret bound of $\tilde{O}(d^{3/2}H^{3/2}\sqrt{T})$, where $d$ is the dimension of the feature mapping, $H$ is the planning horizon, and $T$ is the total number of steps. We apply this approach to deep RL, by using Adam optimizer to perform gradient updates. Our approach achieves better or similar results compared with state-of-the-art deep RL algorithms on several challenging exploration tasks from the Atari57 suite.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper is about online RL algorithm design and theoretical analysis. Different from most previous RL theory works which lack deep RL demonstrations, this work proposes a both practical (scalable to deep RL domains) and provably efficient online RL algorithm (LMC-LSVI) based on the celebrated Langevin Monte Carlo algorithm. Theoretically, it proves that with linear function approximation LMC-LSVI achieves a $\widetilde{\mathcal{O}}(d^{3/2}H^{5/2}T^{1/2})$-online regret. On the practical side, LMC-LSVI is further extended to the Adam LMCDQN algorithm which performs similarly or even better than SOTA explorative deep RL algorithms in some challenging RL domains.

### Strengths
1. Bridging the gap between RL theory and practice is of great importance to the advance of RL research. This work gives a possible and positive answer to this question in the specific setting of online RL where exploration-exploitation balance is a key problem. 
2. The proposed  Langevin Monte Carlo Least-Squares Value Iteration (LMC-LSVI) algorithm turns out to have a quite clean form which simply adds a noise term to the gradient descent update of the Bellman error (Line 9 of Algorithm 1) to incentivize exploration. This advantage thus allows for a deep RL extension where Adam-based adaptive SGLD is further applied.
3. The proposed algorithm enjoys theoretical guarantees (in the linear function approximation setting) which is missing in most previous deep RL exploration methods even for LFAs.

### Weaknesses
The rate of the online regret in linear function approximation setting is far from tight compared with known lower bounds. But from my view this is understandable given that a new approach is derived whose practicability is of higher importance.

### Questions
1. Regarding the theoretical analysis (Theorem 4.2), I am curious why the failure probability $\delta$ must be larger than a certain quantity, say $1/(2\sqrt{2e\pi})$? Is this inevitable for a sampling-stype algorithm and analysis? This will narrow the applicability of the theory since for frequentist regret analysis we always hope that the regret bound can hold for arbitrarily small fail probability.
2. The authors say in the contribution part that "unlike any other provably efficient algorithms for linear MDPs, it can easily be extended to deep RL settings",..., "such unification of theory and practice is unique in the current literature of both theoretical RL and deep RL", which to my knowledge is overclaimed. Even though the perspective of Langevin dynamic are less explored in this regime, which is done by this paper, there do exist other works trying to achieve sample efficiency while being compatible with practical deep RL methods, e.g., [1, 2, 3]. So it seems improper to describe the work as unique given this line of research.

**References:**

[1] Feng, Fei, et al. "Provably Correct Optimization and Exploration with Non-linear Policies." *International Conference on Machine Learning*. PMLR, 2021.

[2] Kitamura, Toshinori, et al. "Regularization and Variance-Weighted Regression Achieves Minimax Optimality in Linear MDPs: Theory and Practice." *International Conference on Machine Learning*. PMLR, 2023.

[3] Liu, Zhihan, et al. "One Objective to Rule Them All: A Maximization Objective Fusing Estimation and Planning for Exploration." *arXiv preprint arXiv:2305.18258* (2023).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
They have introduced online RL algorithms incorporating Langevin Monte Carlo. 

   *They have established the regret bound for Q-learning with Langevin Monte Carlo under linear MDPs. 

   *They have adapted the aforementioned algorithm to use the ADAM optimizer, demonstrating its favorable empirical performance through experiments.

### Strengths
While the literature on Bayesian model-free algorithms is extensive, it is indeed lacking in works that offer both robust theoretical guarantees and practical viability. In my view, this paper effectively bridges this gap. 

* The paper presents a good comparison, highlighting noteworthy contributions compared to existing works.

* Section 3 offers a non-trivial and novel analysis that significantly enhances the paper.

* The practical version's experimental results in Section 5 demonstrate promising performance.

### Weaknesses
* It looks Algorithms they use in practice (Algorithm 2) are not analyzed. So, this algorithm is "practical", but I am not sure it is fair to say this is "provable." 

* Some aspects of the comparisons appear to lack complete fairness.

   * In Table 1, it is unclear what precisely the author intends to convey with "computational traceability" and "scalability." These terms should be defined more formally in the caption of the table. 

   * Furthermore, it may not be entirely fair to say that OPT-RLSVI and LSVI-PHE lack scalability in Table 1. While I acknolwedge that the proposed algorithms (OPT-RLSVI and LSVI-PHE) may not perform well in practice, one could argue that they could exhibit scability with the incorporation of certain simple heuristics, even if formal guarantees are absent. Indeed, the LSVI-PHE paper includes moderate experiments. The author has similarly extended theoretically justified algorithms to practical versions without formal backing as they did. So, in this sense, I am not sure why it is reasonable to say the author's algorithm is scalable, but their algorithms are not scalable.

### Questions
I raised several concerns for the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies usefulness of LMC algorithms for MDPs. It first shows an LMC algorithm deployed with LSVI for linear MDPs and upper bounds the corresponding regret. It also proposes versions of LMC used with Adam and DQN for practical purposes. The performance of the practical algorithm is illustrated on multiple atari games.

### Strengths
1. The paper proposes a regret upper bound for applying LMC with LSVI for linear MDPs.
2. The paper proposes a practical version of LMC to be applied with DQN. It works in practice as shown through multiple experiments.

### Weaknesses
1. The theoretical analysis is shown for linear MDPs and the practical algorithm is applied with DQN. Thus, the theory and applications serve two different purposes and does not compliment each other. It makes the paper look like an assortment of results for different settings than a coherent study.
2. If the aim is to design a practical algorithm, why analysing it for linear MDPs, which is known to be unfit for practice (the version stated in the paper) [1]? Why not analysing it for more practical models like [1], [2], [3]?
3. The regret bound for LMC lsvi is loose in terms of both d and H. Why is it so? Is it due to any fundamental hardness in analysis or inherent to LMC algorithms or just a shortcoming of the analysis done in the paper? Can you explain this?
4. The practical version is claimed to be better than the existing LMC algorithm for RL cause the proposal has theoretical guarantees and also employs better practical techniques. The first is not completely valid as the setting plus algorithm for analysis and practice are really different. The second is also not clear as the experimental results leave the question of performance improvement statistically inconclusive . Can you provide a reason where adam LMC DQN would work better than langevin DQN and where it would be worse?

[1] Zhang, Tianjun, Tongzheng Ren, Mengjiao Yang, Joseph Gonzalez, Dale Schuurmans, and Bo Dai. "Making linear mdps practical via contrastive representation learning." In International Conference on Machine Learning, pp. 26447-26466. PMLR, 2022. 
[2] Ouhamma, Reda, Debabrota Basu, and Odalric Maillard. "Bilinear exponential family of MDPs: frequentist regret bound with tractable exploration & planning." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 8, pp. 9336-9344. 2023.
[3] Weisz, Gellért, András György, and Csaba Szepesvári. "Online RL in Linearly $ q^\pi $-Realizable MDPs Is as Easy as in Linear MDPs If You Learn What to Ignore." arXiv preprint arXiv:2310.07811 (2023).

### Questions
Please check the weaknesses for questions .

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good
