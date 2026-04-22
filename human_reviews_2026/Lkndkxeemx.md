# A Near-Optimal Best-of-Both-Worlds Algorithm for Federated Bandits

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 8

## Abstract
This paper studies federated multi-armed bandit (MAB) problems in which multiple agents work together to solve a common MAB problem through a communication network. We focus on the heterogeneous setting in which no single agent can identify the globally best arm using only locally biased observations. In this setting, different agents may select the same arm at the same time step, but receive different rewards. We propose a novel algorithm called \textsc{FedFTRL} for this problem and, to our knowledge, it is the first to achieve near-optimal regret guarantees in both stochastic and adversarial environments. Notably, in the adversarial regime, our algorithm achieves $O(T^{\frac{1}{2}})$ regret, a significant improvement over the state-of-the-art regret of $O(T^{\frac{2}{3}})$ \citep{yi2023doubly}. We also provide empirical evaluations comparing our algorithm with baseline methods, demonstrating the effectiveness of our approach on both synthetic and real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies distributed multi-armed bandits with heterogeneous losses under both stochastic and adversarial regimes. It introduces FEDFTRL, a novel algorithm that is the first to achieve near-optimal regret in both settings. Comprehensive experiments are provided to validate the theoretical guarantees.

### Strengths
### Contributions
1. First establish an $O(\sqrt{T})$ regret bound in the adversarial regime.
2. Prove a best-of-both-worlds guarantee
3. Paper is well written.

### Weaknesses
1. Requires $O(K + VD)$ bits of communication per round; the protocol differs substantially from prior work (previous only required $O(K)$ ), so direct comparisons are not straightforward.
2. With an additional $O(VD)$ budget per round, the problem essentially reduces to a standard multi-armed bandit with delayed feedback.

### Questions
Please explain weaknesses, and I may significantly revise my review comments accordingly.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work addresses the federated multi armed bandit problem and focuses on the heterogenous setting where agents cannot determine the globally optimal arm using their local biased observations. The major contribution is the proposed Best-of-Both-Worlds algorithm that performs robustly in both stochastic and adversarial environments. The proposed algorithm adapts the Follow-the-Regularized-Leader framework and incorporate a hybrid regularizer typically used for bandits with delayed feedback. The authors view this as analogous to the latency caused by decentralized communication. Novelty includes the use of a communication scheme that tracks ddeviation records and a truncated loss estiator to keep agent action probabilities nearly aligned despite the heterogeneous feedback. They demonstrate that the FEDFTRL algorithm achieves near optimal regret bounds.

### Strengths
1. The core contribution of the paper is the first Best-of-Both-Worlds regret guarantee for the heterogeneou federated bandit setting.

2. Achieving $O(\sqrt{T})$ individual regret in adversarial setting is a clear improvement over previous results.

3. The theoretical analysis seems deep and the derived regret bounds match the known lower bounds.

4. Experiments are comprehensive.

### Weaknesses
1. The major limitation is the communication complexity which is explicitly mentioned by the authors as well. Each agent requires communicating $O(K+VD)$ bits of information every round. This is a potential practical bottleneck for large scale federated systems with many agents or high network diameter.

2. The algorithm relies on special hybrid regularizers and fine tuned time varying learning rates that are defined based on network characteristics. Even though they are necessary for theoretical guarantees, this complexity may hinder deployment and tuning in practice.

3. While the adaptation is novel for federated setting, the theoretical framework relies heavily on importing and combining existing BOBW literature.

### Questions
1. Given that the communication cost is $O(K + V D)$ bits per round for each agent, can youprovide a more detailed discussion on the practical implications of this dependency on the number of agents $V$ and the network diameter $D$ for typical federated systems? Quantifying how $V$ and $D$ affect runtime in the experiments would be beneficial.

2. The parameter $C_t^P$ in Eq. (2) quantifies the delay caused by decentralized communication and $C_T^P$ captures the dependence on network topology. Can the authors further clarify the intuitive meaning of how these complexity measures dictate the regret?

3. The truncated loss estimator $\tilde{\ell}_{v,t}$ is crucial for stabilizing action probabilities. Can the authors comment on the practical robustness of the truncation threshold $(12V C_t^P \gamma_t)$ of the denominator. Especially, regarding its potential sensitivity to misspecified initial parameters or dynamic changes in the network topology captured by $C_t^P$?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper study the Best-of-Both-Worlds (BoBW) problem in decentralized federated bandit setting, where it follows the FTRL algorithm’s idea but handle delay and client heterogeneity in federated settings via a modified learning-rate schedule and truncated loss estimators. Theoretical results are provided for the proposed algorithm and the regret almost matches the lower bound in this setting. Numerical experiments validate the effectiveness of the proposed method.

### Strengths
Theoretical guarantee for the proposed method is strong which almost matches the lower bound in this setting.

### Weaknesses
We need to know the topology and $D$ beforehand.

### Questions
**Learning rates:** How are learning rates in (201) compared with that in the delayed feedback (Masoudian et al., 2022) ?

**Communication cost:** How is $x_{v,t}(k)$ compared to  $12VC_t^{P}\gamma_t$ in (4)? In practice, how many rounds do we need to truncate the loss and broadcast? 

**equation (5):** It seems V multiplies to the loss after communication as well? In this case given $P$ is doubly stochastic, (when the loss estimates of the neighbors are closed), loss estimator seems to have exponential growth as time $t$ increases. Could the author elaborate this more?

**Clarification on matching bounds:** If the graph $G$ is known, could we always construct a doubly stochastic matrix $P$ as in Remark 1 to achieve the nearly matching bounds? 

Typos: 
line 155: (u,v)\not\in E

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper considers a multi-agent multi-armed bandit problem, where $V$ agents each face an identical copy of a set of $K$ arms. At each round, agents can share information with their neighbors according to a communication graph. Due to privacy constraints, agents are only allowed to share statistics of the losses rather than the raw losses themselves. The authors design a best-of-both-worlds (BOBW) algorithm that achieves near-optimal regrets in both stochastic and adversarial regimes. The paper also includes numerical evaluations of the proposed algorithm.

### Strengths
- This paper establishes new state-of-the-art BOBW regrets for the multi-agent privacy-preserving bandit setting.
- The work introduces a truncated loss estimator, which ensures that individual regrets across agents remain similar (Lemma 1). This is a notable contribution, as prior multi-agent bandit works typically require agents to be fully homogeneous to derive individual regret guarantees.

### Weaknesses
- Main concern: The definition of the feedback  $\ell_{v, t}(k_{v, t})$ is unclear. The paper highlights that it is biased, but does not clearly describe the nature or extent of the bias. Moreover, in Section 6.1, the feedback does not appear to show any bias. Could the authors clarify this point?
- It is unclear whether the goal of each agent is to minimize its own individual regret or a global regret across all agents. While the abstract emphasizes that no single agent can identify the globally best arm, this distinction is not explicitly modeled in the problem formulation. Additionally, if each agent is minimizing individual regret, identifying the global best arm may not be necessary. Clarification here would be helpful.
- The use of the term "federated" may be misleading, as federated learning typically involves a central server, whereas the setup in this work appears to be fully distributed.

### Questions
- Is the proof of Lemma 2 in the appendix actually meant to support Lemma 1? Please add clear references in the main text to help readers locate the corresponding proofs.
- Why was IND-FTRL not included in the evaluation shown in Figure 2?
- This work achieves a significant improvement in the regret bound for the adversarial regime. Could the authors elaborate on which algorithmic components, analysis techniques, or assumptions are responsible for this improvement compared to (Yi & Vojnović, 2023)?

### Soundness
3

### Presentation
3

### Contribution
3
