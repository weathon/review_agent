# A Bayesian Multi-agent Multi-arm Bandit Framework for Optimal Decision Making in Dynamically Changing Environments

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
We introduce DAMAS (Dynamic Adaptation through Multi-Agent Systems), a novel framework for decision-making in non-stationary environments characterized by varying reward distributions and dynamic constraints. Our framework integrates a multi-agent system with Multi-Armed Bandit (MAB) algorithms and Bayesian updates, enabling each agent to specialize in a particular environmental state. DAMAS continuously estimates the probability of being in each state using only reward observations, allowing rapid adaptation to changing conditions without the need for explicit context features. Our evaluation of DAMAS included both synthetic environments and real-world web server workloads. Our results show that DAMAS outperforms state-of-the-art methods, reducing regret by around 40% and achieving a higher probability of selecting the best action.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes DAMAS, a Bayesian multi-agent multi-armed bandit framework for non-stationary environments. Each agent specializes in a latent environment, and the system maintains Bayesian beliefs over which environment is active based on reward observations without using explicit contextual features. DAMAS integrates Bayesian updates for environment inference and Bayesian optimization for adaptive exploration tuning. The authors provided a theoretical regret analysis, and conducted numerical experiments on both synthetic and real-world data to show the effectiveness of their method.

### Strengths
- The paper proposes a framework that integrates multi-agent modeling, Bayesian environment inference, and MAB decision-making, which is conceptually novel. I also think this kind of problem has practical relevance as it addresses dynamic systems without contextual signals.
- The paper provides a detailed empirical evaluation across multiple types of non-stationarity.

### Weaknesses
- One major issue with this paper is in its evaluation metrics, as there appears to be a mismatch between theoretical and empirical regret definitions here. If I understand correctly, the theoretical analysis establishes regret bounds relative to a static optimal arm within each stationary phase (as in UCB), whereas the experiments compute regret against a dynamic oracle that knows the best arm for each environment at every time step. This inconsistency makes it difficult to interpret how the theoretical guarantees relate to the reported empirical improvements, and limits the clarity of what kind of optimality DAMAS actually achieves. The authors were also not very clear what is the benchmark they are comparing against. 
- Related to the point above, the main theoretical results (Theorems 4.1 & 4.2) essentially reduce to showing that DAMAS converges to UCB1-style behavior within stationary phases, yielding O(logT) static regret. This result relies on long dwell times and offers little advance over classical stationary-bandit analyses. I think for a non-stationary bandit setting, it'd be much more helpful to investigate the dynamic regret, to enable comparison with related works on non-stationary bandits.
- I also still have concerns over the computational overhead of the current framework. The multi-agent Bayesian updates and Bayesian-optimization tuning is reported as roughly 39.3× slower than UCB. For real-world, large-scale or high-frequency decision systems, this can be too much to handle and further optimization might be required.
- The paper mainly considers UCB/TS-based methods as its baselines, but it omits standard variation-budget and parameter-free non-stationary bandit methods such as RExp3.

### Questions
- The framework requires pre-training each agent in a fixed environment to estimate, then uses Gaussian likelihoods to update environment posteriors online. I wonder if this framework can remain robust if the real non-stationarity deviates from those pre-trained distributions, or when the number/identity of environments is misspecified?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper investigates a multi-state multi-armed bandit setting by proposing an algorithm that dynamically estimates the probability of being in each state. Each state is characterized by changing reward distributions and dynamic constraints. Theoretical guarantees are provided with respect to finite time regret performance of the algorithm. Empirical results also demonstrate some improvement over proposed baseline algorithms.

### Strengths
- This paper addresses relevant problems in non-stationary MAB with valid application areas, such as server allocation, communications etc.

- This paper provides a good amount of empirical evidence to support their claims in comparison to the benchmarks the authors designed.

### Weaknesses
- Seems to me like the framework considers a MAB problem where the states, which govern the reward space is evolving over time. I have a hard time discerning the difference between this model and that of a non-stationary multi-armed bandit model captured by the restless bandit framework (espcially refering to lines 130-137). Even if there are (perhaps marginal) differences between the authors problem setting and the restless bandit setting, it seems like this should have been at least mentioned in the literature review.

- I have a hard time understanding the multi-agent aspect of this work. It seems to me that the algorithm is finding the best agent for the context, and each agent accordingly takes the best action. How is this any different than regular MAB with nested action space? That is we consider the "agent" in this paper just another "action" - could we not also solve this using combinatorial bandit for example? If so then the authors should consider a larger array of baseline algorithms. 

- W.r.t. to the theory, the authors argue that their problem reduces to a UCB algorithm (if I'm not mistaken), and given that UCB has logarithmic regret, so does their algorithm. However, there is more to be desired here, and I encourage the authors to consider per se, whether the algorithm suffers from gap-dependent or gap-free regret, scaling w.r.t. the number of agents, and size of the action space etc. Overall w.r.t. to bandit theory, it is lacking some richness and depth here. It also ties back to my previous point regarding the multi-agent aspect, if this is just UCB with more arms and changing contexts - previous work has shown that in these cases, regret is logarathmic in T [1, 2].

[1] Tekin, Cem, and Mingyan Liu. "Online learning of rested and restless bandits." IEEE Transactions on Information Theory 58.8 (2012): 5588-5611.

[2] Liu, Haoyang, Keqin Liu, and Qing Zhao. "Logarithmic weak regret of non-bayesian restless multi-armed bandit." 2011 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2011.

### Questions
- Are there any interactions between agents and their actions, when acting cooperatively? i.e. are the rewards w.r.t. actions submodular/supermodular? 

- Do the agents have any sense of "agency", or are they purely controlled by the central controller? In which agents may be free to act whatever way they choose under the circumstance and reward structure given - adding complexity to the decision process.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Dynamic Adaptation through Multi-Agent Systems (DAMAS), a new method for non-stationary multi-armed bandits (MABs). DAMAS does not require any contextual information, only observing rewards. The problem is modeled as a set of $E = \{e_0, \dots, e_n \}$ MABs (where the rewards for each arm are sampled from a Gaussian distribution with unknown mean and variance). It is unknown from which of the $e_i, i \in [0,n]$ MABs the sampled reward comes from. The non-stationarity is defined as the change from one $e_i$ to another $e_j$. DAMAS instantiates one agent $\phi_i$ for each environment $e_i$. Additionally, it estimates the likelihood $P(r_t|e_i)$ of an observed reward $r_t$ coming from a specific environment $e_i$. Based on this, it selects the corresponding agent, and pulls an arm using UCB. All agents are updated at each timestep, where the update takes into account $P(e_i)$.

### Strengths
- The problem is well defined, and based on practical applications (web server workloads)
- The proposed approach is justified, and its efficiency demonstrated on multiple non-stationarity scenarios
- I found the weighted Q-value update (line 161)

### Weaknesses
- The framework assumes that 1) the number of environments $|E|$ is known in advance, so it can instantiate the correct number of agents, and 2) that the properties of each $e_i$ are known, since the agents are first trained on the individual environments separately. In a sense, the optimal arm for each environment is known in advance. These seem to me very strong assumptions, that contradict the claim that DAMAS does not rely on external context. 
- The limiting assumptions are conveyed in the real-world experiments (Appendix A), where DAMAS has a higher regret and response time than the single-agent counterpart.
- Even though the theoretical contributions are sound, they seem to me of limited use, as they assume that the agent remains in the current environment for long enough ($T_i \rightarrow \infty$).
- Since only the reward is used to estimate the probability of being in a specific environment, the range of rewards need to be different. This is shown in line 301, where arm-means range in $[0.2,0.7], [0.02,0.09], [1.5,2.5]$ for $e_O, e_1, e_2$, respectively. I believe it would be hard for DAMAS to systematically pull the optimal arm when the ranges of rewards are highly overlapping across environments.

Although the algorithm is sound, well justified and clearly explained, to my understanding, it is based on strong assumptions, limiting its use in practice.

### Questions
Although to my understanding, the assumptions seem strong. Could you clarify how realistic they are?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors consider a multi-armed bandit problem with unbounded rewards and non-stationary environments (i.e., there being $ n $ different reward distributions and the distributions changing over time arbitrarily). The authors propose an algorithm called DAMAS that instantiates $ n $ different instances of a multi-armed bandit algorithm - one for each environment - and uses Bayesian posteriors for updating the algorithms' internal states (e.g., value estimates, etc.) and the posterior probabilities of being in each environment. The authors show a phase-wise asymptotic logarithmic regret bound (i.e., under the assumption of spending an infinite amount of time in each environment before transitioning) and strong experimental results on synthetic setups and real-world web server workloads.

### Strengths
* The paper studies an important sub-area of multi-armed bandits: non-stationary environments and unbounded rewards.

* The paper proposes a simple and principled way of dealing with non-stationary environments, namely, maintaining and updating Bayesian posteriors over the environments.

* The experiments on synthetic setups cover a range of non-stationary characteristics (environments changing abruptly, gradually, randomly, etc.) and show strong empirical performance of DAMAS (the proposed algorithm) over baselines.

### Weaknesses
*  The regret bound is quite weak because it requires the algorithm to spend an infinite amount of time in each environment before transitioning. Not only is the regret bound asymptotic (i.e., it does not hold for a fixed time horizon), it requires each phase length to be asymptotic as well.

* The paper motivates its use of reward and no other context as a strength because it requires less information. However, I think this is a weakness because context is present and is important in many applications. Many other lines of work in the multi-armed bandit literature started out with the non-contextual setting and over time included the contextual setting because it is prevalent in many applications (e.g., bandits with knapsacks -> contextual bandits with knapsacks in online advertising and resource allocation problems). It would be fair to say that considering only rewards is a choice to simplify the problem. However, I don't think it's fair to say that this work overcomes the difficulty of obtaining contextual features.

### Questions
Please see my comments in the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
