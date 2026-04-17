# Minimax Optimal Adversarial Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Consider episodic Markov decision processes (MDPs) with adversarially chosen transition kernels, where the transition kernel is adversarially chosen at each episode. Prior works have established regret upper bounds of $\widetilde{\mathcal{O}}(\sqrt{T} + C^P)$, where $T$ is the number of episodes and $C^P$ quantifies the degree of adversarial change in the transition dynamics. This regret bound may scale as large as $\mathcal{O}(T)$, leading to a linear regret. This raises a fundamental question: *Can sublinear regret be achieved under fully adversarial transition kernels?*  We answer this question affirmatively. First, we show that the optimal policy for MDPs with adversarial transition kernels must be history-dependent. We then design an algorithm of Adversarial Dynamics Follow-the-Regularized-Leader (AD-FTRL), and prove that it achieves a sublinear regret of $\mathcal{O}(\sqrt{(|\mathcal{S}||\mathcal{A}|)^K T})$, 
where  $K$ is the horizon length, $|\mathcal{S}|$ is the number of states, and $|\mathcal{A}|$ is the number of actions. Such a regret cannot be achieved by simply solving this problem as a contextual bandit. We further construct a hard MDP instance and prove a matching lower bound on the regret, which thereby demonstrates the **minimax optimality** of our algorithm.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses Adversarial Reinforcement Learning (RL) where both the transition kernel and the reward function exhibit adversarial dynamics across different episodes. To tackle this, the authors propose AD-FTRL, a novel algorithm based on occupancy measure analysis. This method achieves a regret bound proportional to the square root of the number of episodes ($\sqrt{T}$). Furthermore, the authors include a counterexample to suggest the optimality of their proposed framework.

### Strengths
1. This paper tackles the challenging problem of fully adversarial Reinforcement Learning (RL), where both the transition kernel and the reward function exhibit adversarial dynamics across episodes. The proposed framework comes with a strong theoretical guarantee of $\mathcal{O}(\sqrt{T})$ regret.

2. The authors provide counterexamples that demonstrate the necessity of non-Markovian policies and further illustrate the optimality of their proposed algorithm.

### Weaknesses
1. Even though the lower bound suggests that the regret is optimal, the regret bound has an exponential dependency on the size of the state-action space ($|S||A|$). Consequently, the regret bound is non-trivial only when the number of episodes ($T$) is sufficiently large (specifically, $T > |S||A|^k$). This exponential dependence significantly limits the framework's practical application to large-scale problems.

2. Some claims regarding related work are improper. Specifically, concerning the Upper Bound Analysis, it is crucial to note that prior analyses for adversarial reinforcement learning or bandits operate in three distinct settings, each with a different regret definition. It is unfair to make a comparison without clearly stating these differences.

(a). Fully Adversarial RL: This work falls into this setting (discussed in Lines 118-119), where there is no limitation on the adversarial dynamics, and regret is calculated against the best fixed policy across all episodes. Prior work in this setting did not consider a dynamic transition kernel.

(b). Adversarial Corruption: This setting (Lines 128-129) focuses on corruption where a base reward function and transition kernel exist, and only a small corruption is added. Regret is calculated against the optimal policy in the base model and is efficient only when the total corruption level ($C$) is sublinear in the number of episodes.

(c). Non-Stationary Environments: In this setting (Lines 134-135), the regret is measured against the optimal action in each episode (not fixed across episodes), making the guarantee strictly stronger than the guarantee in this paper. However, this comes at the cost of having a limitation on the cumulative dynamic.

Overall, it is necessary to clarify the different settings to accurately establish the contribution of this work.

3. The claim in Remark 1 is incorrect. There still exists a trivial method that treats each Markovian policy $\mathcal{S} \times \mathcal{H} \to \mathcal{A}$ as a distinct bandit arm or agent, which yields $|\mathcal{A}|^{|\mathcal{S}|H}$ different arms. Even though the result from this trivial method is non-optimal, the authors still need to clarify the mapping.

### Questions
1. The preliminary discussion ("WARM-UP") primarily focuses on the adversarial transition kernel. While it is true that learning transitions in a stochastic environment is often harder than learning the reward function, the analysis should address the additional challenges and complexities that arise when combining adversarial rewards and adversarial transition kernels simultaneously. A deeper discussion on this interaction is warranted.

2. The current framework assumes the action and state sets are non-overlapping across different stages, and, more specifically, assumes uniform sizes such as $|A^k|=|A|$ and $|S^k|=|S|$. What is the regret bound without these simplifying assumptions? The paper should discuss the impact of varying state and action space sizes across stages.

3. Several key related works on Adversarial Reinforcement Learning are missing and should be included for proper context and comparison:

(a): Fully Adversarial RL (Under Function Approximation): The analysis of fully adversarial RL extends beyond tabular settings when incorporating function approximation:

[1] "Provably Efficient Exploration in Policy Optimization" (ICML 2020) considered the adversarial reward function under linear function approximation.

[2] "Near-optimal policy optimization algorithms for learning adversarial linear mixture mdps" (AISTATS 2022) later extended the results to achieve optimal guarantees within the linear function approximation setting.

(b): Adversarial Corruption (General Function Approximation): In the setting of adversarial corruption, work has also addressed robustness with complex function classes:

[3] "Towards robust model-based reinforcement learning against adversarial corruption" (ICML 2024) studied adversarial attacks on the transition kernel with general function approximation, achieving a near-optimal regret guarantee under the adversarial attack model.

### Soundness
3

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
4

### Summary
This work studies how to learn MDPs with both adversarially chosen transition kernels and loss functions. Prior studies only obtain a regret of $\mathcal{O}(\sqrt{T}+C^P)$, where $C^P$ is the degree of adversarial change of transition kernels. This work proposes an FTRL algorithm, which operates over the space of trajectory occupancy measure and achieves a regret of $\mathcal{O}\left(\sqrt{(|\mathcal{S}||\mathcal{A}|)^K T}\right)$. The authors also prove a matching lower bound for this problem.

### Strengths
1. **Novelty**: The idea of the trajectory occupancy measure in this work is simple yet powerful. The matching lower bound is also very interesting.
2. **Presentation**: Most parts of this work are generally well-written and easy to follow.

### Weaknesses
1. **Computation efficiency**: If any, I feel that operating over the space of the trajectory occupancy measure makes the algorithm computationally inefficient. Also, I seemingly cannot find any discussions relating to the computational efficiency of the proposed algorithm. That said, I personally do not think it is a significant drawback, considering that this work proposes the first minimax optimal algorithm for this challenging problem.

### Questions
1. In MDPs with unknown yet fixed transition kernels and adversarially chosen loss functions, we can devise a policy optimization (PO)-based algorithm, which is more computationally efficient and easier to implement. Is it possible to devise a similar PO-based algorithm for the problem considered in this work?
2. Typically, using FTRL with Shannon entropy may lead to suboptimal factors of $\log A$ in adversarial bandits and of $\log (SA)$ in adversarial RL. Why FTRL with Shannon entropy can lead to the minimax optimal regret without having similar factors in the result?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies episodic RL with fully adversarial transitions and bandit feedback. At every round the environment can pick an arbitrary MDP (transitions + losses) from a hard family; the learner only observes a trajectory. The authors propose a trajectory-level FTRL algorithm that maintains a distribution over trajectories and uses importance-weighted loss estimates. They prove a regret upper bound $O((|S||A|)^{\frac{K}{2}}\sqrt{T})$ and then match it with a lower bound of the same order. Thus, the minimax optimal rate is derived for this model. They also show Markov policies can be suboptimal: optimal strategies must be history-dependent.

### Strengths
The paper give sharp information-theoretical result. Upper and lower bounds match with order $(|S||A|)^{\frac{K}{2}}\sqrt{T}$ so the rate is tight for fully adversarial model.

By working directly on trajectory occupancies instead of state–action occupancies, the paper pinpoints where previous bounds picked up a linear term in the transition variation.

The example showing Markov policies are insufficient in this adversarial setting is useful and interesting.

### Weaknesses
Both the regret bound and computational complexity is in exponentially order. Although this is correct and tight in theory, it makes the algorithm less practical. 

In previous papers where $O(\sqrt{T} + C^P)$ regret is derived, it gets sublinear regret when the change of transition is small. This paper, however, always suffer  $(|S||A|)^{\frac{K}{2}} \sqrt{T}$ regret no matter how the transition changes. Thus, the new algorithm is better when $C^P = O(T)$ and $T >  (|S||A|)^{K}$, which is a very restrictive regime.

The change of transition is obliviously adversarial, what is the additional challenge for adaptive adversary?

### Questions
See weakness.

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
4

### Summary
This submission introduces (to the adversarial RL community) the key observation that the optimal policy in episodic MDPs with adversarial trantisions and rewards is not necessarily Markovian. Following this observation, the authors give a FTRL-type algorithm achieving the optimal regret $\tilde{O}(\sqrt{|\mathcal{S}|^K|\mathcal{A}|^K}\sqrt{T})$. The authors also certificate the optimality of the standard FTRL routine with a novel lower bound construction, simutaneously manifesting the exponential-in-$K$ dependency of $|\mathcal{S}|$ and $|\mathcal{A}|$.

### Strengths
- The employment of non-Markovian policy as the correct class to consider is a novel and significant observation
- The construction beyond constant many arms is novel for proving the lower bound
- Lemma B.3, which justify the convexity of the set of trajectory distributions induced by $\epsilon$-greedy non-Markovian policies, appears to be novel

### Weaknesses
- Algorithm 1 a relatively familiar to the adversarial RL community, if not standard. 
  - It is intuitively likely that history-dependent policy has been considered in different contexts but also under the algorithmic framework of occupancy measure; with that said, I respectfully question the originality of Lemma B.3. The authors can correct me if that is not the case.
  - Since Lemma B.3 seems to be the major technical novelty in the upper bound analysis, it is more appropriate for the authors to justify its originality.


### Minor weaknesses

- On Figure 1: There is as a typo because the authors may want to design the initial state in $P_1$ to be $\star$.
- Theorem 5.1: $O$ -> $\tilde{O}$
- Line 320: The first $\pi$ on the RHS should be $\pi_t$
- Line 226: Occupancy measure has been introduced in very classical literature, e.g., those regarding the linear programming formulation of RL in MDPs. I mean this is definitely not a concept that was introduced for the first time in the 21st century.
- Line 169: It seems that the loss function considered within one episode is time-homegeneous, right?

### Questions
- Given the existence of EXP3.P in the literature that achieves high-probability guarantees, do the authors plan to argue about the dfficulty for proving a high-probability regret bound for (potentially certain variant) of their Algorithm 1?
- The lower bound construction is more significant than the upper bound analysis. In fact, is it more appropriate to reduce the length of the description of the relatively standard FTRL-type routines in the main text and elaborate more on how the authors successfully manifest the dependency on $|\mathcal{S}|^{\Omega(K)}$ and $|\mathcal{A}|^{\Omega(K)}$ simultaneously in the lower bound analysis?
- Though it is relatively standard for adversarial RL papers to assume the state space to be disjoint for different steps, it is a bit interesting to also assume the action set to be disjoint for different steps. How does this later assumption simplifes the proof?

### Soundness
3

### Presentation
3

### Contribution
3
