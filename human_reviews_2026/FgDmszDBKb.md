# StaQ: a Finite Memory Approach to Discrete Action Policy Mirror Descent

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
In Reinforcement Learning (RL), regularization with a Kullback-Leibler divergence that penalizes large deviations between successive policies has emerged as a popular tool both in theory and practice. This family of algorithms, often referred to as Policy Mirror Descent (PMD), has the property of averaging out policy evaluation errors which are bound to occur when using function approximators. However, exact PMD has remained a mostly theoretical framework, as its closed-form solution involves the sum of all past Q-functions which is generally intractable. A common practical approximation of PMD is to follow the natural policy gradient, but this potentially introduces errors in the policy update. In this paper, we propose and analyze PMD-like algorithms for discrete action spaces that only keep the last $M$ Q-functions in memory. We show theoretically that for a finite and large enough $M$, an RL algorithm can be derived that introduces no errors from the policy update, yet keeps the desirable PMD property of averaging out policy evaluation errors. Using an efficient GPU implementation, we then show empirically on several medium-scale RL benchmarks such as Mujoco and MinAtar that increasing $M$ improves performance up to a certain threshold where performance becomes indistinguishable with exact PMD, reinforcing the theoretical findings that using an infinite sum might be unnecessary and that keeping in memory the last $M$ Q-functions is a practical alternative to the natural policy gradient instantiation of PMD.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors observe that the exact Policy Mirror Descent (PMD) algorithm produces a policy that depends on the sum of all past Q-functions. However, under function approximation, this accumulation introduces error over time. To address this issue, the authors restrict their analysis to the discrete action space and propose a PMD-like algorithm that retains the most recent M Q-functions. In this formulation, for a sufficiently large but finite M, the policy update error diminishes, while the averaging effect in policy evaluation remains (up to error terms that decay exponentially with respect to M). Furthermore, the authors propose a practical implementation of their algorithm to investigate the benefit of stacking past M Q-functions in practice.

### Strengths
From a theoretical standpoint, it is both important and interesting to explore whether incorporating the past M Q-functions can provide benefits in the function approximation setting. However, I am not convinced that the findings in the paper translate to the practical setting.

### Weaknesses
- The presentation of the paper is at times very difficult to follow.
- The paper is positioned as an investigation into whether the theoretical foundations of PMD translate effectively into practice. However, all empirical results are deferred to the appendix without a clear guideline on what to look for, which limits the reader’s ability to assess the validity of the hypotheses. After carefully reviewing the appendix, I have several concerns:
    - The theoretical analysis focuses on the discrete action setting, yet the experiments start with continuous control tasks where the policy is parameterized as a diagonal Gaussian distribution. 
    - For MuJoCo environments, Soft Actor-Critic (SAC) is widely regarded as a strong baseline, but it is not included in the results. Conversely, DQN and M-DQN, which are designed for discrete action spaces, are evaluated on MuJoCo tasks.
    - In the Atari experiments, DQN and M-DQN use $\epsilon$-greedy exploration, but it is unclear whether the proposed method uses the same exploration strategy.
    - It is not specified whether the authors relied on a standard implementation framework (e.g., spinning up or stable-baselines) for the proposed method, which makes reproducibility and fair comparison difficult to assess.

### Questions
In the preliminaries, the policy is defined as  $\pi \propto \exp(Q(s, \cdot))$. However, based on the standard formulation of PMD, shouldn’t this instead be $\pi_t \propto \pi_{t-1} \exp(Q^{t-1}(s, \cdot))$? This would reflect the iterative update of the policy relative to the previous one, which, when applied recursively, results in the sum over all passed Q-functions?

In the function approximation setting, the early Q-functions are typically of low quality. Could you elaborate on the intuition behind why storing these past Q-functions would be beneficial? This design choice seems somewhat counterintuitive. Additionally, could you clarify which policy is used to sample actions during interaction with the environment?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
StaQ is a finite-memory variant of Policy Mirror Descent for \emph{discrete} actions that averages only the most recent \(M\) Q-functions. A stacked-Q implementation keeps computation efficient. As \(M\) increases, StaQ closely matches exact PMD and yields more stable, higher returns on discretized MuJoCo and MinAtar.

### Strengths
1. The paper provides guarantees that the truncation error from keeping only $M$ past Q-functions rapidly diminishes as $M$  grows, so the policy-update bias effectively vanishes. This bridges the gap between an idealized infinite-history PMD and a practical algorithm, giving users a clear knob $M$ to trade memory for accuracy.

2. The stacked-Q architecture enables parallel evaluation of multiple historical Q estimates in a single forward pass, avoiding extra policy-optimization steps. In practice, this keeps wall-clock overhead modest even for large \(M\), which makes the method easy to scale on standard GPUs.

3. Across several discrete or discretized benchmarks, increasing \(M\) consistently improves return and reduces variability across random seeds. For sufficiently large \(M\) (e.g., hundreds), performance is nearly indistinguishable from exact PMD, supporting the theoretical claims with tangible gains.

### Weaknesses
1 .The method and analysis focus on  discrete actions; extensions to continuous control are not developed. Given that many modern RL tasks are continuous, this limits immediate applicability and may require nontrivial adaptations of both the algorithm and proofs.
2.  Several experiments rely on discretizing originally continuous-control environments, which can alter the dynamics and policy landscape. Conclusions drawn in these settings may not directly transfer to canonical continuous-control benchmarks without further validation.

### Questions
see weakness

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
Policy mirror descent (PMD) algorithms can average out policy-evaluation errors, but their exact implementation is impractical since it requires storing all past Q functions. To address this, the paper proposes a finite-memory variant, called StaQ, which only requires retaining the last $M$ Q functions. The paper shows that the policy update error vanishes for sufficiently large $M$ and the number of iterations, and the averaging effect in PMD can be obtained by introducing a KL divergence penalization during policy update.

### Strengths
The finite-memory formulation is well-motivated, enabling a practical implementation of the PMD algorithm without requiring infinite storage or recomputation.

The paper discusses the similarities and differences between the proposed method and related works, mainly in Section 2, and further connects relevant prior studies throughout later sections where appropriate.

The work is supported by theoretical results. Theoretical analysis for the convergence of Entropy-regularized Policy Mirror Descent (EPMD) is provided in Section 4, to connect EPMD to finite memory EPMD and improve over existing analysis on the convergence rate of EPMD. Theorem 5.1 extends the convergence analysis to the finite memory setting.

The paper considers the applicability of the algorithm by providing practical implementation for EPMD in Section 4.3, and for finite memory EPMD in Section 5.1 (StaQ).

The paper empirically investigates how the number of stored Q functions, $M$, affects StaQ’s performance, demonstrating when the finite-memory approach approximates the exact EPMD’s behavior.

The method is empirically evaluated across a wide range of tasks, and detailed parameter settings are reported to support reproducibility.

### Weaknesses
The paper could be strengthened by quantifying memory usage and explicitly analyzing the trade-off between memory consumption and learning efficiency. StaQ lies between natural policy gradient methods (e.g., TRPO) and PMD. While StaQ reduces storage and computation by keeping only $M$ recent Q functions compared to PMD, it may still require higher memory usage compared to other baselines like TRPO. It may be worth comparing the memory usage of these methods and their learning curves.

The runtime in Table 1 provides limited insight into StaQ’s advantages. It does not capture one of the key benefits of StaQ, which lies in maintaining stable and accurate policy updates under limited memory. Table 1 mainly suggests that varying the number of stored Q functions does not have a large effect on the running time. Including learning curves in the main paper, to highlight StaQ’s learning efficiency, might make the empirical case more convincing.

The method is limited to discrete action space tasks. For continuous control tasks, the paper discretizes the action space, which may lead to loss of precision and suboptimal policies.

**Small thing:**

Line 324-325: “...over states sampled from _some a_ predefined initial state distribution.” You may want to remove _some_.

### Questions
The choice of $M$ may depend on the environment. Do you expect the observed pattern, where StaQ with M=300 approximates PMD, to generalize to other environments, such as those with higher stochasticity or delayed rewards?

In Figure 2, Humanoid-v4, M=500 performs worse than M=300. Similarly, in Table 2, M=300 occasionally outperforms the exact PMD. This result seems to contradict the intuition that a larger M approximates the exact PMD better. Where do you think the StaQ’s performance advantage may come from?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies policy mirror descent (PMD) with a finite memory of past Q-functions in discrete action spaces. It proposes StaQ, an optimization-free policy update rule implemented through a stacked neural network.Policy mirror descent algorithms usually include a distance term, such as KL divergence, to enforce a trust region and ensure stable updates. 

Theoretically, the paper shows that if there is no policy evaluation error and the memory size is large enough, the finite-memory update converges to the same policy as exact policy mirror descent. It also retains the error-averaging effect, with extra terms that decay exponentially with memory size.

The analysis covers both value iteration and policy iteration, providing explicit bounds that separate the effects of evaluation error and truncation. Empirically, the results show that increasing memory size improves performance up to a plateau, matching the exact-PMD baseline within the training horizon on discretized MuJoCo and MinAtar benchmarks.

### Strengths
- Clear problem formulation and principled update:
  - The finite-memory PMD policy is
    $$
    \xi\_{k+1} = \beta\\xi\_k + \alpha q\_k + \frac{\alpha \beta^M}{1-\beta^M}\big(q\_k - q\_{k-M}\big),
    $$
    is derived cleanly and highlights both the deletion of the oldest $q$ and the mild overweighting of the latest $q$.
- Theoretical extensions to Vieillard et al. (2020a) and careful bounds:
  - The paper correctly qualifies that if policy evaluation error vanishes and $M$ satisfies the stated condition, the update introduces no additional policy-update error, i.e., the algorithm converges to the optimal entropy-regularized policy.
- Practical relevance and engineering:
  - The stacked-neural-network (SNN) implementation exploits batched evaluation of many frozen $Q$-networks on GPU, making large $M$ feasible in practice. Precomputing logits for the replay buffer further amortizes cost during policy evaluation.
  - Wall-clock data (Table 1) show minimal runtime increase with $M$ on mid-scale tasks, supporting the feasibility claim.
- Empirical evidence aligns with their theorems:
  - Across discrete-action MuJoCo and MinAtar, increasing $M$ generally improves performance until diminishing returns in some environments; large $M$ becomes empirically close to the exact-PMD variant (that never deletes within the 5M-step window).

**References**
- Nino Vieillard, Tadashi Kozuno, Bruno Scherrer, Olivier Pietquin, Remi Munos, and Matthieu Geist. Leverage the average: an analysis of kl regularization in reinforcement learning. In Advances in Neural Information Processing Systems, 2020a.

### Weaknesses
- The main results and implementation are limited to discrete action spaces. While the paper acknowledges this, a discussion of algorithmic options for continuous actions (e.g., approximate sampling or parameterized policies) would strengthen the practical impact.
- Empirical analysis gaps:
  - Improvement with $M$ is not monotone across all tasks. The paper frames the trend as "beneficial effects with diminishing returns”, but the figures/tables show non-monotonic or even worse behavior on some environments (e.g., Hopper-v4, Breakout-v1, Freeway-v1, HalfCheetah-v4). The analysis could more explicitly discuss when/why performance can dip at intermediate $M$.
  - Baselines: Mirror Descent Policy Optimization (MDPO) is a closely related comparator for policy mirror descent with approximate updates; including it would position StaQ more clearly against the most directly related policy-optimization methods.
  - Sensitivity and interpretability: Although $\epsilon$-softmax exploration is used uniformly in the main study and a “sticky” behavior policy is shown to unlock MountainCar in a focused analysis, a broader sensitivity study of behavior policy design and its interaction with error averaging would be valuable.
- Benchmark choice: Discretized MuJoCo is not ideal; full Atari or other naturally discrete benchmarks would be more appropriate.

### Questions
1. The paper’s methods and theory are limited to discrete action spaces. How difficult would it be to extend StaQ to continuous actions? Could re-parameterized actions such as Gumbel softmax policies preserve similar theoretical guarantees? 
2. The paper assumes vanishing policy evaluation error for convergence proofs. How sensitive is StaQ to nonzero evaluation errors in practice, and could the analysis be extended to handle this more realistic case?
3. The improvement with memory size (M) is not monotonic across tasks. Can the authors explain why intermediate (M) sometimes leads to worse performance? Does truncation interact with learning dynamics or noise in Q-function estimates?
4. The experiments lack baselines such as Mirror Descent Policy Optimization (MDPO). Why was MDPO omitted, and how might StaQ compare to it theoretically and empirically?
5. The paper uses only one behavior policy variant (ε-softmax) except for a small analysis on sticky policies. Could more diverse behavior policies change StaQ’s error-averaging behavior or its convergence?
6. Given the strong theoretical results but limited empirical exploration, how confident should we be that StaQ’s convergence properties translate to real-world deep RL settings?
7. Are there scenarios where finite memory could harm performance due to outdated Q-functions being overweighted or reintroduced?

### Soundness
3

### Presentation
3

### Contribution
2
