# Cooperative Multi-agent RL with Communication Constraints

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Cooperative Multi-agent reinforcement learning (MARL) often assumes frequent access to global information in a data buffer, such as team rewards or other agents’ actions, which is typically unrealistic in decentralized MARL systems due to high communication costs. When communication is limited, agents must rely on outdated information to estimate gradients and update their policies. A common approach to handle missing data is called importance sampling, in which we reweigh old data from a base policy to estimate gradients for the current policy.  However, it quickly becomes unstable when the communication is limited (i.e. missing data probability is high), so that the base policy in importance sampling is outdated. To address this issue, we propose a technique called \textit{base policy prediction}, which utilizes old gradients to predict the policy update and collect samples for a sequence of base policies, which reduces the gap between the base policy and the current policy. This approach enables effective learning with significantly fewer communication rounds, since the samples of predicted base policies could be collected within one communication round. Theoretically, we show that our algorithm converges to an $\varepsilon$-Nash equilibrium in potential games with only $\mathcal{O}(\varepsilon^{-3/4})$ communication rounds and $\mathcal{O}(\max_i |\mathcal{A}_i|\cdot \varepsilon^{-11/4})$ samples, improving existing state-of-the-art results in communication cost, as well as sample complexity without the exponential dependence on the joint action space size. We also extend these results to general Markov Cooperative Games to find an agent-wise local maximum. Empirically, we test the base policy prediction algorithm in both simulated games and MAPPO for complex environments. The results show that our algorithms can significantly reduce the communication costs while maintaining good performance compared to the setting without communication constraints, and standard algorithms fail under the same communication cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a theoretical framework for cooperative multi-agent reinforcement learning (MARL) under communication constraints by unifying two classes of problems: potential games (PGs) and Markov cooperative games (MCGs). The authors first analyze static potential games where agents’ incentives are aligned through a potential function and propose a communication-efficient algorithm that provably converges to an $\epsilon$-Nash equilibrium with quantified sample and communication complexities. They then extend this framework to the sequential setting of MCGs, where all agents share a common reward, and prove convergence to an agent-wise $\epsilon$-local optimum. The paper provides rigorous definitions, assumptions, and proofs establishing both sample and communication complexity bounds, showing that cooperation can be achieved efficiently despite limited information exchange. The theoretical development is elegant and mathematically sound, offering a clear link between cooperative game structures and decentralized optimization.

### Strengths
- The algorithm achieves explicit bounds on both sample and communication complexity, which is an increasingly important issue for large-scale MARL systems.
- The paper rigorously connects Potential Games and Markov Cooperative Games, offering a unifying theoretical lens for cooperative MARL. Provides precise definitions, proof sketches, and clear assumptions

### Weaknesses
- The paper claims that base policy prediction can proactively anticipate a sequence of base policies and thereby reduce the number of communication rounds required for new data collection. However, there is no clear analysis or empirical evidence demonstrating that these predictions are accurate enough or that the number of communication rounds is indeed reduced.
- Some implementation details are unclear or inconsistent with the algorithmic description. For instance, the variable $I_t$ in Algorithm 2 is not defined or initialized anywhere in the manuscript, making it difficult to follow.
- Both Algorithm 1 and Algorithm 2 specify adaptive communication triggers, yet the experiments use predefined communication intervals. This inconsistency indicates a gap between the theoretical algorithm and the implemented version.
- In the congestion game experiment, the authors claim that when the communication interval is large, the naive importance sampling baseline becomes outdated and fails to converge. However, since the communication interval is described as fixed, it is unclear how this conclusion is supported.
- The writing and presentation could be improved for clarity. For example, the explanation of Figure 2 (especially regarding MAPPO) is confusing. It is not clear whether MAPPO is a distinct baseline or whether both curves correspond to the proposed method with and without base policy prediction.
- Since the contribution is on communication-efficient, more related work should be discussed regarding communication in cooperative MARL.
- (minor) Figures 2 and 3 are not explicitly referenced in the main text, which disrupts the flow of reading and makes it hard to connect the results to the discussion.

### Questions
- Can the authors provide quantitative evidence showing that the proposed base policy prediction reduces the total number of communication rounds compared to standard synchronization schemes?
- The experimental baselines are limited. Since the focus is on communication-efficient learning, have the authors considered comparing with other communication strategies such as the multiple synchronization/communication rules described in [1]? More empirical comparison or comprehensive discussion of communication in MARL in related work will make this work stronger.

#### [1] Hsu et al., "Randomized Exploration in Cooperative Multi-Agent Reinforcement Learning", NeurIPS 2024

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses a critical challenge in cooperative multi-agent reinforcement learning (MARL) — communication constraints that prevent agents from frequently sharing global information (e.g., other agents’ actions or team rewards). Traditional MARL methods typically assume constant communication, which is unrealistic in decentralized systems.

The key idea is to improve learning when communication is infrequent by proposing a method called Base Policy Prediction (BPP).

Instead of using stale base policies in importance sampling (IS) (as in TRPO or PPO, which can cause high variance and instability), the authors propose to predict future base policies using old gradients. This keeps the estimated base policy closer to the current policy, reducing variance and allowing fewer communication rounds without sacrificing performance.

### Strengths
Novel Theoretical Advancement:
Introduces Base Policy Prediction, a modification to importance sampling that bridges the gap between outdated and current policies.

Improved Efficiency:
Achieves state-of-the-art results in both communication cost and sample complexity, removing the dependence on the joint action space size.

Strong Theoretical Guarantees:
Provides formal convergence proofs to an ε-Nash equilibrium and clear bounds on communication and sample complexity.

Practical Validation:
Integrates with existing frameworks (such as MAPPO) and demonstrates empirical effectiveness under realistic communication constraints.

### Weaknesses
Dependence on Gradient Prediction Accuracy:
The success of Base Policy Prediction heavily depends on the accuracy of old gradient estimates. Noisy or non-stationary environments could degrade performance.

ε-Nash Equilibrium vs. Global Optimum:
The algorithm converges to an ε-Nash equilibrium or local optimum, which is not necessarily the globally optimal cooperative solution.

### Questions
I am new to this area, so please forgive any possible misunderstandings.

Does the communication cost for broadcasting the replay buffer bound the total amount of data available to each agent? How is it connected to sample efficiency?

If BPP can reduce the variance of importance sampling and enable reusing old trajectories for more policy updates, could it also improve sampling efficiency for general (non–multi-agent) RL?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the challenge of multi-agent reinforcement learning under limited communication, a realistic but underexplored setting where agents cannot frequently share global information such as rewards or others’ actions due to communication costs.
The authors propose a novel method called Base Policy Prediction, which extends importance sampling by predicting future base policies using old gradients. This technique allows agents to maintain more accurate gradient estimates without frequent synchronization, thereby reducing communication rounds while preserving learning stability.

### Strengths
Originality:

The Base Policy Prediction mechanism is a creative modification to classical importance sampling, introducing gradient-based prediction of base policies rather than relying on static ones.
It bridges a clear gap between theoretical MARL under communication constraints and practical distributed implementations like MAPPO, a direction that has been rarely formalized.

Quality:


The sample and communication complexity improvements are significant.
The empirical section validates theory across both toy potential games and complex MARL benchmarks, confirming the benefits in realistic environments.
The comparison table provides a transparent theoretical baseline against prior work.

Clarity:

The presentation of algorithms, changing conditions, and intuition behind communication triggers is commendably clear for a mathematically heavy paper.

Significance:

Addresses a real-world bottleneck in cooperative MARL, which is crucial for scaling to robotics, traffic, and IoT applications.
The framework provides a principled bridge between theory and practice for communication-limited multi-agent learning.

### Weaknesses
While the experiments show promising results, the paper could include quantitative comparisons of communication vs. performance trade-offs across a wider range of intervals.

The SMAC and MPE experiments are promising but lack variance/error bars and statistical significance tests to confirm robustness.

The Base Policy Prediction approach requires computing and storing multiple predicted policies per round.


All agents are assumed homogeneous in terms of policy structure and reward access. Real-world MARL often involves heterogeneous agents; it is unclear how well BPP generalizes there.


The connection between the theoretical oracle in Potential Games and the approximate solver used in experiments could be made more explicit.

### Questions
How does the algorithm scale with continuous or large discrete action spaces? Could the base policy prediction mechanism be adapted using function approximation to handle such cases efficiently?

Since the BPP relies on predicting gradients from old data, how sensitive is the convergence to gradient noise or non-stationarity in the environment?

Could the base policy prediction mechanism extend to non-cooperative or general-sum settings, possibly with bounded regret guarantees?

The paper mentions the codebase in the supplementary material, but it would be helpful to specify the hyperparameters, communication interval values, and exact network architectures for reproducibility.

### Soundness
4

### Presentation
4

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
The paper studies cooperative multi-agent RL under limited communication and proposes base policy prediction, a modified importance-sampling scheme that precomputes a sequence of base policies from old gradients. This keeps importance weights stable, enabling far fewer communication rounds. The authors prove convergence to $\varepsilon$-Nash equilibria in potential games and extend the framework to Markov cooperative games with polynomial sample complexity.

### Strengths
1. The core idea-predicting future base policies via natural-policy-gradient-style updates and collecting data for that whole predicted set in a single communication round-is a clean variance control mechanism for importance sampling under staleness, and it is explicitly tied to the communication trigger condition.

2. The analysis for potential games gives simultaneous bounds on communication rounds $O\left(\varepsilon^{-3 / 4}\right)$ and samples $O\left(\operatorname{poly}\left(\max _i\left|A_i\right|\right) \varepsilon^{-11 / 4}\right)$, improving prior $\varepsilon$-dependence while avoiding exponential dependence on the joint action space.

3. The extension to Markov cooperative games shows the PG oracle can be plugged into a V-learning style outer loop, preserving the communication-saving idea while still yielding an $O(\varepsilon)$ agent-wise local maximum, which demonstrates the method is not limited to the static PG setting.

### Weaknesses
1. The communication-trigger condition (two-part test on reward-difference and on elapsed steps) is chosen to bound IS variance, but the paper does not give a tight or adaptive rule showing this condition is near-optimal for a given environment.

2. The sample complexity still carries a relatively high exponent $\varepsilon^{-11 / 4}$; although better than some baselines, the proof does not clarify whether this exponent is an artifact of handling multiple predicted policies or is information-theoretically necessary.

3. The MCG extension relies on Assumption 4.1 (gap and coverage) for every constructed PG instance; this is a strong structural requirement, and the paper does not discuss how sensitive the guarantees are if the gap is small or state-dependent.

### Questions
1. In the base-policy prediction step, can the authors quantify the maximal KL (or total-variation) drift between a predicted base policy and the actual future policy that still keeps the IS weights uniformly bounded?

2. The changing condition mixes a value-difference test and a hard cap $t^{\prime} \geq c \varepsilon^{-1 / 4}$. Is there a way to remove the hard cap and trigger purely on variance/proxy estimates while preserving the $O\left(\varepsilon^{-3 / 4}\right)$ communication bound?

3. For the MCG algorithm, the PG oracle is called at multiple stages along the horizon. How does error from approximate PG solutions accumulate across stages, and can it be localized so the overall gap stays $O(\varepsilon)$ without tightening each oracle call?

4. The analysis avoids exponential $\prod_i\left|A_i\right|$ factors by counting distinct predicted policies. Can the same counting argument be extended to continuous action spaces via smoothing or entropy regularization, or is discreteness essential to keep $\left|\Pi_t\right|$ small?

### Soundness
3

### Presentation
3

### Contribution
3
