# Instance-Dependent Continuous-Time Reinforcement Learning via Maximum Likelihood Estimation

- Avg Score: 6.67
- Decision: Reject
- Scores: 8, 4, 8

## Abstract
Continuous-time reinforcement learning (CTRL) provides a natural framework for sequential decision-making in dynamic environments where interactions evolve continuously over time. While CTRL has shown growing empirical success, its ability to adapt to varying levels of problem difficulty remains poorly understood. In this work, we investigate the instance-dependent behavior of CTRL and introduce a simple, model-based algorithm built on maximum likelihood estimation (MLE) with a general function approximator. Unlike existing approaches that estimate system dynamics directly, our method estimates the state marginal density to guide learning. We establish instance-dependent performance guarantees by deriving a regret bound that scales with the total reward variance and measurement resolution. Notably, the regret becomes independent of the specific measurement strategy when the observation frequency adapts appropriately to the problem’s complexity. To further improve performance, our algorithm incorporates a randomized measurement schedule that enhances sample efficiency without increasing measurement cost. These results highlight a new direction for designing CTRL algorithms that automatically adjust their learning behavior based on the underlying difficulty of the environment.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces CT-MLE, a model-based algorithm for continuous-time reinforcement learning (CTRL) that uses maximum likelihood estimation (MLE) of the state marginal density instead of directly modeling system dynamics.
The key idea is to achieve instance-dependent adaptivity, where the algorithm’s regret scales with the total reward variance rather than with fixed measurement schedules.
The authors derive theoretical guarantees, showing that regret can become independent of the measurement strategy when observation frequency adapts to problem complexity.
Additionally, they propose a randomized measurement schedule to enhance sample efficiency without additional measurement cost.

### Strengths
The shift from dynamic estimation to marginal density estimation via MLE is elegant and conceptually simple, offering a new angle on continuous-time RL problems.

The authors derive a variance-dependent, nearly horizon-free regret bound, marking a notable advance over existing CTRL works that rely on worst-case analysis or have exponential dependence on the time horizon.

The introduction of reward variance as a measure of instance difficulty is insightful, connecting continuous-time control with established ideas in variance-aware RL.

### Weaknesses
The empirical component appears secondary; the paper would benefit from more thorough experiments demonstrating practical performance, robustness, and scalability of CT-MLE.

The assumptions of finite policy, drift, and diffusion classes simplify analysis but reduce applicability to large-scale or continuous policy spaces.

### Questions
How sensitive is CT-MLE to the choice of function approximator in practice? Are there guidelines for selecting appropriate models?

Could the framework be extended to handle partially observable or non-stationary environments?

In what types of real-world continuous-time systems (e.g., finance, healthcare) do the authors expect the adaptive measurement mechanism to be most impactful?

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
3

### Summary
This paper analyzes the continuous-time RL setting where the dynamics is modelled as an SDE with both a drift and a diffusion terms. In this setting, the authors present an algorithm for minimizing the regret during interaction with the environment. Crucially, the algorithm is based on constructing two confidence sets around the max likelihood estimate of the dynamics, and then acting optimistically w.r.t. them. The paper next provides a theoretical analysis of the algorithm.

### Strengths
- The continuous time RL problem is both interesting and important for the RL community as it can tackle more realistic settings than standard MDPs
- The paper provides a theoretical analysis of the proposed algorithm, proving that its regret grows only logarithmically.

### Weaknesses
- The algorithm cannot be implemented in practice for meaningful and general classes of dynamics F,G and policies Pi. The authors address this in Appendix C with a relaxation of the considered optimization problems. However, this makes the algorithm loose its theoretical properties.
- The analysis assumes Pi,F,G to be finite. Although the authors mention in Remark 5.2 that the analysis can be extended to more general sets using the same arguments as in a previous work (Wang et al. 2024b), they should at least add an appendix where they derive these results explicitly. Indeed, unless the theoretical analysis of this work is merely copied by (Wang et al. 2024b), it is better to add the proof for the more interesting setting explicitly.

### Questions
- I do not get Figure 1. I mean, in the text you criticize equidistant measurements, but here all the three options regard equidistant measurements, and what changes is just the frequency of the measurements
- where does Eq. (4.1) comes from? I would expect a formal proof for it, where you show (or at least reference and formally present) the Markov property of the Ito process, and all the passages to obtain it.
- why the regret bound does not depend on the size of the action space?

typos:
- lines 101-104: Something is missing here
- definitions/propositions/theorems should be all italics

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
3

### Summary
This paper studies instance-dependent guarantees for continuous-time reinforcement learning (CTRL). Under some conditions, it establishes an instance-dependent second-order regret bound for CTRL. The results provides some new insights for CTRL, including robustness on choice of measurements and weaker horizon dependence compared with prior related works.

### Strengths
1. Interesting and novel idea: While instance-dependent analysis has been studied in standard discrete-time RL, it is interesting to see how it adapts to continuous-time RL, where irregular measurement times introduce an extra layer of complexity. This paper tackles that setting.

2. New insights for CTRL: By analyzing instance-dependent regret, the paper offers guidance for designing measurement schedules. In particular, as long as the total measurement budget is upper bounded by the total cumulative variance (the intrinsic difficulty of the problem), it is quite flexible to design measurement strategies. The regret also exhibits only logarithmic dependence on the horizon T under the assumption that cumulative reward is bounded by 1, in contrast to the exponential dependence reported in some prior work.

3. Solid theoretical development: The proofs and main results appear reasonable (although I did not check every step in detail).

4. Empirical illustration: Although this is primarily a theoretical paper, it includes numerical experiments to support the claims.

### Weaknesses
1. Limited intuitive exposition in the main text: The theoretical results are dense. I suggest adding a brief proof sketch in the main paper. It would also help to discuss the key challenges and obstacles specific to the continuous-time setting, and how they differ from standard discrete-time RL (e.g., [1]).

2. My biggest concern is the quadratic density assumption (Proposition 5.9). Although Appendix A.3 provides some discussion, a more general treatment would strengthen the paper. If possible, I suggest the authors to include additional examples or at least more intuition for broader density families. It seems plausible that the Eluder dimension remains tractable for richer classes, but the paper does not currently justify this.

[1] Model-based RL as a Minimalist Approach to Horizon-Free and Second-Order Bounds. Wang et. al.

### Questions
If model misspecification is present, how would it affect the algorithm and the theoretical guarantees? To what extent do the main insights reported here still hold (e.g., grid-shape insensitivity, variance-driven behavior)? These questions are optional.

### Soundness
3

### Presentation
3

### Contribution
3
