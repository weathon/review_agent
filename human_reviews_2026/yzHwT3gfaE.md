# Toward Conservative Planning from Human-AI Preferences in Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
We study reinforcement learning (RL) with trajectory preferences, where the RL agent does not receive explicit rewards at each step but instead receives human-AI preferences over pairs of trajectories. Despite growing interest in preference-based reinforcement learning (PbRL), contemporary works cannot robustly learn policies in offline settings with poor data coverage and often lack algorithmic tractability. We propose a novel **M**odel-based **C**onservative **P**lanning (MCP) algorithm for offline PbRL, which leverages a general function class and uses a tractable conservative learning framework to improve the policy upon an arbitrary reference policy. We prove that, MCP can compete with the best policy within data coverage when the reference policy is supported by the data. To the best of our knowledge, MCP is the first provably sample-efficient and computationally tractable offline PbRL algorithm under partial data coverage, without requiring known transition dynamics. We further demonstrate that, with certain structural properties in PbRL dynamics, our algorithm can effectively exploit these structures to relax the partial data coverage requirement and improve regret guarantees. We evaluate MCP on a comprehensive suite of human-in-the-loop benchmarks in Meta-World. Experimental results show that our algorithm achieves competitive performance compared to state-of-the-art offline PbRL algorithms. Our code is provided at https://github.com/Rshias/MCP.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors study offline reinforcement learning (RL) with trajectory preferences, where the RL agent does not receive explicit rewards at each step but instead receives human-provided preferences over pairs of trajectories. They propose a novel Model-based Conservative Planning (MCP) algorithm for offline PbRL, which leverages a general function class and uses a tractable conservative learning framework to improve the policy upon an arbitrary reference policy. They prove that, MCP can compete with the best policy within data coverage when the reference policy is supported by the data. Lastly, the authors conduct some empirical evaluations of MCP.

### Strengths
1. The problem of conservative planning in RLHF is an important and interesting problem.
2. The technical part of this paper is solid, the proof looks correct to me.
3. There are simulation results supporting the theoretical results.

### Weaknesses
1. My main concern is about the algorithm. The hyperparameter $\lambda_1$ and $\lambda_2$ depend on the concentrability coefficients, which further depends on the real reward or transition kernel. Would you please explain how to calculate (or estimate) such values efficiently?

2. Furthermore, $\lambda_2$ also depends on $M_P$, which is a max over all the intermediate steps during the training. How could the algorithm know this value before the algorithm starts?

3. The idea behind line 5 and 6 of alg.1 is interesting. Would you please analyze the computational complexity of solving the optimization problem in line 5?

### Questions
Please see the weakness

### Soundness
2

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
4

### Summary
This paper proposes Model-based Conservative Planning (MCP), a framework for offline preference-based reinforcement learning (PbRL) that integrates model learning with conservative optimization to ensure reliable performance under partial data coverage. The authors provide generalization guarantees under general function approximation and extend the analysis to structured settings such as kernelized nonlinear regulators. Empirically, the paper evaluates MCP on the Meta-World medium-replay benchmark, showing improved success rates over several preference-based baselines (PT, IPL, DPPO, APPO).

### Strengths
- The paper is clearly written and logically structured.

- The paper provides a solid theoretical framework for offline preference-based reinforcement learning under partial coverage. It establishes generalization guarantees for model-based conservative planning with general function approximation, which, to the best of my knowledge, has not been analyzed in prior PbRL literature.

### Weaknesses
The experimental evaluation exhibits certain limitations.

- The experiments are conducted solely on the Meta-World medium-replay benchmark, which primarily contains deterministic, low-dimensional robotic tasks. This restricted evaluation makes it difficult to assess how well the proposed conservative planning approach generalizes to more diverse or high-dimensional offline preference-based RL settings. Including broader datasets such as D4RL or human-feedback benchmarks (e.g., Atari or MuJoCo preference datasets) would strengthen the empirical evidence.

- The paper lacks ablations isolating the contribution of each design component, such as the relative performance regularization, model-based planning, and the choice of regularization weights.  Without these analyses, it is unclear how much improvement stems from the conservative objective itself versus other implementation factors. Including sensitivity studies or component removals would better validate the claimed effectiveness.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work proposes MCP, a novel model-based conservative planning algorithm for offline preference-based RL that is both sample-efficient and computationally tractable. Under partial coverage, MCP does not require additional structural assumptions and provides a PAC guarantee with a tractable implementation in model-based RL. To the best of the reviewer’s knowledge, this is the first work in model-based RL that addresses the limitations of PbRL. In experiments, MCP demonstrates high performance compared to other baselines.

### Strengths
1. The authors derive PAC bounds for the three variants of MCP (original, factored, and KNR). The paper is clearly structured, making it easy for readers to follow the overall flow and reasoning.

2. This work is a natural extension to the model-based setting and effectively addresses the limitations of previous approaches.

### Weaknesses
1. In Figure 1-(a), MCP is compared only with MR and Oracle. It would strengthen the validity of the theoretical claims if the experiments also included a comparison with APPO, as was done in the theoretical analysis regarding sample efficiency (Line 310).

2. The training curves are presented only for MCP. It would be beneficial to include those of other baselines as well, so that the results highlight not only the final performance but also the faster convergence of MCP, demonstrating the advantage of the model-based RL approach.

### Questions
1. The authors mention that MCP provides an implicit way of encoding conservatism, mainly through the minimax objective function. However, it is somewhat difficult to intuitively understand how this mechanism works in practice. Could the authors provide a simple or toy example to illustrate how the minimax structure implicitly enforces conservatism?

2. In Table 5, $\lambda_{1}$, $\lambda_{2}$, and $\lambda_{3}$ are important parameters for MCP. Therefore, it would be helpful to report their exact values for each dataset. Are these parameters highly sensitive?

### Soundness
3

### Presentation
2

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
This paper tackles incomplete data coverage and high computational cost in offline preference-based RL by introducing Model-based Conservative Planning (MCP). MCP uses a model-based planning framework that implicitly enforces conservatism, enabling sample-efficient and computationally tractable policy learning with general function approximation. Theoretically, MCP competes with the best policy supported by the dataset under partial coverage, with regret bounds improved via dynamic structures (e.g., kernelized nonlinear regulators, factorized models). Empirically, across 8 Meta-World tasks, MCP outperforms baselines like APPO and IPL in average rank and shows strong robustness in low-data regimes.

### Strengths
1. simultaneously achieve sample efficiency and computational tractability in offline PbRL with unknown dynamics and partial coverage. It encodes conservatism via relative performance instead of explicit confidence sets or extra value modeling, simplifying the framework and overcoming the computational bottlenecks of methods like FREEHAND and Sim-OPRL.

2. Supports general function approximation (linear models, neural networks, etc.), and derives adaptive concentration coefficients and regret bounds for structured dynamics (e.g., kernelized nonlinear regulators and factorized models), mitigating the curse of dimensionality and broadening theoretical applicability.

### Weaknesses
1. Compared with the APPO algorithm, this paper mainly introduces a model with conservative terms, but the experimental results do not show much improvement, while introducing additional model computational overhead.

2. What is the robustness of MCP when there is noise in preference labels? Quantitative analysis of the impact of label noise on performance can be conducted to prove that the design of MCP can have better generalization.

3. Each time a policy is updated, a new MDP model should be learned to challenge it, which is very wasteful in terms of compuational cost.

### Questions
see above.

### Soundness
3

### Presentation
3

### Contribution
3
