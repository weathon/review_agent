# On the Computational Limits of AI4S-RL : A Unified $\varepsilon$-$N$ Analysis

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Recent work increasingly adopts AI for Science (AI4S) models to replace expensive PDE solvers as simulation environments for reinforcement learning (RL), enabling faster training in complex physical control tasks. However, using approximate simulators introduces modeling errors that affect the learned policy. In this paper, we introduce a unified $\varepsilon$-$N$ framework that quantifies the minimal computational cost $N^*(\varepsilon)$ required for an AI4S model to ensure that tabular RL can estimate the value function with unbiasedness, with probability at least $1 - \delta$. This characterization allows us to connect surrogate accuracy, grid resolution, and RL policy quality under a shared probabilistic language.  We analyze how the discretization level $K$ of AI4S and RL space governs both PDE surrogate error and RL lattice approximation error, and we employ spectral theory and Sobolev estimates to derive optimal grid strategies that minimize total cost while preserving learning fidelity. Our theory reveals that different systems, such as ODE- and PDE-governed environments, require different allocations of effort between physical simulation and RL optimization. Overall, our framework offers a principled foundation for designing efficient, scalable, and cost-aware AI4S-RL systems with provable learning guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a theoretical framework connecting numerical resolution (in AI4Science simulators) and decision resolution (in RL agents). It claims an analytical form for the optimal resolution ratio and derives scaling laws linking discretization errors to computational cost. Experiments on a learned Cart-Pole dynamics model show non-monotonic performance trends with respect to resolution, suggesting the existence of an optimal middle point.

### Strengths
The paper bridges numerical simulation fidelity and RL resolution, a connection that is conceptually valuable for AI4Science.

The Cart-Pole experiments show that increasing resolution indefinitely does not always improve performance; there exists an intermediate optimum.

### Weaknesses
The central results (universal scaling, closed-form K* and N^1.6 deviation cost) are not empirically verified beyond the Cart-Pole example.

The paper lists tokamak, airfoil, and teppanyaki as examples but provides only theoretical discussion, no numerical results.

The Cart-Pole environment itself is a learned world model, so conclusions about “simulator–RL balance” may not directly generalize to physical simulators.

### Questions
How was the spectral constant estimated in practice? 

Can you provide at least one full experimental validation on a real PDE-based simulator (e.g., tokamak or CFD)?

How sensitive are the results to random seeds and network initialization?

Does the proposed cost function (Eq. 12) correlate with real compute cost (GPU-hours, memory, etc.)?

Could the N^{1.6} exponent be confirmed by sampling multiple off-optimal configurations instead of comparing two extremes?

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
2

### Summary
This work is concerned with analyzing the error introduced by using approximate simulators being used in solving complex physical control problems, governed by PDEs. The authors propose a principled framework, namely $\epsilon-N$ framework to quantify the minimum computational cost required for such approximate simulators to ensure that tabular RL estimates the value function without any bias, with a large probability. Through their theoretical framework, the authors provide optimal resolution trade-offs for some of the AI4S-RL systems such as Tokamak, and Cart-Pole.

### Strengths
1. It is important to understand how accurate the RL environment should be to compute optimal policy for the actual physical system, and this work tries to answer that. 

2. The paper is relatively easy to follow despite being heavy on theory.

### Weaknesses
1. I would have liked to see the empirical result in the main paper instead of it being in the appendix. 

2. I think a careful rearrangement of symbols is necessary as some of the symbols just appear without any definition. For example, $\lambda_1$ in line 374 is mentioned without context, although it is later mentioned in the appendix. $\lambda$, being a crucial parameter, might benefit from a short introduction.

### Questions
1. I understand that the discretization scheme we use will impact the solution we obtain. But, if we use continuous state and time representations, and assuming $\Delta t \approx 0$ (for state evolution),  do we face the similar issues? At least in the case of Cart-Pole, $\Delta t \approx 0$ doesn’t pose a huge computational burden. Is that correct?

2. Perhaps just use $w$ to denote the AI4S domain? $x_p$ just appears for one case; might be easier to read.

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
This paper addresses a fundamental and critical problem at the intersection of AI for Science (AI4S) and Reinforcement Learning (RL): how to manage the deterministic, resolution-dependent errors introduced when using AI-based surrogate models (like neural PDE solvers) as simulation environments for RL. Unlike stochastic noise, these discretization errors do not average out and can systematically bias the learned policy.

The authors' primary contribution is a novel theoretical framework, termed the "$\epsilon-N$ framework," designed to unify these two domains. This framework quantifies the minimal computational cost, $N^*(\epsilon)$, required to train an RL agent to a value function accuracy of $\epsilon$ with high probability.

### Strengths
The originality of this paper is outstanding. The problem itself is timely, but the authors' approach is, to my knowledge, new. Most work in this area either ignores surrogate error, treats it as generic bounded stochastic noise, or uses purely empirical tuning. This paper is the first I have seen to create a formal, unified theory that links PAC-RL sample complexity directly to the spectral and discretization properties of the underlying PDE. The $\rho-K$ analysis and the resulting formula for $K^*$ in Theorem 3 are highly novel and insightful. The technical quality of the theoretical work is exceptionally high. The authors demonstrate a deep and fluent command of both reinforcement learning theory and advanced numerical analysis for PDEs. The error propagation analysis (e.g., in Section 4.2 and Appendix C.2.2) is incredibly detailed, correctly identifying non-obvious error sources like the sublinear $\mathcal{O}(\Delta a_w^{1/2})$ error from boundary control actuation and the reduced $H^{1/2}$ regularity at boundaries (via the trace theorem). Deriving concrete, closed-form scaling laws from this complex analysis is a major technical achievement.

### Weaknesses
* Focus on Tabular RL: The entire theoretical framework is built on tabular, finite-state-and-action PAC-MDP analysis ($S_r, A_r$). However, the motivating examples (Tokamak, Airfoil) are high-dimensional, continuous systems that would never be solvable with tabular methods; they inherently require deep function approximation (e.g., Deep Q-Learning). The paper provides an empirical result for DQN (Fig. 2) and notes it is "consistent," but the theory does not extend to it. It is unclear how the Bellman error from function approximation would interact with the discretization errors ($\rho$). This is a significant gap between the theory and its practical application.

* Linear Dynamics Assumption ($\lambda_1$): The framework's reliance on $\lambda_1$, the dominant eigenvalue of the linearized operator, is elegant but is a simplification. Many complex systems (like turbulence in the airfoil example) are dominated by nonlinear error growth, not just linear modal instability.

* Vagueness in Cost Function Definition: The definition of the final $Cost(K)$ in Theorem 3 is slightly confusing. It appears to be $Cost_{RL} \times (\text{Penalty for } \rho)^2$, where $Cost_{RL}$ is the tabular RL sample complexity $\mathcal{O}(H_r^3 S_r A_r)$. This $Cost_{RL}$ term already scales with $K$ (via $S_r \sim K^\alpha, A_r \sim K^\beta$). However, the cost of a single surrogate call (which should be the primary driver of $N$ and scale with the AI4S grid, i.e., $\mathcal{O}(K^d)$) seems absent. The paper instead posits a "computational balance condition" $H_r^3 S_r A_r \sim H_w S_w A_w$ to relate the RL cost to the (absent) simulator cost. This feels like an assumption to make the optimization tractable, rather than a derivation of total computational cost.

### Questions
1. Could you please provide a more detailed justification for the $Cost(K)$ formulation in Theorem 3 and the "computational balance condition" $H_r^3 S_r A_r \sim H_w S_w A_w$? An alternative might be $Cost_{Total} = N_{episodes}(\rho) \times H_r \times Cost_{step}(K)$. How would the optimization of this (perhaps more direct) cost function compare to the results you presented?

2. You empirically show that DQN benefits from a similar resolution balance. Theoretically, how do you hypothesize the $\rho-K$ analysis would change when function approximation is introduced? Would the NN's approximation error add a new term, e.g., $\rho = \mathcal{O}(1/H_r + 1/K^d + \epsilon_{NN})$? Or is the interaction more complex, perhaps with the NN's generalization ability reducing the required $S_r$ and thus changing the optimal $\alpha$ and $\beta$ in Theorem 3?

3.  The analysis relies on a single $\lambda_1$ for the system. In many of the systems studied (e.g., Tokamak, Airfoil), the instability $\lambda_1$ is highly state-dependent (e.g., the plasma is more unstable in certain configurations). Does your framework assume a single "worst-case" $\lambda_1$ for the entire state space? And if so, does this suggest that a truly optimal system would require adaptive resolutions ($K(s), H_r(s)$) that change based on the local, state-dependent $\lambda_1(s)$?

4. This framework's analysis is centered on discretization error ($\rho$), which is assumed to vanish as resolution ($K, H_r$) increases. Nevertheless, the AI4S surrogates (such as neural operators), can also have systematic errors or model-form errors, if the surrogate is an imperfect representation of the true PDE over even infinite resolution (for example, due to a lack of training data or architectural limitations). How would this irreducible, non-vanishing error term modify the framework? Would it introduce a hard lower bound on the achievable accuracy $\epsilon$, regardless of computational cost $N$?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a theoretical framework to determine the forward prediction error of RL systems control when trained with a neural surrogate. It also provides expressions for the optimal resolution ratio $K^*$ to minimize computational cost of different PDE systems. The authors verify their theoretical results by applying tabular RL to the Cart-Pole environment.

### Strengths
- The paper has some novelty when combining ideas from PAC learning, spectral analysis of PDEs, and coupling the sources of error from RL and AI4S systems.
- The empirical results from the Cart-Pole experiment verify the results from their theoretical analysis.

### Weaknesses
- The paper only has empirical results for the Cart-Pole system, the theoretical results would be better justified if at least one other system was tested.
- The paper assumes the only source of error from the neural surrogate is deterministic discritization error, but this is not always the case for real world systems.

### Questions
- In the Cart-Pole experiment, do you train the RL with a neural surrogate model or just the physics simulator directly? Is this specified in the main text?
- If one of the main problems with RL to control PDE based systems is the expensive cost of using simulators to train the RL, doesn't the same problem exist to train a surrogate neural emulator of the simulator? Are you assuming the amount of data required from a simulator for surrogate training less than that required for RL?
- There are a few small misspellings in the paper which should be corrected:
  - Line 357: "To translate thees PAC-possible AI4S-RL analysis into practical system design guidance, this section establishes a computational resource allocation framework for AI4S-RL systems." Mispelling, should be "To translate these"
  - Line 386, Theorem 3: "yields optimal resolution raio between" misspelled "ratio".

### Soundness
3

### Presentation
2

### Contribution
3
