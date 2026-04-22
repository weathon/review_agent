# Exploring the Entropy Mechanism in LLM Agents On-policy Optimization

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
Training LLM agents in multi-turn environments with sparse rewards, where completing a single task requires 30+ turns of interaction within an episode, presents a fundamental challenge for reinforcement learning. We identify a critical failure mode unique to this setting: the exploration-exploitation cascade failure. This cascade begins with early-stage policy premature convergence, where sparse feed-back causes agents to commit to flawed, low-entropy strategies. Subsequently, agents enter late-stage policy collapse, where conventional entropy regularization becomes counterproductive, promoting chaotic exploration that destabilizes training. We propose Entropy-regularized Policy Optimization (EPO), a general framework that breaks this failure cycle through three synergistic mechanisms:
(1) adopting entropy regularization in multi-turn settings to enhance exploration, (2) an entropy smoothing regularizer that bounds policy entropy within historical averages to prevent abrupt fluctuations, and (3) adaptive phase-based weighting that balances exploration and exploitation across training. Our analysis validates that EPO mitigates entropy variance through smoothing regularizars, which suppresses the oscillations. EPO achieves up to 152% performance improvement on ScienceWorld and up to 19.8% on ALFWorld. Our work demonstrates that multi-turn sparse-reward settings require fundamentally different entropy control than traditional RL, with broad implications for LLM agent training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores instability in large language model (LLM) agent training under multi-turn sparse-reward environments and identifies a phenomenon called the exploration–exploitation cascade failure. To address it, the authors propose Entropy-regularized Policy Optimization (EPO), integrating: **trajectory-level entropy regularization**, **historical-anchored entropy-smoothing penalty**, and **an adaptive coefficient schedule $\beta_k$**. Theoretical analysis claims an improved performance bound and monotonic entropy-variance reduction. Experiments on ScienceWorld and ALFWorld report up to +152% improvement over PPO and GRPO baselines.

### Strengths
**Novel diagnosis:** Identifies a two-phase failure (early blind exploration → late uncertainty propagation), advancing understanding of entropy dynamics in long-horizon on-policy training.

**Methodological integration:** Combines entropy control and adaptive scheduling into a coherent framework compatible with PPO-style optimization. Empirical performance: Achieves consistent improvements on two complex environments.

**Readable and structured:** Algorithm and ablation setups are clear and reproducible.

### Weaknesses
**Theoretical Rigor:**
1. Proposition 5 introduces $|\mathcal{𝐷}|$ and $\mathcal{C}_{\lambda,\beta}$, but only describes them as “problem-dependent.” Their mathematical meanings and relations to observable quantities are never formalized. Appendix B lists numerical values, but these correspond to hyperparameters rather than defined constants in the theorem. 
2. Eq.(5) uses a piecewise-constant indicator function that is non-differentiable at entropy bounds $\(\kappa_{l}\bar{H},\kappa_{r}\bar{H}\)$, contradicting the smoothness assumption required for the Lipschitz continuity later claimed. 
3. The bound assumes $\|\nabla V_{\lambda,\beta}^{\pi_\theta}(s_0)\|\leq\epsilon$, but under softmax parameterization with large action spaces, minimal action probabilities can approach zero, making Lipschitz constants unbounded.
4. The “entropy-smoothing gap” compares expectations under $\pi_θ$ and $\pi*$, whose distributions differ; without importance weighting or proximity assumption, the comparison is mathematically
inconsistent.
5. The term $\frac{1}{2\lambda}\frac{|\mathcal{D}|^2}{\mathcal{C}^{\pi_\theta}_{\lambda}(s_0)}\epsilon^2$ mixes data-scale and regularization constants; the lack of normalization or scaling clarification undermines interpretability.

**Methodological Incompleteness:**
1. The method defines $W_k$ = $\bar{H}, ...,\bar{H}_{k-1} $  as a cumulative mean window rather than a fixed-length sliding window. This induces long-memory bias and may delay adaptation.
2. The monotonic entropy-variance reduction is asserted but not empirically demonstrated, and the appendix does not specify which lemma supports it.

**Presentation and Symbol Consistency:**

The paper does not specify whether $\Phi < 0$ implies degradation or if $\beta_k$ dynamically adjusts to prevent it.

### Questions
1. How are  $|\mathcal{𝐷}|$ and $\mathcal{C}_{\lambda,\beta}$ estimated in practice?
2. Is the indicator penalty in Eq.(5) replaced by a smooth surrogate during implementation?
3. What is the cumulative averaging behavior of $𝑊_𝑘$?
4. Have you empirically verified monotonic entropy-variance decrease? 
5. Under what conditions does EPO fail or degrade?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the instability of exploration in long-horizon episodic decision-making tasks, arguing that entropy in policy learning may fluctuate excessively, causing cascading failures. To address this, the authors introduce Episodic Policy Optimization (EPO): a method that (i) computes average entropy across trajectories, (ii) introduces a smoothing regularizer based on historical entropy statistics, and (iii) adjusts exploration weight β dynamically. Experiments are conducted on several RL benchmarks and code is released.

### Strengths
1. Clear writing & smooth logical flow

    The paper is overall well structured, and the motivation–method–experiment organization is consistent.

2. Code availability

    The authors provide open-source code, which is always beneficial for community verification and reproducibility.

### Weaknesses
1. Limited novelty: it is essentially a technical paper. The proposed method largely stitches together standard RL tricks (entropy regularization, episodic averaging, heuristic weighting) rather than presenting a fundamentally new idea. The contribution is incremental engineering at best. There is no principled theoretical insight bridging entropy dynamics and long-horizon failures beyond high-level motivation.

2. Equation (6) has zero gradient: which means that the  core method does not work mathematically.
Equation (6) uses a hard threshold indicator penalty
$P_{n,t,i} \in {(0,\alpha)}$, which is piecewise constant w.r.t. entropy 𝐻. Therefore, almost everywhere zero gradient w.r.t. policy parameters $\theta$. This means the core contribution has no optimization effect and all claimed behavior improvement must come from other terms. Thus the formal objective contradicts the claimed mechanism. This is a fundamental and fatal flaw.

3. $β$ design is purely heuristic: the dynamic 
𝛽
schedule is described in narrative form without theoretical grounding nor stability analysis. The schedule resembles a manually tuned hyperparameter annealing, lacking justification or ablation isolating its individual effect.

4. Section 4.1 method has no novelty: the episodic entropy computation in Sec.4.1 is simply taking an average over timesteps instead of per-step entropy. This is a trivial formulation change that should be introduced in preliminaries, not a methodological innovation.


5. Paper’s core assumption is unjustified: 
The supposed phenomenon — entropy fluctuation directly causing cascading failures in long-horizon RL — is asserted without rigorous empirical evidence or theoretical reasoning. The link between entropy and cascade failure remains speculative: there is no causal analysis,
no stress test isolating entropy as the factor and no ablation proving removing fluctuations solves the problem.
The claim reads opinion-driven, not science-driven.

6. Evaluation does not demonstrate the claimed benefit:
The paper states solving “catastrophic cascade failure”, but experiments are just on standard benchmarks, not cascade-failure-specific settings.
Thus the problem the paper aims to solve is not convincingly demonstrated.

### Questions
Please see weaknesses above. More analysis and experiments are required.

### Soundness
2

### Presentation
2

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
Traditional entropy regularization methods in reinforcement learning, even when adapted for LLMs, are fundamentally unsuited for multi-turn agent environments because they trigger an exploration-exploitation cascade failure. To address this, the authors propose Entropy-regularized Policy Optimization (EPO), a novel framework that introduces temporal awareness by anchoring the policy's entropy to a dynamically adjusted historical bound, which provides the stability needed to halt the cascade failure while still maintaining essential exploration.

### Strengths
This works introduces Entropy-regularized Policy Optimization (EPO) as a direct solution to this cascade failure. The work's central innovation is identifying that standard methods lack "temporal awareness." Its solution is built around this insight. EPO introduces a concrete mechanism—anchoring the policy's entropy to a dynamically adjusted historical bound—to provide stability and control cross-step entropy propagation.

### Weaknesses
1. The exploration-exploitation dilemma is a challenging problem in reinforcement learning. To address exploration-exploitation failures in multi-turn interactions, the authors adapt entropy regularization. Instead of calculating entropy per step (as in traditional RL), they compute it over the entire trajectory (across all turns) and then average these trajectory entropies across the batch. This method aims to better capture the long-term, compounding effects of early decisions. However, from the definition of the entropy-regularized policy loss, the policy loss is designed to reduce the bias between the real entropy and the expected entropy. The problem is whether the expected entropy will converge to a single value or a fixed range, preventing further exploration.

2. The authors claim that entropy-controlled LLM methods are fundamentally inadequate for multi-turn agent environments because they suffer from an "exploration-exploitation cascade failure." Sparse rewards and standard entropy regularization cause uncontrolled entropy growth early in the trajectory, leading to unstable and suboptimal decision patterns. The uncertainty accumulated from these flawed early steps then compounds, destabilizing the agent's behavior in later turns. It is curious why early-stage entropy exploration leads to accumulated uncertainty. Can the accumulated uncertainty be measured? Will late-stage entropy increase the accumulated uncertainty? Whether the early high entropy is caused by immature agent policies?

### Questions
Please see the Weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies the phenomenon of exploration-exploitation cascade in RL for training LLMs: policy converges at an early state prematurely, and standard PPO/GRPO-style training first lets early steps’ entropy blow up, and then this early chaos “propagates” to later steps, so the policy never settles into a coherent strategy even though entropy regularization is present. To handle this, the authors propose Entropy-regularized Policy Optimization (EPO), which has three pieces: (i) compute entropy trajectory-wise (across all turns) rather than per-step; (ii) add an entropy smoothing regularizer that pulls current entropy into a band around a historical moving average; and (iii) use an adaptive, phase-based weight so exploration is conservative at the start, balanced in the middle, and stabilized at the end.

### Strengths
* The paper is well-organized and clearly written. The paper has clean and good illustrations for their takeaways and empirical findings.
* The trajectory based entropy and entropy smoothing is novel to me---it provides a self-stabilizing strategy that is fairly interesting.

### Weaknesses
Major comments:

* I apologize for my unfamiliarity with the empirical studies---however, it seems to me that the baselines don't include the obvious alternative: single-step entropy PPO. Since the main pitch of the paper is "standard PPO/GRPO entropy is temporally blind" is unstable, the right comparison should be PPO with a per-turn entropy cap. Right now, if I understand correctly, the comparisons are to pure PPO/GRPO and to other agent RL methods, not to the simplest entropy-aware PPO.
* I think the name of the method "Entropy-regularized Policy Optimization" is fairly misleading. The novelty isn't about entropy regularization, but a path-dependent smoothing entropy regularization. Another name like "Entropy-smoothing Policy Optimization" may be more appropriate.

### Questions
* My main questions are in the weaknesses section.
* How does the hyperparameter affects the empirical performance? If the problem is over-exploration due to sparse rewards, what if comparing to a per-turn entropy capped PPO but decaying penalty with turn?

### Soundness
3

### Presentation
2

### Contribution
3
