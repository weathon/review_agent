## Summary
This paper proposes AuxSS, a method for accelerating online reinforcement learning by constructing auxiliary start state distributions from expert state demonstrations (without requiring actions or rewards). The method uses episode length as a proxy for state safety—shorter episodes imply less safe states—and dynamically updates a sampling distribution over demonstration states to prioritize sampling from task-critical bottleneck regions. The approach is evaluated on sparse-reward hard-exploration tasks (Lava Bridge, Miniworld 3D Navigation) and dense-reward MuJoCo tasks, showing sample-efficiency gains and improved robustness to out-of-distribution start states.

## Strengths
- **Compelling problem formulation with minimal affordances:** Unlike competing hybrid RL methods (HySAC, JSRL) that require expert actions and rewards, AuxSS uses only expert state trajectories. Despite this reduced requirement, it matches or exceeds these methods on hard-exploration tasks (Section 5.1, Figure 3) and demonstrates strong sample efficiency with 15× less expert data (Section 5.4, Figure 6).

- **Well-designed ablation study validating the safety motivation:** Section 5.5 and 5.6 provide direct empirical evidence that safety-inspired distributions (AuxSS, Ω-SS) dramatically outperform non-safety alternatives (U-SS, GoalDist-SS) in both sample efficiency and robustness (Figure 7). The comparison between the static Ω-SS and dynamic AuxSS also validates the design choice of online adaptation over a fixed pre-computed distribution.

- **Effective demonstration on hard-exploration tasks:** The Lava Bridge results (Figure 3) clearly show that resampling from difficult regions overcomes the exploration stagnation typical of purely online SAC in sparse-reward environments, and AuxSS is the only method to consistently solve both the easy and hard exploration variants of 3D Navigation (Section 5.2).

## Weaknesses

### Fatal
None.

### Major
- **The smoothing kernel for image-based state spaces is underspecified:** Algorithm 1, Line 4 computes Gaussian smoothing via $\lambda \propto \exp(-(\mathcal{S}_{demo} - \mathcal{S}_{demo}[i])^2 / 2\sigma^2)$ using squared Euclidean distance. Section 5.2 evaluates on Miniworld 3D Navigation with image observations, but the paper never clarifies what representation is used to compute this distance. Computing squared Euclidean distance on raw pixels is highly problematic for defining "neighboring" states in high-dimensional visual spaces—it lacks shift and rotation invariance and suffers from the curse of dimensionality, making bandwidth $\sigma^2$ essentially impossible to tune meaningfully. If the authors instead used underlying physical coordinates (e.g., agent position) for distance while feeding images to the policy, this critical detail is omitted from the algorithm description and pseudocode, leaving the method opaque for the image domains central to the paper's contribution claims.

- **No empirical validation that episode length correlates with the theoretical safety metric:** Section 4 formally defines $\Omega_\pi(s)$ as the probability of non-termination after a $k$-step rollout, and proposes $(H - L_{ep})/H$ as a Monte Carlo approximation. However, the paper never demonstrates that this proxy actually correlates with $\Omega_\pi(s)$ or with empirical task hazard probability. The link between the weighting mechanism and the named "state safety" concept is asserted rather than demonstrated. Without this validation, it is unclear whether the success on hard-exploration tasks stems from the specific safety-motivated weighting or from the more general benefit of any distribution that simply avoids early terminal states.

### Minor
- **The weight update is a single-episode overwrite, not a running estimate:** Algorithm 1, Line 3 directly sets $\mathcal{W}[i] \leftarrow \max((H-L_{ep})/H, \delta)$ based on the outcome of a single trajectory. This means a single stochastic early termination permanently inflates the sampling weight, and a single lucky long episode deflates it. While the smoothing step (Line 5) propagates this signal to neighboring states, it does not reduce the fundamental variance of the per-state estimator. An exponential moving average or similar mechanism would be more stable, and the authors do not provide an ablation comparing the overwrite rule to a smoothed alternative.

### Trivial
- **The $\mu_{OOD}$ benchmark is only briefly described in the main text:** Section 3 mentions that the ID and OOD start state distributions "are shown in Figure 7" (which is actually the distribution ablation plot, not the state space definition), with a full equation deferred to Appendix 7.2. A brief description of how OOD states are constructed in the main text would improve reproducibility.

## Nice-to-Haves
- Visualizing how the sampling distribution $\mathcal{W}$ evolves across training on Lava Bridge (e.g., a heatmap over the maze layout) would help confirm whether it actually concentrates on bottleneck regions or converges to something more uniform.
- Including the original RLPD baseline alongside RLPD+ in comparisons would provide a more complete experimental landscape.
- Reporting confidence intervals or error bands on the training curves would strengthen the empirical claims, particularly for the μ_OOD success rates.

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- ~~"The robustness claim $\mathcal{J}_{\mu_{OOD}}$ relies on incomparable baseline configurations because baseline methods pre-fill replay buffers or use guide policies that alter the effective state distribution."~~ The paper explicitly accounts for this: in Section 5.1, the JSRL comparison only tracks rollout steps beyond the handover point, and Section 5.5 provides a clean comparison (U-SS) that isolates the weighting effect from the reset mechanism. The critic overlooked these sections.

- ~~"BARL's inclusion does not meaningfully demonstrate AuxSS's superiority over modern sparse-reward exploration methods (e.g., episodic curiosity or goal-conditioned RL)."~~ BARL is included because its setting (arbitrary resets, simulator access) closely matches the paper's setup, not as a general comparison claim. The paper correctly notes BARL's architectural limitations in sparse rewards as context. This is a scope-creep criticism.

- ~~"A uniform distribution updated via an EMA of visitation counts would serve as a cleaner control to prove the safety weighting adds value beyond simple dynamic resampling."~~ This is essentially the U-SS baseline already presented in Section 5.6 (uniform static distribution), which performs dramatically worse than AuxSS. The EMA variant would be a refinement, not a missing baseline.

- ~~"The weight update is a high-variance overwrite that permanently inflates/deflates sampling weights."~~ While technically true that this is an overwrite, the smoothing step (Line 5) does propagate information to neighbors, and the empirical results across multiple environments show sufficient stability. This is a minor methodological observation, not a structural flaw that undermines the contribution.

## Novel Insights
The paper makes a compelling case that when expert state trajectories and arbitrary-reset simulators are both available, the design of the start state distribution matters as much as the RL algorithm itself—a point underexplored in hybrid RL literature. The empirical finding that a static safety-motivated distribution (Ω-SS) initially outperforms dynamic AuxSS but later degrades reveals a subtle interaction: the set of task-critical states shifts as the policy improves, making online adaptation necessary for long-horizon robustness. This suggests a useful meta-insight for the broader exploration literature: exploration frontiers should be continually reassessed, not just discovered and frozen.

## Suggestions
- **Clarify the state representation used for Gaussian smoothing in image-based tasks.** If a low-level representation (e.g., agent coordinates) is used for distance computation while the policy receives images, make this explicit in Algorithm 1 and Section 5.2. If raw pixels are used, justify the choice and provide a sensitivity analysis on $\sigma^2$.
- **Report the empirical correlation between $(H - L_{ep})/H$ and a direct estimate of $\Omega_\pi(s)$ (e.g., the fraction of $k$-step rollouts that do not terminate from resampled states).** This would validate or refine the connection between the named "state safety" concept and the actual weighting mechanism.
- **Add a brief 1-2 sentence description of the μ_OOD construction in the main text** (position offset, obstacle randomization, etc.) so the robustness benchmark is self-contained.

## Score and Decision
I calibrated against several papers in the human-review corpus:
- **High-scoring anchors:** M3QXCOTTk4 (scores 8,6,8,8; avg 7.5) — a methodological RL paper with extensive Atari/MuJoCo experiments, novel empirical finding ("curse of diversity"), and comprehensive ablation. AuxSS falls below this due to underspecified methodology for image spaces and lack of proxy-to-theory validation.
- **Medium-scoring anchor:** lF2aip4Scn (scores 6,6,8,6; avg 6.5) — a hybrid RL paper with theoretical guarantees and some clarity gaps. AuxSS is empirically comparable but lacks theoretical backing, offset by its cleaner practical results.
- **Borderline anchor:** Ap344YqCcD (scores 6,5,6,5; avg 5.5) — an empirical IL+RL paper with solid results but questions about generalizability and fairness of comparisons. AuxSS has a similar profile.
- **Low-scoring anchor:** DCg9r2DKKe (scores 1,3,3,3; avg 2.5) — a safe RL paper rejected for fundamental novelty issues, misnamed STL use, and failure to cite prior work. AuxSS is substantially better than this, with no fatal flaws and a well-motivated empirical contribution.

AuxSS sits between the borderline and medium anchors. It has genuine empirical contributions (the ablation study validating safety-motivated distributions is strong) and well-executed experiments across multiple domains. However, the underspecificity of the smoothing kernel for image-based spaces and the absence of correlation analysis between the episode-length proxy and the theoretical safety definition are meaningful methodological gaps. This is not a paper with fatal flaws, but it is also not a paper with the comprehensive rigor of a 7+ acceptance. I position it slightly above the borderline cluster, at the lower end of acceptable.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>