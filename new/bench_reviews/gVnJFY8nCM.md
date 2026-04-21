Now I have a thorough understanding of the paper. Let me synthesize the review.

Key findings from my careful reading:

1. **The log π term vs Greedy-MPPI in MuJoCo**: In HalfCheetah, Swimmer, and Hopper, Residual-MPPI and Greedy-MPPI are essentially identical within standard deviations. The log π term only shows a meaningful advantage in Ant (where the add-on reward is orthogonal to the base). This is a valid concern, but the paper does discuss this in the context of the Ant environment specifically. The paper's framing of "effectiveness" across all environments is somewhat overclaimed.

2. **GTS baselines**: The text says "we only include Greedy-MPPI and Guided-MPPI" but Table 2 includes Full-MPPI too. This is a minor inconsistency. Also, the claim about Full-MPPI and Guided-MPPI having "privileged access to the underlying reward or value function" in Section 4.1 raises the question of how this works in GTS where they say they only have access to the policy network. But the paper seems to have access to the add-on reward in GTS (staying on course), and they could construct some form of Full-MPPI/Guided-MPPI with that.

3. **Proposition 1 assumptions**: The paper acknowledges γ close to 1 with large noise variance but doesn't provide error bounds or sensitivity analysis. This is a fair concern but is common in MPPI literature.

4. **The GTS result is genuinely compelling**: Reducing off-course steps from 93 to 4.43 with only marginal lap time increase, using ~2000 laps of dynamics data vs 80K+ laps for Residual-SAC, is a strong result.

5. **Greedy-MPPI and Guided-MPPI failing catastrophically in GTS**: The paper provides an explanation (finite horizon + long-term reasoning needed), but doesn't validate this with horizon ablations.

Let me now write the final review.</think>Now I have a complete picture. Let me write the consolidated review.

---

## Summary

Residual-MPPI proposes an online planning algorithm for customizing pre-trained continuous-control policies at execution time. By integrating Residual Q-Learning (RQL) into the Model Predictive Path Integral (MPPI) framework, the method uses the prior policy's log-likelihood (log π) as a reward bonus alongside an add-on reward, avoiding the need for the original reward function or value function. The method is validated in MuJoCo and compellingly in Gran Turismo Sport (GTS), where it customizes a champion-level racing agent to reduce off-course steps with minimal lap-time penalty, using orders of magnitude less data than RL-based alternatives.

## Strengths

- **Practical and well-motivated problem formulation**: The policy customization problem—adapting a prior policy to new objectives without access to the original reward or training details—is genuinely important and cleanly defined. The distinction from standard fine-tuning (which discards prior task performance) is clearly articulated (Sections 1, 2.1).

- **Compelling GTS demonstration**: Customizing GT Sophy 1.0 from 93 off-course steps to 4.43 with only a ~5% lap-time increase (Table 2) is a striking empirical result. The sample-efficiency advantage over Residual-SAC (~2K laps vs ~80K laps) is substantial and well-demonstrated. The trajectory visualizations (Figure 2) qualitatively confirm meaningful behavior change.

- **Strong sample efficiency in MuJoCo**: Residual-MPPI (2K steps of dynamics data) achieves total rewards comparable to Residual-SAC (4M steps) on HalfCheetah, Hopper, and Ant (Table 1), while Residual-SAC with matched data (200K steps) fails catastrophically. This directly supports the core claim of online, sample-efficient customization.

- **Outperforms baselines with privileged information**: Residual-MPPI outperforms Guided-MPPI and Valued-MPPI in MuJoCo (Table 1), which have access to the full reward function and prior value function respectively. This makes the advantage over baselines a stronger claim.

## Weaknesses

### Fatal
None.

### Major

- **The log π term—the paper's central mechanism—shows negligible benefit in 3/4 MuJoCo environments**: In HalfCheetah (1936.2 vs 1939.9), Swimmer (−60.0 vs −58.9), and Hopper (7363.0 vs 7367.0), Residual-MPPI and Greedy-MPPI (which omits log π) produce virtually identical total rewards within standard deviations. Only in Ant, where the add-on reward is orthogonal to the base reward, does log π provide a clear benefit (6846.7 vs 6104.2). The paper claims broad "effectiveness" but the core mechanism is only demonstrably necessary when add-on rewards require long-horizon reasoning that conflicts with prior objectives. The contribution's scope should be reframed or supported by additional environments that systematically vary the relationship between base and add-on rewards. The paper's own Section 4.2 acknowledges this for Ant specifically, but the limitation is not discussed for the other environments.

- **No analysis of why baselines fail catastrophically in GTS**: Table 2 shows Full-MPPI, Guided-MPPI, and Greedy-MPPI all failing to complete a single lap. While the paper attributes this to finite-horizon limitations (Section 5.2), no ablation (e.g., varying planning horizon T, reward scaling, or sample count) validates this explanation. Given that Guided-MPPI has access to the full reward and still fails, and that it succeeds in all MuJoCo tasks, the claim that "finite horizons" alone cause this failure needs support. Without ruling out implementation/tuning issues, the headline GTS result rests on unexplained baseline failures.

### Minor

- **Inconsistency in GTS baseline description**: Section 5.1 states "we only include Greedy-MPPI and Guided-MPPI as MPPI variants for comparison in the GTS environment," but Table 2 also lists Full-MPPI with "Failed" results. Additionally, Section 4.1 says baseline MPPI variants have "privileged access to the underlying reward or value function of the prior policy," yet in GTS the authors say they "only have access to the policy network." How Full-MPPI and Guided-MPPI obtain the full reward for GTS is not explained.

- **Proposition 1 relies on γ=1 and infinite noise variance, but uses γ<1 and finite variance in practice**: The paper acknowledges this (Section 3.1: "with γ close to 1 and a large noise variance") but "close" is not quantified. No sensitivity analysis or error bound is provided. While this approximation is common in MPPI literature, the paper's narrative treats the theoretical foundation as exact, which it is not.

- **Missing sensitivity analysis for ω'**: The hyperparameter controlling the trade-off between prior retention and add-on optimization is arguably the method's most important knob, yet no sweep or sensitivity curve is provided. Without this, it is unclear whether ω' requires careful per-task tuning or is robust.

### Trivial
None.

## Nice-to-Haves

- **Planning horizon ablation in GTS**: Varying T on at least one failing baseline would validate the claimed mechanism for Guided-MPPI's failure.
- **Analysis of when log π helps vs. doesn't**: Environments that systematically test orthogonal vs. aligned add-on rewards would sharpen the contribution's scope.
- **Testing with a non-SAC prior**: The theoretical derivation assumes π is the exact max-entropy optimal policy; testing with a PPO or behavioral cloning prior would test practical robustness.

## Removed Points

- **"Zero-shot" framing is misleading because 2K steps of dynamics data are required**: Removed. The paper explicitly defines zero-shot as "zero additional policy training" (Section 1) and describes the dynamics model requirement. The term is used consistently within the paper's scope; the 2K steps are for dynamics learning, distinct from policy training.

- **IS correction term in Algorithm 1 is not discounted but reward is**: Removed as a weakness. This is standard MPPI formulation (the original Williams et al. 2017 paper uses non-discounted IS weights with discounted rewards), and doesn't affect correctness of the algorithm.

- **Including the nominal action sequence as a candidate is a "practical trick" not part of the theory**: Removed. This is a well-known and standard technique in MPPI implementations (Williams et al., 2017 include this), and calling it out as a weakness is not appropriate.

- **Dynamics pipeline techniques are "standard in model-based RL"**: Removed. This is a strawman weakness—using established techniques does not diminish the paper's contribution.

- **Residual-SAC comparison is asymmetric (4M steps vs 2K dynamics steps)**: Removed with caution. The comparison is explicitly designed to show Residual-MPPI's sample efficiency advantage, and the paper includes both 200K-step and 4M-step SAC baselines for transparency. The asymmetry favors the baseline (SAC gets more data), which makes the comparison stronger, not weaker, per our rules.

- **Reproducibility concern about how GTS baselines obtain the full reward**: This is partially kept (in Minor) as an inconsistency concern, but the broader reproducibility claim is removed since the paper cites GT Sophy as a available system.

- **Minor formatting issues**: Removed per rules.

## Novel Insights

The paper reveals an interesting pattern that deserves more acknowledgment: the log π term's benefit is environment-dependent. In tasks where the add-on reward is largely aligned with or easily accommodated by the base reward structure (HalfCheetah joint penalty, Swimmer joint penalty, Hopper height bonus), sampling from the prior already provides sufficient regularization, and explicitly including log π adds nothing. Only when the add-on reward is structurally orthogonal (Ant y-axis movement) or the task requires long-horizon reasoning beyond the planner's horizon (GTS route selection) does log π become essential. This suggests the contribution is more precisely characterized as a method for "orthogonal add-on tasks with long-horizon dependencies" rather than "broad policy customization."

## Suggestions

- Reframe the contribution's scope: explicitly state that log π is most beneficial when the add-on reward conflicts with the prior or requires long-horizon reasoning, and acknowledge that in simpler MuJoCo environments, prior-policy sampling alone provides sufficient regularization.
- Add a planning horizon ablation in GTS to demonstrate that Guided-MPPI's failure is indeed due to finite horizons rather than hyperparameter issues.
- Provide ω' sensitivity analysis (even a simple sweep across a few values) to demonstrate robustness or document the tuning requirement.

## Evaluation

**Originality**: Moderate-to-good. The integration of RQL into MPPI via the log π objective is clean and the Proposition 1 bridge (while relying on prior work for the MDP equivalence) provides theoretical grounding. The problem formulation (policy customization without the original reward) is timely and well-defined.

**Importance of research question**: High. Online policy adaptation without retraining is practically important, and the GTS application demonstrates real-world relevance.

**Claims well-supported**: Partially. The GTS and MuJoCo sample-efficiency results are strong, but the generality of the log π mechanism is overstated relative to the MuJoCo evidence, and the baseline failures in GTS are insufficiently analyzed.

**Soundness of experiments**: Moderate. The core comparisons are well-structured, but the lack of horizon ablation in GTS and absence of ω' sensitivity analysis leave key empirical questions open.

**Clarity**: Good. The paper is well-structured with clear algorithm presentation (Algorithm 1) and well-designed tables.

**Value to community**: Solid. The method addresses a real practical need and the GTS demonstration is compelling enough to interest both the planning and RL communities.

## Calibration Anchors

1. **H-GAP** (avg 7.33, Accept Spotlight): MPC-based planning for humanoid control with strong zero-shot transfer. Residual-MPPI is somewhat weaker—H-GAP has more consistent advantages over baselines across tasks, whereas Residual-MPPI's key mechanism shows negligible benefit in 3/4 MuJoCo tasks. Both have compelling application demonstrations.

2. **DLPA** (avg 6.20, Reject): MPPI-based model-based RL for novel action spaces. Weaker theoretical grounding than Residual-MPPI, but similar pattern of strong empirical claim with incomplete analysis. Residual-MPPI has the stronger GTS demonstration.

3. **ROAM / Adapt On-the-Go** (avg 5.67, Reject): Online adaptation of pre-trained policies at test time. Similar problem setting but with methodological weaknesses. Residual-MPPI is more rigorous and has a more compelling application domain.

4. **Multi-timestep MBRL** (avg 2.50, Reject): Overclaimed empirical results from limited environments. Residual-MPPI is far stronger than this—multiple environments, strong GTS result, clear motivation.

5. **DiffTOP** (avg 8.00, Reject despite score): Differentiable trajectory optimization as policy class. Very strong technical contribution with extensive evaluation. The "Reject" decision despite 8.0 avg score indicates topic-specific considerations. Residual-MPPI is not at this technical level.

Residual-MPPI is clearly above the reject-quality papers (DLPA, ROAM, multi-timestep MBRL) due to its compelling GTS result and clean problem formulation. It falls below the top-tier accept papers (H-GAP, DiffTOP) primarily because its central mechanism shows limited benefit in simpler environments and baseline failures in its strongest result are unanalyzed. The paper sits in the borderline-to-accept range: it makes a real contribution with genuine practical value, but the scope of that contribution is somewhat narrower than claimed.

## Score and Decision

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Borderline</orange>