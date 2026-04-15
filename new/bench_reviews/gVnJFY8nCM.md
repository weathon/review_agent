Now let me look at the human review calibration papers to better anchor the score.Now I have sufficient context to write the final consolidated review. Let me verify one more key claim about the dynamics model training requirements:---

## Summary

Residual-MPPI proposes an online planning algorithm for customizing continuous-control policies at execution time by integrating Residual Q-Learning (RQL) into the MPPI framework. The core idea is to treat the log-likelihood of a prior policy as a reward surrogate for the unknown original task reward, enabling planning over an augmented objective without retraining the policy. The method is evaluated on four MuJoCo environments and on the champion-level racing agent GT Sophy 1.0 in Gran Turismo Sport (GTS), where it successfully customizes the policy toward safer route selection with significantly fewer laps of data than RL-based alternatives.

---

## Strengths

- **Practically significant problem with principled formulation.** Policy customization without access to the original reward function or policy retraining is a real deployment need, and the RQL reduction to an augmented MDP with log π as a reward is theoretically grounded and conceptually elegant.
- **Comprehensive, privileged baselines.** Guided-MPPI and Valued-MPPI have access to the original reward/value function (information unavailable to Residual-MPPI), making their inclusion genuinely strong baselines. The fact that Residual-MPPI outperforms them establishes a meaningful contribution.
- **Compelling GTS demonstration.** Customizing GT Sophy 1.0 with only ~2,000 laps of dynamics training (vs. ~80,000 for Residual-SAC to converge) is a striking data-efficiency result on a realistic, high-fidelity simulator. Guided-MPPI and Greedy-MPPI fail entirely in this environment, directly validating the necessity of the log π term where it matters most.
- **Sample efficiency claim is well-supported.** The comparison against Residual-SAC at equivalent data budgets (200K steps vs. 4M steps) clearly establishes the advantage of the planning-based approach in low-data regimes.
- **Theoretical foundation is present and honest about its limits.** Proposition 1 explicitly states its assumptions (γ=1, infinite-variance Gaussian), and the text uses "suggests" and "approximates" appropriately rather than claiming exactness.

---

## Weaknesses

### Fatal
*None. The paper makes a real, non-trivial contribution and the core mechanism works.*

### Major

- **Residual-MPPI is statistically indistinguishable from Greedy-MPPI in 3 of 4 MuJoCo environments.** From Table 1: HalfCheetah (1936.2±109.3 vs 1939.9±134.7), Swimmer (−60.0±5.2 vs −58.9±5.4), Hopper (7363.0±254.9 vs 7367.0±199.4) are all within overlapping standard deviations. Only Ant shows a clear win (6846.7 vs 6104.2), with the paper's own explanation being that the add-on reward there (y-axis velocity) is genuinely orthogonal to the basic reward. The paper argues the log π term is "the key factor" and "necessity," but 3 of 4 tasks fail to demonstrate this. The MuJoCo add-on tasks (joint angle penalty, height bonus) appear insufficiently challenging to reveal the difference. The GTS result does validate the log π term decisively, but the MuJoCo results undercut the broad narrative about its necessity. The paper should either redesign the MuJoCo tasks to be more discriminating, or more honestly qualify the scope of the MuJoCo claim.

- **No wall-clock inference timing reported for GTS.** The paper's central practical claim is online, deployment-time customization. MPPI with a neural dynamics model requires many parallel rollouts per control step at a fixed frequency in GTS. The absence of any latency or computation-time analysis leaves the "online" feasibility claim unvalidated. For a real-time control setting like racing, this is not optional information.

### Minor

- **Theory-practice gap acknowledged but underdiscussed.** Proposition 1 holds under γ=1 and infinite Gaussian variance (effectively a uniform prior). Algorithm 1 uses γ < 1 and a finite-variance Gaussian. The paper handles this appropriately in Sec. 3.1 ("suggests" and "approximates"), but the GTS section then characterizes the log π term as "theoretically sound" without re-qualifying the approximation. A brief sentence acknowledging the residual approximation gap in the algorithm would improve precision.

- **Small evaluation sample in GTS.** Table 2 reports mean ± std over only 30 laps in a high-variance domain. Off-course steps (4.43±2.39 for few-shot, 9.03±3.33 for zero-shot) have high relative variance, and more laps would strengthen confidence in these values.

- **Abstract understates the dynamics model requirement.** The abstract states the method requires "access to the prior action distribution alone," but the method also requires a usable dynamics model, which in GTS required approximately 2,000 laps of training data. The contribution is genuinely about avoiding *policy retraining* (not all learning), and the abstract should be updated to say so precisely.

- **Residual-SAC framing in GTS is subjective.** The paper describes Residual-SAC (0.87 off-course steps) as "overly conservative" while accepting Residual-MPPI (4.43–9.03 off-course steps) as "safe enough." No target threshold for off-course safety is defined, making this a qualitative judgment rather than an objective evaluation. The paper should either define a safety specification or present this as a Pareto tradeoff between lap time and constraint violation.

- **Dynamics model quality not ablated.** The zero-shot variant's quality is directly bounded by how good the offline dynamics model is, yet there is no experiment showing how performance degrades with reduced model data or increased prediction error. This is relevant to the zero-shot claim and is a missing experiment.

### Trivial

- The paper refers to Residual-SAC and Fulltask-SAC as "upper bounds" when they operate in different data regimes and optimize different objectives; "reference points" is a more accurate label.

---

## Nice-to-Haves

- **Trade-off curves as ω' varies** — A sweep of the prior-vs-add-on weight would reveal whether customization is smooth and controllable, which is essential for practical deployment guidance.
- **More discriminating MuJoCo tasks** — Tasks with structurally orthogonal objectives (like Ant) better demonstrate the method's unique value; replacing some of the near-aligned add-on tasks (joint angle, height) would sharpen the narrative.
- **Multimodality analysis** — If the prior policy captures multimodal behavior, it would be informative to show whether Residual-MPPI preserves or collapses those modes during customization.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **Harsh Critic: "Algorithm 1, line 15 uses 1/λ instead of the normalization constant η."** The critic themselves acknowledge this "may be a parser or truncation issue." Given the hard rule on formatting/notation artifacts from parser issues, this is removed.

- **Harsh Critic: "ω(Ek) in Eq. (4) doesn't look like a normalized importance weight."** This is a presentation-level notation concern. The standard MPPI weighting is well-established in the literature; the paper's notation is a compact variant. Removed as a formatting/presentation nitpick.

- **Harsh Critic: "Full-MPPI is a strawman."** Full-MPPI's failure in continuous control without any policy prior is not trivially expected from the problem setup; it validates that the prior policy guidance is necessary. This is not an unfair comparison that advantages the authors — it is a meaningful ablation. The asymmetry (Full-MPPI loses to Residual-MPPI) is intentional and informative. Removed per hard rule.

- **Human Finder: "Missing comparison with offline-to-online RL baselines (Cal-QL, RLPD, IQL)."** These methods require task-specific training and are not applicable to the zero/few-shot online customization setting the paper targets. Comparing against them would require additional reward access the paper explicitly scopes out. Removed as out-of-scope.

- **Human Finder: "Comparison with TD-MPC2 or DreamerV3 for policy adaptation."** As the paper notes in Sec. 6, these methods require knowledge of the basic reward function and cannot be directly applied to the customization setting. Removed as misunderstanding of scope.

---

## Novel Insights

The synthesis across reviews points to a genuinely useful methodological insight: the log π term in Residual-MPPI is not just a regularizer (as Greedy-MPPI implicitly treats it by sampling from π) but an explicit reward signal that encodes long-horizon information about the original task. The difference between sampling from a prior and *evaluating* the prior's log-likelihood at each trajectory step is subtle but consequential — sampling from π provides implicit regularization that is direction-agnostic when the add-on reward is aligned with the prior task, but breaks down when they are orthogonal or structurally conflicting. The GTS failure of Greedy-MPPI is arguably the paper's clearest proof of this: route selection requires the planner to prefer trajectories that are globally competitive, not just locally not-off-course. The log π term provides the Q-value-like long-range signal that Greedy-MPPI lacks. This distinction between sampling-based and evaluation-based use of the prior is the paper's most insightful contribution and deserves more emphasis.

---

## Suggestions

1. **Redesign or augment 2–3 MuJoCo tasks to have structurally orthogonal objectives** (like Ant). This directly strengthens the core claim about log π being necessary and removes the ambiguity created by the 3/4 ties.
2. **Report wall-clock time per control step for GTS** (both with and without GPU batching for MPPI rollouts). This validates the "online" claim concretely.
3. **Replace "given access to the prior action distribution alone"** in the abstract with "given only access to the prior policy (without knowledge of the original reward or task parameters)" — accurate and still a strong selling point.
4. **Add a dynamics model quality sweep** (training data from 200 to 4000 steps) in appendix or main paper to show robustness of the zero-shot variant.
5. **Define a safety threshold** for GTS off-course steps and present the Residual-SAC comparison as a Pareto tradeoff rather than labeling one policy "too conservative."

---

## Score and Decision

**Calibration:**

| Paper | Topic | Human Scores | Decision |
|---|---|---|---|
| MWHIIWrWWu (MPC² hierarchical control) | MPPI for complex control | 5, 6, 8, 6 (avg ~6.3) | Accept Poster |
| e5jGTEiJMT (Policy Decorator) | Residual policy adaptation | 8, 5, 8 (avg ~7) | Accept Poster |
| JZCxlrwjZ8 (ADM dynamics model) | Dynamics model learning | 5, 5, 6, 8 (avg ~6) | Accept Poster |
| i7jAYFYDcM (BMPC) | MPC + policy learning | 6, 8, 8, 6 (avg ~7) | Accept Poster |

This paper is positioned comparably to or slightly above JZCxlrwjZ8/MWHIIWrWWu in terms of experimental impact (the GTS demo significantly exceeds any of the calibration papers in real-world ambition), but its MuJoCo results have the 3/4 Greedy-MPPI tie weakness that is more concerning than the issues in e5jGTEiJMT or i7jAYFYDcM. The key differentiator is that GTS unambiguously demonstrates the method's value, and the sample-efficiency claim is strongly supported. On balance, I place this at 6.0 — firmly in Accept Poster territory, with real but non-fatal weaknesses.

**Axis Evaluation:**
- *Originality:* Good — combining RQL with MPPI for black-box policy customization is a meaningful step not directly replicated by prior work
- *Importance of research question:* High — online deployment customization without retraining is a genuine practical need
- *Claims supported:* Partially — GTS results strongly support the method, MuJoCo support is weaker than claimed
- *Soundness of experiments:* Moderate — strong baselines, but GTS evaluation is small (30 laps) and inference cost is unreported
- *Clarity of writing:* Good overall, with some abstract overclaiming
- *Value to community:* Solid — both the GTS application and the planning framework for black-box policy customization are useful contributions

**Final Score: 6.0 (Weak Accept / Accept Poster)**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>