=== CALIBRATION EXAMPLE 7 ===

# Final Consolidated Review
## Summary
This paper proposes Guided Reinforcement Learning (GRL), which mixes a guide policy and a learner policy using a derived guide-sampling rate intended to keep online performance above a user-specified degradation threshold. It further introduces GRL-RB, a rollback variant that restores a previous sampling rate when performance drops, and evaluates the approach on Combination Lock and AntMaze tasks using IQL as the learner.

## Strengths
- **The paper tackles a concrete and important control-transfer problem with a specific mechanism rather than a vague heuristic.** The central design choice is explicit: choose a guide-sampling rate from a target degradation threshold, then gradually transfer control to the learner. This is more principled than the static 25/75% schedules and simple linear decay baselines the paper implements.
- **The theory is internally coherent for the scoped settings it actually analyzes.** The three derivation variants are not arbitrary; they build progressively from an optimal guide / terminating learner case to non-optimal guide and dense-reward settings, and the paper explicitly notes relationships such as Eq. 4 reducing to Eq. 3 when \(\beta_l=1\).
- **The paper is unusually explicit about the key failure mode of its own theory.** It clearly states in multiple places that GRL’s guarantee depends on convergence between roll-in steps: e.g., Sec. 1 says “The most challenging of these assumptions is that the agent is able to converge fully between roll-in steps,” and Sec. 3.1.1 states “The assumption of convergence between steps of \(\alpha\) is key to the success of GRL.” This honesty makes the practical role of GRL-RB easier to interpret.
- **GRL-RB is a simple and practically meaningful extension of GRL.** The rollback mechanism is easy to layer on top of an existing learner and directly addresses the observed failure mode of over-aggressive transfer. Figure 4 provides a concrete demonstration that rollback can recover after threshold violations in a way vanilla GRL cannot.
- **The AntMaze results suggest the method is not purely a toy-environment artifact.** Although limited in scope, the comparisons in Figure 5 against JSRL, LD, and IQL indicate that the rollback-based schedule can improve warm-started online fine-tuning on a standard sparse-reward benchmark family.

## Weaknesses
### Fatal
- None.

### Major:
- **The paper’s headline framing overstates the scope of its “performance guarantee.”** The abstract, introduction, and conclusion repeatedly present the method as guaranteeing performance above a user-defined threshold, but the actual derivations are only for narrowly structured settings and rely on a strong convergence assumption that the paper itself says is difficult to satisfy in complex environments. This is not a small caveat. The paper states in Sec. 3.1.1 that “The assumption of convergence between steps of \(\alpha\) is key to the success of GRL,” and introduces GRL-RB precisely because this assumption is hard to meet. As a result, the guarantee applies to the scoped GRL analysis, not to the practical behavior of GRL-RB on the main complex benchmark.
- **The main AntMaze comparison does not directly evaluate the claimed threshold-maintenance objective on the same metric used to define that threshold.** The paper is explicit in the Figure 5 caption that although the threshold was computed from the stepwise reward \(r=-1\), it reports “the standard normalized AntMaze scores for better comparison with the literature.” That makes Figure 5 unsuitable as evidence that the threshold was maintained, even if it is useful as a task-performance comparison. Since maintaining a user-defined degradation threshold is a central claim, the main benchmark should also report results on that thresholded quantity, or at least explicitly show threshold violations/frequency under the same metric.
- **The empirical support is too narrow for the breadth of the claims about robustness, flexibility, and applicability “on top of existing algorithms.”** In practice, the paper evaluates one toy family tailored to the derivation and one non-toy family (AntMaze), all with a single RL backbone (IQL for the substantive experiments). That is promising but not enough to support broader claims such as robustness to hyperparameters, effectiveness as a general warm-starting approach, or flexibility across guide types and learners.
- **A practically critical quantity, \(\beta_l\), is hard to choose and the paper does not sufficiently characterize sensitivity to that choice.** The paper acknowledges this directly in Sec. 3.2.3 (“it can be difficult to choose an appropriate \(\beta_l\)”), and then fixes \(\beta_l=0.1\) conservatively for AntMaze. Because the sampling schedule depends on this quantity, the absence of a sensitivity study leaves it unclear how brittle the derived \(\alpha\) is in realistic settings.

### Minor
- **The flexibility claim regarding guide-policy format is only weakly validated experimentally.** The paper claims the guide can be “heuristics, decision trees, a policy learned through imitation learning/offline RL etc.,” but the experiments use either an oracle guide in Combination Lock or a learned offline policy in AntMaze. This supports some flexibility, but not the broader claim across heterogeneous guide formats.
- **The rollback mechanism is reactive, not preventive.** The paper acknowledges this in Sec. 5: “the roll-back is only triggered once the score has fallen below the threshold.” For safety-critical applications, a method that only recovers after a violation is materially weaker than one that prevents violations.
- **The paper does not analyze the operational cost of rollback.** It would be helpful to know how often rollback triggers, how much transfer speed is lost, and whether GRL-RB can stall near conservative sampling rates on difficult tasks. This matters for assessing the practical tradeoff between safety and transfer efficiency.
- **The positive dense-reward case is less actionable than the others.** Eq. 7 is presented as an inequality rather than a comparably neat closed-form schedule, which weakens the practical contribution of that branch of the theory.

### Trivial
- **Some core method details are deferred to appendices.** Since the curriculum progression and exact update logic matter for interpreting both the guarantee and the rollback behavior, a slightly more self-contained main-text description would improve clarity.

## Nice-to-Haves
- Add at least one additional non-AntMaze environment family to test whether the method generalizes beyond the sparse-navigation setting.
- Report AntMaze results both in normalized score and in the exact threshold metric used by the method, including the number/frequency/magnitude of threshold violations.
- Provide a sensitivity analysis for \(\beta_l\), evaluation cadence, and rollback settings.
- Include one experiment with a genuinely non-neural guide policy (e.g., rules or heuristics) to support the flexibility claim.
- Add ablations separating the effects of derived \(\alpha\), curriculum transfer, and rollback.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the paper is not honest about its assumptions.** Removed because the paper actually does acknowledge the main limitation repeatedly and clearly; the issue is over-framing, not hidden assumptions.
- **Complaint about unfair comparisons because some baselines are weaker or hand-designed.** Weakened/removed as a core criticism because the asymmetry does not obviously favor the proposed method in a way that invalidates the evidence; rather, the broader issue is insufficient empirical breadth.
- **Pure requests for more related work comparisons by naming specific outside methods.** Removed from the core review because missing related works cannot be verified here. The real issue retained is limited empirical coverage, not citation completeness.
- **Reproducibility nitpicks about omitted implementation details or appendix placement.** Removed as non-substantive; the paper provides algorithms in appendices and cites implementation sources.
- **Speculation that AntMaze plots may be single runs or improperly aggregated.** Removed because the paper text provided here is insufficient to verify that claim for Figure 5, and it would be inappropriate to infer misconduct or poor reporting without evidence.

## Novel Insights
The most important synthesis is that this paper is best understood as two different contributions with very different evidential status: (1) a scoped theoretical result showing that threshold-based guide sampling can be derived in highly structured settings, and (2) a practical rollback heuristic that appears useful when those assumptions fail. The paper’s current framing blends these together too aggressively. If separated more cleanly, the work would read as a credible practical method inspired by limited theory, rather than as a broadly guaranteed guided RL algorithm. That reframing would better match both the actual derivations and the empirical evidence.

## Suggestions
- Reframe the main claim more precisely: the guarantee is for GRL under explicit assumptions and structured reward/transition settings, while GRL-RB is a practical recovery mechanism without the same guarantee.
- In the main benchmark section, report the exact threshold metric used internally by the method, not only normalized AntMaze scores.
- Add sensitivity plots for \(\beta_l\) and evaluation frequency, since these directly control the derived sampling rate and rollback behavior.
- Broaden experiments to at least one additional environment family and, ideally, one additional learner backbone.
- Include one non-neural or rule-based guide to substantiate the claimed guide-format flexibility.
- Quantify rollback behavior directly: trigger counts, violation magnitude, time-to-recovery, and whether rollback slows or stalls transfer.

# Actual Human Scores
Individual reviewer scores: [3.0, 3.0, 3.0, 3.0]
Average score: 3.0
Binary outcome: Reject
