## Summary

The paper proposes Locally Constrained Policy Optimization (LCPO), an on-policy continual RL method that mitigates catastrophic forgetting in non-stationary, context-driven environments by constraining KL divergence on out-of-distribution (OOD) past experiences. LCPO requires neither task labels nor architecture changes, and is evaluated on synthetic and real-world context traces spanning MuJoCo, classic control, and a production straggler-mitigation task.

## Strengths

- **Conceptually clean and practically appealing formulation.** LCPO frames policy optimization as a constrained problem (Eq. 1) that anchors outputs on OOD historical samples using only a context-similarity metric. This removes the need for task labels or unstable off-policy rehearsal, and the same hyperparameters are used across all environments.
- **Real-world evaluation on production traces.** Table 1 shows that LCPO achieves tail latencies (1070 and 589) far closer to the prescient ideal (984 and 509) than any other online baseline on a Microsoft workload trace, demonstrating practical utility beyond synthetic benchmarks.
- **Broad experimental sweep and robustness evidence.** The evaluation spans six environments with diverse synthetic context processes (Ornstein-Uhlenbeck, piecewise Gaussian) and real traces. Ablations show LCPO is robust to OOD threshold variation (Fig. 3b) and maintains performance with buffers as small as 500 samples (Fig. 4).

## Weaknesses

### Fatal
*None.* The core mechanism—constraining policy updates via KL divergence on buffered OOD samples—is sound and supported by the gridworld experiments (Fig. 1d,e) and absolute straggler results. The evaluation flaws below do not fully invalidate the contribution, but they substantially weaken the quantitative claims.

### Major

- **Normalized return metric is ill-defined and undermines comparative claims.** §5.1 defines *Normalized Return* by scaling raw lifelong returns to [0, 1] using, *for each environment/trace*, the minimum and maximum returns **observed across all tested agents**. This metric is not anchored to environment-specific bounds (e.g., random policy or optimal return). Because scores depend on which other methods happen to be included, adding a poorly-performing baseline would inflate every other method’s score without improving absolute performance. Figure 3a—the paper’s primary comparative evidence—and the abstract’s claim that LCPO is “on-par with a prescient agent” and “the closest to this idealized baseline” rest on this relative metric. The straggler results in Table 1 provide absolute numbers, but they show LCPO is still 9–16% worse than prescient, which does not clearly substantiate “on-par.”
- **Continuous-control benchmarks were discretized without adequate disclosure.** §5 states: *“Gym environments were modified to accept discrete action space policies, as even prescient policies struggled to learn stable continuous space policies in the presence of contexts.”* This means the reported MuJoCo and classic control experiments were performed on fundamentally altered tasks, yet the abstract and introduction describe results on standard “MuJoCo, classic control” environments without mentioning this modification. Standard baselines such as SAC and TRPO are primarily designed for continuous action spaces; evaluating them on discretized tasks may misrepresent their performance. Moreover, the paper provides no evidence that findings from these discretized tasks transfer to the standard continuous-action setting. This is a major scope-limiting protocol decision that should have been highlighted in the abstract, introduction, and limitations.

### Minor

- **No ablation isolating OOD selection from generic replay regularization.** The paper’s core technical claim is that constraining KL divergence *specifically on OOD samples* (via $W(B_a, B_r)$) is the operative ingredient. However, there is no ablation against a baseline that applies the identical KL constraint on a *random* minibatch from $B_a$ without OOD screening. §5.3 shows LCPO works with as few as $n_b = 500$ samples across 8–20M-step traces; at this scale, the buffer cannot provide meaningful coverage of the state-context space, yet the constraint still functions. Without the random-buffer ablation, it remains unclear whether the gains stem from context-aware local anchoring or from generic trust-region regularization on stale data.
- **Limitations section omits the discretization restriction.** §6 discusses network capacity, exploration, and buffer management, but does not mention that all continuous-control experiments used discretized action spaces—a major restriction on applicability and interpretability.
- **Statistical significance is underreported.** Table 1 presents confidence ranges but no formal significance tests. On Workload 2, the confidence interval for LCPO Agg (589 ± 43) overlaps with A2C (604 ± 109), weakening the unqualified claim that LCPO outperforms all online baselines on this task.

### Trivial
*None.*

## Nice-to-Haves

- **Per-context return trajectories.** Showing episodic returns over time for individual contexts in MuJoCo/straggler tasks (as in Fig. 1c–e for the gridworld) would reveal when forgetting occurs and whether LCPO truly recovers old policies upon context recurrence.
- **Task-label oracle comparison.** In environments with discrete contexts, comparing LCPO against an oracle given perfect task labels would bound the performance cost of using an approximate OOD detector instead of exact labels.
- **Standard continuous-action experiments.** Running at least one benchmark (e.g., Hopper-v4) in the original continuous action space would validate that the findings are not an artifact of discretization.

## Removed Points

These points are flagged to be removed; treat them with caution.

- *“The tabular analogy is theoretically flawed because tabular updates modify exactly one state-context row; LCPO constrains outputs on a finite sample of past inputs, which does not guarantee locality in a neural network.”* The paper presents the tabular analogy as an **illustrative example** and explicitly states the theorem for tabular RL is in Appendix B. It does not claim the neural network case is formally proven by the tabular theorem, so this criticism mischaracterizes the paper’s claims.
- *“The OOD detector may be acting as a generic observation-space outlier detector rather than leveraging context structure.”* The paper acknowledges the counterintuitive Mahalanobis result and notes it is “not surprising” given L2’s lower robustness. Without evidence that the detector is *not* using context, this remains speculation.
- Criticisms about typos, grammar, formatting artifacts, or missing appendix proofs are parser issues or removed-section artifacts, not author errors.
- Criticisms about “unfair comparison” that would *favor* the baseline (e.g., claiming continuous-action baselines are advantaged on discretized tasks) are intentionally asymmetric and not weaknesses of the paper.

## Novel Insights

The paper makes a genuinely novel conceptual distinction between OOD detection and change-point detection for continual RL (§4.1, Fig. 2). The argument that OOD detection defines a natural distance metric on arbitrary context processes—whereas CPD requires piecewise stationarity and is brittle to threshold choice—is well-illustrated and provides a principled motivation for LCPO’s design. This insight, combined with the real-world systems evaluation, gives the work practical relevance beyond standard synthetic benchmarks.

## Suggestions

1. Replace the min/max-across-agents normalization with raw lifelong returns or fixed per-environment bounds (e.g., random policy = 0, prescient = 1) in Figure 3a. This is necessary for the “closest to prescient” claim to be verifiable.
2. Add a random-buffer KL-constraint ablation to isolate whether OOD sampling is the operative mechanism.
3. Prominently disclose the discretization of continuous-control benchmarks in the abstract, introduction, and limitations, and include at least one continuous-action experiment if feasible.

## Score and Decision

**Calibration papers used:**

- **High:** `/home/wg25r/review_agent/human_reviews/m3xVPaZp6Z.md` (avg 7.50, Accept poster) — Policy rehearsing with strong theory and experiments. LCPO has comparable experimental breadth but is below this due to its flawed relative metric and discretization issue.
- **High:** `/home/wg25r/review_agent/human_reviews/HN0CYZbAPw.md` (avg 6.50, Accept poster) — WSRL for offline-to-online fine-tuning with clear analysis and well-supported claims. LCPO is below this because its central comparative evidence (Fig. 3a) relies on an unanchored metric.
- **Medium:** `/home/wg25r/review_agent/human_reviews/Gf15GsnfTy.md` (avg 5.67, Accept poster) — REValueD on discretized DeepMind Control Suite tasks. LCPO is similar in using discretized benchmarks, but additionally suffers from a relative normalization metric that obscures absolute performance, pushing it below this anchor.
- **Medium:** `/home/wg25r/review_agent/human_reviews/FFvCjbhpDq.md` (avg 5.00, Reject) — Experimental study of forgetting in fine-tuning with good empirical analysis but no novel method. LCPO is somewhat stronger due to its novel constrained-optimization formulation and broader environment sweep.
- **Low:** `/home/wg25r/review_agent/human_reviews/8uYJottqTy.md` (avg 4.00, Withdrawn) — CORL with Decision Transformers, very limited experiments and missing details. LCPO is substantially stronger.
- **Low:** `/home/wg25r/review_agent/human_reviews/o10clUzFRH.md` (avg 4.50, Withdrawn) — Rényi regularized RL with theoretical flaws and sparse experiments. LCPO is stronger empirically and conceptually cleaner.

LCPO sits between the medium and low anchors. It has a real methodological contribution and substantial experimental effort, but two structural evaluation issues—the relative normalized metric and the undisclosed discretization—prevent its quantitative claims from being credible as presented. These issues require non-trivial revision (new analysis or experiments) rather than simple rebuttal clarifications. Relative to the anchor cluster, this places the paper below the accepted-poster threshold (≈5.5–6.0 in this area) but above the genuinely weak band (≤4.0).

**Score:** 4.5

**Decision:** Reject

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>