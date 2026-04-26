Now I have enough information to compose the final consolidated review. Let me verify the key claims against the paper.

**Verification of Harsh Critic Claims:**

1. **Missing Steccanella & Jonsson (2022) baseline**: Confirmed. Line 137 explicitly states "MadDist learns state distances using an approach similar to prior work (Steccanella & Jonsson, 2022), but differs in the use of a quasimetric distance function and a scale-invariant loss." This direct predecessor is not in the baselines (QRL, Hilbert). Valid concern.

2. **Headline results on symmetric environments**: Table 1 shows PointMaze results only — these are mazes where the shortest path is symmetric (undirected maze graph). The asymmetric environments (CliffWalking, KeyDoorGridWorld) appear only in Figure 3. Valid concern about disconnect between motivation and strongest results.

3. **No ablation of loss function components**: Confirmed. Paper states (line 229) "Appendix E contains additional ablation studies, and demonstrates that MadDist and TDMadDist are robust to the size of the latent dimension and the choice of quasimetric" — these ablate architecture choices, not the three loss components ($L_o$, $L_r$, $L_c$). Valid concern.

4. **High variance**: Checking Table 1 data — PM Giant Navigate: QRL 0.87±0.21 vs MadDist 0.93±0.17. Some entries are clearly significant (MadDist 1.00±0.00), others less so. Valid but partially mitigated by the clear wins.

5. **TDMadDist underperformance**: Confirmed. The paper acknowledges this honestly at line 229: "While TDMadDist underperforms the MadDist and QRL algorithm."

Now let me compose the final review.

## Summary

The paper introduces two algorithms (MadDist and TDMadDist) for learning the Minimum Action Distance (MAD) — the minimum number of actions to transition between states — from state trajectories alone, without requiring actions or rewards. MadDist extends Steccanella & Jonsson (2022) with a scale-invariant loss, a contrastive term, and quasimetric support; TDMadDist adds temporal-difference bootstrapping. The paper also proposes a simple quasimetric ($d_\text{simple}$) and introduces a benchmark suite of environments with known ground-truth MAD. MadDist achieves near-perfect success rates on downstream planning tasks, significantly outperforming existing baselines.

## Strengths

- **Clean formalization and principled problem definition**: The LP formulation of MAD (Eq. 1) and its equivalence to Floyd-Warshall on the determinized MDP is well-presented, making the problem definition precise and grounding the approach in established theory.

- **Comprehensive benchmark suite with ground-truth evaluation**: The paper introduces environments spanning deterministic/stochastic dynamics, discrete/continuous states, and symmetric/asymmetric transitions, all with computable ground-truth MAD. This enables rigorous evaluation that prior work lacked.

- **Scale-invariant loss design**: Normalizing by $(j{-}i)$ in Eq. 5 is a sensible modification that prevents long-range pairs from dominating, addressing a real limitation of the prior loss (Eq. 2).

- **Strong empirical performance of MadDist**: Table 1 shows MadDist achieving perfect or near-perfect performance (1.00±0.00 on four of six conditions) on downstream planning, decisively outperforming QRL and Hilbert baselines, particularly on Stitch environments that require composing information from disconnected trajectories.

- **Honest reporting of TDMadDist**: The paper transparently reports that TDMadDist underperforms MadDist on most environments rather than omitting it.

## Weaknesses

### Fatal
None.

### Major

- **Missing the most direct baseline undermines attribution of improvements**: MadDist explicitly builds on Steccanella & Jonsson (2022) — reproducing their loss function (Eq. 2) and modifying it with three changes (scale-invariant normalization, contrastive term, quasimetric support) — yet Steccanella & Jonsson's method is not included as a baseline. The two baselines (QRL, Hilbert) differ from MadDist in multiple ways simultaneously (loss function, distance parameterization, training paradigm). Without isolating variables, it is impossible to determine whether improvements come from quasimetric support (the paper's stated motivation), the scale-invariant loss (a straightforward modification), or the contrastive term (a standard anti-collapse strategy). A symmetric variant of MadDist (using L2 instead of any quasimetric) would directly test the central claim. This is not a minor gap — the paper's core claim that quasimetrics are essential for MAD learning remains unsubstantiated. (Affects Sections 6–7, Table 1, Figure 3.)

- **Headline results are on environments where asymmetry is irrelevant**: The paper's core motivation is that MAD is inherently asymmetric and prior symmetric methods are inadequate. Yet all six conditions in Table 1 (the most prominent quantitative results) are on PointMaze environments where the shortest path in an undirected maze graph is symmetric — the quasimetric property that the paper advocates is not needed there. The genuinely asymmetric environments (CliffWalking, KeyDoorGridWorld) only appear in Figure 3 and are small discrete grid worlds where neural representation learning may be overkill. This creates a disconnect between the paper's motivational framing and its strongest evidence. (Affects Sections 1–2, 7, Table 1.)

### Minor

- **No ablation of individual loss function components**: MadDist combines three modifications over prior work ($L_o$ with scale-invariant normalization, $L_r$ contrastive term, $L_c$ constraint loss). The ablations in Appendix E only vary quasimetric choice and latent dimension size — not the individual loss terms. Without component ablation, the contribution of each design choice cannot be isolated. (Affects Section 6.1.)

- **TDMadDist contribution is weakened by consistent underperformance**: TDMadDist is presented as one of two main contributions but underperforms even MadDist on most environments (e.g., PM Large Navigate: 0.70 vs. 1.00; PM Giant Stitch: 0.74 vs. 0.99). The paper offers no analysis of why bootstrapping fails in this setting — understanding these failure modes would either improve the method or provide useful negative findings. (Affects Section 6.2, Table 1.)

### Trivial
None.

## Nice-to-Haves

- Ablation comparing MadDist with a symmetric (L2-based) variant to directly validate the benefit of quasimetric support, especially on asymmetric environments.
- Include Steccanella & Jonsson (2022) as a direct baseline to clarify the incremental contribution.
- Visualization of learned embeddings on asymmetric environments (e.g., KeyDoorGridWorld) showing directional distances diverge.
- Evaluation on downstream goal-conditioned RL tasks beyond planning, as the paper's stated application is goal-conditioned RL.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Unacceptably high variance undermines significance"**: Some Table 1 comparisons have overlapping confidence intervals (e.g., PM Giant Navigate: 0.87±0.21 vs. 0.93±0.17), but MadDist achieves 1.00±0.00 on four of six conditions, which is unambiguously significant. The variance concern applies to only some conditions and does not invalidate the overall conclusion. Demanding larger seed counts is a generic complaint non-standard for this venue.

- **"Ground truth for PointMaze is approximated via discretization"**: The paper explicitly notes this (line 225): "we use in our experiments to approximate the ground truth MAD, by computing the all pairs shortest path using the Floyd-Warshall algorithm over the maze graph." This is a reasonable approximation that is transparently disclosed.

- **"Hilbert baseline is poorly configured"**: This is speculation without evidence. Per hard rules, we cannot question the baseline's implementation.

- **"Sensitivity to $d_\text{max}$ and $H_c$ not discussed in main text"**: This is a minor hyperparameter concern, and per soft rules these are typical implementation details that do not threaten core claims.

- **Strength: "dsimple outperforms more elaborate quasimetrics"**: The evidence for this is in Appendix E and not substantiated in the main text, so this claimed strength is weaker than presented by the Strength Finder.

- **Strength: "Theoretical grounding of MAD as constrained optimization"**: While the LP formulation is clean, this is standard (Floyd-Warshall) rather than a novel theoretical contribution.

## Novel Insights

The paper's most interesting empirical finding is that MadDist achieves its strongest gains on **Stitch** tasks (requiring trajectory composition) rather than on asymmetric environments, suggesting that the scale-invariant loss and trajectory-wide supervision matter more for practical performance than quasimetric support per se. This is somewhat at odds with the paper's framing and would have been a valuable observation for the authors to make.

## Suggestions

- Add a symmetric MadDist ablation (using L2 distance) on both symmetric and asymmetric environments to directly quantify the benefit of quasimetric support.
- Include Steccanella & Jonsson (2022) as a baseline; if their code is available, this requires minimal modification since MadDist's loss is a direct extension of theirs.
- Move one PointMaze-format environment with truly asymmetric dynamics (e.g., one-way corridors) into Table 1 to test the quasimetric advantage in the headline evaluation, or at minimum, add a planning task on an asymmetric environment to Table 1.
- Add per-component ablations ($L_o$ alone, $L_o + L_r$, $L_o + L_r + L_c$) to disentangle the contributions of each design choice.

## Calibration Summary

- **0akLDTFR9x** (CDPC, avg 7.0, Accept poster): Similar topic (temporal distance representation for goal-conditioned RL). This paper has stronger empirical validation and clearer novelty. The paper under review has more experimental design gaps.
- **I7DeajDEx7** (ETD, avg 6.75, Accept poster): Similar topic (temporal distance for RL). ETD also had novelty concerns but had clean experimental comparisons. Our paper has more issues with baseline selection.
- **3ARfhjGfdF** (ReBis, avg 5.5, Reject): Similar issue — missing ablations and questionable baseline comparisons. Our paper has a similar profile but with somewhat stronger headline results.
- **S6j6VQFmpw** (Deep metric learning, avg 3.75, Reject): Overclaimed results with missing baselines. Our paper is notably stronger than this — the formalization and benchmarks are real contributions.
- **s9SVlWOcLt** (Proto Successor Measure, avg 6.75, Reject [scored 5/8/6/8]): Interesting theoretical idea with limited experimental scope. Our paper has better experimentation but similar baseline comparison gaps.

The paper under review has solid contributions (formalization, benchmarks, MadDist algorithm) but significant experimental design issues (missing direct baseline, results mostly on symmetric environments, no component ablation). Relative to calibration anchors, this sits between the ReBis paper (5.5, reject) and ETD (6.75, accept poster), leaning toward the weaker end due to the baseline and attribution issues.

## Score and Decision

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>