## Summary

ProgressCounts introduces a two-step framework for automated reward generation: (1) use an LLM to generate a progress function that coarsely estimates task progress from environment states, and (2) convert progress estimates into count-based intrinsic rewards via discretized bins. On the Bi-DexHands benchmark, ProgressCounts achieves a 0.59 average success rate (4% over Eureka) using only 4 policy samples versus Eureka's 80, demonstrating substantial sample efficiency gains.

## Strengths

- **Clean conceptual decomposition**: Separating "estimate progress" from "generate reward" is a genuine and useful insight that sidesteps the reward weighting/scaling problem plaguing LLM-generated dense rewards. Section 4.2 articulates this clearly: "progress functions offer highly simplified state representations—and given the coarse nature of these representations, we look for a more forgiving mechanism to generate rewards" (p.4).

- **Strong absolute performance on Bi-DexHands**: Matching or exceeding human dense rewards on 17/20 tasks and Eureka on 13/20 tasks (Figure 3) is meaningful, even accounting for methodological concerns. The 0.59 average success rate substantially exceeds human dense rewards (0.52) and sparse rewards.

- **Dramatic sample efficiency**: 4 policy samples vs. 80 for Eureka to achieve comparable/better performance (Figure 2) is a significant practical advantage. The TwoCatchUnderarm experiment (Figure 4) effectively demonstrates the compute-reallocation benefit: same total budget, fewer candidates, more per-candidate compute, achieving 0.55 success where all baselines achieve near-zero.

- **Simplicity and likely reproducibility**: The approach is conceptually simple—LLM generates a short progress function, heuristic discretization, standard count-based reward, PPO training—requiring no evolutionary search or iterative LLM querying. The paper notes binning functions contain "less than 20 lines of code" (Discussion).

- **Targeted design ablations (Table 2)**: Isolating the feature engineering library and heuristic discretization shows each addresses distinct failure modes (e.g., CatchUnderarm failing completely without the feature library; SwingCup dropping from 0.97 to 0.00 without heuristic discretization).

## Weaknesses

### Fatal
None.

### Major

- **Asymmetric ablation undermines the central architectural claim**: The paper's key claim—"both LLM-generated progress functions and count-based intrinsic exploration are necessary" (Section 5.3)—rests on Table 1, where ProgressCounts (0.59) is averaged over 5 trials with best-of-4 selection, while ProgressAsReward (0.45) and SimHashCounts (0.34) are single-trial results. The paper itself acknowledges this in the table caption: "Results are averaged across 5 trials for ProgressCounts, and are single-trial numbers for the ablated methods." This ~5× evaluation-compute asymmetry means the 14-point gap between ProgressCounts and ProgressAsReward could shrink with equal protocols. While the gaps are large enough that the conclusion would likely hold, the current evidence does not rigorously establish it. A fair ablation with equal trial counts and independent model selection is essential for this foundational claim.

- **No variance information on headline comparisons**: The 4% improvement over Eureka (0.59 vs. 0.55) and all per-task results in Figure 3 lack error bars, confidence intervals, or standard deviations. RL results on sparse-reward manipulation tasks are notoriously high-variance. While the paper references Table 8 in the appendix for standard deviations on one comparison point (BlockStack), no variance is reported for any main-result figure or table. The SOTA claim cannot be evaluated without knowing whether the 4% difference exceeds noise. This is especially important because the Eureka numbers are taken from Ma et al. (2023) with potentially different evaluation protocols.

### Minor

- **Progress function specification is incomplete—unused y_i variables**: Section 4.1.1 defines the progress function as outputting both progress variables [x_1, …, x_k] and additional variables [y_1, …, y_k] "that inform our framework whether the progress variables x_i are increasing or decreasing." These y_i variables are never referenced in Section 4.2's binning/reward computation and never discussed in experiments. Either they are used implicitly (e.g., in heuristic discretization to determine normalization direction) and the method is underspecified, or they are vestigial—either way, the reader cannot fully understand or reproduce the method.

- **Discussion overclaims robustness of count-based rewards to non-optimal binning**: The paper hypothesizes that "count-based intrinsic rewards being robust to non-optimal binning functions" (Section 6). While framed as a hypothesis, this is stated as a takeaway rather than an open question. No experiment systematically varies binning quality to test this claim—it remains an untested hypothesis that the paper presents as a conclusion.

- **Single primary benchmark limits generality assessment**: The main paper evaluates only on Bi-DexHands (20 bimanual manipulation tasks sharing substantial structure). MiniGrid results are relegated to the appendix. The approach relies on a domain-specific feature engineering library, and it is unclear how well the framework transfers to domains with less structured progress (e.g., navigation, locomotion) or where subtask decomposition is less natural.

### Trivial
None.

## Nice-to-Haves

- **Alternative binning that preserves subtask identity**: Testing a tuple-based binning B(s) = (x'_1, x'_2, …) instead of the sum B(s) = Σ x'_i would clarify whether the information loss from summing discretized progress variables matters in practice. The current binning conflates states with different subtask completion profiles (e.g., 50% on subtask 1 + 0% on subtask 2 → same bin as 0% + 50%).

- **Progress function quality analysis**: Of the 4 generated progress functions per task, how many produce non-trivial policies? This would clarify how much the best-of-4 selection matters and how reliable the LLM generation is.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh Critic: "Unfair comparison with Eureka due to different evaluation protocols"** — The harsh critic questions whether evaluation protocols (seeds, number of evaluation episodes) match exactly with Eureka. The paper explicitly states it follows "the PPO hyperparameters and sample budgets established by Bi-DexHands Chen et al. (2022), also used in prior work Eureka Ma et al. (2023)" (Section 5.1). This concern is partially addressed.

- **Harsh Critic: "20× fewer samples claim misleading in abstract"** — The abstract states "20× fewer reward function samples," which is accurate. The paper is clear throughout that this refers to policy samples (LLM-generated candidates), not environment interactions. This is not misleading.

- **Harsh Critic: "TwoCatchUnderarm based on single trajectory of single policy"** — While the Figure 4 training curve lacks variance bands, the claim is about demonstrating the compute-reallocation benefit with a single policy sample, which is the intended experimental design. The concern about variance is already captured in the Major weakness above.

- **Strength Finder: "Both components shown necessary through clean ablations"** — This strength is weakened by the verified Major weakness about asymmetric ablation. Retained as a partial strength (the ablations exist and show directional evidence) but cannot be stated as "clean" given the evaluation asymmetry.

- **Harsh Critic: "Table 2 ablations also single-trial"** — While true, Table 2 ablates design choices (feature library, heuristic discretization) rather than the core architectural claim. The single-trial concern is less critical for these supporting ablations than for the main Table 1 comparison. Downgraded to a minor note within the existing weaknesses.

- **Harsh Critic: "Binning conflation from B(s) = Σ x'_i could systematically misguide exploration"** — This is a valid theoretical concern but is speculative and not evidenced by the empirical results. The paper achieves strong performance despite this design choice, and the finer discretization for later subtasks partially mitigates conflation. Moved to Nice-to-Have.

## Novel Insights

The reviews surface an important tension: ProgressCounts's core insight—that LLMs should generate coarse progress estimates rather than fine-grained reward functions—is genuinely elegant and well-supported by the results. However, the paper's own framing creates a paradox. It argues that count-based rewards are more forgiving than dense progress-as-reward (motivating the two-step design), yet the ablation testing this claim uses a less-forgiving evaluation protocol for the dense-reward alternative (single trial vs. 5-trial average). A method that claims robustness through coarseness should be able to prove its advantage under equally robust evaluation conditions.

## Suggestions

- Run ProgressAsReward and SimHashCounts with the same best-of-4, 5-trial-averaged evaluation protocol as ProgressCounts. Even 3-trial averages would substantially strengthen the ablation.
- Add standard deviations or confidence intervals to Figure 2, Figure 3, and Table 1. At minimum, report the per-task standard deviations for ProgressCounts (5 trials) directly in the main paper.
- Clarify the role of the y_i variables—either explain how they are used in the discretization pipeline or remove them from the definition if they are unused.

## Evaluation

**Originality**: The progress-functions + count-based-rewards decomposition is a genuine and novel insight, distinct from prior LLM-for-reward work (Eureka, Text2Reward) which focus on generating full reward functions. The idea that LLMs should generate coarse progress estimates rather than optimized reward functions is cleanly motivated.

**Importance of research question**: Automated reward engineering for sparse-reward RL is an important and active area. The sample efficiency problem (many training runs to find good rewards) is practical and well-identified.

**Claims support**: The core performance claims (SOTA on Bi-DexHands, 20× sample efficiency) are well-supported by Figure 2 and Figure 3. However, the architectural claim that both components are necessary is undermined by the asymmetric ablation, and the SOTA claim (4% over Eureka) lacks variance information.

**Experimental soundness**: The main experimental results are strong, but the ablation methodology has a significant gap (5-trial vs. single-trial) and variance reporting is absent from main results.

**Clarity**: The paper is well-written and clearly structured. The y_i variable underspecification is a notable gap in an otherwise clear presentation.

**Community value**: The framework is simple, practical, and likely reproducible. The insight that progress functions are easier for LLMs to generate than full reward functions could influence how the community approaches LLM-guided RL.

## Score and Decision

**Calibration anchors**:

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Eureka | `/home/wg25r/review_agent/human_reviews/IEduRUO55F.md` | 6.25 | Direct baseline of this paper; evaluated on 29 environments (broader), but had its own issues (fitness function assumptions, environment source code access). ProgressCounts has a simpler/cleaner idea but weaker experimental methodology. |
| Text2Reward | `/home/wg25r/review_agent/human_reviews/tUM39YTRxH.md` | 7.0 | Similar LLM-reward-generation topic; more complete evaluation (multiple benchmarks, real-world deployment). ProgressCounts has a more novel decomposition idea but less thorough evaluation. |
| Q-shaping | `/home/wg25r/review_agent/human_reviews/DlqRpj68xe.md` | 5.67 | Similar pattern: LLM for RL, unfair baseline comparison, overclaimed improvements. ProgressCounts has a stronger and more novel core idea. |
| L2S | `/home/wg25r/review_agent/human_reviews/DBbgasVgyQ.md` | 5.25 | Similar pattern: no error bars, task-specific engineering. ProgressCounts has a cleaner contribution. |
| MaestroMotif | `/home/wg25r/review_agent/human_reviews/or8mMhmyRV.md` | 7.75 | High anchor: LLM for reward/skill design, single benchmark (NetHack), clean framework. ProgressCounts is conceptually similar in scope but has more methodological gaps. |
| LLIT | `/home/wg25r/review_agent/human_reviews/zEhTnQZB3D.md` | 2.33 | Low anchor: fundamentally incomplete, models didn't converge. ProgressCounts is far above this. |

ProgressCounts sits between the rejected medium-scoring papers (5.25–5.75, which had similar methodological issues) and the accepted papers (6.25–7.0). It has a genuinely novel and elegant idea with strong absolute results, which elevates it above the rejected papers. However, the asymmetric ablation and missing variance are significant concerns that prevent it from reaching the scores of cleaner accepted papers. The paper is comparable to Eureka (6.25) but with more serious ablation issues and a narrower evaluation scope.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>