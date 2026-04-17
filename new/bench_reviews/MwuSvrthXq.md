Now I have sufficient calibration data. Let me synthesize the final review.

## Summary

WeCAN proposes an end-to-end reinforcement learning framework for heterogeneous DAG scheduling that features: (1) a weighted cross-attention (WeCA) mechanism to encode task-pool compatibility coefficients in a dimension-adaptive manner, (2) a theoretical analysis identifying an optimality gap in list-scheduling-based generation maps, and (3) a skip-action mechanism for single-pass inference that provably closes this gap at the abstract score level. Empirical evaluations on TPC-H and Computation Graphs benchmarks show improvements of up to 18.1% over heuristics and 7.7% over neural baselines.

## Strengths

- **Well-designed WeCA mechanism.** The placement of compatibility coefficients outside the softmax is cleanly motivated by a concrete example (Section 3.1: two tasks with identical attributes but different compatibility profiles would collapse under inside placement). The ablation in Table 3 confirms the outside placement outperforms the inside version by 3.5–6.3% on TPC-H-30, and removing WeCA layers entirely degrades performance by 13–18%, demonstrating its centrality.

- **Strong empirical performance.** WeCAN-S(256) achieves the best makespan across all six benchmark configurations, improving over the best neural baseline (One-Shot-S(256)) by 7.0–9.5% on Computation Graphs and 5.4–7.7% on TPC-H, while the greedy variant runs 5–15× faster than PPO-BiHyb and comparable to heuristics (Tables 1–2). These are substantial, consistent improvements.

- **Principled theoretical analysis of list scheduling's limitations.** The reduced-space analysis (Section 4) provides formal grounding for why list scheduling cannot represent all optimal solutions, and Theorem 1 establishes that adding skip actions restores surjectivity at the level of the abstract score space. This is a valuable conceptual contribution even if the connection to the learned model is imperfect (see weaknesses).

- **Effective generalization to environment fluctuations.** Figure 2 demonstrates robust performance under variations in pool number, pool features, task number, and task types, validating the adaptability of the WeCA architecture—a genuine practical advantage over fixed-dimensionality approaches.

- **Comprehensive ablation study.** Table 3 systematically varies WeCA (encoder vs. decoder, inside vs. outside, layer count) and LDDGNN vs. GAT, providing clear evidence that each component matters.

## Weaknesses

### Major:

- **Overclaimed theoretical guarantees on "closing the optimality gap."** The paper repeatedly states that the skip mechanism "addresses this gap" (Abstract), "closes this gap" (Contribution 3), and "ensures that TS is a surjection, enabling the generation of the optimal schedule" (Section 4.2). However, Theorem 1 parts (ii) and (iv) are existence statements about the *score space*: they say there *exist* scores (and skip coefficients) under which Algorithm 1 assigns positive probability to an optimal solution. They do not establish that the WeCAN architecture with its specific MLP-based scoring and skip parametrization $u_a(1-k/2n)^{u_b}+u_c$ can *represent* all such scores, nor that REINFORCE training will *find* them. The surjectivity of the abstract map $S^*$ over the enlarged space $B_f^*$ does not transfer to guarantees about the learned policy. The empirical evidence for skip is solid (improvements on heavy-task cases in Figure 3), but the theoretical "gap-closing" narrative is overstated—it should be scoped as "the skip mechanism restores surjectivity of the generation map in principle, and empirical results show practical gains."

- **Skip action contribution not isolated on standard benchmarks.** The main results in Tables 1–2 report WeCAN with skip as the default, but do not include a WeCAN-without-skip ablation on the standard (unmodified) benchmarks. The skip-action benefit is only demonstrated on artificially modified "heavy-task" datasets (1% task replacement, Figure 3 and Appendix C). It remains unclear how much the skip mechanism contributes on typical, unmodified workloads. Including a "WeCAN-no-skip" row in Tables 1–2 would directly address this.

- **Missing directly relevant heterogeneous RL baselines.** The related work (Section 2.1) explicitly discusses several RL methods designed for heterogeneous scheduling with compatibility modeling: Zhou et al. (2022), Zhadan et al. (2023), and Wang et al. (2025). Yet none are included as experimental baselines. The two RL baselines (PPO-BiHyb, One-Shot) are either designed for homogeneous settings or do not handle compatibility coefficients natively. Since the central claim is about handling "diverse task-pool compatibilities," the absence of the most comparable methods is a meaningful gap.

### Minor:

- **Ad hoc skip score parametrization.** The formula $u_{\text{skip}} = u_a(1-k/2n)^{u_b}+u_c$ is introduced without justification for this specific functional form. The paper claims it "clusters poor solutions in the high-$u_a$, high-$u_c$ region" but provides no quantitative evidence (e.g., distribution analysis of skip counts or reward variance). No ablation comparing simpler alternatives (constant, linear) is provided.

- **Non-autoregressive decoder lacks in-main-text justification or comparison.** The decoder makes all action scores time-invariant (Section 3.2: "depends only on the initial state $s_1$"), which is a strong assumption for scheduling problems where later decisions depend on earlier ones. The paper references an autoregressive comparison in Appendix B but does not summarize it in the main text, making it impossible to assess the performance trade-off.

- **One-Shot adaptation to heterogeneous settings is unclear.** The paper acknowledges that One-Shot "does not consider compatibility coefficients or pool allocation" (Section 2.1), but it is not explained how One-Shot is modified to handle heterogeneous environments. If the adaptation is suboptimal, the comparison may unfairly favor WeCAN.

### Trivial:

- The notation in Section 4.1 (spaces A, B, maps T, S) is dense and its formal details are entirely deferred to Appendix A, making the main text's theoretical argument difficult to follow without the appendix.

## Nice-to-Haves

- Comparison against an MILP solver on small instances to measure the gap to true optimality, providing a more complete picture of WeCAN's solution quality.
- An ablation of skip-action on the standard (unmodified) benchmarks to quantify its marginal contribution in typical workloads.
- Training curves, training compute budget, and hyperparameter sensitivity analysis for the skip mechanism.
- A worked example with step-by-step trace showing where list scheduling fails and skip succeeds on a small concrete instance.

## Removed Points

- **"HEFT contradicts the list scheduling optimality gap narrative"** — The paper explicitly labels HEFT as "a non-list heterogeneous scheduling algorithm" (Section 5.1) and acknowledges it outperforms list heuristics. The optimality gap claim is specifically about list-scheduling generation maps, and the paper is formally correct that list scheduling has this property. The reviewer's concern that the narrative overgeneralizes has some validity but HEFT does not contradict the core theoretical claim.

- **"The paper does not provide any analysis or ablation demonstrating the non-autoregressive policy is sufficient"** — The paper does reference an autoregressive comparison in Appendix B and provides strong empirical results with the non-AR design. The concern about justifying NAR is valid but is already partially addressed; the appendix comparison should be summarized in main text rather than dismissed.

- **"Generalization claims are weakly supported"** — Figure 2 does provide quantitative data (percent improvement curves), not just qualitative assessment. The missing detail about the magnitude of distribution shift and baseline comparisons under shift is a valid concern but should not discard the existing evidence.

- **"No confidence intervals for greedy mode"** — Standard single-seed greedy evaluation is typical in this literature; requiring confidence intervals for greedy results is an above-standard demand.

- **"Limited scalability evaluation"** — The paper includes TPC-H-100 (~918 tasks) and references Appendix F for larger problems. Scalability beyond ~1000 tasks is a reasonable concern but not a fundamental flaw given the architectural design.

## Novel Insights

The reduced-space perspective (Section 4) — viewing schedule generation as a map from action sequences to feasible schedules, and formally characterizing when this map fails to be surjective — provides a clean theoretical vocabulary for understanding why certain generation paradigms (like list scheduling) structurally exclude optimal solutions. Even though the connection between the abstract map and the learned model is not fully tightened, the conceptual framing itself is a genuine contribution that could inform future work on schedule generation design beyond this paper.

## Suggestions

1. **Scope the theoretical claims precisely.** Replace "closes this gap" with "restores surjectivity of the generation map in principle, and empirically improves performance on heavy-task instances" throughout the paper. Clearly separate what Theorem 1 guarantees about the algorithm class from what it says about the learned model.

2. **Add WeCAN-without-skip rows to Tables 1 and 2.** This single addition would clarify the practical contribution of skip on standard workloads and significantly strengthen the empirical narrative.

3. **Include at least one heterogeneous RL baseline** from the related work (Zhou et al. 2022 or Wang et al. 2025) and explain how One-Shot is adapted for heterogeneous settings.

4. **Summarize the autoregressive comparison** from Appendix B in the main text, even as a brief paragraph.

5. **Ablate the skip score formula** against simpler alternatives (constant skip score, linear in $k/n$) to justify the chosen parametrization.

## Score and Decision

**Calibration anchors:**
- GOODRL (workflow scheduling, GNN+RL, strong empirical but evaluation gaps): scores 6, 8, 5, 8 (avg ~6.75) → Accept Poster
- DRL improvement heuristic for JSSP (strong novel architecture, comprehensive eval, linear complexity proof): scores 8, 8, 8, 6 (avg ~7.5) → Accept Poster
- L-RHO (novel framework, some theoretical issues, limited comparisons): scores 8, 8, 5, 6 (avg ~6.75) → Accept Poster
- NAR GNN for NCO (overclaimed findings about NAR vs AR, weak baselines): scores 8, 5, 3 (avg ~5.3) → Reject
- Simultaneous generation+improvement for FJSP (poor clarity, weak evaluation): scores 3, 3, 3, 3 → Reject

WeCAN has genuinely novel and well-motivated architectural components (WeCA, LDDGNN), strong and consistent empirical results, and a valuable conceptual contribution in the reduced-space analysis. However, the overclaimed theoretical guarantees about "closing the optimality gap" — where existence results for the abstract score space are conflated with guarantees for the learned model — and the missing heterogeneous baselines are substantive weaknesses. The core empirical contribution is solid (6.5–7 level), but the overclaiming and evaluation gaps pull it down. Relative to L-RHO (which had similar theoretical clarity issues but was accepted), WeCAN has stronger results but larger overclaiming. Relative to GOODRL (similar strength profile, accepted), WeCAN has a comparable profile.

**Score: 6**

The paper makes a meaningful contribution to heterogeneous DAG scheduling with well-designed architectural components and strong empirical results, but the theoretical narrative is overstated and the evaluation lacks key baselines and ablations. With careful revision to scope the claims and fill evaluation gaps, this could be a strong contribution.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>