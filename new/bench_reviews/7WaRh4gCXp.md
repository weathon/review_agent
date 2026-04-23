Now I have all the information I need. Let me compose the final review.

## Summary

This paper introduces Next-Best-Path (NBP), a method for active 3D mapping that shifts from greedy next-best-view selection to predicting long-term goals and obstacle-aware paths via a unified model that jointly predicts coverage gain value maps and obstacle maps. The paper also contributes AiMDoom, a new procedurally generated indoor dataset with four difficulty levels and exact ground truth meshes. NBP achieves state-of-the-art results on both the existing MP3D benchmark and the new AiMDoom dataset.

## Strengths

- **Well-motivated conceptual contribution:** The shift from next-best-view to next-best-path directly addresses the identified failure mode of existing methods — getting trapped in local areas due to short-sighted greedy pose selection. Figure 1 and Figure 4 effectively visualize this problem and the proposed solution. The improvement on AiMDoom Normal (0.734 vs. 0.418 for MACARONS in Table 2) quantitatively confirms the benefit of long-horizon planning.

- **Clever data augmentation via shortest-path properties:** The Dijkstra-based augmentation exploits the property that every sub-path of a shortest path is also a shortest path, yielding O(m²) training samples from a trajectory of length m (Section 4.4). This is a principled and effective strategy for data-efficient training.

- **New benchmark fills a real gap:** AiMDoom provides procedurally generated indoor environments with controllable difficulty (Table 1 shows navigation complexity up to 45.25 vs. 17.09 for MP3D) and clean ground truth meshes, addressing limitations of existing datasets with noisy real-world scans or limited complexity.

- **Informative oracle obstacle map ablation:** Table 4 cleanly disentangles the two model components, showing that using ground-truth obstacle maps improves coverage by 0.074 but remains far from perfect, identifying value map prediction as the primary bottleneck — a clear direction for future work.

- **Multi-task learning benefit confirmed:** Table 5 demonstrates that jointly training value map and obstacle map prediction improves both Final Coverage (0.734 vs. 0.712) and obstacle prediction precision (0.805 vs. 0.754).

## Weaknesses

### Fatal
None.

### Major

- **No variance reported for MP3D results, undermining the "state-of-the-art" claim on the established benchmark.** Table 3 reports only means across 5 test scenes (5 trajectories each = 25 episodes) for the headline comparison. In contrast, AiMDoom results (Table 2) include standard deviations that are often 30–50% of the mean (e.g., Simple Final Cov. = 0.879 ± 0.142). If similar variance exists on MP3D, the 6.23 absolute improvement over ANM (79.38 vs. 73.15) may not be statistically significant. The paper states "We report the mean and standard deviation for each metric across all testing trajectories" (Section 5.1) but omits them for MP3D. Even reporting NBP's own variance would help assess the reliability of the SOTA claim.

- **Strongest learning-based baselines (UPEN, ANM) are absent from the AiMDoom comparison.** The paper acknowledges this limitation (Section 5.2), explaining that retraining their DD-PPO navigation policies requires extensive GPU hours. This is understandable but leaves a meaningful gap: on AiMDoom, NBP is compared only against NBV methods designed for single-object/outdoor settings (SCONE, MACARONS) and the heuristic FBE. Since NBP's core advantage is precisely overcoming the short-sightedness of NBV methods, beating methods ill-suited for complex indoor environments is a relatively low bar. The MP3D comparison does include UPEN and ANM, which partially mitigates this — but on the paper's own new benchmark, the strongest competitors are missing, making it hard to assess whether NBP's AiMDoom advantage is over *all* methods or only over those fundamentally disadvantaged for the task.

### Minor

- **Coarse action space on AiMDoom may systematically favor long-horizon planning.** AiMDoom uses 1.5m translational steps and 45° rotational increments (Section 5.1), an order of magnitude coarser than MP3D (6.5cm, 10°). While this applies equally to all methods evaluated on AiMDoom, coarse discretization inherently amplifies the weakness that NBP is designed to address — NBV methods can only "see" one 1.5m step ahead, while NBP plans paths to distant goals. The paper does not discuss whether this design choice favors NBP, nor how results might change with finer step sizes. This limits the interpretability of the AiMDoom comparison.

- **Re-planning mechanism is under-analyzed.** The inference-time re-planning upon encountering unexpected obstacles (Section 4.3, one sentence) is critical for real-world viability but receives no analysis of frequency, computational cost, or impact on trajectory efficiency.

### Trivial
None.

## Nice-to-Haves

- An oracle value map experiment (ground-truth coverage gains with predicted obstacles) would complement Table 4's oracle obstacle map ablation and confirm the paper's own conclusion that value map prediction is the primary bottleneck.

- Failure analysis on Hard/Insane levels (0.618 and 0.472 coverage) would clarify whether the bottleneck is value map quality, obstacle prediction errors, or path planning failures in complex topologies.

- Ablation of the Dijkstra data augmentation to quantify its contribution to training efficiency.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Navigation complexity metric doesn't capture all aspects of difficulty"**: This is a minor observation about the dataset characterization, not a weakness in the method. The metric serves as a reasonable proxy.

- **"Universal accessibility reduces difficulty by removing doors/windows as obstacles"**: The paper is transparent that doors and windows are open in AiMDoom (Table 1, Section 3). This is a deliberate design choice for the mapping benchmark and not a flaw.

- **"K=4 height layers lose 3D structural information"**: This is a design tradeoff between computational efficiency and representation fidelity. The method works well despite this simplification, so it's not a weakness undermining the claims.

- **"Training instability from online data collection"**: The paper addresses this via replay buffer (Algorithm 1, line 9), curriculum learning (first N_e iterations exclude early steps), and limited training epochs per iteration. While not formally analyzed, these are standard mitigations.

- **"Missing ablations on curriculum learning, replay buffer, Boltzmann temperature β"**: These are reasonable requests for a more complete ablation study but the paper already includes three informative ablations (spatial range, oracle obstacles, multi-task). More ablations are nice-to-have rather than core weaknesses.

- **"UPEN and ANM should be included on AiMDoom"**: Already covered under Major weakness above. The removal is of the duplicated demand as a separate "obvious next step" — the authors have already explained why this is infeasible.

## Novel Insights

The Dijkstra sub-path data augmentation exploits a genuine structural property of shortest paths in a way that is both elegant and broadly applicable to any method that computes optimal paths through an environment. This idea could be useful beyond active mapping — any task that generates trajectory data and predicts values along trajectories could benefit from this O(m²) amplification of training samples. Additionally, the finding that value map prediction (not obstacle prediction) is the primary bottleneck (Table 4) suggests that future work should focus on better long-term goal prediction rather than more accurate obstacle maps, which is an actionable insight for the community.

## Suggestions

- Report standard deviations for NBP on MP3D (even if baselines from prior work lack them) so readers can assess the reliability of the 6.23% improvement claim.
- Discuss the coarse action space on AiMDoom explicitly: acknowledge that 1.5m steps may disadvantage single-step methods more than long-horizon planners, and if possible, include a small sensitivity experiment with finer steps.
- Add a brief discussion of re-planning frequency and typical failure modes at inference to help readers assess real-world applicability.

## Evaluation

**Originality:** The NBV→NBP formulation is a natural but meaningful conceptual shift. The Dijkstra augmentation is novel. The dataset fills a gap. Solid originality for the field.

**Importance:** Active 3D mapping is practically important (digital twins, robotics). The identified failure mode (getting stuck in local areas) is real and well-documented.

**Claims support:** Partially supported. MP3D results are convincing but lack variance. AiMDoom results are strong but the comparison excludes the strongest learning-based baselines.

**Experimental soundness:** Good on MP3D (all major baselines included), limited on AiMDoom (missing UPEN/ANM, coarse action space confound). Ablations are informative.

**Clarity:** Well-written with clear figures. The method is described clearly.

**Community value:** Both the method and the dataset are likely to be useful to the active mapping community.

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| ASID (active exploration, robotics) | jNR6s6OSBT.md | 6.75 | Similar topic (active perception in robotics), oral acceptance despite some baselines/presentation concerns. NBP has comparable novelty but weaker evidential completeness on its own benchmark. |
| LEAP (sparse-view 3D modeling) | P4o9akekdf.md | 7.00 | Clean contribution with solid evaluation. NBP has more moving parts and more gaps in evidence. |
| Coverage Path Planning RL | ZiF1bJ9K6B.md | 4.75 | Similar topic (RL for coverage path planning in unknown environments), rejected. NBP is significantly stronger in methodology and evaluation. |
| Interactive Semantic Map | Z91rwXnJsw.md | 2.0 | Unfair comparison issues, rejected. NBP is far stronger — includes strong baselines on MP3D and acknowledges limitations on AiMDoom. |
| POGEMA (multi-agent pathfinding benchmark) | 6VgwE2tCRm.md | 5.67 | New benchmark + method. NBP has a more novel methodological contribution but similar benchmark contribution. |
| LoTa-Bench | ADSxCpCu9s.md | 6.00 | New benchmark paper. NBP has both method and dataset, slightly stronger overall. |

The paper sits between the medium-scoring benchmark papers (4.75–6.0) and the high-scoring active perception papers (6.75–7.0). It has genuine methodological novelty and a useful dataset contribution, but the evidential gaps (no variance on MP3D, missing strongest baselines on AiMDoom) prevent it from reaching the higher tier.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>