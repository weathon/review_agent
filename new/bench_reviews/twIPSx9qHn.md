Now I have all the information I need. Let me write the final review.

## Summary

CrossDex proposes a method for learning a single dexterous grasping policy that generalizes across multiple robot hand embodiments using RL. Inspired by teleoperation, the method uses human hand eigengrasps as a universal action space (mapped to specific robot joints via retargeting) and fingertip/palm positions as a unified observation space. Experiments in IsaacGym show 80% success on four training hands, 35% zero-shot transfer to two unseen hands, and effective finetuning from the pre-trained policy.

## Strengths

- **Novel and important problem formulation.** Cross-embodiment dexterous grasping for diverse hand morphologies is genuinely unexplored, and the paper is the first to tackle it. The teleoperation-inspired design of using human hand poses as a universal intermediate representation is a sound and well-motivated principle (Section 1, Section 3.2).

- **Clean ablation on unified observations.** Table 4 provides compelling evidence that removing embodiment-specific observations (one-hot labels) dramatically improves unseen-hand generalization (0.352 vs. 0.146 vision-based), directly validating the core design philosophy.

- **Effective finetuning from pre-trained policy.** Table 2 shows consistent and substantial improvements over training from scratch (e.g., 0.740 vs. 0.313 on GRAB multi-task vision) and over other pre-training strategies, demonstrating that CrossDex captures genuinely transferable grasping skills.

- **Zero-shot generalization to unseen hands.** Table 1 shows CrossDex achieving 0.352 vision-based success on unseen hands vs. the best baseline's 0.210, demonstrating meaningful cross-embodiment transfer even without any exposure to the test hands.

- **Cross-embodiment co-training benefits.** Figure 3 shows that training across all hands simultaneously achieves comparable or slightly better performance than individual training, supporting the practical value of the unified representation beyond just generalization.

## Weaknesses

### Fatal
None.

### Major

- **No comparison to external baselines.** The three baseline methods (MT-Raw-OA, MT-Raw-A, MT-Raw-O) are all variants of the authors' own pipeline, differing only in which components of CrossDex are included. There is no comparison to any independently published cross-embodiment method. GET-Zero (Patel & Song, 2024) is the most directly related prior work—it is discussed in Related Work but dismissed as "limited to LEAP Hand variants." However, the LEAP Hand is one of the unseen test embodiments, making a comparison feasible and informative. Without any external baseline, it is impossible to determine whether CrossDex's performance stems from its specific design choices or simply from the generic idea of cross-embodiment training with any reasonable unification scheme. The gap between CrossDex (0.800) and MT-Raw-OA (0.782) on training-hands vision is small enough that the question of whether the improvement is meaningful remains open.

- **No quantitative real-world evaluation.** Section 5.5 consists of one sentence describing a setup (RealMan RM65 + LEAP Hand + RealSense D435i) and a reference to project-page videos, with no success rates, number of trials, or failure analysis. For a paper whose abstract promises deployment across "diverse dexterous hands" and whose most novel claim is zero-shot generalization, the absence of quantitative real-world evidence is a significant gap. The LEAP Hand tested is precisely the unseen-embodiment setting where the paper should be strongest, yet no numbers are provided.

- **Eigengrasp contribution is overclaimed relative to the unified observation space.** Figure 4 shows that varying eigengrasps from k=1 to k=36 produces similar unseen-hand performance (~0.40–0.50), and CrossDex-MANO (raw 45-DOF axis angles) achieves ~0.40 on unseen hands vs. best eigengrasp at ~0.50, with overlapping variance regions. Meanwhile, Table 1 shows that the largest performance gap comes from the unified observation space: MT-Raw-A (unified obs + raw actions) achieves 0.210 on unseen vision vs. CrossDex's 0.352. The paper's framing centers eigengrasps as a key contribution, but the empirical evidence suggests the unified observation space is the more impactful design choice, with eigengrasps providing a modest additional benefit.

### Minor

- **Table 1 lacks variance information.** RL training is high-variance, and Table 2 shows substantial standard deviations (e.g., ±0.373 for No-Pretrain on GRAB). Table 1 reports no variance despite being the paper's central results table, making it difficult to assess the significance of reported differences, particularly the ~2% gap on training hands (0.800 vs. 0.782). Table 3 includes variance for ablation results, making the omission from Table 1 inconsistent.

- **Abstract framing emphasizes the less novel result.** The abstract prominently claims "80% success rate" (on training hands, which is essentially a multi-task RL result) while the more novel and important claim—zero-shot generalization—is associated with a much more modest 35% success rate that is not explicitly stated. This framing could mislead readers about the paper's primary contribution.

- **No analysis of eigengrasp coverage for non-human-like hands.** The eigengrasps are computed from the GRAB dataset of human grasping poses. A 4-fingered hand like LEAP or a 12-DoF hand like Inspire may require grasping strategies poorly represented in a human-grasping PCA subspace. No analysis is provided on whether the eigengrasp subspace adequately covers the reachable grasping configurations of each robot hand.

### Trivial
None.

## Nice-to-Haves

- Failure case analysis on unseen hands (65% failure rate): categorizing failure modes would reveal whether they stem from retargeting errors, action space limitations, or observation ambiguity, and would guide future improvements.
- Quantification of retargeting approximation error (neural network vs. optimization-based), which would help assess how faithfully the policy's intended actions are executed.
- Comparison to GET-Zero on the LEAP Hand (unseen in CrossDex training), even if it requires retraining GET-Zero, which would provide a meaningful external baseline.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic: "Zero-padding creates ambiguity for absent fingers."** The paper addresses this in Section 4.2: "For hands less than four fingers, the method can be generalized by setting the unused finger positions to zero in the observations." This is a reasonable design choice for a method that must handle varying finger counts, and the empirical results show it works. The theoretical ambiguity (finger at origin vs. absent finger) does not appear to cause practical issues given the zero-shot transfer results.

- **Harsh Critic: "Fairer comparison would train state-based baselines without embodiment labels."** The paper already provides MT-Raw-A (unified observations, raw actions, no embodiment labels), which directly addresses this concern. The comparison between MT-Raw-A and CrossDex on unseen hands (0.210 vs. 0.352) isolates the effect of the eigengrasp action space from the observation unification. The distillation mismatch in MT-Raw-OA/MT-Raw-O is acknowledged and explained by the authors in Section 5.2.

- **Harsh Critic: "Retargeting error not quantified."** Table 3 shows that all three retargeting methods produce similar performance, and the paper states the neural networks replace the optimization at much higher speed. The fact that performance is robust to retargeting choice (Table 3) makes the approximation error less critical.

- **Harsh Critic: "Retargeting method doesn't matter, undermining DexPilot claim."** The paper does not overclaim DexPilot specifically—it uses it as one reasonable option and shows the framework is robust to this choice (Table 3). This is actually a strength (robustness), not a weakness.

- **Strength Finder: "Principled handling of varying finger counts" as a strength.** Removed because this is a necessary design choice rather than a notable contribution, and the zero-padding approach has the theoretical ambiguity noted above.

- **Strength Finder: "Neural network retargeting enables scalable training."** While practical, this is an engineering optimization (replacing optimization with a neural network fit) rather than a core contribution. Kept as a supporting point but not elevated to a main strength.

## Novel Insights

The paper reveals an interesting asymmetry in cross-embodiment learning for dexterous hands: the observation space unification (removing embodiment-specific proprioception) matters more than the action space unification (eigengrasps vs. raw joint angles). This is somewhat counterintuitive—one might expect the action space alignment to be the harder problem. The explanation is that the retargeting process implicitly handles action alignment (any hand pose maps to reasonable joint positions), while the observation space must be explicitly designed to avoid embodiment-specific shortcuts that the policy can overfit to. This insight has implications beyond dexterous grasping: in cross-embodiment learning generally, careful observation design may be more critical for generalization than action space design.

## Suggestions

- Add quantitative real-world evaluation: report success rates over 20+ trials per object on the LEAP Hand setup described in Section 5.5. This is the most impactful improvement possible.
- Report standard deviations on Table 1 (main results), consistent with Table 2 and Table 3.
- Rebalance the framing to position the unified observation space as the primary contribution, with eigengrasps as a complementary technique that provides consistent but modest improvement.
- If possible, compare to GET-Zero on the LEAP Hand to provide an external baseline, or clearly acknowledge the absence as a limitation.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| RDT-1B | /home/wg25r/review_agent/human_reviews/yAzN4tz7oI.md | 7.0 | Unified action space for cross-robot manipulation with real-robot experiments. CrossDex is weaker: no quantitative real-world results, smaller scale, ablation-only baselines. |
| HEPi | /home/wg25r/review_agent/human_reviews/7BLXhmWvwF.md | 8.0 | Geometry-aware RL with unified graph representation, simulation-only but very clean methodology and strong baselines. CrossDex has a more novel problem but weaker evaluation rigor. |
| CrossLoco | /home/wg25r/review_agent/human_reviews/UCfz492fM8.md | 6.67 | Cross-embodiment human-to-robot retargeting via RL, simulation + qualitative real-world. Very similar spirit to CrossDex but with more thorough analysis. CrossDex is slightly weaker. |
| VTDexManip | /home/wg25r/review_agent/human_reviews/jf7C7EGw21.md | 5.5 | Dexterous manipulation RL benchmark, no quantitative real-world results. Accepted despite this gap. CrossDex is comparable in experimental depth but more novel in problem formulation. |
| HuWo | /home/wg25r/review_agent/human_reviews/bhUIoQ61pA.md | 5.0 | Humanoid locomotion RL, simulation only, rejected. CrossDex is stronger: more novel problem, better ablations, and at least qualitative real-world demonstration. |
| EbOhZyxIzQ | /home/wg25r/review_agent/human_reviews/EbOhZyxIzQ.md | 5.0 | Simulation-only, ablation-only baselines, rejected. CrossDex has a more significant problem contribution. |
| Pseudo-tactile (xcHIiZr3DT) | /home/wg25r/review_agent/human_reviews/xcHIiZr3DT.md | 2.5 | Dexterous grasping in Isaac Sim, no baselines/ablations, rejected. CrossDex is far stronger on all dimensions. |

CrossDex sits between the 5.0-5.5 rejected papers and the 5.5-6.67 accepted papers. Its genuinely novel problem formulation and promising zero-shot transfer results push it above pure reject territory, but the lack of quantitative real-world evaluation and external baselines prevent it from reaching the 6+ range where similar papers like CrossLoco and VTDexManip sit.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>