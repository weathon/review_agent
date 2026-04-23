Now I have all the information I need. Let me compose the final review.

## Summary

The paper introduces Visuo-Tactile World Models (VT-WM), the first multi-task world model that fuses exocentric vision with fingertip tactile sensing (Digit 360 sensors) to ground robot manipulation predictions in contact physics. VT-WM uses pretrained Cosmos and Sparsh-X encoders with a transformer predictor, and evaluates three claims: (1) tactile grounding improves imagination quality (33% better object permanence, 29% better causal compliance), (2) VT-WM achieves up to 35% higher success rates in zero-shot real-robot planning on contact-rich tasks, and (3) VT-WM is data-efficient, outperforming BC by 3.5× with limited demonstrations on a new task.

## Strengths

- **Well-motivated problem with compelling qualitative evidence.** The paper correctly identifies a real failure mode of vision-only world models—hallucinated contact—and proposes a sensible remedy. Figure 7 directly illustrates the core failure: V-WM hallucinates cloth deformation when the robot moves above it without contact, while VT-WM correctly predicts a stationary cloth. This visual evidence makes the mechanism of improvement tangible and interpretable.

- **First multi-task visuo-tactile world model.** Prior visuo-tactile dynamics models (Sutanto et al., 2019; Tian et al., 2019; Zhang & Demiris, 2023) are task-specific. VT-WM is trained across multiple contact-rich tasks and evaluated for generalization, which is a clear scope advance (Section 2, last paragraph).

- **Principled ablation design for contact perception evaluation.** The V-WM vs. VT-WM comparison in Section 4.1 is well-controlled: both models are conditioned on the same actions and context, and evaluated with CoTracker-based normalized Fréchet distances complemented by paired t-tests with reported t-statistics and p-values. Statistical significance is achieved in 3/5 tasks for each metric (Section 4.1).

- **Planning gains concentrated in contact-rich tasks.** On the simple "reach button" task, both VT-WM and V-WM achieve 100% success (Figure 8, left). The differential improvement appears specifically in contact-rich multi-step tasks (push, wipe, stack), which precisely supports the paper's thesis that tactile grounding helps where contact matters, without hurting where it doesn't.

- **Practical design choice for deployment.** The planning objective remains purely vision-based—tactile improves the world model's reliability indirectly during training and initial-state disambiguation (Section 3.2.3). This means the system does not require tactile goal specifications.

## Weaknesses

### Fatal
None.

### Major

- **Insufficient sample sizes for real-robot planning claims.** The zero-shot planning experiments (Section 4.2, Figure 8) report success rates averaged over only **five trials per task**. With binary success/failure outcomes, five trials yield 95% confidence intervals spanning roughly ±40 percentage points. A claimed 35% improvement on "reach & push" (e.g., from ~1/5 to ~3/5, or 2/5 to 4/5) is statistically indistinguishable from noise. No confidence intervals, standard errors, or statistical tests are reported for the planning results, despite the paper reporting t-tests for contact perception metrics. The data efficiency experiment (Section 4.3) uses only nine trials. These are the paper's most practically important claims, yet they rest on the weakest evidence. The headline claim of "up to 35% higher success rates" cannot be credibly established with five binary trials.

- **Confounded comparison in the data efficiency experiment.** Section 4.3 compares VT-WM (a multi-task pre-trained model fine-tuned on 20 demonstrations) against ACT behavioral cloning trained **from scratch** on the same 20 demonstrations, yielding 77% vs. 22% success. This conflates two factors: (a) the benefit of multi-task pre-training and (b) the benefit of tactile modality. A vision-only multi-task world model (V-WM) fine-tuned on the same 20 demonstrations is the necessary control to isolate the contribution of tactile grounding. Without it, the 3.5× improvement cannot be attributed to tactile grounding—it likely reflects the well-known advantage of pre-training. Additionally, VT-WM uses open-loop CEM planning while ACT is deployed closed-loop; these are fundamentally different control paradigms, further confounding the comparison.

### Minor

- **Inconsistent imagination quality improvements across tasks.** For object permanence, only 3/5 tasks show statistically significant improvements; for causal compliance, again 3/5, with one negative result (scribble with marker, t = −1.22). The headline numbers (33%, 29%) are averages that can be dominated by a single large improvement (e.g., 66% reduction in wipe with cloth for causal compliance) while other tasks show small or negative effects. The paper acknowledges non-significance in the body text, but the abstract states "33% better" and "29% better" without qualification, which overstates the robustness of these findings.

- **CoTracker-on-generated-frames metric validity.** Both evaluation metrics (Section 4.1) depend on running CoTracker—trained on real video—on AI-generated frames. If VT-WM produces frames that are more photorealistically coherent (even if physically incorrect), the tracker may produce smoother trajectories for reasons unrelated to physical fidelity. The paper does not validate that tracking quality on generated frames correlates with physical plausibility. This is mitigated by the qualitative evidence (Figure 7) but remains a potential confound.

- **V-WM baseline not fully characterized.** The paper describes V-WM as a "multi-task vision-only world model" but does not explicitly state whether it shares the identical architecture and training data, differing only in the absence of tactile tokens. Clarifying this would strengthen the ablation interpretation.

### Trivial
None.

## Nice-to-Haves

- A V-WM fine-tuning baseline in the data efficiency experiment, isolating the contribution of tactile from the contribution of pre-training.
- Increased sample sizes for real-robot planning (15–20 trials per task) with confidence intervals or bootstrap intervals.
- Closed-loop planning execution (even simple replanning) to address the open-loop brittleness acknowledged in Section 3.2.3.
- Per-task failure mode analysis for planning—understanding whether failures stem from contact reasoning vs. kinematic errors vs. CEM optimization would clarify what tactile actually provides.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"Architecture lacks novelty"** (Harsh Critic Section 3.2.1 notes): The paper does not claim architectural novelty beyond multimodal fusion. Criticizing the absence of a contribution the authors don't claim is a strawman. The contribution is in the multimodal integration and evaluation, not in individual components.

- **"Deterministic L1 loss limits stochastic dynamics modeling"** (Harsh Critic Section 3.2.2): This is a generic criticism that applies to most world models using L1 reconstruction. The paper follows the established V-JEPA 2 AC paradigm. While valid as a future direction, it's not a weakness of this paper specifically.

- **"Discussion does not acknowledge limitations"** (Harsh Critic Section 5): The paper does acknowledge open-loop execution as a limitation (Section 3.2.3). While the discussion could be more thorough, claiming it doesn't acknowledge limitations at all is overstated.

- **Strength Finder's "Compelling data efficiency result on a new task"**: Dropped because the data efficiency comparison is confounded (see Major weakness). The 3.5× improvement cannot be attributed to tactile grounding without a V-WM fine-tuning control.

- **Strength Finder's "Rigorous evaluation methodology"**: Partially dropped—the t-tests for contact perception are rigorous, but the planning results lack any statistical validation, so calling the overall evaluation "rigorous" overstates the case. Kept the portion about the contact perception evaluation being rigorous.

## Novel Insights

The most insightful observation across the reviews is the **asymmetry in evidence quality**: the paper's most convincing evidence (contact perception metrics with t-tests, qualitative rollouts) supports the least practically important claim (imagination quality), while the most practically important claims (planning success rates, data efficiency) rest on the weakest evidence (n=5 and n=9 binary trials with no variance reporting). This disconnect between evidence strength and claim importance is the central issue—not the absence of a contribution, but the mismatch between what is demonstrated and what is headline-claimed.

## Suggestions

- Report confidence intervals (e.g., bootstrap or Wilson score intervals) for all planning success rates, even with small n. This would make the uncertainty transparent without requiring additional robot trials.
- Add a V-WM fine-tuning baseline for the data efficiency experiment. This is the single most impactful addition the authors could make, as it would isolate the tactile contribution from the pre-training benefit.
- Qualify the abstract claims ("33% better," "29% better") to reflect that these are averages across tasks with mixed statistical significance. For example, "achieving significant improvements in object permanence (average 33% reduction, significant in 3/5 tasks)" would be more accurate.
- In future work, consider even simple closed-loop replanning (e.g., replan every K steps) as a straightforward extension that would substantially strengthen the practical relevance.

## Score and Decision

**Calibration anchors used:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| Efficient RL by Guiding World Models (oBXfPyi47m.md) | 8.0 | Much stronger experimental validation across 72 tasks; VT-WM is clearly below this |
| World-In-World (yDmb7xAfeb.md) | 7.0 | Comprehensive benchmark; VT-WM lacks this breadth of evaluation |
| DexMove (dT3ZciXvNX.md) | 6.0 | Similar scope (tactile+vision, real robot); DexMove had 77.8% success on 6 objects with clearer ablations; VT-WM has weaker sample sizes and confounded comparison |
| Ctrl-World (748bHL2BAv.md) | 6.0 | Large-scale training (95k trajectories), strong video generation; VT-WM has smaller scale but more focused problem |
| 4D Latent WM (iB9qx28gv4.md) | 4.0 | Limited real-robot validation; VT-WM is clearly above this with real robot planning experiments |
| Overclaimed offline MBRL (rbNOhbdQ0v.md) | 3.33 | Overclaimed "first" with only one real task; VT-WM has more tasks but shares overclaiming pattern |
| Geometrically Regularized WMs (TLXp0scq3x.md) | 2.5 | Fundamental issues with claims vs. evidence; VT-WM is well above this |

VT-WM sits above the low-scoring papers (which have fundamental issues or near-zero real-robot validation) but below the medium-scoring papers (which have more robust experimental validation). The core idea is genuinely good and the qualitative evidence is compelling, but the two major weaknesses—insufficient sample sizes for planning claims and a confounded data efficiency comparison—prevent the paper from convincingly establishing its headline claims. This places it in the borderline range, slightly below the 6.0 anchors (DexMove, Ctrl-World) which had cleaner experimental designs despite similar scope.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>