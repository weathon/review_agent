=== CALIBRATION EXAMPLE 11 ===

# Final Consolidated Review
##Summary

VideoZoomer proposes an agentic framework for long video understanding where an MLLM dynamically controls its temporal focus by invoking a `<video_zoom>` tool to retrieve high-frame-rate clips at autonomously chosen moments, starting from a coarse low-frame-rate overview. The method employs a two-stage training strategy: cold-start SFT on distilled exemplar and reflection trajectories, followed by GRPO-based reinforcement learning to optimize the interactive policy. A 7B model trained this way achieves strong results across multiple long video understanding and reasoning benchmarks, surpassing open-source baselines and rivaling proprietary models on certain tasks.

## Strengths

- **Well-motivated agentic formulation of temporal focus.** The "glance-then-zoom" paradigm is a principled departure from static frame selection, enabling the model to iteratively gather evidence and self-correct. The emergence of diverse reasoning patterns (direct-hit, progressive, self-refine; Figure 3) is a genuine benefit of the interactive design that static methods cannot replicate.

- **Reflection data engineering is a meaningful contribution.** The cold-start dataset construction pipeline (Section 3.2, Figure 4)—distilling exemplar trajectories from proprietary models and then augmenting with reflection data where an expert corrects the initial model's failed rollouts—is well-designed. The ablation (Table 3) confirms that removing reflection data causes the average tool call count to collapse to ~1.0, validating that this component teaches multi-step investigation rather than shallow single-call behavior.

- **Comprehensive empirical evaluation across seven benchmarks.** The paper evaluates on MLVU, LongVideoBench, VideoMME, LVBench, VideoMMLU, VideoMMMU, and LongVideoReason-eval, showing consistent improvements. The per-task MLVU breakdown (Table 2) is particularly informative, showing the largest gains on detail-oriented tasks like Action Count (13.6→50.5) and Needle QA (+15.2), which directly benefit from high-temporal-resolution re-sampling.

- **Complementary with frame selectors.** Table 4 demonstrates that combining VideoZoomer with an external frame selector (TSPO) yields additional gains (+2.0 on MLVU, +3.0 on LongVideoBench), showing the learned policy effectively leverages improved starting points rather than overfitting to uniform initialization.

## Weaknesses

### Major:

- **No experimental comparison with agentic baselines.** The paper explicitly positions itself against training-free agentic methods like VideoDeepResearch (Yuan et al., 2025) and Deep Video Discovery (Zhang et al., 2025c) in Section 2, noting they "demonstrate the potential of agentic approach but rely on resource-intensive, closed-source models." Yet neither appears in the experimental tables. Since these are the most conceptually similar baselines—also using iterative tool-based video exploration—their absence is a significant gap. The paper's claim of superiority over "existing" agentic approaches rests on an argument about efficiency and open-source deployability rather than direct empirical evidence. Even a single-benchmark comparison would substantially strengthen the contribution.

- **No ground-truth alignment analysis for zoom selections.** The core mechanism claim is that the model learns to select semantically critical moments for high-resolution inspection. However, the paper provides no quantitative analysis of whether the model's chosen time segments actually overlap with ground-truth relevant moments. Without this, it remains possible that the model's gains come from simply seeing *any* high-resolution clips (providing more total visual information) rather than from correctly *targeting* the right moments. A simple metric—e.g., IoU between selected segments and annotated key moments on a benchmark like LSDBench—would directly validate the mechanism and is a notable omission.

- **Inconsistency between reflection ablation explanation and tool-call-count analysis.** The paper attributes the 5.2-point drop from removing reflection data (80.3→75.1 on LongVideoReason-eval, Table 3) to the model adopting a "shallow" strategy with ~1.0 tool calls versus ~2.0 for the full model. However, Table 11 shows that increasing max tool calls from 1 to 2 improves LongVideoReason-eval by only 0.3 points (79.9→80.2). This means the reflection data's benefit cannot be primarily explained by increased call count—the real driver is likely improved *quality* of reasoning and tool selection, which the paper's explanation obscures. This is more than a presentation issue; it suggests the authors may not fully understand what reflection data contributes, which limits the insight value of the contribution.

### Minor:

- **Reward function formulation is unclear.** Equation 1 presents $R = R_{acc} + R_{format} + R_{tool}$ as a simple sum, but Appendix Table 5 lists reward weights as "0.9/0.1/0.5." The relationship between the equation and the weights is never specified—is this a weighted sum $R = 0.9 \cdot R_{acc} + 0.1 \cdot R_{format} + 0.5 \cdot R_{tool}$? If so, the equation should reflect this. The units and ranges of each reward component are also unspecified, making it impossible to assess the relative magnitude of each term.

- **Conditional $R_{tool}$ creates a credit assignment problem.** The tool-use bonus is only awarded when the final answer is correct. During early RL training, when accuracy is low, the model rarely receives this bonus, making it difficult to discover that tool usage is valuable in the first place. The ablation (w/o $R_{tool}$ → "policy collapse") confirms this fragility, but the paper does not discuss the chicken-and-egg problem this creates or how the cold-start phase mitigates it. A brief analysis of when the tool bonus starts contributing during training would be informative.

- **Efficiency comparison conflates average and fixed frame budgets.** Figure 6 plots VideoZoomer's *average* frames consumed against the baseline's *fixed* frame budget. While this demonstrates average efficiency gains, it does not account for the variance in VideoZoomer's frame usage or the fact that some samples may require the full 128-frame budget. Reporting both mean and maximum frames consumed per benchmark, or showing the full distribution, would make the efficiency claims more rigorous.

- **No failure case analysis.** All case studies (Figures 8–13) show successful reasoning trajectories. No examples are provided where the zoom mechanism leads the model astray (e.g., zooming into an irrelevant segment and failing to recover). This makes it difficult to assess the method's robustness and the limits of the self-correction capability.

- **Missing dedicated limitations discussion.** The paper does not include a limitations section. Key limitations not discussed include: (a) the dependency on proprietary models for cold-start data distillation; (b) the domain specificity of the method (designed for temporally-structured long videos, with minimal gain on abstract reasoning like CLEVRER, Table 8); and (c) the computational cost of RL training (16×H100 GPUs for ~45 hours) relative to the efficiency gains at inference.

### Trivial:

- The GRPO extension with "token-level loss mask over the tool-call trajectory" (Section 3.3) is mentioned in one sentence without elaboration. While this is a potentially important technical detail, the core contribution does not hinge on its specifics.

## Nice-to-Haves

- **Confidence intervals or multiple RL runs.** RL training is known for high variance; reporting results from 2–3 seeds with standard deviations would strengthen confidence in the reported numbers, though single-run evaluation is common in this area.

- **Wall-clock inference time comparison.** The efficiency analysis focuses on frame counts but not actual latency. Multi-turn tool interaction introduces sequential inference overhead that could partially offset frame savings in deployment.

- **Ablation on initial frame count.** The 64-frame coarse overview is treated as fixed. Sensitivity analysis on this hyperparameter would reveal whether the method is robust to coarser or finer initial views.

- **Quantitative analysis of reflection data diversity.** The claim that Gemini-2.5-Pro trajectories show "greater diversity" (Appendix B.1) is qualitative. Metrics like trajectory length variance, unique tool-call sequence patterns, or reasoning-step counts would substantiate this.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Title formatting artifact** ("TEMPORO## RAL") — Parser extraction issue, not a paper problem.
- **Frame budget formula confusion** — The formula $f_{high} \times (t_{end} - t_{start}) \leq B$ per call and total bounded by $B \times N$ is logically consistent; the critic misread this.
- **Missing prompt templates** — Trivial implementation detail; the appendix references them and they will be released with code.
- **Random seeds not reported** — Trivial reproducibility nitpick for RL experiments; the paper commits to releasing code and weights.
- **Unfair comparison with proprietary models** (spark finder's point that GPT-4o/Gemini didn't have the zoom tool) — The asymmetry *favors* the baselines (they are stronger models without needing the tool), so VideoZoomer's competitive performance is a stronger claim, not a weaker one.
- **MLVU tasks showing "-"** — Standard practice for tasks not evaluated; not a weakness.
- **"Not yet released" concerns about cited methods** — Per hard rules, all cited entities are assumed to exist.

## Novel Insights

The reflection data mechanism reveals an interesting training dynamics pattern: the primary value of reflection data may not be teaching the model to make *more* tool calls (as the paper claims), but rather to make *better-quality* calls and develop more sophisticated reasoning strategies. This is evidenced by the tension between the 5.2-point ablation gap and the 0.3-point gap from adding a second tool call (Table 11). The reflection data likely teaches the model *how to evaluate* the utility of retrieved clips and *when to persist* versus when to stop—not just to call the tool more often. This distinction has implications for future work: if call count isn't the bottleneck, then training strategies should focus on improving call quality (e.g., better segment selection, more informative reasoning between calls) rather than simply encouraging more interactions. Additionally, the finding that performance peaks at ~64 frames on LongVideoReason-eval even for the baseline (Figure 6, right) suggests an important open question about the information-to-noise ratio in long video reasoning—beyond a certain temporal resolution, additional frames may actively harm reasoning by introducing distractors, which would fundamentally change how we think about context budget allocation.

## Suggestions

- **Add a direct comparison with at least one agentic baseline** (VideoDeepResearch or Deep Video Discovery) on a shared benchmark, even if using a different evaluation protocol. This is the single most impactful addition for reviewers.

- **Include ground-truth alignment analysis.** On LSDBench or a similar benchmark with annotated key moments, compute IoU between the model's zoom selections and ground-truth temporal segments. This validates the core mechanism claim.

- **Revise the reflection ablation explanation.** Acknowledge that the 5.2-point gap is not primarily attributable to call count differences (per Table 11), and provide analysis of what reflection data actually improves—e.g., does it improve the precision of selected segments, the quality of reasoning between calls, or the model's ability to recognize when it has sufficient evidence?

- **Clarify the reward function.** Either rewrite Equation 1 to include the weights, or explicitly state that the weights are applied as scaling factors and specify the range of each reward component.

- **Add 2–3 failure case examples** in an appendix, showing trajectories where the model zooms to the wrong segment or fails to self-correct, along with brief analysis of what went wrong.

---

**Quality Assessment:**

- **Novelty:** Good. The agentic temporal zoom formulation and the reflection-data-augmented cold-start pipeline are genuine contributions that go beyond existing static selection or single-turn tool-use approaches.

- **Technical soundness:** Adequate with minor concerns. The core method is sound, but the reward function specification is imprecise, and the reflection ablation analysis contains an internal inconsistency that suggests incomplete understanding of the component's contribution.

- **Empirical support:** Broad but has key gaps. Seven benchmarks provide strong breadth, and the ablations are comprehensive. However, the absence of agentic baselines and ground-truth alignment analysis leaves the mechanism claim under-validated. The efficiency analysis is informative but methodologically imperfect.

- **Significance:** High. The problem is important, and the results—particularly a 7B model rivaling GPT-4o on LongVideoReason-eval—are impressive. The framework is likely to influence follow-up work on adaptive temporal resolution in video understanding.

- **Clarity:** Good. Well-structured with clear explanations and helpful figures. Minor notation issues in the reward formulation do not impede understanding.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0, 8.0]
Average score: 5.5
Binary outcome: Accept
