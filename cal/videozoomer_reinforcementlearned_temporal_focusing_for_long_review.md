=== CALIBRATION EXAMPLE 31 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title "VideoZoomer: Reinforcement-Learned Temporal Focusing for Long Video Reasoning" is accurate and descriptive; the two main ideas (temporal zoom + RL) are clearly signaled. The abstract's claim to "rival proprietary systems" is bold but partially supported — the model does surpass GPT-4o and Gemini-1.5-Pro on LongVideoReason-eval, though those are older proprietary models and the comparison is likely in-distribution (training data and evaluation split come from the same benchmark family). The claim of "diverse and complex reasoning patterns" as an *emergent* capability is somewhat oversold; the diversity is explicitly engineered via the reflection data pipeline rather than being spontaneous emergence.

---

### Introduction & Motivation

The motivation is well-constructed: uniform frame sampling ignores temporal dynamics, static pre-selection is non-interactive, and both suffer from irrecoverable initial errors. These are genuine shortcomings. The framing around cognitive science (Kietzmann et al., 2018) is loose — the cited work concerns representational geometry in neural networks and does not strongly motivate the temporal attention metaphor used.

The three contributions listed are clear. However, the introduction's use of LSDBench (Figure 1, Right) as the primary visual demonstration of efficiency is conspicuous, since this benchmark does not appear in any main experimental table (Table 1 or Table 3). If it is the most compelling benchmark for the core efficiency claim, it should be included in the main evaluation. The omission raises a mild concern about benchmark selection.

---

### Method

**Framework Design.** The "first glance, then zoom" strategy is intuitive and well-described. The tool interface is minimal (a single `<video_zoom>` call with `[tstart, tend]`), which is a defensible choice. However, one critical design question is underspecified: **how does the model learn to predict temporally accurate `[tstart, tend]` values?** Long videos may span hours; if the model's initial 64-frame overview is uniformly sampled, the temporal resolution of the overview may be insufficient to localise a short, critical event. The paper does not discuss temporal grounding — how timestamps are represented (relative vs. absolute?), whether there is any grounding signal in the reward, or what happens when the agent specifies a time interval that *contains* the relevant clip but with large padding (which would waste the per-call 16-frame budget). This is perhaps the most important underspecified design choice in the whole system.

**Cold-Start SFT.** The two-component SFT dataset (exemplar + reflection trajectories) is well-motivated. The observation that SFT on exemplar-only data leads to shallow, single-turn strategies is important and the reflection-data fix is elegant. However:
- The exact composition of the 11K trajectories (how many exemplar vs. reflection?) is not provided.
- The distillation from Gemini-2.5-Pro introduces an important **dependency on a proprietary model**. This is not just a reproducibility concern; the downstream performance ceiling is partly set by the quality of the teacher. The paper does not discuss this limitation.
- The "verifiers" used to filter trajectories are not described. What criteria were used? What fraction of candidate trajectories were rejected?

**RL Training.** The GRPO extension to multi-turn is described at a high level: a "token-level loss mask" over tool-call trajectories. This is sensible but the details are thin. In particular: when GRPO computes advantages across a group of rollouts for the same prompt, how are multi-turn trajectories (which may have different numbers of turns) compared? The group sampling at the trajectory level, rather than at the turn level, is a non-trivial design choice.

**Reward Design.** The reward weights (0.9/0.1/0.5 for acc/format/tool, from Table 5) are provided but no sensitivity analysis is reported. The tool bonus weight of 0.5 is quite large relative to accuracy (0.9), meaning a single correct tool call followed by a correct answer earns 1.4 vs. 0.9 for a direct correct answer — a 55% bonus. It is not obvious this balance is optimal, and ablating reward weights would strengthen the analysis.

The reward is assigned **at the end of each trajectory**. For multi-turn sequences, this means intermediate turns receive no direct signal about whether individual tool calls were useful. This is a known challenge in multi-turn RL, and the paper does not discuss it.

---

### Experiments & Results

**Baseline Coverage.** The paper compares against a wide range of open-source models, which is appropriate. However, the most relevant baselines — **VideoDeepResearch** (Yuan et al., 2025) and **Deep Video Discovery** (Zhang et al., 2025c) — are explicitly acknowledged in the related work as the closest conceptual predecessors (agentic, multi-turn, tool-using approaches for long video) yet are entirely absent from Table 1. The authors justify their absence on the grounds that these methods use large proprietary models, but including them even as reference points (even without a head-to-head) would be important context. This gap is a notable weakness.

**LongVideoReason-eval Contamination Risk.** The training data is **LongVideoReason** (52K Q&A pairs, Chen et al., 2025) and the evaluation includes **LongVideoReason-eval**, the evaluation split of the same dataset. This is an in-distribution evaluation. The remarkable result of 80.3 vs. GPT-4o (60.7) and Gemini-1.5-Pro (67.3) is therefore not a fair comparison to those models, which did not train on this data distribution. The paper presents this as evidence of superiority over proprietary systems, which is misleading. This should be stated as a limitation or the proprietary model comparison restricted to out-of-distribution benchmarks.

**Video-R1 Dagger (†) issue.** Table 1 uses a dagger (†) to indicate Video-R1 was evaluated under the authors' own protocol (128 frames). This is a reasonable disclosure, but it means Video-R1 may actually perform better or worse under its intended setting. The protocol modification is not explained in sufficient depth — what constitutes "our own evaluation protocol" vs. the original?

**Ablation Study.** The ablations in Table 3 are comprehensive and convincing. The key finding that "w/o cold-start" fails to converge is important — it confirms that the method is not end-to-end trainable without proprietary model assistance. The w/o reflection finding (tool calls collapse to ~1.0 on average) is compelling and matches the motivation for the reflection data.

However, **no error bars, confidence intervals, or statistical significance tests** are reported anywhere. Given that multiple benchmarks are multi-choice QA where random chance is 25-50%, differences of 1-2 points may not be significant.

**Efficiency Analysis (Figure 6).** The efficiency comparison is presented as: VideoZoomer using *average* X frames outperforms baseline using *fixed* X frames. This is not a controlled comparison. The baseline at "128 frames" uses exactly 128 frames for every sample; VideoZoomer uses *on average* some number of frames, but may use up to 128 (64 + 4×16) for hard samples. For difficult questions where the model exhausts its budget, the comparison is effectively frame-equivalent. The right interpretation is that VideoZoomer achieves better accuracy *at the same average cost*, which is still valuable, but the presentation slightly overstates efficiency by conflating average and worst-case.

**Missing Baselines on Certain Benchmarks.** Many cells in Table 1 are empty (e.g., Kangaroo, LongVU, LongVA, LongVILA have many missing entries), which makes it hard to assess performance across the full spectrum. The authors likely filled in available published numbers, but this patchwork makes the table hard to parse.

---

### Limitations & Broader Impact

The paper's limitations section is essentially absent (the brief ethics statement does not count). Key unacknowledged failure modes:

1. **Temporal grounding failure**: If the agent cannot accurately predict timestamps (especially in hours-long videos with a 64-frame overview), all subsequent zooms may miss the relevant content. The paper has no analysis of how often the zoomed clip actually contains the relevant information.
2. **Proprietary model dependency at training time**: The entire cold-start phase relies on GPT-4o or Gemini-2.5-Pro. This is a reproducibility and accessibility barrier that should be prominently flagged, not buried in implementation details.
3. **Single-axis interaction**: The tool only supports temporal zooming. Many long video questions require spatial precision (e.g., reading text, identifying a specific object's attributes) where temporal zoom alone is insufficient.
4. **Scalability**: The maximum total frames (128) is evaluated. It's unclear how the framework behaves on very long videos (>1 hour) where even 64 frames at the overview level provides extremely sparse coverage (~1 frame per minute for a 1-hour video).
5. **Failure analysis**: No examples of failure cases or error analysis are provided.

---

### Writing & Clarity

The paper is generally well-written and organized. The reflection data pipeline (Section 3.2) is the clearest contribution description. One structural problem: the combined Table 4 appears **twice** on page 9 (the table is repeated verbatim), which is a copy-paste error in the PDF generation. More substantively, the prompt template referenced in the appendix ("We provide the detailed prompt... as follows:") is not actually included in the extracted text, which may be a rendering artifact.

---

### Overall Assessment

VideoZoomer presents a well-motivated and technically sound agentic framework for long video understanding, combining a coarse-to-fine temporal inspection strategy with a two-stage training approach (SFT cold-start + GRPO-based RL). The core idea is natural and the empirical results are broadly positive, showing consistent improvement over the Qwen2.5-VL base model across multiple benchmarks. The ablation study is thorough and the efficiency analysis is a genuine contribution. However, several concerns limit confidence in the reported gains: (1) the most impressive result — surpassing GPT-4o and Gemini-1.5-Pro on LongVideoReason-eval — is almost certainly an in-distribution comparison given the training data source, and should not be presented as demonstrating superiority over proprietary systems; (2) the closest agentic baselines (VideoDeepResearch, Deep Video Discovery) are absent from all experiments; (3) the critical mechanism of temporal localization (how the agent identifies *where* to zoom) is underanalyzed; (4) the entire pipeline depends on proprietary models for cold-start data, a significant reproducibility and accessibility limitation that deserves explicit acknowledgment. Addressing these concerns — particularly the in-distribution evaluation issue and the missing agentic baselines — would be necessary for the claims as currently stated to stand at ICLR's bar.

# Neutral Reviewer
## Balanced Review

### Summary
The paper proposes VideoZoomer, an agentic framework that enables MLLMs to dynamically control temporal visual focus for long video understanding. Starting from a low-frame-rate overview, the model iteratively invokes a temporal zoom tool to retrieve high-frame-rate clips at autonomously selected moments, refining its evidence through multi-turn interaction. The authors introduce a two-stage training pipeline: cold-start SFT on a curated dataset of distilled exemplar and reflection trajectories, followed by multi-turn GRPO to optimize the agentic policy. Extensive evaluations demonstrate that VideoZoomer consistently outperforms open-source baselines and rivals proprietary models while using a flexible, reduced frame budget.

### Strengths
1. **Well-Motivated and Novel Agentic Paradigm:** The "glance-then-zoom" framework directly addresses the rigidity of uniform sampling and static selectors by enabling iterative, goal-directed temporal focusing. This dynamic evidence-gathering mechanism is a clear conceptual and algorithmic step forward for long-video reasoning.
2. **Thoughtful Cold-Start Design with Reflection Data:** The inclusion of on-policy reflection trajectories during SFT effectively mitigates shallow imitation of teacher models. The ablation (`w/o reflection`) provides strong empirical evidence that this diversity is crucial for stabilizing tool use and enabling multi-turn reasoning (average tool calls drop to ~1.0 without it).
3. **Comprehensive Empirical Rigor:** The paper evaluates across 7 diverse benchmarks (covering understanding, reasoning, and proprietary baselines), provides thorough component-wise ablations, and analyzes frame efficiency, FPS selection distributions, and max tool call limits. This breadth aligns well with ICLR's empirical standards.
4. **Clear Methodology and Reproducibility Signals:** The pipeline, reward formulation, and training hyperparameters are transparently documented. The use of established open infrastructure (`verl`, `vLLM`, GRPO) and the explicit commitment to release code, weights, and datasets strongly support reproducibility.

### Weaknesses
1. **Missing Inference Overhead & Compute Analysis:** While the paper emphasizes frame budget efficiency, it does not quantify the computational cost, wall-clock latency, or total token consumption of the multi-turn agentic process. Iterative environment interaction and intermediate reasoning tokens significantly impact practical deployment, and a single-pass vs. agentic trade-off analysis is missing.
2. **Sparse Reward Design Lacks Deeper Analysis:** The reward relies heavily on final-answer accuracy, which is sparse and can induce high variance or optimization instability in long-horizon tool-use scenarios. Although the paper adapts DAPO and uses a conditional `Rtool` bonus, it does not discuss rollout success rates, reward variance, or potential reward-hacking behaviors during RL training.
3. **Uncritical Reliance on Proprietary Distillation:** The ~11K cold-start trajectories are distilled from GPT-4o and Gemini-2.5-pro. The paper does not address the financial/compute cost of this distillation process, nor does it explore how performance scales when using open-source or weaker teacher models, limiting insights into the framework's scalability.
4. **Narrow Generalization Evaluation:** The OOD analysis is limited to short-video captioning and CLEVRER. It remains unclear whether the learned zoom policy generalizes to other critical long-video tasks such as dense temporal grounding, open-ended summarization, or event localization, which would better validate the learned temporal reasoning capabilities.

### Novelty & Significance
The work presents a timely and substantive contribution to the intersection of agentic reasoning, reinforcement learning, and multimodal understanding. By reframing long-video comprehension as a sequential, budget-constrained tool interaction task, VideoZoomer moves beyond the static sampling paradigm that dominates current literature. The combination of reflection-enhanced cold-start initialization and multi-turn RL for dynamic temporal focusing is methodologically sound and empirically validated. Given ICLR's emphasis on learning algorithms that enable complex, interactive reasoning and efficient computation allocation, this paper fits well within the conference's scope and offers a framework that is readily extensible to other temporally intensive reasoning tasks.

### Suggestions for Improvement
1. **Quantify Inference Costs:** Add an analysis reporting average inference latency, wall-clock time, and total token count (including intermediate CoT and tool outputs) per query. Compare these metrics against static baselines to provide a complete accuracy-efficiency-compute trade-off.
2. **Deepen RL Training Diagnostics:** Include metrics such as rollout validity rates, entropy curves, and the frequency of invalid/exceed-budget tool calls during training. A sensitivity analysis on the `Rtool` bonus weight and format reward would strengthen confidence in the reward design.
3. **Analyze Teacher Model Diversity & Cost:** Report the approximate compute/financial cost of generating the 11K cold-start trajectories. Conduct a small-scale ablation using an open-source teacher (e.g., Qwen2.5-VL-72B) to demonstrate that the pipeline does not strictly depend on proprietary APIs.
4. **Expand Task Generalization:** Evaluate the learned policy on a temporal localization or dense captioning benchmark to verify that the zoom mechanism captures generalizable temporal reasoning skills rather than overfitting to multiple-choice QA formats.
5. **Clarify Error Handling & Safety:** Explicitly detail how the environment masks or penalizes invalid segment requests (e.g., out-of-bounds, negative durations, budget violations) and report the model's error recovery rate during inference to bolster robustness claims.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Wall-clock latency vs. accuracy:** The efficiency claim relies on frame count, ignoring the overhead of multi-turn inference. Add end-to-end latency measurements to verify if "fewer frames" actually translates to faster inference compared to single-pass baselines.
2. **Recent adaptive baselines:** Comparisons are mostly against static VLMs (LLaVA, etc.). Include recent dynamic frame selection methods (e.g., Frame-Voyager, TSPO) as standalone agentic baselines to isolate the benefit of RL vs. simple selection.
3. **Fair proprietary constraint:** Claims rival GPT-4o/Gemini, but they use native long context. Evaluate proprietary models under the same 128-frame budget to ensure the performance gap isn't simply due to information density differences.
4. **Invalid action rate:** Report the frequency of hallucinated tool calls or syntax errors during inference. High invalid action rates would undermine the reliability of the agentic policy in deployment.
5. **Video length generalization:** Test performance stratified by video duration (e.g., 5min vs. 60min). The policy might overfit to training lengths, undermining the "long video" generalization claim.

### Deeper Analysis Needed (top 3-5 only)
1. **Reward hacking analysis:** Analyze if the model invokes zoom on questions solvable by the initial overview. Without this, the `R_tool` bonus may encourage unnecessary actions rather than genuine reasoning.
2. **Perception vs. Reasoning failure:** Break down errors into "missed zoom target" vs. "incorrect reasoning after zoom." This determines if the bottleneck is the policy or the base MLLM.
3. **RL Stability across seeds:** Report variance over multiple RL seeds. Single-run RL results are insufficient for ICLR to trust policy convergence and stability.
4. **Teacher Upper Bound:** Quantify the accuracy of the expert model (Gemini/GPT) used for distillation. If the teacher is inaccurate, the SFT ceiling limits the student's potential.
5. **Turn count distribution:** Provide a histogram of tool calls per sample, not just the average. This reveals if the model collapses to a fixed strategy (e.g., always 1 call) despite the RL optimization.

### Visualizations & Case Studies
1. **Zoom timeline alignment:** Plot tool call timestamps against ground-truth event locations. This visually confirms whether the policy actually finds critical moments or guesses randomly.
2. **Failure trajectory case study:** Show a complete trace where the model zooms repeatedly but fails. This exposes whether the model recovers from bad zooms or spirals into error.
3. **Attention heatmaps:** Overlay model attention on frames before and after zooming. This verifies if the "zoom" mechanism actually shifts focus to relevant regions or just adds tokens.
4. **Token usage distribution:** Plot the distribution of input token counts per turn. This reveals if the model conserves context budget as claimed or exhausts it early.

### Obvious Next Steps
1. **Compute-matched efficiency:** Re-evaluate efficiency using FLOPs or GPU-hours instead of frame counts to provide a true cost-benefit analysis.
2. **Open-ended generation eval:** Test on open-ended QA instead of multiple-choice to assess hallucination rates in tool arguments and final answers.
3. **Hard negative testing:** Evaluate on videos with visually similar distractors to test if the zoom policy distinguishes fine-grained details or relies on spurious correlations.
4. **Human preference evaluation:** Conduct human eval on the reasoning traces to verify if the "reflection" steps are logically sound or just verbose filler.

# Final Consolidated Review
## Summary
VideoZoomer proposes an agentic framework for long video understanding where an MLLM dynamically controls its visual focus through iterative temporal zoom operations. Starting from a low-frame-rate overview, the model learns when and where to request high-frame-rate clips via multi-turn tool interactions. The authors introduce a two-stage training pipeline: cold-start SFT on distilled exemplar and reflection trajectories from proprietary models, followed by GRPO-based reinforcement learning.

## Strengths
- **Well-motivated agentic paradigm:** The "glance-then-zoom" framework directly addresses fundamental limitations of uniform sampling and static frame selectors—the inability to correct initial oversights and the rigid allocation of visual context budget. The iterative, goal-directed temporal focusing mechanism is a clear conceptual advance.
- **Thoughtful cold-start design with reflection data:** The insight that SFT on exemplar-only trajectories leads to shallow, single-turn policies is important. The reflection data augmentation—where failed rollouts are corrected by expert models—explicitly teaches error recovery and multi-turn reasoning. The ablation (Table 3) confirms this: without reflection, average tool calls collapse to ~1.0.
- **Comprehensive component-wise ablations:** Table 3 demonstrates that each component (cold-start, reflection data, RL, R_tool bonus) is necessary. The w/o cold-start model failing to converge and w/o R_tool showing policy collapse (Figure 5) provide strong empirical validation of the design choices.
- **Meaningful efficiency demonstration:** Figure 6 shows VideoZoomer achieving higher accuracy than the baseline using fewer average frames (e.g., 0.64 accuracy at 48 frames vs. baseline's 0.581 at 128 frames on MLVU). The FPS analysis (Table 10) shows the model learns to request moderate rather than maximum frame rates, indicating learned efficiency.

## Weaknesses
- **Temporal grounding accuracy is unanalyzed:** The critical question of how accurately the model predicts relevant timestamps remains unaddressed. In videos spanning minutes to hours, a 64-frame overview provides sparse coverage. The paper contains no analysis of zoom precision—what fraction of requested clips actually contain the relevant visual evidence, or how timestamp errors propagate through multi-turn interactions.
- **Closest agentic baselines are missing from experiments:** VideoDeepResearch and Deep Video Discovery are explicitly acknowledged in related work as conceptual predecessors that use agentic, multi-turn tool use for long video understanding, yet they are entirely absent from all experimental comparisons. Including them—even as reference points—would establish whether the learned policy outperforms prompting-based agentic approaches.
- **In-distribution evaluation on LongVideoReason:** The training data is LongVideoReason (52K pairs) and evaluation includes LongVideoReason-eval, the evaluation split of the same dataset. The claim of surpassing GPT-4o (60.7 vs. 80.3) and Gemini-1.5-Pro on this benchmark should be explicitly qualified as an in-distribution comparison where the proprietary models did not train, while VideoZoomer benefited from dataset-specific training.
- **No statistical significance measures:** Tables 1-3 report single numbers without error bars, confidence intervals, or significance tests. Given multi-choice QA formats where chance is substantial (25-50%), small differences may not be meaningful.
- **Inference overhead unquantified:** The efficiency claim is framed in frame counts, but multi-turn inference introduces latency from iterative environment interaction, intermediate reasoning tokens, and tool call overhead. Without wall-clock timing or total token analysis, the practical efficiency advantage remains unclear.
- **Proprietary model dependency for cold-start data:** The entire training pipeline depends on GPT-4o and Gemini-2.5-Pro for trajectory distillation. While the authors commit to releasing code and model weights, the cold-start data generation process itself requires access to proprietary APIs, creating a reproducibility barrier for researchers without such access.

## Nice-to-Haves
- Analysis of zoom precision: report what fraction of requested clips contain ground-truth relevant segments, and how timestamp errors affect downstream accuracy.
- Wall-clock latency comparison against single-pass baselines to verify that frame savings translate to practical inference speedups.
- Multi-seed RL training variance to confirm policy stability.
- Evaluation on open-ended QA rather than purely multiple-choice to assess hallucination in tool arguments.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **LSDBench placement:** The harsh critic notes LSDBench appears in Figure 1 but not main experiments. However, teaser figures commonly highlight compelling results without full reporting—this is not a methodology flaw.
- **GRPO implementation details:** The critic claims GRPO multi-turn details are thin. The paper describes token-level loss masking for tool-call trajectories and provides hyperparameters in Table 5; this level of detail is standard for a methods paper.
- **Trajectory composition specifics:** The exact split between exemplar and reflection trajectories (~11K total) is a minor detail that does not affect reproducibility given the data release commitment.

## Novel Insights
The reflection data design reveals an important insight about imitation learning: SFT on expert trajectories alone teaches the model to mimic successful reasoning but not to recover from failures. By explicitly generating corrected trajectories from failed rollouts, the paper demonstrates a form of "negative example" learning that stabilizes multi-turn policies. The ablation showing single-turn collapse without this data (Figure 5, left panel) suggests that diverse reasoning patterns must be explicitly instantiated during SFT—RL exploration alone is insufficient to discover them from a narrow initialization.

## Suggestions
- **Report zoom hit rate:** Add a quantitative analysis of whether requested timestamp intervals overlap with ground-truth relevant moments (even approximately). This directly addresses the central mechanism of temporal localization.
- **Add closest agentic baselines:** Include VideoDeepResearch or Deep Video Discovery in Table 1, or add a dedicated comparison section. If they cannot be run, report published numbers under comparable settings.
- **Qualify in-distribution results:** Add a note in Table 1 or the text indicating that LongVideoReason-eval shares the same data source as training, making the proprietary model comparison informative but not directly comparable.
- **Report inference latency:** Add average wall-clock time and total token count per query to the efficiency analysis (Figure 6 or a new table) to substantiate the practical efficiency claim.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0, 8.0]
Average score: 5.5
Binary outcome: Accept
