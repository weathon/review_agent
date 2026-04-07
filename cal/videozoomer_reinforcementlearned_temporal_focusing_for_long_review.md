=== CALIBRATION EXAMPLE 63 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me write the review.

---

## Section-by-Section Critical Review

### Title & Abstract

The title "VideoZoomer: Reinforcement-Learned Temporal Focusing for Long Video Reasoning" is descriptive, though "Temporal Focusing" undersells the multi-turn agentic nature of the work. The abstract is generally accurate, but the claim of "rival[ing] proprietary systems" is overstated — the proprietary comparisons are against GPT-4o and Gemini-1.5-Pro, both of which are considerably older than the latest frontier models (Gemini-2.0-Flash is shown in Figure 1 but not in any result table). This framing sets expectations that the experimental section does not fully meet.

---

### Introduction & Motivation

The motivation is strong and well-grounded. The critique of both uniform sampling (all moments weighted equally) and static pre-selection (no correction mechanism) is clearly articulated, and the "glance-then-zoom" analogy to human attention is compelling. Contributions are clearly enumerated.

One concern: the introduction implicitly frames efficiency as a key advantage ("the agent begins with a coarse overview… only consuming a significant context budget when it decides to invoke a tool"), but this claim is about **frame count**, not **wall-clock time or FLOPs**. Multi-turn generation requires sequential forward passes through the full (and growing) context, while a one-shot method processes all frames in a single pass. This latency/compute tradeoff is never addressed anywhere in the paper. Given that efficiency is a central selling point, this omission weakens the claim.

---

### Method (Section 3)

**3.1 Overview (Framework Design)**

The framework is clearly described. The action space is well-constrained: the model specifies `[t_start, t_end]` and an fps value, and receives a clip. The frame budget constraint `f_high × (t_end − t_start) ≤ B` is sensible.

However, some design choices lack justification:
- Why is the initial overview fixed at 64 uniformly sampled frames? This is a significant choice — other approaches (e.g., TSPO) provide intelligent initial selection, and the paper itself shows combining with TSPO boosts performance (Table 4). The choice of uniform initial sampling is pragmatic but unmotivated.
- The maximum number of interaction turns (N=4) and the per-turn budget (B=16 frames) are design choices that seem chosen for convenience but are not ablated systematically beyond Table 11.

**3.2 Cold-Start Initialization**

This is the most novel and well-thought-out component. Two issues deserve scrutiny:

1. **Data composition is opaque.** The final dataset contains ~11K trajectories, but the ratio of expert/exemplar trajectories to reflection trajectories is never reported. Figure 7 shows round-number distributions for both types, but the absolute counts are unclear. This makes it difficult to assess how much of the benefit comes from the reflection data specifically vs. just having diverse multi-turn trajectories.

2. **Verifier details are missing.** The paper states that "all candidate trajectories are passed through verifiers to ensure quality" — but what are these verifiers? Answer-matching against ground truth? A learned reward model? This is a critical quality-control step and deserves explicit description.

3. **Hyperparameter inconsistency.** Section 4.1 states: "we trained our base model with a learning rate of 5 × 10⁻⁶ for 1 epoch." Table 5(a) in the appendix, however, lists "Learning rate: 5e-5." These differ by an order of magnitude. This discrepancy must be clarified.

**3.3 Multi-Turn Tool-Integrated RL**

Extending GRPO to multi-turn tool-calling scenarios is a genuine technical contribution. The reward has three terms: accuracy, format, and a conditional tool-use bonus. The conditional bonus (only awarded if the final answer is correct) is a smart design to avoid spurious tool invocations.

Concerns:
- **Reward weights are not ablated.** The weighting scheme (acc/format/tool = 0.9/0.1/0.5) is non-trivial. The ablation in Table 3 removes the tool bonus entirely (`w/o R_tool`) but does not explore different bonus weights. The chosen 0.5 weight for the tool reward is quite large relative to accuracy; it is plausible that this value significantly influences the learned policy.
- **Conditional tool reward creates confound.** By only granting the tool bonus when the answer is correct, the model is incentivized to call the tool when it is already going to answer correctly. This may not teach *why* calling the tool helps — it may simply co-occur with situations where the model was already capable. A more informative analysis would compare accuracy conditioned on whether the tool was used vs. not.

---

### Experiments & Results (Section 4)

**Main Table (Table 1)**

The most important missing entry is the **base model Qwen2.5-VL itself**. VideoZoomer is initialized from Qwen2.5-VL-7B-Instruct, but this model does not appear in Table 1. Its numbers appear only in Table 6 (appendix) and Table 2, making it impossible to directly read the improvement over the foundation model from the primary comparison table. This should be a mandatory row in Table 1.

Additionally, there are minor numerical inconsistencies: Table 2 reports Qwen2.5-VL (dev) as 58.3, while Table 6 shows 58.1 for the same benchmark; these should be reconciled.

**Proprietary Model Comparisons**

The paper claims VideoZoomer "rivals proprietary systems." The proprietary baselines used are GPT-4o (May 2024) and Gemini-1.5-Pro — both of which are substantially older models. Gemini-2.0-Flash appears in Figure 1 as a teaser comparison but is absent from Table 1. Given that the paper trains using data distilled from Gemini-2.5-Pro, the omission of Gemini-2.0-Flash and Gemini-2.5-Pro from the main table is notable. The "rivals proprietary" claim is misleading in the ICLR 2026 context where much stronger proprietary models exist and are commonly used as references.

**Missing Comparison with Concurrent Agentic Methods**

The most directly comparable baselines — VideoDeepResearch (Yuan et al., 2025) and Deep Video Discovery (Zhang et al., 2025c) — are discussed in related work but never compared against empirically. These methods follow the same agentic paradigm (iterative tool use for long video understanding), differ mainly in using proprietary models vs. a trained small model, and represent the principal alternative to the proposed approach. Their omission from experimental tables is a significant gap. Even a qualitative comparison of frame budgets and accuracy ranges would be informative.

**Training/Test Data Overlap**

The training data is LongVideoReason (52K pairs from Chen et al., 2025), and one of the primary evaluation benchmarks is **LongVideoReason-eval** from the same source. While these may be different splits, the paper does not explicitly address whether the eval set was held out during training. The strongest performance claim — outperforming GPT-4o (60.7→80.3) on LongVideoReason-eval — rests on this benchmark. More assurance about the data split protocol is needed.

**Fair Comparison in Table 1**

Video-R1 results in Table 1 are marked with `†` as "evaluated using our own evaluation protocol under max frames of 128." It is unclear whether Video-R1 was designed and evaluated by its authors at this budget, or whether this is an artificial constraint. This may disadvantage Video-R1. It should be clarified how the authors determined this is a fair setting, or Video-R1's own reported numbers should be included.

**Statistical Significance**

No confidence intervals or variance estimates are reported across any benchmark. On multiple-choice video QA, result variance from random seed choices and sampling during RL can be non-trivial, especially for smaller benchmarks like CLEVRER (Table 8, where the difference is 67.3→68.0). A single run per configuration does not demonstrate robust conclusions.

**Efficiency Analysis (Figure 6)**

The efficiency comparison in Figure 6 plots VideoZoomer's *average* frames against the baseline's *fixed* frames. This is a meaningful comparison in terms of frame count, but the x-axis labels for VideoZoomer are average values across the dataset, not a controllable setting — making the curve for VideoZoomer effectively a single point placed at the average, not a true curve. The presentation implies more granularity than exists.

---

### Ablation Study (Section 4.3)

The ablation study is one of the paper's strongest sections. The four ablation conditions (`w/o RL`, `w/o R_tool`, `w/o cold-start`, `w/o reflection`) are well-chosen and the Figure 5 training dynamics add interpretability. The finding that removing reflection data causes the tool-call rate to collapse to ~1.0 on average is a key insight.

However, one missing ablation: the paper never tests a **non-agentic RL baseline** (i.e., Qwen2.5-VL fine-tuned with RL on LongVideoReason without tool use). This would isolate the contribution of the *agentic* aspect from simply applying RL to the base model. The `w/o RL` condition (SFT-only) does not serve this purpose since it lacks RL altogether.

---

### Appendix

**B.1 (Expert Model Choice):** The comparison of Gemini-distilled vs. GPT-4o-distilled data is useful and the reasoning (Gemini provides more diverse tool-use patterns) is plausible, though the evidence is anecdotal.

**B.4 (FPS Analysis):** Table 10 showing that 66.2% of tool calls request moderate fps (1,2] is genuinely informative and supports the claim that the model learns an efficient policy. This analysis is valuable.

**B.5 (Tool Call Count):** Table 11 showing diminishing returns beyond 2 tool calls is important for understanding the practical operating point. This should be highlighted in the main paper rather than relegated to the appendix.

---

### Writing & Clarity

The overall exposition is clear. One section that genuinely impedes understanding is the efficiency framing throughout: the paper repeatedly claims superior efficiency, but conflates frame count with computation. Section 4.3 says "our model achieves 0.64 accuracy using only 48 frames on average, surpassing the baseline's 0.581 accuracy at a much larger 128 frame budget" — but this ignores that each zoom call requires an additional LLM forward pass with growing context. The actual compute cost could easily exceed that of a one-pass 128-frame baseline. Clarifying this distinction is important.

---

### Limitations & Broader Impact

The limitations section is minimal (ethics statement only, no dedicated limitations section). The following failure modes are unaddressed:

1. **Error in initial glance.** If the coarse 64-frame overview misses a key event entirely (e.g., the event happens very briefly and at the wrong temporal phase for uniform sampling), the model has no mechanism to search globally — it can only zoom into requested segments. The paper demonstrates iterative correction but doesn't discuss how often the initial glance simply provides insufficient information to even know *where* to zoom.

2. **Prompt-injection vulnerability.** In an agentic loop, structured tool calls are parsed by the environment. The paper doesn't discuss what happens if videos contain text that could be crafted to interfere with the `<video_zoom>` tags parsed by the system.

3. **Hallucination under sparse frames.** The model's final answer is conditioned on the zoomed clips plus the initial low-fps overview. There is no analysis of whether the model sometimes hallucinates details from the initial overview that are contradicted by the zoomed clip, or vice versa.

4. **Scalability to hour-long videos.** Even at 64 initial frames, hour-long videos would be sampled at less than one frame per minute — which might miss entire scenes. The paper evaluates on the provided benchmarks without discussing how the approach scales to truly long content (e.g., 1–3 hour films).

---

### Overall Assessment

VideoZoomer addresses a genuine and important problem in long video understanding. The core idea — training a 7B MLLM to perform multi-turn temporal zoom via RL with a carefully designed cold-start — is principled and well-executed. The two-stage training strategy is sensible, ablation results are convincing, and the performance numbers on MLVU and LongVideoReason-eval are impressive. However, the work has several issues that merit revision before acceptance. Most critically: (1) the proprietary model comparisons use outdated baselines (GPT-4o, Gemini-1.5-Pro) while the current frontier (Gemini-2.5-Pro, GPT-4.1) was available at submission time and Gemini-2.5-Pro was even used for distillation; (2) there is no empirical comparison with the most directly relevant concurrent work (VideoDeepResearch, Deep Video Discovery); (3) the efficiency argument throughout conflates frame count with compute, never accounting for the sequential latency overhead of multi-turn inference; (4) there is a notable hyperparameter inconsistency between the main text and the appendix; and (5) the training/test overlap concern for LongVideoReason-eval is unaddressed. The paper is a solid contribution to a fast-moving area and the core method appears to work, but these gaps leave unanswered questions that ICLR reviewers are likely to flag. With appropriate revisions — particularly adding a non-agentic RL baseline, current proprietary comparisons, and an honest compute-cost discussion — the paper would be more convincingly above the acceptance bar.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes VideoZoomer, an agentic framework that enables Multimodal Large Language Models (MLLMs) to dynamically control their visual focus during long video reasoning through a novel "temporal zoom" tool. The authors introduce a two-stage training strategy combining cold-start supervised fine-tuning with reflection trajectory augmentation and subsequent reinforcement learning (using GRPO) to optimize the agent's tool-use policy. Extensive experiments on long video understanding and reasoning benchmarks demonstrate that the 7B model achieves superior performance-efficiency trade-offs compared to open-source baselines and rivals proprietary systems by adaptively sampling frames only when required.

### Strengths
1.  **Agentic Framework for Video Context:** Reframing long video understanding as a sequential tool interaction task is a compelling approach to the context window bottleneck. Unlike static frame selection, the ability to iteratively zoom in on specific segments allows the model to correct initial oversights, effectively managing the information density of long videos.
2.  **Robust Training Strategy (SFT + RL + Reflection):** The two-stage training pipeline addresses common RL instability issues in language models. The inclusion of "reflection data" (correcting failure trajectories from the expert model) is a methodologically sound technique to prevent shallow policy patterns and encourage deeper exploration, which is crucial for complex reasoning tasks.
3.  **Comprehensive Evaluation and Efficiency:** The paper evaluates the model across a wide range of established benchmarks (MLVU, LongVideoBench, VideoMMMU). Crucially, they demonstrate significant efficiency gains, showing that VideoZoomer can outperform baselines with significantly fewer total frames processed, validating the utility of the adaptive sampling policy.
4.  **Transparency and Analysis:** The authors provide thorough ablation studies validating the contribution of each component (reflection data, cold-start, tool reward). Additionally, the analysis of FPS selection and tool call distribution offers valuable insights into the learned policy, showing the model learns to balance detail and budget.

### Weaknesses
1.  **Inference Latency vs. Efficiency Trade-off:** While the paper highlights efficiency in terms of *frame budget* (context window usage), it does not explicitly report *inference latency* or wall-clock time. Multi-turn tool invocation inherently introduces significant latency compared to static sampling methods. For real-time applications, the temporal overhead of iterative tool calls could undermine the claimed efficiency benefits.
2.  **Tool Implementation and Accessibility:** The `<video_zoom>` tool assumes random access to high-frame-rate segments. In many real-world scenarios (e.g., live streaming), such random access might not be feasible or could incur high I/O costs. The paper treats the tool as an idealized oracle without discussing the computational overhead of retrieving these segments, which is a practical limitation for deployment.
3.  **Dependence on Proprietary Distillation Data:** A significant portion of the cold-start data is distilled from closed-source proprietary models (Gemini 2.5 Pro, GPT-4o). While common, this creates a dependency where the reasoning capabilities are inherited from a black box rather than purely learned from open principles. The "less is more" finding (Table 9) suggests quality is key, but the ultimate ceiling is constrained by the expert model's capabilities.
4.  **Comparative Fairness on Proprietary Baselines:** While VideoZoomer claims to rival proprietary systems (Table 1), the comparison relies on specific protocol variations (e.g., max frame counts). Some proprietary baselines (like GPT-4o) are reported with different default settings than the fine-tuned agents. A more rigorous standardization of inference settings (beyond just frame count, such as response time) would strengthen the comparison.

### Novelty & Significance
The novelty lies in the specific integration of **temporal focusing as an active tool within an RL-driven agent** for long video context management. While RL for reasoning (e.g., Video-R1, Math-R1) and agentic search are known trends, combining them specifically for *temporal resolution control* in video is a fresh direction that addresses a distinct gap in MLLM literature. The significance is high; as video data grows, fixed-sampling methods will continue to degrade performance or consume excessive context. An adaptive method like VideoZoomer offers a scalable path forward for open-source models to compete in the long-context video domain without relying solely on context window expansion.

### Suggestions for Improvement
1.  **Analyze Inference Latency:** Add a discussion or table reporting the average inference time (latency) compared to baselines. Since efficiency is a core claim, demonstrating that the "frame savings" compensate for "loop overhead" is essential for a fair assessment of system efficiency.
2.  **Clarify Tool Deployment Feasibility:** Include a brief discussion on the practicality of the `video_zoom` tool in non-ideal environments (e.g., streaming video). If the evaluation assumes offline random access, this should be explicitly stated so readers understand the deployment constraints.
3.  **Error Analysis on Failure Cases:** While case studies are provided, a quantitative failure analysis (e.g., error bars on the benchmarks or a confusion matrix of reasoning types) would help understand when the agent fails to zoom effectively. Are there specific video types where the policy over-consults or under-consults?
4.  **Detail Reward Engineering:** The reward function for the tool usage is conditional on accuracy. Elucidate how the model distinguishes between "unnecessary tool calls" and "necessary tool calls when the first attempt was wrong" during training, beyond just the final answer reward, to ensure the policy learns true utility rather than guessing.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Wall-clock latency and compute cost measurements.** The paper claims efficiency based on frame count, but agentic RL requires multiple sequential forward passes. Without reporting inference time (seconds/query) and GPU cost, the "efficiency" claim is misleading compared to single-pass static baselines.
2. **Controlled input budget for proprietary baselines.** Table 1 claims to surpass GPT-4o and Gemini-1.5-Pro, but does not specify their input frame budgets. If proprietary models used full video while VideoZoomer used 128 frames, the comparison is invalid and undermines the SOTA claim.
3. **Strong retrieval-based baselines.** Compare against standard CLIP-based dense retrieval or Oracle sampling (perfect frame selection). If a simple retrieval method matches performance without RL complexity, the necessity of the proposed agentic framework is unproven.
4. **Tool call failure rate statistics.** Report the frequency of invalid tool calls, hallucinated timestamps, or infinite loops during inference. High failure rates in agentic systems critically undermine reliability claims for real-world deployment.

### Deeper Analysis Needed (top 3-5 only)
1. **Marginal utility of each tool call.** Analyze whether accuracy increases monotonically with the number of zooms. If additional tool calls do not correlate with performance gains, the RL agent may be learning to game the tool bonus reward rather than reasoning effectively.
2. **Sensitivity to initial "glance" quality.** Evaluate performance when the initial low-fps sampling completely misses the critical event. If the agent cannot recover from a bad initial state, the method lacks robustness for truly long videos where key events are sparse.
3. **Quality audit of reflection data.** Quantify the accuracy of the "expert corrections" used in SFT. If the expert model provides noisy or incorrect corrections, the RL policy may learn erroneous recovery strategies, invalidating the training strategy's contribution.
4. **Generalization to unseen video durations.** Test the policy on videos significantly longer than the training distribution (e.g., 1-hour vs. 10-minute). Agentic policies often overfit to specific temporal scales, which would limit the method's broader applicability.

### Visualizations & Case Studies
1. **Failure case trajectories.** Explicitly visualize examples where the model zooms into irrelevant segments and fails to recover. Success cases alone are cherry-picked; failure analysis is required to trust the method's limits.
2. **Zoom interval vs. Ground Truth overlap.** Plot heatmaps showing the temporal overlap between selected zoom intervals and the actual ground truth event timestamps. This proves the agent is targeting evidence rather than guessing randomly.
3. **Inference cost distribution.** Provide a histogram of inference steps and time per query. This reveals the worst-case latency costs which are hidden by average frame budget metrics.

### Obvious Next Steps
1. **Report wall-clock inference time.** Include seconds per query for all baselines and VideoZoomer to validate the efficiency argument practically.
2. **Human evaluation of reasoning chains.** Automated metrics on QA benchmarks are noisy; human judges should verify if the generated reasoning chains logically justify the tool calls.
3. **Zero-shot cross-domain transfer.** Evaluate the trained policy on unseen video domains (e.g., surveillance or sports) without fine-tuning to demonstrate true policy generalization beyond the training dataset.

# Final Consolidated Review
## Summary

VideoZoomer proposes an agentic framework for long video understanding that enables Multimodal Large Language Models (MLLMs) to dynamically control visual focus through multi-turn temporal zoom tool invocations. The method uses a two-stage training strategy: cold-start supervised fine-tuning with curated exemplar and reflection trajectories, followed by reinforcement learning (GRPO) to optimize the tool-use policy. Experiments across multiple long video benchmarks demonstrate improved performance over open-source baselines with adaptive frame budgets.

## Strengths

- **Novel agentic formulation for temporal resolution control.** Reframing long video understanding as sequential tool interaction—where the model can iteratively request high-fps clips at autonomously chosen moments—addresses a fundamental limitation of static frame selection methods. The "glance-then-zoom" paradigm allows the model to correct initial oversights, raising the ceiling on reasoning performance for complex video tasks.

- **Methodologically sound two-stage training with reflection data.** The cold-start phase uses reflection trajectories where an expert model corrects failure modes from an initial agent, preventing the "shallow policy" problem where models learn single-turn tool calls. Figure 5 demonstrates that removing reflection data causes the tool-call rate to collapse to ~1.0, validating this design choice. The ablation study (Table 3) shows each component (cold-start, reflection data, RL, tool bonus) contributes meaningfully.

- **Strong empirical results with efficiency gains in frame count.** The model achieves substantial improvements on MLVU (+10.5 dev, +10.3 test over Qwen2.5-VL), LongVideoBench, and LongVideoReason-eval. Figure 6 shows the model achieves comparable accuracy to baselines using significantly fewer frames on average, supporting the claim that the learned policy allocates visual attention efficiently.

- **Insightful analysis of learned behavior.** Table 10 shows the model selects moderate fps (1-2) in 66.2% of tool calls rather than defaulting to maximum resolution, demonstrating learned efficiency. Table 11 reveals diminishing returns beyond 2-3 tool calls, providing practical guidance for deployment.

## Weaknesses

- **Efficiency claims conflate frame count with actual compute cost and latency.** The paper repeatedly claims "efficiency" based on reduced frame budgets, but multi-turn agentic inference requires sequential forward passes through growing context windows. The latency overhead of iterative tool calls could easily exceed a one-pass baseline that processes more frames upfront. Without reporting wall-clock time or FLOPs, the efficiency claim is incomplete. For real-time applications, this trade-off is critical.

- **Missing comparison with directly relevant concurrent agentic methods.** VideoDeepResearch (Yuan et al., 2025) and Deep Video Discovery (Zhang et al., 2025c) are discussed in related work but not compared against empirically. These methods follow the same agentic paradigm for long video understanding and represent the principal alternative approach. Their omission leaves an important gap in the experimental comparison.

- **Proprietary model comparisons use older baselines.** The paper claims to "rival proprietary systems" but compares against GPT-4o (May 2024) and Gemini-1.5-Pro, while Gemini-2.5-Pro was used for training data distillation and presumably available. Gemini-2.0-Flash appears in Figure 1 but not in the main results tables. Given the rapid pace of proprietary model releases, comparing against current frontier models would strengthen the SOTA claims.

- **Hyperparameter inconsistency between main text and appendix.** Section 4.1 states the learning rate for cold-start training as "5 × 10⁻⁶ for 1 epoch," while Table 5(a) in the appendix lists "Learning rate: 5e-5." These differ by an order of magnitude and need reconciliation.

- **Base model omitted from primary comparison table.** Qwen2.5-VL-7B-Instruct serves as the initialization point, but Table 1 does not include it, making it impossible to directly read the improvement over the foundation model from the main results. The base model numbers appear only in Table 2 (partial) and Table 6 (appendix).

- **Training/evaluation data split protocol unclear for LongVideoReason.** Training uses LongVideoReason (52K pairs from Chen et al., 2025), and one evaluation benchmark is LongVideoReason-eval from the same source. The paper does not explicitly confirm that the evaluation split was held out during training, which matters given the strong performance claim on this benchmark (80.3 vs GPT-4o's 60.7).

- **No mechanism to recover from errors in the initial coarse overview.** If the initial 64-frame uniform sampling completely misses a critical event, the model has no way to discover where to zoom. The paper demonstrates iterative correction when the model identifies relevant segments, but does not analyze how often the initial glance provides insufficient information to even initiate productive search.

- **Verifier details for cold-start data quality control are unspecified.** The paper states that "all candidate trajectories are passed through verifiers to ensure quality" but does not describe what these verifiers are (answer matching? learned reward model? human annotation?). This quality-control step could significantly impact the training data quality.

## Nice-to-Haves

- **Wall-clock latency measurements.** Reporting seconds per query for VideoZoomer vs. static-sampling baselines would validate whether frame savings translate to practical efficiency gains or are offset by multi-turn inference overhead.

- **Non-agentic RL baseline.** An ablation training Qwen2.5-VL with RL on LongVideoReason *without* tool use would isolate the contribution of the agentic framework from simply applying RL to video QA.

- **Confidence intervals for benchmark results.** Single-run results on smaller benchmarks (e.g., CLEVRER's 67.3→68.0) do not establish robustness to random seeds and sampling variance.

- **Failure case analysis.** While case studies show successful reasoning trajectories, quantitative analysis of when the model zooms incorrectly or fails to recover would strengthen trust in the method's reliability.

- **Tool call failure rate statistics.** Reporting frequency of invalid tool calls, malformed timestamps, or budget violations during inference would inform deployment considerations.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Claim about outdated proprietary models being "misleading."* The paper does compare against proprietary models available at the time of submission; calling this misleading overstates the issue. The concern is about completeness of comparison, not deception.

- *Design choice criticism for 64 initial frames and N=4 turns.* Table 11 does ablate the number of tool calls, and 64 frames is a reasonable initial budget. The paper provides justification for these choices, and demanding ablations for every hyperparameter is excessive.

- *Request for CLIP-based retrieval baselines.* While potentially informative, this goes beyond the paper's scope of comparing against existing long video understanding methods. The baselines chosen are appropriate for the field.

- *Concern about unfair Video-R1 evaluation protocol.* The † notation clearly explains that Video-R1 was evaluated under the authors' protocol for fair comparison. This is standard practice and transparently reported.

- *Dependency on proprietary model distillation as a "weakness."* Using GPT-4o/Gemini distillation is standard practice in the field and does not diminish the contribution of training a 7B model to perform agentic video reasoning efficiently.

## Novel Insights

The reflection data mechanism—where failed trajectories from an initial SFT-only agent are corrected by an expert model—provides a principled approach to avoiding policy collapse in agentic RL. The analysis that removing reflection data causes the model to adopt a "shallow" single-call strategy (average tool calls ≈ 1.0 vs. ≈ 2.0 with reflection) is a clear demonstration of how training data diversity shapes exploration behavior. This insight extends beyond video: any agentic system with tool-use could benefit from explicitly including recovery trajectories in cold-start data.

## Suggestions

- Add wall-clock inference time comparison (seconds per query) for VideoZoomer vs. baselines to substantiate the efficiency claim. Even if multi-turn inference is slower, this should be disclosed.

- Include the base model (Qwen2.5-VL-7B-Instruct) in Table 1 for transparent comparison.

- Reconcile the learning rate discrepancy between Section 4.1 (5×10⁻⁶) and Table 5(a) (5e-5).

- Clarify the train/eval split for LongVideoReason and LongVideoReason-eval to address potential data overlap concerns.

- Compare empirically against VideoDeepResearch and/or Deep Video Discovery, even if only with reported numbers from their papers, to situate the contribution among concurrent agentic approaches.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0, 8.0]
Average score: 5.5
Binary outcome: Accept
