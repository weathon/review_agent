=== CALIBRATION EXAMPLE 30 ===

# Final Consolidated Review
## Summary

VideoZoomer proposes an agentic framework for long video understanding where a 7B MLLM dynamically controls its temporal focus by invoking a `<video_zoom>` tool to request high-frame-rate clips at autonomously chosen moments, progressing from a coarse low-frame-rate overview to fine-grained evidence gathering. The method employs a two-stage training strategy: cold-start SFT on distilled exemplar and reflection trajectories, followed by GRPO-based reinforcement learning to optimize the tool-interaction policy. Experiments across seven benchmarks show the model outperforms open-source baselines and rivals proprietary systems while consuming fewer frames.

## Strengths

- **The agentic "glance-then-zoom" paradigm is a genuine conceptual advance over static frame selection methods.** Unlike prior adaptive selection approaches (e.g., TSPO, Frame-Voyager) that decouple frame selection from reasoning, VideoZoomer integrates evidence gathering directly into the reasoning loop, enabling iterative refinement and self-correction. The paper demonstrates three distinct emergent reasoning patterns (direct-hit, progressive, self-refine; Figure 3) that static methods fundamentally cannot produce.

- **The two-stage training strategy with reflection data is well-motivated and empirically validated.** The reflection data augmentation—where expert models correct student failures—is a targeted solution to the "shallow policy" problem (model makes one tool call then guesses). The ablation in Table 3 and Figure 5 confirms this: removing reflection data collapses average tool calls from ~2.0 to ~1.0, with corresponding performance drops (e.g., LongVideoReason: 80.3→75.1). The conditional tool-use reward \(R_{tool}\) similarly prevents policy collapse (Figure 5, w/o \(R_{tool}\)).

- **Comprehensive ablation coverage validates each component's necessity.** Table 3 systematically removes RL, cold-start, reflection data, and \(R_{tool}\), showing substantial drops in each case. The OOD experiments (Tables 7–8) confirm the training does not catastrophically forget short-video or logical reasoning capabilities.

## Weaknesses

- **Efficiency claims are based on frame counts, not inference time or compute cost.** The paper repeatedly claims "superior efficiency" (Abstract, Introduction, Section 4.2, Figure 6), but efficiency is measured solely by the number of video frames consumed. VideoZoomer executes up to 4 sequential tool-call rounds, each requiring a full forward pass through a 7B model with newly injected high-frame-rate clip tokens. The per-sample wall-clock latency and total FLOPs could be substantially higher than a single-pass baseline consuming the same total frames. No latency or compute analysis is provided. This is a critical gap because for practical deployment, frame-budget savings that come at the cost of multi-turn sequential inference may not constitute a net efficiency gain.

- **The 80.3 score on LongVideoReason-eval versus GPT-4o's 60.7 warrants scrutiny.** The model is trained on LongVideoReason (52K pairs) and evaluated on LongVideoReason-eval, which shares the same dataset origin (Chen et al., 2025). While the paper treats these as separate splits, the distributional alignment between training and evaluation data is likely much stronger for VideoZoomer than for the proprietary baselines (GPT-4o, Gemini-1.5-Pro) that were not fine-tuned on this data family. The paper does not discuss to what extent this in-distribution advantage explains the unusually large gap (nearly 20 points over GPT-4o), which is an outlier compared to the more modest gaps on other benchmarks (e.g., MLVU dev: 66.8 vs 64.6). Without such analysis, it is unclear whether this result reflects genuine reasoning capability or distributional familiarity.

- **Heterogeneous frame budgets in Table 1 make cross-model comparison difficult.** The authors evaluate VideoZoomer at up to 128 frames (64 base + 4×16 zoom) but many baselines in Table 1 are evaluated at their default (and often lower) frame counts. Only Video-R1 is explicitly re-evaluated under the authors' 128-frame protocol (marked with †). The efficiency plots in Figure 6 partially address this for three benchmarks but do not cover all results in Table 1. This asymmetry means some performance gaps may reflect greater visual information access rather than the proposed method's superiority.

- **The ablation does not disentangle the contribution of distillation quality from RL optimization.** The cold-start data is distilled from GPT-4o and Gemini-2.5-Pro—models that are themselves highly capable. The paper does not include an ablation using weaker or open-source demonstrators for SFT. This makes it unclear whether RL discovers genuinely novel strategies or primarily refines patterns inherited from proprietary model distillation. If the latter, the contribution of RL may be more stylistic than substantive, and the training recipe's generalizability to settings without strong demonstrators is uncertain.

- **Sparse terminal reward in a multi-turn setting creates a credit assignment challenge.** \(R_{acc}\) is assigned only at trajectory end based on final answer correctness. In trajectories with up to 4 tool calls, the model must learn which intermediate zoom decisions contributed to the correct outcome. While GRPO's group-relative advantage partially mitigates this, the paper does not discuss whether intermediate rewards (e.g., for retrieving clips overlapping with ground-truth key segments) were considered, or why they were excluded. This design choice may slow convergence or lead to local optima where the model learns to terminate early rather than explore.

- **The mechanism for generating timestamps \([t_{start}, t_{end}]\) and fps values is under-specified.** Figure 12–13 show the model outputting continuous float values (e.g., `{"segment": [12.0, 14.0], "fps": 8}`), but the paper does not describe how the base LLM generates precise numerical values, whether there are discretization constraints, or how \(R_{format}\) handles floating-point variations. Given that LLMs are known to struggle with precise numerical generation, this implementation detail is important for both reproducibility and understanding failure modes.

## Nice-to-Haves

- **Oracle and random zoom baselines** to isolate the learned policy's contribution from mere access to high-resolution frames. An oracle that zooms to ground-truth key segments would establish an upper bound; a random-zoom baseline would establish a lower bound. Without these, it is unclear how much performance comes from *where* the model zooms versus the fact that it zooms at all.

- **Temporal localization metrics (e.g., IoU)** between zoomed segments and ground-truth critical events. The paper claims the model learns "temporal focusing" but only validates this indirectly via downstream accuracy. Direct localization metrics would substantiate the core mechanism.

- **Spatial zoom capability.** The current tool only varies temporal resolution; for questions requiring fine-grained spatial detail (reading text, identifying small objects), a spatial crop tool would complement temporal zoom. This is outside the current scope but a natural extension.

- **Error recovery rate quantification.** The paper qualitatively shows self-correction (Figure 3c, Figure 10) but does not quantify how often the agent successfully recovers from an incorrect initial zoom versus failing permanently.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: "Cold-start dataset distilled from proprietary models limits reproducibility."** The paper commits to releasing code, datasets, and model weights upon acceptance (Reproducibility Statement). The dataset of 11K trajectories will be available, so the dependency on proprietary APIs is only for regenerating the pipeline, not for reproducing results. Per hard rules, removed as it questions availability of cited resources.

- **Weakness: "Data contamination of proprietary models' training data."** The spark finder suggested verifying that test sets weren't in the teachers' training data. This is speculative—there is no evidence of contamination, and it cannot be verified for proprietary models. Removed as speculative.

- **Weakness: "Format reward consistency during early RL training."** The balanced reviewer raised concerns about format adherence during early training. The paper explicitly describes \(R_{format}\) and includes format violations in its reward structure. This is a minor implementation detail. Removed as nitpick.

- **Weakness: "Generalization to generative/non-QA tasks."** The paper's stated scope is long video understanding and reasoning, which is conventionally evaluated via QA benchmarks. Criticizing the absence of generative evaluation is scope creep. Removed.

- **Weakness: "Dependency on external tools assumes random access to video, which may not work for streaming."** This is a deployment concern outside the paper's research scope. The paper evaluates in a standard offline setting. Removed as scope creep.

## Novel Insights

The reflection data mechanism reveals an interesting asymmetry: SFT on expert exemplars alone produces a "shallow" policy that calls the tool at most once, while adding reflection data (expert corrections of student failures) doubles the average tool-call depth and significantly improves accuracy. This suggests that in tool-augmented RL settings, the critical bottleneck is not teaching the model *how* to use the tool, but teaching it *when to persist*—a metacognitive skill that pure imitation cannot convey but that error-correction trajectories can. This observation has implications beyond video: any agentic system trained via SFT+RL may need similar reflection-augmented cold-start data to avoid premature termination of tool-use chains.

## Suggestions

- **Report wall-clock inference time and total FLOPs per sample** alongside frame counts. Even a rough comparison (e.g., average seconds per video for VideoZoomer vs. Qwen2.5-VL at equivalent accuracy) would clarify the practical efficiency trade-off. This is the single most important addition for reviewers to assess the work's real-world viability.

- **Add an analysis of the LongVideoReason-eval performance gap.** Break down whether gains come from in-distribution familiarity or from the zooming mechanism—for example, by evaluating a Qwen2.5-VL baseline fine-tuned on LongVideoReason with uniform 128-frame sampling (no zoom tool) on LongVideoReason-eval. If this baseline also achieves a high score, the distributional alignment concern is validated; if not, the zoom mechanism's contribution is clearer.

- **Include oracle and random zoom baselines.** These are inexpensive to implement and would conclusively demonstrate that the learned policy provides meaningful temporal localization beyond mere access to additional frames.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0, 8.0]
Average score: 5.5
Binary outcome: Accept
