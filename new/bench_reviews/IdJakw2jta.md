## Summary

This paper introduces Long-Form Spatio-Temporal Video Grounding (LF-STVG), extending the STVG task from 20-second videos to 1–5 minute videos. The authors propose ART-STVG, an autoregressive transformer that processes frames sequentially with spatial and temporal memory banks augmented by selective memory strategies—temporal memory selection via TextTiling-inspired boundary detection and spatial memory selection via text-similarity ranking—alongside a cascaded spatio-temporal decoder. Experiments on extended HCSTVG-v2 validation sets show ART-STVG outperforms existing methods with growing margins as video length increases.

## Strengths

- **Selective temporal memory is the key insight and is convincingly validated.** Table 2 provides the paper's strongest evidence: using all temporal memories in long videos *hurts* performance (9.6% m_tIoU vs. 16.7% without memory), while selective temporal memory yields a dramatic 23.0% m_tIoU—a 13.4% absolute gain. This directly validates the core claim that irrelevant memories are actively harmful and that relevance-aware selection is essential for multi-event long videos.

- **Growing advantage on longer videos is well-demonstrated.** Table 1 and Figure 2 show ART-STVG's margin over existing methods increases with video length (from +0.7 m_tIoU at 1min to +7.3 at 5min over TA-STVG), consistent with the paper's thesis that autoregressive streaming is better suited for long-form grounding.

- **Competitive SF-STVG performance confirms architectural soundness.** On HCSTVG-v2 validation (Table 7), ART-STVG achieves 59.2/39.2 m_tIoU/m_vIoU, within 1.2/1.0 of SOTA TA-STVG (60.4/40.2), despite being an autoregressive model not specialized for short videos. This rules out the concern that the architecture trades short-form capability for long-form.

- **Practical streaming design resolves computational bottleneck.** The autoregressive frame-by-frame processing naturally avoids the GPU memory issues faced by methods that must load all frames simultaneously, which is a genuine engineering advantage (as illustrated in Figure 1).

- **40-second training experiment supports robustness of the advantage.** Table 6 shows that even when all methods are retrained on 40-second videos, ART-STVG maintains a large lead (28.3 vs. 21.0 m_tIoU for STCAT), confirming the advantage is architectural rather than a training-data artifact.

## Weaknesses

### Fatal

None.

### Major

- **All methods are trained on 20-second videos but evaluated on 1–5 minute videos, making the benchmark measure zero-shot length generalization rather than long-form grounding capability.** The paper states: "all methods including ART-STVG are trained exclusively on the HCSTVG-v2 training set (average video length 20 seconds)" (Sec. 4.1). While the comparison is fair (all methods face the same constraint), the paper's framing of "LF-STVG" as a new problem setting is misleading: what is actually demonstrated is that ART-STVG generalizes better from short training videos to long test videos. A model trained and evaluated on long videos might produce entirely different relative rankings. The partial mitigation in Table 6 (training on 40-second videos) is a step in the right direction but still falls far short of the 1–5 minute evaluation lengths.

- **Baseline inference protocol for long videos is unspecified, making comparison fairness unverifiable.** Existing methods (TubeDETR, STCAT, CG-STVG, TA-STVG) were designed to process all frames simultaneously. At 3.2 FPS, a 5-minute video produces ~960 frames—far beyond what these architectures handle. The paper does not state whether frames are truncated, uniformly subsampled, or processed in sliding windows. This omission makes the headline results in Tables 1(a)–(e) difficult to interpret. If baselines are evaluated on a truncated or subsampled version, the comparison may not be apples-to-apples with ART-STVG, which processes every frame sequentially.

### Minor

- **Spatial memory selection contributes only marginally.** Table 3 shows that spatial memory selection improves m_tIoU by only 0.9% (22.1→23.0), and even adding all spatial memories without selection only adds 0.8% (21.3→22.1). While the paper presents spatial memory selection as a co-equal contribution alongside temporal memory selection, the empirical evidence shows it is nearly negligible compared to the 13.4% gain from selective temporal memory (Table 2). The contribution framing could be more honest about this asymmetry.

- **Dataset construction criteria are underspecified.** The paper states only that extensions are "based on original YouTube videos, not concatenated clips, and we manually review the extended videos to ensure their quality" (Sec. 4.1). What constitutes "quality"? Are the extended portions guaranteed to contain the target? Are there cases where the target leaves and re-enters? These details affect what the benchmark measures.

- **No error propagation analysis for the autoregressive pipeline.** Since ART-STVG updates memory banks based on each frame's predictions, an incorrect bounding box in frame *i* corrupts the memory for all subsequent frames. The paper provides no per-frame error analysis or comparison of prediction quality at the beginning vs. end of sequences. This is an important open question for autoregressive grounding models.

- **Ablations are only on LF-STVG-3min.** The ablations in Tables 2–5 are conducted only on the 3-minute setting. It would be informative to see whether the relative importance of memory selection, cascade design, etc., changes at 5 minutes, where the task is hardest and the advantage is largest.

### Trivial

- The loss function is deferred to supplementary material (Sec. 3.5), which is standard for space-constrained submissions but limits reproducibility from the main text alone.

## Nice-to-Haves

- Training and evaluating on genuinely long training videos (e.g., extending the HCSTVG-v2 training split to 1–3 minutes) would convert the current zero-shot generalization experiment into a proper long-form evaluation and significantly strengthen the paper's claims.

- Reporting GPU memory usage and inference time per method on each LF-STVG benchmark would quantitatively validate the claimed computational advantage of the streaming design.

- A per-frame or segment-level error analysis would address the error propagation concern and provide insight into failure modes of the autoregressive approach.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Autoregressive structure drives most of the performance gain"** — This claim from the harsh critic is factually wrong. Looking at Table 1, the memory mechanisms (ART-STVG vs. Baseline) contribute 5.8–9.0 m_tIoU, while the autoregressive structure alone (Baseline vs. TA-STVG) contributes −8.3 to +2.3 m_tIoU (negative on 1–2 min, marginal on 3–5 min). The memory selection is clearly the primary driver, not the autoregressive structure. The baseline also *underperforms* TA-STVG at 1min (30.1 vs. 38.4) and 2min (23.0 vs. 25.3), and is slightly worse at 4min (9.9 vs. 10.1), contradicting the critic's claim that it "already outperforms all existing methods."

- **"Low absolute scores suggest the task may be poorly calibrated or the evaluation setup is flawed"** — Low absolute scores on a challenging new task do not indicate a flaw in the paper. This reflects task difficulty, not methodological error.

- **"The paper overstates the spatial memory selection contribution"** — While the gain is small (0.9%), the paper does report it accurately in the ablation table. Overstatement is a matter of framing, not factual error; this is downgraded to Minor.

- **"Loss function deferred to supplementary material is a core component"** — Deferring the loss to supplementary due to space constraints is standard practice in the field and not a substantive weakness. Moved to Trivial.

- **"Missing appendix/proofs"** — The parser strips appendices; these exist in the original submission.

- **"Cascaded design only adds 1.5% m_tIoU and is overstated"** — The gain is real and consistent. The paper reports it as "further boosting performance," which is accurate if modest. Downgraded from a weakness to a minor observation.

- **"Failure cases not shown"** — While failure analysis would strengthen the paper, its absence is not unusual for the venue and the paper already has extensive ablation evidence.

## Novel Insights

The most striking finding—which the paper does not fully emphasize—is the asymmetry between spatial and temporal memory selection. Temporal memory selection transforms memory from actively harmful (−7.1 m_tIoU) to highly beneficial (+6.4 m_tIoU), while spatial memory selection adds only 0.9 m_tIoU. This suggests that in multi-event long videos, the critical bottleneck is *event-level* context management (knowing which event you're in), not *instance-level* spatial context. This finding has implications beyond STVG: for any streaming video task with multi-event structure, temporal boundary-aware memory selection should be prioritized over spatial memory curation.

## Suggestions

- Explicitly state the inference protocol for each baseline on long videos (e.g., frame subsampling strategy, max frame limit) and report computational cost (GPU memory, inference time) to make the comparison fully interpretable.
- Reframe the contribution attribution to foreground selective *temporal* memory as the primary mechanism and treat spatial memory selection and cascade design as supporting components, consistent with the ablation evidence.
- If possible, extend the training set to longer videos and re-evaluate all methods to establish whether the current rankings hold when models are trained on genuinely long-form data.

## Score and Decision

**Calibration anchors:**
- **OmniSTVG** (avg 6.67, Accept Poster): New STVG task + benchmark + method. More comprehensive dataset (10K videos) but less methodological depth. Our paper has stronger methodological innovation (selective memory validated by Table 2) but weaker dataset construction.
- **StreamingVLM** (avg 6.0, Accept Poster): Streaming VLM with KV-cache memory for infinite video. Similar streaming design philosophy. Our paper is more targeted to a specific task with clearer ablation evidence for the memory selection insight.
- **ChangingGrounding** (avg 5.0, Reject): New benchmark + zero-shot method for 3D grounding in changing scenes. Similar pattern of new task formulation, but weaker execution and unclear task formulation. Our paper is clearly stronger with more rigorous ablations.
- **Video Detective** (avg 2.67, Reject): Long video QA with memory, overclaimed. Our paper is substantially better with honest ablations and genuine insight.
- **Self-Forcing++** (avg 7.33, Accept Poster): Autoregressive long video generation with error mitigation. Stronger paper overall with deeper analysis of compounding errors.

This paper sits above ChangingGrounding (5.0) and Video Detective (2.67) due to stronger ablation evidence (especially Table 2), a more rigorous experimental setup, and genuine methodological insight. It is somewhat below OmniSTVG (6.67) because the benchmark construction is less comprehensive (extending only the validation set, not creating a full dataset with training splits). The two major weaknesses (training/eval mismatch and missing inference protocol) are real but do not invalidate the core finding about selective temporal memory. The paper makes a genuine contribution to the STVG field by identifying an important problem and providing a well-motivated solution with strong ablation support.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>