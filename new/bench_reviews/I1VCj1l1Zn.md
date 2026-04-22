Now I have a thorough understanding of the paper and relevant calibration anchors. Let me compose the final review.

## Summary

DLP-LoRA proposes a lightweight (5M parameter) mini-MLP plugin that dynamically selects and fuses multiple task-specific LoRAs at the sentence level using top-p sampling, instead of token-level gating. The key insight is that tokens within a sentence typically share the same task, making per-token routing unnecessary. Evaluation across 26 tasks on 4 LLM backbones shows DLP-LoRA achieves near-oracle single-LoRA performance while keeping inference overhead under 2× single-LoRA time, substantially more efficient than token-level MoE methods.

## Strengths

- **Well-motivated efficiency insight**: The observation from prior work (Xu et al., 2024; Lin et al., 2024b; Muqeeth et al., 2024) that tokens within the same sentence are consistently assigned the same LoRA motivates sentence-level routing as a principled simplification over token-level MoE approaches (Section 1). This avoids per-token gating overhead.

- **Strong efficiency gains over competing dynamic LoRA methods**: Table 7 provides concrete efficiency comparisons: DLP-LoRA achieves 1.20× decoding latency ratio and 1.00× memory ratio vs. baseline LLaMA-2 7B, compared to 10.54×/2.04× for MOLA, 3.54×/1.02× for PESC, 3.58×/1.02× for MoRAL, and 1.29×/1.07× for LoRA-Switch. These are substantial improvements.

- **Near-oracle performance despite dynamic routing**: Tables 1–2 show DLP-LoRA stays within −0.35% average accuracy of per-task single-LoRA oracle across four backbones on MCQ tasks, and within −0.51% to +1.32% on QA metrics. This confirms the sentence-level router can effectively replace manual LoRA selection.

- **Broad evaluation scope**: 26 tasks (17 MCQ + 9 QA) across 4 LLM backbones (Qwen-2 1.5B/7B, LLaMA-2 7B, LLaMA-3 8B), with 10-run averaging and composite-task settings.

- **Practical deployment characteristics**: Table 6 shows inference time scales flatly (1.76×→1.83× as LoRA count doubles from 50→100), and all experiments run on a single GTX 2080Ti.

## Weaknesses

### Fatal
None.

### Major

- **No task-performance comparison with competing dynamic LoRA fusion methods on the main evaluation benchmark.** The paper cites Meteora, MoLE, MixLoRA, LoRA-Switch, MoRAL, and PESC as related methods, yet none appear in Tables 1–3 or Table 5. The only comparison with these methods (Table 7) is on efficiency metrics alone, using a different experimental setup (7 LoRAs on ShareGPT rather than the 26-task benchmark used throughout). Without performance numbers for these methods on the same tasks, readers cannot assess whether DLP-LoRA's efficiency advantage comes with any performance cost. A method that is faster but worse on task accuracy is a qualitatively different result than one that is faster *and* competitive on task performance. The paper's primary claims about "outperforming" are measured against no-adaptation baselines (Table 3) rather than against the most relevant alternatives.

- **The "fusion" contribution of top-p is not validated.** The router achieves 98.45% classification accuracy (Section 4.1), meaning that for the vast majority of sentences, top-p likely selects a single dominant LoRA. The paper provides no analysis of how often top-p selects 2+ LoRAs, nor any ablation comparing top-p fusion against simple argmax (top-1) routing. Without this, it is unclear whether the "dynamic fusion" aspect of the method contributes anything beyond single-LoRA selection. The case study in Figure 3 shows one example with 50.5%/49.5% split—suggesting ambiguous routing rather than genuine multi-task fusion—but no systematic analysis is provided.

### Minor

- **Missing ablation on the top-p threshold p.** The value of p is never specified in the methodology or experiments, and no sensitivity analysis is provided. Given that this is a key hyperparameter controlling the number of LoRAs fused per sentence, its omission reduces reproducibility and makes it impossible to assess how performance and efficiency vary with p.

- **Efficiency comparison uses a different setup than the main evaluation.** Table 7 uses 7 LoRAs on ShareGPT, while the main results use 26 LoRAs across diverse benchmarks. Scaling characteristics may differ between these regimes. While Table 6 shows latency ratios at 50–100 LoRAs, no task-performance numbers are reported at that scale.

- **The "92.95% relative improvement" and "smaller LLM outperforms larger LLM" framings are overclaimed.** The 92.95% improvement (Table 3) is over *unadapted* base LLMs, and Table 5 compares a LoRA-adapted 1.5B model against an *unadapted* 13B model—that task adaptation improves over no adaptation is expected, and these comparisons risk misleading readers about the method's specific contribution. The abstract and introduction should more clearly attribute these gains to LoRA adaptation in general rather than DLP-LoRA specifically.

### Trivial
None.

## Nice-to-Haves

- Including Meteora and LoRA-Switch as baselines on the 26-task benchmark with full performance + efficiency numbers would make the contribution much more compelling.
- An ablation comparing top-p fusion vs. greedy (top-1) selection, and a frequency analysis of how many LoRAs are typically selected per sentence, would isolate the contribution of fusion vs. routing.
- Failure analysis on the ~1.5% of sentences misclassified by the router, and analysis of whether multi-LoRA fusion helps or hurts in ambiguous cases.
- Statistical significance information (standard deviations from the 10 runs) to contextualize the small performance differences.

## Removed Points

*These points were flagged by reviewers but are removed or weakened for the following reasons:*

- **"The paper claims DLP-LoRA outperforms single LoRA, but data shows parity/slight degradation"**: The paper's actual claim in Section 4.2 states "DLP-LoRA can match or even exceed the performance of individually fine-tuned single LoRAs," which is accurate given the mixed results across tasks. The abstract's "outperforming different LLMs backbones" refers to base LLMs, not single LoRA. This is a minor framing concern, not a factual error.

- **"Equation 6 notation is unclear"**: Minor notation clarity issue; the batched computation formulation is decipherable with context. Not a substantive methodological concern.

- **"No standard deviations from 10 runs"**: Non-standard reporting in this field; differences shown are at the scale of percentage points, which are meaningful given DLP-LoRA's primary claim is parity with oracle LoRA.

- **"Mini-MLP architecture undisclosed"**: The paper states it is a "4-layer mini-MLP" with "5M parameters" and uses ALBERT tokenizer (Section 3.1). While full architectural details (hidden dimensions, activation functions) are not in the main text, this is a minor reproducibility concern rather than a methodological flaw.

- **"Missing Meteora, MoLE, LoRA-Switch from performance baselines"**: This concern is retained in Major Weaknesses above but the phrasing is adjusted—it is a gap in the experimental comparison, not an assertion that these methods do not exist.

## Novel Insights

The key insight—that sentence-level routing eliminates the per-token overhead of MoE-style LoRA fusion while achieving near-oracle performance—is both simple and effective. The 98.45% routing accuracy suggests that multi-LoRA fusion (the "dynamic fusion" in the title) may be unnecessary for most inputs; the method's practical contribution may be more about efficient LoRA *selection* than LoRA *fusion*. This distinction is not explored in the paper but has implications for how future work should frame similar contributions.

## Suggestions

1. **Add performance comparisons with Meteora and LoRA-Switch on the 26-task benchmark.** These are the most directly comparable methods and the paper already cites them; their absence from the main results is the most important gap to address.

2. **Ablate top-p fusion vs. top-1 greedy selection.** Report task performance with argmax routing (which would be the simplest baseline) alongside top-p, and analyze how many LoRAs are typically selected. This will clarify whether "fusion" matters or whether this is primarily a LoRA-selection method.

3. **Tone down claims about "outperforming" larger LLMs** (Table 5) and "92.95% improvement" (Table 3). These comparisons conflate the general benefit of task adaptation with DLP-LoRA's specific contribution.

## Calibration Anchors

| Anchor Paper | Avg Score | Decision | Comparison |
|---|---|---|---|
| FLoRA (w4abltTZ2f) | 8.0 | Accept (Oral) | Batched LoRA inference, clean formulation, strong efficiency + performance baselines. DLP-LoRA has weaker baseline comparisons but similar efficiency focus. |
| MeteoRA (yOOJwR15xg) | 6.2 | Accept (Poster) | Directly comparable topic (multi-LoRA MoE fusion). MeteoRA was accepted despite limited novelty, with similar efficiency/performance trade-off presentation. |
| SMALLTALK LM (pHOH8FVrTp) | 7.33 | Accept (Spotlight) | Sentence-level routing for MoE with performance baselines. More novel architecture; DLP-LoRA is more applicable but less novel. |
| GLIDER (0gVatTOgEv) | 4.0 | Withdrawn/Reject | LoRA routing method with overclaimed results and missing proper baseline comparisons. DLP-LoRA has similar issues but with more concrete efficiency results. |
| FreeLM (qgLyKwXVDs) | 2.0 | Reject | Overclaimed comparisons against much larger models without proper adaptation baselines. DLP-LoRA has similar issues in Table 5 but the core method is much stronger. |
| CardiCat (vW6rsXAGrz) | 4.0 | Reject | Insufficient/outdated baselines. DLP-LoRA's missing competing method comparisons are similar but less severe since Table 7 provides efficiency comparisons. |

DLP-LoRA sits between GLIDER/CardiCat (rejected for missing baselines, ~4.0) and MeteoRA (accepted poster, ~6.2). It has a real efficiency contribution validated in Table 7, and performance parity with oracle single LoRA is meaningful. However, the absence of task-performance comparisons with competing dynamic fusion methods and the unvalidated fusion mechanism are substantive gaps. I score it slightly above MeteoRA's lower boundary but well below its acceptance threshold due to these comparison gaps.

## Evaluation

- **Originality**: The sentence-level routing idea is straightforward but well-motivated. The top-p fusion mechanism adds limited novelty, especially since its contribution is not validated over simple top-1 selection. Moderate.
- **Importance of research question**: Multi-LoRA serving efficiency is a practical and important problem. High.
- **Claims well supported**: Partially. Efficiency claims are well-supported. Performance claims against the most relevant baselines (other dynamic fusion methods) are absent. Moderate.
- **Soundness of experiments**: Broad (26 tasks, 4 backbones) but incomplete (missing competing method performance baselines, missing ablations on p). Moderate.
- **Clarity of writing**: Generally clear, though some overclaiming in abstract/introduction. Good.
- **Value to community**: Practical deployment value on resource-constrained devices. Good.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>