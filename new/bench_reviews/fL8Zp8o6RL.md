Now let me search the human review calibration papers for comparison.---

## Summary
FTP (FFN Token Pruning) proposes to accelerate the prefilling stage of long-context LLM inference by selectively skipping FFN computation for low-importance tokens, as identified by per-layer attention scores. The key insight is that FFN accounts for ~62% of per-layer walltime under flash attention, and pruned tokens retain their pre-FFN representation through the residual connection, limiting quality degradation. The method is training-free, straightforward to implement, and evaluated on LongBench across 7B–72B models with 1.19–1.45× TTFT speedups reported.

---

## Strengths

- **Targeting the right bottleneck under flash attention.** Unlike prior TTFT methods (e.g., PyramidInfer, LazyLLM) that primarily optimize the attention module, FTP targets the FFN, which is actually the dominant computational cost per layer under flash attention (62.4%/61.3% for Llama3/Qwen2 per Fig. 3). This is a specific, validated observation that distinguishes FTP from contemporaneous work.

- **Elegant use of the residual connection to preserve token information.** By zeroing the FFN contribution rather than dropping the token entirely, pruned tokens pass their pre-FFN hidden state through the residual path unchanged. This avoids the representation collapse seen in methods that discard tokens from the computation graph entirely, and is confirmed to be effective: the attention-based strategy vs. random pruning in Table 3 shows dramatically better preservation (e.g., Llama3 Multi-Doc QA: random=7.56 vs. FTP=34.85 at similar pruning counts).

- **Scalability to larger models.** The method shows larger speedups on 32B/72B models (1.37–1.45×) vs. 7–8B (1.19–1.30×), and results on Qwen2-72B show very small accuracy drops across most tasks. This is practically significant for deployment settings.

- **Controlled ablation establishing selection signal quality.** Table 3 provides a careful random-pruning ablation matching pruned token counts layer-by-layer, directly isolating the contribution of the attention-score criterion and confirming that overhead from attention recomputation is 1–3%.

---

## Weaknesses

### Fatal
*None that single-handedly invalidate the core idea, but Major #1 below constitutes a central-claim failure.*

---

### Major

1. **The headline "negligible decrease in performance" claim is directly violated by the Llama3-8B code completion result.** Table 1 shows Code Completion on Llama3-8B: 55.17→35.91, a ~35% relative drop. This is one of six task families evaluated, not a hidden footnote. The paper's abstract reads: *"only a negligible decrease in performance"* and the conclusion repeats *"significant acceleration while maintaining performance."* These statements are inconsistent with a third of the accuracy being lost on one task. The paper offers zero explanation for why code completion is disproportionately affected or how practitioners should know when FTP is safe to apply. The Qwen1.5-32B Synthetic task also shows a 12% relative drop (52.67→46.25), further complicating the "subtle impact" characterization of Section 4.5. This is not a minor caveat—it means the method's reliability is task-dependent in ways the paper does not characterize.

2. **LazyLLM, the most directly comparable prior method, is discussed in related work but never benchmarked.** Section 2.1 explicitly describes LazyLLM as a prefilling token-pruning approach that uses a dynamic strategy and auxiliary cache. It targets exactly the same setting (TTFT reduction via token dropping during prefilling). Not providing a direct experimental comparison with it is a meaningful gap: without it, the claim that FTP improves over the state of the art in prefilling acceleration is unsubstantiated for the closest prior work.

3. **The pruning criterion is validated only against random selection, not against other plausible importance signals.** The paper's core algorithmic contribution—using cumulative attention mass from the last N queries with threshold η—is only compared to random pruning in Table 3. There is no comparison to: fixed keep-ratio schedules, fixed top-k, prior-layer attention signals, hidden-state norms, or SnapKV-style pooled attention. The paper establishes that attention-based selection is far better than random; it does not establish that the specific proposed formulation is the right or best signal.

4. **No principled justification or sensitivity analysis for key hyperparameters (F, η, P, N).** Different values are used per model (η=0.90 for Llama3, 0.95 for Qwen2-7B, 0.90 for Qwen1.5-32B, 0.93 for Qwen2-72B; F=10 for all but with different subsequent layers). The paper says it empirically found shallow layers are sensitive (Appendix 6.1 is referenced but not available), but there is no systematic sweep showing how performance and speedup vary with F and η. Without this, it is unclear whether the numbers reported reflect optimized per-model tuning or general robustness.

---

### Minor

5. **Evaluation context lengths (5k–15k) do not match the motivating use case (128k–200k).** The introduction explicitly names GPT-4, Qwen2, and Claude-3 with 128k–200k context windows as the motivating scenario. LongBench averages 5k–15k tokens per sample (stated in Sec. 4). The overhead of attention weight recomputation scales as O(N·L) per layer; at 100k tokens, this cost could become non-negligible relative to FFN savings, but this is not tested. Whether the speedup and quality tradeoffs hold at the scales that motivated the paper remains unverified.

6. **Impact of "stale" FFN states on the KV cache during decoding is not analyzed.** Pruned tokens have their FFN update zeroed, so their hidden states stored in the KV cache differ from the full-model states. The paper implicitly assumes this has negligible downstream effect on decoding quality, but provides no analysis (e.g., generation perplexity vs. baseline, or per-step quality tracking). For tasks with long generation (e.g., code completion—exactly the task that fails badly), this could matter.

7. **The PyramidInfer comparison includes a knowingly disadvantaged baseline (PyramidInfer\*).** The paper acknowledges PyramidInfer\* uses PyTorch attention (not flash attention), which is known to be slower and memory-intensive, and runs OOM on Qwen2. Reporting this as a comparison while knowing it is disadvantaged inflates the apparent superiority of FTP. The relevant comparison is the authors' own flash-attention reimplementation of PyramidInfer, which is legitimate and useful, but deserves more documentation on fidelity.

---

### Trivial

8. **Section 4.5 presents speculative post-hoc explanations for why larger models benefit more** (depth, parameter count) without any controlled experiment. This reads as unfounded rationalization and should be framed as a hypothesis.

---

## Nice-to-Haves

- Evaluate on truly long contexts (32k–128k tokens) to validate the motivating premise, even for a subset of models/tasks.
- Ablate cumulative-mass thresholding vs. fixed-ratio and fixed-top-k variants to provide stronger justification for the η formulation.
- Add a per-dataset breakdown (all 16 datasets rather than 6 task averages) to make the code completion and other failure modes fully transparent to readers.
- Combine FTP with KV cache compression methods (e.g., SnapKV) to demonstrate complementarity or potential conflicts in a full-pipeline deployment.
- A layer-wise hidden-state similarity analysis comparing FTP vs. baseline representations would help ground the "residual preserves information" mechanistic claim.

---

## Removed Points

*These points are flagged for removal; treat with caution.*

- **[REMOVED — missing related work rule]** Criticisms about missing comparisons with MInference, FlexPrefill, and SeerAttention. These do not appear in the paper text I can verify, and per the rules, I cannot confirm their existence as bases for criticism.
- **[REMOVED — generic strength]** "The paper is well-written and clearly explains the method." Does not distinguish this paper from any other competently written submission.
- **[REMOVED — generic strength]** "The experiments are extensive." Not specific enough given the issues with evaluation scope.
- **[REMOVED — scope creep]** Requests for theoretical error bounds on FFN approximation quality. This is an empirical systems paper; theoretical proofs are not the community norm.
- **[REMOVED — reproducibility nitpick]** Concerns about undisclosed full training logs or hardware configuration specifics beyond what is stated.
- **[REMOVED — already addressed in paper]** The Neutral Reviewer's concern about "overhead of attention recalculation" being unquantified. Table 3 explicitly reports the overhead (7–10ms for Llama3, 8–15ms for Qwen2) and discusses it as 1–3% of TTFT. This is partially addressed; the remaining concern (scalability to 100k+ lengths) is kept as Minor #5 above.
- **[REMOVED — asymmetry favors baseline]** Criticism that the "PyramidInfer* is disadvantaged by using PyTorch attention, making the comparison unfair." Per the rules, unfair comparisons that favor the baseline (PyramidInfer* is disadvantaged, benefiting FTP) are worth noting but cannot be removed as the asymmetry here favors the *authors*. On reflection, this is kept as Minor #7 above rather than removed.

---

## Novel Insights

The observation that FFN—not attention—is the dominant prefilling bottleneck under flash attention (62%+ walltime) is the paper's most practically impactful contribution. Prior TTFT work optimizes attention because that is where the quadratic complexity lies; but with flash attention, the attention kernel is already so efficient that FFN GEMM operations dominate wall-clock time. Pruning tokens before FFN (while preserving them through the residual path) turns out to be a more effective lever for TTFT reduction than attention-centric methods in this regime. The random ablation in Table 3—showing that the selection criterion, not just the pruning count, is what preserves accuracy—adds a concrete empirical anchor to this claim.

---

## Suggestions

1. **Immediately address the Llama3 code completion failure.** Either add a targeted fix (e.g., task-conditional η, or code-specific first/last token preservation logic), or explicitly narrow the paper's scope to exclude code generation settings and explain *why* code completion is uniquely harmed (hypothesis: code requires uniform sequential processing of all tokens, unlike QA which attends sparsely to relevant passages).

2. **Add a direct comparison with LazyLLM** using the same models and datasets. Since LazyLLM is already cited and described, this comparison is essential to establish where FTP stands relative to the closest prior art.

3. **Report per-dataset results for all 16 LongBench datasets** rather than only 6 task averages. Transparency about failure cases strengthens credibility.

4. **Conduct a hyperparameter sensitivity ablation** on η and F showing accuracy-speedup tradeoffs across a range of values, and propose a general heuristic for selecting these without per-model tuning.

---

## Axes

- **Novelty:** Moderate. Pruning tokens before FFN via residual bypass is a specific and clean idea, but builds directly on the existing token-pruning-during-prefilling paradigm (LazyLLM, PyramidInfer). The contribution is more of a focused engineering insight than a paradigm shift.
- **Technical soundness:** Moderate-weak. The mechanism is sound but the algorithmic choice is justified only against random. The headline claim is demonstrably violated.
- **Empirical support:** Mixed. Speedup numbers are credible and the ablation is useful. But the quality claim is not uniformly supported, and the most direct competitor (LazyLLM) is absent from experiments.
- **Significance:** Moderate. FFN-focused prefilling acceleration is a real gap; the approach is training-free and easy to integrate. Practical impact would be higher if the failure modes were characterized and addressed.
- **Clarity:** Good. The method description (Fig. 4, Algorithm 1) is clear and concise. The results tables are transparent enough that the Llama3 code completion failure is visible—which is why it is a problem that the text ignores it.

---

## Score and Decision

**Calibration:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| LazyLLM (am5Z8dXoaV) | Prefilling token pruning, dynamic selection | 6,5,6,3 | Reject |
| GemFilter (9iN8p1Xwtg) | Early-layer token reduction for prefilling | 6,5,5,5 | Reject |
| FlexPrefill (OfjIlbelrT) | Context-aware sparse attention, prefilling | 8,8,8,8 | Oral |
| KV Prediction (QlvL6eEOC6) | TTFT acceleration | 5,5,5,3 | Reject |
| Recycled Attention (8qYuxV4lRu) | Prefilling acceleration | 6,5,5,5,6 | Reject |

FTP shares the same problem space as LazyLLM and GemFilter (both rejected at avg 5.0 and 5.25). FTP's advantages over those: cleaner engineering insight (FFN vs. attention bottleneck), multi-model evaluation up to 72B, controlled ablation. FTP's disadvantages: LazyLLM missing from comparison; a more severe per-task failure (35% relative drop on code completion, not just a minor miss); headline claim directly contradicted by results. The lack of LazyLLM comparison is particularly damning given that it's cited, and the headline overclaim is worse than what typically sinks a paper in the 5-range. This places FTP below LazyLLM/GemFilter in quality, pointing to ~4.5.

FlexPrefill at 8 is clearly superior: it solves a more general problem (adaptive sparse *attention*), has stronger theoretical grounding, and its results are cleaner. FTP cannot compete with that bar. The paper is closer to the 4-5 range.

**Final score: 4.5 (lean reject)**

The core idea is real and practically relevant, but the empirical support for the central claim is materially undermined by the Llama3 code completion result, the absence of the closest competitor from experiments, and inadequate hyperparameter justification.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>