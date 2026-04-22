Now I have a solid understanding of the paper, the critics' points, and the calibration landscape. Let me synthesize my review.

Key findings from my analysis:

1. **Code Completion failure on Llama3-8B**: This is VERY real. Baseline 55.17 → FTP 35.91 = 35% relative drop. The paper claims "negligible" throughout but this is a catastrophic failure on one model-task combination that is completely ignored.

2. **The "1.30% drop" claim**: This is for Qwen2-7B specifically. For Qwen2, Code Completion drops from 58.43 to 56.74 (~3% relative), which is moderate. The average absolute point drop across 6 tasks is ~1.3 points. The abstract is technically correct for Qwen2 but misleading in not mentioning the Llama3 failure.

3. **Attention as proxy for FFN importance**: Legitimate concern but the ablation (random vs attention) does empirically validate the criterion works. The question of whether FFN-specific metrics would be better is a valid suggestion but not a fatal flaw.

4. **Missing P, N, F ablations**: Real concern - these are important hyperparameters.

5. **LLMLingua2 comparison**: The comparison is informative, not unfair. Remove this criticism.

6. **PyramidInfer re-implementation**: The paper clearly explains why and presents both versions. Remove this criticism.

7. **No error accumulation analysis**: Valid minor concern.

Let me now write the final review.

Now, for scoring calibration:

High anchors (>7): 
- FastGen (8.0): Adaptive KV cache with profiling - much more thorough analysis and strong results
- ReLU Strikes Back (7.33): Exploits activation sparsity with solid experimental backing
- StreamingLLM (7.5): Foundational work on attention sinks
- LLM-Streamline (7.5): Layer pruning with replacement module

Medium anchors (4-6):
- OmniKV (6.0): Token-dropping-free context selection, similar domain, no accuracy loss
- GemFilter (5.25): Token reduction in early layers, rejected, some task-specific issues
- OrthoRank (5.25): Token selection via sink token orthogonality, rejected, limited ablations
- SqueezeAttention (5.5): KV cache compression, poster
- Sparsing Law (5.25): Activation sparsity study, rejected

Low anchors (<3):
- FTP (the other one, gcEhF4nuYI) (3.0): Overclaimed results, unfair comparisons, training required
- Strided Transformers (2.33): Overclaimed "minor degradation" without proper evaluation
- MixAttention (2.0): Overclaimed, limited novelty

This paper FTP sits between:
- Better than the low-scoring papers (it has real experimental results, real speedups, real implementation)
- Worse than the high-scoring papers (FastGen has much better analysis, StreamingLLM is foundational)
- Close to the medium-scoring papers (GemFilter, OrthoRank) which also have speedup-accuracy tradeoffs with limited ablations

The key differentiator: the Code Completion failure on Llama3 (35% relative drop) is a serious problem that the paper hides with "negligible" claims. GemFilter at 5.25 had some performance loss on smaller models noted but less catastrophically. OrthoRank at 5.25 had limited ablations.

But FTP also has clear strengths: well-motivated profiling, clean algorithm, works well on most tasks, flash attention compatible, training-free.

I'd place this around 5.0-5.5 range. The catastrophic Code Completion failure on Llama3 balanced against genuinely useful speedups on most tasks/models. The missing ablations and the misleading "negligible" claim are substantive. But the method does work for most model-task combinations, and the profiling data is legitimate.

Score: 5.0 - borderline, leaning reject due to the unacknowledged failure mode.</think>## Summary

FTP (FFN Token Pruning) is a training-free method that accelerates the prefilling stage of long-context LLM inference by pruning tokens before the FFN module, using cumulative attention scores to dynamically determine which and how many tokens to prune per layer while preserving pruned tokens' information via the residual connection. The method achieves 1.20–1.45× TTFT speedup across models ranging from 7B to 72B parameters on LongBench tasks.

## Strengths

- **Well-motivated with concrete profiling data.** Figures 2 and 3 establish that prefilling dominates inference time (up to 80% on NarrativeQA) and FFN accounts for over 60% of per-layer prefilling walltime on both Llama3-8B and Qwen2-7B. This data-driven motivation for targeting FFN is clear and compelling (Section 1, Figures 2–3).

- **Clean, principled algorithmic design.** The residual connection naturally preserves pruned tokens' hidden states without zeroing (Figure 4, Eq. 1), the adaptive reserve ratio η per layer accounts for varying attention sparsity across layers (Eq. 3, motivated by Figure 5), and the pseudo-code (Algorithm 1, ~15 lines) shows the method is easy to integrate into existing inference pipelines.

- **Strong ablation isolating the attention-based criterion.** Table 3 shows random pruning with identical token counts collapses accuracy (e.g., Llama3 Single-Doc QA: 37.20→11.14) while FTP preserves it (37.20→36.06), with nearly identical TTFT. This cleanly demonstrates that the attention-based selection is essential and its computational overhead is negligible (1–3% of TTFT).

- **Consistent speedups across most model-task combinations.** Tables 1–2 show FTP achieves 1.20–1.45× speedup on 5 of 6 LongBench tasks across Qwen2-7B, Qwen1.5-32B, and Qwen2-72B with accuracy drops of 1–3 absolute points. FTP also consistently outperforms PyramidInfer on accuracy at comparable or better speedup (e.g., Qwen2 Single-Doc QA: FTP 38.75 vs PyramidInfer 29.19 at 1.22× vs 1.21× speedup).

- **Flash attention compatibility.** The method handles the practical constraint that flash attention does not return weights by recalculating only necessary weights at trivial cost (Section 4.1, Table 3). PyramidInfer's official implementation cannot work with flash attention and OOMs on Qwen2 (Table 1).

## Weaknesses

### Major

- **Catastrophic Code Completion failure on Llama3-8B, unacknowledged and hidden by "negligible" claims.** On Llama3-8B-Instruct, Code Completion drops from 55.17 to 35.91 — a **35% relative degradation** — while PyramidInfer retains 55.24 at 1.10× speedup (Table 1). This is the worst accuracy of any method on this task. The paper repeatedly uses "negligible" across the abstract ("negligible decrease in performance"), Section 4.2 ("negligible drop in accuracy score"), Section 4.4 ("negligible performance drop"), and Section 5. The abstract's headline "1.30% performance drop" refers to Qwen2-7B only and is an average of absolute point drops across tasks with very different score ranges (26–70), masking this failure. The paper provides no analysis of why this failure occurs or under what conditions FTP is unsafe to apply. For a method aiming for general applicability to off-the-shelf LLMs, an unexplained 35% relative drop on a standard benchmark task is a serious gap in the contribution.

- **Insufficient ablation of critical hyperparameters P, N, and F.** The method hardcodes P=100 (initial tokens), N=50 (last tokens), and F=10 (unpruned initial layers) in Section 4.1. With P+N=150 tokens always preserved, this is 3% of a 5000-token sequence but a much larger fraction for shorter inputs. F=10 means FTP does not apply to ~30% of the model's layers on 32-layer architectures. These choices significantly affect both speedup and accuracy, yet no experiment varies any of them. The only ablation (random vs. attention-based pruning, Table 3) validates the selection criterion but not the architectural configuration. Without sensitivity analysis, there is no evidence these values are not overfit to the evaluation suite.

### Minor

- **No analysis of attention as a proxy for FFN importance.** The paper prunes tokens from the FFN based on attention scores, but attention scores reflect which tokens are attended to by other tokens, not necessarily which tokens require FFN computation for useful hidden representations. The ablation in Table 3 shows attention-based pruning vastly outperforms random, but this only establishes that *some* structured criterion is needed. No experiment compares attention-based selection to FFN-specific importance metrics (e.g., activation magnitude, gradient-based measures) that might better capture which tokens benefit from FFN updates. The Code Completion failure on Llama3 could be a symptom of this proxy mismatch for code-structured inputs. While the attention score is a practical and cheap choice (available immediately before FFN), the paper should acknowledge this limitation more explicitly.

- **No analysis of error accumulation across layers.** Pruned tokens skip FFN updates at each layer; subsequent layers receive stale hidden states for these tokens. The paper does not analyze how these errors accumulate, how many tokens are pruned per layer, or how hidden states diverge from the full-model baseline as depth increases. This is relevant to understanding failure modes like the Llama3 Code Completion issue.

- **The speedup–pruning rate discrepancy is unexplained.** Figure 6 shows 95% of attention mass concentrates on ~60% of tokens, and η is set to 0.90–0.95. This suggests only 5–10% of tokens might be prunable by the attention criterion, yet the method achieves 1.20–1.45× TTFT speedup. The paper does not report how many tokens are actually pruned per layer, making it hard to understand the source of the speedup. Clarifying the per-layer pruning rates and how they translate to walltime savings would strengthen the empirical analysis.

### Trivial

None.

## Nice-to-Haves

- Investigate the Llama3-8B Code Completion failure with a qualitative analysis of which tokens are pruned vs. retained, and whether FFN-specific metrics would perform better for code domains.
- Report per-layer pruning rates and how cumulative FFN-skipping affects hidden state quality relative to the full model.
- Compare with a uniform token-skipping baseline (same fraction of FFN computation skipped with proper scaling) to further isolate the value of the attention-based criterion.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"LLMLingua2 is not a fair baseline for TTFT speedup"** — The comparison is informative: LLMLingua2 is a prompt compression method that the paper correctly identifies as failing to accelerate prefilling (Table 1 shows 0.66–0.82× TTFT). This motivates the need for in-model approaches. The comparison is not misleading; presenting a weaker baseline is acceptable.

- **"PyramidInfer re-implementation raises concerns about implementation quality"** — The paper clearly explains the reason for re-implementation (official version doesn't use flash attention, OOMs on Qwen2) and presents both versions (PyramidInfer* and PyramidInfer) transparently. This is standard practice and not a weakness.

- **"Criticism that FFN is O(L) while attention is O(L²), so FFN is not the asymptotic bottleneck"** — The paper correctly targets the practical regime of 5k–15k token contexts where FFN dominates walltime (Figure 2–3). Asymptotic arguments at extreme context lengths are outside the paper's stated scope.

- **"Concern about the explanation for larger model speedups in Section 4.5"** — The explanation that deeper models offer more layers for pruning and larger FFN weights amplify savings is reasonable, even if it could be stated more precisely. This is a minor presentation issue, not a methodological weakness.

## Novel Insights

The most insightful observation from the reviews is that the Code Completion failure on Llama3-8B but not Qwen2-7B may reveal that FTP's effectiveness is highly model-architecture-dependent, not just task-dependent. This deserves investigation: Llama3-8B has an 8k context window (requiring truncation of longer Code Completion inputs) while Qwen2-7B supports 32k, which could explain the discrepancy. The paper does not discuss how context window truncation interacts with attention-based pruning.

## Suggestions

- Add a clear "Limitations" paragraph acknowledging the Code Completion failure on Llama3-8B, investigating its cause (architecture? context truncation? η settings?), and providing guidance on when FTP should not be applied.
- Ablate at least one of P, N, or F across a reasonable range to demonstrate the method is not overfit to specific values.
- Report average per-layer pruning rates (tokens pruned / total tokens) alongside speedup numbers so readers can understand the FLOPs-to-walltime mapping.

## Score and Decision

**Calibration anchors:**

- **High (>7):** FastGen (8.0, oral): Adaptive KV cache with profiling, thorough analysis, clean results. StreamingLLM (7.5, poster): Foundational attention sinks work. ReLU Strikes Back (7.33, oral): Activation sparsity exploitation with solid experiments.
- **Medium (4–6):** OmniKV (6.0, poster): Token-dropping-free context selection, 1.68× speedup, no accuracy loss. GemFilter (5.25, reject): Early-layer token reduction, some task-specific issues. OrthoRank (5.25, reject): Token selection via sink token, limited ablations.
- **Low (<3):** FTP-token-wise (3.0, reject): Overclaimed pruning, unfair comparisons, training required. Strided Transformers (2.33, reject): Overclaimed "minor degradation" without real evaluation.

This paper is clearly above the low-scoring anchors — it has real implementation, genuine speedups on most tasks, and a well-motivated approach. However, it falls short of the high-scoring anchors (FastGen, StreamingLLM) which have thorough analysis and no hidden failure modes. Compared to medium-scoring papers: OmniKV (6.0) achieves speedup *without* accuracy loss and has fewer unexplained failure modes; GemFilter (5.25) and OrthoRank (5.25) both had limited ablations and some task-specific issues, similar to this paper. The unacknowledged 35% relative Code Completion drop on Llama3 is more severe than anything in those papers, but the method also works remarkably well on most other combinations. I place this slightly below GemFilter/OrthoRank due to the severity and lack of acknowledgment of the failure mode.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>