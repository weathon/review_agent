Let me re-examine the abstract claim of "1.30% performance drop." The abstract says:

> "Specifically, the Qwen2-7B-Instruct model with FTP achieves a speedup of 1.24× in the prefilling stage with only a 1.30% performance drop compared to the baseline model."

Looking at Table 1 for Qwen2-7B-Instruct, I need to understand how they compute this. Looking at the scores across tasks:
- Single-Doc QA: 39.00 → 38.75 (-0.25)
- Multi-Doc QA: 37.48 → 35.21 (-2.27)
- Summarization: 26.70 → 25.01 (-1.69)
- Few-shot Learning: 70.17 → 69.11 (-1.06)
- Synthetic: 37.50 → 36.75 (-0.75)
- Code Completion: 58.43 → 56.74 (-1.69)

Average baseline score: (39.00 + 37.48 + 26.70 + 70.17 + 37.50 + 58.43) / 6 = 269.28 / 6 = 44.88
Average FTP score: (38.75 + 35.21 + 25.01 + 69.11 + 36.75 + 56.74) / 6 = 261.57 / 6 = 43.595
Average drop: (44.88 - 43.595) / 44.88 = 1.285/44.88 ≈ 2.86%

So the "1.30%" claim in the abstract seems potentially derived differently. But the harsh critic's claim that Code Completion on Llama3-8B drops from 55.17 to 35.91 (a ~35% relative drop) is correct from Table 3 (the random vs ours comparison). Wait, let me recheck Table 1 for Code Completion on Llama3-8B:

From Table 1:
- Llama3-8B-Instruct Code Completion: Baseline 55.17, Ours 35.91 — that IS in Table 1!

So yes, on Llama3-8B, Code Completion drops from 55.17 to 35.91, a substantial relative drop. This is a genuine weakness.

Now let me think about the key issues systematically and form my review.

## Calibration Summary

**High-scoring papers in this area:**
- FlexPrefill: 8/8/8/8 (Oral) - Novel sparse attention mechanism, comprehensive evaluation
- Cascading KV Cache: 6/6/6/6 (Accept Poster) - Training-free method

**Medium-scoring papers:**
- D2O: 6/6/6/5/6 (Accept Poster) - KV cache compression, moderate novelty
- LazyLLM: 6/5/6/3 (Reject) - Pre-filling acceleration, some flaws
- DynamicKV: 5/3/5/6/3 (Reject) - KV cache compression, poor methodology

**Low-scoring papers:**
- FTP (Token Routing): 3/3/3/3 (Withdrawn/Reject) - Learnable router, unclear novelty
- PyramidDrop: 3/3/3/3 (Withdrawn/Reject) - Simple method, insufficient contributions
- HASA: 5/5/5 (Reject)

This paper (FTP for FFN Token Pruning) is more similar to LazyLLM (prefilling acceleration, training-free) and PyramidInfer. LazyLLM got a reject with scores 6/5/6/3. The key question is whether FTP's contributions are substantial enough compared to the weaknesses.

Let me now finalize my assessment:

Key strengths:
1. Well-motivated: FFN dominates prefilling time (60%+) and this is clearly shown with profiling data
2. Simple, elegant mechanism: prune tokens for FFN while preserving attention via residual
3. Training-free, applicable to off-the-shelf models
4. Consistent speedups (1.2-1.45×) across model sizes
5. Strong random-pruning ablation showing the attention mechanism matters

Key weaknesses:
1. **Missing LazyLLM comparison** - this is a genuine gap since LazyLLM is the most directly comparable prior work
2. **Code Completion degradation** on Llama3-8B is severe (55.17 → 35.91)
3. **Metric aggregation issue** - heterogeneous metrics averaged together, "1.30% performance drop" claim is misleading
4. **FlashAttention compatibility / attention recalculation overhead** - acknowledged but not thoroughly analyzed for very long contexts
5. **Hyperparameter sensitivity** not explored systematically

Now let me check if the "1.30% performance drop" is really misleading. The paper says "Qwen2-7B-Instruct model with FTP achieves a speedup of 1.24× in the prefilling stage with only a 1.30% performance drop." This likely refers to averaging across all tasks. For Qwen2-7B, the drops per task are:
- Single-Doc QA: 39.00 → 38.75 (0.64% relative)
- Multi-Doc QA: 37.48 → 35.21 (6.05% relative drop)
- Summ: 26.70 → 25.01 (6.33%)
- Few-shot: 70.17 → 69.11 (1.51%)
- Synthetic: 37.50 → 36.75 (2.00%)
- Code: 58.43 → 56.74 (2.89%)

The 1.30% is likely a percentage point drop relative to the overall average score, not a relative percentage. Average score drops from ~44.88 to ~43.60, which is 1.28/44.88 ≈ 2.85% relative, or 1.28 points absolute. The "1.30%" phrasing is misleading because it sounds like relative drop when it's actually percentage points.

For Llama3-8B, the Code Completion is devastating: 55.17 → 35.91 (34.9% relative drop). The paper actually reports this in Table 1. This is a major issue.

Let me now write the final review.</think>Now I have enough information to write my consolidated review. Let me carefully verify the key claims against the paper text.

## Summary

FTP (FFN Token Pruning) is a training-free method to accelerate the prefilling stage of long-context LLM inference by pruning tokens before FFN computation in each decoder layer. Token importance is determined via attention scores, and pruned tokens retain their attention-updated hidden states through the residual connection (their FFN output is effectively zeroed). The method achieves 1.2–1.45× TTFT speedup across models from 7B to 72B parameters with modest accuracy drops on LongBench.

## Strengths

1. **Well-motivated target**: The paper's strength lies in clearly identifying FFN as the dominant compute bottleneck during prefilling (60%+ of layer walltime, Figure 3) — a departure from prior work that focuses on KV cache compression for decoding or attention sparsity. This is a timely and practical observation given that FlashAttention already optimizes attention computation.

2. **Elegant mechanism**: The insight to prune between attention and FFN modules, leveraging the residual connection to preserve pruned tokens' representation, is clean and simple. The random-vs-attention ablation (Table 3) convincingly demonstrates that the attention-based selection is meaningful — random pruning causes catastrophic degradation (e.g., Synthetic drops to 1.71 on Qwen2) while FTP preserves most performance.

3. **Training-free and broadly applicable**: The method requires no fine-tuning or architectural changes, applicable to any transformer model with residual connections. Demonstrated on 4 model families from 7B to 72B parameters.

4. **Consistent speedups across model scales**: TTFT speedups of 1.20–1.24× on 7–8B models and 1.31–1.45× on 32–72B models, with the speedup increasing with model size (attributed to deeper architectures having more prunable layers).

5. **Dynamic per-layer pruning**: The cumulative attention threshold (η) automatically adapts the number of retained tokens per layer, handling varying attention sparsity patterns — this is a sound design choice versus static pruning ratios.

## Weaknesses

### Major:

- **Severe degradation on Code Completion (Llama3-8B)**: Table 1 shows Code Completion dropping from 55.17 to 35.91 on Llama3-8B-Instruct — a ~35% relative accuracy drop. This is not "negligible" by any standard and suggests the method can fail catastrophically on tasks where most tokens carry structural/semantic importance (code tokens are far less redundant than natural language). The paper does not discuss this failure mode or its implications for when FTP should not be applied.

- **Misleading abstract claim of "1.30% performance drop"**: The abstract states the Qwen2-7B model has "only a 1.30% performance drop," but this averages heterogeneous metrics (F1, Rouge-L, accuracy, edit similarity) across 6 tasks into a single number. Individual task drops range from 0.64% to 6.33% relative on Qwen2-7B, and the Code Completion result for Llama3-8B is catastrophic. The "1.30%" figure obscures significant task-level variation. While the paper does provide per-task results in the tables (which is good), the abstract and conclusion overclaim "negligible decrease in performance."

- **Missing comparison with LazyLLM**: LazyLLM (Fu et al., 2024) is explicitly discussed in related work as a prior prefilling optimization method, but is absent from the experimental comparison. The paper states LazyLLM "defers a portion of computations to the decoding stage," but this should be validated empirically rather than asserted. LazyLLM is the most directly comparable method (prefilling-stage token pruning) and its absence weakens the comparative claims.

- **PyramidInfer comparison is methodologically fragile**: Two PyramidInfer variants are presented — the official implementation (PyramidInfer*) which is slower and OOMs on Qwen2, and the authors' reimplementation using FlashAttention with 20% attention weight recomputation. The authors choose PyramidInfer's hyperparameters from its official settings rather than tuning them for Llama3/Qwen2, while FTP's hyperparameters are tuned per model. This asymmetry could make PyramidInfer appear worse than it is. The paper does not make this limitation explicit.

### Minor:

- **Attention recalculation overhead not analyzed at extreme context lengths**: FlashAttention doesn't output attention weights, so FTP must recompute them. The overhead is claimed to be 1–3% of TTFT on 5k–15k token sequences, but this recomputation has quadratic complexity. It is unclear whether the overhead remains negligible at the extreme context lengths (128k, 200k) that the paper's introduction highlights as a key motivation.

- **No ablation of the "last-N queries" approximation**: The importance scores rely on attention from only the last N=50 queries rather than all queries, justified by one citation (SnapKV). No ablation compares full-query vs. last-N importance, and whether this proxy holds across all task types (especially those requiring attention to earlier context) is not tested.

- **Hyperparameters P=100, N=50, and 𝓕=10 are set without systematic justification**: These are different across models (𝓕=10 for both, but η=0.90 for Llama3 vs. 0.95 for Qwen2) without a clear rationale. No sensitivity analysis is provided in the main text.

### Trivial:

- **No per-dataset breakdown in main tables**: Results are averaged per "task" (which aggregates multiple datasets), making it harder to identify outliers. However, the tables do show per-task results, so the overall trend is visible.

## Nice-to-Haves

- Demonstrate FTP at very long contexts (32k, 128k) and on Needle-in-a-Haystack / RULER benchmarks, where the cost of missing rare but critical tokens is highest.
- Combine FTP with KV cache compression to show end-to-end total inference latency, not just TTFT.
- Analyze error accumulation across layers for pruned tokens — does divergence between pruned and baseline hidden states grow linearly or exponentially with depth?
- Provide an ablation of full-query vs. last-N-query importance scoring to justify the approximation.

## Removed Points

These points are flagged to be removed; treat them with caution:

- **"LazyLLM defers computation to decoding" makes comparison unnecessary**: The harsh reviewer noted LazyLLM "defers a portion of computations to the decoding stage" as a reason not to compare. However, this is exactly the kind of distinction that should be demonstrated empirically, and LazyLLM is listed in related work, so the absence still weakens the paper. (Kept in weaknesses above as a missing comparison.)

- **"Metric aggregation is fundamentally incoherent"**: The harsh reviewer claimed aggregating heterogeneous metrics (F1, Rouge-L, etc.) has "no meaningful probabilistic or decision-theoretic interpretation." This is standard practice in LongBench evaluation and many NLP benchmarks. While the phrasing in the abstract is misleading (claiming 1.30% drop), the aggregation methodology itself is the established convention for this benchmark. The real weakness is the misleading headline claim, not the aggregation per se. (Moved to "misleading claim" above.)

- **"Missing comparisons with MInference, GemFilter, FlexPrefill"**: The human finder raised these. However, FlexPrefill targets sparse attention patterns rather than FFN pruning (different scope), and the other methods are not necessarily released or directly comparable on the same prefilling acceleration axis. The most critical missing baseline is LazyLLM (already noted above). Demanding an exhaustive set of recent baselines when the paper's scope is clear is unreasonable.

- **"Batched inference implications"**: The paper does not discuss how per-sample dynamic pruning affects batching. This is a valid deployment concern but is beyond the paper's stated scope (demonstrating the technique works), and is a nice-to-have rather than a core weakness.

- **"Compute end-to-end total inference time"**: While useful, TTFT is the paper's explicit optimization target, and measuring TTFT specifically is standard for prefilling optimization papers.

- **"Validate that attention importance correlates with FFN importance"**: While an interesting experiment, the core finding of the random ablation (Table 3) already shows that FTP's attention-based selection dramatically outperforms random selection, which is the relevant practical question. Whether attention is the *optimal* proxy for FFN importance is an interesting research direction but not a required experiment.

## Novel Insights

The key insight — that FFN (not attention) is the dominant cost during prefilling in the FlashAttention era, and that token pruning can be applied *between* attention and FFN via the residual connection rather than removing tokens entirely from the computation — is genuinely novel and practical. Prior token pruning methods remove tokens from the entire layer (attention + FFN + KV cache), losing their representations. FTP's observation that you can keep the attention-computed hidden states while skipping only the FFN computation is a clean architectural insight. However, the Code Completion results suggest the residual-connection preservation is insufficient for tasks where FFN contributions matter for most tokens, indicating an important boundary condition on when this method is applicable.

## Suggestions

1. **Acknowledge and discuss the Code Completion failure**: Add analysis of why FTP underperforms on structured tasks (code) and recommend contexts where FTP should be applied cautiously or with higher η values.

2. **Add LazyLLM as a baseline**: Even a partial comparison (e.g., on overlapping models) would substantially strengthen the comparative claims.

3. **Report per-dataset results** or at minimum acknowledge that the "1.30% performance drop" in the abstract is an average that hides significant task-level variation, including a ~35% relative drop on Code Completion for Llama3-8B.

4. **Analyze attention recalculation overhead at longer contexts** (32k+) to verify the "negligible" claim at the context lengths the paper claims to target.

## Score and Decision

**Calibration references**: 
- FlexPrefill (Oral, 8/8/8): Novel prefilling method with comprehensive evaluation on longer benchmarks, clear mechanism novelty.
- LazyLLM (Reject, 6/5/6/3): Pre-filling token pruning, training-free, but with concerns about deferred computation and limited novelty.
- D₂O (Poster, 6/6/6/5/6): KV cache compression, moderate novelty, solid experiments.
- HASA (Reject, 5/5/5): Prefilling acceleration via sparse attention, limited novelty.
- DynamicKV (Reject, 5/3/5/6/3): KV cache method, serious methodology issues.

This paper sits between LazyLLM (reject) and D₂O (poster). The core idea is sound and practically valuable — the FFN-focused pruning with residual preservation is a genuine contribution. The experiments show real speedups. However, the paper has significant weaknesses: a catastrophic failure on Code Completion that goes unacknowledged, a misleading "1.30% drop" abstract claim, a missing LazyLLM comparison, and a fragile PyramidInfer comparison. These are real problems that undermine the paper's claims of "negligible" degradation and comparative superiority. The method's novelty, while interesting, is an incremental architectural insight rather than a fundamentally new paradigm. Compared to FlexPrefill (which earned 8s for truly flexible adaptive sparsity with strong experiments), this paper's scope and rigor are narrower.

Score: **5.5** — A solid idea with useful practical contributions, but significant empirical gaps and overclaimed results that need addressing.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>