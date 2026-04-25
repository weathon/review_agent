Now let me run calibration searches to anchor my score.Now I have enough information to write my final consolidated review.

---

## Summary

FTP (FFN Token Pruning) targets the long-context LLM prefilling bottleneck by pruning less-important tokens *before* the FFN module in each transformer layer, using attention scores to determine which tokens to skip and how many. Unlike prior methods that entirely remove pruned tokens, FTP preserves pruned tokens' hidden states via the residual connection (logically setting FFN output to zero), allowing information to flow through without recomputing FFN. The paper demonstrates TTFT speedups of 1.20–1.45× across four model sizes on LongBench.

---

## Strengths

- **Novel, well-motivated optimization target**: Prior token pruning methods focus on compressing the KV cache for the attention module. FTP is the first to specifically target FFN computation during prefilling, motivated by compelling profiling data (Figure 3: FFN accounts for 62.4% and 61.3% of per-layer walltime for Llama3-8B and Qwen2-7B respectively). This is a concrete, specific observation that differentiates the contribution.

- **Technically clean residual-connection trick**: Setting pruned tokens' FFN output to zero is equivalent to bypassing FFN updates via the residual connection, requiring no extra bookkeeping or data structures. This design is parsimonious and well-reasoned (Section 3.2, Algorithm 1).

- **Dynamic, layer-adaptive pruning ratio** (Equation 3): Rather than a fixed token count, FTP selects the minimal *k* such that top-k tokens cover fraction η of total importance mass, responding to empirically observed inter-layer variability in attention concentration (Figure 5, Observation 3). This is a sensible design choice over fixed-count methods.

- **Strong ablation isolating the selection criterion**: Table 3 compares attention-based vs. random pruning at identical pruning rates across all tasks on two models. Random pruning at the same rate degrades accuracy catastrophically (e.g., Single-Document QA: 36.06→11.14 on Llama3-8B), demonstrating the selection strategy—not just the compression ratio—drives the result.

- **Practical compatibility**: Flash attention is re-used by recalculating only the necessary attention weights, adding only 7–15ms overhead (0.8–3% of TTFT). Training-free.

- **Comprehensive benchmarking**: 16 datasets, 6 task types, 4 model sizes (8B–72B parameters), enabling reasonable claims of generalizability.

---

## Weaknesses

### Fatal

None.

### Major

- **Unexplained catastrophic failure on Llama3-8B Code Completion**: Table 1 shows FTP scores 35.91 vs. a baseline of 55.17 on Llama3-8B Code Completion—a 19.26-point absolute drop and 35% relative drop. PyramidInfer achieves 55.24 on the same setting, making FTP dramatically inferior. This is not an edge case: Code Completion is one of six task categories evaluated on one of two primary models. The abstract claims "negligible decrease in performance," and the conclusion states "delivering significant acceleration while maintaining performance"—both directly contradicted by this result. The paper provides no explanation, no analysis of what causes the failure, no discussion in limitations, and no exploration of whether hyperparameter adjustments (η, F) would fix it. For Qwen2-7B, Code Completion is fine (56.74 vs. 58.43), suggesting a model-specific or context-length-specific failure mode that the authors have not investigated.

- **Missing comparison with LazyLLM**: Section 2.1 explicitly identifies LazyLLM (Fu et al., 2024) as a method that "drops tokens from the prefilling stage" and "proposes an aux cache to avoid redundant computing"—by the paper's own description the most directly comparable prefilling acceleration method. Yet LazyLLM appears in neither Table 1 nor Table 2. The paper's central comparative claim—superior speedup-accuracy tradeoff for prefilling acceleration—cannot be substantiated without this comparison. Comparing only against LLMLingua2 (a prompt compression method, not in-layer token pruning) and PyramidInfer (which primarily targets KV cache and decoding) leaves the central niche inadequately benchmarked.

### Minor

- **Non-trivial accuracy drop on Qwen1.5-32B Synthetic task**: Table 2 shows a 6.42-point absolute drop (52.67→46.25) on the Synthetic task for Qwen1.5-32B—about 12% relative. The paper characterizes these results as "subtle impact on the accuracy score," which is inconsistent with the data. This is not as severe as the Code Completion failure but warrants acknowledgment in the paper.

- **Hyperparameter selection procedure undisclosed**: The values η=0.90/0.95 and F=10 are stated as per-model constants but no selection procedure is described. Without knowing whether these were selected to maximize performance on LongBench test data, it is difficult to assess whether the results could be inflated. Ablations are promised for the appendix but the main paper does not address sensitivity.

- **End-to-end latency not reported alongside TTFT**: Figure 2 shows that prefilling accounts for only 23.71% of inference time on Code Completion (RepoBench-P). A 1.22× TTFT speedup translates to approximately 1.05× end-to-end speedup for generation-heavy tasks. While the abstract correctly qualifies the speedup as "in the prefilling stage," readers cannot easily assess practical benefit without end-to-end numbers, especially for applications where generation dominates.

### Trivial

None.

---

## Nice-to-Haves

- **Layer-wise analysis of retained token sets**: Show the actual fraction of tokens pruned at each depth across representative samples to verify the η-based threshold is meaningfully dynamic rather than concentrating pruning in a narrow depth range.
- **Cascade effect analysis**: Tokens pruned in layer *k* have stale hidden states affecting layers *k+1* through *k+n*. An analysis of whether iterative pruning degrades the "important token" signal in deeper layers would strengthen the theoretical understanding.
- **Investigation of the Llama3-8B Code Completion failure**: Understanding why η=0.90 and shorter sequences (RepoBench-P avg 4206 tokens) lead to catastrophic failure would be valuable for practitioners and for setting safe hyperparameter ranges.

---

## Removed Points

*These points are flagged to be removed. Treat them with caution.*

- **"Speedup is misleading" (harsh critic Issue 3)**: The abstract explicitly qualifies: "achieves a speedup of 1.24× *in the prefilling stage*." Figure 2 is included precisely to show task-specific TTFT proportions. The framing is accurate; the criticism overstates the issue. Demoted to Nice-to-Have (reporting end-to-end latency).

- **"The 1.30% performance drop is not clearly traceable"**: Checking Table 1, Qwen2-7B-Instruct average absolute score drop across six tasks is approximately 1.28 points on the 0–100 scale, or 1.28%—consistent with the abstract's "1.30%" claim. The calculation is valid; this criticism is incorrect.

- **"Comparison with PyramidInfer is confounded"**: Section 4.3 explicitly explains both implementations (PyramidInfer* with native attention vs. re-implemented with flash attention for fairness). Both the baseline and FTP use flash attention. The comparison is internally consistent and the paper is transparent about this choice.

- **"N=50 is not ablated across long sequences"**: The use of N=50 (observation window) is explicitly cited as adopted from SnapKV (Li et al., 2024), an empirically validated prior work. Not ablating borrowed components from established prior work is reasonable.

- **"95%/60% analysis uses unmodified models"**: This is a standard methodology for establishing the motivation for pruning. The analysis of cascade effects would strengthen the paper but is outside its stated scope.

---

## Novel Insights

The genuinely novel contribution is the shift from attention-centric token pruning to FFN-targeted token pruning during prefilling, enabled by the residual-connection zero-output trick. This sidesteps the flash-attention limitation that prevents explicit token eviction in attention (which requires recomputing the score matrix anyway) while targeting the module that actually consumes the majority of walltime (>60%). The observation that attention sparsity can be used to predict *FFN* importance—not just attention importance—is a non-obvious connection that enables clean, zero-overhead pruning of the most expensive per-token operation. This is a real conceptual advance over prior methods.

---

## Calibration and Score

**Anchor papers retrieved and compared:**

| Paper | Avg Human Score | Decision | Comparison |
|---|---|---|---|
| "FTP: Fine-grained Token-wise Pruner" (gcEhF4nuYI) | 3.0 | Withdrawn/Reject | Much weaker: limited novelty, training confusion, minimal ablations. FTP (this paper) is clearly better. |
| DynamicKV (uHkfU4TaPh) | 4.4 | Reject | Similar space (adaptive per-layer token compression for long-context LLMs); rejected for missing practical efficiency analysis and FA compatibility questions. FTP is similarly positioned with a different set of gaps. |
| KV Prediction (QlvL6eEOC6) | 4.5 | Withdrawn | TTFT-focused, limited model coverage, similar scope. FTP has broader model coverage but deeper single-task failures. |
| GemFilter (9iN8p1Xwtg) | 5.25 | Reject | Training-free token reduction for LLM inference; rejected for missing key baselines. FTP's situation is analogous—missing LazyLLM, plus an unaddressed failure case. |
| Recycled Attention (8qYuxV4lRu) | 5.4 | Reject | Long-context inference acceleration with missing baselines (SnapKV, PyramidInfer absent). Similar positioning—novel method, incomplete comparative evidence. |
| OmniKV (ulCAPXYXfa) | 6.0 | Accept (Poster) | Training-free long-context inference, no performance loss claimed—accepted because it credibly delivers performance without accuracy trade-offs. FTP falls below this bar due to the unexplained Code Completion failure and missing LazyLLM comparison. |

**Scoring rationale**: FTP sits between GemFilter/Recycled Attention (5.25–5.40, rejected for missing baselines) and DynamicKV (4.40, rejected for missing practical evaluation). Its novel FFN-targeting angle and strong Qwen2-7B results elevate it slightly above DynamicKV, but the unexplained 35% relative drop on Llama3-8B Code Completion—directly contradicting the core accuracy claim—and the missing LazyLLM comparison keep it below the acceptance threshold of OmniKV (6.0). The paper's quality cluster aligns with the 4.5 range.

**Overall evaluation**: The paper makes a genuine and interesting contribution—FFN-targeted token pruning is a novel and well-motivated angle—and the Qwen2-7B results are solid. However, the Llama3-8B Code Completion failure is a substantive unaddressed concern that undermines the paper's universal accuracy claim, and the absence of a LazyLLM comparison leaves the core competitive claim unsupported. The paper as submitted is not ready for acceptance.

---

**Originality**: Good — the FFN-targeting angle is non-obvious and distinct from prior work.  
**Importance of research question**: High — TTFT is a real bottleneck for long-context deployment.  
**Claim support**: Moderate — Qwen2-7B claims are well-supported; Llama3-8B has a major unaddressed failure.  
**Soundness of experiments**: Fair — comprehensive in coverage, but missing a key baseline and with an unexplained data point.  
**Clarity of writing**: Good — method, algorithm, and ablations are clearly presented.  
**Value to community**: Moderate — the FFN-pruning insight is useful, but the current experimental record is incomplete.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>