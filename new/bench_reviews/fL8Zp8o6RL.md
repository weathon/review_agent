Now let me look at the calibration papers mentioned and search for more.Now I have enough to write the consolidated review.

---

## Summary

FTP (FFN Token Pruning) is a training-free inference-time method that prunes non-critical tokens selectively before the FFN sublayer during prefilling. Token importance is derived from attention scores in each layer; pruned tokens bypass the FFN and pass through unchanged via the residual path. The method is validated on LongBench across four models (7B–72B), reporting TTFT speedups of 1.2–1.4× with reportedly small accuracy loss. The core contribution is shifting the optimization target from the heavily studied attention/KV-cache pathway to the FFN, which profiling shows dominates per-layer prefilling time.

---

## Claims and Support

**Claim 1: FFN dominates prefilling wall time.**
**Well-supported.** Figure 3 shows direct profiling: FFN occupies 62.4% and 61.3% on Llama3-8B and Qwen2-7B respectively. The observation is valid for the tested models and flash-attention deployment setting, though the text occasionally implies universality.

**Claim 2: FTP reduces TTFT while retaining most information via the residual path.**
**Partially supported.** The speedup side is empirically demonstrated. The "retaining most information" argument via residual is a plausible heuristic but not mechanistically established. No ablation isolates the residual path's contribution (e.g., comparing against dropping tokens entirely or zeroing both FFN and residual).

**Claim 3: Attention-based token importance is an effective criterion for FFN pruning.**
**Partially supported.** Table 3 shows attention >> random pruning, which is meaningful but a weak baseline. No comparison against other non-random heuristics (recency, norm-based, aggregated differently) is provided.

**Claim 4: Cumulative attention-mass thresholding (η) is better than a fixed-k pruning rule.**
**Unsupported.** Figure 6 illustrates attention concentration but provides no direct comparison between the η-based rule and a fixed pruning ratio. This is an asserted design choice without ablation.

**Claim 5: Preserving the first F layers is important.**
**Weakly supported in the main text.** Section 3.2.2 cites "Section 4.6" and Appendix 6.1, but the main ablation section (4.6) only compares attention vs. random and does not show results across varying F values. The F=10 choice is not validated in the main paper.

**Claim 6: FTP outperforms prior prefilling-acceleration methods.**
**Partially supported.** FTP generally outperforms reimplemented PyramidInfer, but the comparison relies on a custom reimplementation whose reproduction fidelity is not fully documented. LLMLingua2 is a prompt compression method not a direct architectural counterpart. LazyLLM (Fu et al., 2024), which the paper explicitly acknowledges as a directly competing prefilling-stage method in related work, is absent from the experimental comparison.

**Claim 7: Performance drop is negligible.**
**Overstated.** The abstract claims "only a negligible decrease in performance." However, Table 1 shows Llama3-8B Code Completion drops from 55.17 → 35.91 (~35% relative decline). Table 2 shows Qwen1.5-32B Synthetic drops from 52.67 → 46.25 (~12% relative). The 1.30% figure cited in the abstract refers specifically to Qwen2-7B averaged across all tasks, and does not represent the range of outcomes.

**Claim 8: FTP generalizes across models.**
**Partially supported.** Results on four models covering 7B–72B are presented, which is a reasonable range for the ICLR setting.

---

## Strengths

- **Novel optimization target.** Unlike virtually all prior prefilling-acceleration work (LazyLLM, GemFilter, SnapKV, H2O), which reduces attention computation or KV cache size, FTP targets the FFN sublayer. Backed by profiling showing FFN >60% of per-layer wall time, this is a genuinely underexplored angle with strong empirical motivation.

- **Clean, training-free design.** The method is conceptually simple: prune tokens before FFN, let residual connections preserve pruned token states. Algorithm 1 provides sufficient pseudocode for replication. No finetuning or auxiliary models needed.

- **Attention vs. random ablation (Table 3) quantifies both selection quality and overhead.** The comparison simultaneously shows the attention criterion is essential (random causes catastrophic degradation) and that recomputing attention scores adds only 7–15ms (1–3% of TTFT), directly addressing the overhead concern.

- **Empirical breadth across four models and six task categories.** Presenting results on 7B, 7B, 32B, and 72B models, with explicit handling of memory constraints (OOM on PyramidInfer official), gives a credible foundation for generalization claims.

---

## Weaknesses

### Fatal
*None that would invalidate the core empirical contribution outright, but the headline framing materially misrepresents the results.*

### Major

- **The "negligible decrease" headline claim is directly contradicted by reported results.** Code Completion for Llama3-8B drops from 55.17 → 35.91 (−19.3 points, ~35% relative). The Synthetic task on Qwen1.5-32B drops from 52.67 → 46.25 (−6.42 points). These are not negligible. The paper's own abstract and conclusion use the word "negligible" globally without qualification, misleading readers about the method's reliability. This framing needs correction; the paper should instead describe FTP as showing *modest average degradation on most tasks with notable failures on code-related tasks*, and analyze why code completion is particularly vulnerable.

- **LazyLLM is explicitly described in the related work as a directly competing prefilling-stage method but is absent from empirical comparison.** Section 2.1 states: *"LazyLLM (Fu et al., 2024) also drops tokens from the prefilling stage... However, these methods either yield subtle speedup during prefilling or defer some computation to the decoding stage."* This critique of LazyLLM is the paper's central competitive claim. Without a direct experimental comparison, the claimed superiority over this closest competitor is unverified. This is the most salient baseline gap.

- **Critical design choices are asserted without ablation.** The paper makes at least three specific design claims (1) cumulative-mass threshold η is better than fixed-k, (2) protecting the first F=10 layers is important, (3) residual preservation is the key to accuracy retention. Only the third is tangentially addressed by the random comparison. No ablation compares η-thresholding vs. a fixed pruning ratio, and no ablation in the main text varies F. These are listed as key contributions of the method design (Section 3.2.1–3.2.2) and their empirical justification is deferred entirely to the appendix or absent.

### Minor

- **PyramidInfer reimplementation fairness is not fully established.** The paper appropriately discloses that it reimplements PyramidInfer with flash attention and recalculates 20% attention weights, but does not state whether the same hyperparameter search effort was applied to both methods, nor verifies behavioral equivalence to the official implementation beyond mentioning the 20% attention weight threshold. Since much of the comparative claim rests on this reimplementation, a brief validation step is warranted.

- **Hyperparameters (η, F, P, N) are manually tuned per model with no sensitivity analysis.** Different η values (0.90 vs. 0.93 vs. 0.95) are set for different models without explanation of how they were chosen. A practitioner facing a new architecture has no guidance. This limits plug-and-play applicability.

- **End-to-end latency impact is not discussed.** FTP leaves the full KV cache intact (all tokens are stored). Total query latency depends on both TTFT and decode time. For tasks with long generation outputs (e.g., code completion, where decode time is large), the TTFT gain may not meaningfully improve user-perceived latency. Table 3's TTFT numbers show, e.g., that code completion has the smallest baseline TTFT (449ms for Llama3) while having the largest accuracy failure, suggesting the method's deployment value is limited precisely where generation is long.

### Trivial

- The profiling results in Section 3.1 are presented as universal facts; they should be qualified as specific to flash-attention deployment on A100 GPUs, since the balance may differ under memory-bound regimes or different hardware.

---

## Nice-to-Haves

- A GPU kernel-level profiling of the dynamic token gather/scatter overhead. Dynamic indexing breaks tensor core coalescence and the claimed wall-clock speedup rests on the assumption that FLOP reduction translates linearly to latency.
- A brief failure-case study for code completion on Llama3 to understand whether FFN pruning disrupts syntactic structure or multi-step code reasoning.
- Memory profiling of attention score recomputation at 32k+ contexts, where the O(L²) recomputation cost could approach or dominate the FFN savings at maximum sequence lengths.
- Discussion of compatibility with GQA/MQA architectures (used in most 70B+ models) where importance scoring on shared KV heads may introduce head-specific bias.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Reviewer: "The paper's main quality evaluation is too coarse."** Partially valid but overstated. LongBench's protocol of averaging heterogeneous metrics (F1, Rouge-L, etc.) into a single score is the community-standard evaluation methodology for long-context LLMs — the paper follows the official LongBench pipeline faithfully. The valid version of this criticism (code completion failure) is retained above.

- **Harsh Reviewer: "Recalculating attention adds only negligible cost (unsupported)."** Table 3 directly quantifies this: 7–15ms, 1–3% of TTFT. This is empirically established and the criticism is incorrect.

- **Spark Reviewer: "Claimed 1.2–1.4x speedup is not convincing without kernel-level profiling of gather/scatter overhead."** Moved to Nice-to-Haves. The claimed speedups are measured wall-clock results on real hardware (A100), not theoretical FLOP calculations. While kernel-level profiling would strengthen the paper, the empirical timing measurements in Tables 1–3 are real evidence.

- **Spark Reviewer: Confidence intervals and multiple-run statistical tests.** Single-run evaluation over LongBench's 200 samples per dataset is the community norm. Not holding the paper to a non-standard.

- **Neutral/Spark Reviewer: Memory footprint of attention recomputation.** Moved to Nice-to-Haves. The time overhead is documented; memory profiling is useful but not standard in this setting.

- **Harsh Reviewer: "The residual-path argument requires layerwise representational analysis."** Excessive demand for mechanistic proof in an empirical systems paper. The ablation showing attention >> random partially supports this, and the accuracy results mostly support it (code completion aside). Moved to Nice-to-Have framing.

---

## Novel Insights

The most substantive insight across all reviewers — raised but not fully developed — is the *disaggregation of prefilling bottleneck*. Prior work has universally treated prefilling as an attention/KV-cache problem. FTP demonstrates empirically that, under flash attention (the de facto production standard), the FFN module dominates per-layer wall time at a ratio of ~2:1 over attention. This reframes the optimization problem: methods that only reduce attention computation are leaving >60% of the prefilling bottleneck untouched. If the code completion failure is resolved (either by task-specific operating points or a task-adaptive η), this insight could motivate a broader family of FFN-targeted inference optimizations. The observation that larger models appear more robust to FFN token pruning (Table 2 vs. Table 1) is also interesting but unexplained — it may reflect either parameter redundancy or the 32k context enabling a higher absolute number of retained tokens.

---

## Suggestions

1. **Reframe the accuracy claim honestly.** Replace "negligible decrease" throughout with "modest average degradation on most tasks, with task-specific variation." Explicitly address the code completion failure for Llama3 and discuss why the method is less suitable for that task category.
2. **Add LazyLLM as an experimental baseline** (Section 4.3). Since it is already described in related work as the most directly competing prefilling-stage method, its absence is conspicuous. If LazyLLM cannot achieve real speedup due to implementation constraints, document this honestly as a comparison note.
3. **Add main-text ablations for η-threshold vs. fixed-k and for varying F.** These are design claims, not supplementary curiosities. Even a 2×2 table on a single dataset would substantially strengthen Section 3.2.
4. **Provide a hyperparameter selection heuristic** or show sensitivity across the range (η = 0.85–0.99, F = 5–20) so users can configure FTP without a full grid search on each new model.

---

## Evaluation on Key Axes

- **Novelty:** Moderate-to-high. Targeting the FFN sublayer specifically is genuinely underexplored and well-motivated by profiling. The use of cumulative attention mass as a dynamic pruning criterion is a clean contribution.
- **Technical soundness:** Moderate. The method is well-defined and reproducible. However, several design choices are asserted without ablation, and the residual pathway argument is a plausible heuristic rather than an established principle.
- **Empirical support:** Mixed. Strong on 5 of 6 task categories, with a clear and significant failure on code completion for Llama3 (35.91 vs. 55.17 baseline) that directly contradicts the headline claim. The absence of a comparison with LazyLLM — the most similar prior method — weakens the competitive positioning.
- **Significance:** Moderate. A 1.2–1.4× TTFT speedup is real but modest. The FFN-targeting insight is valuable for the community even if the current implementation has limitations.
- **Clarity:** Good. Algorithm 1, Figure 4, and the performance profiling figures are well-presented. The motivation flows logically from profiling to design. The abstract's "negligible" framing is the main clarity failure.

---

## Score and Decision

**Calibration anchors:**

| Paper | Topic | Scores | Decision |
|---|---|---|---|
| LazyLLM (am5Z8dXoaV) | Dynamic token pruning for prefilling | 6, 5, 6, 3 (~5.0 avg) | Reject |
| GemFilter (9iN8p1Xwtg) | Early-layer token filtering for context reduction | 6, 5, 5, 5 (~5.25 avg) | Reject |
| UNComp (28oMPC5bcE) | Uncertainty-aware KV compression for prefilling | 6, 5, 6, 5 (~5.5 avg) | Reject |
| FlexPrefill (OfjIlbelrT) | Adaptive sparse attention for prefilling | 8, 8, 8, 8 | Accept (Oral) |

FTP is most directly comparable to GemFilter and LazyLLM. It improves on both in presentation and methodological clarity. The evaluation is broader (4 models, 6 tasks). However, the significant code completion failure directly contradicts the headline claim, the ablation suite is inadequate for the design choices claimed, and LazyLLM — explicitly identified in the related work as the nearest competitor — is not empirically compared. These issues collectively mirror the weaknesses that led to GemFilter and LazyLLM being rejected.

Relative to FlexPrefill (Oral, 8s): FlexPrefill has stronger theoretical grounding, multiple novel components with ablations, and a more careful treatment of its accuracy claims. FTP does not approach that bar.

**Final Score: 5.0 — Reject.** The FFN-targeting insight is worth publishing but the current submission overstates its results, has a significant unaddressed failure case (code completion), and lacks the ablations needed to validate its design claims.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>