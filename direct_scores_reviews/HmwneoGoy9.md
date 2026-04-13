## Summary

SeerAttention proposes a learnable sparse attention mechanism that augments standard attention with a lightweight gating module (AttnGate) to adaptively select significant blocks in attention maps. The key enabler is a customized FlashAttention kernel that extracts block-level max-pooled attention maps during training without materializing the full O(n²) matrix, circumventing the memory bottleneck that has limited prior learned sparsity approaches. The method is evaluated in post-training calibration and long-context fine-tuning settings (with YaRN), demonstrating superior accuracy-efficiency tradeoffs over MoA and MInference on perplexity and LongBench.

---

## Strengths

- **Custom FlashAttention training kernel is a genuine engineering contribution.** The technique of storing row-max values (r_ij) during FlashAttention's online pass and rescaling them post-iteration (Equation 2) to recover block-level attention statistics avoids the quadratic memory cost of naïve attention while providing supervision for the gate. Figure 8 confirms near-identical memory usage to FlashAttention-2, enabling training at 64k+ sequence lengths that were previously infeasible for learned sparse attention.

- **Learned sparsity outperforms handcrafted heuristics across most settings.** Table 1 and Table 2 show SeerAttention at equal or higher sparsity achieving lower perplexity and higher LongBench scores than both MoA and MInference in nearly every configuration up to 64k context, demonstrating that the learned gate captures head-specific, input-dependent sparsity patterns that static patterns miss.

- **Fine-tuning integration with YaRN delivers compelling results.** Table 3 shows that YaRN+SeerAttention at 50% sparsity achieves perplexity of 8.81/2.47 vs YaRN baseline 8.79/2.46 (PG19/Proof-pile) — effectively lossless. Even at 90% sparsity, perplexity is 9.16/2.60, a ≤5% relative increase. Figure 1a shows loss curves at both sparsity levels tracking the dense baseline through 400 training steps, indicating stable joint optimization.

- **End-to-end TTFT dominates both competitors.** Table 4 shows SeerAttention achieving 13.37s TTFT at 128k vs MInference's 14.38s despite similar sparsity (0.95), and MoA running out of memory entirely at 128k.

- **Ablations are targeted and informative.** The RoPE ablation (Figure 9) provides convincing evidence: without the re-scaled RoPE in AttnGate, perplexity degrades catastrophically beyond the training context length. The pooling ablation (Figure 10) over 49 combinations identifies a principled best configuration (Qavg, Kmaxmin).

- **Flexibility via a single checkpoint.** Because the gate is trained to predict a distribution and top-k is applied at inference time, users can adjust sparsity ratio post-hoc without retraining — a practical advantage over MoA's per-sparsity search and MInference's fixed patterns.

---

## Weaknesses

- **Figure 1b presents a cross-dataset comparison that is visually misleading.** The orange dashed line labeled "YaRN Baseline" is evaluated on PG19 (perplexity ≈ 10), while the red solid line "YaRN w/ SeerAttention" is evaluated on Proof-pile (perplexity ≈ 3). These are different datasets with inherently different absolute perplexities. Table 3 confirms this: PG19 baseline is 8.79 and Proof-pile baseline is 2.46. The figure should compare both methods on the same dataset; as presented, the apparent dramatic reduction in perplexity is an artifact of the dataset switch, not a model improvement.

- **The 5.67× speedup figure in the abstract and Figure 1c refers to kernel-level computation only, not end-to-end inference.** Table 4 (TTFT) shows that at 128k the end-to-end speedup is 35.54 / 13.37 ≈ 2.66×, and at 32k it is approximately 4.63 / 3.60 ≈ 1.29×. The abstract should clearly label the 5.67× figure as kernel-level and present the corresponding end-to-end figure alongside it to avoid overclaiming. This distinction matters because other LLM components (MLP, normalization, etc.) dilute the attention speedup at the system level.

- **Block size B=64 is a central hyperparameter that is never ablated.** This single choice controls the coarseness of the sparsity approximation and has a first-order effect on the accuracy–efficiency tradeoff. The paper fixes B=64 throughout without justification. Since block size also determines hardware tiling granularity, an ablation over B ∈ {32, 64, 128} is necessary to understand the design space and whether the current choice is optimal.

- **Evaluation is limited to perplexity and LongBench aggregates; Needle-in-a-Haystack (NIAH) and retrieval-intensive tasks are absent.** High attention sparsity could cause disproportionate failure on tasks requiring precise long-range retrieval (e.g., NIAH, multi-hop QA) even when aggregate perplexity changes are small. Without these evaluations, the claim of "minimal loss" is incomplete — perplexity is insensitive to local retrieval failures that matter in practice.

- **No downstream task evaluation for the fine-tuned model.** Section 5.2 evaluates the YaRN+SeerAttention model only on perplexity (Table 3). Given that fine-tuning is presented as a key contribution, the absence of any LongBench or instruction-following results for the fine-tuned model leaves a significant gap in validating that the fine-tuned model retains general capability.

- **Post-training perplexity degradation at 128k/90% sparsity is substantial, not "minimal."** Table 1 shows perplexity rising from 10.03 (dense) to 13.20 at 90% sparsity and 128k context — a 31.6% relative increase. The "minimal perplexity loss" claim in the abstract is accurate for the fine-tuning scenario (which is explicitly scoped to 32k), but the paper should be clearer that post-training at very high sparsity and very long context is a different and weaker regime.

- **The linear projection layer in AttnGate is central to the architecture but not ablated.** It is unclear whether the linear layer is necessary, or whether a direct pooled-Q × pooled-K dot product (analogous to standard attention on pooled tokens) would perform similarly. Given that the gate design is one of the paper's core contributions, this is a meaningful gap.

---

## Nice-to-Haves

- **Per-head variable sparsity.** The paper acknowledges (Table 1 discussion) that MInference's per-head sparsity is likely why it outperforms at 128k in post-training. Extending SeerAttention to learn per-head sparsity budgets would be a natural enhancement and could close this gap.

- **Ranking-based training objective.** The gate is trained with MSE loss against the row-normalized max-pooled attention map. Since inference uses top-k block selection, a ranking or top-k recall loss would more directly optimize the downstream objective. This is not a fatal flaw given the strong empirical results, but it is worth investigating.

- **Calibration cost reporting.** The paper states post-training calibration completes in "hours" on 4 A100 GPUs with 500 steps. A clear breakdown of FLOPs or GPU-hours relative to inference savings would help practitioners assess the trade-off for new models.

- **CUDA kernel implementation.** The Triton kernel is compared against a CUDA FlashAttention-2 baseline, so the speedup numbers reflect both algorithmic and implementation differences. The authors acknowledge this and note CUDA as future work; flagging this more explicitly would strengthen credibility of the efficiency claims.

- **Gate entropy / confidence analysis.** Measuring the entropy of AttnGate outputs across different heads and context lengths would reveal whether some heads are systematically uncertain (high entropy), which would indicate instability in block selection at high sparsity.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Causal masking not explained" (Harsh Critic).** The paper's block-sparse kernel explicitly follows FlashAttention-2's dataflow, which already correctly handles causal masking. There is no evidence the causal structure is mishandled.

- **Statistical rigor / confidence intervals (Harsh Critic).** For large-scale LLM benchmarks (LongBench, perplexity), single-run evaluation is the established norm. Demanding confidence intervals is not standard in this community and would impose a non-standard rigor requirement. Removed per reviewer calibration rules.

- **Missing comparisons with H2O, SnapKV, StreamingLLM (Harsh Critic).** These are KV-cache eviction/compression methods, which operate on a different problem (reducing KV cache memory during decoding) from this paper's focus (sparse prefill computation). The comparison set of MoA and MInference, which are direct sparse prefill competitors, is appropriate. Additionally, per review rules, missing related work comparisons should not be flagged when external sources cannot be confirmed.

- **MoA TTFT at 8k slower than FlashAttn-2 (Harsh Critic).** The critic notes MoA is slower than FlashAttn-2 at 8k (1.29s vs 0.90s) and suggests this needs explaining. However, this reflects a genuine weakness of MoA, not of SeerAttention — the comparison is intentionally asymmetric in favor of the baseline. Including this makes the paper's method look *stronger*, not weaker, and per rules, such comparisons should be removed as "unfair comparisons beneficial to the baseline."

- **The K-outlier hypothesis being speculative (Harsh Critic).** The paper presents this as a possible explanation ("may relate to"), not a claim. This is appropriate scientific hedging and is not a weakness.

- **"Well-structured and clearly written" (Positive Reviewer strength).** This applies to any competently written paper and provides no differentiation.

- **"Topic is important / industry push toward 128k+" (Positive Reviewer strength).** Generic significance claim that applies to any long-context LLM paper.

---

## Novel Insights

The most genuinely novel insight in this paper is the decoupling of *supervision extraction* from *attention computation*: by instrumenting FlashAttention's tiled pass to store intermediate row-max values and rescale them post-iteration, the paper shows that block-level attention statistics can be recovered at near-zero overhead without a separate O(n²) forward pass. This is not merely an engineering trick — it enables a training paradigm where a learned gating module receives dense attention supervision at 64k+ contexts, something that was previously impractical. A secondary insight is that applying a block-rescaled RoPE (θ′=θ/B) to the pooled Q/K positions inside AttnGate is both principled and practically critical: Figure 9 shows it is the difference between smooth extrapolation to 128k from 8k training data versus catastrophic perplexity collapse, suggesting that positional encoding fidelity at the block level is a non-trivial concern for any block-sparse attention scheme that needs to generalize across context lengths.

---

## Suggestions

1. **Fix Figure 1b** to show both YaRN Baseline and YaRN w/ SeerAttention on the *same* dataset (e.g., PG19 or Proof-pile). The current figure is misleading and will be flagged by any reviewer who checks the dataset labels.

2. **Add a block size ablation** (B ∈ {32, 64, 128}) in Section 6. Given that B controls both approximation coarseness and tiling granularity, this is essential for understanding the paper's design space.

3. **Add NIAH evaluation** for the post-training models, ideally at both 32k and 128k context lengths at several sparsity ratios. This would provide evidence that the gate does not miss critical positional targets that perplexity cannot capture.

4. **Add LongBench results for the YaRN+SeerAttention fine-tuned model** in Section 5.2 to demonstrate that fine-tuning with sparsity does not degrade instruction-following on real tasks.

5. **Clarify end-to-end vs kernel-level speedup in the abstract** — present both figures, or at minimum label the 5.67× as kernel-level and cite the corresponding end-to-end TTFT numbers.

6. **Ablate the linear projection layer** in AttnGate by testing a no-linear-layer baseline (direct pooled-Q × pooled-K matmul). If the linear layer is necessary, this experiment will make a stronger case for the current architecture.

---

**Axis Evaluations:**

- *Novelty:* Moderate-to-strong. Learning block sparsity is not new, but the memory-efficient customized training kernel and the RoPE-in-AttnGate design are genuinely novel mechanisms.
- *Technical soundness:* Generally sound. The core kernel design is well-explained and the RoPE ablation is rigorous. Some design choices (block size, linear layer) are unvalidated.
- *Empirical support:* Adequate for post-training claims; weaker for fine-tuning claims (perplexity only, no downstream tasks). Absent for NIAH-type retrieval evaluation.
- *Significance:* Meaningful for the long-context LLM inference community. The fine-tuning result (90% sparsity, ~5% perplexity increase) is practically useful.
- *Clarity:* Good overall structure with targeted ablations; Figure 1b and the abstract speedup presentation introduce avoidable ambiguity.

MY FINAL SCORE: <pineapple>6.8</pineapple>