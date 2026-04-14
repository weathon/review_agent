## Summary
SeerAttention proposes a learned block-sparse attention mechanism that trains a lightweight gating module (AttnGate) to dynamically identify and skip unimportant attention blocks in long-context LLMs. Its central technical contribution is a customized FlashAttention kernel that efficiently extracts block-level max-pooled attention maps as training targets without quadratic memory overhead. The method is evaluated in both post-training and fine-tuning settings, achieving up to 5.47× kernel-level speedup over FlashAttention-2 at 90% sparsity with minimal perplexity degradation.

---

## Strengths

- **Custom FlashAttention kernel for scalable training:** The modification to FlashAttention that stores and rescales local row-maxima to recover block-level max-pooled attention maps (Equations 1–2, Figure 3) is a technically clean and practically load-bearing contribution. It enables training at sequence lengths where naive implementations OOM at just 4k tokens (Figure 8), solving the core infrastructure bottleneck of learned sparse attention on modern LLMs.

- **Block-level RoPE design with demonstrated necessity:** The approach of reusing RoPE parameters with block-level position IDs (θ' = θ/B) for context-length extrapolation is directly validated by Figure 9, which shows dramatic perplexity degradation without this component across all tested sparsity levels — establishing the design choice as load-bearing rather than merely additive.

- **Single model, adjustable sparsity at inference:** Unlike MoA, which requires exhaustive offline search for each sparsity configuration, SeerAttention trains once and allows the Top-k ratio to be adjusted freely at inference. This is a significant practical advantage and a genuine differentiator from prior work.

- **Comprehensive pooling ablation with insightful finding:** Testing all 49 pooling combination for the gate inputs and identifying the best configuration (avg-Q + max+min-K, Figure 10) is thorough, and the connection to K-outlier behavior in LLM quantization provides an interesting interpretive insight that links to a separate line of work.

- **Strong fine-tuning results at extreme sparsity:** Table 3 shows that joint fine-tuning with YaRN yields PG19 perplexity 9.16 at 90% sparsity vs. a baseline of 8.79 — a 4.2% relative increase at 90% sparsity — which is meaningfully better than post-training (10.18 at 90%). This demonstrates that the learned gate benefits significantly from end-to-end optimization.

---

## Weaknesses

### Fatal
None. The core contributions are technically sound and the general direction is well-supported.

### Major

- **Figure 1b is dataset-mismatched and visually misleading:** Panel (b) directly compares "YaRN Baseline (PG19)" (perplexity ≈10) against "YaRN w/ SeerAttention (Proof-pile)" (perplexity ≈3). Proof-pile is mathematical text with inherently far lower perplexity than PG19 prose for LLaMA-family models. The visual implication — that SeerAttention achieves lower perplexity than the dense baseline — is entirely an artifact of this dataset mismatch. Table 3 correctly provides a same-dataset comparison, making Figure 1b redundant and misleading. It should be replaced with a same-dataset comparison or removed.

- **No downstream task evaluation for the fine-tuned model:** Table 3 reports perplexity for YaRN+SeerAttention at up to 90% sparsity, but no LongBench, RULER, or equivalent task-based evaluation is provided for the fine-tuned model. Since the fine-tuning regime is the paper's most compelling contribution (near-lossless accuracy at 90% sparsity), omitting task-level validation leaves the "near-lossless accuracy" claim validated only by perplexity — which is known to sometimes diverge from task performance in long-context settings.

- **Training–inference objective gap not analyzed:** AttnGate is trained with soft MSE against the max-pooled attention map, but used at inference with a hard Top-k decision. The paper never measures the recall rate of genuinely important blocks — i.e., what fraction of the top-K highest-attention blocks in the dense map are correctly identified by the gate's Top-k selection. This would directly validate that the MSE objective leads to correct sparsity prediction rather than merely low reconstruction error on soft scores.

### Minor

- **Block size B=64 is a fixed, unablated hyperparameter:** Block size directly governs sparsity granularity, hardware tile efficiency, and the coarseness of learned patterns. On different GPUs or at different sequence lengths, the optimal block size may differ substantially, and the sensitivity of the accuracy–speedup tradeoff to this choice is unknown.

- **128k performance gap vs. MInference is significant and inadequately addressed:** At 128k context, SeerAttention at 0.9 sparsity yields PG19 perplexity 13.20 vs. MInference's 10.89 (at comparable average sparsity). The paper acknowledges this is due to fixed per-head sparsity and defers to future work. While acknowledged, the gap is 21% relative perplexity increase over MInference at the same length and sparsity, which is a material limitation for the longest contexts.

- **TTFT comparisons may conflate engineering quality with algorithmic merit:** Table 4 shows MoA at 8k is *slower* than dense FlashAttention-2 (1.29s vs 0.90s), and MInference at 8k is 2.6× *slower* (2.33s vs 0.90s), despite both achieving meaningful attention sparsity. SeerAttention at 8k is *faster* than dense (0.78s). This suggests the baselines have suboptimal kernel implementations, and part of SeerAttention's observed TTFT advantage may reflect engineering quality rather than algorithmic merit. The paper should acknowledge this explicitly and ideally compare kernel-level speedup curves (Figure 6) as the primary algorithmic comparison, using TTFT as a secondary systems-level metric.

### Tiny

- **Unexplained super-dense anomaly in Table 2:** SeerAttention at 0.1 sparsity achieves 55.91 on the 0–4k LongBench split, higher than the dense original (55.32). No explanation is offered for why a near-dense sparse model outperforms full attention. While likely within evaluation noise, it should be acknowledged.

- **Causal masking in block-sparse kernel is not discussed:** For decoder-only models, the attention map is lower-triangular. The paper does not explain how future blocks are handled in the block-sparse inference kernel — while presumably implemented correctly, the absence of any discussion leaves correctness implicit.

- **Pooling terminology conflation:** Section 3.1 and Section 4.1 both use "pooling" to mean two different things — pooling of Q/K as gate inputs vs. 2D max-pooling of the full attention map as training target. This causes genuine confusion when reading across sections.

---

## Nice-to-Haves

- **Per-head adaptive sparsity:** The paper identifies this as the likely cause of the 128k performance gap with MInference. Even a simple heuristic or learned head importance score would directly address the paper's acknowledged limitation.
- **Ablation of training target (max-pool vs. avg-pool or sum):** The pooling ablation in Figure 10 covers gate *inputs* but not the training *target*. Since a block's contribution to the output is proportional to its average attention weight × V values, avg-pool or sum-pool of the attention map may be a more principled supervision signal. An ablation would strengthen the max-pool design choice.
- **Quantification of AttnGate parameter count and training compute:** The paper mentions training completes "within hours" on 4 A100s but provides no parameter count for the added linear layers or precise GPU-hour cost, which would help practitioners assess the training investment.
- **Model architecture diversity:** Both evaluated models (Llama-3.1-8B and Mistral-7B) share RoPE and similar GQA-free architectures. Evaluation on models with grouped-query attention or different positional encodings would strengthen generalizability claims.
- **Layer-wise sparsity pattern heatmaps:** Showing how learned patterns evolve across early vs. late transformer layers (beyond the five examples in Figure 7) would deepen understanding of what structure the gate learns.
- **End-to-end training time for the fine-tuning setting:** Wall-clock fine-tuning time with vs. without AttnGate overhead would help users assess the training–inference cost tradeoff.

---

## Removed Points

*These points were raised in the sub-reviews but are removed or substantially deprioritized after cross-checking with the paper.*

- **"5.67× speedup is cherry-picked":** The abstract explicitly conditions the claim on "90% sparsity ratio at a 32k context length" and Figure 1c shows the full sparsity-speedup curve. This is not cherry-picking — the operating point is disclosed. (Removed: factually incorrect criticism.)
- **Missing confidence intervals / statistical significance:** Single-run evaluation is the established norm for LLM benchmarks at this scale. (Removed: non-standard requirement.)
- **Missing baselines H2O, SnapKV, PyramidKV:** These primarily target KV cache compression at the decode stage, a different problem from prefill sparse attention. Comparing against decode-stage methods would be scope creep. (Removed: out of scope given the paper's prefill focus.)
- **Requesting theoretical proofs for the MSE objective:** This is an empirical systems paper; formal proofs are not expected by the community. (Removed: non-standard requirement for this paper type.)
- **Criticism of missing references to Longformer, BigBird, Routing Transformers in related work:** The paper cites the relevant foundational works (Child et al. 2019, Zaheer et al. 2020) and clearly distinguishes its contribution as learned sparsity in *pre-trained* models. Per review instructions, missing related works are not flagged. (Removed.)
- **"The word 'significantly' in the abstract is imprecise":** Pure style nitpick. (Removed.)
- **Requesting user studies or ablation of the linear layer in AttnGate:** The linear layer is a standard design element; ablating its necessity is not required at this detail level. (Moved to non-critical.)
- **MInference sparsity comparison "borderline unfair":** SeerAttention at 0.4 sparsity vs MInference at 0.37 average sparsity is comparable enough; this is not a material unfairness. (Removed.)

---

## Novel Insights

The connection drawn between the pooling ablation results and K-outlier behavior in LLM quantization is a genuine emergent insight: the finding that avg-Q + (max+min)-K yields the best gate performance aligns with the known phenomenon that K tensors in attention have heavy-tailed distributions that are best captured through extreme statistics (max and min) rather than mean. This suggests a structural link between attention sparsity prediction and quantization-robustness challenges in K — a connection not previously articulated in either the sparsity or quantization literature. Additionally, the demonstration that block-level RoPE with rescaled frequencies (θ' = θ/B) enables robust length extrapolation in a compressed-domain module adds a small but useful design principle for any future work building lightweight attention predictors on top of pre-trained LLMs.

---

## Suggestions

1. **Fix Figure 1b immediately:** Replace with a same-dataset comparison (using Table 3's Proof-pile data for both baseline and SeerAttention across sparsity levels), or remove the panel and rely on Table 3 and Figure 4.

2. **Add LongBench/RULER evaluation for the fine-tuned model:** Run the YaRN+SeerAttention (0.5 and 0.9 sparsity) fine-tuned model on LongBench to validate the "near-lossless accuracy" claim at the task level, not just perplexity.

3. **Add a block recall analysis:** Compute the fraction of the top-K highest-attention blocks (by true dense attention) that AttnGate's Top-k successfully identifies, across layers and context lengths. This directly validates the training objective and addresses the training–inference gap concern.

4. **Ablate block size B:** Test at least B ∈ {32, 64, 128} to characterize the granularity–efficiency tradeoff and establish whether B=64 is consistently optimal or hardware/task-dependent.

5. **Clarify the TTFT comparison:** Add a sentence in §5.3.2 acknowledging that MoA and MInference baselines have non-trivial kernel overhead above FlashAttention-2 at short contexts, and position Figure 6's kernel-level speedup as the more controlled algorithmic comparison.

---

**Overall assessment across key axes:**

- **Novelty:** High. Framing attention sparsity as a learned, post-hoc problem with a dedicated training kernel is a clear departure from predefined-pattern methods.
- **Technical soundness:** Solid, with the caveat that the training–inference gap and the choice of max-pool supervision target are not fully justified.
- **Empirical support:** Good in post-training settings; incomplete for the fine-tuning setting (missing task evaluation). Figure 1b is a notable presentation flaw.
- **Significance:** High for the long-context inference community, with practical impact contingent on addressing the prefill-only scope.
- **Clarity:** Generally clear, but the dual meaning of "pooling" and the cross-dataset Figure 1b reduce precision.