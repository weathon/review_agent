Now let me search for calibration papers to properly score this work.Now I have enough information to write the final consolidated review. Let me also quickly look at one more high-scoring paper for calibration.Now I have all the information I need to write the final consolidated review.

---

## Summary

SeerAttention introduces a learnable block-level gating module (AttnGate) that adaptively identifies sparse blocks in attention maps during inference, replacing static or heuristic-based sparse attention methods. To enable scalable training, the authors develop a customized FlashAttention kernel that extracts block-level attention ground truth (via row-max rescaling and column-max pooling) with negligible overhead. The method is evaluated in both post-training and long-context fine-tuning (YaRN) settings, with Table 3 showing near-lossless accuracy (PG19: 8.81 vs. 8.79 baseline) at 50% sparsity and 5.47× kernel speedup at 128k / 90% sparsity.

---

## Strengths

- **Customized FlashAttention training kernel (Section 4.2, Figure 3, Figure 8)**: Modifying FlashAttention to store and rescale the per-block row-max and then apply column-max pooling is a concrete and non-trivial engineering contribution. Figure 8 demonstrates that the overhead is negligible compared to FlashAttention-2, whereas naive PyTorch attention OOMs beyond 4k tokens — enabling scalable training that would otherwise be infeasible.

- **Flexible inference-time sparsity control (Section 4.1, Figure 4)**: A single AttnGate checkpoint can serve arbitrary Top-k ratios at inference time, allowing the accuracy/speed tradeoff to be tuned without retraining. Figure 4 cleanly demonstrates stable perplexity across sparsity ratios 0–0.9 for multiple context lengths from a single model checkpoint.

- **Additional RoPE in AttnGate enables length generalization (Section 3.1, Figure 9)**: The identification that pooling destroys relative positional information and the solution of re-encoding block positions with θ' = θ/B is an insightful and well-supported design choice. Figure 9 shows that without this, perplexity explodes beyond the training length (~16k), while with it, the gate generalizes stably to 128k.

- **Learned pattern diversity (Figure 7)**: The visualization showing AttnGate recovers A-shape, vertical, slash, diagonal, and random patterns without hand-coded priors is a direct empirical demonstration that the learning-based approach subsumes and extends the pattern space of heuristic methods (MoA, MInference).

- **Long-context fine-tuning integration (Table 3)**: Incorporating SeerAttention into YaRN fine-tuning is a practical and well-executed demonstration. Table 3 shows YaRN+SeerAttention at 0.5 sparsity achieves 8.81 (PG19) vs. 8.79 baseline — correctly evaluated on both PG19 and Proof-pile at the same 32k context length.

- **Negligible gate overhead (Figure 5)**: AttnGate and Top-k operations account for only ~1–3% of total latency, with the overhead diminishing further at longer sequence lengths.

- **Practical post-training cost (Section 4.3)**: Training only AttnGate parameters on 500 steps with 4 A100 GPUs (hours of compute) is a concrete, low-barrier deployment claim.

---

## Weaknesses

### Fatal
None.

### Major

- **MoA TTFT comparison is likely polluted by non-attention overhead (Table 4)**. MoA at only 35% sparsity runs at 1.29s (8k), 10.34s (32k), and 36.34s (64k), compared to FlashAttention-2's 0.90s, 4.63s, and 10.09s — a 2.2–3.6× *slowdown* relative to dense attention despite being a sparse method. Figure 6 corroborates this: MoA's kernel speedup sits near 1.0 across all sparsity levels, meaning MoA's block-sparse kernel provides essentially zero kernel-level speedup. The paper explains this by noting "MoA requires an exhaustive search for sparse configurations…therefore we only compared against its default configuration," which suggests MoA's TTFT measurement may include sparse pattern search or other offline-calibration steps, not just inference. If so, Table 4 is not measuring inference latency of sparse attention vs. dense attention: it is measuring MoA's total pipeline cost, which is not a fair efficiency comparison. The efficiency advantage over MoA — one of two headline baselines — therefore cannot be taken at face value. SeerAttention's own speedup over FlashAttention-2 (Table 4, Figure 5) remains credible on its own terms; the problem is specifically the comparison against MoA.

- **Figure 1b compares two different test distributions, making the headline display result misleading (Figure 1b)**. The figure plots "YaRN Baseline (PG19)" (perplexity ≈10) and "YaRN w/ SeerAttention (Proof-pile)" (perplexity ≈3) on the same axis across sparsity levels 0.5–0.9. These are entirely different test distributions with very different absolute perplexity scales. A reader examining Figure 1b and the associated caption ("50% sparsity achieves near-lossless performance, and even at 90% sparsity, the loss remains minimal") cannot extract a valid comparison — the SeerAttention curve is lower simply because Proof-pile has intrinsically lower perplexity than PG19. The correct evidence lives in Table 3, which properly compares on matched datasets, but Figure 1b appears as the headline result and is likely read first. This is a genuine presentation flaw that should be corrected by showing both curves on the same dataset.

### Minor

- **SeerAttention underperforms MInference at the most important setting (128k context, high sparsity)**. Table 1 shows that at 128k context and s=0.9, SeerAttention achieves 13.20 perplexity while MInference achieves 10.89 at matched sparsity (s=0.9). The paper acknowledges this as arising from MInference's per-head sparsity adaptation vs. SeerAttention's fixed global ratio, and flags per-head sparsity as future work. This is a real performance gap at the highest-priority target use case (long context, high sparsity). The framing in Section 5.1 that SeerAttention "consistently outperforms both MoA and MInference in most cases" is technically correct but undersells this failure mode.

- **Prefill-only scope is not prominently disclosed in the abstract or introduction**. Section 5 briefly states "AttnGate currently solely applies in the prefill stage," and the conclusion flags decoding as future work. However, the abstract presents SeerAttention as improving "long-context LLMs" efficiency without qualification. For production deployments, decode latency (which depends on KV cache access, not prefill computation) often dominates total cost. This limitation should appear in the abstract, not solely in a parenthetical clause in the experiments section.

### Trivial

- **Block size B=64 is fixed throughout without ablation**. Block size directly determines the accuracy-efficiency frontier: smaller B allows finer-grained sparse patterns but increases gate overhead, while larger B reduces overhead but introduces coarser approximations. An ablation on at least one model would confirm B=64 is near-optimal rather than arbitrary.

- **Pooling configuration was selected on Llama-3.1-8B at 32k without cross-architecture validation**. The winning Qavg/Kmax+min pooling was identified via Figure 10 on a single architecture. Whether this generalizes to Mistral or other GQA models is not empirically confirmed.

---

## Nice-to-Haves

- **Equal-sparsity accuracy comparison against MInference**: A table holding sparsity fixed (e.g., s=0.5, 0.7, 0.9) for both methods at 32k and 128k context would resolve the sparsity-mismatch ambiguity in Tables 1–2 and cleanly establish whether learned sparsity is strictly better than heuristic sparsity.

- **Block recall analysis**: Measuring recall@k (what fraction of the truly top-k blocks are selected by AttnGate) would provide a more principled characterization of gate prediction quality than perplexity alone, and would directly explain the accuracy degradation at 128k / high sparsity.

- **Training supervision signal comparison (max-pool vs. sum-pool)**: The gate is supervised on the row-max of softmax blocks (Eq. 2), but block contribution to the attention output depends on the softmax-weighted V sum. Whether sum-pool supervision would improve accuracy is an interesting ablation.

- **Head-level sparsity distribution analysis**: Showing the natural per-head sparsity distribution at different context lengths would motivate per-head sparsity adaptation (flagged as future work) and explain why the 128k performance gap vs. MInference appears.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"5.67× vs 5.47× discrepancy" (Harsh Critic)**: The 5.67× quoted in the abstract refers to end-to-end kernel speedup at 32k / 90% sparsity (in the fine-tuning scenario, per Figure 1c), while the 5.47× is the kernel-level speedup at 128k / 90% sparsity (Figure 5 caption). These measure different things (different sequence lengths, different evaluation modes). This is not a real inconsistency.

- **RoPE alternative comparisons demanded (Harsh Critic)**: Requesting comparisons to NTK-aware scaling or ALiBi in the AttnGate is out of scope for the paper; the ablation in Figure 9 is sufficient to justify the design choice made.

- **Max-pool vs. sum-pool supervision "design mismatch" as a flaw (Harsh Critic)**: This is a speculative concern. The paper chose max-pool empirically and it works well (Table 3). This is a nice-to-have ablation, not a structural flaw.

- **Cross-architecture pooling ablation demanded as a Major issue (Harsh Critic)**: Lacking cross-architecture validation for the pooling configuration is a genuine but minor concern (trivial tier at most), not a fatal or major weakness.

---

## Novel Insights

The most genuinely novel observation across both reviewers is the RoPE treatment in AttnGate: because pooling disrupts relative positional information, applying a separate block-level RoPE with reduced rotational angle (θ' = θ/B) is necessary for length generalization, and this is cleanly demonstrated by the degradation in Figure 9 without it. This insight is transferable to other gating mechanisms over pooled sequence representations. The second notable insight is that the customized FlashAttention training kernel — which extracts per-block attention statistics by storing and rescaling intermediate row-maxima — is a broadly applicable technique that may benefit other methods requiring block-level attention supervision.

---

## Suggestions

1. Fix Figure 1b to show both the YaRN baseline and YaRN+SeerAttention on the *same* test dataset, or replace it with a properly matched accuracy vs. sparsity plot. Table 3 has the right data; Figure 1b should reflect it.
2. Clarify in the abstract and early introduction that SeerAttention currently addresses only the prefill (TTFT) phase.
3. Profile and report MoA's latency breakdown separately (kernel time vs. pattern search time) to distinguish whether the MoA TTFT overhead is genuine kernel cost or pipeline overhead, and cite MoA's own reported numbers if available.
4. Add an equal-sparsity head-to-head row in Table 1 at 128k to honestly characterize where SeerAttention stands relative to MInference at the longest context.
5. Include a block size ablation (B=32, 64, 128) on at least one model in the appendix.

---

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Comparison |
|---|---|---|---|
| FastGen (uNrFpDPMyo) | 8.0 | Accept (oral) | High bar: profiling-guided adaptive KV cache, exhaustive task coverage, uniform 8s — SeerAttention is narrower in scope and has a presentation flaw |
| FlashMask (wUtXB43Chi) | 7.0 | Accept (poster) | Engineering contribution to FlashAttention masking; comparable depth to SeerAttention's kernel work |
| SEA (JbcwfmYrob) | 6.67 | Accept (poster) | Learns sparse attention mask, custom Triton kernel — directly comparable scope; SeerAttention has broader evaluation (post-training + fine-tuning, multiple models) |
| ZETA (j9VVzueEbG) | 7.0 | Accept (poster) | Top-k attention variant; comparable novelty |
| ShadowKV (vHO9mU87dc) | 6.75 | Reject | Long-context inference efficiency, rejected despite moderate scores; SeerAttention's efficiency story is partially weakened by the MoA comparison issue |
| Hierarchy-Aided Sparse Attention (Hjk1tWIdvL) | 5.0 | Reject | Prefill sparse attention — narrower contribution, missing baselines, weaker engineering; SeerAttention clearly above this |
| Recycled Attention (8qYuxV4lRu) | 5.4 | Reject | Sparse attention inference for long context; missing baselines, narrower evaluation — SeerAttention is more complete |
| LM-Infinite (pOujzgHIRY) | 4.0 | Withdrawn | Low-quality long-context method; clearly below SeerAttention |

SeerAttention is solidly above the 5.0–5.4 tier (HASA, Recycled Attention): it has stronger baselines, richer ablations, a concrete kernel engineering contribution, and results on two application settings. It is broadly comparable to the 6.67–7.0 tier (SEA, FlashMask, ZETA), sharing the profile of a focused attention-efficiency paper with a custom kernel and competitive empirical results. The major weaknesses (MoA comparison anomaly, Figure 1b mismatch) prevent it from matching the cleanness of the 7.0+ papers, but they do not invalidate the core claim that learned sparsity outperforms static patterns.

**Final score: 6.5**

Originality: Good — learning sparsity rather than predefined patterns is well-motivated, and the RoPE-in-gate insight is novel.
Importance: High — prefill efficiency for long-context LLMs is a pressing practical problem.
Claim support: Partially well-supported — the fine-tuning and kernel results are convincing; the MoA comparison and Figure 1b weaken the efficiency story.
Experimental soundness: Good for self-contained experiments; the MoA TTFT comparison is suspect.
Writing clarity: Reasonable, with Figure 1b being a notable presentation flaw.
Value to community: High — the customized FlashAttention kernel and the RoPE design insight are transferable.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>