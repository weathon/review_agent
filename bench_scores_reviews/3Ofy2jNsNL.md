## Summary

ACT-IN-LLM proposes compressing visual Key/Value tokens **within** LLM decoder layers, rather than before the LLM, to improve efficiency in high-resolution MLLMs. The core mechanism (ACM) uses the previous layer's attention weights from the final token's row to select the top-k most important visual K/V tokens while retaining all Query vectors and thus all hidden-state representations. A theoretical framework is developed framing this as a better low-rank approximation of full-attention than prior Pre-LLM or Early-LLM approaches, and extensive experiments across 0.5B–7B LLMs show a 5.5% improvement over the prior best compression method at similar token budgets.

---

## Strengths

- **Compelling and well-designed motivating experiment.** Figure 2(a) directly shows that dropping tokens at earlier layers causes progressively larger degradation—up to ~15% on high-resolution benchmarks—versus dropping at later layers. Figure 2(b) visualizes that vision tokens with low early-layer attention become highly attended later, providing concrete, model-specific evidence that the problem of pre-LLM compression is real. This goes beyond anecdotal claims.

- **Principled K/V-only compression design.** The architectural decision to compress only K/V while retaining full Q is not arbitrary: it preserves the full hidden-state residual stream so that every visual token still contributes to subsequent layers as a query, and the theoretical analysis in Section 4 provides a coherent justification via low-rank approximation. Figure 5 shows empirically that high-resolution token attention matrices are more low-rank than text matrices, directly supporting the design.

- **Strong controlled comparison in Table 2.** All baselines (Q-former, Avg-pooling, LLaVA-UHD, C-Abstracter, FastV, FlexAttention) are retrained under identical settings—same epochs, data, LLM, vision encoder, and training pipeline. This is the right experimental protocol for comparing compression strategies, and ACT-IN-LLM shows a 5.5% improvement over the next best method (FastV) on the high-resolution average, with large gains on ChartQA (+11.1) and DocVQA (+6.6).

- **Hierarchical compression ratio design is empirically grounded.** The choice r_hr > r_lr (compress high-resolution tokens more aggressively) is motivated by Figure 5's low-rank degree plots, and Table 4a confirms that hierarchical (r_i < r_j < r_p) outperforms flat ratios, matching the observed attention sparsification pattern across layers. This is a specific, verifiable design insight.

---

## Weaknesses

### Fatal
None.

### Major

- **Attention-guided selection barely outperforms simple pooling, undermining the core mechanism claim.** Table 4b shows: Attention-weight (75.04 / **45.35** HR), AvgPool-1D (**75.06** / 45.08 HR). The margin is 0.27 points on HR and essentially zero on general benchmarks. Given that the paper's central novelty claim is "text-guided, layer-wise adaptive compression," this near-identical performance with a non-adaptive, non-text-guided pooling baseline is a significant finding that the paper does not address or discuss. If a static spatial pooling of K/V achieves the same result, the mechanism attribution needs to be re-examined. The paper should either explain why this happens (e.g., the gains come from within-LLM placement, not from text guidance) or provide a more sensitive experimental setup to isolate the contribution of the attention-guided selector.

- **The "error correction mechanism" is claimed but never empirically demonstrated.** The abstract and Section 3.2 assert that retaining all Q tokens provides "an inherent error correction mechanism that mitigates the permanent loss of valuable information." However, there is no experiment that demonstrates this mechanism is active. Specifically: does retaining Q while compressing K/V at layer i actually help recover information in later layers? A direct ablation—(a) compress K/V and keep Q vs. (b) compress K/V and Q—would either validate or undermine this claim. Without it, the mechanism remains speculative.

- **Missing ablation on the critical design choice: why the last token's attention row?** ACM uses A_{i-1}[N+L, :] (the attention of the very last token) to score visual token importance. The paper justifies this with "the last token encodes the complete multimodal context," but this claim is not universal: in VQA tasks, question-word tokens may be more informative selectors; in detail-dense tasks, averaging over text tokens may be better. No ablation compares last-token guidance against mean-over-text-tokens, question-token-only, or max-pooled guidance. This is a central design choice with no comparative support.

- **Performance gap with Full-token model is not trivial and "competitive" is overstated.** Table 2: Full achieves 48.0 HR average vs. ACT-IN-LLM's 45.4—a 5.4% absolute gap. Calling this "competitive performance with non-compression methods" in the abstract and conclusion is misleading. The paper makes a strong case for being the best among compression methods, which is the appropriate claim.

### Minor

- **Default 70% ACM layers is not the best-performing configuration.** Table 5 shows that 50% layers achieves 75.25 general / 46.12 HR—clearly better than the default 70% (75.04 / 45.35). The paper's explanation is that 70% is chosen for the efficiency-performance trade-off, but this is buried in Section 3.3. The default should be presented as an efficiency choice, and the best-performance and best-efficiency configurations should be explicitly distinguished in reported results.

- **Memory savings are disproportionately small relative to claimed token reduction.** The abstract claims ~60% vision token reduction, yet Table 2 shows memory dropping only from 19.9 GB to 18.8 GB (~6%). This discrepancy is never explained. Possible causes (activations outside attention, K/V cache overheads, batch padding, implementation overhead) should be discussed. As currently presented, the practical efficiency claim is partially misleading.

- **Theory does not specifically justify the actual ACM construction.** Theorem 2 establishes *existence* of good C^K, C^V matrices; it does not prove that the specific top-k selection from previous-layer last-token attention achieves or approximates the bound. Theorem 3's advantage claim holds under the conditions of Theorem 2, but ACT-IN-LLM's actual selection algorithm satisfies those conditions only approximately. The theoretical section is directionally sound but overstates the connection between theory and the implemented algorithm.

- **Training time reduction claimed in the abstract but not reported experimentally.** The abstract states "~20% training/inference time reduction," but Table 2 only reports inference time. No training time numbers appear in the paper. The inference reduction shown is ~17% vs. Full (83% × 621ms = 515ms), which is consistent with "~20%," but the training claim should be supported with data.

- **Sharing token indices within each stage reduces the claimed layer-wise adaptivity.** Section 3.3 states "for efficiency, we keep the vision token index to be identical in each stage." This means compression is effectively stage-level, not layer-level, despite framing as "layer-wise adaptive." The impact on performance is not ablated (per-layer vs. per-stage selection).

### Tiny

- There are typographical issues (title reads "ADAPTIVELY COMPRESSION," Theorem 2 has "there there exists") and the notation in Eq. 6 (M̄_i dimensions) is slightly inconsistent with the surrounding text—these do not affect understanding but should be fixed.

---

## Nice-to-Haves

- **Video or multi-image evaluation.** The method's efficiency gains scale with token count; demonstrating on video or multi-image settings would show the upper bound of its practical value, even if such settings were not the original scope.
- **Visualizing which tokens are selected across layers.** Layer-wise heatmaps of retained token positions for the same image at early, middle, and late layers would directly validate whether the hierarchical compression behaves as intended and differs meaningfully from early-layer selection.
- **Plug-and-play evaluation on existing off-the-shelf MLLMs.** The paper shows "w/o train" rows only within its own training pipeline; showing ACM applied to an externally released pretrained MLLM (e.g., LLaVA-NeXT) at inference would demonstrate genuine zero-shot generalizability.
- **Failure mode / per-dataset breakdown.** Reporting where the ~5% HR gap concentrates (e.g., by task type or image resolution) would help users assess deployment risk.

---

## Removed Points

*These points are flagged as removed; treat them with caution.*

- **"Figure 6 table shows identical values for all methods"** — Removed. This is a PDF parser artifact where image-caption data was duplicated into a table; it does not reflect the actual paper content or Figure 6's findings.
- **"Theorem claims ACM is better than any C^Q, C^K, C^V, which seems too strong"** — Partially removed. The harsh critic frames this as the theorem being "universally false," but Theorem 3 is conditional on Theorem 2 and on the MLLM attention structure described in Assumption 1. The theoretical argument is limited but not obviously wrong for its stated scope.
- **"Baseline reproductions in Table 3 may be biased"** — Removed as actionable weakness. The asterisk notation in Table 3 indicates reproductions of SOTA models using *official checkpoints*, not retraining—so this is not a training-recipe fairness issue. Cross-system comparison in Table 3 is inherently noisy (different encoders, data scales), which is noted in the minor weaknesses above, but the specific concern about "identical hyperparameters" mischaracterizes the reproduction method.
- **"Some comparison with baselines is unfair (different architecture inductive biases)"** — Weakened to minor. Table 2 explicitly retrains all methods in the same setting, which is the appropriate controlled comparison; the concern about architecture-specific inductive biases applies to any fair comparison in this space and is not specific to this paper.
- **"Request for confidence intervals / multiple-run statistics"** — Removed. Single-run evaluation on fixed benchmarks is the norm for large-scale MLLM work; this is not a methodological deficiency in this community.
- **"Demanding limitations section"** — Removed as a formal weakness. While a limitations section would strengthen the paper, its absence is a presentation issue, not a scientific one.
- **"Notation mixes affinity and attention weights in Section 4"** — Removed. Eq. 9 makes explicit that A is post-softmax: "softmax(C^Q A (C^K)^T) · C^V V" where A = softmax(QK^T/√D) from Eq. 1. The notation is consistent, if compact.

---

## Novel Insights

One genuine insight emerges from Table 4b that the paper itself does not fully engage with: the near-identical performance of attention-guided selection and simple AvgPool-1D suggests that *the locus of the gain* may be the architectural placement of compression (within-LLM, layer-distributed, K/V-only) rather than the text-guided token selection mechanism. If this is true, it is a stronger and simpler result than claimed—one that shifts the contribution from "text-guided adaptive selection" to "within-LLM K/V compression as a structural design choice." The paper would be more honest and more insightful if it acknowledged this interpretation and tested it directly (e.g., uniform random K/V sampling vs. attention-guided sampling within the same framework).

---

## Suggestions

1. **Add a direct ablation comparing last-token guidance vs. average-over-text-tokens vs. random selection.** This is the single most important missing experiment, as it simultaneously tests the necessity of text guidance and justifies the choice of the last token.

2. **Explicitly test the "error correction" claim.** Add an ablation where K/V compression is applied identically, but Q is also compressed (removing the residual-stream preservation). If the performance drops more sharply than when Q is retained, the mechanism is real. If not, revise the claim.

3. **Reframe the abstract and conclusion.** Replace "competitive performance with non-compression methods" with a precisely calibrated statement (e.g., "achieves 94.6% of full-model performance on HR benchmarks"). Replace "retains all tokens" with "retains all hidden-state representations."

4. **Report or clearly commit to training time numbers.** If "~20% training time reduction" cannot be empirically shown, remove the claim from the abstract.

5. **Present 50%-layers as the high-performance variant and 70%-layers as the efficiency variant** in a clear table, so readers can select the right operating point without confusion.

6. **Discuss the memory discrepancy explicitly.** A short paragraph explaining why 60% token reduction yields only ~6% memory savings (e.g., Q-side dominance, activation buffers, implementation overhead) would significantly improve the credibility of the efficiency story.