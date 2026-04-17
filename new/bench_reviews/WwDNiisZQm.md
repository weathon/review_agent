Now I have enough context from the paper and related reviews to write the final consolidated review.

## Summary

The paper introduces Content-Aware Mamba (CAM) for learned image compression, targeting two limitations of standard Mamba-style SSMs: content-agnostic fixed scan order and strict causal dependency. Two mechanisms are proposed: Content-Adaptive Token Permutation (CTP), which reorders tokens via codebook-based K-Means clustering to group content-similar tokens, and Global-Prior Prompting (GPP), which conditions the SSM output matrix on cluster-derived prompts. The resulting CMIC model achieves state-of-the-art rate-distortion performance, surpassing VTM-21.0 by 15.91–21.34% BD-rate across three benchmarks.

## Strengths

- **Strong RD performance with clear improvements over prior art.** CMIC achieves BD-rate reductions of 15.91%, 21.34%, and 17.58% over VTM-21.0 on Kodak, Tecnick, and CLIC, respectively, and outperforms prior Mamba-based LIC models (MambaVC, MambaIC) by 2.2–10.1 percentage points, consistently across all bitrates. These are substantial, well-documented gains in an actively competitive area.

- **Content-adaptive token permutation is a well-motivated and effective contribution.** Replacing the rigid raster scan with a clustering-driven reordering that prioritizes feature-space proximity over Euclidean adjacency directly addresses a real limitation of 1D SSMs applied to images. The ablation (Tab. 2) shows CTP alone contributes 2.0–2.4% BD-rate improvement, and the clustering visualizations (Fig. 10) provide convincing qualitative evidence that semantically meaningful groupings are formed.

- **Favorable efficiency–performance trade-off.** CMIC uses 69M parameters and 2.39 TFLOPs, compared to MambaIC's 157M/5.56 TFLOPs, while achieving better RD performance. The 78% memory reduction over MambaIC (4.44 GB vs 20.32 GB) is practically significant and aligns with the stated goal of avoiding multi-directional scans.

- **Thorough ablations and analysis.** The paper provides component ablations (Tab. 2), structural alternative comparisons (Tab. 4), cluster number studies (Tab. 6), ERF visualizations (Figs. 7–9), and clustering/activation statistics (Tab. 5), giving a multifaceted picture of the method's behavior.

## Weaknesses

### Major:

- **The "non-causal" / "mitigating strict causality" framing of GPP is a conceptual overclaim.** The paper repeatedly states that GPP "overcomes the sequential dependency," "relaxes the strict causality," and enables "non-causal long-range modeling" (abstract, intro, Sec. 3.4, Fig. 9 discussion). However, examining the SSM equations: h_i = Āh_{i-1} + B̄x_i, O_i = (C + P_i)h_i + Dx_i, where P_i = Γ_i U. The hidden state h_i still depends only on predecessors in scan order. The prompt P_i is determined by token i's cluster assignment and a dataset-learned codebook — it provides a *type-level global prior* (what kind of token this is), not access to future token content. The SSM recurrence remains strictly causal. GPP's true contribution is cluster-conditioned output projection, which enriches each token's readout with global dataset statistics about its semantic category, but this does not constitute breaking or relaxing causality. The ERF visualizations in Fig. 9 showing "non-zero activations after the scanned sequence" are artifacts of the clustering/P-projection path in gradient computation, not evidence of non-causal information flow from future tokens. This overclaim pervades the paper and should be corrected to "cluster-conditioned output projection enhancing global context utilization" rather than "non-causal modeling."

- **Incomplete ablation framework for isolating CTP's contribution from permutation vs. clustering effects.** The CTP mechanism combines two design choices: (1) clustering tokens by semantic similarity, and (2) permuting the scan order to group similar tokens. No ablation tests clustering *without* permutation (e.g., using cluster IDs as auxiliary features while keeping raster order) or permutation with random/simplified sorting versus VQ-codebook clustering vs. k-nearest-neighbor reordering. Without these controls, it is unclear whether the gains come primarily from the scan reordering itself or from the clustering mechanism's ability to capture redundancy structure. Similarly, there is no ablation replacing GPP with a simpler alternative (e.g., learnable per-block bias, single global prompt per image) to verify that the cluster-tied prompt dictionary design is specifically necessary. Given that GPP's incremental contribution is modest (0.7–1.2% over CTP alone), this matters for properly attributing credit.

- **The narrative advantage over multi-directional / 2D Mamba is under-evidenced relative to its prominence.** The paper heavily emphasizes that CMIC avoids "quadrupling computational complexity" from multi-directional scans, but the only comparison is Tab. 4's "2D Mamba" within the CMIC backbone (stages 3–5), which is not a faithful adaptation of the specific multi-directional approaches criticized (e.g., those in MambaIC or Vision Mamba). There is no controlled comparison where a bidirectional scan replaces GPP while keeping CTP, which would directly test whether content-adaptive prompting is superior to the standard approach for addressing causality. The 2D Mamba also has slightly more parameters (71.44M vs 69.11M), partially confounding the comparison.

### Minor:

- **CAM yields negligible gains in the entropy model (acknowledged in Sec. 4.5) but this limitation is not adequately analyzed.** The paper briefly states: "adding CAM yields negligible performance gains while increasing latency, indicating a limitation of CAM in enhancing entropy modeling." This is a notable scope limitation — content-adaptive scanning benefits transform networks but not entropy models — yet receives no further investigation. Understanding *why* (e.g., whether the autoregressive context model already captures local redundancies effectively, or whether permutation disrupts the spatial structure that the context model relies on) would strengthen the paper.

- **FLOPs measurement may not fully include clustering/permutation overhead.** The reported TFLOPs in Tab. 1 is measured on the core network operations. If the K-Means assignment (even at inference, it involves N×K cosine similarity computations per block) and permutation costs are excluded, the efficiency comparison against multi-directional scans is favorable but incomplete. The 5% training overhead claim is about wall-clock time, not FLOPs.

- **The complementarity between CTP and GPP is moderate rather than strong.** Tab. 2 shows that CTP adds ~2.0% over baseline, GPP adds ~1.0%, and together they add ~2.65%. The combined gain falls short of additive (~3.0%), suggesting some redundancy. The paper claims they are "highly complementary" which is a mild overstatement.

### Trivial:

- **The ERF visualization for a single Mamba layer (Fig. 9) uses soft clustering** while the model uses hard argmax assignments. This discrepancy is noted by the spark reviewer; while the visualization still illustrates the qualitative behavior, the authors should acknowledge this gap.

## Nice-to-Haves

- A direct ablation replacing GPP with a bidirectional scan (keeping CTP) would substantiate the claimed superiority of prompting over the standard multi-directional approach for addressing causality in a more controlled manner.
- Analysis of failure cases where content-adaptive clustering may be suboptimal (e.g., images with uniform textures where raster order suffices) would reveal method limitations.
- Per-image BD-rate distributions (rather than just averages) would show whether gains are consistent or driven by specific image types.
- A breakdown of inference-time costs separating K-Means assignment vs. selective scan would clarify practical deployment trade-offs.

## Removed Points

These points were flagged but are removed or weakened for the following reasons:

- **Harsh critic's claim that the conceptual overclaim is "fatal" and warrants rejection.** This overstates the severity. GPP still contributes a measurable improvement (0.7–1.2% BD-rate), and the overall method delivers strong empirical results. The overclaim requires reframing, not invalidation of the contribution — the paper should describe GPP as "cluster-conditioned output projection" rather than "non-causal modeling."

- **Neutral reviewer's concern about non-contiguous memory access from permutation.** This is a hardware-implementation concern that is speculative without measurement. The paper reports actual latency numbers (Tab. 1, Tab. 3), which already reflect real-world performance including any such effects.

- **Spark's demand for MS-SSIM comparisons with MambaVC/MambaIC.** The paper provides a valid reason: those methods are only optimized for MSE, making a fair comparison impossible. This is a scope limitation, not a flaw.

- **Harsh critic's complaint about missing training details (crop size, augmentation, GPU sync for codebook).** These are minor reproducibility concerns; the core architecture and loss are well-specified, and code is promised.

- **Human finder's point about "gains stemming from architectural capacity rather than proposed mechanisms."** Tab. 2's ablation shows the vanilla Mamba baseline (same architecture family, same param regime) achieves -13.26% while CTP+GPP achieves -15.91%, isolating the contribution of the proposed modules to ~2.65% BD-rate. While some confounds remain (the baseline is a simple single-scan variant), the ablation framework does identify meaningful incremental gains.

- **Human finder's concern about K-Means instability during training.** The paper explicitly discusses this in Sec. 3.3 and Appendix A.8–A.10, describing EMA smoothing, initialization strategy, and stability analysis. This concern is addressed in the paper.

- **All formatting/style nitpicks** are removed per the rules.

## Novel Insights

The paper's most genuinely novel insight is that *feature-space proximity is more important than spatial proximity for SSM-based compression*. The CTP mechanism compellingly demonstrates that simply reordering the 1D scan so that content-similar tokens are neighbors in the sequence yields consistent BD-rate improvements, confirming that Mamba's sequential processing is less of a limitation than its default content-blind ordering. The cluster visualization (Fig. 10) showing semantically coherent groups (edges, sky, texture regions) is a clear illustration of this principle. The finding that CAM fails to improve the entropy model (Sec. 4.5) is also an interesting negative result suggesting that the redundancy structure captured by content-aware scanning may be redundant with what context models already exploit locally.

## Suggestions

- Reframe GPP's contribution throughout the paper: replace "non-causal"/"relaxing causality" language with "cluster-conditioned global prior" or "global-prior-conditioned output projection." The current framing overclaims what the mechanism achieves; honest description would maintain the contribution's value while avoiding misleading claims.
- Add an ablation that applies clustering but keeps the raster scan order (feeding cluster IDs as features), to isolate whether scan reordering is the critical component of CTP or whether clustering information alone suffices.
- Add an ablation replacing GPP with a 2-directional scan (keeping CTP), to directly compare the prompting approach against the standard multi-directional alternative on equal footing.

## Score and Decision

**Calibration:**
- FTIC (Accept poster, scores 6,6,6,6): SOTA LIC with novel frequency-aware transformer, moderate novelty, strong results.
- MambaVC (Withdrawn/Reject, scores 3,5,5,6,5): First Mamba-for-compression paper, but incremental (directly copied VSS blocks from VMamba), weak novelty, weaker results.
- This paper occupies a middle ground: much stronger than MambaVC (genuine novel mechanisms, far superior RD results), comparable or slightly stronger empirical results than FTIC, but with a notable conceptual overclaim that inflates one of two main contributions. CTP alone (the stronger contribution) is well-motivated and effective; GPP's contribution is real but modest, and its framing as "non-causal" is incorrect.
- Compared to FTIC (the closest LIC comparator with 6s), this paper has marginally stronger empirical results and a novel architectural idea (CTP), but its central narrative overclaims what GPP achieves. The empirical contribution is sound and substantial regardless of the causality framing. I rate this slightly above the acceptance threshold.

**Score: 5.5** — The paper makes a genuine, well-motivated contribution with CTP and delivers strong empirical results. However, the pervasive overclaim about GPP "mitigating strict causality" / "non-causal modeling" is a significant conceptual inaccuracy that inflates one of the paper's two flagship contributions. With corrected framing (and modest additional ablations), the paper would be a clear accept; in current form, it is borderline.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>