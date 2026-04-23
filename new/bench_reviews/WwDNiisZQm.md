Now I have a thorough understanding of the paper and calibration anchors. Let me write the final review.

## Summary

The paper introduces Content-Aware Mamba (CAM) for learned image compression, comprising two mechanisms: (1) Content-Adaptive Token Permutation (CTP), which reorders tokens by feature-space similarity via learnable codebook clustering so that semantically similar tokens are processed consecutively by the SSM, and (2) Global-Prior Prompting (GPP), which augments the SSM output matrix C with cluster-derived prompts. The resulting CMIC model achieves BD-rate reductions of 15.91%, 21.34%, and 17.58% over VTM-21.0 on Kodak, Tecnick, and CLIC datasets, while substantially reducing parameters, FLOPs, memory, and latency compared to prior Mamba-based LIC models.

## Strengths

- **CTP is well-motivated, cleanly designed, and empirically effective.** The codebook-based cosine K-Means clustering provides stable training (via EMA centroid updates) and deterministic inference, directly addressing the raster-scan limitation. Ablations in Table 2 show CTP alone contributes 1.8–2.4% BD-rate reduction. The cluster visualizations in Figure 10 convincingly demonstrate semantically coherent groupings (e.g., red doors/windows, clouds/sky, feathers), and Table 5 shows adaptive activation (mean 23.27/26.30 active clusters out of 64, with meaningful variance).

- **Strong and consistent rate-distortion improvements over prior Mamba-based LIC models.** CMIC surpasses MambaIC by 2.17–6.48% and MambaVC by 6.80–10.09% in BD-rate across three datasets (Table 1), while also outperforming Transformer-based FTIC by 0.15–0.36 dB in BD-PSNR. CMIC achieves the best BD-rate on Tecnick (−21.34%) and CLIC (−17.58%), and is highly competitive on Kodak (−15.91%).

- **Substantial efficiency gains over competing Mamba-based methods.** CMIC reduces parameters by 56%, FLOPs by 57%, decoding latency by 39%, and GPU memory by 78% compared to MambaIC (Table 1). The single selective scan avoids the computational overhead of multi-directional 2D scans while achieving better compression.

- **Comprehensive ablation design.** Table 2 isolates CTP and GPP contributions, Table 4 compares against Conv/2D Mamba/Attention-only alternatives, Table 6 varies cluster count K, and Table 5 analyzes adaptive activation. Collectively these demonstrate that each component is necessary and the proposed combination is superior.

- **Content-adaptive ERF visualizations provide intuitive validation.** Figure 8 shows CMIC's receptive field aligns with semantic structures (hair/feathers, shorelines, aircraft) across different images, while FTIC and TCM-L produce nearly isotropic, content-agnostic ERFs — directly illustrating the qualitative advantage of content-adaptive processing.

- **Minimal inference overhead.** Table 3 shows decoding latency increases by only 4% (0.387s → 0.405s for 2K images) when adding CTP and GPP, and training throughput drops only from 23.19 to 22.05 samples/s.

## Weaknesses

### Fatal
None.

### Major

- **The GPP mechanism does not genuinely relax causality as claimed, undermining one of the two named contributions.** The paper's second main contribution is framed as "relaxing the strict causality" of the SSM (Abstract, Section 3.4, Conclusion). However, the forward-pass computation O_i = (C + P_i)h_i + Dx_i remains strictly causal: P_i = A(c_{g_i}) depends only on token i's own features and the shared, inference-fixed centroids — not on other tokens in the current image. The hidden state h_i depends only on preceding tokens. Thus GPP provides *token-level semantic conditioning* (a learned category-dependent readout modulation), not non-causal information flow from future tokens. The paper repeatedly calls this "relaxing the strict causal constraint" and claims GPP allows the model to "see beyond" the strictly causal scan, which mischaracterizes the mechanism. GPP empirically helps (0.7–1.2% BD-rate in Table 2), but for a different reason than claimed — it conditions the output readout on the token's semantic category, not on global image information beyond the causal boundary.

- **The ERF evidence for non-causality (Fig. 9) uses soft clustering, a different mechanism than the actual model's hard one-hot assignments.** The paper states: "We compute the ERF between the input and output features of a single state-space model layer **with soft clustering** to analyze the proposed non-causal mechanism." The actual model uses hard one-hot assignments (Γ ∈ {0,1}^{N×K}, Section 3.4). Soft assignments create gradient flow through the assignment matrix that would not exist with hard argmax, potentially showing ERF patterns that the actual model cannot produce. This makes Fig. 9 misleading as primary evidence for the claimed non-causality. The visualization should use the same hard assignments as the model, or the discrepancy should be explicitly discussed.

- **Misleading "distribution-aware" and "sample-specific global prior" framing.** Section 3.4 describes the prompt dictionary as "distribution-aware" and says the prompt "summarizes the overall token distribution of the input." In reality, P_i = A(c_{g_i}) encodes only *which cluster token i belongs to* — determined by comparing its features to shared, inference-fixed centroids. The prompt does not encode the distribution of tokens across clusters (e.g., how many tokens are in each cluster, or which clusters are more prevalent for this image). The "global prior" is a dataset-level codebook lookup, not an image-level global summary.

### Minor

- **"Consistently outperforms leading methods across all evaluated datasets" (Section 4.3) is overstated.** On Kodak, MLICv2 achieves −16.16% BD-rate while CMIC achieves −15.91%, making CMIC 0.25 percentage points worse. CMIC is the best on Tecnick and CLIC and competitive on Kodak — a strong but not uniformly dominant result. The claim should be qualified to reflect this.

### Trivial
None.

## Nice-to-Haves

- It would strengthen the paper to provide an honest re-framing of GPP's contribution as "semantic category conditioning for the output readout" rather than "causality relaxation," which would align the theoretical claims with the actual mechanism. The empirical benefit is real; the theoretical interpretation is what needs correction.
- Showing the ERF in Fig. 9 with the actual hard assignments (or discussing why soft clustering is used for visualization and what differences arise) would make the evidence more trustworthy.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Strength Finder's claim that "GPP provides a principled mechanism for relaxing causality without multi-directional scanning"** — this strength conflicts with the verified Major weakness above. GPP does not relax causality; it provides semantic category conditioning. The mechanism works empirically, but the theoretical claim is incorrect. Moved to Removed Points.

- **Strength Finder's claim that "ERF analysis in Figure 9 provides clear evidence... confirming non-causal behavior"** — the ERF visualization uses soft clustering (different from the actual model's hard assignments), making this evidence unreliable for supporting the non-causality claim. Moved to Removed Points.

- **Strength Finder's claim that "Adaptive cluster activation demonstrates genuine content awareness"** — while Table 5's data is correct, framing this as "genuine content awareness" is somewhat misleading since the cluster activation count is a byproduct of the fixed codebook, not an explicit content-aware design decision. The variance in activation is real but its interpretation as "content awareness" is generous. However, this is a borderline removal — the data does show variation across images, so partial credit is due.

## Novel Insights

The paper's CTP mechanism is a genuinely effective approach to making Mamba's scan order content-adaptive for image compression, and the codebook-based clustering with EMA updates provides a clean and stable alternative to online K-Means. However, the GPP contribution reveals an important conceptual subtlety that the paper misses: augmenting the output matrix C with a category-dependent prompt changes *how* the causal hidden state is read out, not *what* information the hidden state contains. This is analogous to giving a reader different "lenses" based on document type — it helps interpretation but doesn't provide access to unread pages. Recognizing this distinction would lead to a more accurate understanding of when and why prompting helps in SSMs.

## Suggestions

- Re-frame GPP's contribution honestly: it provides semantic-category-dependent output modulation that is independent of scan position, which can reduce the model's sensitivity to scan order without breaking causality. This is still a valuable contribution, just not the one currently claimed.
- Either compute the Fig. 9 ERF with the actual hard assignment mechanism, or explicitly acknowledge the discrepancy and explain why soft clustering is used for visualization and what additional gradient paths it creates.
- Qualify the "consistently outperforms" claim in Section 4.3 to note that MLICv2 slightly outperforms CMIC on Kodak.

## Score and Decision

**Calibration comparison:**

- **InfoTok (7.33, Accept Oral):** CMIC is below this — InfoTok has principled theoretical grounding (ELBO-based optimality proof) with strong empirical results. CMIC's theoretical contribution (GPP causality relaxation) is overstated.
- **InfoScan (6.0, Accept Poster):** CMIC is comparable — both propose content-aware scanning for SSMs. CMIC has stronger domain-specific empirical results and better efficiency numbers, but InfoScan has a cleaner theoretical framing without overclaims.
- **Cluster-Masked Scanning (4.67, Reject):** CMIC is clearly above this — both use cluster-based reordering, but CMIC has much stronger empirical validation and the CTP contribution is well-supported.
- **SF-Mamba (4.5, Reject):** CMIC is clearly above this — SF-Mamba was rejected for limited novelty (overlap with prior work) and marginal downstream performance, while CMIC has genuine CTP novelty and strong compression results.
- **VF-Mamba (4.0, Reject):** CMIC is clearly above this — VF-Mamba was rejected for incremental novelty and limited ablation scope.
- **Combine-ICMH (2.5, Reject):** CMIC is well above this — Combine-ICMH was pure engineering refinement with no architectural novelty, while CMIC has genuine algorithmic contributions.
- **Flow-IB (2.5, Reject):** CMIC is well above this — Flow-IB had fundamental methodological issues (no full RD characterization, extreme overclaims).

CMIC sits between InfoScan (6.0) and the rejected Mamba scanning papers (4.0-4.67). The strong empirical results, well-validated CTP contribution, and efficiency gains push it above the rejected papers, but the overstated GPP contribution and misleading ERF evidence prevent it from reaching the level of cleanly-theorized papers like InfoTok. A score of 6.0 reflects a paper with genuine and significant practical contributions that are partially undermined by overclaimed theoretical framing.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>