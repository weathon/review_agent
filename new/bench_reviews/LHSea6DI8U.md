Now I have all the information I need. Let me write the final review.

## Summary

The paper proposes STBP, a continual spatio-temporal forecasting framework that pairs a frequency-domain backbone (FreNet + Dual-Stream Linear Graph Attention) with a scalable contextual pattern bank. The backbone is frozen after initial training to preserve general knowledge, while the pattern bank expands via parameter concatenation and interacts with the backbone through prompt-based gating and attention mechanisms. STBP achieves substantial improvements over prior CSTF methods on traffic datasets (21.44% and 21.93% MAE reductions on PEMS-Stream and CA-Stream) and more modest gains on air quality data (2.35% on AIR-Stream).

## Strengths

- **Large and consistent empirical improvements on traffic datasets**: STBP reduces average MAE by 21.44% over the best baseline EAC on PEMS-Stream (12.31 vs. 15.67, Table 1) and 21.93% on CA-Stream (15.77 vs. 20.20). Gains hold across all metrics, all forecasting horizons, and all three datasets, leaving little ambiguity about the method's effectiveness on traffic data.

- **Principled stability–plasticity design**: The frozen backbone + expandable pattern bank architecture directly addresses the stability-plasticity tradeoff in continual learning without requiring replay buffers or regularization. The ablation study (Figure 4, "Retrain" vs. "Online" vs. full STBP) confirms that both components contribute — the pattern bank alone is insufficient (Retrain), and the backbone alone suffers from forgetting (Online).

- **Novel DLGA mechanism integrating pattern bank as additional key**: The Dual-Stream Linear Graph Attention (Eq. 7–9) incorporates the pattern bank P^(2)_τ as an additional key in linear attention, enabling stored knowledge to guide spatial correlation modeling while reducing complexity from O(N²) to O(Nd). The ablation removing DLGA (Figure 4) confirms its importance with significant performance degradation.

- **Strong few-shot robustness**: With only 10% training data in later incremental periods, STBP achieves MAE of 13.58 on PEMS-Stream and 17.11 on CA-Stream (Table 2), vastly outperforming EAC (16.13, 20.94). This validates that the pattern bank captures transferable patterns from historical periods.

- **Efficiency with minimal overhead**: Figure 8 shows STBP achieves the best performance with only marginal additional computational cost and memory over EAC, validating that linear attention and frequency-domain processing keep the more expressive backbone practical.

## Weaknesses

### Fatal
None.

### Major

- **No direct evaluation of catastrophic forgetting mitigation**: The paper's central claim is that STBP mitigates catastrophic forgetting (abstract, contributions, Section 4.2). However, the evaluation protocol — model_τ evaluated on test_τ averaged over periods — measures per-period adaptation rather than knowledge retention. While this follows the community-standard protocol used by EAC and TrafficStream, the absence of any direct forgetting measurement (e.g., evaluating the final model on earlier periods' test data, backward transfer metrics, or a breakdown of performance on old vs. newly added nodes) means the improvement over baselines could partly reflect better adaptation rather than better forgetting mitigation. The comparison with the "Online" variant provides indirect evidence, but the core claim remains only partially substantiated. This matters because the paper frames forgetting mitigation as a primary contribution, not a secondary benefit.

- **Incomplete ablation study fails to isolate key claimed contributions**: The paper claims three specific contributions — FreNet for distributional drift, the prompt-based guidance mechanism (Eq. 5), and the three-group pattern bank structure (P^(0), P^(1), P^(2)) — yet none of these are individually ablated. Specifically: (1) There is no "w/o FreNet" variant — only "w/o DLGA" is tested, leaving FreNet's individual contribution unverified; (2) The prompt-based gating interaction (Eq. 5: H' = P^(1)·h(H·(1+P^(0)))) is a claimed contribution with no ablation testing what happens if the pattern bank is used via simple concatenation or additive injection instead; (3) The three-group pattern bank design is central to the framework but the number of groups is never varied. The existing ablations (Retrain, Online, w/o Backbone, w/o DLGA) are valuable but too coarse to attribute improvements to the specific mechanisms the paper claims as novel.

### Minor

- **Marginal improvement on AIR-Stream without discussion**: On AIR-Stream, STBP reduces MAE by only 2.35% over EAC (23.64±0.23 vs. 24.21±0.43), and the RMSE improvement is negligible (37.76 vs. 37.83). The MAE difference is approximately 1.2 standard deviations apart, raising questions about statistical significance. The paper presents this uniformly with the 21%+ traffic improvements without discussing the domain-dependent performance gap or its implications for the generality claim.

- **Misleading presentation of the linear attention approximation (Eq. 8–9)**: The paper presents Softmax(QK^T + QP^T)V (Eq. 8) and then states it "approximates" to φ(Q)(φ(K)^TV + φ(P)^TV) (Eq. 9). While the computation in Eq. 9 is valid (it is the standard linear attention framework of Katharopoulos et al. (2020) applied to concatenated keys [K; P]), the ≈ notation between Eq. 8 and Eq. 9 is misleading. Eq. 9 does not directly approximate Eq. 8; rather, both are approximations to full softmax attention with dual keys, using different methods. The presentation could confuse readers into believing the linear decomposition preserves softmax normalization properties. The paper references Appendix A.3.1 for derivation, which may clarify this, but the main text should be more precise.

- **FreNet's claim of extracting "stable low-frequency components" is overstated**: The learnable frequency embedding Fτ is not constrained to prefer low-frequency components — it could learn to emphasize any frequencies during training. The paper states Fτ "adaptively highlights stable features" (Eq. 6 description), which is more accurate, but elsewhere claims FreNet extracts "stable low-frequency components while suppressing high-frequency noise" (Section 4.3). Without analysis of what frequencies Fτ actually learns to emphasize, this stronger claim is unsupported.

### Trivial
None.

## Nice-to-Haves

- **Backward transfer analysis**: Report performance of the final model (after all incremental periods) on test data from earlier periods, or break down per-period performance by old vs. newly added nodes. This would directly validate the catastrophic forgetting claim and is now becoming expected in the CSTF community.

- **Frequency spectrum visualization of FreNet**: Show which frequency components Fτ learns to emphasize (weight magnitudes across frequency bins), which would directly support or refute the claim about extracting stable low-frequency components.

- **Ablation of the three-group pattern bank**: Test P^(0) only, P^(0)+P^(1), and P^(0)+P^(1)+P^(2) to justify the three-group design choice.

- **Statistical significance testing on AIR-Stream**: Given the small improvement margins, reporting whether differences are statistically significant would strengthen the generality claim.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **"The linear attention derivation in Eq. 8–9 is mathematically invalid as an approximation"** (Harsh Critic, labeled "Structural"): The critic claims the approximation is invalid because "the softmax of a sum cannot be decomposed into a sum of independent linear attention terms." This misreads what Eq. 9 does. Eq. 9 is the standard linear attention framework (Katharopoulos et al., 2020, cited by the paper) applied to concatenated keys [K; P], not a decomposition of softmax(QK^T + QP^T). The computation φ(Q)(φ(K)^TV + φ(P)^TV) is valid and achieves O(Nd) complexity. The issue is presentational (the ≈ between Eqs. 8 and 9 is misleading), not mathematical — downgraded to Minor above.

- **"The 'Online' variant matching EAC undermines the narrative that the pattern bank is essential"** (Harsh Critic): The paper already discusses this finding: "even without the contextual pattern bank on traffic datasets, the spatio-temporal backbone alone attains performance comparable to EAC under online training, highlighting the critical role of real-time dynamic correlation modeling and temporal distribution-drift mitigation" (Section 5.3). This is a reasonable interpretation — the backbone is strong, and the pattern bank adds further improvement. The paper doesn't claim the pattern bank alone is essential; it claims the combination is the contribution.

- **"t-SNE visualizations are not rigorous evidence without quantitative cluster-quality metrics"** (Harsh Critic): While true that t-SNE can produce clusters from many representations, this is standard qualitative analysis in the field. The t-SNE is supplementary evidence, not the primary support for any claim. Downgraded to trivial and removed.

- **"The pattern bank is functionally similar to STID's spatial identity embeddings, STAEformer's adaptive embeddings, and EAC's prompt pool"** (Harsh Critic): The paper explicitly cites and discusses these connections in Section 2 (Related Work) and Section 4.2, noting the distinctions: "Distinct from existing work (Wang et al., 2023a; Chen & Liang, 2025; Wang et al., 2023b), we introduce a Prompt-Based Guidance mechanism." Whether this distinction is sufficient is a matter of degree, not an omission.

- **"Conventional STGNN baselines retrained from scratch inflate the apparent advantage of all continual methods"** (Harsh Critic): The paper follows the evaluation protocol established by prior CSTF work (EAC, TrafficStream) and explicitly states this setup. This is the community-standard comparison, not an unfair advantage specific to this paper.

- **Strength removed: "The method requires no historical data replay"** (Strength Finder): While factually correct, this is a property shared by all prompt-based CSTF methods (EAC, STRAP) and is not a distinguishing strength of STBP specifically.

- **Strength removed: "Interpretability of the pattern bank through clustering analysis"** (Strength Finder): The t-SNE visualization provides only qualitative, uncontrolled evidence. Without quantitative cluster-quality metrics or correlation with known node attributes, this is weak evidence and conflicts with the verified minor weakness about overstated FreNet claims.

## Novel Insights

The most insightful observation across the reviews is the tension between STBP's two components: the ablation study reveals that the backbone alone (without the pattern bank) achieves performance comparable to EAC on traffic datasets. This finding — acknowledged but underexplored by the authors — suggests that the primary driver of STBP's large improvement may be the backbone's architectural design (FreNet + DLGA) rather than the pattern bank's continual learning mechanism. If true, the paper's core narrative (that the pattern bank is key to mitigating forgetting) needs recalibration: the backbone provides the bulk of the performance gain, while the pattern bank provides incremental but important adaptation capability. This does not diminish the contribution but reframes where the novelty primarily resides.

## Suggestions

- Add a backward transfer evaluation: train the model through all incremental periods, then evaluate the final model on test data from period 1. This single experiment would directly validate the catastrophic forgetting claim and significantly strengthen the paper.

- Complete the ablation study with three additional variants: (1) w/o FreNet (replace with standard temporal module), (2) w/o Prompt Guidance (use P_τ via simple addition or concatenation instead of the gating in Eq. 5), and (3) varying the number of pattern bank groups (1, 2, 3). These would isolate the contributions the paper claims but currently does not verify.

- Discuss the domain-dependent performance gap explicitly: acknowledge that STBP's advantage is much larger on traffic data than air quality data, and provide hypotheses for why (e.g., traffic data has stronger periodicity benefiting FreNet, or the pattern bank captures more meaningful heterogeneity in traffic networks).

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| SNIP (expanding-node STF with prompting) | /home/wg25r/review_agent/human_reviews_2026/z45L1eYoHE.md | 5.33 (Reject) | Similar topic (expanding-node STF), but STBP has much stronger empirical results (21%+ gains vs. modest) and more complete framework. STBP clearly above this. |
| StPR (spatiotemporal CL for video) | /home/wg25r/review_agent/human_reviews_2026/VAn2YVMuZC.md | 5.00 (Accept Poster) | Different domain (video CL) with decent results but limited ablations. STBP has stronger empirical gains. |
| SMoPE (prompt-based CL with MoE) | /home/wg25r/review_agent/human_reviews_2026/B8eIc9783S.md | 6.00 (Accept Poster) | Similar prompt-based CL approach with incomplete ablations, accepted at 6.0. STBP is comparable — similar ablation gaps but larger empirical improvements. |
| HippoTune (hippocampal-inspired CL) | /home/wg25r/review_agent/human_reviews_2026/MtDiLnnYgm.md | 6.50 (Accept Poster) | Stronger theoretical grounding (Krylov approximation analysis) with good empirical results. STBP lacks this theoretical depth but has larger empirical gains in its domain. |
| MMCKM (traffic flow with Koopman) | /home/wg25r/review_agent/human_reviews_2026/fhDqFk4DgI.md | 6.00 (Accept Poster) | Traffic forecasting with novel Koopman perspective, accepted at 6.0 despite minor weaknesses. STBP is comparable. |
| DPGNet (dynamic graph STF) | /home/wg25r/review_agent/human_reviews_2026/5FOYGNXRNc.md | 2.00 (Reject) | Limited novelty, inconsistent results. STBP is far above this. |
| FourierFlow (frequency-aware flow matching) | /home/wg25r/review_agent/human_reviews_2026/a3sRspQ62b.md | 2.50 (Reject) | Core methodology issues in turbulence modeling. STBP is far above this. |
| "Heads collapse, features stay" (CL theory) | /home/wg25r/review_agent/human_reviews_2026/IdW0d0mRnG.md | 7.33 (Accept Poster) | Strong theoretical contribution to understanding forgetting. STBP lacks this theoretical depth — below this anchor. |

STBP sits at approximately the 6.0 level — comparable to SMoPE and MMCKM (both accepted at 6.0), above SNIP (5.33, rejected), and below HippoTune (6.5) and the theoretical CL paper (7.33). The large empirical improvements on traffic datasets are a clear strength, but the incomplete ablation study and lack of direct forgetting measurement prevent scoring higher.

## Score and Decision

Score: 6.0

The paper makes a solid contribution to continual spatio-temporal forecasting with impressive empirical improvements on traffic datasets (21%+ MAE reduction) and a well-motivated framework design. The main weaknesses — incomplete ablation study and no direct forgetting measurement — are significant but do not invalidate the core contribution. The paper follows community-standard evaluation protocols and the indirect evidence (comparison with Online/Retrain variants) supports the forgetting mitigation claim. The paper is at the borderline of acceptance, leaning accept due to the strength and consistency of empirical results.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>