Now let me run calibration searches to properly score this paper.Now I have all the information I need. Let me write the final review.

---

## Summary

SuMi addresses **multimodal wild TTA** — a test-time adaptation setting where the target distribution mixes weak OOD samples (one corrupted modality) and strong OOD samples (multiple corrupted or missing modalities). The paper proposes three components: (1) IQR-based sample smoothing for gradual easy-to-hard adaptation, (2) unimodal assistance that selects samples requiring multimodal fusion, and (3) mutual information sharing via cross-modal KL divergence alignment. Experiments on Kinetics50-C and VGGSound-C show clear gains over existing TTA methods and the multimodal-specific baseline READ, particularly in strong OOD scenarios.

---

## Strengths

- **Novel and practical problem formulation**: The paper formalizes *multimodal wild TTA* with a clear taxonomy of weak OOD (one corrupted modality) vs. strong OOD (multiple corrupted or missing modalities). Section 3.1 defines this rigorously, and Figure 1(b-c) demonstrates empirically that existing methods completely collapse under strong OOD while SuMi achieves ~20–45% accuracy.

- **Non-trivial empirical finding motivating unimodal assistance**: Figure 3(c) shows that for unimodal entropy, the [20,40) percentile band outperforms the [0,20) band, contrary to the standard low-entropy-is-best assumption. Table 6 validates this: Area 1 (low multimodal entropy, high unimodal entropy) achieves 39.4% vs. 32.1% for Area 3 (low multimodal, low unimodal), providing principled motivation for the dual-threshold design in Eq. 4.

- **Substantial improvements in the hard strong-OOD setting**: Table 2 shows SuMi at 33.4% vs. READ at 29.1% on Kinetics50-C strong OOD; Table 4 shows 19.7% vs. 14.5% on VGGSound-C. The gap is largest on the "Mix" scenario (18.4% vs. 13.7% on Kinetics50-C; 6.7% vs. 4.5% on VGGSound-C). Figure 5 shows SuMi degrades gracefully as strong OOD ratio increases while all baselines collapse.

- **Comprehensive evaluation**: Two datasets, 21 weak OOD corruption types, 4 strong OOD scenarios, 5 severity levels, mixed-ratio experiments (10 ratios), mixed-severity experiments, and 7 baselines — the experimental scope is adequate and supports the broad claims.

- **Figure 3(a) provides well-controlled motivation**: The controlled comparison (weak-to-strong adaptation ~28% vs. strong-only ~12%) cleanly motivates the IQR smoothing mechanism.

---

## Weaknesses

### Fatal
None.

### Major

- **Table 5 contains two unexplained rows with identical configurations but substantially different results.** Rows 7 and 8 both show IQR✓, UA✓, MIS✓ yet report very different numbers: 44.6 vs. 52.0 (Kinetics50-C sev.5), 54.3 vs. 59.3 (sev.3), etc. — a 7–8 point gap with no explanation in the text or table caption. Section 4.3 only states "IQR smoothing brings the most improvements" without acknowledging the duplication. One plausible interpretation is that row 7 uses MIS for all iterations while row 8 uses MIS only for t₀ iterations (as described in Section 3.4 for strong OOD), which would be an informative ablation of the t₀ design choice — but it is not labeled as such. As presented, the ablation table is unreliable: a reader cannot determine which configuration is the final model, and the unlabeled discrepancy raises questions about result selection.

- **Unexplained IQR+UA anti-synergy.** Table 5 shows IQR+UA (38.1/47.4 on Kinetics50-C sev.5/sev.3) performs *worse* than UA alone (45.1/52.1) across all six settings on both datasets. The paper describes IQR smoothing as "the most important component" and claims all three components work synergistically — yet the IQR+UA combination contradicts this claim. Interestingly, IQR+MIS (51.2/58.0) does perform well, suggesting IQR's benefit is conditioned on MIS being present. This interaction is neither discussed nor explained, and it undermines the modular interpretation of the ablation study.

### Minor

- **IQR mechanism's connection to "easy-to-hard scheduling" is qualitative only.** The core motivation is that IQR progressively selects weak-then-strong OOD samples (Figure 3(b), t-SNE visualization). However, the mechanism operates by selecting samples whose feature dimensions fall within the IQR of the batch representation — a within-batch feature outlier filter — not a direct ranking of samples by OOD severity. The t-SNE is supportive but entirely qualitative. A straightforward quantitative check (tracking what fraction of selected samples are weak vs. strong OOD across iterations) would directly validate the central claim. Without this, the mechanism-claim link remains undemonstrated.

- **Weak OOD improvements over READ are marginal.** On Kinetics50-C weak OOD (Table 1): SuMi 63.9% vs. READ 63.5% (+0.4). On VGGSound-C weak OOD (Table 3): 57.3% vs. 56.4% (+0.9). The paper claims "outperforms all baselines consistently and **significantly**" — this overstatement applies only to strong OOD. In the simpler weak OOD case (which READ was originally designed for), the improvement is marginal and not significant.

- **"Mutual information sharing" is a misnomer.** Eq. 6 implements a KL divergence between each unimodal distribution and a mixture of complementary unimodal and multimodal predictions — a cross-modal consistency regularizer. This is technically sound and functionally reasonable, but calling it "mutual information sharing" (which implies I(X;Y) = H(X) − H(X|Y)) will mislead readers. The design choice of including p^m in the KL target to guard against corrupted-modality contamination is well-motivated.

- **μ requires dataset-specific tuning.** Figure 7(a) shows μ has opposite effects on Kinetics50-C (video-dominant) vs. VGGSound-C (audio-dominant). The paper explains this as expected given modality dominance (Section 4.3), but acknowledges that the user must know which modality is dominant to tune μ — a practical limitation in real-world deployment where modality dominance may be unknown.

### Trivial

- The Eq. 8 formulation uses $\mathbb{1}_{\{x\in\mathcal{H}^t_\theta(x), x\in\mathcal{S}_\theta(x)\}}$ while the Figure 2 caption references a slightly different notation. This is a minor notation inconsistency, not a substantive error.

---

## Nice-to-Haves

- **Quantitative validation of IQR scheduling**: Track and report the fraction of weak vs. strong OOD samples selected at each iteration t. This would directly confirm whether the IQR filter achieves the intended progressive sampling behavior.
- **Ablation of t₀**: Report performance at t₀ ∈ {iter/4, iter/2, 3iter/4, iter} to establish the sensitivity of the MIS stopping criterion (which appears to be the distinction between the two unlabeled full-model rows in Table 5).
- **Extension to 3+ modalities**: Section 3.1 claims generality for M modalities, but all experiments use M=2. A brief experiment or discussion of three-modality settings would strengthen the generality claim.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Baseline comparison is structurally skewed"** — Partially removed as a major criticism. It is true that unimodal methods failing on missing-modality inputs is expected behavior. However, this is NOT a weakness unique to this paper — it is the *point* of the strong OOD setting: to show that purely unimodal methods cannot handle this scenario. The paper does compare to READ (the only genuinely applicable baseline for multimodal TTA), and READ is presented throughout as the primary reference point. This criticism is re-categorized as a **minor** concern about framing ("significantly outperforms" overstatement) rather than a structural flaw.

- **Harsh Critic: "IQR is computed within-sample, not across samples"** — Partially mitigated: the algorithm computes Q1 = quantile(h, 0.25) over the batch (Algorithm 1 takes a batch), so the IQR is batch-level, not within a single sample's feature dimensions. The β criterion for what fraction of dimensions satisfy Eq. 3 is admittedly unusual, but it is a reasonable proxy for sample-level outlier detection. The qualitative evidence in Figure 3(b) supports that it works in practice. This concern is kept but weakened to Minor.

- **Harsh Critic: "Section 4.1 benchmarks are not truly new"** — Removed. Applying standard corruption toolkits to new multi-modal combinations for a new benchmark setting is a standard and accepted practice, as READ itself did similarly for its benchmarks. The benchmark construction is described in the appendix (stripped from the extracted text), so criticizing its absence violates the appendix-stripping rule.

- **Harsh Critic: Eq. 3 vs. Figure 2 discrepancy** — The notation inconsistency between Eq. 8 and the Figure 2 caption is a minor presentation issue, not a methodological discrepancy. Moved to Trivial.

---

## Novel Insights

The paper's most genuinely novel empirical observation is the non-monotonicity of unimodal entropy as a selection criterion for multimodal TTA: samples with very low unimodal entropy actually *underperform* samples with moderately higher unimodal entropy (Figure 3(c), Table 6), because low unimodal entropy signals that the sample can be classified from a single modality and therefore carries little information for multimodal model adaptation. This contradicts the standard TTA assumption that lower entropy is universally better, and it has implications for any multi-modal TTA method that attempts to leverage unimodal information. The IQR smoothing intuition — that gradual adaptation from weak to strong OOD avoids catastrophic forgetting of source-domain knowledge — is well-motivated, though its quantitative grounding needs strengthening.

---

## Suggestions

1. **Explain Table 5 row duplication**: Clearly label the two full-model rows — if they represent "MIS for all iterations" vs. "MIS for first t₀ iterations," this is actually a useful ablation of the t₀ design choice and should be labeled as such.
2. **Explain IQR+UA anti-synergy**: Discuss why IQR+UA underperforms UA alone while IQR+MIS and IQR+UA+MIS (row 8) work well. This likely reflects an interaction between the sample filtering regimes.
3. **Quantify IQR scheduling**: Plot the fraction of weak vs. strong OOD samples selected as a function of iteration to replace the qualitative t-SNE with quantitative evidence.
4. **Reframe "significantly outperforms" for weak OOD**: Be more precise — the significant improvements are in strong OOD scenarios; the weak OOD improvements over READ are modest.
5. **Rename "mutual information sharing"**: Use a more accurate name (e.g., "cross-modal distribution alignment" or "inter-modal consistency regularization") to avoid misleading readers.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Comparison |
|------|----------------|------------|
| `/home/wg25r/review_agent/human_reviews/TPZRq4FALB.md` | 8.0 (Accept) | READ — the multimodal TTA baseline this paper builds on/competes against; stronger methodology clarity, but addresses a simpler problem |
| `/home/wg25r/review_agent/human_reviews/UhKkWHkvfg.md` | 5.0 (Reject) | Multimodal continual TTA; rejected for limited novelty and straightforward combination; this paper has more genuine novelty but similar ablation issues |
| `/home/wg25r/review_agent/human_reviews/BmG88rONaU.md` | 7.5 (Accept) | Cross-modal retrieval TTA; accepted for clean and well-validated methodology |
| `/home/wg25r/review_agent/human_reviews/9w3iw8wDuE.md` | 7.0 (Accept, Spotlight) | DeYO — a TTA baseline used in this paper; accepted for novel confidence metric with clean ablation |
| `/home/wg25r/review_agent/human_reviews/eXrUdcxfCw.md` | 4.8 (Reject) | Continual TTA with prototype-based adaptation; rejected for incremental contribution |
| `/home/wg25r/review_agent/human_reviews/1YSJW69CFQ.md` | 1.67 (Reject) | TTA paper with severe structural issues; much weaker than this paper |

**Assessment relative to anchors:**

SuMi sits clearly above the low-scoring anchor (1.67) and is stronger than MDAA (5.0) in terms of problem novelty and experimental scope. However, it falls short of READ (8.0), DeYO (7.0), and the cross-modal TTA paper (7.5) due to: (a) the unresolved Table 5 duplication that undermines the ablation, (b) the IQR+UA anti-synergy that is inconsistent with the paper's own claims about component contributions, and (c) the qualitative-only support for the central IQR mechanism. The strong OOD results are real and meaningful, but these methodological clarity issues push the paper below the acceptance cluster. The paper is positioned at ~5.0–5.5: closer to the MDAA boundary than to the accepted TTA papers, but with a clearer contribution and better experiments than MDAA.

**Final Score: 5.5**  
**Decision: Reject** (borderline — the core idea is promising and the empirical results in strong OOD are real, but the ablation is unreliable without explanation of the duplicate rows and the IQR+UA anti-synergy, and the central IQR mechanism lacks quantitative validation)

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>