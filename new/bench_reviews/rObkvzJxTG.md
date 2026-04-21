Now I have all the information needed. Let me write the final consolidated review.

## Summary

The paper introduces "multimodal wild TTA," a test-time adaptation setting where the target domain contains a mixture of weak OOD samples (single modality corrupted) and strong OOD samples (multiple modalities corrupted or missing). To address this, the authors propose SuMi, combining three components: IQR-based smoothing for gradual sample inclusion, unimodal assistance for selecting informative samples, and a cross-modal prediction alignment loss (termed "mutual information sharing"). Two new benchmarks (Kinetics50-C and VGGSound-C) are constructed for evaluation. The method achieves meaningful improvements over the only existing multimodal TTA baseline (READ), particularly on the hardest strong OOD scenarios.

## Strengths

- **Valuable problem formulation**: The distinction between weak and strong OOD in multimodal data, and the "wild" setting where both appear simultaneously (Section 3.1), captures a real and underexplored challenge that existing multimodal TTA methods like READ do not address. Figure 1 compellingly demonstrates the failure of existing methods.

- **Insightful unimodal assistance design (Section 3.2.2)**: The counter-intuitive finding (Figure 3(c)) that very low unimodal entropy correlates with *worse* multimodal performance—because samples with very low unimodal entropy are uninformative for multimodal optimization—is non-obvious and well-supported empirically. This dual-threshold sample selection is a genuinely useful insight for multimodal TTA.

- **Meaningful improvements on the hardest scenarios**: On the most challenging Mix setting where one modality is missing and the other is corrupted, SuMi substantially outperforms READ (Table 2: 18.4 vs 13.7 on Kinetics50-C; Table 4: 6.7 vs 4.5 on VGGSound-C). Where other methods catastrophically collapse, SuMi maintains non-trivial performance.

- **New benchmark construction**: Kinetics50-C and VGGSound-C with 21 weak OOD corruption types and 4 strong OOD types (including missing modality scenarios) provide structured testbeds for future work, filling a gap in multimodal TTA evaluation.

- **Robustness under varying mix ratios**: Figure 5 shows that as the ratio of strong OOD samples increases from 0% to 100%, SuMi degrades much more gracefully than all baselines, directly validating the core design motivation.

## Weaknesses

### Fatal
None.

### Major

- **Ablation study directly contradicts the claimed importance of IQR smoothing**: Section 4.3 states "IQR smoothing brings the most improvements to the model." Table 5 shows the opposite: IQR alone is the *worst* individual component on Kinetics50-C severity 5 (31.7 vs UA 45.1 vs MIS 39.4). More critically, IQR+UA (38.1) performs *substantially below* UA alone (45.1), meaning IQR actively hurts when combined with unimodal assistance—a negative interaction the paper never acknowledges. The claim is only defensible if interpreted as "IQR brings the most improvement when added to MIS specifically" (IQR+MIS: 51.2 vs MIS alone: 39.4), but this conditional benefit is a fundamentally different claim. This misattribution shapes how readers understand the method's operating mechanism.

- **"Mutual information sharing" is not mutual information**: Equation 6 defines a KL divergence between unimodal predictions and a mixture of complementary unimodal and multimodal predictions. This is a cross-modal prediction alignment objective—it bears no relation to the information-theoretic definition I(X;Y) = Σ p(x,y) log(p(x,y)/p(x)p(y)). The naming creates a false impression of principled theoretical grounding where none exists. A more honest name (e.g., "cross-modal prediction alignment") would allow the contribution to be assessed on its actual merits.

- **Unexplained duplicate rows in ablation table**: Table 5 contains two rows (rows 7 and 8) with identical component checkmarks (✓✓✓) but vastly different results on Kinetics50-C severity 5 (44.6 vs 52.0—a 7.4 point gap). No explanation is provided. This is not a minor presentation issue; it suggests an important design choice or hyperparameter not captured by the ablation, leaving readers unable to understand what drives the best-performing configuration.

### Minor

- **f(t) = t/iter requires knowledge of total iterations (Section 3.2.1)**: In true streaming TTA, this quantity is unknown. While the paper operates in a fixed test-set setting where this is feasible, it limits applicability to the streaming regime, which is the more practical TTA setting. A data-dependent schedule would strengthen the method's generality.

- **Missing modality handling is not specified in the methodology**: The paper defines Vmiss and Amiss as core strong OOD scenarios (Tables 2, 4) but never explicitly states what input is given to the encoder of the missing modality. This is critical for reproducibility in the setting that constitutes a key contribution. (This detail may be in the appendix, but it should appear in the methodology.)

- **t₀ = iter/2 heuristic for disabling MIS (Section 3.4, Algorithm 1 line 11)**: The paper acknowledges that "strong OOD samples could damage the information sharing," which is why MIS is only applied for the first half of iterations during strong OOD adaptation. This is a significant scope limitation on the MIS component—it essentially admits MIS fails on the hardest samples—and the specific choice of iter/2 is unexplained.

- **VGGSound-C improvements over READ are modest on some corruption types**: On Table 4 (VGGSound-C), SuMi (27.9) underperforms EATA (28.8) on Crowd corruption. The overall strong OOD average improvement (READ 14.5 → SuMi 19.7) is meaningful, but the "consistent and significant" superiority claim in Section 4.2 is somewhat overstated.

### Trivial

- **Figure 2 caption incorrectly states Q1 = ¼ IQR and Q3 = ¾ IQR**: This contradicts Definition 1 in the paper, which correctly defines Q1 as the 25th percentile and Q3 as the 75th percentile.

## Nice-to-Haves

- A data-dependent smoothing schedule replacing f(t) = t/iter would make the method applicable to true streaming settings.
- Investigating and explaining the IQR+UA negative interaction (38.1 < 45.1) would clarify the method's operating regime and could lead to a better design.
- A no-component baseline (plain entropy minimization) in Table 5 would establish the absolute contribution of the full method over naive TTA.
- Evaluation on a real-world distribution shift (beyond synthetic corruptions) would test whether the method generalizes beyond algorithmically corrupted benchmarks.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Unfair baseline comparison because unimodal methods are included**: Including unimodal TTA methods (Tent, EATA, SAR, etc.) is standard practice and serves to establish motivation—showing that existing methods fail on multimodal data is part of the paper's argument. READ is the appropriate multimodal baseline and the paper does compare against it. The improvements over READ are the relevant comparison, and they are reported.

- **"Wild TTA" naming is incremental**: The critic argues that "wild TTA" was introduced by Niu et al. (2023) for unimodal settings. However, the paper itself acknowledges this (line 72: "known as wild TTA... Niu et al. (2023)") and extends the concept to multimodal data, which is a natural but valid extension that introduces new challenges (strong OOD with missing modalities). This is not a hidden incrementality.

- **Circular notation in Equation 8**: While the notation 𝕀_{x ∈ H_θ^t(x)} is indeed confusing because H_θ^t is defined as a set of representation vectors h, the intended meaning is clear from context—the indicator checks whether the representation of x satisfies the IQR bounds. This is a minor notation imprecision, not a logical error.

- **t-SNE visualization is unreliable evidence**: While t-SNE can be sensitive to hyperparameters, Figure 3(b) provides supplementary (not sole) evidence for the IQR smoothing mechanism. The primary evidence is the "Weak to Strong" vs "Strong" adaptation comparison in Figure 3(a).

- **Reproducibility concerns about undisclosed hyperparameters**: The paper provides specific hyperparameter values (γ_m = 0.4×ln(C), γ_u = e⁻¹, β = 0.6/0.9, λ = 5.0, μ = 1.0) and implementation details (Adam optimizer, learning rates, batch sizes). This is adequate for the field.

- **Missing real-world distribution shift evaluation**: This is a nice-to-have, not a core flaw. The paper's contribution is addressing a specific class of distribution shifts (synthetic corruptions simulating real-world noise), which is standard in the TTA literature.

## Novel Insights

The most striking finding that emerges from careful reading of the ablation is that SuMi's performance is driven primarily by the MIS component and its synergy with IQR, not by IQR alone. The IQR+MIS combination (51.2 on Kinetics50-C severity 5) outperforms the full three-component model (row 7: 44.6), suggesting that unimodal assistance, while beneficial in isolation, creates interference when combined with IQR filtering. This points to a fundamental tension: IQR selects samples near the distribution center (initially weak OOD), while UA selects samples with specific entropy profiles—these two selection criteria may be working at cross-purposes. Understanding and resolving this tension could lead to a more effective method than the current three-component combination.

## Suggestions

- Rewrite the ablation discussion to honestly reflect the data: acknowledge that IQR alone is the weakest component, that IQR+UA shows negative interaction, and that the key driver is the IQR+MIS synergy. This would make the paper more credible, not less.
- Explain the two ✓✓✓ rows in Table 5—readers cannot interpret the ablation without knowing what differentiates them.
- Rename "mutual information sharing" to "cross-modal prediction alignment" or similar to avoid creating a false theoretical impression.
- Add a row to Table 5 with no components (plain entropy minimization) to anchor the ablation.

## Score and Decision

**Calibration anchors:**
- **High band (>7)**: READ (avg 8.0, Accept poster) — same topic, cleaner methodology and ablation. SuMi extends READ's setting but has significant presentation issues. DeYO (avg 7.0, Accept spotlight) — TTA method with missing baselines but genuine contribution and honest ablation.
- **Medium band (4-6)**: TR-TTT (avg 5.0, Withdrawn) — contradictory ablation where combining modules hurts. Self-TPT (avg 5.0, Reject) — problematic ablation, overclaimed results. MDAA (avg 5.0, Reject) — incomplete ablation. DART (avg 5.67, Reject) — misleading method naming. SuMi has more genuine contributions than these (new problem formulation, benchmarks, meaningful hard-scenario improvements) but shares the ablation contradiction and naming issues.
- **Low band (<3)**: Active TTP (avg 2.5, Withdrawn) — marginal novelty, unconvincing ablation. SuMi clearly exceeds this tier with its problem formulation and benchmark contributions.

SuMi is stronger than the rejected medium-band papers because it has genuine contributions (novel problem setting, benchmarks, real improvements on the hardest scenarios). However, it is weaker than the accepted papers because the ablation contradicts the narrative, the MIS naming is misleading, and the two unexplained ✓✓✓ rows undermine confidence in the ablation. The improvements over the fair baseline READ are meaningful but moderate on average (29.1→33.4 and 14.5→19.7), with the most significant gains on the hardest Mix setting.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>