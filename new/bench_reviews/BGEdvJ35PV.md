Now I have a thorough understanding of the paper and the calibration anchors. Let me write the final review.

## Summary

The paper formalizes the "dense-concentrated structure" (DC-structure) of molecular data distributions — narrow, densely packed probability peaks separated by near-zero density regions — and shows this causes diffusion reverse steps to overshoot into invalid regions. Building on this analysis, the authors propose DIST, a plug-in method that, at an intermediate reverse timestep, generates multiple candidate trajectories, evaluates them via pilot inference, and filters out those inconsistent with the true distribution. DIST consistently improves stability and validity across three diverse backbone architectures (EDM, GeoLDM, RADM) on QM9 and GEOM-Drugs.

## Strengths

- **DC-structure formalization and overshoot analysis**: Definition 3.1 provides a precise probabilistic characterization of molecular distributions as mixtures of narrow, well-separated peaks, and Eq. 7 derives a concrete overshoot condition (β_t·Δ/σ_*² > cσ_*) showing that small σ_* (narrow peaks) causes reverse steps to overshoot. This goes beyond qualitative observation and offers a mechanistic explanation for why molecular diffusion is fragile — a genuine conceptual contribution.

- **Consistent, substantial empirical improvements across diverse backbones**: Table 2 demonstrates DIST improves every tested backbone (EDM, GeoLDM, RADM) on both QM9 and GEOM-Drugs, with especially large margins on the hardest metric — molecular stability (EDM: 82.0%→89.9%, GeoLDM: 89.4%→93.4% on QM9). The gains hold across GNN-based equivariant, latent-space, and Transformer-based non-equivariant models, confirming the DC-structure issue is architecture-agnostic.

- **Model-agnostic plug-in design requiring no retraining**: DIST uses only pre-trained model weights and a pilot inference/filtering step applied at an intermediate timestep (Sec. 3.2, Fig. 2), making it immediately applicable to existing models. The paper demonstrates this by applying DIST to three independently developed architectures using their officially released weights.

- **TV-contraction analysis linking intermediate and final distribution quality**: Corollary 3.1 establishes ‖q₀ − p₀‖_TV ≤ κ·‖qₜ − pₜ‖_TV, formally justifying that correcting intermediate distributions improves final sample quality. Table 1 provides a clean motivating experiment showing monotonic quality degradation as more reverse steps are taken.

## Weaknesses

### Fatal
None.

### Major

- **The method is fundamentally a selection mechanism, not a distributional correction — the "steering" language overclaims what DIST actually does.** The paper's theoretical diagnosis (Definition 3.1, Eqs. 6–7) identifies that DC-structure causes *inaccurate score estimation in overlap regions*, leading to overshoot. This naturally motivates fixing the score field (e.g., consistency regularization, trajectory averaging). Instead, DIST generates multiple candidate trajectories at an intermediate timestep, evaluates them via pilot inference, and discards the bad ones (Eq. 9, Sec. 3.2). This is selective filtering: the method does not modify the score field, the trajectory dynamics, or the intermediate distribution in any constructive sense — it *selects* from it. Proposition 3.1 bounds the error after selection, which is trivially true for any filtering scheme (removing samples far from the target distribution reduces discrepancy). The intermediate-timestep intervention is a genuine design choice that differentiates DIST from simple best-of-N at the end, but the paper's framing as "steering" and "corrective sampling" obscures this distinction. This matters because the theoretical framework promises a principled correction mechanism but the method delivers candidate selection, creating a disconnect between what is diagnosed and what is addressed.

- **No compute-matched baseline comparison — improvements may be attributable to brute-force compute rather than DIST's mechanism.** DIST inherently uses more total compute than single-sample generation: it generates multiple candidates, runs pilot inference on subsets, and filters. The baselines in Table 2 each generate a single sample per inference pass. A fair comparison would give baselines the same total compute budget and apply simple best-of-N selection (generate N samples with the baseline, keep the one with best validity/stability). If best-of-N at equivalent compute matches or exceeds DIST, the contribution of DIST's specific batching and pilot-scoring mechanism is nullified. This experiment is essential and absent. Without it, the improvements in Table 2 cannot be confidently attributed to DIST's proposed mechanism rather than to spending more compute on candidate generation.

### Minor

- **The pilot scoring function is underspecified in the main paper.** Section 3.2 lists multiple options for the pilot score s_j (e.g., "round-trip residual, self-consistency, ensemble variance, or chemistry-based penalty") but does not state which is used in the experiments. The choice of scoring function is the operational heart of DIST — if it is a chemistry-based validity check (e.g., valence satisfaction), then DIST is simply generating molecules and keeping the chemically valid ones, which is trivially effective. If it uses model-internal signals (e.g., ensemble variance), that is more interesting and non-trivial. The main paper should specify this explicitly, and an ablation over scoring function choices would strengthen the contribution. (The paper references Appendix F for implementation details and Appendix H for hyperparameter ablations.)

- **The efficiency claim is somewhat misleading due to the illustrative calculation omitting pilot inference cost.** The paper gives an example calculation of 307 amortized steps (Sec. 4.3: (1000−300)/100 + 300 = 307), which does not account for pilot inference overhead. The actual values in Table 3 (414–637 steps) are higher, suggesting the table does include some overhead. However, the "nearly half" characterization is only accurate at the lower end (414/1000 ≈ 41%); at the upper end (637/1000 ≈ 64%), the reduction is closer to one-third. The paper references Appendix G.1 for detailed cost quantification, but the main text's example is misleading.

- **Missing standard deviations for baselines and GEOM-Drugs results.** Standard deviations are only reported for DIST variants on QM9 (Table 2); baselines and all GEOM-Drugs results lack variance information, making statistical significance difficult to assess for most comparisons.

### Trivial
None.

## Nice-to-Haves

- A trajectory visualization showing samples being "rescued" at the intermediate timestep vs. simply rejected, to clarify whether any genuine trajectory correction occurs through subsequent denoising of the selected samples.
- Comparison with existing corrective methods (e.g., SMC-based approaches like Monte Carlo guided Denoising Diffusion) under matched compute.
- Ablation of the intermediate timestep choice t and perturbation radius r in the main paper (currently deferred to Appendix H).

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh Critic Claim: "The efficiency claim is misleading — total compute is not reported"** (as stated at full strength). While the illustrative 307-step calculation omits pilot cost, Table 3's caption states values are "computed from the total timestep consumption needed to generate 10,000 molecules," which does appear to include pilot overhead. The actual values (414–637) are higher than 307, consistent with including pilot cost. The real issue is the misleading example and "nearly half" overstatement, not a complete absence of total compute reporting. Downgraded to Minor.

- **Harsh Critic Claim: "Table 1 simply shows that shorter reverse processes produce fewer errors — trivial property."** While it's true that any diffusion model with imperfect scores will show this pattern, Table 1 specifically quantifies the rate of degradation on molecular data and directly motivates the intermediate-timestep correction. It serves its purpose as a motivating experiment even if the trend itself is expected. Removed as a weakness.

- **Harsh Critic Claim: "The condition in Eq. 7 should be treated as a heuristic rather than a precise criterion."** The paper itself treats it as an analytical observation deriving from Definition 3.1, not as a precise criterion for triggering overshoot. It provides a mechanistic explanation. This is a fair observation but not a weakness — the paper doesn't overclaim precision here.

- **Harsh Critic Claim about missing related works (DPS, DDRM, early-stopping + validity filtering).** Per the rules, I do not mention missing related works as I cannot confirm their existence or relevance.

- **Harsh Critic Claim about missing proofs/appendix.** The parser strips appendices; they exist in the original submission.

- **Strength Finder Claim: "Near-halving of inference cost while improving quality."** This strength conflicts with the verified Major/Minor weaknesses about the efficiency claim being misleading. The actual amortized cost reduction ranges from ~41% to ~64% of baseline, and the "nearly half" framing is overstated. Downgraded; kept as a qualified benefit but not as a standalone strength.

- **Harsh Critic Claim: "The analogy to images sets up a straw man — image diffusion models also face sharp distribution boundaries at object edges."** The paper's comparison is specifically about the concentration (narrow peaks) and denseness (tightly packed) of molecular distributions vs. the broader, more smoothly overlapping peaks of image distributions. The harsh critic's counterexample (object edges) does have sharp boundaries but the vast majority of image pixel space is smooth, whereas in molecular space most valid configurations are narrow peaks. This is not a straw man — the distinction is legitimate. Removed.

## Novel Insights

The core tension in this paper — between a principled theoretical diagnosis (inaccurate scores in overlap regions due to DC-structure) and a practical method that operates by selection rather than correction — reveals an interesting open problem: can one design an inference-time mechanism that genuinely *modifies* the score field or trajectory dynamics to address DC-structure-induced overshooting, rather than simply filtering out failed trajectories? The intermediate-timestep intervention point is a potentially valuable lever that DIST identifies but does not fully exploit; future work could combine the selection insight with actual trajectory correction (e.g., using the pilot diagnostic to adjust the score estimate or redirect trajectories at the intermediate step).

## Suggestions

- **Add a compute-matched best-of-N baseline.** Generate N molecules with each baseline using the same total compute budget as DIST, select the best by validity/stability, and compare. This is the single most important experiment for isolating DIST's contribution.

- **Specify and ablate the pilot scoring function.** State exactly which scoring function is used in each experiment. Ablate different choices (chemistry-based vs. model-internal) in the main paper to show whether DIST leverages chemical domain knowledge (trivial) or model-internal signals (non-trivial).

- **Clarify the efficiency metric.** Either report wall-clock time or break down the amortized timestep calculation to explicitly show pilot inference cost, rejected candidate cost, and accepted sample cost separately.

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| Monte Carlo guided Denoising Diffusion (nHESwXvxWK) | 8.5 | Principled SMC-based corrective sampling with strong theory-method connection. DIST is weaker because its method is selection-based rather than a genuine correction of the score/trajectory. |
| On Error Propagation of Diffusion Models (RtAct1E2zS) | 7.5 | Theoretical analysis of error propagation + regularization correction. Similar motivation but more direct theory-method connection. DIST's DC-structure analysis is arguably more novel, but the method-theory gap is larger. |
| SymDiff (i1NNCrRxdM) | 7.0 | Plug-in corrective mechanism (stochastic symmetrisation) for molecular diffusion. More principled correction mechanism; DIST has stronger empirical improvements but a weaker theory-method link. |
| Marginal Matching (kRjLBXWn1T) | 5.25 | Inference-time correction for flow models; overclaimed improvements. DIST has a more original theoretical contribution (DC-structure) and stronger empirical results. |
| C-Code / Controlled Denoising (MBDH5zyxHM) | 4.60 | Best-of-N + SDEdit; overclaimed efficiency, trivial combination. DIST is more principled and has a genuine theoretical contribution, but shares the concern about overclaimed efficiency and selection-vs-correction. |
| ACDC (Zp51wHvoot) | 4.25 | AR + diffusion correction; overclaimed, lacks novelty. DIST has a more original contribution. |
| Active Probabilistic Drug Discovery (mJgymwRsWw) | 2.50 | Iterative selection/filtering, no algorithmic innovation. DIST has a genuine theoretical framework and consistent empirical improvements. |

DIST sits above the medium-scoring inference-time correction papers (4.5–5.5) because it has a genuine and original theoretical contribution (DC-structure formalization + overshoot analysis) and consistent, substantial empirical improvements across diverse backbones. However, it falls below the high-scoring correction papers (7.0–8.5) because the method-theory disconnect (selection vs. correction), the missing compute-matched comparison, and the somewhat misleading efficiency claims prevent the paper from delivering on its theoretical promise. The paper's contribution is real but the gap between what the theory diagnoses and what the method addresses is significant.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>