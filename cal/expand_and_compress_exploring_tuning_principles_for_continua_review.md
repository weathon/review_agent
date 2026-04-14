=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
## Summary
EAC proposes a prompt tuning-based framework for continual spatio-temporal graph forecasting that addresses growing sensor networks. The core idea is to freeze a pre-trained STGNN backbone and maintain a dynamically expanding node-level prompt pool, guided by two empirically and theoretically motivated principles: **Expand** (heterogeneity-guided node-specific prompts) and **Compress** (low-rank factorization to control parameter inflation). The method is evaluated on three streaming datasets across multiple domains, and is tested across six STGNN backbone architectures.

---

## Strengths

- **Genuine universality across diverse STGNN backbones**: Table 3 demonstrates consistent improvement across all six architecture combinations (spatial/spectral × recurrent/convolution/attention), which is a concrete and non-trivial result — prior continual ST methods largely ignore backbone generalization.
- **Simultaneous accuracy and efficiency gains**: Figure 6 shows EAC achieves *both* lower MAE *and* faster training compared to baselines on both Energy-Stream and Air-Stream, rather than presenting a tradeoff. This is a substantive empirical result, not merely a claimed benefit.
- **Novel empirical characterization of prompt pool dynamics**: Figures 3 and 4 document, for the first time in this setting, how prompt dispersion grows monotonically during training and how the prompt pool develops a consistent low-rank spectral structure across seven streaming periods. This empirical characterization is genuinely informative for understanding why the approach works.
- **Principled solution to the growing-graph problem**: The design of expanding the prompt pool exclusively with new-node parameters while freezing the backbone elegantly sidesteps catastrophic forgetting by architectural isolation — a cleaner solution than regularization or replay strategies that require careful hyperparameter balancing.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Table 4 narrative directly contradicts reported numbers**: For the 12-step horizon, LoRA-Based achieves MAE = 10.22 and RMSE = 20.81, whereas EAC achieves MAE = 14.92 and RMSE = 24.17 — LoRA wins by ~46% on MAE and ~16% on RMSE. The paper's conclusion that "simply applying LoRA layers without considering the specific spatio-temporal context of streaming parameters may not be highly effective" is not supported by these figures. EAC's only advantages in Table 4 are MAPE (20.82 vs. 22.78) and training time (224s vs. 337s). A plausible explanation is that LoRA fine-tunes the backbone per-period and therefore achieves lower error on the *current* period while suffering catastrophic forgetting on earlier periods — which would be EAC's actual advantage. If so, the table must report per-period performance across all periods for LoRA (not just final-period or aggregate metrics from a single period) to demonstrate this. As currently presented, the Simplicity Study (RQ5) actively undermines the paper's central claim.

- **Missing ablation separating Expand from Compress**: There is no experiment for EAC with k = d (full-rank, no compression), which would isolate the Compress contribution. The hyperparameter study (Figure 7) varies k from 2 to 12 but never includes k = d as a "Expand-only" reference. Without this, it is impossible to quantify how much of the performance gain comes from heterogeneity-guided expansion versus low-rank compression, and whether the complexity of the Compress module is justified.

- **Theoretical propositions are valid but overclaimed**:
  - *Proposition 1* proves only that introducing node-specific prompt parameters increases average node deviation D(X^θ) — a variance inequality that holds for *any* node-specific perturbation, including random noise. The result does not establish that *learned* prompts improve forecasting, only that they increase feature dispersion. The critical step — connecting higher dispersion to reduced forecasting error — is missing. Moreover, the "invariance" condition required by the proof is never precisely defined, and which STGNNs satisfy it is not discussed.
  - *Proposition 2* applies standard results from random matrix theory (similar to Johnson-Lindenstrauss sketching) to show any matrix P ∈ ℝ^{n×d} admits a rank-k = O(log(min(n,d))) approximation with high probability. This holds for *any* matrix regardless of whether it comes from prompt learning. The paper does not demonstrate that the *learned* prompt pool has lower effective rank than a random matrix of the same dimensions; it only proves the existence of a low-rank approximation, which is not a novel theoretical contribution.

- **Absence of standard continual learning evaluation metrics**: The paper's core claim is that EAC mitigates catastrophic forgetting, yet no Forgetting Measure (FM) or Backward Transfer (BWT) is reported. Average metrics across all periods (Table 1) are insufficient to distinguish genuine anti-forgetting from simply better performance on the most recent period. Figure 5 provides per-year RMSE curves on PEMS-Stream in the few-shot regime, which is informative but not equivalent to standardized CL metrics and is limited to one dataset and one metric.

### Minor

- **Heterogeneity and low-rank empirical analysis is limited to PEMS-Stream**: Both Figure 3 (heterogeneity) and Figure 4 (low-rank spectral structure) use only PEMS-Stream data. Whether these properties hold for Air-Stream or Energy-Stream — which have different node counts, time horizons, and physical processes — is not shown, weakening the claim that the two tuning principles are universally applicable.

- **Fixed shared adjustment matrix B is structurally under-justified**: In the Compress implementation, B is trained in period 1 and frozen for all subsequent periods, defining a fixed subspace for all future node embeddings. The paper provides no analysis of whether this fixed subspace remains adequate as the network evolves over 7 years (PEMS-Stream). An ablation or analysis of reconstruction error in B over time would strengthen confidence in this design choice.

- **How the frozen backbone handles variable node counts is under-explained**: Section 4.1 mentions relying on "node-count-free graph convolution operators" but the methodology section does not explicitly detail how the frozen backbone processes variable input dimensions n_τ ≠ n_{τ-1} across periods, which is central to reproducibility.

- **STKEC reproduced without official code**: As acknowledged in footnote 2, the STKEC results in Table 1 are reproduced from scratch. Given that EAC improves over STKEC by only 0.42% average MAE on PEMS-Stream, reproduction errors could affect the conclusion.

### Tiny

- The formal objective in Eq. (1) minimizes only the current-period loss, while evaluation averages over all periods. A multi-period formulation explicitly penalizing performance degradation on previous periods would be more consistent with the paper's stated motivation and standard CL problem setups.
- The few-shot scenario study (Figure 5) reports only RMSE on PEMS-Stream. Extending this to Air-Stream and Energy-Stream, and adding MAE, would make the few-shot robustness claim more convincing.

---

## Nice-to-Haves

- Provide per-period performance breakdown for LoRA-Based across all periods (not just final-period or aggregate) to demonstrate whether its Table 4 advantage comes at the cost of catastrophic forgetting on early periods — this would clarify the paper's narrative.
- Report Forgetting Measure (FM) and Backward Transfer (BWT) to align with standard CL evaluation and make the anti-forgetting claim verifiable.
- Add an EAC w/o Compress variant (k = d) as an explicit ablation point to quantify each principle's independent contribution.
- Extend the heterogeneity (Figure 3) and low-rank (Figure 4) measurements to Air-Stream and Energy-Stream to validate that these are general properties, not PEMS-Stream-specific.
- Investigate whether B should be periodically updated (e.g., re-estimated every N periods) as the network evolves, to address the fixed-subspace concern.
- Analyze the effect of prompt initialization for new nodes (random vs. nearest-neighbor inheritance) on convergence and early-period performance.
- Report metrics disaggregated by stable vs. newly added nodes to directly verify that EAC benefits newly expanded sensors specifically.

---

## Removed Points
*These points are flagged for removal — treat them with caution.*

- **"Conflation of two separate problems" (forgetting vs. efficiency)**: The paper's framing is that both problems arise from updating all backbone parameters; freezing the backbone is the unified solution to both. This is a coherent design argument, not a logical conflation. Removed.
- **"Overstated critique of prior work"**: The paper's actual claim is specifically that prior methods "still involve optimizing the entire STGNN model" — which is factually correct for TrafficStream, PECPM, STKEC, and TFMoE. The word "principled" is debatable but does not rise to the level of a factual error. Removed.
- **"Expand-only baseline missing from introduction discussion"**: The paper does present Compress as a response to the parameter inflation problem caused by Expand; this is clearly motivated. The missing *experiment* is a legitimate weakness (captured above), but the *conceptual framing* is not deficient. Removed as a separate introduction criticism.
- **Node-only expansion assumption as a weakness**: The paper explicitly scopes its problem to expanding node sets (Definition 3.1: "the network dynamically grows"). Criticizing it for not handling edge rewiring or feature-dimensionality change is scope creep. Removed.
- **"Fixed backbone means poor period-1 training hurts all periods"**: This is a general limitation of any transfer learning approach and is not specific to this paper's weaknesses. Removed.
- **Non-parametric/TTT baseline missing**: Asking the paper to compare against a different continual learning paradigm (test-time adaptation) not in its stated scope. Removed.
- **Formatting/style nitpick (❶ vs ➊ enumeration)**: Pure style issue. Removed.
- **Algorithm 1 deferred to appendix**: Standard practice; not a methodological incompleteness. Removed.

---

## Novel Insights

The most distinctive insight contributed by the reviews (and partially by the paper itself) is the empirical discovery that learned prompt parameter matrices exhibit a *consistent and persistent* low-rank spectral structure across years of streaming data — the heatmap in Figure 4 (right) shows energy concentration >0.75 at rank 6 remaining stable across all 7 periods and training checkpoints. This is not a priori obvious: one might expect the prompt pool to fill higher-rank subspaces as more diverse nodes accumulate. The stability of this property empirically validates the Compress principle as reflecting an intrinsic constraint on the learned representation space, not merely an imposed approximation. Additionally, the paper's framing that GNN message-passing naturally suppresses heterogeneity (by aggregating neighbor information) while leaving correlation modeling intact provides a useful conceptual lens for understanding why node-specific trainable identifiers improve STGNN performance — an observation with implications for the broader static STGNN literature beyond the continual learning context.

---

## Suggestions

1. **Fix the Table 4 narrative immediately**: Either show per-period performance for LoRA-Based (to expose its forgetting) or reframe the comparison honestly as a speed-accuracy tradeoff in which EAC deliberately sacrifices some accuracy for efficiency and forgetting resistance. The current text ("simply applying LoRA may not be highly effective") is directly contradicted by the MAE and RMSE numbers and will be the first thing reviewers challenge.
2. **Add a single row "EAC w/o Compress (k=d)" to Table 1 or a new ablation table** to isolate the contribution of Compress. This is the most impactful missing experiment.
3. **Add FM/BWT metrics to Table 1** or provide a supplementary table with per-period performance for all methods, enabling verification of the anti-forgetting claim.
4. **Revise the theoretical framing of Propositions 1 and 2**: Present them as capacity arguments ("adding node-specific parameters provides the representational budget to model heterogeneity") rather than performance guarantees. Acknowledge that Proposition 2 applies standard matrix approximation theory and that the novel empirical contribution is showing this holds for learned prompt matrices specifically.
5. **Add a paragraph in Section 4.1 explaining, with a small diagram or equation, how the frozen backbone processes growing node sets** — specifically what "node-count-free graph convolution" means operationally when a new period's adjacency matrix A_τ has more rows/columns than A_{τ-1}.

---

**Evaluation summary:**
- *Novelty*: Moderate-to-good — applying prompt learning to the growing-graph continual ST-forecasting setting is genuinely novel; the Expand/Compress formulation is non-trivial.
- *Technical soundness*: Moderate — the method is coherent, but theoretical claims are overreached and Table 4 has a significant narrative problem.
- *Empirical support*: Moderate — three datasets and six backbones are tested, but missing CL metrics and the LoRA comparison issue reduce confidence in the core claims.
- *Significance*: Good — the streaming graph expansion problem is practically important and underserved.
- *Clarity*: Good overall, but the Simplicity Study discussion is confusing and the theoretical section overclaims, which together could be disqualifying without revision.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 3.0]
Average score: 6.8
Binary outcome: Accept
