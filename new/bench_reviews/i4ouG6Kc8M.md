## Summary

The paper investigates checkpoint selection for self-supervised learning (SSL) of vision transformers in histopathology, proposing a procedure that combines task-specific benchmark metrics with task-agnostic rank-based representation quality metrics to identify optimal training checkpoints. Across nine encoders spanning three model scales (21.6M–922.3M parameters) and varied magnification regimes, the authors demonstrate that (i) the best-performing checkpoints consistently occur well before training completion, (ii) rank-based metrics poorly correlate with non-linear segmentation performance, and (iii) small-scale models with proper checkpoint selection can match or exceed the segmentation performance of large foundation models trained on orders-of-magnitude more data.

## Strengths

- **Systematic empirical scope across model scales and architectures:** The paper trains and evaluates nine distinct encoders (ViT-S, ViT-B, SMoE-4/32/128) on a clinically relevant single-tissue type (LUAD), varying parameters from 21.6M to 922.3M and using datasets of 3.27M–10.25M images (Table 1). This is a comprehensive sweep for the histopathology SSL domain. Table 2 shows that across all nine models, the "Final" epoch checkpoint never exceeds the performance of the selected checkpoints on any task—often by a substantial margin (e.g., ViT-S SMoE-128 drops from 0.58 to 0.57 on MoNuSeg and 0.56 to 0.70 on BACH from all-round to final checkpoint).

- **Small models matching large foundation models with proper selection:** Table 2 demonstrates models trained on ~10M images achieve competitive or superior performance to Virchow (632M params, 2B images) and UNI (307M params, 100M) on segmentation benchmarks—e.g., ViT-S SMoE-32 achieves 0.60 on MoNuSeg versus Virchow2's 0.58, and multiple models outperform Virchow's 0.38 on PanNuke 20×. This validates the practical claim that proper checkpoint selection narrows the gap between smaller and larger models.

- **Useful negative result on rank-based metrics and segmentation:** Figures 2 and 3, combined with §5.1 analysis, correctly demonstrate that linear-rank proxies (RankMe, LiDAR, α-ReQ) correlate with classification but fail to correlate with non-linear segmentation performance (PanNuke AJI degrades after ~epoch 80 despite rank metrics continuing to improve). This clarifies an important limitation of these metrics in the medical imaging domain.

- **Validation on held-out downstream tasks:** Figure 4 evaluates selected checkpoints on LUAD subtyping and EGFR slide-level classification—tasks explicitly excluded from the selection procedure (§5.3). Earlier selected checkpoints consistently match or outperform later ones on these independent tasks, supporting the claim that training longer is detrimental to generalization.

## Weaknesses

### Major

- **The "dual-metric" framing is overstated—the final selection depends only on task-specific metrics.** Algorithm 1 Steps 5–6 rank candidate epochs using `r_k = Σ N_{s_k,j}^{ts}`, which sums only normalized task-specific metrics. The task-agnostic metrics influence which epochs enter the candidate set (Step 3: `C_{i,j} = argmax_e (N_{e,i}^{ts} + N_{e,j}^{ta})`) but are then discarded in the final ranking (Step 5). Thus the task-agnostic component functions merely as a filter, not as a dual criterion in the final decision. The paper's title, abstract, and §3 repeatedly emphasize a "dual-metric approach," but the algorithmic implementation reduces this to a task-specific selection procedure with a task-agnostic pre-filter. This undermines the core methodological claim.

- **Missing ablation against a task-specific-only baseline.** The paper does not compare the proposed procedure against standard early stopping using only task-specific benchmark validation scores. Without this ablation, it is impossible to quantify whether the task-agnostic component adds any value beyond selecting the maximum task-specific checkpoint, or whether the procedure simply reduces to benchmark-based early stopping. The marginal contribution of the rank-estimation component remains unverified.

### Minor

- **No statistical significance reporting on held-out evaluations.** Figure 4 reports slide-level AUC from ten train/test splits by concatenating predictions across splits (§5.3), providing point estimates without variance (standard deviation, confidence intervals). Given the small absolute differences in AUC (often <0.02 across checkpoint types), it is not possible to determine whether the observed trends are statistically meaningful or noise.

- **Non-stationary MinMax normalization.** The MinMax normalization in Algorithm 1 (Steps 1–2) is computed across all checkpoints for each metric. This means the normalized values depend on the total training length and checkpoint saving frequency—extending training by 50 additional epochs or changing checkpoint frequency would rescale the entire normalization space, altering the relative rankings without any change in actual representation quality. This makes the selection criterion sensitive to procedural hyperparameters rather than purely to representation quality.

### Trivial

- **Overstated contrast with natural image SSL.** The conclusion (§6, also §5.2, line 170) states that mid-training performance saturation is "in sharp contrast to observations from other data modalities, such as natural language and natural images." Performance degradation in later epochs on small, single-tissue datasets is more likely a domain-overfitting artifact than a fundamental modality difference. This framing inflates the significance of an expected observation.

## Nice-to-Haves

- A unified multi-panel learning curve plot showing training loss, task-agnostic metric evolution, and task-specific benchmark performance for all nine models with selected vs. final epochs marked. This would clarify whether the "mid-training peak" pattern is consistent across all configurations or primarily driven by the ViT-S 20× case.
- Aggregate Spearman's ρ correlation coefficients between each rank metric and each benchmark across all models and epochs, rather than only per-model scatter plots (Figure 3), to provide quantitative correlation evidence for the §5.1 claims.
- Clarification of whether benchmark *validation* or *test* splits were used for task-specific metric computation during selection, and whether any data overlap exists between selection benchmarks and held-out downstream tasks.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Claim that benchmark evaluation during SSL training "contradicts the fundamental use-case of SSL model selection" and constitutes data leakage.** SSL practitioners routinely evaluate on proxy benchmarks during training (e.g., linear probing on ImageNet-1k). The paper treats benchmarks as out-of-distribution proxy validation sets, which is standard SSL practice. If the benchmarks use consistent validation splits not reused in downstream tasks, this is not data leakage. The paper does not suggest test splits were used for selection.

- **Complaint about missing related work on proxy validation sets and stopping criteria.** Per hard rules, missing related work is not a valid weakness.

- **Normalization sensitivity criticism elevated to structural severity.** While the non-stationary normalization is a valid minor concern about algorithmic design, moving it to major overstates the issue—the normalization is applied consistently within each training run for intra-run comparison, and the paper's empirical findings hold across varied configurations (Table 2).

- **Formatting/style nitpicks, grammar issues, and presentation complaints.** Per hard rules, these are parser artifacts, not author errors.

- **Criticisms questioning the contribution because small models are "only" on a single tissue type.** The paper explicitly scopes itself to LUAD in §1.3 ("only one tissue type is chosen"), which is a stated design choice, not a flaw.

## Novel Insights

The paper's most genuinely novel contribution is its systematic empirical demonstration that self-supervised learning for histopathology data peaks mid-training and that continued optimization beyond that point actively harms downstream performance—a finding that challenges the prevailing assumption (borrowed from natural image literature) that SSL benefits monotonically from longer training. The simultaneous observation that rank-based representation quality metrics decouple from non-linear task performance (segmentation) while remaining predictive for classification tasks provides a useful boundary condition for when these increasingly popular metrics are applicable. However, the methodological contribution (the selection procedure itself) does not rise to the level of novelty suggested by the "dual-metric" framing; it is essentially benchmark-based early stopping with a task-agnostic pre-filter.

## Suggestions

- Reconsider the "dual-metric" framing: either (a) revise the final selection step (Step 6) to include a weighted combination of task-specific and task-agnostic scores, making the dual nature genuine; or (b) reframe the paper as a benchmark-based checkpoint selection method and downplay the role of task-agnostic metrics. If the latter, the task-agnostic analysis can be positioned as a separate empirical contribution rather than part of the method.
- Add a direct ablation comparing the proposed procedure against selecting the single epoch with the highest aggregated task-specific score (without the task-agnostic filter). This would directly quantify the marginal contribution of the rank-based component.
- Report mean ± standard deviation (or confidence intervals) for the held-out AUC results in Figure 4, rather than only concatenated point estimates.
- Clarify which benchmark splits (train/val/test) were used for task-specific metric computation and confirm no data overlap with downstream held-out tasks.

## Calibration

I positioned this paper against several calibration anchors:

- **High-scoring anchor (8,8,8,8 — PdaPky8MUn):** Papers with clean methodology, strong theoretical or conceptual contributions, and experiments that directly validate claims score 8. This paper's empirical work is thorough but lacks the methodological novelty and tight claim-experiment alignment of this anchor.

- **Mid-scoring accepted anchor (6,6,6,6 — fszrlQ2DuP):** Transfer Score is a practical, well-evaluated method with clear scope. This paper's empirical breadth is comparable but the algorithmic framing is weaker (the "dual-metric" claim is undermined by the algorithm structure).

- **Mid-scoring rejected anchor (5,3,5,6,3 — aefNwingnS):** Extensive experiments but limited technical novelty, with the best-performing method being a simple combination of existing components. This paper sits slightly above this anchor because its empirical findings (mid-training peak is a genuine, previously undocumented pattern in histopathology SSL) are more cohesive and useful.

- **Low-scoring anchor (3,3,3,3 — V9UsZBbTvZ):** Methodological paper with limited novelty, poor comparison design, and missing baselines. This paper is substantially above this anchor—its empirical scope is much stronger and the core observation is well-supported.

The paper under review is a solid empirical study with genuinely useful findings for the computational pathology community, but its methodological framing is overstated (the "dual-metric" claim does not match the algorithm) and critical ablations are missing. It sits between the mid-accepted and mid-rejected anchors, leaning toward a borderline-accept. The empirical contributions carry meaningful value despite the algorithmic shortcomings.

## Score and Decision

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>