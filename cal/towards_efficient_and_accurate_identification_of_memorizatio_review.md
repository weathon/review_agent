=== CALIBRATION EXAMPLE 56 ===

# Final Consolidated Review
## Summary
This paper proposes pLOO_improved, a method to identify memorized data points in deep neural network training more efficiently and accurately than the standard pseudo-Leave-One-Out (pLOO) approach. The central idea is a simple Accuracy-per-Batch (ApB) proxy — counting the fraction of training batches in which a point is correctly classified — which is used to pre-filter the dataset to only the likely-memorized points before running a reduced pLOO procedure. The claimed outcomes are a >90% reduction in shadow models required (from ~2,000 to ~200) and a >65% reduction in RMSE against the LOO gold standard.

---

## Strengths

- **Targeted diagnosis of pLOO's inefficiency.** The paper precisely identifies *why* pLOO is wasteful: it runs on every point even though most are generalized. This diagnosis is crisp and leads directly to the proposed fix rather than being a motivational preamble.

- **Strong, consistent correlation of the ApB proxy.** The ApB proxy achieves Pearson correlation < −0.95 across five model-dataset combinations (VGG19/ResNet18 × CIFAR-10/CIFAR-100/TinyImageNet, plus MobileNet in the appendix). The breadth of architectures and datasets covered here is a genuine empirical strength, not a generic one — the consistency of the effect across this model zoo rules out architecture-specific artifacts.

- **Concrete documentation that pLOO overestimates memorization.** The paper provides rare empirical evidence (RMSE of 35.5 against LOO) that a widely-used method systematically misbehaves, which is a substantive finding for the community regardless of pLOO_improved's own performance. This erodes unfounded reliance on pLOO in downstream privacy literature.

- **Algorithmic simplicity of ApB.** Algorithm 1 requires only 4 lines of additional code in a standard training loop. This ease of adoption is a genuine distinguishing feature compared to more complex alternatives.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Selection bias in the LOO ground-truth comparison undermines the accuracy claim.** The entire "65% RMSE reduction" claim rests on 160 points explicitly chosen because they exhibit "the largest difference in memorization scores between the original pLOO and pLOO_improved" (Section 5.2). This is a biased sample by design: these are exactly the points where the two methods disagree most strongly, guaranteeing that at least one method is highly wrong on each point. Evaluating only on contentious points and showing pLOO_improved wins is not a fair comparison; it inflates the apparent advantage. A properly randomized sample — or ideally the full dataset — evaluated against LOO on the same architecture is required to make the RMSE claim credible. This is the most critical empirical flaw in the paper.

- **Proxy correlation is measured against pLOO, not against LOO.** Figure 3 and all associated Pearson scores report the correlation between ApB and *pLOO memorization scores*, not true LOO scores. The paper itself shows pLOO has an RMSE of 35.5 against LOO. A proxy that is strongly correlated with a biased estimator does not necessarily correlate well with the ground truth. The actual correlation of ApB with LOO scores is never reported, making the proxy's reliability for identifying truly memorized points uncertain.

- **Inconsistency between 50-run proxy validation and single-model deployment.** Section 4.3 explicitly states "we repeat this process 50 times to ascertain whether we get consistent results," and Figure 3's correlations are computed over these 50 runs. Yet Section 5.2 states that "we train a single model to extract the ApB scores." If the 50-run protocol is needed to establish consistency, it is unclear whether the single-model proxy degrades materially — and no variance analysis or ablation is provided to show the single-run proxy is equally reliable.

- **Different sampling ratios (r = 0.7 vs. r = 0.5) between pLOO and pLOO_improved are an uncontrolled confound.** pLOO uses r = 0.7 (Section 3.3) while pLOO_improved uses r = 0.5 (Section 5.2). The sampling ratio affects both the accuracy of the memorization scores and how many shards are needed. Since the paper's central claim is that restricting the population — not the sampling ratio — is the source of improvement, the different r values introduce a confound that should be ablated away.

### Minor

- **LOO comparison confined to one small, non-representative architecture.** Part 2 of the evaluation (the LOO comparison) is performed only on VGG-6 trained on CIFAR-10, while the main experiments use VGG19, ResNet18, and MobileNet. The paper justifies this with speed considerations and claims pLOO is "model-independent," but the actual magnitude of the RMSE reduction (35.5 → 12.19) may be architecture- and dataset-specific. Even one additional LOO data point on a larger model would substantially strengthen the generalizability claim.

- **No sensitivity analysis for the fixed 5,000-point threshold.** The number of "memorized" points selected for pLOO_improved is hardcoded at 5,000 (footnote 1), yet Figure 1 shows the actual memorization tail varies substantially across datasets and architectures. No ablation studies show how RMSE or compute savings change as a function of this threshold. Given that the efficiency/accuracy trade-off is the paper's main contribution, characterizing this trade-off curve is essential.

- **The "90% compute reduction" refers only to shadow model count, not wall-clock time.** The ApB proxy requires per-sample inference at each batch during the training run, which is a non-trivial overhead not accounted for. The paper should provide end-to-end wall-clock comparisons including this cost, since the overhead may matter in practice, especially for large-batch training.

- **150 vs. 160 point count inconsistency.** Section 5.2 says "we run it over 150 points" while Section 5.3 discusses "the total 160 points." This inconsistency appears throughout the results discussion and should be corrected.

### Tiny

- The RMSE scale (35.5, 12.19) is not explicitly stated to be in percentage points of the memorization score, which lives in [0, 100]. Brief clarification would help readers interpret these values.
- The claim about train-test gap driving memorization differences across number-of-classes settings (Section 4.1) is mentioned but not empirically supported within the paper.

---

## Nice-to-Haves

- Measuring the Pearson correlation of ApB directly against LOO scores (even for the 160 VGG-6 points already evaluated) would directly demonstrate whether the proxy captures true memorization or merely pLOO's biases.
- Robustness checks of ApB under different batch sizes, optimizers, and learning rate schedules would clarify whether the proxy is sensitive to standard training hyperparameter choices.
- A threshold-sensitivity curve (RMSE vs. number of top-K points selected) would help practitioners choose the right operating point for their compute budget.
- An ablation on the number of shadow models (e.g., 50, 100, 200) would clarify the lower bound on compute savings while maintaining accuracy.
- Extending the ApB proxy evaluation to generative or language models would broaden the paper's relevance, though this is outside the current scope.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Core observation (memorized points learned late) is not novel."** The paper explicitly scopes its contribution as validating this hypothesis for *natural* memorization as distinct from artificial (noisy-label) memorization (Section 2), citing Aerni et al. (2024) on the divergence between the two. Criticizing the paper for not being the first to observe training dynamics effects on noisy-label settings misreads the contribution. REMOVED.
- **"No comparison against forgetting events / EL2N as proxy baselines."** Per review policy, no missing related work comparisons are flagged since external sources cannot be confirmed. REMOVED.
- **"Claim that pLOO_improved can improve membership inference attacks is speculative."** The paper explicitly frames this as a belief and future direction (Section 6), not an experimental claim. Flagging speculation in a discussion section as a weakness is not appropriate. REMOVED.
- **"pLOO overestimates because it drops too many points — unfair comparison."** The comparison between pLOO and pLOO_improved against LOO is intentionally asymmetric to demonstrate the benefit of the proposed method, and the asymmetry disfavors pLOO_improved (it uses fewer models). The structural reason for pLOO's overestimation is the paper's own analysis and is presented as a genuine finding, not a strawman. REMOVED.
- **Formatting and style nitpicks** (minor labeling of axes, venue tags). REMOVED per policy.

---

## Novel Insights

The observation that pLOO *systematically overestimates* memorization scores relative to the true LOO baseline — by a large margin (RMSE ~35.5 on a 0–100 scale) — is arguably the most underappreciated finding in the paper. If correct even on the limited VGG-6/CIFAR-10 setting, it implies that a significant body of privacy literature that relies on pLOO scores may be drawing conclusions based on inflated memorization estimates. The proposed mechanism (large per-shard point removal distorts the memorization score upward relative to the true per-point removal) is intuitive and could inform how future approximation methods are designed — specifically, that proximity to the LOO procedure's shard composition matters more than the number of shards trained. This is an insight that reaches beyond the paper's own method and applies to the broader literature.

---

## Suggestions

1. **Replace the biased LOO sample with a random sample.** Draw 160 points uniformly at random from the full dataset (or from the top 5,000 scored memorized points), compute LOO for both pLOO and pLOO_improved on this random sample, and report RMSE. This single change would dramatically strengthen the accuracy claim.

2. **Report ApB-vs-LOO correlation.** Using the same 160 VGG-6 points that already have LOO scores, compute the Pearson correlation between ApB and LOO directly. This is a low-cost analysis that directly validates the proxy.

3. **Ablate the r confound.** Run the pLOO baseline at r = 0.5 (to match pLOO_improved) and also run pLOO_improved at r = 0.7 to isolate the contribution of the search-space restriction from the sampling-ratio change.

4. **Add a threshold sensitivity plot.** Show RMSE (vs. LOO) and compute cost as a function of K (number of top-ApB points selected), ranging from K=1,000 to K=10,000. This characterizes the efficiency-accuracy trade-off and gives practitioners actionable guidance.

5. **Clarify the 50-run vs. single-run discrepancy.** Either (a) report ApB variance across single runs to justify the single-model approximation, or (b) clarify that the method uses 50 runs and account for that cost in the efficiency comparison.

---

**Overall evaluation:** The paper targets an important practical problem and the ApB proxy idea is clean and easy to implement. However, the paper's primary accuracy claim — that pLOO_improved reduces error against the LOO baseline by 65% — rests on a small, deliberately cherry-picked sample from a single small architecture, which is not sufficient for ICLR standards. The proxy's correlation against the ground truth is also never directly measured. These are correctable issues, but in their current state they undermine the core empirical contribution. The efficiency claim (90% fewer shadow models) is more solid but also incomplete without wall-clock accounting. The paper is closer to a workshop contribution in its current form, though the underlying ideas are promising and the critical finding about pLOO's overestimation behavior has real value for the community.

- **Novelty:** Moderate — the proxy concept is new and practically motivated, but the core hypothesis (memorized points learned later) extends known results rather than introduces them.
- **Technical soundness:** Weak — the key experimental design for the accuracy claim has a significant selection-bias flaw and an uncontrolled confound (sampling ratio).
- **Empirical support:** Insufficient — broad proxy correlation results are a genuine strength, but the ground-truth LOO comparison is too narrow and biased to support the headline claim.
- **Significance:** Potentially meaningful, especially the finding that pLOO overestimates memorization, but contingent on stronger validation.
- **Clarity:** Generally good; the inconsistencies (150 vs. 160 points, 50-run vs. single-run) need to be resolved.

# Actual Human Scores
Individual reviewer scores: [6.0, 3.0, 5.0, 6.0]
Average score: 5.0
Binary outcome: Reject
