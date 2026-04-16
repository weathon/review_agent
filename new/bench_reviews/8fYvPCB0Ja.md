Now I have enough context. Let me verify the key claims about the paper's content to validate the harsh critic's points.

The harsh critic raises several points. Let me check each one against the paper:

1. **Fairness metric under-specified**: The paper defines DEO_M and DEO_A in Section 2, and the formula uses `P(Ŷ=y | Y=y, A=a_i)` which is clearly about TA predictions conditioned on true TA and PA. This is the standard EO definition. The critic's concern about ambiguity is partially valid but overstated - the definition is clear.

2. **Theoretical guarantees weak**: The critic argues that Theorems 4.1 and 4.2 don't guarantee fairness (EO) or accuracy in practice. This is a valid point - Theorem 4.1 shows that the feature mean of S is at the unweighted average of group centers, but this doesn't directly imply EO. Theorem 4.2 shows L_FairDD upper bounds L_vanilla for convex D, meaning minimizing L_FairDD also minimizes L_vanilla, which is about feature matching coverage, not about downstream classification accuracy or fairness guarantees. The paper overclaims with phrases like "guarantee fairness and accuracy."

3. **Full PA labels needed**: The paper acknowledges this in Section 7, but doesn't experiment with noisy/incomplete PA. Valid concern.

4. **Accuracy improvements over vanilla DD**: This is a very important point. Looking at Table 2, DM on C-MNIST (FG) at IPC=10 gets 25.01% while FairDD+DM gets 94.61%. This is suspicious - it suggests the vanilla DD baselines are catastrophically failing. The critic suggests the baselines may be misconfigured. However, looking more carefully, these are intentionally *biased* datasets with BR=0.9, meaning vanilla DD could legitimately perform poorly because the model learns to predict PA instead of TA. This would not be a baseline misconfiguration - it could be a genuine effect of the bias on the distillation process. So the critic's concern about misconfigured baselines needs nuance.

5. **No per-group accuracy**: Valid point - aggregate accuracy hides important information about the fairness-accuracy tradeoff.

6. **No fair-preprocessing baselines**: Valid point - no comparison with simple group-balanced distillation.

7. **Novelty**: The method essentially replaces weighted matching with unweighted group-wise matching. This is a simple modification. Valid concern about novelty.

8. **No evaluation on more real-world datasets**: Valid - heavily synthetic benchmarks.

Let me now check if the "suspicious baseline" claim holds:
- C-MNIST FG IPC=10: DM=25.01% vs Whole=97.71%. This is a colored MNIST with 90% color bias - in such a heavily biased dataset, vanilla DD would indeed learn to predict color (PA) instead of digit (TA), leading to very low TA accuracy. When FairDD forces balanced matching across all color groups, the TA information becomes more accessible, which genuinely could improve accuracy dramatically. So this is actually consistent with the paper's claim.

However, looking at CIFAR10-S at IPC=10: DM=37.88% vs DM+FairDD=45.17%, and Whole=69.78%. The improvement is more modest here, which makes sense for a less biased dataset.

Let me now write the final review.</think>Now I have enough calibration context. Let me compile the final review.

## Summary

FairDD is the first framework to explicitly address fairness in dataset distillation (DD). The paper identifies that vanilla DD methods inherit and amplify protected attribute (PA) bias due to the dominance of majority groups in the matching objective, and proposes a "synchronized matching" approach that decomposes the class-level matching target into PA-wise sub-targets and aligns synthetic data to all groups equally. The method applies to any matching-based DD method (DM, DC, IDC, DREAM) without architectural modifications, and is supported by theoretical analysis and experiments on synthetic and real-world biased datasets.

## Strengths

- **Well-motivated and novel problem formulation**: The paper identifies an important and previously unexplored gap—vanilla DD amplifies PA imbalance—and provides a clear mechanistic explanation via the PA-weighted decomposition in Eq. 4-6. This is a genuinely novel contribution to the DD community.

- **Simple, general, and plug-and-play method**: Reformulating the matching objective from a weighted class-level sum to an unweighted group-level sum (Eq. 7) is conceptually clean and requires no changes to DD architectures. Demonstrated across 4 DD methods (DM, DC, IDC, DREAM) and multiple bias types (foreground, background, combined, real-world).

- **Consistent and substantial fairness improvements**: Table 1 shows dramatic reductions in DEO_M and DEO_A across nearly all settings (e.g., DM on C-MNIST FG: DEO_M drops from 100.0 to 17.04). These improvements are large and consistent enough to be compelling.

- **Good theoretical grounding for the failure mode analysis**: The derivation in Section 3 (Eq. 4-6) clearly shows how majority groups dominate the matching objective, providing principled motivation for the fix. The analysis is instructive even if the formal theorems have limitations (discussed below).

- **Helpful qualitative diagnostics**: Visualizations of condensed datasets (Fig. 3), feature coverage (Fig. 4), and t-SNE plots (Fig. 5) effectively illustrate the bias-amplification problem and the FairDD remedy.

## Weaknesses

### Major:

- **Missing simple fairness baselines makes it impossible to assess the contribution of synchronized matching vs. straightforward reweighting**: The core idea is to replace weighted matching (weights proportional to group sizes r_y^{a_i}) with unweighted matching (equal weight 1/|A|). The most natural baseline—applying vanilla DD to a PA-balanced subsample of the original dataset, or simply reweighting the vanilla loss by setting r_y^{a_i} = 1/|A| in Eq. 3—is never compared. Without these, it is unclear whether FairDD's "synchronized matching" offers any advantage over simply balancing the training data before distillation. This was also raised in the closely related paper on bias in dataset distillation (L5yq5KsnEZ, Reviewer 3): "A sample reweighting scheme utilizing kernel density estimation is not novel ... Overcoming the bias problem with a reweighting strategy is a well-known approach."

- **Overclaimed theoretical guarantees**: The paper states that Theorems 4.1 and 4.2 "guarantee the fairness and accuracy of synthetic samples" (Introduction). In reality, Thm 4.1 only shows that the mean feature embedding of S converges to the unweighted average of PA group means under idealized optimization and convex distance metrics—this does not directly imply Equalized Odds fairness (which concerns conditional error rates, not feature means). Thm 4.2 shows L_FairDD upper-bounds L_vanilla for convex D, ensuring that minimizing L_FairDD also minimizes L_vanilla, but this concerns embedding distances, not downstream classification accuracy or fairness. The theorems provide intuition for why FairDD should help but do not constitute the formal guarantees claimed.

- **Evaluation predominantly on synthetic datasets with artificially extreme bias (BR=0.9)**: 4 of 6 benchmarks are constructed by injecting bias at 90% ratio. Real-world biased datasets rarely exhibit such extreme skew, and method performance on moderate bias is not explored (the ablation in Table 4 only tests BR ∈ {0.85, 0.90, 0.95}—all extreme). CelebA is the only natural dataset with a single TA/PA pair. This limits confidence that FairDD is effective under the more nuanced bias patterns typical of real applications. The closely related paper (L5yq5KsnEZ) received similar criticism: "The method proposed in this paper seems to excel only on very simple datasets with sufficiently significant biases ... This raises concerns about the practicality and robustness."

- **The dramatic accuracy improvements over vanilla DD suggest baselines may be catastrophically failing on biased data rather than FairDD specifically improving fairness**: On C-MNIST (FG) at IPC=10, DM achieves only 25.01% accuracy while FairDD+DM achieves 94.61%. While the paper attributes this to "limited representation coverage" of vanilla DD, such extreme gaps may simply indicate that heavily biased DD produces data that is dominated by PA information (color), making the classifier predict PA instead of TA. FairDD forces balanced matching, which incidentally recovers TA predictability—but this is an artifact of the synthetic setup rather than a general property. Without per-group accuracy breakdowns and comparison to simpler balancing baselines, it is hard to assess whether FairDD is fundamentally improving the fairness-accuracy tradeoff or just fixing a pathological degeneration that simpler methods would also address.

### Minor:

- **Trajectory Matching is excluded without empirical validation**: The paper excludes TM methods (like MTT) with a brief theoretical argument about overfitting on minority groups, but provides no empirical evidence. Given that TM is among the strongest DD approaches, this limits the "generalist" claim. The related work (L5yq5KsnEZ, Reviewer 2) raised the same concern.

- **Requires full PA labels with no robustness analysis**: FairDD requires PA labels for all training data. The paper acknowledges this in Section 7 as a limitation but provides no experiments testing robustness to noisy/incomplete PA labels. This is a significant practical concern since in real settings, PA information may be protected, unavailable, or inaccurate.

- **No per-group accuracy analysis**: Only aggregate accuracy is reported. Given that DEO measures conditional accuracy disparities, per-group accuracy tables would directly show whether fairness gains come from improving minority group accuracy or degrading majority group accuracy (or both). Without this, the fairness-accuracy tradeoff interpretation is incomplete.

- **No computational cost analysis**: FairDD multiplies the matching terms by |A| (number of PA groups), but no distillation time comparisons are provided. For datasets with many PA categories, overhead may be nontrivial.

### Trivial:

- Table 3 formatting is difficult to parse due to column alignment issues in the parsed version (this may be a rendering artifact).

## Nice-to-Haves

- Evaluate on additional real-world fairness datasets (e.g., UTKFace, BFFHQ) with moderate natural bias.
- Add a simple group-balanced subsampling/reweighting baseline to establish the value of synchronized matching over naive balancing.
- Report per-group accuracy to make the fairness-accuracy tradeoff transparent.
- Test robustness to noisy or incomplete PA labels.
- Report standard deviations across multiple runs.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Fairness metric is under-specified and ambiguous"**: The paper defines DEO_M and DEO_A clearly in Section 2 with a precise mathematical formula: `P(Ŷ=y | Y=y, A=a)` evaluated on models trained on S. The metric is standard EO. The concern about "what Ŷ is" is clearly the model's TA prediction on a balanced test set—this is standard in fairness evaluation.

- **"t-SNE analysis conflates representational invariance with EO"**: While t-SNE is not a formal fairness metric, the paper correctly frames it as a visualization analysis (Section 5.3, "Visualization analysis on fairness and accuracy") and does not claim it as formal evidence. Qualitative visualizations are appropriate supplementary evidence.

- **"Baselines are misconfigured"**: The extremely low accuracy of vanilla DD on heavily biased synthetic datasets (e.g., 25% on C-MNIST FG) is consistent with the paper's thesis: when 90% of data in each class shares one color, vanilla DD learns to predict color rather than digit. This is not a misconfiguration but the exact problem the paper identifies. However, the absence of sanity checks against original DD results on unbiased versions of these datasets does reduce confidence somewhat.

- **"Theorem 4.2 guarantees are weak"**: While the theorems don't guarantee downstream fairness/accuracy, they do provide meaningful characterization of the optimization landscape. The criticism that "minimizing an upper bound doesn't ensure similar minima" is partially valid but standard in the optimization literature—this is a well-known relaxation argument. The issue is one of *overclaiming*, not of the theorems being wrong.

- **"No confidence intervals or multiple runs reported"**: While desirable, single-run evaluation is the norm in the DD community, and demanding this is a standard-level critique rather than a fundamental flaw.

## Novel Insights

The paper's core observation—that DD matching objectives inherit PA imbalance through the weighted expectation in Eq. 4, leading to biased condensed data—is genuinely insightful and well-derived. The connection between the sample ratio weighting in the standard DD loss and majority group dominance is clearly and formally established. However, the proposed solution (removing the weights) is arguably the simplest possible fix, and the paper's contribution lies more in identifying and formalizing the problem than in the sophistication of the solution. The dramatic accuracy improvements on synthetic data raise important questions about whether FairDD's primary benefit is fairness-specific or is actually about preventing catastrophic PA-dominance in DD—this reframes the contribution from "fairness without accuracy loss" to "making DD work on biased data," which is potentially a stronger claim but needs better-controlled experiments to validate.

## Suggestions

- Add a simple reweighting baseline (e.g., PA-balanced subsampling before vanilla DD, or setting r_y^{a_i} = 1/|A| in the original loss) to establish that synchronized matching provides value beyond naive balancing.
- Moderate the theoretical claims: Thm 4.1 characterizes feature-mean convergence properties, not EO fairness. Thm 4.2 ensures feature-matching coverage, not classification accuracy. This is still meaningful but should be stated accurately.
- Report per-group accuracy tables to validate the fairness-accuracy tradeoff story.
- Test on at least one moderately biased real-world dataset to assess robustness beyond the extreme BR=0.9 setting.

## Score and Decision

**Calibration references:**
- L5yq5KsnEZ (Mitigating Bias in Dataset Distillation): scores 5, 3, 5, 5 → avg ~4.5 (Withdrawn/Reject). This is the most directly comparable paper—same topic (bias in DD), similar method (reweighting), similar limitations (synthetic benchmarks, limited novelty of reweighting). FairDD is somewhat stronger: it provides formal theoretical analysis, tests on more DD methods (4 vs 2), and includes a real-world dataset (CelebA). But it shares the same core limitations.
- 3JsU5QXNru (Group Distributionally Robust DD): scores 6, 8, 6, 6 → avg ~6.5 (Accept Poster). This paper has better theoretical grounding, evaluates on standard benchmarks, and addresses subgroup coverage—but it's evaluated as having similar-level novelty and modularity concerns.
- 39cPKijBed (Training Unbiased Diffusion Models): scores 6, 5, 8, 5 → avg ~6.0 (Accept Poster). Comparable novelty level, stronger theoretical contribution, but similar concerns about computational overhead and limited settings.
- hnrB5YHoYu (Finetuning Diffusion for Fairness): scores 6, 10, 6 → avg ~7.3 (Accept Oral). Much stronger novelty and impact.

FairDD sits below the 3JsU5QXNru paper because it lacks simple fairness baselines, overclaims theoretical guarantees, and relies predominantly on synthetic benchmarks with extreme bias. The absence of a naive reweighting comparison is a notable gap. However, it is above L5yq5KsnEZ because it provides formal analysis, more DD method coverage, and a real-world dataset. The simplicity of the method (equal weighting) is both a strength and a weakness—clean but limited novelty.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>