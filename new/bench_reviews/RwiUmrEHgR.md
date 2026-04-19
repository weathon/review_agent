Now let me run calibration searches in parallel while I continue reading.Now I have enough information to write the final consolidated review.---

## Summary

The paper proposes a Cost-Sensitive Loss (CSL) function for long-tailed image classification that dynamically adjusts per-class weights based on (1) semantic scale values derived from learned feature representations and (2) per-class entropy as a proxy for learning complexity. The loss includes a second term called a "reinforcement_term" designed to encourage continued optimization across epochs. The method is evaluated on CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT, and Tiny ImageNet.

---

## Strengths

- **Entropy-based ETL/DTL distinction (Section 3)**: The observation that tail classes can be easy-to-learn (ETL, e.g., *Airplane*) while head classes can be difficult-to-learn (DTL, e.g., *Dog*) motivates going beyond purely frequency-based re-weighting. This is a legitimate insight under-explored in standard CSL literature, and the *Airplane* vs. *Dog* qualitative example concretely illustrates the problem.

- **Training dynamics visualization (Figures 1 and 2)**: The epoch-level plots of semantic scale and γ_i values for representative classes (including class 91, bicycle) provide genuine interpretability into how the proposed adaptive weighting behaves during training — e.g., a tail class being initially easy sees its γ_i rise then fall, directing the model to harder classes.

- **Plug-in additive structure (Equation 1, Algorithm 1 lines 15/23)**: The CSL term is formulated as an additive regularizer to any base loss, requiring no architectural changes. This is confirmed in the implementation section and makes the approach straightforward to integrate.

---

## Weaknesses

### Fatal

- **The `reinforcement_term` — a stated central contribution — is entirely undefined.** The abstract claims the method "incorporates a reinforcement learning mechanism"; Section 2 states the model is "rewarded with a reward value 'k' depending on the performance improvement it made compared with the previous epoch." Yet no formula, no initialization value, and no update rule are given anywhere in the paper for this term. It appears algebraically in Equations 1 and 2 and in Algorithm 1 line 21 as a bare symbol (`reinforcement`), but its numerical specification is wholly absent. Because it directly enters the loss function that drives all reported results, the method cannot be reproduced from this paper. This is not a presentation gap; the term is undefined at the mathematical level.

- **Two mutually inconsistent formulas for γ_i in the same paper, with an undefined parameter in one.** Algorithm 1 line 19 gives:  
  γ_i ← S_i / (1+ε − α + max(S)·H_i)  
  where α is introduced but never defined anywhere.  
  Section 3 prose gives:  
  γ_i = S_i / ((1+ε)(H_i · max(S_i)))  
  These are structurally different (additive vs. multiplicative denominator; α present vs. absent; max(S) vs. max(S_i)). The paper makes no mention of this discrepancy. Because γ_i drives the method's core re-weighting behavior, a reader cannot know which formula was actually used, making independent re-implementation impossible.

### Major

- **No ablation study of any kind.** The paper provides no experiments isolating the contribution of the entropy term H_i, the semantic scale S_i, or the reinforcement_term individually. On CIFAR-100 at p=100, the proposed method achieves 52.01% vs. LDAM-DRW+SSP at 43.43% — an ~8.6 percentage point gain. Without ablations, it is impossible to determine which component (if any single one) drives this improvement, or whether any component is doing anything beyond noise. The Conclusion itself acknowledges "frequent changes in loss function parameters" cause "erratic gradients" — an implicit acknowledgment of instability that is never quantified or investigated.

- **Comparison limited to baselines from 2019–2021; large CIFAR-100 gains are uninterpretable without modern context.** All baselines in Table 2 are from 2019–2021. The ~8.6% gain over LDAM-DRW+SSP on CIFAR-100-LT at p=100 is a very large margin relative to typical progress on this benchmark, yet no ablation attributes it to any specific design choice and no comparison to methods published in 2022–2024 is provided. Without these, it cannot be determined whether the gains stem from the proposed method or from implementation choices such as the optimizer/scheduler.

### Minor

- **The CSL denominator's index structure is incoherent as written.** Equation 2 has a denominator Σ_k (z_k − e_i)² + ε where z_k indexes data points but e_i indexes the class being summed over in the outer sum. Algorithm 1 line 8 clarifies this as "(inputs − one-hot encoded class vector)²," which appears to be a per-batch quantity. Meanwhile, N_pred,i in the numerator is explicitly per-epoch (validation predictions). Mixing a frozen-per-epoch numerator scalar with a per-batch denominator in a single loss term has no stated principled justification, and the indexing in the equation remains ambiguous.

- **Notation inconsistency: N_{C_i} (Algorithm 1 line 8) vs. N_{pred,i} (surrounding text).** Whether these denote the same quantity is never stated, adding to the ambiguity around the method.

- **Table 1 per-class regressions unacknowledged.** The proposed method achieves lower per-class accuracy than individual baselines in several cases (e.g., Car: 93.75% vs. CB's 96.3%; Frog: 79.49% vs. IB+CB's 86.4%), yet the paper's analysis claims "best accuracy" without acknowledging these cases. The overall average is indeed better, but the selective framing omits relevant information.

### Trivial

- The paper is tagged "Primary Area: datasets and benchmarks" but is clearly a methods paper proposing a novel loss function.
- The abstract and introduction closely follow the phrasing of Yang et al. (2022) over two sentences without adequate paraphrasing.

---

## Nice-to-Haves

- Per-class accuracy breakdown for CIFAR-100 across head/medium/tail splits (standard in the long-tail literature) would help interpret whether gains come from the stated goal of improving tail-class performance.
- Training loss curves comparing CSL vs. CE alone would allow evaluation of the claim that the reinforcement term prevents local optima.
- Clarification of the computational overhead introduced by per-epoch validation-prediction counting and feature storage, especially at ImageNet scale.

---

## Removed Points

*These points were flagged for removal; treat with caution.*

- **"RL framing is a misrepresentation" (Harsh Critic):** Partially valid — the abstract says "incorporates a reinforcement learning mechanism" while Section 2 says the term is "inspired by Reinforcement Learning policies." The problem is not primarily one of framing but of the undefined formula; the overclaiming is real but secondary to the reproducibility issue, which is already captured under Fatal weaknesses.

- **"The L2 regularization connection is a rhetorical gesture" (Harsh Critic):** Minor presentation issue; not a substantive flaw in the method itself. Removed per the soft rule on style nitpicks.

- **"Semantic scale storage cost grows linearly with dataset size" (Harsh Critic):** This is a speculative concern about implementation overhead that the paper does not claim to address; removed as out of scope for this submission.

- **"Table 4 baselines adopted from Park et al. (2021) may be under different conditions" (Harsh Critic):** The paper states results are "adopted from Park et al. (2021)" with the CSL row added. This is standard practice when reproducing controlled comparisons; not a confirmed fairness violation without further evidence.

- **"Strength: 87.64% on Cat shows entropy weighting specifically benefits DTL minority classes" (Strength Finder):** While the number is real (Table 1), a single-class gain in a single dataset is insufficient evidence to support the broader mechanistic claim without ablation. Strength moved here; the ablation gap means this data point cannot cleanly be attributed to entropy weighting.

- **Missing related works (Harsh Critic):** Per hard rules, specific missing paper names (PaCo, BCL, SADE) are not cited in this review.

---

## Novel Insights

The paper's observation that entropy-based learning complexity should modulate per-class weights — independently of raw sample count — is a sensible and partially novel framing. The distinction between ETL and DTL classes is underexplored in the standard CSL literature, which has focused almost entirely on frequency-based adjustments. The Figures 1 and 2 provide preliminary evidence that semantic scale and gamma dynamics exhibit intuitive behaviors during training. However, because the method's two novel components (γ_i and reinforcement_term) are either inconsistently specified or entirely undefined, it is currently impossible to determine whether these dynamics arise from the proposed mechanism or from the interaction of standard cross-entropy training with any scalar additive term.

---

## Suggestions

1. **Define the reinforcement_term completely**: provide its explicit formula, how it is initialized, and how it is updated each epoch. This is a prerequisite for reproducibility.
2. **Reconcile the two γ_i formulas**: choose one, present it in both Algorithm 1 and the prose, and define all parameters (including α if it is retained).
3. **Add ablation experiments** with at least three conditions: CE alone, CE + CSL without reinforcement_term, and CE + full CSL, to attribute the performance gains to specific components.
4. **Report per-class accuracy head/medium/tail breakdowns** for all datasets to verify that gains are genuinely concentrated in the tail as the method is designed to achieve.
5. **Clarify the denominator indexing** in Equation 2 to make the per-batch vs. per-epoch computation explicit and principled.

---

## Score and Decision

**Calibration anchors used:**
- *zvoM1Wastw* (Quantile Regression Adapter, scores 5/3/3/3 avg ≈ 3.5): Inconsistent equation with undefined parameter in one formula. This paper's situation is closely analogous but has *two* inconsistent formulas and also an entirely undefined method component.
- *ZbOSRZ0JXH* (all 3s): Large empirical gains without ablation studies. Directly matches this paper's CIFAR-100 gains concern.
- *Gh1XW314zF* (3/3/3/5 avg ≈ 3.5): Missing ablations and reproducibility details.
- *4SmhpF1nO4* (all 3s): Missing ablations, insufficient method analysis (but the method was at least fully defined).
- *S1IbZssS5a* (6/5/6/5 avg ≈ 5.5): Similar topic (imbalanced learning) but with fully defined method, theoretical analysis, and ablations — a clear higher-quality bar.

This paper falls below all calibration anchors except the incomplete HC26cxtI96 (all 1s). It has *two* fatal reproducibility issues simultaneously (undefined reinforcement_term + contradictory γ_i formulas), which is worse than any single-flaw anchor. The empirical results cannot be attributed to the proposed method given these definitional gaps. The conceptual contribution (entropy-based complexity weighting) is genuine but insufficient to compensate for the inability to reproduce the method.

**Final score: 3.0 — Reject**

MY FINAL SCORE: <pineapple>3.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>