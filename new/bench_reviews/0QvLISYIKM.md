Now let me read the calibration papers to score properly.Now I have enough to write the final review. Let me compile all findings.---

## Summary
This paper proposes using three pointwise information (PI) measures — PMI, PVI, and PSI — as post-hoc confidence estimators for trained deep neural networks. The authors derive theoretical properties (invariance, margin sensitivity, convergence rates) and conduct experiments on failure prediction and calibration on four model/dataset combinations. The central empirical finding is that PVI outperforms both the other PI measures and six softmax-based baselines on AUPR_{f,error} and AURC across all settings.

---

## Strengths

- **Systematic theoretical framework** (Propositions 1–3, Theorem 1, Proposition 5): The paper derives and compares, for the first time in one place, the invariance properties of PMI, PVI, and PSI. Proposition 2's result that PVI is invariant to general invertible linear transformations while PSI is not is a clean and practically relevant finding. Proposition 4 (PMI is constant = 1 for non-overlapping class-conditional distributions regardless of margin) is a crisp theoretical result motivating why PMI is blind to sample difficulty.

- **Theory-driven experimental predictions, mostly confirmed**: Takeaway T5 (Section 3.3) predicts PVI will perform best overall, which is borne out in Table 2 (PVI achieves highest AUPR_{f,error} in all four settings: 51.83, 51.62, 54.07, 56.07 vs. best baselines of 42.50, 44.18, 50.67, 48.54 respectively). Takeaway T4's prediction that PMI/PSI degrade on complex datasets (due to convergence) is confirmed by their drop on STL-10 and CIFAR-10.

- **PVI calibration improvements are substantial** (Table 3): PVI achieves consistently lower ECE across all settings (e.g., 4.91 vs. 7.42 for MSP on VGG16/STL-10, a 34% relative reduction), a noteworthy practical result.

- **Non-obvious finding about margin sensitivity**: Table 1 establishes that PSI has the highest correlation with sample-wise margin (0.846 on CNN/F-MNIST vs. 0.368 for PVI), yet PVI outperforms PSI on failure prediction. Section 5 provides a principled explanation distinguishing decision-boundary sensitivity from predictive reliability. This is a genuinely illuminating finding for the field.

---

## Weaknesses

### Fatal
None.

### Major

- **Computational asymmetry in comparison undermines the empirical claim**: PVI is computed using two independently trained full-architecture networks (Section 2, PVI Estimator: "using the second approach—another trained network as default"), while every baseline (MSP, SM, ML, LM, NE, NG) is a zero-cost algebraic transformation of the original model's single forward pass. PVI's improvements may therefore be partly attributable to the additional model capacity rather than any information-theoretic property. The paper acknowledges this implicitly in the Limitations section ("Our PI measures require training additional models"), but provides no control for it — e.g., no simple second-model ensemble baseline, no MC Dropout, and no ablation that uses PVI with the same weights as the original model. Without such controls, the claim that PVI's advantage stems from its information-theoretic properties (rather than extra training) cannot be established. This is the most significant methodological gap in the paper.

- **Restricted baseline set inflates the scope claim**: All six baselines (MSP, SM, ML, LM, NE, NG) are functions of the same logit vector from the same trained model (Appendix C.3.1, equations 83–88). The abstract's claim "outperforming all existing baselines for post-hoc confidence estimation" is not supported — it is outperforming six variants of the softmax output. Post-hoc methods with comparable or greater compute (e.g., energy-based scoring, ODIN, temperature scaling variants) are absent. The paper should either narrow its scope claim to "outperforms softmax-based baselines" or include representative comparators from other categories.

### Minor

- **Theory-margin connection only partial**: The paper's theoretical emphasis on margin sensitivity (Theorem 1, Proposition 5, T3) is the most developed piece of theory, yet Table 1 shows PVI has the *lowest* margin correlation among the three measures, while PSI — the most margin-sensitive — performs worst on failure prediction in several settings (CNN/F-MNIST: PSI AUROC 90.15 vs. MSP's 92.57). Section 5 provides a reasonable explanation distinguishing the two goals, but this weakens the framing of margin sensitivity as the operative mechanism for PVI's superiority.

- **Temperature scaling interaction with PVI**: Section 4 states "we perform temperature scaling based calibration for all methods," and Section 2 (PVI Estimator) notes "we consider using temperature scaling" for PVI internally. It is unclear whether temperature scaling is applied once or twice to PVI, which matters for interpreting the ECE advantage in Table 3. A clarification is needed.

- **No computational cost reporting**: The overhead of training two additional full-architecture networks per confidence score estimate is practically relevant and should be quantified. Practitioners comparing PVI to a simple MSP call need this information.

### Trivial
None beyond parser artifacts, which are not the authors' errors.

---

## Nice-to-Haves

- Adding a "second-model ensemble" baseline (train an independent network of the same architecture, use its softmax output as the confidence score) would directly control for the additional compute used by PVI and would substantially strengthen the core empirical claim.
- CIFAR-100 or ImageNet-scale validation would test whether PVI's calibration advantage (Table 3) generalizes beyond simple 10-class settings.
- Reliability diagrams accompanying Table 3 ECE numbers would reveal whether PVI's lower ECE comes from consistent improvement across confidence bins or from a few bins.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Proposition 4 ambiguity about log base"** (Harsh Reviewer): The reviewer noted that "pmi(x;y) = 1" only holds under a specific log base. In fact, Proposition 4 uses the natural logarithm convention, and the "= 1" result follows directly when P(Y=0)=P(Y=1)=0.5 and distributions are non-overlapping: log(p(x,y)/(p(x)p(y))) = log(p(x|y)/p(y)) = log(1/0.5) = log(2) ≠ 1 in nats. This is a legitimate minor theoretical imprecision in the paper, but the reviewer's framing is confused and it is a trivial notational issue, not substantive.

- **"T4 prediction directly contradicted"** (Harsh Reviewer): The reviewer claimed T4 predicts PMI/PSI should beat PVI on MNIST, but PVI still wins (97.53 vs. 97.34). T4 says PMI/PSI "may do well for simpler datasets" relative to complex ones — it does not predict they beat PVI absolutely. T5 explicitly states PVI performs best overall. The contradiction is overstated; removed.

- **"Double-temperature scaling invalidates Table 3"** (Harsh Reviewer framing as structural): The concern is valid as a minor clarification request but does not rise to a structural issue; downgraded to minor.

- **Strength Finder: "Post-hoc applicability without architecture modification"**: This is a generic claim and is directly contradicted by the Major weakness that PVI requires training additional full-architecture models. Removed per rules.

- **Strength Finder: "Theory-driven prediction that PMI and PSI struggle on complex data is empirically confirmed"**: This is a supporting strength but partially generic — the paper predicts PVI wins, which it does, but T4's specific prediction about PMI/PSI convergence is only one part. Retained in weaker form in main strengths.

---

## Novel Insights
The paper's most genuinely novel observation is the empirical dissociation between margin-correlation and failure-prediction performance: PSI is the most sensitive to the decision boundary geometry (highest Pearson correlation in Table 1 across all settings) yet performs the worst on failure prediction in complex settings. This result challenges the intuitive assumption that a confidence score should be a proxy for geometric difficulty, and instead suggests that predictive reliability (correctness of the prediction) and proximity to the decision boundary are distinct quantities that should be measured differently. This observation has implications beyond this specific work — it suggests that efforts to improve confidence estimators by making them more "margin-aware" may be misguided if the downstream task is accuracy-based failure detection rather than boundary-based uncertainty.

---

## Suggestions

1. **Add a second-model ensemble baseline** that trains a fresh model of the same architecture and uses its softmax confidence as a baseline. This is the minimum control needed to separate "additional compute" from "information-theoretic advantage" for PVI.
2. **Revise the abstract and Section 5 claims** to scope the comparison accurately: "outperforms six softmax-based post-hoc baselines" rather than "all existing baselines for post-hoc confidence estimation."
3. **Report training time and inference cost** for all three PI measures vs. baselines, even in an appendix. For practitioners, the decision to use PVI depends entirely on whether the calibration/AUPR gain justifies training two extra models.
4. **Clarify the temperature-scaling procedure for PVI** to rule out double application and ensure the ECE comparison in Table 3 is clean.

---

## Score and Decision

**Calibration anchors:**

| Paper | Avg Score | Decision | Relevance |
|---|---|---|---|
| AL4tS0HhJT (Post-prediction confidence training) | 2.50 | Reject | Low scoring anchor; much weaker than paper under review — poor baselines, wrong claims |
| YUefWMfPoc (Post-hoc confidence estimators for selective classification) | 5.75 | Reject | Most topically similar; extensive empirical study with same weakness: insufficient comparison to other calibration methods |
| ruGY8v10mK (Data-driven uncertainty measure for misclassification detection) | 6.50 | Accept | Comparable scope; accepted despite some presentation gaps, proposed genuinely novel measure |
| TId1SHe8JG (Higher-order calibration with formal guarantees) | 7.50 | Accept (Spotlight) | Strong theoretical contribution with formal guarantees; clearly stronger theory than this paper |

**Reasoning:** This paper sits between the YUefWMfPoc (5.75, rejected for insufficient baseline comparison) and ruGY8v10mK (6.50, accepted for novel measure with clear improvements). Like YUefWMfPoc, it performs an empirical comparison of confidence estimators and is criticized for comparing against a narrow baseline set while overclaiming scope. Like ruGY8v10mK, it proposes a systematic framework with a new measure (PVI applied in this context) that shows genuine empirical improvements.

The computational asymmetry issue (PVI trains two full models vs. zero-cost baselines) is a genuine Major weakness not present in either anchor paper. This pushes the score toward the lower end. The theoretical contribution (systematic invariance/margin/convergence analysis) is genuine but moderate in depth compared to TId1SHe8JG. 

**Originality**: Moderate — the PI measures exist; their application and comparison here is novel.  
**Importance of research question**: High — confident uncertainty estimation is broadly important.  
**Claims well-supported**: Partially — PVI beats the specific defined baselines, but "all existing post-hoc baselines" is not supported.  
**Experimental soundness**: Moderate — consistent across 4 settings, but narrow comparison set.  
**Clarity**: Good — Section 3.3 takeaways are helpful.  
**Value to community**: Moderate — useful comparison of PI measures, but the central claim requires the computational asymmetry to be controlled.

Final score: **5.0** (borderline reject). The paper has a genuine theoretical contribution and consistent empirical improvements within its defined scope, but the Major weakness of the computational asymmetry combined with the overclaimed scope in the abstract is sufficient to prevent acceptance without revision.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>