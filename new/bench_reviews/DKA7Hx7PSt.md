Now I have a thorough understanding of the paper and can verify the reviewer claims. Let me synthesize the final review.

## Summary

The paper introduces Learning Embedding Linear Projections (LELP), a knowledge distillation method for binary and few-class problems that extracts pseudo-subclass structure from a frozen teacher's embeddings via PCA-based linear projections, then trains the student on these expanded classes using a single unified cross-entropy loss. LELP's key practical advantage over Subclass Distillation is that it requires no teacher retraining. Experiments across vision (binarized CIFAR) and NLP (Amazon Reviews, Sentiment140, GLUE, LMRD) benchmarks demonstrate LELP outperforms baselines including Vanilla KD, Embedding Distillation, FitNet, VID, Relational KD, CRD, and DKD, and matches or exceeds Subclass Distillation.

## Strengths

- **LELP genuinely does not require teacher retraining**, which is a significant practical advantage over Subclass Distillation, especially for large teacher models. The paper argues this explicitly as a core desideratum (Section 3, "No Retraining the Teacher Model") and notes that SD's hyperparameter optimization "becomes excessively computationally intensive and impractical when dealing with large teacher models" (Section 2). This is the clearest and most impactful contribution.

- **Strong empirical results across multiple settings**: In Table 1, LELP outperforms all practical clustering methods (K-means, Agglomerative, t-SNE+K-means) across all 6 vision settings. In Table 2, LELP achieves the highest performance in 7 of 8 NLP settings, with particularly large margins on Amazon Reviews 5-class (+1.78 over SD, +2.93 over next non-SD baseline) and Sentiment140 (+1.67 over SD).

- **Comprehensive baseline comparison**: The paper compares against 9 distillation methods (Vanilla KD, Embedding Distillation, FitNet, VID, Relational KD, CRD, DKD, and Subclass Distillation) across both vision and NLP, same-architecture and cross-architecture settings, providing a thorough empirical picture.

- **Oracle Clustering upper bound validates the design principle**: Table 1 shows Oracle Clustering substantially outperforms all practical methods (e.g., 86.42% vs. LELP's 79.91% on CIFAR-100bin), confirming that pseudo-subclass information carries significant signal and motivating future improvements.

- **Simple, unified training objective**: LELP converts embedding information into pseudo-subclass probabilities and trains with a single cross-entropy loss (Section 3.3, Eq. 4), avoiding the need to balance multiple loss terms as in Embedding Distillation.

- **Cross-architecture applicability demonstrated**: Table 2 includes same-family/same-dimension (ALBERT-Large→ALBERT-Base), same-family/different-dimension (ALBERT-XXL→ALBERT-Base), and cross-architecture (ALBERT-XXL→MLP+sentence-T5) settings, validating one of the stated desiderata.

## Weaknesses

### Fatal
None.

### Major

- **Confounded comparison with Subclass Distillation due to different teacher models**: The paper acknowledges that "the accuracy of the teacher model in Subclass Distillation usually differs from the one used for LELP (and the other baselines). Therefore, comparing them directly might not be entirely fair" (Section 4.1). Despite this, the abstract, introduction, and conclusion still claim LELP is "consistently competitive with, and typically superior to" SD. Since SD retrains the teacher (e.g., Amazon Reviews 5-class: SD teacher = 78.45 vs. vanilla teacher = 77.58), the student's performance gain may partly come from the retrained teacher's better representations rather than the distillation method itself. The paper partially mitigates this by reporting "best baseline excluding Subclass Distillation" separately, but the overarching "typically superior" framing is stronger than the evidence supports for a confounded comparison. The fairer claim—already partially made—is that LELP matches or exceeds SD *without* the computational cost of teacher retraining, which is a genuinely strong claim on its own.

- **All experiments use α=0, removing ground-truth supervision**: The paper sets α=0 in all experiments (Section 4.1), meaning the student is trained purely on distillation loss. While the paper argues this "focus[es] solely on the effect of the distillation loss" and notes the semi-supervised setting naturally lacks labels, nearly all practical KD deployments combine distillation with ground-truth supervision. The relative ranking of methods could change when α>0, since additional label signal may reduce the marginal benefit of pseudo-subclass information. Without any α>0 experiments, the practical relevance of LELP's demonstrated improvements remains unclear for the most common use case. This is a significant gap given the paper's claims about practical superiority.

### Minor

- **Hyperparameter sensitivity for S and β is addressed only in appendix**: LELP introduces two key hyperparameters—the number of pseudo-subclasses per class (S) and the subclass temperature (β)—that directly control the granularity and softness of the pseudo-subclass decomposition. The main text references ablations in Appendix C but provides no sensitivity analysis in the body. For a method whose core contribution depends on these choices, visibility of robustness in the main text would strengthen confidence.

- **Marginal gains on smaller NLP datasets**: On the three ALBERT-Large→ALBERT-Base settings in Table 2, LELP's gains over the best non-SD baseline are small (0.05–0.24 points), while substantial gains (0.73–4.98 over non-SD baselines) concentrate on the larger ALBERT-XXL→ALBERT-Base settings. The paper does not discuss this scaling pattern, which could illuminate when and why LELP is most beneficial.

- **Theoretical justification for null-space projection and random rotation is limited**: These steps are motivated by intuition ("we have found that it often helps," Section 3.1) with ablations deferred to Appendix C. The neural collapse connection is asserted but not formally established—neural collapse explains convergence to class means, but LELP relies on residual structure after collapse, and why PCA on these residuals preserves useful subclass information is not shown rigorously. This is acceptable for an empirical paper but leaves the theoretical foundations incomplete.

- **Subclass splitting becomes noisy near decision boundaries**: Equation 5 conditions pseudo-subclass probabilities on the teacher's class probability. For examples where the teacher is uncertain (near decision boundaries), the class probability is small, making the subclass structure noisy precisely where teacher knowledge is most valuable. The paper does not address this potential limitation.

### Trivial
None.

## Nice-to-Haves

- Experiments with α>0 under standard KD training conditions, even for a subset of datasets, to verify LELP's advantages persist in the most common practical setting.
- Analysis of why LELP's benefit scales with dataset size (e.g., correlation with embedding dimension, amount of training data, or teacher capacity).
- Sensitivity analysis for S and β in the main text for 2-3 representative datasets.
- Quantitative analysis (beyond t-SNE visualization) of what semantic structure the PCA pseudo-subclasses capture.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's "student outperforming teacher is not surprising" claim**: The reviewer argues the paper's claim that the student surpasses the teacher is "presented as more surprising than it is." The paper states this as an observation without presenting it as paradoxical. This is a subjective framing nitpick, not a substantive weakness.

- **"Missing related works" suggestions**: Per the meta-reviewer rules, these cannot be verified and are removed.

- **Demand for "same teacher" comparison with SD**: The reviewer suggests running SD with the original (non-retrained) teacher. This is impractical—SD *requires* teacher retraining by design, as its method forces the teacher to divide classes into pseudo-subclasses. Using the non-retrained teacher would not be SD at all. The paper already provides the "best excluding SD" comparison to address the confound.

- **Criticism of Oracle Clustering inconsistent margins across architectures**: The reviewer flags that LELP's improvement over K-means varies by architecture (6.02 vs. 2.88 on CIFAR-100bin). This is not a weakness—methods performing differently across architectures is expected and normal, not an inconsistency.

- **Claim that SST-2 contradicts "typically superior"**: The Harsh Critic notes SD (92.85) beats LELP (92.81) on one dataset. However, this difference is within error bars (92.81 ± 0.36 vs. 92.85 ± 0.15), and "typically superior" explicitly allows occasional exceptions. This is not a meaningful contradiction.

- **Formatting/presentation nitpicks**: Claims about overclaiming in the abstract about student>teacher are framing disagreements, not substantive errors.

## Novel Insights

The paper makes a valuable distinction that the KD community should note: information in teacher embeddings that is *orthogonal* to the teacher's class weights (identified via null-space projection) is precisely the information not captured in logits and thus most valuable for distillation. This insight—that PCA on null-space-projected embeddings extracts "residual" structure complementary to the teacher's output probabilities—gives LELP a principled motivation beyond empirical performance. The comparison between LELP and the much more expensive Subclass Distillation establishes that this residual structure can be extracted cheaply and effectively, raising the question of whether more sophisticated (nonlinear) extraction from the null space could further close the gap with Oracle Clustering.

## Suggestions

- Temper the "typically superior to SD" claim to something like "typically matches or exceeds SD without requiring costly teacher retraining"—which is actually a stronger and more defensible claim given the practical advantage.
- Include even a small set of α>0 experiments (e.g., on 2-3 datasets with α∈{0.3, 0.5, 0.7}) to demonstrate robustness under standard training conditions.
- Add a brief sensitivity analysis for S and β in the main text (even 2-3 curves on representative datasets) to show LELP is not overly sensitive to these hyperparameters.

---

<context>
**Original reviewer signal**: The Harsh Critic viewed the paper as having solid ideas but claims that outpace evidence, centered on the confounded SD comparison, α=0 setup, and under-reported hyperparameter sensitivity. The Strength Finder highlighted LELP's strong empirical results across all settings, practical advantage of no teacher retraining, and superior clustering method choice.

**What was dropped and why**: (1) The "student>teacher is not surprising" framing critique — subjective nitpick, not a substantive error. (2) Demand for same-teacher SD comparison — impossible since SD requires teacher retraining by design. (3) SST-2 contradiction of "typically superior" — difference within error bars, and "typically" allows exceptions. (4) Inconsistent margin criticism across architectures — expected variance, not a flaw. (5) Missing related works suggestions — unverifiable. (6) Formatting/presentation nitpicks.

**Cross-checks performed**: Verified that the paper explicitly acknowledges the different-teacher confound on line 158 ("comparing them directly might not be entirely fair") and provides a separate "best excluding SD" metric. Verified α=0 is stated with justification (isolating distillation effect, semi-supervised relevance) on line 152. Verified SSD teacher accuracy differs from vanilla teacher in Table 2 (e.g., 78.45 vs 77.58 on Amazon Reviews 5-class). Checked that one QGLUEval column (likely SST-2) shows SD=92.85 vs LELP=92.81, but within error bars. Confirmed ablations are deferred to Appendix C (line 114). Verified LELP beats all non-SD baselines substantially on large datasets but marginally on small ones (Table 2, columns 1-3 vs 4-8).

**Severity read**: The two major weaknesses (confounded SD comparison and α=0-only experiments) are significant but not fatal. The confounded comparison is partially acknowledged by the authors, and LELP's advantage of no retraining is genuine regardless. The α=0 concern is real but the paper provides a defensible rationale and the semi-supervised setting is a legitimate application. No single weakness threatens the core contribution of a practical, retraining-free distillation method for few-class problems, but together they mean the "typically superior" framing is stronger than fully supported.

**Anything else load-bearing**: The paper explicitly scopes itself to few-class/binary problems and acknowledges LELP doesn't scale to many-class settings (Section 5). The α=0 choice enables a cleaner controlled comparison but limits direct practical applicability claims. The semi-supervised setting (Appendix F) is a natural fit for α=0 but is not in the main text. Gains are strongly dataset-size dependent, which the paper doesn't discuss.
</context>