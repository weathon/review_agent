Now let me look at the human reviews for calibration:Now I have sufficient information to write the final meta-review. Let me synthesize everything carefully.---

## Summary
This paper proposes using three pointwise information (PI) measures—PMI, PSI, and PVI—as post-hoc confidence estimators for deep neural networks. The authors provide theoretical analysis covering invariance properties, sensitivity to sample-wise margin and intrinsic dimensionality, and convergence rates, then empirically compare all three PI measures on failure prediction and confidence calibration across four model-dataset combinations. The central finding is that PVI consistently outperforms PMI, PSI, and a set of simple softmax-based baselines on both tasks.

---

## Strengths

- **Systematic theoretical characterization:** The paper derives and compares invariance properties (Propositions 1–3), margin sensitivity (Proposition 4–5, Theorem 1), and convergence rates for all three PI measures in a unified framework. This fills a genuine gap—prior work had only studied individual measures in isolation.

- **Consistent and practically meaningful empirical finding:** Across all four model-dataset setups, PVI is the best method for AUPR_{f,error} and AURC, which the paper correctly identifies as the most discriminative metrics (Jaeger et al., 2023). The AUPR_{f,error} gains are large and consistent (e.g., MNIST: 51.83 vs 42.50 for MSP; CIFAR-10: 56.07 vs 47.99 for MSP). On calibration (Table 3), PVI's improvements are especially substantial on complex datasets (ECE 4.91 vs 7.42 for MSP on VGG16/STL-10).

- **Insightful disconnect between margin correlation and downstream performance:** The paper honestly reports that PSI has the highest margin correlation (Table 1) yet PVI performs best on downstream tasks, and discusses this in Section 5 with a sound explanation distinguishing "decision-boundary sensitivity" from "predictive reliability." This is a genuinely useful conceptual contribution.

- **Good theoretical-empirical alignment where it applies:** The convergence-rate prediction (T4) that PMI/PSI would struggle on complex datasets with more overlapping distributions is borne out: PMI and PSI are competitive on MNIST but degrade on STL-10 and CIFAR-10, while PVI maintains superiority.

- **Clear writing and well-organized structure:** The paper is well-structured, and the theory-to-experiment narrative is coherent, even if the bridge is imperfect.

---

## Weaknesses

### Fatal
_None that invalidate the core finding._

### Major

- **Structural asymmetry: PVI is not strictly post-hoc in the same sense as competing baselines.** PVI requires training an entirely separate neural network of the same architecture as the original classifier (Section 2, PVI Estimator; Section 4). This is qualitatively different from MSP, SM, ML, LM, NE, and NG, which are all computed directly from the target model's existing outputs. PMI and PSI are also more lightweight (PMI uses a shallow 2-layer network; PSI uses projections with Gaussian fits). PVI effectively involves training a second full model. The claim that this is simply "post-hoc" is therefore misleading: it has the computational footprint closer to a model ensemble or auxiliary predictor. The paper acknowledges in Limitations that "PI measures require training additional models," but does not flag this as an asymmetry affecting the fairness of comparisons. To sustain the comparative claim, PVI should also be compared against other methods that use auxiliary trained models (e.g., a 2-model ensemble, MC Dropout), which would be a fairer baseline.

- **Overclaiming: "outperforming all existing baselines for post-hoc confidence estimation."** The baseline set includes only six simple logit/softmax-based heuristics (MSP, SM, ML, LM, NE, NG). No stronger or more contemporary post-hoc baselines are included—e.g., energy scores, representation-space density methods (Mahalanobis, DDU), or learning-based confidence estimators. The abstract's claim that PVI beats "all existing baselines" is not supported by this limited comparison. A more accurate statement would be: "PVI outperforms a set of simple softmax/logit-based post-hoc baselines on four standard vision benchmarks."

- **Narrow experimental scope:** Evaluation is restricted to in-distribution misclassification detection and selective prediction on four relatively small/simple datasets (MNIST, F-MNIST, STL-10, CIFAR-10) with single architectures per dataset. The paper explicitly scopes out OOD detection (Section 4.1: "This work focuses on the first two tasks"), yet the abstract invokes "safe deployment in high-stakes applications" and lists OOD detection as part of failure prediction. There is no evaluation under distribution shift, no large-scale benchmark (ImageNet), and no non-vision domain. Given the scope of the claims, these are significant gaps.

### Minor

- **Theory-to-practice gap for multi-class settings:** Most theoretical results (Propositions 4–5, Theorem 1) are derived for binary classification with specific distributional assumptions. The paper claims these "trivially extend" to multi-class, but does not demonstrate this, particularly for the margin-sensitivity results. All experiments are 10-class, and the binary theory's connection to the multi-class empirical results is not established.

- **Statistical significance:** For several metrics (AUROC_f, AUPR_{f,success} on STL-10), the improvements over the best baselines fall within one standard deviation. The paper correctly identifies the AUPR_{f,error} and AURC improvements as more significant, but the narrative sometimes treats all improvements as equally convincing. Pairwise significance tests on the more marginal improvements would strengthen the claim.

- **Calibration evaluation is narrow:** Only ECE with 10 bins is reported. No reliability diagrams, Brier scores, or NLL are presented, which limits the reader's ability to assess whether PVI is well-calibrated across the full probability range or only on average.

### Trivial

- The paper scopes out OOD detection explicitly, though it is mentioned in the introduction. A one-sentence clarification in the abstract would help avoid the expectation mismatch.

---

## Nice-to-Haves

- **Comparison against at least one auxiliary-model baseline:** Including a 2-model ensemble or MC Dropout as a point of reference would let readers assess whether PVI's gains come from the information-theoretic formulation or simply from using a second trained network.
- **Reliability diagrams:** Visualizing calibration curves for each method on at least one dataset would add substantial interpretability to the calibration section.
- **Feature-layer ablation:** PMI and PSI use output-layer features, while PVI uses raw inputs. An ablation swapping these input representations would help disentangle whether performance differences stem from the measure itself or from the choice of representation.
- **Larger-scale or harder settings:** At least one result on CIFAR-100 or a distribution-shift scenario would significantly strengthen generalizability claims.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic] Reproducibility gap: hyperparameters for auxiliary estimators not in main text.** → REMOVED per hard rule on reproducibility nitpicks. The paper notes comparisons in the appendix (Appendix C/D), which is standard practice.

- **[Harsh Critic] PSI's Gaussian assumption is nontrivial.** → REMOVED as a structural weakness; the paper compares Gaussian vs. binning estimators in the appendix (D.2.2), and Gaussian estimation for 1D projections is a standard well-studied procedure. This is an implementation detail.

- **[Harsh Critic] Estimator choice validity (JS bound method for PMI, comparisons in appendix).** → REMOVED; estimator comparisons are done in appendix, and the chosen method is well-justified.

- **[Human Finder] Claims about specific other papers and their associated findings.** → REMOVED/WEAKENED to the extent they referenced external papers' specific shortcomings not directly verified against this paper.

- **[Harsh Critic / Spark] Temperature scaling may double-apply to PVI.** → This is a speculative concern not directly verified from the paper text. The paper applies temperature scaling to all methods uniformly (Section 4), and PVI's additional calibration within its estimator is part of its design. REMOVED as unverified.

- **[Harsh Critic] PMI is constant (= 1) for non-overlapping distributions, making the theory incomplete.** → Actually valid but WEAKENED: the paper presents this as a theoretical result (Proposition 4) and explicitly uses it to argue against PMI. It's not a flaw in the paper; it is the paper's stated finding.

---

## Novel Insights

The most genuinely novel observation surfaced across reviewers is the empirical demonstration that *margin sensitivity and downstream confidence quality are not aligned*: PSI is the most geometrically sensitive to the decision boundary (highest Pearson correlation with margin in all four settings) yet performs worst among the PI measures on misclassification detection and calibration. This finding challenges a common intuition that uncertainty estimators should closely track decision-boundary proximity, and the paper's explanation—that margin sensitivity conflates boundary distance with prediction correctness—is a meaningful conceptual contribution to the confidence estimation literature. Combined with the theoretical characterization of invariance properties (PVI unique in being invariant to invertible linear but not arbitrary homeomorphic transformations), this provides a principled basis for preferring PVI that goes beyond empirical cherry-picking.

---

## Suggestions

1. **Reframe PVI's "post-hoc" positioning:** Either include ensembles and MC Dropout as baselines (since PVI similarly uses an extra model) or explicitly reposition PVI as an "auxiliary-model-based" confidence estimator, and limit comparative claims accordingly.
2. **Scale down abstract/conclusion claims** to match the evidence: "outperforming simple post-hoc logit/softmax baselines on four standard vision benchmarks" rather than "all existing baselines."
3. **Add at least one OOD detection experiment** to complete the failure prediction evaluation promised in the introduction, even on a standard pair (e.g., CIFAR-10 vs. SVHN).
4. **Provide a runtime and parameter count comparison** for the three PI estimators (the paper's own limitations section flags this). Without it, practitioners cannot assess the PVI vs. MSP trade-off.
5. **Provide reliability diagrams** alongside ECE to give a more complete picture of calibration quality.
6. **Demonstrate or discuss the multi-class extension** of Theorem 1 (PSI margin bound), even informally, to bridge the binary-theory/multi-class-experiment gap.

---

## Score and Decision

**Calibration against human-reviewed papers:**

- *MNGMpHxi1I* ("On Information-Theoretic Measures of Predictive Uncertainty," withdrawn): Scores 3,1,3,3,5. That paper also compared information-theoretic UQ measures and found "no single measure is universal." This paper under review is strictly stronger: it has a clear winner (PVI), a theoretical explanation for why, and consistent empirical results. It should score above that set.
- *YUefWMfPoc* ("How to Fix a Broken Confidence Estimator," post-hoc selective classification): Scores 6,5,6,6. That paper evaluated post-hoc methods at scale (84 ImageNet models) with a novel metric. The current paper's experimental scope is significantly narrower (4 small datasets), though it adds a more substantive theoretical component.
- *49Tn5mfTy5* (UQ via codebook): Scores 6,5,1,8; average ~5. Similar criticism of small-scale evaluation and limited baseline comparison.
- *cZttUMTiPL* (distribution propagation, accepted): Scores 8,6,6,8,6. That paper demonstrated broader applicability (OOD, distribution shifts, calibrated intervals) and stronger theoretical guarantees. The current paper is below this level.

**Summary positioning:** This paper sits between the failed information-theoretic UQ study (MNGMpHxi1I, ~3) and the borderline-accept post-hoc confidence study (YUefWMfPoc, ~5.75). The narrow experimental scope, limited baselines, and structural asymmetry in the PVI comparison are real impediments. The paper's core contribution—a theoretically grounded comparative study with a consistent finding—is genuine but not yet at the level required by a venue like ICLR given the gaps identified. I position this at **5.0** (marginally below acceptance).

**Originality:** Moderate — systematic comparison of three PI measures is new, but each measure has been studied individually. The invariance propositions are novel.  
**Importance of research question:** High — post-hoc confidence estimation is practically critical.  
**Claims well supported:** Partially — empirical core claim is well-supported for AUPR_{f,error} and AURC; broader claims ("all existing baselines") are not.  
**Soundness of experiments:** Fair — results are consistent but scope is narrow; no statistical tests; baselines are basic.  
**Clarity of writing:** Good.  
**Value to community:** Moderate — provides useful theoretical properties and identifies PVI as a strong practical method; limited by narrow evaluation.

**Decision: Reject** (revise to address scope, baselines, and framing)

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>