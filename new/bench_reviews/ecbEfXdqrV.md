Now I have all the information needed. Let me write the final review.

## Summary

This paper investigates whether the well-known counterintuitive likelihood phenomenon in image anomaly detection—where deep generative models assign higher likelihoods to anomalous/OOD data—also occurs in tabular settings. The authors propose a formal, domain-agnostic definition of the counterintuitive phenomenon (Definition 3.3) based on relative model performance, then conduct extensive experiments across all 47 tabular and 10 embedding datasets from ADBench against 12 baselines, demonstrating that NF-SLT (normalizing flow with simple likelihood test using NICE) achieves the best average AUROC (0.8575), best average rank (3.43), and lowest fail ratio (0.02). The paper further provides theoretical and empirical analysis linking the phenomenon's rarity in tabular data to lower dimensionality and weaker feature correlation (measured via intrinsic dimension ratios).

## Strengths

- **Comprehensive, unbiased empirical evaluation**: The paper evaluates on all 47 tabular + 10 embedding datasets from ADBench without selection bias, against 12 baselines. Table 1 shows NF-SLT achieving the best average AUROC (0.8575), best average rank (3.43), highest Top2 Ratio (0.45), and lowest Fail Ratio (0.02). This is substantially more thorough than prior work (Kirichenko et al., 2020 used only 2 datasets).

- **Practical finding that NF-SLT is a strong tabular AD baseline**: Showing that a simple likelihood test with NICE outperforms 12 established baselines across 47 tabular datasets is an actionable and valuable result for practitioners. This finding holds regardless of the definitional choices in the paper.

- **Novel quantification of feature correlation via intrinsic dimension**: The paper operationalizes the intuitive distinction between image "homogeneity" and tabular "heterogeneity" through the d Ratio (intrinsic dimension / ambient dimension). Table 4 shows image datasets have d Ratios of ~0.2–1.9% while tabular datasets have d Ratios of ~39–81%, and that NF-SLT performs worse on tabular datasets with lower d Ratios—a clean and informative analysis.

- **Controlled dimensionality experiments (Table 2)**: Using ICA to isolate the effect of dimensionality while enforcing independence provides clear evidence that AUROC increases as dimensionality decreases when H(P) > H(Q), supporting the theoretical prediction.

- **Synthetic validation of ID–correlation relationship**: Figure 1 (left and center) uses Gaussian data with autoregressive covariance to demonstrate that increasing correlation decreases the estimated intrinsic dimension relative to ambient dimension, grounding the d Ratio measure.

## Weaknesses

### Fatal
None.

### Major

- **Definition 3.3 measures relative model performance, not likelihood inversion directly**: The original counterintuitive phenomenon (Nalisnick et al., 2019) concerns likelihood inversion—OOD data receiving higher likelihoods than in-distribution data. Definition 3.3 replaces this with a criterion based on relative AUROC performance: a counterintuitive phenomenon occurs when most comparison models significantly outperform the generative model. While the paper argues this is more principled (lines 21, noting that "the view is contradictory since the argument would consider any result outside 100% AUROC as counterintuitive"), these are conceptually distinct. A model with AUROC = 0.75 (no likelihood inversion) could satisfy Definition 3.3 if other models achieve AUROC > 0.85 + γ. Conversely, a model with genuinely inverted likelihoods (AUROC ≈ 0.5) might not satisfy the definition if no comparison model does much better. This means the headline claim—"the counterintuitive phenomenon is consistently rare in tabular data"—is conditional on this specific definition, and its sensitivity to baseline composition deserves more analysis (e.g., how results change when removing the weakest baselines like DAGMM at 0.6467 or GOAD at 0.6086).

- **Theorem 5.4 assumes product distributions (feature independence), and H(P) > H(Q) is unverified on real datasets**: The theorem assumes P = ∏pᵢ(xᵢ) and Q = ∏qᵢ(xᵢ), which is extremely restrictive for real data where features are correlated. While Table 2 uses ICA to enforce independence (partially addressing this), it creates a somewhat circular setup: testing a theorem that assumes independence by creating independent data. More importantly, the condition H(P) > H(Q) is never verified for any of the 47 tabular datasets. If this condition does not hold, the theorem's conclusion reverses—higher dimension would make the likelihood gap *larger*, not smaller. Without showing that H(P) > H(Q) is common in images but rare in tabular data, the dimensional argument cannot fully explain the cross-domain difference. Table 3 (raw images without independence) shows inconsistent results (e.g., CelebA/SVHN improves with lower dimension even when H(P) < H(Q)), which the paper attributes post-hoc to bilinear interpolation effects but does not verify.

### Minor

- **Conflation of domain properties with anomaly-type design**: The image experiments use cross-distribution OOD detection (CIFAR-10 vs. SVHN) while the tabular experiments use within-distribution anomaly detection (unusual samples from the same dataset). The nature of anomalies is fundamentally different: in cross-distribution OOD, the "anomalous" distribution may have lower entropy, directly causing likelihood inversion; in within-distribution AD, anomalies share statistical properties with normal data. The paper does not fully disentangle whether the rarity in tabular data stems from data properties (dimension, correlation) or from the anomaly task design. The CV/NLP embedding experiments partially address this but remain within-distribution AD tasks.

- **No variance or significance measures reported**: While 10 repeated experiments are conducted, the paper reports only average AUROC scores. Without standard deviations, it is impossible to assess whether the performance gaps (e.g., the 0.02 gap on the "yeast" dataset) are statistically meaningful.

- **d Ratio analysis in Table 4 (bottom) lacks a control group**: The analysis shows what fraction of "poor-performing" (rank ≥ 3) datasets have d Ratio below various thresholds, but does not show what fraction of *well-performing* datasets also fall below those thresholds. Without this comparison, the correlation between d Ratio and performance is not fully established.

### Trivial
None.

## Nice-to-Haves

- A cross-distribution tabular OOD experiment (train on one tabular dataset, test on another) would help disentangle domain effects from anomaly-type effects.
- Verifying H(P) > H(Q) on real datasets (or at least on a representative subset) would strengthen the theoretical claims.
- Including results with non-volume-preserving flows (RealNVP, Glow) in the main paper rather than only in Appendix G would strengthen the generalizability of the NF-SLT finding.
- Reporting standard deviations across the 10 repeated experiments.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh Critic's claim that the paper's definition makes it "a claim about NF-SLT's competitive ranking among 12 chosen baselines, not a claim about likelihood behavior"**: While technically true that Definition 3.3 measures relative performance, the paper explicitly justifies its departure from direct likelihood comparison (Section 3, arguing that direct likelihood comparison is contradictory and conflates intrinsic difficulty with the phenomenon). This is a debatable design choice, not an error—it belongs in Major weaknesses as noted above, but should not be framed as if the paper is unaware of the distinction.

- **Harsh Critic's claim that "the phenomenon can appear or disappear based on future methodological progress"**: This is an inherent property of any relative definition and is not unique to this paper. All benchmark-based evaluations are subject to this concern.

- **Harsh Critic's claim about hyperparameter selection being "unusual" and potentially unfair to baselines**: The paper states that all models follow the same protocol (selecting the hyperparameter setting that maximizes average AUROC across all datasets). This is a deliberate fairness design, not an unfair comparison.

- **Harsh Critic's demand for direct likelihood comparison plots**: While useful, this conflates the paper's chosen definition with a different definition. The paper explicitly rejects direct likelihood comparison and justifies why.

- **Strength Finder's claim that Definition 3.3 "correctly distinguishes genuine counterintuitive failure from intrinsic dataset difficulty"**: This is an overstatement—the definition may also mask genuine likelihood inversion when comparison models are weak, as the Major weakness above notes.

- **Harsh Critic's claim about "testing with non-volume-preserving flows"**: The paper mentions Appendix G includes other flows. The concern about NICE's constant log-determinant is valid but addressed in the appendix.

## Novel Insights

The paper's use of intrinsic dimension (d Ratio) as a bridge between the intuitive notion of feature "heterogeneity vs. homogeneity" and measurable dataset properties is a genuine methodological contribution that goes beyond the specific anomaly detection application. The finding that embedding representations of images have higher d Ratios (23/1000 and 18/1000) than raw pixels (~1%) provides a unified explanation for why both tabular data and embedded image data are less susceptible to the counterintuitive phenomenon—connecting Kirichenko et al. (2020)'s observation about embeddings to a dimensional framework.

## Suggestions

- Run a sensitivity analysis on Definition 3.3: show how the "rarity" of the counterintuitive phenomenon changes when removing the weakest baselines (DAGMM, GOAD, OCSVM) from the comparison pool. If the result is robust, this addresses the major definitional concern directly.
- Estimate H(P) and H(Q) for at least a subset of the 47 tabular datasets and canonical image OOD pairs, even approximately, to verify whether Theorem 5.4's key condition holds in practice.
- Report standard deviations for Table 1 results to allow significance assessment of performance gaps.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Train-before-Test | ORv3SAzus1 | 7.0 | Stronger theory + comprehensive empirical; our paper has weaker theory |
| ReTabAD | UFwgg44VZq | 5.5 | Similar domain (tabular AD), benchmark contribution; our paper has comparable empirical scope but different focus |
| Deep Research Definition | 5EmpOCq1Ql | 4.5 | Similar pattern (definition concerns + strong benchmark); our paper's definition concern is more central to the claim |
| Likelihood Paradox NF | jCQVjd4vrX | 4.0 | Same topic (NF likelihood paradox) but rejected; our paper is stronger empirically |
| Solomonoff/LLM | psDyG1FDzX | 2.4 | Non-rigorous theory, rejected; our paper is far stronger with real empirical contributions |

This paper sits between the medium-scoring anchors. It is clearly stronger than rejected papers on similar topics (jCQVjd4vrX at 4.0) due to its comprehensive empirical evaluation and the genuinely useful finding that NF-SLT works well for tabular AD. However, the definitional concern (Definition 3.3 measures something different from the original likelihood inversion phenomenon) and the restrictive assumptions of Theorem 5.4 prevent it from reaching the higher scores. Compared to the ReTabAD anchor (5.5), our paper has a comparable empirical contribution in a similar domain but also carries a more central definitional issue. Compared to the Deep Research Definition paper (4.5, Accept Poster), our paper has a similar pattern but the definition concern is arguably more important since the paper's headline claim directly depends on the definition.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>