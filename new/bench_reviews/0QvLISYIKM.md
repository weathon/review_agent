Now I have all the information I need. Let me compose the final review.

## Summary

The paper proposes using three pointwise information (PI) measures — PMI, PVI, and PSI — as confidence estimators for deep neural networks, conducting a comparative study of their theoretical properties (invariance, margin sensitivity, convergence rates) and empirical performance on failure prediction and confidence calibration tasks. The central empirical finding is that PVI outperforms both the other PI measures and six existing post-hoc baselines, which the authors argue is consistent with their theoretical analysis showing PVI has the most balanced invariance and margin sensitivity properties.

## Strengths

- **Systematic theoretical comparison of PI measures with clear takeaways**: The paper derives and compares invariance properties (Propositions 1–3), margin sensitivity (Proposition 4, Theorem 1, Proposition 5), and convergence rates for all three measures, distilling these into concrete takeaways (T1–T5, Section 3.3) that directly connect theory to experiments. Theorem 1's lower bound relating PSI to both margin and intrinsic dimensionality is a non-trivial result.

- **Revealing disconnect between margin sensitivity and practical confidence estimation quality**: Table 1 shows PSI has the highest correlation with margin, yet PVI consistently outperforms PSI on confidence estimation tasks (Tables 2–3). The paper's discussion in Section 5 about this disconnect — confidence as boundary sensitivity vs. predictive reliability — is genuinely insightful and suggests the community should think more carefully about what properties matter for confidence estimators.

- **PVI shows substantial improvements on key metrics**: On AUPR_{f,error} and AURC (the preferred failure prediction metrics per Jaeger et al., 2023), PVI shows consistent and often large improvements. For example, ResNet50/CIFAR-10 AUPR_{f,error}: PVI 56.07±3.24 vs. best baseline NE 48.54±1.83 (Table 2). PVI also achieves the lowest ECE across all settings (Table 3).

- **Convergence rate theory predicts dataset-dependent patterns**: Takeaway T4 predicts PMI/PSI should degrade on complex datasets, and this is borne out empirically — PMI and PSI perform competitively on MLP/MNIST but substantially worsen on VGG16/STL-10 and ResNet50/CIFAR-10, while PVI remains robust (Table 2).

## Weaknesses

### Fatal

None.

### Major

- **PVI is not genuinely "post-hoc" in the standard sense — it requires training a second full model, making the comparison against zero-cost baselines fundamentally asymmetric**: The paper frames its contribution as using PI measures "in a post-hoc manner, without needing to modify their architecture or training process" (Abstract, line 15). However, the PVI estimator requires training "another trained network" with the same architecture (Section 2, line 78), which doubles the computational cost of deployment preparation. The baselines compared against (MSP, SM, ML, LM, NE, NG) are all truly zero-cost — computed directly from the trained model's outputs. Since PVI's performance advantage may derive partially from having a second trained model rather than from the information-theoretic measure itself, the headline claim that "PVI outperform[s] all existing baselines for post-hoc confidence estimation" (line 38) is unsupported without comparing PVI against what a second model achieves through simpler means (e.g., the second model's MSP, ensembling, or MC Dropout). The paper acknowledges in the Limitations section (line 320) that "PI measures require training additional models" but does not address the implications for fair comparison.

- **The comparison between PI measures is confounded by different input representations**: Section 4 (line 278) explicitly states that PVI is computed between "input features and predicted labels" using a separately trained model, while PMI and PSI are computed between "output layer features and predicted labels" using the frozen original model. PVI thus benefits from a dedicated representation optimized for the prediction task on raw inputs, while PMI and PSI are constrained to the original model's output-layer representation. The paper justifies this by saying it is "more natural" for each measure, but the confound means one cannot attribute PVI's superiority to the measure itself rather than to its representational advantage. A fairer comparison would apply all three measures to the same feature layer with comparable estimation procedures.

### Minor

- **Temperature scaling is applied to all methods for failure prediction despite the paper citing evidence it can harm this task**: The paper cites Zhu et al. (2022) showing "popular confidence calibration methods have been shown to be useless or harmful for failure prediction tasks" (line 19), yet applies temperature scaling to all methods "to ensure a fair comparison" (line 231). If temperature scaling harms failure prediction, it may differentially deflate baselines, since PVI — which already involves a separately trained model — may be less affected. While uniform application provides some parity, reporting results without temperature scaling for the failure prediction task would strengthen the claims.

- **The claim that PI measures "can potentially reduce inherent bias" from class imbalance (Motivation Point 3, line 33) is entirely untested**: All experiments use balanced datasets (MNIST, F-MNIST, STL-10, CIFAR-10). Given that this claim is presented as a core motivation for the approach, the absence of any class-imbalanced experiment is a noticeable gap.

- **Standard deviations are large relative to differences in some calibration comparisons**: In Table 3 (VGG16/STL-10), PVI achieves ECE 4.91±2.63 vs. MSP 7.42±3.09 — the confidence intervals overlap substantially with only 5 runs, raising questions about statistical significance for this setting. The differences are more convincing for the failure prediction metrics (Table 2), particularly AUPR_{f,error} and AURC.

- **The disconnect between margin sensitivity and practical performance weakens the theoretical narrative**: The paper acknowledges in Section 5 that "better sensitivity to margin doesn't necessarily imply better performance" — but this means the theoretical properties studied (margin sensitivity, invariance) do not cleanly predict which measure works best in practice. The explanation offered (confidence as boundary sensitivity vs. predictive reliability) is plausible but somewhat ad hoc, and is not empirically validated with per-class or per-difficulty-regime analysis.

### Trivial

None.

## Nice-to-Haves

- Compare PVI against a second-model MSP baseline (i.e., use the second trained model's softmax probability directly as a confidence score) to isolate the contribution of the information-theoretic formulation from the advantage of having a second model.

- Apply all three PI measures to the same feature layer with comparable estimation procedures, to eliminate the representational confound.

- Report failure prediction results without temperature scaling, given the cited evidence that it can harm this task.

- Test on at least one class-imbalanced dataset to validate Motivation Point 3.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Fully post-hoc approach requiring no architecture or training modification" as a strength** (from Strength Finder): This conflicts with the verified major weakness — PVI explicitly requires training a second full model (line 78). The claim of being "post-hoc" without modification is technically true about the original model but misleading as a strength when the method itself requires significant additional training.

- **Logarithm base not specified** (from Harsh Critic): The paper uses log without specifying the base throughout. While Proposition 4 (pmi=1 for non-overlapping distributions) only holds with log base 2, this is a convention issue that doesn't affect the comparative analysis since all measures would scale uniformly with base change. This is too minor to list.

- **"Proposition 5 gives only an upper bound"** (from Harsh Critic): This is factually correct but is not a weakness — upper bounds are a standard theoretical tool, and the paper does not overclaim what the bound demonstrates.

- **Missing related works** (from Harsh Critic, implicit): Per instructions, we do not flag missing related works.

- **Formatting/style nitpicks and typos**: Per instructions, these are removed.

- **Missing appendix proofs**: Per instructions, these are removed as the parser strips appendices.

## Novel Insights

The most genuinely novel insight from this review is that the paper inadvertently reveals a fundamental tension in confidence estimation research: the theoretical properties one might expect to matter (margin sensitivity) do not predict practical performance, while the property that does correlate with performance (balanced invariance) is justified post hoc rather than derived from first principles. This suggests the field may need to rethink what theoretical guarantees are actually relevant for confidence estimation, rather than importing geometric intuitions from the margin/robustness literature.

## Suggestions

- Add a "second-model MSP" baseline: train the same second model used for PVI but use its raw softmax output as a confidence score. This single experiment would clarify how much of PVI's advantage comes from the information-theoretic formulation vs. simply having a second trained model.

- In the abstract and introduction, qualify the "post-hoc" claim by acknowledging that PVI requires additional model training, and explicitly position the contribution as comparing PI measures where some require additional training. This would prevent the fair-comparison concern from arising in the first place.

- Consider reporting failure prediction results both with and without temperature scaling, given the paper's own citation of work showing temperature scaling can harm this task.

## Evaluation

**Originality**: The systematic comparison of three PI measures for confidence estimation is a useful organizing framework. The theoretical analysis of invariance, margin, and convergence properties for all three measures is a meaningful contribution. However, PVI itself is not novel (Ethayarajh et al., 2022), and the paper's primary contribution is comparative rather than methodological.

**Importance of research question**: Confidence estimation for DNNs is an important and well-motivated problem. The paper addresses both failure prediction and calibration, which are practically relevant.

**Claim support**: The central claim that PVI outperforms all baselines is weakened by the asymmetric comparison (PVI requires training a second model; baselines are zero-cost). Without a second-model baseline, the claim cannot be fully attributed to the information-theoretic measure.

**Experimental soundness**: The experimental design has a structural confound (PVI uses raw inputs + new model vs. baselines using original model outputs + PMI/PSI using output features). Temperature scaling may differentially affect methods. Standard deviations overlap in some calibration comparisons. Datasets are all balanced, leaving the class-imbalance claim untested.

**Clarity**: The paper is generally well-organized with clear takeaway summaries (T1–T5) that connect theory to experiments. The "post-hoc" framing is misleading given PVI's training requirements.

**Value to community**: The theoretical analysis of PI measures and the finding about margin sensitivity vs. practical performance are useful reference points. The paper would be more valuable if the empirical claims were more carefully supported.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| R-EDL (high) | `/home/wg25r/review_agent/human_reviews/Si3YFA641c.md` | 7.2 | Novel method with strong empirical results and theoretical justification for uncertainty estimation — clearly above this paper, which has an unfair comparison problem |
| Provable Uncertainty Decomposition (high) | `/home/wg25r/review_agent/human_reviews/TId1SHe8JG.md` | 7.5 | Formal guarantees on uncertainty decomposition — much stronger theoretical contribution |
| Relative Uncertainty (medium-high) | `/home/wg25r/review_agent/human_reviews/ruGY8v10mK.md` | 6.5 | Novel data-driven uncertainty measure with empirical improvements — similar scope but cleaner comparison setup |
| Broken Confidence Estimator (medium) | `/home/wg25r/review_agent/human_reviews/YUefWMfPoc.md` | 5.75 | Post-hoc confidence estimation comparison study — similar topic, rejected despite cleaner experiments |
| Confidence as Vulnerability (medium-low) | `/home/wg25r/review_agent/human_reviews/0IqriWHWYy.md` | 4.25 | Limited technical contribution and missing baselines — similar concerns about overclaimed generality |
| External Insight Calibration (low) | `/home/wg25r/review_agent/human_reviews/miIE56qM10.md` | 3.0 | Specifically criticized for "potentially unfair" comparison of trained post-processing vs. zero-cost baselines — very similar to this paper's core problem |
| Post-prediction Confidence (low) | `/home/wg25r/review_agent/human_reviews/AL4tS0HhJT.md` | 2.5 | No baseline comparison — clearly below this paper |

This paper sits between the medium-low and medium anchors. It has genuine theoretical contributions (invariance analysis, margin sensitivity, convergence rates) that the low-scoring papers lack, but its central empirical claim is undermined by the same unfair-comparison issue that tanked the miIE56qM10 paper (score 3.0). The paper is stronger than the low anchors because of its systematic theoretical analysis, but weaker than the medium anchors because its main empirical finding doesn't hold up under scrutiny. The paper is roughly comparable to 0IqriWHWYy (4.25) which had similar concerns about overclaimed generality with limited experiments, but with more theoretical content. I place it slightly above at 4.5, reflecting the value of the theoretical analysis even though the empirical claims are weakened.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>