## Summary

This paper presents MedAttention, a framework for forecasting severe diabetic complications 6–12 months ahead using Brazilian TUSS billing codes. The model combines skip-gram embeddings for ~170k billing codes, absolute sinusoidal time embeddings, and a BiLSTM with self-attention, achieving AUC 0.907 and AP 0.631 on a cohort of ~105k diabetic patients from 3.9M individuals. The work demonstrates transfer across health operators and includes blinded field validations with clinical experts.

## Strengths

- **Scale and real-world deployment evidence.** The dataset spans 3.9M individuals and 62.7B claim lines, representing the largest TUSS-based prediction study to date. The blinded field validations at both operators—with clinical experts reviewing flagged patients and confirming previously unrecognized high-risk individuals—go beyond standard hold-out metrics. The finding that 41 flagged patients in Operator 2 were newly enrolled in monitoring programs demonstrates practical utility.

- **Clear empirical finding on temporal encoding.** The ablation study (Table 4) shows that time embeddings alone provide no benefit (AUC 0.735 vs 0.741 for BiLSTM), attention alone yields modest gains (0.817), but their combination produces substantial improvement (0.907). This complementarity is a concrete design insight for modeling irregular clinical sequences.

- **Transfer across operators.** The model trained on Operator 1 achieves AUC 0.92 and AP 0.70 on Operator 2 without retraining, demonstrating that standardized billing vocabularies induce transferable structure across populations with different demographics and benefit mixes.

## Weaknesses

- **Abstract's performance claim is misleading on the primary imbalance metric.** The abstract states the model "outperforms capacity-matched baselines," but Table 3 shows the Transformer achieves *higher* Average Precision (0.641 ± 0.011) than MedAttention (0.631 ± 0.003). The paper itself notes AP is the primary metric under severe class imbalance (~1% prevalence). Claiming uniform superiority when the strongest baseline wins on the most relevant metric requires justification or retraction.

- **TCN baseline shows pathological near-chance performance.** The TCN achieves AUC 0.750, F1 0.064, AP 0.051—dramatically worse than all other models and barely above random on F1/AP. This raises concerns about whether baselines received fair hyperparameter tuning. Using SGD with fixed learning rate (unusual for modern sequence models) may disadvantage architectures designed for Adam-family optimizers. A baseline comparison is meaningful only if all models are given reasonable opportunity to converge.

- **No statistical comparison across models.** The paper reports means and standard deviations over 10 runs but provides no paired statistical tests or confidence intervals on differences. Without this, the significance of AUC gaps cannot be assessed.

- **Ablation results lack variance estimates.** Table 3 provides mean ± sd, but Table 4's ablation results are single point estimates. Whether the differences (e.g., BiLSTM 0.741 vs BiLSTM+TE 0.735) are meaningful or within run-to-run noise cannot be determined.

- **Sequence truncation strategy is unspecified.** The model processes sequences with L ≤ 500 events, but Figure 2 shows many patients have thousands of claims. How are sequences truncated—most recent 500, random sampling, or another strategy? This decision critically affects what temporal patterns the model can learn and is necessary for reproducibility.

- **Why time embeddings alone degrade performance is unexplained.** The ablation shows adding time embeddings to BiLSTM alone *hurts* AUC (0.741 → 0.735). This counterintuitive result deserves analysis—does temporal information interfere with code semantics when added to embeddings? The paper does not address this.

- **Field validation lacks methodological rigor.** The blinded validation describes that flagged patients were reviewed and confirmed as high-risk, but critical details are missing: (1) follow-up period, (2) denominator for rates like "34% hospitalizations," (3) background rates among unflagged patients, (4) whether confirmation was by clinicians or model-generated. Without a matched control group, this cannot demonstrate the model outperforms clinical judgment.

- **Spearman correlations are computed against model predictions, not outcomes.** Section 4.5 correlates code frequencies with *predicted risk*, not actual complications. This is circular—the model's predictions depend on its learned representations, so finding correlations between input features and model outputs is tautological. Interpretability requires correlations with ground truth outcomes.

- **No subgroup performance analysis despite fairness claims.** The paper lists fairness considerations as a contribution and dedicates Section 6 to responsible use, yet reports no subgroup analyses by age, sex, or region. The cohort is 57% female, 43% male—given known sex differences in diabetic complications, stratified performance should be reported.

## Nice-to-Haves

- **RETAIN comparison.** RETAIN is the most directly analogous model (RNN with attention for medical codes) and is cited in related work but not evaluated. Including it would contextualize whether the proposed architecture improves over established clinical sequence baselines.

- **Calibration metrics.** With 1:1 training oversampling on ~1% prevalence data, predicted probabilities require recalibration for use as risk scores. Brier scores or calibration curves would address whether the model's probability outputs are well-calibrated.

- **Breakdown by complication type.** The 1,019 outcomes collapse angiopathy, amputation, and renal failure—conditions with different mechanisms and predictability. Performance by subtype would clarify what the model actually captures.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"No architectural novelty" as weakness.** The abstract's honest self-assessment that the contribution is "a methodological instantiation rather than an architectural novelty" is cited as a weakness. This mischaracterizes the authors' transparent scoping—the paper claims empirical design lessons, not architectural innovation. Evaluating it against that stated scope is appropriate; penalizing it for not claiming something it disclaims is not.

- **"First TUSS analysis is not a scientific contribution."** While dataset contributions carry less weight at ICLR than ML venues, this framing undervalues the reproducibility and transparency in cohort construction, which the paper provides in detail.

- **Cohort definition via HbA1c creates selection bias.** The paper explicitly acknowledges this limitation in Section 6. Criticizing absence of magnitude estimation when the limitation is already stated is scope creep.

- **Outcome codes not validated as diabetes-specific.** Appendix A lists codes for amputation, hemodialysis, and angioplasty procedures. While angioplasty can occur for non-diabetic conditions, amputations and dialysis in this cohort context are strongly diabetes-associated. The criticism overstates the problem without evidence of misclassification.

- **Demanding clinical risk score comparisons (Framingham, UKPDS).** These are actuarial/statistical models from a different tradition. The paper evaluates against neural sequence models with capacity control, which is the appropriate comparison class for its contribution.

## Novel Insights

The complementarity between absolute sinusoidal time embeddings and attention is the paper's most interesting empirical finding: neither component works alone, but together they enable the BiLSTM to exploit temporal structure in sparse claim sequences. This suggests attention requires temporally-informed inputs to meaningfully weight events, and conversely, that time embeddings need a mechanism to selectively apply temporal knowledge. The failure of time embeddings alone (degrading baseline performance) is an underappreciated finding—raw temporal signals may introduce noise without a learned selection mechanism.

## Suggestions

- Revising the abstract to accurately reflect comparative performance, particularly acknowledging the Transformer's AP advantage, would strengthen credibility.

- Adding error bars to Table 4's ablation results and statistical tests comparing models would enable readers to assess significance.

- Specifying the sequence truncation strategy (e.g., "most recent 500 events retained") would close a reproducibility gap.

- Computing subgroup performance (at minimum by sex, given the demographic distribution) would substantiate the fairness discussion.

- Correlating code frequencies with actual outcomes rather than model predictions would provide valid interpretability evidence.