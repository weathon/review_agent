=== CALIBRATION EXAMPLE 19 ===

# Final Consolidated Review
## Summary

This paper investigates whether example selection for In-Context Learning (ICL) amplifies social biases in Large Language Models. The authors construct a new dataset (EEC-paraphrase) by paraphrasing the Equity Evaluation Corpus with GPT-3.5-Turbo, and across 8 LLMs and 4 example selection methods, find that while mean bias often decreases with example selection, the *maximum* bias values tend to increase. To mitigate this, they propose ReBE (Remind with Bias-aware Embedding), which uses contrastive learning with demographic-aware sampling to obtain bias-aware embeddings via prompt tuning.

## Strengths

- **Novel and important empirical finding**: The discovery that example selection for ICL can increase *worst-case* bias even while improving accuracy on average is a significant contribution that the fairness community should be aware of. The experiments across 8 LLMs (LLaMA-2-7/13/70B, OPT-6.7/13/30B, GPT-J-6B, GPT-neo-2.7B) and 4 example selection methods (Random, Similarity, Perplexity, DPP) provide solid empirical breadth.

- **Clear problem formulation and analysis pipeline**: The paper systematically isolates potential sources of bias through the null-prompt experiment (Figure 4), attempting to separate the LLM's native bias from bias induced by example selection. The use of AvgGF, MaxTG, and MaxFG metrics is appropriate for capturing different aspects of fairness violations.

- **Practical compatibility with existing methods**: ReBE is designed to work alongside existing example selection strategies rather than replace them. Table 5 demonstrates that DPP+ReBE achieves both higher accuracy (0.87) and lower maximum bias (MaxTG: 0.250 vs 0.273 baseline) compared to DPP alone, showing the method can improve fairness while maintaining task performance.

## Weaknesses

- **Maximum vs. mean bias framing requires clearer justification**: The paper's central claim—"example selection amplifies the biases of LLMs"—relies specifically on *maximum* bias values, while Figure 2 clearly shows that *mean* bias typically *decreases* with example selection. This distinction is mentioned in Section 3.3 but not foregrounded in the abstract or introduction. Relying on maximum values over random seeds is methodologically fragile: if zero-shot evaluation uses fewer seeds than few-shot conditions, the comparison is asymmetric. The paper does not clearly explain what varies across seeds for zero-shot (since no examples are selected), making the baseline comparison unclear.

- **ReBE requires demographic attribute labels during training, a practical limitation not acknowledged**: Section 4.1 specifies that ReBE requires (x, y, s) where s is demographic information. In real deployment, obtaining ground-truth demographic labels is often impossible or ethically problematic. This constraint significantly limits practical applicability but is never discussed as a limitation.

- **Inconsistent debiasing results undermine effectiveness claims**: Table 3 shows that ReBE *increases* average bias metrics for Perplexity-based selection: GPT-J-6B shows AvgGF +0.024, MaxTG +0.060, MaxFG +0.079 (red subscripts). Similarity-based selection shows MaxTG +0.047 increase for GPT-neo-2.7B. The abstract's claim that "ReBE effectively mitigates biases" overstates the evidence—results are inconsistent across selection methods, and the paper offers no explanation for these failures.

- **Contrastive loss design (Equation 4) lacks justification for asymmetric negative set**: The bias-contrastive loss defines negatives as A(i) = {k: y_k ≠ y_i, s_k = s_i}—same demographic, different label. This asymmetrically excludes cross-demographic, different-label pairs from the negative set. Standard SupCon uses all non-positives as negatives. The paper provides no ablation on alternative negative set definitions to validate this choice.

- **Race bias results relegated to appendix**: The paper prominently includes race as a key motivation and constructs EEC-paraphrase with race attributes, yet all race debiasing results are deferred to Appendix D.2. Given the paper's stated scope, these should appear in the main results.

- **Large models excluded from debiasing experiments**: OPT-30B and Llama-2-70B are excluded due to hardware limitations, leaving the largest deployed models untested. The paper's claims about ReBE's effectiveness cannot be verified for models commonly used in practice.

- **No statistical significance testing**: Bias comparisons rely on point estimates without confidence intervals or significance tests. The maximum bias metric is particularly vulnerable—single pathological seeds can drive results.

## Nice-to-Haves

- **Sensitivity analysis for the α hyperparameter**: While referenced in Appendix D.3, a brief treatment in the main text would clarify the stability of the accuracy-fairness trade-off.

- **Comparison with standard debiasing methods**: The paper compares only against context augmentation baselines. Comparison with inference-time intervention or representation engineering methods would better position ReBE's contribution.

- **Human validation of EEC-paraphrase**: Since the dataset is GPT-3.5-generated, human validation that bias signals were not inadvertently altered during paraphrasing would strengthen confidence in the benchmark.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Weakness about "no dedicated limitations section"**: This is a formatting convention, not a substantive flaw.

- **Weakness about the name "Remind"**: Naming choices are not methodological issues.

- **Weakness about k=18 being unusually large for few-shot**: Figure 7 explicitly shows results across different k values, and the parameter analysis addresses this. The default was empirically chosen based on the accuracy-bias trade-off.

- **Weakness demanding theoretical analysis of L_bias**: This is an empirical systems paper; theoretical proofs of disentanglement are not standard requirements.

- **Request for human evaluation of dataset quality**: While valuable, human evaluation is not required for bias benchmark papers—the EEC original dataset already established the template-based approach.

- **Criticism about "no explanation for ReBE failures on Perplexity"**: While the results show inconsistency, demanding an explanation for negative results is scope creep—the empirical observation itself is a valid finding.

## Novel Insights

Beyond the paper's own contributions, the distinction between *mean* and *maximum* bias reveals an important tension in fairness work: methods that improve average-case fairness may still create rare but severe fairness violations. The paper's focus on maximum bias is well-motivated from a worst-case fairness perspective, but the methodological asymmetry between deterministic zero-shot evaluation and stochastic few-shot evaluation (varying seeds) needs explicit resolution. If zero-shot has no seeds while few-shot has many, comparing maximum values is statistically invalid—the maximum of a distribution with more samples will tend to be higher regardless of whether the underlying distribution differs.

## Suggestions

- **Clarify zero-shot baseline methodology**: Explicitly state what varies across "random seeds" for zero-shot evaluation. If nothing varies, report mean and variance across multiple zero-shot runs (varying test subsets) or clearly state that zero-shot is deterministic and compare against the *distribution* of few-shot bias values (e.g., showing zero-shot is within the few-shot distribution).

- **Acknowledge the demographic label requirement**: Add discussion of ReBE's training-time dependency on demographic attributes as a practical limitation, with potential workarounds (e.g., inferred demographics, synthetic data).

- **Report statistical uncertainty**: Add confidence intervals or bootstrap standard errors for bias metrics, especially for maximum values. Consider comparing distributions via statistical tests rather than point estimates.

- **Explain selection-dependent failures**: Investigate why ReBE fails for Perplexity-based selection but works for DPP and Random. This could reveal important interactions between example selection strategies and debiasing effectiveness.

# Actual Human Scores
Individual reviewer scores: [5.0, 3.0, 6.0]
Average score: 4.7
Binary outcome: Reject
