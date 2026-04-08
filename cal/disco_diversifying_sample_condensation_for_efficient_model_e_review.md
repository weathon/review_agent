=== CALIBRATION EXAMPLE 36 ===

# Final Consolidated Review
## Summary

DISCO proposes a method for efficient ML model evaluation by selecting benchmark subsets based on inter-model disagreement (measured via JSD or PDS) rather than sample representativeness, then predicting full benchmark performance from model signatures (concatenated outputs) on the selected subset using simple predictors (kNN, Random Forest). The method achieves state-of-the-art performance prediction on MMLU, HellaSwag, Winogrande, and ARC, and generalizes to ImageNet in the vision domain.

## Strengths

- **Conceptual clarity of the core insight**: The shift from selecting representative samples (clustering-based) to selecting disagreement-inducing samples is a clean, well-motivated departure from prior work (tinyBenchmarks, Anchor Points). Proposition 1 provides a principled information-theoretic justification, and the empirical results validate that this conceptual shift yields practical gains.

- **Consistent cross-domain effectiveness**: The method is validated on both language benchmarks (424 LLMs across 4 datasets) and vision (400 ImageNet models), achieving strong results in both domains. On MMLU with 100 samples, DISCO achieves 1.07%p MAE and 0.987 Spearman rank, substantially outperforming the best prior method (2.08%p MAE / 0.927 rank from Anchor-corr + gp-IRT).

- **Simplicity of the prediction pipeline**: Using model signatures with Random Forest/kNN instead of IRT-based latent parameter estimation is both simpler and more effective. The ablation in Table 2(e) shows this isn't just a theoretical preference — RF with signatures outperforms all alternatives including complex IRT models.

## Weaknesses

- **The injectivity assumption in Proposition 1 limits the theoretical "optimality" claim**: The abstract states that "inter-model disagreement provides an information-theoretically optimal rule for greedy selection." However, Proposition 1 requires $S(m)$ (model accuracy) to be injective with respect to model index $m$. With 424 models on a bounded accuracy scale, ties are inevitable. Without injectivity, maximizing JSD maximizes information about *which model* produced the response, not necessarily about its *accuracy value*. The paper should explicitly caveat the "optimal" claim to reflect this assumption, or provide bounds on the error introduced by non-injective accuracy mappings.

- **Substantial performance degradation under distribution shift between source and target models**: Appendix F reveals that when target models significantly outperform source models (the "performance w/ gap" split), Spearman rank drops from 0.987 to 0.892 — a meaningful degradation. The authors dismiss this scenario as "unrealistic," but rapid capability jumps (e.g., new SOTA models) are common in practice. This is a genuine failure mode that deserves more than dismissal, especially since the method is designed for deployment in evolving evaluation ecosystems.

- **The offline compute cost creates a practical deployment barrier for smaller users**: The offline stage requires 3284 GPU-hours (Table 4), with a break-even point of ~389 model evaluations (Appendix B.3). This means individual researchers or small labs evaluating fewer than 389 models would incur more total cost than direct evaluation. While the paper argues large-scale training runs justify this, the break-even analysis is buried in the appendix despite being central to the method's practical utility claims.

- **The dramatic improvement on ImageNet may partly reflect implicit class coverage rather than pure disagreement-based informativeness**: The random-sampling baseline on ImageNet achieves only 0.652 Spearman rank (Table 3) versus 0.916 on MMLU (Table 1). With 1000 ImageNet classes and only 100 samples, random subsets miss most classes entirely. PDS-based selection likely implicitly provides class coverage by picking samples where models disagree — which tends to be hard or rare classes. This raises the question: is DISCO's advantage on ImageNet primarily due to better disagreement-based informativeness, or simply to avoiding class-collapse? This deserves explicit analysis.

## Nice-to-Haves

- **Characterize what DISCO actually selects**: Analyze selected samples by difficulty, topic, and class distribution. This would verify whether high-PDS samples are genuinely informative or merely ambiguous/noisy, and would clarify the ImageNet improvement mechanism.

- **Empirically verify Proposition 1**: Correlate per-sample JSD/PDS scores with actual per-sample contribution to prediction accuracy. This would bridge the theoretical justification and empirical results.

- **Cost-matched comparisons**: Show whether simpler methods (e.g., random sampling with more online samples matched to DISCO's total compute budget) achieve comparable accuracy, to contextualize the offline investment.

- **Per-task breakdown on MMLU**: Report DISCO's prediction accuracy across MMLU's 57 subtasks to reveal whether the method works uniformly or fails on specific capability areas.

- **Source model diversity requirements**: Table 2(c) varies model count but not diversity. Testing performance when source models come from only 1–2 architecture families would reveal how much DISCO depends on source pool heterogeneity.

## Removed Points

- **Metabench comparison fairness**: The reviewer flagged that Metabench requires more examples (150–450) vs. DISCO's 100. Per the rules, this asymmetry favors the baseline (Metabench gets more data), not the authors' method. The paper transparently discloses this in Table 1's footnote. Removed as an unfair comparison concern.

- **Restriction to closed-choice tasks as a weakness**: The paper explicitly scopes this out in Section 6 (Limitations): "DISCO requires predictive probabilities for several predefined answer choices." Criticizing the absence of open-ended generation support is scope creep beyond the paper's stated contribution.

- **Logit access for closed-source models**: The paper acknowledges the need for predictive probabilities. Demanding applicability to API-only models with text-only outputs is beyond scope.

- **Missing multimodal benchmark experiments**: The paper claims domain-agnostic applicability and demonstrates it on both language and vision domains. The LMMs-Eval reference in the introduction motivates the problem, not a specific experimental claim. Requesting additional multimodal experiments is scope creep.

- **Proposition 2 tightness scaling with M**: This is a mathematical observation about bound looseness. The empirical results validate that PDS works well in practice; the theoretical tightness of bounds is not a practical concern.

- **Confidence intervals absent from Table 1**: The paper provides confidence intervals in Appendix D. Requesting them in the main table is a formatting preference, not a substantive weakness.

## Novel Insights

The cross-domain gap in random-sampling performance (0.652 on ImageNet vs. 0.916 on MMLU) reveals something important about benchmark structure: when the number of classes far exceeds the sample budget, random selection catastrophically fails, and disagreement-based selection may be effective partly because it implicitly ensures class coverage. This suggests that the value of DISCO's selection criterion has two components — (1) selecting samples that maximally differentiate models (the stated information-theoretic signal) and (2) providing distributional coverage over the output space (a less acknowledged structural benefit). Disentangling these two factors would sharpen the contribution.

## Suggestions

- Temper the "information-theoretically optimal" claim in the abstract to acknowledge the injectivity assumption, or add a remark in Section 4.1.2 bounding the error when $S(m)$ is not injective.

- Move the break-even analysis (Appendix B.3) into the main paper or prominently discuss it in Section 5, as it directly affects the practical applicability of the method.

- Add an analysis of DISCO-selected samples on ImageNet to determine whether PDS selection provides implicit class coverage — e.g., report the number of unique classes covered by the top-100 PDS samples vs. random samples.

- Report per-task MMLU results or at least a variance breakdown to show whether DISCO's gains are uniform or concentrated in specific task categories.

- Provide practical guidance on minimum source model pool requirements (size and diversity) for new benchmarks, to help practitioners decide when DISCO is worth deploying.

---

**Assessment on key axes:**

- **Novelty**: Moderate-to-high. The shift from representativeness to disagreement is a clean conceptual contribution. Model signatures as a direct prediction mechanism simplify prior IRT-based approaches.

- **Technical soundness**: The theoretical framework is sound under stated assumptions, but the injectivity assumption is non-trivial and should be better caveated. Empirical methodology is thorough with appropriate baselines, ablations, and cross-domain validation.

- **Empirical support**: Strong. Results are consistent across 4 language benchmarks and ImageNet, with clear margins over prior methods. The performance gap scenario in Appendix F is honestly reported but deserves deeper discussion.

- **Significance**: High. Reducing evaluation cost by >99% with minimal accuracy loss addresses a critical and growing bottleneck. The method is practical for evaluation hubs and large-scale training pipelines.

- **Clarity**: Good. The paper is well-structured with clear problem setup, method description, and experiments. The theoretical sections are appropriately formal without being impenetrable.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 8.0]
Average score: 6.7
Binary outcome: Accept
