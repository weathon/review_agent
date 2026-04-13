=== CALIBRATION EXAMPLE 13 ===

# Final Consolidated Review
## Summary

This paper introduces "causal concept faithfulness," a metric that quantifies the alignment between the causal effects of input concepts on LLM outputs and the concepts cited in natural language explanations. The authors propose a Bayesian hierarchical estimation method using LLM-generated counterfactuals to measure faithfulness at both question- and dataset-levels, and demonstrate the approach on social bias (BBQ) and medical QA (MedQA) tasks.

## Strengths

- **Concept-level alignment with human interpretation:** By defining faithfulness at the level of high-level concepts rather than tokens, the method aligns with how LLMs generate explanations and how humans interpret them. The formal definitions of Causal Effect (CE) and Explanation-implied Effect (EE) in Section 2 provide a principled grounding.

- **Semantic pattern discovery beyond scalar scores:** The method successfully identifies specific ways models are unfaithful—for example, Figure 2 shows that GPT models hide the influence of safety measures (refusing biased questions while citing ambiguity) and social bias, which a scalar faithfulness score alone would not reveal.

- **Hierarchical modeling for sample efficiency:** The Bayesian hierarchical approach pools information across questions to produce more stable estimates, which is necessary given the cost of LLM sampling. Appendix D.4 provides some evidence that estimates stabilize around N≥15.

## Weaknesses

- **Small sample size limits generalizability:** Both experiments use N=30 questions subsampled from larger datasets. While cost constraints are understandable, the wide credible intervals (e.g., GPT-4o faithfulness [-0.92, 0.28] in Table 2) mean most model comparisons lack statistical separation. The claim that GPT-3.5 is "more faithful" than GPT-4o has substantially overlapping CIs (0.75 [0.42, 1.00] vs. 0.56 [0.24, 0.86]).

- **No quantitative validation of auxiliary LLM accuracy:** The method relies on GPT-4o for concept extraction, counterfactual generation, and mention classification. While Appendix F.3 discusses qualitative errors, there is no quantitative evaluation (e.g., human agreement, precision/recall) for any of these steps. Errors propagate multiplicatively through the pipeline.

- **Circular evaluation when GPT-4o evaluates GPT-4o:** The same model serves as both auxiliary evaluator and target. While Appendix D.6 shows preliminary results with Llama-3.1-8B as a target, there is no ablation testing whether GPT-4o-as-auxiliary systematically biases results for GPT-4o-as-target.

- **PCC limitations with small concept sets:** Faithfulness is measured via Pearson correlation over typically 3-5 concepts per question. PCC is unstable with small vectors and insensitive to systematic miscalibration—a model that uniformly over-mentions every concept would achieve high faithfulness if the relative ordering correlates with causal effects, despite being misleading about actual influence.

- **Correlated concepts violate causal assumptions:** Section 2 assumes concepts are disentangled (each can change independently). The limitation section acknowledges this, but the severity is not characterized. In MedQA, concepts like "mental status" and "refusal of treatment" may be semantically linked—removing one may implicitly alter another, confounding CE estimates.

- **Removal-only counterfactuals in MedQA limit interpretability:** For MedQA, only removal counterfactuals are used because "changing clinical values is hard to assess." This measures a different quantity than BBQ's replacement counterfactuals—absence effects versus change effects. A concept with high removal-based CE could indicate diagnostic importance or prompt ambiguity.

- **Method measures "mention" not "correct reasoning":** If an explanation correctly mentions an influential concept but uses it in logically incorrect ways, the method scores it as faithful. The CE/EE framework captures citation alignment, not reasoning correctness.

## Nice-to-Haves

- **Comparison to baseline faithfulness metrics:** The paper claims conceptual novelty over Siegel et al. (2024), but does not empirically compare against simpler metrics (e.g., token-overlap faithfulness, sufficiency/comprehensiveness) to demonstrate added value from the hierarchical approach.

- **Validation that faithfulness scores predict downstream utility:** It remains unclear whether higher causal concept faithfulness correlates with improved human decision-making or trust calibration.

- **Analysis of concept granularity sensitivity:** Different concept extraction granularities would produce different CE/EE distributions and faithfulness scores; this sensitivity is not analyzed.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Asymmetric EE definition criticism:** The reviewer claimed the EE definition inappropriately excludes the original question. However, the formula correctly averages over counterfactuals to measure whether concept citation is consistent across perturbations—this is appropriate for the intended purpose.

- **"First method" claim overstated:** While the introduction states this is "the first method" to identify semantic patterns, the related work (Section 5) acknowledges Siegel et al. (2024) and differentiates the approach (concept-level vs. token-level edits, original vs. counterfactual question faithfulness). The differentiation is adequate.

- **"Safety measure hiding" not genuinely novel:** The critic argued this finding is well-known from RLHF literature. The paper's contribution is demonstrating this pattern via their measurement framework, providing quantitative evidence rather than anecdotal observation—the methodological contribution is valid.

## Novel Insights

The hierarchical faithfulness estimation approach reveals that different models are unfaithful in *different ways*—GPT-3.5 hides social bias but is otherwise relatively faithful to behavioral concepts, while GPT-4o and Claude show more diffuse unfaithfulness patterns. This suggests that aggregate faithfulness scores may be less actionable than the component-level (CE vs. EE per concept category) analysis, and that safety interventions might have unintended faithfulness consequences. The finding that newer models can be *less* faithful than older ones (GPT-3.5 vs. GPT-4o on BBQ) raises important questions about how safety alignment affects explanation behavior.

## Suggestions

- Conduct a sensitivity analysis using different auxiliary LLMs (e.g., Claude evaluating GPT-4o) to quantify potential circularity bias.
- Report inter-rater agreement or precision/recall for the auxiliary LLM's concept extraction and mention classification against human annotations on a subset.
- Include formal statistical tests (or at least credible interval differences) when comparing faithfulness across models, rather than relying on point estimates.
- Explicitly distinguish between removal-based and replacement-based CE in the interpretation, as they measure different notions of influence.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 6.0]
Average score: 7.5
Binary outcome: Accept
