Now I have enough information to write the comprehensive final review. Let me synthesize everything.

## Summary

This paper introduces a method for measuring the faithfulness of LLM explanations at the concept level, defining "causal concept faithfulness" as the alignment between the causal effects of input concepts on model answers (CE) and the rate at which explanations mention those concepts (EE). The method uses an auxiliary LLM (GPT-4o) to extract concepts, generate counterfactual questions, and parse explanations, combined with a Bayesian hierarchical model for sample-efficient estimation. Experiments on BBQ and MedQA reveal novel patterns of unfaithfulness, including that LLM explanations hide the influence of safety measures and social bias.

## Strengths

- **Concept-level faithfulness measurement is a genuine advance over token-level approaches.** By measuring faithfulness at the level of high-level concepts rather than tokens, the method produces results that are directly interpretable and practically actionable. The category-level analysis (Behavior, Identity, Context in BBQ; Clinical Tests, Symptoms, Demographics in MedQA) reveals distinct patterns of unfaithfulness per category, as shown in Figures 1 and 3.

- **Novel empirical findings about LLM unfaithfulness.** On BBQ, the paper discovers a new pattern — LLM explanations hide the influence of safety measures — that was not reported in the original Turpin et al. (2023) study of the same dataset (Figure 2, middle vs. left plots). On MedQA (Table 3), the paper reveals that Claude-3.5-Sonnet's explanations never mention "the patient's mental status upon arrival" (EE=0.00) despite it having the largest causal effect (CE=0.32). These are genuine new insights.

- **Formal definitions grounded in causal inference.** The three definitions (CE via KL divergence in Def. 2.1, EE in Def. 2.2, faithfulness as PCC in Def. 2.3) provide a principled foundation that goes beyond prior work such as Turpin et al. (2023) which used dataset-specific tests, and Siegel et al. (2024) / Atanasova et al. (2023) which operated at the token level.

- **Bayesian hierarchical model with principled uncertainty quantification.** The hierarchical model partially pools information across questions and concept categories (Section 3), producing 90% credible intervals for all faithfulness estimates. This is important for safety-critical contexts and provides transparency about estimate precision.

- **Method works with opaque, API-accessible models.** The problem setting (Section 2) explicitly assumes the model is opaque and only queryable through discrete samples, enabling evaluation of closed-source models like GPT-4o and Claude-3.5-Sonnet.

- **Compelling motivating example.** Table 1 effectively illustrates the unfaithfulness problem with a concrete, memorable gender-swapping example showing GPT-3.5 prefers the female candidate regardless of gender assignment while never mentioning gender in explanations.

## Weaknesses

### Fatal
None.

### Major

- **Unvalidated auxiliary LLM pipeline.** The method relies on GPT-4o for three critical steps: concept extraction, counterfactual question generation, and explanation parsing. No systematic human validation of accuracy is provided for any of these steps. Errors at any stage propagate through the pipeline — missed concepts become invisible, parsing errors inflate or deflate EE. The paper acknowledges in Section 6 that the auxiliary LLM "sometimes contains errors" and provides examples in Appendix F.3, but this falls short of a systematic accuracy assessment. For a method whose core claim is to provide *causal* faithfulness measurement, the lack of any validation that these pipeline steps are reliable undermines confidence in the reported numerical results. While the most stark qualitative findings (e.g., explanations *never* mention gender despite high CE) are likely robust to moderate parsing errors, the precise CE/EE values and faithfulness scores depend on unvalidated LLM judgments.

- **Unvalidated clean interventions undermine the causal interpretation.** The entire framework depends on the disentanglement assumption (Section 2): that counterfactuals change only the target concept C_m while holding all other concepts fixed. If the auxiliary LLM inadvertently modifies other concepts when editing a question, the measured "causal effect" is actually the effect of a bundle of changes. The paper acknowledges correlated concepts as a limitation in Section 6 and Appendix F.2, but frames it as a minor future-work issue. For a method whose core contribution is a *causal* definition of faithfulness, the lack of any validation that interventions satisfy causal identification conditions (e.g., re-extracting concepts from generated counterfactuals to verify only the target changed) is a significant gap. Without this, the "causal" label on "causal concept effect" is not fully earned.

### Minor

- **Gap between theoretical definition of CE and practical estimation.** Definition 2.1 defines CE as the KL divergence averaged over *all* counterfactual values c'_m ∈ ℂ'_m, but in practice only 1–2 counterfactuals per concept are generated (a removal and a swap for BBQ; only a removal for MedQA). When |ℂ'_m| could be very large (e.g., all plausible pairs of working ages), averaging over 1–2 values may not reliably estimate the defined quantity. The paper does not analyze how this subsampling affects the estimate's bias or convergence. However, this gap between theoretical ideal and practical estimation is common in applied causal inference, and the qualitative findings do not depend on the precise CE values.

- **Wide credible intervals undermine model comparison claims.** On BBQ, the 90% CIs overlap substantially: GPT-3.5 F = 0.75 [0.42, 1.00], GPT-4o F = 0.56 [0.24, 0.86], Claude F = 0.62 [0.28, 0.91]. On MedQA: GPT-3.5 F = 0.50 [0.18, 0.77], GPT-4o F = 0.34 [0.05, 0.65], Claude F = 0.30 [0.01, 0.59]. The paper's claim that "GPT-3.5 produces more faithful explanations than the two more advanced models" is not well-supported by these overlapping intervals. This affects only the comparative ranking claims; the qualitative patterns (which concepts are misattributed) are visible in scatter plots regardless.

- **Inconsistency between EE definition text and formula.** Definition 2.2's text states "The probability that M's explanations in response to original input x and to counterfactual questions...imply that C_m is causal," but the formula EE(x, C_m) = (1/|ℂ'_m|) Σ_{c'_m ∈ ℂ'_m} P_M(C_m ∈ E | x_{c_m→c'_m}) averages only over counterfactual questions, excluding the original. The text and formula are inconsistent; either the original should be included in the sum or the text should be corrected.

- **No comparison to existing faithfulness methods.** The paper proposes a new faithfulness metric but does not compare it against any prior method (e.g., the token-level approach of Atanasova et al. 2023, or Siegel et al. 2024). While the concept-level approach is conceptually distinct, some comparison (even against a simple baseline) would help establish whether the approach provides practical advantages beyond its conceptual appeal.

### Trivial
None.

## Nice-to-Haves

- A small-scale human validation study (even 50–100 judgments) of the critical pipeline steps — concept extraction, counterfactual quality, and explanation parsing — would substantially strengthen confidence in the method.
- A robustness analysis showing how CE estimates change when different counterfactual values are selected (e.g., different swap pairs for gender).
- A comparison to an existing faithfulness method, even a simple token-deletion baseline, to demonstrate the practical advantages of concept-level analysis.
- Re-extracting concepts from generated counterfactuals to verify that only the target concept changed, as a basic sanity check on intervention quality.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that EE averaging over counterfactuals "conflates faithfulness of the explanation a user actually receives with faithfulness of explanations for questions the user never asked."** This is a design choice, not a flaw. The paper is measuring faithfulness as a property of the model-question pair across the concept's domain, which is consistent with the causal inference framing. The concept that a model's faithfulness should be evaluated across counterfactual variants is well-motivated by the problem setting.

- **Harsh critic's claim that "hiding safety measures" is "inferred rather than demonstrated."** The paper uses appropriately hedged language ("it appears that the presence of social identity information...contributes to the models' refusal") and provides direct evidence in Figure 2. This is not an overclaim.

- **Harsh critic's claim about z-scoring with small |C| producing "extremely noisy standardized scores."** The paper explicitly acknowledges this concern (Section 3: "since the number of concepts per question is often small, this can lead to unreliable estimates") and the hierarchical model is designed to address it. The hierarchical model's partial pooling provides more stable estimates than per-question z-scoring alone.

- **Harsh critic's claim that the hierarchical model "may mask genuine question-level variation."** This misunderstands the purpose of hierarchical models. Partial pooling shrinks extreme estimates toward the mean but still captures question-level variation — it produces *better* estimates of question-level faithfulness by borrowing strength across questions, rather than masking it.

- **Harsh critic's claim that the introduction "overstates" by implying the method reveals *why* models are unfaithful.** The introduction says the method can "reveal semantic patterns of unfaithfulness" and "the ways in which" explanations are misleading, which is what the method actually does. The "why" interpretation is the reviewer's inference.

- **Harsh critic's claim that removal-only counterfactuals for MedQA "bias CE estimates toward zero."** The paper provides a reasonable justification for this design choice ("changing the values of clinical concepts could introduce subtle changes that are hard to assess the implications of"). This is a reasonable scope decision, not a flaw.

- **Strength Finder's claim about "open-source code repository" enabling reproducibility.** Generic strength without specific evidence of code quality or completeness.

- **Strength Finder's claim that "method works with opaque, API-accessible models" as a "supporting" strength.** This is implicit in the problem setting and not a novel contribution of the paper.

## Novel Insights

The most insightful observation across the reviews is the fundamental tension in this paper: the qualitative pattern analysis (which is the paper's strongest contribution) is largely robust to the validation gaps that threaten the numerical results. When a model's answer distribution shifts dramatically upon removing gender information but its explanations never mention gender, this finding is visible in raw answer distributions and explanation text — it does not require a perfectly calibrated pipeline to be meaningful. The paper would be stronger if it leaned more into this qualitative robustness rather than presenting precise faithfulness scores whose precision is not well-supported by the evidence. A reader should trust the scatter plots and intervention visualizations more than the point estimates.

## Suggestions

- Add even a small human validation study (50–100 judgments) of the auxiliary LLM's concept extraction and explanation parsing accuracy. This would go a long way toward addressing the biggest concern.
- Implement a simple automated sanity check: re-extract concepts from generated counterfactuals and verify that only the target concept changed. This is straightforward and would address the clean-intervention concern.
- Tone down model comparison claims given the overlapping CIs. Focus on the qualitative patterns, which are the paper's true contribution.
- Correct the inconsistency between Definition 2.2's text (mentioning original input x) and formula (averaging only over counterfactuals).

## Evaluation

**Originality:** The concept-level faithfulness framework and formal causal definitions are novel. The concept-level approach is a genuine advance over prior token-level methods. The discovery of "hiding safety measures" as a distinct unfaithfulness pattern is novel.

**Importance of research question:** Measuring LLM explanation faithfulness is an important and timely problem with clear safety implications. The paper's emphasis on semantic patterns (not just aggregate scores) adds practical value.

**Claims support:** The qualitative claims (which concepts are misattributed, what patterns of unfaithfulness exist) are well-supported by the experimental evidence. The quantitative claims (model rankings, precise faithfulness scores) are less well-supported due to wide CIs and unvalidated pipeline.

**Soundness of experiments:** The Bayesian hierarchical model is methodologically sound. The main gap is the unvalidated auxiliary LLM pipeline and lack of comparison to existing methods.

**Clarity of writing:** The paper is well-organized with clear motivating examples. The formal definitions are precise. The inconsistency in Definition 2.2's text vs. formula is a minor blemish.

**Value to research community:** The paper opens a promising direction for concept-level faithfulness analysis and provides novel empirical findings. The framework could support future methodological improvements.

## Calibration Anchors

| Paper | Score | Comparison |
|-------|-------|-----------|
| Sparse Feature Circuits (I4e82CIDxv) | 8.0 | Much stronger: thoroughly validated, scaled to thousands of circuits, with downstream applications. This paper lacks comparable validation and scale. |
| First-Person Fairness in Chatbots (TlAdgeoDTo) | 7.25 | Stronger: similarly uses LLM-as-research-assistant, but validates with independent human annotations across millions of interactions. The validation gap is the key differentiator. |
| Gaussian Concept Subspace (CvttyK4XzV) | 6.75 | Roughly comparable: both have concept-level interpretability with novel formal frameworks but experimental limitations. This paper has more novel findings but less robust evaluation. |
| Hardness of Faithful CoT (1OyE9IK0kx) | 5.0 | Weaker: studies LLM faithfulness but with limited novelty in findings. This paper's formal framework and novel patterns (hiding safety measures) are clearly above this level. |
| RATE (UnpxRLMMAu) | 5.0 | Weaker: also uses LLM-generated counterfactuals for causal measurement but has weaker experimental validation and more convoluted presentation. This paper's qualitative findings and cleaner formulation place it above. |
| TeLLMe: LLMs Explain Neurons (01ep65umEr) | 5.25 | Weaker: similar LLM-based concept explanation pipeline with validation concerns, but less formal rigor. This paper has a stronger theoretical foundation. |
| Textual Data Valuation (OdoS6cH8MP) | 2.0 | Much weaker: unvalidated pipeline on only synthetic data, essentially no real contribution. This paper is clearly above this level. |

The paper sits above the 5.0–5.5 cluster of faithfulness/interpretability papers with unvalidated LLM-based pipelines due to its stronger formal framework, novel qualitative findings, and cleaner experimental design. It sits below the 7.0+ cluster due to the absence of human validation of the measurement pipeline. The 6.0–6.5 range appropriately reflects genuine contributions offset by significant validation gaps.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>