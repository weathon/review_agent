=== CALIBRATION EXAMPLE 3 ===

# Final Consolidated Review
## Summary
This paper proposes **causal concept faithfulness**, a concept-level framework for assessing whether an LLM’s natural-language explanations align with the concepts that empirically affect its answers under counterfactual edits. The method combines an auxiliary-LLM pipeline for concept extraction, counterfactual generation, and explanation labeling with Bayesian hierarchical models to estimate question-level and dataset-level faithfulness in black-box API settings. Empirically, the paper applies the method to BBQ and MedQA and surfaces interpretable patterns of unfaithfulness, including cases where explanations omit socially influential concepts and cases where explanations seem to conceal the role of safety-related refusal behavior.

## Strengths
- **The paper makes a specific methodological move that is genuinely useful for explanation auditing: shifting from token-level perturbations to concept-level interventions.** This is well matched to the object being studied—natural-language explanations often refer to semantically meaningful concepts rather than tokens—and enables findings like “identity is omitted despite high effect” rather than hard-to-interpret token saliency patterns.
- **The method is designed for the realistic black-box setting and addresses sample efficiency nontrivially.** The Bayesian hierarchical pooling over concept categories and questions is not just generic sophistication; it directly targets the high-cost, low-sample regime created by API-only access and repeated counterfactual querying.
- **The paper goes beyond scalar faithfulness scores and extracts semantic failure modes.** The BBQ analysis is especially compelling here: the method does not merely say explanations are imperfect, but distinguishes patterns such as repeated omission of identity concepts and behavior concepts being cited regardless of their actual effect.
- **The empirical case studies are insightful enough to show the value of the framework even when aggregate scores are only moderate.** In particular, the paper’s observation that a model with a somewhat higher overall faithfulness score can still be unfaithful in a more harmful way (e.g., masking social bias) is an important and concrete contribution.
- **The paper is explicit about scope and limitations in several places rather than hiding them.** It directly notes the use of only 30 questions per dataset, dependence on an auxiliary LLM, prompt engineering needs, and failures under correlated concepts; these are meaningful admissions tied to the actual method rather than boilerplate caveats.

## Weaknesses

### Fatal
None.

### Major:
- **The paper’s headline framing overstates what is being measured.** The formal object is not faithfulness to the model’s internal reasoning process in any strong mechanistic sense; it is alignment between (i) concept-level effects on the **answer distribution** under interventions and (ii) whether explanations **indicate those concepts as influential**. The paper sometimes slides from this operational notion to claims about the model’s “reasoning” more broadly. The formalization itself is coherent, but the framing should be narrower and more explicit that this is a behavioral/causal-intervention notion of explanatory faithfulness, not a direct measurement of latent reasoning.
- **The causal interpretation relies on a strong single-concept intervention assumption that is materially stressed in the showcased domains.** Section 2 explicitly assumes concepts are “disentangled,” and Section 6 acknowledges failures under correlated concepts. This is not merely a peripheral limitation: several highlighted examples involve concepts that are plausibly entangled (e.g., social identity and interpreted behavior in BBQ; mental status, refusal of treatment, and other clinical findings in MedQA). When a concept cannot be changed while meaningfully holding the rest of the case fixed, CE becomes less convincing as the “true influence” target against which explanations are judged.
- **The paper’s measurement pipeline is heavily dependent on the auxiliary LLM, but the validation of that pipeline is too limited for a measurement paper.** GPT-4o is used for concept extraction, alternative-value generation, counterfactual rewriting, concept categorization, and explanation labeling. Section 6 admits the auxiliary model “sometimes contain[s] errors,” but the main paper provides no substantial human validation of concept extraction quality, no systematic check of counterfactual validity, and no sensitivity analysis to changing the auxiliary model. Since the paper’s main contribution is a measurement framework, this weakens confidence that the discovered semantic patterns reflect the evaluated model rather than upstream pipeline choices.
- **Empirical support is promising but still narrow relative to the breadth of the claims.** The method is evaluated on two multiple-choice/contextual QA settings with 30 sampled questions each and three closed models. That is enough to demonstrate potential, but not enough to establish broad applicability as a general-purpose way to assess explanation faithfulness or to strongly support model-selection use cases emphasized in the introduction.

### Minor
- **There is no experimental comparison to existing faithfulness approaches.** The related work section positions the method clearly, but without even a limited baseline comparison, it remains unclear how much the concept-level framework changes conclusions relative to simpler or prior intervention-based methods.
- **Some of the comparative model claims are stronger than the uncertainty warrants.** For example, on BBQ the reported credible intervals for dataset-level faithfulness overlap substantially, so statements that one model is “more faithful” than another should be phrased more cautiously.
- **PCC is a reasonable but imperfect alignment metric.** Correlation captures relative alignment but not calibration: a system could systematically over-mention concepts yet still score highly if rankings align. The hierarchical model helps with small per-question concept sets, but the paper does not really analyze the behavioral consequences of choosing PCC over alternatives.
- **The “hiding the influence of safety measures” interpretation is intriguing but somewhat speculative as presented.** The observed behavior after identity removal is suggestive, but the paper does not fully rule out alternative explanations such as the edited question simply becoming easier or less ambiguous in a way unrelated to safety behavior per se.
- **Scope is limited to context-based multiple-choice QA with answer-choice distributions.** The paper does state this setting in Section 2, so this is not a flaw in execution, but it does narrow significance relative to the broader title and motivation around LLM explanations generally.

### Trivial
- **Definition 2.2 appears textually inconsistent about whether EE includes the original input as well as counterfactuals.** The surrounding prose says “original input and counterfactual questions,” while the displayed formula averages over counterfactuals. This is likely clarifiable, but the definition should be made fully consistent in the final version.

## Nice-to-Haves
- A small **human validation study** of concept extraction, counterfactual faithfulness, and explanation-concept labeling on a random subset would substantially strengthen the method paper.
- A **sensitivity analysis over auxiliary models** would directly test the most important source of pipeline fragility.
- A **baseline comparison** against at least one prior perturbation/correlation-style faithfulness method would help establish the added value of concept-level analysis.
- Reporting **an additional alignment metric** (e.g., rank correlation or a calibration-sensitive variant) would clarify whether conclusions depend strongly on PCC.
- A brief, explicit **normative discussion of what counts as “unfaithful” when safety-aligned refusals influence answers** would improve the interpretation of the most novel empirical finding.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The formula in Definition 2.2 is definitely wrong / the paper ignores original explanations.”**  
  The paper text does contain a mild inconsistency, but it is not enough to support a substantive criticism that the method is conceptually broken. This is better treated as a clarification request rather than a major flaw.

- **“The paper is unfair because it evaluates only closed models / lacks open-source analysis.”**  
  This is factually weakened by Section 6, which explicitly says the main analysis uses closed models but also notes an appendix application to Llama-3.1-8B.

- **Generic reproducibility complaints about omitted implementation details.**  
  The paper provides code and the key methodological ingredients; the real issue is not missing low-level details but limited validation of the auxiliary-LLM pipeline.

- **Purely generic strengths such as “the paper is well-written” or “the problem is important.”**  
  These are omitted because they are not specific enough to distinguish this work.

## Novel Insights
The most interesting synthesis across the reviews is that the paper is strongest not as a definitive faithfulness metric, but as a **semantic auditing framework**: its real value lies in revealing *which kinds of concepts explanations systematically omit or overstate* under black-box access. In that sense, the contribution is more diagnostic than foundational. This reframing also helps reconcile the paper’s most compelling result: aggregate faithfulness scores are less informative than the category-specific structure of mismatch, because two models with similar overall scores can be unfaithful in qualitatively different—and differently harmful—ways.

## Suggestions
- **Tighten the central claim.** Reframe the method explicitly as measuring alignment between explanation mentions and intervention-based concept influence on answers, rather than suggesting it directly captures internal reasoning faithfulness.
- **Add direct validation of the auxiliary-LLM pipeline.** Even a modest human study on a subset of concepts/counterfactuals/labels would materially improve technical soundness.
- **Strengthen the treatment of correlated concepts.** Show concrete failure cases, quantify how often they arise in BBQ/MedQA, and discuss when CE should not be interpreted causally.
- **Temper model-comparison language.** Where credible intervals overlap substantially, describe trends rather than firm rank orderings.
- **Include at least one baseline faithfulness method on a subset.** This would better establish the incremental value of concept-level interventions and the hierarchical estimator.
- **Clarify Definition 2.2 and discuss metric choice.** Make the EE definition fully consistent and explain why PCC is preferred despite its calibration blind spots.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 8.0, 6.0]
Average score: 7.5
Binary outcome: Accept
