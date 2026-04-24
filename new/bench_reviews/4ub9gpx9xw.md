## Summary

This paper introduces *causal concept faithfulness*, a framework for evaluating whether LLM explanations truthfully reflect the high-level concepts that drive model answers. It formalizes faithfulness as the Pearson correlation between causal concept effects (estimated via LLM-generated counterfactuals) and explanation-implied effects (estimated via LLM-based mention detection), and uses a Bayesian hierarchical model to pool information across questions. The authors evaluate three models (GPT-3.5, GPT-4o, Claude-3.5-Sonnet) on two tasks (social bias/BBQ and medical QA/MedQA), reporting both dataset-level scores and qualitative patterns of unfaithfulness, including models hiding the influence of safety alignment and omitting key clinical evidence.

## Strengths

- **Novel, well-motivated conceptual framework.** The paper moves beyond token-level perturbations to define faithfulness at the concept level, which aligns with the semantic nature of LLM explanations (Section 2, Definition 2.1–2.3).
- **Theoretically sensible estimation strategy.** The use of a Bayesian hierarchical model to share information across questions and concept categories is an appropriate response to the high cost of LLM API queries (Section 3).
- **Interesting empirical discoveries supported by direct evidence.** The analysis of BBQ in Section 4.1 and Figure 2 reveals that GPT models shift from “Undetermined” to specific biased answers when identity information is removed, yet never mention identity or safety refusal in explanations. This pattern is visible in the raw answer distributions and individual concept CE/EE values, not only in summary scores. Similarly, Table 3 shows Claude-3.5-Sonnet omitting mention of patient mental status (EE = 0.00) despite its large causal effect (CE = 0.32) on MedQA.
- **Extension to a high-stakes domain.** Applying the method to medical QA (MedQA) is a meaningful expansion beyond the social-bias benchmarks that dominate this literature (Section 4.2).

## Weaknesses

### Fatal
None.

### Major
- **Question-level faithfulness estimates are statistically unreliable and overinterpreted.** The paper acknowledges that “the number of concepts per question … is often small” (Section 3), and the case studies in Tables 2–3 show correlations computed over as few as 3 concepts. At this scale, the hierarchical model cannot produce meaningfully question-specific estimates; the 90% credible intervals span almost the entire range (e.g., GPT-4o on BBQ: [−0.92, 0.28]; Claude on MedQA: [−0.75, 0.22]). The paper nonetheless ranks models by these scores (“GPT-4o is the most unfaithful,” Section 4.1) and builds qualitative narratives around them. While the *individual* CE and EE vectors do reveal some patterns (e.g., high CE + low EE for identity concepts), the correlation metric itself is too unstable to support reliable question-level discrimination.
- **No validation of the auxiliary LLM measurement pipeline.** The entire empirical method depends on GPT-4o to (a) extract concepts, (b) generate counterfactual questions, and (c) detect whether explanations mention a concept. The paper provides no quantitative validation of any step—no human agreement rates, no error analysis, and no quality control for counterfactual naturalness or isolation (Section 3 mentions detection “in a single sentence” with no validation protocol). If the auxiliary model omits influential concepts, introduces confounding edits, or misclassifies mentions, the resulting CE and EE estimates may be measurement artifacts rather than faithful reflections of model behavior.
- **Dataset-level model comparisons lack statistical support.** Figure 1 reports dataset-level faithfulness with 90% credible intervals that overlap almost entirely across models: GPT-3.5 [0.42, 1.00], GPT-4o [0.24, 0.86], Claude-3.5-Sonnet [0.28, 0.91]. The text draws strong comparative conclusions (“GPT-3.5 produces more faithful explanations than the two more advanced models,” Section 4.1), yet the paper provides no test of whether these posterior distributions differ. Given the massive overlap, the claimed ranking is not supported by the paper’s own evidence.

### Minor
- **Formal inconsistency in Definition 2.2.** The text states that the explanation-implied effect averages over “original input $\mathbf{x}$ *and* to counterfactual questions,” but the formula averages only over counterfactual values $c'_m \in \mathbb{C}'_m$, omitting the original input. This creates ambiguity about whether the empirical estimates match the theoretical definition.
- **Undiagnosed failures of the disentanglement assumption.** The paper assumes concepts are “disentangled” (Section 2) and acknowledges in the limitations that “there are cases in which our method fails to handle correlated concepts.” However, it never diagnoses how often interventions inadvertently manipulate correlated factors or quantifies the impact on causal estimates.
- **Removal-only counterfactuals in MedQA may conflate concept influence with input degradation.** The paper focuses on removal counterfactuals for MedQA because changing clinical values is “hard to assess.” However, if removing a symptom makes a question clinically unanswerable, answer changes may reflect degraded input quality rather than the causal influence of that concept on the original reasoning (Section 4.2).

### Trivial
- **Pearson correlation is not justified relative to alternatives.** The paper uses Pearson correlation to measure alignment between CE and EE vectors without explaining why rank correlation or error metrics would be inappropriate, especially given the small number of concepts per question (Section 2).

## Nice-to-Haves
- Human validation of the auxiliary LLM pipeline (concept extraction, counterfactual isolation, and mention detection) to quantify measurement error.
- A baseline comparison of the hierarchical estimator against direct per-question plugin estimation to empirically justify the modeling choice.
- Sensitivity analysis using Spearman or Kendall correlation instead of Pearson.
- A diagnostic analysis measuring whether concept interventions inadvertently change other text properties (e.g., via lexical overlap or semantic similarity).

## Removed Points
These points are flagged to be removed, treat them with caution:
- **Safety-training conflation claim.** The harsh critic argued that hiding safety-measure influence is not necessarily unfaithfulness. The paper’s interpretation—that explanations citing only “ambiguity” while omitting the role of identity/safety information is incomplete—is defensible and does not conflate safety training with unfaithfulness.
- **Hierarchical model details deferred to appendix.** The paper appropriately notes that details are in Appendix C; this is standard practice and not a meaningful weakness (appendices exist in the original submission).
- **Understated limitations section.** The severity of limitations is a subjective presentation judgment, not a substantive methodological flaw.
- **Missing experiments (baselines, sensitivity, significance tests).** While these would strengthen the paper, they are methodological additions rather than flaws in what is presented.

## Novel Insights

A genuinely novel observation that emerges from synthesizing these reviews is the tension between using hierarchical Bayesian models to pool information and the resulting shrinkage that dominates question-level estimates when within-group sample sizes are extremely small (here, 3–5 concepts per question). The paper’s framework is theoretically appealing, but this reveals a fundamental challenge: for concept-level faithfulness, the very granularity that makes the method interpretable (question-level correlations) may be statistically unrecoverable without many more concepts per question or much stronger priors. Future work might need to either collect denser concept sets or shift to rank-based alignment metrics that are more stable with tiny samples.

## Suggestions
- Add a formal statistical test (or at least report posterior probability of difference) for dataset-level model comparisons before drawing rankings.
- Include a quantitative validation study of the auxiliary LLM pipeline, even on a small held-out subset, to establish measurement reliability.
- Consider reporting effect-size measures or rank correlations alongside Pearson correlation for question-level estimates to reduce sensitivity to outliers with tiny concept counts.

## Score and Decision

**Calibration references:**
- *High anchor (6.75)*: `i8IwcQBi74.md` (contrastive explanations for reward models) — accepted because the method was validated with clear experiments and controls. The current paper lacks comparable validation of its auxiliary LLM pipeline and has weaker statistical support for comparisons.
- *High anchor (6.75)*: `NgaLU2fP5D.md` (PSI-KT hierarchical knowledge tracing) — accepted with solid hierarchical modeling and extensive experiments. The current paper’s hierarchical model is similarly motivated but undermined by extremely small within-question sample sizes.
- *Medium anchor (5.20)*: `WqsYs05Ri7.md` (uncertainty-aware concept explanations) — rejected due to poor theory and presentation despite an interesting idea. The current paper has clearer presentation and more compelling empirical findings, but comparable methodological concerns.
- *Medium anchor (5.00)*: `JshLcbPI9J.md` (deep backtracking counterfactuals) — rejected due to limited significance and scope. The current paper has broader scope and more interesting findings, but similar questions about whether the evidence supports the claims.
- *Low anchor (3.00)*: `fSbPwHjdDG.md` (causal interventions in LLMs) — rejected for very limited experiments (one task, one model) and problematic methods. The current paper is clearly stronger, with two datasets, three models, and a well-defined framework.
- *Low anchor (3.00)*: `z1yI8uoVU3.md` (steering evaluation framework) — withdrawn/rejected for unclear contribution and sensitivity to arbitrary normalization. The current paper has a clearer conceptual contribution.

**Comparison:** The paper under review sits between the medium and high anchors. Its conceptual contribution and empirical discoveries are stronger than the medium anchors, but its methodological gaps—especially the completely unvalidated auxiliary LLM pipeline and the statistically unsupported dataset-level rankings—are serious enough that it falls short of the high anchors. It is substantially better than the low anchors. A score of **5.0** reflects this: a borderline submission with real contributions that is currently undermined by insufficient validation and weak statistical support for its comparative claims.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>