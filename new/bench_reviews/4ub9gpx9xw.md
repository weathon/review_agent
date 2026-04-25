Now let me search for calibration papers.Now I have enough to write the final review. Let me compile everything.Now I have enough to write the final consolidated review.

---

## Summary
The paper proposes *causal concept faithfulness*, a framework that measures the faithfulness of LLM natural-language explanations by comparing the causal effect (CE) of high-level concepts on model answers against the rate those concepts are mentioned in the model's explanations (explanation-implied effect, EE). CE and EE are formalized using potential outcomes and KL divergence, faithfulness is defined as their Pearson correlation, and estimation uses a Bayesian hierarchical model for sample efficiency. Experiments on BBQ (social bias) and MedQA (medical QA) with GPT-3.5, GPT-4o, and Claude-3.5-Sonnet reveal two novel patterns: (1) LLMs hide the influence of safety alignment measures, and (2) GPT models mask social bias effects in their explanations.

---

## Strengths

- **Rigorous causal formalization (Definitions 2.1–2.3):** The paper cleanly operationalizes faithfulness using individual treatment effects, KL divergence for CE, and PCC for alignment. The definitions are precise enough to be reproducible and advance beyond ad-hoc or task-specific prior measures.

- **Category-level visualization (Figures 1 and 3):** Plotting CE vs. EE with concepts colored by category (Behavior/Context/Identity; Clinical Tests/Symptoms/Demographics) reveals that concepts cluster by category with distinct faithfulness patterns per cluster — a granular insight invisible to a single scalar score. This is the paper's strongest empirical contribution.

- **Discovery of a novel unfaithfulness pattern — hiding safety measures (Section 4.1, Figure 2):** The paper demonstrates that all three GPT-family models produce answers of "Undetermined" on identity-rich BBQ questions, citing question ambiguity in their explanations — but when identity information is removed, the models shift away from "Undetermined." This is a concrete, non-obvious finding not reported in prior work on this dataset (Turpin et al., 2023).

- **Novel application to medical QA (Section 4.2, Table 3):** The method uncovers clinically relevant unfaithfulness — e.g., Claude's explanations never mention *the patient's mental status upon arrival* (EE = 0) despite this having the highest CE (0.32) among all concepts. This demonstrates utility in a high-stakes domain where faithfulness had not been previously studied.

- **Bayesian hierarchical estimation (Section 3):** Using shared priors across concept categories for CE estimation and hierarchical regression for faithfulness estimation is a principled, cost-aware design for expensive API calls. It addresses the small-sample problem through partial pooling.

---

## Weaknesses

### Fatal
None.

### Major

- **Auxiliary LLM accuracy is unvalidated, yet load-bearing.** The method depends on GPT-4o (a) correctly identifying the concept set for each question, (b) generating counterfactuals that truly isolate concept changes, and (c) correctly classifying whether an explanation implies a concept was causal. Every downstream estimate — CE, EE, and F — inherits errors from all three steps. The paper acknowledges in Section 6 that outputs "sometimes contain errors" and defers to Appendix F.3, but provides *no quantitative validation*: no inter-annotator agreement, no held-out comparison against human labels, no ablation replacing GPT-4o with another model. This is not a peripheral limitation — it is the load-bearing assumption of the pipeline. Furthermore, GPT-4o is used as the auxiliary LLM to evaluate GPT-4o itself. If GPT-4o tends to identify the same elements as "concepts" that GPT-4o tends to mention in explanations (a plausible stylistic correlation within the model family), EE would be systematically inflated for GPT-4o relative to GPT-3.5 and Claude, directly biasing the model comparisons the paper foregrounds. An analogous concern was raised for the CALM paper's LLM-in-the-loop perturbation generation; that paper addressed it experimentally — this paper does not.

- **The 30-question samples produce credible intervals too wide to support model-ranking claims.** As shown directly in the paper: BBQ — GPT-3.5 = 0.75 [0.42, 1.00], GPT-4o = 0.56 [0.24, 0.86], Claude = 0.62 [0.28, 0.91]; MedQA — GPT-3.5 = 0.50 [0.18, 0.77], GPT-4o = 0.34 [0.05, 0.65], Claude = 0.30 [0.01, 0.59]. All pairs overlap substantially. The paper's claim that "GPT-3.5 produces more faithful explanations than the two more advanced models" (Section 4.1) is drawn from point estimates whose 90% CIs span nearly the full [0, 1] range and include all relative orderings. The robustness analysis in Appendix D.4 shows stability of the point estimate for N ≥ 15, but that addresses a different question than whether the evidence is sufficient to *rank* models. The paper acknowledges cost constraints, but these do not change the evidentiary status of the results.

### Minor

- **Disentanglement assumption is acknowledged but theoretically unresolved for MedQA.** Definition 2.1 requires that changing C_m leaves all other concepts fixed. In medical QA, clinical variables are causally entangled (e.g., vital signs co-vary with mental status). The paper's response — restricting MedQA to removal-only counterfactuals (Section 4.2) — is a reasonable practical fix, but removal-based counterfactuals estimate a different estimand than the individual treatment effect in Definition 2.1. The paper acknowledges correlated concepts as a limitation (Section 6), but does not explicitly flag that the MedQA CE estimates are measuring something different from the formal definition. This means the theoretical framing and the empirical results are slightly misaligned.

- **Near-zero CE values in Table 3 (GPT-4o) make the GPT-4o MedQA faithfulness estimate unreliable.** GPT-4o's CE for the most influential concept is 0.04 (Table 3) — a nearly negligible KL divergence. After z-score normalization, small absolute differences between CE and EE are amplified, making GPT-4o's MedQA faithfulness estimate dominated by estimation noise rather than genuine signal. The paper should either flag which CE values fall below a reliability threshold or provide a sensitivity analysis of how faithfulness estimates change with CE scale.

- **Potential conflation in EE: formula vs. textual definition.** The textual definition of EE (Definition 2.2) states it is computed over "original input x *and* counterfactual questions," but Equation 2 sums only over counterfactual values c'_m ∈ C'_m, excluding the original. If EE indeed excludes the original question's explanation, the textual framing should be corrected for precision.

### Trivial
None that survive filtering of parser artifacts.

---

## Nice-to-Haves

- **Validate auxiliary LLM steps against human annotation.** A sample of concept extractions, counterfactuals, and concept-mention labels judged by humans would ground the quantitative claims and substantially increase scientific confidence in the pipeline.

- **Larger dataset or minimum-N analysis.** Reporting the minimum N required to achieve 90% CI width ≤ 0.3 for model comparisons (and justifying why current results are interpretable despite the actual widths) would help readers calibrate the model-ranking claims appropriately.

- **Disentangle safety-alignment effects from explanation unfaithfulness.** Comparing faithfulness on questions that do vs. do not trigger safety behaviors would help separate the mechanism of "model trained to refuse" from the mechanism of "explanation is dishonest about reasoning" — two distinct failure modes with different practical implications.

- **Side-by-side counterfactual examples.** Showing original question, counterfactual text, and CE for a few examples would allow readers to assess whether the auxiliary LLM actually isolated the concept as intended.

- **Extend to open-ended (non-MCQ) settings** (already flagged as future work, but a clear and valuable next step).

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic Claim: "EE conflates two distinct behaviors (original vs. counterfactual explanations)"** — Removed as strawman. The formula in Eq. 2 sums over counterfactual inputs only, not the original. The design choice (measuring EE across counterfactuals) is the paper's intention and is internally consistent. Retained as a trivial precision note about the textual vs. formal definition only.

- **Harsh Critic Claim: "Per-question PCC on 3–5 concepts is meaningless"** — Removed as partially addressed. The paper explicitly addresses this in Section 3 by using a Bayesian hierarchical model that pools information across questions rather than computing raw PCC on 3-5 data points per question. The negative faithfulness values (e.g., F(x) = -0.34 in Table 2) are posterior estimates with wide CIs ([-0.92, 0.28]), not raw correlations, so they are interpretable as "uncertain, possibly negative." The hierarchical model is precisely designed for this.

- **Harsh Critic Claim: "GPT-3.5 being 'more faithful' is an artifact of safety alignment, not explanation dishonesty"** — Retained in weakened form as part of the "Nice-to-Haves" about disentangling safety alignment, but not kept as a standalone weakness because the paper explicitly discusses this pattern (Section 4.1, Pattern 1) and the counterintuitive finding is a feature of the analysis, not a flaw. The paper's point is exactly that the type of unfaithfulness differs between models — a nuanced finding, not a methodological error.

- **Strength Finder claim: "Concept-level perturbations better match LLM explanations"** — Kept only as a methodological motivation (Section 5 discussion), not as an independent strength, since no direct empirical comparison to token-level perturbations is provided.

- **Harsh Critic request for direct empirical comparison with Siegel et al. and Turpin et al. on shared data** — Moved to Nice-to-Haves. Relevant related work comparison, but absence of empirical head-to-head is not standard practice and the paper already discusses the conceptual relationships.

---

## Novel Insights

The most genuinely novel observation is the discovery that LLM explanations hide the influence of safety alignment measures rather than solely masking social bias. On BBQ questions containing identity information, GPT-4o and GPT-3.5 shift substantially toward "Undetermined" — but their explanations attribute this to question ambiguity rather than the presence of identity-sensitive content. This pattern (Pattern 1, Section 4.1) distinguishes LLM-explanation unfaithfulness from the previously documented social-bias masking pattern (Pattern 2), and has important practical implications: a user receiving a "cannot determine" answer with an ambiguity explanation receives *doubly* misleading information — the model is actually detecting identity information and making a safety-motivated choice, but the explanation frames it as epistemically neutral. The paper's CE/EE decomposition is the only current approach capable of separating these two failure modes from a single scalar faithfulness score.

---

## Evaluation on Key Axes

- **Originality:** Solid. The combination of causal inference formalism, concept-level counterfactuals, and Bayesian hierarchical modeling for LLM faithfulness is original. The safety-alignment hiding pattern is a novel empirical discovery.
- **Importance of research question:** High. Faithful LLM explanations matter in high-stakes applications, and the move from scalar faithfulness scores to interpretable patterns is a meaningful advance.
- **Whether claims are well-supported:** Mixed. The qualitative pattern findings (Figures 1–3, Tables 2–3) are supported by the data. The quantitative model-ranking claims are not: the CIs overlap for all model pairs on both datasets.
- **Soundness of experiments:** Moderate. The hierarchical modeling is appropriate; the 30-question subsamples are insufficient for comparative claims; the unvalidated auxiliary LLM is the most significant gap.
- **Clarity of writing:** Good. Definitions are precise, figures are interpretable, and limitations are honestly disclosed.
- **Value to the research community:** Meaningful. The framework is reusable, the code is released, and the safety-alignment hiding pattern is a novel and practically relevant finding.

---

## Score and Decision

**Calibration anchors consulted:**

| Path | Avg Score | How it compares |
|---|---|---|
| 1OyE9IK0kx ("Hardness of Faithful CoT") | 5.00 (Reject) | Similar topic, similar concern about methodology; weaker framework than paper under review |
| o6eUNPBAEc ("Language Models Struggle to Explain Themselves") | 5.00 (Reject) | Similar topic; introduces dataset but weaker methodology, brittle assumptions |
| bpheRCxzb4 ("Measuring Information in Text Explanations") | 6.50 (Reject) | Stronger formal framework; similar concern about unvalidated estimators |
| 3GTtZFiajM ("Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge") | 6.75 (Accept) | Accepted; addressed LLM-in-the-loop validation concern in rebuttal — key difference from this paper |
| wwO8qS9tQl ("ALMANACS") | 3.00 (Reject) | Same area; weaker methodology; mostly negative results; below this paper |
| QQt0MwXA81 ("Do LLMs exhibit human-like response biases?") | 6.20 (Reject) | Similar empirical flavor; also struggled with unvalidated perturbations |

**Positioning:** The paper under review is clearly above the 3.0 anchors (ALMANACS-level weak papers) and above the 5.0 anchors (it has a more rigorous causal framework and genuine novel findings). However, it falls short of the 6.5–6.75 papers because: (a) the auxiliary LLM pipeline is not validated (the CALM paper, which got 6.75, addressed this in rebuttal), and (b) the 30-question subsamples produce overlapping CIs for all model comparisons, undermining the paper's main quantitative claims. The safety-alignment hiding discovery and the causal framework are genuine contributions that push above the 5.0 median, but the unresolved validation gap and small sample size preclude a confident accept. The cluster of related work anchors in the 5.0–6.2 range (with most receiving rejection decisions) suggests this paper lands in the borderline 5–6 band. Given the genuine novelty of the pattern discovery and the Bayesian approach, I score it slightly above the median at **5.5**, aligned with the 5.0–6.2 anchor cluster and below the validated, larger-scale papers at 6.5+.

**Decision: Reject** (the unvalidated auxiliary LLM pipeline and statistically unsupported model comparisons are real barriers; the pattern discoveries and formalism are genuine but insufficient on their own to overcome the methodological gaps in the current submission).

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>