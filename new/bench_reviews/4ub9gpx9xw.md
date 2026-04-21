Now let me run calibration searches to anchor the score.Now I have enough to write the consolidated review.

---

## Summary
This paper proposes *causal concept faithfulness*, a method for measuring whether LLM explanations faithfully reflect the concepts that causally influenced the model's answers. The approach uses an auxiliary LLM to generate concept-level counterfactuals, estimates causal concept effects (CE) via KL divergence, measures explanation-implied effects (EE) as concept mention rates, and computes their Pearson correlation via a Bayesian hierarchical model. Experiments on social-bias (BBQ) and medical QA (MedQA) tasks across three LLMs reveal novel patterns of unfaithfulness, including the discovery that safety-alignment mechanisms cause LLMs to produce explanations that cite "ambiguity" rather than identity information when identity concepts are causally influential.

---

## Strengths

- **Semantic pattern discovery (Figure 1, Figure 3):** The paper goes beyond scalar faithfulness scores and reveals *how* models fail. Category-coloring (Behavior/Context/Identity on BBQ; Clinical tests/Symptoms/Demographics on MedQA) produces scatter plots in Figures 1 and 3 that make category-specific patterns visually apparent. This is arguably the paper's most compelling and reusable contribution.

- **Novel finding: safety-measure masking as unfaithfulness (Section 4.1, Figure 2):** The discovery that GPT-3.5 and GPT-4o explain their choice of "Undetermined" by citing *ambiguity* rather than acknowledging that the presence of identity information triggers safety-alignment refusal is a genuinely new finding. This goes beyond the social-bias masking documented by Turpin et al. (2023) and has immediate practical relevance.

- **Rigorous causal definition (Definitions 2.1–2.3):** The formalization using do-operator notation (Appendix B) and the distinction between individual treatment effects and average treatment effects is methodologically sound and provides a transparent, auditable framework that much prior faithfulness work lacks.

- **Bayesian hierarchical model for sample efficiency (Section 3):** The two-level hierarchical model (pooling across concepts within category for CE; pooling across questions for faithfulness) is well-motivated and addresses the practical constraint of API cost. The motivation in Section 3 is clearly explained.

- **Black-box applicability:** The method requires only input-output access, making it directly applicable to frontier API-only models—exactly the systems for which faithful explanations matter most.

- **Dual-domain validation:** BBQ and MedQA are structurally very different tasks (social bias vs. medical diagnosis), and the method surfaces qualitatively distinct patterns in each, strengthening claims of generality.

---

## Weaknesses

### Fatal
None.

### Major

- **GPT-4o self-referential evaluation (Sections 3–4):** GPT-4o serves simultaneously as (a) the auxiliary LLM for counterfactual generation, concept extraction, and explanation classification, and (b) one of the three models under evaluation. No ablation tests whether using a different auxiliary LLM (e.g., Claude) changes GPT-4o's faithfulness score. Given that GPT-4o receives a meaningfully different score from GPT-3.5 (0.56 vs. 0.75 on BBQ), any systematic vocabulary alignment, familiarity with its own generation patterns, or preferential classification of its own outputs could confound the comparative claims. This is the paper's most significant uncontrolled variable.

- **Definitional inconsistency in EE (Definition 2.2):** The text of Definition 2.2 states the EE is "the probability that M's explanations in response to *original input x* and to counterfactual questions imply that C_m is causal," but the formula sums only over counterfactual inputs — `P_M(C_m ∈ E | x_{c_m→c'_m})` — with no term for the original question `x`. The estimation procedure (Section 3) confirms only counterfactual explanations are used. The authors elsewhere note (in the related work, Section 5) that assessing faithfulness only on counterfactual explanations "may not reflect the faithfulness of the LLM in response to the original questions" — yet EE as implemented does exactly that. The definitional text and formula must be made consistent, and the implications for faithfulness scores should be clarified.

### Minor

- **PCC is directionally insensitive (Definition 2.3):** Pearson correlation does not distinguish between over-attribution (high EE for low-CE concepts) and under-attribution (low EE for high-CE concepts). The paper's own analysis identifies under-attribution — hiding social bias or safety triggers — as the safety-critical failure mode. A model that uniformly over-mentions concepts can achieve a similar PCC to one that correctly tracks influence, depending on variance structure. The paper's discussion (Section 4.1, last paragraph) acknowledges that GPT-3.5's higher faithfulness score coexists with *more dangerous* unfaithfulness (social bias masking), which directly illustrates the metric's limitation. Supplementing PCC with a signed decomposition (e.g., counting high-CE/low-EE vs. low-CE/high-EE pairs) would better serve the safety motivation.

- **Wide question-level credible intervals (Tables 2–3):** With only 3–4 concepts per question, question-level PCC estimates carry enormous uncertainty: e.g., GPT-4o on the BBQ example question has CI = [−0.92, 0.28] (Table 2), spanning nearly the entire range. The paper acknowledges this but still presents question-level scores as interpretable. The practical value of question-level faithfulness assessment (one of the three stated goals in Section 1) is weakened by this issue.

- **MedQA removal-only counterfactuals (Section 4.2):** Restricting to removal counterfactuals is reasonable but means CE captures only the effect of *removing* information, not *changing* it. For concepts that anchor question coherence (e.g., "the patient's mental status"), removal may cause a large distributional shift that inflates CE independently of the concept's genuine causal role. The authors note this design choice but do not discuss its effect on CE estimates.

### Trivial

- The dataset-level credible intervals on BBQ (e.g., GPT-3.5 CI = [0.42, 1.00]) are wide enough that the ranking GPT-3.5 > Claude > GPT-4o is not statistically decisive; the paper presents these comparisons in a confident register that slightly overstates their precision.

---

## Nice-to-Haves

- **Auxiliary LLM ablation:** Run the full pipeline using Claude-3.5-Sonnet as the auxiliary LLM when evaluating GPT-4o, and report whether faithfulness scores change materially. This would directly address the self-referential evaluation concern.

- **Directional faithfulness decomposition:** A visualization plotting signed deviation (EE − CE, normalized) by concept category and model would show whether unfaithfulness is predominantly over-attribution or under-attribution — information PCC alone discards, and highly relevant to the safety motivation.

- **Counterfactual quality audit:** A brief human evaluation (e.g., 50 sampled counterfactuals) assessing whether only the target concept changed would quantify the disentanglement assumption and let readers better assess the CE estimates' reliability. The paper notes errors occur (Section 6, Appendix F.3) but offers no frequency estimate.

- **Sensitivity to hierarchical prior strength:** The shrinkage introduced by the hierarchical CE model is consequential for rare/unusual concepts. A brief sensitivity analysis would clarify how much CE estimates are data-driven versus prior-driven.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic: "EE definition excludes the original explanation, undermining the motivating goal."** Partly valid as a definitional inconsistency (kept as a Major weakness above), but the critic's claim that this renders the entire method invalid is overstated. The counterfactual-based EE is a coherent estimand; the issue is primarily that the definition text misleadingly says "original input x and counterfactual questions" while the formula captures only counterfactuals.

- **Strength Finder: "Dual-granularity faithfulness estimation (question and dataset level)."** Kept as a minor strength, but the wide question-level CIs substantially limit the practical value of that level, as verified in Tables 2–3.

- **Harsh critic: hierarchical model shrinkage causes context concepts to cluster at low CE artificially.** The paper's context concept results (low CE, low EE) are plausible on substantive grounds (context information genuinely matters less in these datasets than identity/behavior). This critique is speculative without evidence the prior is driving the pattern, so not included as a weakness.

- **Harsh critic: "MedQA removal counterfactuals cannot isolate single concepts cleanly, making CE contaminated."** Retained as a minor concern above, but the claim that this makes the estimates "contaminated" in a way that undermines results is too strong.

---

## Novel Insights

The most novel synthesis across reviewers is the observation that the faithfulness *metric* (PCC) and the faithfulness *patterns* revealed by the CE/EE scatter plots point in opposite directions for safety-relevant failure modes. GPT-3.5 earns the highest PCC-based faithfulness score while exhibiting the most harmful pattern (social bias masking), whereas GPT-4o earns a lower score because its safety-alignment refusals create CE/EE mismatches. This exposes a fundamental tension: a single scalar faithfulness metric aggregates over heterogeneous failure modes that differ in severity, and any such metric will misjudge models that trade a benign failure mode for a dangerous one. The paper's category-level scatter plot framework implicitly resolves this tension by making the *pattern* primary, but the scalar score — which is foregrounded in section headings and tables — continues to obscure it. Future work should consider whether faithfulness metrics ought to be disaggregated by concept category rather than averaged.

---

## Suggestions

1. Fix Definition 2.2 so the formula matches the text (either include the original question in EE, or revise the text to say EE is estimated from counterfactual explanations only, with justification).
2. Add at least one auxiliary-LLM ablation for GPT-4o evaluation; even a small spot-check would address the self-referential evaluation concern.
3. Add a directional decomposition (high-CE/low-EE vs. low-CE/high-EE counts by category and model) alongside PCC scores to capture the safety-critical distinction between over- and under-attribution.
4. Clarify that question-level faithfulness scores have very wide credible intervals and are best understood qualitatively; the dataset-level scores are the paper's reliable quantitative result.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Relation to this paper |
|------|----------------|------------------------|
| `/human_reviews/1OyE9IK0kx.md` | 5.0 (Reject) | LLM CoT faithfulness; weaker methodology, less novel findings than this paper |
| `/human_reviews/o6eUNPBAEc.md` | 5.0 (Reject) | LLM self-explanation; narrower scope, brittle behavioral assumptions |
| `/human_reviews/gsShHPxkUW.md` | 5.75 (Poster) | Causal mediation for LLMs; similar methodology, less diagnostic framework |
| `/human_reviews/q2DmkZ1wVe.md` | 6.0 (Poster) | Counterfactual QA benchmark; comparable rigor |
| `/human_reviews/bVTM2QKYuA.md` | 6.75 (Oral) | Concept geometry in LLMs; stronger theoretical grounding |
| `/human_reviews/vfEqSWpMfj.md` | 2.5 (Reject) | LLM explainability with unclear methodology; this paper is clearly stronger |
| `/human_reviews/UHPnqSTBPO.md` | 8.0 (Oral) | Principled LLM evaluation with theoretical guarantees; stronger theory, broader experiments |

**Assessment:** This paper sits above the 5.0-reject papers (stronger methodology, clearer novel insights, rigorous causal framing) and is roughly comparable to the 5.75–6.0 poster-level papers (GSShHPxkUW, q2DmkZ1wVe) — methodologically principled, with genuine and reproducible empirical discoveries, but with real unresolved concerns (GPT-4o self-evaluation, definitional inconsistency, narrow samples). It falls below the 6.75 oral-level paper because of the sample-size limitations, the PCC metric's directional insensitivity, and the lack of ablations on the most structurally important design choice (auxiliary LLM identity). Considering the anchor cluster center around 5.75–6.0 and the paper's genuine novelty in revealing the safety-masking pattern and providing a reusable diagnostic framework, I position it at **6.0** — a marginal accept at poster level. The contributions are real and the insights are valuable; the structural concerns identified above are addressable in revision but do not invalidate the core contributions.

**Originality:** Good — first method to reveal semantic patterns of faithfulness failure in black-box LLMs via concept-level causal intervention.
**Importance:** High — directly relevant to trustworthy AI deployment, especially in safety-critical domains.
**Claim support:** Moderate — dataset-level claims are reasonably supported; question-level scores are unreliable.
**Experimental soundness:** Moderate — Bayesian approach is principled but samples are small, GPT-4o self-evaluation is uncontrolled.
**Clarity:** Good — the framework is clearly described; the definitional inconsistency in EE is a notable flaw.
**Value to community:** High — provides a reusable, black-box-applicable diagnostic framework and a concrete new finding (safety masking as unfaithfulness).

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>