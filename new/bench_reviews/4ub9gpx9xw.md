Now let me run calibration searches to score this paper appropriately.Now I have enough information to write the consolidated review.---

## Summary

The paper proposes *causal concept faithfulness*, a framework for measuring whether LLM natural-language explanations accurately reflect the concepts that truly influenced the model's answer. It formalizes causal concept effect (CE) via KL-divergence over counterfactual interventions and explanation-implied effect (EE) as the rate at which concepts are mentioned in explanations, then uses a Bayesian hierarchical model to estimate their Pearson correlation (faithfulness). Applied to GPT-3.5, GPT-4o, and Claude-3.5-Sonnet on BBQ and MedQA, the method reveals interpretable patterns of unfaithfulness beyond a single scalar score, including a novel finding that LLMs hide the influence of safety-alignment measures in their explanations.

---

## Strengths

- **Causal-inference grounding (Section 2, Definitions 2.1–2.3)**: Formalizing faithfulness via individual treatment effects (rather than average treatment effects) correctly matches the question of interest—whether a *specific* explanation for a *specific* question is faithful—and is a principled advance over ad hoc perturbation tests.

- **Bayesian hierarchical model for sample-efficient estimation (Section 3)**: Pooling information across questions with shared concept-category priors enables more reliable faithfulness estimates when per-question sample counts are small; this is a non-trivial methodological contribution practical for the high cost of LLM API queries.

- **Semantic pattern discovery beyond aggregate scores (Figures 1 and 3)**: The category-colored CE vs. EE scatter plots clearly demonstrate the paper's primary claim: all three models exhibit qualitatively distinct patterns by concept category (identity concepts are systematically omitted despite large CE; behavior concepts are over-cited regardless of CE), which a single scalar score cannot reveal.

- **Novel safety-measure concealment finding (Section 4.1, Figure 2)**: The finding that removing social identity information causes GPT models to shift from "Undetermined" to a specific person—contradicting explanations that attribute the refusal purely to question ambiguity—is a concrete empirical contribution not reported in prior work.

- **Counterintuitive and consequential model comparison (Section 4.1)**: The analysis showing that GPT-3.5 has a higher aggregate faithfulness score yet exhibits the more *harmful* pattern (masking social bias), while GPT-4o's lower score partly reflects safety-measure concealment, is a compelling demonstration of why semantic pattern analysis is necessary.

---

## Weaknesses

### Fatal
None.

### Major

- **Circular evaluation when GPT-4o is both auxiliary LLM and evaluated model**: GPT-4o serves as the auxiliary LLM for concept extraction, counterfactual generation, and EE detection, even when GPT-4o itself is the target model $\mathcal{M}$. If GPT-4o's preferences for relevant concepts and appropriate counterfactuals are systematically aligned with its own explanation behavior, CE and EE estimates for GPT-4o could be inflated or deflated relative to estimates for GPT-3.5 and Claude (which are evaluated by a *different* model). The paper justifies using GPT-4o as auxiliary by citing prior work on GPT-based counterfactual generation, but this does not address the asymmetry. No sensitivity analysis to auxiliary LLM choice is provided, leaving comparative model rankings uninterpretable.

- **Inconsistency between Definition 2.2 (EE) verbal description and formula**: The verbal definition states EE captures the probability that explanations in response to "original input $\mathbf{x}$ **and** to counterfactual questions" mention $C_m$ as causal. The formula is:  
  $$\text{EE}(\mathbf{x}, C_m) = \frac{1}{|\mathbb{C}'_m|} \sum_{c'_m \in \mathbb{C}'_m} \mathbb{P}_{\mathcal{M}}(C_m \in E|\mathbf{x}_{c_m \rightarrow c'_m})$$  
  which averages only over *counterfactual* inputs, excluding the original question entirely. This makes EE a measure of whether the model's explanations for counterfactual variants mention $C_m$—not whether the original explanation is faithful. Compounding the concern, the paper explicitly criticizes Atanasova et al. (2023) for "assess[ing] the faithfulness of explanations given to the counterfactual questions, which may not reflect the faithfulness of the LLM in response to the original questions"—yet the EE formula does exactly that. The definitional inconsistency is not a parsing artifact; it appears in the core section of the framework.

- **Comparative faithfulness claims not statistically supported by the credible intervals**: The paper concludes that "GPT-3.5 produces more faithful explanations than the two more advanced models" based on point estimates: 0.75 vs. 0.56 (GPT-4o) vs. 0.62 (Claude). However, the 90% credible intervals—GPT-3.5 [0.42, 1.00], GPT-4o [0.24, 0.86], Claude [0.28, 0.91]—overlap substantially; the entire GPT-4o interval lies within the GPT-3.5 interval. With only 30 questions, the Bayesian model cannot produce confident model-level rankings. The paper reports these intervals but continues to draw substantive comparative conclusions (e.g., Section 4.1's claim about GPT-3.5 being "least unfaithful") that the evidence does not support. MedQA results show similar overlap (e.g., GPT-4o [0.05, 0.65]).

### Minor

- **MedQA removal-only counterfactuals conflate causal effect with response to incompleteness (Section 4.2)**: The paper uses only removal counterfactuals in MedQA because "changing the values of clinical concepts could introduce subtle changes that are hard to assess." However, removing clinical information transforms a coherent patient vignette into an incomplete one. A model responding differently to the removed version may be reacting to structural incompleteness rather than to the concept's causal role. CE then measures sensitivity to missing information, not counterfactual causal effect. The paper does not discuss this conflation or bound its impact on the MedQA conclusions.

- **No validation of concept extraction and EE detection quality (Section 3)**: The entire pipeline's validity depends on GPT-4o reliably identifying concepts and detecting whether explanations mention them. No quantitative validation of these steps—even on a small annotated subset—is provided. Errors in concept extraction or mention detection propagate directly into CE and EE scores.

### Trivial

- The z-score normalization step (used to re-interpret PCC as a regression coefficient in the hierarchical model) discards the absolute magnitude of causal effects; only within-question relative ranking matters. This design choice is reasonable for the stated goal but is not explicitly stated or justified in the main text.

---

## Nice-to-Haves

- **Sensitivity analysis on auxiliary LLM choice**: Testing at least one alternative auxiliary (e.g., Claude as auxiliary when evaluating GPT-4o) would directly address the circularity concern and make cross-model comparisons more credible.

- **Controlled synthetic validation**: Constructing known-faithfulness examples (e.g., injecting a concept and pairing it with explanations that either mention or omit it while answers change) would provide ground-truth calibration for CE and EE.

- **Replacement counterfactuals for at least a subset of MedQA variables**: For discrete clinical variables (age, sex, categorical diagnoses), replacement counterfactuals are feasible and would avoid the removal-conflation problem.

- **A quantitative comparison to Atanasova et al. (2023) or Siegel et al. (2024) on shared questions**: The paper claims superiority over token-level approaches on conceptual grounds but never demonstrates it empirically on the same examples.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: "Sample size renders comparative claims unestablished" as a FATAL concern**: The paper's primary contribution is the semantic pattern analysis, not the numerical model rankings. Wide CIs on 30-question point estimates are a real limitation (kept as Major) but do not invalidate the framework or pattern-discovery findings.

- **Harsh Critic: "The disentanglement assumption is a structural flaw"**: The paper explicitly acknowledges correlated concepts as a limitation in Section 6 and notes that multi-concept interventions are future work. This is a known limitation, not an undisclosed one.

- **Harsh Critic: "The KL divergence direction is unmotivated and its asymmetry is not discussed"**: KL divergence is standard for measuring distributional shift in intervention studies; the particular direction (counterfactual vs. original) is the natural choice for measuring "how much did the distribution change." This is minor notation-level commentary, not a substantive methodological flaw.

- **Harsh Critic: "The pattern of unfaithfulness (safety measures) is interpreted qualitatively from two or three examples"**: The paper states it identifies this pattern "across many questions in the dataset" and references Appendix D.2, which the parser strips. This is a citation to stripped content, not absent evidence.

- **Harsh Critic: "30 questions could be highly unrepresentative without reporting random seed"**: Reproducibility concern about undisclosed hyperparameters/seeds; falls under the hard rule for trivial reproducibility nitpicks. The paper states the sample was "stratified across nine social bias categories."

- **Harsh Critic: "Prompt sensitivity is a reproducibility concern"**: Removed per the rule about trivial implementation details and practical reproducibility nitpicks. The paper acknowledges prompt engineering is needed and says prompts are available in the appendix.

- **Strength Finder: "Applicable to API-only opaque models"**: Generic strength; most LLM evaluation papers work with opaque models. No specific citation or table to evidence this distinctiveness. Dropped.

---

## Novel Insights

The most genuinely novel insight is the discovery of *safety-measure concealment*: LLMs that invoke "undetermined / ambiguous" responses when social identity information is present revert to selecting a specific individual when that information is removed, yet their explanations attribute refusal entirely to ambiguity. This creates a category of unfaithfulness distinct from social-bias masking—namely, the model's explanation hides the causal role of identity information in triggering safety-alignment behavior, not just in driving stereotyped choices. The paper also makes the under-appreciated point that higher aggregate faithfulness scores can coincide with *more harmful* semantic patterns of unfaithfulness, inverting the intuitive ordering between GPT-3.5 and GPT-4o. These two observations together motivate why scalar faithfulness scores are insufficient and why pattern-level analysis is needed.

---

## Suggestions

1. **Run an auxiliary-LLM sensitivity experiment**: Use Claude-3.5-Sonnet as auxiliary LLM when evaluating GPT-4o, and use GPT-4o as auxiliary when evaluating Claude. Report whether the relative model rankings change. Even one swap would substantially address the circularity concern.

2. **Clarify or fix Definition 2.2**: Either update the formula to include the original question ($\mathbf{x}$ alongside the counterfactuals), or update the verbal description to say EE is measured only over counterfactual explanations and explicitly justify why this differs from Atanasova et al. (2023)'s limitation.

3. **Reframe comparative claims as descriptive rather than confirmatory**: Given the wide CIs, avoid statements like "GPT-3.5 produces more faithful explanations"; instead, frame as "GPT-3.5 shows a higher point estimate, though CIs overlap substantially" and lead with the pattern-level analysis as the primary evidence.

4. **Add at least a small annotated validation of concept extraction quality**: Even manually labeling 20–30 concepts as correctly vs. incorrectly extracted and detected would provide interpretability for how much noise propagates into CE/EE.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Comparison |
|------|----------------|------------|
| `/human_reviews/1OyE9IK0kx.md` | 5.0 (Reject) | LLM CoT faithfulness paper with moderate rigor; weaker methodological framing than this paper but similar scope. |
| `/human_reviews/o6eUNPBAEc.md` | 5.0 (Reject) | LLM self-explanation study; similar topic, somewhat weaker causal grounding, also small-scale. |
| `/human_reviews/wwO8qS9tQl.md` | 3.0 (Reject) | LLM explainability benchmark; limited modeling depth, negative results without actionable insight. |
| `/human_reviews/WCRQFlji2q.md` | 9.0 (Accept Oral) | Mechanistic interpretability via sparse autoencoders; much stronger empirical depth and methodological rigor. |
| `/human_reviews/6NNA0MxhCH.md` | 7.5 (Accept Spotlight) | Transformer multiple-choice interpretation via activation patching; more rigorous and larger-scale evaluation. |
| `/human_reviews/8QTpYC4smR.md` | 1.0 (Reject) | Systematic review with no novel contribution; far weaker than this paper. |
| `/human_reviews/wJVZkUOUjh.md` | 2.0 (Reject) | EXAGREE framework; vague utility and unclear methodology. |

**Assessment:** The paper is clearly above the 1–2 band (it has genuine technical contributions). It competes directly with the 5.0-band LLM-faithfulness papers and is *somewhat* stronger due to its causal framework and Bayesian hierarchical model. However, three concurrent Major weaknesses—circularity, EE definition inconsistency, and statistical insufficiency for the headline comparative claim—prevent a clear accept. The 7.5+ papers in the XAI space have much tighter experimental designs and larger validation. The paper sits meaningfully above the 5.0 anchors (stronger causal grounding, novel empirical insights), but the methodological concerns anchor it below the 6.5 threshold for a clear accept.

**Originality**: Good — the causal concept framework and semantic pattern discovery are genuine contributions.  
**Importance of research question**: High — faithfulness of LLM explanations in high-stakes domains is directly practically relevant.  
**Claims well supported**: Weak — the definitional inconsistency and statistically unsupported comparative claims undermine the core results.  
**Soundness of experiments**: Moderate — the Bayesian model is principled, but circularity and small scale limit conclusions.  
**Clarity of writing**: Good overall, with a notable flaw in Definition 2.2.  
**Value to community**: Moderate-high — the pattern-discovery angle and safety-alignment concealment finding are genuinely useful.

**Final Score: 5.5 — Borderline Reject**

The paper makes real contributions but has real methodological problems that prevent confident acceptance. The authors should address the EE definition inconsistency, the circular evaluation design, and the statistical framing of comparative claims before resubmission.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>