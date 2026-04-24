## Summary

This paper proposes Tabby, a post-training architectural modification for transformer-based LLMs that replaces select MLP or language-modeling-head blocks with column-dedicated parameter sets (framed as Mixture-of-Experts layers) to improve synthetic tabular data generation. The authors evaluate multiple variants (MMLP, MH, MMLP-MH) across six standard datasets and introduce a per-column loss tracking diagnostic. While the problem motivation is timely and some empirical observations are useful, the central empirical claims are inconsistent with the paper's own results table and the main comparisons are structurally confounded by uncontrolled parameter scaling.

## Strengths

- **Per-column training diagnostics.** Tabby’s column-separate loss formulation yields fine-grained visibility into which columns a model struggles to learn (Figure 4, Section 4.3). This is a practical and genuinely novel diagnostic tool for tabular synthesis that standard monolithic losses cannot provide.
- **Honest baseline observation.** The paper shows that simple left-to-right “Plain” training without shuffling or pretraining often matches or outperforms far more elaborate LLM pipelines (GReaT/TapTap/Tabula) (Table 2). This is a valuable negative result for the community.
- **Strong performance on select datasets.** Plain-trained Tabby MH achieves the best overall MLE on the House dataset (0.75) and is competitive on several others, while successfully generating valid samples on Rainfall where prior LLM methods fail completely (Table 2, Section 4.1).

## Weaknesses

### Fatal
None. The methodology is not fundamentally invalid; however, the interpretation of results is sufficiently flawed that the core claims are not credibly established.

### Major

- **Empirical claims are directly contradicted by Table 2.** Section 4.1 states: “*Tabby models achieve the highest MLE in 4 out of 6 datasets*.” Table 2 shows that Tabby is strictly highest overall on only **one** dataset (House). On Diabetes, the best Tabby variant (GTT MMLP-MH, 75.3) merely **ties** with the Non-Tabby Plain baseline (75.3); on Adult, Plain MH (84.5) **ties** with Non-Tabby Plain (84.5); and on Travel, Abalone, and Rainfall, Tab-DDPM outperforms every Tabby variant. The abstract’s claim of “higher-quality synthetic data for 4 out of 6 datasets” and the conclusion’s claim of “two out of three evaluated datasets” (line 365) are similarly inconsistent with the evidence presented. A paper whose headline claims are directly refuted by its own results table cannot stand without major revision.
- **Uncontrolled parameter scaling confounds architectural claims.** Tabby MH increases model parameters from 80M to 270M (Table 3), and MMLP variants increase them further. Yet every LLM baseline in Table 2 uses the 80M Distilled-GPT2. Because parameter count is a primary driver of language-model performance, Claims 1 and 2 are structurally confounded: the observed gains may reflect increased capacity rather than the claimed architectural inductive bias of column-dedicated experts. The paper contains no parameter-matched non-Tabby baseline (e.g., standard GPT-2 Medium or a width-scaled Distilled-GPT2), so the experimental framework cannot isolate the architectural effect.
- **Core “Gated MoE” mechanism is underspecified.** The submission advertises “Gated Mixture-of-Experts layers” (Abstract, Section 2), yet Section 3.1 describes only that “the *i*-th column is modeled by *L*_{a,i}” with no gating function, no routing logic, and no explanation of how variable-length token sequences are mapped to fixed experts during autoregressive generation. Without this description the mechanism is not fully reproducible, and the “MoE” framing may be a misnomer for simple position-dependent layer duplication.

### Minor

- **MMLP variants systematically underperform, undermining generality.** The paper presents both MMLP and MH as core variants, yet Table 2 shows MMLP and MMLP-MH frequently collapse or degrade performance (e.g., Plain MMLP on Adult: 77.4 vs. NT 84.5; Plain MMLP-MH on House: 0.00 vs. NT 0.70). Only MH performs reliably. Because the paper offers no diagnosis for this failure, the evidence supports at best a narrow LM-head modification rather than a general architectural contribution.
- **Conclusion contains factual errors.** The conclusion states results are measured with a “Decision Tree Classifier” (Section 5), but the paper explicitly uses random forest throughout (Section 4.0.3). It also claims evaluation on “two out of three evaluated datasets,” contradicting the body’s six-dataset evaluation.

### Trivial
None.

## Nice-to-Have
- A parameter-matched non-Tabby baseline (e.g., GPT-2 Medium or width-inflated Distilled-GPT2) to disentangle capacity from architecture in Table 2.
- Visualization or formal description of the expert selection mechanism (learned gating, hard-coded column indexing, or token-triggered switching) to justify the “Gated MoE” label.
- An analysis of why MMLP fails where MH succeeds, or a reframing of the contribution around MH alone.

## Removed Points
These points are flagged to be removed; treat them with caution.
- **LoRA confound in Section 4.2:** The observation that Llama 3 uses LoRA while Distilled-GPT2 does not is true but explicitly disclosed by the authors. It is a minor training-regime difference, not a hidden confound.
- **Plain training “weakens the case for architectural modification”:** The fact that simple baselines are strong is an empirical finding the authors report honestly. This is a strength of intellectual honesty, not a weakness.
- **Missing appendix proofs or references:** The parser strips appendices from all papers; these exist in the original submission and should not be criticized.

## Novel Insights
None beyond the paper's own contributions. The per-column loss diagnostic and the Plain-training negative result are the most genuinely useful observations, though they are secondary to the overstated architectural claims.

## Suggestions
1. **Correct all empirical claims to match Table 2.** Rewrite the abstract, Section 4.1, and conclusion to accurately reflect that Tabby is best on only a minority of datasets and that the Non-Tabby Plain baseline is highly competitive.
2. **Add a parameter-matched baseline.** Include a ~270M parameter non-Tabby model in Table 2 to isolate whether the MH modification outperforms simple width scaling.
3. **Specify the routing mechanism.** If expert selection is fixed by column index, state this clearly and justify why it constitutes a meaningful architectural bias; if it is learned, show gate distributions.
4. **Focus on MH or explain MMLP failure.** If MH is the only reliable variant, center the paper on that and either explain MMLP’s poor performance or drop it.

## Score and Decision

**Calibration anchors:**
- **High:** TabSyn (`/home/wg25r/review_agent/human_reviews/4Ay23yeuz0.md`, avg 6.75, Accept oral) — solid tabular synthesis with sound experiments and clear claims. Tabby is well below this due to confounded comparisons and claim-table contradictions.
- **Medium:** TabDAR (`/home/wg25r/review_agent/human_reviews/kkGIbmpCHU.md`, avg 4.75, Reject) — autoregressive tabular synthesis with notation and scope weaknesses but a sound core method. Tabby has more severe claim contradictions.
- **Low:** j5EbZEyK9I (`/home/wg25r/review_agent/human_reviews/j5EbZEyK9I.md`, avg 4.50, Reject) — abstract claims directly contradicted by results tables, similar to Tabby’s “4 out of 6” claim. tFpqGk5hR5 (`/home/wg25r/review_agent/human_reviews/tFpqGk5hR5.md`, avg 4.25, Reject) — unfair baseline comparisons, analogous to Tabby’s parameter confounding.

**Reasoning:** This paper sits between the low and medium anchors. It shares the fatal flaw of claim-table contradiction with j5EbZEyK9I (4.50) and the unfair comparison issue with tFpqGk5hR5 (4.25). However, unlike those papers, Tabby does introduce a coherent architectural idea and a genuinely useful diagnostic (per-column loss tracking). The parameter confounding and overstated claims mean the core empirical contribution is not credibly established, warranting a score below the medium band. Relative to the low anchors, Tabby is less fundamentally broken (no invalid proofs or completely missing baselines), so it sits slightly above the very bottom of the low range.

**Score: 4.0**

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>