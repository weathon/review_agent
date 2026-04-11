=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
## Summary
HUBBLE is a suite of fully open-source large language models (1B and 8B parameters) designed for controlled scientific study of memorization risks. The models are trained with systematic insertions of text (e.g., book passages, biographies, test sets) at known duplication rates across copyright, privacy, and test contamination domains. Core findings suggest memorization can be mitigated by diluting sensitive data via larger training corpora and by ordering it to appear earlier in training. The suite also serves as a benchmark for membership inference attacks and machine unlearning evaluation.

## Strengths
- **Open-source resource enabling causal memorization studies:** The release of models, code, datasets, and training configurations with controlled perturbations at varied duplication rates allows researchers to move beyond observational studies and perform precise causal analyses of memorization dynamics.
- **Systematic and rigorous experimental design:** The paper employs a factorial design (model size × data condition × corpus size) plus specialized runs (timing, interference, paraphrased, architecture) to isolate effects of dilution, ordering, scale, and domain interactions, supported by extensive figures and tables.
- **Policy-aware perturbation framework:** Perturbations are carefully selected based on a survey of legal risks in copyright, privacy, and test contamination, grounding technical findings in real-world regulatory concerns and connecting empirical results to societal impacts.
- **Demonstrated utility for broader research communities:** HUBBLE addresses flaws in existing benchmarks (e.g., spurious features in WIKIMIA) by providing a sound testbed for membership inference attacks and machine unlearning, with controlled member/non-member splits and known duplication levels.

## Weaknesses
### Major
- **Ordering claim is not fully substantiated by the experiments.** The timing runs insert perturbations only in isolated intervals (e.g., first or last quarter of training) and show that data inserted early and not revisited is forgotten. However, the paper does not compare against a baseline where the same total number of duplicates are distributed uniformly throughout training. Without this control, the claim that "ordering sensitive data early reduces memorization risks" as a general best practice for data that appears multiple times is overstated.
- **Contamination flaw in the ELLie dataset undermines test-set analysis.** As acknowledged in Appendix D.3, ELLie examples are minimal pairs, and insertion leads to high accuracy even on zero-duplicate examples, invalidating its use for studying dilution. Despite this, ELLie is included in the test contamination evaluations (Figure 10), casting doubt on the rigor of results in that domain.
- **Critical interference check is absent at the scale of core findings.** The interference experiment, which verifies that perturbations from different domains do not interact, is conducted only with 1B models on 100B tokens. The primary dilution and ordering results rely on 8B models trained on 500B tokens; the lack of interference verification at that scale leaves open the possibility that domain interactions could confound the agnostic conclusions.

### Minor
- **Dilution effect is demonstrated only for fixed absolute frequency.** The experiment holds the number of perturbation tokens constant while varying corpus size (100B vs 500B). It does not explore whether dilution holds when the proportion of sensitive data is varied independently of total tokens, which would strengthen the generalizability of the finding.
- **Insufficient analysis linking metrics to practical risks.** The paper shows that different memorization metrics (loss, k‑eidetic, ROUGE) give different signals but does not discuss which metrics best correlate with real‑world harms like verbatim extraction for copyright or attribute inference for privacy, limiting the policy relevance of the evaluations.
- **Impact of architectural modifications on memorization is unexamined.** HUBBLE modifies the Llama architecture (e.g., OLMo tokenizer, untied weights, different layer counts). No ablation study confirms that these changes do not alter memorization dynamics compared to the base architecture, potentially affecting the generalizability of findings to other model families.

### Trivial
- None after applying review filters.

## Nice-to-Haves
- Include a uniform-distribution baseline for the ordering experiments to directly compare front-loaded, back-loaded, and evenly spread duplicates.
- Conduct the interference experiment with 8B models on 500B tokens to confirm domain independence at the scale of the main findings.
- Provide case studies illustrating success and failure modes for PII extraction to ground evaluation metrics in realistic attack scenarios.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Strengths about general readability or importance of the topic:** Removed as generic praise.
- **Weaknesses about high computational cost limiting reproducibility:** Removed as the paper provides detailed GPU hours and all assets are open-source, making reproduction feasible for the target academic audience.
- **Criticisms about readability and appendix integration:** Removed as stylistic nitpicks.
- **Claim that findings are incremental:** Weakened; the novelty lies in the comprehensive suite and multi‑domain validation rather than isolated mechanisms.
- **Concerns about distributional confounds from inserted data:** Weakened; the paper decontaminates the base corpus and checks interference, though some risk remains, it does not invalidate the core results.

## Suggestions
- Revise the ordering claim to clarify that it is based on insertion in isolated intervals and that further work is needed to test different distributions of duplicates (e.g., uniform vs. front-loaded).
- Remove or prominently caveat the ELLie results in the test contamination analysis due to the minimal-pair issue, or conduct additional decontamination to isolate dilution effects.
- In the discussion, explore how the dilution and ordering principles might extrapolate to larger-scale models (e.g., trillion-token training) and whether diminishing returns or breakdowns could occur.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
