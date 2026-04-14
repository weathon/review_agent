=== CALIBRATION EXAMPLE 46 ===

# Final Consolidated Review
## Summary

StepProof proposes a step-by-step autoformalization strategy in which each sentence of a natural language mathematical proof is individually formalized into an Isabelle `have` sub-proof, pushed onto a formal proof stack, and verified incrementally. This contrasts with the prevailing FULL-PROOF paradigm, which generates and verifies the entire proof in one shot. The paper evaluates StepProof against FULL-PROOF and prior baselines on GSM8K, reports efficiency and stability improvements, and introduces a "step pass rate" metric for partial-credit evaluation of autoformalization systems.

---

## Strengths

- **4× reduction in proof-time variance with concrete evidence**: The variance in proof time drops from 20,864.97 s² (FULL-PROOF) to 5,271.65 s² (STEP-PROOF), a 4× improvement in stability. This is a specific, quantified advantage that is directly attributable to the incremental generation strategy and not a generic property of systems papers.

- **Sample efficiency over DTV**: StepProof with Llama3 8B achieves 27.9% proof passing rate in 10 attempts versus DTV*'s 25.3% in 64 attempts — a 6.4× reduction in the number of tries needed for equivalent performance. The comparison is on equal model footing (both use Llama3 8B), making this a clean methodological advantage.

- **First benchmark of autoformalization on small open-source LLMs**: The entire prior literature (Majority Voting with Minerva 8B, DTV with Minerva/GPT-3.5) relies on closed-source or restricted-access models. Demonstrating a working pipeline on Llama3 8B and GLM4 9B (4-bit quantized) is a concrete contribution to accessibility and reproducibility.

- **Step pass rate (r_s) as a new evaluation metric**: The introduction of a partial-credit step pass rate that measures what fraction of a proof's steps were formally verified is a genuinely useful contribution to the evaluation methodology of autoformalization. Table 3 shows that after 10 attempts, 49.5% of proofs have *some* verified steps even when only 27.9% pass fully — information lost by binary metrics.

---

## Weaknesses

### Fatal

None that individually destroy the core idea of step-wise verification, but the combination of the Major weaknesses below severely limits the credibility of the reported results.

### Major

- **Dataset fundamentally mismatched with the task**: GSM8K consists of grade-school arithmetic word problems whose "proofs" are chains of arithmetic calculations (e.g., "5 apples × 3 = 15"). This is not a dataset of mathematical proofs in the formal sense. Formalizing arithmetic word-problem solutions into Isabelle HOL using only the `Main` library is a poorly motivated task — the library lacks specialized arithmetic decision procedures, which likely explains the extremely low one-attempt pass rates (5–6%). Standard benchmarks for autoformalization (MiniF2F, ProofNet, the MATH dataset with structured derivations) exist precisely for this purpose. All quantitative claims in the paper rest on this inappropriate benchmark, which limits generalizability of the findings to actual formal proof verification.

- **Marginal absolute improvements without statistical validation**: The headline improvement in one-attempt pass rate is 5.30% → 6.10%, an absolute gain of 0.8 percentage points (~10 additional proofs out of ~1,319). This is reported in the paper as a "15.1%" improvement (relative), which is technically accurate but misleading given the absolute scale. No statistical significance test is provided. At these sample sizes and pass rates, a difference of 10 proofs is easily within run-to-run noise, and the paper provides no evidence otherwise.

- **Soundness of HOLD-ed steps is unresolved**: Section 3.2 states that when QED is entered, "the system will combine all the steps to perform the final verification of the proof target." However, the paper does not clarify whether HOLD-ed steps (marked as assumed-correct but not individually verified) are included verbatim in this final Isabelle proof or whether Isabelle actually re-verifies them. If HOLD-ed steps are axiomatically assumed and Isabelle's final check does not catch them, then a proof containing HOLD-ed steps is not a formally valid proof — which would be a fundamental correctness issue for a system claiming formal verification. This is never discussed.

- **Reproducibility gaps in the core method**: Three technically non-trivial components are left undescribed: (1) how GSM8K's natural language solutions are segmented into individual verifiable steps (the paper assumes this is trivial, but multi-sentence steps, equations, and parenthetical remarks complicate this); (2) the prompt template and few-shot example used for formalization (critical for reproducibility); and (3) how individually generated `have` statements are assembled into a coherent Isabelle theory file respecting scoping rules. Without these, the method cannot be reproduced.

- **Novelty over prior decomposition methods is understated**: The paper acknowledges LEGO-Prover (Wang et al., 2023) in the related work and claims the only distinction is that StepProof avoids "extra generation of sub-proof formal statement generation." This single sentence is the entire novelty argument over LEGO-Prover. No ablation, no quantitative comparison, and no formal characterization of what that difference means in practice is provided. The relationship to DSP (Jiang et al., 2022), which also guides formal provers with informal proof sketches decomposed into steps, is similarly underdeveloped.

### Minor

- **Table 1 notation error**: The columns are labeled $\mu_f \pm \sigma_f^2$, mixing mean (seconds) with variance (seconds²). The reported values (e.g., 9.54 ± 12.64s) are almost certainly mean ± standard deviation, not mean ± variance. This is a systematic mislabeling that introduces confusion about what is actually being reported.

- **HOLD frequency is not quantified**: The paper never reports how often users actually invoke HOLD during the experiments. If HOLD is used frequently, the reported proof pass rates conflate model capability with user intervention, and the "autoformalization" framing is misleading. Quantifying the frequency of HOLD usage is essential to interpreting the results.

- **"Comments Rate" is a trivially true metric**: In Table 2, StepProof achieves a 100% Comments Rate by construction — every step gets step-level feedback. Presenting this as a performance metric alongside other methods implies it is an achievement, when it is simply an architectural property. It adds no comparative information.

- **Missing bibliographic entry**: "Qinghua et al." and "SlideRule" are cited in Section 2 but do not appear in the reference list. This is a concrete citation omission.

- **DTV\* re-implementation claim**: The paper replaces DTV's original Minerva/GPT-3.5 backbone with Llama3 8B, labels the result DTV*, and then claims to outperform it. While the authors acknowledge this in a footnote, the claim of outperforming "existing methods" should be more carefully qualified — the re-implemented DTV* is operating at a disadvantage relative to its original design, and comparisons against the original DTV results (not shown) would tell a different story.

### Tiny

- Section 4.3's claim that "the main limitation to step pass rate lies not in the model's formalization ability, but in whether the informal proof steps are suitable for conversion into provable formal steps" is an important observation but is asserted without supporting evidence (e.g., manual categorization of failure modes into formalization errors vs. unformalizable steps vs. tactic failures). This should be substantiated.
- The use of "machine code" in Section 1 to describe ITP scripts is imprecise; standard terminology is "formal scripts" or "proof terms."

---

## Nice-to-Haves

- Evaluate on MiniF2F or ProofNet, which have ground-truth formal proofs, to situate StepProof within the mainstream autoformalization literature and allow direct comparison with methods designed for those benchmarks.
- Provide a failure mode breakdown: what fraction of step failures are due to LLM hallucinating Isabelle syntax, inability to find a tactic, or logical gaps in the informal step? This would validate the claim that step-wise verification helps localize errors.
- Report total token usage and wall-clock time per proof for StepProof vs. FULL-PROOF, since the step-by-step approach incurs multiple sequential LLM calls and the efficiency trade-off deserves explicit treatment.
- Include a side-by-side case study where FULL-PROOF fails on a proof that StepProof succeeds on, demonstrating the core value proposition concretely.
- Ablate the effect of providing accumulated verified steps as context for subsequent step formalization — this is the key mechanism claimed to reduce error propagation, but its contribution is never isolated.
- Release prompt templates, Isabelle interaction scripts, and dataset subsets.

---

## Removed Points

*These points are flagged for removal — treat them with caution.*

- **Unfair comparison (Majority Voting uses Minerva 8B, StepProof uses Llama3 8B)**: The harsh critic raises this as a weakness, but per evaluation policy, comparisons where the unfairness is favorable to the *baseline* (Minerva 8B is a stronger math model than Llama3 8B general-purpose) and not the paper's method are intentionally asymmetric to prove a stronger point. StepProof outperforming Majority Voting despite using a weaker model is actually a stronger statement. **Removed.**

- **No pseudocode/algorithm box**: A formatting nitpick with no bearing on the method's validity. The workflow is adequately described in prose and Figure 1. **Removed.**

- **"Chapter" instead of "Section"**: A pure stylistic choice, not a scientific flaw. **Removed.**

- **Contribution 3 ("significantly improved in all aspects") is contested**: The paper does show improvements in pass rate, formalization time (−38.9%), proof time (−39.5%), and variance across the board in Table 1. The criticism that this is not "significant" may be overstated given the consistency of directional improvements. **Weakened to minor; kept as the statistical significance concern instead.**

- **Contribution 2 ("first to test on small open-source LLMs") is a weak ICLR contribution**: While the harsh critic is right that this alone is not a strong research contribution, the paper frames it as filling a gap in the literature (prior work exclusively uses Minerva/GPT-3.5), which is a legitimate motivation even if not the primary contribution. **Removed as a standalone weakness; absorbed into the broader novelty concern.**

---

## Novel Insights

The spark finder raises one genuinely insightful point beyond the paper's own analysis: the step pass rate distribution in Table 3 reveals that after 10 attempts, 49.5% of proofs achieve some degree of formal verification (r_s > 0) while only 27.9% complete fully — suggesting that the bottleneck is not uniform across proof steps. This implies that selectively targeting the "hardest" steps (those most often failing even after 10 retries) for human annotation or specialized model fine-tuning could unlock substantially higher full-proof pass rates with minimal additional effort. The paper does not exploit this decomposition, which represents a missed analytical and practical opportunity. Additionally, Table 4's finding that minor manual rewrites of informal proofs doubled the full-proof pass rate (6% → 12%) — without changing the model — suggests that proof *writing style* is a first-class variable in autoformalization performance, a finding that has implications for how training data for autoformalization should be curated.

---

## Suggestions

1. **Replace or supplement GSM8K with MiniF2F or ProofNet**: These benchmarks have ground-truth Isabelle/Lean formalizations, making them appropriate for evaluating autoformalization. If GSM8K is retained for comparison to prior baselines, add at least one supplementary experiment on a proper proof benchmark.

2. **Report HOLD usage statistics**: For every experiment, report the mean number of HOLD invocations per proof and the fraction of ultimately passing proofs that used at least one HOLD. This directly addresses the automated vs. interactive ambiguity.

3. **Clarify HOLD soundness semantics**: Explicitly state whether Isabelle re-verifies HOLD-held steps in the final QED check. If it does, say so. If it does not, acknowledge that HOLD-inclusive "passing" proofs are not formally complete and revise the claims accordingly.

4. **Provide a prompt template and example in an appendix**: Even a single full example of the few-shot prompt used for step formalization would substantially improve reproducibility.

5. **Add statistical significance testing**: Given the small absolute improvements (0.8pp), even a simple binomial test for the difference in pass rates between FULL-PROOF and STEP-PROOF would strengthen the empirical claims.

6. **Clarify DTV\* framing**: Either compare to DTV under its original setup (with results from the original paper) and position StepProof+Llama3 as a resource-constrained comparison, or explicitly frame all DTV\* results as a controlled model-matched ablation, not as outperforming the original DTV.

---

**Evaluation summary:**

- *Novelty*: Low-to-moderate. The core idea of step-wise decomposition for autoformalization is incremental over LEGO-Prover and DSP, and the novelty delta is not convincingly demonstrated.
- *Technical soundness*: Weak. The method has reproducibility gaps, the HOLD mechanism's soundness implications are unaddressed, and the notation error in Table 1 reflects insufficient care in experimental reporting.
- *Empirical support*: Weak. The dataset choice (GSM8K) is inappropriate for the task; absolute improvements are marginal without statistical validation; the most important baseline (DTV) is tested in a non-standard configuration.
- *Significance*: Moderate potential, currently limited. The step pass rate metric and the finding that proof writing style strongly affects formalization success are genuinely useful. The system, if demonstrated on proper benchmarks with stronger results, could be a meaningful tool for the community.
- *Clarity*: Below the expected standard for ICLR. Key technical details are missing, and the relative vs. absolute improvement framing is misleading.

# Actual Human Scores
Individual reviewer scores: [6.0, 1.0, 3.0, 3.0]
Average score: 3.2
Binary outcome: Reject
