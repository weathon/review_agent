## Summary

This paper proposes StepProof, an autoformalization strategy that verifies natural language mathematical proofs sentence-by-sentence using an incremental proof stack in Isabelle/HOL, allowing fine-grained backtracking. The authors evaluate the approach against a full-proof baseline on GSM8K using small open-source LLMs (Llama3 8B, GLM4 9B), reporting efficiency gains and higher pass rates, and provide a prototype interactive UI.

## Strengths

- **Novel stepwise verification workflow.** StepProof implements a concrete incremental proof stack where each sentence is formalized, verified, and pushed onto the stack; upon failure, only the erroneous step is cleared while verified steps are retained (Section 3.2, Figure 1). This differs from prior end-to-end full-proof strategies (DSP, DTV) and from LEGO-Prover, which requires extra sub-proof statement generation.
- **Controlled internal comparison showing efficiency and stability gains.** On the same GSM8K dataset and Llama3 8B model, STEP-PROOF reduces average formalization time by 38.9% (5.83s vs. 9.54s), reduces average proof time by 39.5% (130.12s vs. 214.93s), and cuts proof-time variance by roughly 75% (5271.65 vs. 20864.97) compared to FULL-PROOF (Table 1, Section 4.2).
- **Evaluation on accessible small open-source models.** The paper tests on Llama3 8B and GLM4 9B (4-bit), whereas much prior work relies on closed-source Minerva and GPT-3.5, partially filling a gap in the literature (Section 4.1).
- **Empirical observation that proof writing style affects formalization success.** A manual-modification experiment on 100 MATH Number Theory problems shows that adapting informal proofs for step-level verification doubles the full step-pass rate from 6% to 12% (Table 4), suggesting STEP-PROOF benefits from more structured proof styles (Section 4.2).
- **Concrete UI instantiation.** Figure 2 provides a screenshot of the interactive interface supporting per-step PROOF, REGEN, HOLD, and UNDO, clarifying the intended interaction mode.

## Weaknesses

### Fatal

None.

### Major

- **Benchmark mismatch undermines generalizability claims.** The paper motivates its work with broad claims about verifying “natural language mathematical proofs” (Abstract, §1) and “mathematical works” across “many branches of mathematics” (§1). However, all experiments are conducted on GSM8K, a dataset of grade-school arithmetic word problems with chain-of-thought solutions. GSM8K steps are short arithmetic calculations, not the kind of formal mathematical proofs typically studied in autoformalization (e.g., miniF2F, ProofNet). Because the paper never tests on a standard theorem-proving benchmark, its central claim—that StepProof enables sentence-level verification of mathematical proofs—lacks appropriate empirical support.
- **Baseline comparisons are confounded and misleading.** Table 2 compares StepProof against Don’t Trust: Verify (DTV) and Majority Voting, but the comparisons are unfair. Footnote 2 states that the DTV baseline is a reimplementation that substitutes Llama3 8B for the original GPT-3.5 and Minerva 8B models. Majority Voting uses Minerva 8B with 64 attempts, while StepProof uses Llama3 8B or GLM4 9B with only 10 attempts. Claiming that StepProof “surpassed DTV” (§4.2) and “reached the level of state-of-the-art” (§6) is invalid because the reported differences confound model capability, compute budget, and strategy.
- **Interactive benefits are asserted but never empirically evaluated.** The paper motivates STEP-PROOF through user-facing benefits: fine-grained error localization, backtracking to preserve correct steps, and HOLD/UNDO functionality (Figures 1–2, §3.2). Yet the experiments are fully automated batch tests measuring only aggregate pass rates and wall-clock time. No user study, simulated interaction trace, or measurement of iteration cost is provided, leaving the core interactive contribution unsubstantiated.

### Minor

- **No algorithmic specification.** The description of STEP-PROOF lacks pseudocode or a formal algorithm specifying step segmentation, proof-stack management in Isabelle, and backtracking mechanics (§3.2). The current prose description is insufficient to assess full reproducibility and to delineate technical novelty from LEGO-Prover beyond the high-level distinction noted.
- **Step pass rate metric is unvalidated.** The paper introduces $r_s$ (Table 3) to quantify partial proof success, reporting that 38.1% of proofs have at least half of their steps formally verified. However, GSM8K steps are arithmetic calculations, not independently provable lemmas, and the metric is never validated against any ground-truth utility (e.g., whether partial step verification correlates with user utility or eventual full-proof success). Its interpretation is therefore unclear.
- **Time improvements are confounded by generation settings.** STEP-PROOF uses max_new_tokens of 256 per step, while FULL-PROOF uses 1024 for the entire proof (§4.1). The observed 39% reductions in formalization and proof time are therefore partly attributable to shorter generation budgets rather than solely to the algorithmic strategy.
- **Small absolute gain without significance testing.** The one-attempt proof pass rate improves from 5.30% to 6.10% (Table 1), an absolute difference of 0.8 percentage points that is marketed as a “15.1% relative improvement” (§4.2). No statistical significance testing is provided, and on a dataset with rates near floor performance, this gap may be indistinguishable from noise.

### Trivial

None.

## Nice-to-Haves

- Evaluate on a standard autoformalization benchmark (e.g., miniF2F, ProofNet) rather than relying solely on GSM8K.
- Conduct a fair baseline comparison by running FULL-PROOF and StepProof with the same LLM, equivalent total compute budget, and statistical significance tests.
- Perform a human-in-the-loop or simulated interaction study measuring whether step-level feedback and backtracking reduce iteration count or time to a correct proof.
- Include complete end-to-end case studies of actual GSM8K examples showing StepProof and FULL-PROOF outputs with annotated failure points.

## Removed Points

These points are flagged to be removed, treat them with caution.

- *Criticism that §4.3 “indicts the dataset choice rather than supporting the method’s superiority.”* The sentence “many steps in the test set cannot be formalized into provable steps” is an honest author acknowledgment of a limitation, not an inadvertent indictment.  
- *Criticism that complaining about LLMs failing to adjust `max_new_tokens` is “just a hyperparameter issue.”* The paper reasonably identifies this as a practical drawback of FULL-PROOF when proof lengths vary.  
- *Any formatting, grammar, or typo complaints.* These are parser artifacts, not author errors.  
- *Missing appendix or missing proofs.* The parser strips appendix sections; they exist in the original submission.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

- Reframe the evaluation either as autoformalization of chain-of-thought reasoning or, preferably, re-run experiments on a standard theorem-proving benchmark with rigorous, fair baselines.
- Add pseudocode for the proof-stack management and backtracking procedure.
- Either empirically validate the interactive benefits (e.g., via a user study) or narrow the claims to focus on the automated batch strategy.

## Score and Decision

**Calibration reasoning:**

I compared this paper against the following anchors from the human-review corpus:

- **Low anchors (≤4):** *DS-Prover* (avg 3.75, Reject) and *SubgoalXL* (avg 3.75, Withdrawn/Reject). DS-Prover was criticized for misleading SOTA claims and unfair baseline comparisons, though it used the correct benchmark (miniF2F). SubgoalXL used the correct benchmark but was flagged for severely unfair baseline comparisons (search budget 16,384 vs. 100–200) and potential data contamination. StepProof shares the unfair-baseline and misleading-SOTA problems, and additionally uses GSM8K—an inappropriate benchmark for general mathematical proof verification—which is a more fundamental mismatch than DS-Prover’s or SubgoalXL’s issues. However, StepProof’s internal FULL-PROOF vs. STEP-PROOF comparison is cleanly controlled, and its core strategy is conceptually more distinct.

- **Medium anchor (~5):** *Synthetic Theorem Generation in Lean* (avg 5.00, Reject). This paper used the correct benchmark (miniF2F) but showed only marginal improvements (≈1% absolute), lacked strong baselines, and was considered low-impact. StepProof shows larger internal relative gains, but its benchmark mismatch and invalid external baselines make its empirical argument weaker overall.

- **High anchors (≥6):** *LEGO-Prover* (avg 7.50, Accept oral) achieved strong results on miniF2F with a novel growing-library concept and rigorous evaluation. *Lyra* (avg 6.00, Reject despite score) achieved SOTA on miniF2F but was rejected for limited technical novelty; even so, it used the correct benchmark and fair internal comparisons.

Relative to these anchors, StepProof sits below the medium anchor (5.00) because its core empirical claims are undermined by the wrong benchmark and broken baselines, but slightly above the low anchors (3.75) because the stepwise strategy itself is a reasonable, novel contribution and the internal comparison is fair. I therefore assign a score of **4.0**.

**MY FINAL SCORE: <pineapple>4.0</pineapple>**
**MY FINAL DECISION: <orange>Reject</orange>**