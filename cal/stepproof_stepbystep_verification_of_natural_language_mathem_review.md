=== CALIBRATION EXAMPLE 4 ===

# Final Consolidated Review
## Summary
StepProof proposes a step-by-step autoformalization strategy for verifying natural language mathematical proofs at the sentence level, contrasting with traditional "FULL-PROOF" approaches that formalize entire proofs at once. The method incrementally formalizes each proof step, pushes verified steps onto a stack, and allows users to backtrack on failures while preserving previously verified steps. Experiments on GSM8K and a subset of MATH with open-source LLMs (Llama3 8B, GLM4 9B) report improvements in pass rates and efficiency compared to baseline methods.

## Strengths
- **Open-source model evaluation:** Unlike prior autoformalization work relying on closed-source models (GPT, Minerva), this paper explicitly tests on accessible LLMs (Llama3, GLM4), enhancing reproducibility and enabling community follow-up experiments.
- **Granular error localization:** The step-by-step verification strategy addresses a genuine limitation of full-proof approaches—where a single error invalidates the entire proof—by allowing users to isolate, identify, and retry specific failing steps without discarding verified work (Section 3.2, Figure 2).
- **Interactive interface design:** The user interface (Figure 2) provides visual feedback (green/yellow highlighting for verified/held steps) and practical controls (HOLD, REGEN, UNDO), demonstrating thoughtful consideration of human-in-the-loop formal verification workflows.

## Weaknesses
- **Inappropriate primary evaluation dataset:** GSM8K consists of grade-school arithmetic word problems with informal reasoning traces, not deductive mathematical proofs suitable for Isabelle formalization. The paper itself notes "many steps cannot be formalized into provable formal steps" but does not question whether GSM8K is the right benchmark. Standard autoformalization benchmarks like MiniF2F or ProofNet would be far more appropriate for validating the method on actual mathematical reasoning. The low pass rates (~6% single-attempt, ~28% after 10 retries) may reflect dataset mismatch rather than method capability.

- **Confounded baseline comparisons:** Table 2 compares StepProof against Majority Voting and DTV under non-equivalent conditions. Majority Voting uses Minerva 8B (a specialized math model) with 64 attempts, while StepProof uses Llama3 8B with 10 attempts per step. The DTV comparison substitutes Llama3 for the original models but uses different attempt counts. The claim that "StepProof required fewer attempts" (Section 4.2) is misleading—one measures total proof attempts, the other measures per-step retries. These differences make it impossible to attribute performance differences to the strategy rather than model or parameter choices.

- **Missing implementation details for reproducibility:** The paper describes the workflow but omits critical details: (a) the exact prompt templates for step formalization, (b) how step segmentation is performed (sentence boundaries? heuristics?), (c) how dependency links between steps (e.g., `using h1 h2`) are generated, and (d) handling of multi-sentence proof steps. Without these specifications, reproducing the system requires substantial re-engineering.

- **Undefined "Comments Rate" metric:** Table 2 lists a "Comments Rate" column showing 100% for StepProof and 31.3% for DTV, but this metric is never defined in the paper. This is a critical omission for interpreting the experimental results.

- **Non-standard variance reporting:** Table 1 reports $\mu_f \pm \sigma_f^2$ (mean ± variance), which is unconventional and difficult to interpret. Standard practice is to report mean ± standard deviation. The extremely large variance values (e.g., $\sigma_p^2$ = 20864.97s² for FULL-PROOF) suggest high instability that warrants deeper analysis.

- **HOLD semantics partially undermine verification claims:** The system allows users to mark steps as "assumed correct" and proceed (Section 3.2). While useful for human-in-the-loop workflows, this means a "verified" proof may contain formally unverified steps. The paper should clarify what guarantees the system actually provides—pure formal verification only applies when no steps are held.

## Nice-to-Haves
- **Evaluation on formal proof benchmarks:** Testing on MiniF2F, ProofNet, or similar datasets designed for formal verification would better establish the method's applicability to mathematical reasoning beyond arithmetic word problems.
- **Ablation on step segmentation granularity:** Understanding how performance varies with different segmentation approaches (clause-level, sentence-level, paragraph-level) would strengthen the design rationale.
- **Failure mode analysis:** The paper mentions "unformalizable informal steps" (Section 4.3) but does not quantify or categorize failure causes (LLM translation errors vs. ITP timeout vs. inherently unformalizable content).

## Removed Points
These points are flagged to be removed, treat them with caution:
- **"SlideRule citation missing":** The reviewer claims SlideRule/Qinghua et al. is uncited, but without external verification of whether this work exists, this criticism cannot be confirmed and is removed.
- **"Claiming novelty while excluding larger models":** The paper explicitly acknowledges in Section 5 that larger models were not tested due to resource constraints. Criticizing the paper for not doing something it transparently scopes out is not a substantive weakness.
- **"First to test on small open-source LLMs is not a contribution":** While this framing is modest, it does represent a reproducibility benefit for the community and can be retained as a minor positive without inflating it.

## Novel Insights
The step-by-step decomposition insight—that preserving verified partial progress reduces regeneration waste and improves error localization—is well-motivated. The analysis showing that "most steps can be proven with relatively few attempts" (Figure 4) while overall pass rates remain low suggests the bottleneck may lie in step formalization quality rather than ITP proving capability. This points to a useful direction for future work: improving the LLM's ability to produce Isabelle-compatible formalizations rather than just verification strategy. The finding that manually adjusting informal proofs to match step-verification requirements doubles the pass rate (Table 4: 6% → 12%) highlights an important practical consideration—the gap between natural mathematical writing and verifiable formal representations remains substantial.

## Suggestions
- Replace or supplement GSM8K evaluation with formal proof benchmarks (MiniF2F, ProofNet) to establish relevance to actual mathematical verification tasks.
- Define the "Comments Rate" metric explicitly or remove it from the results table.
- Report standard deviations and consider confidence intervals or statistical significance tests when claiming improvements.
- Release prompt templates, step-segmentation logic, and Isabelle interaction scripts to enable reproducibility.
- Clarify in the abstract and conclusion that the HOLD functionality provides human-in-the-loop assisted verification rather than full formal guarantees.

# Actual Human Scores
Individual reviewer scores: [6.0, 1.0, 3.0, 3.0]
Average score: 3.2
Binary outcome: Reject
