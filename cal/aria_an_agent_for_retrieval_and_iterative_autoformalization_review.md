=== CALIBRATION EXAMPLE 25 ===

# Final Consolidated Review
## Summary

Aria is an agent for auto-formalization of mathematical statements into Lean 4 that integrates a Graph-of-Thought (GoT) dependency decomposition, Retrieval-Augmented Generation (RAG) via LeanSearch, and compiler-in-the-loop reflection to handle conjecture-level problems requiring synthesis of novel definitions. The paper also introduces AriaScorer, a semantic verification module that grounds evaluation in authoritative Mathlib term definitions rather than surface textual similarity, and demonstrates state-of-the-art performance across multiple benchmarks.

## Strengths

- **GoT architecture is a genuinely novel and well-justified design for handling conjecture-level formalization.** Unlike single-pass methods, the recursive decomposition-then-synthesis pipeline mirrors how human mathematicians actually work: breaking a complex statement into prerequisite concepts, grounding known ones, and constructing new ones bottom-up. The ablation on the Conjectures dataset (6/14 → 1/14 without GoT) and the case studies in Appendix A (e.g., correctly using `Submodule R R` vs `Submodule (MulOpposite R) R` for one-sided ideals) provide concrete evidence that the architecture solves a real problem that simpler approaches cannot.

- **AriaScorer's term-level grounding is a specific, substantive improvement over prior semantic checkers.** The case studies in Appendix B compellingly demonstrate failure modes that purely textual comparison misses: `QuaternionGroup 1` (order 4) vs `QuaternionGroup 2` (order 8, the actual $Q_8$), and `QuaternionAlgebra R a b c` having multiplication rules $i^2 = a + bi$ rather than $i^2 = a$. These are not hypothetical—they are real errors caught by term-level retrieval that LeanScorer misses. The F1 improvement (93.5% vs 82.1% for LeanScorer at α=0) on expert-annotated data quantifies the benefit.

- **Strong empirical results on FATE-X, where the difficulty is appropriate for the method's claims.** The 44.0% vs 24.0% (Goedel-V2 pass@128) final accuracy on FATE-X—problems from PhD qualifying exams and research literature—is a meaningful gap. Unlike the Conjectures dataset, FATE-X has enough problems to make this result statistically informative.

- **Ablation studies clearly demonstrate that all three components (RAG, GoT, Reflection) are necessary, especially on harder datasets.** The complete collapse on Conjectures when ablating RAG (42.9% → 0%) or Reflection (42.9% → 0%) and the substantial drop from GoT (42.9% → 7.1%) provide clear evidence for the architectural design.

## Weaknesses

- **AriaScorer is validated exclusively on Aria's own output distribution, creating potential evaluation circularity.** Section 4.3.1 states: "The evaluation used the Aria agent's syntactically correct, auto-formalized outputs." Since AriaScorer is then used as the "Final accuracy" metric for *all* methods in Table 1, there is a risk that it is systematically more lenient toward Aria's particular output style (e.g., its preference for decomposed, modular definitions) while penalizing baselines for valid but stylistically different formalizations. Cross-system validation—testing AriaScorer on errors from Goedel-V2, Gemini, and Herald—would be necessary to establish it as a fair general-purpose metric. Without this, the reported final accuracy advantage over baselines may be overstated.

- **The paper does not isolate the contribution of the backbone LLM from the pipeline architecture.** Aria uses Gemini-2.5-Pro as its backbone, while the specialized baselines (Herald, Kimina) are 7B–32B models, and Goedel-V2 is 32B. The paper never ablates the backbone: a simple experiment running Gemini-2.5-Pro with single-step RAG or multi-sample self-consistency (without the full GoT agent) would reveal how much of the performance gap comes from using a stronger base model versus the architectural innovation. Given that Gemini-2.5-Pro (pass@1) already achieves 27.8% final accuracy on ProofNet vs Herald's 18.3%, the backbone alone may account for a substantial portion of the gains. This is a significant gap in the evaluation.

- **The Conjectures dataset (N=14) is too small to support the headline claim of "breakthrough performance."** The 42.9% figure corresponds to exactly 6 successes. A single additional success or failure shifts the rate by ~7 percentage points. The baselines all scoring 0% suggests the task is extremely sensitive to specific library knowledge rather than broadly measuring formalization capability. The paper does not state selection criteria for these 14 conjectures, making it unclear whether they are representative or conveniently suited to Aria's architecture.

- **The metric for the Conjectures column in Table 1 is inconsistent with the paper's own definitions.** The caption states "Results for the Conjectures dataset were manually verified," yet the column is labeled "Final acc." which is defined throughout the paper as passing AriaScorer. The reader cannot determine whether the Conjectures results passed AriaScorer, manual expert review, or both. This inconsistency must be clarified—if Conjectures passed only human review while other benchmarks passed AriaScorer, the metrics across columns are not comparable.

- **The core contribution of definition synthesis is not quantified.** The paper positions definition synthesis as a key differentiator ("the first agent capable of autonomously synthesizing the complex novel definitions"), but never reports how many of the successful formalizations actually required synthesizing new definitions versus retrieving existing Mathlib ones. If most successes relied purely on retrieval with GoT planning, the synthesis contribution is less central than claimed; if all 6 successful conjectures required synthesis, this should be stated explicitly.

- **Synthesized definitions risk creating "island formalizations" incompatible with Mathlib's typeclass hierarchy.** For example, in Appendix A.2, Aria defines `IsNoetherianLocalRing` as a new class extending `IsNoetherianRing` and `IsLocalRing`, rather than using Mathlib's existing composition of these typeclasses. Such locally synthesized definitions may not interoperate with existing Mathlib lemmas and instances, limiting the practical utility of the formalized statements for downstream proof automation. The paper does not discuss this limitation.

- **Termination conditions for GoT recursion are not formally specified.** Section 3.1.1 describes top-down expansion "until all leaf nodes can be grounded in Mathlib," but does not define a maximum depth bound or timeout. If the LLM hallucinates a dependency chain that never bottoms out—generating ever-more-obscure prerequisite concepts—the system could enter an infinite regress. The compiler-in-the-loop reflection catches syntax errors but not logical circularity or unbounded expansion in the dependency graph.

- **The cost comparison does not reflect actual computational expenditure.** The paper compares API call counts (17.7 for Aria vs. pass@k for Goedel-V2), but Aria calls Gemini-2.5-Pro—a large frontier proprietary model—while Goedel-V2 is a 32B open-weight model with substantially lower per-call cost. The actual dollar-cost or FLOPs comparison may be dramatically different from what the call-count comparison implies.

## Nice-to-Haves

- Pairwise ablation studies (e.g., GoT+RAG without reflection) to better understand component interactions, especially given that the three modules are deeply coupled.
- A human expert baseline on the Conjectures dataset (success rate and time required) to contextualize the 42.9% figure.
- A failure mode breakdown on the Conjectures dataset—what types of conjectures does Aria still fail on, and why (compilation error, synthesis failure, semantic mismatch)?
- Release of the 14 homological conjectures as a standardized benchmark with ground truth formalizations.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Weakness: Appendix D omits the AriaScorer prompt.** While the scorer prompt is not included, the paper describes the scorer's architecture in detail (Section 3.2, Figure 2) and provides the prompts for all generation components. The scorer prompt is a specific implementation detail; the evaluation methodology and results are sufficient to assess the claim. Removed as a reproducibility nitpick per hard rules.
- **Weakness: Dependency on external tools (jixia, LeanSearch, Herald dataset) for evaluation.** These are cited in the paper and assumed to exist per hard rules. Concerns about their long-term maintenance are speculative. Removed per hard rules.
- **Weakness: LeanScorer is a re-implementation, not the original system.** The authors explicitly acknowledge this and use the re-implementation as an ablation (their pipeline minus term-level grounding), which is a reasonable and transparent approach. Removed as the comparison is properly contextualized.
- **Weakness: Formatting artifacts in Appendix B.3 obscure the case study.** Removed as a formatting nitpick per hard rules.
- **Weakness: Environmental impact of scaling API calls.** This is outside the paper's stated scope and is scope creep. Removed per soft rules.
- **Weakness: No adaptive mechanism to switch GoT on/off based on difficulty.** The paper already documents the trade-off (Section C.2, Table 5) and explains why GoT hurts compilation on simpler problems. Requesting an adaptive mechanism is a nice-to-have extension, not a weakness of the current contribution. Removed per soft rules.

## Novel Insights

The most striking tension in this paper is the **syntactic risk vs. semantic rigor trade-off** introduced by GoT: on simpler problems (FATE-H), the modular decomposition style *hurts* compilation success (89% → 95% without GoT) because it increases the attack surface for typeclass resolution failures, namespace conflicts, and interface mismatches—yet it *improves* final accuracy (71% vs 54%) because the decomposed structure forces semantic explicitness. This means GoT is not simply "better"—it trades one failure mode (syntactic fragility from modular code) for another (semantic fragility from monolithic code). The paper's own data suggests that an optimal system might conditionally activate GoT based on estimated problem complexity, and that the real open problem is not *whether* to decompose, but *how to ensure modular definitions interoperate* within Lean's typeclass resolution system. This insight has implications beyond this paper for any agentic formalization system.

## Suggestions

- Run Gemini-2.5-Pro with a simple RAG+reflection baseline (no GoT) on FATE-X and Conjectures to isolate the GoT architectural contribution from the backbone model contribution. This single experiment would significantly strengthen (or honestly weaken) the core claim.
- Validate AriaScorer on outputs from at least one baseline system (e.g., Goedel-V2) to demonstrate that it generalizes as a fair metric, not just as a detector of Aria-specific error patterns.
- Clarify the Conjectures evaluation protocol in Table 1: were these results judged by AriaScorer, by human experts, or by both? If both, report both numbers separately.
- Report how many of the successful formalizations across benchmarks actually required synthesizing new definitions (vs. pure retrieval), and release the 14 conjectures with selection criteria.

## Quality Assessment

- **Novelty:** High. The GoT decomposition-synthesis pipeline and AriaScorer's term-level grounding are distinct architectural contributions that go beyond incremental improvements to existing auto-formalizers.
- **Technical soundness:** Moderate. The system design is coherent and well-engineered, but the evaluation has a significant gap (backbone model not ablated) and a validity concern (AriaScorer validated only on Aria's outputs).
- **Empirical support:** Moderate-to-good on FATE-X and FATE-H; weak on Conjectures due to sample size and metric inconsistency. The lack of backbone ablation is a notable gap.
- **Significance:** High potential. If the results hold under fairer evaluation conditions, this represents a meaningful advance in auto-formalization, especially for research-level mathematics where all prior methods fail entirely.
- **Clarity:** Good. The paper is well-organized with clear methodology descriptions, though the Conjectures metric inconsistency and some undefined termination conditions detract from precision.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 4.0, 4.0, 8.0]
Average score: 5.3
Binary outcome: Accept
