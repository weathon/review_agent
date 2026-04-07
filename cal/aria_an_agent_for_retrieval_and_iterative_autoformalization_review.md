=== CALIBRATION EXAMPLE 61 ===

# Harsh Critic Review
Now I have a thorough understanding of the paper. Let me write the review.

---

## Section-by-Section Critical Review

---

### Title & Abstract

The title accurately describes the system. The abstract's claims are largely grounded in the reported results. However, the abstract states "68.5% final accuracy" on ProofNet, framing this as a major result, but ProofNet is an *undergraduate*-level benchmark; the more distinctive contribution is on the Conjectures dataset. The abstract also conflates two separate contributions — the Aria agent and AriaScorer — without making clear that AriaScorer is *also* the evaluation metric used to compute "final accuracy" for all benchmarks except Conjectures (which was manually verified). This creates a potential self-evaluation circularity that should be flagged upfront.

---

### Introduction & Motivation

The motivation is strong and clearly articulated: LLMs hallucinate non-existent Mathlib identifiers, fail to synthesize novel definitions for unseen concepts, and existing semantic checkers rely on superficial textual similarity. The three proposed contributions (Aria agent, AriaScorer, and SOTA results) are clearly enumerated.

One gap: the introduction claims GoT "emulates human expert reasoning," but no evidence or citation for this analogy is provided. The GoT paradigm as presented here (recursive dependency decomposition) is conceptually close to dependency-retrieval methods (e.g., Liu et al., 2025a, which appears in the references). The distinction from Liu et al. 2025a's "dependency retrieval-based approach" — itself published at ICLR 2025 — deserves more explicit treatment than the brief co-citation it receives.

---

### Related Work

Coverage is reasonably comprehensive. However, the paper cites Liu et al. (2025a) as introducing "dependency retrieval" for auto-formalization, yet this method is absent from the experimental baselines in Table 1. Since that method is philosophically closest to Aria's retrieval-and-dependency approach, its exclusion from the baseline comparison requires justification. Similarly, Lu et al. (2025) introduces "conceptual RAG" for auto-formalization, published on arXiv in August 2025, yet is also absent from Table 1. The baseline set (Herald, Kimina, Goedel-V2, Gemini-2.5-Pro) is reasonable, but the most methodologically similar works are curiously omitted.

---

### Methodology (Section 3)

**GoT Decomposition Phase (3.1.1):**

1. *Termination and graph well-formedness.* The paper describes the dependency graph as "acyclic" but provides no mechanism to detect or prevent cycles during recursive expansion. In principle, two concepts could mutually depend on each other (e.g., via type class interdependencies). How is cycle detection handled? This is a genuine correctness concern for the algorithm.

2. *Depth control.* No bound on the depth of recursive decomposition is stated. For a pathological input, could the decomposition expand indefinitely? The paper mentions graphs are "relatively shallow (averaging 2–3 layers)" in FATE-X (Appendix B.4.3), but this is an empirical observation rather than a guarantee.

3. *LeanSearch failures.* When LeanSearch retrieves candidates and the LLM rejects all as non-matching, the node is flagged for synthesis. But what if LeanSearch *should* have found the concept but returned irrelevant results due to search failure? This false-negative path (wrongly triggering synthesis for an existing concept) would produce redundant new definitions that diverge from Mathlib conventions. The paper does not analyze this failure mode.

**GoT Synthesis Phase (3.1.2):**

4. *Maximum retries and fallback.* The reflection loop allows 16 attempts per node. What happens when all 16 fail? The paper doesn't specify: is the entire formalization abandoned, or does the agent proceed with an uncompiled definition? This affects the reported compiler success rates.

5. *Semantic correctness of synthesized definitions.* The paper correctly notes (Section 3.1.2) that compilation success does not guarantee semantic correctness. But AriaScorer is applied *only at the end*, to the final theorem. Semantically wrong intermediate definitions can silently propagate (as illustrated in the Catenary Ring case study, Appendix B.4.2). The paper claims this is statistically rare ("only a single instance" in FATE-X), but this analysis is based on a small dataset and the claim is not supported quantitatively.

**AriaScorer (Section 3.2):**

6. *Threshold selection.* AriaScorer is deployed with α = 0.9 "in all other experiments" (Section 4.3.2), but this threshold was selected post-hoc by observing the precision/recall trade-off table on the very dataset used for evaluation (FATE-X). This creates a risk of threshold overfitting. No held-out validation set was used for threshold selection, and the sensitivity of all downstream results to this parameter is not reported.

7. *Fuzzy integral aggregation.* The paper refers to a "fuzzy integral" for aggregating subtask labels (Perfect Match / Minor Inconsistency / Major Inconsistency → score in [0,1]) but provides no formula, no reference, and no ablation of this specific aggregation method versus simpler alternatives (e.g., majority vote, weighted average). Since AriaScorer is a core contribution, this technical gap is notable.

8. *Jixia reliability.* The term-level grounding relies on jixia (a third-party Lean static analyzer) to extract all referenced terms. No characterization of jixia's coverage, failure modes, or accuracy is provided. If jixia fails to extract a term (or extracts it incorrectly), AriaScorer's grounding step is silently incomplete.

---

### Experiments & Results (Section 4)

**Main Results (Table 1):**

9. *Which LLM drives Aria?* The paper never explicitly states which large language model is used as the backbone for Aria's decomposition, grounding, synthesis, and reflection steps. The baseline comparisons include Gemini-2.5-Pro, which appears also to power AriaScorer. If Gemini-2.5-Pro is also Aria's backbone, the comparison between "Aria" and "Gemini-2.5-Pro (pass@1)" effectively measures the value of the *scaffolding* around the same model — which is a valid and interesting comparison, but should be stated explicitly.

10. *Computational cost comparison is incomplete.* The paper compares Aria (17.7 calls/problem on FATE-X) against Goedel-V2 at pass@{16,32,64,128}. However, these are not equivalent units: Aria's calls are to a powerful frontier model (presumably Gemini-2.5-Pro), while Goedel-V2 is a 32B specialized model. Wall-clock time, dollar cost, or FLOPs are not reported. The claim that "Aria maintains higher final accuracy even when compared to Goedel-V2 at pass@128 (using more than 7x calls)" uses pass@128 as a proxy for compute, but this comparison is between models of very different sizes and costs.

11. *Conjectures dataset: extremely small N.* The Conjectures dataset has only **14 problems**. Aria's "42.9%" corresponds to exactly 6 out of 14 correct formalizations. At this sample size, reporting a percentage to one decimal place implies a precision that the data cannot support. More importantly, no statistical confidence intervals are reported anywhere in the paper. All 14 conjectures are from the same narrow domain (homological conjectures in commutative algebra), so generalizing from "0% for all baselines, 42.9% for Aria" to a general claim about "research-level formalization" is a significant overreach.

12. *How is "final accuracy" defined for baselines on the Conjectures dataset?* Table 1 notes that "Results for the Conjectures dataset were manually verified." But Table 1 shows only a single "Final acc." column for Conjectures (no Compiler column). For baselines scoring 0%, was this verified by manual inspection, or by Aria's compiler alone? The evaluation protocol for baselines on Conjectures is not fully described.

**AriaScorer Validation (Table 2):**

13. *Small evaluation set.* Table 2 implicitly has 69 total data points (50 TP + 5 FP + 12 TN + 2 FN at α=0). This is an extremely small sample for validating a semantic checker. The claimed 89.9% accuracy has a 95% CI of roughly ±7 percentage points. The paper does not report confidence intervals.

14. *Self-referential evaluation.* AriaScorer is evaluated on *Aria's own outputs* (Section 4.3.1: "The evaluation used the Aria agent's syntactically correct, auto-formalized outputs"). This is a circular evaluation: the scorer is validated on exactly the distribution it was designed to evaluate, which may not generalize to outputs from other systems. Additionally, since the AriaScorer ground truth was generated by labeling Aria's outputs, it may systematically favor Aria's formalization patterns.

15. *LeanScorer re-implementation.* The baseline LeanScorer (Xuejun et al., 2025) is re-implemented by the authors since the original is not open-source (Section 4.3.1). The fidelity of this re-implementation is unverifiable by readers and could inflate AriaScorer's relative advantage. The paper should report more detail on the re-implementation choices.

16. *Threshold inconsistency.* Section 4.3.2 states that AriaScorer at α=0 achieves 89.9% accuracy, while Section B.5 claims "achieving a 95.5% alignment rate" with experts. These numbers appear inconsistent and refer to different things, but the distinction is never clarified.

**Ablation Studies (Appendix C):**

17. *RAG ablation on FATE-X is surprisingly small.* Table 4 shows that removing RAG drops FATE-X final accuracy from 44.0% to 43.0% (just 1 absolute point), which is within noise. The compiler rate drops from 69% to 61%. The claim that "RAG is crucial" is convincingly demonstrated on the Conjectures dataset (42.9% → 0%), but the FATE-X results actually suggest the reflection mechanism can compensate for RAG's absence at this difficulty level — weakening the claim that all three components are "indispensable."

18. *The GoT ablation baseline is not a fair one-to-one comparison.* Section C.2 describes the GoT ablation as "extracting a flat list of conceptual keywords" and searching each directly, without recursive decomposition. This is a reasonable ablation, but it does not isolate GoT planning from the number of synthesis attempts — the ablated version uses the same synthesis module but with a shallower graph. A more informative ablation would have been "GoT decomposition with depth limited to 1."

---

### Writing & Clarity

The paper is written clearly overall. However:

- The relationship to the Graph-of-Thought (GoT) framework (Besta et al.) is referenced in name only without explanation of how the use here differs from or extends the original formalism. A reader unfamiliar with GoT would need to look this up.
- Section 4.2's case-study analysis ("We identify distinct shortcomings in baseline models") is presented as a systematic finding, but is actually based on informal inspection of generated codes. This should be framed more cautiously as "qualitative observation."
- Appendix B.3 (Hallucination Suppression) presents a case where Aria itself makes the error (QuaternionGroup 1 vs QuaternionGroup 2), and AriaScorer catches it. This is presented as a strength, but it also reveals that Aria's generation quality is imperfect even on manageable problems, raising questions about how many such errors exist in the passed formalizations.

---

### Limitations & Broader Impact

Section B.4 (Appendix) discusses error propagation, verification strategy, and the decision not to use AriaScorer as iterative feedback — this is commendable. However, the following limitations are not acknowledged:

- **Domain specificity of Conjectures.** All 14 conjectures are homological algebra problems. The claimed ability to handle "research-level formalization" is demonstrated in a single subfield.
- **Scalability to deeper graphs.** The paper notes FATE-X graphs average 2–3 layers, and briefly gestures at future work for textbook-scale formalization. The current system's limitations at greater depth deserve more explicit discussion.
- **Mathlib version lock-in.** LeanSearch is indexed against a specific Mathlib version. As Mathlib evolves, results may degrade or improve unpredictably.
- **No discussion of proof correctness.** The paper correctly scopes its contribution to *statement* formalization, noting that proof generation is future work. However, the value of formalizing a conjecture with `sorry`-based proofs (as in all Appendix A examples) could be more explicitly discussed — the formal statement exists but is unproven, which limits direct utility.

---

### Overall Assessment

Aria addresses a genuinely important problem — formalizing research-level mathematical conjectures — and demonstrates a qualitatively different capability from prior systems: the ability to synthesize novel Lean definitions for concepts absent from Mathlib. The GoT decomposition + RAG grounding + compiler-in-the-loop reflection architecture is technically sound in its design, and the qualitative case studies (Appendix A) make a compelling case that the system does something meaningfully new. AriaScorer's term-level grounding is a well-motivated improvement over purely text-based semantic checkers.

However, the paper has several serious weaknesses that must be addressed before acceptance at ICLR. Most critically: (1) the Conjectures benchmark has only 14 examples in a single narrow domain, making the headline result ("42.9% vs 0%") impressionistic rather than statistically reliable; (2) AriaScorer is used as both the generation metric and the evaluation metric for the main results, with threshold α selected post-hoc on the evaluation data itself; (3) the most methodologically related baseline (Liu et al., 2025a) is cited but not experimentally compared; (4) the backbone LLM powering Aria is never named, making cost/fairness analysis of the compute comparison with Goedel-V2 impossible; and (5) several algorithmic components (fuzzy integral aggregation, cycle prevention in GoT, synthesis fallback behavior) lack specification sufficient for reproducibility. These issues collectively undermine the scientific rigor of what is otherwise a promising and technically interesting contribution.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces Aria, an agentic system for auto-formalizing mathematical statements into Lean 4 that utilizes a Graph-of-Thought (GoT) pipeline for dependency decomposition/re-synthesis and a term-grounded semantic checker (AriaScorer). By integrating Retrieval-Augmented Generation (RAG) and compiler-in-the-loop reflection, Aria demonstrates state-of-the-art performance on standard benchmarks and achieves breakthrough results on research-level conjectures where baselines fail.

### Strengths
1.  **Robust Architectural Design for Complex Reasoning:** The Graph-of-Thought approach effectively handles the deep semantic dependencies of research-level mathematics. The ablation studies (Table 3/4) quantitatively demonstrate that without GoT, the system fails to generate novel definitions, as seen in the drop from 6/14 correct conjectures to 1/14 without GoT.
2.  **Rigorous Semantic Evaluation Framework:** The introduction of AriaScorer, which grounds evaluation in retrieved Mathlib definitions, addresses a critical gap in auto-formalization evaluation. Table 2 shows AriaScorer significantly outperforms baselines like Back Translation and LeanScorer in F1 score (93.5% vs 84.9%), proving its ability to catch "semantically correct but syntactically hallucinated" errors that simpler metrics miss.
3.  **Strong Performance on Research-Level Benchmarks:** The paper provides compelling evidence on the Homological Conjectures dataset (14 instances), achieving 42.9% final accuracy while competing models score 0%. This highlights a specific capability of the system to synthesize concepts not found in pre-training, which is a major bottleneck in the field.

### Weaknesses
1.  **Statistical Reliability on Key Results:** The strongest claims (42.9% on conjectures) rely on a very small dataset (N=14). The absolute number of successes is only 6 out of 14. Without variance estimation or a larger holdout set, it is difficult to determine if this result generalizes or is sensitive to specific problem selection.
2.  **Evaluation Evaluator Bias:** AriaScorer uses Gemini-2.5-Pro as its underlying judge (Section 4.3.1). Given that baselines and Aria's core agent also rely on powerful LLMs, there is a risk of "self-referential bias" or correlation between the generator's training distribution and the scorer's, despite the claim that the modules are decoupled. The 95.5% alignment with human experts needs more detail on how many samples were human-annotated for this metric.
3.  **Computational Efficiency Comparison:** While the paper compares Aria's 17.7 API calls against Goedel's pass@128, the comparison of "cost" is slightly asymmetric. Goedel's pass@128 is essentially 128 attempts, whereas Aria's 17.7 is an *average* successful inference path that still involves heavy RAG and GoT computation overhead. A more granular latency or token-budget analysis is needed to claim efficiency beyond just accuracy scaling.

### Novelty & Significance
The work is **significant** for the intersection of AI and Formal Math, as solving the auto-formalization of conjectures is widely recognized as a major open challenge. The **novelty** lies primarily in the application of GoT planning to the specific constraints of Lean's dependency graph, combined with the term-level grounding for evaluation. While GoT and RAG are known in NLP, their specific orchestration for syntactic/semantic consistency in theorem proving, validated by a static analyzer (jixia) and compiler, is a distinct contribution. It advances the field by moving beyond single-shot generation to iterative, dependency-aware reasoning.

### Suggestions for Improvement
1.  **Expand and Clarify Evaluation:** Increase the size of the Conjectures dataset for the main result disclosure or provide a cross-validation analysis to demonstrate statistical robustness. The current 14-instance set is too small for definitive claims of "state-of-the-art" on research-level problems without further validation.
2.  **Address Computational Cost Transparency:** Provide a detailed breakdown of token usage and latency per problem for Arena compared to the best performing single-pass baselines. Clarify if the 17.7 calls is a hard requirement for the full 42.9% or if fewer calls yield diminishing returns, to better contextualize the trade-off between accuracy and resource usage.
3.  **Reproducibility and Tooling:** Ensure the `jixia` static analyzer integration and specific Mathlib versions are clearly versioned in the code release. The paper relies heavily on external tools (LeanSearch, Herald), and specifying exact API versions and retrieval hyperparameters is crucial for ICLR reproducibility standards.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Expand the Conjectures dataset beyond 14 samples to ensure statistical validity of the 42.9% vs 0% claim, as small N makes SOTA claims unreliable.
2. Evaluate AriaScorer on baseline models' outputs (e.g., Goedel-V2) to prove it generalizes beyond Aria's specific generation style rather than overfitting to its own errors.
3. Quantify the percentage of successful formalizations that required *new* definition synthesis versus pure retrieval to validate the core claim of synthesizing unseen concepts.
4. Report actual wall-clock inference time per problem, as API call counts do not accurately reflect computational cost, latency, or practical viability.

### Deeper Analysis Needed (top 3-5 only)
1. Perform statistical significance testing (e.g., Fisher's exact test) on ablation results, as N=14 makes the 0% vs 42.9% difference potentially noisy.
2. Report inter-annotator agreement scores for the expert-validated Ground Truth used to evaluate AriaScorer to establish evaluator reliability.
3. Provide a detailed error taxonomy for the 57% of failed Conjectures to identify systematic weaknesses in the GoT planner rather than just highlighting successes.

### Visualizations & Case Studies
1. Include a detailed case study of a *failed* conjecture formalization to expose failure modes currently hidden by success-only examples in the appendix.
2. Visualize a specific example of a *newly synthesized* definition confirmed not to exist in Mathlib to empirically prove the synthesis capability.
3. Plot accuracy vs. computational cost (API calls) to demonstrate whether the 17.7 call average yields diminishing returns or is strictly necessary.

### Obvious Next Steps
1. Release the full source code and agent pipeline publicly to satisfy ICLR reproducibility standards, as prompts alone are insufficient.
2. Conduct a human expert review of the 42.9% "successful" conjectures to verify if they are actually merge-ready for Mathlib or merely compilable.
3. Expand the Conjectures benchmark to at least 50 problems to solidify the claim of research-level capability before claiming a breakthrough.

# Final Consolidated Review
## Summary

Aria is an agent for auto-formalizing mathematical statements into Lean 4 that combines three key components: (1) a Graph-of-Thought planner that recursively decomposes statements into dependency graphs and synthesizes novel definitions for concepts absent from Mathlib, (2) RAG-based grounding via LeanSearch that anchors concepts to existing library definitions, and (3) AriaScorer, a semantic correctness checker that retrieves authoritative Mathlib term definitions to enable more rigorous verification than text-based comparison. The paper demonstrates state-of-the-art performance on ProofNet (68.5%), FATE-X (44.0%), and notably achieves 42.9% on a dataset of homological conjectures where all baselines score 0%.

## Strengths

- **Novel synthesis capability:** Unlike prior auto-formalizers that rely purely on retrieving existing definitions, Aria's GoT architecture can synthesize new definitions for concepts not in Mathlib. The ablation study (Table 3) validates this quantitatively: on the Conjectures dataset, removing GoT drops success from 6/14 to 1/14. The case studies in Appendix A provide concrete examples (e.g., synthesizing `IsNil` for Koethe's Conjecture, `CatenaryRing` definitions) demonstrating this capability.

- **Rigorous semantic evaluation with term-level grounding:** AriaScorer improves over prior methods by retrieving actual Mathlib definitions for Lean terms rather than relying on textual similarity. Table 2 shows AriaScorer achieves 93.5% F1 versus 84.9% for the best baseline, with particularly notable improvements in catching definition discrepancies (e.g., QuaternionGroup 1 vs QuaternionGroup 2 in Appendix B.2-3).

- **Strong empirical performance on challenging benchmarks:** The 42.9% success rate on research-level conjectures where all baselines fail completely represents a qualitative advance. The paper also demonstrates generalization across domains (Table 6: ProofNet breakdown by algebra, analysis, number theory, topology), and Appendix E shows successful formalization of Borel's Conjecture in topology, suggesting domain generativity beyond the primary algebraic focus.

- **Well-designed ablation studies:** The component-wise ablation (Tables 3, 4) clearly quantifies the contribution of each architectural component. The finding that RAG removal causes complete collapse (42.9%→0%) on Conjectures but only modest impact on FATE-X provides insight into where retrieval versus reflection matters most.

## Weaknesses

- **Conjectures dataset has severe statistical limitations:** The headline "42.9% vs 0%" result is based on N=14 problems, where 42.9% corresponds to exactly 6 successes. This sample is too small to support reliable statistical claims, and all 14 problems are from a single narrow domain (homological conjectures in commutative algebra). Generalization claims about "research-level formalization" based on this dataset are overextended.

- **AriaScorer creates a circular evaluation concern:** AriaScorer is both a claimed contribution and the evaluation metric for all reported results (except Conjectures, which was manually verified). The scorer is validated on Aria's own outputs (Section 4.3.1), creating a distribution mismatch that may favor Aria's formalization patterns. While the paper claims 95.5% alignment with human experts (Section B.5), this validation set size and methodology are not detailed, and the scorer is not evaluated on baseline model outputs to demonstrate generalization.

- **Backbone LLM is never explicitly named:** The paper specifies that AriaScorer uses Gemini-2.5-Pro (Section 4.3.1) and compares against Gemini-2.5-Pro as a baseline, but never states which LLM powers Aria's decomposition, grounding, synthesis, and reflection modules. This omission makes it impossible to assess whether the comparison against "Gemini-2.5-Pro (pass@1)" is fair (i.e., whether Aria's scaffolding improves the same underlying model).

- **Threshold selection for AriaScorer is post-hoc and potentially overfit:** The threshold α=0.9 is selected after examining the precision/recall trade-off on FATE-X (Table 2), which is the same dataset used for downstream evaluation. No held-out validation set is used, and sensitivity analysis of results to threshold choice is not reported.

- **AriaScorer validation set is very small:** Table 2 implies only 69 total labeled examples (50 TP + 5 FP + 12 TN + 2 FN). The 89.9% accuracy at α=0 has a 95% confidence interval of roughly ±7 percentage points, yet no confidence intervals are reported.

- **LeanScorer baseline is a re-implementation:** The paper re-implements LeanScorer since "the original version is not open-source" (Section 4.3.1), making the comparison unverifiable. The reported F1 gap (93.5% vs 82.1%) could partially reflect implementation differences rather than genuine methodological advantage.

## Nice-to-Haves

- Expand the Conjectures dataset beyond N=14 to provide statistically meaningful evaluation of research-level formalization capability. A sample of 50+ problems across multiple mathematical domains would substantially strengthen the claims.

- Evaluate AriaScorer on outputs from baseline models (e.g., Goedel-V2, Gemini) to demonstrate that the scorer generalizes beyond Aria's specific formalization patterns and is not overfitting to the agent's error distribution.

- Report wall-clock inference time and token costs, not just API call counts. The comparison "17.7 calls vs pass@128" does not account for the computational overhead of GoT planning, RAG retrieval, and reflection iterations.

- Provide statistical significance testing (e.g., confidence intervals, Fisher's exact test) for key results, particularly the ablation studies where N=14 makes percentage differences noisy.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Fuzzy integral aggregation formula not provided":** While the paper mentions "fuzzy integral" aggregation for subtask scores, the lack of explicit formula is a minor detail that doesn't undermine the method. The contribution is the term-level grounding concept, not the specific aggregation function.

- **"Jixia reliability uncharacterized":** Jixia is an external static analyzer. While its reliability matters, demanding characterization of external tool coverage is scope creep beyond the paper's contributions.

- **"Cycle detection in dependency graph not specified":** The paper describes the dependency graph as acyclic. Mathematical definition dependencies are naturally acyclic in well-formed formalizations. This is a speculative correctness concern not observed in practice.

- **"Liu et al. 2025a not experimentally compared":** While this related work is cited, demanding comparison with every cited method is excessive. The baseline set (Herald, Kimina, Goedel-V2, Gemini) is reasonable for ICLR standards.

- **"Lu et al. 2025 baseline missing":** This is an August 2025 arXiv preprint. Per review guidelines, papers should not be criticized for missing very recent or unavailable baselines.

## Novel Insights

The paper reveals an important trade-off between "syntactic risk" and "semantic rigor" in formalization systems. The GoT ablation on FATE-H (Table 5) shows a counter-intuitive pattern: removing GoT *increases* compilation rate (89%→95%) but *decreases* final accuracy (71%→54%). The modular definition synthesis approach that GoT enables introduces more syntactic complexity (and thus more compilation failures) but produces semantically correct formalizations more often. This suggests that for complex mathematical reasoning, syntactic compilation metrics alone are misleading proxies for true formalization quality—a finding with implications beyond this specific system.

## Suggestions

- Explicitly state which backbone LLM powers Aria's agent components in the main text (not just in AriaScorer), and clarify whether comparisons against "Gemini-2.5-Pro (pass@1)" represent the same underlying model with different scaffolding.

- Add a held-out validation split for AriaScorer threshold selection, or report results across multiple threshold values to demonstrate robustness.

- Provide confidence intervals for all reported percentages, especially the Conjectures dataset where N=14 renders point estimates unreliable.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 4.0, 4.0, 8.0]
Average score: 5.3
Binary outcome: Accept
