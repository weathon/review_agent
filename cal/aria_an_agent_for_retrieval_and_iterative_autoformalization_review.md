=== CALIBRATION EXAMPLE 21 ===

# Final Consolidated Review
## Summary

Aria is an agent for auto-formalizing mathematical conjecture-level statements into Lean 4, combining a Graph-of-Thought (GoT) decomposition-and-synthesis pipeline with retrieval-augmented generation (RAG) for grounding in Mathlib and compiler-in-the-loop reflection. It also introduces AriaScorer, a semantic correctness checker that performs term-level grounding by retrieving authoritative Mathlib definitions, achieving state-of-the-art results on ProofNet, FATE-H, FATE-X, and a set of 14 homological conjectures where all baselines score 0%.

## Strengths

- **Effective orchestration for novel definition synthesis**: The GoT pipeline's ability to recursively decompose statements, ground known concepts via RAG, and synthesize new definitions bottom-up addresses a genuine limitation of prior single-pass formalizers. The ablation on the Conjectures dataset (42.9% → 7.1% without GoT; 42.9% → 0% without RAG) compellingly demonstrates that the architectural design is critical for handling concepts absent from Mathlib, rather than being an incremental combination of existing techniques.

- **Term-level semantic grounding in AriaScorer**: Moving beyond surface-level textual comparison by injecting retrieved Mathlib definitions into the evaluation prompt is a substantive methodological contribution. The case studies in Appendix B (e.g., detecting that `QuaternionGroup 1` is cyclic of order 4, not the quaternion group Q₈) concretely demonstrate failure modes of text-based checkers that AriaScorer addresses.

- **Clear empirical advantage on challenging benchmarks**: The 44.0% vs. 24.0% gap on FATE-X and the 42.9% vs. 0% on Conjectures represent meaningful progress on problems where current methods fail entirely. The comparison against Goedel-V2 at pass@128 (using 7x more calls) is informative and shows that the improvement is not simply a function of additional compute.

## Weaknesses

### Major:

- **The 14-conjecture dataset is too small to support the paper's strongest claims**. The abstract's claim of "breakthrough performance" on research-level mathematics rests on 6 successful formalizations out of 14. While the result is impressive, n=14 makes the "all other models score 0%" claim statistically fragile—a single baseline success would substantially change the narrative. The paper would be significantly stronger with a larger and more diverse conjecture set, or with appropriately hedged claims. This directly affects the paper's core novelty claim about conjecture-level capability.

- **No comparison with other agentic or multi-step planning approaches**. All baselines (Goedel-V2, Kimina, Herald, Gemini-2.5-Pro) are single-pass or multi-sample systems. The paper's key architectural contribution—an agentic GoT pipeline with reflection and RAG—has no methodologically similar comparator. Without this, it is impossible to disentangle the contribution of the specific GoT design from that of simply using an agentic loop with a strong base model. This is a significant gap for evaluating novelty.

- **AriaScorer validation has circularity risk and limited scale**. AriaScorer is developed alongside Aria and validated on Aria's own outputs (Section 4.3.1: "auto-formalized output of Aria on FATE-X"). While expert ground truth is constructed, the evaluation set contains only 69 examples, no inter-annotator agreement statistics are reported, and the checker's performance on outputs from other formalization systems is not evaluated. This means AriaScorer may be overfit to Aria's specific failure modes, directly impacting the reliability of "final accuracy" numbers reported throughout.

### Minor:

- **No systematic failure analysis**. With 56% of FATE-X attempts and 57% of Conjecture attempts failing final accuracy, the paper provides no categorization of failure types (decomposition errors, grounding errors, synthesis errors, semantic mismatches). Understanding *where* the system fails is as important as where it succeeds, and its absence makes it difficult to assess limitations or future improvement directions.

- **No evaluation of synthesized definition correctness beyond compilation**. A core claim is that Aria synthesizes definitions absent from Mathlib, but the paper evaluates these only by whether they compile and pass AriaScorer. Whether synthesized definitions are *mathematically* correct representations of the intended concepts—rather than merely syntactically valid approximations—is not assessed by independent expert review.

- **Missing statistical rigor**. No confidence intervals, standard deviations, or significance tests are reported for any results. This is particularly concerning for the small Conjectures dataset and the 69-example AriaScorer evaluation.

- **Computational cost analysis is incomplete**. The paper reports 17.7 API calls per problem but provides no wall-clock time, token consumption, or dollar cost. Without this, practical deployability cannot be assessed, and the comparison to Goedel-V2 pass@k remains incomplete (one cannot determine if Aria's 17.7 calls are cheaper than Goedel-V2's 128 parallel samples).

- **No analysis of dependency graph depth vs. performance**. The paper mentions average depth of 2–3 layers but does not examine whether success degrades for deeper graphs, which is critical for assessing scalability to more complex formalization targets.

### Trivial:

- The α=0 vs. α=0.9 threshold choice for AriaScorer could be better justified in the main text (currently deferred to a brief mention in Section 4.3.2), though this is a minor presentation issue.

## Nice-to-Haves

- Expand the Conjectures dataset beyond 14 examples from a single domain (commutative algebra) to include analysis, topology, and number theory conjectures, which would substantially strengthen the generality claim.
- Include human expert evaluation of the mathematical correctness of synthesized definitions, not just compilation + AriaScorer.
- Report wall-clock time and cost per problem, and plot accuracy vs. compute curves comparing Aria to multi-sample baselines.
- Evaluate AriaScorer on outputs from other formalization systems to demonstrate its generality as a checker beyond Aria's specific error profile.
- Add failure case dependency graph visualizations to complement the current success-case diagrams.

## Removed Points

These points are flagged to be removed; treat them with caution—they may still contain useful context but do not belong as criticisms in the main review.

- **"Wikipedia conjectures are not research-level"**: The harsh critic implies the conjectures are merely from Wikipedia and thus not genuine research problems. This misrepresents the paper—the conjectures are well-known open problems in commutative algebra compiled by Melvin Hochster, a leading researcher in the field. The Wikipedia page documents them but does not diminish their status as open mathematical conjectures. The paper's framing of these as "real-world mathematical conjectures proposed by mathematicians" is accurate.

- **"No discussion of neuro-symbolic equivalence checking alternatives"**: The critic claims the paper should discuss why existing neuro-symbolic methods (Liu et al., 2025a; Wu et al., 2025) couldn't be adapted. The paper does discuss these in Section 2 under "Semantic Check," and AriaScorer addresses different failure modes (definition discrepancy, implicit semantic inclusion) that purely equivalence-based methods cannot detect. This is not a missing comparison but a different approach.

- **"Contribution 3 is just an outcome"**: The critic argues listing state-of-the-art results as a contribution is inappropriate. It is standard and acceptable to list empirical achievements as contributions alongside methodological ones.

- **"Missing Graph-of-Thoughts citation (Yao et al.)"**: Per the rules, we do not flag missing related works, as we cannot confirm their existence or relevance without external sources.

- **"Reproducibility concerns about LeanSearch, jixia, Mathlib version-locking"**: These are standard engineering dependencies in this field; the paper cites them and they exist. Version-locking Mathlib is an ongoing community challenge, not a specific flaw of this paper.

- **"Kimina contamination—no discussion of other models' contamination"**: The paper already flags the most relevant contamination risk (Kimina on ProofNet). There is no evidence other baselines have similar issues.

- **"Termination criteria for dependency graph expansion"**: The paper describes the process clearly—expansion terminates when leaf nodes are grounded in Mathlib. If a concept cannot be grounded, it becomes an internal node for synthesis. This is explained in Section 3.1.1.

- **"Formatting and terminology inconsistencies"**: Pure style nitpicks; removed per rules.

- **"The hierarchical decomposition assumption may not hold for circular/mutually recursive definitions"**: The paper explicitly claims the system handles *acyclic* dependency graphs (Section 3.1.1), acknowledging this limitation. Circular dependencies are outside the stated scope.

- **"No theoretical bounds on AriaScorer"**: Requesting theoretical verification guarantees for an empirical semantic checker is scope creep beyond what is standard in this area.

## Novel Insights

The most insightful observation across the reviews is the "syntactic risk vs. semantic rigor" trade-off revealed by the FATE-H ablation (Table 5): removing GoT *increases* compilation rate (89% → 95%) but *decreases* final accuracy (71% → 54%). This suggests that the GoT planner's modular style—explicitly synthesizing intermediate definitions—introduces syntactic fragility (namespace conflicts, type class resolution failures) even as it improves semantic correctness. This is a genuine architectural insight: modularity helps semantics but hurts compilation, and the trade-off flips as problem difficulty increases. This finding deserves deeper discussion in the paper, as it has implications for the design of any agentic formalization system.

## Suggestions

- **Add a systematic failure taxonomy**: Categorize the 56% of FATE-X failures into decomposition errors, grounding failures, synthesis errors, and semantic mismatches. This would make the contribution much more actionable for future work.
- **Evaluate AriaScorer on non-Aria outputs**: Run AriaScorer on Goedel-V2 and Gemini outputs to demonstrate it is a general-purpose checker, not just tailored to Aria's error profile. This would significantly strengthen the AriaScorer contribution.
- **Report confidence intervals or bootstrap analysis**: Even simple Wilson score intervals for the small Conjectures dataset would substantiate the "breakthrough" claims.
- **Include at least one failed dependency graph visualization**: Show where the pipeline breaks down, not just where it succeeds, to help readers understand limitations.
- **Soften the abstract's "breakthrough" and "all other models score 0%" language**: These claims are accurate for n=14 but overstate the evidence. Consider "substantial improvement" and "no baseline achieved any success."

---

**Axis Evaluation:**

- **Novelty**: Moderate-to-high. The GoT + RAG + reflection orchestration for definition synthesis is genuinely novel, and AriaScorer's term-level grounding addresses a real gap. However, the individual components (RAG, reflection, graph-based decomposition) each have precedents, and the lack of comparison with other agentic approaches makes it hard to assess how much novelty comes from the architecture vs. simply applying an agent to this task.

- **Technical soundness**: Mostly sound, but weakened by the circular AriaScorer validation and the absence of statistical testing. The ablations are well-designed and the main claims are supported, but the reliability of "final accuracy" numbers depends on trusting AriaScorer, which is validated on Aria's own outputs.

- **Empirical support**: Strong on ProofNet and FATE benchmarks; limited on Conjectures due to small n. The ablation studies are thorough and informative. Key gap: no evaluation on other formalization systems' outputs for the checker, and no comparison with agentic baselines.

- **Significance**: High potential impact. If the results hold up at scale, the ability to formalize conjecture-level statements would be a meaningful advance for automated mathematics. The significance is somewhat limited by the small Conjectures dataset and the open question of how the system scales to domains beyond algebra.

- **Clarity**: Good. The paper is well-structured, the pipeline is clearly described, and the case studies in the appendices are detailed. The main paper could benefit from an integrated end-to-end example rather than delegating all case studies to the appendix.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 4.0, 4.0, 8.0]
Average score: 5.3
Binary outcome: Accept
