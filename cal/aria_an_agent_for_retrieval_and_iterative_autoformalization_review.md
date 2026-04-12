=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
## Summary
This paper presents **Aria**, an agent for Lean statement auto-formalization that combines a top-down **Graph-of-Thought dependency decomposition**, retrieval against Mathlib, bottom-up synthesis of missing definitions, and compiler-guided reflection. It also introduces **AriaScorer**, a semantic checker that augments LLM-based subtask evaluation with retrieved term definitions from Lean/Mathlib. Empirically, the system reports strong gains on ProofNet and FATE, and especially notable performance on a small conjecture set where the baselines fail.

## Strengths
- **The paper targets a genuinely hard setting that most auto-formalization work does not really solve: conjecture-level formalization requiring new intermediate definitions.** The central architectural move—explicitly decomposing a statement into conceptual dependencies, grounding what exists in Mathlib, and synthesizing the rest bottom-up—is specific and meaningful. This is not just “use RAG + reflection”; the graph-based dependency treatment is the key distinctive idea.
- **AriaScorer’s term-grounding is a substantive improvement over surface-form semantic judging.** The paper gives convincing examples where plain textual or lightly decomposed comparison would fail, but retrieved definitions reveal genuine semantic mismatches, e.g. the `QuaternionGroup 1` vs. `Q_8` case and the quaternion algebra interface mismatch. This is one of the more technically interesting parts of the paper.
- **The empirical gains are large enough to matter, especially on harder settings.** Table 1 reports 68.5% final accuracy on ProofNet, 71.0% on FATE-H, 44.0% on FATE-X, and 42.9% on the 14-conjecture set where all compared baselines score 0. Even allowing for some uncertainty in the semantic metric, these are not marginal improvements.
- **The ablations are unusually informative about what each module is doing.** Reflection is shown to drive compilation success; GoT matters more as problems become structurally harder; RAG is especially critical on the conjecture set. The pattern across Tables 3–5 supports the paper’s claim that different components address different failure modes rather than contributing redundantly.
- **The paper is transparent about the actual prompting structure and about limits of the semantic checker.** Appendix D exposes the prompts for decomposition, grounding, synthesis, and reflection, and Appendix B.4 explicitly discusses why AriaScorer is used as a terminal evaluator instead of an iterative controller. That level of methodological visibility is useful.

## Weaknesses

### Fatal
- **The main headline metric (“Final accuracy”) depends on the authors’ own LLM-based semantic checker rather than an independent evaluation on the main benchmarks, which substantially weakens the central comparative claim.**  
  This is the most important issue. Table 1 defines final accuracy as passing both compilation and **AriaScorer**; the paper states this explicitly in the introduction and experiment sections. While the paper does validate AriaScorer on a human-labeled subset of FATE-X (Table 2), it does **not** provide human verification for the main benchmark comparisons in Table 1 on ProofNet/FATE-H/FATE-X, and the conjecture set is only said to be “manually verified.” Because the system’s headline “surpasses previous methods” claim is mediated through a custom LLM judge, the empirical claim is not as definitive as the paper presents it. This does not make the experiments useless—the scorer validation is nontrivial—but it does mean the strongest comparative claim is not independently established.

### Major:
- **The validation of AriaScorer is narrower than the role the paper assigns to it.**  
  The checker is validated on **Aria’s own syntactically correct outputs on FATE-X** (“The evaluation used the Aria agent’s syntactically correct, auto-formalized outputs”), not on a broader and more adversarial mixture including baseline outputs or structurally diverse correct alternatives. That leaves an important uncertainty: whether AriaScorer remains equally reliable when judging outputs with different stylistic/structural biases than Aria’s own generations. Since this checker defines the paper’s main end-to-end metric, this selection of validation data matters.
- **The evidence for the paper’s strongest novelty claim—successful synthesis of genuinely new definitions—is still too anecdotal.**  
  The paper repeatedly emphasizes that Aria can “autonomously synthesize” missing concepts and definitions, and the conjecture case studies support that qualitatively. However, the experiments never quantify how often successful formalizations actually required synthesized definitions versus pure retrieval plus compiler-reflection. Without that breakdown, it is hard to tell how much of the gain comes from the proposed definition-synthesis capability as opposed to stronger grounding and iterative repair.
- **The human ground-truth construction for AriaScorer is credible but under-reported for a semantic-equivalence task of this difficulty.**  
  Section 4.3.1 states that labels were produced by one algebra PhD candidate with Mathlib experience and independently verified by a second expert. That is a reasonable start, but for a paper leaning heavily on semantic evaluation, the absence of agreement statistics, adjudication details, or discussion of ambiguous cases leaves the claimed 89.9%/93.5% performance of the checker less secure than it should be.
- **The planner’s failure modes are under-analyzed despite being central to the method.**  
  The methodology clearly relies on an LLM to recursively decompose concepts into prerequisite concepts before grounding. The paper argues that grounding handles hallucination after decomposition, but it does not really analyze decomposition failures themselves: wrong prerequisite choices, unnecessary expansion, graph explosion, or dead-end concepts. Appendix B.4 acknowledges semantic error propagation is rare in FATE-X, but there is no analogous diagnostic analysis for planning errors, even though the planner is one of the main claimed innovations.

### Minor
- **Cost/efficiency analysis is too limited for an agentic method.**  
  The paper does better than many submissions by reporting an average of 17.7 calls/problem on FATE-X and comparing against Goedel-V2 pass@k, but this is still incomplete. There is no token, latency, or monetary cost accounting, and API calls are only a rough proxy because calls can differ greatly in size and complexity.
- **Generalization beyond the reported domains remains only partially supported.**  
  The paper includes ProofNet subfield breakdowns and one topology case study, which is useful. Still, the most compelling results are concentrated in algebra-heavy settings plus a very small 14-example homological conjecture set. The broader claim of research-level generality would be stronger with a more diverse challenging benchmark.
- **The paper does not analyze threshold sensitivity of AriaScorer in the main evaluation deeply enough.**  
  Table 2 shows a meaningful precision/recall tradeoff between α=0 and α=0.9, and the main experiments use α=0.9. Given that final benchmark performance depends on this threshold, a more explicit sensitivity analysis would strengthen confidence in the robustness of the reported rankings.

### Trivial
- **The paper would benefit from a clearer accounting of where failures occur in the full pipeline.**  
  The current ablations are useful, but a direct error taxonomy over benchmark failures—decomposition failure, grounding miss, synthesis compile failure, semantically wrong but compilable, scorer rejection—would make the system easier to diagnose and improve.

## Nice-to-Haves
- Include a **blind human evaluation** on a representative subset of ProofNet/FATE-H/FATE-X to calibrate the Table 1 “final accuracy” numbers against independent judgment.
- Report the **fraction of successful cases requiring synthesized intermediate definitions**, ideally broken down by dataset and graph depth.
- Add a **failure-case appendix for the planner**, showing incorrect decompositions or retrieval dead ends, not only successful blueprints.
- Provide fuller **compute accounting**: total prompt tokens, average latency, and approximate cost per problem.
- Give a small **downstream proof-search analysis**, even preliminary, to test whether better statement formalization as judged by AriaScorer translates into improved proving success.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Claim that the paper lacks exact numbers of reflection attempts or other implementation details.**  
  This is factually wrong: Appendix C.1 explicitly says the full agent allows **16 reflection attempts**.
- **Criticism about exact release status / independent verifiability of LeanSearch, Jixia, or other cited tools.**  
  The paper cites these tools, so questioning their existence or availability should not be held against the paper.
- **Complaint that the paper should compare against many additional external baselines from related work.**  
  This is not a reliable criticism here because it depends on outside knowledge of what systems are actually comparable and available. The current baseline set is imperfect but nontrivial, and lack of additional named baselines should not be overstated.
- **Claim that GoT is contradicted by the FATE-H ablation because compilation success increases without it.**  
  This misstates the paper’s claim. The paper itself already explains that GoT can trade syntactic simplicity for semantic rigor on easier problems; the relevant metric is final accuracy, where GoT still helps substantially (71% vs. 54% on FATE-H).
- **Pure reproducibility nitpicks such as missing temperatures/top_p/complete environment specifications.**  
  These would improve replication, but they are not core scientific flaws by ICLR standards for an empirical systems paper of this type.
- **Requests to benchmark open-weight models specifically to prove the method is not driven by Gemini alone.**  
  This would be informative, but it is a nice-to-have rather than a substantive flaw in the current paper.

## Novel Insights
The most interesting synthesis across the reviews is that the paper’s strongest idea may actually be **AriaScorer**, not just Aria. The generation pipeline is promising and appears meaningfully stronger on difficult formalization problems, but the paper’s own evaluation setup makes the scorer part of the claim being tested. This creates an unusual situation: the paper’s main empirical story depends on the trustworthiness of a second contribution that is itself only partially validated. In other words, the work is potentially quite strong, but its two headline contributions are not independent—they are entangled. Untangling them with external human evaluation and a broader checker-validation set would likely sharpen the paper considerably.

## Suggestions
- Add **independent human evaluation** for a representative sample from each main benchmark, and report agreement with AriaScorer.
- Expand AriaScorer validation to include **baseline outputs and structurally diverse correct statements**, not only Aria outputs.
- Quantify **how often definition synthesis is actually necessary and successful**, with examples and failure rates.
- Provide a **pipeline failure taxonomy** and concrete planner-failure case studies.
- Add **compute/cost statistics** beyond API-call counts.
- Moderate the strongest novelty phrasing unless backed by stronger quantitative evidence, especially around being the “first” to autonomously synthesize complex new definitions.



# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0, 4.0, 4.0, 8.0]
Average score: 5.3
Binary outcome: Accept
