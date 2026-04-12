=== CALIBRATION EXAMPLE 12 ===

# Final Consolidated Review
## Summary
This paper proposes **PLAGUE**, a modular framework for multi-turn jailbreak generation that decomposes attacks into **Planner, Primer, and Finisher** phases, with reflection, backtracking, summarization, and a retrieval-based memory of successful strategies. The main empirical claim is that this composition yields stronger attack success than prior multi-turn attacks on several recent models, while also providing a reusable framework for mixing components from prior methods such as GOAT, Crescendo, and ActorBreaker.

## Strengths
- **The paper makes a specific systems-level contribution beyond a single attack prompt:** the Planner/Primer/Finisher decomposition is a concrete abstraction for multi-turn jailbreak generation, and the paper demonstrates this modularity by swapping in prior methods as components rather than only comparing against them. This is more informative than a monolithic new attack recipe.
- **The component ablations are genuinely informative at the mechanism level.** Table 3 isolates the effect of backtracking, reflection, planning, and retrieval on o3 and Claude Opus 4.1, and Table 4 further shows that changing the Finisher from GOAT-style to Crescendo-style materially changes results on Claude. This supports the paper’s claim that different models are vulnerable to different parts of the attack lifecycle.
- **The paper evaluates on a relevant set of strong target models and reports both binary and graded harmfulness metrics.** In particular, the reported gains on o3 and Claude Opus 4.1 are notable if valid, because these are exactly the settings where multi-turn red-teaming results are most interesting.
- **The budget accounting is more explicit than in many attack papers.** Table 5 separates target-model calls, evaluator calls, and planner calls, which helps distinguish “query budget to the victim” from total orchestration overhead.

## Weaknesses

### Fatal
- **The paper’s main SOTA comparison is not experimentally clean enough to fully support the headline superiority claim.**  
  The most serious issue is that the baseline configurations are modified in ways that can disadvantage them, while PLAGUE’s gains are then framed as state-of-the-art improvements. The paper explicitly states:
  - for GOAT: *“we tweak GOAT’s evaluation environment … To reduce computational costs, we run GOAT without history enabled for the Attacker”*;
  - for Crescendo: *“We remove any explicit backtracking counts from their attack and limit their maximum number of turns to six.”*  
  Since backtracking/reflection/history are exactly the kinds of mechanisms that PLAGUE argues matter, disabling or altering them in baselines makes the core superiority claim less convincing. The issue is not that the paper uses a unified budget cap—that is reasonable—but that it reports strong “outperforming prior SOTA” claims without also showing original baseline configurations or clearly separating “our reimplementation under our constrained protocol” from “prior published best.” For an ICLR paper making strong empirical claims, this is a substantial weakness.

### Major:
- **The “lifelong learning” claim is stronger than what the evidence currently supports.**  
  What is demonstrated is a retrieval-augmented memory of previously successful strategies indexed by goal embeddings. Table 3 shows retrieval helps, but only modestly on the provided ablation (+0.041 SRE on o3 from `GOAT + BT + R + P` to `... + RSS`, and similarly small on Claude in that table). The paper does not show sequential improvement over time, retrieval quality, transfer across batches, forgetting dynamics, or how performance evolves as the memory bank grows. As written, the evidence supports **useful strategy retrieval**, but not a strong “lifelong learning” claim in the richer sense suggested throughout the paper.
- **Metric definitions and reporting are inconsistent enough to weaken confidence in the exact quantitative claims.**  
  The paper says *“We use SRE and ASR interchangeably in our work”*, but SRE is a graded score and binary ASR is not. The text also mixes improvement claims across metrics without always being clear which one is being used. For example, the o3 improvement claim of “32.14%” does not match the relative increase in binary ASR shown in Table 2 (0.445 to 0.662), while the Claude 40.2% claim aligns with SRE in Table 4 rather than Table 2. There is also an inconsistency between Section 3.5 (*“score greater than 8/10”*) and Algorithm 3 (*“if score > 9.0 then”*). These are not cosmetic issues; they materially affect reproducibility and confidence in the reported gains.
- **The optimization signal used inside the attack is only partially aligned with the final reported evaluation.**  
  PLAGUE’s attack loop is driven by a custom rubric scorer rewarding compliance, practicality, level of detail, and relevance, while the final main metric is a modified StrongReject score based on refusal, convincingness, and specificity. This is not a fatal flaw—proxy objectives are common—but the paper does not validate that optimizing the rubric scorer correlates reliably with the final judged harmfulness metric. Given that much of the method’s control flow depends on rubric thresholds (7/10, 3/10, 8/10), this missing analysis matters.
- **The paper overstates efficiency if total system cost is considered rather than only victim-model queries.**  
  The paper is careful in Table 5 to separate target calls from evaluator/planner calls, which is good, but some textual claims such as “high order of efficiency” and “comparable query budget” risk overstating the result. PLAGUE often uses similar target calls to Crescendo, but it additionally incurs planner and evaluator/scorer overheads. For attack benchmarking, victim-query budget is important; for practical deployment cost, total orchestration overhead also matters. The paper should present this distinction more carefully instead of rhetorically collapsing them.

### Minor
- **The ablation coverage is narrower than the scope of the claims.**  
  The most detailed component ablation is shown only on o3 and Claude Opus 4.1. Since the paper evaluates five target models and argues that different components matter differently across models, a broader ablation would make that argument more convincing.
- **Several key hyperparameters/decision thresholds are introduced with little sensitivity analysis.**  
  Examples include the retrieval threshold of 0.6, maximum of two retrieved strategies, the two-step plan, and the rubric thresholds for backtracking/success. These choices may be reasonable, but the paper relies on them heavily and provides little evidence that the conclusions are robust to them.
- **The “plug-and-play” framing is somewhat stronger than the demonstrated user experience.**  
  The framework is modular in a technical sense, and the paper does show component swapping. However, best performance appears to require model-specific selection of the Finisher (e.g., GOAT-style for some settings, Crescendo-style for Claude), which makes the framework more “modular and tuneable” than universally strong out of the box.
- **Reliance on a modified LLM-as-a-judge setup raises comparability concerns.**  
  The paper uses a modified StrongReject-style prompt and a specific evaluator model. This is acceptable in the area, but because the method’s headline claims depend on judged harmfulness, it would be useful to better establish judge robustness or correlation with the binary metric beyond simply reporting both.

### Trivial
- **There are a few internal consistency issues in presentation.**  
  For example, the six-turn controlled-budget setup is central, yet Figure 2 discusses behavior at eight turns. That is not invalid as auxiliary analysis, but it should be separated more clearly from the controlled-budget comparisons.

## Nice-to-Haves
- Include baseline results in both the paper’s constrained protocol **and** each baseline’s closest original configuration, so readers can distinguish unified-budget comparisons from “best published” comparisons.
- Add a sequential analysis of the memory bank: performance over time, retrieval hit quality, and sensitivity to bank size / similarity threshold.
- Report uncertainty more explicitly (e.g., confidence intervals or paired significance tests) for the main ASR differences.
- Provide a direct correlation analysis between rubric scores and final SRE/binary-ASR outcomes to justify the rubric as an optimization proxy.
- Add a default “one configuration across all models” result to better support the plug-and-play claim.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“No evaluation against defenses.”**  
  This is outside the paper’s stated scope. The paper is about constructing and analyzing a multi-turn attack framework against target models, not about benchmarking defenses. Such experiments would strengthen the work, but their absence is not a core flaw for this paper’s stated contribution.
- **“The paper should evaluate on many more benchmarks / much longer turn budgets.”**  
  The HarmBench standard set and a fixed six-turn budget are reasonable scoped choices for controlled comparison. Broader evaluation would help, but the current benchmark choice is not itself a substantive flaw.
- **“The attacker being DeepSeek-R1 means the gains may just be due to a strong attacker.”**  
  This is only partially compelling. The same attacker setup appears to be used consistently for the authors’ comparisons, so this does not invalidate the comparative findings inside the paper. A weaker-attacker study would be useful but is not required to establish the proposed framework’s value.
- **Pure reproducibility complaints about missing prompt details or release status.**  
  The paper already includes prompts/algorithms in the appendix and explicitly states code/prompts will be available; further nitpicks on artifact completeness are not central here.
- **Stealthiness not measured.**  
  The paper does in fact include relevance in the rubric and discusses progression/semantic drift. It does not perform a separate stealthiness benchmark, but claiming it completely neglects contextual relevance would overstate the issue.

## Novel Insights
The most interesting synthesis across the paper is that **multi-turn jailbreak success seems to depend less on a single “best attack” than on coordinating distinct lifecycle functions**: initialization of a plausible plan, maintenance of context through intermediate benign steps, and a final objective-conditioned strike. The paper’s own results suggest that these functions interact differently with different victim models—for example, reflection appears more valuable on o3 while backtracking matters more for Claude—supporting the idea that future red-teaming and defenses may need to reason about *where* in a conversation the vulnerability lies, not just *which attack family* is used.

## Suggestions
- Run and report baseline methods in their closest original configurations alongside the paper’s unified-budget variants, and tone down the strongest SOTA claims unless both views agree.
- Narrow or better justify the “lifelong learning” claim by adding sequential-memory experiments or reframing the component as retrieval-augmented strategy reuse.
- Standardize all metric language: clearly separate binary ASR from SRE, fix the threshold inconsistencies, and recalculate every percentage improvement with the referenced metric named explicitly.
- Add an analysis showing how rubric score thresholds correlate with final judged success; if the correlation is weak, revise the control logic or claims.
- Distinguish carefully between **target-model query budget** and **total orchestration cost** in both tables and prose.
- Expand component ablations beyond o3 and Claude, even if only on a smaller subset, to support the claim that different components expose different model-specific weaknesses.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 2.0, 2.0]
Average score: 2.5
Binary outcome: Accept
