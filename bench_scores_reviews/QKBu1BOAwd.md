## Summary

DRAFT is a framework that iteratively refines tool documentation for LLMs through a three-phase self-driven loop: an Explorer probes the tool with diverse queries and captures execution results, an Analyzer identifies discrepancies between documentation and observed behavior and proposes targeted revision suggestions, and a Rewriter synthesizes these inputs to produce an updated documentation version plus directions for the next exploration round. The process is governed by a diversity-promoting exploration strategy (cosine similarity constraint + self-reflection) and a tool-adaptive termination mechanism (BLEU + embedding cosine similarity convergence). Experiments on ToolBench and RestBench with three LLMs (GPT-4o, GPT-4o-mini, Llama-3-70B) consistently outperform static documentation rewriting baselines, with secondary benefits demonstrated on tool retrieval.

---

## Strengths

- **Consistent, non-trivial gains across heterogeneous LLMs and datasets.** On ToolBench (I3-Instruction), CP% improvements over EasyTool reach 5–7 points across all three tested LLMs (Table 1), and on RestBench-TMDB the gains are even larger (e.g., GPT-4o: 79→88 CP%). The fact that gains transfer to GPT-4o-mini and Llama-3-70B using documentation refined with GPT-4o specifically demonstrates that the refined documentation captures model-agnostic usability improvements, not just idiosyncrasies of the backbone.

- **Cross-model generalization with a weaker backbone (Figure 7).** The paper tests Llama-3-70B as a refinement backbone—a weaker, open-source model—and shows that its refined documentation still benefits all three evaluation models on RestBench-TMDB. This is a practically important finding: organisations without GPT-4o access can still benefit from DRAFT.

- **Secondary retrieval benefit (Table 3).** Showing that DRAFT-refined documentation improves both sparse (BM25) and dense (Contriever) tool retrieval is a meaningful bonus, demonstrating that the improvements reflect genuine semantic enrichment rather than being a narrow inference-time artifact.

- **A notable cross-class result (Table 1, ToolBench).** GPT-4o-mini + DRAFT achieves 47 CP% on ToolBench, exceeding the raw GPT-4o at 37 CP%. This is a striking concrete demonstration of the practical value of documentation quality.

- **Human evaluation confirms improved completeness and accuracy.** Table 4 shows strong human preference for DRAFT documentation, especially on ToolBench (68% DRAFT vs 4% raw for completeness, 56% vs 0% for accuracy). The zero raw-preferred accuracy on ToolBench is notable.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing single-pass rewrite with execution traces baseline.** The paper's central thesis is that *iterative* trial-and-error is essential for documentation quality. However, the critical missing comparison is: give the backbone LLM the tool's original documentation plus a set of real execution traces (i.e., the same observations DRAFT generates), and ask it to rewrite the documentation in a single pass. Without this ablation, the paper cannot substantiate the claim that iteration—rather than simply having execution feedback at all—drives the improvement. EasyTool is a static rewrite without execution feedback, but it is not a single-shot rewrite *with* execution feedback. This gap directly undermines the necessity of the iterative framework.

- **No ablation removing the Analyzer module.** DRAFT's modular decomposition (Explorer → Analyzer → Rewriter) is presented as a contribution. However, there is no ablation comparing against simply feeding the Explorer's execution traces directly to a Rewriter without the intermediate Analyzer pass. Without this ablation, it is unclear whether the Analyzer's structured suggestion layer adds value over an end-to-end "observe and rewrite" prompt.

- **Algorithm 1 contains a pseudocode bug.** In Lines 15–18, when the convergence condition (Δ > τ) is satisfied at iteration *i*, the algorithm breaks *before* reaching Line 19, meaning t_i (the converged version) is never added to the output set D̃. The last version added would be t_{i-1}. Conversely, if no convergence is detected and all *I* rounds complete, D̃ accumulates t_1 through t_I—multiple intermediate versions per tool—rather than a single refined documentation. The intended semantics (use only the final version per tool) are not implemented as written, and the paper should clarify or correct this.

### Minor

- **Termination criterion is unvalidated.** The claim that BLEU + cosine similarity convergence "prevents overfitting" is not empirically confirmed. Figure 6 shows performance peaks and then declines, but the paper does not demonstrate that the adaptive termination mechanism actually halts at or near the peak iteration for each tool. A concrete analysis—e.g., showing the round at which the mechanism fires vs. the round of peak performance—would substantiate this claim.

- **Query diversity ≠ behavioral coverage.** The similarity constraint (Eq. 2) enforces semantic dissimilarity between generated *queries*, not between *API parameter configurations* or *response regimes*. Two semantically distinct queries may invoke identical tool behavior, while two semantically similar queries may exercise different parameter paths. For tools where important edge cases arise from parameter combinations, boundary values, or authorization conditions, the proposed diversity mechanism may leave meaningful behavioral regions unexplored. This is not acknowledged.

- **No cost or efficiency reporting.** DRAFT requires multiple LLM calls (Explorer + Analyzer + Rewriter) per iteration per tool, potentially up to five rounds. The paper provides no reporting on average API call counts, token usage, wall-clock time, or total refinement cost per tool. For a practically motivated system, this absence makes it impossible to assess the cost–benefit trade-off relative to one-shot methods like EasyTool.

- **Cross-model generalization evidence is narrower than claimed.** The abstract states "robust cross-model generalization capabilities." In reality, the main experiments use three models from two provider families (GPT-4o, GPT-4o-mini, Llama-3-70B), and the backbone generalization experiment (Figure 7) covers only one additional backbone (Llama-3-70B) on one dataset (RestBench-TMDB). The word "robust" is not warranted by this evidence. The explanation offered (shared transformer structure, pretraining corpora) is speculative and untested.

- **Retrieval results are mixed.** Table 3 shows gains on TMDB with both retrievers but Contriever @10 on Spotify slightly *decreases* (49.6→49.2). The paper interprets this section positively without acknowledging the mixed pattern; the claim should be softened to "can improve retrieval in most settings."

- **Ablation scope is limited.** Table 2 reports ablations for only one dataset (TMDB) and one model (GPT-4o) in the main paper. The performance drops for w/o diversity and w/o adaptive are modest (88→84 and 88→80 CP%), and without variance, it is unclear whether these are robust. The paper states appendix results show similar trends, which partially addresses this.

- **No limitations section.** The paper does not discuss when DRAFT may fail or should not be used: tools with rare dangerous failure modes, stateful or rate-limited APIs, tools that cannot be safely probed, or cases where exploration quality is poor because initial documentation is nearly empty. These are important practical caveats.

### Tiny

- **§2.5 contains unsubstantiated claims.** "Dynamically maintaining an accurate and up-to-date representation of evolving features" is listed as a strength (§2.5) but is never tested—no experiment with temporally evolving tools is presented. This should be framed as a potential benefit or future direction rather than an established strength.

- **High "Equal" rates in human evaluation are not discussed.** Table 4 shows Equal rates of 44–70% on several criteria for RestBench. While DRAFT is still clearly preferred, the substantial "Equal" proportion implies improvements are real but not dramatic in many cases; acknowledging this candidly would strengthen the paper's credibility.

- **The termination threshold τ and similarity threshold φ are not sensitivity-tested.** The paper uses τ=0.75 and φ=0.9 without showing how performance changes with alternative values, making it unclear how sensitive results are to these choices.

---

## Nice-to-Haves

- **Documentation length control.** Longer documentation can trivially provide more information. Reporting average length before and after refinement, and optionally including a length-matched baseline, would confirm that DRAFT improves information *density* rather than just quantity.

- **Iteration trajectory visualization.** A case study showing what concretely changes in the documentation across iterations—what is added, removed, or restructured—would make the improvement mechanism transparent and provide intuition beyond aggregate metrics.

- **Failure case analysis.** Win% is not 100%, meaning DRAFT occasionally produces worse documentation. Characterising when and why refinement degrades performance would increase trust in deployment.

- **Hyperparameter sensitivity study.** A sensitivity analysis for τ (e.g., 0.6, 0.75, 0.9) and φ (e.g., 0.8, 0.9, 0.95) would support reproducibility across different tool domains.

- **Testing a more distinct model family** (e.g., Mistral, Qwen) for cross-model generalization would substantially strengthen that finding.

---

## Removed Points

*These points are flagged for removal; treat with caution.*

- **"No statistical significance / confidence intervals"** (harsh critic, major): WEAKENED to minor. Single-run evaluation with expensive API-based LLM benchmarks is the norm in this community. The lack of variance reporting is a limitation, but it does not invalidate results where gains are consistently 5–10+ CP points across multiple datasets and models.

- **"Analyzer and Rewriter use the same LLM is a problem"** (harsh critic): REMOVED. Using the same backbone for all roles is standard practice in LLM agent papers. The paper need not justify this choice.

- **"text-embedding-ada-002 is dated / proprietary"** (harsh critic): REMOVED. Using an available, well-established embedding API is a reasonable engineering choice. Criticising the specific embedding model as a weakness is not substantive.

- **"No formal objective is defined"** (harsh critic): REMOVED. This is an empirical systems paper; formal objectives are not expected.

- **"The method reads as advocacy in §2.5"** (harsh critic): REMOVED as style nitpick; the substance (evolving tools not tested) is captured under Tiny weaknesses.

- **"Figure 1(c) is potentially confusing"** (harsh critic): REMOVED. The confusion appears to be a PDF-to-text rendering artifact, not a genuine paper flaw; the caption clearly states the figure highlights that DRAFT documentation is more favored.

- **"EasyTool comparison is unfair because DRAFT uses more compute"** (harsh critic): WEAKENED to Minor (subsumed into cost reporting weakness). The comparison itself is intentionally asymmetric in favour of EasyTool (simpler, cheaper); the authors' stronger claim is that even with a harder comparison, iterative feedback adds value. This is an acceptable scientific choice.

- **"The problem statement mixes multiple failure modes"** (harsh critic): REMOVED. Categorising documentation failure modes (incompleteness, redundancy, inaccuracy) is appropriate motivational framing, not a methodological flaw.

- **"DRAFT cannot guarantee correctness for unobserved tool behaviors / overconfident docs"** (harsh critic safety concern): REMOVED as outside the paper's stated contribution scope; not expected for this type of paper.

---

## Novel Insights

The most genuinely novel observation in this paper—partially surfaced by reviewers but underemphasised by the authors—is the **information-asymmetry gap between execution feedback and documentation text**. Human-authored documentation fails not because of bad writing, but because its authors cannot exhaustively probe tools in the way an automated framework can. DRAFT demonstrates that systematically inducing diverse tool executions and using the gap between observed outputs and documented behaviour as a revision signal is more effective than documentation rewriting guided purely by linguistic analysis. The secondary finding that documentation refined by a weaker open-source backbone (Llama-3-70B) still generalises to improve stronger models suggests that the informational content captured through tool interaction is largely model-agnostic—an insight with implications for building shared, model-independent tool interfaces. The key open question the paper leaves unaddressed is whether the **iterative** structure is necessary or whether a single, well-designed batch of diverse execution traces is sufficient for a one-shot rewrite to achieve equivalent gains.

---

## Suggestions

1. **Add a single-shot baseline with execution traces.** Run one round of DRAFT's exploration to collect N diverse execution examples, then ask the backbone to rewrite the documentation once using all examples simultaneously. Compare this to the iterative DRAFT. This is the most important missing experiment and should be in the main paper.

2. **Add an Analyzer ablation.** Compare DRAFT against a version where the Rewriter is given execution traces directly (no Analyzer intermediate step). This tests whether the structured suggestion layer contributes.

3. **Fix Algorithm 1.** Either move Line 19 (̃D update) so it fires before the convergence break (and only the final version per tool is retained), or explicitly specify that D̃ stores only the last accepted documentation per tool and clarify the pseudocode accordingly.

4. **Report average iteration counts and token/API cost per tool** in a table alongside performance, so readers can assess practical efficiency.

5. **Add a figure or table** showing the round at which adaptive termination fires per tool vs. the round of peak performance, to validate the termination mechanism's effectiveness.

6. **Add a limitations section** covering: poor initial documentation stalling exploration, stateful/rate-limited APIs, tools that cannot be safely probed, and cost at scale.

---

## Evaluation on Key Axes

**Originality:** Moderate. The application of iterative self-refinement to *tool documentation* via *actual tool execution feedback* is a genuinely novel angle; the individual components (self-refinement, similarity-based diversity, convergence detection) are not new. The integration is the contribution, and it is well-motivated.

**Importance of research question:** High. Documentation quality is a genuine bottleneck for LLM agents, and the problem is practically significant at scale.

**Claims well supported:** Partially. The claim that DRAFT improves tool-use performance is well supported. The claim that the *iterative* framework is necessary (vs. a single-shot rewrite with execution feedback) is not supported due to the missing baseline.

**Soundness of experiments:** Moderate. Multiple datasets, multiple models, and a secondary retrieval analysis are solid. However, the missing single-shot baseline, limited ablation scope, and absent variance reporting reduce confidence in the sufficiency of the experimental validation.

**Clarity of writing:** Good. The high-level idea and staged decomposition are clear. Method details (algorithm correctness, evaluation criteria for the Analyzer) are under-specified relative to ICLR standards.

**Value to the research community:** Moderate-to-good. A practical, plug-and-play documentation improvement pipeline that benefits both retrieval and execution is immediately deployable. The cross-model generalization finding enhances this value.

**Contextualization relative to prior work:** Adequate. The distinction from EasyTool (no execution feedback) is clear; the paper could more sharply position against broader self-improvement and active probing literature.