Now let me search the human reviews for calibration.Now I have enough information to write the consolidated review. Let me compile it.

---

## Summary

DRAFT is a framework for iteratively refining tool documentation to help LLMs use external tools more effectively. It operates through three phases: an Explorer that simulates diverse tool usage scenarios, an Analyzer that diagnoses documentation problems from execution feedback, and a Rewriter that updates the documentation. Two supporting mechanisms — a diversity-promoting exploration strategy and a tool-adaptive termination mechanism — prevent redundancy and overfitting. Experiments on ToolBench (I3-Instruction) and RestBench (TMDB and Spotify) show consistent improvement over static baselines, with additional cross-model generalization, tool retrieval, and human evaluation analyses.

---

## Strengths

- **Well-motivated and concretely diagnosed problem.** The paper identifies three distinct, real failure modes in tool documentation (incomplete, redundant, inaccurate) with concrete API examples in Figure 1(a). This is more specific and actionable than prior framings.

- **Modular, interpretable architecture.** The Explorer/Analyzer/Rewriter decomposition assigns clear responsibilities to each component. The natural-language feedback chain makes the refinement process inspectable, and Figure 2 / Appendix E provide a concrete revision trajectory.

- **Broader evaluation than comparable work.** The paper goes beyond end-to-end success rate to include: tool retrieval evaluation (Table 3, BM25 and Contriever on two datasets), cross-model generalization (Figure 7, using Llama-3-70B as backbone and evaluating on all three LLMs), and human evaluation (Table 4, completeness / conciseness / accuracy). This breadth is a genuine strength compared to papers like PLAY2PROMPT which evaluated on a single dataset.

- **Cross-model generalization is genuinely tested.** Refining with Llama-3-70B as backbone and then applying documentation to GPT-4o (and vice versa) is the most important property for a documentation refinement system to have, and the paper confirms it holds.

- **Clear performance gains over EasyTool on two datasets.** On RestBench-TMDB (GPT-4o), DRAFT achieves 88% CP% vs EasyTool's 79%; on ToolBench (GPT-4o), 51% vs 45%. These margins exceed 6+ percentage points and are consistent across all three LLMs tested.

---

## Weaknesses

### Fatal
None.

### Major

- **EasyTool missing from Spotify (one-third of evaluation).** Table 1 marks EasyTool as "-" on RestBench-Spotify with the note "EasyTool has not been implemented on the Spotify dataset." No technical justification for this absence is provided. Spotify is a complete evaluation split, and EasyTool is the most directly comparable baseline (static documentation rewriting vs. DRAFT's iterative version). On the two splits where comparison exists, DRAFT's margins are real but not overwhelming (9 pp on TMDB, 6 pp on ToolBench). The absence of this comparison on Spotify prevents a full head-to-head evaluation and leaves DRAFT's claimed superiority over EasyTool partially unsubstantiated.

- **No statistical significance testing on small test sets.** RestBench-TMDB has approximately 50 test instances (CP% values are multiples of 2, consistent with N=50). At this scale, a 4-point difference in CP% corresponds to ~2 samples. The ablation in Table 2 (88% vs. 84% vs. 80%) and the tool retrieval improvements in Table 3 (differences of 2-5 NDCG points) all fall within noise ranges that should be characterized with confidence intervals or bootstrap significance testing. The paper reports no variance across runs, no significance tests, and no seed analysis. Given the stochastic nature of LLM generation, this is a substantive methodological gap — particularly for the ablation results which form the primary evidence for the two proposed mechanisms.

### Minor

- **Table 3 tool retrieval shows mixed results on Spotify that are underdiscussed.** The paper claims "the modifications not only improve the readability of the documentation but also enhance its semantic quality." However, BM25 on Spotify shows no improvement at @1 (43.9% → 43.9%), and Contriever @10 on Spotify *decreases* (49.6% → 49.2%). These results partially contradict the stated narrative and should be acknowledged rather than glossed over with a uniformly positive framing.

- **Win% metric does not enable direct DRAFT-vs-EasyTool comparison.** As stated in the paper, Win% is computed relative to ReAct as a common reference point. A DRAFT Win% of 71% and an EasyTool Win% of 62% on TMDB (GPT-4o) both measure wins *against ReAct*, not against each other. The CP% columns do enable direct comparison, but the paper's discussion conflates the two metrics at points. A direct pairwise DRAFT-vs-EasyTool evaluation using the same ChatGPT judge would be more informative.

- **Computational cost is absent.** DRAFT requires multiple GPT-4o calls per tool across up to 5 iterations (Explorer + real API invocation + Analyzer + Rewriter per round). EasyTool uses a single ChatGPT call per tool. The preprocessing cost difference is potentially substantial and should be quantified to let readers assess the cost-performance tradeoff. This is especially relevant for large-scale tool repositories.

- **Human evaluation methodology is underpowered.** Three annotators ("well-educated doctor students") evaluated 50 cases with no inter-annotator agreement reported, no blinding protocol described, and no analysis of edge cases or failure modes. With only three raters, ICC or Cohen's κ cannot be meaningfully computed. The results (Table 4) are suggestive but should be presented with appropriate caveats.

### Trivial

- **Algorithm 1 termination condition wording is ambiguous.** Line 16 reads "if Δ > τ then Break." Since Δ (Eq. 5) is an average of cosine *similarity* and BLEU, Δ is a *similarity* measure — stopping when Δ is high means stopping when the two consecutive versions are nearly identical (i.e., minimal change). This is correct but the text description "when there is minimal change" could be read as describing a dissimilarity metric. Explicitly noting that Δ is a similarity metric and Break happens when similarity *exceeds* τ (meaning the documents are already very alike) would eliminate ambiguity.

- **ToolBench I3-Instruction subset size not stated.** The paper notes "Due to budget constraints, we focus on the most challenging subset of ToolBench, namely I3-Instruction" but never reports the number of test instances used. This is a minor reproducibility gap.

---

## Nice-to-Haves

- **Characterize what edits DRAFT actually makes.** The paper demonstrates that CP% improves but never analyzes which types of documentation edits drive the gains (e.g., adding parameter constraints, fixing return value descriptions, removing redundancy). An edit-type breakdown would give actionable insight into which documentation flaws matter most.

- **Cost-benefit analysis.** A comparison of the number of API calls and tokens consumed per tool in DRAFT vs. EasyTool would help practitioners decide when DRAFT's additional cost is worthwhile.

- **Sensitivity analysis on φ (similarity threshold) and τ (termination threshold).** These are the two critical hyperparameters governing the core mechanisms. Table 2 ablates presence/absence but not threshold values. A brief sensitivity sweep would strengthen trust in the reported settings.

- **Failure case analysis.** The appendix shows a successful revision trajectory; showing a case where DRAFT degrades documentation quality would reveal the method's failure modes and boundary conditions.

- **Evaluate on "dynamic tools" use case.** The introduction explicitly motivates DRAFT against tools with frequently updated documentation. No experiment directly tests whether DRAFT can detect and correct documentation that has drifted from actual tool behavior—the most distinctive use case in the stated motivation.

- **Deeper analysis of GPT-4o-mini surpassing GPT-4o.** The paper notes that GPT-4o-mini with DRAFT outperforms GPT-4o without DRAFT on ToolBench (47% vs. 37%), a noteworthy finding about the relative value of documentation quality vs. model capacity. This deserves more than a passing mention.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Win% table in Figure 1(c) reads backwards (parser artifact).** The harsh reviewer noted that Figure 1(c) shows "Raw Documentation" at 92.5%/77.5%/87.9% Win, appearing to favor raw documentation. This is almost certainly a PDF-parsing artifact (the bars likely show DRAFT winning those percentages against raw documentation). This is not a paper problem.

- **Cross-model generalization explanation is "hand-wavy."** The paper's explanation that decoder-only models share similar transformer structures and pre-training corpora is brief but plausible for an empirical paper. Demanding a deeper mechanistic analysis is scope creep for a systems-oriented contribution.

- **Circular optimization / self-bias from using GPT-4o as backbone.** The neutral reviewer raises this concern, but (a) cross-model generalization experiments directly address it — documentation refined with GPT-4o improves Llama-3-70B and vice versa; (b) the paper explicitly tests whether this holds. The concern is addressed, if not in depth.

- **CP% metric does not verify argument correctness.** This is a valid observation but applies equally to all baselines evaluated with the same metric; it is a limitation of the benchmark, not of DRAFT's design. The paper follows standard evaluation practice in this field (same as ToolBench's original evaluation).

---

## Novel Insights

The most genuinely novel observation that emerges is the cross-model generalization finding: iterative documentation refinement driven by one model's execution feedback produces documentation that improves *all* tested models. This suggests that LLMs as a class have shared comprehension needs around tool documentation that transcend model-specific idiosyncrasies — a finding with implications for how tool libraries should be maintained. The paper gestures at an explanation (shared transformer architecture and pretraining corpora) but does not characterize which documentation elements are universally beneficial. Identifying the universal vs. model-specific components of documentation quality is a promising direction.

---

## Suggestions

1. Implement and report EasyTool on RestBench-Spotify to complete the primary comparison table.
2. Report bootstrap confidence intervals or at minimum multi-run variance for Table 1, Table 2, and Table 3.
3. Add a direct pairwise DRAFT-vs-EasyTool Win% evaluation, clearly distinguished from the ReAct-relative Win%.
4. Report per-tool API cost statistics for DRAFT vs. EasyTool (number of API calls, token count, wall-clock time).
5. Add a sentence explicitly defining Δ as a similarity metric in Section 2.4 to resolve algorithmic ambiguity.
6. Acknowledge the mixed Contriever @10 result on Spotify in the tool retrieval discussion.
7. Report the number of test instances used from ToolBench I3-Instruction.

---

## Score and Decision

**Calibration:**

- **PLAY2PROMPT (EBaMTeWi2K)** — Rejected, scores 3/5/5/3/5 (avg ~4.2). Most directly comparable: same problem (iterative tool documentation refinement, LLM exploration, feedback loop). Weaker than DRAFT: single dataset (StableToolBench), no cross-model generalization study, no retrieval evaluation, no human evaluation.

- **ToolEVO (wtrDLMFU9v)** — Accepted (Poster), scores 6/3/6/1. Related: tool learning in dynamic environments. Different in contribution (MCTS, new benchmark). Accepted despite a 1/10 outlier and documented presentation concerns.

- **MTU-Bench (6guG2OlXsr)** — Accepted (Poster), scores 6/5/6/6 (avg 5.75). Benchmark paper, broader in coverage but different contribution type.

- **PromptWizard (VZC9aJoI6a)** — Rejected, scores 5/3/3/5/6 (avg ~4.4). Iterative prompt optimization; rejected partly for missing baselines and weak novelty claims.

**Assessment:** DRAFT is meaningfully stronger than PLAY2PROMPT (the closest comparator) in evaluation breadth, analysis depth, and number of datasets. Its core findings are supported on 2 of 3 dataset comparisons with EasyTool, and the cross-model generalization result is genuine. However, the missing EasyTool baseline on Spotify (one-third of evaluation), small test sets without significance reporting, and the underdiscussed mixed retrieval results are real weaknesses. The paper sits above PLAY2PROMPT but below a strong poster like MTU-Bench. Anchored against the calibration papers, a score of **5.0** (borderline/weak reject → weak accept) is appropriate: the contribution is real and the evaluation is reasonably thorough for the community's standards, but the experimental design has addressable but non-trivial gaps that prevent confident acceptance.

**Evaluation axes:**
- *Originality*: Moderate — applying iterative feedback-based refinement to tool documentation is novel in emphasis but extends prior ideas (APE, DSPy, self-reflection). The specific problem decomposition is clean.
- *Importance of research question*: High — tool documentation quality is a genuine bottleneck for real-world LLM deployment.
- *Claims well-supported*: Partially — main claims hold on two of three datasets, but the third is incomplete.
- *Soundness of experiments*: Adequate but flawed — multi-LLM and multi-metric evaluation is commendable, but small test sets and missing significance testing and missing baselines weaken confidence.
- *Clarity of writing*: Good — methodology is clearly described; some minor ambiguities around the termination condition.
- *Value to community*: Moderate-high — the retrieval and cross-model generalization analyses extend the contribution beyond what similar papers have done.

**Decision: Borderline Reject** — the core contribution is meaningful but the incomplete EasyTool comparison on Spotify and lack of statistical rigor prevent confident acceptance. These are addressable issues; the paper has a clear path to acceptance.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>