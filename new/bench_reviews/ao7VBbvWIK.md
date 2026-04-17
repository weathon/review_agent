Now I have enough calibration data. Let me synthesize my final review.

## Summary

HASTE proposes a hybrid pipeline for code context retrieval that combines AST-guided structural chunking, hybrid information retrieval (BM25 + semantic search with Reciprocal Rank Fusion), call-graph expansion, and token-budget-constrained extraction to provide LLMs with structurally coherent yet semantically relevant code context for automated editing tasks.

## Strengths

- **Well-motivated problem formulation.** The paper identifies a real and important tension between structure-aware and relevance-focused context retrieval for LLM-based code editing. The framing is clear and the dichotomy is practically meaningful.

- **Thoughtful, modular pipeline architecture.** The HASTE pipeline is systematically designed: Scanner → AST-aware Chunker → Identifier Extraction → Payload Builder → Embedding/Indexing → Hybrid Retrieval with RRF → Call-graph expansion → Token-bounded export. Each module is clearly described and the overall design is coherent.

- **Qualitative example illustrating structural expansion's value.** The type-hint example in §5.1 (where graph expansion correctly includes a dependent class definition) is a compelling illustration of why structural context matters for downstream edit quality.

- **Honest discussion of failure modes.** §5.3 transparently describes instances where HASTE's context was insufficient or where the underlying suggestion was flawed, adding credibility to the evaluation.

## Weaknesses

### Major:

- **Evaluation scale is far too small to support core claims.** The primary quantitative evidence for RQ1/RQ2 consists of six hand-picked Python files, each with a single synthetic edit task (Table 1). The SWE-PolyBench evaluation (§5.3) lacks aggregate statistics—no mean/median scores, no per-category breakdown, and crucially, excludes instances with "processing errors" without quantifying how many were dropped or why. The paper's abstract claims HASTE "resolves" the relevance-structure trade-off and represents a "key step towards enabling reliable and scalable AI-assisted software development," but six data points plus an anecdotal SWE-PolyBench analysis cannot substantiate claims of generality or scalability.

- **Baseline comparison is fundamentally incomplete.** The central claim is that HASTE resolves the trade-off between structure-aware and relevance-focused methods, yet: (1) No results for the three defined baselines (IR-only, AST-only, naïve truncation) are reported in §5—only HASTE's scores appear in the table. Without side-by-side numbers, readers cannot assess whether HASTE actually outperforms either side of the trade-off. (2) The "IR-only" baseline operates on AST-chunked units (§3.1), meaning it already has structural awareness—it is not a pure "structure-agnostic" retrieval baseline. The token-level pruning methods critiqued in §2.2 are never empirically compared. (3) No ablation study isolates the contribution of hybrid retrieval, call-graph expansion, or AST-bounded pruning. Without this, the claim of "synergistic integration" (Abstract) is unsupported.

- **Hallucination Rate and AST Fidelity metrics are defined but never reported.** These metrics (§4.2.2–4.2.3) are central to the paper's narrative—reducing hallucinations and maintaining structural fidelity are core selling points. Yet §5 contains zero numerical values for either metric. The hallucination-related claims in the Abstract and §2.4 ("confirmed empirically") are not backed by any quantitative evidence.

- **LLM-as-Judge protocol is opaque and uncalibrated.** The primary metric is an LLM-as-judge score (0–100), but the prompt template, scoring rubric, and weighting across correctness/readability/instruction-alignment dimensions are not provided. No human validation is reported. Three runs per task are averaged, yet no standard deviations or confidence intervals are shown. For a paper whose core evidence rests on these scores, this is a significant gap.

### Minor:

- **Pearson's r = -0.97 on six points is over-interpreted.** The paper treats this as a "strong negative correlation" revealing a meaningful trade-off frontier, but with n=6, this correlation is extremely unstable and should be presented as a tentative observation, not a substantive finding.

- **Only Python is evaluated, and edit tasks are narrowly scoped.** All six curated tasks are simple, localized edits (adding type annotations, adding try-except blocks). This limits confidence that HASTE handles more complex refactoring or cross-file edits.

- **SWE-PolyBench exclusions introduce potential selection bias.** The paper drops instances with "processing errors" without reporting how many or analyzing failure patterns, which could mask HASTE's limitations on harder cases.

- **"Token-bounded Extraction" lacks algorithmic specificity.** The title promises token-bounded extraction, but the paper does not detail the algorithm for resolving conflicts when expanded context exceeds the budget—which AST subtrees are pruned first, and how structural integrity is maintained under aggressive compression.

## Nice-to-Haves

- Execution-based evaluation (e.g., pass@k or resolve rate on SWE-PolyBench test suites) to validate that syntactically correct edits are also functionally correct.
- Testing on at least one additional LLM (beyond Gemini 1.5 Flash) to assess generalizability.
- Latency/overhead analysis of the full pipeline, which has multiple stages (parsing, embedding, indexing, retrieval, call-graph traversal).
- Reporting standard deviations across the 3 runs per task.
- Multi-language evaluation, since Tree-sitter support is claimed but never demonstrated.

## Removed Points

- **Claim that SpeechPrune and related token-level pruning methods should be directly compared as baselines.** While the harsh critic flags the absence of token-level pruning baselines, the paper explicitly positions HASTE against structure-*aware* code methods and RAG for code—token-level pruning is critiqued conceptually in §2.2 as inapplicable to code due to structural breakage, and the paper's IR-only baseline already serves as a relevance-focused comparator. The absence of a token-level pruning baseline is a fair methodological concern but not a "straw-man" issue; however, it is somewhat offset by the fact that IR-only is not a pure structure-agnostic baseline either.

- **Demand for comparison with RepoCoder or similar published systems.** This asks for baselines outside the paper's scope. HASTE is a context-retrieval pipeline, not a full code-editing system; comparing against end-to-end systems with different goals would be an apples-to-oranges comparison. The appropriate baselines are the ones the paper defines—except they need to actually report results for them.

- **Formatting/style nitpicks (e.g., Tree-sitter CST vs. AST distinction).** The AST-T5 reviewer flagged a similar concern about Tree-sitter generating CSTs rather than ASTs. While technically accurate, this is a common usage in the SE literature and does not affect HASTE's functionality.

- **Reproducibility concerns about code not yet released.** The paper states code will be released upon acceptance, which is standard for double-anonymous review. Flagging this is inappropriate per review rules.

## Novel Insights

The paper's most insightful observation is that the compression-quality frontier for code context retrieval is not monotonically worse with more compression—rather, the key is *what* is compressed and *how*. HASTE's core idea of using AST boundaries as the unit of compression (rather than arbitrary token spans) to maintain structural coherence while achieving high compression ratios is a genuinely useful engineering insight. However, the current evaluation is too thin to empirically validate this insight at scale.

## Suggestions

1. **Report baseline results side-by-side with HASTE.** The most critical gap is the absence of any comparative numbers for IR-only, AST-only, and naïve truncation. Without this, the evaluation cannot support claims of improvement over alternatives.

2. **Report Hallucination Rate and AST Fidelity numbers.** These are the paper's own metrics and their absence undermines core claims. If they were not computed, acknowledge this and soften claims accordingly.

3. **Expand the curated evaluation to at least 30-50 tasks across multiple languages and edit types**, or present the current results as a pilot study with appropriately scoped claims.

4. **Provide the LLM-as-Judge prompt and rubric.** Either in an appendix or supplementary material, to enable reproducibility and assessment of the evaluation's reliability.

5. **Add an ablation study** removing hybrid IR, call-graph expansion, and AST-bounded pruning separately, to demonstrate which components drive the observed performance.

## Score and Decision

**Calibration comparison:**

- **PKG (EHfn5fbFHw)** — Context-augmented code generation with knowledge graphs, limited evaluation, limited baselines → Reject (scores 5/5/3/8, avg ~5)
- **RCC (GYk0thSY1M)** — Context compression for LLMs, incomplete baselines, small model experiments → Reject (scores 3/3/5/5, avg ~4)
- **FRAPPE (MjR5LcAGXJ)** — Prompt compression, empirical but straightforward, some missing ablations → Reject (scores 5/3/3/5/3, avg ~3.8)
- **AST-T5 (TS8PXBN6B6)** — Structure-aware code pretraining, reasonable evaluation, some gaps → Reject (scores 5/6/6, avg ~5.7)
- **Codev-Bench (c2C2NQKjZw)** — Benchmark paper, small dataset concerns → Reject (scores 3/5/6/3, avg ~4.25)

HASTE shares the same pattern as PKG: promising system design for code context, but evaluation that is far too limited to support strong claims. The PKG paper at least compared against established benchmarks (HumanEval, MBPP) with actual baseline numbers. HASTE is weaker: only 6 curated tasks, no baseline comparison numbers shown, and its own defined metrics (hallucination rate, AST fidelity) never reported. Compared to RCC and FRAPPE, HASTE's evaluation is even thinner—those papers at least had proper baseline tables.

The core idea is sound and the pipeline design is a genuine contribution, but the paper in its current form makes claims ("resolves the trade-off," "dramatically improving," "reducing hallucinations") that are far beyond what the evidence supports. The evaluation is essentially a proof-of-concept on a handful of tasks, with no comparative results against the defined baselines.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>