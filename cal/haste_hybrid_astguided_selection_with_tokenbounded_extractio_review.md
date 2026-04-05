=== CALIBRATION EXAMPLE 13 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Accuracy & Clarity:** The title accurately reflects the core mechanism (AST-guided selection, token budgeting). The abstract clearly outlines the context-window trade-off, the proposed hybrid retrieval + AST pipeline, and the main empirical claim (compression + improved edit success).
- **Supported Claims:** The claim that HASTE "significantly improving the success rate of automated code edits" is currently **unsupported by the abstract and the paper body**. While the abstract states improvement over the trade-off dilemma, it implies comparison against baselines, yet baseline results are never quantitatively presented in the text or tables. Additionally, "reducing model-generated hallucinations" is highlighted, but the Hallucination Rate metric defined in Section 4.2.3 is never reported in Section 5, making this claim unverified.

### Introduction & Motivation
- **Motivation & Gap:** The motivation is clear: developers face a tension between structure-preservation and semantic relevance when retrieving code for LLMs. The gap is reasonably identified, though the dichotomy is presented as strictly binary, overlooking recent hybrid or graph-based retrieval methods that already attempt to bridge this.
- **Contributions:** Contribution 1 (the pipeline) is more of an engineering integration of established components (AST chunking, BM25+Dense retrieval with RRF, call-graph expansion) rather than a novel algorithmic advance. Contribution 3 ("analysis of the trade-off") is largely a descriptive observation of Contribution 2 rather than a standalone contribution.
- **Over/Under-claiming:** The introduction slightly over-claims by framing this as resolving a "dichotomy" and presenting the pipeline as "novel." For an ICLR submission, the framing should more explicitly distinguish what methodological or empirical advance this specific combination provides over existing code-RAG or repo-level retrieval systems (e.g., RepoCoder, LSP-augmented context windows).

### Method / Approach
- **Clarity & Reproducibility:** The high-level architecture (Section 3) is modular and readable, but the **core mechanism is under-specified**. Section 3.3 states the expanded call-graph set is "filtered under a strict token budget," but offers no algorithm for *how* this filtering preserves AST integrity. Does it greedily drop lowest-RRF nodes? Does it backtrack if dropping a node breaks a dependency? The title promises "AST-guided selection," yet the method does not detail how the AST actively governs the budget constraint vs. simply chunking whole functions.
- **Assumptions:** The method assumes static call-graph extraction is sufficient. In Python (the language used in experiments), dynamic imports, reflection (`getattr`), and higher-order functions frequently break static analysis, leading to incomplete call graphs. This is not discussed.
- **Edge Cases & Failure Modes:** No discussion of how the system handles massive files where the call graph itself exceeds the token budget, or cyclic import dependencies. The "Exporter" mentions ordering chunks to "minimize forward references," but provides no heuristic for resolving them if budget cuts prevent inclusion of dependent definitions.

### Experiments & Results
- **Testing Claims & Baselines:** The experiments **do not currently test the comparative claims**. Section 4.1.3 lists three baselines (IR-only, AST-only, Naïve truncation), but Table 2 and the discussion in Section 5 **only report HASTE results**. Without baseline scores, claims of "improvement" and navigating the trade-off "effectively" cannot be validated. This is a critical omission for ICLR standards.
- **Ablations & Metrics:** Section 4.2.2 and 4.2.3 define *AST Fidelity* and *Hallucination Rate*, yet **neither metric appears in Section 5**. Only LLM-as-judge scores are reported. I expect a full results table including these metrics to substantiate the structural integrity and hallucination reduction claims. Furthermore, the choice of $k=60$ for RRF is stated without ablation; different fusion weights could materially affect the structure-relevance balance.
- **Statistical Rigor & Data Scale:** The curated analysis relies on **$n=6$ files**. Computing a Pearson correlation ($r = -0.97$, Section 5.2) on six points is statistically unstable and highly sensitive to the single outlier (`test3.py`). ICLR expects larger-scale evaluations, especially for a "robust framework." Error bars or confidence intervals are absent despite mentioning "each task was executed three times" (only averages are shown).
- **SWE-PolyBench:** The discussion in Section 5.3 notes that instances with "processing errors" were excluded but provides no count or analysis of error types. This introduces potential survivorship bias. Reporting raw pass/fail rates or patch accuracy against executable tests would strengthen validity over relying solely on an LLM-judge.

### Writing & Clarity
- The paper is generally well-structured and easy to follow.
- **Figure/Table Alignment:** Table 2 lists files and scores but lacks a unified view with baselines, which reduces its informativeness. Figures 2(c) and 2(d) illustrate the negative correlation clearly, but the axes labels and sample sizes would benefit from explicit notation in the caption.
- **Conceptual Clarity:** Section 3.4 (Observability) reads like product documentation rather than research content. While useful for deployment, it distracts from the methodological contribution and doesn't tie back to the experimental results (e.g., no latency trade-off analysis against retrieval quality).

### Limitations & Broader Impact
- **Acknowledged Limitations:** Section 5.3 honestly discusses failure modes related to ambiguous prompts and flawed suggestions. Section 6 mentions cross-file analysis and multi-language support as future work.
- **Missed Limitations:** The paper fails to acknowledge that AST-based chunking assumes syntactically valid input; broken or legacy codebases may fail parsing entirely. Furthermore, the static analysis limitation in dynamic languages (Python) is a fundamental constraint that must be discussed.
- **Societal/Broader Impact:** There is no discussion. While optional for some venues, ICLR papers increasingly expect reflection on AI-for-SE impacts. For example, does aggressive context compression risk omitting security-critical functions (e.g., input validation), potentially guiding an LLM to generate unsafe patches? A brief discussion would strengthen the paper's maturity.

### Overall Assessment
HASTE addresses a genuine bottleneck in code LLMs—balancing structural coherence with semantic relevance within token limits—and proposes a sensible pipeline combining AST chunking, hybrid retrieval, and budget-aware selection. However, the paper currently falls short of ICLR's empirical rigor standards. The most critical concern is the absence of baseline results in Section 5, which renders comparative claims unsubstantiated. Additionally, key metrics defined in the methodology (AST Fidelity, Hallucination Rate) are never reported, and the primary analysis relies on a sample size of $n=6$ with weak statistical grounding. The method section also lacks algorithmic detail on how the token budget interacts with AST preservation, leaving the core novel claim under-specified. The contribution stands as a promising engineering integration, but acceptance would require: (1) inclusion of baseline and ablation results with error bars, (2) reporting all defined metrics, (3) expansion to a larger benchmark or scale, and (4) clarification of the budgeted selection algorithm and static analysis limitations.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces HASTE, a retrieval and context-compression framework that integrates lexical/semantic IR, Reciprocal Rank Fusion (RRF), call-graph expansion, and AST-guided pruning to extract code snippets that fit within strict LLM token budgets while preserving syntactic coherence. Evaluated on a small curated dataset and tasks from SWE-PolyBench, the method reports high LLM-as-judge scores and compression ratios up to 85%, arguing that structurally sound context reduces model hallucinations during localized code edits.

### Strengths
1. **Clear Problem Motivation and Pipeline Design:** The paper effectively articulates the tension between semantic relevance and structural integrity in code context retrieval (Sec 1) and provides a well-structured, modular architecture (Sec 3) that logically integrates established IR and code-parsing techniques into a cohesive pipeline.
2. **Practical Relevance and Transparency on Trade-offs:** The focus on token-bounded extraction directly addresses a critical industry bottleneck for LLM code assistants. Table 2 transparently reports both compression ratios and judge scores per file, and the authors honestly acknowledge failure modes on SWE-PolyBench where ambiguous prompts or flawed suggestions degrade performance (Sec 5.3).
3. **Explicit Hallucination Mitigation Strategy:** Rather than treating hallucination as a post-hoc decoding problem, HASTE intervenes at the retrieval stage by guaranteeing structurally valid, AST-constrained context (Sec 2.4). This proactive framing is conceptually sound and aligns well with current trends in reliable RAG.

### Weaknesses
1. **Limited Empirical Scale and Weak Baselines:** The primary quantitative analysis (RQ2) relies on only six curated Python files (Table 1), which is insufficient to establish statistical significance. Baselines (IR-only, AST-only, naive truncation) omit strong, recent code-RAG and graph-based retrieval methods, making it unclear how HASTE compares to state-of-the-art.
2. **Unreported Metrics and Overstated Claims:** Section 4.2.2 and 4.2.3 define AST Fidelity and Hallucination Rate as core evaluation metrics, yet Section 5 provides zero quantitative results for them. Additionally, claiming a "strong negative correlation (r = -0.97)" between compression and quality is statistically fragile with N=6 and is used to conclude "compression is not harmful" without rigorous ablation or larger-scale validation.
3. **Reliance on Single LLM-as-Judge and Lack of Ablations:** The primary evaluation metric is a single LLM (Gemini 1.5 Flash) without human calibration, prompting robustness checks, or multi-model aggregation, which introduces potential bias and limits reliability. The paper also lacks component-level ablations (e.g., impact of RRF vs. single-index retrieval, call-graph depth, AST pruning thresholds), obscuring which design choices actually drive performance.
4. **Methodological Novelty is Primarily Integrative:** HASTE combines known techniques (BM25, dense embeddings, RRF, AST chunking, call-graph traversal, and token budgeting) into a pipeline rather than proposing a new algorithmic, theoretical, or architectural contribution. ICLR typically expects stronger methodological novelty or large-scale empirical insights beyond system integration.

### Novelty & Significance
**Novelty: Moderate-Low.** The contribution lies in engineering integration rather than algorithmic innovation. Hybrid retrieval (BM25 + dense + RRF), AST-aware chunking, and graph-based expansion are individually established in code intelligence and general RAG literature. HASTE's specific combination and token-bounded constraint are practical but do not substantially advance underlying ML methodology.
**Significance: Moderate-High (Practical), Moderate (Academic).** The problem of context-window-constrained code retrieval is highly relevant to deployed AI-assisted development tools. HASTE provides a functional, interpretable baseline that demonstrates the value of structural guarantees in code context. However, without larger-scale, statistically robust evaluation and ablation studies, its significance for the broader ML research community remains constrained by empirical scope rather than conceptual impact. For ICLR, the paper falls short of the expected rigor in experimental design and novelty.

### Suggestions for Improvement
1. **Expand Evaluation and Strengthen Baselines:** Scale the empirical evaluation to at least dozens or hundreds of files/tasks across multiple languages. Include strong contemporary baselines (e.g., CodeRAG, RepoFusion, GraphRAG adaptations, StarCoder2 context strategies) and report aggregate metrics with variance/confidence intervals. Replace or heavily supplement the LLM-as-Judge with objective metrics (compilation rate, exact-match/pass@k, unit test pass rates, human evaluation).
2. **Report All Defined Metrics and Provide Component Ablations:** Add tables/figures quantifying AST Fidelity and Hallucination Rate. Conduct systematic ablation studies isolating the impact of (a) BM25 vs. dense vs. hybrid ranking, (b) call-graph expansion depth, and (c) AST pruning strictness. This will clarify which components contribute most to performance and justify architectural choices.
3. **Rigorize Statistical Analysis and Temper Claims:** With N=6, correlation coefficients should not be used to infer generalizable trends. Use proper statistical tests, acknowledge sampling limitations, and reframe conclusions to reflect the pilot-scale nature of the curated dataset. Provide exact token budget values, embedding model names/sizes, FAISS/configuration details, and RRF weight tuning procedures to meet ICLR reproducibility standards.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add comparisons against state-of-the-art code-specific context selection methods (e.g., ContextRAG, RepoCoder, or retrieval-augmented code editors from recent top venues), because benchmarking only against naive truncation, IR-only, and AST-only fails to prove HASTE outperforms modern alternatives.
2. Implement a rigorous component-wise ablation (removing hybrid ranking, call-graph expansion, and AST-bounded pruning individually), because without it there is no evidence that the claimed performance stems from architectural synergy rather than from simply adding more context retrieval steps.
3. Evaluate using execution-based ground-truth metrics (e.g., test pass@1 or exact diff matching on patch datasets), because LLM-as-a-judge scores alone cannot objectively verify functional correctness or prove that hallucinations are actually reduced rather than just stylistically masked.

### Deeper Analysis Needed (top 3-5 only)
1. Report the AST Fidelity and Hallucination Rate results defined in Sections 4.2.2/4.2.3, because the paper’s central claim that structural constraints reduce hallucinations is unsupported without quantitative measurements of these specific metrics.
2. Formalize and analyze the token-budget arbitration logic during call-graph expansion and RRF fusion, because the current description leaves it unclear whether relevance or syntactic integrity is silently sacrificed when the budget is exhausted mid-expansion, directly undermining reproducibility.
3. Statistically contextualize the strong negative correlation (r=-0.97) between compression ratio and judge score across all baselines, because presenting quality degradation as a "trade-off" without showing HASTE stays significantly ahead of baselines at matched compression levels misrepresents the actual performance frontier.

### Visualizations & Case Studies
1. Provide a Pareto-front plot showing compression ratio vs. task accuracy/hallucination rate for all methods, because this standard visualization would immediately reveal whether HASTE genuinely achieves a superior trade-off curve or merely sits on an inferior frontier.
2. Include a side-by-side annotated diff of a high-compression success and a failure, explicitly highlighting retained vs. pruned AST nodes and call-graph dependencies, because this is the only way to visually verify that the claimed "semantic + structural" preservation actually prevents gap-induced hallucinations.

### Obvious Next Steps
1. Calibrate the LLM-as-a-judge with a formal rubric, inter-annotator (LLM) agreement analysis, and human correlation checks, because relying on a single unanchored judge model leaves the primary evaluation metric highly vulnerable to prompt bias and generative sycophancy, which ICLR reviewers routinely flag.
2. Scale evaluation to a statistically robust corpus (hundreds of real-world tasks across multiple languages), because drawing generalizable claims about "reliable and scalable AI-assisted development" from six hand-picked Python files and a narrow benchmark subset is insufficient for publication.
3. Report end-to-end preprocessing, indexing, and retrieval latency alongside token savings, because demonstrating that 85% compression justifies the upfront computational overhead of dual indexing, AST parsing, and call-graph construction is mandatory for practical claim validity.

# Final Consolidated Review
## Summary
HASTE proposes a retrieval and context-compression pipeline that combines hybrid lexical/semantic search with AST-aware chunking and call-graph expansion to extract structurally coherent, token-bounded code contexts for LLMs. Evaluated on a small curated dataset and SWE-PolyBench, the method reports compression ratios up to 85× while maintaining high LLM-as-judge scores, positioning structural preservation as a proactive defense against code hallucinations.

## Strengths
- **Proactive structural constraint for hallucination mitigation:** By enforcing AST boundaries during token-bounded extraction, the framework addresses code hallucination at the retrieval stage rather than relying on post-hoc decoding or prompt adjustments. This reframes structural coherence as a prerequisite for reliable generation, a conceptually sound shift from conventional RAG practices (Sec 2.4).
- **Transparent performance trade-offs:** Table 2 explicitly pairs per-file compression ratios with judge scores, and Section 5.3 honestly isolates failure modes to ambiguous prompts or flawed suggestions rather than retrieval deficits. This clear boundary-setting provides realistic expectations of the system's operational limits.
- **Cohesive, problem-aligned pipeline:** The integration of hybrid ranking (RRF), dependency expansion, and syntax-aware pruning directly targets the documented tension between semantic relevance and executability in code LLMs, offering a practical architecture for context-window-constrained editing tasks.

## Weaknesses
- **Unreported core metrics and absent baseline data:** Sections 4.2.2 and 4.2.3 define *AST Fidelity* and *Hallucination Rate* as central evaluation criteria, yet Section 5 provides zero quantitative results for them. Furthermore, Section 4.1.3 specifies three baseline strategies, but Table 2 and the results discussion exclusively report HASTE scores. Without baseline comparisons and the explicitly defined structural/hallucination metrics, the paper’s core claims that HASTE "effectively navigates" the relevance-structure trade-off and "reduces model-generated hallucinations" remain empirically unsubstantiated.
- **Underspecified token-budget arbitration:** Section 3.3 states that the call-graph expanded candidate set is "filtered under a strict token budget," but provides no algorithm for how pruning decisions are prioritized once limits are reached. It is unclear whether the system greedly drops lowest-RRF nodes, backtracks to preserve dependency chains, or truncates mid-AST. This omission critically undermines reproducibility and leaves the core "AST-guided selection" mechanism inadequately defined.
- **Pilot-scale evaluation with overconfident statistical claims:** The primary compression-quality analysis (RQ2) relies on exactly six Python files (Table 1), yet reports a Pearson correlation of $r = -0.97$ and generalizes the observed trend. With $n=6$, correlation coefficients are highly volatile to single outliers (e.g., `test3.py` at 6.8× compression). The absence of variance reporting despite triplicate runs, combined with survivorship bias in SWE-PolyBench ("excluded processing errors" without counts), falls short of the statistical rigor expected for empirical ML/systems claims.

## Nice-to-Haves
- Add component-wise ablations (e.g., hybrid ranking vs. single-index, call-graph depth, AST pruning thresholds) to quantify the marginal contribution of each architectural module.
- Report end-to-end indexing, retrieval, and preprocessing latency to contextualize computational overhead against the reported token savings.
- Supplement or calibrate the single-model LLM-as-judge with execution-based ground truth (e.g., test pass@1) or human agreement metrics to reduce potential prompt bias.

## Novel Insights
The paper effectively reframes code context compression from a pure information-retention problem to a dependency-aware structural filtering problem. By treating AST coherence as a hard constraint rather than a soft preference, HASTE demonstrates that preventing hallucinations is most efficiently achieved by eliminating structurally invalid prompts before generation begins, rather than attempting to correct syntactic or logical gaps during LLM decoding.

## Suggestions
- **Report all defined metrics and baseline scores:** Populate Table 2/Section 5 with AST Fidelity, Hallucination Rate, and scores for the three specified baselines to directly validate the claimed performance frontier.
- **Formalize the budget-constrained pruning algorithm:** Provide pseudocode or a detailed breakdown of the selection logic when candidates exceed token limits, explicitly describing how AST parent-child relationships are guaranteed, how cycles are resolved, and what backtracking or tie-breaking rules apply.
- **Scale empirical validation and correct statistical overreach:** Expand evaluation to a statistically meaningful corpus (dozens/hundreds of tasks), report mean ± variance across runs, quantify excluded SWE-PolyBench errors, and remove or clearly qualify correlation statistics derived from $n \leq 6$ samples.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0]
Average score: 1.0
Binary outcome: Reject
