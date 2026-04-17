---
job_id: c82aad9f-0c1a-40b3-b40f-41937938f931
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: ao7VBbvWIK.pdf
paper: HASTE: Hybrid AST-Guided Selection With Token-Bounded Extraction
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper focuses on LLM context compression, retrieval, and AST-based code representation for automated code editing, which falls squarely within representation learning, RAG, and applied ML for software engineering.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results/Discussion, Conclusion) are present and written in English. The work is incomplete empirically (no baseline numbers) but not so empty or flawed to justify an automatic desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No signs of prompt injection, hidden instructions targeting LLM reviewers, or other manipulative content are present in the provided main paper text.

---

# Expected Review Outcome:

## Summary

The paper introduces HASTE, a modular framework for constructing LLM prompts from large codebases under tight context budgets. HASTE combines AST-aware chunking, lexical and embedding-based hybrid retrieval with Reciprocal Rank Fusion, and call-graph-based expansion, followed by token-budget-aware pruning to preserve syntactic integrity while achieving strong compression. Experiments on a small curated set of six Python files and a subset of SWE-PolyBench tasks, evaluated via an LLM-as-a-judge, suggest that HASTE can obtain up to about 85% code compression with high judged code-edit quality and low hallucination.

## Strengths

1. **Clear articulation of a real and important problem.** The paper squarely targets the context-window bottleneck for LLM-based code editing, a central obstacle in practical code-assistance systems. The introduction and related work (Pages 1–3) make a coherent case that existing structure-only or relevance-only approaches are insufficient in isolation.

2. **Reasonable and well-engineered system design.** The HASTE pipeline in **Figure 1** is thoughtfully decomposed into Scanner, AST-aware Chunker, Identifier Extraction, Embedding + BM25 indexing, hybrid RRF retrieval, call-graph expansion, and token-bounded export. This modular architecture is easy to map onto real systems and clarifies how structure and relevance signals are fused.

3. **AST-aware chunking and budget-aware expansion/pruning are conceptually appropriate for code.** Using AST boundaries for chunking (Section 3.1) and enforcing AST integrity when expanding via call graphs (Section 3.3, “Selection”) directly attacks the brittleness of token-level pruning in code. That design choice is sound and aligns with known failure modes of line- or token-based retrieval.

4. **Hybrid retrieval design is well-specified and grounded in IR practice.** Section 3.2 and 3.3 give a clear description of lexical (BM25) and embedding-based indexes, with **Equation (1)** (RRF formula) specifying how scores from BM25 and semantic search are combined. This is precise enough to reproduce and is a reasonable choice for merging heterogeneous signals.

5. **Some empirical evidence of high-quality edits under strong compression.** On the curated six-file dataset, **Table 2** combined with **Figures 2(a) and 2(b)** shows that Judge Scores are consistently high (≥90, mostly ≥98) while compression ratios reach up to 6.8× (≈85% token reduction) for `test3.py`. **Figures 2(c) and 2(d)** also visualize the correlation between compression and quality, making the trade-off quite transparent.

6. **Qualitative analysis of failure modes.** Section 5.3’s discussion of SWE-PolyBench outcomes (and **Figure 3**) explicitly calls out low-scoring cases (scores 0–10) and ties them to misinterpretation of ambiguous tasks or flawed suggestions. This is more honest and informative than purely reporting averages and suggests the authors have at least inspected failures.

7. **Implementation and release plan.** The paper states that HASTE is already implemented as a Python package (Page 9, “Data Availability”), with a plan to release code and data. For a systems-style contribution, that substantially increases potential practical impact.

## Weaknesses

1. **No empirical comparison with any baseline, despite defining them.** Section 4.1.3 explicitly lists three baselines (IR-only retrieval, AST-only retrieval, naive truncation). However, **no table or figure anywhere in Section 5** reports performance of these baselines on either the curated dataset or SWE-PolyBench. All quantitative results (Table 2, Figures 2 and 3) appear to show only HASTE. This is a fundamental methodological gap: the paper’s core claim is that combining AST guidance with hybrid retrieval and budget-aware pruning improves over structure-only or relevance-only methods. Without measured baseline scores, it is impossible to assess how much value each component adds or whether HASTE actually improves over simpler heuristics like “take the whole function” or “BM25 over functions.”

2. **Extremely small, handpicked evaluation for the core RQ1/RQ2.** For the compression–quality trade-off (RQ2) and primary edit-quality analysis (RQ1), the curated dataset consists of only **six Python files** with one automatically generated edit task each (Table 1 and Table 2). That is 6 tasks in total. With $n=6$, the claimed Pearson correlation $r=-0.97$ in **Figure 2(c)** and $r=-0.81$ in **Figure 2(d)** is very fragile; a single outlier (e.g., `test3.py`) substantially determines the slope. This is far too small a sample to make strong claims about the general trade-off between compression and quality.

3. **LLM-as-a-judge evaluation is underspecified and potentially fragile.** Section 4.2.1 states that a “general-purpose LLM” judges correctness, readability, and alignment, but provides no details on:  
   - the exact scoring rubric, weighting of dimensions, or prompt template,  
   - how the three runs per task (Page 5, Section 4.1.4) map into a single Judge Score (mean? median? majority?),  
   - inter-judge reliability or any sanity checks (e.g., self-consistency of the judge when re-scoring identical patches).  
   Since all central quantitative claims in **Table 2**, **Figures 2** and **Figure 3** hinge on this metric, its opacity and lack of calibration significantly weaken the empirical support.

4. **Very limited use of SWE-PolyBench and incomplete characterization.** Section 5.3 references “a series of tasks from SWE-PolyBench” but does not state the number of tasks actually used or the selection criterion, only mentioning that instances with “processing errors” are excluded. **Figure 3** shows 12 instances (judging from the x-axis labels), which is still a tiny fraction of the benchmark. Moreover, we only see HASTE’s scores, without baseline comparison or per-task context length, AST fidelity, or hallucination rate. As a result, it is impossible to know whether HASTE is competitive in a realistic benchmark setting.

5. **Missing quantitative evidence for AST fidelity and hallucination rate.** Section 4.2 introduces AST Fidelity and Hallucination Rate as two key metrics. Yet **no table or figure** reports numeric values for these metrics. The results section mostly restates LLM-as-judge scores and qualitative comments on hallucinations. Since much of the paper’s motivation is reducing hallucinations and preserving structure, the absence of actual AST fidelity or hallucination statistics undermines the central claims.

6. **No ablations to justify architectural choices.** HASTE includes many design components: AST-aware chunking (Section 3.1), identifier-based lexical fingerprints, hybrid BM25 + embeddings with RRF (Sections 3.2–3.3), call-graph-based expansion, and token-budget enforcement. Yet the experiments never vary or ablate these components. For example:
   - How much does call-graph expansion contribute relative to just selecting the top-$k$ retrieved chunks?  
   - Does RRF outperform “semantic-only” or “BM25-only” retrieval in this setup?  
   - Is AST-aware chunking necessary, or would simple line-based or function-level chunking suffice?  
   Without such analysis, it is unclear which parts of HASTE are essential and which are just engineering choices.

7. **Mathematical and algorithmic specification is superficial in critical places.** While Equation (1) for RRF is correctly stated, several key operations in the pipeline remain underspecified in technical terms:  
   - In Section 3.1, the Chunker “uses AST-aware logic” to produce compilable units, but there is no formal definition of what AST nodes are grouped, how multi-function files are handled when the token budget is exceeded, or whether partial classes are allowed.  
   - In Section 3.3, the “Selection” step states that HASTE expands via the call graph up to some depth and then filters under a strict token budget, but there is no description of the exact objective (e.g., is it greedily adding chunks while counting tokens, or an optimization problem balancing relevance score vs. token size?), nor how call-graph edges are constructed when dynamic features or polymorphism are present.  
   - The definition of AST Fidelity in Section 4.2.2 is vague: “comparing the AST of the system's output against the reference modified AST.” Does this mean exact tree equality, tree edit distance, or some structural similarity measure? Without a precise formula, the metric is not reproducible.

8. **Over-interpretation of very limited correlation plots.** In Section 5.2, the paper claims that “HASTE successfully navigates the frontier of this trade-off” based largely on **Figures 2(c) and 2(d)** with six points. With no baselines and tiny sample size, the strong Pearson correlations are not meaningful evidence of a general relationship. At best, they reflect a monotonic trend on a handcrafted mini-dataset. Presenting this as a robust insight risks overstating the empirical findings.

9. **Related work on context extension/compression for LLMs is incomplete.** The related work section extensively covers AST-based code models and RAG for code, but it omits several directly relevant lines of work on semantic compression and context-window extension for LLMs (see “Potentially Missing Related Work” below). This makes it harder to understand what is truly new compared to existing semantic compression strategies, and whether HASTE is just an application of those ideas to code with AST-aware chunking.

10. **Evaluation protocol does not convincingly measure “hallucination reduction.”** The title and abstract emphasize that HASTE “reduces model-generated hallucinations,” but there is no systematic comparison of hallucination frequency with and without HASTE or against the baselines. The qualitative examples in Section 5.3 (e.g., generic try-except, copying generic examples) are informative, but they do not isolate the effect of context compression vs. inherent LLM limitations or poor task prompts. Without a controlled comparison, the claim of hallucination reduction remains largely speculative.

11. **Narrow task diversity for code edits.** All curated tasks in Table 2 are small, local modifications: adding try/except, type hints, default checks, or simple return annotations. These are relatively “friendly” tasks for LLMs, especially with high-quality surrounding context. It is unclear whether HASTE’s approach scales to more complex edits (e.g., cross-file refactoring, protocol changes, large multi-function modifications) that truly stress both semantics and structure.

12. **Presentation is somewhat informal and lacks rigorous framing in parts.** The writing style occasionally reads more like an engineering blog than a research paper, for example referring to “Frankenstein context” (Page 1) or using anecdotal language like “our replication of these approaches revealed a critical flaw” (Section 2.2) without showing the replication or its quantitative outcome. This informality detracts from the perceived rigor of the work.

## Potentially Missing Related Work

1. **Ratner et al., “Parallel Context Windows for Large Language Models”, 2022.**  
   This work proposes methods for handling long contexts by processing multiple windows in parallel and merging representations, directly addressing the same context-window bottleneck that motivates HASTE. It should be discussed in the related work on context handling (probably in Section 2.4 or a new subsection) to clarify how HASTE’s AST-guided compression differs from or complements parallel-window approaches.

2. **Fei et al., “Extending Context Window of Large Language Models via Semantic Compression”, 2023.**  
   This paper introduces semantic compression to extend effective context length, which is conceptually very close to HASTE’s goal of compressing code while preserving relevant semantics. It should be cited and contrasted in Sections 2.2 or 2.4, particularly when motivating why AST-guided structural constraints are necessary beyond generic semantic compression. A short discussion after the introduction of HASTE’s compression (Section 3.3 / Section 4) could clarify what is specific to code and ASTs compared to general semantic compression techniques.

## Questions

1. **Baselines and quantitative comparison.** You define three baselines in Section 4.1.3 but never report their results. Can you provide a full comparison table (e.g., for both the 6 curated tasks and the SWE-PolyBench subset) that includes HASTE, IR-only, AST-only, and naive truncation, measuring at least Judge Score, AST Fidelity, and Hallucination Rate? This would significantly strengthen the empirical claims.

2. **AST Fidelity metric definition and results.** How exactly is AST Fidelity computed (e.g., tree edit distance, fraction of unchanged nodes, structural similarity)? Please provide a formal definition and add a table (perhaps alongside **Table 2**) showing AST Fidelity for each file and baseline, to substantiate the claim that HASTE maintains structural integrity.

3. **Hallucination Rate measurement.** How was Hallucination Rate operationalized in practice? Was it annotated manually, judged by the LLM-as-judge, or heuristically derived from diffs? Please clarify and provide quantitative hallucination statistics across methods.

4. **Details of the retrieval and selection heuristics.** In Section 3.3, what is the exact algorithm for selecting chunks under a token budget once you have RRF scores and call-graph expansions? Is it a greedy algorithm starting from top-ranked chunks, or do you solve a knapsack-like objective combining score and token count? Clarifying this (and ideally giving pseudocode) would make the method more reproducible.

5. **SWE-PolyBench coverage and selection.** How many tasks from SWE-PolyBench did you attempt, how many failed due to processing errors, and how were the reported tasks in **Figure 3** selected? Are they representative or cherry-picked? A more systematic description of the benchmark usage and an aggregate metric (e.g., average score across all valid tasks) would be very helpful.

6. **Sensitivity to LLM choice and judge.** All experiments use Gemini 1.5 Flash both as editor and judge. Have you tried any other LLMs, even on a subset of tasks? If not, do you expect significantly different behavior for models with different strengths (e.g., weaker code models, or models with different context lengths)? Any evidence or argument here would help gauge robustness.

7. **Ablation on call-graph depth and hybrid retrieval weights.** Can you provide ablation experiments varying call-graph expansion depth and the relative weighting of lexical vs. semantic retrieval in RRF? This would clarify whether those choices are critical to performance, and might also reveal interesting patterns about how much additional context is useful.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The core architectural ideas are plausible and reasonably specified, but the empirical methodology is incomplete: baselines are defined but never evaluated, key metrics (AST Fidelity, Hallucination Rate) are not reported, and dataset sizes are very small, which limits the strength of the conclusions.

## Presentation Rating

2: fair.  
The paper is generally readable and **Figure 1** plus **Figures 2–3** and **Tables 1–2** help convey the pipeline and results, but important details of the metrics and algorithms are missing, related work on semantic compression is incomplete, and the tone drifts toward informal in several places.

## Contribution Rating

2: fair.  
The idea of combining AST-aware chunking with hybrid retrieval for code context selection is useful and practically relevant, but the lack of rigorous empirical comparison and limited benchmark coverage make it hard to judge the real advance over simpler baselines or existing context-compression methods.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper addresses an important, practical problem and presents a coherent and seemingly useful system design, with some promising initial results and clear potential for impact as an open-source tool. However, the experimental section is substantially underdeveloped for ICLR: no baseline numbers despite listing baselines, tiny curated evaluation (6 tasks), incomplete reporting of the advertised metrics, and missing connections to closely related work on semantic compression. With stronger and more systematic experiments and clearer mathematical specification of metrics, this could evolve into a solid contribution.

## Reviewer Confidence

4: confident.  
I am familiar with LLM RAG systems, AST-based code representation, and evaluation methodologies, and have carefully checked the described pipeline, equations, and results. Some details remain unspecified in the paper, but the assessment of strengths and weaknesses is unlikely to change dramatically.