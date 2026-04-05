=== CALIBRATION EXAMPLE 11 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- Does the title accurately reflect the contribution?
  - The title is broadly accurate: the paper proposes a hybrid, AST-guided retrieval/compression method with token budgets.
  - However, “Selection with Token-bounded Extraction” is slightly underspecified relative to the actual system, which also includes hybrid lexical/semantic retrieval and call-graph expansion. The title emphasizes extraction, but the method is more of a retrieval-and-pruning pipeline.

- Does the abstract clearly state the problem, method, and key results?
  - The abstract does identify the core problem: LLM context limits for code.
  - It also states the general method: AST + lexical/semantic search + structure-preserving extraction.
  - The key results are stated in high-level terms, but the abstract is too confident given the evidence presented later. Claims like “significantly improving the success rate” and “reducing model-generated hallucinations” are not backed in the paper by rigorous comparative evidence or statistically robust evaluation.

- Are any claims in the abstract unsupported by the paper?
  - Yes. The abstract claims “robust LLM-as-a-judge framework” and “reducing hallucinations,” but the paper does not provide a sufficiently rigorous judge protocol, inter-rater validation, or human evaluation to establish robustness.
  - The “up to 85% code compression” claim is supported by the curated dataset, but the sample is very small and the largest compression is reported on just one file.
  - The claim that HASTE “maintain[s] high structural fidelity” is asserted via an AST fidelity metric, but the paper does not provide enough detail to assess whether this metric is meaningful or correctly computed.

### Introduction & Motivation
- Is the problem well-motivated? Is the gap in prior work clearly identified?
  - The motivation is reasonable and aligned with ICLR expectations: context compression for code LLMs is an important practical problem.
  - The paper clearly frames a real tension between structure-preserving and relevance-focused context selection.
  - That said, the “gap” is somewhat overstated. The paper presents the field as if there were a sharp dichotomy between AST-based structure and IR-based relevance, but it does not engage deeply with existing code-retrieval or repository-context methods that already combine multiple signals or use structured retrieval.

- Are the contributions clearly stated and accurate?
  - The contributions are stated explicitly in the introduction.
  - But they are not all equally substantiated. “A novel pipeline” is plausible, yet “maintaining high structural fidelity” and “mitigating hallucination risks” are stronger claims than the evidence supports.
  - The contribution list also blurs what is algorithmic novelty versus system integration. Much of the pipeline reads like an engineering composition of standard pieces (BM25, embeddings, AST chunking, call graph expansion, RRF).

- Does the introduction over-claim or under-sell?
  - It over-claims. The paper repeatedly implies broad, general advances in code intelligence and hallucination reduction, but the evaluation is limited and not comparative enough to justify that level of generality.
  - The framing suggests a principled resolution of a fundamental trade-off, yet the empirical section mostly demonstrates that the method can work on a small curated benchmark and some SWE-PolyBench tasks, not that it dominates the trade-off across realistic settings.

### Method / Approach
- Is the method clearly described and reproducible?
  - Partially. The architecture is described at a high level, but the paper omits too many implementation details for reproducibility.
  - Key missing details include: the exact embedding model, chunking rules, call-graph construction method, how AST fidelity is computed, how token budgets are enforced, how expansions are ordered and truncated, and how the “Suggestion Generator” creates tasks.
  - The method section also uses many generic statements that sound like a product description rather than a precise algorithmic specification.

- Are key assumptions stated and justified?
  - Not sufficiently.
  - A major assumption is that AST-guided expansion preserves usefulness while retaining syntactic coherence, but the paper does not justify why the chosen traversal depth, candidate selection, or budget allocation should be robust across languages or repositories.
  - Another assumption is that call graphs and ASTs are available and accurate; the paper does not discuss cases where static analysis is incomplete, dynamic dispatch is present, or code is not parseable.

- Are there logical gaps in the derivation or reasoning?
  - Yes, especially in Section 3.3 and Section 4.
  - The retrieval pipeline describes RRF over BM25 and semantic search, then call-graph expansion, then filtering under a token budget. But it does not specify how expansion interacts with ranking. For example, does a highly relevant candidate bring in many irrelevant dependencies? How are conflicts resolved?
  - The claim that AST-aware pruning “guarantees” syntactic validity is too strong. AST-bounded selection can help, but syntactic validity is not guaranteed in all cases unless the final assembly process is formally constrained and validated.
  - The RRF equation is presented, but the formula text is not enough to establish the system’s actual behavior, especially when later stages alter the retrieved set substantially.

- Are there edge cases or failure modes not discussed?
  - Yes, several important ones:
    - Files with weak or missing ASTs, generated code, or syntax errors.
    - Languages or frameworks where call graphs are hard to compute statically.
    - Highly coupled code where preserving syntax still leaves missing semantic dependencies.
    - Queries requiring cross-file or repository-wide reasoning, which the authors themselves mention only as future work.
    - Tasks where retrieving “complete” functions is not enough because behavior depends on configuration, tests, or external APIs.

- For theoretical claims: are proofs correct and complete?
  - There are no real theoretical claims or proofs, so this is not applicable. But some claims are stated as guarantees without proof and should be weakened.

### Experiments & Results
- Do the experiments actually test the paper's claims?
  - Only partially.
  - The curated six-file benchmark does test whether compressed context can support local edits, but it is too small to support strong general conclusions.
  - The SWE-PolyBench evaluation is more promising, but the paper’s description is selective and incomplete; it explicitly excludes processing errors, which may bias results.
  - The main claim of outperforming structure-agnostic pruning is not directly tested because the baselines do not include a strong token-level pruning or context-compression baseline.

- Are baselines appropriate and fairly compared?
  - The baseline set is weak for an ICLR paper.
  - IR-only, AST-only, and naïve truncation are reasonable sanity checks, but they are not sufficient against a method that claims hybrid retrieval and compression novelty.
  - Missing baselines likely matter materially:
    - a strong code-RAG baseline using only retrieval,
    - a line-level or token-level compression baseline,
    - a repository-context baseline such as function-level retrieval with neighboring context,
    - possibly a learned context selection baseline.
  - Without these, it is hard to attribute gains specifically to HASTE rather than to generic retrieval plus structural padding.

- Are there missing ablations that would materially change conclusions?
  - Yes, and this is one of the paper’s biggest weaknesses.
  - The paper does not ablate:
    - hybrid retrieval vs BM25 alone vs embedding alone,
    - AST expansion depth,
    - call-graph expansion contribution,
    - effect of AST-aware pruning versus simpler full-function retrieval,
    - effect of identifier extraction,
    - effect of different token budgets.
  - Since HASTE is presented as a multi-stage hybrid system, ablations are essential to show which component is responsible for gains.

- Are error bars / statistical significance reported?
  - No.
  - The curated benchmark has only six samples, so mean scores and correlations are highly unstable. Reporting Pearson’s r = -0.97 on six points is not convincing evidence.
  - There are no confidence intervals, no significance tests, and no variance measures despite stating that tasks were run three times and averaged.
  - The repeated-run setup is not clearly reflected in the reported results, and it is unclear whether variability was negligible or simply omitted.

- Do the results support the claims made, or are they cherry-picked?
  - The curated results are vulnerable to cherry-picking concerns.
  - The six files were manually selected, and all reported scores are very high. That is not necessarily invalid, but it makes the benchmark look curated to showcase success rather than stress-test the method.
  - The SWE-PolyBench discussion also appears selective: it emphasizes perfect and near-perfect cases and only briefly discusses failures.
  - The paper explicitly excludes processing errors in Section 5.3, which can inflate apparent performance.

- Are datasets and evaluation metrics appropriate?
  - The choice of SWE-PolyBench is appropriate in spirit, but the paper does not explain the subset used, selection criteria, or exclusion rate, which is important for interpretation.
  - The six-file curated dataset is too small for claims about general compression-quality trade-offs.
  - The LLM-as-judge metric is reasonable as a proxy, but the paper does not validate it against human judgment or against an objective edit metric such as exact patch match, test pass rate, or compile success.
  - AST Fidelity and Hallucination Rate are interesting, but both are underdefined. “Hallucination” in code generation should ideally be operationalized more concretely, e.g., as compilation errors, extraneous insertions, or unsupported edits.

### Writing & Clarity
- Are there sections that are confusing or poorly explained?
  - Yes, mainly the method and evaluation sections.
  - Section 3 is polished at the level of system description, but too many phrases remain abstract to be actionable.
  - Section 4.1 and 4.2 do not give enough procedural detail to reproduce the experiments.
  - Section 5.3 is especially hard to assess because it summarizes results without clearly stating the exact task set, the number of excluded instances, or how the judge scored outputs.

- Are figures and tables clear and informative?
  - Table 1 is clear as a descriptive summary, though small.
  - Table 2 is too limited to support broad conclusions; it reports only compression ratio and judge score, without the underlying token counts, task complexity, or baseline comparisons.
  - Figures 2 and 3 are referenced, but the paper’s interpretation of them is stronger than what the small datasets justify.
  - The figures are useful for illustrating trends, but they do not establish robustness.

### Limitations & Broader Impact
- Do the authors acknowledge the key limitations?
  - Only partially.
  - The conclusion mentions future work on cross-file analysis and richer ranking signals, but the limitations are not clearly acknowledged as limitations of the current method.
  - The paper does not explicitly admit that the current evaluation is small-scale, task-specific, and not yet sufficient for a general claim of superiority.

- Are there fundamental limitations they missed?
  - Yes:
    - dependence on static analysis quality,
    - brittleness to repositories with incomplete call graphs,
    - inability to handle semantic needs beyond syntactic coherence,
    - reliance on judge models that may not reflect actual code correctness,
    - limited evidence for generalization beyond Python and a narrow task class.
  - The method may also struggle when the key context is not in the same function or call neighborhood but in tests, configs, or documentation.

- Are there failure modes or negative societal impacts not discussed?
  - The paper does not meaningfully discuss broader impacts.
  - There are mild safety concerns: if such a system is used to automate code edits, overconfidence in high judge scores could mask subtle bugs.
  - There is also a risk that a model-assisted workflow may propagate incorrect edits more efficiently if the retrieval system is wrong, but this is not discussed.
  - No serious negative societal impacts are apparent, but the absence of a broader-impact discussion is notable given the application domain.

### Overall Assessment
HASTE addresses an important and timely problem: how to provide LLMs with compact yet syntactically coherent code context under tight token budgets. The system idea—hybrid lexical/semantic retrieval plus AST-guided expansion and budgeted extraction—is plausible and practically interesting. However, for ICLR standards, the paper’s empirical support is too weak to justify its strongest claims. The evaluation is small, the baselines are not competitive enough, the ablations are missing, and the judge-based metrics are undervalidated. As written, the paper reads more like a promising system prototype than a fully established method with demonstrated generality. It could still be a useful contribution if the authors substantially strengthen the experimental evidence and clarify the algorithmic details, but in its current form it does not yet meet the bar for a strong ICLR acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes HASTE, a hybrid code-context retrieval and compression pipeline that combines lexical retrieval, semantic embeddings, AST-guided expansion/pruning, and call-graph traversal to produce token-bounded code snippets for LLM-based code editing. The central claim is that preserving syntactic structure while retrieving semantically relevant code improves downstream edit quality and reduces hallucinations under context-window constraints.

### Strengths
1. **Addresses an important and timely problem for code LLMs.**  
   The context bottleneck is a real limitation in software engineering applications, and the paper targets a practically relevant setting: retrieving compact, structured code context for edits rather than just whole-file understanding.

2. **Reasonable high-level design that combines complementary signals.**  
   The architecture explicitly combines BM25-style lexical search, semantic embeddings, AST-aware chunking, and call-graph expansion. This hybrid formulation is sensible for code, where exact identifier matches and structural dependencies both matter.

3. **Attempts to evaluate both utility and compression.**  
   The paper does not only report a single performance number; it also examines compression ratio, AST fidelity, and hallucination rate conceptually, and discusses a trade-off between compression and quality. That framing aligns with what ICLR reviewers often appreciate: an explicit efficiency-versus-performance analysis.

4. **Uses a real software engineering benchmark in addition to curated examples.**  
   Including SWE-PolyBench, even if only for part of the evaluation, is a positive sign because it moves beyond toy demonstrations and toward more realistic tasks.

5. **The paper is generally readable and well-motivated.**  
   The problem statement, the relevance-vs-structure tension, and the intended role of ASTs are explained clearly at a conceptual level, making the system easy to understand.

### Weaknesses
1. **The empirical evidence is far too limited for ICLR standards.**  
   The main quantitative analysis on the curated dataset appears to use only six Python files, which is not sufficient to support broad claims about robustness, compression behavior, or general effectiveness. For ICLR, this is a major concern: the evidence base is too small and too narrow to establish reliability.

2. **The evaluation methodology is under-specified and heavily dependent on LLM-as-judge.**  
   The paper says a general-purpose LLM scores outputs on correctness/readability/alignment, but it does not specify the judge prompt, calibration, inter-rater reliability, or whether the judge was blinded. This makes the primary metric difficult to trust and hard to reproduce.

3. **Key claims are stronger than what the reported results justify.**  
   The abstract claims HASTE “significantly improves” success rate and “reduces hallucinations,” but the results shown are mostly high scores on small examples, with no statistical significance testing, no confidence intervals, and no strong baselines. The evidence does not convincingly establish superiority.

4. **Baselines are weak and incomplete.**  
   The comparisons are only against IR-only retrieval, AST-only retrieval, and naive truncation. For an ICLR paper in this area, stronger baselines would be expected, such as recent code-RAG methods, tree-aware pruning methods, repository-level retrieval systems, and simple ablations of the hybrid scoring and call-graph expansion components.

5. **Ablation study is missing.**  
   The paper claims the benefit comes from the combination of hybrid retrieval, AST-guided filtering, and call-graph expansion, but it does not isolate the contribution of each component. Without ablations, it is impossible to know which part actually drives performance.

6. **The reported compression metrics are not fully interpretable.**  
   The paper mixes compression ratio, token reduction percentage, and “up to 85%” claims, but it does not clearly define the exact denominator or whether compression is measured before or after expansion. The numbers are therefore hard to compare across methods.

7. **Generalization is not convincingly demonstrated.**  
   The curated set is Python-only, and the broader benchmark discussion is qualitative and limited. The paper claims extensibility to other languages, but does not present cross-language experiments or evidence that the approach transfers well.

8. **Hallucination reduction is asserted more than measured.**  
   Although hallucination rate is listed as a metric, the paper does not present a clear quantitative hallucination analysis, operational definition, or comparative results showing that HASTE reduces hallucinations relative to baselines.

9. **Several methodological details needed for reproducibility are missing.**  
   Important implementation choices are unclear: how the AST chunker handles edge cases, how call-graph traversal is computed, what embedding model is used, how candidate fusion is parameterized beyond one RRF constant, and how token budgets are enforced exactly.

10. **The novelty is incremental relative to existing hybrid retrieval and AST-based code processing.**  
   The paper combines established ideas—hybrid retrieval, AST-aware chunking, and context selection—but does not yet demonstrate a clearly new algorithmic principle or compelling theoretical insight beyond a sensible system integration.

### Novelty & Significance
**Novelty: moderate to low.** The overall idea of combining lexical and semantic retrieval with AST-aware structural constraints is sensible, but the paper mostly assembles known components into a pipeline rather than introducing a clearly distinct method. The most novel aspect seems to be the specific packaging of hybrid retrieval plus AST-guided expansion under a token budget for code-edit context construction.

**Significance: potentially useful, but not yet convincingly established.** The problem is important for ICLR’s interests in language models, retrieval, and efficient context use, but the current empirical support is too limited to justify strong significance claims. In ICLR terms, the paper is directionally relevant but does not yet clear the bar for a strong acceptance because the evidence is not sufficiently rigorous or broad.

**Clarity: fairly good at the conceptual level, weaker at the technical level.** The motivation and pipeline description are understandable, but the exact algorithm, implementation choices, and evaluation protocol need much more precision.

**Reproducibility: moderate to weak.** The paper promises open-source release, which is good, but the current manuscript lacks enough detail to reproduce results independently from the text alone.

### Suggestions for Improvement
1. **Add a much stronger empirical evaluation.**  
   Evaluate on substantially more files and more diverse repositories, and report aggregate results with confidence intervals. Include both Python and at least one other language if possible.

2. **Provide rigorous ablations.**  
   Separate the effects of BM25, embeddings, AST chunking, call-graph expansion, and the token budget. This is essential to show what actually matters.

3. **Strengthen the baselines.**  
   Compare against recent code retrieval and code-context compression methods, not just truncation and simple IR/AST variants. Include strong repository-level retrieval baselines and any relevant pruning methods.

4. **Define and validate the judge-based evaluation carefully.**  
   Specify the exact LLM-as-judge prompt, scoring rubric, and any calibration procedure. Ideally supplement judge scores with functional tests, compilation checks, or human evaluation on a subset.

5. **Report statistical uncertainty.**  
   Add confidence intervals, variance across multiple runs, and significance tests where appropriate. Small-sample claims should be clearly marked as preliminary.

6. **Clarify compression measurement.**  
   Define precisely how compression ratio is computed, whether it is token-based or character-based, and how expansion affects the final budget. Report both pre- and post-expansion token counts.

7. **Quantify hallucination reduction directly.**  
   If hallucination is a central claim, measure it explicitly with a well-defined protocol and compare against baselines.

8. **Make reproducibility concrete.**  
   Document the embedding model, AST parser, call-graph construction, fusion weights, token-budget policy, and judge setup. Provide pseudo-code for the end-to-end pipeline.

9. **Demonstrate cross-language or cross-project transfer.**  
   Since the paper claims language-agnostic potential, include experiments beyond Python or at least a transfer setting on unseen repositories.

10. **Tone down over-strong claims unless supported.**  
   The manuscript should avoid broad statements like “significantly improving” or “reducing hallucinations” unless backed by robust comparative evidence. A more cautious framing would better match the current results.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add strong retrieval/compression baselines beyond IR-only, AST-only, and truncation: e.g., line-level/statement-level pruning, tree-sitter chunking without hybrid retrieval, and at least one recent code-context compression or repo-retrieval method. Without these, the claim that HASTE is a meaningful advance over prior context-selection methods is not convincing at ICLR’s bar.

2. Evaluate on substantially more than six curated Python files; add larger, public benchmarks with diverse repositories and languages, not just SWE-PolyBench and a tiny hand-picked set. The current evidence is too narrow to support claims of generality, robustness, or practical utility.

3. Include an ablation study separating the contribution of each component: BM25, dense retrieval, RRF fusion, call-graph expansion, AST-bounded pruning, and identifier extraction. Right now it is impossible to tell which part of HASTE drives performance, so the core architectural claim is under-supported.

4. Compare against an end-to-end “retrieve more context, let the LLM sort it out” baseline under the same token budget, plus oracle-context upper bounds. ICLR reviewers will want to know whether HASTE actually improves over simpler budget allocation strategies and how far it is from the best possible context selection.

5. Report statistics over enough tasks and repetitions to establish significance and variance, not just averaged judge scores on a few instances. With the current sample size, the reported compression-quality tradeoff could easily be an artifact of task selection.

### Deeper Analysis Needed (top 3-5 only)
1. Analyze failure cases by categorizing when HASTE breaks: missing dependency, wrong retrieval, over-expansion, or AST-pruning errors. Without this, the paper’s claim that it reduces hallucination is not credible because the mechanism of failure is unknown.

2. Validate the LLM-as-judge metric against objective outcomes such as test pass rate, patch correctness, compilation success, and exact edit localization. A judge score alone is too weak for code-edit claims, especially when the benchmark includes ambiguous or no-op tasks.

3. Quantify structural fidelity directly with compilation/parsing rates and syntactic validity before and after context compression. The paper claims AST coherence and executable context, but it does not show whether selected snippets are actually parseable or compilable at a meaningful rate.

4. Provide a real compression-quality frontier across multiple budget settings, not just a few examples and correlations from six files. The current evidence does not establish where HASTE sits relative to alternatives under tight budgets, which is the central claim.

5. Analyze whether gains come from better context selection or simply from easier tasks / trivial edits in the benchmark. The SWE-PolyBench discussion suggests many high scores are on no-op or low-complexity instances, which weakens the argument that HASTE solves hard software engineering cases.

### Visualizations & Case Studies
1. Show side-by-side examples of raw code, retrieved context from each baseline, and HASTE’s final context for both success and failure cases. This would reveal whether HASTE באמת preserves the right dependencies or merely returns cleaner-looking snippets.

2. Add a failure-case visualization on difficult edits that require cross-function or cross-file dependencies, showing where the AST-guided expansion misses critical code. This is necessary to test the central structural-completeness claim.

3. Plot task success versus token budget across multiple methods, with confidence intervals. A single correlation plot is not enough; reviewers need to see whether HASTE dominates alternatives at realistic compression levels.

4. Show retrieval heatmaps or AST trees highlighting which nodes were kept, expanded, or pruned for representative tasks. This would make it clear whether the method is structurally principled or just heuristic chunk selection.

### Obvious Next Steps
1. Extend the method and evaluation to cross-file and repository-level edits, since many real SWE tasks require dependencies outside one file. The paper itself admits this is the next step; without it, the scope remains too limited for the claimed impact.

2. Release a fully reproducible benchmark suite with task generation procedure, exact prompts, judge rubric, and code-edit verification scripts. For ICLR, reproducibility is not optional when the main evidence depends on an LLM judge.

3. Add a compile-and-test closed-loop evaluation where generated patches are applied and validated against repository tests. That is the most direct way to show the method improves real code editing, not just subjective judge scores.

4. Test on more languages and parser ecosystems than Python/Tree-sitter assumptions, or explicitly constrain the claim to Python. The current framing is broader than the evidence supports.

# Final Consolidated Review
## Summary
This paper proposes HASTE, a hybrid code-context retrieval and compression pipeline that combines BM25-style lexical retrieval, dense semantic retrieval, AST-guided chunking/pruning, and call-graph expansion to produce token-bounded context for LLM-based code editing. The motivation is important: under tight context windows, the paper aims to preserve both semantic relevance and syntactic coherence, rather than sacrificing one for the other.

## Strengths
- The paper addresses a real and practically important problem for code LLMs: context selection under tight token budgets, where naive truncation or relevance-only retrieval can easily produce unusable prompts.
- The high-level design is sensible and well-motivated: hybrid lexical/semantic retrieval plus AST-aware structuring and call-graph expansion is a plausible way to improve code-context quality, and the paper clearly explains the intended trade-off between relevance and structural integrity.
- The evaluation does include a real software engineering benchmark (SWE-PolyBench) in addition to curated examples, which is better than a purely toy demonstration and suggests the authors are trying to test the method in more realistic settings.

## Weaknesses
- The empirical evidence is far too limited to support the paper’s broad claims. The main curated study uses only six hand-picked Python files, and the SWE-PolyBench discussion is selective, excludes processing errors, and does not provide enough detail to judge representativeness. This makes the reported “up to 85% compression” and high-quality edit claims look preliminary rather than convincing.
- The evaluation is heavily dependent on an under-specified LLM-as-judge protocol. The paper does not give the judge prompt, calibration, blinding, inter-rater reliability, or validation against objective outcomes such as compilation success or test pass rate. For code-editing claims, this is a major weakness because judge scores can easily overstate actual correctness.
- Baselines and ablations are insufficient. Comparing against IR-only, AST-only, and naive truncation is not enough for a hybrid retrieval/compression method; the paper does not isolate the contribution of BM25, dense retrieval, RRF, AST expansion, identifier extraction, or call-graph traversal. Without this, it is impossible to tell which component actually drives the results.
- Several of the paper’s strongest claims are overstated relative to the evidence. In particular, claims about “significantly improving” edit success and “reducing hallucinations” are not convincingly established, since hallucination is not rigorously operationalized and the results do not include robust comparative analysis or uncertainty estimates.

## Nice-to-Haves
- A clearer, more formal description of the algorithmic pipeline, including exact embedding model, token-budget enforcement, call-graph construction, and AST-fidelity computation, would improve reproducibility.
- More direct reporting of compression-quality trade-offs across multiple budget settings, with confidence intervals, would make the results easier to interpret.
- A small set of qualitative examples showing retrieved context from HASTE versus baselines, including failure cases, would help readers understand when the method succeeds or breaks.

## Novel Insights
The main conceptual insight is not a new primitive, but a useful synthesis: for code, “good retrieval” is not just about semantic relevance, and “good pruning” is not just about keeping syntax intact. HASTE’s core idea is that these two constraints should be coupled through AST-aware selection and budgeted expansion, so the model sees snippets that are both topically relevant and structurally coherent. That is a plausible and timely direction, but the paper currently shows more of a system integration story than a clearly demonstrated methodological advance.

## Potentially Missed Related Work
- Code repository context retrieval / repo-level code RAG methods — relevant because they often already combine lexical and semantic signals for code search and would be strong baselines.
- Tree-sitter-based code chunking and structure-preserving code retrieval methods — relevant to the AST-guided extraction aspect.
- Recent context compression or pruning work for code LLMs — relevant because the paper’s central claim is about token-bounded extraction, and such methods should be compared directly.
- Repository-level edit or patch generation benchmarks beyond SWE-PolyBench — relevant for evaluating whether the method generalizes beyond the narrow curated set.

## Suggestions
- Add a substantially larger evaluation on public repositories, and report results with variance or confidence intervals.
- Include ablations for each major component: lexical retrieval, dense retrieval, fusion, AST pruning, call-graph expansion, and identifier extraction.
- Replace or supplement LLM-as-judge with objective code-edit outcomes such as compile success, unit-test pass rate, and patch correctness on a subset.
- Strengthen baselines with at least one recent code-context retrieval/compression method and one stronger repository-context baseline.
- Tone down the strongest claims unless they are backed by more rigorous comparative evidence.

# Actual Human Scores
Individual reviewer scores: [2.0, 0.0]
Average score: 1.0
Binary outcome: Reject
