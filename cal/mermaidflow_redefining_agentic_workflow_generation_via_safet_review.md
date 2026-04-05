=== CALIBRATION EXAMPLE 62 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title:** The title accurately reflects the contribution: using Mermaid for workflow representation and evolutionary search for generation. It is specific and informative.
- **Abstract:** Clearly states the problem (fragile, unexecutable plans from unconstrained LLM generation), the method (Mermaid as verifiable IR, safety-constrained evolutionary operators), and results (consistent improvements, faster convergence).
- **Unsupported Claims:** The abstract claims MermaidFlow "guarantee[s] static graph-level correctness." This is a very strong claim. Later sections clarify that the LLM generation can violate constraints and requires regeneration via a checker. The "guarantee" applies to the *search space* and *post-filtered* candidates, not the generation process itself. This phrasing risks overstating the robustness. The authors should temper the "guarantee" language to reflect the generation-filter loop.

### Introduction & Motivation
- **Motivation & Gap:** Well-motivated. The critique of code-centric workflows (e.g., AFlow, ADAS) as brittle, implicit, and hard to verify is accurate and aligns with recent findings on multi-agent failures (Cemri et al., 2025). The proposal of a declarative, statically verifiable intermediate representation (IR) directly addresses this gap.
- **Contributions:** Clearly stated: (1) Mermaid declarative representation, (2) EP-based search framework, (3) Empirical performance gains. These are accurate to the paper's content.
- **Over-claiming:** The introduction claims this is "the first agentic workflow framework to guarantee static graph-level correctness across the entire generation process." As noted in the Abstract assessment, this is misleading. The framework relies on an LLM that must *attempt* to follow constraints, followed by a rejection-sampling checker. It does not guarantee that every LLM call produces a valid graph, only that the pipeline eventually yields one. Furthermore, GPTSwarm and similar graph-based methods also aim for structural validity; the authors should nuance the "first" claim to emphasize the specific safety-constrained *evolutionary* aspect rather than absolute graph correctness.

### Method / Approach
- **Clarity & Reproducibility:** The method is described with formulas and algorithms (Algorithms 1-3), and detailed prompts are provided in Appendix A.3. This aids reproducibility. However, there is a conceptual ambiguity between the **formal operators** and the **implementation**.
    - Section 4.1 defines atomic operators (Node Substitution, Addition, Crossover, etc.) as graph transformations. Lemma 1 claims closure under these operators.
    - Section 4.2 and Appendix A.3.1 reveal that the system actually prompts an LLM to generate a new graph *describing* modifications based on parent graphs, and then uses a checker to validate. The system does not appear to programmatically apply graph operators to data structures; rather, it relies on the LLM to simulate these operators in its output.
    - *Critical Question:* If the "operators" are merely instructions in a prompt rather than algorithmic manipulations of the graph object, the "Evolutionary Programming" framing is partially metaphorical. The authors should clarify whether any programmatic graph manipulation occurs, or if this is effectively "LLM-guided refinement with structural constraints."
- **Assumptions & Logical Gaps:**
    - **Lemma 1:** This lemma and its proof are trivial. Lemma 1 states that if $G \in S$ and $O$ is defined to preserve constraints, then $O(G) \in S$. This is true by definition of the operators in Section 4.1 and does not require a proof by induction. It adds mathematical weight without substance. The authors could remove the lemma and simply state that operators are designed to preserve validity.
    - **LLM-as-Judge Bias:** The method uses an LLM-as-judge to select candidates to avoid expensive rollouts during search. However, the final results depend on actual execution. If the Judge's scoring correlates poorly with execution success, the evolutionary search could drift toward "persuasive" but non-functional workflows. The paper does not analyze the correlation between the Judge's scores and actual pass rates.
- **Edge Cases/Failure Modes:** The method handles invalid Mermaid via regeneration (checker), but what about valid Mermaid that translates to buggy Python? This is addressed implicitly via the validation score, but the Mermaid-to-Python translation is LLM-dependent (Algorithm 1, line 8). The translation step is a known failure point for code generation, and its failure modes are not integrated into the search feedback loop (e.g., if translation fails, does the system feedback to the graph evolution?).

### Experiments & Results
- **Testing Claims:** The experiments compare against strong baselines including AFlow and MaAS, which are appropriate. The tasks (GSM8K, MATH, HumanEval, MBPP) are standard for this domain.
- **Baselines:** Fairly compared. The authors constrained baselines like AFlow to the same LLM (`gpt-4o-mini`) and iteration count, which is good practice.
- **Missing Ablations:**
    - **Representation vs. Search:** The paper attributes gains to the *Mermaid representation* and the *EP search*. A critical missing ablation is "MermaidFlow without EP" (i.e., single-shot or iterative refinement without population crossover) to quantify the benefit of the evolutionary component. Conversely, comparing EP on a different structured representation (e.g., JSON trees) against Mermaid would isolate the representational benefit. While the comparison with AFlow helps, AFlow uses MCTS, not EP, confounding the variables.
    - **Seed Variance:** Table 1 notes results are "averaged over three runs." For LLM reasoning benchmarks, three runs is often insufficient to establish statistical significance, especially given the high stochasticity of the optimization loop. Error bars or standard deviations should be reported.
- **Cherry-picking & Metrics:** The results appear consistent across domains. The token efficiency claim (Section 5.3, "MermaidFlow requiring only about half the cost of AFlow") is a valuable practical insight.
- **Truncated Content:** Section 5.3 ("Optimal Stopping Point Analysis") appears incomplete in the provided text, cutting off after "We use the round index of optimal stopping points to demonstrate this." This prevents full evaluation of that specific analysis. The authors must ensure the camera-ready version includes this missing discussion/figure.

### Writing & Clarity
- **Confusing Sections:**
    - The distinction between the **Mermaid Graph** and the **Python Code** is sometimes blurred. For example, Figure 1 caption mentions "workflow lifecycle," but the relationship between the Mermaid node types and the Python operator classes is primarily explained in the Appendix. The main text could better explain how the typed Mermaid graph maps to the Python execution engine.
    - The "Optimal Stopping Point Analysis" subsection is disjointed due to missing text/figures, disrupting the flow of the ablation study.
- **Figures & Tables:** Table 1 is clear and well-formatted. Figure references in the text (e.g., "Figure 5.3") seem to have numbering errors likely due to parser or drafting issues, which should be corrected.
- **Formalism:** The use of Lemma 1, Definition 1, and set notation for the search space is formal but ultimately superficial given the reliance on LLM generation. The prose would be clearer if it focused on the algorithmic workflow (Prompt -> Generate -> Check -> Evolve) rather than over-formalizing simple constraint properties.

### Limitations & Broader Impact
- **Acknowledged Limitations:** The authors acknowledge in Section E that the current Mermaid node types lack control flow constructs (if/for-loops) and that the Mermaid-to-Python translation relies on an LLM, which can fail.
- **Missed Limitations:**
    - **Scalability of LLM-as-Judge:** As workflows grow in complexity (more nodes, more edges), the cognitive load on the LLM-as-judge to evaluate "workflow coherence" and "complexity balance" increases. The paper does not discuss whether the Judge's reliability degrades with graph size.
    - **Domain Specificity of Node Types:** The Mermaid node types (e.g., `Programmer`, `ScEnsemble`) appear tailored to the current benchmarks (Math/Code). It is unclear how easily this type system generalizes to domains requiring tool use, web browsing, or database queries without redesigning the type schema and prompts. The "task-agnostic" claim in the Introduction may be overstated regarding the *schema* definition.
- **Negative Societal Impacts:** Standard risks of automated agentic workflows apply (e.g., generation of malicious code if safety filters are bypassed). Since MermaidFlow optimizes for task success, there is a risk it could evolve workflows that exploit benchmark quirks or use unsafe tools. This is not discussed.

### Overall Assessment
MermaidFlow presents a compelling and timely contribution to the agentic workflow research agenda. The idea of replacing brittle code generation with a declarative, statically verifiable graph representation (Mermaid) addresses a real bottleneck in automated agent design. The empirical results are promising, showing that this representation, combined with an evolutionary search, can outperform code-based methods like AFlow and MaAS while being more token-efficient.

However, the paper has notable weaknesses that must be addressed. First, the formalization of "Evolutionary Operators" (Lemma 1) is trivial and disconnected from the implementation, which relies on LLM prompting and rejection sampling rather than algorithmic graph manipulation. Second, the ablation studies are insufficient to disentangle the benefits of the Mermaid representation from the benefits of the population-based search strategy. Third, the reliance on only three experimental seeds without error bars limits the statistical confidence in the reported gains. Finally, there are minor presentation issues, including a truncated section in the ablation study and over-claimed "guarantees" of correctness that don't fully align with the generation-filter mechanism.

With clarifications on the implementation vs. formalism, additional ablations to isolate the representation's impact, more robust statistical reporting, and a discussion of the LLM-as-judge reliability, this paper would be a strong addition to the ICLR proceedings. The core insight—that a constrained, structured search space improves workflow optimization—is valuable and likely to influence subsequent work in reliable agentic systems.

# Neutral Reviewer
## Balanced Review

### Summary
The paper introduces MermaidFlow, a framework that represents agentic workflows as declarative, statically verifiable Mermaid graphs and optimizes them via safety-constrained evolutionary programming. By enforcing type compatibility and structural constraints during graph mutation and crossover, the method guarantees valid workflow candidates throughout search, which are subsequently translated to executable Python code. Empirically, MermaidFlow achieves consistent improvements in success rates, higher code generation validity, and better token efficiency compared to strong code-based workflow optimization baselines across math and programming benchmarks.

### Strengths
1. **Well-Motivated Problem & Declarative Design**: The paper correctly identifies the brittleness of implicit, code-bound workflow generation in multi-agent systems. Using Mermaid as an intermediate representation cleanly separates planning from execution, enabling human-readable visualization and static verification (Sec 3.1, Fig 1).
2. **Constraint-Preserving Search Operators**: The evolutionary operators (node addition/deletion, edge rewiring, subgraph mutation, crossover) are carefully designed to maintain type signatures and role consistency. The closure argument (Lemma 1) effectively shows that the search remains within the valid subspace, directly addressing the high failure rates of unconstrained LLM code generation (Sec 4.1, Sec 5.3).
3. **Strong Empirical Validation & Practical Engineering**: The framework outperforms 13 baselines across GSM8K, MATH, HumanEval, and MBPP, with consistent average gains (Table 1). The appendices provide substantial reproducibility value, including detailed checker logic (regex soft checks + Mermaid CLI hard checks), complete algorithm pseudocode, full prompt templates, and concrete Mermaid/Python case studies (Appendix A, B).
4. **Demonstrated Robustness & Efficiency Gains**: The authors show a >90% success rate in generating executable code versus ~50% for AFlow, alongside roughly half the token consumption to reach comparable performance (Sec 5.3). These metrics directly support the claim of improved search reliability and cost-effectiveness.

### Weaknesses
1. **Modest Absolute Performance Gains**: Improvements over strong baselines (AFlow, MaAS) range from ~1% to 2.6% absolute. While robust, these margins are incremental, and the paper does not report statistical significance tests or confidence intervals to confirm they exceed experimental noise (Table 1).
2. **LLM-Dependent Translation Reintroduces Brittleness**: The method guarantees static validity at the Mermaid graph level, but relies on an LLM to translate Mermaid back to Python. As acknowledged in Appendix E, this step can still produce runtime errors, partially undermining the "safety-constrained" guarantee at execution time.
3. **Under-Specified Evolutionary Dynamics & Compute Overhead**: The paper reports token efficiency but omits critical search dynamics: population size management, diversity metrics, stagnation/early-stopping criteria, and wall-clock/API latency per iteration. The LLM-as-Judge scoring rubric is also heuristic (Sec 4.2) without ablation on judge reliability or potential reward hacking.
4. **Lemma 1 Presented Without Formal Proof**: The closure invariant is stated as a lemma but functions more as a design property. A rigorous structural induction proof or a reduction to existing type-system guarantees would strengthen the theoretical contribution expected at ICLR.
5. **Limited Task Scope**: Evaluation is restricted to closed-domain math and code synthesis tasks under gpt-4o-mini. The framework's scalability to long-horizon planning, tool-use environments, or open-domain multi-agent tasks remains untested, limiting claims of broad generalization beyond narrow benchmarks.

### Novelty & Significance
**Novelty**: Moderate-High. Repurposing Mermaid.js as a typed, verifiable DSL for agentic workflows and coupling it with constraint-aware evolutionary programming is a fresh and pragmatic systems contribution. The novelty lies not in algorithmic breakthroughs, but in the thoughtful intersection of declarative graph representation, static verification, and black-box evolutionary search for LLM agents.
**Significance**: High practical relevance. Workflow brittleness is a widely recognized bottleneck in multi-agent LLM systems. MermaidFlow provides a principled, modular, and interpretable pathway that could become a standard design pattern for reliable agent orchestration. While absolute gains are incremental, the substantial improvements in validity rates, token efficiency, and interpretability align well with ICLR's emphasis on robust, verifiable, and scalable AI systems. For acceptance, the paper would benefit from tighter statistical validation and a more rigorous treatment of the translation bottleneck.

### Suggestions for Improvement
1. **Add Statistical Rigor**: Report standard deviations across runs and apply paired statistical tests (e.g., bootstrap or t-tests) to confirm that performance gains over baselines are significant and not within noise margins.
2. **Quantify Translation & Judge Reliability**: Provide empirical error rates for the Mermaid-to-Python translation step and conduct an ablation comparing the LLM-as-Judge against rule-based or random selection. Discuss mitigation strategies for potential judge bias.
3. **Detail Search Dynamics & Compute**: Include convergence curves with wall-clock time, API call counts, and population diversity metrics. Clarify how the history buffer is capped, how stagnation is handled, and the exact cost-benefit trade-off per evolution step.
4. **Strengthen Theoretical Claims**: Either provide a formal structural induction proof for Lemma 1 or reframe it as a "Design Guarantee" with explicit type-system references, aligning the terminology with the empirical nature of the contribution.
5. **Explore Broader Generalization**: Extend evaluation to at least one long-horizon or tool-use benchmark (e.g., AgentBench, WebArena) to demonstrate that the static graph constraints scale to dynamic, stateful environments beyond isolated math/code problems.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3 only)
1. Isolate the representation from the search algorithm by running identical EP operators and LLM-judge selection over a Python-code search space. Without this, the claimed >90% valid-code-generation rate could be driven by the selection mechanism or prompt templates rather than the Mermaid representation.
2. Report performance under strictly normalized compute budgets (total LLM API calls, token consumption, and wall-clock time). Comparing 20 rounds for MermaidFlow against 30 for ADAS, while ignoring judge overhead, invalidates the superior efficiency claim.
3. Conduct a factorial ablation of the four EP operators (crossover, mutation, insertion, deletion) by zeroing out their sampling probabilities one at a time. The assertion that domain-aware crossover and mutation drive search lacks empirical backing without isolating each operator's marginal contribution.

### Deeper Analysis Needed (top 3 only)
1. Quantify the correlation between LLM-as-Judge scores and actual execution/validation success rates across datasets. If the judge correlates poorly with ground truth, the framework is optimizing for superficial structural heuristics rather than functional correctness, directly undermining the "faster convergence" claim.
2. Audit the end-to-end pipeline failure rate introduced by the LLM-based Mermaid-to-Python translation. The paper claims "static correctness guarantees," but Appendix C highlights imperative translation bugs; without quantifying how often valid graphs yield broken Python, the safety guarantee is incomplete.
3. Analyze the relationship between workflow complexity (node count, edge density, ensemble size) and final solve accuracy. Without this, improvements on MATH/HumanEval could stem from trivial prompt inflation or redundant ensembling rather than meaningful structural evolution.

### Visualizations & Case Studies
1. Provide an execution trace of a workflow that passes all static Mermaid checks (W1–W5, CLI) but fails during translation or runtime. Visualizing the exact structural-to-imperative mismatch exposes the real-world validity gap.
2. Plot the evolutionary trajectory of a single high-performing workflow over 5-10 generations, annotating the exact operator applied at each step alongside the resulting validation score. This proves the EP framework performs meaningful structural exploration rather than random prompt tweaking.
3. Generate a scatter plot comparing LLM Judge predicted scores against ground-truth execution scores across all candidates. Visual outliers or systematic over/under-prediction would immediately reveal judge bias as a confounding factor in the selection process.

### Obvious Next Steps
1. Replace the LLM-based Mermaid-to-Python translator with a deterministic, compiler-backed AST/code generator. Leaving translation to an LLM reintroduces the exact imperative brittleness the paper criticizes and leaves the "guaranteed static safety" claim fundamentally incomplete.
2. Implement and benchmark basic control-flow primitives (conditional branching, iterative loops) using Mermaid's supported node types or extensions. Deferring dynamic routing to future work severely limits the framework's applicability to non-trivial agentic reasoning tasks.
3. Report a performance-vs-cost Pareto frontier (success rate vs. tokens/API calls) against baselines. ICLR requires explicit efficiency trade-off curves; claiming "roughly half the cost" without a formalized budget analysis is insufficient for an optimization paper.

# Final Consolidated Review
## Summary
MermaidFlow introduces a framework that represents agentic workflows as declarative, statically verifiable graphs using the Mermaid DSL, and optimizes them via safety-constrained evolutionary programming. By enforcing type compatibility, role consistency, and structural connectivity, the method guarantees valid workflow candidates throughout the search, which are subsequently translated to executable Python. The framework consistently outperforms strong code-based workflow optimization baselines across math and code generation benchmarks, demonstrating significantly higher generation validity rates and improved token efficiency.

## Strengths
- **Declarative, Statically Verifiable IR:** The substitution of brittle, implicit code generation with a typed Mermaid graph cleanly separates planning from execution. This enables pre-execution structural validation (type compatibility, connectivity) and directly reduces the high failure rates typical of unconstrained LLM-driven workflow search.
- **High Generation Validity & Token Efficiency:** The framework achieves a >90% success rate in producing executable code (compared to ~50% for Python-code baselines like AFlow) and reaches comparable performance with roughly half the token consumption. This demonstrates that constraining the search space to a compiler-verifiable substrate drastically improves sample quality and cost-effectiveness.
- **Comprehensive Empirical Validation & Reproducibility:** Outperforms 13 strong baselines across GSM8K, MATH, HumanEval, and MBPP. The paper provides exceptional reproducibility through detailed checker implementations (regex soft checks + Mermaid CLI hard checks), complete algorithmic pseudocode, full prompt templates, and end-to-end case studies.

## Weaknesses
- **Confounding of Representation and Search Benefits:** The performance gains are jointly attributed to the Mermaid representation and the evolutionary search strategy. However, the absence of a controlled ablation isolating these factors leaves this claim under-supported. Comparing MermaidFlow against AFlow conflates the graph IR with the optimization algorithm (MCTS vs. EP). Without evaluating EP over an alternative IR (e.g., JSON/Python AST) or comparing single-shot Mermaid generation against population-based evolution, it is unclear whether the gains stem from the declarative structure or the search dynamics.
- **Unverified LLM-as-Judge Correlation:** The optimization loop relies on an LLM-as-judge to score candidates and avoid expensive rollouts. The paper does not quantify the correlation between these heuristic judge scores and actual execution/validation success rates. If the judge correlates poorly with ground-truth performance, the evolutionary search risks optimizing for superficially structured or "persuasive" workflows that fail during runtime, directly undermining the convergence claims.
- **Translation Brittleness Contradicts End-to-End Safety Claims:** While the Mermaid search space guarantees static graph-level validity, the final pipeline depends on an LLM to translate valid graphs into executable Python. As noted in the appendices, this translation step remains prone to imperative logic errors (e.g., unreliable control flow, incorrect instance initialization). This reintroduces the exact runtime brittleness the method aims to solve, making the "guaranteed static safety" claim incomplete for the execution phase.
- **Insufficient Statistical Reporting:** Results are averaged over only three optimization runs without standard deviations, confidence intervals, or statistical significance testing. Given the stochastic nature of LLM-based optimization loops and the tight margins over strong baselines (~1–5% absolute), this reporting standard is inadequate to confirm that the improvements exceed experimental noise or LLM variance.

## Nice-to-Haves
- Reframe Lemma 1 explicitly as a "Design Guarantee" rather than a formal mathematical lemma, as the closure property is currently tautological by construction of the operators. A rigorous structural induction proof is unnecessary for this empirical systems contribution.
- Include a factorized ablation of the EP operators (crossover, mutation, insertion, deletion) to quantify the marginal contribution of each evolutionary mechanism to the overall search trajectory.
- Extend evaluation to at least one open-domain or tool-use benchmark to test whether the current node schema and type constraints generalize beyond tightly closed math/code tasks.

## Novel Insights
The paper highlights a fundamental tension in automated agentic workflow design: the generation flexibility required by LLMs directly conflicts with the structural guarantees needed for reliable multi-agent execution. By adopting a constrained, compiler-verifiable DSL, MermaidFlow demonstrates that restricting the optimization search space to a statically analyzable format not only drastically reduces syntactic dead-ends but also improves token efficiency. This suggests a broader principle for LLM-driven system design: imposing strict, verifiable intermediate representations as "guardrails" yields more productive optimization trajectories than unconstrained generation targeting free-form implementations, even when the final output must eventually be compiled to imperative code.

## Suggestions
- Report standard deviations across optimization seeds and apply appropriate statistical tests to verify that performance gains over baselines are significant.
- Quantify the correlation between LLM-as-Judge predicted scores and ground-truth execution pass rates. An ablation comparing the judge against random or rule-based selection would clarify its necessity and potential for selection bias.
- Audit the end-to-end failure rate introduced by the Mermaid-to-Python translation step. Discuss transitioning to a deterministic, template-based or AST-driven translator to close the safety gap between graph validation and runtime execution.
- Add a controlled ablation disentangling the representation from the search algorithm (e.g., running the same EP mechanics over a less constrained IR, or comparing MermaidFlow against single-pass prompt refinement).
- Clarify the missing text in the "Optimal Stopping Point Analysis" subsection and correct figure reference numbering in the camera-ready version.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0]
Average score: 5.3
Binary outcome: Reject
