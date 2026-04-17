Now I have enough information from the reviews, the paper, and calibration papers to write the final review.

## Summary

MermaidFlow proposes using Mermaid, a declarative graph markup language, as an intermediate representation for multi-agent LLM workflows, replacing brittle Python/JSON-based representations. Workflows are modeled as typed, annotated directed graphs, and an evolutionary programming (EP) framework with constraint-preserving operators (crossover, mutation, insertion, deletion) searches this structured space. The paper claims that this design "guarantees static graph-level correctness" throughout generation, and demonstrates consistent improvements over 13 baselines on four reasoning benchmarks (GSM8K, MATH, HumanEval, MBPP).

## Strengths

- **Well-motivated and appealing core idea.** The insight that replacing implicit, code-bound workflow representations with a declarative, human-readable, structurally verifiable IR addresses a genuine failure mode is clearly articulated and supported by prior empirical findings (Cemri et al., 2025; Zhang et al., 2024a; 2025c) that identify workflow fragility as the primary source of multi-agent system failures.

- **Consistent empirical improvements across all benchmarks.** MermaidFlow achieves the best average score (80.75%) across all four benchmarks, exceeding the next-best (MaAS) by 1.40% on average and AFlow by 2.08–5.54%. These gains are consistent, even if margins on some individual benchmarks are modest.

- **Demonstrated search efficiency.** The MATH ablation shows MermaidFlow reaches ~52% solve rate using ~2.7e4 tokens vs. AFlow's ~6.9e4 tokens, with >90% valid code generation rate vs. AFlow's ~50%. This practical efficiency benefit is tangible and well-illustrated.

- **Clean operator design grounded in the representation.** The five EP operators (node substitution, addition, deletion, edge rewiring, subgraph mutation/crossover) are well-scoped and leverage the typed graph structure. The crossover case study (Figure 4) effectively illustrates how compositional search over declarative structures works in practice.

## Weaknesses

### Major:

- **The "guaranteed static correctness" claim is overclaimed and contradicted by the paper's own description.** The paper repeatedly asserts that the system "guarantees" validity and that "every candidate is valid by construction" (Abstract; §1; §3.2; §4.1). However, Section 4.1 explicitly states: "when using an LLM to generate a new Mermaid graph, the resulting Mermaid code may sometimes violate predefined safety constraints. To address this, we implement a checker to verify whether the newly generated candidates conform to the defined workflow and operation rules. If any violations are detected, new workflows are regenerated." This is rejection sampling, not construction-time guarantee. The "guarantee" is enforced post-hoc, not by construction. Lemma 1 is essentially tautological: it states that constraint-preserving operators produce valid outputs, but the operators are *defined* as constraint-preserving, and the paper provides no formal proof or empirical verification that the LLM-implemented operators actually satisfy the stated preconditions (e.g., type compatibility in crossover, subgraph mutation input/output matching). This gap between the formal framing and actual implementation materially weakens the paper's central conceptual contribution—the paper has a strong heuristic filter on invalid plans, not a system with provable correctness guarantees. This should be reframed honestly.

- **Key ablation missing: representation vs. search algorithm.** The experiments compare MermaidFlow (Mermaid + EP) against AFlow (Python + MCTS) and ADAS (Python + heuristic search). Since both the representation language and the search algorithm differ simultaneously, the observed improvements cannot be causally attributed to the Mermaid representation alone. The paper lacks a "Mermaid-with-MCTS" or "Python-with-EP" variant. Without this controlled ablation, the evidence supports "this system performs well" but not the stronger claim that "the declarative, verifiable graph representation is the core driver of improvement."

- **LLM-as-Judge selection is critical but opaque and unvalidated.** The EP loop relies on an LLM-as-judge to score and select candidate workflows (§4.2), but the judging prompt, scoring criteria, and—crucially—calibration against actual task performance are not provided in the main text. If judge scores correlate poorly with true solve rates, the evolutionary search may be effectively random exploration, and the performance gains could be incidental. There is no analysis of judge-reward alignment, no comparison to simpler selection strategies (e.g., random sampling with execution-based validation), and no robustness analysis.

### Minor:

- **Modest absolute margins without statistical testing.** Results are averaged over only 3 runs with no standard deviations or confidence intervals reported. Over AFlow, the margins are 2.28% (GSM8K), 2.61% (MATH), 2.79% (HumanEval), and 0.64% (MBPP). At these scales, statistical significance is unclear.

- **"Safety" framing is misleading.** The title and abstract use "safety-constrained," which in the broader AI/ML context implies preventing harmful actions or ensuring operational safety. The actual constraints enforced are structural (type compatibility, connectivity, role consistency)—more accurately described as "structural validity" or "type-safe" rather than "safety."

- **DAG expressiveness limitation.** The workflow graphs appear to be DAGs, which cannot natively represent conditional branching (decision points), loops (repeated execution), or other richer workflow patterns. The paper does not discuss this limitation or how it might be addressed. Current benchmarks (math reasoning, code generation) may not surface this, but real-world multi-agent scenarios often require conditional or iterative control flow.

- **Regeneration overhead is unquantified.** The checker-based regeneration loop is mentioned but not analyzed: how often does regeneration occur, how many attempts are needed on average, and what is the computational overhead?

### Trivial:

- **"First" novelty claims are overstated.** The paper claims to be "the first agentic workflow framework to guarantee static graph-level correctness" and "the first workflow optimization framework built atop a statically verifiable workflow representation." Given prior DSL/graph-based approaches (MetaGPT's SOPs, MAS-GPT's DSLs, GPTSwarm's agent graphs), these "first" claims are likely too strong and should be softened.

## Nice-to-Haves

- **Controlled ablation isolating representation from search.** Running EP on Python-based workflows or MCTS on Mermaid-based workflows would directly test whether the declarative representation or the search strategy drives improvements.

- **LLM-as-Judge calibration analysis.** Reporting rank correlation between judge scores and actual task performance would validate the core selection mechanism.

- **Error bars and significance tests.** With margins of 0.64–2.79%, reporting standard deviations across runs and conducting paired significance tests would strengthen the empirical claims.

- **More diverse benchmarks.** Evaluation on tasks requiring richer workflow structures (conditional branching, iterative refinement) would test the framework's generalizability beyond sequential/ensemble patterns.

## Removed Points

- **"Cannot be independently verified" / reproducibility concerns about baselines or models.** The paper cites AFlow, MaAS, ADAS, etc., and uses gpt-4o-mini. Per the hard rules, we treat all cited tools and models as existing and available.

- **Missing hyperparameter details / implementation reproducibility nitpicks.** The harsh reviewer's complaint about undisclosed α and λ values, prompt details, etc., falls under the rule against nitpicking about unspecified implementation details. The paper provides the key algorithmic design; full prompt texts and hyperparameters are appropriate for appendices or code releases.

- **Demand for theoretical proofs.** The harsh reviewer's request for formal type system soundness proofs goes beyond what is standard for an empirical systems paper in this venue. The formalization, while shallow, is appropriate for the paper's scope.

- **ScoreFlow baseline absence.** The neutral reviewer and spark both suggest including ScoreFlow. However, ScoreFlow uses learned/gradient-based optimization, which is a fundamentally different paradigm from evolutionary search. The paper already compares against 13 baselines spanning multiple paradigms. Adding one more is a nice-to-have, not a core flaw.

- **Concerns about unfair comparison favoring the authors' method.** The harsh reviewer raises this, but the asymmetry actually works against MermaidFlow in some cases (e.g., MaAS uses trainable modules, ADAS gets 30 iterations vs. MermaidFlow's 20). Per the hard rules, this type of concern should not be flagged if the asymmetry favors the baseline.

## Novel Insights

The paper reveals an important practical insight: representing agent workflows in a declarative graph language (rather than generating Python code directly) dramatically improves the *search efficiency* of workflow optimization—over 90% of generated Mermaid candidates produce valid executable code vs. ~50% for Python-level search. This structural reliability benefit, rather than formal "guarantees," may be the more honest and practically valuable contribution. The gap between the paper's formal framing (closed subspaces, Lemma 1) and the actual mechanism (LLM generation + rejection sampling) suggests the community would benefit from clearer terminology distinguishing between "design-level invariance" (operators are *designed* to preserve constraints) and "system-level guarantee" (the system *ensures* no invalid workflow is ever output).

## Suggestions

1. **Reframe the "guarantee" claims honestly.** Replace "guarantees" and "valid by construction" with language like "designed to preserve structural validity" and "enforces validity through a combination of constraint-aware operators and a post-generation checker that rejects invalid candidates." This is still a meaningful contribution but does not overstate it.

2. **Add a representation-isolated ablation.** The single most impactful experiment would be running the EP search framework on Python-based workflows (keeping all other settings the same), which would directly answer whether the Mermaid representation itself drives the improvements.

3. **Validate the LLM-as-Judge.** Report correlation (e.g., Spearman rank) between judge scores and task-based validation scores on a held-out subset. Even a small-scale calibration analysis would significantly strengthen the claim that the EP loop is performing directed search rather than random exploration.

## Score and Decision

**Calibration context:** AFlow (accepted as Oral, avg. ~7.5) established the Python-code-based workflow search paradigm with comprehensive experiments across 6 benchmarks and detailed cost analysis; ADAS (accepted as Poster, avg. ~6) had a simpler code-generation search with strong transfer results; Flow (accepted as Poster, avg. ~6.25) proposed AOV-graph-based workflow refinement with similar formalization gaps. MermaidFlow is closer to Flow in offering a graph-structured workflow representation with claimed formal properties that are not fully substantiated. Its empirical results are solid but modest-margin, and the representation-vs-search confound is a notable gap. On the higher end, AFlow's comprehensiveness (6 benchmarks, cost analysis, ablations) sets a bar this paper doesn't fully meet.

MermaidFlow has a genuinely good idea executed well enough to produce consistent improvements, but its core claim (static correctness guarantees) is overclaimed in a way that, once corrected, reduces the novelty framing. The paper would be significantly stronger with honest reframing of the guarantees and the key representation ablation. I place this below AFlow (7.5) and around Flow/ADAS (6.0) territory, but the overclaiming of guarantees pulls it slightly below what the empirical results alone would justify.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>