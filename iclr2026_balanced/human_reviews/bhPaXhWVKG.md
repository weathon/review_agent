## Human Reviewer 1

### Summary
This paper proposes MermaidFlow, a framework for agentic workflow generation based on the Mermaid graph language. The core idea is to represent workflows as declarative typed graphs G(V[τ,α], E[ρ]) and optimize them through safety-constrained evolutionary programming (EP). The authors claim that Mermaid representation provides advantages such as static verifiability, human readability, and modularity. Experiments are conducted on GSM8K, MATH, HumanEval, and MBPP, showing an average improvement of 2.08% over AFlow with approximately 50% reduction in token cost.

### Strengths
1. **Clear formalization**: The graph representation of workflows, type system, and evolutionary operators are well-defined
2. **Static verification mechanism**: Two-layer checking (soft + hard) ensures syntactic correctness of generated workflows
3. **Comprehensive experiments**: Cover multiple domains including math reasoning and code generation, with comparisons against multiple baselines
4. **Token efficiency**: Approximately 50% reduction in token cost compared to AFlow
5. **Case study**: Figure 4 clearly demonstrates the workflow evolution process

### Weaknesses
**1. Limited Novelty**

- **Essentially a variant of known paradigms**: The method still follows a per-task iterative evolutionary search paradigm, introducing new representation and constraints on top of existing frameworks.
- **Questionable necessity of Mermaid**: The paper does not sufficiently justify why Mermaid is superior to Python. For LLMs, both are structured text, but LLMs have higher affinity for code representations. Moreover, current LLM-based workflow generation is far from the stage where "constraint benefits > exploration benefits", making the actual gains from safety constraints (efficiency? effectiveness?) unclear. Workflow failures primarily stem from early systems (like ADAS) building from scratch with inadequate controllable and supervised generation granularity, rather than inherent issues with Python representation itself.


**2. Experimental Design Issues**

- **Marginal performance gains**: Average improvements of 2.08% over AFlow and 1.40% over MaAS. Given that agentic workflows can achieve high upper bounds through automated structural design, the paper fails to demonstrate breakthrough on apparent bottlenecks (e.g., AFlow's restricted XML generation format causing significant performance degradation on certain tasks, and the inability to reuse good designs from non-selected nodes—the simplest example being that poor workflow structures may contain good prompts). Moreover, I noticed in the appendix that the evolved MermaidFlow ensemble uses 5-sample voting while AFlow uses 3-sample voting, which could be a major contributing factor to the performance improvement rather than the Mermaid representation itself.
- **Unfair cost comparison**: The reported token consumption of MermaidFlow does not clearly specify whether it includes the Mermaid→Python translation step. According to the experimental setup, each generated Mermaid workflow requires translation via gpt-4o-mini. This cost should be explicitly stated regarding inclusion, along with the specific cost breakdown for this component alone.


**3. Technical Detail Issues**

- **Unclear role of the Checker**: If Mermaid provides strong type constraints, the checker should rarely trigger, as LLMs have very low probability of generating syntactically invalid Python code. However, the checker is actually a critical component in MermaidFlow (Section A.2), suggesting that the constraints are not strong enough. A >90% success rate still means 10% of generations require retry, which remains quite high.
- **Incomplete operator definitions**: The paper claims "safety-constrained", but specific safety properties (such as deadlock-freedom, termination) are not defined. There is also a lack of case study analysis on how directly generating Python code workflows violates these safety properties.


**4. Inappropriate Baseline Categorization (Minor)**

- **Conceptually flawed taxonomy**: The authors categorize CoT, ComplexCoT, and Self-Consistency as "Single-agent execution methods" (Section 5.1, Table 1). This classification is deeply problematic. The term "agentic workflow" was introduced precisely to distinguish workflows with fixed resource scheduling from agents capable of autonomously allocating computational resources—with LLM calls being the most representative computational resource. CoT is a classic prompting strategy, while Self-Consistency is a classic (non-agentic) workflow. Neither involves autonomous agents that can dynamically decide when and how to invoke LLMs based on intermediate states or task requirements. By labeling these non-agent baselines as "single-agent methods," the authors conflate fundamentally different paradigms and misrepresent the conceptual boundaries of their contribution.

### Questions
1. Could you provide more concrete case studies, including but not limited to:
   - **Mermaid vs. Python comparison**: Show side-by-side examples where (a) Python-based workflow generation fails but Mermaid succeeds, and (b) demonstrate what specific safety properties are violated in the Python case (e.g., deadlock, non-termination, type mismatch).
   - **Safety constraint effectiveness**: Provide concrete examples showing how safety constraints improve efficiency or effectiveness during evolution. What specific invalid mutations are prevented? How much retry overhead is avoided?
   - **Checker analysis**: Given that the checker has >90% success rate (implying ~10% retry), provide examples of the 10% failed cases. What constraint violations occur? Why doesn't Mermaid's type system prevent these?
   - **Component reuse**: Demonstrate how MermaidFlow enables reusing good components (e.g., effective prompts) from suboptimal workflows, addressing the AFlow limitation you mentioned. Show concrete examples from your evolution process.

2. Could you provide a more detailed cost analysis?
   - Explicitly state whether the reported token consumption includes Mermaid→Python translation costs
   - Break down token costs by component: (a) workflow generation, (b) translation, (c) execution, (d) evaluation
   - Compare the per-iteration cost breakdown between MermaidFlow and AFlow

3. I would greatly appreciate if you could discuss the concerns I raised in the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper introduces MermaidFlow, a framework for generating agentic workflows using Mermaid as an intermediate representation. The core contribution is representing workflows as declarative, statically verifiable graphs instead of imperative code, combined with a safety-constrained evolutionary programming (EP) approach for workflow optimization. Also, MermaidFlow proposes a set of mutation operators that preserve senmantic correctness during search. Experiments on GSM8K, MATH, HumanEval, and MBPP show improvements over baselines including AFlow and ADAS, with the framework achieving higher success rates and better learning and token efficiency.

### Strengths
- **Novel Representation**: The use of Mermaid as a declarative intermediate representation is innovative and well-motivated. It cleanly separates planning from execution, addressing a key weakness in prior code-based methods.
- **Comprehensive System Design**: The paper thoroughly describes the type system, operators, validation mechanisms (soft and hard checks), and the complete pipeline from Mermaid to executable code.
- **Consistent Empirical Improvements**: The method shows improvements across all four benchmarks (average 80.75% vs. 79.35% for the best baseline).

### Weaknesses
- **Mermaid DSL Limitations**: Mermaid DSL is static by design and may be hard to express loops, conditionals or some runtime operations. The paper does not show how these limitations. Extending the DSL or documenting its expressive limits would strengthen the contribution.
- **Presentation Issues**: Figures 1 and 2 appear as raster images rather than vector graphics (like pdf or svg) and lose clarity when zoomed.
- **Execution Model Ablations**: The paper uses only gpt-4o-mini as the execution LLM. Performance with other models (e.g., Claude, Llama) is only explored for a brief ablation on optimization LLM.

### Questions
- **Mutation Operator Effectiveness**: Which evolutionary operators contribute most to performance improvements? The paper mentions crossover occurs with only 10% probability but provides no analysis of operator frequency, success rates, or relative contributions.
- **Iteration Scaling and Convergence**: Is 20 iterations sufficient for convergence, or would extended search yield further gains? The paper doesn't justify this choice or show whether performance plateaus, and provides no analysis of optimal stopping points across different tasks. Additionally, if there are more iterations, can different workflows be observed, and will their complexity increase with the iteration?

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces a unified framework combining multi-agent learning and evolutionary optimization. The proposed system models workflow automation as a constrained single-objective optimization with an evolutionary loop governed by variation, selection, and reflection. A meta-controller dynamically updates strategies and rules, ensuring adaptive improvement. Theoretical results guarantee convergence and monotonic rule-quality growth, while experiments demonstrate strong cross-domain generalization.

### Strengths
* Solid theoretical formalization.
* Innovative integration of evolutionary search and multi-agent optimization.
* Well-structured proofs and explicit assumptions.
* Broad empirical evaluation demonstrating clear advantages.
* Clear, consistent, and professional presentation.

### Weaknesses
* **Incomplete experimental reporting**: Missing per-benchmark hyperparameters and statistical variance.
  *Suggestion:* Add full configuration tables and multi-run results.
* **Unclear curriculum mechanism**: Difficulty-level staging lacks quantitative definitions.
  *Suggestion:* Provide formal thresholds and curriculum ablations.
* **Assumptions not empirically tested**: Theoretical premises like positive information gain remain unchecked.
  *Suggestion:* Visualize empirical distributions of related quantities.
* **Limited strong baselines**: Comparison with advanced workflow retrieval or graph-based frameworks is limited.
  *Suggestion:* Extend experiments to include such baselines.

### Questions
* How does the strategy distribution evolve across curriculum phases?
* How sensitive is convergence to the meta-controller update interval?

### Soundness
2

### Presentation
3

### Contribution
4

### Rating
6

### Confidence
3