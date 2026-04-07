=== CALIBRATION EXAMPLE 60 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is descriptive and reflects the core contribution: using safety-constrained evolutionary programming for workflow generation via a Mermaid-based representation. The abstract clearly states the problem (fragile, unexecutable plans), the proposed solution (MermaidFlow), and the claimed benefits (improved success rates, faster convergence). The claims are specific and appear to be supported by the experimental results later in the paper. The abstract appropriately sets expectations for a technical ICLR submission.

### Introduction & Motivation
The introduction effectively frames the problem of brittle, code-entangled agentic workflows and the need for verifiable, structured representations. The three-layer lifecycle model (planning, realization, execution) is a useful conceptual tool. The core argument—that implicit representations hinder verifiability and search—is well-motivated by citing recent studies on multi-agent failures (Cemri et al., 2025; Zhang et al., 2024a). The contributions are listed clearly at the end of the section.

**Critical Concern:** The claim that MermaidFlow is the "first agentic workflow framework to **guarantee** static graph-level correctness across the entire generation process" (Section 1) is very strong. The guarantee depends on the correctness of the Mermaid checker/compiler and the strict adherence of the LLM-based generator to the defined operators. The paper later notes that LLM-generated Mermaid code sometimes violates constraints (Section 4.1), requiring regeneration. This suggests the "guarantee" is enforced through a rejection-sampling loop, not a formal, unconditional guarantee. This claim should be tempered to reflect the actual, best-effort enforcement mechanism.

### Method / Approach
**3. A Novel Declarative Graph Representation:** The use of Mermaid as a declarative, typed graph representation is creative and leverages an existing, human-readable standard. The formal definitions of nodes, edges, types, and the search space \(S\) are clear. A key strength is the separation of planning (Mermaid graph) from execution (generated Python code).

**Critical Concern (Novelty/Substance):** While using Mermaid is novel in this context, the core conceptual advance—a typed, declarative intermediate representation for verification—is reminiscent of classical program synthesis and compiler design. The paper could more deeply articulate how this representation fundamentally enables new capabilities beyond what a custom-designed DSL or XML schema could provide. The benefits (human-readability, built-in renderer) are practical but not theoretically profound.

**4. Constraint-Aware Evolutionary Workflow Optimization:** The EP operators (node substitution, addition, edge rewiring, etc.) are well-defined and incorporate type-safety checks. Lemma 1 (Transformation Invariance) is stated but not proven; a proof sketch or argument based on the operator definitions would strengthen this section. The use of an LLM-as-Judge for selection is pragmatic but introduces a non-deterministic and potentially expensive component to the search.

**Critical Concerns (Reproducibility & Rigor):**
1.  **Implementation Heavy:** The method's success hinges on extensive, domain-specific prompt engineering (evidenced by the extremely long prompts in Appendix A.3). The "Mermaid checker" (Appendix A.2) is described as using regex and the Mermaid CLI, but its full logic (e.g., the exact regex patterns, the handling of all edge cases) is not provided. This creates a significant reproducibility barrier.
2.  **Operator Scope:** The operators are defined at a high level, but their instantiation via LLM prompts (e.g., how does an LLM perform "Subgraph Mutation" while preserving I/O types?) is left opaque. The gap between the formal operator definition and its prompt-based implementation is a potential source of error or inconsistency.
3.  **Search Process:** The parent sampling distribution \(P_{\text{mixed}}\) and the LLM-as-Judge scoring are critical to the evolutionary search's efficiency and quality. However, the paper provides no ablation studies on the importance of these components or their hyperparameters (\(\alpha\), \(\lambda\)). The choice of generating four candidates per round is also not justified.

### Experiments & Results
**5.2 Experimental Results:** The comparison against 13 baselines across four benchmarks is comprehensive. Using a consistent base LLM (gpt-4o-mini) is good practice. The results show consistent but often modest improvements (e.g., +2.61% on MATH over AFlow, +1.40% average over MaAS).

**Critical Concerns:**
1.  **Statistical Significance:** The paper reports averages over three runs but provides no measures of variance (standard deviation, confidence intervals) or statistical significance tests. For ICLR, it is essential to demonstrate that the reported improvements are statistically significant and not due to random variation, especially given the modest margins.
2.  **Fairness of Comparison:** The baseline **MaAS** incorporates a trainable module, while MermaidFlow is purely a search/prompting framework. This is noted, but the implications are not discussed. Is MermaidFlow's improvement achieved without any training a key advantage, or does it put MaAS at an unfair disadvantage in a one-off evaluation? The comparison to **AFlow** (also search-based) is more direct and convincing.
3.  **Missing Ablation:** A crucial ablation is missing: **How important is the evolutionary search itself?** A simple baseline where an LLM generates a Mermaid workflow from scratch in one shot (or via iterative refinement without a population-based history) would help isolate the contribution of the EP framework versus the Mermaid representation alone.
4.  **Claim vs. Evidence:** The abstract claims "faster convergence to executable plans." Figure 3 shows MermaidFlow reaching a higher final performance, but the "convergence" speed (iterations to reach a given performance threshold) is not quantitatively compared. The text mentions token efficiency (half the cost of AFlow), which supports the claim, but a clearer, quantitative comparison of convergence speed is needed.

**5.3 Ablation Study:** The evolution efficiency analysis and the optimal stopping point discussion are valuable. The study on optimization LLM scale is interesting but unsurprising (better LLMs yield better workflows). Table 2 is referenced but not provided in the main text (it appears to be missing from the parsed content), which hinders evaluation.

### Writing & Clarity
The paper is generally well-structured and clear. The use of figures to illustrate the workflow lifecycle, framework overview, and case study is effective. The formal definitions in Sections 3 and 4 are precise.

**Critical Concern:** Some sections are overly verbose due to the inclusion of extremely long prompts and code listings in the appendices. While providing prompts is good for openness, the core method description in Sections 3 and 4 should more clearly summarize the *principles* of how the LLM is guided, rather than pointing to thousands of lines of prompt text. The main narrative is sometimes interrupted by references to missing elements (e.g., "Figure 5.3", "Table 2").

### Limitations & Broader Impact
The conclusion briefly mentions that "integration with real-world multi-agent systems and user-in-the-loop workflows introduces nuances that merit further exploration." This is a minimal acknowledgment of limitations. A dedicated limitations section would strengthen the paper. Key limitations to address include:
1.  **Expressiveness:** As noted in Appendix E, the current Mermaid representation cannot directly express control flow (loops, conditionals). This is a significant limitation for general-purpose workflow generation.
2.  **Scalability & Cost:** The approach requires multiple LLM calls per optimization cycle (for generation, judging, translation). The computational and financial cost of the search process is not discussed.
3.  **Dependence on Proprietary LLMs:** The entire framework relies on the capabilities of a closed-source LLM (GPT-4o-mini) for both optimization and execution. The performance and validity guarantees may not transfer to open-source or smaller models.
4.  **Societal Impact:** Not discussed. Potential negative impacts could include the environmental cost of extensive LLM queries or the risk of generating workflows that automate harmful tasks. These should at least be acknowledged.

### Overall Assessment
MermaidFlow presents a thoughtful and empirically validated approach to improving the robustness of LLM-based agentic workflow generation. Its core idea—using a declarative, verifiable graph representation as a search space for evolutionary optimization—is sound and demonstrates clear practical benefits over direct code generation, as shown in improved success rates and better token efficiency. The work is suitable for ICLR as it addresses a timely problem in agent foundations with a novel methodology.

However, the paper's contributions are tempered by several important concerns. The **strongest claims regarding guarantees need qualification**. The **empirical improvements, while consistent, are modest and lack statistical validation**. The **reproducibility of the method is heavily dependent on undisclosed prompt engineering and checker implementation details**. Finally, a more thorough discussion of **limitations and ablation studies** (especially on the necessity of the evolutionary component) is required.

**Recommendation:** The paper presents a promising framework but requires revisions to solidify its claims, provide rigorous statistical evidence, improve reproducibility, and thoroughly address limitations before meeting the high bar for ICLR acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces MermaidFlow, a framework that redefines agentic workflow generation by representing workflows as declarative, statically verifiable graphs using the Mermaid language. It proposes a safety-constrained evolutionary programming approach with graph-level operators that preserve correctness, enabling efficient search over a structured workflow space. Empirical results on math and code benchmarks show consistent improvements in success rates and convergence speed compared to existing methods.

### Strengths
1. **Innovative Representation**: The use of Mermaid graphs as a declarative, typed intermediate representation is novel and well-motivated. It enables static verification (e.g., type safety, connectivity) and human interpretability, addressing limitations of imperative code-based workflows (Sections 3.1-3.2, Figure 1).
2. **Safety-Constrained Evolutionary Operators**: The formally defined operators (e.g., node substitution, crossover) are designed to maintain workflow correctness by construction, supported by Lemma 1 on transformation invariance. This reduces invalid candidates during search (Section 4.1).
3. **Comprehensive Empirical Evaluation**: The paper demonstrates consistent performance gains over 13 baselines across four benchmarks (GSM8K, MATH, HumanEval, MBPP), with detailed ablation studies on evolution efficiency, LLM scale impact, and optimal stopping points (Table 1, Section 5.2-5.3).
4. **Clear Framework Design**: The separation between planning (Mermaid graphs) and execution (Python code) is clearly articulated, enhancing modularity and reducing brittleness. The algorithmic details and prompt templates in the appendix aid reproducibility (Algorithms 1-3, Appendix A.3).

### Weaknesses
1. **Limited Domain Evaluation**: Experiments are restricted to math reasoning and code generation tasks, lacking validation on more complex agentic scenarios (e.g., planning, dialogue, or real-world interactions). This may overstate generality (Section 5.1).
2. **LLM-Dependent Translation**: The conversion from Mermaid to Python code relies on an LLM, which can introduce errors and undermines the claimed reliability. The paper acknowledges this in Appendix E but does not resolve it empirically.
3. **Incomplete Workflow Expressiveness**: The representation currently lacks support for control structures like loops or conditionals, limiting its applicability to dynamic workflows. This is noted as future work but not addressed in experiments (Appendix E).
4. **Potential Baseline Inconsistencies**: Some baseline results (e.g., MaAS on MBPP) are taken from prior papers rather than re-implemented, which may affect fairness due to differences in setup or hyperparameters (Table 1 footnote).

### Novelty & Significance
The paper introduces a novel combination of Mermaid-based workflow representation and safety-constrained evolutionary programming, offering a principled approach to improve robustness and interpretability in multi-agent systems. The emphasis on static verification and correctness-preserving search is significant for scalable agentic reasoning. However, the core evolutionary concepts build upon existing work, and the practical impact depends on extending the representation to broader domains.

### Suggestions for Improvement
1. **Broaden Evaluation**: Include benchmarks from diverse domains (e.g., HotpotQA, DROP, or interactive tasks) to demonstrate broader applicability and robustness.
2. **Develop a Rule-Based Translator**: Implement a deterministic compiler from Mermaid to Python (e.g., using LangGraph) to eliminate LLM dependency and enhance reproducibility, as suggested in Appendix E.
3. **Extend Representation for Control Flow**: Incorporate node types for loops, conditionals, and recursion to handle more complex workflows, possibly through an expanded Mermaid schema or domain-specific extensions.
4. **Conduct Component Ablations**: Isolate the contributions of the Mermaid representation, evolutionary operators, and LLM-as-judge through controlled experiments (e.g., removing safety constraints or using random mutations).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Direct comparison of workflow validity rates.** The paper claims Mermaid ensures "static graph-level correctness" and a >90% valid code generation rate vs. ~50% for AFlow, but this is only mentioned in text. A direct, quantitative comparison of the percentage of *generated workflows* that are syntactically valid and executable (before LLM-as-Judge selection) between MermaidFlow and AFlow/ADAS is missing. Without this, the core claim of a more reliable search space is not substantiated.
2. **Ablation of evolutionary operators.** The contribution includes novel safety-constrained operators (crossover, mutation, etc.), but there is no experiment showing the impact of each operator type or the "safety-constrained" aspect. An ablation where operators are applied without constraints (or individually removed) would test if the constraints are actually necessary for the performance gains.
3. **Comparison against a rule-based Mermaid-to-Python translator.** The current method uses an LLM to translate Mermaid to Python, reintroducing the unreliability the paper criticizes. A baseline where this translation is done via a deterministic, rule-based compiler (mentioned as future work in Appendix E) is essential to isolate the benefit of the Mermaid representation from the hazards of the LLM-based translation step.
4. **Experiments on more diverse or complex agent tasks.** Evaluation is limited to math and code generation, which are largely single-agent reasoning tasks masquerading as multi-agent workflows. Testing on benchmarks requiring true coordination, tool use, or dynamic planning (e.g., WebShop, ALFWorld, or a custom multi-step collaboration benchmark) is needed to validate the framework's utility for *agentic* workflows.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of what "safety" and "correctness" actually mean and guarantee.** The paper claims "guarantee static graph-level correctness," but this is only syntactic/structural validity (e.g., nodes are connected). It does not guarantee the workflow is semantically sensible or effective for the task. A deeper analysis distinguishing syntactic validity from semantic quality, and quantifying how often "correct" graphs are useless, is critical.
2. **Analysis of search dynamics and population diversity.** The learning curve (Fig. 3) is shallow and only for MATH. A deeper analysis is needed: how does the population fitness distribution evolve? Does the method converge to local maxima? How diverse are the final workflows? This is needed to trust that EP is doing more than just cherry-picking from random generation.
3. **Cost-Benefit analysis of the full pipeline.** The token efficiency claim (half the cost of AFlow) is under-explained and given without variance or statistical testing. A detailed breakdown of token usage (for generation, validation, LLM-as-Judge, translation) versus the performance gain per token is necessary to claim "faster convergence" and efficiency.
4. **Analysis of the LLM-as-Judge's role and biases.** The final workflow selection relies on an LLM-as-Judge. How aligned are its scores with actual execution scores? Could the gains simply be from using a better judge? An analysis of judge accuracy and its correlation with final performance is missing.

### Visualizations & Case Studies
1. **Visualization of failure modes and limitations.** The paper shows successful evolved workflows but not where the method fails. Case studies of typical failures (e.g., semantically incoherent graphs that are syntactically valid, or graphs that pass the Mermaid checker but produce wrong answers) would reveal the practical limits of the "safety" guarantees.
2. **Side-by-side evolution traces.** Show a sequence of graph mutations from initial to final workflow for a few problems, illustrating the evolutionary path. This would make the "safety-constrained" process concrete and show whether the operators lead to interpretable improvements or random walks.
3. **Comparison of workflow complexity.** Visualize the distribution of workflow size (nodes, edges) for MermaidFlow vs. baselines. If MermaidFlow simply generates larger, more ensemble-heavy graphs, the gain might be from brute-force complexity, not smarter search.

### Obvious Next Steps
1. **Implement a rule-based Mermaid-to-Python compiler.** This is mentioned in Appendix E as future work, but it is an obvious prerequisite for a paper claiming reliability benefits from a structured representation. Using an LLM for translation undermines the core thesis.
2. **Benchmark on true multi-agent coordination tasks.** The current tasks (GSM8K, MATH, HumanEval, MBPP) are not strong proxies for the "agentic workflow" problem the paper motivates. Testing on environments requiring role specialization, communication, and dynamic task decomposition is a necessary next step that should have been included.
3. **Formalize and prove the "safety" properties.** Lemma 1 states the space is closed under the operators, but this is a trivial claim if the validator `Q` is defined as checking membership in `S`. A more meaningful analysis would be: given a set of semantic constraints (e.g., "no disconnected components"), prove that the operators preserve them.
4. **Compare against a simpler baseline: direct LLM generation of Mermaid graphs with rejection sampling.** If the gains come from the Mermaid representation, then generating Mermaid graphs via a single LLM call and discarding invalid ones (without evolution) should already outperform Python-based methods. An ablation comparing the full EP pipeline to this simpler baseline is needed to justify the evolutionary complexity.

# Final Consolidated Review
## Summary
MermaidFlow introduces a declarative graph representation for agentic workflows using the Mermaid language, coupled with a safety-constrained evolutionary programming framework to search this space. The method separates planning (verifiable graphs) from execution (generated code), aiming to improve robustness and search efficiency over direct code-generation approaches.

## Strengths
- **Novel, Verifiable Intermediate Representation**: Using Mermaid as a typed, declarative graph representation enables static verification of structural properties (type safety, connectivity) and improves human interpretability, directly addressing the paper's motivation of moving away from brittle, code-entangled workflows (Sections 3.1-3.2, Figure 1).
- **Effective Constrained Search Framework**: The formally defined evolutionary operators (e.g., node addition, crossover) are designed to preserve graph correctness, and the empirical results demonstrate consistent performance gains across four standard benchmarks, with clear advantages in token efficiency and higher valid code generation rates (>90% vs. ~50% for AFlow) (Table 1, Section 5.3).
- **Comprehensive Implementation Details**: The appendices provide extensive algorithmic descriptions, prompt templates, and validation checker logic, which significantly aids understanding and reproducibility (Algorithms 1-3, Appendices A.2, A.3).

## Weaknesses
- **Limited Evaluation Scope Undermines General Claims**: The paper motivates the problem as fundamental to "agentic workflow generation," but evaluations are confined to math reasoning (GSM8K, MATH) and code generation (HumanEval, MBPP). These are largely single-agent reasoning tasks; the framework's utility for true multi-agent coordination, planning, or tool-use scenarios remains unproven, weakening the claim of a general solution for agentic systems (Section 5.1).
- **Core Reliability Claim is Undermined by LLM-Dependent Translation**: The promised robustness stems from a verifiable representation, but the final step—translating Mermaid to executable Python code—is performed by an LLM. This reintroduces the very unreliability the representation aims to circumvent, as the LLM can generate incorrect code. The paper acknowledges this in Appendix E but provides no empirical mitigation (e.g., error rates of the translation step), leaving the end-to-end guarantee incomplete.
- **Missing Ablation on the Evolutionary Component**: The contribution combines the Mermaid representation with an evolutionary programming (EP) search. It is unclear how much performance gain is attributable to the representation itself versus the EP framework. A critical baseline is absent: generating Mermaid graphs via a single LLM call with rejection sampling (no evolution) compared to the full EP pipeline. Without this, the necessity of the evolutionary complexity is not justified.
- **Empirical Improvements are Modest and Lack Statistical Validation**: While results show consistent gains, the margins are often small (e.g., +2.61% on MATH, +1.40% average over MaAS). The paper reports averages over three runs but provides no measures of variance, confidence intervals, or statistical significance tests, making it difficult to assess if improvements are robust or due to random variation (Table 1, Section 5.2).

## Nice-to-Haves
- A detailed cost-benefit analysis breaking down token usage (generation, validation, judging, translation) versus performance gain would strengthen the "faster convergence" and efficiency claims.
- Visualizing evolution traces (sequences of graph mutations from initial to final workflow) would make the safety-constrained search process more concrete and interpretable.
- Extending the Mermaid schema to natively support control-flow constructs (loops, conditionals) would enhance the framework's expressiveness for dynamic workflows, as noted in Appendix E.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Strength/Weakness on Guarantees**: The harsh critic's point that the "guarantee of static graph-level correctness" is overstated is partially valid but requires nuance. The paper does not claim unconditional guarantees; it implements a checker and regeneration loop (Section 4.1: "If any violations are detected, new workflows are regenerated"). The claim is contextualized by this mechanism, so it is not a factual misrepresentation but could be phrased more precisely. The point is removed as a direct weakness.
- **Weakness on Novelty/Substance**: The criticism that the core idea is "reminiscent of classical program synthesis" is a generic critique applicable to many learning-based systems and does not engage with the specific contribution of adapting Mermaid for this domain in a novel way. Removed.
- **Weakness on Implementation Heavy / Reproducibility**: While prompt engineering is extensive, the paper provides the full prompts in the appendix (A.3), which is sufficient for reproducibility. Demanding the exact regex patterns of the checker is overly specific; the description of using regex and the Mermaid CLI (Appendix A.2) is adequate. Removed.
- **Weakness on Fairness of Comparison to MaAS**: The paper explicitly notes MaAS incorporates a trainable module, while MermaidFlow does not (Section 5.1). This is a difference in approach, not an unfair comparison. The comparison to purely search-based methods (AFlow, ADAS) is the most direct and convincing. Removed.
- **Weakness on Statistical Significance for Convergence Speed**: The claim of "faster convergence" is supported by a token efficiency comparison (half the cost of AFlow) in Section 5.3, which is a quantitative metric. While more rigorous timing analysis would be nice, the provided evidence is reasonable. Downgraded to a Nice-to-Have.
- **Weakness on Missing Table 2**: The referenced Table 2 (comparison of optimization LLMs) is present in the parsed content (lines 1088-1091). The reviewer's complaint is factually incorrect. Removed.
- **Suggestion to "Formalize and prove safety properties"**: For an empirical systems paper, providing Lemma 1 and defining the validator is sufficient. Requiring full formal proofs is not a standard expectation. Removed.
- **Suggestion to "Compare against a simpler baseline (direct LLM generation of Mermaid)"**: This is a substantive ablation point and has been moved to the core Weaknesses section.

## Novel Insights
The reviews collectively highlight a core tension in the work: the introduction of a structured, verifiable representation (Mermaid graphs) is a genuine insight that addresses a known brittleness problem in agentic systems. However, the framework's end-to-end reliability is partially compromised by retaining an LLM for the final translation step, and its evaluation on simplified tasks leaves open whether the benefits will hold in truly agentic, multi-step coordination settings. The most novel observation is that a declarative graph representation can create a search space where validity is easier to enforce, enabling more efficient evolutionary exploration, but the ultimate semantic correctness and practical utility for complex agency remain to be fully demonstrated.

## Suggestions
- Broaden the empirical evaluation to include at least one benchmark requiring genuine multi-agent coordination, dynamic planning, or tool use (e.g., WebShop, ALFWorld, or a custom collaborative task) to substantiate the claim of applicability to agentic workflows.
- Implement and evaluate a rule-based, deterministic compiler from Mermaid to executable code (e.g., using LangGraph) to replace the LLM-based translator, thereby solidifying the end-to-end reliability argument.
- Add basic statistical reporting (e.g., standard deviation across runs) and, if feasible, significance testing for the key performance comparisons to strengthen the empirical claims.
- Include an ablation study comparing the full EP pipeline against a baseline that generates Mermaid graphs via a single LLM call with rejection sampling (no evolutionary history or operators) to isolate the contribution of the evolutionary search component.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 4.0]
Average score: 5.3
Binary outcome: Reject
