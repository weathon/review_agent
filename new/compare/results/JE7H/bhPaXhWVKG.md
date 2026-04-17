---
job_id: aab1e77c-a84f-4533-82c4-f097e95f8919
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: bhPaXhWVKG.pdf
paper: MermaidFlow: Redefining Agentic Workflow Generation via Safety-Constrained Evolutionary Programming
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper focuses on representation and optimization of LLM-based multi-agent workflows via a declarative graph language and evolutionary search, which fits ICLR’s core areas (representation learning, neurosymbolic / hybrid systems, optimization, safety).

## Minimum Quality
Pass ✅.  
The paper has all required sections (Abstract, Introduction, Related Work, methodology, Experiments/Results, Conclusion). The methods are technically coherent, experiments are non-trivial on standard benchmarks (GSM8K, MATH, HumanEval, MBPP), and the exposition is adequate for review.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see hidden instructions to reviewers or other manipulative content; the only prompts are clearly marked as methodological components of the system.

---

# Expected Review Outcome:

## Summary

The paper introduces MermaidFlow, a framework for generating and optimizing LLM-based agentic workflows using a declarative, Mermaid-style graph representation. Workflows are specified as typed, statically verifiable graphs whose nodes map to LLM agents and tools, and the authors design a set of type-safe evolutionary operators (node addition/deletion, edge rewiring, subgraph mutation, crossover) that preserve graph-level correctness. An evolutionary programming loop, assisted by an LLM-as-judge, searches this structured space and the resulting workflows are translated into executable Python for math and code benchmarks, where MermaidFlow outperforms several non-agentic, hand-crafted agent, and automated workflow baselines.

## Strengths

1. **Clear problem formulation and representation contribution.**  
   The paper identifies a real pain point in agentic systems: workflows are usually encoded as brittle Python or JSON, which makes search extremely noisy. Section 3 formalizes a typed workflow space \(G(\mathcal{V}_{[\tau,\alpha]}, \mathcal{E}_{[\rho]})\) with explicit node/edge types, and Equation (3) makes the connection between a node and an LLM agent configuration precise. This explicit representation is a meaningful contribution independent of the specific search algorithm.

2. **Safety-preserving evolutionary operators grounded in typing.**  
   Section 4.1 defines operators (node addition, deletion, edge rewiring, subgraph mutation, crossover) with explicit type constraints such as  
   \(\texttt{T}_{\text{out}}(v_a)=\texttt{T}_{\text{in}}(v')\) for insertions and interface matching for subgraph swaps. While the “Lemma 1” is trivial, the *operator design itself* is thoughtful and demonstrates how to systematically constrain workflow evolution instead of ad-hoc code edits.

3. **Strong empirical comparison on standard benchmarks.**  
   Table 1 (Page 8) shows consistent improvements over 13 baselines across GSM8K, MATH, HumanEval, and MBPP, including strong code-based search methods like ADAS and AFlow and trainable systems like MaAS. The average performance gain of +1.4 percentage points over the best prior (MaAS) is non-trivial given that many baselines, especially on GSM8K and HumanEval, are already near the capability ceiling of the base LLM.

4. **Evidence that the representation yields better search dynamics and efficiency.**  
   Figure 3 (Page 8) compares train/test solve rates on MATH across optimization iterations for MermaidFlow vs AFlow. MermaidFlow’s curves are smoother, with fewer plateaus and faster attainment of high test accuracy, which supports the claim that staying in a statically valid graph space leads to more stable evolution. The token cost comparison (2.7e4 vs 6.9e4 tokens at similar performance) is an important practical benefit.

5. **Good qualitative case study of evolution / crossover.**  
   Figure 4 (Page 9) nicely visualizes how two parent workflows (Workflow_4 and Workflow_5) are recombined to produce Workflow_8, highlighting where test nodes and ensemble nodes are inherited. The accompanying Python snippet shows a faithful translation, reinforcing that the Mermaid representation is not just decorative but actually drives executable code.

6. **Well-documented implementation and prompts.**  
   Appendix A provides detailed node types (CustomOp, ProgrammerOp, ScEnsembleOp, TestOp, etc.), the Mermaid checker (W1–W5), and the actual prompt templates for updating graphs, generating Python, and LLM-as-judge. This makes the system reproducible in spirit and clarifies how the abstract operators are instantiated in practice.

7. **Figures effectively communicate the main ideas.**  
   Figure 1 (Page 2) illustrates the workflow lifecycle from Mermaid script to visualization and execution, and Figure 2 (Page 4) contrasts imperative vs declarative representations while showing the EP loop. These visuals make the semantics of the proposed “Mermaid field” much easier to grasp than text alone.

## Weaknesses

1. **Static “correctness” is overstated and not rigorously defined.**  
   The abstract and introduction claim “guarantee static graph-level correctness” and “every candidate is valid by construction”, but Section 4.1 quietly admits that raw LLM outputs may violate constraints and must be filtered/regenerated by a checker. Definition 1 and Lemma 1 (Equations (4)–(5)) only formalize closure *assuming* the operator inputs already lie in \(\mathcal{S}\), and say nothing about the LLM generation step or semantic correctness (e.g., does the workflow meaningfully solve the task). The checker described in A.2 focuses on syntax and a few structural rules (W1–W5), not on semantics. This matters because much of the paper’s motivation is about safety and robustness; the current theory supports only “syntactic well-formedness under hand-designed operators,” which is noticeably weaker than the claims.

2. **Theoretical part is minimal and mostly tautological.**  
   Lemma 1 is essentially restating the closure property encoded in the operator definitions, and its “proof” is a one-line induction. There is no analysis of search efficiency, coverage of the task-relevant workflow manifold, or trade-offs between constraint tightness and expressivity. Equation (2) defines \(\mathcal{S}\) abstractly via a static validator \(Q\), but there is no discussion of how restrictive \(\mathcal{C}_{\text{static}}\) is, or whether it could systematically *exclude* useful workflows (e.g., ones with control flow constructs that Mermaid currently cannot represent, as acknowledged in Appendix E). For a representation + optimization paper at ICLR, this level of theory feels shallow.

3. **Heavy reliance on closed-source LLMs and an LLM-as-judge, with limited analysis of their impact.**  
   The optimization, judging, and code-generation all use GPT-4o(-mini) or Claude 3.5 (Section 5.1, Table 2). Table 2 shows performance increasing when the optimization LLM is scaled up, but the paper does not disentangle how much of MermaidFlow’s advantage over AFlow / ADAS comes from the *representation* vs simply feeding more structured prompts to the same powerful LLM. There is also no robustness analysis to judge miscalibration or bias in LLM-as-judge scoring. This limits the generality of the results and raises doubts about whether improvements would persist with weaker or open-source models.

4. **Search-space and operator ablations are missing or very limited.**  
   Section 5.3 focuses on evolution efficiency vs AFlow, but there is no ablation on the set of operators \(\mathbb{O}\). For instance, what happens if you remove crossover or subgraph mutation and only allow local node substitutions? How much do the type constraints vs general Mermaid syntax matter? Similarly, Table 3 (Page 9) only reports the index of the selected round; it does not show performance at each iteration for AFlow (beyond MATH) or for variants with different mutation probabilities. Without such ablations, it is hard to attribute gains to specific design choices rather than to generic “more careful search”.

5. **Benchmarks cover only math/code reasoning and are relatively small in training size.**  
   While GSM8K, MATH, HumanEval, and MBPP are standard, they are all single-instance question-answering or code synthesis tasks. The paper’s motivation, however, is about multi-agent workflows, roles, and complex coordination. The workflows shown in Figures 5–8 (Appendix B) are essentially star-shaped ensembles around a single problem node with an ensemble or test node; there is no example involving deep branching, asynchronous tools, or long-horizon planning. Moreover, the training splits are small (e.g., 33 HumanEval and 86 MBPP problems per Table 4), which may not stress the search process enough to exhibit failure cases or overfitting behaviors. This reduces confidence in claims of generality to “multi-agent systems” broadly.

6. **Lack of strong baselines in the *same* representation space.**  
   Most baselines in Table 1 are either (a) non-agentic prompting methods or (b) agent systems operating over Python or custom DSLs (AFlow, ADAS, MaAS, GPTSwarm). There is no comparison to other declarative or graphical encodings, such as using JSON schemas with type-checkers or LangGraph-like DAGs with static validation, nor to recent robustness-oriented workflow frameworks. Without baselines that share a similar representation but differ in search algorithm, it is difficult to isolate how much of the improvement stems from Mermaid itself versus the specific EP + LLM-as-judge design.

7. **Some claims about Python brittleness and Mermaid robustness are anecdotal and under-quantified.**  
   Section 5.3 and Appendix C give qualitative examples of Python failures (unreliable if-conditions, pointless loops, bad imports). These examples are reasonable, but the only quantitative evidence is a single 50% vs >90% “executable code success rate” statement without details on measurement protocol (e.g., number of workflows attempted, distribution over operators, what counts as executable). By contrast, Table 5 (Page 38) provides a nice breakdown of Mermaid error types and frequencies for W1–W5; a similar analysis for Python-based generation would make the argument more compelling.

8. **Mermaid-to-Python step still depends on an LLM and can reintroduce brittleness.**  
   The end-to-end guarantee of “safe, verifiable” workflows is weakened by the fact that Python code is generated from Mermaid via GPT-4o-mini (Algorithm 1, Step 8, and GENERATE_PYTHON_CODE prompt in A.3). The paper does not report failure rates or error analyses for this translation step, nor does it quantify how often the generated Python violates the semantics implied by the Mermaid graph. Appendix E acknowledges the need for a rule-based converter as future work, which implicitly admits this is a real issue.

9. **Clarity and notation issues in places.**  
   - In Equation (3), the set notation  
     \(\mathcal{V}_{[\tau,\alpha]}=\{(m,p(\tau,\alpha),f(\tau)\mid m\in M,\ p\in P,\ f\in F\}\)  
     is missing a closing parenthesis and vertical bar; it should be of the form \(\{(m,p(\tau,\alpha),f(\tau)) \mid ...\}\). Small, but it appears in the core formalization of nodes.  
   - Section 4.2 describes sampling parents from \(W_{\text{history},t}\) with distribution \(P_{\text{mixed}}(i)\), but does not specify the actual values used for \(\lambda\) and \(\alpha\) nor how sensitive results are to these hyperparameters.  
   - Algorithm 2 has several typos (`prev Attempt` instead of `prev_attempt`, missing assignment arrows), which slightly hampers unambiguous reproduction.

10. **Positioning relative to very recent workflow-robustness work is incomplete.**  
    The Related Work section is comprehensive for many multi-agent systems and workflow search methods, but misses closely aligned recent efforts that specifically target robustness of agentic workflows and hybrid evolutionary designs (see “Potentially Missing Related Work” below). Incorporating these would clarify what exactly is new in MermaidFlow beyond “yet another search framework over workflows”.

## Potentially Missing Related Work

1. **Xu et al., “RobustFlow: Towards Robust Agentic Workflow Generation,” 2025.**  
   This work tackles robustness in agentic workflow generation and appears directly relevant to the paper’s core goal of improving workflow reliability and safety. It should be discussed in Section 2 (probably under “Workflow Search and Optimization”), with an explicit comparison of the constraints / verification mechanisms used in RobustFlow versus MermaidFlow’s typed graph approach. If RobustFlow has empirical results on similar reasoning benchmarks, adding it as a baseline in Table 1 or at least a qualitative comparison would strengthen the positioning.

2. **Xu et al., “HyEvo: Self-Evolving Hybrid Agentic Workflows for Efficient Reasoning,” 2026.**  
   HyEvo proposes hybrid probabilistic–deterministic agentic workflows that evolve over time, which is conceptually quite close to the evolutionary programming perspective in Section 4. It should be referenced in Section 2 (likely under “Workflow Search and Optimization”) and contrasted with MermaidFlow’s strict type-based constraints. If HyEvo also uses evolutionary or population-based search, a discussion about differences in search operators, safety guarantees, and empirical domains would help clarify novelty.

## Questions

1. **Clarifying the notion of “correctness”.**  
   When you state that candidates are “valid by construction” or that MermaidFlow “guarantees static correctness”, do you mean only syntactic validity (no isolated nodes, proper node types, etc.), or do you claim some semantic property (e.g., no deadlocks, meaningful flow from problem to answer)? Please sharpen the definition, and if it is only syntactic, adjust the language throughout to avoid overclaiming.

2. **Quantifying the LLM-as-judge and code-generation failure modes.**  
   Can you provide statistics on (a) how often the LLM-as-judge picks a candidate that later underperforms a different candidate in the same round, and (b) how often Mermaid → Python translation fails or produces incorrect wiring relative to the original graph? Even approximate rates would help assess the reliability of these two critical components.

3. **Operator ablations and sensitivity.**  
   Could you run an ablation on MATH (or at least one dataset) where you: (i) disable crossover, (ii) disable subgraph mutation, and (iii) restrict to node substitution only, keeping everything else fixed? Also, how sensitive are the results to the 10% crossover probability, the candidate pool size \(N\), and the number of optimization rounds?

4. **Representation capacity vs current limitations.**  
   Appendix E notes that MermaidFlow cannot express if-conditions or loops. In practice, how often do the learned workflows require such constructs, especially for code generation tasks? Are there examples where AFlow discovers a loop-/branching-heavy workflow that MermaidFlow cannot represent, and if so, how does performance compare?

5. **Generality beyond math/code and to other LLMs.**  
   Do you have preliminary results (even anecdotal) on non-numeric domains such as QA, tool-augmented retrieval tasks, or interactive multi-step planning (e.g., web agent tasks)? Also, have you tried using a smaller open-source LLM (e.g., Llama-3 instruct) for either optimization or execution to see if the benefits of MermaidFlow persist?

6. **Clarification on the Python success-rate claim.**  
   In Section 5.3 you state that AFlow has “only a 50% success rate in generating executable code” vs >90% for MermaidFlow. How exactly were these rates measured (number of workflow-generation attempts, time horizon, dataset, definition of “executable”)? Providing a small table akin to Table 5 for this would help substantiate this important claim.

## Flag For Ethics Review

No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is technically coherent, the operator definitions are consistent, and experiments are reasonably thorough, but static “correctness” guarantees are limited to syntax/typing and the dependence on proprietary LLMs and LLM-as-judge is not deeply analyzed.

## Presentation Rating

3: good.  
The paper is generally clear, with helpful figures (especially Figures 1, 2, 3, and 4) and extensive appendices; however, some notation inconsistencies and over-claims about correctness detract slightly from clarity.

## Contribution Rating

3: good.  
The idea of using a Mermaid-based typed workflow representation together with constraint-preserving evolutionary operators is a meaningful step toward safer workflow search, and the gains over strong baselines on standard benchmarks make it relevant to the community, even if the theoretical depth and domain breadth are limited.

## Overall Rating

6: marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper presents a well-motivated and practically useful representation + search framework, with clear empirical gains and good qualitative insights. At the same time, several central claims about safety and correctness are stronger than the actual guarantees; ablations and robustness analyses are missing; and the evaluation domain is relatively narrow. I lean toward acceptance because the representation idea and the concrete system design are likely to be of interest to researchers working on agentic workflows, but substantial revisions would be needed for a higher-tier recommendation.

## Reviewer Confidence

4: confident.  
I am familiar with multi-agent LLM systems and workflow search methods, and I carefully checked the mathematical definitions and experimental setup. There may be very recent related work not covered here, but my overall assessment is unlikely to change dramatically.