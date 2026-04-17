# DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Large Language Models (LLMs) demonstrate strong performance on mathematical problems when prompted with Chain-of-Thought (CoT), yet it remains unclear whether this success stems from search, rote procedures, or rule-consistent reasoning. To address this, we propose modeling CoT as a certain rule-based stochastic process over directed acyclic graphs (DAGs), where nodes represent intermediate derivation states and edges encode rule applications. Within this framework, we introduce \textbf{logical closeness}, a metric that quantifies how well a model’s CoT trajectory (i.e., the LLM's final output) adheres to the DAG structure, providing evaluation beyond classical PASS@$k$ metrics. Building on this, we introduce the \emph{DAG-MATH} CoT format and construct a benchmark that guides LLMs to generate CoT trajectories in this format, thereby enabling the evaluation of their reasoning ability under our framework. Across standard math reasoning datasets, our analysis uncovers statistically significant differences in reasoning fidelity among representative LLM families-even when PASS@$k$ is comparable-highlighting gaps between final-answer accuracy and rule-consistent derivation. Our framework provides a balance between free-form CoT and formal proofs systems, offering actionable diagnostics for LLMs reasoning evaluation. Our benchmark and code are available at https://github.com/YuanheZ/DAG-MATH.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a DAG-based framework for representing CoT reasoning and introduces the concept of logical closeness, enabling fine-grained evaluation of LLM mathematical reasoning beyond final-answer accuracy. Its approach assesses the coherence and consistency of logical dependencies along a CoT trajectory, rather than focusing solely on whether the solution is correct. Furthermore, the authors construct a benchmark of DAG-formatted mathematical problems derived from existing datasets and provide empirical analyses linking graph-level characteristics—such as size, density, and branching complexity—to problem difficulty.

### Strengths
- This work formalizes the notion of logical closeness and proposes a metric, the perfect reasoning rate, based on this notion to measure LLMs' logical consistency beyond the final output. This indeed addresses an important question of whether an LLM arrives at a correct answer through genuine logical reasoning or mere pattern matching.
- The formalization in Sections 2 and 3 is clearly presented and easy to follow.
- Section 5 and Appendix B offer several interesting insights, such as the correlation between graph structure and problem difficulty, and how a correct final answer may still arise from unclosed or flawed reasoning.

### Weaknesses
- The DAG-MATH benchmark presented in the paper is validated using symbolic correctness and an LLM-as-Judge approach. I assume that SymPy is employed to verify mathematical equivalence, while the logical dependencies between nodes (i.e., whether an edge should exist) are assessed by the LLM-as-Judge. However, as the paper itself demonstrates, LLMs can often produce superficially consistent but logically inconsistent reasoning. While using LLMs for judgment is a practical solution, it would be helpful for the authors to further justify the reliability of this dataset construction and evaluation methodology. 
- Building on this point, reliable automation of the logical closeness check appears challenging, if not infeasible, since formalizing a DAG from natural-language CoT inherently involves subjective interpretation, particularly for more complex problems. For instance, reasonable disagreement could arise over whether a given edge should connect two specific nodes.

### Questions
- How can we trust an LLM-as-Judge to reliably evaluate the logical coherence of DAG constructions, particularly when edges are intended to represent valid inference paths? Have the authors conducted any analyses or validation studies to assess the consistency or accuracy of these judgments?
- At first glance, Figure 4 being a line plot was somewhat confusing. Do the authors plot accuracy against varying levels of logical correctness rates and then smooth the resulting curve? A brief clarification in the caption or text might help.
- I also wonder whether different node segmentation choices could exist for the same reasoning trajectory. If so, how sensitive are the proposed approach and the PRR metric to such segmentation differences?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes to model mathematical reasoning’s CoT traces as a rule-based stochastic process over task-specific DAGs, where nodes represent reasoning states and edges encode inference rules or justifications. Building on this formulation, the paper introduces logical closeness as a new metric to evaluate the model's reasoning trajectory. It also presents a benchmark of gold-standard DAG-MATH graphs with verified logical structures and statistical analyses relating graph properties to problem difficulty. Finally, the paper evaluates several large language models by prompting them to produce formatted CoT reasoning on mathematical benchmarks and examines how their reasoning abilities correlate with the proposed DAG-MATH framework.

### Strengths
- The paper is clearly written and well-organized. 
- The idea of representing CoT reasoning with DAG-MATH is interesting and novel. The proposed metrics are also new and conceptually sound. 
- The empirical results are informative, showing how graph structures reflect problem difficulty and reasoning quality.

### Weaknesses
- Enforcing the DAG-MATH format may degrade the natural reasoning flexibility of LLMs. It would help to include an analysis or ablation comparing performance with and without this formatting constraint. Furthermore, if the few-shot examples are drawn from a specific model family, models of the same family might have an advantage because their reasoning patterns are similar.
- The analysis is primarily quantitative. Some qualitative examples or case studies of the generated DAG-MATH graphs, especially highlighting common reasoning errors or structural failures, would strengthen the insights.
- In Section 2.2, the paper mentions that thinking LLMs can be viewed as “an exploration of the task-specific DAG with self-correction or backtracking, but its final output … is still consistent with our transition rule.”, but empirical results did not include reasoning models, which could strengthen the value of the paper. 
- DAG-MATH has limited scalability. Complex problems with very long and multiple reasoning traces are computationally expensive to construct. Those with cyclic reasoning or backtracking are hard to capture in a strictly acyclic form.

### Questions
- How exactly is the branching of reasoning paths determined?
- Since the canonicalization turns reasoning steps into SNF, does it limit the type of mathematical questions that DAG-MATH can apply to? 
- In lines 63-64, what does it mean that the other works “fail to capture long-range and cross-branch dependencies, as well as the goal-directed, absorbing-state nature of CoT”?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DAG-MATH, which is a framework to evaluate mathematical reasoning of LLMs through modeling Chain-of-Thought as stochastic process over directed acyclic graphs. The authors propose metric called "logical closeness" for distinguishing between when model solves problem by search versus real logical inference. They make benchmark with 2,894 gold-standard DAGs and test five LLMs, discovering that although PASS@1 scores vary much between models, their perfect reasoning rates stay quite similar, which suggests search is inflating the accuracy metrics.

### Strengths
The paper tackles really important problem about understanding whether LLMs achieve correct answers through systematic search or through genuine logical reasoning, which is fundamental question for the field. The DAG-based formalization is quite novel approach that sits nicely between completely free-form CoT and very formal systems like LEAN verification, making it more practical to use. The logical closeness metric gives us insights that go beyond simple PASS@k metrics that everyone uses.

The empirical analysis is quite comprehensive, showing interesting patterns about how DAG statistics like number of nodes, edges, density and branching correlate with problem difficulty. It reveals that harder problems create larger and more sparse graphs with higher branching, which makes sense intuitively. The finding that search and exploration inflate PASS@1 while actual reasoning ability measured by PRR stays comparable across different models is really actionable insight that changes how we should think about evaluating these systems. Also good that authors released the benchmark and code for others to use.

### Weaknesses
There is concerning circularity in how the benchmark was constructed - using GPT-4 and Qwen to create the "gold standard" DAGs means the benchmark is essentially created by same type of models that are being evaluated, which introduces obvious biases. The theoretical justification feels not enough developed. Why should we believe this specific DAG formalization captures what "true" reasoning means? The stochastic process described in Equation 1 seems somewhat arbitrary choice without proper justification.

The statistical analysis lacks rigor with only 32 samples per problem and no significance testing provided for the differences claimed between PASS@1 and PRR. There is no analysis about robustness - what happens when same problem can be formulated with different but equivalent DAG structures? The scope is quite limited, restricting to mathematics problems with difficulty below 6, and missing comparisons with other approaches like process reward models that also try to do step-level verification. The three-stage prompting methodology might be imposing particular reasoning patterns that are not universal. Most importantly, there is no human validation beyond what the models themselves produce, which is problematic for claiming these are "gold standard" solutions.

### Questions
How did you validate that the DAGs generated by GPT and Qwen actually represent correct reasoning structures? It seems crucial to have human experts verify at least subset of these DAGs to ensure the benchmark quality. Many mathematical problems have multiple valid solution approaches - how does the logical closeness metric handle cases where completely different but valid DAG structures could exist for same problem?

Could you provide proper statistical significance tests for the differences you claim between PASS@1 and PRR? How sensitive are all these results to the specific prompting strategies you used? If you changed the prompts slightly, would the DAG structures and evaluation results change significantly?

What is the practical path from these evaluation insights to actually improving LLM reasoning capabilities? The paper identifies interesting patterns but doesn't suggest how to use this knowledge for making better models. How does this framework compare empirically with recent work on process reward models from OpenAI and others that also try to verify reasoning at step level?

Why did you choose to require exactly one assertion per node - this seems quite restrictive and arbitrary? And why make logical closeness binary measure instead of having gradient that could capture partial correctness better? Finally, have you considered that the models might be following completely different internal reasoning process and the DAG structure is just post-hoc rationalization that we impose on their outputs?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DAG-MATH, a framework designed to formalize and evaluate the Chain-of-Thought (CoT) trajectories generated by Large Language Models (LLMs) in mathematical reasoning. The CoT is modeled as a rule-based stochastic process over a Directed Acyclic Graph (DAG), where nodes are intermediate states. The core proposal is the Logical Closeness metric, which quantifies the fidelity of an LLM's path against a "gold standard" DAG, yielding the Perfect Reasoning Rate (PRR) and AUC scores. The authors claim this provides a superior diagnostic tool compared to simple final-answer metrics like PASS@k. The resulting DAG-MATH benchmark is built using LLM-generated structured outputs.

### Strengths
1. The underlying idea of treating CoT as a DAG traversal is fundamentally sound and offers a pathway for structured reasoning analysis beyond token-level checks. This is the paper's primary and most important strength.
2. The authors have created impressive few-shot prompts to enforce their complex output format, which is a valuable demonstration of structured generation control in LLMs. The visual examples of the DAGs are convincing.
3. The metric correctly isolates failure modes like speculative branching and imperfect reasoning, which are invisible to simple PASS@k.

### Weaknesses
1. The PRR/AUC metric confuses adherence to the authors' custom template with true logical reasoning ability. The paper must provide evidence that this metric holds up when applied to non-formatted, naturally generated CoT.
2. A critical omission is the lack of comparison with or contextualization against MCTS or similar graph-based search methods. If the goal is to improve reasoning, how does the DAG-MATH diagnosis inform or relate to these established LLM search strategies?
3. The use of LLMs to generate the ground truth DAGs for their own evaluation introduces a circular dependency. This casts significant doubt on the objectivity and reliability of the Logical Closeness scores.
4. The Acyclicity Assumption restricts the framework to simple forward derivation, excluding crucial reasoning patterns like planning, iterative refinement, or proof by contradiction, thereby limiting its general applicability.

### Questions
1. Can you demonstrate the utility of PRR/AUC by heuristically parsing DAGs from unconstrained, free-form CoT outputs (without the DAG-MATH template) on a subset of problems? If the metric collapses here, it confirms the dependency on the template is too strong.
2. Please elaborate on the relationship between DAG-MATH's diagnostic insights and existing MCTS/Tree-of-Thought techniques. How can the PRR/AUC scores be used to guide the search policies or reward functions in such systems?
3. Given the reliance on LLM-generated ground truth, what specific human review or verification process was applied to the 2,894 gold-standard DAGs to ensure their canonical logical structure? What were the human agreement statistics on the logical decomposition?

### Soundness
2

### Presentation
3

### Contribution
2
