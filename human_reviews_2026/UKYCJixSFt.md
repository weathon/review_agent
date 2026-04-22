# Towards Reliable Benchmarking: A Contamination Free, Controllable Evaluation Framework for Multi-step LLM Function Calling

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4, 4

## Abstract
As language models gain access to external tools through structured function calls, they become increasingly more capable of solving complex, multi-step tasks. However, existing benchmarks for tool-augmented language models (TaLMs) provide insufficient control over factors such as the number of functions accessible, task complexity, and input size, and remain vulnerable to data contamination.
We present FuncBenchGen, a unified, contamination-free framework that evaluates TaLMs by generating synthetic multi-step tool-use tasks to stress-test TaLMs.  
The key idea is to cast tool use as traversal over a hidden function-dependency DAG where nodes are function calls and an edge between nodes represents one function consuming the output of another.
Given a set of external function schemas, initial variable values, and a target variable, models must compose the correct call sequence to compute the target variable.  FuncBenchGen allows users to precisely control task difficulty (e.g., graph size, dependency depth, and distractor functions) while avoiding pretraining/test-time leakage.

We apply our FuncBenchGen framework to evaluate seven open and closed LLMs on tool use tasks of varying difficulty.  Reasoning-optimized models consistently outperform general-purpose models with GPT-5 significantly outperforming other available models. Performance declines sharply as dependency depth increases.  Furthermore, connected distractors---irrelevant functions sharing type-compatible variables with relevant functions---prove especially difficult to handle. 

We find that strong models often make syntactically valid function calls but propagate incorrect or stale argument values across steps, revealing brittle state tracking by LLMs in multi-turn tool use.
Motivated by this observation, we introduce a simple mitigation strategy that explicitly restates prior variable values to the agent at each step.  Surprisingly, this lightweight change yields substantial gains across models. e.g., yielding an improvement in success rate from 62.5\% to 81.3\% for GPT-5, without modifying the underlying architectures or training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents FuncBenchGen, a evaluation framework for tool-augmented LLMs. the main contributions are: (1) proposing a framework that can generate contamination-free function-calling tasks. Besides, they conduct extensive experiments with 7 LLMs. They have revealed some interesting findings. Finally, they proposed a simpel augmentation mitigation strategy that restates variable values from prior calls.

### Strengths
1. Overall, the presentation is good. The paper is well-written and easy to follow.
2. The contribution is significant, and many experiments are conducted for demonstration. They proposed a evaluation framework that is contamination-free, requiring many task complexities including function, depth and irrelevant function types. Their experiments with 7 LLMs, including the most-up-dated GPT5.

### Weaknesses
1. failure analysis and mitigation strategy are not extensively studied. For example, how do you find these 4 types of these failures? how it widely spreading in the cases? How do you define the boundary of each failure type? Section 4.6.1 mitigation strategy, is there more experiments to demonstrate its effectiveness?

### Questions
See above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents FuncBenchGen, an automated contamination-free benchmark for testing tool-augmented LMs on multi-step function calls with fine-grained control over difficulty. It treats tool use as walking a hidden function-dependency DAG and generates synthetic function schemas at eval time to prevent train/test leakage. The setup lets you tune depth, function count/types, and add distractors. The authors test seven leading LMs, analyze where they fail, and show a simple fix—explicitly restating variables—significantly boosts success.

### Strengths
- The framework generates tasks and functions on the fly, preventing pretraining/test-time leakage and removing the data-contamination issues that plagued prior function-calling benchmarks.
- FuncBenchGen exposes fine control over function-set size, dependency depth (DAGs), and irrelevant distractors (connected/disconnected), enabling targeted evaluation along key axes (Table 1).
- The study is broad and careful. The leading LMs are tested across graph sizes, complexities, and distractors; The results (Table 2; Fig. 3–4) give a multi-view of behavior, and the executor breakdown (Table 3) shows variable-tracking errors dominate—pointing to clear fixes.
- The proposed fix is simple and effective. Based on the reported results, for example, GPT-5 show gains from 62.5% to 81.3%. Besides, modeling multi-step tool use as DAGs and enforcing schema/type matching aligns with real API workflows and applications.

### Weaknesses
- The missing prior work and limited experiments. The study does not directly discuss or compare to the prior work on contamination-free or fine-grained function-calling benchmarks (HammerBench [1], LiveBench [2], etc.). This may significantly limits novelty and contributions. This paper should include the missing work also in Section 2, so to more competently introduce the related work and explain how FuncBenchGen is distinct in controllability, expressivity, or realism.
- Shall include more examples to better show the motivations and potentials/limitations of FuncBenchGen. For example, the paper may show qualitative examples of actual failed call traces or state-tracking failures, such as incorrect call propagation, classic state-tracking errors. Besides, it shall be better if this paper shows more complex control flow, such as conditional logic, iteration, probabilistic outputs, which is commonly appearing in real-world systems.
[1] Wang, Jun, et al. "Hammerbench: Fine-grained function-calling evaluation in real mobile device scenarios." arXiv preprint arXiv:2412.16516 (2024).
[2] White, Colin, et al. "LiveBench: A challenging, contamination-limited LLM benchmark." arXiv preprint arXiv:2406.19314 (2024).

### Questions
- Could you share more about how FuncBenchGen samples DAGs? In particular, how do you make sure the sampled graphs are truly representative for a given size and depth?
- How well does the synthetic type-matching approach hold up with real-world function calls where types are deeply nested and often ambiguous? What is the suggested way to use real API sets to stress-test realism and semantic complexity?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces FuncBenchGen, a benchmark for tool-augmented language models. The authors synthetically generate function-calling tasks in the form of DAG's, with each node taking in some input/output variables and edges representing function composition. The authors benchmark seven LLMs on FuncBenchGen and propose a mitigation strategy to improve performance on their benchmark.

### Strengths
- The proposed benchmark is synthetic, which allows the authors to study various properties in a controlled manner. The authors vary multiple dimensions such as the number of irrelevant premises, the size of the graph, and so on.
- An aspect of tool-use is measured: being able to retrieve the correct functions and use them in a way that allows the LLM to solve the execution prediction problem.
- The paper is presented clearly and the evaluation is sound. Most SoTA models are tested on the benchmark, and quite a few of them struggle to complete the task.

### Weaknesses
- While this synthetic task is interesting from an analysis point of view, it is unclear how it applies to real-world function calling scenarios. While the authors briefly address this in Sec 3.3, I find the abstraction to simplify the task a lot and would like to see at least one of the described real-world scenarios evaluated in a more realistic setting to validate the simplified formulation being a reliable proxy.
- The papers fails to mention a large relevant body of literature focused on following the flow of programs and logic chains (e.g. https://arxiv.org/abs/2209.00840, https://arxiv.org/abs/2401.03065, https://arxiv.org/abs/2403.16437). These tasks, in spirit, are very similar, as they relate to being able to follow relevant chains of logic in a graph. Many of the research questions have been studied in similar contexts (e.g. RQ2 in https://arxiv.org/abs/2302.00093, RQ3 in https://arxiv.org/abs/2406.17169).

### Questions
- How would fine-tuning on the task affect the performance on the benchmark? One advantage of the synthetic setup is that it is very easy to generate more examples of the task to study something like this.
- While you mention is an abstraction of a more complex real-world task, do you have any evidence performance on this task correlates in some way to performance on a real-world function calling task? Often, tasks that are equivalent down to some abstraction can have large differences when evaluating a model on them.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FuncBenchGen, a contamination-free evaluation framework designed to address the key limitations in existing benchmarks for multi-step function calling of TaLMs. Current benchmarks face the risk of data contamination and lack fine-grained control over task complexity factors such as function dependency depth, the number of core functions, and the presence of irrelevant functions. FuncBenchGen resolves these issues by formulating multi-step function calling as traversal over a hidden function-dependency DAG. The framework dynamically generates synthetic functions and tasks during evaluation to ensure no data leakage, while allowing users to precisely adjust complexity parameters, including the number of core functions, dependency depth, and the type and quantity of irrelevant distractor functions. Extensive experiments on seven state-of-the-art large language models show that reasoning-optimized models consistently outperform general-purpose models, with GPT-5 achieving the highest success rate but performing poorly in long-sequence tasks. CINs severely impair performance by interfering with relevant paths, and model performance declines sharply as dependency depth increases. The most common failure mode is the use of "unknown variable values," while syntax errors are relatively rare. A simple mitigation strategy, explicitly restating all known variable values at each step, can significantly improve performance.

### Strengths
1.Proposes an innovative DAG-based FuncBenchGen framework enabling fine-grained complexity adjustment.
2.Conducts extensive experiments on 7 state-of-the-art LLMs with rigorous variable control and repeats several times, yielding reliable results.
3.Presents a practical lightweight mitigation strategy with significant performance gains without model modification/training.

### Weaknesses
1.Given a task, how can a framework automatically generate a test suite with a specified depth of n? How should the functionality of each core node be constructed? It is crucial to ensure that exactly n nodes are sufficient and necessary to complete the task and that each node is feasible; however, this is not mentioned in the text.
2.There is a discrepancy between the "type + subtype" matching of synthetic functions and the parameter semantic matching of real APIs. The framework's migratability in real API scenarios has not been verified.

### Questions
1.After a task is split into m core nodes, will m function APIs actually be built as tools for the large model to call? If querying addresses, web searches, etc., are involved, how is each of these tools constructed? 
2.When the model is only provided with the meta-information (function description, type + subtype annotations) of synthetic functions without knowing their preset "input→output" mapping rules, how does the model ultimately obtain the expected target variable value? How does the model infer the correct function call sequence without knowning the semantic meaning? If the aforementioned questions and shortcomings are explained, a reassessment of the score may be conducted.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes FuncBenchGen, a benchmark framework for evaluating LLMs on multi-step function-calling tasks under controlled and contamination-free conditions. The benchmark allows fine-grained control over task complexity (e.g., function count, depth, and distractors) and reveals key limitations in models' state tracking. A simple mitigation strategy, restating known variables, significantly improves performance.

### Strengths
1. This paper proposes a novel benchmark framework that addresses a critical and under-explored challenge, evaluating LLMs on multi-step tool use in a contamination-free, controlled setting.
2. Experiments are comprehensive, covering a range of models and various settings.
3. The proposed mitigation strategy, while simple, is effective and general.

### Weaknesses
1. The paper suffers from significant formatting and presentation problems. For example, the abstract is split into three paragraphs, disrupting its coherence.
2. The paper does not provide concrete examples, making it difficult to understand what the actual data instances look like. Without such examples, it is unclear how complex the generated tasks truly are, what kind of reasoning is required at each step, or how function inputs and outputs are represented.
3. Function schemas and variable types are generated entirely at random without semantic constraints, which reduces interpretability and realism. As a result, the benchmark evaluates abstract structural reasoning rather than meaningful function-calling behavior, limiting external validity.
4. Tasks are purely synthetic, with limited connection to real-world API use or natural semantics.
5. The proposed "variable reminder" mitigation, while effective, is arguably a hack rather than a principled solution. It sidesteps the problem rather than enabling models to learn better internal state tracking.

### Questions
1. Since function schemas and variable types are randomly assigned, how do you ensure that the generated tasks remain semantically coherent or representative of real-world function-calling scenarios?
2. How do you validate that randomly generated functions produce diverse yet meaningful reasoning chains, rather than trivial or redundant ones?
3. The process for generating CIN and DIN nodes appears partly heuristic. Could you clarify how these nodes are chosen and connected in practice?

### Soundness
3

### Presentation
2

### Contribution
3
