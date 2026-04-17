# Point-and-Click: A Procedural Benchmark for 2D Adventure Puzzle Solving

- Decision: Reject
- Scores: 4, 8, 4, 6

## Abstract
Point-and-click adventure games offer an ideal platform for testing multimodal large language model agents on long-horizon reasoning, commonsense knowledge, and language-perception grounding. Such games demand creative, compositional reasoning and the deduction of implicit goals. However, existing benchmarks provide limited support for compositional and generative puzzles, and often suffer from data contamination. To bridge this gap, we present Point-and-Click, a benchmark for 2D adventure games that procedurally generates rich puzzles and provides ground-truth solutions for evaluation. The environment instantiates controllable directed acyclic graphs of puzzle dependencies over primitives like keys/locks, codes, and pattern matching, spanning an exponentially scaling number of layouts with tunable difficulty. Experiments reveal the limitations of current multimodal LLM/VLM agents on this benchmark. We hope Point-and-Click serves as a rigorous testbed for progress on general-purpose embodied reasoning and implicit goal deduction in interactive environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes Point-and-Click, a procedurally generated benchmark for evaluating multimodal agents on long-horizon, goal-inference puzzle solving in 2D adventure-game environments. Each puzzle is represented as a DAG that defines the dependencies between sub-tasks. The benchmark currently supports click-only interactions. By evaluating 3 different agents, the results reveal a significant performance gap between agents and humans, especially on hard set, where all 3 agents achieve a 0% success rate.

### Strengths
- The framework leverages DAGs to generate puzzles, allowing difficulty control by adjusting the number of steps, DAG size, and branching factors.
- Unlike standard datasets with clearly defined goals or ground-truth answers, this benchmark does not have an explicit goal. Instead, it requires agents to explore the environment and recognize what constitutes success, which makes it stand out from standard datasets.

### Weaknesses
-  Although Section 4 describes error types and includes a subsection on case studies, no actual visual examples are provided in the main paper, only textual summaries. The text references the appendix, but the appendix and supplementary material are not available in the submission. Including a few annotated examples would make the diagnostics clearer.
- The paper spends most of its space describing the framework and implementation details, which is good, but some of this content could be moved to the appendix to leave more space for the experimental setup and results analysis. I found the current experimental setup a little vague, and the experimental results do not provide many insights.
- The framework relies on a limited set of puzzle primitives (3 main primitives). While this makes puzzle generation controllable and analysis easier, it also restricts the logical diversity of the benchmark. Even with longer DAGs, the underlying reasoning patterns might remain relatively simple and repetitive.
- A small suggestion, Figure 1(b) font is too small.

### Questions
- Could the authors elaborate on the human baseline setup? For example, how many participants were involved, what was their background, and what was their experimental setting? Were the human participants also restricted to a given action budget?
- The paper mentions 30 puzzles across 3 difficulty levels, each ranging from 10 to 100 steps. How is the difficulty determined in this case? Is it defined strictly by the average step count, or also by other factors such as graph branching and dead-end density?

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
4

### Summary
The paper introduces Point-and-Click, a benchmark for evaluating the ability of vision-language models to perform well on point-and-click puzzle games, testing long-horizon planning, grounded reasoning, and extrapolating world knowledge. The authors create a system for procedurally generating huge amounts of puzzles, spanning up to hundreds of steps with complex compositional dependencies (locks, keys, combinations, etc), and having no explicitly stated goal; e.g., the model must itself figure out what actions to take in order to complete the puzzle. The authors evaluate the performance of frontier models, discuss the types of errors that often occur, and explore the implications of the results on model design.

### Strengths
This is a great benchmark!

1. It measures an important ability of models in an interesting and intuitive way;
2. Procedural generation guarantees that there's no dataset contamination, a major issue with similar benchmarks;
3. The ground-truth causal graph allows evaluation of partial completion, which isn't the case for many other puzzle-related benchmarks.

The paper is clearly written, well-motivated, and describes the bechmark well. The evaluations are well-done and highlight the usefulness of point (3) above.

### Weaknesses
I think this paper lacks significant weaknesses; of course, it would always be nice if the benchmark was more diverse (e.g. more complex puzzle primitives, more 3D reasoning, etc), but this is trivially a weakness of every benchmark; the dataset introduced in the paper is good and logically self-contained.

Minor comments

- On lines 103–104, the performance is broken down into three categories, but I don't think the categories have been defined yet.

### Questions
As this is a benchmark paper that introduces its benchmark very clearly, I don't have any immediate questions for the authors.

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
Point-and-Click introduces a procedurally generated benchmark for multimodal agents to solve 2D point-and-click adventure puzzles with implicit goals, long-horizon dependencies, and language-perception grounding. Each game instance is built from controllable DAG-structured puzzle graphs, rendered as interactive rooms, and comes with ground-truth causal graphs for fine-grained evaluation. Experiments show a large human–agent gap: state-of-the-art computer-use/VLM agents achieve low success while humans remain high, revealing failures in perception/attention, riddle solving, memory, and long-horizon planning.

### Strengths
1. Tests GUI agents on implicit-goal puzzle solving (agent must infer what “success” even is), which is distinct from instruction-following paradigms.
2. Reports success, step completion, optimality vs. reference plan, and error categories (perception/attention misses, riddle-reasoning, forgetting clues) are useful diagnostics beyond accuracy.

### Weaknesses
1. The main tables lack uncertainty bars and budget-sensitivity (timeouts vs. success); it’s unclear how robust rankings are to interaction budgets and prompt seeds.
2. The paper evaluates only a small set of agents (two commercial “computer-use” systems and one open-source GUI agent), which makes it hard to separate paradigm effects from vendor-stack biases, and limits confidence that conclusions generalize across model families. I highly recommend the authors to add results in Qwen2.5-VL-72B-Instruct, InternVL 2.5/3, Llama-3.2-Vision ....

### Questions
I recommend bucketing instances by puzzle primitives:Key–Lock, Code–Lock, Pattern-Match, and their compositions, and reporting per-bucket Success and Efficiency@Success (optionally add last-3-step failure rate, input/typo errors for Code–Lock, and localization misses for Pattern-Match). This single table will localize whether performance drops are driven primarily by reasoning/planning & memory (Key–Lock), symbolic extraction and precise entry (Code–Lock), or visual perception/targeting (Pattern-Match), thereby pinpointing which capability family is the bottleneck and guiding targeted model/ablation improvements.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a procedurally generated 2D point-and-click puzzle benchmark to evaluate multimodal agents on long-horizon, visually grounded planning with implicit goals. The task seem to be far from existing paradigm of agent evaluations. Tasks are posed as dependency-DAG escape puzzles, solved through pixel-click interaction. The system provides ground-truth graphs, controllable difficulty, and fine-grained metrics. Experiments show large human–agent gaps.

### Strengths
- the overall design of the task is fairly innnovative, existing paradigm does not seem to exist. 

- The procedural generation with DAG can avoid data contamination for future evaluations, it also offers the opportunity of structured metrics. 

- the study also introduces human evaluations to demonstrate the gap.

### Weaknesses
- There are only 30 puzzles tested (10 per tier). The number seem too few to claim generalization across a procedural domain.

- The benchmark mixes visual recognition and planning without controlled ablations, although there are some analysis to infer where the procedure fails, some additional versions of the benchmark that evaluates part of the process can be valuable. 

- It is not so clear whether the puzzles will allow alternative solutions (even generated with a underlying DAG), is there a possiblity that a puzzle can be solve by other paths different from the underlying DAG? 

- The paper does not seem to use this oppoturnity to clearly discuss the strengths and limitations under this new evaluations.

### Questions
please address the limitations discussed.

### Soundness
3

### Presentation
3

### Contribution
3
