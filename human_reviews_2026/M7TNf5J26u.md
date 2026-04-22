# AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite

- Avg Score: 7.00
- Decision: Accept (Oral)
- Scores: 6, 8, 8, 6

## Abstract
AI agents hold the potential to revolutionize scientific productivity by automating literature reviews, replicating experiments, analyzing data, and even proposing new directions of inquiry; indeed, there are now many such agents, ranging from general-purpose "deep research" systems to specialized science-specific agents, such as AI Scientist and AIGS. Rigorous evaluation of these agents is critical for progress. Yet existing benchmarks fall short on several fronts: they often (1) lack reproducible agent tools necessary for a controlled comparison of core agentic capabilities; (2) do not account for confounding variables such as model cost and tool access; (3) do not provide standardized interfaces for quick agent prototyping and evaluation; (4) fail to provide holistic, product-informed measures of real-world use cases such as science research; and (5) lack comprehensive baseline agents necessary to identify true advances. In response, we define principles and tooling for more rigorously benchmarking agents. Using these, we present AstaBench, a suite that provides a holistic measure of agentic ability to perform scientific research, comprising 2400+ problems spanning the entire scientific discovery process and multiple scientific domains, and including many problems inspired by actual user requests to deployed Asta agents. Our suite comes with the first scientific research environment with production-grade search tools that enable controlled, reproducible evaluation, better accounting for confounders. Alongside, we provide a comprehensive suite of nine science-optimized classes of Asta agents and numerous baselines. Our extensive evaluation of 57 agents across 22 agent classes reveals several interesting findings, most importantly that despite meaningful progress on certain individual aspects, AI remains far from solving the challenge of science research assistance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces AstaBench, a benchmark suite and surrounding tooling intended to holistically evaluate agents for scientific research assistance. Contributions include: (i) a 2400+ problem task suite spanning literature understanding, coding/execution, data analysis, and end-to-end discovery; (ii) a standardized agent environment with date-restricted, production-grade literature search and a computational notebook to foster controlled comparison; (iii) a leaderboard and cost-accounting toolkit that normalizes API prices and reports efficiency; and (iv) a set of baseline agent classes (including several “Asta” agents) used to evaluate 57 agents across 22 classes. Key empirical finding: despite progress on some sub-skills, science assistance remains far from solved; even competitive agents struggle outside literature tasks, and end-to-end success rates remain low.

### Strengths
1. Ambitious scope & breadth. The suite spans four categories with explicit date cutoffs and tool restrictions, a step toward controlled, reproducible comparison in agent settings where tool access often dominates outcomes. 

2. Standardized environment. The Asta Scientific Corpus (date-restricted search/snippet tools) plus the Computational Notebook provides an agent-friendly interface decoupled from bespoke agent code—useful for generalization across architectures. 

3. Cost-aware leaderboard. Normalized, time-invariant cost accounting and reporting of confounders (openness, tooling) encourage more honest comparisons across agents/providers. 

4. Comprehensive baselines. Evaluation across 57 agents / 22 classes (including ReAct, Smolagents, and specialized Asta agents) provides a broad empirical snapshot rather than narrow, self-selected comparisons. 

5. Clear high-level findings. The paper documents that literature tasks are comparatively mature, while coding/execution, analysis, and end-to-end discovery remain weak—useful signal for the community.

### Weaknesses
1. Potential product-driven bias and external validity. Several tasks are “inspired by actual user requests to deployed Asta agents,” which risks distributional alignment with the authors’ own agent design and data pipelines. The paper could perhaps quantify how many tasks are product-derived, how they were sampled, and present held-out non-Asta sources to reduce bias?

2. Judging protocols and reliability. The work leans on a “Rubric & LLM Judge” but provides limited detail on prompting, calibration, adjudication, and human validation (e.g., double-blind spot-checks, disagreement resolution, inter-rater reliability). Perhaps more rigorous reporting (e.g., bootstrap CIs, per-task variance, adjudication rates) could be done to support “rigorous” claims?

3. Fairness of comparisons across tooling. Many agents are evaluated with custom or fully-custom tools, while others are restricted to the standard toolset. Even with “tooling” labels on the leaderboard, this remains an apples-to-oranges comparison. Stronger tool access ablations (standard-tools-only vs customized) could accompany headline numbers?

4. Domain skew & representativeness. The benchmark is weighted toward CS with relatively shallow non-CS coverage (e.g., limited biomed beyond LitQA2). For a “scientific research” suite, the domain diversity is modest. Perhaps the authors could provide a breakdown by field, difficulty, and reasoning type to justify claims of “holistic” science?

5. End-to-end metrics under-specified. The paper argues end-to-end discovery is hard (e.g., step-wise success compounding) but lacks task-level pass criteria, failure taxonomies, and human-rated scientific quality of the final reports/code for E2E-Bench(-Hard). Perhaps clarify grading and include expert review samples?

### Questions
My questions are reflected in the weaknesses I mentioned above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces AstaBench, a suite of benchmarks for evaluation AI agents' potential for conducting scientific tasks. The suite aggregates scientific tasks and the authors run a comprehensive evaluation of different models on these tasks with different agent scaffolds. The suite controls for several factors, such as which tools are used, and provides a set of agent baselines that the authors run these evaluations on.

The paper's contributions to AI for science evaluations are impressive. Evaluating agents is a challenging problem, and the authors make a serious and compelling effort in that direction. I recommend acceptance, and offer suggestions for further improving the paper below.

### Strengths
- The authors calculate the cost for all evaluations, which is key for reliable benchmarking. Most existing frameworks don't do this (as the authors correctly identify).
- The scope of benchmarks is broad, including agents from many stages of the scientific research pipeline
- The number of models and agent scaffolds tested is impressive
- The insights from the analysis are solid; I appreciate the focus on in-depth analysis and insights as well as the focus on openly releasing all of the materials from the analysis

### Weaknesses
- Some parts in comparisons with other frameworks are overclaimed. For example, as far as I know, frameworks like HAL and Inspect do have open agents and account for variation in tool use/agent scaffolds. 
- Similarly, some of these frameworks have support for general agents too, for example the Inspect ReAct agent or the HAL generalist agent that are available across benchmarks 
- Claims like benchmarks in other suites not being "product informed" also seem like a stretch, especially since the other frameworks also incorporate many of the same benchmarks
- CORE-Bench has 45 tasks in train and 45 in validation; why does the table say 35 and 37?
- Could you clarify how the budget or number of steps for agents is set, especially over different models? How do you avoid infinite loops?
- In figure 2, how are the 11 benchmarks collapsed into 5 figures? What is the aggregation function?

### Questions
Overall, I think the paper is a great contribution and would be a strong fit for ICLR. I am happy to recommend acceptance. Moreover, I would raise my score further and recommend a spotlight or oral ***if the following points are addressed***:
- Please make sure the text of the paper matches the contributions of the paper. The results in the paper speak for themselves; there is no need to claim this is the "first" suite to do many of these things (in particular with the comparisons against past agent frameworks)
- Please clarify how you selected the agents you picked for the analysis. What other agents did you consider? Why did you select this subset?
- Please update the text to clarify the contributions based on my concerns in the weaknesses section (differences in the number of tasks for benchmarks, aggregation of results across benchmarks, and more generally updating the clarity of the writing and presentation).

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Paper features an AstaBench - benchmark for evaluating scientific LLM agents on the variety of tasks. It integrates multiple benchmark for each task and consists not only with raw data, but with tools essential for testing. Benchmark can be used for testing agents on individual tasks (e.g. literature review or data analysis) as well as on end-to-end research.

### Strengths
- AstaBench provides a set of tools for testing agents, which makes it easier to setup experiments.
- Benchmark can be used for testing agents for specific tasks as well as for end-to-end research tests.
- AstaBench have a potential to become strong and stable benchmark for testing scientific LLM agents.

### Weaknesses
- Currently benchmark is heavily weighted for computer science and machine learning domains

### Questions
- Could you please provide at least brief description of scoring metrics in the main part of the paper? It's an important part of the benchmark, yet it's currently hidden in the appendix.
- It probably would be better to replace "AI" with "LLM" in title and abstract as benchmark made mainly for LLM agents, not for any AI agents (which also may refer to RL and other areas).

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper puts forward outstanding issues in the evaluation practices for AI science-specific agents. They introduce AstaBench, an evaluation suite that provides a holistic measure of agentic ability to perform scientific research.

### Strengths
- The paper does identify open issues in agent evaluations and provides a thorough analysis of the field. The paper provides a holistic evaluation of various models and scaffolds. 
- The set of benchmarks included in the analysis is broad and offers good coverage of the AI science benchmark domain
- The writing is clear and overall easy to parse.

### Weaknesses
- Treating tools as confounding variables only makes sense when we want to collapse agent evals to model evals. For measuring how well agents with the latest model-backbones can do on a task, we should compile each model with the best-possible scaffold around it. This often involves using the tools the agentic models was e.g. trained with natively (web search, code interpreter, etc.)
- Other confounders aside from standard tools have been motivated and described in previous work [1]
- The Pareto efficiency frontier should be convex because every point on the convex envelope can be achieved by mixing between two agents. 

[1] https://arxiv.org/abs/2407.01502

### Questions
Why are non-standard tools a confounding variable if we don't focus on comparing models but agent performance?

### Soundness
3

### Presentation
3

### Contribution
3
