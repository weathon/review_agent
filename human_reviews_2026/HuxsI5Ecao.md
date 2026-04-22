# ReplicationBench: Can AI Agents Replicate Astrophysics Research Papers?

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 2, 8

## Abstract
Frontier AI agents show increasing promise as scientific research assistants, and may eventually be useful for extended, open-ended research workflows. However, in order to use agents for novel research, it is crucial to first assess the underlying faithfulness and correctness of their work. To evaluate agents as research assistants, we introduce ReplicationBench, an evaluation framework that tests whether agents can replicate entire research papers drawn from the astrophysics literature. Astrophysics, where research relies heavily on archival data and computational study while requiring little real-world experimentation, is a particularly useful testbed for AI agents in scientific research. We split each paper into tasks which require agents to replicate the paper’s core contributions, including the experimental setup, derivations, data analysis, and codebase. Each task is co-developed with the original paper authors and targets a key scientific result, enabling objective evaluation of both faithfulness (adherence to original methods) and correctness (technical accuracy of results). ReplicationBench is extremely challenging for current frontier language models: even the best-performing language models score under 20\%. We analyze ReplicationBench trajectories in collaboration with domain experts and find a rich, diverse set of failure modes for agents in scientific research. ReplicationBench establishes the first benchmark of paper-scale, expert-validated astrophysics research tasks, reveals insights about agent performance generalizable to other domains of data-driven science, and provides a scalable framework for measuring AI agents' reliability in scientific research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ReplicationBench, a benchmark designed to evaluate LLM agents as research assistants by measuring their ability to automatically reproduce results from astrophysical research papers, a research area where most of the day-to-day workflows are entirely computational. The authors curate 107 research-grade tasks from 19 papers and assess how well cutting-edge LLMs like Claude 4 Sonnet, Gemini 2.5 Pro, and OpenAI o3 perform when attempting to re-implement experiments based solely on paper text and supplementary material. The study reveals that current models struggle significantly with automation of reproducibility, achieving only 19% success even with multiple attempts. ReplicationBench aims to facilitate future work on automating scientific experimentation and offers a dataset, baseline models, and evaluation methodology for this new challenge.

### Strengths
1. The astrophysics area is well chosen to match this paper's purpose to evaluate LLM agents as research assistants. The collaboration with the original authors of the collected astrophysical papers also ensures the quality and authenticity of the collected tasks.

2. The grading strategy is carefully designed and well-illustrated, and typical error cases are shown and analyzed, making it easier to understand the bottleneck of current LLM agents in solving tasks in ReplicationBench.

### Weaknesses
1.  Although the automated grading metrics are carefully designed, some fields still seem to be too subjective. For example, how is the "tolerance" exactly assigned?

2. There lack of a detailed result analysis of comparison among different LLMs. For example, (1) even as reasoning models, o3/o4-mini still tend to generate much fewer tokens (~10k vs. 36k or 91k,) which takes much shorter runtime. Why? (2) Gemini 2.5 takes the longest runtime but results in the second-last performance (Table 2). Is there any useful conclusion that can be drawn from comparing different models' trajectories?

### Questions
1. What's the unit of 'Runtime' in Table2? Second, minute or hour?

2. How is 'Partial Credit' specifically defined?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents ReplicationBench, a benchmark of 107 astrophysics research tasks from 19 papers to evaluate LLMs and agents. The benchmark provides a suite of containerized environments and automated graders for the tasks to ensure evaluation reproducibility. The experimental results and qualitative analysis show that LLMs are still lagging behind human experts in conducting astrophysics research.

### Strengths
1. The paper ensures the proposed benchmark’s real-world usefulness by grounding the tasks on published papers and working with their expert authors to create the tasks and analyze LLM performance qualitatively.
2. The tasks in this benchmark take several important factors into consideration, such as coverage, objectivity, and guess-proofing.

### Weaknesses
1. A substantive assessment of the weaknesses of the paper. Focus on constructive and actionable insights on how the work could improve towards its stated goals. Be specific, avoid generic remarks. For example, if you believe the contribution lacks novelty, provide references and an explanation as evidence; if you believe experiments are insufficient, explain why and exactly what is missing, etc.
2. This benchmark focuses solely on astrophysics tasks. However, the paper does not clearly position itself in the related literature and justify what are the unique challenges and new insights are brought by this design choice to the community. The task collection process, evaluation results, and analysis are similar to those in existing benchmarks, such as CORE-Bench, DiscoveryBench, and ScienceAgentBench (the last related work is also not cited and discussed by the authors). Thus, it is unclear how ReplicationBench adds a new dimension to LLM benchmarking and evaluation research, and this evaluation of astrophysics tasks adds new findings or insights into existing LLMs and agents for (data-driven) science. 
3. The proposed “scalable data collection” method does not add much value to the paper, since its utility is not demonstrated in this paper by training or evaluating the LLMs. It is unclear why this part is included in the submission.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces ReplicationBench, a benchmark for testing LLM agents on replicating astrophysics papers. Each paper is decomposed into objectively gradable tasks with author-defined inputs, outputs, and numeric tolerances. Experiments show <20% success across frontier models.

### Strengths
1. Using actual astrophysics papers with real data and non-trivial tooling evaluates the natural scientific capabilities of an agent, unlike SWE-Bench or MLE-Bench.
2. Objectively gradable numeric and coding tasks make evaluation cheaper and reproducible.

### Weaknesses
1. The paper sort of positions ReplicationBench as the "first" at paper-scale replication, but works like PaperBench, DiscoveryBench, exist to evaluate agent capabilities across multiple domains. This paper is, in practice, an instance of that paradigm specialized to astrophysics. 2. A conference/journal for astrophysics could be more suitable for this paper.
It is not clear what the insight is here beyond that longer horizon, real-domain scientific tasks are hard for agents. This result has been repeatedly shown in multiple benchmarks.
3. There are data leakage concerns -- there is no evaluation or result on whether all necessary tokens were masked in the paper. It seems like a LLM was prompted to mask tokens based on relevance to the task, no results provided on accuracy of masking.
4. Reproducibility concern: detailed LLM sampling parameters, evaluation metrics are not explained. It is not clear how accuracy is computed.
5. Provide an analysis of different failure modes. How often do agents produce methodologically valid but numerically wrong outputs?

### Questions
See weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper contributes a benchmark for reproducing peer-reviewed astrophysics papers with ai agents. It collects 107 expert-authored tasks from 19 papers. A secondary dataset, ReplicationBench-Plus, adds 58 semi-automatically generated tasks. The paper finds that current frontier systems are only able to reproduce a small fraction of tasks.

### Strengths
- The paper peer-reviewed astrophysics papers with real datasets and collaboration with the authors to generate tasks
- Qualitative failure analysis with human experts
- Thorough evaluations in the experimental section report token runtime efficiency, error bars, and five frontier LLMs 
- Focus on Physics as domain going beyond ML.

### Weaknesses
- Simple scaffold used for agent implementation: I would like to see a more task-specific scaffold with optimized prompt and tooling to better understand the absolute performance claims we can make from the benchmark.
- The benchmark only includes tasks from a small number of papers. Overall number of tasks is also low with only 107/165. This produces large error bars. 
- Together with point 2: 20% of runs showed some signs of cheating/memorization by the agent of results in the paper. Adding more tasks and more papers would overall improve the robustness of the findings.

### Questions
- I would like to see a broader discussion and analysis of the qualitative failure modes with anectodal examples of what caused agents to not be able to reproduce work and how the authors assess the implications for using AI agents in science. For example, how feasible is it to solve the current bottlenecks the scaffold faces.

### Soundness
3

### Presentation
4

### Contribution
3
