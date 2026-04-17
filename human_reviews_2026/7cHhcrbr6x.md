# WebDS: An End-to-End Benchmark for Web-based Data Science

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Many real-world data science tasks involve complex web-based interactions: finding appropriate data available on the internet, synthesizing multimodal data from different locations, and producing summarized analyses. Existing web benchmarks often focus on simplistic interactions and often do not require diverse tool-using capabilities. Conversely, traditional data science benchmarks typically concentrate on static, highly structured datasets and do not assess end-to-end workflows that encompass data acquisition, cleaning, analysis, and insight generation. In response, we introduce WebDS, the first end-to-end web-based data science benchmark. It comprises 870 web-based data science tasks across 29 diverse websites from structured government data portals to unstructured news media, challenging agents to perform complex, multi-step, tool-based operations, across heterogeneous data formats, to better reflect the realities of modern data analytics. Evaluations of current SOTA LLM agents indicate significant performance gaps in accomplishing these tasks. For instance, Browser Use, which accomplishes 80% of tasks on Web Voyager, completes only 15% of tasks in WebDS, which our analysis suggests is due to new failure modes like poor information grounding, repetitive behavior and shortcut-taking that agents performing WebDS' tasks display. By contrast, humans achieve around 90% accuracy, highlighting a substantial gap between current agents and human performance. By providing a more robust and realistic testing ground, WebDS sets the stage for significant advances in the development of practically useful LLM-based data science.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents WebDS, a benchmark for evaluating LLM agents on realistic web-based data science tasks. It integrates 29 containerized websites and 870 tasks involving data retrieval, multi-hop reasoning, tool use, and report generation. Results show humans succeed ~90%, but state-of-the-art agents achieve <15%, revealing a major performance gap. The key contribution is providing a comprehensive and reproducible benchmark that mirrors end-to-end data science workflows.

### Strengths
The paper is highly original in proposing the first end-to-end benchmark for web-based data science, combining web navigation with real data analysis in a way not covered by prior benchmarks. It demonstrates strong quality and clarity by designing 29 containerized websites and 870 tasks across multiple domains, with clear task categorization that ensures reproducibility. Finally, its significance is clear as results reveal a striking human–agent performance gap, establishing WebDS as a critical benchmark for advancing research on practical, data-science–capable AI agents.

### Weaknesses
1. Limited Task Scope Beyond Data Science. Although the paper claims to be about Web Agents + Data Science, in reality the data science aspect is minimal, and the amount of required coding is very small.

2. Overly Simple Baseline Design. The paper does not propose its own unique baselines. Considering this is naturally a CLI + GUI hybrid setting, many approaches from WebArena or similar frameworks could have been adapted, but they were not included. Moreover, many closed-source agent frameworks and both open/closed-source products were ignored. Overall, the baseline coverage feels insufficient.

3. Lack of Multiple-Trial Evaluation (pass@n). For each example, it is unclear whether multiple trials were conducted to calculate pass@n. I suspect the low reported performance may be partly due to annotation ambiguity, which makes it hard for agents to succeed on a single attempt.

4. Subjective Task Evaluation Is Too Arbitrary. For open-ended tasks, evaluation is simply based on a 5-level scoring scheme. Ideally, one should measure agreement between humans and LLM-as-a-Judge. If a more reliable scoring method were adopted, I would increase my rating, but the current LLM-based scoring feels too random, and results may vary significantly across repeated runs.

### Questions
1. While WebDS tasks involve data retrieval and basic analysis, many real-world data science workflows include steps such as data cleaning, preprocessing, visualization, and modeling. Do the authors plan to expand WebDS to cover these stages, and how would they envision incorporating them into the benchmark ?
2. Why not adopt **execution-based evaluation**, which would be more realistic than the current rule-based approach?


Miss citations:

I strongly recommend that the authors cite more **data science–related work**, such as:

* *DSBench: How Far Are Data Science Agents from Becoming Data Science Experts?*
* *DA-Code: Agent Data Science Code Generation Benchmark for Large Language Models*
* *Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows*
* *Spider2-V: How Far Are Multimodal Agents From Automating Data Science and Engineering Workflows?*
* *DABstep: Data Agent Benchmark for Multi-step Reasoning*
* *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*

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
The paper proposes WebDS, a new benchmark to evaluate the end-to-end web-based data science capabilities of large language model agents. Unlike previous web agent benchmarks that focus on simple browsing, or data science benchmarks that focus on static datasets, WebDS integrates both dimensions — requiring agents to autonomously navigate, collect, analyze, and synthesize data from web-based environments.

### Strengths
This work fills a major gap in current evaluationno existing benchmark assesses full end-to-end data science workflows involving both web interaction and analytical reasoning, captures realistic web-based tasks that better reflect real-world data analysis behavior. The authors evaluates a wide range of SOTA models consistently and highlights key performance bottlenecks.

### Weaknesses
1. The subjective scoring relies on GPT-4o, creating potential evaluation circularity and bias toward similar model families. Including more human evaluations or open-source LLM judges would strengthen reliability.
2. While qualitative error categories are given, there is no quantitative breakdown of which task attributes (multi-hop, tool-use, multi-site) most contribute to failures.

### Questions
1. How do you plan to maintain or expand WebDS over time while ensuring reproducibility and preventing overfitting?
2. Could future versions include community-contributed tasks to diversify domains and evaluation scope?

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
This paper proposes a new benchmark for the web agent domain: WEBDS, which includes 870 data science tasks across 29 different websites. It also provides a dockerized implementation, ensuring stability and reproducibility of experiments. This work lays a foundation for the development of intelligent agents in the data science domain.

### Strengths
- The benchmark environment is very comprehensive, covering 29 websites, 10 domains, and 870 tasks. It focuses on the **data science domain**, evaluating the **entire data processing pipeline**.

- The test environment is **dockerized**, offering fixed environments, stable experiments, and reproducible results.

- The authors use **vision-language models** to analyze complete execution trajectories, providing **richer evaluation metrics** beyond success rate.

### Weaknesses
- **Lack of innovation:** The benchmark design does not significantly differ from existing mature web agent benchmarks. The main contributions remain in expanding the testing environment and task set.

- **Insufficient rigor:** Although using vision-language models for trajectory evaluation is common in GUI agent research, the validation experiments for scoring accuracy and stability (on a 1–5 scale) are rather cursory. The authors only mention comparing 50 human-evaluated tasks, yet the claimed 88% agreement seems questionable. Moreover, since most tested agents achieve **less than 10% success rate**, with most scores below 2, the majority of ratings are clustered around 1. Such a narrow score distribution weakens the credibility of the validation.

- **Limited model comparison:** The benchmark does not test more advanced GUI agents such as **UI-TARS** and **GPT-5**. Since most baseline models achieve accuracy below 5%, this raises the question of whether the benchmark is **too difficult**.

- **Lack of analysis on low success rates:** The paper merely lists potential reasons without detailed trajectory-based analysis.

### Questions
Please refer to the weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces WebDS, the first end-to-end benchmark for web-based data science workflows. It addresses the gap between existing benchmarks that focus either on simple web navigation or static data analysis. WebDS comprises 870 human-written tasks across 29 dockerized, data-rich websites from 10 domains (e.g., government data, news media). Tasks require complex, multi-step operations (data acquisition, synthesis, analysis, report generation) across heterogeneous sources. Evaluation of current SOTA LLM agents reveals a substantial performance gap: the best agent achieves only $13.2\%$ success (vs. $90\%$ human baseline), exhibiting novel failures like poor information grounding and failed repetition. WebDS is containerized for reproducibility and offers fine-grained evaluation across task attributes and difficulty tiers, positioning it as a new frontier for developing practical, end-to-end data science agents.

### Strengths
1. The benchmark's grounding in practitioner interviews and the inclusion of 29 diverse, data-rich websites ensures the tasks are highly realistic and demand complex generalization, not simple pattern matching.

2. The failure analysis (Table 5) provides excellent, fine-grained insights into unique failure modes not captured by simplistic metrics. Specifically, "Failed Repetition" (due to lacking state-checking heuristics) and "Poor Groundedness" (contradiction between perceived and latent knowledge) are critical problems for future research.

3. The evaluation structure allows for granular analysis across Domain-wise, Attribute-wise (7 labels), and Difficulty tiers (Easy/Medium/Hard), giving researchers precise targets for iterative model improvement.

### Weaknesses
1. The analysis and outputs primarily focus on text (reports, answers). Since the input sites are described as being rich in graphics and non-textual data, the benchmark's current focus may under-represent the full multimodal synthesis challenge (e.g., generating a visual chart or interpreting a complex image-based trend).

2. The action space is not fixed, allowing researchers flexibility, but the implementation relies on existing WebArena/BrowserGym abstractions. A brief discussion on whether these existing abstractions adequately capture the fine-grained data manipulation needed for the "Tool Usage" attribute would be beneficial.

3. The current distribution of tasks may slightly overemphasize QA tasks versus downstream action tasks. While the authors mention future work to include more action-based tasks, the current lack of balance might skew initial model optimization toward data retrieval over action execution.

### Questions
1. The benchmark includes Tool Usage as an attribute. Could the authors clarify which specific tools (e.g., Python code execution, SQL query, spreadsheet manipulation) were enabled for the SOTA agents during the 13.2\% success rate evaluation? Did the 13.2\% success rate fully integrate external code execution or primarily rely on the LLM's internal reasoning over extracted text?

2. The analysis highlights "Poor Information Grounding" as a key failure mode. Can the authors provide a more detailed example of how the "Groundedness" failure was scored? Is the model judged based on whether it extracts the correct data from the HTML (perception error) or whether it uses the correct data in the final report (synthesis/reasoning error)?

3. To maintain relevance and avoid saturation, the paper mentions a plan to "periodically refresh/expand the private test pool." Could the authors provide a brief plan or estimate of the expected frequency or size of this refreshment (e.g., quarterly, or annually adding 100 new tasks)?

### Soundness
4

### Presentation
3

### Contribution
3
