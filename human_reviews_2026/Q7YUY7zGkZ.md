# WideSearch: Benchmarking Agentic Broad Info-Seeking

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
From professional research to everyday planning, many tasks are bottlenecked by wide-scale information seeking, which is more repetitive than cognitively complex. With the rapid development of Large Language Models (LLMs), automated search agents powered by LLMs offer a promising solution to liberate humans from this tedious work. However, the capability of these agents to perform such "wide-context" collection reliably and completely remains largely unevaluated due to a lack of suitable benchmarks. To bridge this gap, we introduce WideSearch, a new benchmark engineered to evaluate agent reliability on these large-scale collection tasks. The benchmark features 200 manually curated questions (100 in English, 100 in Chinese) from over 15 diverse domains, grounded in real user queries. Each task requires agents to collect large-scale atomic information, which could be verified one by one objectively, and arrange it into a well-organized output. A rigorous five-stage quality control pipeline ensures the difficulty, completeness, and verifiability of the dataset. We benchmark over 10 state-of-the-art agentic search systems, including single-agent, multi-agent frameworks, and end-to-end commercial systems. Most systems achieve overall success rates near 0\%, with the best performer reaching just 7\%. However, given sufficient time, cross-validation by multiple human testers can achieve a near 100\% success rate. These results demonstrate that present search agents have critical deficiencies in large-scale information seeking, underscoring urgent areas for future research and development in agentic search.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces WideSearch, a benchmark for broad, multi-entity, verifiable information-seeking by LLM agents. Each task pairs a query with a predefined table schema.
The agent is required to 1) identify the complete entity set and 2) fill attributes for each entity using web tools. 
A five-stage, human-in-the-loop curation pipeline ensures difficulty, verifiability, and temporal stability.
LLM-as-judge is used as an automated evaluator for semantically variable cells. 
Experiments on a few models show very low table-level success, while item-level F1 can be much higher with test-time scaling.

### Strengths
1. Clearly scoped evaluation setting: The paper frames `wide information seeking` as distinct from general deep research style tasks, emphasizing breadth, completeness, and structured outputs rather than deep synthesis of a single topic.
2. Rigorous curation: The five-stage pipeline is thoughtful and practical for benchmark validity.
3. Hybrid automatic evaluation that mixes exact or approximate rules with LLM-as-judge, including column types and primary keys for deterministic alignment. This is a careful table-centric evaluation.

### Weaknesses
1. Although 18 topics are covered, the distribution still shows heavier representation in Business/Finance/Arts & Culture. A sensitivity analysis showing robustness of conclusions across domains/languages would be useful.
2. To the best of my knowledge, Manus is the only product claiming wide research. Is there any result from that system?

### Questions
1. Fonts are too small for Fig.1 and Fig. 2
2. In line 476, note that the left quote is wrong.
3. Too many numbers in Table 1. Could you bold important numbers?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces WideSearch, a benchmark targeting “agentic broad information-seeking” tasks where an LLM agent must find a complete set of entities and fill a structured table from the live web (e.g., “all DOJ attorney openings in a date window”). The dataset contains 200 manually curated questions (100 EN / 100 ZH) spanning 15 domains, built with a five-stage curation pipeline to ensure breadth, verifiability, and temporal stability, and scored with a hybrid, table-alignment evaluation (rule checks + LLM-as-judge). Benchmarking 10 systems shows very low table-level success (most ≈0%; best single/multi-agent ≈7%), while humans achieve ~20% in single-annotator mode; the gap stems from strict completeness requirements rather than difficulty of single facts (item-level F1 can be high with retries). Multi-agent frameworks improve F1 via divide-and-conquer but still rarely achieve perfect tables, highlighting deficiencies in planning, reflection, and evidence use.

### Strengths
Overall, it's a very good work that extends the frontiers of evaluating newly emerged deep research systems and tools.

1. This paper proposes a novel dataset aimed at breadth and completeness across many entities. Rigorous five-stage human-centered curation ensures realistic and verifiable tasks.

2. Results expose a fundamental limitation of current agents—completeness at scale—and show multi-agent setups help but don’t solve it, offering a concrete target for future research.

### Weaknesses
1. The evaluation protocol might not be robust enough. This paper uses markdown format as a protocol, applies several fuzzy or exact matches to number/date/urls, and finally applies LLM-as-a-judge to evaluate complex answers. This paradigm replies on pre-crafted ground truths and would be sensitive to those queries whose answer might change with time (for example, the top-selling electrical toothbrush in the past week).

2. The current success criterion may be too brittle, which requires perfect table equality; small, arguably negligible deviations lead to failure, which can conflate usability with strict exactness.

3. Tasks are claimed “temporally invariant”, yet the benchmark relies on the live web; concrete procedures for refreshing ground truths and handling site changes are under-specified.

### Questions
1. What exactly do the source queries come from? Are they purely curated by human annotators or modified from existing data sources (like NaturalQuestions)? Are there any time-sensitive queries (for example, the top-selling electrical toothbrush in the past week)?

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
5

### Summary
The paper introduces WideSearch, a benchmark designed to evaluate search agents on broad information-seeking tasks, focusing on collecting atomic factual data across many entities that are objectively verifiable. The benchmark comprises 200 manually curated tasks (100 English, 100 Chinese). It also evalutes a broad set of LLMs and the LMs under agent frameworks.

### Strengths
The benchmark is unique from previous ones, which extend the commonly used browsecomp-style simple-answer questions to broad information gathering.

### Weaknesses
- Most importantly, even though the paper's motivation is very clear and reasonable, it remains a straightforward extension to the existing browsecomp-style benchmarks (from simple fact to a list of fact). Despite its potential usefulness (given the success of browsecomp), it's hard to admire from a research perspective. Also, it's still constrained to this very specific type of tasks, for the convenience of evaluation.

- Regarding the experiments: 
  - 1) the tasks appear to be too challenging for simple LLMs (with searches) because they usually only execute very limited search steps. This makes related analysis trivial to know. And it's also not necessary to call them as "single agent" 
    2) The tasks appear to be more suitable for Deep Research agents such as Deep Research from OpenAI/Gemini/Grok/Qwen/Kimi/.... But according to the content in the appendix, the evaluted systems are more or less from self-implemented agent scaffoldings, while leaving those important frontier deep research systems omitted? The paper could be much more benefited by including experiments on more frontier agents and their analysis.

- Writing and experiments: might as well include more details in the main content.

### Questions
- It's suggested to include a table of comparison to the recent deep research benchmarks on key dimensions.
- The paper would be benefited by including more evaluated agents, and at least OpenAI Deep Research, which is usually the leading agent across recent benchmarks.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces WideSearch, a benchmark designed to evaluate the reliability and comprehensiveness of LLM-based search agents in large-scale information-gathering tasks. WideSearch assesses agents on tasks requiring exhaustive and structured collection of numerous atomic facts across multiple entities. The benchmark consists of 200 manually curated queries (100 English, 100 Chinese) spanning 18 diverse domains, each paired with a well-defined schema and gold-standard table. 

A five-stage human-in-the-loop curation pipeline ensures that all tasks are complex, objectively verifiable, and reliant on real web tool usage. The evaluation framework integrates rule-based scoring with LLM-as-a-judge scoring, achieving over 97.8% agreement with human judgment.

The authors further perform an extensive benchmarking study across state-of-the-art agentic systems, revealing severe limitations in current models—particularly in maintaining completeness and fidelity at scale—highlighting the urgent need for more advanced planning, reflection, and evidence-grounding capabilities in next-generation agents.

### Strengths
(1) The paper tackles an underexplored yet practically crucial dimension of agent evaluation—broad, high-fidelity information gathering—which complements existing reasoning- and synthesis-oriented benchmarks. The conceptual framing of WideSearch as the “breadth” counterpart to DeepSearch and DeepResearch is clear, coherent, and well-motivated.

(2) The five-stage human-in-the-loop curation pipeline is rigorous and systematic, ensuring that all tasks are complex, verifiable, and genuinely dependent on real web search. The inclusion of both English and Chinese queries further enhances the benchmark’s generality and cross-lingual coverage.

(3) The evaluation and analysis are comprehensive and insightful. The experiments benchmark over ten single-agent, multi-agent, and end-to-end commercial systems using multiple aggregation metrics (Avg@N, Pass@N, Max@N), providing a well-rounded assessment of agent reliability at scale. The error taxonomy—separating advanced agentic failures (planning, reflection, evidence use) from basic operational errors—is particularly insightful. The test-time scaling analysis further isolates completeness, rather than retrieval accuracy, as the key bottleneck, offering a concrete and data-driven direction for future work.

### Weaknesses
The benchmark construction and evaluation are solid in general. It would be great if the authors could further conduct quantitative analysis regarding the major challenges and failure patterns mentioned in sections 4.1 and 4.2.

### Questions
(1) The paper reports 97.8% agreement between LLM-as-judge and human evaluation, which is strong. Could you provide more details on the specific disagreement cases? It would be great if the authors could provide more insights into this. 

(2) The benchmark relies on live web data. How do you plan to ensure its stability over time? Is there a plan for versioning or periodic re-validation as the web content evolves?

### Soundness
4

### Presentation
3

### Contribution
3
