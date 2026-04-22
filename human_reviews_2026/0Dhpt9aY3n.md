# A Benchmark for Deep Information Synthesis

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 6

## Abstract
Large language model (LLM)-based agents are increasingly used to solve complex tasks involving tool use, such as web browsing, code execution, and data analysis. However, current evaluation benchmarks do not adequately assess their ability to solve real-world tasks that require synthesizing information from multiple sources and inferring insights beyond simple fact retrieval. To address this, we introduce DEEPSYNTH, a novel benchmark designed to evaluate agents on realistic, time-consuming problems that combine information gathering, synthesis, and structured reasoning to produce insights. DEEPSYNTH contains 120 tasks collected across 7 domains and data sources covering 67 countries. DEEPSYNTH is constructed using a multi-stage data collection pipeline that requires annotators to collect official data sources, create hypotheses, perform manual analysis, and design tasks with verifiable answers. When evaluated on DEEPSYNTH, 11 state-of-the-art LLMs and deep research agents achieve a maximum F1 score of 8.97 and 17.5 on the LLM-judge metric, underscoring the difficulty of the benchmark. Our analysis reveals that current agents struggle with hallucinations and reasoning over large information spaces, highlighting DEEPSYNTH as a crucial benchmark for guiding future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DeepSynth, a very challenging benchmark for evaluating LLM agents. DeepSynth consists of 120 diverse tasks created by 16 experts, where each task requires an agent to navigate through about 4 web pages and read up to 15 documents and tables. The tasks are designed to reflect real-world analysis demand, and cover knowledge across 42 countries. The authors evaluated 5 state-of-the-art models (o4-mini, GPT-4.1, GPT-5, Gemini-2.5-Pro and DeepSeek-R1), as well as 3 state-of-the-art agentic frameworks (o3-deep-research, smolagents and OWL) on DeepSynth. It is observed that this benchmark is challenging for all these models and frameworks, with the best F1 score being only 8.97 points. The authors also conducted a bunch of ablation studies to understand the challenges in DeepSynth. They found models perform worse as the number of intermediate steps increases, and most failures are caused by either navigation error or synthesis error, which suggests future directions for developing LLM agents.

### Strengths
1. There lack good benchmarks for developing LLM agents. The DeepSynth benchmark in this paper will help accelerate the research and development of LLM agents.
2. The DeepSynth benchmark is designed to reflect real-world demand. The data curation looks reasonable and solid, though the dataset scale of 120 tasks may be a little bit small.
3. The benchmark is challenging enough even for state-of-the-art models and agentic frameworks. It reveals failures in existing LLM agents and shows substantial room for future improvement.

### Weaknesses
1. Information synthesis may not be an intuitive title for the community. To my understanding, what you meant by information synthesis is closely related to multi-step reasoning or agentic tasks. Maybe you can consider agentic tasks as your title?
2. There is a substantial gap between the F1 score and LLM judge score. In my opinion, this indicates that your ground truth answer may have different semantic forms, and string matching doesn’t work well there. Could you evaluate this issue with experiments, and provide some guidelines for those who will use your benchmark?
3. There isn’t a human baseline for this benchmark. A human baseline will let us understand whether the benchmark is solvable for average humans or require specific domain knowledge. I understand this may not be completed during rebuttal since each task requires 5.5 hours for humans on average, but it’s nice to have human baseline results later.

### Questions
1. In Figure 5, the space between types should be larger for better readability.

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
3

### Summary
This paper proposes DEEPSYNTH, a high-quality, well-designed benchmark comprising 120 challenging and diverse information synthesis tasks across 42 countries and 7 domains. The authors clearly articulate the motivation, methodology, data collection and curation process, followed by a detailed analysis of the dataset characteristics. The paper presents comprehensive experiments evaluating popular SOTA models and specialized agent frameworks, with detailed performance analysis revealing significant limitations in current systems.

Overall, this is a solid paper with rigorous benchmark design and high-quality implementation. The presentation effectively explains the design choices and underlying rationale. The work addresses an important gap in agent evaluation by focusing on information synthesis rather than simple fact retrieval. Recommend accept.

### Strengths
- Excellent observation regarding the lack of real-world tasks that require synthesizing information from multiple sources, with strong motivation for designing such a benchmark that is well-suited for advancing current LLM and agent systems which are becoming increasingly powerful and saturating existing benchmarks.
- Dataset collection and curation process is exceptionally well-designed, executed, and presented through a rigorous 4-stage pipeline (data source identification, hypothesis generation, validation, task formulation), enabling readers and users to easily understand the characteristics and quality standards of the benchmark.
- The experiments with SOTA models are comprehensively designed and the analysis and ablations are thoroughly conducted, unveiling the significant incapabilities of current models (best F1 score only 8.97) and underscoring the substantial value and challenge posed by this benchmark.
- The findings derived from the dataset creation process and experimental analysis are insightful and well-supported with quantitative evidence.

### Weaknesses
- Overly difficult questions might limit the practical applicability of the benchmark, making evaluations of less powerful models even more challenging. For example, if less powerful models all achieve nearly zero performance, it becomes difficult to obtain meaningful signals to differentiate and evaluate which model performs better. In comparison, benchmarks with graduated difficulty levels generally suffer less from this limitation.
- Since the benchmark requires gathering information from the dynamic Internet that constantly changes, some websites may become unavailable over time (e.g., pages/tables expire or are restructured). This could make certain tasks inaccessible or make long-term maintenance of the benchmark particularly challenging, potentially affecting reproducibility and consistency of evaluations.

### Questions
- Regarding the inherently challenging nature of the benchmark, what are the authors' thoughts on improving the applicability so that less powerful models can also derive meaningful evaluation signals from the benchmark? Would incorporating graduated difficulty levels be beneficial?
- Although one design principle emphasizes 'robustness against memorization,' once the benchmark becomes public, won't it inevitably face contamination issues? How do the authors plan to prevent models from over-optimizing specifically for this benchmark, which could cause unfairness when comparing against other models that haven't been exposed to these tasks?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents DEEPSYNTH, a benchmark of 120 tasks for evaluating LLM agents on multi-source information synthesis across 7 domains and 42 countries. It uses a multi-stage expert-driven pipeline for task creation and shows SOTA agents (e.g., GPT-5, o3-deep-research) achieve low F1 scores (max 8.97), exposing gaps in reasoning and tool use.

### Strengths
- Multi-stage pipeline ensures non-memorizable, diverse tasks.
- Covers 9 models/agents with metrics like F1/EM and ablations.
- Breaks down performance by steps/operations.

### Weaknesses
- Lacks full code/prompts, data promised post-acceptance.
- Annotator selection biased; uneven regional coverage.
- Overlaps with prior benchmarks without direct comparisons.
- Low baselines may reflect poor prompting.

### Questions
- Can you provide inter-annotator agreement metrics (e.g., Cohen's kappa) for hypothesis validation and task formulation to address subjectivity concerns.

### Soundness
2

### Presentation
3

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
This paper introduces DeepSynth, a benchmark that evaluates agent on realistic and time consuming problems. DeepSynth covers 120 tasks across 7 domains and data sources covering 42 countries.  Each task requires agents to navigate ~4.2 web pages, and read between 1 to 15 context documents. The benchmark has been proven to be challenging even by state-of-the-art models like GPT-5 and Gemini-Pro-2.5 and agentic systems like o3-deep-research and smolagent, showing DeepSynth can serve as an arena of evaluating the capabilities of agentic systems.

### Strengths
- The construction process of DeepSynth is rigorous. It consists of five steps: (1) data source identification, (2) hypothesis generation, (3) hypothesis validation, (4) task formulation, and (5) data validation. All steps is done by expert annotators. The demographic information of the annotators & average time consumption are also reported in this paper.
- DeepSynth is challenging for many state-of-the-art models and systems, showing huge gap between human capabilities and current models.
- The analysis section gives interesting insights, and also provides insights on how the agents fail, whether the groudtruth intermediate steps will help, and how the agent behaves across tasks from different regions.

### Weaknesses
- I understand the benchmark is challenging. However, the evaluation metrics are hard to interpret (especially EM and LLM Judge Score, see the questions section). The value of EM is almost all 0s and LLM Judge Score might favor their own model family (GPT-4.1).
- (minor) The annotator demographic might also be biased (75% male, 81.25% PhD).

### Questions
- Is EM reliable as a metric? The correlation between EM score and LLM Judge Score are not strong. Is there some examples where the EM is low but the LLM Judge Score is high & EM high but LLM Judge score low?
- Is the model used for LLM Judge Score GPT-4.1 (the same in Wolfson et al. (2025))? In this case, is it likely that it is biased towards its own generation? How do we interpret that Smolagent (GPT-4.1) has higher LLM Judge Score than Smolagent (GPT-5), but F1 scores show otherwise.
- Thank you for providing the time estimation! What are the approximate cost of compute for running different agents on this benchmark?
- Is there an inter-annotator agreement rate for hypothesis validation?
- Is the human performance 100 among experts since DeepSynth only include tasks where answers from both annotators were identical?
- With only 120 tasks, what are the confidence intervals on the
reported scores?

### Soundness
3

### Presentation
3

### Contribution
3
