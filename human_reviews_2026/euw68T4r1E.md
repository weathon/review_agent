# BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 8, 4, 2

## Abstract
Crowdsourced model evaluation platforms, such as Chatbot Arena, enable real-time
evaluation from human perspectives to assess the quality of model responses. In
the coding domain, manually examining the quality of LLM-generated content
is extremely challenging, as it requires understanding long chunks of raw code
and deliberatively simulating code execution. To this end, we introduce BigCodeArena, an open human evaluation platform for code generation back-ended
with a comprehensive and on-the-fly execution environment. Built on top of Chatbot Arena, BigCodeArena features to enable the execution of LLM-generated
code and allows humans to interact with the execution process and outcomes. We
collected over 14K raw code-centric conversation sessions across 10 widely used
LLMs, spanning 10 programming languages and 8 types of execution environments. Among these conversations, we identify more than 4.7K multi-turn samples
with pairwise human preference. Further analysis uncovers the underexplored
preferences of LLMs in fine-grained domains characterized by tasks, languages,
and frameworks. To systematically examine code understanding and generation capabilities of frontier LLMs, we curate two benchmarks based on the collected data,
namely BigCodeReward and AutoCodeArena. For BigCodeReward, we
postprocess the 4.7K conversations and evaluate the consistency between reward
models and human preference. The evaluation shows that most LLMs have superior performance in judging coding preferences when the execution results are
given. Inspired by the findings, we propose AutoCodeArena, an automatic Elo
rating benchmark designed to assess the coding quality of LLMs without humans.
We find that proprietary LLMs like GPT-5, Claude-Sonnet-4, and Claude-Opus-4
still lead the performance in code generation among the recent emerging models.
To democratize transparent evaluation of code generation in the wild, we aim to
establish BigCodeArena as a long-term project.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes an open crowdsourced evaluation platform for code generation that uses an on-the-fly execution environment.
This helps collecting more reliable human preference data, by mitigating errors from human judgment based only on static code review.

### Strengths
* Human preference collection: Execution feedback towardsmore robust human preference data. 
* Extensive collection: 10 languages and 8 execution environments with interactive debugging. 
* Benchmarks: BIGCODEREWARD and AUTOCODEARENA would help future research.

### Weaknesses
* Sensitivity to quality:  As its robustness heavily depends on the quality and consistency of human-provided feedback, more analysis to justify possible low-quality inputs from human (see questions below)
* Human cost: Collecting human preference data, especially across multiple languages and interactive environments can be expensive and time-consuming. Reporting time/infracost per each data point would be interesting
* Model train:  It remains unclear how effectively these signals could be integrated into training pipelines when sufficiently collected. Discussion whether this can be used for training better model would be useful.

### Questions
Any answer to weaknesses would help rebuttal
Specifically, how can we ensure reliability and consistency of human feedback?
How can we quantify or filter out low-quality feedbacks?
How sensitive are the evaluation metrics or model performance results to noisy feedbacks?
What is the average time and cost per annotation or preference collection, and whether automation can reduce cost.
How scalable to expand to new languages or execution contexts?

### Soundness
4

### Presentation
4

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
This paper introduces BigCodeArena, an open human evaluation platform for code generation, featuring real-time execution and interactive debugging environments. The authors collected over 14K code-centric conversation sessions across 10 large language models, including 4.7K multi-turn sessions with pairwise human preference labels.
Building on this dataset, the paper further proposes two benchmarks: (i) BigCodeReward — evaluating the consistency between reward model judgments and human preferences over 4.7K annotated conversations; and (ii) AutoCodeArena — an automated Elo-based benchmark for assessing coding quality without human evaluators.
The study highlights that (1) proprietary and open-source LLMs show comparable reliability in judging code quality, (2) execution feedback substantially improves preference alignment, and (3) GPT-5 achieves the best overall code generation quality among recent models.

### Strengths
- The work establishes a robust and transparent data collection framework for execution-based human evaluation. 
- The paper provides comprehensive design details on sandboxing, environment configuration, and preference aggregation, enhancing reproducibility.
- The proposed benchmarks (BigCodeReward, AutoCodeArena) form a meaningful step toward automated and scalable evaluation of code generation systems.

### Weaknesses
- The presentation could be improved for clarity and precision. Some parts, including figures and methodological descriptions, are confusing and would benefit from clearer exposition (see Questions).
- The analysis feels somewhat limited, particularly on the code generation side. While Elo-based comparisons are informative, a deeper breakdown of error types, execution failures, or qualitative behavior differences across models would strengthen the findings.

### Questions
- Section 3 states that annotators conducted "multi-turn conversations with at least two user–model exchanges,"" but it remains unclear whether these interactions occurred independently with each model or jointly within the pairwise evaluation. 

- In Figure 3, the meanings of All Data, Environment Matched, and Language Matched are insufficiently specified. It is unclear whether these settings refer to pair-sampling constraints during evaluation (i.e., which model pairs are compared) or to post-hoc averaging/grouping criteria applied when aggregating Elo scores.

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
-BigCodeArena builds on Chatbot Arena to collect a dataset of 4.7k preferences.
- From this data, the authors introduce BigCodeReward to evaluate the consistency between reward models and human preferences.
- Additionally, the authors also introduce AutoCodeArena to automate the process of preference collection.

### Strengths
- The paper tackles an important topic, addressing the need for reliable human evaluation of LLM-generated code.
- It is generally well-written and easy to follow. 
- The authors demonstrate multiple uses of the collected data, showing how human preferences can power both BigCodeReward and AutoCodeArena.

### Weaknesses
- The paper lacks actionable insights or design recommendations for practitioners—either developers using coding assistants or researchers building LLMs—reducing its practical significance. Relatedly, Several components of the work appear incremental relative to prior platforms such as Chatbot Arena and WebDevArena; similarly, AutoCodeArena resembles prior automated evaluation setups like Arena-Hard. The paper could better articulate what unique insights BigCodeArena contributes beyond integrating existing ideas or infrastructure.
- It is unclear how BigCodeReward handles noisy or inconsistent human preferences, which could affect the reliability of results.
- I could not find analysis or results on the response rate or completion statistics for the optional sub-questions, leaving unclear how representative these annotations are. 
- There is no comparison to existing leaderboards or ranking trends (e.g., Chatbot Arena), and the observation that frontier proprietary models remain strongest is unsurprising. At a cursory glance, there seems to be a strong correlation.

### Questions
Please address each of the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces BIGCODEARENA, an open and execution-based human evaluation platform for code generation models. Specifically, the platform is built upon Chatbot Arena, which integrates real-time code execution, interactive debugging, and pairwise preference collection to produce a more reliable measure of model performance. Moreover, the authors collect 14K code-centric conversations across 10 LLMs, 10 programming languages, and 8 execution environments, from which 4.7K multi-turn preference samples are curated.

### Strengths
1. The authors correctly identify a key limitation in existing human evaluation: humans often cannot judge code quality without execution. By integrating executable environments, this work provides a more realistic and reliable evaluation protocol.
2. The writing is good, and the paper is easy to follow.

### Weaknesses
1. The novelty is limited. While execution-based evaluation is impactful, the conceptual novelty mainly lies in combining existing ideas (Chatbot Arena + executable sandbox). The methodology may be viewed more as engineering integration than a new algorithmic or theoretical contribution.
2. AUTOCODEARENA relies on LLM-as-a-Judge (Claude-3.7-Sonnet), which may itself introduce bias. The paper lacks calibration or agreement analysis between automated and human judgments.
3. Although expert volunteers are mentioned, the annotation process (e.g., inter-annotator agreement, quality metrics, error analysis) is not deeply evaluated, which could raise concerns about reliability.
4. I think the topic of the paper is not well-suited for ICLR. The paper looks more like a platform description rather than a research paper with clear methodological novelty.

### Questions
The author should provide more insight into the paper, for example, whether we can use the analysis behind the platform to further improve the current LLMs. Or how we can better help humans to use LLMs to generate code.

### Soundness
2

### Presentation
4

### Contribution
2
