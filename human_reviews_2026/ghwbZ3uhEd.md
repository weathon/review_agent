# LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
Deep research---producing comprehensive, citation-backed reports by searching across hundreds of live websites---marks an important frontier for agentic systems. To rigorously evaluate this ability, three principles are essential: tasks should be (1) user-centric, reflecting realistic information needs, (2) dynamic, requiring up-to-date information beyond parametric knowledge, and (3) unambiguous, ensuring consistent interpretation across users. Existing benchmarks fall short of these principles, often focusing on narrow domains or posing ambiguous questions that hinder fair comparison. Guided by these principles, we introduce LiveResearchBench, a benchmark of 100 expert-curated tasks spanning daily life, enterprise, and academia, each requiring extensive, dynamic, real-time web search and synthesis. Built with over 1,500 hours of human labor, LiveResearchBench provides a rigorous basis for systematic evaluation. To evaluate citation-grounded long-form reports, we present DeepEval, a comprehensive suite covering both content- and report-level quality: checklists for coverage and presentation, rubric-tree assessments of citation accuracy and traceability, and metrics for consistency and depth of analysis. Using LiveResearchBench and DeepEval, we conduct a comprehensive evaluation of frontier deep research systems, including single-agent web search, single-agent deep research, and multi-agent systems. Our analysis reveals current strengths, recurring failure modes, and key system components needed to advance reliable, insightful deep research. Our code is available at: https://github.com/SalesforceAIResearch/LiveResearchBench.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents LiveResearchBench, a new 100-task benchmark for "deep research" agents, and an evaluation suite, DeepEval. The authors use this framework to test 18 different systems, offering some useful insights into their common failure points. 
Overall, this is a valuable and thoughtfully designed contribution to the field.

### Strengths
The paper’s main strength is its design philosophy. The authors do an excellent job motivating why this kind of evaluation needs to be user-centric, dynamic, and unambiguous, and they've built the tasks accordingly. The benchmark itself feels realistic, with tasks that specify an audience and output format while requiring up-to-date information, which is a convincing improvement over prior work. 
The breadth of the DeepEval suite is also good. It combines high-level checks with more granular metrics, like checklist-based coverage and a rubric-tree for citation validation, which seem to align well with human judgment. Evaluating 18 systems provides a solid snapshot of the current landscape and surfaces recurring errors that will be helpful for the community.

### Weaknesses
My main issue is the evaluation pipeline's heavy reliance on LLMs for generating checklists and judging outputs. While the authors took steps to mitigate bias, it's still a risk. I'd be more convinced by more quantitative stats on model-human agreement or some ablation studies on the judging process.

### Questions
1. How do you define `wide information search`? Any idea related to the existing Wide Research?
2. How did you handle potentially harmful queries or unreliable web sources during the evaluation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces LiveResearchBench, a benchmark designed to evaluate deep research agents that produce comprehensive, citation-grounded reports through multi-source web investigation. The benchmark consists of 100 expert-curated, realistic, and dynamically updating tasks across domains such as daily life, enterprise, and academia, constructed under three principles: user-centricity, dynamism, and unambiguity. Alongside it, the authors propose DeepEval, a multifaceted evaluation suite assessing report quality through coverage, presentation, citation accuracy, and analytical depth. Empirical evaluation across single- and multi-agent setups highlights limitations in current deep research capabilities.

### Strengths
1. The experiment is extensive, including 17 deep research agents' performance on the proposed benchmark.
2. The finding that existing deep research agents struggle with citation reliability and analytical depth is useful.
3. The checklist design is interesting.

### Weaknesses
1. The novelty is not enough and the description of related work is not precise. E.g., Mind2Web2 is ambiguous and focuses on short answers, etc. This makes the distinction from the existing work not convincing.

### Questions
1. The clearly defined specific questions are not practical. In reality, are people always asking such specific questions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes **LiveResearchBench**, a benchmark for evaluating *Deep Research* agents. It comprises **100 expert-curated, time-sensitive, multi-faceted tasks**, each specified with a clear audience, scope, and expected output to minimize ambiguity. The authors also present a comprehensive evaluation suite (DeepEval) covering six dimensions that span both report-level and content-level assessment. The paper evaluates 18 agent systems and reports a range of observations about the current agents.

### Strengths
- While there are a surge of new deep research benchmarks, this work remains different from vey recent works (e.g., unambiguous and time varying research tasks). 

- The paper includes a user study and human-agreement check for parts of the evaluation pipeline. These are important for the benchmark design as well as the reliability of the evaluation.

- Comprehensive evaluation on most existing frontier systems.

### Weaknesses
Some high-level points:

- Although the benchmark aims to evaluate open-ended research and report generation, the tasks still appear biased to tasks that are over-prescribed (explicit audience, formatting requirements, and tight scopes). This is understandable for the convenience of the evaluation, but still undermines its claimed scope.
- The evaluation is central to the contribution, yet the paper does not clearly explain how each of the six metrics was designed, what prior work they extend, and why these particular choices were made. 
- The evaluation heavily depends on LLM-as-a-Judge, which could make it both expensive and unreliable. Given this, the details of human agreement studies should be further detailed in the paper.
- The paper defines six metrics but the main table averages only on four of them? Why omit **Citation Accuracy** in the main comparison when accurate grounding is arguably the linchpin for user need? This is particularly concerning given the text itself notes that models struggle most with **citation correctness/formatting**. Even more worrying, the method that tops the main table (Open Deep Research) is reported to have extremely high overall error rates on some tasks

Regarding the individual metrics:

- Report-presentation metric may have overweighted style and citation formatting (and actually many of them are directly related to citation formats). If the presentation score is driven by citation entries, duplicates, or formatting, it's very likely an agent can overfit to style easily. Might as well elaborate more about how the 10 error patterns were selected and discuss the risk of RL/finetuning leading to superficial gains on these patterns.
- Factual & Logical Consistency at scale. Longer documents are more likely to contain contradictions, but they also become harder for an LLM judge to evaluate reliably. This may be a challenge for the judge model.
- Analysis Depth: Looking at the prompt, it looks too heavy for the LMs to reason accurately over all the points.
- Citation-related two: 
  - It is **unclear how claims are extracted** from long reports and how accurate this extraction is (likely another LLM call; but now the writing in the main text, Appendix D, the figure are not clear enough). This step is crucial and should be evaluated itself given that the extraction could have been blocked by the websites.
  - The **rubric-tree framing** of citation accuracy feels unnecessary for what is essentially a simple three-step pipeline

Writing:

- Writing should be further polished. Some details are hard to comprehend and some details in the appendix appear to be made in a rush. The references are also loose and arbirtrary.
- In the relatde work section, the "Evaluation of Long-form Answer" part should discuss more about how previous (search-related) benchmarks handled long-form answers.

### Questions
- (Some questions are already mentioned in the weakness section)
- The paper mentions a very large time investment (1500 hours), but it is not clear what is included (task ideation, rubric creation, judge prompts, code, pilot runs, re-annotation, etc.). A breakdown would help much to understand the number.
- DeepEval appears to involve many LLM calls. What is the end-to-end cost and latency to evaluate a single system on the full benchmark? 
- Will the benchmark be fully opensourced? (including the tasks, task-specific checklists, code, etc.)
- I'd suggest including a summary table (ideally in the main paper) contrasting LiveResearchBench with representative benchmarks across key dimensions. The current writing is not clear enough and will be hard for people to comprehend.
- OpenAI Deep Research seems more like a multi-agent system, given the information revealed?

### Soundness
3

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
This paper addresses LiveResearchBench, a benchmark of 100 expert-curated, time-varying tasks designed around four principles: user-centricity, unambiguous scope, temporal dynamism, and multi-faceted reasoning. The authors further propose DeepEval, an evaluation suite measuring presentation, factual consistency, coverage, analytical depth, and citation reliability through rubric-tree scoring and ensemble LLM-as-a-Judge evaluation. Evaluations on 18 single- and multi-agent systems show existing models excel at information collection but fall short in analytical depth and citation faithfulness.

### Strengths
(1) The paper is well-motivated by the lack of realistic, dynamic benchmarks for deep research. LiveResearchBench is constructed with clear design principles—user-centric, unambiguous, time-varying, and multi-faceted—and backed by a solid six-stage curation and five-stage verification process. The use of clarification questions, checklist-based unit tests, and expert validation ensures task clarity and evaluation consistency.

(2) DeepEval provides a six-dimensional framework that goes beyond accuracy to assess coverage, analytical depth, and citation reliability. The rubric-tree and checklist design make LLM-as-a-Judge both scalable and interpretable, achieving strong alignment with human evaluation.

(3) The evaluation and analysis are comprehensive, offering insights into the current bottlenecks and future directions for deep research agents.

### Weaknesses
One of the benchmark’s core strengths—its dynamic, time-varying nature—also poses practical challenges. Time-sensitive webpages and evolving content make long-term maintenance difficult. It would be helpful if the authors could provide evidence or analysis showing the benchmark’s stability and validity over time.

### Questions
(1) How stable are the system rankings across repeated runs or different judge ensembles? 
(2) Could you provide inter-judge agreement or human–LLM agreement for each evaluation dimension separately?

### Soundness
3

### Presentation
3

### Contribution
3
