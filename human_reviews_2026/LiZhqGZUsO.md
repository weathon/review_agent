# Understanding DeepResearch via Reports

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
DeepResearch agents represent a transformative AI paradigm, conducting expert-level research through sophisticated reasoning and multi-tool integration. However, evaluating these systems remains critically challenging due to open-ended research scenarios and existing benchmarks that focus on isolated capabilities rather than holistic performance. Unlike traditional LLM tasks, DeepResearch systems must synthesize diverse sources, generate insights, and present coherent findings, which are capabilities that resist simple verification. To address this gap, we introduce DeepResearch-ReportEval, a comprehensive framework designed to assess DeepResearch systems through their most representative outputs: research reports. Our approach systematically measures three dimensions: quality, redundancy, and factuality, using an innovative LLM-as-a-Judge methodology achieving strong expert concordance. We contribute a standardized benchmark of 100 curated queries spanning 12 real-world categories, enabling systematic capability comparison. Our evaluation of four leading commercial systems reveals distinct design philosophies and performance trade-offs, establishing foundational insights as DeepResearch evolves from information assistants toward intelligent research partners. We will release the prompts and data we use.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a benchmark and an evaluation pipeline for assessing DeepResearch systems in the research report generation task. The proposed benchmark consists of 100 queries across 12 categories. The evaluation pipeline is based on LLM-as-a-judge, with prompts tuned through a manual adjustment framework inspired by TextGrad. The authors evaluate DeepResearch in terms of the quality, redundancy, and factuality of the generated reports and claim that their LLM-as-a-judge-based pipeline achieves reasonable alignment with human evaluation, with 61.11% agreement in the task of ranking three reports.

### Strengths
* This paper introduces a new dataset consisting of 100 queries across 12 categories for evaluating DeepResearch on the report generation task.

* They propose a framework to manually improve prompts for LLM-as-a-judge, which is inspired by TextGrad. The idea is not academically novel, but the framework can be practically useful.

### Weaknesses
The proposed dataset is not substantially novel compared to existing datasets such as [1]. Therefore, I consider the evaluation pipeline to be the main contribution. However, I am not fully convinced that the proposed evaluation framework is substantially novel or offers clear advantages over existing methods.

* The rationale behind the criteria selection and categorization is unclear. In particular, Quality includes Comprehensiveness, Coherence, Clarity, and Insightfulness, which are largely different. It is unclear why Redundancy and Factuality are specifically isolated from these criteria.

* Doubt on the reliability of LLM-as-a-judge. I agree that redundancy and factuality can be evaluated without references. However, without prior information such as reference reports, it is not persuasive to assume that LLM-as-a-judge can reliably evaluate comprehensiveness, which is one of the important criteria. For such criteria, reference-based evaluation [1] would be more reasonable and reliable. This concern is consistent with the result in Table 1, which shows that the Quality MAD is relatively high.

* Novelty of the evaluation pipeline is not substantial. The proposed pipeline represents a standard and straightforward application of LLM-as-a-judge.

[1] Du et al. (2025). DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents. https://arxiv.org/abs/2506.11763.

### Questions
I expect responses to the points listed in the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a framework for evaluating  deep research systems by producing long research-style reports. The framework measures quality, redundancy, and factuality using an LLM-as-a-Judge. They benchmarked four commercial systems across 100 queries from 12 categories. They report low MAD between LLM and human scores  to show that their evaluation aligns with human evals.

### Strengths
- very timely problem
- studying tradeoffs between characteristics of reports vs evaluation metrics 
- Tend to also include  llm-human eval agreement 
- promised for public release of prompts and benchmark queries
- easy to read and follow

### Weaknesses
- The restriction to three metrics (quality, redundancy, factuality) omits other critical aspects such as relevance, novelty, interpretability, etc. I think it would be more exciting to focus on more novel aspects of evaluation.

- I might have missed it but it was not clear what is the soruce of 150 k real-world queries, how did they classify them?  some more details about the benchmark would be appreciated.

- I could not understand exactly how the 61.11 % agreement between llm and human was measured. If it is pairwise agreement, then random baseline would give 50% and thus 61% is not really significant agreement

- I am not sure that using MAD between LLM and human scores is the best way to calibrate alignment between human and model ratings. This metric emphasizes absolute score matching rather than relative ranking consistency, which might be the more meaningful signal in evaluation. For instance, the LLM and human annotators may operate on different scoring scales; yet, if their relative ordering of reports aligns well, that would better reflect evaluation reliability. In contrast, a system with a constant absolute deviation (e.g., always two points higher or lower than humans) could still achieve low MAD but fail to capture true ranking agreement.

- I think pairwise redundancy scoring can be exploited easily e.g., a nonsensical report with unrelated paragraphs could appear perfectly non-redundant.

- The factuality evaluation only tests claim–source overlap; it does not check whether unsupported claims exist, nor citation precision. 
Please check the following papers which have more accurate explanation of citation precision and coverage. 
https://arxiv.org/abs/2411.17375
https://arxiv.org/abs/2304.09848

- some open-source and task-specific DeepResearch frameworks are missing for example
-- Deepresearcher
- openscholar https://arxiv.org/abs/2411.14199
- LLMs + web search for example https://github.com/sentient-agi/OpenDeepSearch

- The framework’s calibration is dataset-specific; there is no evidence it generalizes to unseen queries. It would be useful to show how would calibration before and after alignment changes between llm and human annotation.

-  Section 4 provides general reflections  which could be briefly mentioned in intro. I did not find section 4 informative or necessary to keep in the main part of the paper

### Questions
- I am not sure if I missed this but how were the 150 k real-world queries obtained, filtered, and categorized?
- Why were only three dimensions (quality, redundancy, factuality) chosen? Why not additional ones  like relevance, novelty, etc?
- Who were the three human annotators?were they domain experts or crowdworkers? Since some topics are very niche, If they are not expert, it might not be a good idea to calibrate llm as a judge with their opinion.
- In page 5, what agreement metric does the 61.11 % refer to? Pairwise accuracy, Kendall tau? simple agreement? if it is simple agreement between 2 options, then the random results will lead to 50% agreement and then 61% is not really impressive. 
- How does the calibration process generalize beyond the 120 aligned reports? how did the scores changed before and after calibration? 
- Is redundancy metric gamifiable? If I have  non-coherent but dissimilar text (high non-redundancy but low quality)?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents a framework for evaluating AI research agents based on their final research reports instead of single tasks. It measures three aspects, quality, redundancy, and factuality, using LLM-as-a-Judge approach aligned with human experts. The authors benchmark four commercial systems (OpenAI, Perplexity, Gemini, Qwen) on 100 real-world research queries across 12 categories. Results show that Qwen and OpenAI perform best overall, producing reports that are more coherent, factually grounded, and insightful.

### Strengths
- The paper addresses the growing need to evaluate full research-capable AI systems, not just text generators or search agents.  
- The three dimensions (quality, redundancy, factuality) give a clear and balanced way to judge complex research reports.  
- The authors test multiple commercial DeepResearch systems on real-world queries covering 12 diverse categories, providing rich insights.  
- The paper discusses trade-offs (e.g., length vs. clarity) and highlights distinct design philosophies among systems, showing awareness of practical challenges.

### Weaknesses
- The framework only evaluates final reports and does not analyze intermediate reasoning steps or tool usage during the research process.  
- The evaluation focuses on AI-related and factual domains; it’s unclear how well it applies to creative or interdisciplinary research.  
- The framework doesn’t capture how agents perform over multiple iterations or collaborations with human researchers.

### Questions
- Real research often involves iteration and feedback. Could the framework be extended to evaluate multi-turn or collaborative research settings, where AI systems revise their reports based on critique or new data?
- The framework focuses mainly on assessing the final reports. It would be interesting to also evaluate intermediate steps, such as how the systems plan, search, and integrate evidence during the research process.

### Soundness
3

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
This paper proposes DEEPRESEARCH-REPORTEVAL, a framework for evaluating DeepResearch agents through their generated research reports. It builds a benchmark of 100 real-world research queries and scores reports from four commercial systems on quality, redundancy, and factuality using an LLM-as-judge aligned with human ratings.

### Strengths
1, The work studied here is intereting and the conclusion is valuable for real-world applications as LLM-generated reports are expected to exist more frequently in industry.

2, It designs a somewhat reasonable scoring scheme (quality, redundancy, factuality) with LLM-human alignment, showing strong correlation with expert judgments.

3, It provides a novel realistic benchmark of 100 research queries.

### Weaknesses
1, Although the paper iteratively aligns LLM-as-judge with human ratings, the evaluation pipeline does not explicitly control for superficial textual cues (e.g., length, citation count, formatting quality, bullet-point structure, narrative polish) that may inflate perceived report quality.
Those info may mislead the conclusion.
2, I would recommend to measure cross-LLM specificity, i.e., whether different DeepResearch systems produce meaningfully distinct reports for the same query. This can avoid the evaluation that rewards safe, popular, and consensus-shaped outputs instead of research originality or perspective depth.

### Questions
What's the agreement among human evaluators, and how close the LLM-human agreement to the human-by-human ones?

### Soundness
3

### Presentation
3

### Contribution
3
