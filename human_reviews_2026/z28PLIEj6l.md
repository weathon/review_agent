# FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction

- Avg Score: 7.33
- Decision: Accept (Poster)
- Scores: 8, 6, 8

## Abstract
Future prediction is a complex task for LLM agents, requiring a high level of analytical thinking, information gathering, contextual understanding, and decision-making under uncertainty. Agents must not only gather and interpret vast amounts of dynamic information but also integrate diverse data sources, weigh uncertainties, and adapt predictions based on emerging trends, just as human experts do in fields like politics, economics, and finance. Despite its importance, no large-scale benchmark exists for evaluating agents on future prediction, largely due to challenges in handling real-time updates and retrieving timely, accurate answers. To address this, we introduce FutureX, a dynamic and live evaluation benchmark specifically designed for LLM agents performing future prediction tasks. FutureX is the largest and most diverse live benchmark for future prediction, supporting real-time daily updates and eliminating data contamination through an automated pipeline for question gathering and answer collection. We evaluate 25 LLM/agent models, including those with reasoning, search capabilities, and integration of external tools such as the open-source Deep Research Agent and closed-source Deep Research models. This comprehensive evaluation assesses agents’ adaptive reasoning and performance in dynamic environments. Our goal is to establish a dynamic, contamination-free evaluation standard that drives the development of LLM agents capable of performing at the level of professional human analysts in complex reasoning and predictive thinking.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces FutureX, a large-scale live benchmark designed to evaluate LLM agents on future-prediction tasks that require reasoning, information gathering, and decision-making. Unlike prior static benchmarks, FutureX uses a semi-automated pipeline that continuously crawls, curates, and resolves questions from 195 websites across 11 domains. The system automatically collects future-oriented questions, gathers LLM predictions at event start dates, and scores them after resolution, enabling a contamination-free, real-time evaluation. The benchmark assesses 25 models, from base LLMs to research agents (e.g., Grok-4, Gemini Deep Research), with human-expert comparisons.

### Strengths
- Contamination-free design: The design of the closed environment to prevent information contamination is well-executed and addresses a critical challenge in LLM evaluation.
- Comprehensive model evaluation: The study includes 25 models spanning reasoning, search, and deep-research agents with human expert baseline, providing a comprehensive evaluation landscape.

### Weaknesses
- The pipeline's automated expansion might favor scale over fidelity. No quantitative validation is provided to ensure the "extra" questions truly add value.
-  The paper lists 195 websites but will be good to include the full list of the selected websitesl. More details on domain balance, filtering reliability would help.
- The authors may consider including Brier score as most prior works in forecasting adopt this metric, which would facilitate comparison with existing literature.

### Questions
- What are the daily compute and maintenance costs for running FutureX, given the live crawling and 25-model evaluation pipeline?
- Could future versions compare LLM predictions with crowd forecasts (e.g. prediction-market aggregates) to test alignment with collective prediction?
- Will the full list of websites, question templates, and answer-extraction code be released for transparency and reproducibility?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FutureX, a large-scale, dynamic benchmark for evaluating the future prediction capabilities of LLM agents. The authors argue that existing benchmarks fail to test complex reasoning and decision-making under real-world uncertainty.

FutureX uses a fully automated pipeline that continuously gathers future-oriented questions from 195 diverse websites. It runs 25 different LLM agent models to get their predictions on a "start date." After the event's "resolution date," the system automatically finds the ground-truth outcome and scores the predictions. This live-updating design inherently eliminates data contamination, as the answers do not exist at the time of prediction.

The benchmark includes four difficulty tiers, ranging from simple single-choice questions (Level 1) to high-volatility, open-ended numerical and ranking tasks (Level 4). The paper's findings show that while agents with search tools (like Grok-4) perform best, all models struggle significantly with the harder tiers. Furthermore, all evaluated agents still perform substantially worse than a baseline of 40 human experts.

### Strengths
The primary strength of FutureX is its novel "live" evaluation design. By focusing on future events whose outcomes are not yet known, it provides a robust and scalable solution to the critical problem of data contamination that plagues static benchmarks.

The automated pipeline for question collection and answer verification is a significant technical achievement, ensuring the benchmark remains current and challenging.

Another strength is the logical stratification of tasks into four difficulty tiers. The results validate this structure, showing a clear performance drop as tasks move from simple retrieval (Level 1/2) to complex, open-ended reasoning under uncertainty (Level 3/4).

I like the the inclusion of a human expert baseline. It provides crucial context, grounding the model scores and highlighting the significant gap that still exists between current AI agents and human-level analytical reasoning.

### Weaknesses
As far as I understand, the benchmark's "prediction window." is only "one-week". This restricts the evaluation to short-term predictions, not long-term forecasting. It fails to test an agent's ability to reason about events months or years in the future, which is a different and critical skill for human analysts. I think this heavily limited this benchmark.

The evaluation metrics, while appropriate for correctness, do not capture the probabilistic nature of forecasting. The benchmark does not assess an agent's ability to express calibrated confidence (e.g., providing a probability or a confidence interval). It only scores the accuracy of a single-point answer unlike some recent works that assess the confidence too.

The comparison to human experts is a bit ambiguous. It is not specified whether the 40 human experts and the 25 LLM agents had access to the exact same information retrieval tools (I think this is very important as the author pointed out about potential leakage of data to models; such leakage could happen to human experts too), making the performance gap difficult to interpret fairly.

the benchmark tests the entire agent system at once. This makes it difficult to isolate the point of failure. When an agent performs poorly, it's unclear whether the fault lies with the underlying LLM's reasoning or with the agent's planning and tool-use framework.

### Questions
What are the computational and financial costs associated with maintaining this live benchmark? Its complexity might make it difficult for other researchers to reproduce or build upon.



How does the benchmark disentangle the performance of the base LLM from the performance of the agent framework? For example, are the failures of open-source agents due to poor planning logic or the base model's inability to follow the plan?

Are there plans to expand the benchmark's scope beyond the one-week prediction window to include long-term forecasting tasks, which require different reasoning skills than short-term information synthesis?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents FutureX, a large-scale, live benchmark for evaluating base LLMs and LLM agents on real-world future prediction tasks. It features a four-stage pipeline for event database construction, future event curation, agent prediction, and answer acquisition, with all stages executed daily to ensure dynamic, contamination-free evaluation across diverse domains and difficulty levels. Experiments with 25 models show that search- and tool-augmented agents outperform base LLMs but still lag behind humans on complex tasks, establishing FutureX as a scalable framework for assessing LLM reasoning in uncertain, real-world environments.

### Strengths
1. Future prediction is a good testbed for evaluating the capabilities of LLMs and agents in information gathering, reasoning, and predictive analysis, while naturally mitigating data contamination since ground-truth answers are not available yet at prediction time.

2. FutureX provides an automated and scalable pipeline for data construction, future event curation, and answer verification, offering good practical value with minimal manual effort.

3. The tiered event categorization in FutureX (Basic, Wide Search, Deep Search, Super Agent) enables systematic assessment of reasoning depth and tool usage in LLM agents.

4. The evaluation spans 25 models covering base, search-augmented, and tool-augmented agents, which offers comprehensive and comparative insights across different model classes.

5. The paper is clearly written and well-organized, with informative figures and tables.

### Weaknesses
1. Some details on human annotation are unclear. For example, the number of questions per category, whether human experts had access to all information sources or relied solely on their own knowledge, and the consistency of their answers on the same question. Such information would clarify the human–agent performance comparison and also serve as an indicator of the robustness and quality of the questions in the benchmark.

2. While the performance analysis across 25 models is comprehensive, the paper offers limited discussion on the diagnostic utility of FutureX. It is not clear whether the benchmark can help identify specific weaknesses in agents or provide actionable insights for improving their reasoning and prediction capabilities. Such discussion will further benefit the benchmark users and the development of new agents.

### Questions
1. In Line 252, it is mentioned that "the answer acquisition success rate exceeds 97%". How is this success rate calculated?

### Soundness
3

### Presentation
3

### Contribution
3
