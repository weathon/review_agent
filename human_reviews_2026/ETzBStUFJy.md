# USTBench: Benchmarking and Dissecting Spatiotemporal Reasoning Capabilities of LLMs as Urban Agents

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Large language models (LLMs) have shown emerging potential in spatiotemporal reasoning, making them promising candidates for building urban agents that support diverse urban downstream applications. Despite these benefits, existing studies primarily focus on evaluating urban LLM agent on outcome-level metrics (e.g., prediction accuracy, traffic efficiency), offering limited insight into their underlying reasoning processes. As a result, the strengths and limitations of urban LLM agents in spatiotemporal reasoning remain poorly understood. To this end, we introduce USTBench, the first benchmark to evaluate LLMs’ spatiotemporal reasoning abilities as urban agents across four decomposed dimensions: spatiotemporal understanding, forecasting, planning, and reflection. Specifically, USTBench supports five diverse urban decision-making and four spatiotemporal prediction tasks, all running within our constructed interactive city environment UAgentEnv. The benchmark includes 62,466 structured QA pairs for process-based evaluation and standardized end-to-end task assessments, enabling fine-grained diagnostics and broad task-level comparison across diverse urban scenarios. Through extensive evaluation of fourteen leading LLMs, we reveal that although LLMs show promising potential across various urban downstream tasks, they still struggle in long-horizon planning and reflective adaptation in dynamic urban contexts. Notably, recent advanced reasoning models (e.g., DeepSeek-R1) trained on general logic or mathematical problems do not consistently outperform non-reasoning LLMs. This discrepancy highlights the need for domain-specialized adaptation methods to enhance urban spatiotemporal reasoning. Overall, USTBench provides a foundation to build more adaptive and effective LLM-based urban agents and broad smart city applications. Our project is available at https://github.com/usail-hkust/USTBench.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces USTBench, a large‐scale benchmark for evaluating the spatio-temporal reasoning abilities of large language models (LLMs) acting as urban agents. USTBench is built on an interactive environment (UAgentEnv) that simulates nine realistic urban prediction and decision-making tasks (e.g., congestion forecasting, traffic-signal control, urban planning). It decomposes reasoning into four processes (understanding, forecasting, planning, and reflection with feedback) and provides 62,466 QA pairs plus standardized end-to-end task metrics. Thirteen state-of-the-art LLMs (generalist and reasoning-enhanced) are benchmarked, revealing strong performance in understanding/forecasting but persistent weaknesses in long-horizon planning and reflection.

### Strengths
- First dataset to systematically dissect spatio-temporal reasoning in urban-agent settings and to evaluate a “reflection-with-feedback” capability missing from earlier city-scale benchmarks. Also explicitly compares with STBench, CityBench, CityGPT, and UrbanPlanBench, highlighting gaps such as process-level metrics and long-horizon planning evaluation.
- Covers 9 heterogeneous tasks, 5 data modalities, and ~62k QA pairs, providing both breadth and depth for evaluation.
Benchmarks 13 state-of-the-art LLMs across four reasoning facets, plus ablations on reflection, yielding actionable insights for model developers.
- Demonstrates that general reasoning post-training does not fully transfer to complex urban domains, motivating domain-adaptive techniques.
- Detailed ablation studies and examination of failure cases offer valuable insights into the spatio-temporal reasoning challenges of current LLMs.

### Weaknesses
- The study would be strengthened by an explicit modality ablation or corruption experiment to verify which data sources (e.g., socio-economic vs. POI) are most influential in reasoning outcomes.
- The related work section could be strengthened. In particular, the authors may consider discussing the following relevant works, which could help better situate the current contribution within the existing literature: 
   - Urban Agent [1, 2, 3]

[1] Wang, Jiawei, et al. "Large language models as urban residents: An llm agent framework for personal mobility generation." Advances in Neural Information Processing Systems 37 (2024): 124547-124574.

[2] Li, Bowen, et al. "LogiCity: Advancing neuro-symbolic ai with abstract urban simulation." Advances in Neural Information Processing Systems 37 (2024): 69840-69864.

[3] Jiang, Yue, et al. "Urbanllm: Autonomous urban activity planning and management with large language models." arXiv preprint arXiv:2406.12360 (2024).

### Questions
Mentioned in weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper integrates several scenarios related to urban spatio-temporal understanding and their datasets for LLM Agent. Based on this, this paper further splits the process of solving the above scenarios using LLM Agent into four dimensions including  spatiotemporal understanding, forecasting, planning, and reflection, and then designs USTBench to evaluate LLM performance according to each of these four dimensions, as well as to provide an end-to-end overall performance evaluation.

### Strengths
1. The idea of splitting the LLM agent into multiple key dimensions and benchmarking them independently proposed in this paper helps to help researchers understand the role played by each dimension in the whole agent autonomous execution process in a more refined way, and can deepen the understanding of the LLM capability boundary.
2. In this paper, a significant amount of work has been invested in completing the processing of raw scenarios and datasets to QA pairs that can be used for benchmarking.
3. This paper gives the review results and analysis of commonly used LLMs, which helps researchers to quickly understand the existing LLMs in the field.

### Weaknesses
1. This paper considers LLM agents as spatiotemporal understanding, forecasting, planning and reflection. The statement is not given any reason in this paper so that I find it very confusing. This seems to be a claim forced by the authors to match the benchmark task they designed. Therefore, a reasonable derivation to show that agents used for urban spatio-temporal reasoning tasks include mainly and only these aspects of capabilities is necessary and will complete the logical chain throughout the text.
2. Following the weakness above, Figure 2 also shows a worrying imbalance, with forecasting, planning, and reflection occupying only a relatively small portion of the overall performance radar graph, while the rest is all about the spatial capabilities and temporal capabilities of the LLM. This seems to indicate that the benchmark proposed in this paper does not focus on or is not capable of assessing the aforementioned high-level capabilities of LLM in a holistic manner, but is only a repetitive assessment of the spatio-temporal comprehension capabilities of LLM mainly (such work already exists).
3. In the table in the experimental section, the different models are split by horizontal lines to divide them into groups, but there is no explanation for this, which is equally confusing.

### Questions
1. Why are LLM Agent capabilities for urban spatio-temporal understanding limited to four dimensions?
2. How representative is the LLM chosen for this paper? How was the list of models reviewed determined for this paper? There seems to be a limited number of closed-source LLMs here, with a higher number of DeepSeek-R1 variants.
3. How are the lists of LLMs included selected in Figure 4 and Figure 5, especially Qwen2.5-7B vs. Qwen2.5-32B used for comparison in Figure 5?
4. Why are Table 4/Figure 4 and Table 5/Figure 5 so close? This gives the false impression that the tables and figures are related content.
5. With this benchmark, is it possible to quantitatively assess the impact of a single dimension of performance on an end-to-end LLM agent? Or are there deeper and more comprehensive insights beyond reflection to reflect the “DISSECTING” in the title.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes USTBench, an evaluation benchmark to systematically measure large language models’ (LLMs) spatiotemporal reasoning abilities in urban environments. Built on an interactive environment called UAgentEnv, the benchmark covers four reasoning facets: spatiotemporal understanding, forecasting, planning, and reflection, and supports both fine-grained (process-based) and end-to-end task evaluation. The authors argue that existing urban LLM benchmarks mostly rely on outcome-based metrics (e.g., traffic efficiency, prediction accuracy) and thus may hide reasoning deficits; USTBench instead makes the intermediate reasoning step explicit and measurable, showing, for example, that reasoning-style LLMs like DeepSeek-R1 do not always outperform strong non-reasoning LLMs such as GPT-4o on urban tasks. The paper also releases datasets (62k+ structured QA pairs), environment configs, and scripts to support reproducibility.

### Strengths
1. Timely and well-scoped problem. Urban LLM agents are an emerging but under-evaluated direction. Focusing on spatiotemporal reasoning (not just traffic, not just planning) makes the benchmark conceptually coherent.

2. Process-based evaluation is clearly motivated. The paper illustrates, with concrete cases (e.g. congestion trend vs prediction), that outcome-only metrics can give a misleading ranking of reasoning vs non-reasoning models. This is an actual pain point in current LLM-for-planning work.

3. Comprehensive baselines. They evaluate both non-reasoning and reasoning LLMs, including recent RL-on-reasoning models (DeepSeek-R1, QwQ, GLM-Z1, o4-mini, GPT-4o), which makes the claim “general reasoning does not always transfer to urban tasks” convincing.

### Weaknesses
1. Ground-truth construction and label fidelity need more quantification. Many QA instances are generated via interactions with UAgentEnv, but the paper does not report inter-annotator agreement or error bounds for the simulation-derived “optimal” actions. For planning QAs, the “exhaustive search” over horizon H could itself be suboptimal or environment-specific.
2. Evaluation still has a strong API/hardware assumption. The runtime table shows some models become very slow (e.g. DeepSeek-R1 via Alibaba API), which limits the practical use of the benchmark, but the paper doesn’t analyze how this affects comparability across models.
3. Numerous grammatical errors: L107 LLMs excels; L322 Reasoning models achieving...
4. The paper repeatedly claims that “UAgentEnv is an interactive environment for urban agents,” but in fact, the entire environment consists only of static JSON tasks (in QA format), without any state transitions, action feedback, or interactive loops.
5. Tasks are single-step question-answering without any decision–feedback loop.
6. Amap data is not an open API and requires a commercial license.
7. The evaluation of CoT and few-shot results is lacking.

### Questions
None

### Soundness
3

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
The paper introduces USTBench, a novel benchmark designed to evaluate large language models as urban agents, focusing on their ability to reason about and act within dynamic urban environments. It proposes a comprehensive framework that decomposes urban reasoning into four stages: spatiotemporal understanding, forecasting, planning, and reflection. The benchmark is built around an interactive city simulation (UAgentEnv) based on real-world data and includes 62,466 QA pairs for fine-grained evaluation. The paper shows that while LLMs excel at spatial and temporal understanding, they struggle with long-term planning and reflection, highlighting the need for domain-specific adaptation. By providing detailed insights into each reasoning stage, USTBench aims to advance LLMs' applications in urban systems and other dynamic domains.

### Strengths
The paper presents a novel and comprehensive benchmark, USTBench, for evaluating large language models as urban agents, marking an original contribution in the field of AI for urban systems evaluation. The contributions are listed below:
1. It decomposes of urban reasoning into four distinct capabilities (spatiotemporal understanding, forecasting, planning, and reflection), which allows process-based diagnostics instead of only outcome-level evaluation. This modular view introduces a new problem formulation that bridges urban computing and cognitive assessment of LLMs.
2. The benchmark integrates real-world urban datasets, nine realistic urban tasks, and an interactive simulation environment (UAgentEnv). The dataset has over 62,000 structured QA pairs, and is meticulously designed to isolate and measure reasoning sub-skills, enhancing the reliability of evaluation. The experimental section is extensive, comparing multiple state-of-the-art models and providing consistent metrics for each reasoning stage.
3. The paper is well-organized and logically presented. It effectively motivates why decomposing reasoning is critical for diagnosing model weaknesses. Figures and tables are intuitive, and the design of each reasoning subtask is clearly justified.
4. Urban intelligence is an emerging but crucial domain for LLM applications, and the proposed benchmark fills a clear gap by providing standardized, fine-grained evaluation protocols.

### Weaknesses
While conceptually strong, the paper has several areas for improvement:
1. Limited coverage of real-world complexity. Although UAgentEnv incorporates multiple tasks, it still simplifies many aspects of urban systems (e.g., policy constraints, multi-agent interactions). The lack of uncertainty modeling may limit generalizability to truly dynamic urban settings.
2. Evaluation breadth. Experiments cover a good range of models, but the paper lacks ablation or fine-tuning studies showing whether models can improve with urban-specific instruction tuning or multimodal inputs. This limits insight into potential solution directions.
3. Reflection dimension underdeveloped. The reflection tasks are intriguing, but the methodology for quantifying reflection accuracy feels underexplained. More qualitative analysis (e.g., error types, improvement dynamics) would clarify why reflection fails and how it could be improved.

### Questions
1. Simulation validity: How closely does the “optimal” action derived via simulation match human expert planning in urban contexts? Have the authors conducted any expert validation?
2. Data diversity: Does UAgentEnv include data from multiple cities? If not, how might this bias model generalization?
3. Interpretability: Can the benchmark support interpretability analysis, such as tracing which reasoning stage most strongly correlates with task success, to guide future model improvements?

### Soundness
3

### Presentation
3

### Contribution
3
