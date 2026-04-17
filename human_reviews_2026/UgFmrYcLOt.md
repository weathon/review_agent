# TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Large Language Model (LLM) agents are rapidly emerging as powerful systems for automating tasks across domains. Yet progress in the open-source community is constrained by the lack of high quality permissively licensed tool-agentic training data. Existing datasets are often limited in diversity, realism, and complexity, particularly regarding multi-tool and multi-turn interactions. To address this gap, we introduce Toucan, the largest publicly available tool-agentic dataset to date, containing 1.5 million trajectories synthesized from nearly 500 real-world Model Context Protocols (MCPs). Unlike prior work, Toucan leverages authentic MCP environments to generate diverse, realistic, and challenging tasks with trajectories involving real tool execution. Our pipeline first produces a broad spectrum of tool-use queries using five distinct models, applies model-based quality filtering, and then generates agentic trajectories with three teacher models using two agentic frameworks. Rigorous rule-based and model-based validation ensures high-quality outputs. We also introduce three extension mechanisms to further diversify tasks and simulate multi-turn conversations. Models fine-tuned on Toucan outperform larger closed-source counterparts on the BFCL V3 benchmark and establish a new Pareto optimum on MCP-Universe Bench.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces TOUCAN, a tool-agentic dataset containing 1.5 million training samples synthesized from nearly 500 real-world MCP servers. The authors also describe the details of their construction pipeline, based on the target to generate diverse, realistic and challenging tasks. This includes in overall 5 steps: MCP Server Onboarding, Task Synthesis, Task Filtering, Trajectory Generation, and Rule&LLM-Based Post-Filtering. The analytical and experimental results show that TOUCAN preserves high quality and helps on improving performance on diverse benchmarks based on Qwen2.5 series models.

### Strengths
1. This paper introduces the largest agentic dataset with 1.5 Million samples for LLMs.
2. The dataset is generated with real MCP servers, providing realistic feedback that is important for LLM tool calling capability acquisition.
3. The analysis shows the diversity of the dataset, as well as the quality to some extent, based on LLM judges.
4. Experimental results show that fine-tuning with the dataset on Qwen2.5 models improves at least 3% scores in several benchmarks.

### Weaknesses
1. The technical contribution is limited. The pipeline introduced is heuristic, where many of previous work has applied similar methods. I cannot find any interesting or new things according to the description. For example, rule&LLM-based quality evaluation have been explored in ToolACE and APIGen, and no any distinct design is found. While the authors emphasize generating realistic, diverse and challenging data, the diversity and difficulty things are not found in the generation pipeline.
2. The experiments are not convincing:
   - Only Qwen2.5 models are tested with the proposed dataset. 
   - The experiments are conducted on a selected subset of approximately 119K samples, rather than the full 1.5M dataset. This raises concerns about potential overstatement of their contribution, as the released 1.5M data may not all be of high quality.
   - The improvement is also limited, e.g., only achieving about 3% improvement on Qwen2.5-7B-Instruct with the data. This raises the concerns whether we need so many data samples for one single capability (In my personal experience, 10 - 20 samples per tool are enough for learning the usage of the tool. Given that the total tool amount is about 2k in TOUCAN, it is much less data needed). An ablation study on the data amount is needed.
   - The comparison with the exsting public datasets is needed, not just the statistical level, but experimental level.
   - More analysis on the data quality is needed, showing that the accuracy and difficulty of the data is sufficient.
   - Ablation on the stages proposed is needed. We need evidences proving that the proposed five-stage pipeline is optimized.

### Questions
1. What is your unique design in your pipeline?
2. Have you tried using more advanced models for data generation? For example, GPT-5 or Claude 4.

### Soundness
2

### Presentation
2

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
This work introduces TOUCAN, the largest publicly available tool-agentic dataset to date, comprising 1.5 million trajectories synthesized from nearly 500 real-world MCP servers. The TOUCAN Generation Pipeline consists of five stages: 1. MCP Server Onboarding 2.Task Synthesis 3.Task Filtering 4. Trajectory Generation 5. Rule&LLM-Based Post-Filtering, and 3 distinct procedures post-core pipeline (Steps 1 to 5) to generate new instances targeting specific objectives:1. Irrelevance, 2. Persona-based Diversification, 3. Multi-Turn. 

Empirically, models fine-tuned on TOUCAN significantly outperform both open- and closed-source baselines across diverse benchmarks.

### Strengths
- The paper is well-organized and easy to follow, with clear presentation of dataset statistics, filtering criteria, and evaluation metrics.

- The data generation pipeline is described in detail, systematically addressing the common challenges in tool-calling dataset construction.

- The work is highly valuable, I believe tool-use capability is a crucial step for LLMs toward AIGC.

### Weaknesses
- Although the reported performance gains are significant, there remains a gap to the latest SOTA. For instance, xLAM-2-70B-fc-r achieves 75.38 on the BFCL-V3 multi-turn benchmark, higher than the TOUCAN-tuned counterparts.

- The 495 MCP servers used may still be insufficient to cover the full spectrum of real-world scenarios. Expanding the dataset to include more domains.

### Questions
I briefly take a look at the released dataset and noticed that many MCP servers contain overlapping or functionally similar tools (e.g., search tools).
Did the authors consider deduplication of MCP servers during the MCP Server Onboarding stage to reduce redundancy?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents TOUCAN, a dataset of 1.5M tool-agentic trajectories from 495 real-world MCP servers. Unlike prior work using simulated responses, TOUCAN uses authentic tool execution. The systematic pipeline includes task synthesis, quality filtering, trajectory generation, and three extensions for irrelevance, diversification, and multi-turn dialogues. Models fine-tuned on TOUCAN outperform larger baselines on BFCL V3 and MCP-Universe benchmarks.

### Strengths
1. TOUCAN is the largest open-source tool-agentic dataset, using real MCP servers with authentic tool execution rather than simulated responses. The coverage of 495 servers across diverse domains addresses a critical gap in permissively licensed training data.

2. The five-stage pipeline with rigorous filtering (LLM-based quality assessment, rule-based validation) and three thoughtful extensions (irrelevance handling, persona-based diversification, multi-turn conversations) demonstrates careful engineering. The ablation study validates each component's contribution.

3. Extensive documentation including prompts, schemas, hyperparameters, and detailed appendices. The modular pipeline design allows future extensions and customization.

### Weaknesses
1. Excluding servers requiring authentication (reducing 2,800 to 495 servers) systematically removes widely-used production services like GitHub, Notion, and Slack. This likely underrepresents enterprise workflows and authenticated API interactions common in real deployments. The impact on domain coverage and practical applicability needs better characterization.

2.  LLM-based quality assessment using Kimi-K2 shows only 0.264 Pearson correlation with human annotations on 50 samples. This small sample size and modest correlation raise concerns about annotation reliability, especially since similar models are used for both annotation and training.

3. All benchmarks may share tool characteristics with the training set. The paper lacks evaluation on completely unseen tool ecosystems or domains, making it unclear whether models learn general tool-use capabilities or memorize specific patterns.

### Questions
1.  What is the estimated domain coverage loss from excluding authenticated servers? Have you quantified how many critical real-world use cases are missing?

2.  What is the inter-annotator agreement among human evaluators? With only 0.264 correlation, how confident are you in the quality scores? Have you considered ensemble approaches with multiple judges?

3. Have you evaluated on completely unseen tool ecosystems (domains not in training)? What about zero-shot performance on novel tool combinations?

4. How do you ensure test sets from benchmarks were not included through overlapping MCP servers or similar task formulations?

5. Since you filter out tool failures, how do models learn robust error recovery? Could you include failure cases with recovery strategies?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TOUCAN, a large-scale tool-agentic dataset comprising over 1.5 million trajectories from 495 real-world Model Context Protocols (MCPs). The goal of TOUCAN is to address the limitations of current open-source datasets by providing more diverse and realistic tool-agentic interactions. It presents a comprehensive data generation pipeline involving task synthesis, trajectory generation, and rigorous filtering for high-quality outputs. The paper also highlights improvements in model performance when fine-tuned on TOUCAN, particularly on benchmarks such as BFCL V3 and MCP-Universe.

### Strengths
1.  TOUCAN offers a large-scale, high-quality dataset that provides diverse real-world tool-agentic interactions, filling a significant gap in available open-source datasets for training agentic LLMs.

2. The paper demonstrates clear improvements in model performance after fine-tuning on TOUCAN, outperforming existing models on multiple benchmarks. This provides evidence for the practical impact of the dataset.

3. TOUCAN covers a wide variety of domains, tools, and interaction patterns, making it a versatile resource for various LLM-based tasks.

### Weaknesses
1. The method relies heavily on large language models and teacher models to generate tool-agentic interactions, which is computationally expensive. This may limit its accessibility for research groups without sufficient resources.

2. Models trained on this dataset may be overfitted to the 2000 tools. How to extend the models to other domains with different tools? These paper may have limited usefulness to practical tasks.

3. The pipeline described for task generation and filtering is similar to previously proposed methods, such as in [a] (Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage, ICLR 2025). While the data volume is significantly larger in this manuscript, the methodological novelty may be questioned. The relationship and differences should be discussed in the manuscript.

4. The dataset involves more than 2,000 tools. It is unclear how the inclusion of so many tools adds unique value to the dataset. A discussion of the redundancy and actual diversity of these tools would help clarify whether they introduce meaningful variety or if many of them are effectively duplicates.

### Questions
see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
