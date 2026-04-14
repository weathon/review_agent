# CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents

- Decision: Reject
- Scores: 5, 3, 3, 6

## Abstract
The development of autonomous agents increasingly relies on Multimodal Language Models (MLMs) to perform tasks described in natural language with GUI environments, such as websites, desktop computers, or mobile phones. Existing benchmarks for MLM agents in interactive environments are limited by their focus on a single environment, lack of detailed and generalized evaluation methods, and the complexities of constructing tasks and evaluators. To overcome these limitations, we introduce CRAB, the first agent benchmark framework designed to support cross-environment tasks, incorporating a graph-based fine-grained evaluation method and an efficient mechanism for task and evaluator construction. Our framework supports multiple devices and can be easily extended to any environment with a Python interface. Leveraging CRAB, we developed a cross-platform CRAB Benchmark-v0 comprising 120 tasks in computer desktop and mobile phone environments. We evaluated 6 advanced MLMs using different single and multi-agent system configurations on this benchmark. The experimental results demonstrate that the single agent with GPT-4o achieves the best completion ratio of 38.01%.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces CRAB, a new framework for evaluating multimodal language model (MLM) agents across different computing environments like desktop and mobile platforms. The key contributions include: (1) The first benchmark framework supporting cross-environment tasks, (2) A graph-based evaluation method for fine-grained assessment of agent performance, and (3) An efficient mechanism for constructing tasks and evaluators. The authors implement Crab Benchmark-v0 with 120 tasks across Ubuntu desktop and Android environments, evaluating several MLMs in different agent configurations. The best performance achieved was 38.01% completion ratio using GPT-4o in a single-agent setup.

### Strengths
1. The paper considers an important problem in MLM agent evaluation: cross-environment evaluation. Evaluating MLM agents across different platforms better reflects real-world scenarios where tasks often require coordination between devices.
2. The benchmark demonstrates good diversity in its evaluation coverage, which helps provide a more comprehensive assessment of MLM agents' capabilities.

### Weaknesses
1. Despite its apparent comprehensiveness, the empirical analysis lacks depth, especially considering the benchmark's cross-platform nature. The paper merely reports metrics without providing meaningful interpretation. It fails to offer an in-depth analysis of the key challenges posed by the cross-platform environment and the capabilities that best correlate with addressing these challenges.
2. The paper inadequately explains the construction process for graph evaluators. It provides a vague description of human involvement in the annotation process and lacks clear criteria for decomposing tasks into sub-tasks. Furthermore, there's limited explanation of how cross-platform dependencies are handled within the graph structure.
3. The evaluation framework is unnecessarily complex with multiple metrics (Success Rate, Completion Ratio, Execution Efficiency, Cost Efficiency). A single, well-designed metric would be more practical and interpretable.
4. (minor) The paper has several writing issues like inconsistent descriptions (e.g., "6 popular MLMs" vs "four advanced MLMs") and redundant entries in tables (e.g., repeated OSWorld entry in Table 1).

### Questions
1. What efforts were made to ensure CRAB's tasks reflect real-world scenarios? Can you provide a breakdown of task distributions by category?
2. How long does a single evaluation take?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces a benchmark framework  (CRAB) designed to evaluate the performance of MLM agents across different environments, such as desktop and mobile platforms. CRAB addresses the limitations of existing benchmarks by supporting cross-environment tasks, employing a graph-based fine-grained evaluation method, and offering an efficient mechanism for task and evaluator construction. The framework is extensible to any environment with a Python interface and includes a benchmark, Crab Benchmark-v0, consisting of 120 tasks across computer desktop and mobile phone environments. The study evaluates four advanced MLMs using various single and multi-agent system configurations, with GPT-4o achieving the highest completion ratio of 38.01%.

### Strengths
1. CRAB's ability to handle tasks across different environments is an advancement, reflecting the multi-platform nature of real-world applications.
2. The graph evaluator provides a nuanced and detailed assessment of agent performance, capturing intermediate progress and multiple valid pathways to task completion.
3.  The framework's design allows for easy adaptation to new platforms and devices, and its sub-task composition method streamlines task creation.
4. The inclusion of 120 real-world tasks in Crab Benchmark-v0 enhances the benchmark's applicability and usefulness for developing and testing MLM agents.
5. The paper presents a thorough evaluation of various MLMs and agent configurations, offering insights into their relative strengths and weaknesses.

### Weaknesses
This following comment reflects my personal bias, if the authors can convince me, I would be happy to change my score:

Although web agent recently is a hot topic, I feel that it is an engineering problem, not a research problem. In another words, as long we we gather enough web interaction data, all kinds of problems listed in those papers in the related works will not exist anymore. Therefore, just like 10 years ago, we spend quite a lot of effort in tweaking different models and benchmarks to solve vision and language problem and turn out of be useless, we should spend more time on studying how to scale up the training data if we want to solve the web agent problem. And even if it is solved, it won't teach us anything because the same lesson has learnt before. Therefore, it would be great for the author to elaborate on the broader lessons we can learn from reading this paper and how this problem can advance the AI field in general.

### Questions
Please see the comment in weakness section and please respond directly.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces the CRAB framework, designed to evaluate multi-modal language models MLMs in a cross-platform setting. CRAB supports multi-agent configurations and uses a graph-based task decomposition approach to allow flexible, multi-path task completion across environments like desktop and mobile platforms. A unique feature is the graph-based evaluator that provides a detailed, granular assessment of task progress by tracking completion across multiple paths, which extends beyond traditional single-path evaluations. Experiments with commercial and open-source MLM models on Ubuntu and Android environments demonstrate CRAB’s potential for flexible multi-agent, cross-platform assessments.

### Strengths
A key contribution of the paper is the cross-platform evaluation capability, allowing MLMs to be assessed across desktop and mobile platforms. The graph-based task decomposition enables flexible, multi-path task completion, and the detailed evaluation metrics offer a comprehensive assessment of agent performance. The diverse experimental setup with various MLM models shows CRAB’s broad applicability.

### Weaknesses
1.	The paper lacks sufficient review of existing work, missing comparisons with related frameworks such as Mobile-ENV [1] and GUI-World [2].
2.	Table 1 only presents the number of apps, whereas similar papers like AndroidWorld [3] provide detailed counts of tasks and templates per app. Additionally, "osworld" appears twice in the table, which could cause confusion and detracts from a thorough task-level comparison.
3.	The explanation of GDT (Graph of Decomposed Tasks) is unclear; it appears more like a detailed "tree of thought" rather than a true graph structure, which may confuse readers regarding the intended data structure.
4.	The necessity of a multi-agent system is not well justified, as no significant performance improvements are shown over a single-agent configuration.
5.	The absence of a comparison with traditional trajectory-based evaluation methods limits the paper’s ability to showcase the distinct advantages of CRAB. Additionally, each task is based on a simple 4-node graph, raising questions about how significantly the results could differ from those of trajectory-based evaluation.
Reference:
[1] Zhang, Danyang, Lu Chen, and Kai Yu. 2023. "Mobile-ENV: A Universal Platform for Training and Evaluation of Mobile Interaction." arXiv preprint arXiv:2305.08144.
[2] Chen, Dongping, Yue Huang, Siyuan Wu, Jingyu Tang, Liuyi Chen, Yilin Bai, Zhigang He, Chenlong Wang, Huichi Zhou, Yiqiang Li, et al. 2024. "GUI-World: A Dataset for GUI-Oriented Multimodal LLM-Based Agents." arXiv preprint arXiv:2406.10819.
[3] Rawles, Christopher, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. 2024. "AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents." arXiv preprint arXiv:2405.14573.

### Questions
N/A

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
This paper introduces CRAB, a novel benchmark framework for evaluating multimodal language model (MLM) agents on cross-environment tasks. CRAB supports multiple devices, incorporates a graph-based fine-grained evaluation method, and provides an efficient mechanism for constructing tasks and evaluators. The authors also present Crab Benchmark-v0, comprising 120 tasks across computer desktop and mobile phone environments. They evaluate four advanced MLMs in single and multi-agent configurations on this benchmark. The results show that a single agent with GPT-4o achieves the best completion ratio of 38.01%. It highlights the need for more effective autonomous agents.

### Strengths
Originality & Quality: CRAB is the first benchmark to incorporate cross-environment tasks, reflecting real-world scenarios. Novel graph evaluator and sub-task composition methods address the limitations of existing evaluation methods and benchmarks.

Clarity: Contain detailed descriptions of framework design, task dataset, agent implementations. Visual case studies provide examples to illustrate agent performance on specific tasks. Good analysis of results by platform, model, agent structure, and termination reasons.

Significance: This work addresses key limitations in previous methods, including platform diversity, evaluation metrics.
It enables standardized evaluation of MLM agents on complex real-world cross-platform tasks. Modular and extensible framework design allows easy adaptation to new platforms and tasks.

### Weaknesses
- Limited coverage of applications. It focuses on original apps in Ubuntu & Android on Pixel devices. Expanding to more apps and devices would further improve real-world coverage. The number of data instances (120 tasks) is relatively small. While the tasks are designed to be more complex than those in other benchmarks, extending the dataset to around 500 instances would enable more comprehensive and statistically significant comparisons between different models and agent settings. A larger dataset would better test the generalization abilities of the agents and increase the overall significance of the benchmark. I am willing to increase the score if the authors can consider it.

- Visual prompting methods do not fully recognize all interactive elements, potentially hurting agent performance. More advanced visual understanding techniques could be explored.

- Lack of human performance baselines makes it difficult to assess how far MLM agents are from human-level task completion abilities.

### Questions
- Can the authors comment on the costs associated with running the benchmark?

### Soundness
3

### Presentation
3

### Contribution
3
