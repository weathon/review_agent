# FedAgentBench: Towards Automating Real-world Federated Medical Image Analysis with Server–Client LLM Agents

- Decision: Reject
- Scores: 4, 6, 2, 8

## Abstract
Federated learning (FL) allows collaborative model training across healthcare sites without sharing sensitive patient data. However, real-world FL deployment is often hindered by complex operational challenges that demand substantial human efforts in cross-client coordination and data engineering. This includes: (a) selecting appropriate clients (hospitals), (b) coordinating between the central server and clients, (c) client-level data pre-processing, (d) harmonizing non-standardized data and labels across clients, and (e) selecting FL algorithms based on user instructions and cross-client data characteristics. However, the existing FL works overlook these practical orchestration challenges. These operational bottlenecks motivate the need for autonomous, agent-driven FL systems, where intelligent agents at each hospital client and the central server agent collaboratively manage FL setup and model training with minimal human intervention. To this end, we first introduce: (i) an agent-driven FL framework that captures key phases of real-world FL workflows from client selection to training completion, and (ii) a benchmark dubbed FedAgentBench that evaluates the ability of LLM agents to autonomously coordinate healthcare FL. Our framework incorporates 40 FL algorithms, each tailored to address diverse task-specific requirements and cross-client characteristics. Furthermore, we introduce a diverse set of complex tasks across 201 carefully curated datasets, simulating 6 modality-specific real-world healthcare environments, viz., Dermatoscopy, Ultrasound, Fundus, Histopathology, MRI, and X-Ray. We assess the agentic performance of 14 open-source and 10 proprietary LLMs spanning small, medium, and large model scales. While some agent cores such as GPT-4.1 and DeepSeek V3 can automate various stages of the FL pipeline, our results reveal that more complex, interdependent tasks based on implicit goals remain challenging for even the strongest models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes FedAgentBench, a benchmark and multi-agent FL framework in which LLM-based server and client agents automate the full operational pipeline of federated learning in healthcare. This spans selecting participating hospitals, cleaning and harmonizing their heterogeneous data and labels, and choosing among 40 FL algorithms, and initiating training. The benchmark spans across 201 datasets from 6 clinical imaging domains. The authors evaluate 24 open and proprietary LLMs and find that frontier models (e.g. GPT-4.1, DeepSeek V3) can reliably execute many of these coordination steps, but all models struggle with the most interdependent tasks.

### Strengths
1. Addresses a real bottleneck in FL deployment. The work explicitly targets the coordination layer of FL (onboarding sites, cleaning data, resolving label taxonomies, starting training), which is widely known to be the main practical blocker in clinical federations rather than just the optimization algorithm itself. The benchmark makes that whole layer but also its parts measurable and quantifiable. 
2. Scale and heterogeneity of environments. The benchmark covers 6 imaging modalities (Dermatology, Ultrasound, Fundus, X-Ray, MRI, Histopathology) with 201 datasets across different clinical tasks, which is very comprehensive.
3. System-level realism and scale. The whole pipeline is evaluated across 24 LLMs spanning open source, proprietary, small and large models.

### Weaknesses
1. Not really real-world. All datasets in FedAgentBench are from public datasets with only artificial noise injections. Usually the data comes directly from the PACS where no description of a whole dataset is available. Also that some folder structure and some label definition exists in advance is not real-world to name just a few points. In general it is interesting to see how LLMs can perform the proposed tasks but the prerequisites do not exist in a real-world setup. 
2. The 'Training-start' metric is weak. The FL training phase is scored by “Training Start Verification”: did the agent generate a valid config, select an algorithm, distribute instructions, and start training. There is no evidence that the chosen algorithm is actually appropriate beyond superficial matching of task descriptors, nor that it converges.

### Questions
1. It would be nice to have one table which aggregates all the results from all the different environments. 
2. Unclarity about the different environments. While there can be a range of tasks modeled per environment, it is not clear which tasks are actually evaluated beyond the Dermatology environment, where also, from my understanding, only a single task (skin cancer detection) is evaluated. For MRI for example some of the datasets are segmentation datasets. Some clarification here is needed.

### Soundness
2

### Presentation
1

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
This paper introduces FedAgentBench, a benchmark framework for automating federated medical image analysis through collaboration among server–client LLM agents. The system includes seven specialized agents that autonomously handle key stages of federated learning—client selection, data cleaning, label harmonization, and model training—using sixteen functional tools. FedAgentBench integrates 201 medical imaging datasets across six modalities and 40 federated algorithms, providing a realistic testbed to evaluate agent reasoning, adaptability, and privacy-aware collaboration. Experiments on 24 large language models (e.g., GPT-4.1, Claude-Opus, DeepSeek-V3) show that strong closed-source models outperform open ones, while label harmonization remains the most difficult task. The work contributes the first end-to-end benchmark for agent-driven federated learning, highlighting a step toward automated, large-scale healthcare FL systems.

### Strengths
1.	Novel problem formulation – The paper is among the first to conceptualize federated medical image analysis as a multi-agent LLM coordination task, bridging federated learning and autonomous AI research.
2.	Comprehensive and realistic benchmark – Integrates 201 datasets across six imaging modalities and 40 federated algorithms, offering a broad, heterogeneous, privacy-respecting testbed.
3.	Well-structured system design – The seven-agent server–client architecture, combined with 16 tool-based operations, demonstrates strong implementation quality with clear modular decomposition of the FL workflow.
4.	Thorough evaluation – Benchmarks 24 large language models (both proprietary and open-source) with well-defined metrics, providing valuable empirical insights into LLMs’ reasoning and collaboration capabilities.
5.	Potential research impact – The benchmark is likely to become a useful reference for studying agentic automation in scientific and healthcare FL settings, encouraging reproducible system-level research beyond algorithm design.

### Weaknesses
1.	Limited technical novelty – The work focuses mainly on benchmark and system construction; it does not introduce new algorithms or methodological advances in federated learning or agent reasoning.
2.	Evaluation depth – While broad in scope, the analysis remains mostly descriptive. There is limited discussion on why certain models fail or how specific reasoning strategies affect success rates.

### Questions
1.	Role of tools and autonomy: Since the agents rely heavily on predefined tools, how much of the system’s performance comes from LLM reasoning vs. scripted automation? A controlled ablation (e.g., removing certain tool functions) might clarify how much genuine reasoning each agent contributes.
2.	Definition of task success: The benchmark measures task completion and reasoning accuracy, but the success criteria for each sub-task (e.g., client selection vs. label harmonization) seem uneven. How were these thresholds determined, and do they reflect meaningful operational success in real FL workflows?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to use agents to train federated learning tasks without direct human coding. To do so, the authors develop both client-side agents and server-side agents, and discuss two modes: one with fine-grained supervision, where the instructions are very precise about how the agents should perform the task, and another “goal-oriented” mode. 
More precisely, the authors use a large quantity of existing data assets, grouping them by task and then splitting them across clients, asking the agents to find a way to learn a model despite very different inputs. This implies that the agents should be able to select relevant data, decide whether a client should participate in the training, and handle standard preprocessing tasks such as data deduplication, noise filtering, or correction of incorrect labels. The results are then presented as a benchmark comparing several open-source and proprietary large language models.

### Strengths
The topic of the paper is very relevant, and the method is quite surprising. The idea of adjusting for heterogeneity by asking agents to modify the data is novel and could, under careful supervision, be useful for developing use cases in federated learning. The paper likely required a significant amount of work and computation, as the number of large language models and datasets used is impressive. Some figures and visualizations are also quite polished and could help reach a broader audience.

### Weaknesses
The paper does not meet the scientific standards expected at ICLR. In particular, it does not formulate or test clear scientific hypotheses. It is more of an interesting project than a reproducible benchmark or a proposal of a new scientific method with clear results. There is no clear analysis of what has been done and what could be improved.

The blind application of agents to manipulate very sensitive data such as health data, and to output predictions in critical systems, raises major ethical issues that are completely overlooked by the paper. The methodology is also very naive and ignores many challenges of training federated systems.

For example, the computation of the final model accuracy is never discussed, and some tables report F1 scores reaching 1, which is implausible for health data tasks. It is likely that the reported values are meaningless or that the training and testing data overlap in some way. Similarly, client selection seems to consist only of selecting clients with data related to the task, but the fact that integrity constraints can limit the incentive to share data across clients is ignored. The paper also claims to be privacy-preserving thanks to federated learning, without acknowledging that federated learning alone is not sufficient to prevent data leakage through updates or the final model.

The length of the paper is also inappropriate, as it includes a large amount of low-quality content, such as lengthy, uncritical descriptions of different models or very wordy passages, for example:

The paper is also disproportionately long and contains much low-quality content, such as verbose descriptions of models without critical analysis. For example:
"o4-mini OpenAI (2025c) is a presumed lighter version of the GPT family with unclear lineage,
possibly a prototype or early mini model." "Although it lacks vision
capabilities, it provides consistent JSON formatting and supports Chinese-English reasoning tasks
effectively." "Table 15 compares open-source and proprietary LLM agents in the XRay environment Proprietary
models continue to dominate the list. GPT-4.1 reaches ceiling performance with consistent 5/5
across all sub-tasks and the best Overall scores (100.00 under both fine-grained and goal-oriented
guidance). A strong second tier includes GPT-o4-mini (91.43 / 85.71) and GPT-4.1-mini (88.57
/ 77.14), followed by GPT-4o-mini (77.14 / 74.29), GPT-4-Turbo (71.43 / 77.14), GPT-4 (71.43
/ 65.71), GPT-4o (71.43 / 68.57), and GPT-o3-mini (71.42 / 74.29). Claude-3-7 is mid-pack
(57.14 / 57.14), while GPT-3.5-Turbo trails (25.71 / 34.29)" "These methods are essential for benchmarking and provide the backbone
upon which many subsequent algorithms are built."

The references are not properly formatted.

The work clearly relies on heavy use of LLMs, which is not what is declared.

### Questions
Feel free to comment on the above weaknesses, in particular on the evaluation and limits of your pipeline.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents FedAgentBench, the first systematic benchmark applying the LLM Agent framework to FL operation automation, particularly in medical image analysis, representing a meaningful step toward agent-driven FL automation.

### Strengths
(1) Novel and timely direction: Applying LLM Agents to FL operations is highly original and addresses practical deployment challenges beyond traditional algorithmic design.

(2) Comprehensive benchmark design: FedAgentBench systematically covers key FL workflow stages, providing a valuable platform for future research.

(3) Insightful empirical findings: The authors conduct a thorough benchmarking of existing models across diverse FL tasks, providing valuable practical guidance for future model selection and resource allocation.

### Weaknesses
While the analysis of agent failure modes is detailed, the paper lacks discussion on how these insights could guide future LLM Agent or prompt design. Adding a short section in Future Work to outline such directions would further strengthen the contribution.

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
4
