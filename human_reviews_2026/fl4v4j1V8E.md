# Team, Then Trim: An Assembly-Line LLM Framework for High-Quality Tabular Data Generation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
While tabular data is fundamental to many real-world machine learning (ML) applications, acquiring high-quality tabular data is usually labor-intensive and expensive. Limited by the scarcity of observations, tabular datasets often exhibit critical deficiencies, such as class imbalance, selection bias and low fidelity. To address these challenges, building on recent advances in Large Language Models (LLMs), this paper introduces team-then-trim, a framework that synthesizes high-quality tabular data through a collaborative team of LLMs, followed by a rigorous data quality control (QC) pipeline. In our framework, tabular data generation is conceptualized as a manufacturing process: specialized LLMs, guided by domain knowledge, are tasked with generating different data components sequentially, and the resulting products, i.e., the synthetic data, are systematically evaluated across multiple dimensions of QC. Empirical results on both simulated and real-world datasets demonstrate that our framework outperforms the state-of-the-art methods in producing high-quality tabular data, highlighting its potential to support downstream models when direct data collection is practically infeasible.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes the team-then-trim framework for tabular data generation in low-data regimes (n<100).

It’s two parts (1) Multi-agent synthetic data generation & (2) QC pipeline.

Assessed on different tabular data settings vs other generative and LLM based synthetic data generators.

Core contribution: LLM decomposition via multi-agent workers and a different curation mechanism (not the idea of the generation+curation pipeline itself, which prior work proposed)

### Strengths
- Significance: Tackles an important and well-studied problem with high impact in many domains

- Originality: Proposes an interesting idea of multi-agent synthetic data generation + a multi-step QC pipeline.
The idea of feature decomposition and the generation to respect dependencies is great.

- Quality: Good set of experiments in many scenarios: (imbalance, incompleteness, noise, scarcity) + multiple downstream models.
Seems to outperform existing methods on the settings tested (albeit minimally)

- Clarity: clearly written paper

### Weaknesses
(1) Limited novelty: basically the same idea as CLLM, just with a multi-agent approach + different QC approach. Better positioning is needed to understand the gain because there are more LLM calls (via more agents)

(2) Inconsistent results: while train-then-trim mostly outperforms other approaches, it is not universally the case. Understanding when it helps and when it doesn’t and why is important

(3) Missing computational cost: given the extra LLM calls via multi-agent, it is important to understand the performance cost vs performance gain trade-off

(4) Source of gain: It’s useful to understand if the source of gain is from the multi-agent generation or the QC approach. What if the CLLM curation mechanism were applied to the multi-agent generation, how would it perform with the new QC mechanism? i.e. an important ablation is Train-then-team + CLLM curation, CLLM generation + new QC mechanism

(5) Analysis of teaming — more in-depth analysis on when teaming provides value. What types of datasets, what dataset sizes, when should it be used and when does it provide minimal value

(6) Additional LLMs: the paper only uses Llama as the backbone LLM. It is important to try different architectures of LLMs, different sizes and more recent LLMs, such that we know as of today if multi-agent is needed and for all LLMs.

### Questions
- Please can you had the computational costs and number of LLM calls

- Please can you add analysis with other LLMs bases, sizes/parameters (and generally more recent)

- Please can you add this abaltion to understand which component (generation vs curation) drives improvement? Team-then-Trim generation + CLLM's learning dynamics curation vs CLLM's single-LLM generation + Team-then-Trim's 3-stage QC

- Please can you add std dev for all results over the 10 seeds

- The datasets used are public and might be known to the LLM. Some analysis needs to be done on newer/private datasets to assess generalisation

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a framework “Team, then trim” which consists of aggregated workers (LLMs) specialized to (conditionally) generate specific (subset of) columns thus capturing the inter-feature logics and domain dependencies. Once the synthetic dataset has been generated, it undergoes a quality check (QC) process, namely sanity check, objective based cost assessment (by comparing synthetic samples with bootstrapped original data) and diversity based monitoring (to make sure the samples are not skewed) to make sure that the synthetic dataset is of high quality.

### Strengths
- Overall idea and analogy of assembly line workers is intuitive enough.
- The paper is clear to understand and well presented.

### Weaknesses
- [**Experiment on recent baselines**] Addition of more recent baselines, especially the ones that explored the usage of LLMs for tabular generation [1, 2, 3] will strengthen the paper. Moreover, ‘team-then-trim’ has some similarities with [1] in terms of using specialized model components per column/subset of columns (MoEs for [1], worker LLMs here), so it is also important to compare and contrast the pros and cons in related works.

- [**Experiment on model sizes**] Varying model sizes will be interesting to understand its importance on data quality. Questions around design choices such as “Larger/smaller task manager + smaller/larger role specialists” i.e do we need more capable task manager with average workers or an average task manager with highly capable workers; will be interesting to understand. 
    - Questions like “how’s varying model size affects data quality” will also come under this experimental design choice. 

- [**Experiment on model families**] Related to above and a follow-up one would be to look at different families of LLMs for Task Manager and role specialists. Will there be any bias coming up due to collaborations among LLMs coming from different families?
    - Eg: in LLM-as-a-judge literature, there’s bias associated with model preferring responses given by it’s family [4] i.e self-preference bias. So, any analysis and observation in that direction would be interesting. 
    - Appendix E is an interesting starting point for addressing this kind of follow-up questions.

- [**Discussion on time complexity**] Please add time complexity analysis for proposed framework. The size of task manager and role specialist workers (if they are different), number of columns and rows to be generated, number of LLM workers to be used, costs associated with data quality checks (clustering) etc; contribute to overall time complexity. As some of it is data-specific (columns) and task-manager specific decisions (how many workers to assign), it is important to understand the time complexity from practical standpoint beforehand. 
    - [Question, L804] Please add more details in this section in terms of how many samples were generated for each dataset?

- [**Discussion/Experiment on additional metrics**] Inclusion of additional metrics such as MLE (Machine Learning Efficacy) [5], DCR (Distance to closest record) [6], Discrimination [7] is important for discussions associated with privacy preservation, synthetic-vs-real data quality validation. 
    - This will complement some of the discussions in Sec 2.2, especially for objective and diversity cost assessments.
    - Consider adding descriptions of various metrics in Appendix (including AUC, Accuracy, F1, Precision, Recall etc;) complementing sec 3.2.2.

- [**Discussion/Experiment on construction and evaluation of `G`**] From Fig 5 (L760-765), I see that task manager LLM is responsible in forming the relationships among data (i.e construction of `G`, eq. 1), and would like to know how it fairs with manual-human graph construction and assignment of worker LLMs. And how can one evaluate the quality of `G` i.e discard it or regenerate the work assignments.

- [**Discussion/Possible Experiment**] How can one extend the framework for their use case specific requirements for which LLMs doesn’t have enough domain knowledge, let’s say rare data which LLMs didn’t learn in their training process? For example, to generate UUIDs, distinct IDs which is rare/might be spurious from training process. Is it possible to do some fine-tuning with the current framework to get reliable predictions? 

- [**Discussion/Experiment on column dependencies**] Following up from previous point, how does the conditional order of data generation affect in scenarios when columns has a bidirectional relationship i.e there can be different choices to resolve a scenario such as: 
  - Generate column A, then column B vs 
  - Generate column B, then column A or 
  - Generate both A and B together. So, understanding how task-manager (LLM) and human might resolve role conflicts would be interesting.   
    - A quick experiment would be to pick a dataset and have task-manager generated roles and human generated roles and compare the performance differences on role conflicts and worker assignment differences. 
    - Consider adding discussion on different scenarios i.e “independent columns, unidirectionally causal columns and bidirectionally causal columns”.

1. Tabby: Tabular Data Synthesis with Language Models: https://arxiv.org/abs/2503.02152
2. Language Models are Realistic Tabular Data Generators: https://arxiv.org/abs/2210.06280 
3. HARMONIC: Harnessing LLMs for Tabular Data Synthesis and Privacy Protection: https://arxiv.org/abs/2408.02927v1 
4. Self-preference bias in LLM-as-a-judge: https://arxiv.org/abs/2404.13076, https://arxiv.org/abs/2410.21819 
5. A Multi-Dimensional Evaluation of Synthetic Data Generators: https://ieeexplore.ieee.org/document/9686689 
6. SynthEval: A Framework for Detailed Utility and Privacy Evaluation of Tabular Synthetic Data: https://arxiv.org/abs/2404.15821  
7. Synthcity: facilitating innovative use cases of synthetic data in different data modalities: https://arxiv.org/abs/2301.07573

### Questions
- L462-463: Is there a hypothesis on why some metrics are not better for the proposed framework. 
- L338: Is there a sentence continuing post “generated data”, I didn’t get that part.
- Consider adding a section describing the datasets used along with representative examples.

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
3

### Summary
The paper proposes using an agentic AI approach for generating tabular data. A coordination LLM splits the generation problem into K parts assigned to different LLMs, each generating a subset of the tabular features in a way coordinated by the coordination LLM. Here, the prompt requires this LLM to handle dependencies between features for improved quality. The generated data is then passed into a three-stage quality check pipeline ensuring: 1) a sanity check for data types and values; 2) the provided learning potential for a given downstream model, and 3) a good level of diversity. The generated data is evaluated on the downstream task utility against baselines from related work.

### Strengths
- Leverages structural knowledge of the data during generation
- Incorporates multi-level quality checks to ensure high-quality data from different points software view: sanity, utility, and diversity
- Allows for the recovery of data subgroups missing in the original data

### Weaknesses
- Evaluation against related work misses typical tabular generators, e.g., GReaT [1] and Tabula [2], and in particular also any other agentic LLM, e.g., [3] or diffusion-based ones, e.g., [4].
- All LLMs in the evaluation seem to be of the same type, i.e., Llama 3.3 70B Instruct, but the power of this method could also be to use more targeted LLMs for the different roles, coordinator vs worker, or for specific features. No evaluation in this direction has been done.
- Following that, the same LLM is used for all roles, the paper should stress more what the advantage of this approach is with respect to some kind of chain-of-thought/in-context learning type of guidance of a single LLM during the data generation.
- Only full or no quality control is considered as an ablation study. It could be interesting how much each of the three QC steps contributes.

Minor:
- The type of data noise (label flip) has not been specified. Did you use symmetric or class-specific flipping?
- Table 1, 2, 4, 5: font is way larger than surrounding text.
- Figure 3 is placed before the text referencing it
- Figure 3: to ease comparison I suggest to use the same y range on all subfigures


[1] Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M., Kasneci, G.: Language models are realistic tabular data generators. arXiv preprint arXiv:2210.06280 (2022)
[2] Zhao, Z., Birke, R., & Chen, L. Y. (2025, June). Tabula: Harnessing language models for tabular data synthesis. In Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp. 247-259).
[3] Benoît Ronval, Pierre Dupont, Siegfried Nijssen. TAGAL: Tabular Data Generation using Agentic LLM Methods. arXiv preprint arXiv:/2509.04152 (2025)
[4] Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, Artem Babenko. TabDDPM: Modelling Tabular Data with Diffusion Models. ICML 2023: 17564-17579

### Questions
- How does Team-then-Trim perform against other baselines from related work, such as the ones referenced under weaknesses?
- What is the noise transition matrix used for label flipping?
- What is the benefit of the different QC steps?

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
3

### Summary
This paper introduces Team-then-Trim, a framework for synthetic tabular data generation using coordinated large language models. A task-manager LLM partitions the feature space into semantically aligned components and schedules specialized worker LLMs to generate each subset sequentially based on dependency structure. The resulting partial outputs are concatenated into full samples, which then pass through a three-stage quality control process assessing validity, task utility, and diversity preservation. Across simulated and real-world datasets, the proposed method yields synthetic data that improves downstream model performance and maintains distributional fidelity compared to both traditional oversampling and single-LLM baseline.

### Strengths
- The team-then-trim structure separates generation from post-hoc quality control, providing robustness against LLM hallucination.
- The three-stage quality control pipeline (sanity, objective-driven filtering, diversity enforcement) is systematic and targets well-known challenges in synthetic data generation, including invalid entries, distributional bias, and limited incremental information.
- The use of model-based scoring and information-gain comparison to filter batches offers a principled framework beyond heuristic rejection rules that previous work used.
- The method demonstrates downstream performance better than existing tabular data generation baselines.

### Weaknesses
- The quality control pipeline assumes access to a reasonably performant base model and sufficient initial real data to bootstrap quality signals, which can limit applicability in low-data or scarce-label settings (including simulated data incompleteness setting in the paper).
- The method incurs non-trivial computational overhead due to repeated generation, batch scoring, and rejection loops. The generation resource trade-offs are not fully addressed.
- The reliance on a single trained classifier for qualifying the cost of synthetic data raises the possibility that the QC process overfits to the specific classifier used, rather than reflecting true data utility. It would be valuable to evaluate whether the selected batches remain consistent when multiple different classifiers are used for the scoring stage.

### Questions
- The evaluation reports performance using 500 generated and original samples. How does downstream performance scale as the number of synthetic samples increases? Specifically, does performance continue to improve with additional synthetic data, or does it plateau or degrade? 
- In scenarios where the number of original samples is limited, can the synthetic data still recover or cover the full cluster structure that would be observed if the complete real dataset were available? In other words, does the proposed method retain the ability to approximate the true distributional clusters when starting from a partially observed dataset?
- Which LLM was used for the curated generation process in CLLM? The original CuratedLLM paper reports that stronger LLMs exhibit better performance, particularly on under-represented samples. Therefore, it would be helpful to clarify the specific model used in your reproduction to understand the reported results.
- The proposed pipeline appears to rely on data-specific prompt construction for effective synthetic sample generation. Could the authors evaluate the robustness of the method with respect to prompt variations? Such an analysis would strengthen the novelty claim by demonstrating that performance is not overly reliant on manually curated prompt engineering.

### Soundness
3

### Presentation
3

### Contribution
2
