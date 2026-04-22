# EcomEval: Towards Reliable Evaluation of Large Language Models for Multilingual and Multimodal E-Commerce Applications

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 6, 2

## Abstract
Large Language Models (LLMs) excel on general-purpose NLP benchmarks, yet their capabilities in specialized domains remain underexplored. In e-commerce, existing evaluations—such as EcomInstruct, ChineseEcomQA, eCeLLM, and Shopping MMLU—suffer from limited task diversity (e.g., lacking product guidance and after-sales issues), limited task modalities (e.g., absence of multimodal data), synthetic or curated data, and a narrow focus on English and Chinese, leaving practitioners without reliable tools to assess models on complex, real-world shopping scenarios.
We introduce EcomEval, a comprehensive multilingual and multimodal benchmark for evaluating LLMs in e-commerce. EcomEval covers six categories and 37 tasks (including 8 multimodal tasks), sourced primarily from authentic customer queries and transaction logs, reflecting the noisy and heterogeneous nature of real business interactions. To ensure both quality and scalability of reference answers, we adopt a semi-automatic pipeline in which large models draft candidate responses subsequently reviewed and modified by over 50 expert annotators with strong e-commerce and multilingual expertise. We define difficulty levels for each question and task category by averaging evaluation scores across models with different sizes and capabilities, enabling challenge-oriented and fine-grained assessment. EcomEval also spans eight languages—including four low-resource Southeast Asian languages—offering a multilingual perspective absent from prior work.
We evaluate 19 open and proprietary LLMs on EcomEval, revealing substantial performance disparities and highlighting scenarios where these general-purpose models perform poorly in the e-commerce domain. By combining diversity, authenticity, quality, difficulty awareness, multilinguality and multimodality, EcomEval establishes a rigorous and representative testbed for advancing research and deployment of LLMs in e-commerce. Upon acceptance, we will release the full dataset to support reproducible research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents EcomEval, a dataset for evaluating Large Language Models (and Large Multimodal Models, later both referred to as   LLMs) on e-Commerce tasks. Compared to existing datasets like eCeLLM and Shopping MMLU, EcomEval features multi-modal tasks, a broader coverage of rare languages (in southeast Asia), and a broader inclusion of tasks (e.g. post sales, multi-turn dialogue). The data curation pipeline also becomes semi-automatic with a difficulty labeling. Extensive results on open and closed-source LLMs demonstrate the limitations of existing AI systems in real-world e-commerce tasks.

### Strengths
1. A more diverse and real-world dataset for e-commerce. Compared to existing datasets like eCeLLM datasets and Shopping MMLU, it is clear that EcomEval features a broader range of tasks and skills. For example, EcomEval features multi-modal tasks such as multimodal product similarity, brand recognition, image summarization, etc. EcomEval includes dialogue tasks such as shopping guide. EcomEval includes after-sales tasks. These features make EcomEval a better benchmark on specific domains (eCommerce) than existing ones. 

2. A semi-automatic benchmark reconstruction pipeline from raw LLM logs. Existing benchmarks in e-commerce are mostly sourced from existing ones (e.g. eCeLLM and Shopping MMLU are more or less sourced from open Amazon datasets), with already good task definition and input-output pairs. In contrast, EcomEval designs a semi-automatic task construction pipeline from raw LLM logs. This is of practical value in identifying important tasks and formulating it as a problem for future LLM fine-tuning. The proposed pipeline is also reasonable (e.g. clustering of prefixes as instructions, LLM-based task summary).

3. Extensive evaluations of existing models and interesting results. The evaluation results of existing AI systems bear interesting insights into this specific domain of e-commerce. First, GPT-5 is not always the best model, which is a good finding as users may choose to use other models (potentially cheaper) for substitutes. Second, the gap between open-source and close-source models are not significant in terms of language models, which shows that future efforts in curating e-commerce training datasets do not have to rely on proprietary models, and thus become more affordable. Finally, the gap between open-source and close models are significant on multi-modal tasks, pointing out potential room for improvements.

### Weaknesses
1. Taxonomy can be improved as the categories are not completely independent and mutually exclusive. I find that the categories in Figure 1 do not seem to be mutually exclusive and thus may be slightly confusing. For example, EcomQA (e.g. Shopping Guide) may also be highly related to user understanding. Product recommendation in "Shopping Reasoning" may also be related to user understanding, "description similarity" in shopping reasoning may also be related to "shopping concepts". Therefore, the taxonomy may bear room for improvements to make it more mutually exclusive and clear. 
2. Details about LLM-judge are somewhat insufficient. Different LLM-judges may have different preferences. In addition, according to results in this paper, most of them are less than 75% correct. Therefore, at least some LLM-judge results may be incorrect. I am wondering, since the authors use LLM-judge for all QA pairs, have the authors evaluated the consistency between different LLM-judges, and between LLM-judges and human annotators? Such details will be useful for practitioners who want to curate training datasets and would also need LLM-judges to filter the data. In addition, as the authors did not provide a detailed list of problem types (e.g. choice, open form generation, extraction, etc.), it is not clear why full LLM-judge is necessary.

### Questions
Please refer to 'weaknesses'.

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
This paper introduces EcomEval, a comprehensive multilingual and multimodal benchmark for evaluating LLMs in e-commerce applications. The benchmark covers six categories and 37 tasks across seven languages, with approximately 3,100 items sourced primarily from authentic customer queries and transaction logs. The authors employ a semi-automatic construction pipeline where LLMs generate initial responses that are subsequently reviewed by over 50 expert annotators. They define difficulty levels for tasks and questions, and evaluate 19 state-of-the-art LLMs, revealing significant performance disparities across different e-commerce scenarios.

### Strengths
- The benchmark addresses a critical gap in e-commerce LLM evaluation by providing extensive task diversity (37 tasks across 6 categories) and incorporating authentic data from real customer interactions rather than synthetic datasets. The benchmark also includes multimodal tasks and covers important but underexplored areas like shopping guidance and after-sales service.

- The benchmark involves seven languages. The coverage of five low-resource Southeast Asian languages helps multilingual e-commerce evaluation.

- The semi-automatic pipeline combining LLM generation with expert human review from over 50 annotators ensures both scalability and quality. The difficulty-aware design with calibrated difficulty levels at both task and item granularity enables fine-grained model assessment and provides actionable insights for researchers and practitioners.

### Weaknesses
- The heavy reliance on GPT-4.1 as the primary judge raises concerns about evaluation bias, especially when evaluating competing models. While expert review is mentioned, the extent and systematic nature of this review are unclear. The 0-3 scoring rubric is quite coarse and may not capture subtle differences in model performance, particularly for complex generative tasks.

- The paper lacks a deep analysis of why certain models perform poorly on specific tasks or languages. The discussion of failure cases is limited, and there's insufficient exploration of the relationship between model capabilities and e-commerce-specific requirements. 

- Some individual tasks may have relatively few examples (3100 items across 37 tasks), potentially limiting the statistical significance of comparisons.

- The distribution of tasks across categories and languages isn't clearly documented, which could affect the interpretation of aggregate results.

### Questions
Please see the weakness section

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
This is a comprehensive and timely paper introducing EcomEval, a much-needed benchmark for evaluating Large Language Models (LLMs) and Multimodal LLMs (MLLMs) specifically in the specialized, global domain of e-commerce.
The authors successfully address several critical limitations found in prior e-commerce benchmarks, such as limited task diversity, lack of multimodal data, reliance on synthetic data, and a narrow linguistic focus. The results of evaluating 19 state-of-the-art models confirm the benchmark's value by clearly differentiating model capabilities and exposing performance gaps in domain-specific and cross-lingual tasks.
Overall, this work makes a substantial contribution to domain-specific LLM evaluation by prioritizing authenticity, breadth, and difficulty calibration.

### Strengths
1. Comprehensive Scope and Multimodality: EcomEval achieves broad coverage by encompassing six primary categories and 37 distinct tasks, including eight essential multimodal tasks. This comprehensive classification system moves beyond the limited taxonomies of prior work, addressing real business needs ranging from product QA to intent understanding and multimodal content analysis.

2. Authenticity and Real-World Data Grounding: A significant strength is the reliance on authentic data, with most items derived from real user queries and transaction logs. This methodology captures the "noisy and heterogeneous nature" of genuine customer–merchant interactions, overcoming the limitations of benchmarks based purely on synthetic or curated instruction data.

3. Critical Multilingual Breadth: The benchmark supports evaluation across seven languages, including English, Chinese, and critically, five low-resource Southeast Asian languages (Vietnamese, Thai, Indonesian, Malay, and Portuguese). This directly addresses the narrow English/Chinese focus of previous work and reflects the truly global scale of e-commerce. The results demonstrate that performance variance is substantial in these low-resource languages.

4. Rigorous Quality Control via Expert Annotation: The construction employs a quality-assured, semi-automatic pipeline where LLMs draft initial responses, which are then refined and verified by a team of over 50 expert annotators. These experts possess strong e-commerce and multilingual expertise, ensuring the factual correctness and coherence of the questions and reference answers across languages.

### Weaknesses
1. Limitation to Single-Turn Tasks: A primary recognized limitation is that the benchmark mainly consists of single-turn questions. While the authors note that multi-turn tasks relevant to online shopping scenarios are crucial for assessing true conversational ability (such as interactive product guidance dialogues), the current version lacks this multi-turn capability, which is identified as an area for future work.

2. Modest Dataset Size: Although the data quality is high, the overall size of EcomEval is stated as approximately 3,100 items. Given that this dataset is divided across six categories, 37 tasks, and seven languages, the sample size per specific task/language combination might be relatively small, potentially affecting the statistical significance for deep-dive analysis into low-resource language performance.

3. Dependency on Proprietary LLM-as-a-Judge: The evaluation method relies on GPT-4.1 as the LLM-as-a-judge to score model responses. While expert annotators review the scores, the initial dependence on a closed-source, proprietary model for assigning scores introduces a challenge regarding transparency and guaranteed reproducibility for researchers outside the ecosystem of the judging model.

4. Observed Weakness in Complex Generative Tasks: The experimental analysis reveals that both proprietary and open-source models perform poorly in e-commerce generative tasks, such as product tag generation and product title refinement. The examples provided show models failing to adhere to complex, multi-step instructions, suggesting that while the "hard" tasks successfully test LLM limitations, the high complexity of the prompt rules might occasionally lead to brittleness or lack of generalization in current models, requiring practitioners to simplify real-world tasks.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- **Introduction of EcomEval:** EcomEval is a comprehensive benchmark to evaluate Large Language Models (LLMs) and Multimodal LLMs (MLLMs) on complex, real-world e-commerce tasks.  
- **Addressing Previous Benchmark Limitations:** It resolves issues such as limited task variety, absence of multimodal elements, and narrow linguistic scope.  
- **Diverse Task Coverage:** EcomEval includes 37 tasks across six categories and incorporates eight multimodal components for robust evaluation.  
- **Data Authenticity:** The dataset is derived from real customer queries and transaction logs, capturing authentic and noisy customer–merchant interactions unlike synthetic datasets.  
- **Validation through State-of-the-Art Models:** 19 leading models were evaluated, revealing performance gaps in areas like conversational recommendation, complex cross-lingual tasks, and specialized generative tasks.  
- **Multilingual Focus:** The benchmark spans seven languages, including five low-resource Southeast Asian languages, enabling comprehensive multilingual evaluation.  
- **Rigorous Data Quality Assurance:** The data pipeline leverages LLMs for initial draft creation and ensures high quality and factual correctness through human verification by 50 expert annotators.  
- **Exposure of Performance Gaps:** EcomEval highlights key weaknesses in shopping guidance, after-sales service, and other critical e-commerce areas previously overlooked.  
- **Actionable Insights for LLM Development:** By exposing model limitations, EcomEval provides valuable guidance for improving domain-specific LLM research and deployment.

### Strengths
- **Comprehensive Scope:** EcomEval covers **37 diverse tasks** across six primary categories, reflecting genuine business needs.  
- **Multilingual Breadth:** The benchmark spans **seven languages**, including **five low-resource Southeast Asian languages**, addressing a critical gap in prior work.  
- **Multimodal Integration:** **Eight tasks** incorporate multimodal data, a feature largely absent in other e-commerce evaluations.  
- **Data Authenticity:** Most items are derived from **real user queries and transaction logs**, ensuring the data reflects real-world noise and complexity.  
- **Inclusion of Novel Tasks:** EcomEval incorporates important, previously missing **real-world scenarios**, such as **shopping guides** and **after-sales service tasks**.  
- **Difficulty Calibration:** The benchmark includes **calibrated difficulty levels** at both the task and item granularity to create challenge-oriented evaluations.  
- **High-Quality Assurance:** The dataset construction involves a **semi-automatic pipeline** rigorously verified by **over 50 expert annotators** proficient in e-commerce and target languages.  
- **Extensive Benchmarking:** Validation included **comprehensive experiments** on **19 state-of-the-art closed-source and open-source LLMs/MLLMs**.  
- **Revealing Insights:** The evaluation effectively differentiates model capabilities, exposing **specific weaknesses** (e.g., generative tasks and low-resource languages).  
- **Reproducibility:** The full dataset, including **difficulty tags**, will be released to support **reproducible research**.

### Weaknesses
- Despite the benchmark’s rigor and comprehensive coverage across 37 tasks, its scope is currently limited regarding **conversational complexity** . The authors explicitly note that **EcomEval** mainly consists of **single-turn questions**, suggesting that it does not fully assess the **multi-turn capability** required for complex, real-world shopping assistants. This is a critical gap, as many e-commerce interactions, such as interactive product guidance dialogues and recommendation sessions, inherently require multi-hop reasoning, sequential context and dialogue management. 

- Furthermore, the methodology relies on a **proprietary model** (GPT-4.1) to function as the primary judge for evaluating model responses, introducing an **external dependency** that must be manually reviewed by experts to ensure correctness. While the evaluation setup is robust, this reliance on a closed-source system for crucial scoring diminishes the self-contained nature of the benchmark

- Actionable insights for improving the work should center on expanding the dataset depth in crucial areas. Although EcomEval boasts 37 tasks and 7 languages, its total size of approximately **3,100 items** may limit the number of data points available for fine-grained analysis of specific low-resource language tasks or within the new, challenging categories like shopping guidance and after-sales service. 

- Additionally, while the dataset is primarily authentic, certain tasks (e.g., product inquiry, numerical reasoning, recommendation) were partially derived from and translated from prior benchmarks like **eCeLLM** and **Shopping MMLU**, which may contain synthetic or curated instruction data, slightly diluting the overall authenticity of every item in the evaluation suite.

-  **Lack of Multi-Turn Tasks:** The benchmark primarily uses **single-turn questions**, which fails to assess the crucial **multi-turn capability** required for handling real-world interactive **shopping-assistant dialogues**. Future work should incorporate **multi-turn tasks** relevant to online shopping scenarios to provide a more holistic evaluation.

- **Weakness in Generative Tasks:** LLMs, both closed-source and open-source, perform poorly in e-commerce **generative tasks**, specifically struggling with **product tag generation** and **product title generation** by often missing selling points. The benchmark could be strengthened by providing more fine-grained insights or auxiliary data (e.g., related product catalogs) to challenge models specifically on synthesizing effective marketing content.

- **Difficulty in User Understanding:** Models frequently underperform in the **user understanding** category, often overlooking crucial information such as product categories and targeted buyers in tasks like query-product relevance measurement. This suggests the need for more complex reasoning chains or examples demonstrating nuance in buyer intent.

- Presentation of the paper can be greatly improved. Also there are no mentions of sampling parameters (temperature, top-k etc) used for LLM inference. 

- Why more popular languages such as Spanish not considered as they have a large number of speakers across the world and would be good to have in a comprehensive benchmark.

### Questions
1. What is the **size** of **benchmark dataset** being proposed? Is it just 3100 items? (Line 105)
2. Why not consider other **languages** such as **Spanish** which are also widely spoken across many countries and have many speakers? It is well known that LLMs struggle with low resource languages but the dataset would be more useful if authors consider other important languages apart from English, Chinese (for which decent benchmarks exist).
3. In Tables 2 and 4, authors are doing a **simple average** across the scores that are linearly scaled from 0-100 with arbitrary scoring (>80 easy, 70-80 : moderate, <70 : difficult). There are 2 major issues with this **methodology**. Firstly without knowing how many samples were being considered for each individual task, a simple average doesn't reflect the underlying reality. If the authors believe this is reasonable why are there no averages calculated for Table 3? Secondly the difficulty scoring criteria (Section 3.2) is arbitrary and not very scientific.
4. Using **GPT4.1** as a **judge** is ok but evaluating responses generated by the same GPT4.1 model could lead to bias and hence incorrect inference especially for the same model. Also why not use the best state of the art model as a judge?
5. Some of the models evaluated are **thinking models** whereas others are not but authors do not evaluate any simple **prompting techniques** such as Chain-of-thought or Few shot prompting etc which are very commonly used in e-commerce domain when using LLMs/MLLMs for downstream tasks. Also no details of sampling parameters (temperature, top-k etc) are provided. 
6. In Table D1, why is the correct answer **A**? The **shape mismatch** (oval vs rectangular) cannot be ignored and hence the query-product pair is actually irrelevant and not a substitute. Table D3 and D4 also show cherry picked examples which could be ambiguous and need more context to arrive at a conclusion.

### Soundness
2

### Presentation
2

### Contribution
3
