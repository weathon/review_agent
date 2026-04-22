# Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
Understanding and reasoning with abstractive information from the visual modality presents significant challenges for current multi-modal large language models (MLLMs). Among the various forms of abstractive information, Multi-Modal Relational Knowledge (MMRK), which represents abstract relational structures between multi-modal entities using node-edge formats, remains largely under-explored. In particular, STructured and Abstractive Reasoning (STAR) on such data has received little attention from the research community. To bridge the dual gaps in large-scale high-quality data and capability enhancement methodologies, this paper makes the following key contributions: (i). An automatic STAR data engine capable of synthesizing images with MMRK to build multi-modal instruction data with reliable chain-of-thought thinking for various STAR tasks and (ii). A comprehsive two-stage capability enhancement training framework, accompanied by a suite of evaluation protocols tailored to different STAR tasks. Based upon these contributions, we introduce STAR-64K, a dataset comprising 64K high-quality multi-modal instruction samples, and conduct experiments across 5 open-source MLLMs. Experimental results show that our two-stage enhancement framework enables smaller 3B/7B models to significantly outperform GPT-4o in STAR. Additionally, we provide in-depth analysis regarding the effectiveness of various designs, data transferability, and scalability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a method for automatically generating data to support multimodal relational knowledge and structured abstractive reasoning. Using this approach, the authors construct both a training dataset and a benchmark for evaluating MLLMs. The dataset consists of multimodal knowledge graphs along with questions spanning eight categories: entity counting, relation counting, image counting, triple counting, subgraph description, error detection, entity reasoning, and relation reasoning. To leverage this data, the authors propose a two-stage finetuning pipeline—supervised finetuning followed by preference alignment. Their experiments show that existing models perform poorly on the benchmark, while finetuning on the proposed dataset leads to substantial performance gains.

### Strengths
- The paper is well-written and clearly organized.
- Training on the proposed data leads to clear performance improvements.
- The analysis is thorough, exploring important dimensions such as task transferability, data scalability, and the effectiveness of chain-of-thought reasoning.

### Weaknesses
- The task appears somewhat artificial, with limited discussion of potential real-world applications. The experimental setup further highlights this artificiality: if I understand correctly, the visual format of the multimodal knowledge graphs remains fixed across all instances. Even if such a task were practically relevant, it is unlikely that real-world data would exhibit such consistent structure.
- Given that the dataset encodes factual information (e.g., concepts found in Wikipedia, a common pretraining source for LLMs), it would be useful to disentangle the contribution of visual input from the models’ reliance on prior textual knowledge. One way to validate this would be to include a baseline trained without image input (e.g., an additional row in Table 2 where only the question text is provided). Another option would be to construct counterfactual variants of the data and compare model performance under those conditions.

### Questions
- What are some concrete real-world scenarios where a multimodal knowledge graph of this kind would be useful?
- Have you evaluated the extent to which image information influences model predictions (see Weaknesses point 2)?
- Beyond the benchmark, what potential applications do these data have? Can the proposed method transfer to related tasks that use different schematics or alternative ways of representing relationships?
- What might explain the large performance differences between LLaVA1.5-Next and Qwen2.5VL? In the zero-shot setting, their performance is similar, with Qwen slightly ahead when controlling for model size. However, after training, this relationship reverses. What factors could be driving this change?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the problem of structured and abstractive reasoning over multimodal relational knowledge in MLLMs. The authors developed a data pipeline to construct the MMRK dataset and related evaluation strategies. Based on Qwen2.5-VL and LLaVA-Next, they conducted in-depth experiments, analyzing training strategies and the characteristics of models trained on the MMRK data.

### Strengths
* The authors built a comprehensive data pipeline that covers data source collection, data/instruction/task synthesis, as well as data validation strategies.
* The authors conducted an in-depth exploration of the impact of training strategies on learning MMRK, involving two different mainstream multimodal foundation models and multiple post-training approaches. The experimental results are highly insightful.
* Moreover, the small models trained on the authors’ synthesized data can achieve performance comparable to, or even better than, that of large closed-source models, further confirming the scarcity of data in this field and the significance of the dataset constructed in this work.

### Weaknesses
* The authors’ experiments are mainly centered on in-domain tasks that are similar to the training data, lacking evaluation on real-world problems. As shown in Figure 1, the authors present a variety of scenarios. Experimental results on these scenarios would help to better understand the impact of the authors’ contribution on the model’s real-world performance.
* The authors use an LLM (Qwen2.5-VL-72B) as the judge, but its reliability has not been thoroughly discussed. This is especially noteworthy given that the zero-shot Qwen2.5-VL-32B demonstrates performance comparable to that of Qwen2.5-VL-72B.
* The authors have thoroughly demonstrated the performance improvements brought by the MMRK data. But the baseline models compared by the authors are relatively outdated, and they did not include comparisons with the latest models such as Qwen3-VL, Claude 4.5, or Gemini 2.5 Pro. Comparing against state-of-the-art models would help readers better understand the current capabilities of multimodal models in solving STAR tasks.

### Questions
refer to the weaknesses

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
This paper tackles Structured and Abstractive Reasoning (STAR) with Multi-Modal Relational Knowledge (MMRK), an underexplored area for MLLMs. Key contributions include an automatic STAR data engine for multi-modal instruction synthesis and a two-stage enhancement framework. Introducing the **STAR-64K** dataset, experiments show smaller 3B/7B models outperform GPT-4o in STAR tasks. The work provides valuable insights into model performance, data scalability, and transferability.

### Strengths
1.**Clear Motivation**: The paper presents a well-defined objective, introducing a data engine to synthesize Multi-Modal Relational Knowledge (MMRK) data. Training on this data effectively enhances models' abilities in Structured and Abstractive Reasoning (STAR).

2.**Novel Contribution**: This is the first work to transform multi-modal knowledge graphs into reasoning tasks involving multi-modal relational knowledge. The authors design training data, evaluation benchmarks, and validate the approach on 3B/7B models, addressing a previously unexplored gap in research.

### Weaknesses
While incorporating STAR training effectively enhances multi-modal models' structured and abstractive reasoning capabilities, it remains unclear whether other abilities, such as OCR (TextVQA,DocVQA)and general multi-modal question answering (MME,MMMU), may suffer from catastrophic forgetting. If the improvement in STAR comes at the expense of reducing the model's general versatility or impairing core functions like OCR, the practical significance of the framework would be greatly diminished. Further exploration into the trade-offs between STAR enhancement and general multi-modal model performance is necessary to validate its broader applicability and impact.

### Questions
The key question is whether the addition of STAR capabilities is genuinely impactful or simply included for novelty. It is critical to establish whether integrating STAR tasks enhances the general versatility and performance of multi-modal models. To address this, I strongly recommend the authors conduct an ablation study comparing the performance of a model trained with and without STAR data on comprehensive multi-modal benchmarks, using a baseline such as the LLAVA framework. This experiment would demonstrate the necessity and real-world significance of STAR tasks.  

If this concern is adequately addressed and the results prove STAR's contribution to improving general multi-modal reasoning, I will consider revising my score positively.

### Soundness
2

### Presentation
3

### Contribution
3
