# MediTab: Scaling Medical Tabular Data Predictors via Data Consolidation, Enrichment, and Refinement

- Decision: Reject
- Scores: 6, 3, 6

## Abstract
Tabular data prediction has been employed in medical applications such as patient health risk prediction. However, existing methods usually revolve around the algorithm design while overlooking the significance of data engineering. Medical tabular datasets frequently exhibit significant heterogeneity across different sources, with limited sample sizes per source. As such, previous predictors are often trained on manually curated small datasets that struggle to generalize across different tabular datasets during inference.

This paper proposes to scale medical tabular data predictors (MediTab) to various tabular inputs with varying features. The method uses a data engine that leverages large language models (LLMs) to consolidate tabular samples to overcome the barrier across tables with distinct schema. It also aligns out-domain data with the target task using a "learn, annotate, and refinement'' pipeline. The expanded training data then enables the pre-trained MediTab to infer for arbitrary tabular input in the domain without fine-tuning, resulting in significant improvements over supervised baselines: it reaches an average ranking of 1.57 and 1.00 on 7 patient outcome prediction datasets and 3 trial outcome prediction datasets, respectively. In addition, MediTab exhibits impressive zero-shot performances: it outperforms supervised XGBoost models by 8.9% and 17.2% on average in two prediction tasks, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes an approach for developing learning prediction models using heterogenous tabular data sources and schemas and across tasks. The approach relies on large language models to represent structured data in natural language as a common representation across contexts, aggregate datasets from like and unlike tasks and populations, and do zero and few-shot prediction. The approach is applied to several medical data (primarily clinical trials with some retrospective observational data). The approach generally outperforms fully-supervised baselines and further performs well in zero and few-shot settings.

### Strengths
* The method appears to yield performant predictive models across datasets and tasks, and without requiring significant amounts of labeled target data. 
* The approach to representing and harmonizing structured data in natural language using LLMs is general and plausibly would continue to work well outside of the context evaluated in this work.

### Weaknesses
* I have several concerns related to clarity and lack of detail given to some important aspects of the methodology and experiments. These are elaborated on in the Questions section below.
* The datasets chosen are relatively small and relatively low-dimensional (e.g. ~10s of fields at most). An area where this method might be useful is with tabular data of much greater dimensionality, as is typical in healthcare contexts.

### Questions
* How are the “External Patient Databases” used (MIMIC-IV and PMC-Patients)? Are they used as supplementary databases during the psuedolabeling step? 
* How is the MIMIC-IV data processed? The information in Table 1 that shows that there are only 2 categorical, 1 binary, and 1 numerical feature in MIMIC-IV. As MIMIC-IV is much richer (potentially thousands of features) it is unclear which components of the database are actually used and no details are provided.
* In section 2.4, why is the initial model trained on all available training data from $T_1$ considered a multi-task model (designated by $f_{MTL}$)? If I understand correctly, this model is trained on one task, but several datasets.
* The description of the psuedolabeling step is not entirely clear to me. Is the idea to take the initial model for the target task, make predictions for the target task on data collected for other tasks, and then use those predictions as pseudo-labels for further training? This seems peculiar because it is not clear that this should fundamentally improve performance for the target task given that the pseudo-labels are essentially just predictions of the target label derived from information in the target task database(s).
* If available, it would be relevant to compare to baselines that pool over datasets with rule-based schema harmonization. For example, in the context of electronic health records and claims data, there are standards such as the OMOP Common Data Model that provide the means of mapping data from disparate sources to a shared schema.
* An ablation experiment that removes the auditing steps (both the LLM sanity check and the Data Shapley checks) and the pseudolabeling step would help gain insight into the marginal value that they provide, especially as they are positioned as the novel methodological contributions of this work relative to TabLLM (if I understand correctly).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The manuscript proposes a framework utilizing LLM to perform alignment between different datasets for the same and different tasks. The framework prompts ChatGPT to summarize each row in a table into text and utilizes BioBERT as a classification model that takes text as input. Moreover, it trains an init model to annotate data from datasets of other tasks and clean such data with Shapley scores into supplementary data samples.

### Strengths
- The paper is well-written and easy to follow.

### Weaknesses
- The framework is quite straightforward, and there is not much technical contribution. It is mostly a combination of multiple existing models. And the idea of transferring tabular data into text is not novel at all. There are a bunch of existing works [1][2][3], including one of their baselines TabLLM[4]. The further incorporation of text information from samples from other datasets is just one trivial step forward. Furthermore, [4] actually proved that a template for transferring the tabular data works better than an LLM. Yet, in this paper, there is no comparison for such serialization methods.

- The author didn’t specify what exact features are included in these experimental datasets. Also, it is unclear how many columns are overlapped between different datasets. Yet, if there is a large portion of feature overlapping, maybe simple concatenation and removing or recoding of the missing columns will work just as well. There is no discussion regarding this whatsoever.

- The step 2 in section 2.2 is confusing:
    - The authors claimed that they used active learning in step 2. Is the “active learning pipeline” method the same as traditional active learning that select informative samples to label? If not, the description can mislead the readers.
    - The authors claimed that they cleaned supplementary dataset T_{1, sup} with a data audit module based on data Shapley scores. More experiments are expected to demonstrate the effectiveness of the audit module. Moreover, it would be better if the authors conducted more ablation studies to show whether the supplementary dataset improve the prediction performance. 

- The datasets in Table 1 contain less than 3000 patients. It is very easy for the LLMs (e.g., BioBERT) to overfit the training set. It is unclear how the authors prevent overfitting during the fine-tuning phase.

- In Table 3, the proposed MediTab exhibits the capability to access multiple datasets during its training, in contrast to the other baseline models, which are constrained to employing a single dataset. This discrepancy in data utilization introduces an element of unfairness in the comparison. It would be more appropriate to conduct a comparison against models that have undergone training on multiple datasets. For instance, TabLLM, being a large language model, can readily undertake multi-dataset training with minor adjustments to its data preprocessing procedures. Therefore, a more equitable comparison would involve evaluating MediTab and TabLLM under identical conditions, both in the context of training on a single dataset and across multiple datasets. 

- Most medical data, like MIMIC-IV, includes timestamp information of the patients’ multiple visits or collections. This framework completely ignores this part of the medical data, which limits their application to real-world clinical environments.

Reference:
1. Bertsimas, Dimitris & Carballo, Kimberly & Ma, Yu & Na, Liangyuan & Boussioux, Léonard & Zeng, Cynthia & Soenksen, Luis & Fuentes, Ignacio. (2022). TabText: a Systematic Approach to Aggregate Knowledge Across Tabular Data Structures. 10.48550/arXiv.2206.10381.
2. Yin, Pengcheng & Neubig, Graham & Yih, Wen-tau & Riedel, Sebastian. TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data. ACL 2020.
3. Li, Y., Li, J., Suhara, Y., Doan, A., and Tan, W.-C. (2020). Deep entity matching with pre-trained language models. Proc. VLDB Endow., 14(1):50–60.
4. Stefan Hegselmann, Alejandro Buendia, Hunter Lang, Monica Agrawal, Xiaoyi Jiang, and David Sontag. Tabllm: Few-shot classification of tabular data with large language models. arXiv preprint arXiv:2210.10723, 2022.

### Questions
1. All questions in the above section.

2. Are there any overlaps of columns between the tabular data for the same tasks? Is it hard to do a simple concatenation? What’s the traditional method for dealing with the missing columns? Are they applicable to this situation?

3. For the choice of BioBERT and the QA model for salinity check, the author did not provide a reason for choosing these models.

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes to scale medical tabular data predictors (MediTab) to handle diverse tabular inputs with varying features. The approach involves utilizing a large language models (LLMs) to merge tabular datasets, addressing the challenges presented by tables with different structures.  Additionally, it establishes a process for aligning out-of-domain data with the specific target task through a "learn, annotate, and refinement" pipeline.

### Strengths
1. The core concept behind MediTab, involving the consolidation, enrichment, and refinement modules, is well-founded in its aim to improve the scalability of predictive models designed for medical tabular data.
2. Implementing a sanity check through Large Language Model (LLM) Reflection is particularly important in the medical domain.
3. The paper is clearly written; the quality is sound.
4. The empirical evaluation in the paper is strong; relevant baselines are considered.
5. The coverage of related work is extensive, with clear distinctions drawn from other studies.

### Weaknesses
1. It is not clear, how was splitting into train/val/test organised?
2. The statement “After the quality check step, we obtain the original task dataset $T$ and the supplementary dataset $T_{sup}$ and have two potential options for model training. The first is to combine both datasets for training, but we have found that this approach results in suboptimal performance.” is a bit unclear. Why it is suboptimal, could authors elaborate on this?
3. Are these datasets prone to missing values, which is a critical concern in the medical domain? If so, what would be the recommended strategy for handling these missing values?
4. Results on Ablation studies on Different Learning strategies are provided. Could authors provide  Ablation studies on the different model components?

### Questions
1. Could the authors elaborate more on the LLM sanity check, as well as the results and tests provided in Appendix C.4-C.5? It has been discussed that it is crucial to conduct thorough evaluations of LLMs in healthcare, with particular attention to aspects of safety, equity, and bias [a]. Could the authors provide their thoughts on why they believe their model satisfies these requirements?
2. Could authors provide more detailed explanation of empirical studies and address points in Weakness section?

a. Singhal et al., Large language models encode clinical knowledge. Nature 620, 172–180 (2023).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
