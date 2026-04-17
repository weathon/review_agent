# DIYHealth Suite: Dataset, Model, and Benchmark for Health Management at Home

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Generative AI is reshaping healthcare by enhancing multimodal data interpretation, clinical insight generation, and personalized decision support. However, existing advances remain tightly coupled with hospital-grade devices, restricting accessibility and use for anytime, anywhere health management in non-clinical settings. With the proliferation of wearables, mobile sensors, and telemedicine, healthcare is
shifting toward the home, giving rise to the emerging field of Diagnosis-It-Yourself (DIY) at home, i.e., home care. Despite this promise, several distinctive challenges remain: (i) home-collected data are heterogeneous, exacerbated by the absence of standardized large-scale datasets; (ii) models require adaptation to highly variable task demands and dynamically evolving individual conditions; (iii) the broad spectrum of home care tasks lacks a unified benchmark for systematic evaluation. In this paper, we present DIYHealth Suite, a comprehensive framework designed to address these challenges through a tailored dataset, model, and benchmark. We first curate DIYHealth-900K, a large-scale multimodal dataset capturing diverse real-world home care scenarios. Building on this, we propose DIYHealthGPT, an adaptive foundation model for home-based health management, powered by the novel Hybrid Hyper Low-Rank Adaptation technique, which integrates expert mixtures with hypernetwork-driven modulation to balance cross-task generalization and instance-level personalization. Finally, we establish DIYHealthBench, the first benchmark to evaluate foundation models on home care tasks. Extensive experiments demonstrate that DIYHealthGPT delivers state-of-the-art performance over both general-purpose and medical-specific baselines on 11 home care tasks in both open-QA and closed-QA settings, laying the groundwork for the next generation of AI-driven, personalized, and scalable health management at home.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents DIYHealth Suite, a comprehensive framework for AI-driven home healthcare. To address challenges such as data heterogeneity, task variability, and lack of benchmarks in non-clinical settings, the authors introduce three key components: DIYHealth-900K, a large-scale multimodal dataset of real-world home care scenarios; DIYHealthGPT, an adaptive foundation model leveraging a novel Hybrid Hyper Low-Rank Adaptation technique for balancing generalization and personalization; and DIYHealthBench, the first benchmark for evaluating home care tasks. The proposed model achieves state-of-the-art results on 11 tasks, demonstrating its effectiveness for personalized, scalable health management at home.

### Strengths
1. The home healthcare scenario addressed in this work is practical and useful.
2. The dataset and benchmark are comprehensive, covering a wide range of home healthcare tasks.
3. The effort in constructing the dataset—despite relying solely on public sources—is commendable.
4. The proposed adaptation method is empirically validated and demonstrates considerable effectiveness.

### Weaknesses
1. DIYHealth-900K is constructed entirely from existing public datasets, so no new data or tasks are introduced in this work.
2. The strong performance of DIYHealthGPT is largely attributable to training on in-distribution data, as the training and test sets are drawn from the same dataset. Comparisons against other models—whether general-purpose or medical—are therefore unfair, since those baselines are effectively evaluated in a zero-shot setting. This shifts the novelty from an "adaptive" foundation model to a "unique" dataset, which itself is not produced by this work.
3. The proposed evaluation tasks appear relatively easy, with even small models achieving strong results and the proposed method approaching near-perfect performance in many cases. This suggests that larger models may already saturate the benchmark, limiting its usefulness for measuring state-of-the-art progress.

### Questions
1. The paper lacks an explanation of the used evaluation metrics, such as Matthews Correlation Coefficient (MCC). Providing their definition, interpretation, and justification for use would improve the presentation.

### Soundness
2

### Presentation
3

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
This paper introduces a unified framework for AI-based home healthcare. It includes DIYHealth-900K, a large multimodal dataset, DIYHealthGPT, a foundation model using a novel Hybrid Hyper Low-Rank Adaptation (H2LoRA) for personalized reasoning, and DIYHealthBench, a benchmark for 11 home care tasks. Experiments show that DIYHealthGPT outperforms both general and medical foundation models in accuracy and biomedical fidelity, advancing personalized health management beyond clinical settings.

### Strengths
1. The experiments are comprehensive. Comparisons between DIYHealthGPT and baselines in both general and medical domains demonstrate the effectiveness of DIYHealthGPT, while ablation studies further showcase the correctness of rationale behind the design of each component of DIYHealthGPT. 

2. Abundant content regarding DIYHealthBench notably enhances the readability of this study, facilating readers' understanding on the data. 

3.Repo has been provided with detailed instructions and clear codes, ensuring decent reproducibility.

### Weaknesses
1. Technically, the proposed DIYHealthGPT is not considerably novel. Previous works in both general [1] and medical [2] domains have already applied similar MoE-based architecture to benefit the training of foundation model. In my perspective, the major reason that DIYHealthGPT outperforms other baselines is that it has been further trained with data in DIYHealth-900K, thus customized to this task, rather than that it has apparently more advanced architecture or training strategy. It will be beneficial if the authors could involve more discussions regarding the performance in this perspective. 

[1]. Chen, Zeren, et al. "Octavius: Mitigating Task Interference in MLLMs via LoRA-MoE." The Twelfth International Conference on Learning Representations.
[2]. Wang, Xiaochen, et al. "FEDKIM: Adaptive Federated Knowledge Injection into Medical Foundation Models." Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 2024.

2. It is still not fully clear to me that how the DIY healthcare scenario becomes different from other healthcare applications [3]. While paying notable attention to the method and data, the authors did not fully demonstrate the uniqueness of the setting, obscuring the motivation and undermining the evaluation of this study.

[3]. Liu, Jie, et al. "Asclepius: A spectrum evaluation benchmark for medical multi-modal large language models." Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2025.

### Questions
I notice that although the authors have created a repo for the codes, the data are still unavailable. I am personally curious whether and how the authors would like to publicize the data.

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
3

### Summary
To solve the problem that most medical multimodels are built for hospitals. The paper introduces DIYHealth Suite, which includes DIYHealth-900K, DIYHealthGPT, and DIYBench. Research shows an average of 22.7% improvement in ACC under closed-QA and 16.7% in F1-Bio under open-QA. In addition, DIY HealthGPT receives the most Rank-1 preferences for conciseness, correctness, relevance, and actionability.

### Strengths
1. Colorful and clear diagrams that are helpful for comprehension.
2. This paper is well written.
3. It achieves state-of-the-art performance across all home care tasks, exceeding the robust general and medical LVLM baseline by an average of 22.7% in closed-ended question accuracy and by 16.7% in open-ended question F1-Bio accuracy.
4. This research is interesting and has clear application potential. It is built for real home settings using consumer-grade phones and wearables, covers 11 diverse tasks, delivers personalized and actionable guidance, and is clinician-validated, making it directly usable for at-home health management.

### Weaknesses
1. Most training labels are machine-generated. Only 10% of labels are audited by humans, which could lead to the model learning the wrong things. 
2. The paper does not provide real user feedback for assessing usability.  
3. The paper does not compare performance across subgroups such as gender, sex, race or ethnicity, age, skin tone, primary language, device type, and lighting conditions. Without subgroup checks, hidden disparities may persist, and some users could receive less accurate or less safe guidance. This weakens claims of real-world reliability and raises deployment risk.

### Questions
See weaknesses.

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
The paper investigates the problem of home care and medicine access through a multimodal question-answering framework that integrates both textual and imaging data. It presents three primary contributions: (1) the creation of a large-scale dataset, (2) the development of a GPT-based model designed for this domain, and (3) benchmark results evaluated on a curated test set.

### Strengths
- The paper addresses an important gap in current research related to automated medical assistance in home care settings.
- It is comprehensive in scope, combining dataset development, model design, and evaluation. 
- The proposed model is distinctive in its integration of task-level and instance-level cross-information, which enhances multimodal reasoning capabilities.

### Weaknesses
- Figure 3 does not accurately reflect the corresponding mathematical formulation presented in the text.
- The description of the publicly available datasets—collected from 20 open sources—is missing from the main paper.
- It is unclear whether the baseline models were evaluated in a zero-shot setting. If so, this would constitute an unfair comparison with the proposed model, which is fine-tuned with task-specific data.

### Questions
- How can matrix A be described as a shared projection if a separate one exists for each task?
- There is ambiguity around Equation (4) and the explanation on line 286; these should be clarified.
- The statement “We sample 10% of training data per task”—is this sampling performed in each epoch or only once during training?
- In Stage 3, a dedicated H2LoRA block per task is mentioned, please elaborate on its implementation.
- DIYHealthBench is derived from the designated DIYHealth-900K test set and reportedly includes 12,167 carefully sampled examples. How were these examples selected, and according to what criteria?
- Were hyperparameter tuning procedures conducted, and if so, what methodology was followed?
- What is the loss function used during training?
- Please specify the model size and architectural parameters.
- How were prompts designed for both training and evaluation?
- What temperature settings were used for the baseline models during inference?
- How was interrater agreement assessed, and what was the resulting Cohen’s kappa score?

### Soundness
3

### Presentation
3

### Contribution
4
