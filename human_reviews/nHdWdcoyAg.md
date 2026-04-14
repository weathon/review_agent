# SURE-VQA: Systematic Understanding of Robustness Evaluation in Medical VQA Tasks

- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
Vision-Language Models (VLMs) have great potential in medical tasks, like Visual Question Answering (VQA), where they could act as interactive assistants for both patients and clinicians. Yet their robustness to distribution shifts on unseen data remains a critical concern for safe deployment. Evaluating such robustness requires a controlled experimental setup that allows for systematic insights into the model's behavior. However, we demonstrate that current setups fail to offer sufficiently thorough evaluations, limiting their ability to accurately assess model robustness.
To address this gap, our work introduces a novel framework, called SURE-VQA, centered around three key requirements to overcome the current pitfalls and systematically analyze the robustness of VLMs: 1) Since robustness on synthetic shifts does not necessarily translate to real-world shifts, robustness should be measured on real-world shifts that are inherent to the VQA data; 2) Traditional token-matching metrics often fail to capture underlying semantics, necessitating the use of large language models (LLMs) for more accurate semantic evaluation; 3) Model performance often lacks interpretability due to missing sanity baselines, thus meaningful baselines should be reported that allow assessing the multimodal impact on the VLM.
To demonstrate the relevance of this framework, we conduct a study on the robustness of various Parameter-Efficient Fine-Tuning (PEFT) methods across three medical datasets with four different types of distribution shifts. 
Our study reveals several important findings: 1) Sanity baselines that do not utilize image data can perform surprisingly well; 2) We confirm LoRA as the best-performing PEFT method; 3) No PEFT method consistently outperforms others in terms of robustness to shifts. Code is provided at https://github.com/KOFRJO/sure-vqa.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The robustness of vision-language models (VLMs) to distribution shifts is critical for safe deployment. However, existing benchmarks fail to provide realistic data shifts, rely on traditional metrics that do not capture the underlying semantics, and lack sanity baselines for interpreting performance. The paper proposes a flexible framework, called SURE-VQA, that addresses the above pitfalls in the evaluation of VLM robustness in the medical domain. The relevance of SURE-VQA is demonstrated via comparing the robustness of VLMs fine-tuned on medical domains. LLM-based metrics are also proposed and validated via a human rater study.

### Strengths
- The paper is well-written and easy to follow.

- A new dataset that consists of realistic shifts in both image and text domains is constructed. The dataset can be used to evaluate the robustness of various medical VLMs.

- The sanity baselines highlight the pitfall of VLMs in using linguistic shortcuts for predictions without the information in the images.

### Weaknesses
- The constructed dataset is a simple combination of three existing datasets. Although the paper aims to create more realistic shifts by considering both vision and text variations, in each VQA pair, only the shift in one modality is presented. Therefore, the SURE-VQA dataset looks incremental and does not provide novel insights into the robustness of VLMs against multimodal distribution shifts. It would be helpful to introduce shifts in both modalities in a VQA pair.

- The experiments in the paper only focus on one VLM. Results from other VLMs, such as Med-Flamingo [1] and PathChat [2],  may also be given to provide a more comprehensive view on the robustness of VLMs and to demonstrate the utility of the proposed dataset.

- It would be beneficial to introduce some mechanism to control the shifts in SURE-VQA to provide a more holistic view on the robustness of VLMs against distribution shifts. For example, control the number of shifted samples from each type in the test set.

- The use of large language models as evaluators does not appear to be very effective and may not be the best choice for the SURE-VQA dataset. First, on the SLAKE dataset, the correlation score achieved by the Mistral model is comparable to that of traditional metrics. Second, on the MIMIC dataset, traditional metrics yield higher correlation scores than the Mistral model. More importantly, as shown in Figure 3, the two human raters do not achieve much consensus on the SLAKE and OVQA datasets (with low correlation scores), making it unclear whether a high score by the Mistral model truly reflects the evaluation quality. 

[1] Moor et al., Med-Flamingo: a Multimodal Medical Few-shot Learner, Machine Learning for Health Symposium, 2023.\
[2] Lu et al., A multimodal generative AI copilot for human pathology, Nature, 2024.

### Questions
- With multiple LLMs available, is there a specific reason for choosing the Mistral model as the evaluator?
- Why does the paper focus on analyzing different PEFT methods using the constructed dataset instead of different VLMs?

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
"SURE-VQA" introduces a framework to evaluate the robustness of Vision-Language Models (VLMs) in medical Visual Question Answering (VQA), specifically in the context of real-world clinical shifts. The framework addresses three limitations in current robustness testing: reliance on synthetic shifts, lack of semantic-aware metrics, and absence of interpretability-oriented sanity baselines. The authors test various parameter-efficient fine-tuning (PEFT) methods on multiple medical datasets, comparing their robustness against distribution shifts such as gender, ethnicity, and age.

### Strengths
SURE-VQA’s focus on clinically relevant shifts (e.g., gender and ethnicity) is valuable for medical VQA applications, making it more practical for real-world use.
The code is open-sourced.

### Weaknesses
The authors present different types of shifts in the dataset in Figure 2 and this work is to evaluate the robustnees of VLMs in medical VQA tasks. So how do you consider the potential shifts between pre-training distribution in VLMs and dataset distribution? I mean, there may be a case that distribution shifts indeed exist in the dataset but the dataset distribution is totally included in the pre-training data of VLMs, then that makes no sense to evaluate the robustness of VLMs. I think a definition regarding 'distribution shifts' here is required [Han et al.] and hope authors can clarify this. Or have you performed any investigation to ensure the 'distribution shifts' are indeed exist among pre-training, training and test distributions?

Han, Z., Zhou, G., He, R., Wang, J., Wu, T., Yin, Y., Khan, S., Yao, L., Liu, T. and Zhang, K., 2023. How well does gpt-4v (ision) adapt to distribution shifts? a preliminary investigation. arXiv preprint arXiv:2312.07424.

### Questions
1. In human rater study, the kendall correlation between two human raters is supposed to be the highest and the Mistral is expected to be closest to humans' scores. But in figure 3, the correlation between humans is even the worst in SLAKE and not the highest in OVQA. How to understand this phenomenon?
2. In Figure 2, the authors define several distribution shifts. I was wondering if these shifts are commonly ackonwledged in OoD fields [Ye et al., Gulrajani et al.] or what's the basis for the authors to propopse these shift types?
3. As the paper focuses on medical setting, the selected shifts, though realistic, could be expanded to reflect more clinically meaningful scenarios. Shifts in patient demographics (e.g., age, gender, ethnicity) are useful here, but modality and especially question type shift may not correlated with medical setting.



Ye, N., Li, K., Bai, H., Yu, R., Hong, L., Zhou, F., Li, Z. and Zhu, J., 2022. Ood-bench: Quantifying and understanding two dimensions of out-of-distribution generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7947-7958).

Gulrajani, I. and Lopez-Paz, D., 2020. In search of lost domain generalization. arXiv preprint arXiv:2007.01434.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper discusses a framework called SURE-VQA, which aims to systematically assess the robustness of visual-language models (VLMs) in medical visual question answering (VQA) tasks. The authors point out that while there are benchmarks for evaluating the robustness of VLMs, they fail to provide a sufficient framework to effectively address real-world issues, particularly concerning more realistic data variations. To address these problems, the authors propose the SURE-VQA framework, based on three key requirements: 1. Robustness should be evaluated against the inherent real-world shifts in VQA data; 2. Large language models (LLMs) should be used for more accurate semantic evaluations, replacing traditional label-matching metrics; 3. Meaningful baselines should be provided to assess the multimodal impacts of VLMs. The authors demonstrate the relevance of the SURE-VQA framework by studying the robustness of different parameter-efficient fine-tuning (PEFT) methods on three medical datasets.

### Strengths
1. The paper proposes a new evaluation framework (SURE-VQA) that addresses issues present in current evaluation methods for visual question answering (VQA) tasks, particularly in dealing with real-world distribution shifts. This framework may incorporate new evaluation metrics or methods, enhancing the evaluation standards for VQA models.

2. It utilizes abundant and representative datasets and proposes a comprehensive taxonomy that simulates four types of real-world data distribution shifts. The shifts encompass Acquisition, Question Type, Manifestation, and Population, broadly simulating various shifts that may occur in medical VQA datasets, achieving a comprehensive and reliable evaluation.

3. The study tests the impact of different PEFT algorithms on model robustness and includes comparisons with the sanity baseline, analyzing the model's dependence on visual information.

4. The figures are clear and significantly enhance the readability of the article.

### Weaknesses
1. The article briefly mentions the scope of robustness discussed in the Introduction and R1 sections, but it lacks a more explicit and comprehensive definition of the robustness of VLMs in Med VQA tasks. It may necessary to further compare with the existing work on the evaluation of VLM generalization ability to illustrate the innovation points.

2. Based on the descriptions in the article, the focus primarily lies on the interference resistance of VLMs against data distribution shifts. The authors list several scenarios of data distribution variations, which appear to resemble an evaluation of the model's domain adaptation capability. Are there any related works on VLM domain adaptation capabilities in the medical field that are comparable to this work?

3. Regarding the inconsistency in the model's robustness on open-ended questions versus closed-ended questions mentioned in sections 481-485 of "Comparison Between Shifts," where performance shows opposing results between SLAKE and OVQA, is there an explanation for this? Particularly in the SLAKE dataset, the performance on the Open-Ended Question Type Shift data appears to be higher on the out-of-distribution (OoD) dataset. Can this be elucidated?

4. The experimental results do not yield a unified conclusion or significant implications. It would be beneficial to summarize the key factors that significantly influence the robustness of VLMs in Med VQA tasks. Additionally, further discussion on methods for improving model robustness on out-of-distribution (OoD) data, based on the analysis of the impact of PEFT methods on model robustness, would be valuable. Relevant literature from general domains, such as "Ma J, Wang P, Kong D, et al. Robust visual question answering: Datasets, methods, and future challenges[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024;" and "Yoon J S, Oh K, Shin Y, et al. Domain generalization for medical image analysis: A survey[J]. arXiv preprint arXiv:2310.08598, 2023," could provide useful insights.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper investigates the out-of-distribution (OOD) robustness of vision-language models (VLMs) in medical visual question answering (VQA) tasks. To improve the evaluation of model robustness, the authors introduce an evaluation framework, SURE-VQA, which encompasses three key requirements: Realistic Shifts, Appropriate Metrics, and Relevant Sanity Baselines. Experiments were conducted using parameter-efficient fine-tuning (PEFT) methods on three existing medical VQA datasets. The study concludes by highlighting three major shortcomings in current approaches and underscoring findings from the PEFT robustness analysis, which shows that no single PEFT method consistently outperforms the others in robustness.

### Strengths
1. The paper is well-structured and easy to follow, with clear figures and visualizations that enhance comprehension.
2. The three requirements proposed address key concerns in the field of OOD robustness.
3. The codebase is open-sourced, which may facilitate future researchers to evaluate other VLMs.

### Weaknesses
1. Only LLaVA-Med is evaluated in the experiments, which restricts the generalizability of the findings. Testing additional models, especially state-of-the-art general-purposed VLMs, would provide a more comprehensive view of robustness across different architectures.
2. The experimental section focuses heavily on PEFT methods, which may divert focus from the broader aim of evaluating overall robustness. This narrow focus could limit insights into robustness aspects that might arise from other fine-tuning or training approaches relevant to VLMs in medical VQA.
3. The primary contribution is the SURE-VQA framework itself. Although it establishes a valuable structure for evaluation, the paper lacks contributions in model development or methodological advancements that could directly improve robustness.
4. The paper uses Mistral and LLM-based metrics to evaluate semantic similarity, but it may not thoroughly address possible biases or limitations inherent in these evaluators, especially in the medical context.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
