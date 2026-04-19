# Interpretable Bilingual Multimodal Large Language Model for Diverse Biomedical Tasks

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
Several medical Multimodal Large Languange Models (MLLMs) have been developed to address tasks involving visual images with textual instructions across various medical modalities, achieving impressive results. 
Most current medical generalist models are region-agnostic, treating the entire image as a holistic representation. However, they struggle to identify which specific regions they are focusing on when generating a sentence.
To mimic the behavior of doctors, who typically begin by reviewing the entire image before concentrating on specific regions for a thorough evaluation, we aim to enhance the capability of medical MLLMs in understanding anatomical regions within entire medical scans.
To achieve it, we first formulate \textbf{Region-Centric tasks} and construct a \textbf{large-scale dataset, MedRegInstruct,} to incorporate regional information into training. Combining our collected dataset with other medical multimodal corpora for training, we propose a \textbf{Region-Aware medical MLLM, MedRegA}, which is the first bilingual generalist medical AI system to simultaneously handle image-level and region-level medical vision-language tasks across a broad range of modalities. Our MedRegA not only enables three region-centric tasks, but also achieves the best performance for visual question answering, report generation and medical image classification over 8 modalities, showcasing significant versatility. Experiments demonstrate that our model can not only accomplish powerful performance across various medical vision-language tasks in bilingual settings, but also recognize and detect structures in multimodal medical scans, boosting the interpretability and user interactivity of medical MLLMs. The codes and model will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents MedRegA, a region-aware medical multimodal language model capable of handling both image-level and region-level vision-language tasks across diverse medical modalities. The authors propose three Region-Centric tasks and create a large-scale dataset called MedRegInstruct to enhance the model's focus on specific anatomical regions. They introduce a Regional Chain-of-Thought approach to further improve performance. Experiments show that MedRegA outperforms existing models on tasks like visual question answering, report generation, and medical image classification, demonstrating significant versatility and interpretability.

### Strengths
- The comprehensive MedRegInstruct dataset is valuable and likely to have a significant impact on the community.
- Introducing Regional Chain-of-Thought is insightful and significantly improves the model's performance.
- The model achieves state-of-the-art performance without fine-tuning on specific benchmark datasets, indicating strong generalization across different domains.

### Weaknesses
- Human validation of the automatically generated annotations is needed to prove the quality of the dataset.
- The paper lacks comprehensive comparisons in report generation with state-of-the-art methods, such as those on the ReXrank[1] leaderboard.
- Detection task of multiple regions still remain challenging for purposed method.

[1] Rajpurkar Lab. ReXrank Leaderboard. 22 July 2024, https://rajpurkarlab.github.io/rexrank/.

### Questions
Please address the weaknesses mentioned above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work introduces a novel region-centric task and presents a large-scale Chinese medical dataset named MedRegInstruct. To address this new task, the authors developed a bilingual multimodal large language model, MedRegA, which outperforms baseline methods across most evaluation metrics.

### Strengths
1.	The paper is well-organized.
2.	This work introduces a novel task alongside a large-scale medical dataset.
3.	The authors propose a new model that significantly outperforms other models across the majority of datasets and evaluation metrics, particularly for the newly proposed task.

### Weaknesses
1.	The article’s description of the model structure in the main text is incomplete, relying heavily on Figure 4 to convey the model’s architecture. For example, the structure of the encoder and tokenizer remains ambiguous. The authors should clarify which components utilize existing models, which are fine-tuned, and which are trained from scratch.
2.	The fairness of the model comparison is questionable. In Table 1, only a subset of models has been fine-tuned based on the test dataset, leaving others unmodified. Is this due to an inability to fine-tune certain models, or for another reason? The authors should provide detailed criteria for choosing to fine-tune specific models and address the fairness of this experimental design.
3.	On the newly proposed task, the authors’ model demonstrates a substantial performance advantage. They mention that existing open-source models cannot accurately capture coordinates, but it would be insightful to know if fine-tuning on the authors’ proposed dataset could improve this. If fine-tuning other models would yield better results, the authors should do so for a more persuasive comparison. Conversely, if fine-tuning has no impact, the authors should explain the model’s unique advancements to provide deeper insights.
4.	The role of the CoT component is somewhat unclear, with insufficient ablation experiments to support its effectiveness. The authors should conduct a more specific ablation study to illustrate the impact of CoT rather than merely reporting improved results with its inclusion.
5.	While the authors claim that the proposed model can perform diagnostic tasks, there is no clear evaluation or user study provided. A brief evaluation or explanation of the diagnostic results would add clarity.
6.	Several prior studies focus on medical image segmentation, which aligns well with the proposed region-centric task. Have the authors considered using such methods as feature extractors, backbones, or even as comparative models?
7.	The experimental results on report generation are insufficient, as the authors rely solely on descriptive analysis without quantitative evaluation or user studies. Similar to point 5, if the authors wish to validate effectiveness without explicit evaluation metrics, a user study would provide valuable support.

### Questions
1.	Please explain why certain models were fine-tuned in the comparison while others were not.
2.	The authors state that adding CoT improves model performance; however, no comprehensive ablation experiments were conducted to confirm this.
3.	Have the authors considered comparisons using other large models as the backbone?
4.	Do the authors plan to open-source this dataset?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this work, the authors tackle the current problem in MLLMs (multimodal large language models) of how they don't necessarily "focus" on particular regions, but rather take the entire image as context and then have to solve a variety of downstream tasks. To address this problem they offer two main sets of contributions. The first, is the formulation of several region specific tasks, namely region-to-text identification, text-to-region detection, and grounded report generation, which they combine into a dataset called MedRegInstruct. The second is the training strategy and modeling paradigm for a MLLM they propose, which they refer to as MedRegA. MedRegA is a bilingual, as it was trained on both English and Chinese paired datapoints, model that they apply to this larger dataset they prepare in comparison to currently available medical MLLMs.

### Strengths
• The paper really introduces another method of interaction, in the bounding boxes, that can be quite useful as MedRegA is able to reason about specific regions. This is an immensely useful tool for narrowing of problems as opposed to current trained MLLMs have to do.

• The proposed training strategy for learning how to incorporate the bounding boxes is simple and scalable, seems like it could be generally applied to other strategies that want to apply bounding boxes.

• Regional CoT makes a lot of sense intuitively and provides a nice bias for solving region specific problems.

• MedRegA has very strong performance on the chosen set of benchmarks and metrics in comparison to existing state of the art methods.

### Weaknesses
• The section of the report grounding dataset is quite confusing as to how its constructed, and how much it relies on the models it uses for automation for providing correct labels/groundtruth. For example, is InternLM perfect at sub-select different parts of the reports? 

• The paper claims to work across a broad set of modalities, but when considering the grounding dataset it seems that it only considers, as other MLLMs do, Chest Xrays?

• It's unclear if the comparison to existing baselines are exactly fair as the trained MedRegA has access to more data. In fact, compared to the baseline of InternLM, it should be strictly better because MedRegA is trained with InternLM as an initialization. While it's true that the baselines don't provide a mechanism for highlighting particular regions, they still should be adapted to the new data that the authors are evaluating on.

### Questions
• What was the filtering criterion for reducing SA-Med2D-20M?

• Is there a noticeable performance difference between English and Chinese tasks? Is there paired data that this can be evaluated on? • • • 

• Overall, it is unclear from the paper why exactly these two particular languages are used, other than being able to source data for both, rather than just arbitrary languages with medical report data.

• Is the MedRegInstruct dataset planning to be release?
For the Region-Text dataset, why is the case that there needs to be this split into half for region-to-text and half for text-to-region? Can't every datapoint server the purpose of being either during training?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new large-scale, multi-site, medical visual-question answering (VQA) dataset, called MedRegInstruct, with fine-grained region-specific annotation and multiple corresponding region-centric tasks. The proposed dataset contains multiple region-centric evaluation tasks including, region-to-text generation, text-to-region detection, and grounded medical report generation. The paper further proposed a new region-aware multi-modal large language model (MLLM), called, MedRegA, pre-trained with the proposed dataset and regional chain of thought (CoT). The proposed method outperforms multiple existing SOTA medical VQA models on the regional-centric evaluation.

### Strengths
1. The proposed new region-centric medical dataset MedRegInstruct addressed the need for fine-grained, region-level, training and evaluation datasets in the medical domain. It is novel to the field and will help the development of the field as provides a more fine-grained training and evaluation annotation. It is proven that the method pre-trained with the new dataset can demonstrate more robust performance in the targeted region-specific tasks and also other related medical tasks. 
2. The paper has provided enough details about the data curation and evaluation, including the process of question-answering pair generation from medical data with region annotation, a prompt for each downstream evaluation, and detailed settings for each task. The author also seems to claim they will release the code and pre-trained model later.
3. The paper is overall well written with nice figures and a clean presentation. It is easy to follow even if there are many details in it.

### Weaknesses
1. While the proposed dataset is novel and important to the field, the corresponding MedRegA model is not that novel, though I hate to say that, in comparison. The overall model design is basically the same as the base InternVL model and the major change is the training scheme and prompt formulation with the new dataset, which can be extended to other medical VQL models as well. While the author further proposed single-step regional CoT in the paper, it is still relatively straightforward. **Yet**, this is not a major drawback as the performance of this model demonstrates the superiority of training with the proposed dataset.
2. Similarly, it seems that the baselines used in the evaluation are not all fine-tuned with the proposed dataset, which enables the model to understand regional information. Comparing the model trained on the proposed dataset with baselines that are not trained on this dataset is not very fair, to some extent. It would be interesting to see how well each baseline would improve if they were fine-tuned on the proposed dataset, even if just in a few-shot manner.
3. Another concern about this paper is the quality of the data. Most of the regional bounding box-text pairs in the dataset were taken from the existing dataset and generated from an automated rule-based system and an LLM. The evaluation of the data quality is missing in this paper, which is very critical as the main purpose is **fine-grained regional evaluation,** where the quality of the bounding box and the correctness of the corresponding text description are very important. It would be better if some sort of data quality check could be done before releasing it to the public.

### Questions
Overall, this is a good paper with solid contributions and it has a significant meaning to the field. Still, the reviewer has a few questions here.

1. How is the English version of the text report/caption generated for the in-house Chinese data? What kind of translation was used here?
2. According to the “Structure Detection” section on page 5, the structure bounding box in the dataset was generated with a fine-tuned MLLM on the existing Region-Text dataset given the corresponding organ prompt. The reviewer wonders if any kind of evaluation was done on this fine-tuned MLLM, how accurate is the extract bounding box? And why not train a more straightforward detection model for this purpose? Considering that the specific detection model usually performs better than a general MLLM.
3. Also, as mentioned above, the reviewer wonders how much will the other baseline medical MLLMs improve if they were trained with the proposed dataset. The comparison between InternVL and MedRegA in the paper is not very fair as the InternVL was only trained with natural image-text pairs. The reviewer understands it may take tons of extra time/money to conduct such an evaluation, but it would still be an interesting question to explore.
4. According to the abstract, the code and model will be released to the public, but the reviewer wonders if the data will be released as well. This is pretty critical as the dataset is the major contribution of the paper. The answer to this question will influence the reviewer's final score for the paper.

### Soundness
3

### Presentation
3

### Contribution
4
