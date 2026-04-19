# Seeking the Right Question: Towards High-Quality Visual Instruction Generation

- Decision: Reject
- Scores: 3, 3, 6

## Abstract
Large language models achieve significant improvements in instruction following through training with synthetic data. The self-instruct method generates instructions based on manually selected tasks, establishing an annotation-free paradigm for synthesizing instructions. However, the experience of synthesizing language instructions for LLMs does not directly transfer to visual instruction generation. Visual instructions encompass both images and questions, and questions generated directly from images often struggle to form high-quality instructions.
By analyzing real user queries, we summarize the characteristics of high-quality instructions: they require image perception, reasoning, and answerability. We propose a three-stage visual instruction generation pipeline, named "Seeking the Right Question" (SRQ), to produce high-quality instructions. In stage 1, we select 160 instructions that meet high-quality standards as seed questions, categorizing them into eight groups based on multi-modal task types. In stage 2, we introduce capability-driven prompting to generate high-quality questions. In stage 3, we implement an Image Dependency Scoring Mechanism to filter the generated questions. Additionally, we use GPT-4o to directly generate answers, forming $<$image, question, answer$>$ triples for model training.
To demonstrate the effectiveness of SRQ, we construct the high-quality instruction dataset Allava-SRQ from 125,000 images sampled from the Allava dataset. Experimental results show that Allava-SRQ significantly improves the performance of multiple baseline models across various benchmarks. We plan to open-source SRQ and the high-quality instruction dataset Allava-SRQ to promote advancements in the field of visual instruction generation.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The author believes that traditional methods struggle to create high-quality visual instructions because they often rely on questions generated directly from images, which may not effectively challenge the models to improve their reasoning or perception skills. The paper introduces a new visual instruction generation method, "Seeking the Right Question" (SRQ). The SRQ pipeline consists of three stages. In stage 1, high-quality seed questions are selected, meeting standards of image perception, reasoning, and answerability. These questions are then categorized into eight groups, such as Mathematics and Creative Content Creation. In stage 2, capability-driven prompting is employed to generate diverse questions using these categories as prompts, leveraging the power of GPT-4o. In stage 3, an Image Dependency Scoring Mechanism is used to filter questions, ensuring they depend heavily on image content, resulting in effective <image, question, answer> triples for model training. The effectiveness of SRQ is demonstrated by constructing a dataset, Allava-SRQ, from 125,000 images, showing significant performance improvements over previous methods like LLAVA and Allava.

### Strengths
The paper presents its concepts and findings with considerable clarity, making it easy for readers to follow the arguments and grasp the significance of the work. The authors clearly define the problem, specifically addressing the mismatch issues between questions and images in current visual instruction datasets, which is a well-recognized challenge in the field. Additionally, the approach introduced in the paper is straightforward and intuitive, making it easier for other researchers or practitioners to implement and build upon.

### Weaknesses
(1) Overclaim: "The eight categories encompass nearly all task scenarios applicable to the LVLM model." I completely disagree with the author’s statement. These eight categories do not cover all task scenarios and some classifications are not typically applicable to the LVLM model. Please provide further widely accepted theory or experimental evidence, rather than arbitrary assumptions.

(2) Scoring is unreasonable: Many real-world problems involve a coupling of multiple sub-tasks. Some basic sub-problems fit "The question can be completely answered based on the information in the image without inference or assumption (score = 5)," but subsequent sub-questions may require additional reasoning. Using such a standard for filtering can lead to questions that are too simple and do not match the needs of real-world users. Similarly, the subsequent reasoning is flawed: "Consequently, the assessment of answerability is implicitly included in the scoring mechanism that relies on image content, offering a more reliable approach than explicit judgments of whether a question is answerable."

(3) The method is inefficient and difficult to scale: The author's pipeline essentially combines synthesis and filtering. If there is enough original instruction data, using a conventional synthesis method and then filtering to retain high-scoring data could achieve the same effect. Moreover, the author's approach requires generating 8 different questions per image. If a baseline method allows the VLM to randomly generate 8 times more data and then uses a filtering strategy, it might achieve similar or even better results. Since your eight types of questions are completely different, as shown in your case, there is necessarily redundancy. From this perspective, letting the model generate by itself might be better. Thus, important baselines/ablation experiments are missing here.

(4) The method is limited to improving quality at the case level and doesn't ensure or enhance diversity at the dataset distribution level, which is also very important. The authors set eight different categories and 160 seed samples, but there are no additional restrictions on the distribution of input images. If there are many similar images in the vast source collection, how can they ensure the diversity of the final synthesized dataset?

(5) I believe the authors have underestimated the capabilities of VLM, especially GPT-4o. Consolidating the pipeline into a dynamic prompt (including CAPABILITY-DRIVEN PROMPTING FOR QUESTION GENERATION and IMAGE DEPENDENCY SCORING MECHANISM) and integrating it into GPT-4o for synthesizing questions and answers should be more efficient. Therefore, strong experimental comparisons are needed to demonstrate that the effectiveness gained by sacrificing efficiency is significant.

(6) The complete pipeline prompt template is not provided, making it difficult to verify the method's effectiveness at the case level.

### Questions
(1) In the experiments, does the version of GPT-4o remain consistent with the model version used in the baseline? There are capability differences between various 4o versions.
(2) As VLMs become increasingly powerful, will the benefits of the proposed solution diminish?
(3) Has there been any testing on the effectiveness of open-source model synthesis?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes "Seeking the Right Question" (SRQ), a novel three-stage pipeline for high-quality visual instructions generation.
Based on this data pipeline, authors construct the Allava-SRQ dataset based on the 125K images samples from Allava. Experimental results show that the model trained with Allava-SRQ performs better than the counterpart trained with the dataset proposed in LLaVA, demonstrating the effectiveness of the proposed dataset.

### Strengths
1. This paper concentrates on the quality of questions in the dataset, which is a crucial component determining the overall quality of the dataset.

2. The proposed data construction pipeline is straightforward.

3. Experimental results show that the model trained on Allava-SRQ outperforms its counterpart trained on the LLaVA dataset, demonstrating the effectiveness of the proposed dataset.

### Weaknesses
1. The writing in this paper is lacking clarity. The authors do not adequately explain the motivation behind the design of the proposed data construction pipeline. Additionally, the description of the image dependency scoring mechanism is confusing—how is the score derived for a given image or question?

2. This paper focuses on question quality. However, aside from Figure 5, it does not provide additional quantitative analyses regarding the question quality of other datasets.

3. Since questions are generated using 160 seed instructions, the diversity of generated instructions may be limited.

4. The improvements over the baseline appear to stem from the distillation of GPT-4. The effectiveness of the proposed data pipeline remains unclear.

### Questions
Please see weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new method for generating visual instruction data with better quality, which leads to better performance of downstream VLM trained with the generated data.

### Strengths
The quality of synthetic dataset is important in fine-tuning popular visual language models nowadays. The proposed method may effectively improve the quality of generated data, leading to better performance of the resulting visual language models.

Comprehensive experiments are conducted, showing that the proposed method is indeed effective, and outperforms baseline methods.

In experiments, the authors try to make sure that all the compared methods are evaluated under the same setting, by using the same scale of dataset and using carefully chosen fine-tuning strategy, which is important for fair comparison.

### Weaknesses
Directly using the scoring for comparison as in Figure 5 may not be appropriate. The scoring criteria described in section 2.3 does not cover all the three aspects the authors mentioned, it may lean towards questions which can be easily answered from the image without reasoning. To compare the quality, we may need human evaluation or better scoring strategy.


The authors mentioned that "for each query image, we sample from the eight categories ... to generate questions across all eight categories simultaneously. ... we randomly sample three from each category as few-shot examples". While for baseline methods, the authors "set the number of few-shot examples to 8. If I understand correctly, the baseline method should also be provided more few-shot (24) samples, to make sure that the generation are compared fairly. 


A cost/computation comparison between the proposed method and baseline generation method are needed. The proposed instruction generation may need more computation cost because it may need both generation and scoring. What will the result be if we also apply scoring and filtering in vanilla instruction generation?

### Questions
If we use different or ensemble of language models as evaluator for scoring, will it lead to better results?

### Soundness
3

### Presentation
3

### Contribution
3
