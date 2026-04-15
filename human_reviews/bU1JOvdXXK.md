# Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs).
However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in anonymous github page \url{https://github.com/Benchmark-Dysca/Dysca}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This is a data paper. The authors introduce Dysca, a new benchmark for evaluating Large Vision-Language Models (LVLMs) from the perspective of assessing multi-stylized images and noisy scenarios, as well as test environments involving corruption, print attacks, and adversarial attacks. Dysca uses SDXL images to dynamically synthesize varied image styles and questions, enabling evaluation across 51 styles and 20 subtasks under four scenarios: clean, corrupted, print attacked, and adversarially attacked. They also provide interesting insights into LVLMs' capabilities and performance under diverse conditions.

### Strengths
1. The reviewer appreciates the authors' attempt to address a unique aspect of LVLM benchmarks: the evaluation of multi-stylized images and noisy scenarios. This perspective is intriguing and relevant.
   
2. The paper is well-written with clear motivations and a logical structure. The presentation is visually appealing with sophisticated diagrams. The detailed introduction of the benchmark data facilitates reader comprehension.
   
3. The experimental analysis provides several valuable insights. Notably, there is a high correlation between their evaluation rankings and those of non-synthetic benchmarks, highlighting deficiencies in current LVLMs when facing various question types, image styles, and scenarios.

4. The overall effort in the work is solid. Using an automated synthesis method to generate an evaluation benchmark is an unconventional but intriguing and worthwhile exploration.

### Weaknesses
1. The major concern with this dataset is the use of image generation tool for automatic vision creation. For such images to serve as a gold standard for MLLM evaluation, they must meet a critical assumption: the generated images are perfect and error-free. Unfortunately, even today's state-of-the-art SDXL models, which can produce high-quality images, still blend real and synthetic styles rather than producing wholly realistic images. As humans, we can typically recognize diffusion-synthesized images. SD models excel at creating artistic-style images, but for sure, non-realistic images inevitably introduce bias in evaluations on realistic images. Also, many automatically synthesized images that contain scene texts are often of low quality, i.e., containing inaccurate scene text. Such biases could largely skew model evaluations.
Therefore, the reviewer suggests that the authors add ample discussion regarding this issue. For example, the authors could include some in-depth analytical experiments to validate the concerns raised above.


2. The entire process is fully automated, including the use of CLIP for automated quality assessment of generated QA image-text pairs, without any human inspection, which is problematic.
The reviewer suggests that the authors analyze the impact of all automatically constructed data on quality, or consider using a more reasonable evaluation method for assessment.

3. While using automatically synthesized images significantly reduces annotation costs, as a testing set, it is a dubious claim without (or with less) real value. Conversely, generating a large-scale training set through automated image synthesis is undoubtedly positive.
Therefore, the reviewer hopes that the authors can further explain and clarify this matter.



4. The main focus of this data should be on testing LVLMs' performance in multi-stylized and noisy images and in scenarios involving corruption, print attacks, and adversarial attacks. If a LVLM benchmark is limited to these aspects, its utility and contribution are significantly diminished.
Therefore, the reviewer hopes that the authors can further explain and clarify this matter.


5. "_Although they claim that the questions are re-annotated, previous work (Chen et al., 2024) has demonstrated that these benchmarks have unintentionally leaked into the training data of LLMs and LVLMs_": This claim might largely be inaccurate, as Chen et al. (2024) only discussed very early LVLM benchmarks, which were simplistic and not re-annotated. However, most current benchmarks have re-annotated QA texts, almost solving the data leakage issue.
Please provide further details.

6. The authors claim a large dataset size with 617K visual-language QA pairs. However, for evaluation benchmarks, sheer size is not as important as having a more varied set of subsets assessing different aspects and capabilities [1]. Unfortunately, Dysca only covers 20 image perception subtasks, offering no significant advantage over SEED-Bench2 and BenchLMM. Moreover, it appears to be surpassed by existing work (e.g., [2]) in both data quantity and evaluation scope, which the authors have not compared or mentioned.
The reviewer suggests adding more comparisons for this.


7. As a benchmark, it evaluates too few (comparatively) LVLMs, limited to just a small subset of models (actually 16; the 26 LVLMs are somewhat overclaimed). There are many more state-of-the-art multimodal LLMs that should be included at least in the appendix. the reviewer suggests the authors include evaluations of more models next.
The reviewer strongly recommends adding more MLLMs for experimental comparison.



## Summary

Overall, if the authors had focused solely on evaluating multi-stylized images and noisy scenarios, it would have been a safer claim, even with automated image synthesis. However, by comparing it to current general-purpose LVLM benchmarks, the limitations and significant issues of automated image synthesis are exposed, particularly its inability to replace the importance of realistic images. This is my key suggestion for the authors.  
Thus, "_We demonstrate for the first time that evaluating LVLMs using large-scale synthetic data is valid_" might be an overclaim, at least not sufficiently and safely proved in this paper.

Overall, the reviewer is open-minded. If the authors can actively and effectively address the concerns with more discussions, the reviewer would consider raising my rating.

---

[1] Are We on the Right Way for Evaluating Large Vision-Language Models? 2024.

[2] MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks. 2024

-----------

After the reviewer-author interaction, the reviewer decided to increase the evaluation to positive as the authors provided enough detailed responses and clarification.

### Questions
1. Can the authors present the complete prompts used for image generation in the appendix? Could some prompts be displayed?

2. Did the authors assess the quality of automatically synthesized images? How is quality ensured when generating such a large scale of images, especially those containing text?

3. How were the QA pairs (text) constructed on such a large scale? Was ChatGPT utilized? or just all by rule-based template?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Dysca, a dynamic and scalable benchmark that uses large language models (LLMs) and text-to-image diffusion models (T2I) to generate synthetic images, questions, and answers. They filter the low-quality examples by leveraging off-the-shelf models. It is dynamic and scalable. Prior benchmark might have potential data leakage issues and scalability limitations. Evaluation experiments show their findings, such as inconsistency in question types and the vulnerability of attacks. It shows the limitations of current vision language models (VLMs).

### Strengths
Their strengths come from the idea of automatically synthesizing dataset curation. The metadata contains information to generate prompt, image, and question-answer pairs. 
1. Scaling is easy because of adopting the synthetic data.
2. They propose the findings, the importance of the language model, the inconsistent performance to different question types, the difference perceptual performance on different sub-tasks, and the degradation in attack. 
3. They demonstrate the validity of Dysca in the section 4.2.2.

### Weaknesses
Their weaknesses come from the idea of simply scaling data. Note that "simple" does not mean "easy." Papers such as [R1], [R2], and [R3] consider the specific points of cultural understanding, Humor, and hallucination. It would be useful to have specific evaluation key points like these benchmarks.
1. How does the proposed benchmark differ from unifying existing benchmarks into one benchmark? Does it evaluate the model behavior that existing benchmarks fail to cover? Combining the existing benchmarks proposed in Table 1 creates a benchmark covering a wide range of perceptual tasks and question types. Therefore, it should clearly show something that existing benchmarks have not been able to do. For example, it would be valuable to show researchers the characteristics of the samples that all VLMs get wrong.
4. What specific insights can the proposed benchmark provide that are not observable through existing benchmarks? It would be interesting to propose a new metric that is only available in the proposed benchmark. This is a weakness that can be linked to Weakness 1. Metadata is an important role in sample generation. Similar to [R4], an explainable metric using that metadata could be an example. Figure 7 may be good for fine-grained analysis, but showing 51 styles or considering other meta data at once is difficult to understand at once.
5. With the growing body of research on training with synthetic data, suppose we follow the training approach: (1) generate data using the same process with Dysca, and (2) train VLMs. Afterward, how should we handle potential data leakage due to the similar data distribution compared to Dysca? It can be useful to know the sub-populations where all VLMs underperform, even when trained on synthetic images.
7. T2I diffusion models have the limitation of creating all possible images corresponding to question-answer pairs. Could authors discuss this limitation? For example, benchmarks that consider specific vulnerabilities of VLMs [R1, R2, R3, R5, R6] seem to be hard to generate automatically. 

[R1] Do Androids Laugh at Electric Sheep? Humor “Understanding” Benchmarks from The New Yorker Caption Contest, ACL'23

[R2] Benchmarking Vision Language Models for Cultural Understanding, EMNLP'24

[R3] BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-language Models, ECCV'24

[R4] Attribute Based Interpretable Evaluation Metrics for Generative Models, ICML'24

[R5] NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples, NeurIPS'24

[R6] Vision language models are blind, arxiv'24

### Questions
1. Would authors change "" to the right double quotes?
2. Would authors fix the image captions to be below the image and the table captions to be above the table? i.e., Table 2, Figure 6.
3. In Section 4.2.2, is there any reference that a correlation is high if it is higher than 0.6?

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
This work propose a large-scale dataset, i.e., Dysca, for evaluating the perceptual capability of LVLMs. To build such a large-scale dataset, the author propose a flexible pipeline which defines the metadata and a rule-based approach for generating the text-to-image prompt, image and question-answer (QA) pairs. In summary, Dysca consists of 617K Vision-language QA pairs, covering 20 perceptual subtasks, 4 image scenarios and 3 question types. The author conduct comprehensive evaluation for 24 existing LVLMs on Dysca, demonstrating the weakness of existing LVLMs.

### Strengths
1. The proposed dataset is comprehensive and large in scale. Dysca not only considered different perceptual scenarios, it also provided a testbed for evaluating the perceptual capability of LVLMs under both corruption and printing attack situations.
2. The data construction pipeline is flexible to be extended for assembling the samples of other sub-tasks. Benifit from the powerful generation capability of T2I models, the pipeline also demonstrated the good quality of generative large-scale datasets.
3. The experiments are comprehensive and proved the claimed merits of the proposed dataset.

### Weaknesses
There are two main concerns:
1. No data leakage is one of the main claimed contribution for the dataset, but there is no comprehensive statistical comparison of Dysca with existing benchmarks. Only the blind experiment in Appendix D, which should be moved to the main content considering its importance, trying to explain this point which is not enough from my perspective.
2. The setup of adversarial attack, i.e., only PGD, is too simple. Compare to the corruption and printing attack setup, only PGD is considered for adversarial attack is not convincible, especially considering there are bunch of emerging studies on the adversarial attack of LVLMs.


There are also several unclear points,
1. There is no description of the rule-based approach adopted when getting the prompt and QA from the metadata.
2. No explanatio of the metadata itself. Why the metadata is combined with the four aspects? Is there any dataset- or task-specific relationship when considering this?
3. It is unclear that how many examples for each type of corruption and printing attack in the dataset.

### Questions
It is very common that the text and image cannot be aligned well for T2I generation. As far as i know, only the data clean part in the pipeline is related to this problem. How do you ensure the threshold and biding (top 6 models) way can work for it?

### Soundness
3

### Presentation
2

### Contribution
2
