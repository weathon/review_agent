# Distill Visual Chart Reasoning Ability from LLMs to MLLMs

- Decision: Reject
- Scores: 8, 6, 5, 5

## Abstract
Solving complex chart Q&A tasks requires advanced visual reasoning abilities in multimodal large language models (MLLMs). Recent studies highlight that these abilities consist of two main parts: recognizing key information from visual inputs and conducting reasoning over it. Thus, a promising approach to enhance MLLMs is to construct relevant training data focusing on the two aspects. However, collecting and annotating complex charts and questions is costly and time-consuming, and ensuring the quality of annotated answers remains a challenge. In this paper, we propose Code-as-Intermediary Translation (CIT), a cost-effective, efficient and easily scalable data synthesis method for distilling visual reasoning abilities from LLMs to MLLMs. The code serves as an intermediary that translates visual chart representations into textual representations, enabling LLMs to understand cross-modal information. Specifically, we employ text-based synthesizing techniques to construct chart-plotting code and produce ReachQA, a dataset containing 3k reasoning-intensive charts and 20k Q&A pairs to enhance both recognition and reasoning abilities. Experiments show that when fine-tuned with our data, models not only perform well on chart-related benchmarks, but also demonstrate improved multimodal reasoning abilities on general mathematical benchmarks such as MathVista.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes an efficient method (CIT) that aims to use code as a bridge and incorporates certain characteristics of diagram research (such as visual complexity) to generate data (REACHQA). On this basis, the paper deconstructs the capabilities required for the Chart QA task into recognition and reasoning, and demonstrates through experiments that the proposed method is effective.

### Strengths
1. CIT incorporates the characteristics of the chart tasks to generate high-quality data.
2. This paper attempts to deconstruct the capability of chart-based question answering into recognition and reasoning and provides thorough justification based on this approach.

### Weaknesses
1. The authors should consider and attempt to provide data on the overlap between the generated data, the validation set, and relevant benchmark datasets to further substantiate the validity of the experiments.
2. I hope the authors can provide more evaluation results on benchmarks that are specific to recognition [1] and reasoning [2] to demonstrate that the method has strong generalization capabilities. If the results do not decline, that would be an even more interesting observation.

[1] OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models
[2] We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?

### Questions
Please refer to the weaknesses section.

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
4

### Summary
This paper purpose CiT (Code-as-Intermediary Translation), a data synthesis method for distilling visual reaoning abilities from LLM to MLLMs. In CiT, code are rendered to charts and translated to instruction tuning by LLMs. Using this pipeline, this paper developed ReachQA, which is an effective benchmark and training dataset.

### Strengths
- The overall writing is clear and easy to follow.
- The idea of CIT is straightforward, effective and easily scalable.
- CIT addressed an important gap in MLLM training in lacking accurate textual annotations of visual diagrams.
- The resulting dataset has high quality. Althought only with 3K images and 20K QAs, the perforamnce of MLLMs improves by at most 35 percent. In addition, when trained a mixture with general multimodal dataset, the model can effectively retain its general MM benchmarks.

### Weaknesses
- As authors claimed, the MLLMss ability consist of two main parts, 1. recognizing key information from visual inputs 2. conducting reasoning over it. The design of ReachQA is rich in both parts, so it's unclear which part improves models' over performance the most, the reviewer is aware that similar analysis is conducted in section 5, but an error analysis (similar with figure 1) on before/after ReachQA training can make it more clear. 
- Dataset volume is a concern. As a training dataset, only 3K images is not sufficient, especially for cross-modal training. So it's likely that the improvement gained by training on ReachQA is that the model learns reasoning better rather than recogonition.

### Questions
Please refer to above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper aims to distill the visual chart reasoning ability from LLMs to MLLMs through a code-as-intermediary translation (CIT) method. The motivation behind the work lines the concern of current MLLMs and then distill the LLMs knowledge about reasoning to MLLMs is a way. The method separates two steps, recognition and reasoning, then the authors try to generate more codes for chart generation, and then generate instruction-answer pairs. Besides, the authors also propose a ReachQA dataset for training and evaluation. Through multiple experiments, the authors demonstrate the effectiveness of the method, the dataset with strong performances.

### Strengths
1. The inspiration of the method is interesting and leverages the translation principle to incorporate an intermediate language. By using code as an intermediate language, the method is interesting and novel. 
2. The steps behind the method are clear and reasonable, first use code to generate charts, and then generate QA pairs. The authors use multiple methods, such as Evol-instruct, self-instruct, llm-as-judge and so on to make the generated data to be high quality and diversity.
3. The experiments are extensive to show the effectiveness of the method and the study is also good.

### Weaknesses
1. The main concern is about the motivation or the story. While the authors mention that the existing MLLMs are struggle in the recognition and reasoning abilities, the authors then want to use distill LLMs' reasoning ability to MLLMs. However, it is clear that Claude and GPT-4o are smart in these ways. Therefore, the problem is not about existing MLLMs but the open-sourced, or freely sourced MLLMs. I encourage the authors to rephrase the story, which then could introduce the method more clearly. 
2. Though the steps are great and the authors use multiple existing methods to generate and guarantee the data quality. The novelty of the method/framework is clearly not that big, but I like the code-as-intermediate translation method. 
3. In detail, the analysis of the question types should be reported, which could show the diversity and quality; from the experiments, strong improvements are achieved but still largely fail behind the close models; the self-repair method is good but the prompt is too simple, I am curious about other clear prompts; as the authors show figure 1 as a motivation, it is still necessary to better analyse the two type errors in the final trained model, not like figure 3, but like figure 1 to show the improvement.

### Questions
See above

### Soundness
3

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
4

### Summary
This paper proposes a novel pipeline for constructing multi-modality chart datasets. The central idea is the introduction of the “Code-as-Intermediary Translation” concept, which facilitates the creation of chart datasets for MLLMs (Multimodal Large Language Models). Experimental results demonstrate that the proposed datasets not only enhance chart-related QA capabilities but also improve overall reasoning performance.

### Strengths
Utilizing code as a generative engine is a sound approach to creating new chart images.

Experimental results indicate that the new dataset significantly enhances specific performance metrics.

### Weaknesses
The use of code as a tool for rendering the chart dataset appears similar to the concept presented in [1]. Furthermore, data augmentation through code is reminiscent of [2]. Could you elaborate on how your approach differentiates from these works?

The model fine-tunes general MLLM models. How would the performance change if an equivalent amount of chart data were selected from general datasets for fine-tuning the MLLMs? A comparative analysis might be insightful.

Why is the chart type described as ∞? Are there any specific chart types that cannot be generated using Python code? 

What is the accuracy or success rate of MLLMs when generating code?

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
