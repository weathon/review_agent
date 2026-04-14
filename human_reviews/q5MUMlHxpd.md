# VOILA: Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 6, 6

## Abstract
Multimodal Large Language Models (MLLMs) have become a powerful tool for integrating visual and textual information. Despite their exceptional performance on visual understanding benchmarks, measuring their ability to reason abstractly across multiple images remains a significant challenge. To address this, we introduce VOILA, a large-scale, open-ended, dynamic benchmark designed to evaluate MLLMs' perceptual understanding and abstract relational reasoning. VOILA employs an analogical mapping approach in the visual domain, requiring models to generate an image that completes an analogy between two given image pairs, reference and application, without relying on predefined choices. Our experiments demonstrate that the analogical reasoning tasks in VOILA present a challenge to MLLMs. Through multi-step analysis, we reveal that current MLLMs struggle to comprehend inter-image relationships and exhibit limited capabilities in high-level relational reasoning. Notably, we observe that performance improves when following a multi-step strategy of least-to-most prompting. Comprehensive evaluations on open-source models and GPT-4o show that on text-based answers, the best accuracy for challenging scenarios is 13% (LLaMa 3.2) and even for simpler tasks is only 29% (GPT-4o), while human performance is significantly higher at 70% across both difficulty levels.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper introduces VOILA, a benchmark designed to assess multimodal large language models (MLLMs) on tasks requiring visual perception and abstract relational reasoning. VOILA primarily focuses on visual analogical reasoning, where models must generate an image that completes an analogy given two pairs of images. The benchmark includes two subsets, VOILA-WD (with distractions) and VOILA-ND (without distractions), testing MLLMs on varying levels of complexity. Contributions:
1. A dataset creation pipeline allowing large-scale generation of visual analogy questions, and a large-scale, open-ended benchmark to evaluate MLLMs’ high-level visual reasoning capabilities. 
2. Comprehensive evaluation of state-of-the-art MLLMs, highlighting performance gaps in relational reasoning compared to human baselines.
3. Ablation studies reveal model limitations in handling complex visual relationships, suggesting areas for improvement in future MLLM development.

### Strengths
1. The proposed problem is a novel multimodal reasoning problem, which is well-defined and advances abstract relational reasoning in MLLMs​.
2. The benchmark design is comprehensive, encompassing various rule configurations and levels of task difficulty.
3. The results reveal the weaknesses of current multimodal LLMs.

### Weaknesses
1. The paper does not address why the proposed problem is an essential and meaningful problem to solve. In what applications/scenarios is the capability essential?
2. The comparison between the model and human performance may not be fair: (1) Humans are given two examples. (2) It seems the task for humans is to choose properties from available options, while MLLMs need to predict those properties.
3. Table 3 takes up a lot of space and has a lot of numbers, but it is barely mentioned or explained in the text.

### Questions
1. The prompt4 for generating images seems unnecessary for evaluating the "reasoning" capabilities of MLLMs. Also, the reviewer is curious how the authors use GPT4o to generate images since it is not clear whether they call another image generation model to generate images in the web interface. And it seems gpt4o API does not support image output. 
2. Why Table 4 and Table 5 only use the selected models for the ablation study?

### Soundness
2

### Presentation
2

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
This paper introduces a new benchmark for evaluating multi-modal large language models’ perceptual understanding and abstract relational reasoning through analogy completion. It contributes a dataset creation pipeline, a large evaluation dataset consisting of 10K+ questions, and extensive evaluations of both open-source and proprietary models on this benchmark as well as detailed analyses of their performance on different subtasks.

### Strengths
- The idea of using visual analogy completion for evaluating multi-modal large language models is well motivated and somewhat unique 
- The visual analogy completion problem is well defined and formulated, with clear definitions of the three properties and four rules.  
- The dataset has been manually cleaned to ensure that the generated images match the texts.

### Weaknesses
- The experiments can be more complete. Currently, most evaluations focus on the first three steps of the analogy completion task with only a few data points on the image generation stage. 
- The paper presentation can be improved. For example, the writing can be improved to highlight the most insightful and interesting takeaways. 
- While the visual analogy task is interesting, the reviewer is uncertain about the practical value of this benchmark especially given the abundance of multi-modal benchmarks – for example, why should the community use this benchmark to evaluate models’ relation understanding instead of existing VQA benchmarks that can also test for relation understanding?

### Questions
- Did the authors include examples of each subtask in the prompts? 
- The authors mention they evaluated Emu-2 on the image generation subtask in 5.1, but Table 3 is missing its numbers? 
- In Figure 6, it’s interesting that LlaMa 3.2 is the only model with higher performance on Voila-WD than Voila-ND, while all other models exhibit large drops from Voila-ND to Voila-WD regardless of their absolute accuracies. Can the authors provide insights into why this occurs for LlaMa 3.2?
- Nit: table 4, it’s VOILA-ND in both columns – should one of them be VOILA-WD instead?

Suggestions: 
- While the authors motivate the visual analogy task by saying that it tests for higher-order reasoning, they break down the task into four subtasks, each of which is an easier task (despite being still difficult to the model). While the reviewer agrees with the authors that this task decomposition enables fine-grained analyses of models’ weaknesses, it’d also be interesting to see an evaluation of the end-to-end analogy completion task. 
- Table 2 can highlight the performance drops from the previous to the next stage
- The authors mentioned that models struggle with the visual analogy task when images are combined in a collage format due to resolution constraints. It’d be interesting to see if this finding holds with high resolution image collages for multi-modal models that support the AnyRes strategy such as Llava-OneVision.

### Soundness
3

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
4

### Summary
This paper proposed a new benchmark named VOILA, aiming to evaluate the performance of visual language models in understanding abstract relation understanding by designing visual analogy questions.
Experiments show that many models can describe the images and idenitify the relationship between images, but cannot apply the relationship very reliably.

### Strengths
1. The visual analogy perspective of evaluating vision language models is interesting.
2. The paper gives a very detailed the description of how the benchmark is collected.

### Weaknesses
1. The analogies considered in this paper seems to be restricted to only numbers, subjects, and actions.

### Questions
1. One line of related works I think this paper need to mention is on the benchmarking of multiple image understanding, such as works of Muirbench and MIRB, especially MIRB, which has already included visual analogy as one sub task for testing current state-of-the-art visual language models.
2. The data generation pipeline uses diffusion models to geerate images from given text prompts. I'm wondering how reliable is the diffusion model? how many images are discarded. Perhaps include a performance of human on the final dataset can answer this (I noticed that in table 3 there are some human performance, but why is human performance only tested on the 'Applying relationship' subset?)
3. In my understanding, this paper only considers analogy on properties like number, subject, and action. Is these enough to cover all possible visual combinations that can form analogies?



Muirbench: https://arxiv.org/abs/2406.09411
MIRB: https://arxiv.org/abs/2406.12742

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose a new benchmark called VOILA which evaluates the perceptual understanding and analogical reasoning capabilities of multimodal language models, particularly its reasoning capabilities across multiple images. VOILA requires a MLLM to understand relations between images and generate a new image that follows the pattern. This task is open-ended and results are evaluated using a strong LLM. Results indicate that even the best MLLM performs much worse than an average human (30% vs 70%)

### Strengths
This is generally a well organized and well written paper.
The proposed benchmark is interesting and highlights a critical shortcoming in existing MLLMs
The benchmark is well curated, there is also manual filtering involved to ensure quality

### Weaknesses
This task is not exactly novel, [1] propose a new task that evaluates visual cognition of MLLMs. The new novelty comes from the requirement that MLLM generate the output image. 
The evaluation is not grounded in existing psychological assessments for instance the reasoning used in [1] is widely used in neurodevelopmental and neuropsychological research.
GPT4o seems to be poor at identifying relationships (from image) in VOILA, does this affect the evaluation which in turn also uses GPT4o. 

References
[1] https://arxiv.org/abs/2406.10424

### Questions
Can the authors conduct a small experiment to validate that their evaluation strategy is accurate?
The prompt to identify relation asks the model to identify differences in subject, types and actions all in a single prompt. What if this problem was broken down to three sub-prompts? It seems intuitive that performance should improve

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces VOILA, a benchmark specifically designed to evaluate Multimodal Large Language Models (MLLMs) in perceptual understanding and abstract relational reasoning across images. By using analogical mapping, VOILA requires MLLMs to generate an image that completes an analogy between two image pairs, testing the models' relational reasoning without predefined answer choices. The benchmark includes challenging tasks, with models struggling to match human performance, especially in higher-level reasoning steps. Performance improves with least-to-most prompting strategies, but there remains a substantial gap between the best-performing model and human results.

### Strengths
• VOILA stands out by focusing on abstract visual analogies, making it a valuable addition to existing MLLM evaluation benchmarks, particularly in assessing perceptual and relational reasoning.

• The multi-step reasoning approach and comprehensive ablation studies offer a detailed examination of the current limitations in MLLMs.

### Weaknesses
• The paper emphasizes that this is a dynamic benchmark. For a static benchmark, once a configuration is provided, it can be fully generated and then annotated for correctness by human evaluators. However, as a dynamic benchmark, once it is generated, how to ensure the correctness of the benchmark and how to make it scalable, which remain challenging and difficult to guarantee. This issue is unsolved in this paper.

• VOILA’s reliance on GPT-4o for model evaluation makes the evaluation process both resource-intensive and costly for practitioners.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
