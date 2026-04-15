# ACE: All-round Creator and Editor Following Instructions via Diffusion Transformer

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8, 6

## Abstract
Diffusion models have emerged as a powerful generative technology and have been found to be applicable in various scenarios. Most existing foundational diffusion models are primarily designed for text-guided visual generation and do not support multi-modal conditions, which are essential for many visual editing tasks. This limitation prevents these foundational diffusion models from serving as a unified model in the field of visual generation, like GPT-4 in the natural language processing field. In this work, we propose ACE, an All-round Creator and Editor, which achieves comparable performance compared to those expert models in a wide range of visual generation tasks. To achieve this goal, we first introduce a unified condition format termed Long-context Condition Unit (LCU), and propose a novel Transformer-based diffusion model that uses LCU as input, aiming for joint training across various generation and editing tasks. Furthermore, we propose an efficient data collection approach to address the issue of the absence of available training data. It involves acquiring pairwise images with synthesis-based or clustering-based pipelines and supplying these pairs with accurate textual instructions by leveraging a fine-tuned multi-modal large language model. To comprehensively evaluate the performance of our model, we establish a benchmark of manually annotated pairs data across a variety of visual generation tasks. The extensive experimental results demonstrate the superiority of our model in visual generation fields. Thanks to the all-in-one capabilities of our model, we can easily build a multi-modal chat system that responds to any interactive request for image creation using a single model to serve as the backend, avoiding the cumbersome pipeline typically employed in visual agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ACE (All-round Creator and Editor), a unified foundational model capable of handling a diverse array of visual generation tasks. By incorporating Long Contextual Units (LCU) and an efficient multimodal data collection methodology, ACE demonstrates exceptional performance in multi-task joint training, encompassing a wide range of tasks from text-guided generation to iterative image editing. Experimental results indicate that ACE significantly outperforms existing methods across multiple benchmark tests, showcasing its robust potential for practical applications.

### Strengths
ACE introduces LCU, a novel approach that unifies various modal conditions, enabling the model to handle complex multimodal tasks. LCU allows ACE to flexibly adapt to different tasks, including generation and editing, which is lacking in current models. By integrating historical information into LCU, ACE can handle multi-turn editing tasks, enhancing its practicality in continuous interaction scenarios. ACE covers eight basic generation tasks and supports multi-turn and long-context tasks, establishing a comprehensive evaluation benchmark, significantly outperforming existing methods, especially in image editing tasks. User studies show that ACE is more in line with human perception. This paper not only makes significant contributions and proposes a practical and innovative solution but also excels in writing and figure drawing, with clear diagrams and rigorous logic, providing an excellent reading experience for the audience.

### Weaknesses
Model Efficiency and Scalability:
The paper should include a more detailed discussion on the computational efficiency and scalability of the model:
It is important to evaluate the model's performance when processing large-scale data to understand its practical applicability.

In-depth Analysis of Specific Tasks:
The paper should provide a thorough performance analysis for specific tasks.
This analysis should include comparisons with models that are specifically designed for those tasks, such as image editing models and inpainting models.

Data Annotation Quality:
While MLLM-assisted annotation improves efficiency, the quality of automatic annotations may not always be on par with manual annotations.
A quantitative analysis of the data annotation quality would enhance the credibility of the paper.

### Questions
Discussion on Model Efficiency and Scalability: Could you provide more details on the model's performance across different scales of data? This would help in understanding its computational efficiency and scalability.

In-depth Analysis of Specific Tasks: For key tasks, could you offer a detailed comparison with state-of-the-art models specifically designed for those tasks? This would provide a clearer picture of the model's relative performance.

Enhancing Model Interpretability: Could you explore the decision-making process of the model and provide an interpretability analysis of the generated results? This would help in understanding how the model arrives at its outputs.

### Soundness
3

### Presentation
4

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
This work proposes a unified visual generation and editing framework that supports a wide range of predefined tasks. To train and evaluate the proposed ACE, this work also introduces a data curation pipeline and an overall benchmark. Experimental results and numerous use cases demonstrate the superiority of the proposed method.

### Strengths
The method provides a unified visual generation and editing framework that supports a wide range of predefined tasks.
 The benchmark is comprehensive, designed to evaluate visual generation and editing models effectively.

### Weaknesses
The paper lacks some ablation studies to help readers understand the authors' design choices. Additionally, the results in Table 2 may not be entirely fair, as the superiority of ACE might be attributed to the scale of the data.

### Questions
1. What would happen if the Text Encoder T5 were replaced with an LLM? Would it be able to understand more diverse instructions?
2. Will the collected data be made public?

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
1. Propose ACE, a unified foundational model framework that supports a wide range of visualgeneration tasks, achieve a best task coverage.
2. Define the CU for unifying multi-modal inputs across different tasks and incorporate long context CU.
3. Design specific data construction pipelines for various tasks to enhance the quality and eff-ciency of data collection.
4. Establish a more comprehensive evaluation benchmark compared to previous ones, cover-ing the most known visual generation tasks. 

It's a lot of work, from method to data to data construction pipelines to benchmark, very systematic and complete work.
And all-in-one models are really interesting and is consistent with the general trend of generate model development.
But,
drawings are terrible, and the method is a little weak. Maybe you could use the all-in-one methods in low-level works as reference.

### Strengths
1. Propose ACE, a unified foundational model framework that supports a wide range of visualgeneration tasks, achieve a best task coverage.
2. Define the CU for unifying multi-modal inputs across different tasks and incorporate long context CU.
3. Design specific data construction pipelines for various tasks to enhance the quality and eff-ciency of data collection.
4. Establish a more comprehensive evaluation benchmark compared to previous ones, cover-ing the most known visual generation tasks
5.Analyze and categorize these conditions from textual and visual modalities respectively, includeTextual modality and Visual modality.

### Weaknesses
1. The drawings are terrible!!! In particular, Figure 3. Incongruous text proportions and strange colour scheme...It's in the lower-middle range of T2I work. 
2. The method is a little weak. All-in-one methods have been far dicussed in the field of low-level and it's ripe for the picking. Compared with them, the ACE module is not that impressive.

### Questions
Please DRAW better.
I don't find the computing resource? I think it would be big, maybe you could have a discuss.
Related work? i think there should be other works that make try building an all-in-one visual generation model. Maybe you could list them clearly, I'm not an expert on this.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes an All-round Creator and Editor as a unified foundation model for visual generation tasks. The main technical contribution lies in introducing a Long-context Condition Unit that standardizes diverse input formats. Built upon diffusion transformers, the architecture incorporates condition tokenizing, image indicator embedding, and long-context attention blocks to achieve unified visual generation capabilities. To address the scarcity of training data, the authors develop a data collection pipeline that combines synthesis/clustering-based approaches. Additionally, they establish a comprehensive benchmark for evaluating model performance across various visual generation tasks.

### Strengths
1) The framework unifies multiple image generation and editing tasks through a single model, avoiding the hassle of calling separate specialized models. The proposed LCU provides a structured approach to incorporating historical context in visual generation.

2) The paper presents systematic methodologies for data collection and instruction construction, which contributes to the development of all-in-one visual generative foundation models.

3) The evaluation benchmark provides comprehensive coverage across diverse image manipulation and generation tasks, enabling thorough performance assessment.

### Weaknesses
Technical Issues:
1. Formatting inconsistencies: in lines 417-418, the image placement obscures instruction text.

2. The authors are encouraged to provide discussions on task-specific performance trade-offs during training, specifically how optimizing for one task might affect the performance of others.

3. It would be helpful to provide methodological details regarding parameters in data preparation (lines 321-325), such as cluster number determination and data cleaning criteria.

4. The qualitative results in Figure 5 reveal some limitations. 1) Row 1 (left): ACE generates a distorted hand. 2)  Row 2 (right) and Row 4 (left): The model exhibits undesired attribute modifications not specified in the instructions, including unintended gesture alterations / head rotation changes, and camera perspective shifts.

### Questions
1. Regarding Figure 6, the authors are encouraged to elaborate on the empirical or theoretical basis for the chosen data distribution and its specific advantages for the ACE model.

2. The paper would benefit from addressing the practical challenges of model updates. Specifically, how might one efficiently incorporate new functionalities without complete model retraining? This consideration is crucial for the model's practical deployment and ongoing development.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a method to train a unified model for 8 different tasks: Text-guided Generation, Low-level Visual Analysis, Controllable Generation, Semantic Editing, Element Editing, Repainting, Layer Editing and Reference Generation. The idea is intuitive. The main contribution of the paper is the framework for generating paired training data. The source of the data generation comes from two aspects: 1. synthetic generation and 2. from publicly available datasets (LAION-5B). To verify the results of this task, authors also create a new benchmark called ACE Benchmark.

### Strengths
1. The dataset generated in this paper is beneficial to the community. This will help other researchers follow this series of research works.
2. A unified model for all tasks is also more efficient compared to have several individual models specific to certain type of tasks.

### Weaknesses
1. It seems the author does not have clear discussions on how those tasks affect each other. Are they beneficial to each other? Or some of the tasks are reducing the performance of other tasks? How to select the most reasonable tasks that should be unified with the single model? I believe adding this type of discussion with corresponding experiments will make the paper more solid.

### Questions
Indeed, as I mentioned in the weakness, how those tasks affect each other?

### Soundness
3

### Presentation
2

### Contribution
3
