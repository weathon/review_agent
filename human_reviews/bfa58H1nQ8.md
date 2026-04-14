# Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want

- Decision: Accept (Poster)
- Scores: 6, 6, 5

## Abstract
In this paper, we present the Draw-and-Understand framework, exploring how to integrate visual prompting understanding capabilities into Multimodal Large Language Models (MLLMs). Visual prompts allow users to interact through multi-modal instructions, enhancing the models' interactivity and fine-grained image comprehension. In this framework, we propose a general architecture adaptable to different pre-trained MLLMs, enabling it to recognize various types of visual prompts (such as points, bounding boxes, and free-form shapes) alongside language understanding. Additionally, we introduce MDVP-Instruct-Data, a multi-domain dataset featuring 1.2 million image-visual prompt-text triplets, including natural images, document images, scene text images, mobile/web screenshots, and remote sensing images. Building on this dataset, we introduce MDVP-Bench, a challenging benchmark designed to evaluate a model's ability to understand visual prompting instructions. The experimental results demonstrate that our framework can be easily and effectively applied to various MLLMs, such as SPHINX-X and LLaVA. After training with MDVP-Instruct-Data and image-level instruction datasets, our models exhibit impressive multimodal interaction capabilities and pixel-level understanding, while maintaining their image-level visual perception performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The manuscript aims at improving the multi-modal LLMs’ capability of understanding user input and generating corresponding responses. 
For this new task, the authors have curated a training dataset and evaluation benchmark based on open-source datasets. 
For the model to be capable of handling the visual prompt input from the users, an architecture of VP-MLLM is proposed, which mainly includes an additional visual prompt encoder based on existing MLLM architecture.
The curated training data and the model architecture allows MLLM to generate responses based on point and box inputs. 
The evaluation of VP-MLLM shows notable improvements on the capability of classifying and describing image contents based on specified image regions, compared to existing MLLMs. 
Ablation studies demonstrate the effectiveness of the proposed visual prompt encoder and the training strategy.

### Strengths
- The authors have curated a training dataset as well as its corresponding evaluation benchmark for the visual prompt understanding task, which would be valuable to future research on the task. 
- A new architecture for performing the visual prompt understanding task is proposed, which is shown to be effective in the experiments. 
- Compared to existing approaches such as Ferret, the proposed VP-MLLM seems to be better at responding to the questions based on user inputs.

### Weaknesses
- It seems that the proposed approach have negative impacts on the general ability of MLLMs, as shown in Table 2. 
- Lack of comparison to important baselines. It seems that the VP-LLaVA-8B introduces limited improvement over ViP-LLaVA-Base-7B (in Table 7). But the comparison is only drawn on the VCR dataset. I would be interested to see more detailed comparisons between these two approaches.

### Questions
- How does different design choices affect the general capabilities of the MLLM in Table 4? 
- According to Sec 3, visual prompt encoder only supports points and bounding boxes. How does it accept free-form inputs as in Table 1?
- Is there any relations between the <region 1> token and the first token given to the visual prompt encoder?

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
3

### Summary
The paper proposes a framework for enhancing Multimodal Large Language Models (MLLMs) by integrating visual prompting capabilities. The framework allows MLLMs to interact through visual and language-based prompts, such as points, bounding boxes, and free-form shapes, enabling users to guide the model’s focus to specific image regions. Key components include a Visual Prompting MLLM (VP-MLLM) architecture, which combines an image encoder, a visual prompt encoder, and a language model. It is supported by a large multi-domain dataset, MDVP-Instruct-Data, and evaluated on a benchmark, MDVP-Bench, designed to test visual prompt comprehension. The framework shows superior performance in multimodal interaction and fine-grained visual understanding tasks.

### Strengths
The integration of visual prompting with MLLMs addresses limitations in user interactivity and allows nuanced image region referencing, which enhances model usability. By enabling point-based, box-based, and free-form prompts, the framework gives users greater flexibility in interacting with models, thus expanding the application scope in real-world tasks. I like the idea of interaction between human and models.

The MDVP-Instruct-Data and MDVP-Bench provide a rich variety of images and tasks, which improve the model’s versatility across different domains and promote robust model evaluation.

The VP-MLLM framework is adaptable to existing MLLMs, enabling straightforward integration of visual prompts across different models with minimal disruption to their original capabilities

### Weaknesses
The paper provides limited information on key implementation specifics, particularly regarding model parameter settings and data preprocessing steps. This lack of detail may impact reproducibility, making it challenging for other researchers to achieve consistent results in similar settings.

The framework's reliance on pre-trained models may limit its performance based on the initial capabilities of the base MLLMs, potentially constraining generalization across drastically different datasets or new models.

The two-stage training process and reliance on large datasets are resource-intensive, requiring considerable computational power for alignment and fine-tuning, which may not be accessible to all practitioners.

### Questions
How easily can the Draw-and-Understand framework be adapted to other MLLMs? Are there specific architectural features required in the MLLM for seamless integration?

The paper mentions that box prompts sometimes underperform compared to point prompts. Could you explain why this discrepancy occurs?

Could you provide information about the hardware used for your experiments? Knowing the specifications would help in understanding the computational requirements for reproducing your results.

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
4

### Summary
The paper introduces a framework that enhances multimodal large language models (MLLMs) with robust visual prompting capabilities. It proposes a novel architecture, VP-MLLM, that integrates a vision encoder, a visual prompt encoder, and an LLM, allowing models to interpret diverse visual prompts like points, bounding boxes, and free-form shapes. Additionally, the authors present the MDVP-Instruct-Data, a large-scale, multi-domain dataset designed to support visual prompting and image-level perception. The experimental results, tested on MDVP-Bench, demonstrate that VP-MLLMs outperform existing methods in pixel-level understanding and multimodal interaction, enhancing MLLMs' capacity for detailed image analysis and spatial reasoning.

### Strengths
1.  The paper introduces an innovative visual prompt encoder and adapts existing MLLMs for enhanced visual prompting, enabling user-friendly interactions that incorporate spatial and region-specific cues.
2. The experimental results are comprehensive, covering a wide range of tasks and benchmarks. The performance on MDVP-Bench demonstrates the effectiveness of the proposed approach in terms of pixel-level understanding and multimodal interaction.
3. The paper is generally well-organized, with a logical flow from the problem statement to the methodology and results. The figures and tables support understanding.

### Weaknesses
1. The two-stage training strategy is presented as a key component, but details on the alignment stage (stage 1) are sparse. Additional clarification regarding the pre-training tasks and specific data used in this phase would enhance reproducibility.
2. While the paper presents ablation studies, there is limited insight into the specific impact of the visual prompt encoder's internal mechanisms. A more granular ablation of this component would clarify its effectiveness relative to simpler alternatives.
3. The model’s reliance on the quality of visual prompt data, particularly the GPT-4V-constructed dataset, raises concerns about its robustness across datasets with varying annotation quality. An analysis of performance variation based on prompt quality would be beneficial.
4. Although the framework is promising, the potential computational cost associated with handling multiple visual prompts and spatial references is not discussed. Assessing the model's performance in terms of processing time and scalability would address practical implementation concerns.

### Questions
1. Could the authors provide more information on the data used for alignment pre-training in stage 1? Specific details on the dataset types and the nature of the pre-training tasks would enhance reproducibility.
2. How does VP-MLLM's performance vary with lower-quality or inconsistent visual prompt data? A discussion on the impact of prompt quality on model accuracy would clarify the robustness of the proposed framework.

### Soundness
3

### Presentation
3

### Contribution
3
