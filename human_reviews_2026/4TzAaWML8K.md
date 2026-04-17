# Towards Enhanced Image Generation via Multi-Modal Chain of Thought in Unified Generative Models

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Unified generative models have shown remarkable performance in both text and image generation. When faced with image synthesis tasks, they adopt straightforward text-to-image (T2I) generation. However, we find that direct T2I generation limits unified generative models in handling complex compositional instructions. Such instructions frequently occur in realistic application scenarios. Although this is a vital issue, existing works predominantly focus on improving the basic image generation capability of unified generative models. While improvements in basic image generation can contribute to complex image generation to some extent, they still fail to adequately resolve the problem. Inspired by Chain of Thought (CoT) solving complex problems in a step-by-step manner, this work aims to introduce CoT into unified generative models to address the challenges of complex image generation that direct T2I generation cannot effectively solve, thereby endowing models with enhanced image generation ability. To achieve this, we first introduce Functionality-oriented eXperts (FoXperts), an expert-parallel architecture in our model FoX, which assigns experts based on function. In this way, FoXperts disentangles the potential conflicts in current mainstream modality-oriented designs and provide a sound foundation for CoT. When introducing CoT, the first question is how to design a CoT approach specifically for complex image generation. To this end, we emulate a human-like artistic workflow---planning, acting, reflection, and correction---and propose the Multimodal Chain of Thought (MCoT) approach, since the data here involves multiple modalities (text and image). In response to the subsequent challenge---how to design an effective MCoT training paradigm---we develop a multi-task joint training paradigm that equips the model with all capabilities required for each MCoT step in a disentangled manner. This paradigm overcomes the difficulty and impracticality of collecting consistent multi-step data tuples for training. Extensive experiments demonstrate that FoX consistently outperforms existing unified models on various T2I benchmarks, delivering notable quantitative improvements in complex image generation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses a critical limitation in contemporary unified generative models: their struggle with ​complex compositional image generation​ (e.g., multi-object scenes, spatial relationships). The authors identify that direct text-to-image (T2I) generation is insufficient for these challenges and propose a novel reasoning-based paradigm inspired by Chain-of-Thought (CoT).

The authors introduce ​FoXperts, an expert-parallel architecture that assigns experts based on function:1. A unified ​Linguistic Expert​ for text. 2. A dedicated ​Semantic Vision Expert​ for visual understanding tasks. 3. A dedicated ​Generative Vision Expert​ for visual generation tasks. Also, they propose ​MCoT, a four-step (Planning, Acting, Reflection, Correction) reasoning framework that emulates a human artistic workflow. The proposed model, ​FoX​ (1.3B parameters), demonstrates highly competitive performance across diverse benchmarks.

### Strengths
1. The idea of applying a Chain-of-Thought reasoning process to complex image generation is inspiring. The functionality-oriented expert architecture presents a fresh alternative to mainstream modality-oriented designs, effectively addressing a fundamental conflict in multimodal modeling.

2. The experimental evaluation is thorough. The paper validates its approach across a wide range of well-established benchmarks for both image generation and understanding, demonstrating the high performance of their model.

3. By moving from one-shot generation to a reasoned, multi-step process, it enhances the reliability and controllability of models for complex tasks. The proposed training paradigm is particularly significant as it provides a practical solution to a major data availability challenge.

### Weaknesses
1. What was the original intention behind using the VAE encoder for the "Image for Understand" component? Why not using a model with richer semantic features, like CLIP?

2. The overall training and inference process is multi-staged. What is the rationale for integrating and training these stages within a single model? For the same workflow, what are the advantages compared to using two expert models (e.g., a VLM for understanding and a generative model for creation)? For instance, can the understanding and generation tasks mutually enhance each other?

3. Although the total parameter of the model  is 1.3B, the entire image generation workflow is relatively long and time-consuming. How is the trade-off between performance and efficiency balanced?

4. Can the "reflection and correction" process be performed multiple times? If so, how does it affect the model's performance? Does the long-context issue impact this process? Could the authors provide some illustrative examples?

5. If an error occurs during the reflection stage, it will inevitably lead to mistakes in the subsequent correction. This may even result in a corrected image that is poorer than the initially generated one? How to prevent this issue as much as possible?

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the task of complex image generation by firstly introducing a unified generative model named FoX, which consists of FoXperts that disentangle experts by functions of generation and understanding. It also proposes an MCoT method to address image generation as a multi-step process of planning, acting, reflection, and correction, with a multi-task joint training paradigm to train the model without consistent multi-step data. Experiments are conducted on image generation and understanding benchmarks.

### Strengths
1. The model shows satisfying performance with a small number of parameters.
2. The generated images seem to effectively solve the complex image generation task, judging from the qualitative results.
3. The overall writing is clear and easy to follow.

### Weaknesses
1. The proposed method shows limited novelty compared with previous work. Assigning experts by functionality, the core contribution of FoX, is already introduced by BAGEL[1]. Using CoT in image generation is also present in works like T2I-R1[2], GoT-R1[3], Uni-CoT[4] but are not discussed in this paper. The main difference between them and the proposed MCoT lies in the layout planning, but this is somewhat confined to the compositional image generation task (e.g., T2I-CompBench) and cannot be extended to more general scenarios.
2. Benchmarks used in multimodal understanding are not sufficient. The experiments only consider MME-P, MMBench, and VQAv2. Commonly used ones like MMMU, MM-Vet, TextVQA, and InfoVQA are missing.
3. The model scale is still small (1.3B). It is unclear whether the proposed method can be scaled to larger models.


[1] Emerging Properties in Unified Multimodal Pretraining

[2] T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT

[3] GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning

[4] Uni-CoT: Towards Unified Chain-of-Thought ReasoningAcross Text and Vision

### Questions
1. Baselines in Table 2 are too old. What about recent models?
2. Can MCoT be extended to more general reasoning-related image generation benchmarks like WISE[5], T2I-ReasonBench[6] and PhyBench[7]?
3. Line 879: why is Semantic Visual Expert initialized from Generative Visual Expert? Is there an ablation on initializing from scratch or from Linguistic Expert (like in [8])?
4. Some writing issues:
- Line 170-172: I suppose these papers released in 2024 are not concurrent.
- Line 874: Qwne -> Qwen
- Table 9: No textual explanation. Better explain that the results are on T2I-CompBench.

[5] WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation

[6] T2I-ReasonBench: Benchmarking Reasoning-Informed Text-to-Image Generation

[7] PhyBench: A Physical Commonsense Benchmark for Evaluating Text-to-Image Models

[8] Mono-internvl: Pushing the boundaries of monolithic multimodal large language models with endogenous visual pre-training

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces FoXperts, a unified generative model with a Functionality-oriented eXperts architecture that mitigates function-domain conflicts in modality-oriented designs, while seamlessly integrating both understanding and generation across textual and visual modalities.

### Strengths
* The method is simple and easy to understand.
* MCoT splits complex drawing into four quick passes—plan, execute, reflect, refine—each targeting one goal so error drops round-by-round.
* FoXperts assigns “seeing” and “painting” to two separate vision experts, plus a language expert, eliminating internal conflict and giving later iterations a solid base.

### Weaknesses
* The paper mainly combines an expert architecture with a unified understanding–generation model. Technically, it divides the visual expert into two parts—semantic understanding and generation—which is rather straightforward. Essentially, it does not differ significantly from common mixture-of-experts models, so the innovation seems to lie more in integration than in architectural novelty.
* The proposed MCoT adopts a four-step process, which appears to be a standard approach in Chain-of-Thought (CoT) methods, without specific adaptations for image generation tasks.
* The idea of a planning → execution → reflection → revision process has already been reflected in existing CoT-based visual-language models (e.g., CoT-VLA), yet the paper lacks comparisons or discussions regarding these related works.
* Although the paper proposes a four-step process, it does not explain why such a complex procedure is necessary, as opposed to simpler two-stage (e.g., planning + generation) or three-stage designs.
* The paper claims that “a single visual expert leads to functional conflicts,” but provides neither experimental evidence nor theoretical analysis to demonstrate the existence or impact of such conflicts on performance.
* The paper argues that “direct T2I generation cannot handle complex compositional instructions,” yet current models like SD3 and DALL·E 3 already show strong performance in complex scene generation. The authors do not sufficiently justify why introducing CoT is essential rather than further optimizing existing generative models.

### Questions
See the weaknesses.

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
The study introduces a unified generative model that improves complex image creation through a functionality-oriented expert design and a multimodal chain of thought process. The model separates visual understanding and generation to strengthen both abilities and follows a four-step reasoning workflow of planning, acting, reflection, and correction. A multi-task training scheme enables each step to be learned independently without costly supervision. Experiments on several benchmarks show clear gains in compositional accuracy and visual quality, highlighting the value of combining functional expert design with stepwise reasoning for better image generation.

### Strengths
1. The separation of generation and understanding experts brings performance gains, which shows potential in unified MLLMs.
2. The multi-task joint training paradigm enables efficient learning of each reasoning step without requiring expensive multi-step supervision, making the approach scalable in terms of data efficiency.
3. The proposed FoX achieves best results on most generation benchmarks, which shows the effectiveness for this framework.

### Weaknesses
1. All experiments are conducted using the Qwen2 0.5B backbone without comparisons across larger scales or alternative architectures, leaving uncertainty about the method’s scalability and general applicability.
2. The proposed MCoT framework introduces multiple reasoning steps (planning, acting, reflection, correction), which likely increase inference time and computational cost compared with direct text-to-image generation.

### Questions
1. The multi-step MCoT process (planning, acting, reflection, correction) likely increases inference time. Can the authors provide quantitative comparisons of inference latency and memory usage versus baseline text-to-image generation?
2. Could the authors clarify whether the proposed FoX and MCoT framework can generalize to other backbone architectures or larger models beyond Qwen2 0.5B?

### Soundness
3

### Presentation
3

### Contribution
3
