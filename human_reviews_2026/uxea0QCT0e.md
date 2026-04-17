# ViPER: Empowering the Self-Evolution of Visual Perception Abilities in Vision-Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
The limited capacity for fine-grained visual perception presents a critical bottleneck for Vision-Language Models (VLMs) in real-world applications. Addressing this is challenging due to the scarcity of high-quality data and the limitations of existing methods: supervised fine-tuning (SFT) often compromises general capabilities, while reinforcement fine-tuning (RFT) prioritizes textual reasoning over visual perception. To bridge this gap, we propose a novel two-stage task that structures visual perception learning as a coarse-to-fine progressive process. Based on this task formulation, we develop ViPER, a self-bootstrapping framework specifically designed to enable iterative evolution through self-critiquing and self-prediction. By synergistically integrating image-level and instance-level reconstruction with a two-stage reinforcement learning strategy, ViPER establishes a closed-loop training paradigm, where internally synthesized data directly fuel the enhancement of perceptual ability. Applied to the Qwen2.5-VL family, ViPER produces the Qwen-Viper series. With an average gain of 1.7\% on seven comprehensive benchmarks spanning various tasks and up to 6.0\% on fine-grained perception, Qwen-Viper consistently demonstrates superior performance across different vision-language scenarios while maintaining generalizability. Beyond enabling self-improvement in perceptual capabilities, ViPER provides concrete evidence for the reciprocal relationship between generation and understanding, a breakthrough to developing more autonomous and capable VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents ViPER, a self-supervised loop that lets a VL model bootstrap its own visual-perception skills without human labels or external teachers. Key technical novelty is an internal critic that detects its own failures, converts them into executable image-editing prompts, synthesises hard negatives, and continues fine-tuning itself.

### Strengths
- first fully self-driven loop that converts its own perception failures into targeted training images.
- The algorithm is presented through a modular pipeline, illustrated with a step-by-step running example, and accompanied by released code and 2.1 M generated images that together ensure reproducibility.
- By enabling 3–8 B parameter models to outperform 10× larger proprietary counterparts without additional labels, ViPER offers an immediately practical and conceptually new route toward continual self-improvement of visual perception abilities.

### Weaknesses
- The quality of viper10k data is constrained by the performance of the models used for its creation, potentially leading to a self-reinforcing loop of model-centric biases rather than ground-truth visual reasoning.
- The paper contrasts VIPER's efficiency with the computationally inefficient nature of large-scale SFT/distillation. However, the proposed framework introduces a high computational requirement for data generation.

### Questions
Please refer to weaknesses.

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
This paper introduces a self-bootstrapping paradigm that enhances VLMs in recognizing fine-grained details and capturing dynamic differences. The core contribution is a two-stage RL optimization algorithm used to bootstrap the base model. The first stage requires the model to self-reflect on its initial caption and perform refinements. The second stage then requires the model to predict the visual operations based on the refined information. Evaluation across several benchmarks proves the method’s effectiveness in improving fine-grained perception and mitigating hallucinations. Ablation studies further validate the necessity of the two-stage structure over mixed RL, confirming that a cold start is unnecessary. Finally, the authors introduce an automated online data construction system designed for training this model.

### Strengths
* The idea of self-bootstrap VLMs through a two-stage training paradigm is interesting and novel, which improves the model in fine-grained visual perception.
* The ablation studies are extensive, validating the key design choices of VIPER.
* The results on several multimodal benchmarks show the effectiveness of the proposed training paradigm.

### Weaknesses
* This method heavily relies on a diffusion model, which is used not only for caption-to-image generation but also for image editing. This places high demands on the diffusion model's accuracy and subsequently becomes a bottleneck. The instruction-following capability of multimodal generative models, particularly for tasks requiring fine-grained compositional generation, is currently not very satisfactory.
* The paper's data construction pipeline is not very clear, and some points remain ambiguous even after reviewing the appendix. For example, it is unclear how the refined caption is obtained.

### Questions
* How many RL steps do you train for each stage?
* Maybe the author can try to report the results on MMVIP[1] and Inst-IT[2] benchmarks to further evaluate the fine-grained, instance-related perception capabilities.


[1] Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs

[2] Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ViPER, a self-evolutionary framework designed to enhance visual perception capabilities in Vision-Language Models. The authors identify a critical bottleneck in VLMs: the limited fine-grained visual perception, which is difficult to address with traditional methods like supervised fine-tuning and reinforcement fine-tuning. To overcome this challenge, they propose a novel two-stage task formulation for visual perception learning, structured as a coarse-to-fine process. This approach is implemented in the ViPER framework, which integrates a self-bootstrapping mechanism for iterative self-critiquing and self-prediction, allowing the model to evolve by generating its own training data. The framework employs a dual-granularity reconstruction process—image-level and instance-level—and integrates a two-stage reinforcement learning strategy. ViPER was applied to enhance the Qwen2.5-VL model, producing the Qwen-Viper series, which showed improvements across various benchmarks.

### Strengths
The paper presents a clear and well-structured approach to enhancing visual perception in Vision-Language Models, with an innovative self-evolutionary framework, ViPER, supported by robust experimental design and comprehensive benchmarks. The detailed illustrations and methodical experimental setup effectively demonstrate the framework's performance improvements.

### Weaknesses
* While the paper demonstrates the effectiveness of ViPER in improving visual perception, it primarily compares the Qwen-Viper models against a limited set of benchmarks. Including comparisons with more diverse and recent methods would provide a broader context for the proposed approach.

* Although the paper conducts some ablation studies, a more thorough analysis of the individual components of the ViPER framework—such as the specific impact of the data synthesis module or the two-stage reinforcement learning—would help clarify the contributions of each part to the overall improvements.

* The method relies heavily on self-generated data, which could potentially lead to issues with scalability and efficiency, especially for larger datasets. Future work could explore optimizations to reduce computational overhead and improve the model’s scalability without sacrificing performance.

* The study focuses heavily on visual perception tasks, but the integration of visual understanding with textual reasoning could be further explored. The interplay between these two components could be better examined to understand their combined impact on multimodal reasoning tasks.

### Questions
Please refer to Weakness.

### Soundness
3

### Presentation
2

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
This paper proposes ViPER, a self-bootstrapping training framework to improve fine-grained visual perception in VLMs without relying on external curated data. The core idea is a two-stage, coarse-to-fine framework: (1) Caption Self-Refining, in which a VLM captions an image, a diffusion model redraws the image from that caption, and the VLM critiques and then refines its caption by comparing the redraw with the original; (2) Visual-Operation Predicting, in which the model infers the edit applied to an image pair, thereby learning to attend to small, local changes. These yield a self-synthesized training set, Viper10K, and feed a coupled two-stage RL process, producing the Qwen-Viper variants from Qwen2.5-VL bases. On seven benchmarks encompassing single-image, multi-image, and hallucination tasks, Qwen-Viper shows ~1.6–1.7% average gains and up to 6.0% on fine-grained perception axes, with ablations supporting the need for both stages and for two-stage (vs. mixed) RL.

### Strengths
- This work presents a coarse-to-fine framework that separates holistic image captioning from localized visual-operation prediction. This design cultivates both scene-level understanding and region-specific editing reasoning, forming an executable synthesis pipeline.
- The system forms a closed loop that uses mismatches between the initial captions and the redrawn images as supervision, enabling label-free improvement.
- Across seven benchmarks covering multi-image tasks and hallucination diagnostics, the models improve consistently with largest gains on fine-grained perception up to +6.0% on the 7B variant. 
- Ablations confirm that the two-stage design surpasses mixed RL approaches, with both stages providing complementary benefits.

### Weaknesses
- Results are limited to Qwen2.5-VL (3B/7B) while framework generality across other architectures/sizes is not demonstrated.
- Data synthesis relies on Qwen-Image and OmniGen2 to reconstruct/edit images without  cross-generator report or real-edit robustness tests, leaving a risk of generator-specific artifacts/shortcuts.

### Questions
- How sensitive are your results to the choice of BGE-M3? 
- Have you run a supervised-only baseline on the same data/budget to quantify the gain from RL?
- Can you report results on at least one non-Qwen VLM (e.g., LLaVA/InternVL)?

### Soundness
3

### Presentation
3

### Contribution
3
