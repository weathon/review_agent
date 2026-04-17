# Detective SAM:  Adaptive AI-Image Forgery Localization

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Image forgery localization in the generative AI era poses new challenges, as modern editing pipelines produce photorealistic, semantically coherent manipulations that evade conventional detectors while model capabilities evolve rapidly.
In response, we develop Detective SAM, a framework built on SAM2, a foundation model for image segmentation, that integrates perturbation-driven forensic clues with lightweight feature adapters and a mask adapter to convert forensic clues into forgery masks via automatic prompting.
Moreover, to keep up with the rapidly evolving capabilities of diffusion models, we introduce AutoEditForge: an automated diffusion edit generation pipeline spanning four edit types. This supplies high-quality data to maintain localization accuracy under newly released editors and enables continual fine-tuning for Detective SAM.
Across seven benchmark datasets and seven baselines, Detective SAM delivers stable out-of-distribution performance, averaging 36.99 IoU / 44.19 F1, a 33.67% relative IoU gain over the best baseline. Further, we show that state-of-the-art edits cause localization systems to collapse.
With 500 AutoEditForge samples, Detective SAM quickly adapts and restores performance, enabling practical, low-friction updates as editing models improve.
The pretrained weights, AutoEditForge, and evaluation script are available at https://github.com/Gertlek/DetectiveSAM

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Detective SAM, a model for localizing AI-generated image forgeries. The proposed framework builds on top of SAM2 by incorporating perturbation-driven signals to fine-tune two auxiliary modules, i.e., a mask adapter and a feature adapter, while keeping the backbone frozen. In addition to forgery localization, the authors introduce an automated pipeline that integrates instruction-based local editing methods, allowing them to evaluate the generalization ability of Detective SAM with respect to various editing techniques.

### Strengths
1. The paper is well-structured, and the presentation is clear and easy to follow.

2. The task of forgery localization is relevant to current research in trustworthy AI.

3. Leveraging a large foundation model like SAM2 for general-purpose forgery detection is a well-motivated and promising direction.

4. The experiments support the performance claims well. In particular, the integration of local editing methods into the evaluation pipeline adds an interesting perspective on the model’s generalization.

### Weaknesses
While I do not find any major flaws, I have a few questions and suggestions for improvement:

1. Perturbation design: According to Section 3.2, each image is associated with N perturbed variants (e.g., Gaussian blur, Gaussian noise, JPEG compression). Table 3(a) suggests that localization performance improves with the number of perturbations, but it is unclear whether this improvement saturates. Could the authors explore a wider range of perturbations and analyze how increasing N continues to affect performance?

2. Failure case analysis: It would be helpful to include more examples of failure cases, particularly for fine-grained or challenging forgery categories. A qualitative and/or quantitative discussion of current model limitations would enhance the reader’s understanding of where Detective SAM might still fall short.

### Questions
Please see the Weaknesses for details.

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
3

### Summary
This paper proposes Detective SAM, a lightweight and adaptive framework for localizing forgeries in diffusion-edited images. Rather than relying on differences between clean and edited images, the method utilizes perturbation-induced embedding shifts extracted from a single edited image. These perturbation-based forensic signals are injected into a frozen SAM2 backbone using two lightweight modules: feature adapters that refine the decoder inputs, and a mask adapter that generates an automatic heatmap prompt to guide forgery segmentation. To maintain robustness against evolving editing techniques, the authors further introduce AutoEditForge, an automated pipeline that generates diverse and realistic diffusion-based forgeries for continual fine-tuning. Evaluations across seven datasets and seven baselines show that Detective SAM achieves strong out-of-distribution performance while enabling rapid adaptation to new editing models with minimal computational overhead.

### Strengths
1. The paper clearly identifies the limitations of existing forgery localization methods when applied to diffusion-edited images, and effectively addresses both the performance degradation and the challenge of adapting to continuously emerging editing models.
2. The proposed approach fully leverages the segmentation capabilities of SAM2 by introducing lightweight adapter modules that can be efficiently trained and reliably guide localization across a variety of recent diffusion-based editing pipelines.
3. Extensive experiments against seven competitive baselines across seven diverse datasets provide strong empirical evidence of the robustness and generalization performance of Detective SAM.

### Weaknesses
1. While the authors emphasize the importance of continual fine-tuning using AutoEditForge (Table 2) and set catastrophic forgetting as one of persistent problems in current IFL systems, it remains unclear whether the proposed adapter modules are genuinely resilient to catastrophic forgetting. The current evidence does not directly show whether older edit distributions remain preserved when new diffusion models are introduced.
2. The paper consistently refers to unseen datasets as “out-of-distribution” (OOD). However, the training set MagicBrush and the test set AutoSplice share the same editing model, DALL-E. Given that the diffusion models are trained to estimate training image distributions, the distinction between in-distribution and OOD becomes ambiguous.
3. The feature adapters are described as learning residual delta corrections between clean and perturbed embeddings, yet the paper provides limited analysis to verify what these corrections represent or how they contribute to performance improvements.
4. The manuscript repeatedly emphasizes the lightweight and efficient nature of Detective SAM. While this is valuable, the motivation for efficiency as a primary design goal is not entirely clear, given that forgery localization typically prioritizes accuracy over speed.
5. Although AutoEditForge is central to continual adaptation, the paper does not quantitatively evaluate the realism or consistency of its generated edits.

Note: Weaknesses 1-5 correspond directly to Questions 1-5.

### Questions
1. It would be helpful if the authors could clarify how the adapter design explicitly mitigates forgetting, especially considering that fine-tuning uses only 500 images from recent diffusion models. If future models generate substantially more new edits, would Detective SAM still maintain localization accuracy on prior model generations? A brief discussion or empirical analysis around Lines 81–83 could strengthen this point.
2. Could the authors clarify their OOD definition in this context? Specifically, when two different image inputs are processed by the same diffusion model (e.g., DALL-E), how is it justified to treat them as distinct distributions?
3. Including visualization, ablation, or interpretability results showing what the delta corrections capture--such as which regions or features are being adjusted--would make the role of this component clearer. A short qualitative example could effectively illustrate its necessity and benefit.
4. The authors could further clarify in what deployment scenarios the efficiency aspect is critical. For instance, is the method intended for frequent online updates or resource‑limited forensic systems? Discussing the practical motivation would help contextualize the design choice.
5. It would strengthen the paper if the authors could include either a quantitative assessment (e.g., perceptual quality metrics or human evaluation) or a qualitative justification for the realism and diversity of AutoEditForge outputs. This would better support the claim that the generated data effectively represents real‑world forgery scenarios.

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
5

### Summary
This paper presents Detective SAM, a framework that localizes AI forgeries by detecting feature shifts in a perturbed image using a frozen SAM2 model. It is paired with AutoEditForge, an automated pipeline that generates up-to-date training data. This dual approach allows the model to significantly outperform existing methods and rapidly adapt to counter new, state-of-the-art image editors.

### Strengths
The paper's primary strength is its methodological design, which leverages the powerful SAM2 foundation model for the task of forgery localization. Rather than designing a new architecture, the authors integrate a perturbation-driven forensic clue using lightweight feature adapters and an automatic mask adapter.  This design is effectively paired with the AutoEditForge pipeline for automated data generation. The adaptable model and data pipeline present a structured approach to the challenge of evolving forgery techniques.

### Weaknesses
1. While this work performs very well, it appears more like a system-level optimization from an engineering perspective, particularly the AUTOEDITFORGE component. I do not object to leveraging the capabilities of existing models to improve a specific task, but the authors' overall framework seems more like an integration of existing work. For instance, numerous studies have explored perturbations at both the image and feature levels, and the concept of feature adapters is also well-established. I would like the authors to discuss the scientific contribution of this paper.

2. The method is tightly coupled with the promptable architecture of SAM2. While this is a clever design choice, it raises critical concerns about the generality of the proposed paradigm. It is unclear whether this 'perturb-adapt-prompt' approach can be successfully transferred to other prompt-based segmentation models, or if its success is inextricably linked to the specific properties of SAM2's decoder. The paper would be strengthened by a discussion on the conditions required for this transferability, or an explanation of why it might be limited to this specific model family.

3. Using image perturbations to distinguish manipulated regions is a proven and effective method. However, when external noise is added to an edited image, it is unclear whether this approach can maintain its consistent generalization performance.

### Questions
1. The discussion around Tab 3(a) feels insufficient. Could the authors provide more intuition as to why the combination of Gaussian Noise and Blur emerges as the optimal pairing? Given the paper's claim that  the perturbation type has a significant impact on
the localization performance, it would be beneficial to see experiments with a more diverse set of perturbation types to fully substantiate this argument.

2. How would the proposed method perform in a classic forgery scenario where an image is first subtly edited (not visible to the naked eye) and then globally post-processed with transformations like Gaussian noise to mask the tampering artifacts? Is the model robust enough to distinguish its own diagnostic perturbations from pre-existing, confounding noise in the image?

3. Could you clarify the evaluation protocol regarding the results in Table 1? The fine-tuned Detective SAM_{SOTA} is presented in Table 2 as the final version of the model, yet it is not benchmarked against the initial baselines in Table 1.

4. What quality control mechanisms are in place to ensure the usability of images generated by AUTOEDITFORGE? Could you provide statistics on the pipeline's reliability, such as the approximate proportion of automatically generated samples that require manual intervention or are discarded? Furthermore, what are the common failure modes of this generation process?

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
The paper proposes Detective SAM, a framework for image forgery localization that adapts the Segment Anything Model 2 (SAM2) to detect and precisely mark regions of AI-generated edits. Instead of retraining SAM2, the authors freeze its backbone and attach small trainable “feature” and “mask” adapters that use differences between an image and its perturbed versions (blurred, noised, or compressed) as forensic clues indicating manipulation. These perturbation responses are converted into an automatic heatmap prompt guiding SAM2 to segment the tampered areas. To keep pace with rapidly changing diffusion-based editors, they also introduce AutoEditForge, an automated pipeline that continually produces realistic, labeled image edits (Replace, Remove, Add, Change Partially) for fine-tuning and evaluation. Together, these components aim to build an adaptive, lightweight, and continually updatable system that localizes forgeries across diverse generative-model edits more reliably than prior detectors.

### Strengths
1- The integration of perturbation-driven forensic signals with SAM2 is technically neat and explained with adequate mathematical detail.

2- Results across seven datasets demonstrate stable performance under moderate domain shift, which is rare among IFL systems.

### Weaknesses
1- In Abstract & Sec. 3.3, the system is framed as a continual, lifelong learning pipeline that updates smoothly as new editors appear. But in Sec. 4.1–4.2, the experiments are all offline fine-tuning on small fixed batches (500 samples), without any demonstration of online or streaming adaptation. The “continual” claim is thus aspirational rather than empirically validated.

2- In Abstract & Intro, Detective SAM supposedly handles arbitrary new diffusion editors “without retraining.” but in Sec. 4 in the Training specification, the model is trained and tested only on diffusion-edited datasets (SIDA, MagicBrush, AutoSplice, CoCoGLIDE, NanoBanana) and fails sharply on unseen SOTA editors until retrained. The paper’s own results contradict the “no-retraining” implication.

3-  In Sec. 3.2, they emphasize lightweight adapters—81 k + 887 k parameters—trained in “two hours on an H100.” but in Sec. 4 Baselines, inference comparisons are made against massive MLLMs but all are run on a single H100 GPU, ignoring throughput, preprocessing (multiple perturbations per image), and repeated SAM2 encodings. The real computational cost per sample is much higher than implied; “lightweight” refers only to trainable weights, not runtime.

### Questions
Your method is described as a general framework for image forgery localization, yet the core signal it relies on is perturbation-induced embedding instability (Section 3.1–3.2). This assumes that forged regions respond differently to Gaussian noise, blur, or compression compared to authentic ones. However, not all forgeries—e.g., copy-move, GAN inpainting, or carefully composited human edits—necessarily exhibit such perturbation sensitivity. Could you clarify whether Detective SAM is actually detecting forgery semantics or merely perturbation sensitivity, and how the method would generalize to forgery types that do not produce perturbation-based cues?

This question matters because, the authors blur the distinction between “detecting forgeries” (a semantic problem) and “detecting perturbation-sensitive regions” (a statistical proxy). The two overlap for diffusion artifacts but diverge for other forgery classes. Hence, the approach is domain-specific rather than general, despite the paper’s repeated framing as a “comprehensive framework for modern IFL.”

### Soundness
3

### Presentation
3

### Contribution
3
