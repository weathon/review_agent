# Self-Improvement Anomaly Detection via Large Language Model for Unsupervised Zero-shot Anomaly Detection

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Zero-shot anomaly detection has emerged to overcome the limitations of conventional methods, which depend on learning the distribution of normal data and struggle to generalize to unseen class. However, existing zero-shot methods often rely on anomalous data during training, which is impractical in real-world settings where such data are scarce or unavailable. To address these limitations, we propose a novel unsupervised zero-shot anomaly detection framework, self-improvement anomaly detection with large language model that requires no anomalous data during training. It leverages self-improvement large language model-based architecture that refines textual responses grounded in input images. To support semantic interpretation, we design stage prompts that guide the large language model using visual features spanning from local patterns to global semantics. Our approach not only produces interpretable anomaly maps but also enhances semantic understanding of normality, offering a new direction for zero-shot anomaly detection under realistic anomaly-free constraints. Extensive experiments on nine real-world datasets from both industrial and medical domains demonstrate the effectiveness of our approach. Our self-improvement anomaly detection with large language model outperforms state-of-the-art methods across various unsupervised zero-shot anomaly detection benchmarks, validating its robustness and generalizability across diverse datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focus on the unsupervised zero-shot anomaly detection task and proposes Self-Improvement Anomaly Detection with Large Language Model (SIAD-LLM). Unlike existing zero-shot methods that rely on labeled anomalies, SIAD-LLM operates in fully anomaly-free environments and generalizes to unseen datasets. MLLM is leveraged to obtain rich-contextual information and augment the text for detect anomaly. A stage prompt template integrates multi-scale features to enhance localization, while an enhancement module and adapter are proposed for better alignment between vision and language representations. Experiments on nine industrial and medical datasets show that SIAD-LLM achieves good performance.

### Strengths
1. Proposing a new task called unsupervised zero-shot anomaly detection. The main difference with zero-shot anomaly detection is that the current zero-shot anomaly detection methods often need to use auxiliary dataset containing both normal and anomamous samples for finetuning, while unsupervised zero-shot anomaly detection only requiring to access to auxiliary normal samples. The authors argue that this could be useful because the anomalies are scarces in practice.

2. Leveraging MLLM for self-improvement of text prompt to augment and enrich the expression capability of text.

### Weaknesses
1. The motivation for necessity of the unsupervised zero-shot AD task is in doubut. Although the current zero-shot AD methods mostly requires to use dataset containing both normal and anomalous samples for finetuning, I don't think this is a problem because the anomalies refers to those in the auxiliary dataset. The methods does not have specific requirement for the auxiliary datasets, so we can simply choose the datasets with anomalies as the auxiliary dataset for finetuning. Since the method developed for the unsupervised zero-shot AD alos need a auxiliary dataset for finetuning, why not directly choose one that contains anomalies. Therefore, I doubt the necessity to specifically develop methods for unsupervised zero-shot AD.

2. The motivation of each module is weak and lacks supporting evidence. Although the paper focuses on the unsupervised zero-shot anomaly detection (UZSAD) task, the motivations for the enhancement modules and the self-improvement mechanism mainly emphasize enriching contextual representations rather than directly addressing the UZSAD challenge.

3. Training objective is omitted in the main paper. While the framework appears to involve a training process, the paper does not clearly define or explain the optimization objective, making it difficult to understand how the model learns the adaptor.

4. The overall framework design seems fragmented, with multiple modules loosely combined rather than forming a coherent strategy for the UZSAD task. As a result, the novelty of the proposed approach is limited.

5. Although the method claims to operate in an anomaly-free setting, it still relies on pseudo-anomalies. When other zero-shot anomaly detection methods also employ pseudo-anomalies, SIAD-LLM performs weaker than AnomalyCLIP, even with the use of MLLMs.

6. The use of multimodal large language models (MLLMs) significantly increases computational cost, and the self-improvement mechanism further exacerbates this inefficiency.

7. It is unclear how the self-improvement mechanism generalizes to unseen datasets. Since the self-improvement operates on the training data, it remains uncertain how the generated textual enhancements transfer during inference. The inference process should be clearly described.


8. The ablation study is insufficient. The effectiveness of the stage-wise and patch-wise enhancement modules should be analyzed individually to verify their actual contribution to performance.

### Questions
1. The paper does not describe how pseudo-anomalies are generated during training. Could the authors clarify the specific augmentation strategies or mechanisms used to create these pseudo-anomalies?
2. The training objective is omitted in the main paper. What loss functions or optimization criteria are used to train the proposed framework?
3. The inference process is unclear. How does the model utilize the self-improved textual prompts when applied to unseen datasets during inference?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces SIAD-LLM, a self-improvement anomaly detection framework leveraging large language models (LLMs) for unsupervised zero-shot anomaly detection (UZAD), a more realistic and challenging setting where only normal data is available during training and both unseen normal and anomalous data are encountered during testing. SIAD-LLM combines stage-specific visual features via tailored prompt templates and dynamically refines its prompt and reasoning process using LLM-driven, image-grounded question answering, aiming to enhance semantic representation of “normality” and “abnormality.” The model is experimentally evaluated across nine industrial and medical datasets, achieving strong results in both image-level and pixel-level anomaly detection, alongside ablations and qualitative comparisons to prior work.

### Strengths
- The paper formally proposes the Unsupervised Zero-Shot Anomaly Detection (UZAD) task, which unifies unsupervised and zero-shot paradigms. This definition is both conceptually clear and practically valuable.
- The paper is logically structured. The methodology section systematically introduces each module, helping readers understand how the framework achieves self-improvement.
- The proposed method demonstrate excellent performance on multiple datasets.

### Weaknesses
- major:
- W1: The related work omits direct comparison and discussion to multiple recent LLM-driven anomaly detection approaches and prompt-based few-shot/generalist frameworks. For example, CLIP-AD, GPT-4V-AD, AD-LLM, FiLo, InCTRL are all recent, impactful, highly relevant, and unmentioned. This not only undercuts the claim of novelty but also limits the reader’s ability to judge where SIAD-LLM truly advances the field.
- W2: The paper combines focal loss, dice loss, and cross-entropy loss in its training objective (Appendix B), but does not explain why these terms are equally weighted or whether any of them are redundant or sensitive in the UZAD setting.
- W3: Despite strong baselines, some recently proposed models, such as CLIP-AD, FiLo, are not included, either as ablation, comparison, or discussion, which could mislead regarding true state-of-the-art.

- minor:
- W4: The paper describes the full SIAD-LLM pipeline in text and figures, but does not provide any pseudocode or algorithm box for the overall method. This omission makes it harder to follow the implementation logic and understand how the self-improvement loop is executed step by step.
- W5: The paper briefly mentions the high computational burden of LLMs but provides no concrete measurements or efficiency metrics. Without quantitative evidence, it is unclear how SIAD-LLM trades off accuracy for inference time, GPU load, or memory usage—especially compared with prior methods.
- W6：The adapter module in Section 3.4 introduces a balancing coefficient λ = 0.1 to merge updated and original feature representations, but the paper provides no explanation or sensitivity analysis for this choice. Without justification, it is unclear whether the model’s performance or stability depends on this parameter.

### Questions
- Q1: In Section 3.4, the adapter module introduces a balancing coefficient λ = 0.1 to combine the updated and original feature representations. Could the authors elaborate on how this value was determined — for example, through empirical tuning, theoretical reasoning, or stability considerations?
- Q2: Given the acknowledged high computational cost, could the authors report (1) average inference/training time per image versus prior work, and (2) the primary bottlenecks in GPU/memory use?
- Q3: Regarding the loss terms used in Appendix B: why were focal, dice, and CE losses all equally weighted, and are any of them especially sensitive for UZAD?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an unsupervised zero-shot anomaly detection framework that does not require any anomalous samples during training. The authors introduce a self-improvement anomaly detection pipeline, where textual responses generated by an LLM are iteratively refined based on input images. In addition, a stage-wise prompt design is adopted to capture information from local patterns to global semantics. Extensive experiments across multiple datasets demonstrate the method’s effectiveness.

### Strengths
1. The paper presents a clear motivation and clearly identifies the gap in existing zero-shot anomaly detection research.

2. The proposed method is evaluated on multiple diverse datasets, demonstrating consistent performance gains.

3. The writing is clear and easy to follow, making the methodological flow understandable.

### Weaknesses
1. Although the paper claims to be “real-world applicable,” the full pipeline relies on LLM inference and iterative prompt-looping, which is computationally expensive. A discussion on the cost–performance trade-off is missing.

2. The pseudo-anomaly generation strategy is described only briefly. The method relies on data augmentations to synthesize anomalies, yet no systematic comparison or theoretical justification of these augmentations is provided.

3. While the paper provides prompt templates for different datasets, it is unclear whether manual prompt editing is required when deploying the method in new domains.

4. Some baselines (e.g., AnomalyCLIP, AA-CLIP) typically rely on anomaly samples or fixed prompt templates. It is unclear whether their hyperparameters were properly re-tuned under the UZAD setting to ensure a fair comparison.

### Questions
See Weaknesses:

1. Although the paper claims to be “real-world applicable,” the full pipeline relies on LLM inference and iterative prompt-looping, which is computationally expensive. A discussion on the cost–performance trade-off is missing.

2. The pseudo-anomaly generation strategy is described only briefly. The method relies on data augmentations to synthesize anomalies, yet no systematic comparison or theoretical justification of these augmentations is provided.

3. While the paper provides prompt templates for different datasets, it is unclear whether manual prompt editing is required when deploying the method in new domains.

4. Some baselines (e.g., AnomalyCLIP, AA-CLIP) typically rely on anomaly samples or fixed prompt templates. It is unclear whether their hyperparameters were properly re-tuned under the UZAD setting to ensure a fair comparison.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a self-improvement anomaly detection framework that integrates large language models (LLMs) into unsupervised zero-shot anomaly detection. The central idea is to leverage the semantic reasoning ability of an LLM to iteratively refine textual prompts or pseudo-anomaly descriptions, thereby enhancing the alignment between language-guided semantics and visual features extracted from pre-trained vision encoders such as CLIP. The approach is intuitively appealing and aligns with current trends in multimodal learning; however, the overall contribution is relatively limited. The framework mainly combines existing components—LLM-based prompt generation, CLIP feature extraction, and iterative pseudo-label refinement—without introducing fundamentally new mechanisms or theoretical insights. Moreover, the paper lacks a detailed explanation of how the self-improvement process operates and why LLM feedback concretely enhances anomaly detection capability. The experimental section demonstrates some performance improvement, but the analysis is not sufficiently comprehensive to isolate the contribution of each module. Overall, while the idea of incorporating LLM reasoning into zero-shot anomaly detection is timely and potentially valuable, the current work remains incremental and underexplored, requiring clearer methodology, stronger empirical evidence, and a more rigorous justification of its claimed advantages.

### Strengths
1. The concept of iteratively refining pseudo-anomaly prompts or descriptions through LLM feedback is appealing and could inspire further exploration of self-adaptive mechanisms in ZSAD.

2. The authors conduct experiments on several benchmark datasets, showing that the proposed method achieves competitive or improved results compared to existing ZSAD baselines. The results indicate that LLM-based semantic refinement can positively improve detection performance.

### Weaknesses
1. Although the paper includes experiments, the analysis is shallow. There are no detailed ablation studies, sensitivity tests, or visualizations to demonstrate why the proposed method works.

2. The proposed method relies heavily on both CLIP and LLM to perform anomaly detection, which significantly increases computational complexity. Moreover, the framework requires two forward passes through CLIP for each input during inference to generate the final results, further increasing the computational overhead. This design raises concerns about scalability and practical deployment, especially in real-time or resource-constrained scenarios. The authors should provide a more detailed analysis of the computational cost.

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
