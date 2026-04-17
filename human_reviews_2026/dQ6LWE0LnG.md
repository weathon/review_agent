# GranViT: A Fine-Grained Vision Model For Autoregressive Multimodal Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Vision encoders are indispensable for allowing impressive performance of Multimodal Large Language Models (MLLMs) in vision–language tasks such as visual question answering and reasoning. However, existing vision encoders focus on global image representations but overlook fine-grained regional analysis. They are limited in fine-grained perception due to the scarcity of fine-grained annotated data and the lack of a fine-grained pre-training paradigm. In this paper, we propose GranViT, a novel Vision Transformer that integrates fine-grained feature extraction with semantic alignment to Large Language Models (LLMs) via region-level autoregressive training. We first construct Gran-29M, a dataset comprising 29 million natural and OCR images paired with over 180 million high-quality region-level annotations, to enable large-scale fine-grained pretraining. Consequently, we develop a pretraining-adaptation framework along with a self-distillation mechanism to train fine-grained GranViT on Gran-29M. We sufficiently exploit the fine-grained annotations from Gran-29M to resort to bounding-box-to-caption regression to enhance localized visual representation of the vision encoder in the pretraining and caption-to-bounding-box regression to improve vision feature utilization and localization for LLM in the adaptation. We further incorporate a self-distillation mechanism that imposes explicit localization constraints on the vision encoder to strengthen its regional reasoning capability. Extensive experiments show that GranViT surpasses existing vision encoders and attains strong transferability to varying LLMs. Remarkably, it achieves state-of-the-art results on fine-grained recognition, multimodal VQA, and OCR understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles existing MLLMs’ vision encoders’ poor fine-grained perception (caused by scarce annotated data and no fine-grained pre-training). It builds the Gran-29M dataset (29M images, 180M+ region annotations) and proposes GranViT with a pretraining-adaptation framework to enhance localized representation. Experiments show GranViT outperforms other encoders, transfers well to LLMs, and hits SOTA in fine-grained recognition, VQA, and OCR tasks.

### Strengths
1. GranViT builds Gran-29M (29M natural/OCR images, 180M+ region annotations) to solve fine-grained data scarcity for pretraining.
2. It uses a pretraining-adaptation framework (Bbox2Caption/Caption2Bbox) plus self-distillation to boost fine-grained perception.
3. GranViT achieves SOTA in fine-grained tasks, VQA, OCR, and works well with various LLMs.

### Weaknesses
1. It is recommended to add the results of SAILViT in Figure 1(a). 
2. In Figure 1(b), why does the visualization of SAILViT differ completely from that of other methods?
3. How was the two-stage strategy used in the paper determined, and would changing its order or combining the two stages have any impact?
4. This paper proposes Gran-29M for training, does this result in the model using far more training tokens or requiring much longer training time compared to previous methods?

### Questions
Refer to Weaknesses

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
The paper introduces a fine-grained visual representation learning framework for multimodal large language models through a two-stage training paradigm with self-distillation. The model leverages Bbox2Caption and Caption2Bbox tasks to strengthen both visual detail extraction and local region understanding, achieving strong results on OCR and fine-grained benchmarks. Experiments show that GranViT outperforms prior methods like SAILViT and SigLip2, maintaining high transferability across different LLM backbones and tasks.

### Strengths
1. The two-stage pretraining pipeline with both Bbox2Caption and Caption2Bbox tasks effectively enhances visual localization and detailed reasoning.
2. Table 1 results are good, demonstrating good improvements over baselines across different capabilities.
3. The experiments are comprehensive, covering relatively strong vision-language models of various sizes and benchmarks of different domains.

### Weaknesses
1. In Table 2, it seems that the model does not scale well with respect to the model size. Compared with the SAILViT baseline, the improvement is around 0.8 with Qwen2.5-3B and 0.4 with Qwen2.5-7B. Thus, it is unclear if the method can scale well to larger sizes.
2. From Table 1/2/3, it seems that their method doesn't lead to improvements, or even the model performance in some domains, such as MMMU and other reasoning tasks. 
3. The authors claim they achieve state-of-the-art performance on certain tasks (line 109-110) while only comparing with other small-size models, which is clearly an overclaim.

### Questions
1. How can you expand beyond existing data sources?

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
3

### Summary
This paper proposes a method to enhance the fine-grained understanding capabilities of vision encoders in multimodal large language models (MLLMs). To this end, the authors first construct a dataset containing 29 million image-text pairs with fine-grained annotations, including localized bounding boxes and descriptive captions. They then introduce a two-stage training strategy that combines autoregressive learning with distillation loss to fully leverage the rich supervision provided by the dataset. Experiments on various benchmarks demonstrate the effectiveness of the proposed method and framework, achieving higher accuracy than previous approaches.

### Strengths
* This paper addresses an important problem and proposes an effective strategy to enhance the fine-grained understanding capabilities of vision encoders.

* The work introduces a large-scale dataset with fine-grained captions, which could be a valuable resource for the community—especially if it is open-sourced.

* The paper is well-organized and clearly written, making it easy to follow.

### Weaknesses
* The performance improvement over prior methods appears marginal. For example, in Table 2, the proposed method achieves only +0.83 and +0.36 average gains over SAIT-ViT on Qwen-3B and Qwen-7B, respectively. Given the relatively small margins, the practical significance and effectiveness of the proposed approach warrant further discussion.

* Qualitative examples are missing. It would be highly informative to include case studies that illustrate: (1) examples where the baseline ViT fails but GranViT succeeds, demonstrating the benefits of the proposed method; and (2) failure cases where GranViT still struggles, which would help clarify the current limitations.

* The attention visualizations in Figure 1 are not fully convincing. The images shown are relatively simple, containing only one or a few objects, and the attention maps of other ViT models also highlight local regions. To better demonstrate the advantages of GranViT, it would be more compelling to include examples from complex or text-heavy scenes (e.g., documents, charts, or cluttered environments), where precise local attention is critical.

* The impact of the hyperparameter λ is not analyzed or compared in the ablation experiments. A sensitivity study on λ would help validate the design choices and improve the reproducibility of the method.

* The title "Autoregressive Perception" may be misleading. As far as I understand, the vision encoder remains bidirectional rather than employing a causal (autoregressive) attention structure. A more descriptive title that better reflects the actual modeling approach could 
improve clarity.

** Minor comments:** 

There is a typo in Table 1: In the OCRBench row and SAIL-ViT column, the value "590" should likely be "59.0", as it is inconsistent with the scale of other entries.

### Questions
* The motivation behind the distillation loss is not fully clear. The teacher model receives a cropped image as input, while the student model processes the full image—without any textual guidance to align their feature representations. Since both are standard ViTs without task-specific conditioning, it is unclear how they are expected to produce semantically consistent features at corresponding spatial locations. Could this setup introduce ambiguity in the distillation target, and if so, how is it mitigated?

* Table 8 shows that fine-tuning the ViT leads to improved performance. However, the authors choose to keep it frozen during training due to concerns about computational cost. Given that ViTs typically have significantly fewer parameters than the LLM, it would be helpful to quantify the additional computational overhead of updating the ViT. Is the cost primarily in memory, FLOPs, or training stability? A more detailed discussion would strengthen the justification for this design choice.

* The ViT is optimized using gradients from the autoregressive loss of the LLM, which requires end-to-end training with a large language model. An alternative approach—used in several open-vocabulary object detectors such as RegionCLIP and GLIP—is to directly supervise the ViT using region-level captions in its feature space, without involving the LLM during training. This can be more computationally efficient. Could the authors discuss whether their LLM-dependent strategy offers advantages over such direct, LLM-free supervision methods, in terms of performance, scalability, or generalization?

### Soundness
2

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
4

### Summary
The paper proposes GranViT, a vision encoder for MLLMs aimed at fine-grained perception. The key ideas are: 1) Proposes a pretraining corpus Gran-29M of 29.5M images with 183.6M region-level annotations built by combining natural-image and OCR sources and auto-generated captions and bounding boxes. 2) Two-stage training recipe: tage-1 tunes the vision encoder + projector with LLM frozen using Bbox2Caption; Stage-2 freezes the vision encoder and tunes projector + LLM using Caption2Bbox to strengthen localization and transferability. The results show SOTA or strong performance on fine-grained detection and OCR tasks, comparable VQA, and slight trade-offs in general reasoning.

### Strengths
- Clear problem focus and a method explicitly tailored to address the problem. The 2-stage training procedure is well-motivated and empirically supported.
- Contribution of a large, diverse, and structured pretraining data with concrete filtering procedure.

### Weaknesses
- Data quality control: Gran-29M relies heavily on auto-generated annotations, it remains unclear whether the dataset quality is good enough for large-scale training. A data quality auditing process needs to be applied.
- The same Qwen2.5-VL family was used to generate captions and later to consume the vision features in adaptation stage. It remains unclear whether the model performance gain was inflated by distributional closeness to its own generated text style or by leakage.

### Questions
1. Is there any verification or auditing of the auto-generated data annotations? How noisy are the Qwen-generated local captions?
2. The Qwen2.5-VL models were used to generate the captions then also used to serve as the LLM backends. Did you evaluate with non-Qwen LLMs for captioning or LLM backends?
3. Localization baselines: It would be great to include comparisons against task-specific localization and grounding models, beyond the existing MLLM-encoder baselines.

### Soundness
2

### Presentation
3

### Contribution
3
