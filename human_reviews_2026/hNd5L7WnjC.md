# A Rich Knowledge Space for Scalable Deepfake Detection

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 0

## Abstract
The proliferation of realistic deepfakes has driven the development of numerous benchmark datasets to support detection research. Despite their increasing volume and diversity, no prior effort has systematically consolidated these resources into a unified framework for large-scale model training, nor has there been a massively pre-trained model tailored to deepfake detection. In this work, we introduce Multi-modal Multi-type Integrated Deepfake Dataset (MMI-DD), a large-scale resource containing 3.6 million facial images, the largest collection to date. It unifies diverse benchmarks with uniform preprocessing, and further provides fine-grained annotations across four deepfake types, as well as VLM-generated descriptions capturing both facial and environmental attributes for each image. By leveraging this comprehensive multi-modal dataset, we construct a foundational deepfake knowledge space that empowers our model to discern a broad spectrum of synthetic media. Our method, $SD^2$ (Scalable Deepfake Detection), refines CLIP for deepfake detection, optimizing image-text classification with rich, type-specific labels. We enhance this with intermediate visual features capturing low-level cues and text label separation loss for stability. We further leverage VLM-generated descriptions and contrastive learning to expand the scope of forgery knowledge, reducing overfitting and enhancing generalization. Extensive experiments on challenging deepfake datasets and AIGC benchmark demonstrate the effectiveness, scalability, and real-world applicability of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SD2 (Scalable Deepfake Detection), a CLIP-based visual-language model trained on a newly built MMI-DD dataset containing 3.6 million multimodal facial images across 11 datasets. It proposes three main components: the Cross-Layer Attention Module (CLAM), the Text Label Separation Loss, and the Dual Contrastive Loss to enhance generalization and scalability in deepfake detection. Experimental results demonstrate strong performance across multiple benchmarks, suggesting that SD2 scales effectively with increasing data size.

### Strengths
1. The authors integrate eleven existing datasets into a unified corpus with over 3.6 million multimodal samples, covering four manipulation types and both real and fake instances (Section 3.1, Table 1). They also include human-verified textual labels and VLM-generated descriptions, enhancing the dataset’s semantic richness and enabling cross-modal learning (Figure 2). This large-scale integration and annotation pipeline offer a solid infrastructure that benefits future research in multimodal forensics.

2. The model introduces the Cross-Layer Attention Module (CLAM) to fuse visual features from different transformer layers and employs Dual Contrastive and Label Separation losses to strengthen cross-modal alignment. Each component is separately evaluated in the ablation study, where removing CLAM or either loss leads to performance drops of 1.24–1.97% on the cross-domain mAUC. This structured experimental setup provides convincing evidence that the proposed components meaningfully contribute to the final performance. Across intra-domain and cross-domain tests, SD2 achieves 95.76 mAUC and 87.79 mAUC respectively, surpassing prior models and methods. What I like is that the authors further report that performance continues to improve as training data size increases from 1M to 3.6M samples (Table 6). These findings support the paper’s claim that SD2 scales effectively and avoids the overfitting observed in earlier CLIP-based detection frameworks.

3. The extension to non-facial AIGC data highlights broader applicability. The experiments on non-facial synthetic datasets, such as synthetic scene and object generation benchmarks, show that SD2’s vision encoder generalizes beyond facial forgeries. The model outperforms prior works on the averaged mAUC, confirming the potential of multimodal pretraining for generalized forgery detection. This result is a noteworthy attempt to bridge the gap between face-centric and content-agnostic AIGC detection, which is important in the current stage.

### Weaknesses
1. The overall novelty of SD2 is somehow limited, as most techniques extend existing designs rather than introduce new concepts. The proposed Cross-Layer Attention Module (CLAM) aggregates multi-level visual features from intermediate transformer layers. However, similar multi-scale fusion strategies have long existed in CNN-based architectures, and recent ViT variants also exploit cross-layer interactions. As a result, CLAM feels more like a standard adaptation of established ideas to a CLIP backbone rather than a novel architectural contribution.

2. Parts of the loss terms largely reuse ideas already present in prior multimodal learning frameworks. For example, the Dual Contrastive Loss directly follows CLIP-style image-text alignment, while the Text Label Separation Loss resembles class-wise embedding decorrelation objectives used in prior works.  Without further explanations and comparisons, the learning objective feels incremental rather than innovative.

3. Certain implementation details remain unclear, for example,  parameter updates in Eq. (4). The paper introduces the Text Label Separation Loss (Section 3.3, Eq. 4) but I can not find out which parts of the model are updated under this loss. Clarifying which components receive gradient updates would  strengthen the technical clarity of the paper.

### Questions
Please check Weaknesses for the details.

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
The paper finetunes CLIP image and text encoder with image-text deepfake pairs by curating all the existing datasets. The text and caltegorical annotation for each image is generated using InternVL and gpt o1 respectively.

### Strengths
1. The paper is well motivated. I agree that the current deepfake methods in the community train on FF++ and test on other datasets which doesn't necessarily reflect their in-the-wild performance and are overfitted to the training domain. The proposed methods aims to scale the data bottleneck by training in a contrastive manner using image-text pairs, ands highlight that the model performance improves with dataset size.
2. The paper is well-written. 
3. The performance improvement with data scaling is impressive.
4. The data scaling shows emergent behavior and the performance improves on datasets "beyond facial manipulations".

### Weaknesses
The paper doesn't compare with recent deepfake detection methods. The authors should compare the recent methods by training them using the bigger curated dataset proposed, and then compare it with SD^2, to see whether there is a performance improvement or not . This will highlight whether the CLIP-based LoRA training is actually useful or it is just because of data scaling where the CLIP-based methods are not scalable. It is still not clear that other non-CLIP based methods are scalable or not. 

Please compare with 1 or 2 SOTA methods from NeurIPS 25, CVPR 25, AAAI 25, etc.

### Questions
1. Can you show the results of recent methods when trained on the bigger dataset and show the performance comparison.

### Soundness
4

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
5

### Summary
- This paper presents a visual–language framework for deepfake detection together with a large-scale, multi-modal knowledge space constructed by unifying 11 datasets into a 3.6M-image corpus with consistent preprocessing, five-way type labels (REAL/FS/FR/EFS/FE), and VLM-generated facial and scene descriptions. The goal is to achieve scalable, cross-domain generalization beyond single-dataset training by leveraging richer supervision from text and multi-level visual cues. Specifically, the method (SD2) adapts CLIP with a cross-layer attention module that fuses intermediate and final vision features, a fine-grained image–text classification objective using type-specific labels, and a text-label separation loss to stabilize class embeddings, together with a dual image–text contrastive loss that aligns each image with both facial and scene descriptions. To enable efficient and robust training at scale, the approach fine-tunes CLIP encoders with LoRA and employs a SigLIP-style contrastive objective to avoid very large batches. Furthermore, extensive experiments show superior intra-domain and cross-domain AUC over CNN/Transformer and CLIP-based baselines, and competitive accuracy on a non-facial AIGC benchmark using the frozen vision encoder with linear probing, underscoring the method’s effectiveness, scalability, and practical applicability.

### Strengths
- Overall, the paper is clearly written and easy to follow, with a well-articulated connection between the unified corpus and the proposed vision–language (V–L) objectives.
- The data contribution is significant: the authors unify 11 sources into a 3.6M-image corpus with consistent preprocessing, five-way labels, and VLM-generated descriptions.
- The proposed method is both practical and modular, building on CLIP by introducing cross-layer attention, text–label separation, and dual image–text contrastive training.
- The authors also address training efficiency, employing LoRA fine-tuning and a SigLIP-style objective to reduce batch requirements. The results are strong across both intra- and cross-domain settings, with additional linear-probe transfer experiments on a non-facial AIGC benchmark further demonstrating robustness.

### Weaknesses
- The unified corpus may inherit dataset biases and spurious correlations; issues such as deduplication, identity overlap, and source-specific artifacts are not sufficiently quantified.
- VLM-generated descriptions can introduce noise or hallucinations, yet the paper does not assess caption quality or analyze its impact on downstream accuracy.
- The reported performance gains may be partly attributable to corpus scale or data curation rather than architectural improvements; stronger experimental controls, e.g., equalizing data size, testing alternative prompts, or using simpler adapters.

### Questions
- How are near-duplicates and identity overlaps managed across the 11 sources? Could clarify the criteria for duplicate removal, report per-source identity counts, and describe any identity-disjoint checks implemented to prevent data leakage.
- How accurate are the VLM-generated facial and scene descriptions? Have authors conducted a small-scale human audit (e.g., inter-annotator agreement, error type analysis) or performed an ablation study in which captions are shuffled or masked to quantify their causal impact?
- How are heterogeneous source labels mapped to the five target categories (REAL/FS/FR/EFS/FE)? Does proposed framework support open-set or “other manipulations” (e.g., audio-visual inconsistencies, partial edits, diffusion-based reenactment), and how are such cases handled?
- How sensitive are the results to prompt templates, temperature, and LoRA rank choices? Report the random seeds, hyperparameter ranges, and effect sizes (with confidence intervals) for main comparisons.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces a new large-scale deepfake dataset called MMI-DD, which consolidates 3.6 million images from 11 different sources annotated with fine-grained DeepFake types. The data also contains VLM-generated text descriptions of facial and environmental attributes.​ The authors also propose SD², a scalable, CLIP-based detection baseline. The model uses a multi-modal learning strategy that leverages the detailed annotations through specialized classification and contrastive loss functions.

### Strengths
- The authors make a noteworthy attempt to address a significant challenge in the field: the lack of a large-scale, unified benchmark for DeepFake detection and reasoning. By integrating 11 distinct datasets into a single resource (MMI-DD), they provide a valuable foundation that could encourage the community to move beyond single-dataset training paradigms.
- The work highlights a critical and often overlooked issue, which is the performance saturation or degradation of existing models when trained on increasingly large and heterogeneous datasets.

### Weaknesses
- The paper states that six researchers manually categorized the images, but it does not report the inter-annotator agreement score or detail the protocol for resolving disagreements. This omission makes it difficult to assess the reliability and consistency of the fine-grained labels, which are crucial for the proposed classification loss.
- The annotations are generated using proprietary models like GPT-4o. This assumes that GPT-4o is a good DeepFake detector and reasoning model, which is wrong. MLLMs are inherently built for global semantics rather than capturing subtle inconsistencies that occur in DeepFakes. Papers like [1, 2] show that subtle changes are ignored, and these models are broadly good in overall content understanding. The paper also provides no information on how the quality, accuracy, and potential biases of these machine-generated captions were validated. This means that the annotations are highly unreliable and noisy.
- The paper makes claims about how its components work, for example, that CLAM captures low-level forgery artifacts. However, there is no qualitative evidence, such as attention maps or feature visualizations, to verify these claims and provide insight into the model's decision-making process.
- Section 5.2.2 states the model's vision encoder is frozen for linear probing but also mentions that the protocol involves fine-tuning on new data (SD v1.4 and ImageNet). This contradiction makes it difficult to determine if the strong performance is due to the pre-training on the MMI-DD dataset or the subsequent task-specific tuning.

[1] Tong, Shengbang, et al. "Eyes wide shut? exploring the visual shortcomings of multimodal llms." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024. [2] Huynh, Ngoc Dung, et al. "Vision-Language Models Can't See the Obvious." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

### Questions
- What procedure was followed to resolve labeling disagreements among the researchers?
- How did the authors verify the factual accuracy of the VLM-generated facial and environmental descriptions? Was there a human review process to filter out hallucinations or irrelevant details?
- What is the rationale behind the claim that the model is able to ground the reasoning to the visual cues through the CLAM module?
- The dual contrastive loss aims to disentangle forensic cues from spurious correlations. Have you conducted experiments to test this, for example, by evaluating the model on images where a known fake face is placed in diverse and unseen environments?
- In the general synthetic image detection experiment (Table 4), could the authors clarify whether the performance gains come from the MMI-DD pre-training or from the subsequent fine-tuning on ImageNet and SD v1.4 data?
- Figure 1 and Table 6 show that the proposed model scales better than baselines. How does the computational cost (e.g., total GPU hours) of training SD2 on the full 3.6 million image dataset compare to the other methods? And what are the inference times of each of these models?

### Soundness
1

### Presentation
2

### Contribution
1
