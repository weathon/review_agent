# Target-Oriented Single Domain Generalization

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Deep models trained on a single source domain often fail catastrophically under distribution shifts, a critical challenge in Single Domain Generalization (SDG). While existing methods focus on augmenting source data or learning invariant features, they neglect a readily available resource: textual descriptions of the target deployment environment. We propose Target-Oriented Single Domain Generalization (TO-SDG), a novel problem setup that leverages the textual description of the target domain, without requiring any target data, to guide model generalization. To address TO-SDG, we introduce \textbf{S}pectral \textbf{TAR}get Alignment (STAR), a lightweight module that injects target semantics into source features by exploiting visual-language models (VLMs) such as CLIP. STAR uses a target-anchored subspace derived from the text embedding of the target description to re-center image features toward the deployment domain, then utilizes spectral projection to retain directions aligned with target cues while discarding source-specific noise. Moreover, we use a vision-language distillation to align backbone features with VLM's semantic geometry. STAR further employs feature-space Mixup to ensure smooth transitions between source and target-oriented representations. Experiments across various image classification and object detection benchmarks demonstrate STAR’s superiority. This work establishes that minimal textual metadata, which is a practical and often overlooked resource, significantly enhances generalization under severe data constraints, opening new avenues for deploying robust models in target environments with unseen data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new approach for single-domain generalization where they aim to inject the target domain's information, e.g., art or foggy, to let a model change its behavior depending on the target domain. Specifically, they employ the CLIP model to encode such information given as text and fuse the information into the image embedding. Additionally, they employ distillation loss from the CLIP model and a feature-level mixup approach. Empirically, their approach outperforms baseline methods with a large margin in image classification and object detection.

### Strengths
1. The idea of injecting the target domain's meta information into the image encoder is interesting and sounds reasonable. 
2. The proposed approach seems to be strong in both image classification and object detection. 
3. Empirical results are strong, though I am not sure if the results support their claims enough.

### Weaknesses
1. Their main claim is about the effectiveness of the textual description of the target domain. However, their experiments and methods are not well-designed to support their claim. It is true that combining several techniques outperforms existing approaches, yet it is not clear whether the improvement comes from their main claim. For example, according to Table 6, distillation from the CLIP model appears to significantly improve performance, suggesting that the use of CLIP enhances performance. 

2. Related to 1, it is not clear if the comparison in results is fair in terms of backbone, e.g., Table 2. If they employ the CLIP model during training and other baselines do not have access to CLIP, the comparison might not be fair. 

3. Is the feature-level mix-up not explored in domain generalization? The technique was proposed a long time ago, and applying it to the DG problem sounds like a simple idea. Then, the use of feature mix-up is not very novel.

### Questions
The authors need to make an effort to show how much gain is obtained from their proposed main module. So far, many factors are involved, and not easy to see the pure gain from their novel approach. Please respond to the weaknesses above. 

Minor comments. Some tables are far from the corresponding sentences. Better to fix it.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a Target-Oriented Single Domain Generalization (TO-SDG), which leverages textual descriptions from the target domain during training to improve generalization from a single source domain. To bridge the semantic gap between source and target domains, the authors propose a Spectral Target Alignment framework composed of three components: a spectral target orientation module, a vision-language distillation strategy, and a feature-space mixup technique. Extensive experiments across three standard benchmarks demonstrate the method's effectiveness and improvements over baseline approaches.

### Strengths
1. The proposed method contributes a simple yet effective framework that incorporates target domain text descriptions.

### Weaknesses
1. The experimental section lacks comparisons with several state-of-the-art single domain generalization (SDG) approaches, such as [1-3], many of which demonstrate stronger performance. The omission of these baselines limits the confidence of the performance claims and weakens the empirical contribution of the work.
[1] Chen, Jin, et al. "Meta-causal learning for single domain generalization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.
[2] Li, Deng, et al. "Prompt-driven dynamic object-centric learning for single domain generalization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
[3] Xu, Zhipeng, et al. "Adversarial Domain Prompt Tuning and Generation for Single Domain Generalization." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

2. The technical contribution of this method is insufficient. The Vision-Language Distillation is very common, and has been proposed in many literatures, e.g.,[1].
[1] Yang, Chuanguang, et al. "Clip-kd: An empirical study of clip model distillation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
The feature-level mixup that fuses source feature with target-oriented feature, which is similar to the domain-level mixup in literatures such as [2],
[2] Cao, Meng, and Songcan Chen. "Mixup-induced domain extrapolation for domain generalization." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 10. 2024.
Please show comparisons with these related works, and provide detailed analysis on how your motivation and method differ from these ones.

3. While the proposed TO-SDG task is conceptually novel, its real-world applicability is debatable. In practical scenarios, target domain styles (e.g., “art painting”) are often abstract, noisy, or multi-modal, making it difficult to summarize them with a single descriptive phrase. In this perspective, the proposed setting risks being seen as a promising research task with limited practicality. It’s recommended to discuss the limitations of TO-SDG more explicitly in your manuscript, and consider to conduct experiments on more existing settings.

4. The evaluation on PACS is only conducted using the ‘photo’ domain as the source, which is inconsistent with standard SDG protocols. Typically, performance is averaged over all possible single-source domain settings to ensure fairness and robustness. It’s recommended to extend your experiments to include all four PACS domains as source domains. 

5. SDG methods are mostly evaluated under multiple benchmarks for the image classification task. It’ll be better to provide results on Digits/DomainNet.

### Questions
Please refer to weaknesses.

### Soundness
2

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
4

### Summary
This paper proposes Target-Oriented Single-Domain Generalization (TO-SDG), where a model trained on one labeled source domain leverages a text description of the unseen target domain. The authors introduce STAR (Spectral TARget Alignment), which aligns source features to a CLIP text anchor, projects them onto a target-oriented subspace via SVD, applies vision–language distillation, and performs feature-space Mixup. STAR achieves notable improvements on PACS, DomainNet, and Diverse-Weather benchmarks.

### Strengths
1.  The paper is well organized and easy to follow.

2. Consistent improvements in both classification (PACS, DomainNet) and detection (Diverse-Weather).

### Weaknesses
1. For classification, baselines focus heavily on augmentation-style SDG methods with ResNet-18. It would be informative to see stronger modern backbones (e.g., ConvNeXt/Vit-B) and CLIP-tuned baselines.

2. The ablation removes “projection” entirely or uses bottom-k, but it would be helpful to plot performance vs k (η) across domains (beyond a single sensitivity curve) and report the standalone effect of the mean-shift without projection to disentangle the two parts of STO more cleanly.

### Questions
1. What is the per-iteration overhead of STO (SVD + projection) relative to baseline training for classification and detection?

2. What happens if T is partially wrong or noisy? Can you sweep paraphrases/synonyms to test stability?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies the task of generalization on the target domain when only the source data and high-level information about the target domain (i.e., target-oriented Single-Domain Generalization). The paper claims that this is a novel and more practical setting as it lets you to utilize lightweight information regarding the target domain. To tackle TO-SDG, they propose the Spectral Target Alignment (STAR) method. STAR consists of three main components: (1) re-centers the features towards the target domain using CLIP encoded target's textual information, (2) Aligns image encoder features with CLIP's features via a distillation objective, and (3) boosts generalization by interpolating between source and target cues/features. 

The paper comprehensively compares STAR on TO-SDG with baselines such as open vocabulary object detection and single-domain generalization (SDG) methods, as well as classification methods. Experimental results show that STAR is more effective than these baselines on different types of benchmarks (e.g., adverse-weather generalization and artistic domain shift).

### Strengths
I appreciate the comprehensiveness of the experimental setup on multiple benchmarks, different types of domain shift types, backbones and different tasks. This helps to understand effectivness of generalization the proposed method in various settings.

### Weaknesses
1- The paper proposes a new task, Target-Oriented Single Domain Generalization (TO-SDG), where only high-level information from the target domain (text description) is available. However, this setting is essentially **very similar or identical to zero-shot domain adaptation (ZDA)**, which has been extensively studied. **The paper neither explains the differences from existing ZDA formulations nor compares with representative methods**. Some relevant works are [1–6]. The implementation details and how different backbones (e.g., different models or different resnets) are adopted needs to be better explained in the implementation details. 

2- Several parameters and model choices are unclear. For instance, what versions  CLIP or LLaVA variants are used? how the LLaVA language model is employed to encode textual descriptions?  Clearer descriptions are needed for reproducibility and understanding.

3- In Eq. (2) and L688, the image encoder appears to be ImageNet-pretrained, not CLIP-pretrained. If so, the image features and CLIP text embeddings lie in different spaces, and the claim in L239 ("because φt originates from the same CLIP space as the image embeddings, the shift operates in a modality-aligned coordinate system, avoiding the semantic drift that often plagues purely heuristic normalization schemes,") would not hold? I'd appreciate if this could be further clarified

4- Some statements are inaccurate or exaggerated. For example, L52:  "Yet all prior CLIP-based robustness methods ultimately rely on target images ...." is not accurate and may need to be adjusted. For instance, methods such as CLIP-the-Gap [7] and prompt-tuning approaches like CoOp and CoCoOp do not require target data. While CoOp/CoCoOp adapt prompts using labeled source data (e.g., ImageNet), they are evaluated on unseen target domains (e.g., ImageNet-V2, Sketch, A, R), demonstrating genuine generalization without target access.



[1] Unified Language-driven Zero-shot Domain Adaptation, CVPR 2024
[2]  PØDA: Prompt-driven Zero-shot Domain Adaptation, ICCV 2023
[3] Zero-Shot Deep Domain Adaptation, ECCV 2018
[4] Text-driven Zero-shot Domain Adaptation with Cross-modality Graph Motif Matching
[5] Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation, CVPR 2024
[6]  UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement, ICCV 2025
[7] CLIP the Gap: A Single Domain Generalization Approach for Object Detection, CVPR 2023 
[8] PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization

### Questions
1- What is the difference between TO-SDG and SDG setting? Also, in this setting, don't we need to adapt to different domains separately? This is more like a domain adaptation setting than generalization? I believe addressing this point is essential, as it seems like the task is the main contribution of this paper. Specifically, each component of STAR method has been extensively studied for DA/DG domain. E.g., [7-8] leverage image-text transferability property in the clip space, AdaIN (as scited in the paper) uses batch statistics for augmentation and interpolation between samples and the knowledge distillation has been studied in many works such as studend-teacher framework. 

2- A comprehensive comparison against zero-shot DA methods is needed both in related works and experimental results.

3- Are image encoder backbone and clip text encoder in the same space? if not how would Eq2 and L239 hold?

### Soundness
2

### Presentation
2

### Contribution
2
