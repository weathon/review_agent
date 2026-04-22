# CoT-PL: Visual Chain-of-Thought Reasoning Meets Pseudo-Labeling for Open-Vocabulary Object Detection

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Open-vocabulary object detection (OVD) seeks to recognize and localize object categories beyond those seen during training. Recent approaches typically leverage vision-language models (VLMs) to generate pseudo-labels using image-text alignment, allowing detectors to generalize to unseen classes without explicit supervision. However, these methods depend heavily on direct image–text matching, neglecting the intermediate reasoning steps essential for interpreting semantically complex scenes. This results in limited robustness when confronted with crowded or occluded visual contexts. In this paper, we introduce CoT-PL, a new framework that employs structured visual chain-of-thought (CoT) reasoning into the pseudo-labeling process. CoT-PL decomposes object understanding into three interpretable steps: (1) region perception even for unseen objects, (2) category recognition via zero-shot reasoning, and (3) background grounding to separate semantically complex objects. Crucially, the third step naturally motivates our contrastive background learning (CBL) that uses the pre-computed background cues as negatives to promote feature disentanglement between objects and background. In this way, CoT reasoning and CBL form an integrated pipeline tailored to robust pseudo-labeling in crowded or occluded scenes. Notably, in these two settings, our novel-class pseudo-label quality achieves relative improvements of 103.4% and 168.4% over the best prior, respectively. Our extensive experiments demonstrate that CoT-PL achieves +7.7 AP50 on open-vocabulary COCO and +2.9 mask AP on LVIS for novel classes, setting a new state of the art.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces CoT-PL, a novel framework that integrates visual chain-of-thought (CoT) reasoning into the pseudo-labeling process for open-vocabulary object detection (OVD). The method decomposes object understanding into three interpretable steps: region perception using SAM, zero-shot category recognition via MLLMs, and background grounding. It further proposes contrastive background learning (CBL) to disentangle object and background features. The authors demonstrate state-of-the-art performance on OV-COCO and OV-LVIS, with significant gains in challenging crowded and occluded scenes.

### Strengths
The idea of reformulating pseudo-labeling as a multi-step visual reasoning process is innovative and well-motivated. It effectively addresses key limitations of existing VLM-based methods (noisy boxes, caption dependency, background collapse).

The method achieves impressive gains on standard OVD benchmarks (+7.7 AP₅₀ on OV-COCO, +2.9 mask AP on LVIS for novel classes), with particularly large improvements in crowded and occluded settings.

The paper includes thorough ablation studies (e.g., CoT steps, MLLM variants, preprocessing strategies) that clearly demonstrate the contribution of each component.

### Weaknesses
Dependence on MLLM Quality: The method’s performance is tied to the zero-shot reasoning ability of the MLLM. Weaker models lead to lower-quality pseudo-labels, which may limit generalizability.

Computational Cost: The reliance on SAM and large MLLMs (e.g., Qwen2-7B) for pseudo-labeling makes the training pipeline computationally expensive and inference time relatively longer.

### Questions
CoT-PL relies on MLLM for multimodal reasoning, which could significantly increases its runtime compared to the baseline methods. To clarify the practical cost of CoT-PL, could you provide metrics on the pseudo-labeling speed (e.g., images/second) and the total GPU hours required to generate labels for a dataset like COCO?

Could the method be adapted to use smaller or more efficient MLLMs without significant performance drop? Have you explored model distillation or lightweight reasoning modules?

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
This paper proposes a pipeline (which is called a chain of thought in this paper) to utilize large vision-language models to produce pseudo labels for training open-vocabulary object detectors. The pipeline consists of three steps: first, generate object proposals by SAM and validate by MLLMs; second, generate free-form classnames for proposals containing an object; third, separate objects from background to encourage object–background disentanglement in training. The proposed pseudo-labeling method mitigates the limitations of region matching from CLIP and some missing objects in coarse captions. Training with improved pseudo labels, the model outperforms the baseline.

### Strengths
1. The proposed pseudo-labeling method can generate pseudo labels for all objects in the image, greatly eliminating the missing annotations.
2. The proposed pseudo-labeling method undergoes a rigorous pipeline to validate the quality of proposals, ensuring proposals contain the whole objects.
3. The proposed method takes background collapse into account, encouraging better object–background disentanglement.

### Weaknesses
1. There are also many improved pseudo-labeling methods. Comparing with them is necessary. For example:
- To tackle the **noisy pseudo boxes** problem, we can use a well-trained / self-trained open-vocabulary object detector as the labeler instead of the CLIP.
- To tackle the **caption dependency** problem, we can use an MLLM to rewrite the caption to get a more detailed one.
- Moreover, [1] also proposes a pseudo-labeling method using captions and achieves superior performance.

2. Comparing with generating pseudo labels from captions, the proposed method, which generates free-from classnames, has some problems that should be taken into account:
- Captions are annotated by humans, and the nouns in captions truly exist in the image. While MLLMs may have hallucinations. How to mitigate these noises.
- The nouns in the caption are a closed set. For example, all the people in the single image will be annotated as the same label 'person'. But the proposed method generates free-form labels and may generate 'person' and 'man' for different instances of the same class. How to handle the problem caused by synonyms and superclasses?

3. There are also many MLLMs with great region perception ability [2, 3]. Generating pseudo labels with them may get better performance and may eliminate preprocessing steps.
4. As shown in Table 6, filtering noisy and uncertain predictions will get higher performance. But why is the filtering not applied to OV-LVIS (threshold is set to 1 as shown in Line 924) ?



[1] A Hierarchical Semantic Distillation Framework for Open-Vocabulary Object Detection. In TMM 2025.

[2] Describe Anything: Detailed Localized Image and Video Captioning. In ICCV 2025.

[3] The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World. In ICLR 2024.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
3

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
This paper proposes CoT-PL, a new framework for pseudo-labeling in open-vocabulary object detection (OVD).
Unlike prior methods that rely on direct image–text alignment using vision-language models (VLMs), CoT-PL formulates pseudo-label generation as a multi-step visual reasoning process—a “chain-of-thought” approach for visual understanding.

The framework explicitly tackles three key weaknesses of current OVD pseudo-labeling methods:

1. Noisy pseudo boxes, caused by co-occurrence bias in image-level VLM supervision.

2. Caption dependency, where objects missing from captions remain unlabeled.

3. Background collapse, where occluded or unlabeled instances are incorrectly learned as background.

To address these, CoT-PL integrates three reasoning stages:

1. Region Perception: SAM generates candidate masks, and an MLLM verifies object existence to remove spurious boxes.

2. Category Recognition: A zero-shot reasoning module assigns labels to each region without relying on captions.

3. Background Grounding: Contrastive Background Learning (CBL) separates unlabeled background from true objects by using grounded background cues as negative training signals.

The method achieves consistent improvements on OV-COCO and OV-LVIS benchmarks, producing higher-quality pseudo labels and more robust detectors.

### Strengths
- Clear problem identification: The paper clearly articulates three major weaknesses of existing OVD pseudo-labeling pipelines (noisy boxes, caption dependency, background collapse) and provides a coherent reasoning-based solution.

- Conceptual originality: Reinterpreting pseudo-labeling as a structured reasoning process is both novel and well motivated.

- Unified and interpretable design: The integration of SAM, MLLM reasoning, and contrastive background learning forms a consistent, interpretable pipeline that can be easily understood and reproduced.

- Strong empirical results: The proposed approach improves pseudo-label quality and achieves competitive or superior performance under complex, occluded conditions.

### Weaknesses
- Limited comparison with latest baselines: The paper does not include direct comparisons with very recent (2024–2025) state-of-the-art OVD models, which makes it difficult to fully gauge its competitiveness.

- Backbone limitation: Most experiments rely on ResNet-50, which may not capture the performance trends of newer backbones (e.g., ViT, Swin).

- Efficiency and scalability: The computational cost of the multi-step reasoning pipeline (especially SAM and MLLM inference) is not analyzed, raising concerns about scalability to large-scale or real-time settings.

- Dataset scope: Evaluation is limited to COCO and LVIS. Broader tests on real-world or open-set data would strengthen claims about generalization.

### Questions
- How does CoT-PL perform when compared directly with the most recent SOTA OVD models?

- What is the runtime overhead introduced by the SAM + MLLM reasoning process? Could the authors discuss the computational trade-offs?

- How sensitive is CoT-PL to the choice and scale of the multimodal language model used?

- Could the chain-of-thought reasoning be extended to temporal or video-based object detection tasks?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a three-step pipeline—comprising pseudo-box generation, pseudo-label assignment, and background extraction—to improve pseudo-label quality for open-vocabulary object detection. The method demonstrates strong results on the OV-COCO and OV-LVIS benchmarks.

### Strengths
The paper is well-organized and easy to follow. 

The proposed three-step pipeline is reasonable and shows potential for mitigating the long-tail problem in object detection.

The proposed method demonstrates strong performance on the OV-COCO and OV-LVIS benchmarks.

### Weaknesses
A fundamental question is: what is the difference between CoT based pseudo-boundary box generation and CoT-based bounding box prediction? If the goal is to use pseudo-annotations to train a more efficient model, then the long-tail issue of partial and small objects still exists.

Although the authors emphasize the use of CoT, the method functions more like a fixed, three-step pipeline than a dynamic reasoning process. A key limitation is its failure to reason about the contextual relationships between objects in a scene. For instance, instead of inferring that the partial object is a "person" because the object is on a "skateboard" (as seen in Figure 2a), the pseudo-label assignment module appears to process each object independently.

Consequently, the work is perceived more as a clever system optimization built upon existing methods, rather than a deep investigation into how fine-grained visual reasoning, such as Chain-of-Thought, can advance the field of open-vocabulary object detection.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
