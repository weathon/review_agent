# No time to train! Training-Free Reference-Based Instance Segmentation

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 4, 6

## Abstract
The performance of image segmentation models has historically been constrained by the high cost of collecting large-scale annnotated data. The Segment Anything Model (SAM) alleviates this original problem through a promptable, semantics-agnostic, segmentation paradigm and yet still requires manual visual-prompts or complex domain-dependent prompt-generation rules to process a new image. Towards reducing this new burden, our work investigates the task of object segmentation when provided with, alternatively, only a small set of reference images. Our key insight is to leverage strong semantic priors, as learned by foundation models, to identify corresponding regions between a reference and a target image. We find that correspondences enable automatic generation of instance-level segmentation masks for downstream tasks and instantiate our ideas via a multi-stage, training-free method incorporating (1) memory bank construction; (2) representation aggregation and (3) semantic-aware feature matching. Our experiments show significant improvements on segmentation metrics, leading to state-of-the-art performance on COCO FSOD (36.8% nAP), PASCAL VOC Few-Shot (71.2% nAP50) and outperforming existing training-free approaches on the Cross-Domain FSOD benchmark (22.4% nAP).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a training-free method for reference-based instance segmentation by leveraging two powerful pretrained vision foundation models: SAMv2 for segmentation mask proposals and DINOv2 for semantic feature extraction. The method includes a three-stage process: (1) memory bank construction from reference images; (2) feature aggregation into class prototypes; and (3) feature matching with a semantic-aware soft merging strategy for mask selection. The framework is simple, scalable, and effective, achieving state-of-the-art performance on multiple few-shot detection/segmentation benchmarks without any additional training.

### Strengths
1.The paper is well-written and clearly structured. The motivation is timely and compelling—reducing annotation and training cost is highly relevant in the era of foundation models.

2. The method is elegant and efficient, with minimal overhead and no need for finetuning, making it broadly applicable to low-resource or rapid-deployment scenarios.

3. The method performs competitively (or even better) than fine-tuned approaches on COCO-FSOD, PASCAL-FSOD, and CD-FSOD, with good cross-domain generalization.

4. Simple but effective use of DINOv2 + SAM: The paper demonstrates how powerful pretrained models can be combined in a modular, reproducible way to tackle challenging tasks like instance segmentation.

### Weaknesses
1. Limited discussion of some generalist model such as DINO-X, SINE, and T-REX:
These methods seems to also support reference-based Instance Segmentation. The current paper does not systematically analyze how its method compares in terms of design, generalization, or efficiency.

2.Evaluation scope: results emphasize benchmarks; no demonstration on real-world deployment or interactive scenarios.

### Questions
1.Have you tested the approach using different foundation encoders (e.g., CLIP,MAE,DINOV3 vs DINOv2)? Is the method sensitive to the embedding geometry?

2.How does the method handle reference ambiguity—for example, when multiple objects of the same class vary significantly in appearance or scale?

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
Image segmentation faces high annotated data costs; SAM eases this via promptable segmentation but needs manual prompts/domain-specific rules, while existing reference-based methods require fine-tuning or costly metrics. The paper proposes a training-free three-stage framework fusing SAM (mask generation) and DINOv2 (semantics): building a reference feature memory bank, two-step feature aggregation, and cosine similarity matching with semantic-aware soft merging. It achieves SOTA on COCO-FSOD (36.8% nAP), PASCAL VOC Few-Shot (71.2% nAP50), and outperforms training-free methods on CD-FSOD (22.4% nAP).

### Strengths
1. Practical Significance: The training-free paradigm reduces deployment costs for low-annotation domains (e.g., underwater, microscopic imaging), aligning with real-world needs for fast adaptation.
2. Efficiency-Accuracy Balance: It outperforms prior methods (e.g., Matcher) on accuracy and runs ~129x faster (0.929s/img vs. 120.014s/img), striking a rare balance.
3. Clarity & Reproducibility: The three-stage framework is visualized clearly (Fig.2), with detailed implementation details (resolution, IoU threshold) and standard experiments enabling reproducibility.
4. Cross-Domain Robustness: It works across diverse domains (cartoon, aerial) with fixed hyperparameters, expanding application scope.

### Weaknesses
1. Limited Originality: The method combines off-the-shelf models (SAM/DINOv2) with classic techniques (cosine similarity, soft merging)—no novel methodology or theoretical insights, making it an engineering implementation rather than an innovation.
2. Incomplete Experiments: No ablation for two-step aggregation (e.g., instance-only vs. class-only prototypes) or memory bank design; no comparison with mainstream “VLM+SAM” pipelines, weakening competitiveness arguments.
3. Shallow Analysis: Failure cases (similar-class confusion) lack root-cause attribution (e.g., DINOv2’s bias); visualizations (Fig.3/4) only show predictions, no core process (feature similarity) visuals.
4. Weak Prior Work Context: Only compares performance with prior methods (e.g., Matcher) but no deep logic contrast (e.g., why cosine similarity outperforms Earth Mover’s Distance)

### Questions
1. Could you add ablation for two-step aggregation (instance-only/class-only vs. two-step) and memory bank design (e.g., CLIP vs. DINOv2) to verify module necessity?
2. Why not compare with “VLM (Qwen-VL)+SAM” pipelines? Please supplement CD-FSOD/COCO-FSOD experiments to show your method’s advantages.
3. For similar-class confusion, could you visualize DINOv2’s feature embeddings (t-SNE) to check semantic overlap? Can adjusting prototype construction mitigate this?
3. How does the method perform with low-quality reference images (blurred/occluded)? Please provide model size and GPU memory for edge deployment

### Soundness
3

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
Existing methods—such as frameworks like SAM—generate masks that lack semantic awareness. They require manual intervention or complex prompt-generation pipelines, which creates limitations in automated processing and cross-domain scenarios. Analyses show that these existing methods need fine-tuning for novel categories, which gives rise to issues including task-specific data requirements, overfitting, and domain shift. Additionally, methods that integrate pre-trained models suffer from drawbacks in computational efficiency and instance-level segmentation capabilities.
To address these challenges, this paper proposes constructing a category-specific feature memory bank, refining feature representations through two-step aggregation, and performing feature matching with a semantic-aware soft merging strategy.

### Strengths
1、This method not only addresses the automation challenges of frameworks like SAM—such as their "lack of semantic awareness and need for manual intervention"—but also avoids issues like overfitting and domain shift caused by the "requirement for fine-tuning on novel categories" in traditional methods. Its effectiveness has been verified through laboratory experiments.

2、The paper achieves better optimization for scenarios (e.g., camouflaged objects) that existing vision foundation models—such as SAM (designed for segmenting everything) and DINO-V2 or CLIP (used for vision-language alignment), as illustrated in Figure 1—struggle to handle. This is beneficial for the expansion of existing methods in vertical domains.

### Weaknesses
1、A key limitation of traditional DINO/CLIP-based frameworks for training-free open-vocabulary semantic segmentation (OVSeg) lies in their requirement for a predefined category list during evaluation—this prevents them from being classified as genuine open-vocabulary methods. By comparison, generative vision-language model (VLM)-based methods possess intrinsic properties that make them more adept at realizing open-domain perception. Please analyze and compare the strengths of the aforementioned method (the one in question) and VLM-based approaches.

2、In the ablation study, the "Variance in Reference Set" experiment demonstrates that different reference images lead to performance variations, and increasing the number of shots (shot count) can reduce such deviations. I believe this is because the model requires certain typical, high-value references to enhance its generalization ability, while a larger shot count is more likely to improve the diversity of reference images. Is there any attempt to achieve the highest possible performance using the fewest possible high-value reference images? Alternatively, could a guiding method be proposed for selecting reference images?

3、Has there been any attempt to replace the segmenter (SAM-based) and the model used for classification (DINOv2)? This would demonstrate that the method can be plug-and-play replaced and generalized when better foundation models emerge in the future.

### Questions
See weakness part.

### Soundness
3

### Presentation
2

### Contribution
2
