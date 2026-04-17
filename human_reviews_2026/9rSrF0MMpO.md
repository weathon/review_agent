# Enhancing Open-Vocabulary Object Detection through Multi-Level Fine-Grained Visual-Language Alignment

- Decision: Reject
- Scores: 4, 6, 2, 8

## Abstract
Traditional object detection systems are typically constrained to predefined categories, limiting their applicability in dynamic environments. In contrast, open-vocabulary object detection (OVD) enables the identification of objects from novel classes not present in the training set. Recent advances in visual-language modeling have led to significant progress of OVD. However, prior works face challenges in either adapting the single-scale image backbone from CLIP to the detection framework or ensuring robust visual-language alignment. We propose Visual-Language Detection (VLDet), a novel framework that revamps feature pyramid for fine-grained visual-language alignment, leading to improved OVD performance. With the VL-PUB module, VLDet effectively exploits the visual-language knowledge from CLIP and adapts the backbone for object detection through feature pyramid. In addition, we introduce the SigRPN block, which incorporates a sigmoid-based anchor-text contrastive alignment loss to improve detection of novel categories. Through extensive experiments, our approach achieves 58.7 AP for novel classes on COCO2017 and 24.8 AP on LVIS, surpassing all state-of-the-art methods and achieving significant improvements of 27.6% and 6.9%, respectively. Furthermore, VLDet also demonstrates superior zero-shot performance on closed-set object detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper identifies that previous open-vocabulary object detection methods face challenges in adapting single-scale image backbones for object detection tasks. It further introduces a fine-grained visual-language alignment framework to bridge the image-region gap.

### Strengths
**Originality**:
- This paper's motivation is sound, as current open-vocabulary object detection methods have not adequately addressed the challenge of multi-scale features.

**Clarity**:
- This paper clearly explains its motivation and methodology.

**Significance**:
- This paper addresses the problem of region-text misalignment, proposes a reasonable method, and proves its effectiveness through experiments.

### Weaknesses
- The definition of "fine-grained" in this paper is unclear. If it means region-level alignment (fine-grained) compared with image-level alignment (coarse-grained), I don't think that region-level alignment is enough to be called 'fine-grained alignment'.
- In Table 1, the backbones of compared methods are different from the backbone used in this paper, which leads to an unfair comparison. Some methods that also use ViT as the backbone should be included.
- The visualization examples in this paper are very rough and can not clearly show the effect they are intended to demonstrate.

### Questions
- I'm wondering about what "fine-grained" exactly means in this paper, or how it is different than what is called "coarse-grained"?

### Soundness
4

### Presentation
2

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
This paper presents VLDet, a framework for Open-Vocabulary Object Detection (OVD) that enhances fine-grained visual-language alignment by effectively adapting CLIP encoders for detection tasks.

Previous OVD methods either use CLIP’s single-scale image encoder, which limits spatial precision, or train multi-scale backbones from scratch using pseudo-labeled data from zero-shot detectors such as GLIP, resulting in inefficiency and bias. Moreover, they emphasize region-level alignment while neglecting image-level alignment, which is also crucial for robust visual-language modeling.

The main contributions are as follows:

- Visual-Language Pyramid Upscale Block (VL-PUB) adapts CLIP’s image and text encoders into an open-world detector, leveraging multi-scale image features for improved detection accuracy.

- Sigmoid-based Region Proposal Network (SigRPN) introduces sigmoid-based visual-language contrastive learning to differentiate background from objects of any class and integrates image-wise contrastive loss for stronger alignment.

- Extensive Experiments show that VLDet achieves state-of-the-art performance on COCO2017 and LVIS, improving novel class mAP from 42.7 to 58.72 and from 23.2 to 24.83, demonstrating strong generalization to unseen categories.

Overall, the paper proposes a unified and effective approach for adapting CLIP to open-vocabulary detection by improving spatial precision, cross-modal alignment, and generalization.

### Strengths
- The paper is clearly written and logically structured, with a well-motivated problem and methodology.

- The proposed VL-PUB and SigRPN modules directly address critical limitations of prior OVD methods.

- Extensive experiments on standard benchmarks show consistent and significant performance improvements.

### Weaknesses
- The introduction could be better organized to separate the problem definition, motivation, and main contributions.

- The VL-Fuse layer employs bi-directional cross-attention, but its mechanism is not clearly explained or supported with mathematical details.

- The paper lacks comparisons with more recent OVD baselines that use stronger backbones or alignment mechanisms.

### Questions
- In the VL-Fuse Layer, why do the text features differ between the first and second fusion stages (from $l_{cls}$ and $l_{cap}$ to $l'_{cls}$), and what is the intuition behind this design choice?

- What is the rationale for subtracting the mean similarity across all object classes when computing background similarity, and how does this normalization improve detection stability?

- Can VLDet be applied to other open-vocabulary detectors or backbones beyond CLIP’s ViT architecture?

- How does the proposed sigmoid-based loss differ from previous sigmoid or softmax approaches, and why does it achieve better alignment for novel object detection?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposed three improvements to tackle the open-vocabulary object detection task. First, the Visual-Language Pyramid Upscale Block extracts multi-level feature pyramid from the single-scale image feature to detect objects at various scales. Second, the Visual-Language Region Proposal Network fuses image features with text embeddings and uses an Anchor-Text Binary Alignment Loss for classification. Third, a Mini-Batch Image Contrastive Loss is used to transfer image-level knowledge. The proposed method is evaluated on  COCO and LVIS.

### Strengths
1. The paper provides some visualization examples (Figure 1 and Figure 4) for comparison, showing the differences between various models intuitively. 
2. The paper provides detailed ablation studies in Table 3, Table 4, and Table 5, showing that each component of the proposed method is useful and the improvements are orthogonal.
3. The formulas in the paper are well-written, making the proposed method easy to understand.

### Weaknesses
1. The paper proposed three independent innovations (Visual-Language Pyramid Upscale Block, Visual-Language Region Proposal Network, and Mini-Batch Image Contrastive Loss) without a unified motivation, making the paper A + B + C.
2. Using a multi-scale feature pyramid is a common practice in object detection. I don't think it is a common problem that needs to be tackled. As ViT-based CLIP produces single-scale features, we can use ViTDet [1,2] to produce a feature pyramid.
3. Vision-language deep fusion is also a common practice in open-vocabulary object detection. Authors should compare the proposed VLFuse with related works [3,4,5,6]. I don't think the challenges mentioned from Line 164 to Line 167 are a big problem. All these methods work well with high performance.
4. The experiments in Table 1 are extremely unfair as the proposed method uses much more pretraining data (objects365) and a significantly larger backbone (ViT-Large). Authors should include other strong baselines for comparison, including but not limited to [2, 3]. Further, objects365 contains all 80 classes in COCO. Evaluating the open-vocabulary performance on COCO is meaningless.
5. The number of decimal places is inconsistent (Line94, Table 1, Table 2).





[1] Exploring Plain Vision Transformer Backbones for Object Detection. In ECCV 2022.

[2] CLIPSelf: Vision Transformer Distills Itself for Open-Vocabulary Dense Prediction. In ICLR 2024.

[3] Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. In ECCV 2024.

[4] YOLO-World: Real-Time Open-Vocabulary Object Detection. In CVPR 2024.

[5] Grounded Language-Image Pre-training. In CVPR 2022.

[6] Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone. In NeurIPS 2022.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes VLDet, a two-stage OVD framework that (i) adapts CLIP backbones into a multi-scale detector via a VL-Pyramid Upscale Block (VL-PUB) and (ii) introduces a Sigmoid-based RPN (SigRPN) trained with anchor–text binary alignment, plus mini-batch image–caption and region–text contrastive losses for multi-level alignment. On COCO and LVIS, the method reports large gains on novel categories (e.g., 58.72 AP novel on COCO; 24.83 AP novel on LVIS) after pretraining only on Objects365 with generated captions. Ablations cover freezing strategies, multi-scale vs single-scale, caption usage, and loss components.

### Strengths
1. The model achieves state-of-the-art novel-class average precision on COCO and LVIS benchmarks, given its specified pretraining data. A series of well-designed ablations—including those on captions, language-image contrastive learning, visual-language fusion, and multi-scale features—credibly demonstrate the source of its performance gains.

2. The proposed SigRPN module is an elegant design. By framing region proposal as a binary visual-language alignment task for foreground and background separation, it provides a clear and effective method for improving proposal quality for unseen object categories.

3. The paper provides sufficient implementation details for reproducibility, reporting specific choices such as batch sizes, components kept frozen during training, and comparative results against single-scale baselines.

### Weaknesses
1. The substantial requirement of 64 A100 GPUs for pretraining will likely hinder widespread adoption. Furthermore, the paper does not profile the inference cost, making it difficult to assess its practical efficiency compared to prior two-stage open-vocabulary detectors.

2. While the method of using a foundation vision-language model to generate captions for the Objects365 dataset is described, the exact model, prompts, and data cleaning procedures are not fully specified. This lack of detail complicates strict reproducibility and fair comparison with future work.

3. The comparison to some baselines, such as recent grounding-detector variants, is complicated by differences in their pretraining data. Although the paper acknowledges this, a stricter, like-for-like evaluation protocol would strengthen its claims.

### Questions
1. Please specify which captioning model and prompts were used for the Objects365 dataset, and will this generated caption set be released to ensure reproducibility?

2. Is it possible to report the inference latency and FLOPs for the VLDet-B/L models, both with and without SigRPN, and compare them to common open-vocabulary detection baselines?

3. How sensitive are the final results to the freezing of the vision-to-language module, and to the choice of the underlying CLIP model variant, when evaluated under a consistent computational budget?

### Soundness
3

### Presentation
3

### Contribution
3
