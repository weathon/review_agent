# Redefining Generalization in Visual Domains: A Two-Axis Framework for Fake Image Detection with FusionDetect

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
The rapid development of generative models has made it increasingly crucial to develop detectors that can reliably detect synthetic images. Although most of the work has now focused on cross-generator generalization, we argue that this viewpoint is too limited. Detecting synthetic images involves another equally important challenge: generalization across visual domains. To bridge this gap, we present the OmniGen Benchmark. This comprehensive evaluation dataset incorporates 12 state-of-the-art generators, providing a more realistic way of evaluating detector performance under realistic conditions. In addition, we introduce a new method, FusionDetect, aimed at addressing both vectors of generalization. FusionDetect draws on the benefits of two frozen foundation models: CLIP & Dinov2. By deriving features from both complementary models, we develop a cohesive feature space that naturally adapts to changes in both the content and design of the generator. Our extensive experiments demonstrate that FusionDetect not only delivers a new state-of-the-art, which is 3.87% more accurate than its closest competitor and 6.13% more precise on average on established benchmarks, but also achieves a 4.48% increase in accuracy on OmniGen, along with exceptional robustness to common image perturbations. We introduce not only a top-performing detector, but also a new benchmark and framework for furthering universal AI image detection.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper re-defines “generalization” for fake-image detection as a two-axis problem: detectors must perform well on both unseen generators and unseen visual domains. To evaluate this rigorously, the authors introduce OmniGen, an open-source benchmark that unites 12 state-of-the-art text-to-image models. They further propose FusionDetect, a lightweight yet powerful detector that concatenates frozen CLIP and DINOv2 features. Extensive experiments show that FusionDetect surpasses 8 recent competitors by 3.87\% average accuracy on legacy datasets and by 4.48\% on OmniGen. FusionDetect remains robust under common corruptions such as JPEG compression and Gaussian blur, dramatically outperforming artifact-based rivals. Moreover, it displays the smallest cross-domain volatility, underscoring its superior adaptability to unseen generators and visual domains.

### Strengths
1. This paper proposes a previously overlooked challenge and casts it through a two-axis formulation that jointly considers generator shift and semantic-domain shift. 

2. This paper introduce a novel approach, FusionDetect, which achieves state-of-the-art performance.

3. A curated image dataset encompassing state-of-the-art generators is presented, providing a catalyst for continued progress in the field.

### Weaknesses
1. The paper fails to establish that generalization across visual domains is actually challenging. The proposed method performs very well on OmniGen, suggesting that the task is not hard for a trained model. Current synthetic datasets (such as WildFake [1] and DiffusionForensics [2]) are already semantically diverse enough that modern detectors rarely over-fit to specific objects. Moreover, the domain shift typically manifests as a move from natural photographs to paintings, oil canvases, or pixel art, none of which are represented in Figure 4. So I think the proposed dataset fails to demonstrate a meaningful train-test gap. 

2. The method’s novelty is quite limited: it merely concatenates features from two off-the-shelf backbones. Would performance climb further if additional encoders—BLIP, SigLIP, or others—were fused in the same way?

3. The CLIP and DINOv2 features shown in Figure 3 appear to be used without any MLP training; comparing these raw embeddings with the fused, task-tuned representation is therefore unfair.

[1] Yan et al. Wildfake: A large-scale and hierarchical dataset for ai-generated images detection. AAAI 2025. 

[2] Jeongsoo et al. Community forensics: Using thousands of generators to train
fake image detectors. CVPR 2025.

### Questions
Question: 

As described in Weakness.
 
Suggestions:
1. The title is somewhat misleading: “A TWO-AXIS FRAMEWORK” suggests that the model produces separate outputs for the two axes, whereas it in fact delivers only a single binary prediction.
2. Additional experiments that quantify how differences in visual domains influence generator training should be provided to substantiate the paper’s core claim.
3. The layout of Figure 3 is a disaster; the plots should be merged into a single row, with labels positioned directly beneath them.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a new benchmark that includes 12 state-of-the-art generators to comprehensively evaluate detectors under realistic conditions. Besides, they introduce a new detection method by incorporating two frozen foundation models to achieve a generalizable and robust detection.

### Strengths
This paper constructs a diverse evaluation dataset and introduces a detection method.

### Weaknesses
1. Figure 2, as the motivational illustration, does not align with the description of the motivation. Specifically, the figure does not clearly distinguish between the absence of cross-generator overlap and cross-semantic overlap, making it difficult to visually demonstrate how the detector struggles to generalize across these two variables, namely the generator and semantics.
2. The paper identifies the issue that the detector struggles to generalize between the generator and semantics. However, the experimental section does not demonstrate the relationship between the evaluation datasets and these two variables, making it impossible to substantiate the effectiveness of the detection method with respect to both generator and semantics. For example, the results in Table 1 only show the performance of different methods across three distinct test sets, but how these datasets relate to the generator and semantics variables remains unclear.
3. Some of the methods in Tables 1 and 2 use their pre-trained weights, which makes the comparison unfair. It is likely that the higher generalization performance of these methods is due to the more diverse training data used in the paper. To ensure a fair comparison, all methods should be trained on the same unified training set.
4. The paper uses images from the Unsplash platform as real images for the test set and generates fake images based on these. However, how can it be guaranteed that the images from this platform are indeed real? There seems to be no assurance that the images from this platform are authentic.

### Questions
Please see the Weaknesses

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
The paper introduces a two-axis generalization framework for AI-generated image detection (generalization to unseen generators and unseen types of visual content). The authors show that combining CLIP and DINOv2 features is an effective method of detecting AI-generated images. They also present the OmniGen benchmark, which contains 12 modern generators and over 400 different subjects of generation.

### Strengths
The paper demonstrates that combining CLIP (for semantic feature) and DINOv2 (for structural features) can create a robust AI image detector with adequate ablations that show the contributions of each of these feature extractors. The authors trained the detector only on SD1.4 and SD2.1 and showed the model generalizes to unseen generators, which is a strong indicator of generalizability.

### Weaknesses
One of the primary contributions of the paper is the two-axis generalization framework (generalization to unseen generator and unseen visual content), which leads to the proposal of the OmniGen benchmark. However, the description of the OmniGen benchmark lacks sufficient detail about its unseen visual content. While the authors mention using a prompt template with “over 400 subjects”, they do not provide a categorical breakdown of these subjects or other variables, which makes it difficult to assess its semantic diversity over existing datasets. 

Moreover, it is important to strongly establish the problem of generalizing to unseen semantic content. Further information or citation is required to establish that a detector trained on any existing dataset’s performance noticeably degrades on a single generator across multiple unseen different visual content.

### Questions
1. Could you provide more details on the semantic diversity of the OmniGen Benchmark?
2. Can you provide more direct proof to establish the problem of generalizing to unseen semantic content?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper argues that prior work on AI-generated image detection focuses too narrowly on cross-generator generalization and introduces a “two-axis” framework that adds cross-semantic (visual) generalization. It presents *FusionDetect*, a detector that concatenates frozen CLIP and DINOv2 embeddings and trains an MLP classifier, along with a new benchmark (*OmniGen*) containing twelve modern generators.

### Strengths
- The empirical evaluation is extensive, covering multiple established benchmarks plus a new, large benchmark (OmniGen).  
- The method is simple, computationally efficient, and achieves strong quantitative results across generators.

### Weaknesses
1. **Conceptual framing** — The “two-axis generalization” claim overstates novelty.  
   Cross-class or cross-concept evaluation has already been studied (e.g., *Improving Synthetically Generated Image Detection in Cross-Concept Settings*, Dogoulis et al., 2023; *Breaking Semantic Artifacts for Generalized AIGI Detection*, Zheng et al., NeurIPS 2024). These works explicitly examine semantic or scene-level generalization and propose stronger methodological responses than simple dataset aggregation.

2. **Feature-fusion originality** — Combining CLIP and DINOv2 is now mainstream. “Eyes Wide Shut?” (Tong et al., CVPR 2024)  already fuse or compare these backbones. Without a novel fusion mechanism or clear ablation versus stronger CLIP-only or DINO-only heads, the contribution is largely incremental.

3. **Evaluation framing** — Cross-generator generalization is the field’s central challenge with real-world implications (social media, Sora-style video, closed-source APIs). Cross-semantic tests are useful diagnostics but do not justify a redefinition of “generalization.” The paper’s positioning should be moderated to reflect that this is a *comprehensive evaluation* rather than a conceptual overhaul.

### Questions
1. How this work is different earlier studies on cross-concept / class?
2. Why it's important? Given social media mostly we're concerned about cross-generator.

### Soundness
2

### Presentation
2

### Contribution
1
