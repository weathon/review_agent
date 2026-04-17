# Improving Visual Discriminability of CLIP for Training-Free Open-Vocabulary Semantic Segmentation

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 4, 2

## Abstract
Extending CLIP models to semantic segmentation remains challenging due to the misalignment between their image-level pre-training objectives and the pixel-level visual understanding required for dense prediction. While prior efforts have achieved encouraging results by reorganizing the final layer and features, they often inherit the global alignment bias of preceding layers, leading to suboptimal segmentation performance. In this work, we propose LHT-CLIP, a novel training-free framework that systematically exploits the visual discriminability of CLIP across layer, head, and token levels. Through comprehensive analysis, we reveal three key insights: (i) the final layers primarily strengthen image–text alignment with sacrifice of visual discriminability (e.g., last 3 layers in ViT-B/16 and 8 layers in ViT-L/14), partly due to the emergence of anomalous tokens; (ii) a subset of attention heads (e.g., 10 out of 144 in ViT-B/16) display consistently strong visual discriminability across datasets; (iii) abnormal tokens display sparse and consistent activation pattern compared to normal tokens. Based on these findings, we propose three complementary techniques: semantic-spatial reweighting, selective head enhancement, and abnormal token replacement to effectively restore visual discriminability and improve segmentation performance without any additional training, auxiliary pre-trained networks, or extensive hyperparameter tuning. Extensive experiments on 8 common semantic segmentation benchmarks demonstrate that LHT-CLIP achieves state-of-the-art performance across diverse scenarios, highlighting its effectiveness and practicality for real-world deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes LHT-CLIP, a training-free framework designed to adapt CLIP models for open-vocabulary semantic segmentation. The core problem it addresses is the misalignment between CLIP's image-level pre-training and the pixel-level understanding needed for segmentation, specifically the poor visual discriminability in the final layers.

### Strengths
The combined LHT-CLIP module consistently improves the performance of existing training-free methods (like SCLIP, ClearCLIP, and ResCLIP) and achieves better results on eight semantic segmentation benchmarks.

### Weaknesses
### Major

**1. Limited Novelty**

The core problems addressed in this paper (performance degradation in final layers, discriminability of specific attention heads, and the emergence of "abnormal tokens") have been previously identified and studied in prior work. It appears the authors have combined these known issues and applied relatively simple or existing techniques to address them. Consequently, the paper's novelty seems limited.

1) The observation that final layers are suboptimal for dense prediction is not new. This has been identified by several prior works, including Perception Encoder [1], CorrCLIP [2], Cascade-CLIP [3], and CLIPer [4]. These works have also explored multi-layer fusion strategies.

2) The finding that a subset of attention heads displays strong visual discriminability is acknowledged by the authors as being derived from recent work and is not an original contribution. Furthermore, this phenomenon was also studied in [5], which is not discussed or cited by the authors.

3) The issue of "abnormal tokens" has also been well-discussed in previous works [6, 7]. In particular, the analysis in [7] appears very similar to the one in this paper, yet the authors provide no comparison or discussion of this related work.

**2. Lack of Generality Experiments**
While the authors report performance on ViT-B based models, results on ViT-L or ViT-H based models and stronger baselines [8, 9] appear to be missing. For a plug-in module, this omission weakens the claims about the method's generality.

**3. Inaccurate State-of-the-Art (SOTA) Claim**
The authors claim to achieve SOTA performance. However, existing methods [2, 8, 9] appear to outperform LHT-CLIP by a large margin, which contradicts this claim.


## Minor

1. The PRELIMINARIES subsection is overly long. Basic concepts could be introduced more briefly to save space.
2. The paper lacks a dedicated Related Work section. While some related literature is mentioned, a consolidated section would improve the paper's context and clarify its position relative to existing methods.

[1]. Perception Encoder: The best visual embeddings are not at the output of the network, 2025

[2]. CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation, 2024

[3]. Cascade-clip: Cascaded vision-language embeddings alignment for zero-shot semantic segmentation, 2024

[4]. Cliper: Hierarchically improving spatial representation of clip for open-vocabulary semantic segmentation, 2024

[5]. INTERPRETING CLIP’S IMAGE REPRESENTATION VIA TEXT-BASED DECOMPOSITION, 2023

[6]. Explore the Potential of CLIP for Training-Free Open Vocabulary Semantic Segmentation, 2024

[7]. SEE WHAT YOU ARE TOLD: VISUAL ATTENTION SINK IN LARGE MULTIMODAL MODELS, 2025

[8]. Harnessing vision foundation models for high-performance, training-free open vocabulary segmentation, 2025

[9]. Proxyclip: Proxy attention improves clip for open-vocabulary segmentation, 2024

### Questions
Please check the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work provides an in-depth analysis of the internal mechanisms of the CLIP model, identifying three key characteristics. **Existence of anomalous tokens**: The authors find that anomalous tokens emerge in the intermediate layers of CLIP. These tokens exhibit high similarity to all visual tokens, thereby capturing global contextual information, and their spatial positions remain fixed in subsequent layers. **Decay in visual discriminability**: In the final layers of CLIP, the visual discriminability of tokens experiences a sharp decline, despite marginal gains in semantic alignment. **Visual properties of specific attention heads**: The authors observe that the outputs of certain attention heads retain strong visual discriminability.

Based on these observations, this work proposes three targeted optimization strategies. **Anomalous Token Replacement (ATR)** eliminates the negative impact of anomalous tokens by replacing them with the mean of their surrounding tokens. **Spatial-Semantic Reweighting (SSR)** mitigates the decay of visual discriminability by reweighting the residual connections in the final layers. **Selective Head Enhancement (SHE)** refines the final layer's features by leveraging a similarity matrix computed from the outputs of select heads that demonstrate strong visual discriminability.

### Strengths
- **In-depth analysis**. This work presents a thorough and convincing quantitative and qualitative analysis for each of the three findings.
- **Coherent design**. The proposed components (ATR, SSR, and SHE) directly correspond to the three key observations, showcasing a well-reasoned and targeted design.
- **Comprehensive experiments**. The proposed method is supported by extensive and detailed experiments that rigorously validate the effectiveness of each module, providing strong evidence for the authors' claims.

### Weaknesses
- The obervation of anomalous tokens and ATR share similarities with prior work, such as CLIPTrase[1] and SC-CLIP[2]. It would further strengthen the paper if the authors could clearly discuss the conceptual differences from and advantages over these existing approaches 
- The experiments are primarily focused on methods that use a pure CLIP. However, many state-of-the-art methods achieve superior performance by integrating other visual foundation models and these methods also make use of the final layer of CLIP. Therefore, the generalizability of the proposed techniques remains to be validated on more advanced hybrid architectures, such as ProxyCLIP[3], Trident[4], and CorrCLIP[5].

[1]"Explore the potential of clip for training-free open vocabulary semantic segmentation." ECCV 2024.

[2]"Self-calibrated clip for training-free open-vocabulary segmentation." arXiv preprint arXiv:2411.15869.

[3]"Proxyclip: Proxy attention improves clip for open-vocabulary segmentation." ECCV 2024.

[4]"Harnessing vision foundation models for high-performance, training-free open vocabulary segmentation." ICCV 2025.

[5]"CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation." ICCV 2025.

### Questions
Was the proposed operation applied to the attention pooling layer of SigLIP? Given SigLIP's architecture, applying it at the attention pooling layer might be a more effective alternative.

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
4

### Summary
The paper introduces TLH-CLIP, a training-free framework that adapts CLIP for open-vocabulary semantic segmentation by leveraging spatial cues across Token, Layer, and Head levels. The authors find that: (1) some final-layer tokens degrade spatial clarity, (2) later layers favor global alignment over local detail, and (3) only a few attention heads provide strong spatial signals. Based on this, they propose three techniques—abnormal token replacement, semantic-spatial reweighting, and selective head enhancement—to improve segmentation without additional training. TLH-CLIP achieves state-of-the-art results on 8 benchmarks.

### Strengths
- The proposed techniques are conceptually simple yet provide novel and valuable insights into leveraging CLIP’s internal structures—such as tokens, layers, and attention heads—for dense prediction tasks.
- The paper presents a fresh and in-depth examination of CLIP’s internal mechanisms, revealing that its final layers tend to emphasize semantic alignment over spatial detail, among other insightful observations.
- The manuscript is clearly written and well-organized, with logically structured sections and comprehensive experimental validation that effectively supports its conclusions.
- TLH-CLIP demonstrates notable and consistent performance gains over strong baseline methods across a diverse set of benchmark datasets.

### Weaknesses
- While the authors claim that the proposed method improves segmentation performance without extensive hyperparameter tuning, it in fact requires tuning four hyperparameters, as indicated in Tables 2, 3, and 4.
- The results highlighted in light gray in Tables 2–4 are described as reflecting the optimal settings; however, these values differ across tables (e.g., 28.0, 28.1, and 28.4 mIoU), which may cause confusion. A more appropriate hyperparameter sensitivity analysis should include results obtained with default parameters, allowing readers to clearly observe the performance gap between the default and optimal settings. The authors are encouraged to clarify whether the reported results correspond to default or tuned configurations and to explain the observed discrepancies.
- In addition, the ablation studies in Tables 2, 3, and 4 show that the performance improvements under the optimal settings are relatively minor (< 1.0 mIoU), which may raise questions about the overall significance of the proposed method.

### Questions
- Issues mentioned in the weaknesses section.
- How is the performance gain of this method over the CLIP baseline?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the poor performance of CLIP models on pixel-level semantic segmentation, which stems from their image-level pre-training. The authors analyze CLIP's vision encoder and find that final layers sacrifice "visual discriminability" (pixel-level distinction) for "semantic alignment" (image-text matching). They propose LHT-CLIP, a training-free framework that modifies the inference pipeline with three components: Abnormal Token Replacement (ATR), Spatial-Semantic Reweighting (SSR) to favor earlier layers' features, and Selective Head Enhancement (SHE) to aggregate features from known discriminative attention heads. These modules are applied as a plug-and-play enhancement to existing methods, consistently improving state-of-the-art performance on multiple segmentation benchmarks.

### Strengths
1. This framework is designed as a set of plug-and-play modules (ATR, SSR, SHE) that can be applied to other existing training-free methods (like SCLIP, ClearCLIP, and ResCLIP).
2. The authors also extend experiments to SigLIP, suggesting transferability of the proposed approaches.
3. The method requires no gradient descent, extra trainable modules, or large-scale retraining. This is a significant advantage over weakly-supervised or adapter-based methods, which require computational and annotation resources.

### Weaknesses
1. **Hand-crafted heuristics**: The framework feels like a collection of disparate, heavily-tuned heuristics rather than a fundamental and unified solution: ATR identifies tokens by a specific sparsity threshold; SSR requires manually identifying a model-specific range of layers; SHE requires a pre-computed list of "good" heads identified by averaging discriminability scores across multiple datasets and more operations. The pipeline is complex and sensitive to hyperparameters.
2. **Misleading claim**: The paper positions itself in the "training-free" category. However, the SHE component is actually data-driven. The authors state that all hyperparameters, including the selection of the top-k discriminative heads, are "tuned on 1,000 randomly sampled training images from {Context, ADE, Stuff} datasets". While this isn't "training" via backpropagation, it is a data-dependent tuning process on held-out data. This makes the "training-free" claim misleading; the method is not "zero-shot", as it requires access to a representative sample of the target datasets.
3. **Incomprehensive literature review**: This paper claims itself "the first work to explicitly modify the inference procedure prior to the final layer". This is inappropriate. CAT-Seg [1] and even earlier works have pointed out that features before pooling convey better semantics; Perception Encoder [2] is cited but only discussed in Appendix. The authors should discuss such works as a "Related Work" section in the main paper.
4. **The contribution is incremental**: The problem that this paper tackles is somewhat obsolete. The paper focuses on adapting an old foundation model for a task (semantic segmentation) that it was not designed for. While it pushes the SOTA in this specific sub-field, it feels like an incremental patch. The broader field is moving towards models explicitly designed for dense tasks (e.g., SAM) or more capable MLLMs.


[1] CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation, CVPR 2024

[2] Perception Encoder: The best visual embeddings are not at the output of the network

### Questions
This paper's bibliography section includes multiple repeating references. This does not affect my ratings but the authors should check and avoid such issues. Examples: "Segclip: Patch aggregation with learnable centers for open-vocabulary semantic segmentation", "Reco: Retrieve and co-segment for zero-shot transfer", "Learning open-vocabulary semantic segmentation models from natural language supervision".

### Soundness
2

### Presentation
1

### Contribution
2
