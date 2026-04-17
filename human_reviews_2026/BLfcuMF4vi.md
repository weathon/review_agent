# DiSa: Saliency-Aware Foreground-Background Disentangled Framework for Open-Vocabulary Semantic Segmentation

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Open-vocabulary semantic segmentation aims to assign labels to every pixel in an image based on text labels. State-of-the-art approaches typically utilize vision-language models (VLMs), such as CLIP, for dense prediction. However, VLMs, pre-trained on image-text pairs, are biased toward salient, object-centric regions and exhibit two critical limitations when adapted to semantic segmentation: (i) Foreground Bias, which tends to ignore background regions, and (ii) Limited Spatial Localization, resulting in blurred object boundaries. To address these limitations, we introduce DiSa, a novel saliency-aware foreground-background disentangled framework. By explicitly incorporating saliency cues in our designed Saliency-aware Disentanglement Module (SDM), DiSa separately models foreground and background ensemble features in a divide-and-conquer manner. Additionally, we propose a Hierarchical Refinement Module (HRM) that leverages pixel-wise spatial contexts and enables channel-wise feature refinement through multi-level updates. Extensive experiments on six benchmark open-vocabulary semantic segmentation datasets demonstrate that DiSa consistently outperforms current state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces DiSa, a new framework for open-vocabulary semantic segmentation . The authors try to solve two key limitations in existing Vision-Language Models (VLMs) like CLIP, Foreground Bias and Limited Spatial Localization.  The paper claims that this dual-branch, saliency-guided approach allows the model to learn specialized representations that better handle both foreground objects and background context.

### Strengths
The paper's primary strength lies in its clear and accurate motivation. The identification of "Foreground Bias" and "Limited Spatial Localization" as two distinct, critical failures of VLMs in dense prediction tasks is sharp and well-articulated. The proposed disentanglement strategy bring additional performance improvement over baseline.

### Weaknesses
The submission, in its current form, suffers from several critical weaknesses that prevent me from recommending acceptance. My primary concerns are as follows:

1.  **Fundamentally Unsound Methodology and Notations:** The paper's core contribution, the Saliency-aware Disentanglement Module (SDM), is described in a manner that is mathematically incoherent and contradictory.
    * In Section 3.2, the authors define their key tensors, the saliency map $S$ and correlation map $C$, in two conflicting ways *in the same sentence*: first as a set (e.g., $S = \{S_i, ...\}$) and immediately after as a single tensor (e.g., $S \in \mathbb{R}^{H \times W \times N_C \times D}$). This is a fundamental error.
    * The specified tensor dimensions are nonsensical. A "saliency map" or a "correlation map" (derived from cosine similarity) should be scalar-valued, with dimensions $\mathbb{R}^{H \times W \times N_C}$. The authors' inclusion of an arbitrary feature dimension $D=128$ is technically incorrect and is never justified. This error makes the central mechanism of the paper—using $S$ to filter $C$—impossible to understand or reproduce.

2.  **Overly Convoluted and "Chaotic" Pipeline:** The pipeline in Fig.2 generates correlation maps $C$, while *in parallel* generating saliency maps $S$ by "sharpening" cross-attention maps $A$ with gradients from an $\mathcal{L}{itm}$. The paper provides no clear justification for why this complex, gradient-based saliency generation (using $\partial \mathcal{L}_{itm} / \partial A$) is necessary or superior to simpler, more direct methods. The design feels arbitrary and poorly motivated.

3. **Marginal Ablation Gains:** The ablation study fails to justify the added complexity. As highlighted, row (VI) (`(IV) + Category`) shows scores (11.8, 19.1, 32.0, 57.9, 95.4, 78.3) that are only marginally better, and in some cases worse, than the main (I) Baseline (12.0, 19.0, 31.8, 57.5, 94.6, 77.3). This suggests the proposed refinement modules offer minimal practical benefit for their complexity.

4.  **Formatting Inconsistencies:** Figure 6, labeled as a "Figure," is in fact a table presenting numerical data.

### Questions
please check weakness

### Soundness
2

### Presentation
1

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
The paper introduces DiSa, a novel saliency-aware foreground-background disentangled framework for open-vocabulary segmentation. DiSa introduces a Hierarchical Refinement Module (HRM) that captures spatial context through Pixel-, Category-, and Semantic-wise Refinement. DiSa achieves performance gains over six benchmarks.

### Strengths
1. The paper is clearly written. 
2. The idea is novel and make senses to me. 
3. The modular design can easily slot into CLIP-based baselines.

### Weaknesses
1. The saliency is derived from cross-attention + Grad-CAM reweighting via an auxiliary ITM loss tied to segmentation supervision. It may introduce label leakage.
2. The performance gains are limited from Table1. 
3. The foreground/background Token Selection is a bit ambiguous. It is better to elaborate more. And is it possible to use the model to do foreground/background segmentation as well?

### Questions
1. How do you select top-k visual tokens in the correlation maps?
2. What happens if you freeze CLIP completely (no key/query projection tuning) to preserve open-vocabulary behavior?
3. It would be better if you could show some failure cases.

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
This paper effectively addresses the foreground bias in Vision-Language Models (VLMs) for open-vocabulary semantic segmentation by introducing a saliency map module to decouple foreground and background regions. The authors further propose a Hierarchical Boundary Modulation (HBM) module that refines the segmentation output at the pixel, semantic, and category levels. Their method also achieves State-of-the-Art (SOTA) performance on multiple official benchmarks.

### Strengths
1. The fundamental problem is well-identified and critical: The authors pinpoint the crucial issue of Vision-Language Models (VLMs) being inherently biased toward foreground objects in dense prediction tasks like semantic segmentation. Their proposed saliency map-based method provides an elegant and direct solution for decoupling foreground and background representations.

2. The empirical results are well-validated: The model not only achieves State-of-the-Art (SOTA) performance but demonstrates robustness and superior generalization by outperforming competitors across all official benchmarks tested.

### Weaknesses
1. The proposed module to extract foreground/background region is based GradCAM. While maintaining a comparable GFLOP count, the reliance on a gradient-based method like GradCAM may lead to a slower inference speed due to the required backward pass.

### Questions
1. Please check and correct the typo on Line 245: "re-weigh" should be reviewed for proper hyphenation or spelling (e.g., "reweight" or "re-weight").

2. Does  Variant I in Table 5 represent the performance of the DiSa model without the Saliency Boundary Module (SBM)?

3. For 'stuff' classes (e.g., wall or sky), what precisely does the saliency map capture? Is the definition of saliency in this work consistent with prior literature, or does it merely indicate a region of higher model confidence?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes DiSa, a saliency-aware method for open-vocabulary segmentation that disentangles foreground-background using saliency for decomposition and hierarchical feature refinement. The key contributions are:
- Thew saliency aware disentanglement module that uses saliency for adaptive foreground to background separation to mitigate foreground bias.
- Hierarchical Refinement Module that is used to capture detailed spatial and channel context in 3 stages (Pixel, category, Semantic) to improve limited spatial localization.
- SOTA performance across six large open vocabulary semantic segmentation benchmarks.

### Strengths
- Both the DSM and HRM are shown to effectively mitigate their issues and the ablation studies show a strong motivation for the current elements in the method.
- Good experimental results and the increase in PAS-20b indicates the results are supported by the theory.
- Good performance with no additional datasets and low computational cost (GFLOPs).

### Weaknesses
- The entire disentanglement pipeline is dependent on the quality and accuracy of the ITM loss. If it fails to localize the object for novel classes the split will be flawed.
- While GFLOPs are low the pipeline is complex which can introduce a higher training overhead and fragility (more failure points).
- k=96 is a fixed value. This doesn't account for object that vary widely in size or partially visible which can impact performance on diverse scenes.

### Questions
- To isolate the novelty of the SDM what is the performance gap between using $L_{itm}$ gradient cue vs a simpler method like CLIP attention maps with the dual branch structure?
- How does the weighted feature aggregation block compare to simple concatenation or element-wise summation?
- The authors should provide an analysis of failure cases and visualizations where the disentanglement fails (e.g. background regions misclassified as foreground or ambiguous boundaries) to illustrate the limitations of the SDM.

### Soundness
3

### Presentation
3

### Contribution
3
