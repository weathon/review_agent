# ContextGen: Contextual Layout Anchoring for Identity-Consistent Multi-Instance Generation

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Multi-instance image generation (MIG) remains a significant challenge for modern diffusion models due to key limitations in achieving precise control over object layout and preserving the identity of multiple distinct subjects. To address these limitations, we introduce **ContextGen**, a novel Diffusion Transformer framework for multi-instance generation that is guided by both layout and reference images. Our approach integrates two key technical contributions: a **Contextual Layout Anchoring (CLA)** mechanism that incorporates the composite layout image into the generation context to robustly anchor the objects in their desired positions, and **Identity Consistency Attention (ICA)**, an  innovative attention mechanism that leverages contextual reference images to ensure the identity consistency of multiple instances. To address the absence of a large-scale, high-quality dataset for this task, we introduce **IMIG-100K**, the first dataset to provide detailed layout and identity annotations specifically designed for Multi-Instance Generation. Extensive experiments demonstrate that ContextGen sets a new state-of-the-art, outperforming existing methods especially in layout control and identity fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces ContextGen, a novel Diffusion Transformer-based framework designed for multi-instance image generation, a task that requires precise control over the layout and identity of multiple subjects. The authors identify key challenges in existing methods, namely inadequate position control, weak identity preservation, and a lack of high-quality training data. To address these, they propose a model guided by both layout and reference images, featuring two main technical contributions: Contextual Layout Anchoring (CLA) for spatial control and Identity Consistency Attention (ICA) for identity preservation. Furthermore, they introduce a new large-scale dataset, IMIG-100K, with detailed layout and identity annotations. The experimental results show that ContextGen achieves state-of-the-art performance, outperforming existing open-source and proprietary models on several benchmarks. The strength of the paper lies in the successful integration of these components to produce a highly effective system. However, the work is weakened by concerns about the novelty of its individual methodological components and the conclusiveness of the ablation studies used to validate them.

### Strengths
The primary strength of this work is its empirical success. The authors have successfully engineered a complete system that achieves a new state-of-the-art in the complex task of layout-controlled, multi-instance image generation. The combination of a large, well-curated dataset (IMIG-100K), a strong foundation model (FLUX.1-Kontext), a tailored architecture (CLA and ICA), and a robust fine-tuning strategy (including DPO) has proven to be highly effective. The quantitative results across multiple benchmarks, particularly the comprehensive comparison in Table 1, demonstrate a clear advantage over a wide range of baseline methods in overall performance. This indicates that the holistic approach and the specific combination of techniques are well-suited for the problem at hand.

### Weaknesses
Despite the strong empirical results, the paper has several weaknesses concerning its methodological contributions and the clarity of their validation.

1.  **Limited Novelty in Core Mechanisms:** Beyond the ICA mechanism, the core methodological contributions appear to be adaptations of existing techniques. Contextual Layout Anchoring (CLA) seems to be a general strategy for multi-reference generation where each reference's attention is constrained, a concept that is not entirely new. The idea of using a composite layout image as an explicit input is interesting, but the paper lacks a specific ablation study to demonstrate its effectiveness compared to a baseline without it. Similarly, the use of Direct Preference Optimization (DPO) is an application of an existing fine-tuning technique. While Table 4 shows it is effective for this task, it does not represent a novel methodological contribution from this work.

2.  **Unconvincing Ablation Study for ICA:** The novelty of the Identity Consistency Attention (ICA) mechanism is the paper's most significant claim, but its validation in Table 3 is not fully convincing. While the final proposed model (applying ICA to the middle blocks) achieves the highest average score and IDS, the performance differences among the last four configurations are marginal and likely within the noise margin. The configuration that completely omits ICA (row beginning with "ITC = 90.26") performs almost as well, which raises questions about whether ICA provides a truly significant or necessary improvement. Furthermore, there is a large, unexplained gap in the IDS metric between the first three configurations and the last four. The paper does not offer an explanation for this pattern, making it difficult for the reader to understand the specific properties or impact of the ICA mechanism. Overall, the ablation study does not provide compelling evidence for the effectiveness of ICA.

3.  **Overstated Contribution of the Dataset:** While the effort in creating the IMIG-100K dataset is substantial, the claim that it is the "first large-scale, hierarchically-structured image-guided multi-instance generation dataset" is questionable. The pipeline described—using generative models to create images and segmentation models to extract references—has become a common practice in the field, with similar efforts seen in works like OmniGen2, Qwen-Image, and MS-Diffusion. As such, while the dataset is a valuable asset, its creation does not constitute a significant research innovation.

4.  **Confounded Baseline Comparisons:** The main results in Table 1 are difficult to interpret due to a lack of controlled variables. The proposed method benefits from a stronger base model (FLUX), a larger and task-specific training dataset (IMIG-100K), and an additional DPO tuning stage, whereas many baselines (e.g., MS-Diffusion) use older, less capable base models and different data. This makes it impossible to attribute the performance gains specifically to the novel CLA and ICA mechanisms. While it is understood that re-implementing all baselines under identical conditions is costly, this confounding of variables limits the scientific insight that can be drawn from the results. The paper demonstrates that a better system can be built with better components, but it fails to provide clear evidence for the isolated impact of its core architectural innovations.

### Questions
1.  Could you provide an ablation study that demonstrates the specific benefit of using the composite layout image as an additional input? For instance, a comparison with a model variant that receives only the noise image and the individual reference images, without the pre-composited layout image. This would help clarify the contribution of this specific design choice.

2.  Could you provide a more detailed analysis of the ICA ablation results in Table 3? Specifically, please elaborate on why the performance variations are so small among the last four configurations and justify the final choice. More importantly, please explain the large and systematic performance gap (especially in the IDS metric) between the first three configurations and the last four. What does this reveal about the properties and effectiveness of the ICA mechanism?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper advances multi-instance generation (MIG) with not only precise layout control, but identity preservation. To simultaneously achieve  these two goals, the authors devise CLA and ICA to manipulate attention masks for better identity and location understanding. Moreover, they also propose a large-scale dataset to facilitate future research in this field. Extensive experiments on three large-scale benchmarks demonstrate their method's efficacy.

### Strengths
(1) The proposed dataset, IMIG-100K, has a significant meaning to the IMIG field.

(2) The overall training process is clear and could be easily followed.

(3) Comparison with ``Closed-Source Commercial Models'' in Tab.1 strongly demonstrates the efficacy of proposed method.

### Weaknesses
Major:
(1) The inference process is not clear on pure "layout-to-image" benchmarks. Based on Fig. 2, the ref images seems to be mandatory for model's inference. However, the included layout-to-image benchmark (COCO-MIG and LayoutSAM-eval) only provide layout and instance captions. How to perform inference on these benchmarks with yoour model? Is it possible to infer your model without any reference images? Please clarify this.

(2) Instance-Wise Position Indexing is one of the contributions for this work, but I cannot see any ablation study about this.

(3) In Fig.7, the generated image exhibits noticeable artifacts (crab legs on the bag) with DPO. Therefore, it is necessary to report metrics about image quality (FID or IS). The authors can compute FID and report the result on LayoutSAM-eval as CreatiLayout has also reported FID on this benchmark.


Minor:
(1) Typo: Tab.1 #L338, "FLUX" is mis-spelled.

Ref.
[a] CreatiLayout: Siamese Multimodal Diffusion Transformer for Creative Layout-to-Image Generation. In ICCV'25.

### Questions
Please refer to "weaknesses"

### Soundness
3

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
4

### Summary
The authors introduce ContextGen, a novel Diffusion Transformer (DiT) framework designed to tackle the significant challenges of multi-instance image generation (MIG). The method aims to solve two key problems: poor layout control and the failure to preserve the identities of multiple distinct subjects. The framework integrates two primary technical innovations: 1) a Contextual Layout Anchoring (CLA) mechanism to precisely position objects, and 2) an Identity Consistency Attention (ICA) mechanism that uses reference images to maintain the identity of each instance. Recognizing a lack of training data, the authors also created IMIG-100K, a new large-scale dataset for this task. The paper reports state-of-the-art performance in control precision, identity fidelity, and image quality.

### Strengths
- The work's primary strength is its direct and simultaneous attack on the two most critical failure points in multi-instance generation: layout control and identity preservation.
- The paper proposes two clear, novel technical components (CLA and ICA) integrated into a modern DiT framework. The Identity Consistency Attention, in particular, appears to be a sophisticated mechanism tailored specifically for the multi-subject challenge.
- The introduction of the IMIG-100K dataset is a valuable contribution in its own right. A large-scale, hierarchically-structured dataset with detailed layout and identity annotations addresses a major resource gap and will be beneficial for the wider community.

### Weaknesses
- Fidelity vs. Generative Diversity Trade-off: The model appears to struggle with the balance between preserving identity and generating novel depictions. As suggested by the results (e.g., Figure 4, rows 1 and 3), the generated instances often exhibit a "copy-paste" artifact, adhering so rigidly to the reference images that they lack meaningful diversity. This suggests a potential overfitting to the reference context and limits the model's true generative flexibility.
- Sub-optimal Performance on Key Metrics: Despite the SOTA claims, the method does not achieve top performance across all evaluated metrics. This performance gap may be linked to the newly introduced IMIG-100K dataset. By relying heavily on synthetic data, the model may not learn the full complexity and nuances of real-world image distributions, which could be limiting its generalization and robustness. Incorporating more authentic, real-world images into the training curriculum might be necessary to close this gap.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenging task of multi-instance image generation (MIG), which requires simultaneous control over the layout of multiple objects and the preservation of their distinct identities. The authors propose ContextGen, a novel Diffusion Transformer (DiT) based framework. The framework features two core innovations: 1) Contextual Layout Anchoring (CLA), which uses a "composite layout image" as context to precisely anchor object positions; and 2) Identity Consistency Attention (ICA), a novel attention mechanism that leverages reference images to maintain high-fidelity identity for multiple instances. Furthermore, to address the lack of high-quality training data in this domain, the authors have constructed and introduced a large-scale, hierarchically-structured dataset, IMIG-100K. The experimental evaluation is extensive. On three different benchmarks (LAMICBench++, COCO-MIG, and LayoutSam-Eval), ContextGen achieves state-of-the-art performance, outperforming not only all open-source models but also top-tier proprietary models (like GPT-4o and Nano Banana) on key identity preservation metrics.

### Strengths
1. Empirical Results: As mentioned, the paper's experimental results are its greatest strength. It comprehensively outperforms all competitors (both open-source and proprietary) on three distinct and challenging benchmarks. The ability to handle multiple (>3) subjects, shown in Tables 1 and 4, is a capability that other models clearly lack.
2. High-Quality Dataset Contribution: The release of IMIG-100K  is a major contribution. The authors not only created a dataset but thoughtfully designed it in three subsets (Basic, Complex, Flexible) to support the model's progressive training curriculum .
3. Architectural Design: The hierarchical attention design is excellent. Rather than blindly applying attention masks, the authors discovered through experiments (Table 3) that different layers of the DiT indeed have functional specializations—CLA is applied to the first/last blocks for structure, while ICA is applied to the middle blocks for identity details.

### Weaknesses
1. Confusing DPO Ablation: This is the only confusing part of the paper. Table 3  shows the best (non-DPO) model (F+M+B) achieving an AVG score of 64.66. However, in Table 4, the "w/o DPO" baseline has an AVG score of only 62.55. Both scores are presumably from LAMICBench++, but they do not match. The authors claim DPO provides a benefit (62.67 vs 62.55) , but this seems to be a degradation compared to the 64.66 score in Table 3. The authors need to clarify this discrepancy.
2. Reliance on Base Model: This work is a LoRA fine-tune on a very strong (and until recently, non-public) base model, FLUX.1-Kontext. This is not a flaw, but rather an efficient research methodology. However, it does mean the success is built, in part, on the inherent in-context capabilities of the Kontext model. The authors are transparent about this in Appendix A.1 , which is good.
3. Dataset Limitations: The IMIG-100K dataset is synthetic, meaning it may not fully capture the entire complexity and long-tail distribution of real-world imagery.

### Questions
Please referring Weakness.

### Soundness
3

### Presentation
3

### Contribution
3
