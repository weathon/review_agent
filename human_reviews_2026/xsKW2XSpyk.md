# Patch-Level Kernel Alignment for Dense Self-Supervised Learning

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Dense self-supervised learning (SSL) methods showed its effectiveness in enhancing the fine-grained semantic understandings of vision models. However, existing approaches often rely on parametric assumptions or complex post-processing (e.g., clustering, sorting), limiting their flexibility and stability. To overcome these limitations, we introduce Patch-level Kernel Alignment (PaKA), a non-parametric, kernel-based approach that improves the dense representations of pretrained vision encoders with a post-(pre)training. Our method propose a robust and effective alignment objective that captures statistical dependencies which matches the intrinsic structure of high-dimensional dense feature distributions. In addition, we revisit the augmentation strategies inherited from image-level SSL and propose a refined augmentation strategy for dense SSL. Our framework improves dense representations by conducting a lightweight post-training stage on top of a pretrained model. With only 14 hours of additional training on a single GPU, our method achieves state-of-the-art performance across a range of dense vision benchmarks, demonstrating both efficiency and effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates SSL loss for dense-prediction tasks. The paper proposes a lightweight post-training phase with a new SSL loss to improve the dense-feature quality of a DINOv2-pretrained backbone. The paper's main contributions are:
1) A new PaKA loss that adopts Centered Kernel Alignment (CKA) to compare patch distributions between student and teacher views, after alignment of the local patches through an ROI process.
2) A new augmentation strategy that maximizes overlap between student and teacher views and reduces the strength of teacher data augmentation.

The authors validate their approach by post-training a DINOv2R baseline and testing it on VOC and ADE20K using both visual in-context learning and linear probing. Their approach shows improvement over DINOv2R or NeCo baselines. The authors also perform an ablation to show the impact of the different contributions.

### Strengths
- Paper introduces a lightweight post-training to performance of DINOv2 backbone
- Paper performs careful ablation to study the impact of the different components (CKA loss, data-augmentation)

### Weaknesses
My main issue with the paper is that the empirical results do not seem to fully support the claim:
- It seems that the reported numbers are quite low for the baselines. For instance, Table 3 in [1] reports that DINOv2 obtains a score of 83.1 on PascalVOC and 49.5 on ADE20K. In contrast, DINO2R is getting around 74.2 and 35.0 on VOC/ADE20K in Table 2. Are you using a different base model, and would the proposed approach transfer to a stronger base model?
- CKA is proposed as one of the main contributions in the paper but does not lead to a significant improvement in the linear probing protocol in Table 5a.
- The authors do not control for post-training on a different data distribution (COCO). What would be the performance of the DINO baseline, post-trained on COCO using the regular DINO loss and gram anchoring?
- Other supervised baselines could be included, such as PESpatial or AM-RADIOv2.5, and the evaluation could go beyond the linear probing protocol.

[1]: DINOv3, Siméoni et al., 2025.

### Questions
My main question is related to the first weakness, why do we see a gap in term of performance on dense linear probing compare to result reported to the litterature and would the gain obtain by the approach transfer to stronger backbone ?

### Soundness
1

### Presentation
2

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
This paper presents PaKA (Patch-level Kernel Alignment), a method for improving dense visual representations via self-supervised learning. Unlike clustering- or sorting-based approaches (e.g., iBOT, NeCo), PaKA proposes a non-parametric kernel-based alignment objective grounded in Centered Kernel Alignment (CKA). The method aligns the relational structure of patch embeddings between a student and teacher network, trained under an EMA framework.
Key design choices include:
- An CKA loss aligning pairwise patch similarities between student and teacher.
- A “clean teacher” (minimal augmentations) and high-overlap crops for consistent patch correspondence.
- Lightweight post-pretraining fine-tuning applicable to pretrained ViT encoders.
Empirically, PaKA achieves state-of-the-art performance on dense vision benchmarks (PASCAL VOC, COCO-Stuff, ADE20K) and demonstrates improved efficiency (–37% training time, –24% memory vs. NeCo). The paper is positioned as a generic refinement step rather than a full pretraining framework.

Summary of the review: The paper proposes PaKA, a simple and efficient refinement to dense self-supervised learning using a CKA-based objective that improves segmentation and depth performance while reducing compute and memory cost. The approach is practical and well-engineered, showing robustness and strong applicability to pretrained encoders. However, it lacks deeper conceptual novelty, mechanistic insight, and breadth of validation—key ablations, teacher analysis, and evaluations on classification, detection, or multimodal models are missing, and efficiency comparisons need clearer fairness. Overall, the work is technically sound and useful but remains primarily incremental rather than foundational, justifying a rating of 4 for solid engineering with limited conceptual advance.

Reproducibility Comment: Code provided is clean and structured with clear instructions. In the paper there is a clear provision of different evaluations and setups used. The main details for reproducibility are provided.

### Strengths
1) Simplicity: CKA provides a clean, non-parametric alternative to cluster-based dense SSL.
2) Empirical robustness: Consistent gains on dense segmentation and depth benchmarks.
3) Efficiency: Lower compute and memory cost than NeCo.
4) Practical relevance: A lightweight post-pretraining recipe applicable to strong pretrained encoders.

### Weaknesses
1. Batch-size dependence (missing ablation).
The authors acknowledge batch importance but show no study of how varying batch size affects convergence or downstream accuracy.

--> include 8–64 batch ablation for segmentation/hummingbird evaluation metrics.

2. Insufficient validation of “clean teacher.”
All evidence for the augmentation-free teacher is based on unsupervised segmentation, which is noisy.

--> Test the teacher-augmentation ablation using linear segmentation or Hummingbird-style stable probes that are less dependent on clustering variance.

3. Weak mechanistic motivation.
The paper demonstrates empirical gains but lacks an analysis explaining why the CKA objective enhances semantics (would also be interesting to see if it improves other metrics like object-detection as well and whether the motivation also provides insights for their task generalisability, therefore further explanation is expected ).

4. Limited understanding of PaKA's capabilities in global understanding.
PaKA’s effect on global understanding remains untested.

--> evaluate k-NN or linear classification on ImageNet vs. DINOv2R to ensure that dense refinement does not degrade global representations.

5. No exploration of VLM encoders (CLIP, SigLIP).
Given their prevalence, applying PaKA to CLIP or SigLIP vision encoders would clarify if the method benefits or harms text aligned vision models.  

--> Report zero-shot classification, retrieval, segmentation performance, and hummingbird evaluation pre/post PaKA on them.

6. Finetuning evaluation limited to linear heads.
There are no non-linear decoders used for tasks such as segmentation (as well as depth estimation)

--> Evaluate with non-linear decoders (e.g.,Mask2Former) to ensure benefits persist in end-to-end settings.

7. Limited generalisation to non-dense tasks.
No other evaluations to simple segmentation, and depth estimation.

--> Evaluate with other benchmarks like object-detection (VITDET), Panoptic Segmentation (e.g., Mask2Former), since both PaKA and its predecessor seem very good in the hummingbird evaluation (fetching related patches) the question that is raised from my end is how good the features are also in key point matching (use the features from PaKA as feature descriptors to evaluate how good are they in key point matching eg on the HPatches Dataset [3]), and the Multiview feature consistency [1]

8. Dataset choice unexplained.
Why is COCO preferred for post-training instead of ImageNet or mixed-domain corpora? 

--> A dataset ablation would clarify domain dependence.

9. Lacks of novelty to the creation of a new paradigm, seems more as an engineering investigation.
The method feels like a well-tuned refinement of existing self-distillation pipelines rather than a fundamentally new paradigm.

--> clarify more the position of PaKA ( is it an empirical improvement or a conceptual shift)

10. Computational (Execution and Memory) Cost Improvements are inconclusive
The reported efficiency gains (−37% time, −24% memory) may partly arise from engineering or implementation differences rather than inherent algorithmic simplicity. If PaKA employs xformers, torch compiled models, lighter augmentations, smaller intermediate tensors, mixed-precision optimization, or simply uses newer and more optimised libraries and/or has optimised their code further while NeCo’s public implementation does not, the comparison becomes uneven. 

-->  (a) clarify the exact setup of execution comparisons, (b) since the code structure is very similar to NeCo provide a side by side comparison of computational costs (time and memory) on a per part of the code that you introduced eg (data processing, augmentations, forward pass, loss, etc),  (c) potentially average the computational costs over multiple runs to avoid any hardware noise,


[1] Banani, M. E., Raj, A., Maninis, K.-K., Kar, A., Li, Y., Rubinstein, M., … Jampani, V. (2024). Probing the 3D Awareness of Visual Foundation Models. arXiv [Cs.CV]. http://arxiv.org/abs/2404.08636

[2] Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). Masked-attention Mask Transformer for Universal Image Segmentation. arXiv [Cs.CV].  http://arxiv.org/abs/2112.01527

[3] Balntas, V., Lenc, K., Vedaldi, A., & Mikolajczyk, K. (2017). HPatches: A benchmark and evaluation of handcrafted and learned local descriptors. arXiv [Cs.CV].  http://arxiv.org/abs/1704.05939

### Questions
Questions for Authors are derived from the weaknesses:

1. Batch-size dependence
Could you include an ablation studying how varying batch size (e.g., 8–64) affects convergence, stability, and downstream accuracy?
This is particularly relevant since relational losses (like CKA) are sensitive to sample diversity and normalization across the batch.

2. Validation of the “clean teacher” design
Can you validate the effect of the augmentation-free teacher using more stable metrics, such as linear segmentation or Hummingbird-style evaluations, instead of unsupervised clustering-based segmentation, which is inherently noisy?
This would strengthen the causal link between the teacher’s augmentation level and representation quality.

3. Mechanistic motivation and conceptual clarity
Could you expand on why the CKA objective improves semantics?
Do you observe improved structure preservation or inter-class separation in the learned representation?
Would this mechanistic intuition also explain potential improvements in object detection or other structured tasks?

4. Global understanding and classification ability
How does PaKA influence global-level understanding?
Please evaluate on k-NN or linear classification (e.g., ImageNet-1k) compared to DINOv2R to ensure that dense refinement does not harm global semantic performance.

5. VLM encoder evaluation (CLIP, SigLIP)
Have you tested PaKA on vision encoders from multimodal models such as CLIP or SigLIP (with frozen text towers)?
Reporting zero-shot classification, retrieval, and segmentation performance (and optionally Hummingbird scores) before and after PaKA would clarify whether the method generalizes to text-aligned encoders without degrading cross-modal alignment.

6. Finetuning with non-linear decoders
Can you evaluate PaKA-pretrained encoders with decoder-based architectures (e.g., Mask2Former [2]) for segmentation and/or depth estimation?
This would show whether improvements persist in end-to-end fine-tuning beyond linear heads.

7. Generalisation to broader visual tasks
Could you test PaKA on additional dense and geometric benchmarks such as:
a. Object detection (e.g., ViTDet),
b. Panoptic segmentation (e.g., Mask2Former),
c. Keypoint matching using PaKA features as descriptors on HPatches [3], and
d. Multiview feature consistency or 3D awareness tests as in Banani et al. (CVPR 2024) [1]?
This would better establish the method’s versatility and geometric consistency.

8. Dataset choice for post-training
Why was COCO selected as the post-training dataset instead of ImageNet or a mixed-domain corpus?
Could you provide a small dataset ablation (e.g., ImageNet vs COCO) to demonstrate that PaKA’s gains are not domain-dependent?

9. Positioning and novelty of PaKA
How should readers interpret PaKA in the broader SSL landscape—do you consider it a conceptual advance or a well-engineered refinement of self-distillation pipelines?
Clarifying this positioning would help reviewers and readers understand its intended scope and contribution level.

10. Fairness of computational efficiency claims
The reported runtime and memory reductions (−37% time, −24% memory) might depend on implementation-specific optimizations.
Could you:
a) Clarify whether both PaKA and NeCo were benchmarked under identical optimization conditions (precision, compiler, libraries, data loading, etc.)?
b) Provide a side-by-side breakdown of time and memory usage per module (data loading, augmentations, forward pass, loss computation, etc.)?
c) Average timings over multiple runs to reduce hardware noise?
This would ensure the efficiency comparison is methodologically fair.

[1] Banani, M. E., Raj, A., Maninis, K.-K., Kar, A., Li, Y., Rubinstein, M., … Jampani, V. (2024). Probing the 3D Awareness of Visual Foundation Models. arXiv [Cs.CV]. http://arxiv.org/abs/2404.08636

[2] Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). Masked-attention Mask Transformer for Universal Image Segmentation. arXiv [Cs.CV].  http://arxiv.org/abs/2112.01527

[3] Balntas, V., Lenc, K., Vedaldi, A., & Mikolajczyk, K. (2017). HPatches: A benchmark and evaluation of handcrafted and learned local descriptors. arXiv [Cs.CV].  http://arxiv.org/abs/1704.05939

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
This paper presents a novel post-(pre)training method for improving the low-level quality of self-supervised learning representations, and thus increase the downstream performance on tasks such as segmentation or overclustering. Experimental results show that applying on a DINOv2 model leads to significant improvements on benchmarks such as Pascal VOC, COCO, or ADE20k.

### Strengths
- The presented method is well-motivated, in particular the need for a non-parametric local-matching approach in dense self-supervised learning, and matching gram matrices on locally aggregated patch-level representations. The Centered Kernel Alignment (CKA) approach is also well motivated as opposed to using the actual gram matrix.

- The paper is very clear and well-written, the proposed method is correctly formalized and the figures go straight to the point and help the understanding. 

- The evaluation setup is strong and makes the results convincing. Large-scale datasets are used (COCO, ADE20k), against strong baselines (DINOv2R), with meaningful evaluation protocol (linear probing).

### Weaknesses
- The presented method should be agnostic to any pretraining method. It would have been great to show this by post-training on more than a single model, right now there is no way of saying if this post-training is specific to DINO or if it works in the general case.

- The specific backbone used for most of the experiment is not specified in the main text or in the figures captions. I think this is a very important detail for practitioners that should be highlighted.

- Where does DINOv2R line performance come from in Table 1,2,3,4 ? Is it your reproduction or from another paper ? I can’t see these results on the original paper. This is why specifying the backbone is important, otherwise comparisons are meaningless.
Why do you compare your results to this specific version of DINO ? Why not DINOv2 or v3 ?

- I don’t think Section 4 is relevant, at least it should as much importance because these findings on data augmentation are very specific to the particular method that is proposed, and particular to the fact that this is post-training and not pre-training. The writing might be a bit misleading “Motivated by the limitations of this inherited augmentation paradigm” makes it believe that this is a new go-to augmentation strategy.

- There are a number of small presentation issues: 

- L851 typo “evalutaions”

- L23 typo “Kernal”

- “For a fair comparison, we also post-trained same backbone model by NeCo (Pariza et al., 2025),” This is confusing , do you post-train from neco, which is pretrained from dinov2r according to the Table 1 ?

- Line 41 “another line of research (Lebailly et al., 2024; Pariza et al., 2025; Stegmüller et al., 2023; Ziegler & Asano, 2022) has focused on dense representation learning via self-distillation” DINOv2 and Ibot can also be considered as self-distillation methods

- Some figures are blurry, make sure to use .pdf instead of .png or .jpeg

- “while reducing computation by 37% and memory usage by 24% compared to prior methods” which one ? This needs to be precise and only can compare to post-training techniques. The answer is actually in section 5.4: “more memory-efficient compared to NeCo” it should be mentioned above.

### Questions
The method is used as a post-(pre)training refinement stage. Have you thought about making it a pretraining method ?

### Soundness
3

### Presentation
3

### Contribution
3
