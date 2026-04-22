# Consistent Region-Informed Self-supervised Pretraining

- Avg Score: 3.60
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 4, 4

## Abstract
Dense prediction tasks such as semantic segmentation require representations that capture both global semantics and local structure. Most self-supervised learning methods prioritise image-level invariance, producing strong features for classification but offering limited guidance for tasks requiring (or depending on) spatial coherence. In parallel, several approaches have been proposed specifically for dense prediction, but their improvements in local fidelity often come at the cost of weaker global transfer.
We present CRISP (Consistent Region-Informed Self-Supervised Pretraining), a framework that enhances patch-level learning with explicit region-level alignment. CRISP discovers coherent regions in a reference image, projects them to augmented views via geometric correspondences, and aggregates their patch features into concept tokens with a mask-guided module. By enforcing consistency at the region, patch, and global levels, CRISP learns representations that are both semantically strong and spatially coherent.
Pretraining on ImageNet-1K shows that CRISP achieves substantial gains on dense prediction benchmarks while maintaining strong performance on global benchmarks. These results establish region-level consistency as a critical ingredient for advancing universal visual representations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper builds on iBOT-stype self-distillation visual self-supervised pretraining pipeline. It introduces a method that discovers a concept region in an image, and performs self-distillation between the teacher and student networks on the discovered region through a concept token. The region is defined by a patch similarity metric, in which all patches have large cosine similarities to a uniformly sampled seed patch. The mask information is preserved in augmented views by directly imposing the same geometric augmentation as the RGB input receives. A concept token performs cross-attention to the ViT backbone encodings to summarize the information inside the selected region. The iBOT loss is applied on this concept token to perform self-distillation, along with the original iBOT class token and patch token losses.

### Strengths
1. The paper is easy to follow.
2. The proposed method increases the performance compared to some selected baseline methods.
3. Ablation studies on some hyperparameters are performed. 
4. Visualizations are good.
5. The code is provided.

### Weaknesses
1. The proposed method, CRISP, does not train from scratch and essentially is a finetuning method on top of iBOT. This raises concerns about the fairness of the comparison between CRISP and the baselines. The authors perform the finetuning for another 200 epochs, which is effectively 800 epochs if the authors strictly follow the iBOT training pipeline (see the original iBOT paper for the effective training epochs). The authors do not provide information or experiments on this huge additional training budget compared to the baseline methods. This point undermines the Soundness score.
2. The proposed concept token method resembles the ViT-register approach [1], in which additional cls-token-like tokens attend to different regions of the input image (Figure 9 of [1]), without requiring a pre-trained teacher network for region proposal. The register approach contributes to the dense prediction task performance of ViTs, while CRISP is not compared to it. This point undermines the Contribution score. Could the authors elaborate on the relationship between CRISP and the ViT-register method, and compare them on the dense prediction tasks?
3. CRISP involves feeding the concept token through additional ViT blocks for aggregating the regional information. This additional model size leads to additional training costs, which also leads to unfairness in the comparison. While evidence suggests that more layers of the concept-token ViT do not translate to better performance (Table 6), it is still unclear how this additional training budget/model size contributes to the baseline performance, especially in the case where the proposed method is trained longer than the baseline ones. This point undermines the Soundness score. 
4. The reported metrics are all based on frozen backbones. While this pipeline can yield a metric of the feature quality, it might not correspond to the actual performance after finetuning the model on the specific task [2]. The authors do not provide information about the finetuning performance. This point undermines the Soundness score.
5. The region proposal scheme in CRISP is not novel in the MIM family. Several methods [2,3,4] find that the MIM model itself can provide semantically-meaningful region-level masks, such as the foreground object. There is no elaboration and comparison between CRISP and these methods. This point undermines the Contribution/Soundness score. 
6. The selected baseline methods are not very strong. Recent advancements [5,6] in visual self-supervised learning introduce several improvements over the selected methods. They show gains in dense prediction tasks compared to the baseline methods. The authors do not experiment with these advanced methods. This point undermines the Contribution/Soundness score. 
7. The dataset used for benchmarking is small. While ADE20K, Pascal VOC, and Cityscapes are commonly used for evaluating the semantic segmentation tasks, they are small compared to, e.g., MS COCO. Evaluation on the small datasets might not give conclusive and reliable results. This point undermines the Soundness score. 
8. This is minor, but the authors do not elaborate on the effect of CRISP on high-level tasks, such as image classification (although a multi-label result is presented, no result on a common image classification dataset, say ImageNet1K, is reported). A visual representation will not be qualified as "universal" if it cannot achieve competitive performance in all aspects. However, this can be a wording issue.

[1] Vision transformers need registers. ICLR 2024

[2] Evolved Hierarchical Masking for Self-Supervised Learning. TPAMI 2025.

[3] Self-guided masked autoencoder. NeurIPS 2024.

[4] What to hide from your students: Attention-guided masked image modeling. ECCV 2022.

[5] Contrastive masked autoencoders are stronger vision learners. TPAMI 2023.

[6] Context Autoencoder for Self-supervised Representation Learning. IJCV 2024.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes CRISP: a self-supervised pretraining framework that adds a region-level consistency pathway to a standard teacher–student ViT setup. It (i) discovers coherent regions in a reference view from teacher features, (ii) warps those regions exactly into two augmented views using known geometric transforms, and (iii) aggregates the region’s patches via a masked-attention “concept token,” aligning student/teacher region embeddings with a distillation loss. The goal is to bridge global semantics and local spatial coherence so that features transfer well to dense tasks (segmentation, correspondence, mask propagation) without sacrificing image-level performance. The idea is positioned against invariance-only pretraining (e.g., DINOv2, iBOT) and dense-SSL variants (e.g., DenseCL, PixPro, DetCon, CrOC/CrIBo).

Summary of the review: The paper tackles the gap between globally discriminative yet spatially fragmented features and dense task requirements through a simple, elegant approach using geometric mask warping and a lightweight concept token. The method is conceptually clear and technically sound, offering a principled way to improve spatial coherence. However, it relies heavily on threshold sensitivity and region selection, lacks comprehensive comparisons to related methods, and could benefit from ablations and multi-region analyses for robustness. Overall, it’s a promising and thoughtful contribution with solid potential but limited breadth, warranting a rating of 6 for its clarity, soundness, and room for further validation.


Reproducibility Comments: Code is provided and it seems clean. The results seem reproducible based on the code provided as well.

### Strengths
1) Clear problem framing: addresses the known gap between globally discriminative but spatially fragmented features and dense tasks’ needs. 
2) Method simplicity & elegance: exact geometric mask warping avoids ambiguous cross-view matching; the concept token is an intuitive, lightweight pooling mechanism.

### Weaknesses
1) Region discovery sensitivity. The method hinges on a similarity threshold and on teacher strength. Please add ablations: performance vs threshold; effect of averaging blocks; behaviour from random init; and whether adaptive/learned thresholds outperform fixed. 

2) Coverage: one region per iteration. Many-object scenes may be under-regularized. Evaluate sampling multiple regions per image per step (with care to avoid collapse), and report compute implications.

3) No Comparative breadth Experiments: Does not include  head-to-heads (same backbone, same data) versus DenseCL, PixPro, STEGO, DetCon, CrOC, CrIBo, DINOv2 to solidify the benefits

### Questions
1) Discovery stability: How stable are discovered regions across seeds/epochs? Any temporal ensembling or EMA helps? Please include a region repeatability metric.

2) From-scratch feasibility: Can CRISP train without an iBOT warm start? If unstable, could you ramp the region loss or pretrain the teacher briefly? 

3) Multiple regions: What fails if you select multiple seeds per image per step? Any signs of representational collapse or compute blow-up?

4) In-Context Visual Understanding: Does the new model gain any benefits on the recent proposed Hummingbird evaluation [1] ( implemented openly by [2])?


[1] Balažević, I., Steiner, D., Parthasarathy, N., Arandjelović, R., & Hénaff, O. J. (2023). Towards In-context Scene Understanding. arXiv [Cs.CV]. http://arxiv.org/abs/2306.01667 [2] https://github.com/vpariza/open-hummingbird-eval

### Soundness
3

### Presentation
3

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
The paper proposes a dense self-supervised learning method based on the following steps:  
1. Extract object regions using an SSL-based approach.  
2. Generate two views of each object region.  
3. Apply a random mask to one of the views. Feed the first (unmasked) view to the teacher network and extract its features.  
4. Feed the second (masked) view to the student network and extract its dense features.  
5. Append a concept token to the student and teacher features and pass them through an additional transformer block, where the concept token is allowed to attend only to the patches within the **region** mask, while no cross-patch attention is permitted.  
6. Finally, enforce the concept token’s representation to be similar to the teacher’s region features.

The method is initialized from iBOT to facilitate the mask extraction phase for the teacher network.   The results show superior performance compared to CrIBO and competitive results with CAPI.

### Strengths
1. The paper addresses a topic of growing importance in the self-supervised learning (SSL) domain. Dense self-supervised learning has numerous applications that have not been sufficiently explored in the SSL literature, where the main focus has largely been on improving classification performance. More works like this are needed to advance research in this direction.

2. The experiments demonstrate scalability from ViT-S to ViT-L, indicating that the method is likely to benefit further from larger models and higher-quality datasets.

3. The inclusion of qualitative comparisons strengthens the experimental section and increases the trustworthiness of the results.

### Weaknesses
1. The experimental benchmarks are insufficient and inconsistent. There are established benchmarks for dense self-supervised learning such as [1], which have been reported in prior works [1,2,3,4, 5]. Although the paper cites [1] and compares with it in the tables, it does not perform experiments on any of the established benchmarks. Furthermore, the comparison methods are inconsistent across ViT-S and ViT-B models, making it difficult to obtain a fair and comprehensive view of the method’s performance across different model sizes.

2. The paper argues (lines 70–76) that existing dense self-supervised methods rely on hand-crafted region extraction techniques and perform poorly on classification benchmarks. However, the proposed method also employs a hand-crafted algorithm to extract object regions—at least as hand-crafted as CrIBO—and similarly does not report performance on classification benchmarks.

3. The method is built upon a pretrained iBOT network and aims to enhance its dense representations. In this regard, it would be more appropriate to compare it with post-training methods that also start from pretrained architectures (e.g., [ 2, 3, 4]). Such comparisons are currently missing from the paper.

4. Additional ablations are needed to analyze the impact of initialization (e.g., DINOv1/v2/v3) and to justify the necessity of the concept token, given that a CLS token already exists. In Table 6, the configuration with L=2 achieves the best results, yet the authors choose L=1 for efficiency. However, no efficiency metric or column is provided to support this decision, which should be added to strengthen the justification.

5. While there are interesting visualization on the sparse correspondence ability of the method, there is no quantitative table for that. Please use the correspondence benchmarks used in [3] to quantify this section. 

6. The method section requires further clarification. For example, it is not clearly explained why only the concept token is allowed to attend to all patches, while other tokens are restricted. Additionally, the figure should explicitly illustrate both the random mask and the region mask to make the overall design and data flow clearer.


[1] - CrIBo: Self-Supervised Learning via Cross-Image Object-Level Bootstrapping, ICLR24

[2] - Time Does Tell: Self-Supervised Time-Tuning of Dense Image Representations, ICCV23

[3] - Near, far: Patch-ordering enhances vision foundation models' scene understanding, ICLR25

[4] - MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning, ICCV25

[5] -Towards In-context Scene Understanding, NeurIPS23

### Questions
Please refer to the weakness section. I explained my requests and questions there.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a self-supervised learning objective, CRISP, that explicitly leverages spatial coherence. Specifically, CRISP first identifies semantically coherent regions in a reference image and tracks them during the process of geometric augmentations such as cropping. It then maximizes the similarity between region-level representations from two augmented views. Experiments demonstrate the effectiveness of CRISP on various dense prediction tasks, including semantic segmentation.

### Strengths
- The proposed idea is simple yet effective.
- The paper is generally well-written and easy to follow.
- The method consistently outperforms existing self-supervised learning baselines pretrained on ImageNet-1k in both segmentation and classification tasks.

### Weaknesses
- For dense prediction tasks, this paper only evaluates on semantic segmentation. Other dense prediction tasks such as object detection and depth estimation are also important to assess whether the model truly learns spatially fine-grained, region-level representations from the proposed objective.
- CRISP requires a strong initialization (e.g., iBOT) to obtain reliable region masks. Such dependency may limit its applicability across different datasets or domains. It would be valuable to investigate whether CRISP can also be trained from scratch without relying on pretrained models.
- It is unclear whether CRISP can be integrated with other self-supervised frameworks beyond iBOT. For instance, starting from DINOv2, would fine-tuning with CRISP improve the quality of region- or patch-level representations? If not, this limitation could weaken the practical impact of the paper.
- Scalability is a critical factor in self-supervised learning. It would strengthen the paper to demonstrate the scalability of the proposed method beyond ImageNet-1k.
- Recently, CLIP-based models (e.g., SigLIP, Perception Encoder) have shown strong performance on dense prediction tasks, sometimes surpassing dense SSL models such as DINOv2. It would be interesting to discuss whether the idea of CRISP could be extended or adapted to such vision-language models.
- I am also curious about the sparse correspondence results shown in Section 4.3. How does CRISP affect the correspondence quality compared to other SSL methods such as iBOT, DINOv2, and CAPI? Does CRISP explicitly improve sparse correspondence performance?

### Questions
See the weakness part.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
CRISP augments the iBOT self-supervised pipeline with a third, region-level objective: it first discovers coherent visual regions in a reference view by thresholding patch-to-patch cosine similarities in the teacher network, warps the resulting mask to two augmented views via the known geometric transforms, and then aggregates the masked patch tokens into a single “concept” token with a lightweight, mask-gated transformer block. Student and teacher concept tokens are aligned with a distillation loss that operates alongside the original global [CLS] and patch-level losses, yielding representations that are simultaneously semantic and spatially precise. ImageNet-1K pre-training of ViT-S/B/L models produces state-of-the-art frozen-feature results on ADE20K, PASCAL VOC and Cityscapes (k-NN and linear protocols), competitive video segmentation on DAVIS/YouTube-VOS/MOSE, and classification numbers on par with DINOv2 despite using ∼100× less pre-training data. Extensive ablations show that (i) averaging the last four teacher blocks for similarity maps, (ii) a cosine threshold β = 0.75, and (iii) a single concept-transformer block give the best accuracy/efficiency trade-off; visualisations confirm sharper object boundaries and robust cross-image correspondences. The work therefore provides the first demonstration that explicit, geometry-warped region consistency can be injected into an invariance-based SSL framework without extra labels, heuristics, or heavy compute, closing a long-standing gap between global and dense self-supervised learning.

### Strengths
region-level consistency is enforced across augmented views by re-using the known geometric transforms that created the views—an idea that is simple, parameter-free, and complementary to existing global or patch losses. It directly addresses the spatial-misalignment weakness of DINO-style methods without resorting to offline correspondence or external segmentation priors.

### Weaknesses
The paper does not analyse why aligning concept tokens should improve downstream dense tasks beyond the intuitive “better spatial coherence”. There is no discussion of collapse modes, no gradient analysis, and no information-theoretic argument that relates the new region loss to the original patch and global losses.

Regions are discovered by a single cosine threshold applied uniformly across all images and throughout training. The authors acknowledge that this fixed threshold may break for scenes with very different object scales or for thin structures, but no adaptive or learnable alternative is explored; hence the method may fail on datasets that are visually unlike ImageNet.

CRISP is initialised from iBOT and fine-tuned for 200 epochs, yet several competitors (CAPI, DINOv2*) are compared in their “fully-converged” state without matching initialisation or epoch budget. A controlled experiment that starts every method from the same random weights and trains for exactly the same schedule would strengthen the claim that region consistency, rather than longer optimisation, drives the gains.

All dense benchmarks are reported with frozen features; there is no end-to-end fine-tuning comparison. Consequently it remains unclear whether CRISP still helps when task-specific heads and full network updates are allowed—arguably the more common deployment scenario.

### Questions
How sensitive are the results to the cosine threshold β? Did you try a schedule that lowers β during training to capture progressively larger context?

What happens if the region loss weight λ_region is pushed well above 1? Is there a regime where region consistency starts to hurt global classification?

Does CRISP still outperform iBOT when both are trained from scratch (no iBOT warm-start) for 400 or 800 epochs?

Have you tested end-to-end fine-tuning on ADE20K or Cityscapes? Does the gap persist, shrink, or invert?

How does the concept-token aggregation complexity scale when patch size decreases (e.g., 8×8) and the number of candidate regions increases?

Could you clarify the failure modes—what fraction of discovered regions miss object boundaries or spill heavily into background?

### Soundness
2

### Presentation
2

### Contribution
2
