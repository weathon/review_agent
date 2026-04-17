# Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction For 3D-Aware Distillation

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Vision Foundation Models (VFMs) have achieved remarkable success when applied to various downstream 2D tasks. Despite their effectiveness, they often exhibit a critical lack of 3D awareness. To this end, we introduce Splat and Distill, a framework that instills robust 3D awareness into 2D VFMs by augmenting the teacher model with a fast, feed-forward 3D reconstruction pipeline. Given 2D features produced by a teacher model, our method first lifts these features into an explicit 3D Gaussian representation, in a feedforward manner. These 3D features are then "splatted" onto novel viewpoints, producing a set of novel 2D feature maps used to supervise the student model, "distilling" geometrically grounded knowledge. By replacing slow per-scene optimization of prior work with our feed-forward lifting approach, our framework avoids feature-averaging artifacts, creating a dynamic learning process where the teacher’s consistency improves alongside that of the student. We conduct a comprehensive evaluation on a suite of downstream tasks, including monocular depth estimation, surface normal estimation, multi-view correspondence, and semantic segmentation. Our method significantly outperforms prior works, not only achieving substantial gains in 3D awareness but also enhancing the underlying semantic richness of 2D features. Our project page and code are available [here](https://davidshavin4.github.io/Splat-and-Distill/)

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work proposes a distillation framework to enhance 3D awareness of a self-supervised encoder. The idea is to leverage a feed-forward model for multi-view 3DGS inference (pre-trained) to synthesise a 3D-consistent target for a student network. Specifically, an EMA teacher uses segmentation masks to fuse its patch-level representation into 3DGS by guided usampling. The resulting high-resolution feature grid is downsampled for student distillation.
Experiments on six datasets demonstrate notable improvement of the base model across four tasks: depth estimation, multi-view correspondence, surface normal estimation and semantic segmentation.

### Strengths
* Creating a 3DGS representation on-the-fly for 3D-aware feature distillation is a very nice idea.
* The experimental scope and the demonstrated results are laudable. The approach demonstrates consistent and often significant improvement across all tasks.
* The approach is relatively simple and leads to representations, which can be used standalone (without the need for concatenating them with the baseline features in evaluation).

### Weaknesses
* Overall, it is unclear where the gains come from. Conceptually, the approach designs a view-invariant representation and uses segmentation masks. This explains the gains on semantic segmentation, which requires view-invariance, but not the 3D-aware tasks (e.g. depth), where the representation is covariant with the camera pose. 
* The ablation study is not very informative, also in the context of the above point. Even the worst configuation here (“without blending”) already outperforms the baseline DINOv2 model by a significant margin. All other configuations lead only to a marginal decrease in the task accuracy. I would like to see the most basic configuration instead: a fixed teacher model, without masked upsampling and blending.
* Related to the above points, introduction stresses “a least-squares” compromise from previous work, citing multi-view inconsistency of the base feature representation. The argument seems to  imply that there is no feature averaging in this work. However, this is exactly what Eqs. (2,4) do. The only difference is that this averaging occurs within each segment, which is not particularly critical (as experiment in Tab. 5B shows)

### Questions
* What aspect of the framework design encourages improvement on view co-variant tasks (e.g. depth, surface normal estimation)?
* How would the most basic variant of the proposed framework (no mask-guided usampling, no blending, no EMA) perform?
* How does the approach work around the issues of a “least-squares compromise” and not “enforcing feature similarity at corresponding points” from previous work?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The method extends DINO-style self-distillation by integrating a feed-forward 3D reconstruction pipeline. Two context views are used to build a 3D Gaussian scene via a pretrained model (MVSplat), while the teacher’s 2D features are lifted onto this geometry to form 3D feature Gaussians. These are rendered from a novel viewpoint to generate a 3D-aware supervisory map for the student. Mask-aware feature lifting preserves semantic boundaries, and semantic blending regularizes noisy regions. The student, given only the target image, learns to match the rendered teacher features through a cross-entropy loss, with the teacher updated via EMA. This yields 2D features enriched with 3D geometric awareness.

### Strengths
- The method demonstrates clear performance gains over DINOv2, the baseline framework it builds upon.
- The use of a feed-forward 3D reconstruction module to inject geometric awareness into 2D foundation models seems a promising and timely research direction.

### Weaknesses
The evaluation setup is limited, as DINOv2 is seldom used directly for downstream tasks. It typically serves as a feature backbone for task-specific heads. Hence, direct comparison against DINOv2 provides a very limited insight. Demonstrating improvements when replacing DINOv2 with the proposed method within state-of-the-art pipelines (e.g., VGGT) would make the results substantially more compelling.

Many recent methods seem to be missing from the comparisons:
- DINOv3
- Sarıyıldız, M.B., Weinzaepfel, P., Lucas, T., de Jorge, P., Larlus, D. and Kalantidis, Y., 2025. DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 30084-30094).
- Govindarajan, H., Wozniak, M.K., Klingner, M., Maurice, C., Kiran, B.R. and Yogamani, S., 2025. CleverDistiller: Simple and Spatially Consistent Cross-modal Distillation. BMVC 2025 (on arxiv since March)
- Huang, X., Wu, J., Xie, Q. and Han, K., 2025. MLLMs Need 3D-Aware Representation Supervision for Scene Understanding. arXiv preprint arXiv:2506.01946.
- You, Y., Li, Y., Deng, C., Wang, Y. and Guibas, L., 2024. Multiview Equivariance Improves 3D Correspondence Understanding with Minimal Feature Finetuning. ICLR 2025

Minor thing:
- Section 3.3 closely resembles the feature lifting strategy used in OccamLGS; the authors should probably cite it.

### Questions
- How does the proposed approach perform when used as a drop-in replacement for DINOv2 within task-specific architectures such as VGGT?
- The evaluated downstream tasks (e.g., depth estimation) are interesting, but 2D foundation models are rarely applied to them directly. It would be helpful to also include results from task-specific state-of-the-art methods on the same benchmarks as reference to see where the proposed approach stands in comparison.
- How does the method compares to the missing baselines?

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
3

### Summary
This paper introduces SnD, a new pipeline to improve the 3D awareness of 2D foundation features. Similar to previous methods such as FiT3D, SnD uses 3DGS as 3D representation to improve 3D awareness. Instead of per-scene optimization, SnD uses feed-forward 3DGS (MVSplat) to improve efficiency. In addition, a student-teacher framework, similar to DINO, is used to distill the 3D awareness from teacher model. SnD outperforms baselines in various downstream tasks.

### Strengths
1. The paper is well-written and easy to follow.

2. Based on MVSplat, a feed-forward 3DGS method, SnD is more efficient than previous methods that require optimization. 

3. SnD outperforms baselines in various downstream tasks, including depth estimation, normal estimation, and semantic segmentation.

### Weaknesses
1. The authors should add an ablation study without student-teacher framework. For example, a potential experiment is finetuning a model with feature rendering loss, similar to Fit3D. Currently, it is unclear to me why student-teacher framework is necessary. 

2. The improvement that mask-aware feature lifting brings is not explicit.

3. Quantitative evaluation of multi-view feature correspondence should be added, instead of using visualization only.

### Questions
1. In Table 1, I suggest clarifying that SnD is finetuned on ScanNet++. 

2. Semantic mask: As shown in Fig. 11, the mask used is actually instance mask, instead of semantic mask (two monitors have different masks). This should be clarified in Sec. 3.3. 

3. Typos: 
    * L123: of are -> are
    * L294: teacher’s are -> teacher’s parameters are

### Soundness
3

### Presentation
3

### Contribution
2
