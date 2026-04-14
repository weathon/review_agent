# View-Independent 3D Feature Distillation with Object-Centric Priors

- Decision: Reject
- Scores: 5, 5, 5, 6

## Abstract
Grounding natural language to the physical world is a ubiquitous topic with a wide
range of applications in computer vision and robotics. Recently, 2D vision-language
models such as CLIP have been widely popularized, due to their impressive capa-
bilities for open-vocabulary grounding in 2D images. Subsequent works aim to
elevate 2D CLIP features to 3D via feature distillation, but either learn neural fields
that are scene-specific and hence lack generalization, or focus on indoor room
scan data that require access to multiple camera views, which is not practical in
robot manipulation scenarios. Additionally, related methods typically fuse features
at pixel-level and assume that all camera views are equally informative. In this
work, we show that this approach leads to sub-optimal 3D features, both in terms
of grounding accuracy, as well as segmentation crispness. To alleviate this, we
propose a multi-view feature fusion strategy that employs object-centric priors to
eliminate uninformative views based on semantic information, and fuse features
at object-level via instance segmentation masks. To distill our object-centric 3D
features, we generate a large-scale synthetic multi-view dataset of cluttered tabletop
scenes, spawning 15k scenes from over 3300 unique object instances, which we
make publicly available. We show that our method reconstructs 3D CLIP features
with improved grounding capacity and spatial consistency, while doing so from
single-view RGB-D, thus departing from the assumption of multiple camera views
at test time. Finally, we show that our approach can generalize to novel tabletop
domains and be re-purposed for 3D instance segmentation without fine-tuning, and
demonstrate its utility for language-guided robotic grasping in clutter.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper addresses the limitations of 3D feature distillation in previous methods, which assumed all camera views are equally informative. To tackle this, it introduces MV-TOD, a synthetic dataset with detailed 3D masks and 6DoF grasp pose annotations for tabletop scenes. The paper then proposes DROP-CLIP, a method that employs object-centric priors for weighted multi-view feature re-projection. Additionally, to simplify the application for robotics, it introduces view-independent feature distillation, enabling the extraction of 3D open-vocabulary features from a single view. Experimental results demonstrate the method's effectiveness, including its utility in language-guided (open-vocabulary) robotic grasping.

### Strengths
1. The method demonstrates strong results in CLIP/DINO feature distillation and performs well on object-level segmentation.
2. The experiments are comprehensive, with sufficient ablation studies provided.
3. The paper is well-written and clearly communicates its ideas.

### Weaknesses
1. A broader set of baselines should be included in experiments, such as LERF [1], D3Field [2], and Semantic Gaussians [3].
2. The MV-TOD dataset, while useful, is synthetic and introduces a sim-to-real gap when using modified CLIP/DINO for dense feature extraction, which limits its impact.
3. For open-vocabulary grasping, comparing the proposed 3D representation to baselines like F3RM [4], LERF-TOGO [5], and GaussianGrasper [6] would help validate its advantages.
4. The novelty of the method may fall short for an ICLR submission, as it relies on projecting 2D CLIP or DINO features into 3D points via simple averaging, which is not a new approach.
5. The results from real-world scenes (Fig. 7) are not particularly impressive; comparisons with recent methods, such as Semantic Gaussians [3] and GaussianGrasper [6], would strengthen the evaluation.

[1] Kerr, Justin, et al. "Lerf: Language embedded radiance fields." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[2] Wang, Yixuan, et al. "D3Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation." arXiv preprint arXiv:2309.16118 (2023).

[3] Guo, Jun, et al. "Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting." arXiv preprint arXiv:2403.15624 (2024).

[4] Shen, William, et al. "Distilled feature fields enable few-shot language-guided manipulation." arXiv preprint arXiv:2308.07931 (2023).

[5] Rashid, Adam, et al. "Language embedded radiance fields for zero-shot task-oriented grasping." 7th Annual Conference on Robot Learning. 2023.

[6] Zheng, Yuhang, et al. "GaussianGrasper: 3D Language Gaussian Splatting for Open-vocabulary Robotic Grasping." arXiv preprint arXiv:2403.09637 (2024).

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a multi-view feature fusion strategy that employs object-centric priors to eliminate uninformative views based on semantic information, and fuse features at object-level via instance segmentation masks.

### Strengths
1. This paper addresses the issue of sub-optimal 3D feature distillation resulting from the assumption that all camera views are equally informative and presents view-independent feature distillation, allowing the extraction of 3D open-vocabulary features from a single view. 
2. This paper proposes DROP-CLIP, which utilizes an object-centric prior for weighted multi-view feature re-projection, enhancing the accuracy of feature extraction. 
3. A new dataset MV-TOD has been collected, which includes detailed 3D masks and 6DoF grasp pose annotations specifically designed for table-top scenes.
4. Experimental results validate the effectiveness, generalization and applicability of the proposed methods.

### Weaknesses
1. My main concern is the **novelty** of the proposed method. As far as the reviewer knows, the authors just replace the pixel-level features by object-level features through cropping the images. It seems that the contributions are mostly on the dataset and downstream applications, which is not enough for this venue.
2. It is beneficial to provide some comparisons with **more baselines** like sparsedff[1] or d3field[2].
3. The dataset was constructed in a simulated environment (Blender). Its **impact on real-world scenarios** is unknown. I suggest the authors provide clarification or experiments to demonstrate its generalization ability in real-world scenarios.



[1] Wang, Qianxu, et al. "Sparsedff: Sparse-view feature distillation for one-shot dexterous manipulation." *arXiv preprint arXiv:2310.16838* (2023).

[2] Wang, Yixuan, et al. "D $^ 3$ Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation." *arXiv preprint arXiv:2309.16118* (2023).

### Questions
1. The symbols in the paper are poorly represented. I suggest the author to simplify these symbols and remove some meaningless ones.
2. Some of the fonts in the tables and figures are too small to recognize, like Tab.1, Fig.2, Fig.3.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a method for fusing 2D semantic features from foundation models into 3D representations, applying this approach to downstream tasks in robotics, such as manipulation. The focus is on leveraging information across different views to maximize useful insights, while also addressing the fusion of priors in an object-centric manner. Additionally, the paper introduces a new dataset designed to tackle the challenges posed by cluttered scenes in real-world environments.

### Strengths
Two areas that are less well addressed in the literature include:

Semantics-informed View Selection: This approach utilizes an informative matrix to balance information across different views while accounting for uncertainty.

Object-wise Fusion and Features: The emphasis on object-centric features is potential to enhance generalization in complex scenes.

Table 1 provides an informative summary of the datasets.

### Weaknesses
My primary concern lies in the fact that feature distillation has been a popular topic in recent research, making it difficult to identify the paper's key contributions in this context. While the dataset is certainly valuable for the community, the fundamental challenge of achieving multi-view consistency when fusing 2D features into 3D representations remains a critical issue that the paper does not fully address.

The robotics experiments presented are interesting; however, they do not effectively demonstrate the performance of the proposed method. The research presented in this paper only finds the parts of interested objects. As noted in the discussion, the current manipulation pipeline appears to be limited to a two-stage open-loop demonstration.

### Questions
How can we ensure that the fused 3D features are consistent across multiple views?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel method for 2D->3D feature distillation using point cloud encoder, but focus on the multi-view fusion strategy. In particular, the paper is centered around the distillation of pretrained 2D CLIP models into 3D encoder, dubbed DROP-CLIP. Compared to existing multi-view fusion and per-scene optimization method, the proposed method does not require expensive optimization, and work with as few as a single RGB-D view. To facilitate the training of DROP-CLIP, the paper also uses blender to construct a large synthetic dataset termed MV-TOD for training the 3D encoder. Experiments show significant improvement of segmentation results on the MV-TOD dataset over existing baselines (OpenScene and OpenMask3D).

### Strengths
- **Technical contributions.** The paper proposes a series of techniques and datasets that are well-justified. To start with, viewpoint uncertainty is an important factor in 2D->3D distillation. Many existing methods suffer from inaccurate predictions caused by such uncertainty. The paper not only proposes a method to address this, but proposes an object-centric dataset that attempts to eliminate the bias of room-scale datasets. The individual components are also well-ablated in Tab. 2.
- **Good ablation and motivating applications.** The experiment sections ablate individual components well and show improvement over existing baselines on the proposed MV-TOD dataset. Though more real-world data experiments would be appreciated, the experiments seem to justify each component to some extent.

### Weaknesses
- **Unclear performance on real-world data.** Though the paper demonstrates that DROP-CLIP outperforms recent methods (OpenScene and OpenMask3D), the results were obtained on the synthetic MV-TOD dataset. For the experiments with real-world data in Tab. 4, the paper compares with relatively outdated methods. So the improvement of DROP-CLIP on real-world data seems a bit inconclusive.
- **Efficiency is unreported.** The paper provides a motivating application that uses DROP-CLIP for robotics grasping, which is good. However, efficiency is crucial for robotics applications. The paper is unclear as to whether the proposed method is real-time.
- **Unclear pipeline.** The paper does not seem to have a figure that shows how all components are connected. For example, L240-L242 mentions that sets of 2D segmentation masks are used. However, the method overview figure does not show how 2D masks are integrated, and there does not seem to be a description of how these masks are obtained (especially for real data) in the main paper.
- **Justification of the dataset contribution.** In the introduction, the paper discusses the necessity of constructing the MV-TOD dataset from synthetic data by comparing it to an existing room-level dataset. However, there is no quantitative evidence to support this. For example, how would the proposed component work if it is trained on an existing dataset and evaluated on the proposed dataset? Would joint-training improve the overall performance?
- **Discussion of related work.** The paper compares with several 2D->3D distillation methods. However, there are several more recent methods that are closely related to the paper. Specifically, [A] and [B] also distill 3D features, and [B] also uses 2D input masks to improve 3D segmentation results. Citing and providing differentiation with these work would benefit the presentation.

[A] Qin, Minghan, et al. "Langsplat: 3d language gaussian splatting." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[B] Qiu, Ri-Zhao, et al. "Feature Splatting: Language-Driven Physics-Based Scene Synthesis and Editing." ECCV. 2024.

### Questions
Reflecting on weaknesses above, my questions are

- Is it possible to include additional experiments on real-world data?
- What is the efficiency of the proposed method?
- Can the figures be improved to be more informative?
- Is it possible to do joint training on both MV-TOD and some other datasets to show possible improvement? If not, why?
- Discussion of related work.

### Soundness
3

### Presentation
3

### Contribution
3
