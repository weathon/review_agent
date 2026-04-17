# HART: Human Aligned Reconstruction Transformer

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
We introduce HART, a unified framework for sparse-view human reconstruction. Given a small set of uncalibrated RGB images of a person as input, it outputs a watertight clothed mesh, the aligned SMPL-X body mesh, and a Gaussian-splat representation for photorealistic novel-view rendering. Prior methods for clothed human reconstruction either optimize parametric templates, which overlook loose garments and human-object interactions, or train implicit functions under simplified camera assumptions, limiting applicability in real scenes. In contrast, HART predicts per-pixel 3D point maps, normals, and body correspondences, and employs an occlusion-aware Poisson reconstruction to recover complete geometry, even in self-occluded regions. These predictions also align with a parametric SMPL-X body model, ensuring that reconstructed geometry remains consistent with human structure while capturing loose clothing and interactions. These human-aligned meshes initialize Gaussian splats to further enable sparse-view rendering. While trained on only 2.3K synthetic scans, HART achieves state-of-the-art results: Chamfer Distance improves by 18–23% for clothed-mesh reconstruction, PA-V2V drops by 6–27% for SMPL-X estimation, LPIPS decreases by 15–27% for novel-view synthesis on a wide range of datasets. These results suggest that feed-forward transformers can serve as a scalable model for robust human reconstruction in real-world settings. Code and models will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
HART represents a strong empirical contribution through a well-engineered combination of recent achievements on geometric and human reconstruction. It’s elegant and practical to integrate point-map prediction, occlusion-aware geometry completion, and SMPL-X alignment under a transformer framework.

### Strengths
1. The integration of VGGT-based multi-view fusion with per-pixel SMPL-X attribute prediction (tightness vectors and body-part labels) is an interesting adaptation of general 3D reconstruction transformers to the human domain.
2. The paper is clearly structured, with explicit loss formulations, architecture diagrams, and references to code/model release. These increase its reproducibility and transparency.

### Weaknesses
1. Although cross-domain generalization is demonstrated, the model is trained entirely on synthetic THuman 2.1 data. How does the model perform when trained or fine-tuned on real-world multi-view captures? Are there domain adaptation or photometric normalization strategies considered?
2. The method heavily depends on VGGT and DPSR. Beyond combining existing modules, what new representational capability does HART itself introduce? Could the same gains be achieved by tuning VGGT + DPSR with human-specific priors?
3. The paper does not provide a detailed sensitivity study on the loss weights or robustness to view number reduction (e.g., from 4 to 2 views).
4. How sensitive is HART to small pose misalignments or temporal inconsistencies (e.g., slight motion between views)? Could it handle unsynchronized or monocular sequences?
5. Despite being a feed-forward system, the Gaussian splatting stage still involves iterative optimization. Could a fully feed-forward rendering head be learned to replace this step for true real-time NVS?

### Questions
1. How does the model perform when trained or fine-tuned on real-world multi-view captures? Are there domain adaptation or photometric normalization strategies considered?

2. Besides the existing modules, what new representational capability does HART itself introduce? Could the same gains be achieved by tuning VGGT + DPSR with human-specific priors?

3. Could a fully feed-forward rendering head be learned to replace this step for true real-time NVS?

### Soundness
3

### Presentation
3

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
The paper introduces HART, a transformer-based framework for sparse-view human reconstruction (clothed mesh, SMPL-X estimation, novel-view synthesis) with occlusion-aware DPSR and residual normal learning.

### Strengths
1. One strength of HART is its unified transformer-based framework, which jointly outputs watertight clothed meshes, aligned SMPL-X body meshes, and Gaussian-splat representations for novel-view rendering from sparse uncalibrated RGB images—avoiding fragmented workflows of prior methods that handle these tasks separately. It achieves SOTA results (e.g., 18–23% CD improvement in clothed mesh reconstruction) across datasets like THuman 2.1 and DNA-Rendering, even with training on only 2.3K synthetic scans .

2. Another strength lies in its practical optimizations for human-specific challenges: it uses occlusion-aware DPSR with a 3D U-Net to recover complete geometry in self-occluded regions (a limitation of general 3D reconstruction backbones like VGGT) and leverages residual normal learning (with Sapiens priors) to enhance surface detail. These tweaks, though built on existing modules, better adapt feed-forward transformers to sparse-view human reconstruction scenarios .

### Weaknesses
1. HART’s novelty is not in new conceptual ideas, but in systematically integrating and adapting existing modules for sparse-view human reconstruction: it combines transformer architecture, occlusion-aware DPSR, and residual normal learning (no new network/reconstruction paradigm); applies Gaussian splatting (mainstream rendering) with human mesh constraints (no new rendering method); tailors pre-trained VGGT/DINOv2 (for general tasks) to human-specific tasks (no new extractors/encoders), focusing on cohesive pipeline optimization.

2. The method heavily relies on external components: removing the Sapiens model (for base normal prediction) results in blurrier surfaces, and disabling the indicator grid refinement leads to incomplete geometry in self-occluded regions, showing its weak independence from these specific auxiliary modules

3. HART is trained solely on 2,345 synthetic scans from the THuman 2.1 dataset, which lacks the diversity of real-world human variations (e.g., plus-size bodies, sheer fabrics, heavy coats, or diverse ethnic features). This limits the validity of claims about "robust generalization to real-world settings," as the training data does not fully represent real-world complexity.

4. While HART uses RANSAC and PnP to estimate camera parameters (instead of assuming centered principal points), the paper does not quantify the accuracy of these estimated parameters (e.g., comparison with ground-truth camera poses on test datasets). This omission hides potential biases—if camera estimation is inaccurate under certain conditions (e.g., extreme poses), it could confound the evaluation of reconstruction/rendering performance.

### Questions
Please refer to the weakness part.

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
The paper introduces a feed-forward transformer that takes a set of >=3 uncalibrated RGB images as input, and produces a watertight clothed mesh, aligned SMPL body mesh, and 3DGS representation for photorealistic novel-view rendering. The transformer design is initialized from VGGT, with DPT heads for residual normals (on top of Sapiens), pointmaps (which are combined with normal predictions for differentiable Poisson surface reconstruction and clothed mesh reconstruction), and SMPL tightness vectors (as in ETCH) as well as body part label maps (which are together decoded into SMPL-X meshes). The method is trained on 2.3k human meshes from THuman2.1 dataset. 2D Gaussian surfels can be optionally outputted for novel-view synthesis.

### Strengths
- Developing feed-forward VGGT-style architectures for clothed human reconstruction from multiple views is an important and interesting problem.
- The method is straightforward and well-motivated. The key introduced components (normal residual prediction and indicator grid refinement) help improve the quality of the geometry according to an ablation study
- The method produces better results than baselines such as VGGT, MAtCha, PuzzleAvatar, LaRa, etc.

### Weaknesses
- The reconstructed meshes lack very fine-grained details, such as detailed clothing textures, fingers, and hair. In the conclusion, the authors claim this is due to limited indicator grid resolution. Fundamentally, despite predicting residuals on top of a Sapiens normal prior, it seems the geometry quality is slightly better than VGGT but not substantially better.
- The impact of predicting residual normals on top of Sapiens and indicator grid refinement seems qualitatively very small, at least according to Figure 9 and Table 8. It is not clear to me whether the impact is significant.
- The method requires >=3 uncalibrated human images captured in the same body pose, which may be difficult to obtain in the wild outside of synchronized, sparse-view camera capture setups. Extending the method to dynamic monocular videos may be an interesting avenue for future research

### Questions
- Which dataset and sequences are the ablation studies in A.7 performed on?
- Regarding the statistical significance of the ablation studies in A.7, is it possible to show per-sequence ablation results or error bars with multiple randomly-selected view sets per scene, to confirm whether the proposed design choices yield consistent improvements on every sequence?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes HART, a unified feed-forward transformer framework for sparse-view human reconstruction, which  simultaneously predicts clothed human geometry, body pose, and GS. The work builds upon  VGGT and extends it to human-centric reconstruction by introducing multiple prediction heads and an occlusion-aware differentiable Poisson reconstruction (DPSR) module. The experiment shown improvements against baselines in geometric, rendering and pose estimation.

### Strengths
- The transformer based feed-forward pipeline should be much higher efficient compared to optimization- or diffusion-based methods；
- It is reasonable to achieve various human-centric tasks, like pose, pointmap or geometry and potential tracking, in a unified model.

### Weaknesses
1. I think the claim of  “scalable model” in abstract is not proper, because hart is trained with 3d scans dataset.
2. The performance is not impressive for me, especially the geometry and smplx results. The human mesh in fIg.3 and video demos, are over-smoothed and on the same par with Vggt predictions. The challenges of  pose estimation tasks mainly lies in hand region and in-the-wild challenging cases. However, this work dose not preform very well. 
3. I have great concerns about the normal predictions of DPT heads, even if it is designed for dense prediction tasks. It will be better to include some normal outputs   before/after DPT refinement.  
4. Another concern is the accuracy of the PnP-predicted camera poses. It is critical, as the estimated poses are used in both DPSR  and GS fitting, where accumulated errors could degrade overall performance. 
5. The comparisons are not sufficient. although this work focus on mv setting, the comparisons should include single view based methods, like SiTH and PSHuman. With more input views, this work should perform better than them.

### Questions
as listed in weakness.

### Soundness
2

### Presentation
2

### Contribution
2
