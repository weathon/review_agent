# ReCAP: Recursive Prompting for Self-Supervised Category-Level Articulated Pose Estimation from an Image

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 2, 8

## Abstract
Estimating category-level articulated object poses is crucial for robotics and virtual reality. 
Prior works either rely on costly annotations, limiting scalability, or depend on auxiliary signals such as dense RGB-D sensing and geometric constraints that are rarely available in practice. 
As a result, articulated pose estimation from a single RGB image remains largely unsolved.
We propose ReCAP, a Recursive prompting for self-supervised Category-level Articulated object Pose estimation from an image. 
ReCAP adapts a pre-trained foundation model using a Recursive Prompt Generator with residual injection, introducing less than 1\% additional parameters.
This mechanism enables parameter-efficient scaling through recursive refinement, while residual injection preserves token alignment under dynamic reconfiguration, yielding robust articulated-object adaptation.
To further resolve structural ambiguities, we introduce $\mathcal{X}$-SGP, a multi-scale fusion module that adaptively integrates semantic and geometric cues, an aspect often overlooked by geometry-centric approaches. 
Experiments on synthetic and real benchmarks demonstrate state-of-the-art monocular articulated pose estimation without requiring 3D supervision or auxiliary depth input. 
To the best of our knowledge, ReCAP is the first self-supervised framework to accomplish this task from a single image.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a self-supervised, single-image, category-level articulated object pose estimation framework that avoids depth/3D supervision. The method adapts VGGT with lightweight prompting, fuses semantic and geometric cues, predicts a dense point cloud, and canonicalizes it via a learnable category template to regress global and per-part poses. Experiments show competitive performance.

### Strengths
+ Proposed a self-supervised, single-image articulated pose estimation framework built on a frozen VGGT backbone, avoiding any ground-truth annotations while tackling a challenging setting.
+ Introduces a  prompt strategy for VGGT is meaningful.

### Weaknesses
+  While adapting VGGT may be reasonable, the reported ablations show only marginal gains; it remains unclear whether the proposed prompting/recursion is necessary versus simpler alternatives or no adaptation at all.
+  Benchmark coverage is limited and qualitative results focus largely on rotational joints; prismatic/mixed-DOF categories, heavy occlusions, and classes with larger intra-class variation are underexplored.
+  For symmetric objects, closed configurations, or low-texture surfaces, joint type/axis is not uniquely recoverable from a single image; results read as the most plausible hypothesis under shape/semantic priors rather than demonstrably identifiable solutions.
+ For symmetric shapes, near-closed poses, or texture-poor views, a single image does not uniquely determine the joint type or axis; the predictions read as prior-conditioned best guesses rather than uniquely identifiable solutions.

### Questions
+ How is the joint type obtained (assumed, predicted, or inferred)? 
+ Using a learnable category-level template with DCD may bias predictions toward an average shape, suppressing instance-level details and skewing axis estimation. Quantifying this effect is valuable.
+ Since core geometry comes from a frozen VGGT, to what extent do the gains stem from the backbone prior rather than the proposed self-supervised training?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces RECAP, a self-supervised method for single image category-level articulated object pose estimation. To tackle the depth uncertainty problem, the authors exploit a geometry foundation model to learn the corresponding complete point cloud for the input object, with the proposed recursive prompt for adapting articulated objects. Then an alignment method is used for optimizing the per-part 6D pose using the reconstructed point cloud into the RGB image.

### Strengths
1. The authors solve the pose estimation problem using an only RGB image, which is promising.
2. The technique presentation is sound and convincing.
3. Enough experiments are provided.

### Weaknesses
1.To address the depth missing problem, the authors employ a geometric foundation model for point cloud learning. However, the comparison of point cloud reconstruction with methods that do not utilize such foundation models is arguably unfair. Although the RECAP method introduces external knowledge for the pose estimation task, it only achieves marginal improvements.
2.Several relevant works on category-level articulation pose estimation are not adequately cited or discussed, such as R2-Art (AAAI 2025), U-COPE (ECCV 2024), and "Toward real-world category-level articulation pose estimation" (TIP 2022). Additionally, the well-established render-and-compare methodology, widely used for single-image pose estimation, is also overlooked in the discussion.
3.The authors utilize the OP-Align dataset as a benchmark; however, its scale is relatively limited, encompassing only four categories. Given that existing datasets contain over 2,000 objects across numerous categories, the selection of merely four categories appears insufficient.
4.The evaluation does not include two widely recognized articulation datasets—ArtImage and ReArtMix. The authors are encouraged to provide an explanation for this omission.

### Questions
please refer to weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents ReCAP, a self-supervised framework for category-level articulated object pose estimation from a single RGB image — a task that traditionally requires depth or multi-view supervision. ReCAP adapts a large geometry foundation model (VGGT) using a novel Recursive Residual Prompting mechanism, which refines prompts through iterative recursion and stabilizes them via residual injection, introducing less than 1% additional parameters. This enables parameter-efficient adaptation of rigid-object priors to articulated settings.

To address occlusion and symmetry ambiguities, the paper further introduces a Cross Semantic–Geometry Pyramid (X-SGP) module that hierarchically fuses semantic and geometric cues. Experiments on OP-Align, HOI4D, and PartNet-Mobility show that ReCAP achieves state-of-the-art performance among self-supervised methods and even surpasses some supervised RGB-D baselines.

### Strengths
* The proposed Recursive Residual Prompting (RRP) is a well-motivated and technically elegant solution to two core issues in applying prompt tuning to large geometry backbones such as VGGT: (1) limited capacity of shallow prompts to capture complex articulation patterns, and (2) instability caused by VGGT’s dynamic token reconfiguration. 
* The proposed Cross Semantic–Geometry Pyramid (X-SGP) effectively fuses semantic and geometric cues via adaptive FiLM modulation and multi-scale refinement. Ablation studies show clear performance drops when removing pyramid layers or FiLM components, confirming their necessity in handling occlusion, symmetry, and fine-grained part alignment.
* The authors provide quantitative ablations for recursion depth, prompt placement (input/output), and parameter scaling, showing that the recursive approach achieves comparable or better performance than multi-layer stacking with only 0.8% additional parameters. This level of analysis supports the soundness and reproducibility of the proposed design.

### Weaknesses
* The work thoughtfully adapts prompt tuning and DEQ-style recursion to articulated pose estimation, which is a valuable and nontrivial contribution. Still, the methodological core builds on established ideas, with innovation mainly in integration and application rather than new theoretical development.
* Despite its parameter efficiency (<1% trainable), the recursive refinement introduces noticeable latency (≈13 FPS vs. 41 FPS in baselines). Discussion of this trade-off or adaptive recursion strategies would improve clarity on practical feasibility.

### Questions
* The paper could better illustrate how recursive prompting reshapes geometric or semantic representations. For example, visualizing token attention or feature evolution across recursion steps would clarify how the mechanism contributes to improved articulation reasoning beyond empirical gains.

### Soundness
3

### Presentation
4

### Contribution
4
