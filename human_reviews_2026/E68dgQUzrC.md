# Open-Set Semantic Gaussian Splatting SLAM with Expandable Representation

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
This work enables everyday devices, e.g., smartphones, to dynamically capture open-ended 3D scenes with rich, expandable semantics for immersive virtual worlds. While 3DGS and foundation models hold promise for semantic scene understanding, existing solutions suffer from unscalable semantic integration, prohibitive memory costs, and cross-view inconsistency. To respond, we propose Open-Set Semantic Gaussian Splatting SLAM, a GS-SLAM system augmented by an expandable semantic feature pool that decouples condensed scene-level semantics from individual 3D Gaussians. Each Gaussian references semantics via a lightweight indexing vector, reducing memory overhead by orders of magnitude while supporting dynamic updates. Besides, we introduce a consistency-aware optimization strategy alongside a Semantic Stability Guidance mechanism to enhance long-term, cross-view semantic consistency and resolve inconsistencies. Experiments demonstrate that our system achieves high-fidelity rendering with scalable, open-set semantics across both controlled and in-the-wild environments, supporting applications like 3D localization and scene editing. These results mark an initial yet solid step towards high-quality, expressive, and accessible 3D virtual world modeling. Our code will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper, "Open-Set Semantic Gaussian Splatting SLAM with Expandable Representation“, introduces a system designed to enable everyday devices like smartphones to capture and reconstruct 3D scenes with rich, open-set semantics. The core idea is to integrate an expandable semantic feature pool with a Gaussian Splatting SLAM (GS-SLAM) framework. This feature pool decouples scene-level semantics from individual 3D Gaussians, using lightweight indexing vectors to reduce memory overhead and support dynamic updates of semantics. The system also incorporates a consistency-aware optimization strategy and a Semantic Stability Guidance mechanism to improve cross-view semantic consistency.

### Strengths
The paper presents a compelling and timely contribution at the intersection of dense SLAM, 3D Gaussian splatting, and open-vocabulary semantic understanding. 

The work demonstrates high originality through several key innovations: 

1. Expandable Semantic Feature Pool: Rather than embedding high-dimensional semantic features directly into each Gaussian (which is memory-prohibitive), the authors propose a shared, dynamic, and expandable semantic feature pool. Each Gaussian references semantics via a lightweight indexing vector. This design is both novel and pragmatic—it decouples scene-level semantics from per-point storage, enabling scalability and dynamic updates. 

2. Open-Set Semantic Integration in SLAM: While prior SLAM systems typically handle closed-set semantics (e.g., fixed object categories), this work enables open-set semantic understanding—supporting queries like “fruit in plastic bag” or “xbox”—by leveraging foundation models (e.g., CLIP) within a real-time SLAM pipeline. This bridges a critical gap between foundation model capabilities and real-world 3D reconstruction. 

3. Consistency-Aware Optimization: The paper introduces a dual mechanism—an Intra-Inter Semantic Consistency Objective (via contrastive learning) and Semantic Stability Guidance (via cosine-similarity-based reweighting)—to enforce cross-view and temporal semantic coherence. This addresses a fundamental limitation of naively lifting 2D foundation model outputs into 3D without considering geometric consistency.

### Weaknesses
1. While the paper claims to reduce memory overhead "by orders of magnitude" (line 037) through the expandable semantic feature pool, the memory footprint analysis in Table 5 and Table 10 still shows significant memory usage, particularly as the semantic feature dimension ($D_s$) or pool size ($L$) increases. For real-time applications on "everyday devices (e.g., smartphones)" (line 025), the current memory requirements might still be prohibitive.

2. The runtime analysis in Table 8 shows that the proposed method incurs an "approximately 8% – 21% additional time overhead" compared to SplaTAM and LoopSplat. While this might be acceptable for some applications, achieving truly real-time performance on mobile devices for complex scenes with continuous semantic updates remains a challenge. The paper mentions that the "additional computational overhead stems from semantic assignment from Fp" and the consistency objectives.

3. The paper states that the system "demonstrates limited robustness in highly dynamic or large-scale (e.g., city-level) environments" (lines 1150-1151). This is a significant limitation for a SLAM system designed for "in-the-wild 3D scenes" (line 026). While the Semantic Stability Guidance mechanism is introduced to mitigate semantic ambiguity, a more thorough analysis of its effectiveness in highly ambiguous or dynamic scenarios would be beneficial.

4. The system relies on pre-trained 2D foundation models like CLIP [11] or SAM [12] for semantic distillation. While this is a common approach, the performance and generalizability of the system will be inherently tied to the capabilities and biases of these underlying models. The paper mentions that the framework is not dependent on "any specific semantic extraction method" (line 826), but the choice of these models can still impact the quality of the semantic representations.

### Questions
See the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a Gaussian-based SLAM framework that integrates open-vocabulary semantic features derived from pretrained visual-language models. It introduces an updating mechanism for semantic features across frames, aiming to improve temporal consistency and robustness in semantic mapping. The method is evaluated on multiple RGB-D datasets, showing improvements in both geometric accuracy and semantic segmentation performance.

### Strengths
- Leveraging pretrained models for enhancing scene-level semantic awareness is meaningful and practically relevant for scalable 3D mapping.

- The integration of semantic information into a Gaussian-based SLAM framework is valuable and timely, addressing the growing need for open-vocabulary 3D understanding on real-time systems.

- The experimental section covers multiple datasets and tasks， typically three in-the-wild scenes captured by everyday devices are interesting.

- The visualization in the paper is clear and informative, effectively illustrating the semantic reconstruction results and helping understand the pipeline of the proposed method.

### Weaknesses
- The quantitative comparison in Table 1 lacks several recent SOTA RGB-D SLAM baselines such as MonoGS, Gaussian-SLAM, RTG-SLAM, and SplatSLAM. It is highly recommended to include these baselines to ensure a fair, comprehensive, and up-to-date evaluation of the proposed module's performance.

- The organization of the paper needs substantial improvement. Essential information about how semantic features are extracted, processed, and integrated into the pipeline is relegated to Appendix A.1, while the main body spends excessive space on general preliminaries and related work (Eq. 1–5, 10–11). As a result, readers struggle to understand the core contribution without repeatedly consulting the supplementary material. The authors should move the key descriptions of the semantic feature pipeline, backbone choice, 2D-to-3D mapping, update rules, and training losses into the main Method section, and condense redundant preliminary equations.

- The use of pretrained semantic features from CLIP and DINOv2 is no longer novel, as several recent works, such as SemGauss-SLAM and OVO-SLAM, have already demonstrated open-vocabulary mapping with similar embeddings. The only distinctive component here is the recurrent update mechanism applied to explicit per-map semantic features. However, the motivation for updating explicit feature representations across frames is not clearly justified. Updating/Memorization is intuitive for latent representations that capture temporal dependencies, but less so for fixed explicit map representations, as per pixel level, their cross-frame correspondence is not recurrently updated but shifted from adjacent pixels in a continuous trajectory.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
2

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
This paper proposes an open-set semantic SLAM system based on 3D Gaussian Splatting. The key innovation is the introduction of an expandable semantic feature pool, which decouples high-level semantics from individual Gaussians. This design significantly reduces memory usage and allows for dynamic integration of new semantic concepts during SLAM operation. To address the inconsistency issue from 2D foundation models, the authors also propose a consistency-aware optimization strategy and a semantic stability guidance mechanism. Extensive experiments on both synthetic and real-world datasets demonstrate the system's advantages in rendering quality, tracking accuracy, and open-set semantic understanding tasks, such as 3D localization and scene editing.

### Strengths
The core idea of a learnable and expandable semantic feature pool is novel and clever. It effectively separates the scene-level semantics from the geometric representation, enabling efficient and scalable open-set learning.  

The paper is structured logically, and the framework is explained step by step, making it understandable.   

This work pushes the boundary of semantic SLAM from closed-set to open-set, which is a crucial step towards general-purpose 3D scene understanding. The ability to run on commodity hardware and support applications like 3D editing greatly enhances its practical value.

### Weaknesses
While the comparison with existing SLAM methods is comprehensive, I feel the comparison with state-of-the-art open-set 3D understanding methods (e.g., OpenMask3D, OpenScene) in terms of pure semantic segmentation accuracy is somewhat lacking.   

The runtime analysis in the supplement shows a non-negligible overhead. Although the authors attribute this to the baseline, a more in-depth discussion on how to optimize the efficiency of the semantic module itself would be helpful for practical deployment.  

The threshold parameters for the pool expansion (e.g., the similarity thresholds in Algorithm 1) seem crucial but their selection process is not deeply analyzed.

### Questions
1. The expansion factor n and the threshold for determining "empty slots" seem to be empirical. Could you discuss the sensitivity of the model's performance to these hyperparameters? Is there a risk of the pool expanding too aggressively in very large-scale scenes?
﻿
2. In the semantic stability guidance, how is the "object that appears for the first time" specifically identified? Is this based purely on the lack of correspondence in previous frames?
﻿
3. The paper demonstrates excellent results on in-the-wild data. Could you comment on the system's performance in highly dynamic environments where objects move frequently? Does the semantic pool and consistency mechanism handle such cases robustly?

### Soundness
3

### Presentation
4

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
The paper introduces an open-set semantic 3D Gaussian Splatting (3DGS) framework designed to enhance scene reconstruction and semantic segmentation under open-set and real-time conditions. The proposed method integrates semantic understanding into the Gaussian Splatting pipeline by associating per-Gaussian semantic embeddings and enabling online label adaptation. The system claims to handle unknown categories and domain shifts, and to improve both reconstruction quality and scene-level segmentation.

Experiments are conducted mainly on the Replica dataset, showing improved reconstruction quality and semantic accuracy compared to several baseline methods. The results demonstrate promising performance in synthetic indoor environments, suggesting potential for robust open-world mapping.

### Strengths
1. Good presentation and writing quality.
The paper is clearly structured, logically consistent, and easy to follow. Figures are visually clean and the methodology is well explained.

2. Strong results on benchmark datasets.
The proposed approach achieves good reconstruction and segmentation results on Replica and other small-scale datasets, showing the method’s effectiveness in controlled settings.

3. Direct and effective idea.
The proposed framework is conceptually straightforward yet functional. It extends 3D Gaussian Splatting toward semantic understanding and contributes a practical perspective to open-set semantic reconstruction.

### Weaknesses
1. Overreliance on synthetic data.
Using Replica as the main evaluation dataset is limiting. Since Replica is synthetic and lacks real-world sensor noise, it is less meaningful for evaluating SLAM or localization robustness. The paper would be stronger with experiments on real-world semantic datasets such as ScanNet++, SemanticKITTI.

2. Missing comparisons with recent baselines.
Several recent 3DGS-based SLAM and semantic mapping methods (e.g., MonoGS, S3PO-GS, SEGS-SLAM) are not included in comparisons. Without these baselines, it is difficult to gauge the real advancement of the proposed approach relative to the current state of the art.

3. Limited improvement from semantics to pose estimation.
While the paper introduces semantic components, the pose estimation accuracy remains very close to previous methods, suggesting that semantics may not significantly contribute to the geometric optimization process.

### Questions
The questions are the same as weakness. This paper presents a solid and well-written approach that meaningfully extends semantic understanding into 3D Gaussian Splatting. However, the experimental evaluation is too narrow and overly dependent on synthetic data, which weakens claims of real-world robustness and open-set generalization. Including stronger baselines and more diverse datasets would substantially improve the paper’s credibility.
Typos:
1. In Figure 2, the label “Ground Truth RGB Frame” appears misplaced — please verify and correct this annotation.

### Soundness
3

### Presentation
3

### Contribution
2
