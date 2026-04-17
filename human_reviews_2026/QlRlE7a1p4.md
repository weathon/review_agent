# FullPart: Generating each 3D Part at Full Resolution

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 8

## Abstract
Part-based 3D generation holds great potential for various applications. Previous part generators that represent parts using implicit vector-set tokens often suffer from insufficient geometric details. Another line of work adopts an explicit voxel representation but shares a global voxel grid among all parts; this often causes small parts to occupy too few voxels, leading to degraded quality.
In this paper, we propose \textit{FullPart}, a novel framework that combines both implicit and explicit paradigms.
It first derives the bounding box layout through an implicit box vector-set diffusion process, a task that implicit diffusion handles effectively since box tokens contain little geometric detail.
Then, it generates detailed parts, each within its own fixed full-resolution voxel grid. Instead of sharing a global low-resolution space, each part in our method—even small ones—is generated at full resolution, enabling the synthesis of intricate details.
We further introduce a center-point encoding strategy to address the misalignment issue when exchanging information between parts of different actual sizes, thereby maintaining global coherence. Moreover, to tackle the scarcity of reliable part data, we present \textit{PartVerse-XL}, the largest human-annotated 3D part dataset to date with 40K objects and 320K parts. Extensive experiments demonstrate that FullPart achieves state-of-the-art results in 3D part generation. 
Code, model, and dataset are available at https://fullpart3d.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
FullPart: Generating each 3D Part at Full Resolution presents a novel framework for part-based 3D generation that addresses a critical limitation in existing methods: the resolution bottleneck for small parts. The authors identify that prior work, whether using implicit latent tokens or explicit voxel grids, forces all parts to share a single global representation space. This causes small but complex parts to be allocated insufficient resolution, leading to a loss of geometric detail.

Furthermore, the authors introduce PartVerse-XL, a large-scale, high-quality dataset of 40K 3D objects and 320K human-annotated parts, to address the scarcity of reliable training data for part-level generation. Extensive experiments show that FullPart achieves state-of-the-art performance, particularly in generating plausible geometries for small and occluded parts.

### Strengths
1. Introduction of a High-Quality, Large-Scale Dataset: PartVerse-XL

The creation and release of PartVerse-XL is a significant contribution that addresses a major roadblock in the field.

Addressing a Critical Data Scarcity: The paper correctly identifies that existing 3D datasets lack high-quality, semantically consistent part-level annotations. Artist-created metadata is often noisy and inconsistent, hindering the training of robust models. The two-stage pipeline of automated pre-segmentation followed by expert human refinement ensures both high quality and semantic consistency. This is a costly but necessary step to create a reliable benchmark. PartVerse-XL provides a foundational resource that will likely accelerate future research in 3D part generation, much like how datasets like ImageNet propelled 2D vision. By training and evaluating on this dataset, FullPart establishes a strong, reproducible baseline for the community.

2. A Well-Designed and Effective Hybrid Pipeline

The core pipeline of FullPart is a thoughtfully engineered solution that effectively combines the best of two worlds. The pipeline is cleverly designed to use the right representation for the right task:

Stage 1 (Implicit for Layout): Bounding boxes contain minimal geometric detail but require an understanding of global structure and inter-part relationships. Implicit diffusion models are perfectly suited for this "conceptual" generation task.

Stages 2 & 3 (Explicit for Geometry): Generating high-fidelity geometry demands precise spatial control. By allocating a dedicated voxel grid to each part, FullPart bypasses the fundamental resolution constraint of shared-grid methods. The ablation study in Figure 6(c) powerfully demonstrates the dramatic improvement this brings, especially for small parts like thin chair legs.

### Weaknesses
1. Significant Architectural Similarity to Preceding Work

While the pipeline is effective, its core structure bears a strong resemblance to contemporaneous work, particularly Co-Part (Dong et al., 2025), which may dampen its perceived novelty.

Parallels with Co-Part: Co-Part also proposes a multi-stage, diffusion-based framework for part generation that utilizes a layout stage followed by geometry generation. Both methods employ transformer architectures with intra-part and inter-part attention mechanisms to maintain coherence.


2. Insufficient Qualitative Comparison with Key Part-Based Baselines

The experimental section, while including quantitative metrics, lacks the qualitative depth needed to fully convince the reader of its superiority over other part-aware generators.

Sparse Visual Comparisons: Figure 4 provides only a limited set of visual comparisons against PartCrafter and OmniPart. To truly showcase its advantages, the paper needs a more extensive gallery of side-by-side comparisons, especially on challenging cases.

Failure Cases of Baselines: It would be highly informative to include examples where methods like OmniPart fail dramatically (e.g., a chair where the legs are voxelized into blobs) and show how FullPart succeeds.

Missing Baseline: Given the noted similarity, a direct qualitative comparison with Co-Part is crucial but is absent. Showing a visual comparison would allow for a direct assessment of whether the full-resolution grid leads to tangibly better results than Co-Part's approach.

User Study for Coherence: A user study evaluating the global coherence and structural plausibility of the assembled objects compared to other part-based methods would provide stronger evidence than metrics like Part-CD alone.

3. Under-Exploration of the Editing Application

The editing application presented in Figure 7 is promising but is only briefly mentioned. The paper would be strengthened by a more thorough exploration of this capability. For instance, it could demonstrate more complex edits like part replacement, rotation, or deletion, and quantitatively evaluate the preservation of unedited parts. How does this editing capability compare to other editable part-based generators? A discussion or comparison on this front would highlight a key practical advantage of the proposed pipeline.

### Questions
1. Significant Architectural Similarity to Preceding Work
2. Insufficient Qualitative Comparison with Key Part-Based Baselines
3. Under-Exploration of the Editing Application

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes FullPart, a 3D part generation framework integrating implicit and explicit paradigms. It first generates bounding box layouts via implicit box vecset diffusion, then creates each part in an independent full-resolution voxel grid, and uses a center-corner encoding strategy to maintain global coherence. It also introduces PartVerse-XL, a large human-annotated 3D part dataset. Experiments show FullPart achieves state-of-the-art 3D part generation results.

### Strengths
1. It identifies the problem that the resolution given to small parts is limited in 3D part generation by explicit voxel representation. And solves this problem by generating each part in an independent full-resolution voxel grid.
2. It creatively proposes a new positional embedding strategy to maintain global coherence in part generation with independent voxel grids.
3. The paper introduces PartVerse-XL, a large human-annotated 3D part dataset, which is a valuable resource for the research community.
4. The figures are clear and attractive, effectively illustrating the framework and results.

### Weaknesses
1. It looks like generating each part in full-resolution voxel grids will significantly increase the computational cost of training and inference. However, the paper does not provide any analysis or discussion on the computational efficiency of FullPart compared to existing methods.
2. The evaluation is only conducted on the PartVerse-XL dataset, which is introduced by the authors themselves. It would be more convincing if the authors could also evaluate their method on other established 3D part generation datasets (or synthesize from 3D part understanding datasets) to demonstrate its generalizability.

### Questions
1. Is there any innovation in the layout generation stage? Can you introduce how prior work handles layout generation and what are the differences compared to your method?
2. How does the computational cost (in terms of training time, inference time, and memory usage) of FullPart compare to existing 3D part generation methods that use explicit voxel representations?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents FullPart, a framework for part-level 3D object generation in which each part is produced at full resolution. The main contribution lies in the integration of implicit and explicit representations:
1. An implicit vecset-based diffusion model learns the layout of parts,
2. An explicit voxel-based representation generates complete high-resolution 3D parts, and
3. The resulting coarse 3D parts are refined into textured meshes.

The authors also introduce PartVerse-XL, a new dataset for part-level 3D generation. Compared with PartCrafter and OmniPart, FullPart demonstrates superior geometric fidelity and structural coherence. Because each part is generated at full resolution, the overall 3D objects contain richer details than those produced by TRELLIS or Direct3D-S2. Ablation studies confirm the effectiveness of Center-Corner Encoding, human-refined annotations, and the Per-Part Full-Resolution Grid.

### Strengths
1. The combination of implicit and explicit representations is well-motivated and effectively leverages the strengths of both paradigms.
2. The construction of the PartVerse-XL dataset is a valuable contribution that can advance research in part-level 3D generation.
3. The proposed Center-Corner Encoding mechanism is conceptually simple yet empirically effective.

### Weaknesses
1. Section 2.2 discusses mainly recent (2025) part-level 3D generation works. The related work could be more comprehensive by including earlier implicit Tri-plane–based approaches, such as Frankenstein [Yan et al., 2024] and StdGen [He et al., 2024].

### Questions
1. The Related Work section could be expanded to discuss a broader range of part-level 3D generation methods, emphasizing the types of representations they employ (e.g., implicit, voxel-based, point-based, or hybrid).
2. Add a section on the use of large language models (LLMs).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces FullPart, a novel framework for high-resolution part-aware 3D generation. This framework overcomes the resolution and coherence limitations of previous approaches by combining both implicit and explicit paradigms. Specifically, FullPart consists of three stages: (1) implicit layout generation for spatial arrangement; (2) explicit structure generation at full resolution for fine geometric details; (3) refinement for realistic appearance. Besides, the center-corner encoding mechanism is proposed for global part coherence. To support this framework, this work further introduce PartVerse-XL, the largest human-annotated 3D part dataset with semantically consistent part labels and textual descriptions. Experiments show that FullPart achieves state-of-the-art performance in both part- and object-level evaluations, outperforming existing methods such as PartCrafter, OmniPart, and TRELLIS.

### Strengths
1. The proposed per-part full-resolution generation paradigm is technically sound and yields satisfactory results. As shown in Figures 4 and 5, this method can improve the generation performance of both part-level and object-level. Besides, the proposed center-corner encoding mechanism utillizes a unified positional encoding strategy to preserve global spatial coherence across parts of different scales, effectively addressing the spatial discrepancy between different parts.

2. This work presents the largest human-annotated 3D part dataset, including 40K high-quality 3D objects and 320K semantically consistent 3D parts. The scale and diversity of this dataset will facilitate future research on part-aware 3D generation.

3. The paper provides quantitative and qualitative performance comparisons as well as a comprehensive ablation analysis. Experimental results (Table 1, Figure 4, Figure 5) demonstrate the superior performance of FullPart compared to previous part-level and object-level 3D generation methods.

4. The paper is clearly written and well-organized, making it easy to follow.

### Weaknesses
1. Generating each part within a full-resolution voxel grid substantially increases memory usage and computational costs, particularly for complex objects with numerous parts. The paper does not include an analysis of computational efficiency or stress testing in this regard.

2. The test set is limited to 100 manually selected untrained objects, which may not comprehensively evaluate the model's generalization capability and overall performance.

3. Quantitative ablation analysis results are missing.

4. This paper lacks a discussion of its limitations and failure cases.

### Questions
The sequential sampling strategy for generating over 30 parts appears promising. Could the authors provide results to demonstrate whether the structural coherence or overall geometric quality diminishes as the number of parts increases?

### Soundness
4

### Presentation
3

### Contribution
3
