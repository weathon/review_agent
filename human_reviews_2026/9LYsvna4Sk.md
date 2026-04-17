# Unsupervised Representation Learning for 3D Mesh Parameterization with Semantic and Visibility Objectives

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Recent 3D generative models produce high-quality textures for 3D mesh objects. However, they commonly rely on the heavy assumption that input 3D meshes are accompanied by manual mesh parameterization (UV mapping), a manual task that requires both technical precision and artistic judgment. Industry surveys show that this process often accounts for a significant share of asset creation, creating a major bottleneck for 3D content creators. Moreover, existing automatic methods often ignore two perceptually important criteria: (1) semantic awareness (UV charts should align semantically similar 3D parts across shapes) and (2) visibility awareness (cutting seams should lie in regions unlikely to be seen). To overcome these shortcomings and to automate the mesh parameterization process, we present an unsupervised differentiable framework that augments standard geometry-preserving UV learning with semantic- and visibility-aware objectives. For semantic-awareness, our pipeline (i) segments the mesh into semantic 3D parts, (ii) applies an unsupervised learned per-part UV-parameterization backbone, and (iii) aggregates per-part charts into a unified UV atlas. For visibility-awareness, we use ambient occlusion (AO) as an exposure proxy and back-propagate a soft differentiable AO-weighted seam objective to steer cutting seams toward occluded regions. By conducting qualitative and quantitative evaluations against state-of-the-art methods, we show that the proposed method produces UV atlases that better support texture generation and reduce perceptible seam artifacts compared to recent baselines. We will make our implementation code publicly available upon acceptance of the paper.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The proposed method builds upon previous work, such as FlexPara and Nuvo, which address UV unwrapping. These methods propose a neural network optimization-based unwrapping approach with mapping cycles. The neural networks employed are neural fields similar to those utilized in NeRF. Compared to previous methods, the proposed approach incorporates more meaningful cut control based on visibility and semantics. This significantly improves the cut positions and generates more meaningful islands. For semantics, the authors propose using the shape diameter function (ShDF), while also demonstrating a version employing a semantic network. Regarding visibility-based cutting, they propose utilizing the ambient occlusion (AO) as a measurement. Both losses can be utilized jointly during the optimization process.

### Strengths
- The utilization of AO as a measurement to identify concealed cut areas is a novel approach. The authors’ integration of this information (soft-selection) into the optimization process is noteworthy.
- Parallel to the AO integration, the incorporation of part segmentation employing ShDF and Gaussian mixture models to learn the soft assignment is also interesting.
- While the SAMesh version appears more preferable to me from the supplementary materials, the application of ShDF yields remarkably convincing results, considering the simplicity of the technique.
- The authors comprehensively evaluate their method employing various metrics, demonstrating the enhanced part segmentation.

### Weaknesses
- The current level of UV unwrapping is still not on artist-levels, but it does make significant improvements over previous works.
- The chart layout is intentionally straightforward, but I could envision easy modifications to maintain a similar texture density.
- Overall, the method only adds a small delta on-top of the existing methods (Semantic and Visibility-based cutting).

### Questions
- Could you please provide the training duration per mesh? n Table 2, there is an inconsistency in the inference times. Since training is performed per asset, it should be factored in. FlexPara estimates times of approximately 400 seconds for a global unwrap and 1300 seconds for a multi-chart, which is comparable to Nuvo’s speed. I would anticipate that the method would require slightly longer training times.

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
3

### Summary
This paper proposes a fully differentiable and unsupervised framework for 3D mesh parameterization that jointly accounts for semantic consistency and visibility awareness.
The method introduces two key objectives:

(1) a semantic-aware objective, which leverages the Shape Diameter Function (ShDF) to segment the mesh into semantically meaningful regions and then learns UV parameterizations for each region in an unsupervised manner; 

(2) a visibility-aware objective, which employs a differentiable Ambient Occlusion (AO) loss to guide seam placement toward less visible areas, thereby improving texture continuity and perceptual quality.

Experimental comparisons with FlexPara, OptCuts, Blender SmartUV, Maya, and xatlas show improvements in semantic alignment, seam invisibility, and geometric distortion.

### Strengths
1.The paper is well motivated, aiming to address the challenges of 3D asset generation and to narrow the quality gap between generated assets and artist-created 3D models.

2.The use of off-the-shelf tools to obtain part segmentation information for UV parameterization is both reasonable and effective.

3.Leveraging geometric properties, such as Ambient Occlusion (AO), to determine UV cut seams in a differentiable manner is novel and demonstrates clear effectiveness.

### Weaknesses
1.The authors should clarify their specific contributions, as providing coarse semantic information and applying an alpha-cut strategy to optimize per-vertex or per-face labels have already been explored in prior 3D segmentation works. Moreover, the UV parameterization learning approach appears largely adapted from existing methods.

2.Since the semantic partitioning relies on off-the-shelf tools and involves hard segmentation of the mesh into parts before subsequent processing, it is difficult to regard the proposed method as a fully differentiable framework.

3.The proposed UV chart aggregator is straightforward but not very effective. It can lead to significant wasted space on UV maps and may compromise several desirable properties of UV parameterization, such as continuity and uniform distortion.

### Questions
The method for determining cut seams appears to be crucial in the visibility-aware UV parameterization, yet Eq. (5) does not clearly explain how the seams are effectively identified. Could the authors elaborate on how this formulation contributes to seam detection and clarify its effectiveness in guiding the visibility-based UV cutting process?

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
This paper presents an unsupervised, differentiable framework designed to automate 3D mesh parameterization (UV mapping), addressing this critical and often manual bottleneck in 3D content creation. The core contribution is augmenting standard geometry-preserving UV learning with two novel perceptual criteria: semantic awareness and visibility awareness.

The semantic-aware pipeline utilizes a partition-and-parameterize strategy, first segmenting the mesh into meaningful 3D parts using a Shape Diameter Function (ShDF), and then independently parameterizing and aggregating these parts into a unified atlas.

Conversely, the visibility-aware objective leverages ambient occlusion (AO) as a differentiable proxy for viewer exposure, actively steering the placement of cutting seams toward regions of low visibility. By minimizing this AO-weighted seam loss, the framework ensures that UV cutting seams are less likely to be observed, thereby reducing perceptible artifacts after texturing and rendering.

Overall, qualitative and quantitative comparisons confirm that the proposed method generates UV atlases that maintain low geometric distortion while achieving better semantic coherence and superior, perceptually favorable seam placement compared to existing baselines.

### Strengths
- The paper is easy to follow. The framework successfully transitions from the standard foundation of geometry-preserving mesh parameterization learning to its specific novel contributions: semantic awareness and visibility awareness.

- The research includes comprehensive comparisons against state-of-the-art baselines, encompassing both recent open-source learning methods (e.g., FlexPara, xatlas) and established industrial software tools (e.g., Autodesk Maya and Blender’s SmartUV)

- The proposed semantic-aware and visibility-aware parameterizations are effective. 

  - The semantic-aware pipeline generates practical, high-utility UV atlases by employing a partition-and-parameterize strategy based on the ShDF. This produces UV charts that align with meaningful 3D parts, making them cleaner, more interpretable, and directly beneficial for downstream tasks such as texture painting, editing, and transfer. 
  - The visibility-aware objective significantly improves perceptual quality by actively minimizing an (AO–weighted loss, successfully steering cutting seams toward less-exposed, more occluded regions. Using AO as a proxy supervision signal is a concise and elegant way to control visibility.

### Weaknesses
- The methodology’s initial phase, "geometry-preservation mesh-parameterization learning" (Sec. 3.2), adopts a network architecture based on the bi-directional cycle mapping backbone proposed in prior work (e.g., Zhao et al., 2025). This phrasing may lead readers to believe Stage I is a novel geometry parameterization method, rather than emphasizing that the true contribution lies in integrating semantic-aware and visibility-aware objectives as downstream tasks atop an existing geometric backbone. The manuscript could improve clarity by explicitly framing Stage I as an adoption of a known geometry-preserving backbone (e.g., FlexPara) and focusing the novelty on the downstream integration of the semantic and visibility objectives.
- The ablation comparing the Shape Diameter Function (ShDF) partitioner to the SAMesh baseline is confusing. Quantitatively, SAMesh output is treated as GT for semantic metrics (Hamming Distance and Rand Index). Yet, the ablations show that the proposed ShDF partitioner yields UV parameterizations that are geometrically superior (more conformal and more equiareal) than those using SAMesh. Since ShDF is primarily geometry-driven (local thickness), the paper lacks external validation against industry-standard or human-annotated semantic datasets (e.g., PartNet, PartNeXt) to confirm whether ShDF’s partitions align with true human semantic decomposition.

### Questions
- The final stage of the semantic pipeline employs an "intentionally simple and deterministic" aggregation and packing strategy, placing UV islands into a $G\times G$ regular grid. While this is simple and beneficial for texture editing and transfer workflows, the authors note it can be replaced by "more advanced packing solvers." This choice implies the final UV atlas likely suffers from suboptimal UV space utilization or packing efficiency—a critical metric in 3D production that is not reported in the quantitative analysis.

- Although the authors claim that adding perceptual objectives causes only "minor changes" in geometry-preserving metrics, the quantitative results show a noticeable drop in equiareality (area preservation) when the visibility objective is enabled—from 0.6759 (Base Neural UV Arch., i.e., FlexPara) to 0.6093 (Ours—Visibility-Aware Param). While the authors consider this “minor,” such degradation may be significant for applications requiring precise triangle area preservation (e.g., lightmap baking). The paper does not provide an in-depth analysis of whether this loss of geometric fidelity is the minimal necessary cost to achieve perceptually superior seam placement.

Could the authors provide more in-depth analysis and discussion on the above two points?

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
This work proposes a method for predicting a UV parameterization for a mesh that not only respects common geometric properties such as minimizing distortion and cuts, but also tries to keep semantically related components nearby in UV space and place cuts / seams in low visibility regions. To accomplish this, the method uses the ShDF to segment the mesh into semantic components and then uses ambient occlusion as a proxy for visibility. Each semantic segment is independently parameterized and then all segments are recombined and packed into a single atlas. The method is evaluated against several baselines both quantitatively with comparison figures and quantitatively using specific semantic and visibility metrics as well as standard distortion metrics. Ablations of several key method components are also included.

### Strengths
- This work identifies a gap in the current state of parameterization methods. It then addresses this by proposing a method producing parameterizations that group semantic components and place seams at low visibility regions.
- The soft seam membership introduced to make the cutting differentiable is an interesting contribution and can be used in future research aiming to take a learning based approach to UV parameterization.

### Weaknesses
- Distortion looks higher on this method in Fig 3 and is quantitatively worse in Tab. 1 and Tab. 2., indicating that this approach sacrifices distortion for semantics and less visible seams
- As noted in this paper, a good parameterization is subjective and varies a lot artist to artist. Given this, it would be beneficial to include a user study where actual 3D artists can evaluate the quality of the parameterizations.
- Quantitative metrics for seam visibility evaluate using the same ambient occlusion that this method optimizes for. While AO might be a good metric for visibility, it is likely not perfect in all situations and thus optimizing for it and then measuring on it feels unfair.

### Questions
- Based on my interpretation of eq 7 it seems the relationship is backward and the $AO_i$ should be $(1 - AO_i)$ in the numerator?
- Have authors explored using other modern semantic segmentation methods? I.e Partfield [1]? I see authors have ablated the use of SAMesh but perhaps Partfield would work better. I wonder if on harder shapes learning based segmentation methods would work better than ShDF. Could the authors provide more intuition for why the ShDF segmentations produced more conformal and equiareal parameterizations?

References: [1] Liu, Minghua, et al. "Partfield: Learning 3d feature fields for part segmentation and beyond." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.

### Soundness
3

### Presentation
2

### Contribution
3
