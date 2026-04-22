# Mesh Splatting for End-to-end Multiview Surface Reconstruction

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Surfaces are typically represented as meshes, which can be extracted from volumetric fields via meshing or optimized directly as surface parameterizations. Volumetric representations occupy 3D space and have a large effective receptive field along rays, enabling stable and efficient optimization via volumetric rendering; however, subsequent meshing often produces overly dense meshes and introduces accumulated errors. In contrast, pure surface methods avoid meshing but capture only boundary geometry with a single-layer receptive field, making it difficult to learn intricate geometric details and increasing reliance on priors (e.g., shading or normals). We bridge this gap by differentiably turning a surface representation into a volumetric one, enabling end-to-end surface reconstruction via volumetric rendering to model complex geometries. Specifically, we soften a mesh into multiple semi-transparent layers that remain differentiable with respect to the base mesh, endowing it with a controllable 3D receptive field. Combined with a splatting-based renderer and a topology-control strategy, our method can be optimized in about 20 minutes to achieve accurate surface reconstruction while substantially improving mesh quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel framework that bridges the gap between volumetric and surface-based 3D reconstruction methods. The core idea is to soften a base mesh into multiple semi-transparent, differentiable layers, effectively transforming it into a pseudo-volumetric representation while preserving the explicit structure of the mesh. These softened layers are rendered through a splatting-based differentiable renderer and optimized end-to-end with image-level supervision. By combining volumetric gradient propagation with explicit topology control, the method achieves high-quality surface reconstructions with fewer vertices and significantly reduced training time compared to existing volumetric or mesh-based approaches.

### Strengths
1. The proposed concept of differentiably softening meshes into volumetric layers is both innovative and practical. It effectively bridges the gap between surface- and volume-based approaches, enabling volumetric supervision while preserving explicit mesh controllability and topology refinement.
2. The paper demonstrates convincing quantitative and qualitative results on the DTU and BlendedMVS benchmarks, showing clear visual and numerical improvements over prior methods.
3. The overall presentation is clear, well-structured, and easy to follow.

### Weaknesses
1. Since the proposed framework claims to reconstruct intricate geometric details, I recommend including comparisons on the Ship or Ficus scenes from the NeRF Synthetic dataset, as these cases involve thin structures that are particularly challenging to recover accurately.
2. The overall novelty of the framework appears slightly limited, as the pipeline is largely built upon DMTet and splatting. Prior approaches such as VMesh [1] and Radiance Surfaces [2] have also explored incorporating volumetric supervision into surface-based representations.
3. The method still relies on tetrahedral grids, which may restrict its scalability and generalization to large-scale or unbounded scenes, potentially limiting its broader applicability.

[1] Guo, Y. C., Cao, Y. P., Wang, C., He, Y., Shan, Y., & Zhang, S. H. (2023, December). Vmesh: Hybrid volume-mesh representation for efficient view synthesis. In SIGGRAPH Asia 2023 Conference Papers (pp. 1-11).

[2] Zhang, Z., Roussel, N., Muller, T., Zeltner, T., Nimier-David, M., Rousselle, F., & Jakob, W. (2025, August). Radiance surfaces: Optimizing surface representations with a 5D radiance field loss. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (pp. 1-10).

### Questions
1.  The paper primarily evaluates object-centric datasets, while scene-level reconstructions are only briefly discussed in the ablation section. For larger-scale environments (e.g., Mip-NeRF360), it would be helpful to clarify how the authors adapt or extend the DMTet initialization to achieve more stable and accurate reconstruction at scene scale.
2. Since the framework relies on DMTet for base mesh reconstruction, exploring FlexiCubes as an alternative could potentially enhance the quality and robustness of the initial mesh. Such a comparison or discussion would further strengthen the technical completeness of the work.

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
The authors propose a method for reconstructing 3D shapes from multi-view images. They introduce a pseudo-volumetric representation called a soft mesh, assigning per-vertex opacity and features to enable alpha blending. Furthermore, they use DMTet to initialize mesh topology and continuously remesh the shape to improve topological stability. Experiments show that the proposed method achieves high quality shape reconsturction on DTU and BlendedMVS datasets.

### Strengths
Strengths
1.The paper shows that optimizing the proposed soft mesh outperforms optimizing a single layer.
2.The reconstruction is both fast and memory-efficient.

### Weaknesses
Weeknesses:
1.The performance advantage of the soft mesh is not well justified. While the authors claim that its pseudo-volumetric nature facilitates optimization, an alternative is to convert the base mesh into an SDF and train with NeuS or VolSDF, which remain differentiable with respect to the mesh. The proposed soft mesh seems to approximate this idea primarily for faster rendering. Intuitively, the SDF-based variant should be weaker than standard NeuS/VolSDF because it is constrained by the topology of the base mesh and thus sacrifices flexibility; nevertheless, the reported results indicate the soft mesh surpasses NeuS. Please explain the mechanisms behind this improvement.
2.The procedure for optimizing DMTet in the initial 5,000 iterations is not described. Please elaborate on the optimization objectives, and constraints used during this phase.
3.The explanation of the alpha weights is insufficient. The method adopts VolSDF’s mapping, which was originally used to compute attenuation coefficients; the authors should justify its suitability for alpha weights, additionally clarify whether the expected depth or the most visible point lies on the base mesh.
4.The paper does not report first-stage reconstruction quality. Including quantitative metrics for this stage would clarify the effectiveness of the subsequent stage.

### Questions
The author should provide more explanation. I am open to adjusting my rating since representing 3D scenes with multi-surface is critical and might be a potential research direction in the future, although the performance of the proposed method is poor.

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
*Motivation:*
- volumetric methods produce overly dense sampling and can be hard to mesh
- surface methods do not req meshing but make it hard to model intricate geometric details

*Method:*
- authors introduce a "soft mesh" representation that comprises multiple semi-transpatent
mesh layers that allow for underlying base mesh to be optimized
- this mesh representation is optimized via "differentiable Mesh Splatting" - a differentiable
rasterization algorithm that supports transpatent triangles, directly supervised by photometric loss. The final color is obtained by passing a set of interpolated (via barycentrics) features to an MLP and integrated via volumetric rendering.

*Evaluation:*
- Method performs favorably compared to several strong baselines (SuGaR, 2DGS, etc)

### Strengths
*Clarity:*
- The method description is clear and experimental evaluation is overall well-documented.

*Originality / significance:*
- Proposed approach is interesting as it can be seen as a hybrid formulation
between volumetric and surface-based representations, and allows for direct
mesh extraction (which is very useful e.g. for physics simulations).

*Evaluation:*
- Method performs favorably with respect to several strong baselines on
surface reconstruction task.
- The mesh splatting method seems more efficient than existing
differentiable rendering approaches (nvidiffrast).

### Weaknesses
*Clarity/Motivation:*
- The motivation for the approach is not particularly clear - authors point out to
issues with volumetric rep-s and mesh rep-s, and use the concept of receptive field - which is
not clearly defined. My best guess to what they mean is: meshes are typically optimized per-vertex - and thus the corresponding parameterization has a large number of effective degrees of freedom, and thus. However, there is a number of very established parameterizations (control points, ARAP, spectral methods) that do not share this weakness. Similarly, additional
reguralization terms and multi-node constraints (e.g. local smoothess) effectively act
as a way to restrict the "receptive field" (aka reduce effective degrees of freedom).

*Method Limitations:*
- As indicated by authors themselves (Sec 4.4), the method is severely limited by the
quality of the underlying base mesh - so it would not work well in settings with poor quality
of base meshes obtained by off-the-shelf mesh reconstruction. Because of this, the representation and algorithm introduced in the paper itself can be considered more
of a mesh refindement strategy than a standalone algorithm.
- (minor) 20 minutes per scene does not seem like performance suitable
for some real-world applications.

*Novelty:*
- (minor) The idea of using a soft mesh is not particularly new: e.g. [Esposito'2024] introduces
a very similar representation. And more generally, it can potentially be seen as a variant of control point parameterization or neural cages [Yifan'2020].
- (minor) MeshSDF [Remelli'2020] introduces a differentiable iso-surface extraction method, which allows differentiating through meshing. This sounds like a relevant related work.

*Experimental Evaluation:*
- A reasonable set of baselines is used for volumetric reconstruction, but potentially it
would also be interesting to compare against other differentiable rendering frameworks with "standard" differentiable rendering (nvdiffrast or DRTK) and an off-the-shelf mesh parameterization (e.g. mesh + smoothness).
I do appreciate the performance comparison with nvdiffrast-based iterative rendering implementation, but I wonder if there is a way to dissect
quality wrt to a) mesh representation b) rendering algorithm.
- It is not fully clear why the set of methods is different between Table 1 and Table 2.
- I understand that authors focus specifically on mesh reconstruction, but since the method
actually uses volumetric rendering and produces opacity and color, it seems natural to conduct
evaluation in terms of rendering quality - this might also bring up potential issues on complex
geometries for which meshes are a poor representation. I believe it would be helpful for the readers to understand if the method is also useful as a NVS approach.

### Questions
1. Is the "receptive field" a common terminology in the context of geometry representations?
- L034-035: "have a large receptive field along rays" - unclear what is meant here?
- L037: "... only boundary geometry without "
Both remarks are a bit confusing (and both abstract and intro contain the exact same wording which does not make it any more clear in intro) because some volumetric surface representation (e.g. SDF) also in a way only model boundary / surface.
Why relying on priors is a inherently bad thing? And what exactly is meant by "shading or normals"? Does this mean that one has to introduce an image formation model for the underlying
mesh representation to be optimizable? Clarification of this would be appreciated.

2. What is the main bottleneck in the speed/memory wrt to e.g. Gaussian splatting?
Authors did allude that implementation can be improved, but it would be helpful
to understand if there are any fundamental constraints? E.g. since there is a need for sorting triangles - potentially one might have performance issues on object borders/scenes where large
number of triangles fall in the same pixel.

3. Why not have the same set of methods in evaluation on different benchmark datasets (Table 1 vs Table 2)?

### Soundness
3

### Presentation
2

### Contribution
2
