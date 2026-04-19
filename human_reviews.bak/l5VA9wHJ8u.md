# GeoSplating: Towards Geometry Guided Gaussian Splatling for Physically-based Inverse Rendering

- Decision: Reject
- Scores: 5, 5, 6, 5

## Abstract
We consider the problem of physically-based inverse rendering using 3D Gaussian Splatting (3DGS) representations (Kerbl et al., 2023b). While recent 3DGS methods have achieved remarkable results in novel view synthesis (NVS), accurately capturing high-fidelity geometry, physically interpretable materials, and lighting remains challenging, as it requires precise geometry modeling to provide accurate surface normals, along with physically-based rendering (PBR) techniques to ensure correct material and lighting disentanglement. Previous 3DGS methods resort to approximating surface normals but often struggle with noisy local geometry, leading to inaccurate normal estimation and suboptimal material-lighting decomposition. In this paper, we introduce GeoSplatting, a novel hybrid representation that augments 3DGS with explicit geometric guidance and differentiable PBR equations. Specifically, we bridge isosurface and 3DGS together, where we first extract isosurface mesh from a scalar field, then convert it into 3DGS points and formulate PBR equations for them in a fully differentiable manner. In GeoSplatting, 3DGS is grounded on mesh geometry, enabling precise surface normal modeling, which facilitates the use of PBR frameworks for material decomposition, achieving excellent decomposition performance, especially for reflective cases. This approach further maintains the efficiency and quality of NVS from 3DGS while ensuring accurate geometry from the isosurface. Comprehensive evaluations across diverse datasets demonstrate the superior efficiency and competitive inverse rendering performance of GeoSplatting compared to state-of-the-art inverse rendering baselines.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper is dedicated to addressing the long-standing challenges of inverse rendering using a 3DGS-based approach. The authors first introduced geometric priors to 3DGS through Flexicubes, achieving impressive geometric quality. They then adopted the standard rendering equation constraints to solve for BRDF. To further enhance the rendering quality and improve the robustness of geometry, the authors introduced residual color and a second-stage optimization that increases geometric freedom. The experimental results demonstrate that GeoSplatting achieves excellent NVS and geometric quality in both synthetic and real-world scenes, along with relatively good BRDF estimation results.

### Strengths
- This paper is well-written and easy to understand.
- The geometric quality and NVS quality are excellent, and introducing geometric priors and guidance through Flexicubes is novel.
- The training time is impressive, ranking among the fastest in its type of methods.

### Weaknesses
1. Missing reference:
   - 3DGSR: Implicit Surface Reconstruction with 3D Gaussian Splatting, Lyu et al. 3DGSR is very similar to GSDF, with the biggest difference being that the former is based on 3D-GS, while the latter is based on Scaffold-GS.
   - The related work section of this paper lacks a discussion on neural field-based inverse rendering. Before the emergence of methods based on 3D-GS, many technically innovative approaches were not adequately discussed. Examples include NeRFactor, PhySG, NVDiffrec, InvRender, NeRO and NVDiffrecMC.
2. Dataset choice on the evaluation of BRDF estimation and relighting. In lines L401-L405, the authors used the Synthetic4Relight dataset from InvRender to demonstrate the advantages of GeoSplatting. However, this dataset is not challenging because there are no shadows in the input images, masking the long-standing issues in inverse rendering related to shadow and BRDF decoupling. Using the vanilla NeRF dataset would be more convincing. For instance, in Figure 15, the hotdog scene from NeRF, which contains strong shadows, would be a better choice than the trick scene in InvRender where lighting intensity has already been reduced.
3. Questions regarding the evaluation of NVS in inverse rendering. The authors dedicated a significant portion of the paper to demonstrating the advantages of GeoSplatting in NVS (e.g., Figure 6, Table 1). This is commendable, as it highlights GeoSplatting's fidelity in reconstruction. However, I can't help but wonder—how important is NVS really in the context of inverse rendering? If NVS is truly a metric worth comparing in IR, why not use methods that don't require PBR, such as vanilla 3D-GS? In most cases, PBR constraints significantly reduce the quality of NVS. From my perspective, the most important metrics in IR are the quality of BRDF and relighting. Given that the reconstruction fidelity of base models is already sufficient today, NVS doesn't seem to be a metric that articles in the IR field should emphasize to this extent.
4. Poor decomposition results. The biggest issue with GeoSplatting is its poor decomposition quality, which is a common problem among 3D-GS-based methods. For example, in Figure 7, there is still a significant amount of shadow baked into the albedo and roughness, reflecting GeoSplatting's inability to distinguish between shadows and roughness, even though the scene in Figure 7 has minimal shadows. While ambient lighting shows improvements compared to NVDiffrec, Relightable-GS, and TensoIR, it still falls short of InvRender's level. The same problem appears in Figures 12 and 14. In contrast, Relightable-GS provides more Blender- and relighting-friendly decomposition results. Furthermore, GeoSplatting still suffers from significant hue discrepancies, which affect relighting quality. These issues are amplified in real scenes, as shown in Figure 9, where the quality of albedo and roughness becomes unacceptable (scan 65).
5. Some minor questions. In Table 1, there might be issues with the SSIM for 3D-GS and LPIPS for 2DGS. Additionally, under typical circumstances, the training time for 3D-GS at a resolution of 800x800 should not exceed 10 minutes. Since 2DGS modifies the EWA used in 3D-GS, its computational efficiency will inevitably be slower than that of 3D-GS under the same conditions, likely by around 40%. And why the training of Mip-NeRF only cost around 2.5h, which is impossible on a single RTX 4090 GPU.

### Questions
Are there some typo errors in the title? 
- GeoSplating (-> GeoSplatting): Towards Geometry Guided Gaussian Splatling (-> Splatting) for Physically-based Inverse Rendering

### Soundness
3

### Presentation
4

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
The paper introduce GeoSplatting, a novel hybrid representation that augments 3DGS with explicit geometric guidance and differentiable PBR equations. 
Contributions:
1. extract isosurface mesh from a flexicube, then sample 3D gaussian points on triangle faces.
2. apply PBR equations in a deferred rendering manner.

### Strengths
1.The paper presents a novel pipeline which extract issosurface from flexicube and then convert to 3DGS for rendering.
 2. The experiment results show good performance in various datasets.

### Weaknesses
1. Many typos.
- in the title "geosplating", "splatling"
- equation 2: ","
- line254-255: "rasterizationto"
- Figure14(ours,Relit2)
2. lack of novelty: the physically-based rendering after rastization, sample 3D gaussian on triangle face, shift gaussian position on surface, these are all proposed by previous methods
3. I think the geometry compare in Table3 in unfair, as GeoSplatting only achieves 1 out of 4 best quality. And the only best scene (spot) is too simple, which can be represent by the 96^3 grid, but for a complex scene, such as the lego in Figure10, it's obvious that the geometry is not good at all.

### Questions
1. how to compute the normal map in stage2 (Figure10)

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
This paper proposed a novel 3DGS based inverse rendering framework, which rely on mesh-based representation as a geometry guidance. The starting point of this paper is that 3DGS cannot reconstruct accurate geometry, which is crucial for physically-based rendering. On the technical level, the main contribution of this paper is a new 3D Gaussian construction method, which aligns the gaussian primitives with the mesh by sampling 3D Gaussians on the vertices and triangle faces of the mesh.

### Strengths
This paper focuses on how to improve the quality of geometric reconstruction of 3DGS so that it can better meet the needs of physically-based rendering. The proposed novel isosurface guided gaussian splatting pipeline can accurately reconstruct the surface and normals of objects, thereby achieving high-quality inverse rendering results with satisfactory training time. In addition, even if the PBR part of the method is removed, the MGadapter proposed in the paper can also be considered as a new 3DGS construction method that can improve the geometric representation ability of 3DGS.

### Weaknesses
- The proposed method uses flexicubes as geometry guidance and requires object mask loss as supervision during training, which limits the method’s to the object level.
- For indirect lighting, the author only considered using MLP to estimate inter-reflection, but did not consider ambient occlusion.
- The quality of roughness reconstruction is not very satisfactory. This may be because the method does not take occlusion into account, so the material cannot be accurately estimated for the shadowed parts, which is also confirmed by the visualization results in the paper.
- Language issues. There are many spelling and grammatical errors in the article, even in the title of the article.

### Questions
- Please provide more details on the second stage optimization. Does the second stage optimization use the pruning and densification strategy in the vanilla 3DGS or just optimize the existing 3D Gaussian?
- Why can jointly training 3DGS and isosurface improve the quality of inverse rendering? Are there any experimental results to prove this? In addition, the CD socre of the mesh in Table.3 seems to be inferior to the result of directly extracting the mesh using flexicubes. Does this mean that joint training affects the reconstruction quality of the geometry?
- Given that indirect lighting is modeled as an attribute of each Gaussian, which represents the inter-reflection under the lighting conditions corresponding to the training stage, Is it possible to construct indirect lighting during relighting?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces an approach, GeoSplatting, which combines 3DGS with explicit geometric guidance and PBR. Firstly, the authors propose to extract meshes by utilizing Flexicubes. Then, the paper introduces the MGadapter, which constrains gaussians on the mesh surface. GeoSplatting incorporates a physically-based rendering framework that leverages the split-sum model to efficiently compute high-order lighting effects. This allows for fast rendering speeds and improved material and lighting disentanglement.

### Strengths
This paper makes full use of 3D Gaussian to optimize mesh, and propose a densification strategy. They introduce physics-based rendering equation to replace the implicit SH functions representation, which has good performance on the inverse rendering task.

### Weaknesses
1. The main contribution of this paper is the proposal of a hybrid representation combining mesh and Gaussian, but the discussion on mesh-based methods is not sufficiently thorough. Some omitted references are listed below, some of which are highly relevant. The authors need to state the main differences and comparisons.
   - DreamMesh4D: Video-to-4D Generation with Sparse-Controlled Gaussian-Mesh Hybrid Representation 
   - Mesh-based Gaussian Splatting for Real-time Large-scale Deformation 
   - High-Quality Surface Reconstruction using Gaussian Surfels
   - Direct Learning of Mesh and Appearance via 3D Gaussian Splatting
2. The comparison in terms of geometry is insufficient. The authors claim that they can reconstruct "precise" normals, which is inappropriate. Besides, here is insufficient experimental evidence to support this claim. The comparative experiments in Table 3 are too few, and the method proposed in this paper performs well only on the spot model. Including an evaluation metric for normal consistency might be better.
3. The impact of resolution on both the results and speed in the Flexicube method has not been discussed. 
4. The citation formats for references within the text are improper, which causes reading difficulties.

### Questions
1. I have some doubts about the efficiency, as compared to 3DGS, this method requires using an MLP to learn and also needs to optimize the SDF field to extract the mesh. Could the authors please explain the reasons for the improvement in time efficiency?
2. For Eq(1), What does a heuristic function mean? So, when $\mu$ is determined, are $\mathbf{S}$ and 
$\mathbf{R}$ essentially fixed?
3. The authors' experiments indicate that stage 2 is crucial for improving NVS performance, but this part does not show a significant difference compared to 3DGS.

### Soundness
2

### Presentation
2

### Contribution
2
