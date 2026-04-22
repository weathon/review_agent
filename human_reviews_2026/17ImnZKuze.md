# GSPlane: Concise and Accurate Planar Reconstruction via Structured Representation

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Planes are fundamental primitives of 3D sences, especially in man-made environments such as indoor spaces and urban streets. Representing these planes in a structured and parameterized format facilitates scene editing and physical simulations in downstream applications. Recently, Gaussian Splatting (GS) has demonstrated remarkable effectiveness in the Novel View Synthesis task, with extensions showing great potential in accurate surface reconstruction. However, even state-of-the-art GS representations often struggle to reconstruct planar regions with sufficient smoothness and precision. To address this issue, we propose GSPlane, which recovers accurate geometry and produces clean and well-structured mesh connectivity for plane regions in the reconstructed scene. By leveraging off-the-shelf segmentation and normal prediction models, GSPlane extracts robust planar priors to establish structured representations for planar Gaussian coordinates, which help guide the training process by enforcing geometric consistency. To further enhance training robustness, a Dynamic Gaussian Re-classifier is introduced to adaptively reclassify planar Gaussians with persistently high gradients as non-planar, ensuring more reliable optimization. Furthermore, we utilize the optimized planar priors to refine the mesh layouts, significantly improving topological structure while reducing the number of vertices and faces. We also explore applications of the structured planar representation, which enable decoupling and flexible manipulation of objects on supportive planes. Extensive experiments demonstrate that, with no sacrifice in rendering quality, the introduction of planar priors significantly improves the geometric accuracy of the extracted meshes across various baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
GSPlane injects 2D planar priors (SAM masks + Metric3Dv2 normals) into Gaussian Splatting by re-parameterizing planar Gaussians as convex combinations of three learned basis points; a Dynamic Gaussian Re-classifier (DGR) reverts false planar assignments; mesh layout refinement and an optional Supportive Plane Correction (SPC) yield cleaner, topology-consistent planar meshes while preserving NVS quality on ScanNetV2 and Tanks & Temples with large vertex reductions.

### Strengths
1. Elegant plane parameterization (weights over three basis points) that directly enforces coplanarity during training.
2. DGR automatically corrects misclassified planar Gaussians without hurting non-planar training.
3. Clear practical gains: big drops in planar/overall vertex counts with comparable or slightly better NVS metrics.

### Weaknesses
1. Dependence on off-the-shelf priors. Quality hinges on SAM/Metric3Dv2; the authors acknowledge the limitation and domain-shift sensitivity. 
2. Heuristic thresholds in DGR. The 5%/20% gradient rule is plausible but somewhat ad hoc; stability vs. missed corrections is not deeply analyzed.
3. Comparability caveat for GOF. For GOF they replace Marching Tetrahedra with TSDF fusion “to avoid mesh in actually empty areas,” which may confound apples-to-apples comparisons.

### Questions
1. Basis-point stability: How do you prevent degeneracy/collinearity of the three basis points per plane during training, and do you re-seed them if the plane rotates significantly?
2. Fairness of meshing choices: Can you report GOF results with both (a) original meshing and (b) TSDF, to isolate GSPlane’s contribution from mesher effects?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work focuses on improving surface reconstruction of Gaussian Splatting (GS) with structured plane representations. To this end, the method extracts plane priors based on monocular normal predictions and SAM model. Based on plane priors, the method defines structured plane representations by using a normalized linear combination of three non-collinear points. With this representation, the method introduces a dynamic Gaussian re-classifier to correct false-positive Gaussians based on the average gradients of planar Gaussians and non-planar Gaussians. In addition, the method mesh simplification based on the structured representation. The experiments show that the method can improve surface reconstruction of different GS methods for indoor scenes and outdoor scenes.

### Strengths
1. The method defines a structured plane representation to constrain the Gaussian points on the same plane, different from the squeezing Gaussians as planes in 2DGS and PGSR.
2. The introduced plane representation can help simplify meshes.
3. The method can improve different GS surface reconstruction on ScanNet dataset.

### Weaknesses
1. Althought the introduced plane representation can help simplify meshes, it needs to extract meshes by TSDF fusion first. This means that it cannot direclty extract lightweight meshes, and  aslo require large storage.
2. According to the experiments in Tables 1 and 2, the introduced mesh refinement improves surface reconstrucion a lot. However, the overall improments on Tanks and Temples (Table 2) are limited. Therefore, I wonder maybe some components do not work in Tanks and Temples dataset.
3. According to the right table in Figure 4, if SOTA GS methods direclty use monocular normal estimation as supervisions, their performance is very competitive with the proposed method. In this way, the advantages of the introduced structured plane representation seem somewhat weak.

### Questions
1. For Tanks and Temples dataset, the improvements of the method are very limited. It is better to show the performance without mesh refinement. This can help understand what components are most important in the method.
2. For the right table of Figure 4, if PGSR is combined with monocular normal supervision, how about the performance? I suspect the performance improvement of the method comes from the constraint of the monocular normal estiamtion.
3. It is better to compare the effieciency of different methods.
4. For the dynamic Gaussian re-classifier, accodring to the left table in Figure 4, its boost is very limited. I wonder if it works for other GS baselines. Moreover, in Line 235-236, how to determine the thresholds, 5% and 20%?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes GSPlane, a 3D reconstruction method that enhances the geometric accuracy and structural simplicity of reconstructed meshes, particularly in planar regions (e.g., floors, walls, tables). The main idea is to take plane detections from 2D methods to serve as priors in 3DGS. They also propose strategies such as a dynamic Gaussian re-classifier and supportive plane correction to further refine the results.

### Strengths
Reparameterizes Gaussian coordinates using basis points on detected planes by 2D methods, ensuring Gaussians adhere to planar constraints during training. 
Identifies and corrects misclassified planar Gaussians during training to improve planar integrity  (DGR). 
An application that preserves planar integrity when decoupling objects (SPC).

### Weaknesses
Very few visualizations. In all figures, I cannot find comparisons that presents in the same modality. For PGSR in the teaser, it present rendered image, but GSPlane presents rendered geometry. In Figure 3, other methods presents rendered geometry while +Ours presents rendered normal maps. I cannot tell the improvements. 

The accuracy of planar detection relies heavily on off-the-shelf models (e.g., SAM, Metric3Dv2). Errors in these models can propagate into the 3D reconstruction. 

The paper does not address how to extend the structured representation to non-planar surfaces, which remains an open and bigger challenge for detailed geometry reconstruction. 

The VGGT model gives a very smooth plane reconstruction, which is not compared. 

Overall, I cannot find a significant performance improvement over other methods in the current version.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2
