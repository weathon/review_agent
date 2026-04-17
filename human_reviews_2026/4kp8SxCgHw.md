# GaussianMorphing：Mesh-Guided 3D Gaussians for Semantic-Aware Object Morphing

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
We introduce GaussianMorphing, a novel framework for semantic-aware 3D shape and texture morphing from multi-view images. Unlike conventional approaches constrained to point clouds or correspondence-aligned untextured data, our approach leverages mesh-guided 3D Gaussian Splatting (3DGS) to achieve high-fidelity appearance and geometry representation. On the one hand, our unified mesh-guided Gaussian deformation strategy ensures geometrically consistent deformation by binding 3DGS points to reconstructed mesh patches while preserving texture fidelity through topology-aware constraints. On the other hand, the framework establishes unsupervised semantic correspondence by exploiting mesh topology as a geometric prior, while maintaining structural integrity through physically plausible point trajectory constraints. This integrated approach maintains both local geometric details and global semantic coherence throughout the morphing process without requiring labeled data. Experimental results show that GaussianMorphing outperforms prior 2D/3D morphing methods, with a color consistency ($\Delta E$) reduction of  22.2%  and an EI reduction of 26.2%  on our proposed TexMorph.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the task of semantic-aware 3D shape and texture morphing between source and target objects given only multi-view images, without requiring pre-aligned 3D models or labeled data. The core technical pipelines involves 1) reconstructing 3DGS representations from the input images, 2) extracting surface meshes to anchor the Gaussians via barycentric coordinates, 3) establishing unsupervised semantic correspondences using GCN, 4) learning a neural morphing flow for non-linear interpolation of geometry and appearance. Experimental results on TexMorph show  improved performance of color consistency error, edge integrity over baseline methods, with qualitative improvements in handling non-isometric deformations and better user study preferences.

### Strengths
This paper integrates meshes with 3D Gaussians, enabling semantic-aware morphing that bridges the gap between unstructured point-based representations and structured topology.

The proposed method exhibits outstanding results on the TexMorph benchmark, achieving state-of-the-art performance.

The paper introduces novel benchmarks and metrics (i.e. TexMorph, MSE-SSIM, ∆E, EI) that advance evaluation standards for 3D morphing

### Weaknesses
The mesh-anchored Gaussian binding using barycentric coordinates and normal offsets is very similar to prior works like Dynamic Gaussians Mesh, which introduces Gaussian/Mesh Anchoring for aligning Gaussians to mesh faces in dynamic scenes, and Mesh-based Gaussian Splatting, which defines Gaussians over meshes for deformation. The deformation via neural morphing flow also seems to be studied in MaGS

-Dynamic Gaussians Mesh: Consistent Mesh Reconstruction from Dynamic Scenes Liu et al.

-Mesh-based Gaussian Splatting for Real-time Large-scale Deformation Gao et al.

-MaGS: Reconstructing and Simulating Dynamic 3D Objects
with Mesh-adsorbed Gaussian Splatting

Ablation studies are not sufficient. Ablation studies only test mesh guidance and geometric distortion loss separately, but ignores the impact of ARAP energy or smoothness loss.

The dual-domain optimization is incremental, combining standard losses (geodesic distortion, ARAP) from prior shape interpolation works like NeuroMorph and Spectral Meets Spatial, without substantial novel formulations. I think this contribution is somewhat marginal.

The color initialization by averaging Gaussian RGB assumes uniform lighting, which may introduce biases in real-world varying illumination.

Some important technical details are missing. For example, the Correspondence Morphing Flow (Ψ) is very vaguely described as a neural network without any network architecture detail; the parameters like KNN value or epsilon for geodesic distance approximation in smoothness loss are omitted.

### Questions
The paper uses SuGaR which is based on Poisson reconstruction for mesh extraction, so it assumes watertight surfaces. Thus, will it fail for fragmented objects or open surfaces? This leads to inconsistent anchoring.

Do you rigorously verify if “semantic correspondences” are established in an unsupervised manner? as claimed in the paper? Can we integrate 2D image priors to enhance robustness to non-isometric deformations and improve semantic accuracy?

Can we add a perceptual loss (e.g., LPIPS) for color consistency, capturing higher-level texture semantics and leading to more visually pleasing transitions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a framework for 3D shape and texture morphing from multi-view images. The approach leverages mesh-guided 3D Gaussian Splatting (3DGS) to overcome the limitations of previous methods that rely on point clouds or pre-defined homeomorphic mappings. The core idea is a unified deformation strategy that anchors 3D Gaussians to reconstructed mesh patches, ensuring geometrically consistent transformations and preserving texture fidelity through topology-aware constraints. The framework also establishes unsupervised semantic correspondence using mesh topology and maintains structural integrity via physically plausible point trajectories. The method is evaluated on a new benchmark called TexMorph and shows improvements over existing 2D/3D morphing techniques.

### Strengths
1. The paper presents a new approach to 3D morphing by combining mesh-guided deformation with 3D Gaussian Splatting. 

2. The method incorporates semantic awareness by using mesh topology as a geometric prior, enabling more meaningful and coherent morphing results.

3. The method requires only multi-view images as input, reducing the need for high-quality 3D data or manual annotations.

4. The paper introduces a new benchmark (TexMorph) and evaluation metrics tailored for 3D morphing, allowing for a more thorough assessment of the proposed method.

### Weaknesses
1. The smoothness requirement in the loss function, particularly the "Appearance Consistency" term (Lsmooth), may lead to over-smoothed textures and loss of fine details. The paper acknowledges this to some extent, but it remains a significant concern.

2. While the method reduces the need for specialized 3D assets, the computational cost of generating the initial mesh-Gaussian representation and optimizing the morphing framework is relatively high, requiring significant GPU resources and time.

3. Although the method introduces several metrics, the qualitative results do not look that good.

### Questions
-How does the method handle significant topological changes during morphing (e.g., when objects merge or split)?

-What are the limitations of using a mesh as a guiding structure? Are there scenarios where the mesh might hinder the morphing process?

-How well does the method generalize to datasets with different characteristics than TexMorph? （e.g. with highly structured and detailed texture/appearance)

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
This paper introduces GaussianMorphing, a novel framework that unifies 3D geometry and texture morphing through a mesh-guided 3D Gaussian Splatting (3DGS) representation. The method anchors unstructured Gaussians to reconstructed mesh faces, integrating geometric and texture consistency via geodesic distortion and color-smoothness losses, achieving the balance between image-based pipelines and 3D-centric methods

### Strengths
1.The manuscript is well organized and easy to follow. The formulas and methodological details are clearly presented, making the technical contributions easy to understand.

2.The first-frame editing pipeline is a well-established paradigm in video editing, and I appreciate the exploration of extending this idea to 3D editing.

3.The presented editing results are promising and appear to be on par with the state-of-the-art performance of current 3D editing methods.

### Weaknesses
My concerning mainly lies on the setting. In my understanding, the main contribution of this paper is the view sampling issue. In another world, after optimally selecting the novel views, the rest processing is feed-forwarding the views to current 3D editing models. Current experiments seem to focus more on the comparison between the results with view expansion and that without view expansion to emphasis the effect of view expansion. In my opinion, the idea of 
using view expansion itself is not sufficiently novel, as it is a common sense in reconstruction. Instead, I would like to see the improvement of proposed view sampling strategy corresponding to the baseline random sampling or uniform sampling, presenting how and why the proposed sampling strategy outperforms a vanilla strategy. I would temporarily give a borderline reject rating and raise my score upon this contribution is well illustrated.

### Questions
See weaknesses. I would suggest the authors respond to the concern.

### Soundness
2

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
4

### Summary
The paper introduces GaussianMorphing, a novel framework for semantic-aware 3D shape and texture morphing that generates high-fidelity 3D outputs directly from multi-view images.

The core of the method is a hybrid paradigm that anchors unstructured 3D Gaussians to reconstructed mesh patches, utilizing the explicit mesh topology as a scaffold to guide geometrically consistent transformations while preserving texture fidelity. To handle complex transformations, the framework establishes unsupervised semantic correspondence using a Graph Convolutional Network (GCN) to capture local geometric context, thereby eliminating the need for labeled data or pre-aligned 3D assets. The process is governed by a dual-domain optimization strategy that employs geodesic-aware geometric distortion constraints ($L_{geo}$) and a texture-aware color smoothness loss ($L_{smooth}$) to ensure stable, structurally sound, and visually seamless transitions.

Through comprehensive experiments on the new TexMorph benchmark, GaussianMorphing substantially outperforms prior 2D and 3D methods, demonstrating superior structural consistency and texture preservation.

### Strengths
- GaussianMorphing introduces a novel hybrid paradigm that integrates 3D Gaussian Splatting (3DGS) with mesh-guided deformation. By using the mesh as a topological scaffold to anchor unstructured Gaussians, the method enables geometrically consistent transformations while preserving high-fidelity texture and appearance.
- GaussianMorphing achieves state-of-the-art performance on the newly proposed TexMorph benchmark, substantially outperforming existing 2D and 3D techniques. The approach demonstrates robust generalization across diverse sources, including complex synthetic models, real-world scanned objects, and in-the-wild photographs.
- The paper proposes unsupervised semantic correspondence and a dual-domain optimization strategy that combines geodesic-aware geometric distortion constraints with texture-aware color smoothness, ensuring stable, structurally sound, and visually seamless transitions.

### Weaknesses
While the final generation of the morphing sequence is fast (around 2 minutes), the initial setup and training phase suggest high computational demand. Generating the initial hybrid mesh-Gaussian representation takes about 1 hour for a typical object pair, followed by optimization requiring 500 to 1000 iterations. This multi-stage process indicates that the time and resource cost to enable morphing between a new pair of objects is still relatively high compared to some image-based methods (e.g., FreeMorph is tuning-free).

### Questions
- Does the current method require the source and target to be topologically homeomorphic when establishing correspondences? For example, would transformations like "apple to donut" pose inherent difficulties? If such cases can be accommodated, could the revision include corresponding qualitative and quantitative results?

- Does the network for extracting semantic-aware mesh correspondences require pretraining, or is it trained jointly with the morphing flow network?

- If providing the full code is not feasible, pseudocode would be sufficient to clarify the method. Including incomplete code in the supplementary material to suggest actual training code may cause confusion and is not needed.


- Yang et al., 2025: "Textured 3D regenerative morphing with 3D diffusion prior" has goals closely aligned with GaussianMorphing, namely achieving textured 3D morphing using 3D priors. As this work is closer in paradigm, it would be helpful to include more discussion.

- Does the method risk collapsing intermediate transformations to a trivial state before reaching the target? For example, could the process degenerate from the source to a gray spherical Gaussian representation (which might serve as a generic initialization) and then to the target? If this does not occur, it would be better to explain why the method avoids such collapse. In cases where the source and target are highly dissimilar in geometry and texture, such as morphing from a plant to an animal, does a similar issue arise?

### Soundness
3

### Presentation
3

### Contribution
3
