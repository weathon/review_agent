# Radiant Triangle Soup with Soft Connectivity Forces for 3D Reconstruction and Novel View Synthesis

- Decision: Reject
- Scores: 2, 6, 8, 2

## Abstract
We introduce an inference-time scene optimization algorithm utilizing triangle soup, a collection of disconnected translucent triangle primitives, as the representation for the geometry and appearance of a scene. Unlike full-rank Gaussian kernels, triangles are a natural, locally-flat proxy for surfaces that can be connected to achieve highly complex geometry. When coupled with per-vertex Spherical Harmonics (SH), triangles provide a rich visual representation without incurring an expensive increase in primitives. We leverage our new representation to incorporate optimization objectives and enforce spatial regularization directly on the underlying primitives. The main differentiator of our approach is the definition and enforcement of soft connectivity forces between triangles during optimization, encouraging explicit, but soft, surface continuity in 3D. Experiments on representative 3D reconstruction and novel view synthesis datasets show improvements in geometric accuracy compared to current state-of-the-art algorithms without sacrificing visual fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a new idea of using triangles as the primitive to replace 3D Gaussians for the splatting. This method adopts a soft triangle representation to allow gradients to propagate to triangle positions and also considers the connectivity of triangles. The method is evaluated on the DTU dataset for surface reconstruction and the Mip-NeRF-360 dataset for novel view synthesis. The results demonstrate better geometry quality than Triangle splatting on the DTU dataset but worse rendering quality than triangle splatting on the novel view synthesis.

### Strengths
The idea of associating different triangles seems to be interesting and improved performance of geometry reconstruction is demonstrated.

### Weaknesses
1. The idea of making rasterization of triangles differentiable has already been studied in previous differentiable renderers like SoftRasterizer, and so on.
2. Why learning such connectivity is useful is not discussed clearly, which is the main contribution of the proposed method. I'm not sure in any cases, we need to connect the surface for a better rendering quality.
3. The search for connected triangles is costly. In the supp, the runtime for the DTU scene is 1.5h, which is much slower than baselines like GoF, PGSR, and so on.


In summary, the paper is not well motivated with new techniques. The proposed triangle connection seems to have an obvious weakness in terms of computation inefficiency. The results (for both geometry and NVS tasks) are not impressive enough either.

### Questions
I have no other questions than the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Radiant Triangle Soup (RTS), a novel 3D scene representation for reconstruction and novel view synthesis that uses a set of translucent triangle primitives, with an additional contribution to enhancing a soft connectivity force between neighboring triangles, enabling explicit cross-primitive coordination during optimization.

### Strengths
1. The paper is well written and easy to follow.
2. RTS strikes an impressive balance between geometric fidelity and color expressiveness, offering a promising primitive for neural rendering and 3D reconstruction.
3. The soft connectivity force is interesting and enables direct information exchange among primitives, and it may be a unique characteristic of the triangular representation.
4. Rich implementation details are documented meticulously, ensuring the work can be reliably reproduced and extended.
5. I encourage the authors to continue exploring this type of representation, as it may offer better compatibility with modern rendering engines compared to 3D Gaussians.

### Weaknesses
I think the main weaknesses are as follows:
1. From the quantitative metrics, there is still a noticeable gap compared with SOTA methods, both in novel view synthesis and geometry reconstruction. The authors could consider introducing a regularization similar to that in 2DGS to further enhance the credibility of the experiments. Although this may make the approach more complex, it could lead to a fairer quantitative comparison.
2. If my understanding is correct, Triangle Splatting is a work that is quite similar in terms of representation. The authors should further clarify the core contributions of this paper, especially at the representational level.
3. Training efficiency and rendering FPS is not reported if I don't miss anything.

Point 2 is the main reason I gave a initial score of 6, and it strongly influences my inclination to adjust the score during rebuttal. I hope the authors can respond to this point carefully.

### Questions
1. How do the authors view the future development of the triangle-based representation? I believe that, given the current rapid progress in feed-forward reconstruction networks, the triangle-based form should also start exploring reconstruction approaches beyond optimization-based methods. On this basis, if the triangle representation evolves toward feed-forward reconstruction networks, would it have any advantages over 3D Gaussians?

2. From a deployment perspective, it seems that the triangle representation is significantly more compatible with modern rendering engines compared to 3D Gaussians. Is this true?

### Soundness
3

### Presentation
3

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
Scene reconstruction / analysis-by-synthesis method using triangles as a primitive compared to existing primitive reconstruction works like gaussian splats. Furthermore learns connectivity between primitives allowing you to get surfaces. This blurs the line between mesh based methods for which optimizing over strange topologies can be difficult, and more loose primitive methods like gaussian splats for which you don't get surface normals that make a lot of sense.

### Strengths
I really enjoyed reading the paper, it was well motivated, and is a really original and interesting idea. Lots of well designed qualitative results and a lot of quantitative results. Not quite SOTA but demonstrates competitive results.

### Weaknesses
I really want there to be an additional table to include losses used in all these different methods. My main concern is that this uses normal information during training while other methods like 3DGS don't require that information ahead of time, and while you ablated the other loss terms, you didn't ablate this term. I'm concerned that this limits the usage of this to synthetic or highly constrained setups where you have the normal.

### Questions
I'd love some more perspective to ease my concerns about normals being required.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Radiant Triangle Soups as a new 3D representation that can be optimized via differentiable rendering and aims for high-quality 3D surface reconstruction and novel view synthesis at the same time. It leverages triangles as primitives, parameterized by the center, per-vertex Spherical Harmonic color coefficients, scales of vectors from the center to the vertices, a 3D rotation, a scalar opacity, and a diffuse strength, with the latter being necessary to obtain gradients for optimization.
Following the previous work of 3D Gaussian Splatting and follow-ups, the triangles are initialized using a point cloud obtained via SfM, rasterized in a differentiable fashion from training views, and optimized for reconstructing the ground truth images. Additionally to established losses like SSIM and depth smoothness, the authors propose a scene loss that softly encourages connectivity between neighboring triangles with aligned normals and rotations with the goal being emerging closed surfaces during optimization.
The paper includes an empirical evaluation for surface reconstruction on DTU and novel view synthesis on the Mip-NeRF360 dataset.
Furthermore, the authors ablate on the choice and weightings of depth smoothness and connectivity / scene loss terms.

### Strengths
- The paper is well written and very easy to follow and understand.
  - The introduction motivates the task and the proposed approach well.
  - The related work is very detailed and covers many existing approaches.
  - The method is is mostly clearly described.
  - The paper is honest about its experimental results.
- The paper includes multiple technical contributions:
  - The RTS representation based on triangle primitives intuitive.
    - Colors are interpolated based on the Spherical Harmonic coefficients of the triangle's vertices.
    - Surface normals per triangle are computed based on cross-products between edges and then alpha blended.
  - I find the idea of the connectivity (scene) loss to encourage emergence of closed surfaces consisting of aligned triangles as in meshes especially interesting and novel.
  - The adaptive density control leverages the characteristics of the representation, e.g., by splitting triangles into 4, or making use of the edge connectivity for pruning criteria.
- The method achieves the best performance in 3D surface reconstruction for some scenes of DTU and the best LPIPS result in novel view synthesis on indoor scenes of the Mip-NeRF360 dataset.
- The ablation study w.r.t. loss terms and weighting shows the effectiveness of the connectivity loss.
- The appendix provides additional implementation details as well as qualitative and quantitative results.

### Weaknesses
- The quantitative comparison with baselines is not convincing.
  - PGSR [1] is overall better in both surface reconstruction (Tab. 1) and novel view synthesis (Tab. 2) than the proposed method.
    - The authors attribute this to "the algorithm utilizes a full suite of multi-view objective functions that significantly improve the geometric reconstruction quality" (line 443). It remains unclear (also from related work) what these technical differences are exactly and whether comparison is still fair or not.
    - In any way, what prevents the authors from using the same objective functions for a fair comparison with PGSR?
  - While on indoor scenes, the method's NVS performance is on par with baselines, the performance on outdoor scenes is quite poor, lacking significantly behind GOF [2] and PGSR [1] (3.41 less PSNR and 0.147 higher LPIPS).
- The experimental evaluation is limited.
  - The provided qualitative results are insufficient.
    - The main paper includes only Fig. 7, which does shows three DTU examples without any comparison to baseline methods, and Fig. 8, which shows one novel view each for two DTU examples compared to only one baseline 2DGS, which is quantitatively only the 3rd strongest baseline for surface reconstruction, according to Tab.1.
  - The paper misses to compare their method with baselines in terms of training and rendering speed as well as GPU memory requirements.
    - The authors mention in the limitations (Sec. 6) that "due to periodic nearest-neighbors search, there is a minor increase in run-time".
- The paper lacks a detailed comparison of their method with TriangleSplatting [3].
  - This paper seems to be an extremely important related work, but the paper does not mention it at all in the introduction and mentions it in one sentence in the related work (line 146f.).
  - It is very difficult to evaluate the technical novelty compared to this paper, especially regarding the proposed representation, as the authors inly mention that TriangleSplatting "does not support any mechanism for [the triangles] to interact directly with each other", i.e., claiming that their soft connectivity forces loss is novel.
- Lack of clarity:
  - The authors claim that previous approaches like 3D Convex Splatting [4] are "more expensive than RTS" (cf. lines 153-163). However, the difference to RTS here is not clear to me, especially since in terms of diffuseness of primitives this method seems to share a lot of similarities with 3DCS (cf. 181 and Sec. 3.3). Furthermore, as mentioned above already, the paper misses to evaluate efficiency in terms of time and memory and compare with baselines to support this claim.
  - The triangles are parameterized by the three bisector lengths. It would be helpful for the reader to mention that a triangle is indeed uniquely represented by this.
  - The initialization of the rotation matrix is unclear. First, the rotation is defined based on triangle edges and normals, but then you apply a random rotation to this matrix. Is the outcome then not just random too?
  - "The diffuse scalar is also set as a function of the average distance to the three nearest neighboring points." (line 212). What is the function? Or point to the appendix, if it is described there.
  - In Sec. 3.5, the authors first describe the intuition and the behavior of the soft connectivity forces they introduce without actually defining the loss term. This is done later in Sec. 4.2. These two section should be merged to improve readability.
   - The description of the second term in the connectivity loss (the normal part) is missing in line 356.

- Minor weaknesses:
  - Missing references for use of triangles as fundamental primitives in computer graphs (lines 58f.).
  - The Fig. 4 is unnecessarily large / has a suboptimal layout for its message.
  - Missing references: Which previous works do you refer to in line 209 "Similar to previous works...".

References:
- [1] PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction. TVCG 2024
- [2] Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes. SIGGRAPH Asia 2024
- [3] Triangle Splatting for Real-Time Radiance Field Rendering. arxiv 25 May 2025
- [4] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes. CVPR 2025

### Questions
- The authors need to compare their method with TriangleSplatting in terms of technically novel contributions.
- I also suggest that the authors either describe in detail what they mean with PGSR's "algorithm utilizes a full suite of multi-view objective functions" as the reason for their worse performance compared to PGSR, why it is not applicable to their approach, if that is the case, or provide additional experimental results for using the same objective functions as PGSR but with the RTS representation and connectivity loss, hopefully boosting performance.
- Since the authors claim that previous works relying on volumetric primitives are "more expensive than RTS" (line 163), I suggest that the authors provide empirical evidence for this in form of an time and memory efficiency comparison for both training and test time.
- Further open questions are:
  - What is the intuition of applying a random rotation to the rotation matrices of triangles at initialization? Is it maybe a rotation only a long a certain axis?
  - The authors emphasize that they "directly generate a 3D point cloud for geometric evaluation from the rendered depth maps without performing any TSDF fusion" (line 405f.). Is this consistent with baselines? If not, how does it change the conclusion of the comparisons?

### Soundness
2

### Presentation
3

### Contribution
2
