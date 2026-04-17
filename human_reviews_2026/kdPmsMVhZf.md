# G4Splat: Geometry-Guided Gaussian Splatting with Generative Prior

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Despite recent advances in leveraging generative prior from pre-trained diffusion models for 3D scene reconstruction, existing methods still face two critical limitations. First, due to the lack of reliable geometric supervision, they struggle to produce high-quality reconstructions even in observed regions, let alone in unobserved areas. Second, they lack effective mechanisms to mitigate multi-view inconsistencies in the generated images, leading to severe shape–appearance ambiguities and degraded scene geometry.
In this paper, we identify accurate geometry as the fundamental prerequisite for effectively exploiting generative models to enhance 3D scene reconstruction. 
We first propose to leverage the prevalence of planar structures to derive accurate metric-scale depth maps, providing reliable supervision in both observed and unobserved regions. 
Furthermore, we incorporate this geometry guidance throughout the generative pipeline to improve visibility mask estimation, guide novel view selection, and enhance multi-view consistency when inpainting with video diffusion models, resulting in accurate and consistent scene completion.
Extensive experiments on Replica, ScanNet++, DeepBlending and Mip-NeRF 360 show that our method consistently outperforms existing baselines in both geometry and appearance reconstruction, particularly for unobserved regions.
Moreover, our method naturally supports single-view inputs and unposed videos, with strong generalizability in both indoor and outdoor scenarios with practical real-world applicability. Project page: https://dali-jack.github.io/g4splat-web/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes G4SPLAT, a geometry-guided framework for sparse-view 3D Gaussian Splatting reconstruction. Existing generative prior methods often suffer from degraded performance due to insufficient geometric supervision and multi-view inconsistency. G4SPLAT addresses these issues by deriving scale-accurate plane-aware depth maps  that leverage the prevalent planar structures in scenes. These accurate geometric cues are integrated into a geometry-guided generation process to refine visibility mask estimation, guide novel view selection, and improve cross-view consistency during video diffusion model inpainting. Experiments on Replica, ScanNet++, and DeepBlending datasets show that G4SPLAT outperforms state-of-the-art baselines in both geometric accuracy and appearance quality.

### Strengths
- This paper is clearly written, with well-explained derivations of plane-aware depth maps and the geometry-guided generation process, making G4SPLAT’s contributions and implementation easy to understand.
- The proposed G4SPLAT takes advantage of plane structures in a scene to figure out depth with the right scale, which gives solid guidance even in areas that don’t have much information in sparse-view setups. Ablation results show that generative priors alone offer little geometric benefit and may even degrade quality, while adding plane-aware geometry modeling greatly improves accuracy and consistency.

### Weaknesses
- The proposed method relies on the presence of significant planar structures in the scene, which may not be applicable in certain complex or natural environments. In scenes lacking prominent planar features, the generated plane-aware depth maps may not provide sufficient geometric cues, potentially affecting the quality of the final 3D reconstruction.

### Questions
In constructing the Visibility Grid, how do the choices of voxel size and number of sampling points Q affect visibility accuracy and runtime?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces G4Splat, a new method for sparse-view 3D scene reconstruction. Building upon existing work MAtCha Gaussians [1], the paper proposes to leverage estimated 3D planes throughout the entire reconstruction pipeline that additionally leverages a pre-trained video diffusion model as a prior. Making use of the usual existence of planar structures in 3D scenes created by humans, the authors propose to extract per-view 2D planes with SAM in order to merge them into global 3D planes.
These can then be used to obtain more accurate depth than from sparse-view SfM approaches like MASt3R-SfM [2] used in [1], and to compute improved visibility masks compared to rendered alpha maps, which are required as inpainting masks when combined with a generative prior.
The authors further leverage the 3D planes for a plane-aware selection of novel views for inpainting and finally for selecting supervision signals from the geometry-guided inpainting.
An experimental evaluation on ScanNet++, DeepBlending, and Replica shows that G4Splat consistently outperforms all baselines w.r.t. both 3D surface reconstruction and novel view synthesis quality. The paper further includes qualitative results for different numbers of input views and an ablation study w.r.t. the use of the generative prior, the 3D planes, and both combined.

- [1] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views. CVPR 2025
- [2] MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion. 3DV 2025

### Strengths
- The paper is well written and mostly easy to follow and understand.
  - The introduction motivates the topic well and discusses the idea following the Manhattan world assumption and the shortcomings of existing works leveraging generative priors.
  - The related work section is comprehensive. I found Sec. 2.3. about the plane assumption in reconstruction especially interesting.
  - The method section first introduces the 3D plane estimation step by step (first 2D and per view, then 3D and global), followed by all applications of the obtained 3D planes throughout the entire optimization pipeline.
- I appreciate the method and paper story of making use of 3D planes everywhere throughout the 3D reconstruction pipeline, following the Manhattan world assumption.
  - The step-wise estimation of the 3D planes using SAM in 2D, then merging these with RANSAC, is a reasonable and intuitive approach.
  - The paper shows how to obtain more accurate depth maps and visibility masks using 3D planes as well as more useful novel view selection for inpainting with generative priors.
    - Fig. 3 visualizes all of these improvements over naive alternatives well.
- The quantitative and qualitative comparisons with baselines are overall very convincing.
  - The proposed method consistently and significantly outperforms all baselines for both 3D surface reconstruction and novel view synthesis.
  - The authors make sure to include a baseline of 2DGS combined with the video diffusion model that they use (See3D), as this model seems to be quite new and not used by baselines.
  - The authors choose very recent and strong baselines, representing the current state of the art.
- The runtime comparison shows that the proposed method is roughly as fast as previous works leveraging generative priors, while obtaining better quality.
- The ablation study shows that all individual design choices contribute to the overall performance of the method.
- The authors provide additional convincing qualitative results in form of videos on the anonymous website.
- The appendix provides more results, detailed background to prior works that this paper leverages, training details, and limitations.

### Weaknesses
- Since the method relies on 3D planes, it seems to be quite tailored for indoor and possibly city-like outdoor scenes, or in other words, scenes that actually consist of enough 3D planes. 
  - I am wondering how this method would perform on other kind of outdoor scenes, e.g., the garden or bike scenes in the MipNeRF360 dataset. Are there still enough planar structures to leverage?
    - In the limitations section in the appendix, the authors do touch on this topic but mainly for obtaining scale-accurate depth.
  - The paper, appendix, and website mostly show almost only indoor scenes.
- The paper would benefit from a more precise evaluation of 3D reconstruction for observed and unobserved regions separately.
  - It would be very interesting to see quantitatively how much each component improves observed and unobserved regions, respectively.
- Concern about fairness w.r.t. generative prior:
  - While the authors do provide a baseline of 2DGS + See3D, i.e., the video generative prior that they use in this paper, would MAtCha + See3D not be the better baseline in terms of fairness?
    - In Fig.4, MAtCha looks quite accurate, just incomplete, whereas 2DGS + See3D has a lot of artifacts. Is this solely due to the generative prior See3D or also due to 2DGS being worse than MAtCha?
  - Regarding the use of generative priors, the main paper (related work) would need some more details about which approach uses what kind of generative prior and how does it affect results.
  - To this end, it would be beneficial for the paper to evaluate their method with the generative prior used by baselines, if that is possible. This would be important in order to precisely attribute the performance gains to a better generative prior or to the other technical contributions, e.g., using the 3D planes.
- Some lack of clarity:
  - The need for the background Sec. 3.1 about MAtCha Gaussians became only clear much later in Sec. 3.4 that then states that G4Splat builds on MAtCha as initialization. There is no description of the structure of Sec. 3 (after the Method title), but the paper just goes immediately into the background section after the related work, leaving the reader confused why another "related work" section in the method section is needed. Writing one or two sentences about the structure of the Method section and the individual subsections and why they are necessary at the beginning of the Method section would resolve this.
  - From the main paper, it is unclear what role 3D plane extrapolation plays in the pipeline. Are 3D planes extrapolated to unobserved regions? Does this involve any additional assumptions about the layout of an indoor scene, for example?
    - Line 74f.: "planar surfaces allow depth extrapolation: a 3D plane can be reliably estimated from partial depth observations and then extended across the entire surface" states something along those lines.
  - In paragraph "Per-view 2D Plane Extraction" (lines 193ff.), it is unclear how you obtain the normals.
  - Is the depth from the monocular depth estimator and scaled using the 3D plane geometry really better than the depth estimated by MASt3R-SfM? If this is the case, why? Are more recent DUSt3R follow-ups like VGGT maybe more performant?
  - The color supervision description in lines 314f. is quite vague and refers to the appendix. If possible, it would be beneficial to move this information to the main paper to make it more self-contained, as otherwise there remain open questions that require the appendix.
  - Tab.3 misses to explain what "DS" in "Ours (DS)" stands for. Actually, I was not able to find that information anywhere in the paper. I assume it uses a distilled video model maybe?
  - The PM setting in the ablation study is not completely clear to me. Are the planes there only used for initialization and for depth supervision or for what parts of the pipeline exactly?
- The paper misses a related work and potential baseline: Spurfies [1].


References:
- [1] Spurfies: Sparse Surface Reconstruction using Local Geometry Priors. 3DV 2025

### Questions
My suggestions to the authors are already detailed in the weaknesses section but here again concretely:
- It would be interesting to evaluate the approach on more outdoor scenes with less planar structures like the garden and bike scenes in the MipNeRF360 dataset.
  - While I do see the value in a method that performs particularly well for indoor scenes, this would be beneficial for the paper to better understand possible limitations.
- Splitting the quantitative (and possibly qualitative) evaluation into observed and unobserved regions would be very interesting to evaluate the behavior of the individual contributions more precisely.
- I suggest that the authors evaluate MAtCha + See3D (or is that already the second row, i.e., GP only of the ablation study maybe) instead of 2DGS + See3D. It would be the better comparison as a baseline in qualitative and quantitative evaluations in terms of fairness.
- It would be interesting to also evaluate G4Splat in combination with different generative priors to see whether the 3D planes can make the use of generative priors more effective irrespective of the particular choice which prior to use.
- Further open questions are:
  - Are 3D planes extrapolated to unobserved regions? Does this involve additional assumptions? If so, how does it affect results?
  - How are surface normals obtained (cf. lines 193ff.)?
  - Why is the depth from monocular depth estimation and scaling better than from dense reconstruction approaches like MASt3R-SfM or VGGT?
  - What does "DS" stand for in Tab.3?
  - Could you give details about the PM setting in the ablation?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
G4Splat proposes a reconstruction pipeline for sparse-view 3D scene reconstruction that integrates metric-scale geometry estimation (leveraging planar priors) with generative priors from pretrained diffusion/image models. Geometry guidance is injected at multiple stages (depth estimation, visibility masks, view selection, and inpainting) to reduce shape–appearance ambiguities and improve multi-view consistency in novel-view completion. Experiments on Replica, ScanNet++ and DeepBlending report improvements in both geometry and appearance, especially in unobserved regions.

### Strengths
Identifies a real limitation in prior generative-prior reconstructions (poor geometry in observed areas and inconsistency in unobserved areas) and supplies a concrete fix via planar metric depth and guided inpainting. 

End-to-end incorporation of geometry at multiple pipeline points (visibility masks, view selection) is sensible and likely to increase multi-view consistency. 

Broad evaluation on several standard datasets and claimed improvements on both geometry and appearance metrics.

### Weaknesses
Reliance on planar priors: in scenes without significant planar structure (e.g., natural outdoor scenes, complex organic interiors), the metric-depth derivation may fail; robustness experiments for such cases are not prominent in the material on the forum page. 

Integration with generative priors can still propagate biases from the generative model (style/appearance biases) - the paper does not analyze or mitigate such biases.

Computational cost: combining geometry estimation, diffusion-based inpainting, and splatting can be expensive; readers would benefit from runtime and resource-use breakdowns and ablations on where the gains come from.

### Questions
Provide quantitative robustness experiments on scenes with few planar structures — how does the depth prior behave and how does it affect final reconstructions? 

How does G4Splat handle scale ambiguity when planar cues are incorrect or scarce? Provide failure cases.

Please report computational cost and latency for a representative scene, and ablate which component (plane-based depth, guided visibility, diffusion inpainting) provides the largest gain.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes G4SPLAT, a sparse-view 3DGS reconstruction method that uses planar depth priors and geometry-guided video diffusion refinement. The idea is to extract global 3D planar surfaces from input images, convert them into scale-accurate depth maps, and use these to supervise Gaussian optimization, select novel viewpoints, and constrain generative inpainting. The method iteratively refines the 3D model using inpainted novel views and recomputed plane-aware depth maps. Experiments on Replica, ScanNet++, and DeepBlending claim substantial gains in unobserved regions and single-view/unposed settings. Overall, the pipeline combines MAtCha-style plane-scaled depth, 2DGS, and video diffusion priors, but introduces significant engineering complexity in return for modest conceptual novelty.

### Strengths
- Clear motivation: improving geometry for generative 3DGS. The paper correctly identifies geometry accuracy as a limiting factor in generative scene completion and explicitly seeks to address shape-appearance ambiguity. This motivation is grounded in observed weaknesses of recent diffusion-enhanced 3DGS methods.
- Plane-based scale recovery is well-explained and technically coherent. The method extends plane fits across views using SAM, normal clustering, and RANSAC with multi-view consistency checks, yielding scale-aligned depth even in weakly observed areas. While not new, the pipeline is carefully engineered, and ablations show benefit over raw monocular depth.
- The approach uses plane-aware visibility masks, plane-driven novel-view planning, and plane-based inpainting weighting to enforce consistency with the goal of stabilizing diffusion-assisted NVS. This produces fewer floaters and sharper planar regions compared to GenFusion / Difix3D+.

### Weaknesses
- **Fundamental Reliance on Planar Structures:** The method's core contribution and primary advantage are fundamentally tied to the Manhattan-world assumption. While effective for artificial environments, this reliance makes the approach far less suitable for organic, non-planar scenes (e.g., natural landscapes, complex statues, foliage). The paper's solution for non-planar regions is to fall back on monocular depth estimation, which is the very technique it criticizes for scale ambiguity.
- **Heavy Engineering Pipeline** - G4Splat is not a single model but a complex, multi-stage pipeline that glues together numerous off-the-shelf components like MAtCha, MASt3R-SfM, SAM, K-means clustering, RANSAC, a monocular depth estimator, and a video diffusion model. This high complexity makes the system brittle - a failure in any one component could compromise the entire pipeline. 
- The approach inherits a strong geometric scaffold from MAtCha, including scale-aligned depth, plane priors, and reliable surface initialization. This prior stabilizes 3DGS optimization in sparse-view regimes and likely contributes to the improved geometric integrity. In contrast, several baselines (e.g., GenFusion and Difix3D+) do not assume a comparable geometric initialization and instead operate in a more challenging setting where depth must be inferred solely from generative consistency. This makes the comparison somewhat imbalanced, and it becomes difficult to isolate how much of the improvement stems from the proposed refinements versus MAtCha's initialization advantage.
- The qualitative comparisons in Fig. 4 appear to show holes and structural failures for generative baselines that are not typically reported in their original papers. This behavior is plausible when such models are applied without a metric depth prior or explicit plane constraints, especially under sparse or weakly posed conditions. However, the visual degradation suggests that the baselines may not have been given an equally stabilized geometric starting point. A clearer description of how pose supervision, depth alignment, and initial surfaces were handled for each competing method would help ensure confidence in the reported gaps.
- All datasets are indoor and relatively structured - no explicit experiments test failure modes (irregular geometry, non-planar scenes, outdoor clutter). Standard view splits of 3, 6, and 9 views are followed in relevant literature (ReconFusion, CAT3D), but are not followed in this paper. 
- A strong baseline in ViewCrafter (TPAMI'25) is missing.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
1
