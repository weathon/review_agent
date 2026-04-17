# ComGS: Efficient 3D Object-Scene Composition via Surface Octahedral Probes

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Gaussian Splatting (GS) enables immersive rendering, but realistic 3D object–scene composition remains challenging. Baked appearance and shadow information in GS radiance fields cause inconsistencies when combining objects and scenes. Addressing this requires relightable object reconstruction and scene lighting estimation. For relightable object reconstruction, existing Gaussian-based inverse rendering methods often rely on ray tracing, leading to low efficiency. We introduce Surface Octahedral Probes (SOPs), which store lighting and occlusion information and allow efficient 3D querying via interpolation, avoiding expensive ray tracing. SOPs provide at least a 2x speedup in reconstruction and enable real-time shadow computation in Gaussian scenes. For lighting estimation, existing Gaussian-based inverse rendering methods struggle to model intricate light transport and often fail in complex scenes, while learning-based methods predict lighting from a single image and are viewpoint-sensitive. We observe that 3D object–scene composition primarily concerns the object’s appearance and nearby shadows. Thus, we simplify the challenging task of full scene lighting estimation by focusing on the environment lighting at the object’s placement. Specifically, we capture a 360° reconstructed radiance field of the scene at the location and fine-tune a diffusion model to complete the lighting. Building on these advances, we propose ComGS, a novel 3D object–scene composition framework. Our method achieves high-quality, real-time rendering at around 26 FPS, produces visually harmonious results with vivid shadows, and requires only 36 seconds for editing. The code and dataset are available at https://nju-3dv.github.io/projects/ComGS/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes ComGS, a Gaussian-splatting-based framework that performs inverse rendering for a scene and supports inserting new objects into the scene with consistent lighting appearance. Main contributions include (1) Surface Octahedral Probes (SOPs) for capturing indirect lighting and occlusion, (2) a diffusion model to estimate HDR 360° lighting based on the Gaussian scene, (3) the occlusion caching technique to efficiently cast shadows of the new object without ray tracing. Experimental results demonstrate superior quality to baseline.

### Strengths
1. The idea of using light probes for the inverse rendering task is novel. Light probes are a technique in traditional graphics rendering that cache indirect lighting to avoid costly ray tracing, which is well-suited to the inverse rendering task for performance gains.
2. The proposed occlusion caching further avoids costly ray tracing for the inserted object. Ray tracing has been a widely used but costly technique in recent years' inverse rendering paper, and this paper successfully bypasses it while preserving quality.
3. The inverse rendering community can benefit from the SynCom Dataset created by this paper.

### Weaknesses
## Major
1. Missing citations. The related work section only introduces NeRF- and GS-based inverse rendering methods, while a rich literature of mesh-based inverse rendering methods has also emerged in recent years and is worth mentioning, for example,
```
@article{hasselgren2022shape,
  title={Shape, light, and material decomposition from images using monte carlo rendering and denoising},
  author={Hasselgren, Jon and Hofmann, Nikolai and Munkberg, Jacob},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={22856--22869},
  year={2022}
}
@inproceedings{dai2025inverse,
  title={Inverse Rendering using Multi-Bounce Path Tracing and Reservoir Sampling},
  author={Dai, Yuxin and Wang, Qi and Zhu, Jingsen and Xi, Dianbing and Huo, Yuchi and Qian, Chen and He, Ying},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year = {2025},
  url = {https://openreview.net/forum?id=KEXoZxTwbr}
}
```
The idea of using a diffusion model to estimate 360° environment map lighting is also explored by previous work
```
@article{lyu2023diffusion,
  title={Diffusion posterior illumination for ambiguity-aware inverse rendering},
  author={Lyu, Linjie and Tewari, Ayush and Habermann, Marc and Saito, Shunsuke and Zollh{\"o}fer, Michael and Leimk{\"u}hler, Thomas and Theobalt, Christian},
  journal={ACM Transactions on Graphics (TOG)},
  volume={42},
  number={6},
  pages={1--14},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```
2. Lambertian assumption. The training scheme regularizes the roughness to be near 1 and metallic to be near 0 (Eq. 12). While this may improve training stability, it also limits the expressivity of the inverse rendering pipeline, especially for scenes with specular reflections. Also, Eq. 15 further approximates the rendering equation with Lambertian assumption, which may introduce bias for non-Lambertian scenes.
3. Paper writing. (1) The proposed Surface Octahedral Probes are one of the main contributions of this paper, but I found the corresponding description to be too concise (L224-235). The authors should expand the corresponding paragraph in the revision. (2) The paper does not contain a limitation subsection.

## Minor
1. Monte Carlo rendering. Strictly speaking, the rendering estimator in Eq. 7 with a low-discrepancy sequence should be a deterministic quasi-Monte-Carlo scheme, compared to traditional Monte Carlo methods that require stochastic importance sampling. Despite avoiding noise and variance, the shortcoming of such a scheme is the lack of ability to capture high-frequency details, such as sharp specular reflections [1]. This weakness coincides with the Lambertian assumption weakness I mentioned before.
2. The inverse rendering results on TensoIR Dataset (Fig 14) still contain significant artifacts, e.g. the albedo map. It would also be preferred to show the inverse rendering results on the scene-level data.

[1] Zhu et al. "Learning-based inverse rendering of complex indoor scenes with differentiable Monte Carlo raytracing." SIGGRAPH Asia 2022 Conference Papers. 2022. Fig 4

### Questions
1. Is the proposed 360° environment map lighting diffusion model (Sec 3.2.1) similar to the "Diffusion posterior illumination" paper? If so, the author should clarify this and weaken the contribution claim on it.
2. Can authors provide a clarification on whether the method can handle non-Lambertian scenes? Experimental results are also preferred.
3. How accurate are the estimated cast shadows by the occlusion caching compared to the ray-traced ground truths, given that Eq. 15 may introduce significant approximation error?

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
This paper presents a novel object-scene composition framework, ComGS, which consists of an efficient 2DGS-based inverse rendering method and a diffusion-based lighting estimation model. The proposed inverse rendering pipeline utilizes Surface Octahedral Probes to cache the indirect illumination and occlusion, eliminating the need for costly ray tracing . Experiment results are provided to demonstrate the good performance of the framework, highlighting the effectiveness of SOPs.

### Strengths
1. This paper proposes a novel inverse rendering framework. Unlike previous methods that use expensive ray tracing to obtain indirect lighting and occlusion, this method uses probes to store indirect lighting and occlusion, significantly improving efficiency.
2. The proposed lighting estimation model conditioned on rendered partial panorama envmap from GS demonstrates good performance.
3. Extensive experiments show that the method achieve better performance than inverse-rendering and generative baselines.

### Weaknesses
1. It seems that the proposed methods cannot be used in single image scenario. During editing stage, since the object is inserted into the scene, the occlusion probes near the object should be baked first.  But if the scene is represented by single image, it is impossible to use GS to reconstruct it, which means we cannot obtain the occlusion SOPs.
2. Even the position of probes are generated from the surface point clouds using FPS, the sampled probes may still be within the Gaussians’ effective ranges (mixing with the Gaussians), which will cause light leakage. But I think there shouldn't be many such examples, so this isn't a big problem.

### Questions
1. Is there a way to obtain occlusion probes even when the scene is represented by a single image? Or are there other methods to obtain the shadow?
2. Could you provide the performance of two extra baselines MV-CoLight and GI-GS on your SynCom-Objectdataset?

[MV-CoLight, NIPS'25](https://github.com/InternRobotics/MV-CoLight)

[GI-GS, ICLR'25](https://github.com/stopaimme/GI-GS/tree/master)

### Soundness
2

### Presentation
3

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
This paper presents a realistic 3D object-scene composition pipeline that achieves plausible shadows and lighting effects in real time.
The framework consists of three major stages:
(1) Reconstruction, where both the object and scene are represented using Gaussian splatting;
(2) Editing, which includes environment lighting estimation and Surface Octahedral Probe (SOP)-based occlusion caching; and
(3) Rendering, performing relighting and shadow synthesis for seamless composition.
The proposed system is efficient and demonstrates high-quality visual realism on various examples of single object insertion and relighting.

### Strengths
1. The proposed framework demonstrates good performance and maintains efficiency, outperforming several baselines.
2. The proposed SOP structure is convenient, enabling real-time shadow and lighting computation.
3. The paper is well organized and easy to follow.

### Weaknesses
1. Self-occlusion or ambient occlusion effects on the inserted object are barely noticeable (e.g., Figure 7), suggesting that the occlusion modeling may not be effective enough. (Please also see question 3).
2. The experiments mostly showcase single-object insertions. It remains unclear how the system performs when inserting multiple objects, or when an object is placed freely in the scene (for example, partially under the shadow area). More results could further demonstrate the effectiveness of the proposed method. 
3. The performance of object inverse rendering depends on the input images. The synthetic dataset from the simulator largely aligns with the proposed method, which may be biased. More discussion about real-data capture conditions or constraints would improve clarity and generalizability.

### Questions
1. For the scene reconstruction, are the 2D Gaussians also equipped with albedo, roughness, and metallic attributes as described in lines 155–161? If so, how are these optimized since they do not appear in Step 2? For the object reconstruction, is the RGB buffer C still optimized and used? If so, how does it involve in the rendering equation?
2. Eq. 2 seems incorrect, any typos?
3. Is the self-occlusion term computed in Eq. (8) reused in Section 3.3 (``Rendering'')? Does the relighting step re-bake indirect illumination under new lighting conditions? From Figure 7 and explanations in Section 3.3, it seems these two terms are not re-used at the relighting stage.
4. What if there are multiple light sources in the scene (may correspond to various peak values in the environment map)? Will the generated shadow composite of layered shadows with darker overlapped regions?
5. Does a mismatch between the captured object’s original environment map and the target scene environment influence the final performance?

I will raise the score if the major concerns are addressed.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents a novel pipeline for realistic 3D object-scene compositions, which is formulated as composing given scene and object multi-view images into a harmonious and visually-natural 3D representation. The main contributions of this work are surface octahedral probes (SOP) and local lighting completion. These designs empower high-quality relightable object reconstruction and object-scene composition with significantly improved efficiency.

### Strengths
1. The proposed method is well-motivated. This work targets two long-standing obstacles in the object-scene composition task: relightable object reconstruction and scene lighting estimation, and introduces specific designs to tackle these challenges.
2. The SOP design mitigates the reliance on per-point ray tracing during optimization and rendering, thus improving the inverse rendering efficiency, leading to (near) real-time relightable rendering speed.
3. The local lighting completion builds upon the insight that only part of the scene interacts with the to-be-composed object, thus focusing on only the salient parts of the representation.
4. The proposed pipeline achieves superior performance from both the quantitative and qualitative evaluations while demonstrating high rendering efficiency, verifying its optimal effectiveness-efficiency trade-off.

### Weaknesses
1. The SOP initialization raises concerns. It is assumed that the SOPs are initialized by ray tracing and then optimized in a supervised manner. This initial ray-tracing step and progressive SOPs require careful designs to avoid light leaks and placement offsets. Although a heuristic solution is provided for the offset = 1% object size, its robustness is not clearly verified.
2. Assumption on the pipeline settings. The proposed pipeline focuses on scenes with low or moderate occlusions (<40%), and builds on assumptions that the object is small relative to the scene and its placement affects only its own appearance and nearby regions. These assumptions are fundamental to the overall pipeline and prompt an efficient solution. However, it is not clear whether this assumption accounts for most real-world cases, and it is interesting to investigate how the method performs for hard cases and failure cases.

### Questions
1. It is recommended to add ablations or verifications on the SOP initializations. How the offset affects the final results and whether per-scene tuning is needed should be clarified.
2. The memory consumption introduced by SOPs should be shown, as thousands of probe operations are executed in this implementation.
3. How sensitive are results to SOP count and probe texture resolution? Is there a principled way to choose probe density per scene scale?
4. A deduction on the limit of the proposed method regarding the assumptions is favorable. For example, what is the threshold for a normal pipeline proceeding, and how do these assumptions affect the method’s effectiveness?
5. Is it possible to incrementally update the SOP as the object or camera moves to mitigate the calculation of re-caching for each editing? This may further enhance the pipeline completeness.

### Soundness
2

### Presentation
3

### Contribution
3
