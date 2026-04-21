# 6DGS: Enhanced Direction-Aware Gaussian Splatting for Volumetric Rendering

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
Novel view synthesis has advanced significantly with the development of neural radiance fields (NeRF) and 3D Gaussian splatting (3DGS). However, achieving high quality without compromising real-time rendering remains challenging, particularly for physically-based rendering using ray/path tracing with view-dependent effects. Recently, N-dimensional Gaussians (N-DG) introduced a 6D spatial-angular representation to better incorporate view-dependent effects, but the Gaussian representation and control scheme are sub-optimal. In this paper, we revisit 6D Gaussians and introduce 6D Gaussian Splatting (6DGS), which enhances color and opacity representations and leverages the additional directional information in the 6D space for optimized Gaussian control. Our approach is fully compatible with the 3DGS framework and significantly improves real-time radiance field rendering by better modeling view-dependent effects and fine details. Experiments demonstrate that 6DGS significantly outperforms 3DGS and N-DG, achieving up to a 15.73 dB improvement in PSNR with a reduction of 66.5\% Gaussian points compared to 3DGS. The project page is: https://gaozhongpai.github.io/6dgs/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
3DGS excels at synthesizing novel views of scenes, but it struggles with showing effects that depend on the viewer's perspective, like reflections, refractions, and scattering caused by complicated light transport. To solve this issue, the paper introduces a method called 6D-GS, which is based on N-dimensional GS. This method represents a scene as a 6D signal to capture both the spatial position and view direction at the same time. The paper enhances N-dimension GS by incorporating a view-dependent opacity function and an optimized density control. Extensive experiments on two synthetic datasets demonstrate the effectiveness of the proposed method.

### Strengths
- The paper is well-written and easy to follow. The details are clearly described, and I believe the results can be reproduced by an expert in this field.

- The method is simple and easy to implement. It allows slicing the 6D Gaussian given a view direction into a 3D Gaussian, which can then be directly fed into the existing 3D Gaussian rasterizer.

- Experiments show impressive results in the 6DGS-PBRT datasets, achieving 10dB over 3DGS and 1.5dB over N-D Gaussian. The results may interest research that aims to overfit high-dimension data.

### Weaknesses
The main weakness is the absence of supplemental video, which makes it difficult to access the novel view synthesis results. 

From my understanding, allocating more capacity to model view dependency of a scene might hinder the geometry and novel view synthesis. For instance, early NeRF uses a relatively small MLP with fewer Fourier features to strike a balance. 

Furthermore, the introduction of view-dependent opacity also complicates the results. For instance, while making the opacity view-dependent allows for easy fitting of high specular reflection, it is possible that when you view the specular spot from a slightly different angle, it becomes transparent. Since the PSNR metric may not provide a clear indication, it is recommended to include a video demonstration.

### Questions
I am interested in learning more about the potential applications of the algorithm. Can the algorithm be used for general radiance field reconstruction  instead of using synthetic data in dense view settings? If so, can you provide results from real-world datasets such as NeX? 
If not, it would be helpful to understand the specific scenarios where algorithms can be applied and to discuss any limitations.

### Soundness
3

### Presentation
4

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
This paper aims to address the challenge of handling view-dependent effects in scenes, a common issue encountered in both NeRF and 3DGS. Inspired by N-DG, this paper further enhances the 6D spatial-angular representation to better model scene details, particularly in contexts exhibiting complex view-dependent effects. By leveraging additional directional information, the proposed method achieves an improved representation of opacity and color while utilizing fewer Gaussians, thereby enhancing rendering speed. In addition, this paper offers a theoretical analysis of the conditional Gaussian parameters derived from the 6DGS representation, revealing its physical significance and contributions to enhancing rendering quality.

### Strengths
- The method proposed in this paper enhances the modeling of opacity and color while reducing the scale of Gaussians. By slicing the 6DGS to conditional 3DGS, it further improves rendering speed.
- This paper offers a theoretical analysis of 6DGS that enhances rendering quality, aiding in the understanding of its contribution to modeling complex scenes, particularly the scenes with view-dependent effects.
- This paper clearly defines the problem to be addressed and proposes the 6DGS representation, which is beneficial for overcoming the limitations of previous representations in applications. Additionally, it provides some validation across multiple datasets.

### Weaknesses
- The experimental validation is insufficient. (1) The method proposed in this paper emphasizes its capability to model complex scenes exhibiting view-dependent effects; however, the qualitative experiments are limited to results derived from the 6DGS-PBRT dataset proposed in this paper. The authors should provide additional visualization results, such as those derived from the Synthetic NeRF dataset to strengthen the effectiveness of the proposed method. (2) The paper only presents results for two synthetic datasets, which does not adequately demonstrate the robustness and applicability of the proposed method in real-world scenarios.

- Some important analyses are missing. Due to the challenges of camera pose estimation in real-world scenes with view-dependent effects, and considering that camera pose influences Gaussian density control, the authors should consider conducting further analysis on this issue.

### Questions
Based on the aforementioned weaknesses, the following issues also require discussion:
- Given that N-DG demonstrates rendering results on real data, and as mentioned in line 62 of the paper, the proposed method can handle scenes with glossy surfaces and transparency, can the method achieve better modeling and rendering results when applied to the real-world dataset collected by NeX? As NeX proposes view-dependent pixel representation that can handle non-Lambertain surfaces and model fine details, the author should conduct comparative experiments with NeX.

[1] NeX: Real-time View Synthesis with Neural Basis Expansion, CVPR 2021.

- Since N-DG can be extended to a higher-dimensional representation, was the dimensional setting for this method in the comparative experiments set as 6D?

- As the method proposed in the paper is based on per-scene optimization, the authors should consider providing a comparison of training times across different scenes. This would help understand the contributions of the method better.

### Soundness
2

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
This paper proposes an extension of the existing 3D Gaussian Splatting technique to a six-dimensional Gaussian Splatting method. 6DGS combines 3D positional information with 3D view direction information using a six-dimensional Gaussian, enabling it to precisely capture view-dependent variations in color and transparency. To enhance computational efficiency, a restricted covariance method is introduced, which selectively incorporates only the necessary correlations, and spherical harmonics are employed to accurately model view-dependent color changes. Additionally, a new synthetic dataset called 6DGS-PBRT was constructed for performance evaluation, with experimental validation of view-dependent effects. This dataset, comprising synthetic scenes with various lighting and material properties, demonstrates that 6DGS can approximate complex visual effects effectively.

### Strengths
1. The fact that the proposed 6DGS is compatible with the existing 3D Gaussian Splatting (3DGS) framework and requires minimal modification appears to be a significant strength. This compatibility suggests that 6DGS can be effectively integrated into various existing datasets and application environments, potentially achieving impressive performance with minimal adjustments.

2. The theoretical explanations, figures, and experimental results are intuitively presented, making it easy to understand the contribution and performance of the proposed approach. This strong presentation quality greatly enhances readability.

3. The authors constructed a new 6DGS-PBRT dataset specifically to evaluate view-dependent effects, which seems to add considerable value. This dataset enables a more precise validation of 6DGS’s performance and appears to be an effective foundation for demonstrating its utility.

4. The approach provides important insights for the optimization of multi-dimensional Gaussian splatting techniques. By introducing restricted covariance modeling to refine view-dependent effects, this work offers valuable clues for future optimization of higher-dimensional Gaussian models.

### Weaknesses
1. Although experiments on synthetic datasets are meaningful, the lack of experiments on real-world captured datasets may limit the understanding of the model’s performance in real environments.

2. While the authors’ 6DGS-PBRT dataset seems well-suited for view-dependent effects, the limited variety of classes may fall short in representing a wider range of optical phenomena.

3. Additional explanation about the visual characteristics of each dataset would enhance clarity. For instance, clarifying that “cloud” captures absorption and scattering effects, or “glass” represents refraction effects, could make the paper more accessible. It also seems that adding reflective objects or other classes could provide a more comprehensive evaluation.

### Questions
1. Why does the proposed 6DGS outperform Ray Tracing-based Gaussian models on PBRT-rendered datasets? Intuitively, Ray Tracing-based models might be expected to perform better. Could it be due to incomplete implementation or insufficient optimization of the Ray Tracing-based model?

2. Although 6DGS is not Ray-Tracing-based, it appears to approximate the complex rendering effects and physical phenomena modeled by PBRT effectively. Could it also achieve high performance on more complex effects, such as subsurface scattering involving multiple scattering events?

3. While the synthetic dataset results already establish the value of this paper’s contributions, is there a specific reason or technical difficulty that prevented experiments on real-world datasets?

### Soundness
3

### Presentation
4

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
This paper presents 6D Gaussian Splatting(6DGS), an advanced method for volumetric rendering that enhances traditional 3D Gaussian splatting (3DGS) by incorporating view-dependent information through a 6D spatial-angular Gaussian representation.

The authors address key limitations in 3DGS, such as handling specular reflections and anisotropic effects, by leveraging a 6D Gaussian model that adapts both color and opacity based on viewing angles. Experimental results demonstrate significant improvements in both image quality (up to a 15.73 dB PSNR increase) and rendering efficiency, achieving real-time speeds on complex datasets.

### Strengths
The paper presents a novel 6D Gaussian Splatting (6DGS) method, which effectively models view-dependent effects through spatial and directional information.

By improving color and opacity handling, 6DGS produces high-quality, realistic renderings, particularly useful in scenes with complex reflections and lighting. Its compatibility with the 3DGS framework enhances its practical relevance, allowing integration with minimal adjustments. The experiments are well-designed and validate the method's efficiency, achieving significant reductions in Gaussian points without compromising image quality.

### Weaknesses
The method is primarily tested on synthetic datasets, lacking real-world scene evaluations that would confirm its robustness in practical applications. The new parameter λopa adds complexity, as it requires careful tuning. Additionally, while efficient for static scenes, the model’s performance on dynamic, interactive scenes remains untested, limiting its demonstrated versatility. Finally, the initialization phase can be computationally intensive, posing a potential challenge for scaling to large or complex scenes.

### Questions
- Could the authors provide more insights into how the choice of λopa influences performance across diverse datasets?

- How would the model adapt or perform in dynamic scenes with frequent view changes?

- Would implementing 6DGS directly in CUDA offer significant speed improvements for real-time applications, and if so, are there plans for such an implementation?

### Soundness
4

### Presentation
3

### Contribution
4
