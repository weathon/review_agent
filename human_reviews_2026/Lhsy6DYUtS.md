# RingLight-GS: A compact and expressive framework for modeling scene color in 3DGS

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
3D Gaussian Splatting (3DGS) achieves impressive novel view synthesis in real-time by directly rendering Gaussian primitives. However, it incurs substantial storage demands and struggles to model high-frequency, view-dependent appearance effects under complex illumination. We introduce RingLight-GS, a compact framework that effectively models scene color in 3DGS, delivering high-quality rendering under complex lighting while greatly reducing storage costs. The scene color is separated into a view-independent base color and a view-dependent residual color by disentangling static albedo from dynamic lighting, with the base color learning similarity to the 3DGS opacity. Specifically, the residual color is derived from view-dependent appearance features via a neural tensor ring regression model, influenced by spatial positions and viewing directions. Extensive experiments on synthetic and real-world datasets demonstrate that RingLight-GS consistently outperforms both NeRF-based and 3DGS-based baselines. It delivers sharper highlights, better material consistency, and lower perceptual error with minimal memory overhead.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RingLight-GS, a compact and expressive framework for modeling scene color in 3D Gaussian Splatting (3DGS) that aims to reduce storage costs while improving rendering quality, especially under complex lighting. The core contribution is the separation of scene color into a view-independent base color and a view-dependent residual color. This residual color, which captures high-frequency details like specular highlights, is modeled using a neural tensor ring regression. This approach allows for sharper highlights, better material consistency, and lower perceptual error with minimal memory overhead compared to previous methods.

### Strengths
Originality: The paper presents a original approach by integrating existing ideas into a novel 3D Gaussian Splatting framework. Its main contribution lies in modeling view-dependent residual colors using a neural tensor ring (TR) regression, combining tensor decomposition for parameter efficiency with implicit neural representations. The network dynamically generates tensor cores from continuous inputs (position and view direction), enabling a compact, continuous, and highly expressive appearance function that captures high-frequency details beyond prior tensor- or SH-based methods. This design demonstrates both technical novelty and practical effectiveness in enhancing appearance modeling in 3DGS.

Quality: The paper presents solid technical work, supported by a rigorous and comprehensive experimental evaluation. The authors validate their method on three diverse datasets—Tanks and Temples, NeRF-Synthetic, and Shiny-Blender—testing performance on complex geometry, clean indoor scenes, and challenging specular reflections, respectively. Comparisons with a wide range of NeRF- and 3DGS-based baselines are thorough and fair.

Clarity: The paper is clearly structured and written. It follows a logical progression from motivation and background to the proposed method and evaluation. Figures are used effectively: Figure 1 summarizes the method's performance against baselines, Figure 2 shows the architectural pipeline, and later figures illustrate the underlying tensor concepts. The methodology section first defines notations and preliminaries, then explains the core model components, making the technical details more accessible.

Significance: This work addresses a key challenge in 3D Gaussian Splatting (3DGS)—accurate and efficient modeling of complex, view-dependent lighting—by introducing a neural tensor representation that captures high-frequency details such as specular highlights. The method improves rendering fidelity while substantially reducing memory footprint compared to high-order Spherical Harmonics, making 3DGS more compact and practical. These improvements are relevant for applications with memory or bandwidth constraints, and the neural tensor approach may inform future research on compact representations of other complex scene properties.

### Weaknesses
-1. In the Optimization Strategy section, the authors introduce γ and β as hyperparameters. However, only the value of γ (set to 10,000) is briefly mentioned, without further discussion. It would strengthen the paper’s rigor if the authors could include an analysis—either in the ablation study or in the appendix—showing how different choices of these hyperparameters (e.g., γ, β, etc.) affect the results.

-2. The method separates color into a view-independent "base color" (diffuse albedo) and a view-dependent "residual." However, this decomposition is learned without direct physical supervision. The paper lacks multi-view renderings of the same scene, making it unclear whether RINGLIGHT-GS can maintain high-frequency detail consistency across views. Including such results would clarify this capability.

### Questions
-1. RINGLIGHT-GS leverages the TR function to compute appearance features and subsequently derive the residual color, enabling the reconstruction of high-frequency details such as specular highlights. Have the authors verified whether RINGLIGHT-GS is also effective in reconstructing diffuse surfaces?

-2. The results in Table 5 indicate that RingLight-GS requires a significantly longer training time (roughly 2-3x) compared to the original 3DGS. This represents a substantial trade-off for the achieved compactness and quality. Could you please elaborate on the primary computational bottlenecks introduced by your method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes RingLight-GS, a 3DGS method that improves the color representation from SH coefficients with a neural tensor ring regression model. It decomposes each Gaussian’s color into a base color and a view-dependent residual color. The residual color is predicted via a neural TR module that takes spatial positions and view directions as input. The authors claim this design provides higher-quality, compact rendering under complex lighting conditions. Experiments show small performance improvements and storage reduction.

### Strengths
- The paper is easy to follow.

### Weaknesses
- The main claimed contribution, separating scene color into a view-independent base color and a view-dependent residual color, is a common idea and can barely be considered an original contribution. Exactly, it's just the basic rationale for the SH-based color representation that is natively adopted in 3DGS, where the degree 0 of the SH is actually the view-independent base color, and the high degrees are for the view-dependent residual color. To predict the view-dependent color more precisely, some previous works like EnvGS [1] have been proposed with the same idea, which are ignored in this work. In comparison, this work just represents a less accurate solution with weaker significance.

- 1. The introduced Neural Tensor Ring Regression is essentially not different from NeRF-style MLP with Fourier positional encoding. The TR structure merely rearranges the computation into a multilinear contraction of MLP outputs but does not change the representational capacity or introduce a meaningful inductive bias. Similar tensor decompositions have already been applied to plenty of NeRF and also 3DGS previous works. The paper does not explain why the TR form is advantageous and non-trivial. 

  2. Empirical, no sufficient justification for using Tensor Ring. The only ablation (“w/o TR”) is inconclusive, considering it's unclear about the parameter count and computational cost, or the hyperparameter adjustments. From the experimental results, the paper does not show why the TR module captures high-frequency, view-dependent effects better than existing SH, MLP-based or other tensor decomposition models. No in-depth study is provided.

- Reported performance gains over 3DGS and 3iGS are small. Moreover, the comparison does not include the SOTA related methods like EnvGS for complex lighting condition, Scaffold-GS [2] equipped with MLP for better quality and compression, and other SOTA compression 3DGS works released in 2025.

[1] Xie, Tao, et al. "Envgs: Modeling view-dependent appearance with environment gaussian." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[2] Lu, Tao, et al. "Scaffold-gs: Structured 3d gaussians for view-adaptive rendering." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
See the weaknesses. Overall, the method is mostly an engineering by combining existing ideas without offering new theoretical insights or a clear performance breakthrough.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes RingLight-GS, a compact extension of 3D Gaussian Splatting that separates scene color into a view-independent base and a view-dependent residual. By modeling residuals with a neural tensor ring regression conditioned on position and view direction, the method improves rendering quality under complex illumination while significantly reducing storage requirements.

### Strengths
The paper presents a novel approach to modeling view-dependent appearance in Gaussian Splatting using Neural TR Regression, which dynamically generates high-dimensional tensor slices for each spatial position and view. This method is original in combining tensor decomposition with neural feature learning, effectively capturing complex lighting effects without explicit BRDF parameters. The approach is well-motivated, clearly explained, and demonstrates potential for high-quality, compact scene representation, addressing key limitations of prior SH-based methods.

### Weaknesses
- While the Neural TR Regression effectively models view-dependent appearance with compact storage, it introduces additional model complexity and relies on high-dimensional tensor decomposition, which may increase training time and memory usage. Furthermore, the approach assumes accurate input positions and viewing directions, making it potentially sensitive to noise or imperfect geometry.

- The presentation of experimental results could benefit from supplementary video demonstrations (e.g., depth maps, normal maps). I strongly encourage the authors to include video comparisons on test scenes to better showcase the effectiveness of the proposed method. 

- Can this framework be extended to 2D Gaussian Splatting (2DGS)? I am curious about the performance implications if 2DGS were used instead of 3DGS. If I understand correctly, subsequent designs such as the “Neural TR decomposition” appear to be agnostic to the choice of primitive representation. How would the rendering quality compare if 2DGS is adopted? If 2DGS cannot be directly integrated into the current framework, what are the main limitations preventing this?

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper presents RingLight-GS, a method that enhances 3D Gaussian Splatting (3DGS) by improving compactness and handling of complex lighting effects. Traditional 3DGS relies on large Spherical Harmonics (SH) coefficients to represent color, leading to high memory consumption. RingLight-GS addresses this by decomposing each Gaussian’s color into a view-independent base color and a view-dependent residual color. The core contribution is a Neural Tensor Ring (TR) Regression model that learns the view-dependent component. This TR model maps spatial coordinates and viewing directions to appearance features through a decomposed tensor representation that is both compact and expressive. As a result, RingLight-GS achieves approximately 2.5–3× storage reduction compared to standard 3DGS, while improving rendering quality, particularly for specular highlights and reflections.

### Strengths
1. Separating the base color from a tensor-factorized residual is an elegant and effective design. It reduces 3DGS’s memory overhead and improves its ability to represent complex, view-dependent effects like specular highlights that SHs handle poorly.

2. The method achieves both compactness and improved performance: it reduces model size substantially (e.g., from 68 MB to 22 MB on NeRF-Synthetic) while also attaining higher PSNR, SSIM, and LPIPS scores across multiple datasets, outperforming existing compression and lighting-aware 3DGS variants.

### Weaknesses
The advanced Neural TR model incurs a computational cost. The paper reports substantially longer training times (e.g., ~29 minutes versus ~15 minutes for standard 3DGS on Shiny-Blender) and increased VRAM usage during training, making it less efficient to train and deploy than the original 3DGS.

The method introduces several new components (neural TR, render module, loss scaling factor γ) and associated hyperparameters (TR rank, feature dimension). The ablation studies show performance degrades if these are not set correctly, adding complexity and potentially limiting its ease of adoption.

In Figure 2, which aims to illustrate the overall scene color modeling framework, is too schematic and abstract. It uses generic blocks (e.g., "TR Function Regression," "Render Module") without visually hinting at their unique internal mechanics (e.g., how the neural TR cores connect). This makes it difficult for a reader to grasp the innovative data flow and architecture without constantly cross-referencing the complex text in Section 3.2. A more detailed or annotated diagram would be greatly beneficial.

### Questions
The Neural TR module plays a central role in modeling view-dependent effects. It remains unclear how its efficiency and compactness scale with scene complexity. In scenes with substantially more Gaussians or a larger spatial extent, would preserving rendering quality necessitate a proportional increase in TR rank or feature dimension, thereby diminishing the storage benefits?

### Soundness
2

### Presentation
2

### Contribution
2
