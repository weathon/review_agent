# MTGS: Multi-Traversal Gaussian Splatting

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Multi-traversal data, commonly collected through daily commutes or by self-driving fleets, provides multiple viewpoints for scene reconstruction within a road block. This data offers significant potential for high-quality novel view synthesis, which is crucial for applications such as autonomous vehicle simulators. However, inherent challenges in multi-traversal data often result in suboptimal reconstruction quality, including variations in appearance and the presence of dynamic objects. To address these issues, we propose Multi-Traversal Gaussian Splatting (MTGS), a novel approach that reconstructs high-quality driving scenes from arbitrarily collected multi-traversal data by modeling a shared static geometry while separately handling dynamic elements and appearance variations. Our method employs a multi-traversal scene graph with a shared static node and traversal-specific dynamic nodes, complemented by color correction nodes with learnable spherical harmonics coefficient residuals. This approach enables high-fidelity novel view synthesis and provides flexibility to navigate any viewpoint. We conduct extensive experiments on a large-scale driving dataset, nuPlan, with multi-traversal data. Our results demonstrate that MTGS improves LPIPS by 23.5% and geometry accuracy by 46.3% compared to single-traversal baselines. Code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Multi-traversal data enables multi-view road scene reconstruction and high-quality novel view synthesis (vital for autonomous vehicle simulators). However, it faces challenges like appearance variations and dynamic objects, leading to suboptimal reconstruction. We propose Multi-Traversal Gaussian Splatting (MTGS), which models shared static geometry while handling dynamic elements and appearance variations separately for high-fidelity novel view synthesis. Experiments on nuPlan show MTGS improves LPIPS by 23.5% and geometry accuracy by 46.3% over single-traversal baselines.

### Strengths
1. Interesting questions and clear definitions of the questions.
2. Clear expression of methods and presentation of algorithms. 
3. Reasonable ablation experiments.

### Weaknesses
The paper has several areas that need improvement:
- **Lack of comparison with the latest methods**: The paper fails to compare with the latest novel view synthesis methods such as ReconDreamer, FreeVS, and Dist - 4D. By comparing with these methods, the performance, advantages, and disadvantages of the proposed method can be more comprehensively evaluated, providing more valuable reference for readers.
- **Absence of novel view synthesis visualization**: The paper does not provide novel view synthesis visualization. It is recommended to visualize translations of 1m, 2m, and 4m. This kind of visualization can more intuitively show the effect of the method in novel view synthesis, helping readers better understand the performance of the method.
- **Poor - quality images**: The image quality in the paper is poor and cannot be used in real - world scenarios, and there is a large gap compared with the novel view synthesis of ReconDreamer. High - quality images are crucial for demonstrating the effectiveness of the method, and poor - quality images may affect readers' understanding and evaluation of the method.
- **Poor - quality reconstruction results in the demo**: The reconstruction results in the demo seem to be of poor quality, and there is still a large gap compared with the demo of OmniRe. The demo is an important means to show the practical effect of the method. Poor - quality reconstruction results may make readers question the practical value of the method.

### Questions
The main problem lies in the above aspects. Additionally, the innovation of the thesis is limited.

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
In this paper, the authors propose a novel approach MTGS that reconstructs high-quality driving scenes from multi-traversal data. Specifically, MTGS employs a multi-traversal scene graph consisting of a static node and dynamic nodes. To address appearance variations across different traversals, the scene graph is further complemented by appearance nodes with learnable spherical
harmonics coefficient residuals. Experiments demonstrate that MTGS achieves state-of-the-art performance in both driving scene reconstruction and novel view synthesis.

### Strengths
1. This paper focuses on high-fidelity driving scene reconstruction using multi-traversal data, which is valuable for cross-lane simulations of AV. However, appearance variations in multi-traversal data disrupt scene consistency and introduce geometric errors, making the reconstruction process more challenging.
2. The authors decompose the entire driving scene using a multi-traversal scene graph with three core nodes: shared static nodes, appearance nodes, and transient nodes, which significantly improves the fidelity and geometric consistency of scene reconstruction.
3. Extensive experiments on the nuPlan dataset demonstrate that MTGS achieves superior performance.

### Weaknesses
1. One of the core innovations of MTGS—the "Scene Graph Node Decomposition"—has similarities to prior 3D Gaussian Splatting-related methods (such as StreetGS, DrivingGaussian) and lacks breakthrough ideas.
2. MTGS’s performance advantages are highly dependent on multi-traversal data, which poses significant drawbacks in real-world autonomous driving scenarios and hinders its practical deployment due to high data collection costs and inefficiencies.

### Questions
Please check the weaknesses.

### Soundness
3

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
This paper proposes MTGS, a method for reconstructing driving scenes using multi-traversal data, where multiple recordings of the same road are available. The goal is to achieve high-quality view extrapolation and photorealistic driving scene reconstruction. To do so, the authors propose a multi-traversal scene graph with shared static nodes and traversal-specific appearance and transient nodes as welll as some tricks for better quality.  The authors conduct comprehensive experiments using the large-scale nuPlan dataset, which contains multi-traversal data.

### Strengths
- The paper is technically solid, well-organized, and supported by comprehensive ablation studies.
- The proposed framework effectively aligns appearances and reconstructs static environments with high visual fidelity in drivable areas.
- Leveraging multi-traversal data for reconstruction is an important and underexplored direction for autonomous driving and digital twin simulation.

### Weaknesses
- The handling of transient and dynamic objects remains problematic. Since these objects vary across traversals, the current model cannot establish consistent geometry or appearance. While the approach has potential as a strong background reconstruction technique, it is still incomplete as a holistic driving-scene solution.

- The method’s technical novelty is moderate. While the multi-traversal setting is valuable, the main modules (scene graph design, affine correction, normal/depth regularization) extend prior work rather than introducing fundamentally new formulations. The problem formulation is also very similar to reconstruction methods using in the wild images.

- As can be seen from the videos in the supplementary materials,  roadside vehicles are almost broken (in traversal_test).

- In Table 1, for ST setting, StreetSG and OmniRe are even worse than the original 3DGS in terms of PSNR, which needs better explanations.

### Questions
See weaknesses.

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
4

### Summary
This paper presents a method for reconstructing realistic driving scenes from multiple trips along the same route. It separates static geometry from dynamic elements and adjusts for lighting or appearance changes, leading to cleaner, more consistent results. Experiments show that MTGS achieves noticeably better image quality and geometry accuracy compared to single-traversal methods.

### Strengths
1. The paper is well written and easy to follow.
2. The paper contributes to an interesting and important problem of reconstructing a scene across multi-traversals. Multi-traversal data often involve significant temporal and appearance variations such as changes in illumination, weather, and dynamic objects, which make achieving consistent, high-fidelity reconstruction difficult. This paper shows that their solution achieves strong results in multi-traversal reconstructions.

### Weaknesses
1. While I think this paper tackles an interesting topic. However, I think the most important challenge in the multi-traversal reconstruction is not enough handled. The decomposition of static and dynamic objects is common in self-driving gaussian splatting. While it is a natural solution in the multi-traversal scenarios, I believe it is not a significant contributions here. I feel the solutions to handle the illumination, weathers, etc. need to be strengthened in this paper. \
2. This paper seems lacking of visual comparisons, can the author provide more results comparisons on novel-view synthesis?

### Questions
1.I understand that the traversal-specific residual coefficients are meant to capture appearance differences such as lighting and reflections. However, since these residuals are learned independently for each traversal, wouldn’t that make it difficult for the model to generalize or interpolate lighting changes between traversals? In particular, because higher-order SHs (which normally encode directional illumination) are now traversal-specific, isn’t there a risk that the model just memorizes traversal-specific lighting instead of learning a shared representation of how lighting varies? From table 2, it seems without the Appr.Node, the novel-view psnr/ssim shows better results. \
2. You mentioned that sharing Y_0,0 forces the Gaussian to learn albedo. But since this is enforced only implicitly by sharing, without explicit supervision or intrinsic decomposition, how can you be sure that Y_0,0 doesn’t still capture some average lighting components?\
3. For the LiDAR-guided exposure alignment, view-dependent lighting effects such as shadows or specular highlights might be mistaken for exposure differences. Could this cause the method to over-correct and distort the true scene appearance?

### Soundness
3

### Presentation
3

### Contribution
2
