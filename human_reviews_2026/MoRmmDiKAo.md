# HDR-NSFF: High Dynamic Range Neural Scene Flow Fields

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
Radiance of real-world scenes typically spans a much wider dynamic range than what standard cameras can capture. While conventional HDR methods merge alternating-exposure frames, these approaches are inherently constrained to 2D pixel-level alignment, often leading to ghosting artifacts and temporal inconsistency in dynamic scenes. To address these limitations, we present HDR-NSFF, a paradigm shift from 2D-based merging to 4D spatio-temporal modeling. Our framework reconstructs dynamic HDR radiance fields from alternating-exposure monocular videos by representing the scene as a continuous function of space and time, and is compatible with both neural radiance field and 4D Gaussian Splatting (4DGS) based dynamic representations. This unified end-to-end pipeline explicitly models HDR radiance, 3D scene flow, geometry, and tone-mapping, ensuring physical plausibility and global coherence. We further enhance robustness by (i) extending semantic-based optical flow with DINO features to achieve exposure-invariant motion estimation, and (ii) incorporating a generative prior as a regularizer to compensate for  limited observation in monocular captures and saturation-induced information loss. To evaluate HDR space-time view synthesis, we present the first real-world HDR-GoPro dataset specifically designed for dynamic HDR scenes. Experiments demonstrate that HDR-NSFF recovers fine radiance details and coherent dynamics even under challenging exposure variations, thereby achieving state-of-the-art performance in novel space-time view synthesis. Project page: https://shin-dong-yeon.github.io/HDR-NSFF

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Paper proposes HDR-NSFF for reconstructing dynamic High Dynamic Range (HDR) radiance fields from alternatively exposed monocular videos. ​ Unlike traditional HDR methods operating on 2D images, HDR-NSFF models 3D scene flow, HDR radiance, and tone mapping in a unified end-to-end pipeline, which incorporates exposure-robust semantic flow estimation using DINO features, robust depth estimation, and generative priors to address challenges like color inconsistencies and information loss due to saturation. ​ The method is valuated on real-world and synthetic datasets, showing its outperformance over state-of-the-art methods in novel view synthesis, time interpolation, and combined view-time synthesis, achieving superior reconstruction quality and temporal coherence. ​

### Strengths
1. HDR-NSFF explicitly models 3D scene flow, enabling robust handling of occlusions, large motions, and dynamic changes in geometry and radiance. ​The use of DINOv2-based semantic features for optical flow estimation enhances robustness to exposure variations, addressing a key challenge in HDR reconstruction under large motion. ​

2. The proposed generative priors compensate for information loss due to saturation and sparse-view limitations. ​

3. A real-world GoPro dataset is introduced with synchronized multi-exposure captures. ​

### Weaknesses
1. The problem setting, capture (alternative exposures), and employment of generative methods for HDR were explored previously or similarly or at least a straightforward extension in hindsight to the current dynamic HDR for neural radiance fields. 

2. The GoPro dataset may not adequately capture the diversity of real-world lighting conditions, object types, and extreme motion scenarios, as the examples demonstrated in this paper involves simple (but large motions) and often times sunny outdoor scenes, where HDR may not be  ecessarynto reveal lost details due to saturation and under exposure with noise corruption. 

3. Generative priors may introduce artifacts or hallucinations, potentially affecting the fidelity of the reconstruction. ​Please point out if current experiments address this issue. 

4. The framework requires significant training time (10+ hours per scene). ​

### Questions
Note to authors: my preliminary rating would’ve been more like 5 (currently the next rating from 4 is 6) which will be finalized during/after rebuttal. 

1. Have you explored strategies to optimize the training process or reduce the computational cost for real-world applications? ​

2. How do you ensure that the generative priors do not introduce significant artifacts or hallucinations during the reconstruction process? ​ Have you evaluated the impact of these priors on the overall fidelity of the HDR radiance fields? ​

3. How well does the model generalize to other real-world scenarios with more drastic lighting conditions, object types, and motion patterns? E.g. indoor rave party scene with laser and abrupt lighting changes with multiple dancers in complex motions under full and partial occlusion.

4. How do you address inaccuracies propagated by DINOv2 and Depth-AnythingV2?

5. What are the prospects for adapting HDR-NSFF for real-time applications, such as live video processing or interactive systems?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents HDR-NSFF, a novel framework for reconstructing dynamic HDR radiance fields from alternatively exposed monocular videos. The approach integrates 3D scene flow estimation, HDR radiance reconstruction, and tone mapping into a unified end-to-end framework. The authors also introduce a real-world GoPro dataset with synchronized multi-exposure captures to support systematic evaluation.

### Strengths
The paper is well-structured, and the experiments are fairly comprehensive. Reconstructing dynamic HDR radiance fields is also an interesting and meaningful idea.

### Weaknesses
1. The visual quality of the reconstructed images is not satisfactory; noticeable blurriness can be observed in several results (e.g., Figure 1 and Figure 6). Could the authors provide an explanation for this issue?

2. Long exposure tends to introduce motion blur (e.g., in the Big Jump scene), while short exposure often leads to noise. How do the authors address this trade-off in their method?

3. When comparing with other HDR reconstruction methods, was the same tone-mapping function used for visualization and evaluation?

4. How are the flow and depth representations integrated into the proposed system?

### Questions
1. Are the nine cameras perfectly parallel to each other, or are they converging toward a common focal point?

2. Were all cameras fixed and synchronized during the capturing process?

3. The proposed framework involves multiple loss terms. Is the optimization process stable, and are the chosen hyperparameters suitable for all scenes?

4. Since the method adopts a per-scene optimization strategy to reconstruct dynamic HDR radiance fields, can it generalize to render arbitrary viewpoints and timestamps?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents HDR-NSFF, a method that builds a high dynamic range random field of dynamic scenes. It proposes a framework that incorporates a learnable tone-mapping to adapt to high dynamic range. It also collects an HDR dynamic scene dataset for the task.

### Strengths
1. It proposes a learnable tone-mapping function for HDR reconstruction.

2. It introduces a new HDR dynamic scene dataset.

3. Experimental results are better than other SOTA methods.

### Weaknesses
1. I'm wondering why this paper is conducted on NSFF methods, as there are many advanced methods in NeRFs and Gaussians in recent years. Using an outdated baseline somehow weakens the contributions of the paper.

2. The writing of the method section includes so many details on pretrained model selections, whereas the main contribution about the dataset and the learnable tone mapping is relatively introduced far behind. This may mislead the reader's judgment of an important part of the paper.

### Questions
See above.

### Soundness
2

### Presentation
2

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
This paper proposes HDR-NSFF, which reconstructs an implicit 4D scene representation from a monocular dynamic scene video with random exposures, enabling the rendering of HDR novel views or controllable LDR results. The method leverages DINO-Tracker to enhance the robustness of optical flow against illumination variations and incorporates generative priors with multi-view information to address the ill-posed nature of monocular video reconstruction. Experimental results demonstrate the effectiveness of the proposed method compared to dynamic NeRF and 4DGS approaches when handling input videos with random exposures.

### Strengths
- The paper focuses on novel view synthesis for dynamic scenes with randomly exposed inputs, which is a relatively unexplored area in the current novel view synthesis field and holds considerable research value.
- The pipeline of the proposed method is concise and effective, and the introduced Robust Learning Strategies effectively mitigate the sensitivity of the optical flow prior to variations in input exposure.
- Experimental results demonstrate that the proposed method can handle multi-exposure input videos more effectively than dynamic NeRF and 4DGS methods.

### Weaknesses
1. The proposed method is built upon the NSFF framework, which relies on COLMAP to obtain camera parameters from the input images. However, when the input video exhibits large exposure variations, the quality of the camera parameters estimated by COLMAP may degrade or even cause COLMAP to fail entirely. The paper does not provide a clear solution to this issue, which may limit the practical applicability of the proposed method.
2. The paper points out that modeling HDR in 3D scenes can effectively alleviate the exposure inconsistency problem found in 2D methods. However, in the 2D HDR domain, NECHDR[1] specifically addresses the issue of exposure inconsistency in 2D approaches. Therefore, the authors should take the NECHDR work into consideration when performing HDR rendering on training views, and demonstrate through experiments that the proposed method outperforms NECHDR in terms of exposure consistency, or, if the exposure consistency performance is comparable, that the proposed method offers advantages in other aspects such as handling occlusions. 
3. The proposed method should be compared with an additional pipeline: first, applying a 2D single-image or video-based HDR algorithm to the input multi-exposure video to obtain HDR results, and then using existing dynamic NeRF or 4DGS methods to represent the 4D scene based on the generated HDR video.
4. How is the ablation study in Table 3 designed? Both Dino-Tracker and the generative prior are supposed to be the key technical contributions of this paper. Why is there an entry labeled “Ours with Dino-Tracker”—does this imply that the main Ours method does not use Dino-Tracker? The authors should clearly explain the design of the ablation study. It is also recommended to conduct ablation experiments on the full model by individually removing modules such as Dino-Tracker or the generative prior. Moreover, Figure S3 in the supplementary material shows that the improvement brought by the generative prior (GP) to the visual results appears to be minimal, as the visualizations before and after adding GP look almost identical. The authors should clarify the reason for this observation.

[1] Exposure Completing for Temporally Consistent Neural High Dynamic Range Video Rendering. ACMMM 2024.

### Questions
In most 2D HDR methods, the scenes being processed involve small or even fixed camera motions. How does the proposed method perform on the commonly used 2D HDR datasets under such conditions?

### Soundness
2

### Presentation
4

### Contribution
3
